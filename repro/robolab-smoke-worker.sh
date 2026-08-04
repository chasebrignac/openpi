#!/usr/bin/env bash
# Reproducible one-node RoboLab/Isaac camera smoke for the pinned R580 eval AMI.

set -uo pipefail

readonly AWS_REGION="us-east-2"
readonly RESULT_BUCKET="pi05-repro-752160877725-us-east-2"
readonly EXPECTED_AMI_ID="ami-06517bc7fad3c6a48"
readonly EXPECTED_DRIVER="580.126.09"
readonly EXPECTED_ROBOLAB_REVISION="0aef241fb088ca21bb4ebd24448940ed56620d17"
readonly EXPECTED_CLIENT_REVISION="aa6420561529593114160d05e5ad155792b272f3"
readonly EXPECTED_ISAACLAB_BASE_DIGEST="sha256:b4d8e96cbfb9a6c40067bec6cc5ee180e36d4c0164b25f7215c5f47e31897b94"
readonly ROBOLAB_IMAGE_URI="752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro@sha256:2d17c15e62887c9fc8b4c41b7ee3d39c4c187348eb55b4273fd24e785a3325e7"

install -d -m 0755 /opt/pi05/logs
readonly STARTED_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
readonly LOG_PATH="/opt/pi05/logs/robolab-smoke-${STARTED_UTC}.log"

metadata_token="$(curl -fsS -X PUT \
  -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600' \
  http://169.254.169.254/latest/api/token)"
metadata() {
  curl -fsS \
    -H "X-aws-ec2-metadata-token: ${metadata_token}" \
    "http://169.254.169.254/latest/meta-data/$1"
}

instance_id="$(metadata instance-id)"
instance_type="$(metadata instance-type)"
ami_id="$(metadata ami-id)"
readonly instance_id instance_type ami_id
readonly RESULT_KEY="manual-smoke/robolab/${STARTED_UTC}-${instance_id}.log"

smoke_status=0
(
  set -euo pipefail

  echo "started_utc=${STARTED_UTC}"
  echo "instance_id=${instance_id}"
  echo "instance_type=${instance_type}"
  echo "ami_id=${ami_id}"
  echo "image_uri=${ROBOLAB_IMAGE_URI}"
  test "${ami_id}" = "${EXPECTED_AMI_ID}"

  driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u)"
  echo "driver_version=${driver_version}"
  test "${driver_version}" = "${EXPECTED_DRIVER}"
  nvidia-smi
  docker version

  aws ecr get-login-password --region "${AWS_REGION}" |
    docker login --username AWS --password-stdin \
      752160877725.dkr.ecr.us-east-2.amazonaws.com
  docker pull "${ROBOLAB_IMAGE_URI}"
  docker image inspect "${ROBOLAB_IMAGE_URI}"

  test "$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' \
    "${ROBOLAB_IMAGE_URI}")" = "${EXPECTED_ROBOLAB_REVISION}"
  test "$(docker image inspect --format '{{index .Config.Labels "ai.openpi.client-revision"}}' \
    "${ROBOLAB_IMAGE_URI}")" = "${EXPECTED_CLIENT_REVISION}"
  test "$(docker image inspect --format '{{index .Config.Labels "ai.openpi.isaaclab-base-digest"}}' \
    "${ROBOLAB_IMAGE_URI}")" = "${EXPECTED_ISAACLAB_BASE_DIGEST}"

  docker run --rm \
    --gpus all \
    --network host \
    --ipc host \
    --ulimit core=0 \
    -e OMNI_KIT_ACCEPT_EULA=YES \
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    --entrypoint /workspace/isaaclab/_isaac_sim/python.sh \
    "${ROBOLAB_IMAGE_URI}" \
    -c "import importlib.metadata as m, torch, torch._dynamo, typeguard; assert m.version('typing_extensions') == '4.12.2'; assert m.version('typeguard') == '4.4.2'; print(torch.__version__)"

  docker run --rm \
    --gpus all \
    --network host \
    --ipc host \
    --ulimit core=0 \
    -e OMNI_KIT_ACCEPT_EULA=YES \
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    --entrypoint /workspace/isaaclab/_isaac_sim/python.sh \
    "${ROBOLAB_IMAGE_URI}" \
    -m pytest -q \
    tests/test_isaaclab.py \
    tests/test_registered_envs.py \
    tests/test_tasks_valid.py \
    tests/test_run_empty.py
) >"${LOG_PATH}" 2>&1 || smoke_status=$?

printf '\nsmoke_exit_code=%s\nfinished_utc=%s\n' \
  "${smoke_status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"${LOG_PATH}"
cat "${LOG_PATH}"

log_sha256="$(sha256sum "${LOG_PATH}" | cut -d ' ' -f1)"
if ! aws s3api put-object \
  --region "${AWS_REGION}" \
  --bucket "${RESULT_BUCKET}" \
  --key "${RESULT_KEY}" \
  --body "${LOG_PATH}" \
  --server-side-encryption AES256 \
  --metadata "instance-id=${instance_id},ami-id=${ami_id},driver=${EXPECTED_DRIVER},image-digest=2d17c15e62887c9fc8b4c41b7ee3d39c4c187348eb55b4273fd24e785a3325e7,log-sha256=${log_sha256},smoke-exit-code=${smoke_status}"; then
  echo "failed to upload ${LOG_PATH} to s3://${RESULT_BUCKET}/${RESULT_KEY}" >&2
  if (( smoke_status == 0 )); then
    smoke_status=90
  fi
fi

exit "${smoke_status}"
