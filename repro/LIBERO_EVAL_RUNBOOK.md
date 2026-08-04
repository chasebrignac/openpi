# Reproducible LIBERO evaluation

This procedure evaluates one eager or TensorRT checkpoint on all four official ten-task LIBERO suites without
changing OpenPI's WebSocket policy-server protocol. The policy server and simulator run as two processes inside
one `--network none` worker container and communicate only over loopback. Eager evaluation uses `g6e.4xlarge`.
Compiled evaluation uses `g7e.4xlarge` and is deliberately tied to the exact still-running instance that built the
engines. During the manual-first phase, export, engine build, server startup, and rollout evaluation are direct
commands in the final evaluator image on that retained instance. The compiled spec renderer is validation-only
until an exact-existing-instance dispatcher has been implemented and replayed.

## Immutable evaluator boundary

The OpenPI source pins `third_party/libero` as a gitlink at
`f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`. A normal `git archive` or bundle checkout does **not** contain the
submodule files; the old `examples/libero/Dockerfile` therefore cannot build from the worker source artifact.
`repro/Dockerfile.libero` closes that gap by:

- requiring an account-local policy or combined TensorRT-policy image URI pinned by `sha256` as its parent;
- fetching exactly the gitlink commit during the image build and deleting its Git metadata only after verification;
- installing a separate CPython 3.8.20 simulator environment from
  `repro/libero-evaluator-requirements.txt` while retaining the parent policy runtime;
- embedding the reviewed source tree while the worker independently mounts and verifies the same source bundle;
- hashing the lock, loading every suite's fixed init-state files, and requiring 10 tasks with at least 50 states each;
- reasserting LeRobot-v2 and evaluator labels, recording `ai.openpi.policy-backend=eager|tensorrt`, and adding the
  LIBERO revision and lock hash as OCI labels.

The final account-local ECR digest, not a mutable tag, is the runtime dependency boundary. Do not use an image if
its local smoke result or labels differ from `repro/libero-evaluator-contract.json`.

## 1. Build and inspect the image manually

Commit the reviewed source first. Set the exact parent policy digest that already passed the policy-container smoke
tests; do not use a tag.

```bash
export PI05_EVALUATOR_SOURCE_COMMIT=e30480a6de404c74a996863c4fde89367350cf70
export PI05_PARENT_POLICY_SOURCE_COMMIT=229c08ea2a13a70cbbf1a9c8a1f31cb1ca674dee
export PI05_SOURCE_COMMIT="$PI05_EVALUATOR_SOURCE_COMMIT"
export PI05_SOURCE_CHECKOUT="/absolute/path/to/verified/openpi-$PI05_EVALUATOR_SOURCE_COMMIT"
export PI05_POLICY_BASE_IMAGE='752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro@sha256:d76e6d73fca409e998304a6a8997f80fab1252fe0301d667a072f99dd6624f24'
export PI05_LIBERO_LOCAL_IMAGE="pi05-repro-libero:${PI05_SOURCE_COMMIT}"
export PI05_ECR_REPOSITORY='752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro'
export PI05_LIBERO_TAG="libero-evaluator-${PI05_SOURCE_COMMIT}"
export PI05_POLICY_BACKEND=eager

test "$(git -C "$PI05_SOURCE_CHECKOUT" rev-parse HEAD)" = "$PI05_SOURCE_COMMIT"
test -z "$(git -C "$PI05_SOURCE_CHECKOUT" status --porcelain)"
aws ecr get-login-password --region us-east-2 | \
  docker login --username AWS --password-stdin 752160877725.dkr.ecr.us-east-2.amazonaws.com
docker pull "$PI05_POLICY_BASE_IMAGE"
docker image inspect "$PI05_POLICY_BASE_IMAGE" --format '{{json .Config.Labels}}' | jq .
git -C "$PI05_SOURCE_CHECKOUT" archive --format=tar "$PI05_SOURCE_COMMIT" | docker build --pull=false \
  --file repro/Dockerfile.libero \
  --build-arg POLICY_BASE_IMAGE="$PI05_POLICY_BASE_IMAGE" \
  --build-arg POLICY_BACKEND="$PI05_POLICY_BACKEND" \
  --build-arg SOURCE_SHA="$PI05_SOURCE_COMMIT" \
  --tag "$PI05_LIBERO_LOCAL_IMAGE" -

docker image inspect "$PI05_LIBERO_LOCAL_IMAGE" \
  --format '{{json .Config.Labels}}' | jq .
docker run --rm --network none "$PI05_LIBERO_LOCAL_IMAGE" \
  /opt/libero-venv/bin/python -c \
  'import json, os, platform; value=json.load(open(os.environ["LIBERO_RUNTIME_CONTRACT"])); assert platform.python_version()=="3.8.20"; print(value)'

if aws ecr describe-images --region us-east-2 --repository-name pi05-repro \
  --image-ids imageTag="$PI05_LIBERO_TAG" >/dev/null 2>&1; then
  echo "refusing existing immutable evaluator tag: $PI05_LIBERO_TAG" >&2
  exit 1
fi
docker tag "$PI05_LIBERO_LOCAL_IMAGE" "$PI05_ECR_REPOSITORY:$PI05_LIBERO_TAG"
docker push "$PI05_ECR_REPOSITORY:$PI05_LIBERO_TAG"
export PI05_LIBERO_DIGEST="$(aws ecr describe-images --region us-east-2 \
  --repository-name pi05-repro --image-ids imageTag="$PI05_LIBERO_TAG" \
  --query 'imageDetails[0].imageDigest' --output text)"
test "${PI05_LIBERO_DIGEST#sha256:}" != "$PI05_LIBERO_DIGEST"
export PI05_LIBERO_IMAGE="$PI05_ECR_REPOSITORY@$PI05_LIBERO_DIGEST"
docker pull "$PI05_LIBERO_IMAGE"
test "$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' \
  "$PI05_LIBERO_IMAGE")" = "$PI05_SOURCE_COMMIT"
test "$(docker image inspect --format '{{index .Config.Labels "ai.openpi.image-purpose"}}' \
  "$PI05_LIBERO_IMAGE")" = libero-evaluator
test "$(docker image inspect --format '{{index .Config.Labels "ai.openpi.parent-policy-image"}}' \
  "$PI05_LIBERO_IMAGE")" = "$PI05_POLICY_BASE_IMAGE"
test "$(docker image inspect --format '{{index .Config.Labels "ai.openpi.policy-backend"}}' \
  "$PI05_LIBERO_IMAGE")" = "$PI05_POLICY_BACKEND"
```

The final labels must include the exact source commit, `ai.openpi.image-purpose=libero-evaluator`, LeRobot v2 revision
`0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`, LIBERO revision
`f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`, backend, and lock SHA-256. Mirror/push the reviewed image to the
project ECR, resolve its registry manifest digest, pull that digest, and repeat both inspections. Record the parent
digest, final digest, build log, labels, and smoke output in the manual runbook ledger. The eager worker and the
compiled contract validator reject a final image whose source, purpose, or LeRobot labels differ from its spec.

For TensorRT, set `PI05_POLICY_BASE_IMAGE` to the reviewed **LeRobot-v2 combined TensorRT-policy** digest and set
`PI05_POLICY_BACKEND=tensorrt`. The build verifies TensorRT 11, `trtexec`, and `/opt/modelopt/bin/python`; a policy
image without the complete compiler/runtime cannot masquerade as a compiled evaluator. Build and push this final
LIBERO evaluator image before exporting or compiling. Its final ECR digest—not its parent digest—must be passed as
`--image-digest` to validation, engine build, serving, and evaluation.
Also record the combined parent's exact
`ai.openpi.parent-tensorrt-compiler-image` label as
`PI05_TENSORRT_COMPILER_IMAGE`; its
`ai.openpi.parent-tensorrt-compiler-source-revision` must equal
`PI05_SOURCE_COMMIT`.

Docker is not available in the local development environment used to author this contract, so this image build is
the first required manual replay step. Do not begin a retained GPU replay until it passes.

## 2. One-trial local GPU replay

Use a writable output directory and a read-only checkpoint. The process below starts the existing
`scripts/serve_policy.py`, waits for its loopback socket, runs one fixed init state for each of 40 tasks with
`examples/libero/main.py`, validates the exact 40-record result, writes hashes/metrics to a manifest, and stops the
server. No inbound port or container network is needed.

`projected_cost_usd` is still nonzero when the smoke runs on an already-active
workbench: it records this evaluation's non-overlapping share of the paid
reservation, not the cost of launching another instance. Set that allocation
from the cost ledger before starting.

The block below is the exact retained-host record for accepted attempts 05 and
06, including their create-once image and ledger inputs. Do not rerun it over
those existing output directories; any future replay uses new attempt numbers.

```bash
set -euo pipefail
export PI05_LIBERO_IMAGE='752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro@sha256:51b352c1a7205d6bdae668f99060ebd05049042e1d89916993830acbdc63b374'
export PI05_EVALUATOR_SOURCE_COMMIT=e30480a6de404c74a996863c4fde89367350cf70
export PI05_PARENT_POLICY_SOURCE_COMMIT=229c08ea2a13a70cbbf1a9c8a1f31cb1ca674dee
export PI05_SOURCE_COMMIT="$PI05_EVALUATOR_SOURCE_COMMIT"
export PI05_SOURCE_CHECKOUT="/opt/pi05/source/openpi-$PI05_EVALUATOR_SOURCE_COMMIT"
export PI05_PARENT_POLICY_IMAGE_DIGEST=sha256:d76e6d73fca409e998304a6a8997f80fab1252fe0301d667a072f99dd6624f24
export PI05_CHECKPOINT=/opt/pi05/checkpoints/pi05_libero_pytorch
export PI05_CONVERTED_MANIFEST=/opt/pi05/checkpoints/_manifests/pi05_libero_pytorch.converted-manifest.json
export PI05_MODEL_REVISION=c73bb6ff5cbaa3c7bba5f03ea38c22bd95e8274308285e2f17b6ed2d73688dd0
export PI05_COST_LEDGER_VERSION_ID=WwdchX.Da46XNc5.cVFjkU7.qqryrA7h
export PI05_COST_LEDGER_SHA256=13eb67119d0261a58f52a2b1633e125b2ff8e47214095c70807f96e88c316db9
: "${PI05_EAGER_SMOKE_PROJECTED_COST_USD:?set the nonzero ledger allocation}"
python3 -c 'import os; assert float(os.environ["PI05_EAGER_SMOKE_PROJECTED_COST_USD"]) > 0'

imds_token="$(curl --fail --silent --show-error --request PUT \
  --header 'X-aws-ec2-metadata-token-ttl-seconds: 60' \
  http://169.254.169.254/latest/api/token)"
instance_identity="$(curl --fail --silent --show-error \
  --header "X-aws-ec2-metadata-token: $imds_token" \
  http://169.254.169.254/latest/dynamic/instance-identity/document)"
unset imds_token
test "$(jq -r .accountId <<<"$instance_identity")" = 752160877725
test "$(jq -r .region <<<"$instance_identity")" = us-east-2
test "$(jq -r .instanceType <<<"$instance_identity")" = g6e.4xlarge
export PI05_INSTANCE_ID="$(jq -r .instanceId <<<"$instance_identity")"
printf '%s\n' "$PI05_INSTANCE_ID" | grep -Eq '^i-[0-9a-f]{17}$'

test "$(git -C "$PI05_SOURCE_CHECKOUT" rev-parse HEAD)" = "$PI05_SOURCE_COMMIT"
test -z "$(git -C "$PI05_SOURCE_CHECKOUT" status --porcelain)"
test -d "$PI05_CHECKPOINT"
test -f "$PI05_CONVERTED_MANIFEST"

# /mnt/openpi is a retained-host symlink to /opt/pi05. Keep host inputs on the
# canonical /opt/pi05 path; only the read-only container destination uses /mnt.

for attempt in 05 06; do
  run_id="libero-base-runtime-smoke-$attempt"
  output="/opt/pi05/evidence/$run_id"
  test ! -e "$output"
  install -d -m 0700 -o 1000 -g 1000 "$output"
  test ! -e "$output/replay.log"
  test ! -e "$output/timing.json"
  test ! -e "$output/timing.json.tmp"

  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  set +e
  docker run --rm --gpus all --network none --hostname pi05-libero \
    --add-host pi05-libero:127.0.0.1 --ipc host --shm-size 32g \
    --user 1000:1000 --workdir /workspace/openpi \
    --mount type=bind,src="$PI05_SOURCE_CHECKOUT",dst=/workspace/openpi,readonly \
    --mount type=bind,src="$PI05_CHECKPOINT",dst=/mnt/openpi/checkpoints/pi05_libero,readonly \
    --mount type=bind,src="$output",dst=/output \
    --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/cache \
    --env USER=pi05 --env LOGNAME=pi05 \
    --env TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env PYTHONPATH=/workspace/openpi/src:/workspace/openpi \
    --env PI05_SOURCE_SHA="$PI05_SOURCE_COMMIT" \
    --env PI05_IMAGE_DIGEST="${PI05_LIBERO_IMAGE##*@}" \
    --env PI05_INSTANCE_ID="$PI05_INSTANCE_ID" \
    --env PI05_RUN_ID="$run_id" \
    --env PI05_SEED=7 \
    "$PI05_LIBERO_IMAGE" \
    python scripts/repro_libero_eval.py run \
      --policy-config pi05_libero \
      --checkpoint /mnt/openpi/checkpoints/pi05_libero \
      --model-revision "$PI05_MODEL_REVISION" \
      --stage base --trials-per-task 1 --seed 7 \
      --instance-type g6e.4xlarge \
      --projected-cost-usd "$PI05_EAGER_SMOKE_PROJECTED_COST_USD" \
      --output-root /output \
    2>&1 | tee "$output/replay.log"
  pipeline_status=("${PIPESTATUS[@]}")
  smoke_exit_code="${pipeline_status[0]}"
  if test "$smoke_exit_code" -eq 0; then
    smoke_exit_code="${pipeline_status[1]}"
  fi
  set -e
  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  jq -n --arg started_at "$started_at" --arg finished_at "$finished_at" \
    --argjson exit_code "$smoke_exit_code" \
    '{started_at:$started_at,finished_at:$finished_at,exit_code:$exit_code}' \
    > "$output/timing.json.tmp"
  mv "$output/timing.json.tmp" "$output/timing.json"
  test "$smoke_exit_code" -eq 0
done
```

Attempts 01 through 04 remain immutable failure evidence. The two accepted clean
outputs are explicitly `libero-base-runtime-smoke-05` and `-06`; never reuse or
publish a failed attempt directory. Record both successful manifests and
artifact hashes. If a dependency or command needs correction, make the edit in
source, append it to the manual ledger, rebuild to a new digest, and restart the
two-clean-replay count with new attempt numbers. Never patch a running
container.

### Seal and publish each direct replay

The host wrapper makes `replay.log` and an exact three-key `timing.json` part of
the smoke output. Seal each otherwise complete root with the reviewed
control-plane checkout. The evaluator and final evaluator image use
`e30480a...`, which includes the explicit policy-client WebSocket close needed
after the observed suite-shutdown hang. The parent policy image and converted
teacher remain separately bound to `229c08e...`; neither identity substitutes
for the other. Local `validate` and an `upload` without `--execute` make no AWS
API calls or mutations, but both deliberately make a fresh IMDSv2 identity
read on the executing workbench. The executing
form snapshots the eight exact payloads, publishes a deterministic claim first,
uses conditional AES256/SHA256 uploads with version-specific round trips, writes
the publication receipt, and writes the manifest last. Its final gate requires
exactly eleven sole/latest object versions, no delete markers, and no incomplete
multipart uploads. The exact `worker_artifact` emitted by the retained
converted-teacher publication is tracked at
`repro/libero-teacher-pytorch.worker-artifact.json`; do not reconstruct or copy
it into an untracked host path. The publisher hashes the three local checkpoint
files against the converted manifest and requires their hashes and every S3
VersionId to match that tracked object. It also requires a locally hashed
exact-version cost-ledger copy with a paid entry covering the instance,
complete wrapper interval, and projected cost. Before writing the publication
claim, `upload --execute` downloads the teacher manifest, all three teacher
objects, and the ledger again by exact S3 VersionId and reproduces every pinned
SHA-256.

```bash
set -euo pipefail
export AWS_REGION=us-east-2
export AWS_DEFAULT_REGION=us-east-2
: "${PI05_CONTROL_COMMIT:?set the reviewed commit containing the LIBERO evidence publisher}"
printf '%s\n' "$PI05_CONTROL_COMMIT" | grep -Eq '^[0-9a-f]{40}$'
export PI05_CONTROL_CHECKOUT="/opt/pi05/source/openpi-$PI05_CONTROL_COMMIT"
export PI05_EVALUATOR_SOURCE_COMMIT=e30480a6de404c74a996863c4fde89367350cf70
export PI05_PARENT_POLICY_SOURCE_COMMIT=229c08ea2a13a70cbbf1a9c8a1f31cb1ca674dee
export PI05_LIBERO_IMAGE='752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro@sha256:51b352c1a7205d6bdae668f99060ebd05049042e1d89916993830acbdc63b374'
export PI05_PARENT_POLICY_IMAGE_DIGEST=sha256:d76e6d73fca409e998304a6a8997f80fab1252fe0301d667a072f99dd6624f24
export PI05_CHECKPOINT=/opt/pi05/checkpoints/pi05_libero_pytorch
export PI05_CONVERTED_MANIFEST=/opt/pi05/checkpoints/_manifests/pi05_libero_pytorch.converted-manifest.json
export PI05_MODEL_REVISION=c73bb6ff5cbaa3c7bba5f03ea38c22bd95e8274308285e2f17b6ed2d73688dd0
export PI05_COST_LEDGER_VERSION_ID=WwdchX.Da46XNc5.cVFjkU7.qqryrA7h
export PI05_COST_LEDGER_SHA256=13eb67119d0261a58f52a2b1633e125b2ff8e47214095c70807f96e88c316db9
export PI05_EVIDENCE_S3_ROOT='s3://pi05-repro-752160877725-us-east-2/manual-smoke/libero'
export PI05_COST_LEDGER_S3_URI='s3://pi05-repro-752160877725-us-east-2/control/cost-ledger.json'
export PI05_CONVERTED_CHECKPOINT_ARTIFACT="$PI05_CONTROL_CHECKOUT/repro/libero-teacher-pytorch.worker-artifact.json"
test "$(git -C "$PI05_CONTROL_CHECKOUT" rev-parse HEAD)" = "$PI05_CONTROL_COMMIT"
test -z "$(git -C "$PI05_CONTROL_CHECKOUT" status --porcelain)"
test -f "$PI05_CONVERTED_CHECKPOINT_ARTIFACT"
test -d "$PI05_CHECKPOINT"
test -f "$PI05_CONVERTED_MANIFEST"

imds_token="$(curl --fail --silent --show-error --request PUT \
  --header 'X-aws-ec2-metadata-token-ttl-seconds: 60' \
  http://169.254.169.254/latest/api/token)"
instance_identity="$(curl --fail --silent --show-error \
  --header "X-aws-ec2-metadata-token: $imds_token" \
  http://169.254.169.254/latest/dynamic/instance-identity/document)"
unset imds_token
test "$(jq -r .accountId <<<"$instance_identity")" = 752160877725
test "$(jq -r .region <<<"$instance_identity")" = us-east-2
test "$(jq -r .instanceType <<<"$instance_identity")" = g6e.4xlarge
export PI05_INSTANCE_ID="$(jq -r .instanceId <<<"$instance_identity")"
printf '%s\n' "$PI05_INSTANCE_ID" | grep -Eq '^i-[0-9a-f]{17}$'

install -d -m 0700 /opt/pi05/evidence-inputs
install -d -m 0700 /opt/pi05/evidence-publications

export PI05_COST_LEDGER_PATH=/opt/pi05/evidence-inputs/cost-ledger.exact-version.json
if test -e "$PI05_COST_LEDGER_PATH"; then
  test -f "$PI05_COST_LEDGER_PATH"
  test ! -L "$PI05_COST_LEDGER_PATH"
else
  ledger_tmp="$(mktemp /opt/pi05/evidence-inputs/.cost-ledger.XXXXXX)"
  set +e
  ledger_get_receipt="$(aws s3api get-object \
    --bucket pi05-repro-752160877725-us-east-2 \
    --key control/cost-ledger.json \
    --version-id "$PI05_COST_LEDGER_VERSION_ID" \
    --checksum-mode ENABLED \
    --expected-bucket-owner 752160877725 \
    --region us-east-2 --output json "$ledger_tmp")"
  ledger_get_status=$?
  set -e
  if test "$ledger_get_status" -ne 0; then
    rm -f "$ledger_tmp"
    exit "$ledger_get_status"
  fi
  test "$(jq -r .VersionId <<<"$ledger_get_receipt")" = "$PI05_COST_LEDGER_VERSION_ID"
  test "$(sha256sum "$ledger_tmp" | cut -d ' ' -f1)" = "$PI05_COST_LEDGER_SHA256"
  chmod 0400 "$ledger_tmp"
  ln "$ledger_tmp" "$PI05_COST_LEDGER_PATH"
  rm "$ledger_tmp"
fi
test "$(sha256sum "$PI05_COST_LEDGER_PATH" | cut -d ' ' -f1)" = "$PI05_COST_LEDGER_SHA256"

for attempt in 05 06; do
  run_id="libero-base-runtime-smoke-$attempt"
  output="/opt/pi05/evidence/$run_id"
  common_args=(
    --output-root "$output"
    --run-id "$run_id"
    --evaluator-source-commit "$PI05_EVALUATOR_SOURCE_COMMIT"
    --evaluator-image-digest "${PI05_LIBERO_IMAGE##*@}"
    --parent-policy-source-commit "$PI05_PARENT_POLICY_SOURCE_COMMIT"
    --parent-policy-image-digest "$PI05_PARENT_POLICY_IMAGE_DIGEST"
    --model-revision "$PI05_MODEL_REVISION"
    --instance-id "$PI05_INSTANCE_ID"
    --checkpoint-root "$PI05_CHECKPOINT"
    --converted-manifest "$PI05_CONVERTED_MANIFEST"
    --converted-checkpoint-artifact "$PI05_CONVERTED_CHECKPOINT_ARTIFACT"
    --cost-ledger-path "$PI05_COST_LEDGER_PATH"
    --cost-ledger-s3-uri "$PI05_COST_LEDGER_S3_URI"
    --cost-ledger-version-id "$PI05_COST_LEDGER_VERSION_ID"
    --cost-ledger-sha256 "$PI05_COST_LEDGER_SHA256"
  )

  python3 "$PI05_CONTROL_CHECKOUT/scripts/repro_stage_libero_evidence.py" \
    validate "${common_args[@]}"
  python3 "$PI05_CONTROL_CHECKOUT/scripts/repro_stage_libero_evidence.py" \
    upload "${common_args[@]}" --s3-root "$PI05_EVIDENCE_S3_ROOT" \
    --config "$PI05_CONTROL_CHECKOUT/repro/reproduction.json"

  publication_receipt="/opt/pi05/evidence-publications/$run_id.json"
  test ! -e "$publication_receipt"
  publication_tmp="$(mktemp "/opt/pi05/evidence-publications/.${run_id}.XXXXXX")"
  set +e
  python3 "$PI05_CONTROL_CHECKOUT/scripts/repro_stage_libero_evidence.py" \
    upload --execute "${common_args[@]}" --s3-root "$PI05_EVIDENCE_S3_ROOT" \
    --config "$PI05_CONTROL_CHECKOUT/repro/reproduction.json" \
    > "$publication_tmp"
  publication_status=$?
  set -e
  if test "$publication_status" -ne 0; then
    rm -f "$publication_tmp"
    exit "$publication_status"
  fi
  jq -e '.s3.manifest.version_id and (.s3.publication.payload | length == 8)' \
    "$publication_tmp"
  chmod 0400 "$publication_tmp"
  sync -f "$publication_tmp"
  ln "$publication_tmp" "$publication_receipt"
  rm "$publication_tmp"
  sync -f /opt/pi05/evidence-publications
done
```

Persist each local publication receipt outside its sealed output root and append
its SHA-256, evidence revision, manifest URI/VersionId/hash, and all eight
payload VersionIds to the manual-edit ledger. A prefix containing unknown
history, multiple versions, or a delete marker is rejected rather than cleaned
or overwritten. An interrupted exact publication can resume only when its
deterministic claim and every existing version-specific byte are identical.

### TensorRT one-trial replay

Do not run a compiled smoke on an arbitrary replacement instance. First build and numerically validate the engines
inside the final TensorRT LIBERO evaluator image on one On-Demand `g7e.4xlarge`. Both `build_tensorrt_engines.py`
and `serve_tensorrt_policy.py` bind the final image digest, instance ID, live G7e GPU UUID/name/driver inventory
(RTX PRO 6000 Blackwell Server Edition on `g7e.4xlarge`), TensorRT major version, source SHA, track, dataset
revision, checkpoint assets, and precision.

For both clean manual replays, use one deadline-bounded retained G7e session. The launcher bootstrap only prepares
the host and returns; it is not a completed ephemeral worker that somehow remains available. From SSM, run every
LIBERO export, validation, FP8, engine-build, latency, serve, and evaluation command directly with the same final
evaluator image digest. Mount a fresh host output root as `/output` for each replay, rebuild the engines in each
root, and do not change instance, image, driver, or GPU between that replay's build and evaluation. Keep the instance
alive through both replays; the guest timer and independent scheduler remain the hard stop.

Each replay must write the complete directory to
`/output/artifacts/tensorrt/libero/fp8`—never under the read-only `/mnt/openpi`
input tree—and publish that directory, not selected `.plan` files, as one
worker-compatible `artifact` output:

```json
{
  "name": "libero_fp8_engines",
  "kind": "artifact",
  "path": "artifacts/tensorrt/libero/fp8",
  "publish_destination": "tensorrt/libero/fp8"
}
```

The directory includes both plans, the timing cache, validation report, source manifests, layer reports, build logs,
and `tensorrt-manifest.fp8.json`. A direct publication command emits the resulting `kind=asset` descriptor and
records it in the retained-session ledger while the host is still running. Do not invoke an ordinary ephemeral
worker for any compiled manual phase, and do not terminate, stop/start, resize, replace, or relaunch the instance
between build and evaluation.

On the control workstation, create the explicit same-instance contract from the retrieved build manifest. The
values must be copied from evidence, not typed from memory:

```bash
export PI05_BUILD_RUN_ID=libero-fp8-build-01
export PI05_BUILD_INSTANCE_ID=i-0123456789abcdef0
export PI05_ENGINE_BUILD_MANIFEST=/retrieved/build/tensorrt-manifest.fp8.json
export PI05_ENGINE_BUILD_MANIFEST_SHA256="$(shasum -a 256 "$PI05_ENGINE_BUILD_MANIFEST" | awk '{print $1}')"

jq -n \
  --arg build_run_id "$PI05_BUILD_RUN_ID" \
  --arg source_commit "$PI05_SOURCE_COMMIT" \
  --arg image_digest "${PI05_LIBERO_IMAGE##*@}" \
  --arg instance_id "$PI05_BUILD_INSTANCE_ID" \
  --arg dataset_revision 'a4336d589d589045d1c56423ffdf3b88a0e19b1f' \
  --arg manifest_sha256 "$PI05_ENGINE_BUILD_MANIFEST_SHA256" \
  '{schema_version:1,kind:"pi05-tensorrt-build-instance",
    execution_constraint:"evaluate-before-exact-build-instance-stop",
    build_run_id:$build_run_id,source_commit:$source_commit,image_digest:$image_digest,
    instance_type:"g7e.4xlarge",instance_id:$instance_id,track:"libero",
    dataset:{name:"physical-intelligence/libero",revision:$dataset_revision},precision:"fp8",
    engine_build_manifest_sha256:$manifest_sha256}' \
  > /tmp/libero-fp8-build-instance.json
```

The evaluator independently hashes the staged build manifest and checks every field above before starting the
server. The strict TensorRT loader then checks the actual GPU UUID/name/driver inventory, so supplying the old
instance ID to a fresh G7e still fails before any rollout.

Run the compiled smoke directly in the final evaluator image. `PI05_INSTANCE_ID` must come from the IMDS query made
on this host, not from the build contract; the evaluator compares the two independent values. Populate the compiled
descriptor variables from the direct publication receipt. The evaluator starts `serve_tensorrt_policy.py` itself,
waits for loopback readiness, runs all four suites, and stops the server:

The retained session is prepaid but not free. Allocate a non-overlapping part
of its reserved `g7e.4xlarge` cost to this evaluation—hourly rate times the
evaluation runtime budget—and record that value in both the manifest and the
same reservation's ledger notes. This is cost attribution within the existing
reservation, not a second reservation or an incremental launch charge.

```bash
export PI05_CHECKPOINT=/absolute/path/to/pi05_libero_l09_snapflow
export PI05_SOURCE_CHECKOUT=/opt/pi05/source/openpi
export PI05_MODEL_REVISION=64_CHARACTER_CHECKPOINT_REVISION
export PI05_COMPILED_ARTIFACT=/absolute/path/to/replay-build/artifacts/tensorrt/libero/fp8
export PI05_COMPILED_REVISION=64_CHARACTER_COMPILED_ARTIFACT_REVISION
export PI05_COMPILED_MANIFEST_S3_URI='s3://pi05-repro-752160877725-us-east-2/runs/BUILD_RUN/manifests/worker-input-compiled.sha256.json'
export PI05_COMPILED_MANIFEST_VERSION_ID=VERSION_ID_FROM_PUBLICATION
export PI05_COMPILED_MANIFEST_SHA256=64_CHARACTER_DESCRIPTOR_MANIFEST_SHA256
export PI05_COMPILED_PAYLOAD_S3_URI='s3://pi05-repro-752160877725-us-east-2/runs/BUILD_RUN/artifacts/tensorrt/libero/fp8/'
export PI05_EVAL_OUTPUT_HOST=/absolute/path/to/fresh-retained-session-evaluation-attempt
: "${PI05_RETAINED_EVAL_PROJECTED_COST_USD:?set the nonzero retained-session ledger allocation}"
python3 -c 'import os; assert float(os.environ["PI05_RETAINED_EVAL_PROJECTED_COST_USD"]) > 0'

test -z "$(git -C "$PI05_SOURCE_CHECKOUT" status --porcelain)"
export PI05_SOURCE_SHA="$(git -C "$PI05_SOURCE_CHECKOUT" rev-parse HEAD)"
test "$PI05_SOURCE_SHA" = "$PI05_SOURCE_COMMIT"
export PI05_IMAGE_DIGEST="${PI05_LIBERO_IMAGE##*@}"
: "${PI05_INSTANCE_ID:?capture from the retained host IMDSv2 identity document}"
: "${PI05_INSTANCE_TYPE:?capture from the same IMDSv2 identity document}"
test "$PI05_INSTANCE_TYPE" = g7e.4xlarge
test "$PI05_INSTANCE_ID" = "$PI05_BUILD_INSTANCE_ID"
test ! -e "$PI05_EVAL_OUTPUT_HOST"
install -d -m 0700 "$PI05_EVAL_OUTPUT_HOST"

docker run --rm --gpus all --network none --ipc host --shm-size 32g \
  --workdir /workspace/openpi \
  --mount type=bind,src="$PI05_SOURCE_CHECKOUT",dst=/workspace/openpi,readonly \
  --mount type=bind,src="$PI05_CHECKPOINT",dst=/mnt/openpi/checkpoints/pi05_libero_l09_snapflow,readonly \
  --mount type=bind,src="$PI05_COMPILED_ARTIFACT",dst=/mnt/openpi/assets/tensorrt/libero/fp8,readonly \
  --mount type=bind,src="$PI05_EVAL_OUTPUT_HOST",dst=/output \
  --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/cache \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONPATH=/workspace/openpi/src:/workspace/openpi \
  --env PI05_SOURCE_SHA --env PI05_IMAGE_DIGEST \
  --env PI05_INSTANCE_ID --env PI05_INSTANCE_TYPE \
  --env PI05_RUN_ID=libero-final-fp8-smoke-01 \
  --env PI05_SEED=7 \
  "$PI05_LIBERO_IMAGE" \
  /opt/modelopt/bin/python scripts/repro_libero_eval.py run \
    --backend tensorrt \
    --policy-config pi05_libero_l09_snapflow \
    --checkpoint /mnt/openpi/checkpoints/pi05_libero_l09_snapflow \
    --model-revision "$PI05_MODEL_REVISION" \
    --compiled-artifact-dir /mnt/openpi/assets/tensorrt/libero/fp8 \
    --compiled-artifact-revision "$PI05_COMPILED_REVISION" \
    --compiled-manifest-s3-uri "$PI05_COMPILED_MANIFEST_S3_URI" \
    --compiled-manifest-version-id "$PI05_COMPILED_MANIFEST_VERSION_ID" \
    --compiled-manifest-sha256 "$PI05_COMPILED_MANIFEST_SHA256" \
    --compiled-payload-s3-uri "$PI05_COMPILED_PAYLOAD_S3_URI" \
    --precision fp8 \
    --dataset physical-intelligence/libero \
    --dataset-revision a4336d589d589045d1c56423ffdf3b88a0e19b1f \
    --build-instance-id "$PI05_BUILD_INSTANCE_ID" \
    --build-run-id "$PI05_BUILD_RUN_ID" \
    --engine-build-manifest-sha256 "$PI05_ENGINE_BUILD_MANIFEST_SHA256" \
    --stage final --trials-per-task 1 --seed 7 \
    --instance-type g7e.4xlarge \
    --projected-cost-usd "$PI05_RETAINED_EVAL_PROJECTED_COST_USD" \
    --output-root /output
```

Use a distinct build output root, `PI05_EVAL_OUTPUT_HOST`, build run ID, evaluation run ID, and artifact publication
prefix for replay 2. Do not copy replay 1's engines into replay 2. Both end-to-end replays remain on the same
still-running G7e; if that instance stops, the clean-replay count restarts on a replacement with newly built engines.

Every stage retry also gets a never-before-created output directory. Stage manifests and artifact records are
create-once, written atomically, flushed with `fsync`, and refuse overwrite. If a command fails, retain its entire
attempt directory and ledger row as evidence; do not edit, delete, or resume it. Fix the source/runbook, rebuild the
digest if needed, and rerun in a new `attempt-N` directory. A successful immutable engine directory may be mounted
read-only into a fresh evaluation attempt, but a failed or partially published directory is never reused.

## 3. Eager worker spec and future compiled placement contract

The checkpoint staging/publishing tools emit a `worker_artifact` JSON object. Save that exact object for the selected
base or candidate checkpoint. Render the eager worker spec locally; this command has no AWS side effects and calls
the same validator as `scripts/repro_worker.py`.

```bash
python scripts/repro_libero_eval.py render-worker-spec \
  --run-id libero-base-intermediate-01 \
  --controller-source-s3-uri s3://pi05-repro-752160877725-us-east-2/source/openpi-CONTROLLER_GIT_COMMIT-complete.bundle \
  --controller-source-version-id CONTROLLER_SOURCE_VERSION_ID \
  --controller-source-sha256 CONTROLLER_SOURCE_BUNDLE_SHA256 \
  --controller-source-commit "$PI05_CONTROLLER_COMMIT" \
  --source-s3-uri s3://pi05-repro-752160877725-us-east-2/source/openpi-SOURCE_GIT_COMMIT-complete.bundle \
  --source-version-id SOURCE_VERSION_ID \
  --source-sha256 SOURCE_BUNDLE_SHA256 \
  --source-commit "$PI05_SOURCE_COMMIT" \
  --image-uri "$PI05_LIBERO_IMAGE" \
  --parent-policy-image "$PI05_POLICY_BASE_IMAGE" \
  --backend eager \
  --checkpoint-artifact /absolute/path/to/base-worker-artifact.json \
  --policy-config pi05_libero \
  --stage base --trials-per-task 10 --seed 7 \
  --instance-type g6e.4xlarge --projected-cost-usd 25 \
  --output /tmp/libero-base-intermediate-01.json

python scripts/repro_worker.py run --spec /tmp/libero-base-intermediate-01.json
```

The rendered spec binds the host controller and container model source as two
independently versioned complete-history bundles. It uses one On-Demand
`g6e.4xlarge`, one digest-pinned image with purpose `libero-evaluator`, one
immutable checkpoint input, and six expected outputs: four suite JSONL files,
one combined JSONL file, and one evaluation manifest. `repro_worker.py` adds
`--network none`, verifies controller/model/image/input identities, uploads only
complete expected outputs, and writes the authoritative instance identity,
command, receipts, timing, and final cost evidence to its run manifest.

Review the rendered spec and normal worker bootstrap dry-run before using the separately gated `--execute` workflow
in `repro/WORKER_RUNBOOK.md`. The evaluator renderer itself never creates or terminates AWS capacity.

For the compiled candidate, the renderer can preserve both immutable input descriptors and the same-instance
contract as a machine-validated future orchestration artifact:

```bash
python scripts/repro_libero_eval.py render-worker-spec \
  --run-id libero-final-fp8-official-01 \
  --controller-source-s3-uri s3://pi05-repro-752160877725-us-east-2/source/openpi-CONTROLLER_GIT_COMMIT-complete.bundle \
  --controller-source-version-id CONTROLLER_SOURCE_VERSION_ID \
  --controller-source-sha256 CONTROLLER_SOURCE_BUNDLE_SHA256 \
  --controller-source-commit "$PI05_CONTROLLER_COMMIT" \
  --source-s3-uri s3://pi05-repro-752160877725-us-east-2/source/openpi-SOURCE_GIT_COMMIT-complete.bundle \
  --source-version-id SOURCE_VERSION_ID \
  --source-sha256 SOURCE_BUNDLE_SHA256 \
  --source-commit "$PI05_SOURCE_COMMIT" \
  --image-uri "$PI05_LIBERO_IMAGE" \
  --parent-policy-image "$PI05_POLICY_BASE_IMAGE" \
  --parent-tensorrt-compiler-image "$PI05_TENSORRT_COMPILER_IMAGE" \
  --parent-tensorrt-compiler-source-revision "$PI05_SOURCE_COMMIT" \
  --backend tensorrt \
  --checkpoint-artifact /absolute/path/to/snapflow-checkpoint-worker-artifact.json \
  --compiled-artifact /absolute/path/to/libero-fp8-engines-worker-artifact.json \
  --build-instance-contract /tmp/libero-fp8-build-instance.json \
  --policy-config pi05_libero_l09_snapflow \
  --precision fp8 \
  --dataset physical-intelligence/libero \
  --dataset-revision a4336d589d589045d1c56423ffdf3b88a0e19b1f \
  --stage final --trials-per-task 50 --seed 7 \
  --instance-type g7e.4xlarge --projected-cost-usd 25 \
  --output /tmp/libero-final-fp8-official-01.json
```

The rendered TensorRT JSON is **validation-only, non-launchable** today. Do not pass it to `scripts/repro_worker.py --execute`
or to the normal EC2 launcher. No replay-tested dispatcher currently attaches a
worker lifecycle to an already running retained instance. Manual replays must use the direct command in section 2.
After replay-driven exact-instance orchestration exists, this contract is the intended input to that dispatcher.

The rendered image contract includes `policy_backend=tensorrt`, the final
evaluator digest, combined parent digest, compiler digest/source revision, and
all six pinned TensorRT/CUDA/ModelOpt/Torch/ONNX/ONNX Runtime versions. The
future dispatcher must verify the inherited OCI labels before starting the container.

The future contract carries `placement.mode=exact-existing-instance`; that is an execution constraint, not a
fresh-capacity launch request. A future dispatcher must compare IMDS `instanceId` with the placement, inject the
independently observed value as `PI05_INSTANCE_ID`, stage the complete immutable artifact tree and checkpoint, and
invoke `/opt/modelopt/bin/python`. The evaluator compares the observed ID again with `--build-instance-id` before
starting the unchanged WebSocket server. If the retained session stops, discard the placement-specific engines and
rebuild them on the replacement instance. There is no valid override.

## 4. Intermediate and official paired runs

Use a fresh server/run ID for every stage and the same seed, suite order, and trial count for base and candidate.
The fixed pair identity is `(suite, task_id, init_index, seed)` and does not include the stage.

| Evaluation | `--trials-per-task` | Episodes per stage | Paired comparisons |
|---|---:|---:|---:|
| Runtime smoke | 1 | 40 | 40 |
| Intermediate | 10 | 400 | 400 |
| Official final | 50 | 2,000 | 2,000 |

Render and run an eager base spec on `g6e.4xlarge`. Run the compiled candidate directly on the same still-running `g7e.4xlarge`
that built its engines; optionally render its non-launchable future contract as additional identity
evidence. Preserve source SHA and seed, and record both image digests as stage identities. After retrieving the
eager worker output and the direct retained-session output, combine the two `episodes.jsonl` files and run the
existing quality gate:

```bash
cp /retrieved/base/episodes.jsonl /tmp/libero-base.jsonl
cp /retrieved/final/episodes.jsonl /tmp/libero-final.jsonl
sed -n '1,$p' /tmp/libero-base.jsonl /tmp/libero-final.jsonl > /tmp/libero-paired.jsonl

# Set this only to the reviewed health floor recorded before candidate
# evaluation (for example, from the accepted released-teacher baseline). It is
# deliberately required rather than silently defaulted by the gate.
: "${PI05_LIBERO_BASELINE_FLOOR:?set the pre-registered aggregate base-success floor}"
python - <<'PY'
import os

value = float(os.environ["PI05_LIBERO_BASELINE_FLOOR"])
assert 0.0 < value <= 1.0
PY

python scripts/repro_quality_report.py /tmp/libero-paired.jsonl \
  --mode intermediate --expected-pairs 400 \
  --minimum-baseline-success "$PI05_LIBERO_BASELINE_FLOOR" \
  --base-stage base --candidate-stage final \
  --output /tmp/libero-intermediate-report.json
```

For the official run, render a new eager spec and run a fresh direct compiled evaluation with
`--trials-per-task 50`, use `--mode official-final`, and pass the same pre-registered
`--minimum-baseline-success "$PI05_LIBERO_BASELINE_FLOOR"`. That gate requires exactly 2,000 complete
pairs, exactly 500 pairs in each suite, a healthy base, aggregate candidate success within two points of base, and
every suite within three points. A rollout, policy-transport, inference, or schema exception aborts evaluation and
cannot become a failed quality record; missing episodes, mismatched revisions, duplicate pairs, short init-state
sets, modified dependency locks, or partial outputs also fail closed.

## 5. Manual-edit ledger and CloudFormation handoff

For every manual replay, append: source SHA, backend, parent/final image digests, image labels, checkpoint and compiled
artifact descriptor revisions/VersionIds/hashes, engine-build manifest hash, exact build instance ID and GPU
inventory, direct-command hash (plus eager spec hash where applicable), seed, start/finish time, eager worker
run-manifest or compiled retained-session ledger evidence, evaluator manifest hash, quality report, projected/actual
cost, failure, and the precise source edit that fixed it. Rebuild and replay rather than making an unrecorded
in-container change.

Only after two clean one-trial replays and one clean intermediate replay should the image digest contract, IAM/SSM
launch path, output declarations, logging, and replay-driven exact-instance dispatch be transcribed into
CloudFormation. The rendered compiled contract remains non-launchable until that dispatcher has its own clean
manual replay. The CloudFormation replay remains an abbreviated one-trial pilot; it must not repeat full evaluation
or training.
