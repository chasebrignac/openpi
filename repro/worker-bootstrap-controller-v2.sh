#!/bin/bash
# Fetch and independently verify the model and host-controller source bundles,
# then execute the controller while exposing only the model checkout to Docker.
# This bootstrap must be published under a key qualified by controller commit,
# fetched by exact S3 VersionId, and checked by SHA-256 by the launch command.
set -euo pipefail
umask 077

required=(
  EXPECTED_ACCOUNT_ID
  EXPECTED_AWS_REGION
  WORKER_SPEC_S3_URI
  WORKER_SPEC_VERSION_ID
  WORKER_SPEC_SHA256
  WORKER_EXECUTE
)
for name in "${required[@]}"; do
  if test -z "${!name:-}"; then
    echo "missing required bootstrap variable: ${name}" >&2
    exit 2
  fi
done
if test "${EXPECTED_ACCOUNT_ID}" != "752160877725" || test "${EXPECTED_AWS_REGION}" != "us-east-2"; then
  echo "bootstrap account/region boundary mismatch" >&2
  exit 2
fi
if test "$(aws sts get-caller-identity --region "${EXPECTED_AWS_REGION}" --query Account --output text)" != "${EXPECTED_ACCOUNT_ID}"; then
  echo "bootstrap refuses the active AWS account" >&2
  exit 2
fi
for command in aws git python3 sha256sum; do
  command -v "${command}" >/dev/null
done

install -d -m 0700 /opt/pi05

readarray -t spec_location < <(python3 - "${WORKER_SPEC_S3_URI}" <<'PY'
import sys
import urllib.parse

parsed = urllib.parse.urlsplit(sys.argv[1])
key = parsed.path.lstrip("/")
if parsed.scheme != "s3" or not parsed.netloc or not key or parsed.query or parsed.fragment:
    raise SystemExit("invalid worker spec S3 URI")
print(parsed.netloc)
print(key)
PY
)
if test "${#spec_location[@]}" -ne 2; then
  echo "failed to resolve worker spec S3 URI" >&2
  exit 2
fi
spec_path=/opt/pi05/worker-spec.json
aws s3api get-object \
  --bucket "${spec_location[0]}" \
  --key "${spec_location[1]}" \
  --version-id "${WORKER_SPEC_VERSION_ID}" \
  --expected-bucket-owner "${EXPECTED_ACCOUNT_ID}" \
  --region "${EXPECTED_AWS_REGION}" \
  "${spec_path}" >/dev/null
printf '%s  %s\n' "${WORKER_SPEC_SHA256}" "${spec_path}" | sha256sum --check --status

read_source_pin() {
  local field="$1"
  python3 - "${spec_path}" "${field}" <<'PY'
import json
import sys
import urllib.parse

with open(sys.argv[1], encoding="utf-8") as stream:
    spec = json.load(stream)
source = spec[sys.argv[2]]
if set(source) != {"s3_uri", "version_id", "sha256", "commit"}:
    raise SystemExit(f"invalid {sys.argv[2]} schema")
parsed = urllib.parse.urlsplit(source["s3_uri"])
key = parsed.path.lstrip("/")
if parsed.scheme != "s3" or not parsed.netloc or not key or parsed.query or parsed.fragment:
    raise SystemExit(f"invalid {sys.argv[2]} bundle S3 URI")
for value in (parsed.netloc, key, source["version_id"], source["sha256"], source["commit"]):
    if not isinstance(value, str) or not value or "\n" in value or "\r" in value:
        raise SystemExit(f"invalid {sys.argv[2]} pin")
    print(value)
PY
}

readarray -t source_pin < <(read_source_pin source)
readarray -t controller_pin < <(read_source_pin controller_source)
if test "${#source_pin[@]}" -ne 5 || test "${#controller_pin[@]}" -ne 5; then
  echo "failed to resolve both source bundle pins" >&2
  exit 2
fi

materialize_source() {
  local label="$1"
  local bucket="$2"
  local key="$3"
  local version_id="$4"
  local expected_sha256="$5"
  local expected_commit="$6"
  local bundle="$7"
  local verify_repo="$8"
  local checkout="$9"

  for path in "${bundle}" "${verify_repo}" "${checkout}"; do
    if test -e "${path}"; then
      echo "${label} path already exists; refusing stale reuse: ${path}" >&2
      return 2
    fi
  done
  aws s3api get-object \
    --bucket "${bucket}" --key "${key}" --version-id "${version_id}" \
    --expected-bucket-owner "${EXPECTED_ACCOUNT_ID}" --region "${EXPECTED_AWS_REGION}" \
    "${bundle}" >/dev/null
  printf '%s  %s\n' "${expected_sha256}" "${bundle}" | sha256sum --check --status
  git init --bare "${verify_repo}" >/dev/null
  git -C "${verify_repo}" bundle verify "${bundle}" >/dev/null
  local bundle_head
  bundle_head=$(git bundle list-heads "${bundle}" HEAD | awk '$2 == "HEAD" {print $1}')
  if test "${bundle_head}" != "${expected_commit}"; then
    echo "${label} bundle HEAD mismatch" >&2
    return 2
  fi
  git clone --no-checkout "${bundle}" "${checkout}" >/dev/null
  git -C "${checkout}" cat-file -e "${expected_commit}^{commit}"
  git -C "${checkout}" checkout --detach "${expected_commit}" >/dev/null
  if test "$(git -C "${checkout}" rev-parse HEAD)" != "${expected_commit}"; then
    echo "${label} checked-out commit mismatch" >&2
    return 2
  fi
  if test -n "$(git -C "${checkout}" status --porcelain=v1 --untracked-files=all)"; then
    echo "${label} checkout is not clean" >&2
    return 2
  fi
  # git bundle verify alone accepts a bundle cut from a shallow repository.
  # A fresh clone can still contain a broken parent edge, so require the full
  # object graph to be self-contained before this source can control a run.
  git -C "${checkout}" fsck --full --no-dangling >/dev/null
}

source_bundle=/opt/pi05/model-source.bundle
source_verify_repo=/opt/pi05/model-source-verify.git
source_checkout=/opt/pi05/model-source
controller_bundle=/opt/pi05/controller-source.bundle
controller_verify_repo=/opt/pi05/controller-source-verify.git
controller_checkout=/opt/pi05/controller-source
materialize_source model \
  "${source_pin[0]}" "${source_pin[1]}" "${source_pin[2]}" "${source_pin[3]}" "${source_pin[4]}" \
  "${source_bundle}" "${source_verify_repo}" "${source_checkout}"
materialize_source controller \
  "${controller_pin[0]}" "${controller_pin[1]}" "${controller_pin[2]}" "${controller_pin[3]}" \
  "${controller_pin[4]}" "${controller_bundle}" "${controller_verify_repo}" "${controller_checkout}"

for controller_file in scripts/repro_worker.py scripts/repro_checkout_permissions.py; do
  if test ! -f "${controller_checkout}/${controller_file}" || test -L "${controller_checkout}/${controller_file}"; then
    echo "verified controller checkout lacks ${controller_file}" >&2
    exit 2
  fi
done

evidence=/opt/pi05/source-verification.json
python3 - "${evidence}" "${WORKER_SPEC_S3_URI}" "${WORKER_SPEC_VERSION_ID}" "${WORKER_SPEC_SHA256}" \
  "${source_pin[0]}" "${source_pin[1]}" "${source_pin[2]}" "${source_pin[3]}" "${source_pin[4]}" \
  "${controller_pin[0]}" "${controller_pin[1]}" "${controller_pin[2]}" "${controller_pin[3]}" \
  "${controller_pin[4]}" "${source_bundle}" "${source_checkout}" "${controller_bundle}" \
  "${controller_checkout}" <<'PY'
import json
import os
import sys

(
    output,
    spec_uri,
    spec_version,
    spec_sha,
    source_bucket,
    source_key,
    source_version,
    source_sha,
    source_commit,
    controller_bucket,
    controller_key,
    controller_version,
    controller_sha,
    controller_commit,
    source_bundle,
    source_checkout,
    controller_bundle,
    controller_checkout,
) = sys.argv[1:]
value = {
    "schema_version": 2,
    "worker_spec": {"s3_uri": spec_uri, "version_id": spec_version, "sha256": spec_sha},
    "source": {
        "s3_uri": f"s3://{source_bucket}/{source_key}",
        "version_id": source_version,
        "sha256": source_sha,
        "commit": source_commit,
    },
    "controller_source": {
        "s3_uri": f"s3://{controller_bucket}/{controller_key}",
        "version_id": controller_version,
        "sha256": controller_sha,
        "commit": controller_commit,
    },
    "bundle_sha256_actual": source_sha,
    "head_commit": source_commit,
    "source_clean": True,
    "source_fsck_full": True,
    "bundle_path": source_bundle,
    "checkout_path": source_checkout,
    "controller_bundle_sha256_actual": controller_sha,
    "controller_head_commit": controller_commit,
    "controller_source_clean": True,
    "controller_source_fsck_full": True,
    "controller_bundle_path": controller_bundle,
    "controller_checkout_path": controller_checkout,
}
temporary = output + ".tmp"
with open(temporary, "w", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")
os.replace(temporary, output)
PY

# Only the independently verified model checkout crosses the container trust
# boundary. The controller checkout remains behind root-only /opt/pi05.
python3 "${controller_checkout}/scripts/repro_checkout_permissions.py" \
  --checkout "${source_checkout}" \
  --control-root /opt/pi05 \
  --control-file "${spec_path}" \
  --control-file "${source_bundle}" \
  --control-file "${controller_bundle}" \
  --control-file "${evidence}" \
  --control-file /opt/pi05/launch-metadata.json

worker_args=(
  run
  --spec "${spec_path}"
  --source-evidence "${evidence}"
  --launch-metadata /opt/pi05/launch-metadata.json
)
if test "${WORKER_EXECUTE}" = "1"; then
  worker_args+=(--execute)
elif test "${WORKER_EXECUTE}" != "0"; then
  echo "WORKER_EXECUTE must be 0 or 1" >&2
  exit 2
fi
exec python3 "${controller_checkout}/scripts/repro_worker.py" "${worker_args[@]}"
