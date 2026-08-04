#!/bin/bash
# Fetch and verify an exact worker spec and source bundle, then hand off to the
# repository-owned Python worker. This file itself must be fetched by S3
# VersionId and SHA-256 by the launcher command rendered by repro_worker.py.
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

readarray -t source_pin < <(python3 - "${spec_path}" <<'PY'
import json
import sys
import urllib.parse

with open(sys.argv[1], encoding="utf-8") as stream:
    spec = json.load(stream)
source = spec["source"]
parsed = urllib.parse.urlsplit(source["s3_uri"])
key = parsed.path.lstrip("/")
if parsed.scheme != "s3" or not parsed.netloc or not key or parsed.query or parsed.fragment:
    raise SystemExit("invalid source bundle S3 URI")
for value in (parsed.netloc, key, source["version_id"], source["sha256"], source["commit"]):
    if not isinstance(value, str) or not value or "\n" in value or "\r" in value:
        raise SystemExit("invalid source pin")
    print(value)
PY
)
if test "${#source_pin[@]}" -ne 5; then
  echo "failed to resolve source bundle pin" >&2
  exit 2
fi
source_bundle=/opt/pi05/openpi.bundle
aws s3api get-object \
  --bucket "${source_pin[0]}" \
  --key "${source_pin[1]}" \
  --version-id "${source_pin[2]}" \
  --expected-bucket-owner "${EXPECTED_ACCOUNT_ID}" \
  --region "${EXPECTED_AWS_REGION}" \
  "${source_bundle}" >/dev/null
printf '%s  %s\n' "${source_pin[3]}" "${source_bundle}" | sha256sum --check --status
verify_repo=/opt/pi05/bundle-verify.git
if test -e "${verify_repo}"; then
  echo "bundle verification repository already exists; refusing stale reuse" >&2
  exit 2
fi
git init --bare "${verify_repo}" >/dev/null
git -C "${verify_repo}" bundle verify "${source_bundle}" >/dev/null
bundle_head=$(git bundle list-heads "${source_bundle}" HEAD | awk '$2 == "HEAD" {print $1}')
if test "${bundle_head}" != "${source_pin[4]}"; then
  echo "source bundle HEAD mismatch" >&2
  exit 2
fi

checkout=/opt/pi05/repo
if test -e "${checkout}"; then
  echo "checkout path already exists; refusing destructive reuse" >&2
  exit 2
fi
git clone --no-checkout "${source_bundle}" "${checkout}" >/dev/null
git -C "${checkout}" cat-file -e "${source_pin[4]}^{commit}"
git -C "${checkout}" checkout --detach "${source_pin[4]}" >/dev/null
head_commit=$(git -C "${checkout}" rev-parse HEAD)
if test "${head_commit}" != "${source_pin[4]}"; then
  echo "checked-out source commit mismatch" >&2
  exit 2
fi
if test -n "$(git -C "${checkout}" status --porcelain=v1 --untracked-files=all)"; then
  echo "checked-out source is not clean" >&2
  exit 2
fi

evidence=/opt/pi05/source-verification.json
python3 - "${evidence}" "${WORKER_SPEC_S3_URI}" "${WORKER_SPEC_VERSION_ID}" "${WORKER_SPEC_SHA256}" \
  "${source_pin[0]}" "${source_pin[1]}" "${source_pin[2]}" "${source_pin[3]}" "${source_pin[4]}" \
  "${source_bundle}" "${checkout}" <<'PY'
import json
import sys

(
    output,
    spec_uri,
    spec_version,
    spec_sha,
    bucket,
    key,
    source_version,
    source_sha,
    commit,
    bundle,
    checkout,
) = sys.argv[1:]
value = {
    "schema_version": 1,
    "worker_spec": {"s3_uri": spec_uri, "version_id": spec_version, "sha256": spec_sha},
    "source": {
        "s3_uri": f"s3://{bucket}/{key}",
        "version_id": source_version,
        "sha256": source_sha,
        "commit": commit,
    },
    "bundle_sha256_actual": source_sha,
    "head_commit": commit,
    "source_clean": True,
    "bundle_path": bundle,
    "checkout_path": checkout,
}
temporary = output + ".tmp"
with open(temporary, "w", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")
import os
os.replace(temporary, output)
PY

# The bootstrap's umask deliberately keeps specs, hashes, and launch controls
# private. Expose only the verified checkout to the UID-1000 container, and
# make that checkout non-writable before Docker ever sees it.
python3 "${checkout}/scripts/repro_checkout_permissions.py" \
  --checkout "${checkout}" \
  --control-root /opt/pi05 \
  --control-file "${spec_path}" \
  --control-file "${source_bundle}" \
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
exec python3 "${checkout}/scripts/repro_worker.py" "${worker_args[@]}"
