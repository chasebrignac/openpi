# Compiled TensorRT Artifact Seal and Publication

Run this on the retained `g7e.4xlarge` immediately after the passing TensorRT
build, before writing latency reports or any other files into the artifact
directory. The directory must be the original flat export/build directory. The
sealer rejects symlinks, nested paths, temporary files, empty files, unlisted
extras, missing files, hash changes, dirty-source manifests, or any mismatch in
source, image, track, dataset, precision, EC2 instance, GPU UUID/name, or driver.

Set the values from the retained session and build command, not from memory:

```bash
set -euo pipefail
cd /opt/pi05/source/openpi

export PI05_SOURCE_SHA="$(git rev-parse HEAD)"
test -z "$(git status --porcelain)"
export PI05_IMAGE_DIGEST="${LIBERO_RUNTIME_IMAGE##*@}" # use the matching DROID image for that track
export PI05_INSTANCE_ID="$INSTANCE_ID"                 # value independently read from IMDSv2
export PI05_INSTANCE_TYPE=g7e.4xlarge
export PI05_TRACK=libero
export PI05_PRECISION=fp8
export PI05_DATASET=physical-intelligence/libero
export PI05_DATASET_REVISION=a4336d589d589045d1c56423ffdf3b88a0e19b1f
export PI05_ARTIFACT_DIR=/opt/pi05/manual-replays/replay-01/artifacts/tensorrt/libero/fp8
export PI05_ARTIFACT_BUCKET=pi05-repro-752160877725-us-east-2
export PI05_COMPILED_S3_ROOT="s3://${PI05_ARTIFACT_BUCKET}/compiled"
export PI05_PUBLICATION_RECEIPT=/opt/pi05/manual-replays/replay-01/libero-fp8-publication.json

# This must print the option; an older CLI cannot perform a create-once upload.
AWS_PAGER='' aws s3api put-object help | grep -- '--if-none-match'
```

For DROID, use `PI05_TRACK=droid`, dataset
`allenai/MolmoAct2-DROID-Dataset`, revision
`e44d3138c64cfeb1c24fbbce087b475fb1233728`, and the exact flat artifact
directory `/opt/pi05/manual-replays/replay-01/artifacts/tensorrt/droid/fp8`.

First print the non-reading plan, then run the read-only local validation. The
validator reads live `nvidia-smi` inventory and checks it against the build
manifest. It also requires protected `PI05_SOURCE_SHA`, executing Git HEAD, and
`--source-commit` to be identical and the checkout to remain clean. Neither
command calls AWS or writes a seal file:

```bash
python3 scripts/repro_stage_compiled_artifact.py plan \
  --artifact-dir "$PI05_ARTIFACT_DIR" --track "$PI05_TRACK" --precision "$PI05_PRECISION" \
  --source-commit "$PI05_SOURCE_SHA" --image-digest "$PI05_IMAGE_DIGEST" \
  --dataset "$PI05_DATASET" --dataset-revision "$PI05_DATASET_REVISION" \
  --instance-type "$PI05_INSTANCE_TYPE" --instance-id "$PI05_INSTANCE_ID" \
  --s3-root "$PI05_COMPILED_S3_ROOT"

python3 scripts/repro_stage_compiled_artifact.py validate \
  --artifact-dir "$PI05_ARTIFACT_DIR" --track "$PI05_TRACK" --precision "$PI05_PRECISION" \
  --source-commit "$PI05_SOURCE_SHA" --image-digest "$PI05_IMAGE_DIGEST" \
  --dataset "$PI05_DATASET" --dataset-revision "$PI05_DATASET_REVISION" \
  --instance-type "$PI05_INSTANCE_TYPE" --instance-id "$PI05_INSTANCE_ID"
```

An `upload` command without `--execute` is also a local-only dry run. Review its
content-addressed target before authorizing the only AWS-writing form:

```bash
python3 scripts/repro_stage_compiled_artifact.py upload \
  --artifact-dir "$PI05_ARTIFACT_DIR" --track "$PI05_TRACK" --precision "$PI05_PRECISION" \
  --source-commit "$PI05_SOURCE_SHA" --image-digest "$PI05_IMAGE_DIGEST" \
  --dataset "$PI05_DATASET" --dataset-revision "$PI05_DATASET_REVISION" \
  --instance-type "$PI05_INSTANCE_TYPE" --instance-id "$PI05_INSTANCE_ID" \
  --s3-root "$PI05_COMPILED_S3_ROOT"

python3 scripts/repro_stage_compiled_artifact.py upload --execute \
  --artifact-dir "$PI05_ARTIFACT_DIR" --track "$PI05_TRACK" --precision "$PI05_PRECISION" \
  --source-commit "$PI05_SOURCE_SHA" --image-digest "$PI05_IMAGE_DIGEST" \
  --dataset "$PI05_DATASET" --dataset-revision "$PI05_DATASET_REVISION" \
  --instance-type "$PI05_INSTANCE_TYPE" --instance-id "$PI05_INSTANCE_ID" \
  --s3-root "$PI05_COMPILED_S3_ROOT" \
  | tee "$PI05_PUBLICATION_RECEIPT"
```

The executing form first requires zero object, version, and delete-marker
history beneath the content-addressed prefix. It then uses owner-scoped
`s3api put-object --if-none-match '*'` for a create-once AES256 upload beneath
`TRACK/PRECISION/CONTENT_REVISION/artifact/`. It requires the returned version
to match `head-object`, re-downloads that exact version to verify bytes and
SHA-256, and requires the final listing to contain exactly one version per
expected key and no delete markers. It also fails if the local directory
changes during upload. The clean protected source check is repeated immediately
before this AWS preflight, so changing the checkout after local validation
cannot publish the earlier in-memory seal.

Use only the emitted descriptor when filling `repro_libero_eval` or a DROID
evaluation worker spec:

```bash
jq -e '.worker_artifact.kind == "asset"' "$PI05_PUBLICATION_RECEIPT"
jq '.worker_artifact' "$PI05_PUBLICATION_RECEIPT" > /tmp/compiled-worker-artifact.json
```

Append the command, receipt SHA-256, content revision, manifest version ID, and
object-version receipts to the retained-session manual ledger. Do not retry a
partially completed publication or delete its history: the prefix is poisoned
by design. A failed check requires a fresh artifact directory and rebuild; do
not delete or patch files to make an existing directory pass.
