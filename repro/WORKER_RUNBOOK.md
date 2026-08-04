# Ephemeral AWS worker runbook

This is the manual, fail-closed bootstrap contract for every non-workbench GPU. The launcher remains dry-run unless its separate `--execute` flag is present, and the worker remains dry-run unless the rendered bootstrap was created with `--execute`.

The examples use the unique attempt ID `20260804T120000Z-a1`. Generate a new
sortable ID for every fresh attempt and replace every occurrence consistently.
Experiment names accept only ASCII letters, digits, `.`, `_`, and `-`, may not
contain `..`, and must begin and end with a letter or digit. Fresh jobs omit
`--overwrite`; an accidental retry with an existing ID therefore fails closed.

## 1. Prepare an exact source artifact

Commit the reviewed tree first. A source bundle must contain that commit; never bundle uncommitted files.

```bash
set -euo pipefail
test -z "$(git status --porcelain)"
SOURCE_COMMIT="$(git rev-parse HEAD)"
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
git bundle create /tmp/openpi.bundle HEAD
git bundle verify /tmp/openpi.bundle
test "$(git bundle list-heads /tmp/openpi.bundle HEAD | awk '$2 == "HEAD" {print $1}')" = \
  "$SOURCE_COMMIT"
SOURCE_BUNDLE_SHA256="$(sha256sum /tmp/openpi.bundle | awk '{print $1}')"
SOURCE_BUNDLE_BYTES="$(wc -c </tmp/openpi.bundle | tr -d '[:space:]')"
SOURCE_BUNDLE_KEY="source/openpi-$SOURCE_COMMIT.bundle"
SOURCE_BUNDLE_S3_URI="s3://pi05-repro-752160877725-us-east-2/$SOURCE_BUNDLE_KEY"
SOURCE_HISTORY_JSON="$(
  aws s3api list-object-versions \
    --bucket pi05-repro-752160877725-us-east-2 \
    --prefix "$SOURCE_BUNDLE_KEY" --region us-east-2 \
    --expected-bucket-owner 752160877725 --output json
)"
test "$(jq --arg key "$SOURCE_BUNDLE_KEY" \
  '[.Versions[]? | select(.Key == $key)] | length' <<<"$SOURCE_HISTORY_JSON")" -eq 0
test "$(jq --arg key "$SOURCE_BUNDLE_KEY" \
  '[.DeleteMarkers[]? | select(.Key == $key)] | length' <<<"$SOURCE_HISTORY_JSON")" -eq 0
SOURCE_PUT_JSON="$(
  aws s3api put-object --bucket pi05-repro-752160877725-us-east-2 \
    --key "$SOURCE_BUNDLE_KEY" --body /tmp/openpi.bundle --region us-east-2 \
    --expected-bucket-owner 752160877725 --server-side-encryption AES256 \
    --if-none-match '*' \
    --metadata "source-commit=$SOURCE_COMMIT,sha256=$SOURCE_BUNDLE_SHA256" --output json
)"
SOURCE_VERSION_ID="$(jq -er '.VersionId | strings | select(length > 0)' <<<"$SOURCE_PUT_JSON")"
SOURCE_HEAD_JSON="$(
  aws s3api head-object --bucket pi05-repro-752160877725-us-east-2 \
    --key "$SOURCE_BUNDLE_KEY" --version-id "$SOURCE_VERSION_ID" --region us-east-2 \
    --expected-bucket-owner 752160877725 --output json
)"
test "$(jq -r '.VersionId' <<<"$SOURCE_HEAD_JSON")" = "$SOURCE_VERSION_ID"
test "$(jq -r '.ContentLength' <<<"$SOURCE_HEAD_JSON")" = "$SOURCE_BUNDLE_BYTES"
test "$(jq -r '.ServerSideEncryption' <<<"$SOURCE_HEAD_JSON")" = AES256
test "$(jq -r '.Metadata["source-commit"]' <<<"$SOURCE_HEAD_JSON")" = "$SOURCE_COMMIT"
test "$(jq -r '.Metadata.sha256' <<<"$SOURCE_HEAD_JSON")" = "$SOURCE_BUNDLE_SHA256"
SOURCE_FINAL_HISTORY_JSON="$(
  aws s3api list-object-versions \
    --bucket pi05-repro-752160877725-us-east-2 \
    --prefix "$SOURCE_BUNDLE_KEY" --region us-east-2 \
    --expected-bucket-owner 752160877725 --output json
)"
test "$(jq --arg key "$SOURCE_BUNDLE_KEY" --arg version "$SOURCE_VERSION_ID" \
  '[.Versions[]? | select(.Key == $key and .VersionId == $version and .IsLatest == true)] | length' \
  <<<"$SOURCE_FINAL_HISTORY_JSON")" -eq 1
test "$(jq --arg key "$SOURCE_BUNDLE_KEY" \
  '[.Versions[]? | select(.Key == $key)] | length' <<<"$SOURCE_FINAL_HISTORY_JSON")" -eq 1
test "$(jq --arg key "$SOURCE_BUNDLE_KEY" \
  '[.DeleteMarkers[]? | select(.Key == $key)] | length' <<<"$SOURCE_FINAL_HISTORY_JSON")" -eq 0
SOURCE_ROUNDTRIP="$(mktemp /tmp/openpi-bundle-roundtrip.XXXXXX)"
aws s3api get-object --bucket pi05-repro-752160877725-us-east-2 \
  --key "$SOURCE_BUNDLE_KEY" --version-id "$SOURCE_VERSION_ID" --region us-east-2 \
  --expected-bucket-owner 752160877725 "$SOURCE_ROUNDTRIP" >/dev/null
printf '%s  %s\n' "$SOURCE_BUNDLE_SHA256" "$SOURCE_ROUNDTRIP" | sha256sum --check --status
rm -f -- "$SOURCE_ROUNDTRIP"
printf 'source.commit=%s\nsource.s3_uri=%s\nsource.version_id=%s\nsource.sha256=%s\n' \
  "$SOURCE_COMMIT" "$SOURCE_BUNDLE_S3_URI" "$SOURCE_VERSION_ID" "$SOURCE_BUNDLE_SHA256"
```

Copy the four printed values into the worker spec. Never substitute a later
unversioned `head-object` result for the captured `source.version_id`.

## 2. Write and validate a worker spec

Use this shape. Replace every placeholder with a recorded immutable value. A
Shallow worker needs three distinct inputs: the dataset, the original JAX
teacher (for released normalization assets), and the converted PyTorch teacher
(for KD and layer transplantation). Dataset revisions are 40-character Hugging
Face commits. Checkpoint revisions are 64-character content/provenance hashes.
Every manifest itself needs an exact S3 `VersionId` and SHA-256; payload files
are accepted only when their byte sizes and SHA-256 values match that manifest.
Use the `worker_artifact` object printed by
`repro_stage_converted_checkpoints.py upload --execute` for the third entry.
The worker rejects old weak DROID manifests unless their validation records the
`molmoact2-v3-exact-media-references-v1` layout and exactly 518 exterior-left
plus 316 wrist-left MP4 references. Dataset, released-teacher, converted-teacher,
and worker-checkpoint tracks must match the image's LeRobot v2/v3 runtime. A
converted teacher is accepted only alongside the selected original JAX teacher
whose revision, GCS source, and expected config it records. These cross-input
checks complete before Docker starts.

```json
{
  "schema_version": 1,
  "project": "pi05-aws-repro",
  "run_id": "libero-shallow-20260804T120000Z-a1-2k",
  "aws": {
    "account_id": "752160877725",
    "region": "us-east-2",
    "artifact_bucket": "pi05-repro-752160877725-us-east-2"
  },
  "source": {
    "s3_uri": "s3://pi05-repro-752160877725-us-east-2/source/openpi-SOURCE_GIT_COMMIT.bundle",
    "version_id": "SOURCE_VERSION_ID",
    "sha256": "SOURCE_BUNDLE_SHA256",
    "commit": "SOURCE_GIT_COMMIT"
  },
  "image": {
    "uri": "752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro@sha256:IMAGE_MANIFEST_DIGEST",
    "digest": "sha256:IMAGE_MANIFEST_DIGEST",
    "purpose": "policy",
    "lerobot_runtime": "v2",
    "lerobot_revision": "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
  },
  "artifacts": [
    {
      "name": "libero",
      "kind": "dataset",
      "revision": "a4336d589d589045d1c56423ffdf3b88a0e19b1f",
      "manifest": {
        "s3_uri": "s3://pi05-repro-752160877725-us-east-2/datasets/libero/a4336d589d589045d1c56423ffdf3b88a0e19b1f/manifest.sha256.json",
        "version_id": "MANIFEST_VERSION_ID",
        "sha256": "MANIFEST_SHA256"
      },
      "payload_s3_uri": "s3://pi05-repro-752160877725-us-east-2/datasets/libero/a4336d589d589045d1c56423ffdf3b88a0e19b1f/snapshot/",
      "destination": "libero"
    },
    {
      "name": "libero_teacher_jax",
      "kind": "checkpoint",
      "revision": "b00d25ec1a1284656ccfd0cf00597fced40fa20c9c7c39ebfdf256db6e844fb7",
      "manifest": {
        "s3_uri": "s3://pi05-repro-752160877725-us-east-2/checkpoints/pi05_libero/b00d25ec1a1284656ccfd0cf00597fced40fa20c9c7c39ebfdf256db6e844fb7/manifest.sha256.json",
        "version_id": "oX5OL_hTQDoYmYD7bTZ.sM7.4KxB5FX3",
        "sha256": "9140fa118b1a2b627726519cb3d21a0a98f2b1b736b5909a49520fc75d8dd8ad"
      },
      "payload_s3_uri": "s3://pi05-repro-752160877725-us-east-2/checkpoints/pi05_libero/b00d25ec1a1284656ccfd0cf00597fced40fa20c9c7c39ebfdf256db6e844fb7/checkpoint/",
      "destination": "pi05_libero"
    },
    {
      "name": "libero_teacher_pytorch",
      "kind": "checkpoint",
      "revision": "CONVERTED_64_CHARACTER_REVISION",
      "manifest": {
        "s3_uri": "s3://pi05-repro-752160877725-us-east-2/checkpoints/pi05_libero_pytorch/CONVERTED_64_CHARACTER_REVISION/manifest.sha256.json",
        "version_id": "CONVERTED_MANIFEST_VERSION_ID",
        "sha256": "CONVERTED_MANIFEST_SHA256"
      },
      "payload_s3_uri": "s3://pi05-repro-752160877725-us-east-2/checkpoints/pi05_libero_pytorch/CONVERTED_64_CHARACTER_REVISION/checkpoint/",
      "destination": "pi05_libero_pytorch"
    }
  ],
  "container": {
    "command": [
      "torchrun", "--standalone", "--nproc-per-node=2",
      "scripts/train_pytorch.py", "pi05_libero_l09_distill",
      "--exp-name", "libero-shallow-20260804T120000Z-a1",
      "--checkpoint-base-dir", "/mnt/openpi/runs",
      "--teacher-pytorch-weight-path", "/mnt/openpi/checkpoints/pi05_libero_pytorch",
      "--seed", "42",
      "--num-train-steps", "2000", "--save-interval", "2000"
    ],
    "environment": {"WANDB_MODE": "offline"},
    "shm_size_gib": 64
  },
  "expected_outputs": [
    {
      "name": "pilot_checkpoint",
      "kind": "checkpoint",
      "path": "checkpoints/pi05_libero_l09_distill/libero-shallow-20260804T120000Z-a1/2000",
      "publish_destination": "pi05_libero_l09_distill/libero-shallow-20260804T120000Z-a1/2000"
    }
  ],
  "output": {
    "s3_uri": "s3://pi05-repro-752160877725-us-east-2/runs/libero-shallow-20260804T120000Z-a1-2k/"
  },
  "timing": {
    "sync_interval_seconds": 60,
    "upload_buffer_seconds": 900,
    "stop_grace_seconds": 30
  },
  "scratch": {
    "model": "Amazon EC2 NVMe Instance Storage",
    "expected_count": 1,
    "ordinal": 0,
    "mount": "/mnt/openpi",
    "filesystem_label": "PI05_SCRATCH"
  },
  "seed": 42
}
```

`image.purpose` is mandatory; the worker rejects older ambiguous specs. Use
`policy` for training, policy export, dataset/calibration generation, and eager
non-simulator policy evaluation. A policy image must declare the approved
LeRobot v2 or v3 commit, and the pulled image must carry matching
`ai.openpi.image-purpose`, `ai.openpi.lerobot-runtime`, and
`ai.openpi.lerobot-revision` labels.

Closed-loop LIBERO uses a separate, fail-closed evaluator purpose. Its spec must
bind the evaluator image to the exact v2 parent policy image, simulator commit,
and evaluator dependency lock used at build time:

```json
"image": {
  "uri": "752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro@sha256:EVALUATOR_IMAGE_DIGEST",
  "digest": "sha256:EVALUATOR_IMAGE_DIGEST",
  "purpose": "libero-evaluator",
  "policy_backend": "eager",
  "lerobot_runtime": "v2",
  "lerobot_revision": "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5",
  "libero_simulator_revision": "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c",
  "libero_requirements_sha256": "124e74d09719941c9e3e75a61330808a8d32ae35a1ebee00c18e1222e966d0c8",
  "parent_policy_image": "752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro@sha256:PARENT_POLICY_DIGEST"
}
```

The eager schema above is exact: compiler and TensorRT toolchain fields are
rejected. The pulled image must carry those values under
`ai.openpi.policy-backend=eager`,
`ai.openpi.libero-simulator-revision`,
`ai.openpi.libero-requirements-sha256`, and
`ai.openpi.parent-policy-image`, in addition to the exact source, purpose, and
LeRobot labels. The evaluator command must be `scripts/repro_libero_eval.py
run`, must declare `--backend eager --output-root /output`, and its container
environment must include these exact entries:

```json
"environment": {
  "MUJOCO_EGL_DEVICE_ID": "0",
  "MUJOCO_GL": "egl",
  "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
  "PYOPENGL_PLATFORM": "egl"
}
```

The worker rejects a normal v2 policy image for this command and renders both
GPU/EGL identity variables into Docker as literal argv entries.

Container environments may add application settings such as `WANDB_MODE`, but
cannot override worker-owned `PI05_*`, `HOME`, `PATH`, `PYTHONPATH`,
`PYTHONDONTWRITEBYTECODE`, or `XDG_CACHE_HOME` values. AWS and Docker control
variables are likewise rejected.

Graph-only ModelOpt, `trtexec`, TensorRT validation, and engine building use the
compiler image contract below. Compiled policy latency uses the combined image
described afterward. The compiler purpose deliberately has no LeRobot fields or
labels; the worker rejects a compiler image that claims LeRobot or omits any
pinned toolchain component.

```json
"image": {
  "uri": "752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro-tensorrt@sha256:IMAGE_MANIFEST_DIGEST",
  "digest": "sha256:IMAGE_MANIFEST_DIGEST",
  "purpose": "tensorrt-compiler",
  "toolchain": {
    "tensorrt_version": "11.0.0.114",
    "cuda_version": "13.3.0",
    "modelopt_version": "0.45.0",
    "torch_version": "2.8.0",
    "onnx_version": "1.21.0",
    "onnxruntime_gpu_version": "1.24.2"
  }
}
```

Do not mix the three stanzas: evaluator provenance is mandatory only on the
LIBERO evaluator, policy fields are unexpected on a compiler image, and
`toolchain` is unexpected on an eager policy image. Every image remains pinned
by an account-local ECR digest and the source commit label.

Export and serving with the compiled policy stack use a combined image. The v2
and v3 schemas differ only by their exact LeRobot runtime/revision pair:

```json
"image": {
  "uri": "752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro@sha256:COMBINED_DIGEST",
  "digest": "sha256:COMBINED_DIGEST",
  "purpose": "tensorrt-policy",
  "lerobot_runtime": "v2",
  "lerobot_revision": "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5",
  "parent_tensorrt_compiler_image": "752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro@sha256:COMPILER_DIGEST",
  "parent_tensorrt_compiler_source_revision": "FULL_40_CHARACTER_SOURCE_COMMIT",
  "toolchain": {
    "tensorrt_version": "11.0.0.114",
    "cuda_version": "13.3.0",
    "modelopt_version": "0.45.0",
    "torch_version": "2.8.0",
    "onnx_version": "1.21.0",
    "onnxruntime_gpu_version": "1.24.2"
  }
}
```

DROID substitutes runtime `v3` and revision
`0b067df57d21d3a02d6c511f1609172fa39ac29b`. The compiler source must equal
`source.commit`. OCI inspection independently checks the combined image's
source, compiler digest/source, policy-runtime labels, and all six toolchain
labels.

A TensorRT LIBERO evaluator sets `policy_backend` to `tensorrt`, keeps the eager
evaluator fields, and adds the same three compiler fields shown above. Its
`parent_policy_image` is the v2 combined image digest. It must also pin the
running engine-build machine:

```json
"placement": {
  "mode": "exact-existing-instance",
  "instance_id": "i-0123456789abcdef0"
}
```

The worker compares that ID with live IMDSv2 before staging or starting Docker,
requires `--build-instance-id` to match, and injects the validated live values
as the worker-owned `PI05_INSTANCE_ID` and `PI05_INSTANCE_TYPE` pair. Neither
value is copied from the worker spec. A replacement instance is rejected.

Ephemeral compile-pipeline specs invoke exactly one reviewed Python entry point:
`export_pi05_onnx.py`, `validate_pi05_onnx.py`, `quantize_pi05_fp8.py`, or
`build_tensorrt_engines.py`; latency benchmarking has the same output contract.
Shell wrappers and help-only commands are rejected;
the instance and executing image digest must match the spec. Writable
`--output-dir`, `--output`, or `--artifact-dir` paths must be below
`/output/artifacts` and covered by `expected_outputs`. `/mnt/openpi` is the
read-only input namespace (apart from explicit training-resume overlays), so it
is never a valid compiled-stage output path.

The `expected_count` must come from the selected EC2 type's documented instance-store layout. If discovery finds a different number, if the selected model is EBS, if the root device cannot be resolved exactly, or if the selected disk contains an unknown filesystem, the worker exits without formatting anything.

The bootstrap keeps `/opt/pi05`, the worker spec, source bundle, launch
metadata, and verification evidence owner-only. After the commit is checked
out, `repro_checkout_permissions.py` makes only that checkout readable and
non-writable for the UID-1000 container. The worker performs the equivalent
read-only handoff for each input only after checking every manifest size and
SHA-256.

Each execution creates exactly one new
`/opt/dlami/nvme/pi05-runs/RUN_ID` workspace. An existing run directory is a
hard failure, even when it appears empty, so a retry cannot consume old input,
checkpoint, marker, receipt, or spool state. Docker sees only the four payload
directories `checkpoints`, `logs`, `manifests`, and `artifacts` under
`/output`; host-owned `.ready`, `.receipts`, `.spool`, and `.active` control
directories are root-owned and never mounted into the container.

Validate locally without AWS writes:

```bash
python scripts/repro_worker.py run --spec /tmp/libero-shallow-20260804T120000Z-a1-2k.json
```

Upload the validated spec with AES256 encryption, then record its S3 `VersionId` and local SHA-256.

## 3. Publish the tiny bootstrap

```bash
sha256sum repro/worker-bootstrap.sh
aws s3api put-object --bucket pi05-repro-752160877725-us-east-2 \
  --key bootstrap/worker-bootstrap.sh --body repro/worker-bootstrap.sh \
  --region us-east-2 --expected-bucket-owner 752160877725 \
  --server-side-encryption AES256
aws s3api head-object --bucket pi05-repro-752160877725-us-east-2 \
  --key bootstrap/worker-bootstrap.sh --region us-east-2 \
  --expected-bucket-owner 752160877725
```

Render the launcher's command file in dry-run mode first:

```bash
python scripts/repro_worker.py render-bootstrap \
  --bootstrap-s3-uri s3://pi05-repro-752160877725-us-east-2/bootstrap/worker-bootstrap.sh \
  --bootstrap-version-id BOOTSTRAP_VERSION_ID \
  --bootstrap-sha256 BOOTSTRAP_SHA256 \
  --spec-s3-uri s3://pi05-repro-752160877725-us-east-2/specs/libero-shallow-20260804T120000Z-a1-2k.json \
  --spec-version-id SPEC_VERSION_ID \
  --spec-sha256 SPEC_SHA256 > /tmp/libero-shallow-20260804T120000Z-a1-2k.command.sh
```

Pass that file to `scripts/repro_aws_launch.py --command-file ...` without `--execute`. The bootstrapped worker validates the checked-out source and prints its plan but does not touch NVMe, stage data, pull an image, or run a container.

After reviewing both plans, render again with `render-bootstrap --execute`, then use the launcher's explicit `--execute`. These are two independent mutation gates.

## Output completion protocol

PyTorch checkpoints are saved as `tmp_STEP`, flushed, and atomically renamed to
previously nonexistent numeric `STEP` directories. Numeric steps are immutable;
the trainer refuses replacement. Before publication it checks every model and
optimizer tensor for finite values. The worker recognizes only numeric
checkpoint directories as complete and snapshots/uploads them on each sync
interval.

Other container outputs are uploaded only after the host worker validates the
declared output and atomically commits a marker in its container-inaccessible
`.ready` control directory. A marker has this exact form:

```json
{
  "schema_version": 1,
  "kind": "artifact",
  "artifacts": [
    {"path": "artifacts/metrics.json", "bytes": 123, "sha256": "LOWERCASE_SHA256"}
  ]
}
```

Allowed kinds are `checkpoint`, `log`, `manifest`, and `artifact`; allowed path
roots are the matching directories under `/output`. The worker copies each
ready file to a root-owned stable local spool before upload, refuses to
overwrite a run key with prior version history, verifies the S3 object's
size/hash metadata/version, and writes a root-owned local receipt. Container
stdout is rotated into host-owned atomic log segments and synced by the same
path.

An `expected_outputs` entry with `publish_destination` is also turned into a
worker-compatible input manifest after every file has an S3 receipt. Its
revision binds the output hashes and object VersionIds to the producing run,
source commit, image digest, seed, and declared destination. The final run
manifest contains a complete `published_inputs` object; copy that object
unchanged into the next worker's `artifacts` list. For example, the pilot above
stages on the next worker at
`/mnt/openpi/checkpoints/pi05_libero_l09_distill/libero-shallow-20260804T120000Z-a1/2000`. Only checkpoint and
artifact outputs can be published this way; logs and manifests are evidence,
not model inputs. Published worker outputs carry an S3 VersionId per payload
file, and the next worker fetches each of those exact versions before checking
its SHA-256; it does not resolve those handoff files through an unversioned
prefix sync.

### Continue a bounded run on a fresh worker

A fresh ephemeral worker cannot use bare `--resume`: its durable input is
read-only while `/mnt/openpi/runs` starts empty. Copy the prior descriptor from
`run-manifest.json.published_inputs` unchanged into `artifacts`, then add this
contract (shown for the 2k-to-5k LIBERO continuation):

```json
{
  "resume_checkpoint": {
    "artifact_name": "pilot_checkpoint",
    "target": "pi05_libero_l09_distill/libero-shallow-20260804T120000Z-a1/2000"
  },
  "container": {
    "command": [
      "torchrun", "--standalone", "--nproc-per-node=2",
      "scripts/train_pytorch.py", "pi05_libero_l09_distill",
      "--exp-name", "libero-shallow-20260804T120000Z-a1",
      "--checkpoint-base-dir", "/mnt/openpi/runs", "--resume",
      "--seed", "42",
      "--num-train-steps", "5000", "--save-interval", "5000"
    ],
    "environment": {"WANDB_MODE": "offline"},
    "shm_size_gib": 64
  },
  "expected_outputs": [
    {
      "name": "checkpoint_5000",
      "kind": "checkpoint",
      "path": "checkpoints/pi05_libero_l09_distill/libero-shallow-20260804T120000Z-a1/5000",
      "publish_destination": "pi05_libero_l09_distill/libero-shallow-20260804T120000Z-a1/5000"
    }
  ]
}
```

Before Docker starts, the worker requires a `pi05-worker-output` manifest,
fetches every payload object by its exact VersionId, rechecks every byte hash,
and requires `model.safetensors`, `optimizer.pt`, `metadata.pt`,
`wandb_id.txt`, and schema-v2 `resume-state.json`. The sidecar contains a
canonical SHA-256 fingerprint binding the config/experiment, model
architecture, PyTorch precision, immutable dataset revision, normalization
content, data-factory configuration, prompt-sidecar content, recovery
provenance, teacher model content (for Shallow), initial model lineage, seed,
effective batch inputs, optimizer, deterministic learning-rate schedule,
whole-episode split, and counter-derived model-microstep and loader-epoch
stochastic schedule. The worker verifies that embedded
fingerprint before copying bytes. The trainer reconstructs the same contract
from its invocation and rejects any mismatch. The worker atomically restores the directory at
`/mnt/openpi/runs/CONFIG/EXPERIMENT/STEP`, refuses any existing target or path
ambiguity, and does not upload that restored input as a new output. The trainer
then restores model and optimizer state, verifies metadata step equality, and
advances the deterministic data iterator past already-consumed microbatches.
Use the same shape for Shallow 5k-to-10k/20k/30k and SnapFlow
5k-to-10k/20k/30k continuations.

At the soft deadline (hard launcher deadline minus the upload buffer), the wrapper stops the container, performs final sync, uploads `manifests/run-manifest.json`, and finally uploads `manifests/final-sync-evidence.json`. The launcher's independent hard shutdown timer remains the last-resort cutoff.
The run manifest's `launch` object includes the authoritative cost-ledger
reservation ID, On-Demand purchase option, reserved hours, and projected
maximum compute cost. The worker also hashes `/opt/pi05/run-command.sh` against
that launch metadata before staging; actual post-run cost reconciliation stays
in the versioned S3 ledger rather than being guessed by the worker.
