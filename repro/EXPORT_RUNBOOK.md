# ONNX, TensorRT, and FP8 runbook

This stage is run on the same `g7e.4xlarge` used for final latency
measurements. It does not alter the OpenPI policy-server protocol. Prefix KV
state crosses the engine boundary as fixed tensors, and the decoder returns the
same normalized one-step action chunk as SnapFlow `sample_actions`.

## Preconditions

- Accepted nine-layer SnapFlow checkpoint for the selected track.
- A cost-guard reservation in `export_compile_quantize` and its ID.
- One digest-pinned combined TensorRT-policy image per LeRobot track, built
  from `repro/Dockerfile.tensorrt-policy`. For LIBERO, that v2 image is only an
  intermediate parent: build the final `POLICY_BACKEND=tensorrt`
  `repro/Dockerfile.libero` image before export, then use that final evaluator
  digest for export, validation, ModelOpt, engine build, latency, serving, and
  rollout evaluation. DROID uses its combined v3 image directly. ModelOpt 0.45
  remains isolated from OpenPI's Torch 2.7.1 training image.
- A JSONL calibration manifest with at least 1,024 distinct samples and two or
  more task/scene strata. Each record is `{ "path": "chunk.npz", "index": 0,
  "stratum": "task/scene" }`. Relative paths resolve beside the manifest.
- Each NPZ contains channel-first `image_0` through `image_2`, corresponding
  scalar masks, `lang_tokens`, `lang_mask`, `state`, normalized/internal
  `actions`, and fixed noise. `actions` is provenance input for the corpus
  envelope and is never fed to the exported graphs. Arrays may be unbatched or
  batched; the manifest index chooses one sample.

Record every manual dependency or command correction in the main runbook
before replaying it. Do not add this stage to CloudFormation until two clean
smoke replays need no undocumented edits. Commit the source first; every build
below uses its clean `git archive`, never a mutable working tree.

The NVIDIA multi-platform source was manually resolved to linux/amd64, mirrored
to the account, and GPU-smoked by digest. Build the compiler only from that
account-local mirror:

```bash
set -euo pipefail
test -z "$(git status --porcelain)"
export SOURCE_COMMIT="$(git rev-parse HEAD)"
export ECR_REPOSITORY='752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro'
export TENSORRT_BASE_IMAGE="$ECR_REPOSITORY@sha256:2a5a0a9a32ec5ddc1c384c15ddcf3b89ddc4f8647e7ee7ae708d844210183a1e"
export COMPILER_LOCAL="pi05-tensorrt-compiler:$SOURCE_COMMIT"
export COMPILER_TAG="tensorrt-compiler-$SOURCE_COMMIT"

git archive --format=tar "$SOURCE_COMMIT" | docker build --platform linux/amd64 --pull=false \
  --file repro/Dockerfile.tensorrt \
  --build-arg TENSORRT_IMAGE="$TENSORRT_BASE_IMAGE" \
  --build-arg SOURCE_SHA="$SOURCE_COMMIT" \
  --tag "$COMPILER_LOCAL" -

test "$(docker image inspect --format '{{index .Config.Labels \"org.opencontainers.image.revision\"}}' \
  "$COMPILER_LOCAL")" = "$SOURCE_COMMIT"
test "$(docker image inspect --format '{{index .Config.Labels \"ai.openpi.image-purpose\"}}' \
  "$COMPILER_LOCAL")" = tensorrt-compiler
docker run --rm --gpus all --network none "$COMPILER_LOCAL" bash -ceu '
/opt/modelopt/bin/python - <<PY
import numpy as np
import modelopt
import onnx
import onnxruntime as ort
import tensorrt
import torch

assert tensorrt.__version__ == "11.0.0.114"
assert torch.cuda.is_available()
x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])
y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])
z = onnx.helper.make_tensor_value_info("z", onnx.TensorProto.FLOAT, [1])
graph = onnx.helper.make_graph(
    [onnx.helper.make_node("Add", ["x", "y"], ["z"])],
    "cuda-smoke",
    [x, y],
    [z],
)
model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
model.ir_version = 10
onnx.save(model, "/tmp/pi05-cuda-smoke.onnx")
options = ort.SessionOptions()
options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
session = ort.InferenceSession(
    model.SerializeToString(),
    sess_options=options,
    providers=["CUDAExecutionProvider"],
)
assert session.get_providers()[0] == "CUDAExecutionProvider", session.get_providers()
actual = session.run(
    None,
    {"x": np.asarray([1], np.float32), "y": np.asarray([2], np.float32)},
)[0]
np.testing.assert_array_equal(actual, np.asarray([3], np.float32))
print(modelopt.__version__, onnx.__version__, ort.__version__, tensorrt.__version__, torch.__version__)
PY
trtexec --onnx=/tmp/pi05-cuda-smoke.onnx \
  --saveEngine=/tmp/pi05-cuda-smoke.engine \
  --warmUp=0 --duration=0 --iterations=1 \
  > /tmp/pi05-trtexec-smoke.log 2>&1
test -s /tmp/pi05-cuda-smoke.engine
grep -F "&&&& PASSED TensorRT.trtexec" /tmp/pi05-trtexec-smoke.log
'

docker tag "$COMPILER_LOCAL" "$ECR_REPOSITORY:$COMPILER_TAG"
docker push "$ECR_REPOSITORY:$COMPILER_TAG"
export COMPILER_DIGEST="$(aws ecr describe-images --region us-east-2 \
  --repository-name pi05-repro --image-ids imageTag="$COMPILER_TAG" \
  --query 'imageDetails[0].imageDigest' --output text)"
export TENSORRT_COMPILER_IMAGE="$ECR_REPOSITORY@$COMPILER_DIGEST"
docker pull "$TENSORRT_COMPILER_IMAGE"
```

Then build both track-specific combined policy runtimes from that compiler
digest and the identical source commit:

```bash
export LIBERO_LEROBOT_SHA=0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
export DROID_LEROBOT_SHA=0b067df57d21d3a02d6c511f1609172fa39ac29b

git archive --format=tar "$SOURCE_COMMIT" | docker build --platform linux/amd64 --pull=false \
  --file repro/Dockerfile.tensorrt-policy \
  --build-arg TENSORRT_COMPILER_IMAGE="$TENSORRT_COMPILER_IMAGE" \
  --build-arg TENSORRT_COMPILER_SOURCE_SHA="$SOURCE_COMMIT" \
  --build-arg SOURCE_SHA="$SOURCE_COMMIT" \
  --build-arg LEROBOT_RUNTIME=v2 --build-arg LEROBOT_SHA="$LIBERO_LEROBOT_SHA" \
  --tag "pi05-tensorrt-policy-libero:$SOURCE_COMMIT" -

git archive --format=tar "$SOURCE_COMMIT" | docker build --platform linux/amd64 --pull=false \
  --file repro/Dockerfile.tensorrt-policy \
  --build-arg TENSORRT_COMPILER_IMAGE="$TENSORRT_COMPILER_IMAGE" \
  --build-arg TENSORRT_COMPILER_SOURCE_SHA="$SOURCE_COMMIT" \
  --build-arg SOURCE_SHA="$SOURCE_COMMIT" \
  --build-arg LEROBOT_RUNTIME=v3 --build-arg LEROBOT_SHA="$DROID_LEROBOT_SHA" \
  --tag "pi05-tensorrt-policy-droid:$SOURCE_COMMIT" -

for TRACK in libero droid; do
  LOCAL="pi05-tensorrt-policy-$TRACK:$SOURCE_COMMIT"
  TAG="tensorrt-policy-$TRACK-$SOURCE_COMMIT"
  test "$(docker image inspect --format '{{index .Config.Labels \"org.opencontainers.image.revision\"}}' \
    "$LOCAL")" = "$SOURCE_COMMIT"
  test "$(docker image inspect --format '{{index .Config.Labels \"ai.openpi.image-purpose\"}}' \
    "$LOCAL")" = tensorrt-policy
  test "$(docker image inspect --format '{{index .Config.Labels \"ai.openpi.parent-tensorrt-compiler-image\"}}' \
    "$LOCAL")" = "$TENSORRT_COMPILER_IMAGE"
  docker tag "$LOCAL" "$ECR_REPOSITORY:$TAG"
  docker push "$ECR_REPOSITORY:$TAG"
done

export DROID_POLICY_TAG="tensorrt-policy-droid-$SOURCE_COMMIT"
export DROID_POLICY_DIGEST="$(aws ecr describe-images --region us-east-2 \
  --repository-name pi05-repro --image-ids imageTag="$DROID_POLICY_TAG" \
  --query 'imageDetails[0].imageDigest' --output text)"
export DROID_RUNTIME_IMAGE="$ECR_REPOSITORY@$DROID_POLICY_DIGEST"
docker pull "$DROID_RUNTIME_IMAGE"
docker image inspect --format '{{json .RepoDigests}}' "$DROID_RUNTIME_IMAGE" \
  | jq -e --arg value "$DROID_RUNTIME_IMAGE" 'index($value) != null'
test "$(docker image inspect --format '{{index .Config.Labels "ai.openpi.image-purpose"}}' \
  "$DROID_RUNTIME_IMAGE")" = tensorrt-policy
test "$(docker image inspect --format '{{index .Config.Labels "ai.openpi.lerobot-runtime"}}' \
  "$DROID_RUNTIME_IMAGE")" = v3
test "$(docker image inspect --format '{{index .Config.Labels "ai.openpi.lerobot-revision"}}' \
  "$DROID_RUNTIME_IMAGE")" = "$DROID_LEROBOT_SHA"
```

Resolve each pushed repository digest, pull it by digest, repeat the label and
network-disabled GPU/import smokes, and record the compiler plus combined image
digests. A tag is never a runtime identity. The pinned Dockerfile arguments are
part of the reproduction contract and must not be overridden.

LIBERO has one additional image layer because the official simulator needs its
separate pinned Python 3.8 environment. Resolve the v2 combined digest, then
build and push the final evaluator from that digest:

```bash
export LIBERO_POLICY_TAG="tensorrt-policy-libero-$SOURCE_COMMIT"
export LIBERO_POLICY_DIGEST="$(aws ecr describe-images --region us-east-2 \
  --repository-name pi05-repro --image-ids imageTag="$LIBERO_POLICY_TAG" \
  --query 'imageDetails[0].imageDigest' --output text)"
export LIBERO_TENSORRT_POLICY_IMAGE="$ECR_REPOSITORY@$LIBERO_POLICY_DIGEST"
export LIBERO_EVALUATOR_LOCAL="pi05-libero-tensorrt-evaluator:$SOURCE_COMMIT"
export LIBERO_EVALUATOR_TAG="libero-tensorrt-evaluator-$SOURCE_COMMIT"

docker pull "$LIBERO_TENSORRT_POLICY_IMAGE"
git archive --format=tar "$SOURCE_COMMIT" | docker build --platform linux/amd64 --pull=false \
  --file repro/Dockerfile.libero \
  --build-arg POLICY_BASE_IMAGE="$LIBERO_TENSORRT_POLICY_IMAGE" \
  --build-arg POLICY_BACKEND=tensorrt \
  --build-arg SOURCE_SHA="$SOURCE_COMMIT" \
  --tag "$LIBERO_EVALUATOR_LOCAL" -

test "$(docker image inspect --format '{{index .Config.Labels "ai.openpi.image-purpose"}}' \
  "$LIBERO_EVALUATOR_LOCAL")" = libero-evaluator
test "$(docker image inspect --format '{{index .Config.Labels "ai.openpi.policy-backend"}}' \
  "$LIBERO_EVALUATOR_LOCAL")" = tensorrt
test "$(docker image inspect --format '{{index .Config.Labels "ai.openpi.parent-policy-image"}}' \
  "$LIBERO_EVALUATOR_LOCAL")" = "$LIBERO_TENSORRT_POLICY_IMAGE"
docker tag "$LIBERO_EVALUATOR_LOCAL" "$ECR_REPOSITORY:$LIBERO_EVALUATOR_TAG"
docker push "$ECR_REPOSITORY:$LIBERO_EVALUATOR_TAG"
export LIBERO_EVALUATOR_DIGEST="$(aws ecr describe-images --region us-east-2 \
  --repository-name pi05-repro --image-ids imageTag="$LIBERO_EVALUATOR_TAG" \
  --query 'imageDetails[0].imageDigest' --output text)"
export LIBERO_RUNTIME_IMAGE="$ECR_REPOSITORY@$LIBERO_EVALUATOR_DIGEST"
docker pull "$LIBERO_RUNTIME_IMAGE"
```

The exact same final `LIBERO_RUNTIME_IMAGE` digest must be running for every
LIBERO command in sections 0 through 7. Set `IMAGE_DIGEST` from that URI here;
capture the live G7e identity from the retained host's identity document after
launch, as shown below. Do not substitute the combined parent:

```bash
export IMAGE_DIGEST="${LIBERO_RUNTIME_IMAGE##*@}"
export LIBERO_REVISION=a4336d589d589045d1c56423ffdf3b88a0e19b1f
export DROID_IMAGE_DIGEST="${DROID_RUNTIME_IMAGE##*@}"
export DROID_REVISION=e44d3138c64cfeb1c24fbbce087b475fb1233728
docker image inspect --format '{{json .RepoDigests}}' "$LIBERO_RUNTIME_IMAGE" \
  | jq -e --arg value "$LIBERO_RUNTIME_IMAGE" 'index($value) != null'
```

Keep this one On-Demand `g7e.4xlarge` running until export, both numerical
gates, optional FP8, engine build, latency, engine publication, and LIBERO
compiled smoke/evaluation have finished. Stop/start is not valid because GPU
UUID and driver inventory are part of the engine manifest.

For the two manual replays, launch this as the deadline-bounded retained manual
session documented in `RUNBOOK_AWS.md`: category `export_compile_quantize`, one
`g7e.4xlarge`, a reviewed bootstrap command, and `--retain-after-command`. The
bootstrap only prepares the host and returns. This is not a completed ephemeral worker that remains available, and
no worker completion path is being reused. It is an explicitly retained instance whose prepaid guest timer and
independent external scheduler still enforce the reserved deadline.

During the manual-first replay, export, build, serve, and evaluate as reviewed
direct commands in the retained session, always in the final digest-pinned
evaluator image. Do the same for every LIBERO command in sections 0 through 7.
Both clean replays use fresh output roots on this same still-running instance;
neither may reuse the other's engines. Do not render any compiled phase as an
ordinary ephemeral worker spec. The exact-instance renderer is a future,
non-launchable validation contract until replay-driven dispatch onto an
already-running instance exists.

On the retained host, stage the reviewed Git checkout and inputs, then capture
the source, image, and live EC2 identity independently. The source checkout
must be clean. The account, region, instance ID, and instance type come from
one IMDSv2 identity document rather than from an engine manifest:

```bash
export PI05_SOURCE_CHECKOUT=/opt/pi05/source/openpi
test -z "$(git -C "$PI05_SOURCE_CHECKOUT" status --porcelain)"
export PI05_SOURCE_SHA="$(git -C "$PI05_SOURCE_CHECKOUT" rev-parse HEAD)"
test "$PI05_SOURCE_SHA" = "$SOURCE_COMMIT"

export PI05_IMAGE_DIGEST="${LIBERO_RUNTIME_IMAGE##*@}"
test "$PI05_IMAGE_DIGEST" = "$IMAGE_DIGEST"
test "$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' \
  "$LIBERO_RUNTIME_IMAGE")" = "$PI05_SOURCE_SHA"
test "$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' \
  "$DROID_RUNTIME_IMAGE")" = "$PI05_SOURCE_SHA"
test "$DROID_IMAGE_DIGEST" = "${DROID_RUNTIME_IMAGE##*@}"

PI05_IMDS_TOKEN="$(curl --fail --silent --show-error -X PUT \
  -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600' \
  http://169.254.169.254/latest/api/token)"
PI05_IDENTITY_DOCUMENT="$(curl --fail --silent --show-error \
  -H "X-aws-ec2-metadata-token: $PI05_IMDS_TOKEN" \
  http://169.254.169.254/latest/dynamic/instance-identity/document)"
test "$(jq -r .accountId <<<"$PI05_IDENTITY_DOCUMENT")" = 752160877725
test "$(jq -r .region <<<"$PI05_IDENTITY_DOCUMENT")" = us-east-2
export PI05_INSTANCE_ID="$(jq -r .instanceId <<<"$PI05_IDENTITY_DOCUMENT")"
export PI05_INSTANCE_TYPE="$(jq -r .instanceType <<<"$PI05_IDENTITY_DOCUMENT")"
test "$PI05_INSTANCE_TYPE" = g7e.4xlarge
export INSTANCE_ID="$PI05_INSTANCE_ID"
```

Use a new empty host directory for each replay and mount it at `/output`.
Mount the clean checkout read-only at `/workspace/openpi` and staged inputs
read-only at `/mnt/openpi`. This wrapper injects all four protected identities
into every network-disabled LIBERO phase; the stage guards reject omissions or
mismatches:

```bash
export PI05_REPLAY_OUTPUT_HOST=/opt/pi05/manual-replays/replay-01
test ! -e "$PI05_REPLAY_OUTPUT_HOST"
install -d -m 0700 "$PI05_REPLAY_OUTPUT_HOST"

run_libero_phase() {
  test -z "$(git -C "$PI05_SOURCE_CHECKOUT" status --porcelain)"
  test "$(git -C "$PI05_SOURCE_CHECKOUT" rev-parse HEAD)" = "$PI05_SOURCE_SHA"
  docker run --rm --gpus all --network none --ipc host --shm-size 32g \
    --workdir /workspace/openpi \
    --mount type=bind,src="$PI05_SOURCE_CHECKOUT",dst=/workspace/openpi,readonly \
    --mount type=bind,src=/mnt/openpi,dst=/mnt/openpi,readonly \
    --mount type=bind,src="$PI05_REPLAY_OUTPUT_HOST",dst=/output \
    --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/cache \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env PI05_SOURCE_SHA --env PI05_IMAGE_DIGEST \
    --env PI05_INSTANCE_ID --env PI05_INSTANCE_TYPE \
    "$LIBERO_RUNTIME_IMAGE" "$@"
}

run_droid_phase() {
  test -z "$(git -C "$PI05_SOURCE_CHECKOUT" status --porcelain)"
  test "$(git -C "$PI05_SOURCE_CHECKOUT" rev-parse HEAD)" = "$PI05_SOURCE_SHA"
  docker run --rm --gpus all --network none --ipc host --shm-size 32g \
    --workdir /workspace/openpi \
    --mount type=bind,src="$PI05_SOURCE_CHECKOUT",dst=/workspace/openpi,readonly \
    --mount type=bind,src=/mnt/openpi,dst=/mnt/openpi,readonly \
    --mount type=bind,src="$PI05_REPLAY_OUTPUT_HOST",dst=/output \
    --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/cache \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env PI05_SOURCE_SHA --env PI05_IMAGE_DIGEST="$DROID_IMAGE_DIGEST" \
    --env PI05_INSTANCE_ID --env PI05_INSTANCE_TYPE \
    "$DROID_RUNTIME_IMAGE" "$@"
}
```

Every LIBERO `python ...` example below is invoked as
`run_libero_phase python ...`; it is never executed by an already-completed
worker. Every DROID example is invoked through `run_droid_phase`; bare host
Python is not part of the compiled reproduction. That wrapper binds the v3
combined digest independently instead of inheriting LIBERO's final evaluator
digest. Use `/opt/pi05/manual-replays/replay-02` and rebuild both full artifact
trees for the second clean replay. If the retained instance stops before both
evaluations finish, restart the two-clean-replay count and rebuild on its
replacement.

Every manual stage attempt uses a never-before-created output directory. The
compiled scripts publish create-once artifacts and manifests with atomic
create-if-absent linkage and `fsync`, and refuse overwrite. Preserve a failed attempt directory and its
ledger row as evidence; never patch, clear, or reuse it. Make the documented
fix, select a new `replay-N-attempt-M` output root, and rerun the prerequisites
needed for that stage. A successful immutable stage can be mounted read-only as
input to a later fresh attempt; a partial or failed stage cannot.

The combined image-purpose contract below describes the intermediate v2
parent. DROID uses the same contract shape with its v3 revision and combined
digest directly. This v2 contract is **not** the LIBERO export/build/serve
identity:

```json
"image": {
  "uri": "752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro@sha256:IMAGE_MANIFEST_DIGEST",
  "digest": "sha256:IMAGE_MANIFEST_DIGEST",
  "purpose": "tensorrt-policy",
  "lerobot_runtime": "v2",
  "lerobot_revision": "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5",
  "parent_tensorrt_compiler_image": "752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro@sha256:COMPILER_IMAGE_DIGEST",
  "parent_tensorrt_compiler_source_revision": "FULL_SOURCE_COMMIT",
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

For DROID use the v3 revision and v3 combined digest. For LIBERO, the worker
image purpose is `libero-evaluator`, its parent is the v2 combined digest, and
its `ai.openpi.policy-backend` label is `tensorrt`. Engine manifests bind the
runtime image digest, instance ID, GPU inventory, driver, and TensorRT build.
The worker rejects a graph-only compiler image for policy serving.
For LIBERO, export, validate, quantize, compile, benchmark, serve, and evaluate
only with the final evaluator image. Never use its combined v2 parent for one of
those stages, and never evaluate engines on a replacement instance; either
split breaks the strict contract.

## 0. Generate and validate calibration corpora

Run each command in the matching final runtime image after the dataset and
released normalization assets have been staged at the paths in its config. For
LIBERO this means `LIBERO_RUNTIME_IMAGE`, including calibration generation.
The generator uses the configured transforms and fixed dataset revision. It
chooses one unique midpoint from each of 1,024 evenly spaced bins across the
full dataset, avoiding both task-local first-frame bias and DROID's enormous
random-permutation allocation. It preserves model camera order as `image_0`
through `image_2`, derives strata from the tokenized task prompt, and generates
noise from a per-sample deterministic seed. It performs no AWS calls and
refuses to overwrite a non-empty directory.

```bash
run_libero_phase python scripts/repro_make_calibration.py generate \
  --config-name pi05_libero_l09_snapflow \
  --dataset-revision a4336d589d589045d1c56423ffdf3b88a0e19b1f \
  --output-dir /output/artifacts/calibration/libero

run_droid_phase python scripts/repro_make_calibration.py generate \
  --config-name pi05_droid_l09_snapflow \
  --dataset-revision "$DROID_REVISION" \
  --output-dir /output/artifacts/calibration/droid
```

Generation only succeeds after replaying the export reader against the corpus
and proving exactly 1,024 unique records, contiguous sample ordinals, valid
chunk hashes and deterministic noise, plus at least two non-empty prompt
strata. Re-run the same final gate without decoding the source dataset with:

```bash
run_libero_phase python scripts/repro_make_calibration.py validate \
  --manifest /output/artifacts/calibration/libero/manifest.jsonl

run_droid_phase python scripts/repro_make_calibration.py validate \
  --manifest /output/artifacts/calibration/droid/manifest.jsonl
```

## 1. Export BF16 graphs

Run once for LIBERO and once for DROID, changing dataset metadata and paths:

```bash
export ARTIFACT_DIR=/output/artifacts/tensorrt/libero/bf16
run_libero_phase mkdir -p "$ARTIFACT_DIR"
run_libero_phase python scripts/export_pi05_onnx.py \
  --config pi05_libero_l09_snapflow \
  --checkpoint /mnt/openpi/checkpoints/pi05_libero_l09_snapflow \
  --calibration-manifest /output/artifacts/calibration/libero/manifest.jsonl \
  --output-dir "$ARTIFACT_DIR" \
  --track libero \
  --dataset physical-intelligence/libero \
  --dataset-revision "$LIBERO_REVISION" \
  --image-digest "$IMAGE_DIGEST" \
  --instance-id "$INSTANCE_ID" \
  --cost-reservation "$COST_RESERVATION"

export DROID_ARTIFACT_DIR=/output/artifacts/tensorrt/droid/bf16
run_droid_phase mkdir -p "$DROID_ARTIFACT_DIR"
run_droid_phase python scripts/export_pi05_onnx.py \
  --config pi05_droid_l09_snapflow \
  --checkpoint /mnt/openpi/checkpoints/pi05_droid_l09_snapflow \
  --calibration-manifest /output/artifacts/calibration/droid/manifest.jsonl \
  --output-dir "$DROID_ARTIFACT_DIR" \
  --track droid \
  --dataset allenai/MolmoAct2-DROID-Dataset \
  --dataset-revision "$DROID_REVISION" \
  --image-digest "$DROID_IMAGE_DIGEST" \
  --instance-id "$INSTANCE_ID" \
  --cost-reservation "$COST_RESERVATION"
```

The export has no dynamic axes. It emits `encode-prefix.bf16.onnx`,
`decode-denoise.bf16.onnx`, external weights, fixed inputs, paired PyTorch
goldens, SHA-256 hashes, and an export manifest.

## 2. Seal the corpus-derived action envelope

After export, bind the calibration targets, fixed held-out golden corpus,
exported PyTorch decoder reference/input, and released normalization stats into
one artifact. The default `0.01` normalized-unit margin is deterministic and
prevents harmless floating-point drift at an exact empirical extremum. It is a
configurable regression tolerance, not a physical safety margin.

```bash
run_libero_phase python scripts/repro_make_action_limits.py \
  --track libero --config-name pi05_libero_l09_snapflow \
  --calibration-manifest /output/artifacts/calibration/libero/manifest.jsonl \
  --golden-corpus /mnt/openpi/evidence/libero-heldout.npz \
  --artifact-dir "$ARTIFACT_DIR" \
  --norm-stats-json /mnt/openpi/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json \
  --internal-margin 0.01 \
  --output "$ARTIFACT_DIR/action-limits.normalized.npz"

export ARTIFACT_DIR=/output/artifacts/tensorrt/droid/bf16
run_droid_phase mkdir -p "$ARTIFACT_DIR"
run_droid_phase python scripts/repro_make_action_limits.py \
  --track droid --config-name pi05_droid_l09_snapflow \
  --calibration-manifest /output/artifacts/calibration/droid/manifest.jsonl \
  --golden-corpus /mnt/openpi/evidence/droid-heldout.npz \
  --artifact-dir "$ARTIFACT_DIR" \
  --norm-stats-json /mnt/openpi/checkpoints/pi05_droid_jointpos/assets/droid/norm_stats.json \
  --internal-margin 0.01 \
  --output "$ARTIFACT_DIR/action-limits.normalized.npz"
```

The command refuses to overwrite an existing artifact and emits an adjacent
JSON sidecar. Both contain the track/dataset identity and hashes for the
calibration manifest and chunks, held-out golden, normalization stats, export
manifest, and decoder reference/input. The active mask gates only LIBERO's
seven or DROID's eight robot dimensions; padded model dimensions are ignored.

Every emitted range is labeled a **corpus envelope**, not a robot hardware
safety limit. LIBERO's post-normalization mapping is state-independent, so its
physical corpus envelope is recorded. DROID's first seven internal actions are
joint deltas that become absolute targets only after adding the current
unnormalized state; their static physical bounds therefore remain unset. Only
the state-independent gripper corpus envelope is recorded. Simulator/robot
environment bounds remain a required rollout-worker gate for both tracks.

## 3. Gate PyTorch to BF16 ONNX

```bash
export ARTIFACT_DIR=/output/artifacts/tensorrt/libero/bf16
run_libero_phase python scripts/validate_pi05_onnx.py \
  --artifact-dir "$ARTIFACT_DIR" \
  --precision bf16 --cosine-threshold 0.999 \
  --action-limits-npz "$ARTIFACT_DIR/action-limits.normalized.npz" \
  --track libero --dataset physical-intelligence/libero \
  --dataset-revision "$LIBERO_REVISION" \
  --image-digest "$IMAGE_DIGEST" --instance-id "$INSTANCE_ID" \
  --cost-reservation "$COST_RESERVATION"

export DROID_ARTIFACT_DIR=/output/artifacts/tensorrt/droid/bf16
run_droid_phase python scripts/validate_pi05_onnx.py \
  --artifact-dir "$DROID_ARTIFACT_DIR" \
  --precision bf16 --cosine-threshold 0.999 \
  --action-limits-npz "$DROID_ARTIFACT_DIR/action-limits.normalized.npz" \
  --track droid --dataset allenai/MolmoAct2-DROID-Dataset \
  --dataset-revision "$DROID_REVISION" \
  --image-digest "$DROID_IMAGE_DIGEST" --instance-id "$INSTANCE_ID" \
  --cost-reservation "$COST_RESERVATION"
```

`--action-limits-npz` is accepted only with its matching hash-sealed JSON
sidecar and explicit corpus-envelope/non-safety metadata. The active
`action_low`, `action_high`, and `action_mask` are expressed in the same
normalized units as the graph boundary. Physical environment-limit validation
remains an evaluation-worker gate. Per-joint mean bias is gated at 0.01
normalized action units by default. Do not compile on failure. Validation
includes both an isolated decoder comparison and the compounded candidate
prefix-cache to candidate-decoder action path.

## 4. Calibrate selective FP8

Create FP8 as a second self-contained bundle. Set
`ARTIFACT_DIR=/output/artifacts/tensorrt/libero/fp8`, then rerun the LIBERO
commands in sections 1 through 3 into that directory before quantization. Do
the same through `run_droid_phase` with
`DROID_ARTIFACT_DIR=/output/artifacts/tensorrt/droid/fp8` before the DROID
quantization command. Do not copy a partial BF16 tree or write into the staged
`/mnt/openpi` inputs.

```bash
export ARTIFACT_DIR=/output/artifacts/tensorrt/libero/fp8
run_libero_phase python scripts/quantize_pi05_fp8.py \
  --artifact-dir "$ARTIFACT_DIR" \
  --calibration-manifest /output/artifacts/calibration/libero/manifest.jsonl \
  --track libero --dataset physical-intelligence/libero \
  --dataset-revision "$LIBERO_REVISION" \
  --image-digest "$IMAGE_DIGEST" --instance-id "$INSTANCE_ID" \
  --cost-reservation "$COST_RESERVATION"

export DROID_ARTIFACT_DIR=/output/artifacts/tensorrt/droid/fp8
run_droid_phase python scripts/quantize_pi05_fp8.py \
  --artifact-dir "$DROID_ARTIFACT_DIR" \
  --calibration-manifest /output/artifacts/calibration/droid/manifest.jsonl \
  --track droid --dataset allenai/MolmoAct2-DROID-Dataset \
  --dataset-revision "$DROID_REVISION" \
  --image-digest "$DROID_IMAGE_DIGEST" --instance-id "$INSTANCE_ID" \
  --cost-reservation "$COST_RESERVATION"
```

The command refuses any chunk count other than 1,024, round-robins across
strata, and allow-lists only semantically named transformer MLP MatMul/Gemm
nodes. Attention, softmax, normalization, rotary operations, graph inputs and
outputs stay BF16/FP32. If exporter names lose module paths, inspect the graph
and use a narrowly anchored `--include-regex`; record it in the runbook. The
selector fails closed rather than quantizing broadly.

Repeat step 3 with `--precision fp8 --cosine-threshold 0.995`. No QAT is part of
this reproduction. The explicit DROID FP8 gate is:

```bash
run_droid_phase python scripts/validate_pi05_onnx.py \
  --artifact-dir /output/artifacts/tensorrt/droid/fp8 \
  --precision fp8 --cosine-threshold 0.995 \
  --action-limits-npz /output/artifacts/tensorrt/droid/fp8/action-limits.normalized.npz \
  --track droid --dataset allenai/MolmoAct2-DROID-Dataset \
  --dataset-revision "$DROID_REVISION" \
  --image-digest "$DROID_IMAGE_DIGEST" --instance-id "$INSTANCE_ID" \
  --cost-reservation "$COST_RESERVATION"
```

## 5. Build TensorRT engines

First print and review the exact commands (omit `--execute`):

```bash
run_libero_phase python scripts/build_tensorrt_engines.py \
  --artifact-dir /output/artifacts/tensorrt/libero/bf16 \
  --precision bf16 \
  --validation-report /output/artifacts/tensorrt/libero/bf16/onnx-validation.bf16.json \
  --export-manifest /output/artifacts/tensorrt/libero/bf16/export-manifest.json \
  --export-image-digest "$IMAGE_DIGEST" \
  --track libero --dataset physical-intelligence/libero \
  --dataset-revision "$LIBERO_REVISION" \
  --image-digest "$IMAGE_DIGEST" --instance-id "$INSTANCE_ID" \
  --cost-reservation "$COST_RESERVATION"

run_droid_phase python scripts/build_tensorrt_engines.py \
  --artifact-dir /output/artifacts/tensorrt/droid/bf16 \
  --precision bf16 \
  --validation-report /output/artifacts/tensorrt/droid/bf16/onnx-validation.bf16.json \
  --export-manifest /output/artifacts/tensorrt/droid/bf16/export-manifest.json \
  --export-image-digest "$DROID_IMAGE_DIGEST" \
  --track droid --dataset allenai/MolmoAct2-DROID-Dataset \
  --dataset-revision "$DROID_REVISION" \
  --image-digest "$DROID_IMAGE_DIGEST" --instance-id "$INSTANCE_ID" \
  --cost-reservation "$COST_RESERVATION"
```

Add `--execute` after review. After FP8 validation passes, print and review its
separately sealed build command:

```bash
run_libero_phase python scripts/build_tensorrt_engines.py \
  --artifact-dir /output/artifacts/tensorrt/libero/fp8 \
  --precision fp8 \
  --validation-report /output/artifacts/tensorrt/libero/fp8/onnx-validation.fp8.json \
  --export-manifest /output/artifacts/tensorrt/libero/fp8/export-manifest.json \
  --export-image-digest "$IMAGE_DIGEST" \
  --fp8-manifest /output/artifacts/tensorrt/libero/fp8/fp8-manifest.json \
  --track libero --dataset physical-intelligence/libero \
  --dataset-revision "$LIBERO_REVISION" \
  --image-digest "$IMAGE_DIGEST" --instance-id "$INSTANCE_ID" \
  --cost-reservation "$COST_RESERVATION"

run_droid_phase python scripts/build_tensorrt_engines.py \
  --artifact-dir /output/artifacts/tensorrt/droid/fp8 \
  --precision fp8 \
  --validation-report /output/artifacts/tensorrt/droid/fp8/onnx-validation.fp8.json \
  --export-manifest /output/artifacts/tensorrt/droid/fp8/export-manifest.json \
  --export-image-digest "$DROID_IMAGE_DIGEST" \
  --fp8-manifest /output/artifacts/tensorrt/droid/fp8/fp8-manifest.json \
  --track droid --dataset allenai/MolmoAct2-DROID-Dataset \
  --dataset-revision "$DROID_REVISION" \
  --image-digest "$DROID_IMAGE_DIGEST" --instance-id "$INSTANCE_ID" \
  --cost-reservation "$COST_RESERVATION"
```

Add `--execute` only after that review. TensorRT 11 is always
strongly typed and has removed the old precision flags; precision comes from
BF16 graph types or explicit FP8 Q/DQ. The launcher verifies the numerical
report passed before compiling, records the exact TensorRT/GPU versions and
commands, and hashes engines, logs, layer information, and timing cache.

Upload the complete artifact directory and manifests to its versioned S3 run
prefix while keeping the instance running. The publication must emit the
immutable `kind=asset` directory descriptor consumed by
`scripts/repro_libero_eval.py`. Engines are GPU/TensorRT-build-specific; only
compare BF16 and FP8 latency from engines built, timed, served, and evaluated on
the same `g7e.4xlarge` and final evaluator image digest.

## 6. Stagewise G7e latency

Run `scripts/benchmark_pi05_latency.py` five times on the same
`g7e.4xlarge`: eager base (`--backend torch --stage base`), eager nine-layer
student (`shallow`), eager one-step model (`snapflow`), then TensorRT BF16 and
FP8 (`--backend tensorrt`). Torch stages also require their `--config`,
`--checkpoint`, and expected denoising count. For example:

```bash
run_libero_phase python scripts/benchmark_pi05_latency.py \
  --backend tensorrt --stage tensorrt_fp8 \
  --artifact-dir /output/artifacts/tensorrt/libero/fp8 \
  --track libero --dataset physical-intelligence/libero \
  --dataset-revision "$LIBERO_REVISION" \
  --image-digest "$IMAGE_DIGEST" --instance-id "$INSTANCE_ID" \
  --cost-reservation "$COST_RESERVATION"

run_droid_phase python scripts/benchmark_pi05_latency.py \
  --backend tensorrt --stage tensorrt_fp8 \
  --artifact-dir /output/artifacts/tensorrt/droid/fp8 \
  --track droid --dataset allenai/MolmoAct2-DROID-Dataset \
  --dataset-revision "$DROID_REVISION" \
  --image-digest "$DROID_IMAGE_DIGEST" --instance-id "$INSTANCE_ID" \
  --cost-reservation "$COST_RESERVATION"
```

The official defaults are batch one, 500 warmups, and 10,000 timed iterations
for each of prefix, complete denoising, and total policy. Every iteration emits
paired CUDA-event and synchronized wall time; reports include mean, p50, p95,
and p99. Nonstandard counts require the explicit
`--allow-nonstandard-counts` smoke flag and cannot enter the aggregate report.

Combine the five reports with `scripts/summarize_pi05_latency.py`. It applies
all stagewise gates from `repro/reproduction.json` and labels the cumulative
result as an AWS G7e relative speedup, never as Thor latency or an 11x Thor
claim.

## 7. Compiled LIBERO quality on the build instance

Before stopping the G7e, follow the direct TensorRT command in
`repro/LIBERO_EVAL_RUNBOOK.md`. Run it through the same retained-session
container boundary, final evaluator digest, clean checkout, protected identity
environment, checkpoint, and replay output root used above. The evaluator
starts `scripts/serve_tensorrt_policy.py` with `/opt/modelopt/bin/python` and
uses the unchanged loopback WebSocket protocol. Because the normal EC2 launcher creates fresh capacity, it is
invalid for this step; the current compiled
worker spec is non-launchable and must not be passed to
`scripts/repro_worker.py --execute`. If the instance has stopped, rebuild the engines on the replacement
rather than overriding the identity checks.

## 8. Compiled DROID serving startup on the build instance

Before stopping the same retained G7e, prove that the accepted DROID engine can
be loaded by the unchanged policy-server implementation under the protected v3
runtime identity. Run this in a retained-session terminal, wait for the
WebSocket listening message, record the output, then interrupt it with
`Ctrl-C`. This network-disabled command is a server startup/load gate, not a
RoboLab rollout; the separate RoboLab procedure must preserve the same engine
build identity when it supplies its client transport.

```bash
run_droid_phase /opt/modelopt/bin/python scripts/serve_tensorrt_policy.py \
  --artifact-dir /output/artifacts/tensorrt/droid/fp8 \
  --checkpoint-dir /mnt/openpi/checkpoints/pi05_droid_l09_snapflow \
  --precision fp8 \
  --track droid \
  --dataset allenai/MolmoAct2-DROID-Dataset \
  --dataset-revision "$DROID_REVISION" \
  --image-digest "$DROID_IMAGE_DIGEST" \
  --instance-type "$PI05_INSTANCE_TYPE" \
  --instance-id "$PI05_INSTANCE_ID" \
  --port 8000 --seed 42
```

Do not invoke this example as bare host Python or substitute the compiler,
LIBERO evaluator, eager DROID image, mutable tag, or a replacement G7e.
