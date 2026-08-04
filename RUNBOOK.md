# AWS-only pi0.5 optimization reproduction runbook

This runbook is the source of truth for the manual-first reproduction. Record
every command and manual correction here while it is discovered. CloudFormation
is written only after two clean smoke replays need no undocumented correction.

## Non-negotiable constraints

- Account `752160877725`, region `us-east-2`, On-Demand capacity only.
- The committed plus projected AWS spend may never exceed `$3,000`.
- No P5/P6, Spot, Reserved Instances, Capacity Blocks, QAT, 100k-step DROID run,
  or broad hyperparameter sweep.
- Local NVMe is scratch. Checkpoints, logs, manifests, graphs, engines, dataset
  revisions, and evaluation results must be copied to versioned encrypted S3.
- Every paid launch goes through `scripts/repro_aws_launch.py --execute` with
  explicit spend-category and workload identities; it reserves with the cost
  guard before its idempotent On-Demand API call.
- Every run writes a manifest with `scripts/repro_manifest.py` and is not promoted
  until its artifact hashes exist in S3.
- AWS results validate the method and relative G7e speedups. They do not validate
  Thor's 24 ms latency or an 11x Thor claim.

## Run entry template

Copy this block for every manual run or correction:

```text
UTC time:
Operator:
Git SHA and dirty state:
Purpose/gate:
Instance/AZ/AMI or image digest:
Dataset and revision:
Cost reservation ID and maximum hours:
Exact command:
Observed result/metrics:
Manual correction (exact edit/command, or none):
Artifact S3 URI and SHA-256:
Instance termination verified at:
```

## Pre-run checklist

The exact foundation and launch commands are recorded in `RUNBOOK_AWS.md`.

1. `aws sts get-caller-identity` returns account `752160877725`.
2. `AWS_REGION` and CLI region are explicitly `us-east-2`.
3. Working tree SHA and container digest match the intended run manifest.
4. Dataset and checkpoint S3 object versions are resolved, never `latest`.
5. Projected hours include setup, checkpoint upload, and a shutdown margin.
6. The cost guard reservation passes both its category cap and the hard cap.
7. An instance-side shutdown deadline is installed before the workload starts.
8. SSM works; the security group has no inbound rules.

## Promotion order

1. Foundation, checkpoint conversion, golden vectors, and baseline quality
   smoke.
2. Shallow-pi: 300-step overfit, 2k pilot, then 5k/10k/20k/30k empirical gates.
3. Bounded RoboLab BC recovery only when the Stack3RubiksCube trigger fires;
   follow `repro/ROBOLAB_BC_RECOVERY_RUNBOOK.md`.
4. SnapFlow: 5k pilot, then 10k/20k/30k only while gates require it.
5. BF16 ONNX validation, BF16 TensorRT, selective-MLP FP8 calibration, and the
   fixed batch-one benchmark inputs emitted by export.
6. On the same retained `g7e.4xlarge`, run eager base, Shallow, SnapFlow,
   TensorRT BF16, and TensorRT FP8 latency; then run intermediate and paired
   final quality evaluation.
7. Two clean abbreviated manual replays, then CloudFormation and Change Set.

The foundation baseline smoke is a quality/startup gate, not official latency
evidence.  Official eager-base latency is deliberately deferred until step 6:
the accepted SnapFlow export has then emitted `encode-inputs.npz` and
`decode-inputs.npz`, and all five stages can be timed on one retained instance.
Do not keep an early G7e instance alive across training or compare an early
baseline measurement with engines built on a replacement instance.

## Dataset and released-teacher staging

This stage needs network and disk only; it does not need or authorize a GPU
instance. `repro_stage_data.py` reads the repository and full 40-character
revision from `repro/reproduction.json`. There are deliberately no command-line
repo or revision overrides. `plan`, and mutating actions without `--execute`,
are dry runs and do not contact Hugging Face or AWS.

First print both transfer plans:

```bash
python3 scripts/repro_stage_data.py plan \
  --dataset libero \
  --local-root /mnt/openpi/datasets \
  --s3-root s3://pi05-repro-752160877725-us-east-2/datasets

python3 scripts/repro_stage_data.py plan \
  --dataset droid \
  --local-root /mnt/openpi/datasets \
  --s3-root s3://pi05-repro-752160877725-us-east-2/datasets
```

Download and validate each immutable snapshot. These are the only commands in
this section that download dataset payloads:

```bash
python3 scripts/repro_stage_data.py download \
  --dataset libero \
  --local-root /mnt/openpi/datasets \
  --execute

python3 scripts/repro_stage_data.py download \
  --dataset droid \
  --local-root /mnt/openpi/datasets \
  --execute
```

The DROID command refuses to start unless at least 105% of the configured
259 GB estimate is free. Both commands validate `meta/info.json`, the declared
LeRobot codebase version and required fields before hashing every payload file.
The DROID validation additionally requires one non-empty
`meta/tasks_annotated.parquet` row for each episode, checks the scalar/action
policy-input schema in every data parquet, and reconstructs the exact data and
required-camera file sets from all `meta/episodes` references. It rejects null
or negative file references, missing or orphan files, empty required MP4s, and
non-finite or unordered camera timestamps. Image features are MP4 subtrees in
this LeRobot v3 snapshot and are deliberately not required as parquet columns.
For the pinned revision it also requires exactly 518 exterior-left and 316
wrist-left MP4 references; the validation report records those counts and each
camera's minimum start, minimum duration, and maximum end timestamp. Hugging
Face's `.cache` transfer state is neither hashed nor uploaded.

If a download was resumed or copied from another volume, rerun the read-only
validation and SHA-256 manifest generation explicitly:

```bash
python3 scripts/repro_stage_data.py validate \
  --dataset libero \
  --local-root /mnt/openpi/datasets

python3 scripts/repro_stage_data.py validate \
  --dataset droid \
  --local-root /mnt/openpi/datasets
```

Only promote a DROID manifest whose validation block contains
`"layout_contract": "molmoact2-v3-exact-media-references-v1"`. A manifest
created by an earlier, weaker validator must be regenerated and uploaded; use
the new manifest SHA-256 and S3 `VersionId` in worker specifications. The
versioned bucket retains the older manifest version, but it must not be replayed.

Upload only after inspecting the validation JSON. The uploader rejects the
operation before `aws s3 sync` unless the effective region is `us-east-2`, STS
returns account `752160877725`, and the destination bucket is in Ohio with
versioning and default encryption enabled. It uploads beneath a path containing
the pinned source commit, uploads the SHA-256 manifest separately, and verifies
the manifest object's source-revision metadata and S3 version ID.

```bash
AWS_REGION=us-east-2 python3 scripts/repro_stage_data.py upload \
  --dataset libero \
  --local-root /mnt/openpi/datasets \
  --s3-root s3://pi05-repro-752160877725-us-east-2/datasets \
  --execute

AWS_REGION=us-east-2 python3 scripts/repro_stage_data.py upload \
  --dataset droid \
  --local-root /mnt/openpi/datasets \
  --s3-root s3://pi05-repro-752160877725-us-east-2/datasets \
  --execute
```

Expected immutable destinations are:

```text
s3://pi05-repro-752160877725-us-east-2/datasets/libero/a4336d589d589045d1c56423ffdf3b88a0e19b1f/
s3://pi05-repro-752160877725-us-east-2/datasets/molmoact2-droid/e44d3138c64cfeb1c24fbbce087b475fb1233728/
```

### Why there are two LeRobot images

The released `physical-intelligence/libero` snapshot is LeRobot `v2.0`; the
released `allenai/MolmoAct2-DROID-Dataset` snapshot is LeRobot `v3.0`. LeRobot
performs a strict major-format compatibility check, so one installed runtime
must not be used to silently reinterpret both layouts. Dataset staging is
runtime-neutral and preserves both raw releases. Build and digest-pin the
LIBERO image with the v2 commit and the DROID image with the v3 commit. First
publish the final commit's source bundle exactly as described in
`repro/WORKER_RUNBOOK.md` and retain its VersionId and SHA-256. Build from a
`git archive` of that same commit, rather than the live directory, so ignored
credentials, local artifacts, and edits made while a long build is running
cannot enter an image whose label claims to be the committed tree.

The AWS DLC base is in a separate private ECR registry. Authenticate Docker to
both registries before either build. The worker AMIs are x86-64, so the build
also pins and verifies `linux/amd64` explicitly:

```bash
set -euo pipefail
test "$(uname -m)" = x86_64
test -z "$(git status --porcelain)"
export SOURCE_COMMIT="$(git rev-parse HEAD)"
export OPENPI_SHA="$(python3 -c 'import json; print(json.load(open("repro/reproduction.json"))["source"]["openpi_commit"])')"
export LIBERO_LEROBOT_SHA="$(python3 -c 'import json; print(json.load(open("repro/reproduction.json"))["source"]["lerobot_v2_commit"])')"
export DROID_LEROBOT_SHA="$(python3 -c 'import json; print(json.load(open("repro/reproduction.json"))["source"]["lerobot_v3_commit"])')"
export PALIGEMMA_TOKENIZER_SHA256="$(sed -n 's/^ARG PALIGEMMA_TOKENIZER_SHA256=//p' repro/Dockerfile)"
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$OPENPI_SHA" =~ ^[0-9a-f]{40}$ ]]
test "${#PALIGEMMA_TOKENIZER_SHA256}" -eq 64
export BASE_ECR_REGISTRY=763104351884.dkr.ecr.us-east-2.amazonaws.com
export ECR_REGISTRY=752160877725.dkr.ecr.us-east-2.amazonaws.com
export ECR_REPOSITORY="$ECR_REGISTRY/pi05-repro"
export LIBERO_TAG="libero-v2-$SOURCE_COMMIT"
export DROID_TAG="droid-v3-$SOURCE_COMMIT"

for registry in "$BASE_ECR_REGISTRY" "$ECR_REGISTRY"; do
  aws ecr get-login-password --region us-east-2 | \
    docker login --username AWS --password-stdin "$registry"
done

git archive --format=tar "$SOURCE_COMMIT" | docker build \
  --platform linux/amd64 \
  --build-arg SOURCE_SHA="$SOURCE_COMMIT" \
  --build-arg LEROBOT_RUNTIME=v2 \
  --build-arg LEROBOT_SHA="$LIBERO_LEROBOT_SHA" \
  --file repro/Dockerfile \
  --tag "pi05-repro:$LIBERO_TAG" -

git archive --format=tar "$SOURCE_COMMIT" | docker build \
  --platform linux/amd64 \
  --build-arg SOURCE_SHA="$SOURCE_COMMIT" \
  --build-arg LEROBOT_RUNTIME=v3 \
  --build-arg LEROBOT_SHA="$DROID_LEROBOT_SHA" \
  --file repro/Dockerfile \
  --tag "pi05-repro:$DROID_TAG" -

verify_training_image() {
  local image="$1" runtime="$2" lerobot_sha="$3"
  test "$(docker image inspect --format '{{.Architecture}}' "$image")" = amd64
  test "$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$image")" = \
    "$SOURCE_COMMIT"
  test "$(docker image inspect --format '{{index .Config.Labels "ai.openpi.upstream-revision"}}' "$image")" = \
    "$OPENPI_SHA"
  test "$(docker image inspect --format '{{index .Config.Labels "ai.openpi.image-purpose"}}' "$image")" = \
    policy
  test "$(docker image inspect --format '{{index .Config.Labels "ai.openpi.lerobot-runtime"}}' "$image")" = \
    "$runtime"
  test "$(docker image inspect --format '{{index .Config.Labels "ai.openpi.lerobot-revision"}}' "$image")" = \
    "$lerobot_sha"
  test "$(docker image inspect --format '{{index .Config.Labels "ai.openpi.paligemma-tokenizer-sha256"}}' "$image")" = \
    "$PALIGEMMA_TOKENIZER_SHA256"
  test "$(docker image inspect --format '{{index .Config.Labels "ai.openpi.video-decoder"}}' "$image")" = \
    pyav
  test "$(docker image inspect --format '{{index .Config.Labels "ai.openpi.onnxruntime-gpu-version"}}' "$image")" = \
    1.26.0
}
smoke_policy_image() {
  local image="$1" runtime="$2"
  docker run --rm --gpus all --network none --user 1000:1000 \
    --interactive --env HOME=/tmp --env EXPECTED_LEROBOT_RUNTIME="$runtime" "$image" \
    python - <<'PY'
import importlib
import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import torch

from scripts.smoke_lerobot_video import smoke_pyav_decoder

assert torch.cuda.is_available()
torch_value = torch.ones(1, device="cuda")
assert torch_value.item() == 1
smoke_pyav_decoder()

devices = jax.devices()
assert any(device.platform == "gpu" for device in devices), devices
jax_value = jnp.ones((1,), dtype=jnp.float32)
jax.block_until_ready(jax_value)
assert {device.platform for device in jax_value.devices()} == {"gpu"}

x_info = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])
y_info = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])
z_info = onnx.helper.make_tensor_value_info("z", onnx.TensorProto.FLOAT, [1])
graph = onnx.helper.make_graph([onnx.helper.make_node("Add", ["x", "y"], ["z"])], "cuda-smoke", [x_info, y_info], [z_info])
model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
options = ort.SessionOptions()
options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
session = ort.InferenceSession(
    model.SerializeToString(), sess_options=options, providers=["CUDAExecutionProvider"]
)
assert session.get_providers()[0] == "CUDAExecutionProvider", session.get_providers()
actual = session.run(None, {"x": np.array([1], np.float32), "y": np.array([2], np.float32)})[0]
np.testing.assert_array_equal(actual, np.array([3], np.float32))

module = (
    "lerobot.datasets.lerobot_dataset"
    if os.environ["EXPECTED_LEROBOT_RUNTIME"] == "v3"
    else "lerobot.common.datasets.lerobot_dataset"
)
importlib.import_module(module)
cache = pathlib.Path(os.environ["HF_HOME"])
cache.mkdir(parents=True, exist_ok=True)
marker = cache / ".write-smoke"
marker.write_text("ok")
marker.unlink()
print(torch.__version__, torch.version.cuda, jax_value.devices(), module, session.get_providers())
PY
}
verify_training_image "pi05-repro:$LIBERO_TAG" v2 "$LIBERO_LEROBOT_SHA"
verify_training_image "pi05-repro:$DROID_TAG" v3 "$DROID_LEROBOT_SHA"
smoke_policy_image "pi05-repro:$LIBERO_TAG" v2
smoke_policy_image "pi05-repro:$DROID_TAG" v3

docker tag "pi05-repro:$LIBERO_TAG" "$ECR_REPOSITORY:$LIBERO_TAG"
docker tag "pi05-repro:$DROID_TAG" "$ECR_REPOSITORY:$DROID_TAG"
docker push "$ECR_REPOSITORY:$LIBERO_TAG"
docker push "$ECR_REPOSITORY:$DROID_TAG"

export LIBERO_IMAGE_DIGEST="$(aws ecr describe-images --region us-east-2 \
  --repository-name pi05-repro --image-ids imageTag="$LIBERO_TAG" \
  --query 'imageDetails[0].imageDigest' --output text)"
export DROID_IMAGE_DIGEST="$(aws ecr describe-images --region us-east-2 \
  --repository-name pi05-repro --image-ids imageTag="$DROID_TAG" \
  --query 'imageDetails[0].imageDigest' --output text)"
[[ "$LIBERO_IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]]
[[ "$DROID_IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]]
test "$LIBERO_IMAGE_DIGEST" != "$DROID_IMAGE_DIGEST"
export LIBERO_IMAGE_URI="$ECR_REPOSITORY@$LIBERO_IMAGE_DIGEST"
export DROID_IMAGE_URI="$ECR_REPOSITORY@$DROID_IMAGE_DIGEST"

docker pull "$LIBERO_IMAGE_URI"
docker pull "$DROID_IMAGE_URI"
verify_training_image "$LIBERO_IMAGE_URI" v2 "$LIBERO_LEROBOT_SHA"
verify_training_image "$DROID_IMAGE_URI" v3 "$DROID_LEROBOT_SHA"
smoke_policy_image "$LIBERO_IMAGE_URI" v2
smoke_policy_image "$DROID_IMAGE_URI" v3
```

The exact runtime commits are also pinned in `repro/reproduction.json`. Record
the two immutable URI/digest pairs in the manual log and every matching
training/evaluation manifest. Do not run a LIBERO loader in the DROID image or
a DROID loader in the LIBERO image. Worker specs always use the digest URI,
never the temporary source tag, and must pair it with the same `SOURCE_COMMIT`,
source-bundle VersionId, and source-bundle SHA-256. Because ECR tag immutability
is enabled, a retry after one tag was already published must reuse and verify
that tag's recorded digest; do not invent a mutable replacement tag or rebuild
the other track from a different commit.

### Released teacher checkpoints

The released teachers are the approximately 12.4 GB GCS directories
`gs://openpi-assets/checkpoints/pi05_libero` and
`gs://openpi-assets-simeval/pi05_droid_jointpos`. The joint-position checkpoint
is the public teacher used by RoboLab's π0.5 client; using the generic DROID
checkpoint would silently evaluate the wrong action-space contract. Unlike the datasets, these URLs do
not name a repository commit. A repeatable staging run must therefore record
the GCS generation, byte count, MD5/CRC32C metadata and local SHA-256 for every
object; the URL alone is not an acceptable source revision. Do not copy either
teacher into the versioned S3 prefix until that object-generation inventory has
been captured. `repro_stage_checkpoints.py` computes a deterministic SHA-256 of
that inventory and uses it as the immutable S3 path component. As with dataset
staging, the default is a network-free dry run:

```bash
python3 scripts/repro_stage_checkpoints.py plan \
  --checkpoint libero \
  --local-root /mnt/openpi/checkpoints \
  --s3-root s3://pi05-repro-752160877725-us-east-2/checkpoints

python3 scripts/repro_stage_checkpoints.py plan \
  --checkpoint droid_jointpos \
  --local-root /mnt/openpi/checkpoints \
  --s3-root s3://pi05-repro-752160877725-us-east-2/checkpoints
```

Download generation-specific objects, verify source sizes and available GCS
MD5s, refetch the inventory to detect a mid-transfer source change, then hash
the local trees:

```bash
python3 scripts/repro_stage_checkpoints.py download \
  --checkpoint libero \
  --local-root /mnt/openpi/checkpoints \
  --execute

python3 scripts/repro_stage_checkpoints.py download \
  --checkpoint droid_jointpos \
  --local-root /mnt/openpi/checkpoints \
  --execute
```

Upload only after inspecting each source manifest. These commands perform the
same account, region, bucket-versioning and encryption preflight as dataset
staging:

```bash
AWS_REGION=us-east-2 python3 scripts/repro_stage_checkpoints.py upload \
  --checkpoint libero \
  --local-root /mnt/openpi/checkpoints \
  --s3-root s3://pi05-repro-752160877725-us-east-2/checkpoints \
  --execute

AWS_REGION=us-east-2 python3 scripts/repro_stage_checkpoints.py upload \
  --checkpoint droid_jointpos \
  --local-root /mnt/openpi/checkpoints \
  --s3-root s3://pi05-repro-752160877725-us-east-2/checkpoints \
  --execute
```

Convert each teacher from a clean committed checkout in the matching
digest-pinned image. The converter refuses a non-empty output directory, copies
the nested normalization assets, and writes a self-describing `config.json`.
The image's OCI revision must equal `SOURCE_COMMIT` before doing the expensive
restore. Reuse the two ECR digest URIs recorded by the image-build step; tags
are not accepted as provenance.

```bash
export SOURCE_COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --porcelain)"
: "${LIBERO_IMAGE_URI:?set the digest URI emitted by the image-build step}"
: "${DROID_IMAGE_URI:?set the digest URI emitted by the image-build step}"
export LIBERO_IMAGE_DIGEST="${LIBERO_IMAGE_URI##*@}"
export DROID_IMAGE_DIGEST="${DROID_IMAGE_URI##*@}"

for image in "$LIBERO_IMAGE_URI" "$DROID_IMAGE_URI"; do
  test "$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$image")" = \
    "$SOURCE_COMMIT"
done

docker run --rm --gpus all --network none --user "$(id -u):$(id -g)" \
  --workdir /workspace/openpi \
  --mount "type=bind,src=$PWD,dst=/workspace/openpi,readonly" \
  --mount type=bind,src=/mnt/openpi/checkpoints,dst=/mnt/openpi/checkpoints \
  --env HOME=/tmp "$LIBERO_IMAGE_URI" \
  python examples/convert_jax_model_to_pytorch.py \
    --checkpoint-dir /mnt/openpi/checkpoints/pi05_libero \
    --config-name pi05_libero \
    --output-path /mnt/openpi/checkpoints/pi05_libero_pytorch \
    --precision bfloat16

docker run --rm --gpus all --network none --user "$(id -u):$(id -g)" \
  --workdir /workspace/openpi \
  --mount "type=bind,src=$PWD,dst=/workspace/openpi,readonly" \
  --mount type=bind,src=/mnt/openpi/checkpoints,dst=/mnt/openpi/checkpoints \
  --env HOME=/tmp "$DROID_IMAGE_URI" \
  python examples/convert_jax_model_to_pytorch.py \
    --checkpoint-dir /mnt/openpi/checkpoints/pi05_droid_jointpos \
    --config-name pi05_droid_jointpos \
    --output-path /mnt/openpi/checkpoints/pi05_droid_jointpos_pytorch \
    --precision bfloat16
```

Build worker-compatible converted manifests only after the source JAX trees,
conversion config, copied asset ID, and every output hash validate. The
converted revision binds those bytes to the original GCS inventory, exact
source commit, config, precision, and image digest:

```bash
python3 scripts/repro_stage_converted_checkpoints.py validate \
  --checkpoint libero \
  --local-root /mnt/openpi/checkpoints \
  --source-commit "$SOURCE_COMMIT" \
  --image-digest "$LIBERO_IMAGE_DIGEST"

python3 scripts/repro_stage_converted_checkpoints.py validate \
  --checkpoint droid_jointpos \
  --local-root /mnt/openpi/checkpoints \
  --source-commit "$SOURCE_COMMIT" \
  --image-digest "$DROID_IMAGE_DIGEST"
```

Before either converted teacher is promoted, build exactly one fixed 64-sample
validation corpus per track in its matching LeRobot image and run the
JAX-to-PyTorch velocity gate. These two canonical files are also the held-out
corpora used by every later Shallow/SnapFlow offline gate in
`repro/TRAINING_RUNBOOK.md` and by the action-envelope step in
`repro/EXPORT_RUNBOOK.md`; do not create a second framework-only corpus.
The `--data-split-seed 42` argument is the training-config seed: it selects the
same deterministic, task/site-stratified whole-episode validation split used by
training before any delta/action query. The separate `--seed` values fix only
the corpus noise/timestep vectors.
The image contains the SHA-256-pinned PaliGemma tokenizer, so all four commands
run with networking disabled. The comparison rejects a corpus whose bytes,
dataset revision, resolved config fingerprint, teacher link, tensor contract,
source manifest, converted manifest, conversion config, or converted model hash
does not match. It requires every sample—not only the mean—to have velocity
cosine similarity at least `0.999` and writes both frameworks' velocities with
their hash into the report. Each comparison refuses to start the expensive
full-depth forward passes if either its report path or adjacent velocity NPZ
already exists (including a symlink), and both final writes use exclusive
creation. If a run is interrupted after creating only one file, retain that
partial evidence for diagnosis; do not delete it and blindly reuse the same
canonical output name.

```bash
export LIBERO_DATASET_REVISION="$(jq -er '.source.libero_revision' repro/reproduction.json)"
export DROID_DATASET_REVISION="$(jq -er '.source.molmoact2_droid_revision' repro/reproduction.json)"
export REPRO_RUN_ID=pi05-aws-repro-001
mkdir -p /mnt/openpi/evidence

docker run --rm --gpus all --network none --user "$(id -u):$(id -g)" \
  --workdir /workspace/openpi \
  --mount "type=bind,src=$PWD,dst=/workspace/openpi,readonly" \
  --mount type=bind,src=/mnt/openpi/datasets,dst=/mnt/openpi/datasets,readonly \
  --mount type=bind,src=/mnt/openpi/checkpoints,dst=/mnt/openpi/checkpoints,readonly \
  --mount type=bind,src=/mnt/openpi/evidence,dst=/mnt/openpi/evidence \
  --env HOME=/tmp "$LIBERO_IMAGE_URI" \
  python scripts/repro_make_golden.py \
    --run-id "$REPRO_RUN_ID" \
    --config-name pi05_libero_l09_distill \
    --samples 64 \
    --seed 7001 \
    --data-split-seed 42 \
    --dataset-revision "$LIBERO_DATASET_REVISION" \
    --output /mnt/openpi/evidence/libero-heldout.npz

docker run --rm --gpus all --network none --user "$(id -u):$(id -g)" \
  --workdir /workspace/openpi \
  --mount "type=bind,src=$PWD,dst=/workspace/openpi,readonly" \
  --mount type=bind,src=/mnt/openpi/datasets,dst=/mnt/openpi/datasets,readonly \
  --mount type=bind,src=/mnt/openpi/checkpoints,dst=/mnt/openpi/checkpoints,readonly \
  --mount type=bind,src=/mnt/openpi/evidence,dst=/mnt/openpi/evidence \
  --env HOME=/tmp "$DROID_IMAGE_URI" \
  python scripts/repro_make_golden.py \
    --run-id "$REPRO_RUN_ID" \
    --config-name pi05_droid_l09_distill \
    --samples 64 \
    --seed 7002 \
    --data-split-seed 42 \
    --dataset-revision "$DROID_DATASET_REVISION" \
    --output /mnt/openpi/evidence/droid-heldout.npz

docker run --rm --gpus all --network none --user "$(id -u):$(id -g)" \
  --workdir /workspace/openpi \
  --mount "type=bind,src=$PWD,dst=/workspace/openpi,readonly" \
  --mount type=bind,src=/mnt/openpi/checkpoints,dst=/mnt/openpi/checkpoints,readonly \
  --mount type=bind,src=/mnt/openpi/evidence,dst=/mnt/openpi/evidence \
  --env HOME=/tmp "$LIBERO_IMAGE_URI" \
  python scripts/repro_compare_frameworks.py \
    --config-name pi05_libero \
    --jax-checkpoint /mnt/openpi/checkpoints/pi05_libero \
    --pytorch-checkpoint /mnt/openpi/checkpoints/pi05_libero_pytorch \
    --source-manifest /mnt/openpi/checkpoints/_manifests/pi05_libero.source-manifest.json \
    --converted-manifest /mnt/openpi/checkpoints/_manifests/pi05_libero_pytorch.converted-manifest.json \
    --corpus /mnt/openpi/evidence/libero-heldout.npz \
    --device cuda:0 \
    --output /mnt/openpi/evidence/pi05_libero.framework-equivalence.json

docker run --rm --gpus all --network none --user "$(id -u):$(id -g)" \
  --workdir /workspace/openpi \
  --mount "type=bind,src=$PWD,dst=/workspace/openpi,readonly" \
  --mount type=bind,src=/mnt/openpi/checkpoints,dst=/mnt/openpi/checkpoints,readonly \
  --mount type=bind,src=/mnt/openpi/evidence,dst=/mnt/openpi/evidence \
  --env HOME=/tmp "$DROID_IMAGE_URI" \
  python scripts/repro_compare_frameworks.py \
    --config-name pi05_droid_jointpos \
    --jax-checkpoint /mnt/openpi/checkpoints/pi05_droid_jointpos \
    --pytorch-checkpoint /mnt/openpi/checkpoints/pi05_droid_jointpos_pytorch \
    --source-manifest /mnt/openpi/checkpoints/_manifests/pi05_droid_jointpos.source-manifest.json \
    --converted-manifest /mnt/openpi/checkpoints/_manifests/pi05_droid_jointpos_pytorch.converted-manifest.json \
    --corpus /mnt/openpi/evidence/droid-heldout.npz \
    --device cuda:0 \
    --output /mnt/openpi/evidence/pi05_droid_jointpos.framework-equivalence.json
```

Both commands must exit zero and report `gate_pass: true`. A failure blocks
teacher upload and Shallow-pi training; keep the corpus and velocity artifacts
for diagnosis instead of regenerating them with a different seed. Record each
NPZ and adjacent JSON SHA-256 once, then carry those exact hashes into every
framework-equivalence, offline-metrics, promotion, and export manifest.

Validate and immutably stage that evidence before the workbench can stop. The
helper below is only shell abbreviation: each `validate` is local/read-only,
each first `upload` is an AWS-free dry run, and only the final invocation with
`--execute` can write. The uploader rechecks all cross-file identities and the
64-sample cosine gate, refuses any existing object/version history at its
content-addressed destination, then records an AES256-encrypted S3 VersionId,
SHA-256 checksum, and size for all four files plus the manifest written last.

```bash
stage_equivalence_evidence() {
  local action="$1" track="$2" corpus="$3" teacher="$4" image_digest="$5"
  shift 5
  python3 scripts/repro_stage_equivalence_evidence.py "$action" \
    --track "$track" \
    --golden-npz "/mnt/openpi/evidence/${corpus}-heldout.npz" \
    --golden-sidecar "/mnt/openpi/evidence/${corpus}-heldout.json" \
    --equivalence-report "/mnt/openpi/evidence/${teacher}.framework-equivalence.json" \
    --velocity-npz "/mnt/openpi/evidence/${teacher}.framework-equivalence.npz" \
    --source-manifest "/mnt/openpi/checkpoints/_manifests/${teacher}.source-manifest.json" \
    --converted-manifest "/mnt/openpi/checkpoints/_manifests/${teacher}_pytorch.converted-manifest.json" \
    --source-commit "$SOURCE_COMMIT" \
    --image-digest "$image_digest" \
    "$@"
}

stage_equivalence_evidence validate libero libero pi05_libero "$LIBERO_IMAGE_DIGEST"
stage_equivalence_evidence validate droid_jointpos droid pi05_droid_jointpos "$DROID_IMAGE_DIGEST"

stage_equivalence_evidence upload libero libero pi05_libero "$LIBERO_IMAGE_DIGEST" \
  --bucket pi05-repro-752160877725-us-east-2 --prefix evidence/framework-equivalence
stage_equivalence_evidence upload droid_jointpos droid pi05_droid_jointpos "$DROID_IMAGE_DIGEST" \
  --bucket pi05-repro-752160877725-us-east-2 --prefix evidence/framework-equivalence

AWS_REGION=us-east-2 stage_equivalence_evidence upload \
  libero libero pi05_libero "$LIBERO_IMAGE_DIGEST" \
  --bucket pi05-repro-752160877725-us-east-2 --prefix evidence/framework-equivalence --execute
AWS_REGION=us-east-2 stage_equivalence_evidence upload \
  droid_jointpos droid pi05_droid_jointpos "$DROID_IMAGE_DIGEST" \
  --bucket pi05-repro-752160877725-us-east-2 --prefix evidence/framework-equivalence --execute
```

Record both emitted evidence revisions, all object VersionIds, the final
manifest VersionIds, and manifest hashes in the manual ledger. Later workers
must retrieve these exact versions; an unversioned S3 key is not evidence.

Review each manifest and the network-free upload plan, then execute. Original
and converted teachers deliberately use separate immutable S3 prefixes. The
converted publisher is create-once rather than an `aws s3 sync`: it
conditionally creates a content/provenance claim, conditionally publishes each
small object or completes each multipart object with `If-None-Match: *`, and
performs a version-specific SHA-256 round trip for every object. It then writes
a durable receipt and publishes the converted manifest last. A retry after an
interruption resumes only when the existing claim and every partial object are
byte-for-byte exact and have one version with no delete marker; unknown keys,
changed bytes, or ambiguous version history stop the run. Therefore retry the
same command after a transport interruption—never clear or overwrite the
content-addressed prefix. Record the emitted claim, payload, receipt, and final
manifest VersionIds from `s3.publication` along with the copy-ready
`worker_artifact`.

```bash
python3 scripts/repro_stage_converted_checkpoints.py upload \
  --checkpoint libero \
  --local-root /mnt/openpi/checkpoints \
  --source-commit "$SOURCE_COMMIT" \
  --image-digest "$LIBERO_IMAGE_DIGEST" \
  --equivalence-report /mnt/openpi/evidence/pi05_libero.framework-equivalence.json \
  --s3-root s3://pi05-repro-752160877725-us-east-2/checkpoints

AWS_REGION=us-east-2 python3 scripts/repro_stage_converted_checkpoints.py upload \
  --checkpoint libero \
  --local-root /mnt/openpi/checkpoints \
  --source-commit "$SOURCE_COMMIT" \
  --image-digest "$LIBERO_IMAGE_DIGEST" \
  --equivalence-report /mnt/openpi/evidence/pi05_libero.framework-equivalence.json \
  --s3-root s3://pi05-repro-752160877725-us-east-2/checkpoints \
  --execute

python3 scripts/repro_stage_converted_checkpoints.py upload \
  --checkpoint droid_jointpos \
  --local-root /mnt/openpi/checkpoints \
  --source-commit "$SOURCE_COMMIT" \
  --image-digest "$DROID_IMAGE_DIGEST" \
  --equivalence-report /mnt/openpi/evidence/pi05_droid_jointpos.framework-equivalence.json \
  --s3-root s3://pi05-repro-752160877725-us-east-2/checkpoints

AWS_REGION=us-east-2 python3 scripts/repro_stage_converted_checkpoints.py upload \
  --checkpoint droid_jointpos \
  --local-root /mnt/openpi/checkpoints \
  --source-commit "$SOURCE_COMMIT" \
  --image-digest "$DROID_IMAGE_DIGEST" \
  --equivalence-report /mnt/openpi/evidence/pi05_droid_jointpos.framework-equivalence.json \
  --s3-root s3://pi05-repro-752160877725-us-east-2/checkpoints \
  --execute
```

Copy the emitted `worker_artifact` object into each worker spec rather than
hand-assembling its revision, manifest VersionId, SHA-256, payload URI, and
destination.

## Manual execution log

### 2026-08-03 - source import and guardrails

- Imported upstream OpenPI commit `15a9616a00943ada6c20a0f158e3adb39df2ccac`
  on branch `codex/pi05-aws-repro`.
- Verified Ohio G/VT quota is 64 vCPUs: one `g7e.12xlarge` plus one
  `g6e.4xlarge` exactly. Verified G7e offerings in `us-east-2a` and `us-east-2b`.
- Captured current Linux On-Demand rates from AWS Pricing in
  `repro/reproduction.json`.
- Queried checkpoint metadata only (no payload transfer). The current LIBERO
  teacher inventory is 16 objects / 12,439,085,481 bytes with inventory revision
  `b00d25ec1a1284656ccfd0cf00597fced40fa20c9c7c39ebfdf256db6e844fb7`; the
  DROID/RoboLab joint-position teacher is 26 objects / 12,435,136,033 bytes with
  revision `6487c08461e26cac570a2781f477474e6573c7a6e0a4ba93a9f0efb146c2db5b`.
- No paid instance was launched and no cost reservation was made.
