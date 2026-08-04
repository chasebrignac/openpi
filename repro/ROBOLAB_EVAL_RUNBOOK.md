# Pinned RoboLab evaluation on `g6e.4xlarge`

This is the manual, AWS-only closed-loop bridge for the DROID track. It uses
RoboLab's built-in Pi0-family client without modifying its websocket request or
response. It evaluates exactly `BananaInBowlTask` and
`Stack3RubiksCubeTask`, then turns native `episode_results.jsonl` into
provenance-complete input for `scripts/repro_promotion_report.py`.

Do not run camera-enabled validation on the persistent workbench's base AMI:
its NVIDIA `595.71.05` driver reproduced the known Isaac Sim RTX startup crash
in the untouched Isaac Lab base, while non-camera startup succeeded. Launch
category and workload `evaluation` through `scripts/repro_aws_launch.py`; it
has no arbitrary AMI override and selects the pinned R580 evaluation AMI below.
Record every command or dependency correction in the main run log. Do not add
the procedure to CloudFormation until it has completed two clean abbreviated
replays with no undocumented edit.

## Immutable inputs and fixed evaluation contract

| Input | Required value |
|---|---|
| Host AMI | `ami-06517bc7fad3c6a48` — `Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20260403` |
| Host AMI owner/shape | `amazon` / `898082745236`, `x86_64`, `Linux/UNIX`, HVM, `/dev/sda1` |
| Expected host driver | `580.126.09` (AWS release notes; AMI supports G6e/G7e) |
| RoboLab repository | `https://github.com/NVLabs/RoboLab.git` |
| RoboLab commit | `0aef241fb088ca21bb4ebd24448940ed56620d17` |
| RoboLab release | `0.2.1` |
| Isaac Sim | `5.0.0` |
| Isaac Lab | `2.2.0` |
| Published evaluator image | `752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro@sha256:2d17c15e62887c9fc8b4c41b7ee3d39c4c187348eb55b4273fd24e785a3325e7` |
| RoboLab OpenPI client commit | `aa6420561529593114160d05e5ad155792b272f3` |
| Policy/client variant | `pi05`, open-loop horizon `15` |
| Environment seed | `1`, fixed in the pinned DROID joint-position registration |
| Policy-server seed | `7003`, with a fresh server process for each stage |
| Tasks | `BananaInBowlTask`, `Stack3RubiksCubeTask` |
| Rendering/instruction | realtime, balanced, `default`, headless |
| Intermediate count | exactly `50` episodes/task (`10` envs x `5` runs) |
| Final count | exactly `200` episodes/task (`10` envs x `20` runs) |

Do not use `--num-episodes-adaptive` for acceptance runs: an adaptive stop does
not produce the fixed paired sample count this reproduction requires. If ten
parallel environments do not fit after the policy server is resident, record
the OOM and use `5 x 10` for intermediate or `5 x 40` for final. Do not change
the total episode count.

RoboLab fixes this DROID registration at 15 Hz. Its native result object has
the task, episode identity, success, `ee_path_length`, and `ee_sparc`. The
adapter rejects missing/non-finite metrics and mismatched paired episode
metadata. SPARC is descriptive motion evidence (more negative is smoother),
not a threshold invented by this reproduction; the acceptance condition is at
most a five-point success drop on each task.

Before any full evaluation, run the one-hour host/driver/camera smoke in
read-only plan mode, review its selected `ami_id`, cost, deadline, and command
hash, then add `--execute` to the identical invocation only when authorized:

```bash
python3 scripts/repro_aws_launch.py \
  --category evaluation \
  --workload evaluation \
  --instance-type g6e.4xlarge \
  --hours 1 \
  --label robolab-r580-camera-smoke \
  --scheduler-role-arn arn:aws:iam::752160877725:role/pi05-repro-scheduler-deadline-role \
  --command-file repro/robolab-smoke-worker.sh
```

`repro/robolab-smoke-worker.sh` verifies the host AMI and exact driver, pulls
the digest-pinned evaluator image, checks its provenance labels and dependency
pins, runs the camera-enabled empty-task tests, and uploads the log to encrypted
S3. The paid smoke completed on instance `i-011eb2c219aea0e3e`: all 128 tests
passed with `smoke_exit_code=0`, and the instance entered termination
immediately afterward. The immutable evidence is
`s3://pi05-repro-752160877725-us-east-2/manual-smoke/robolab/20260804T005329Z-i-011eb2c219aea0e3e.log`,
VersionId `AIVks2vssJ5y8yT5.WDeKbiJA0rsvngM`, SHA-256
`dba12538bda077a51f2de816196dfa54d285da47a4fbb8f6cde1d998e6794c5d`.

## Ephemeral base intermediate worker

Use `scripts/repro_robolab_worker.py` for the first policy-connected AWS run.
It deliberately does not relax the generic worker's one-container,
network-disabled contract. Instead, it stages the exact 229c model source and
both versioned teacher manifests, starts the 229c DROID policy server and the
pinned RoboLab evaluator as two named containers, and always removes the
policy container before terminal evidence is published. The worker creates a
uniquely named Docker `--internal` bridge, verifies `Internal=true`, attaches
both containers without publishing a host port, and gives the evaluator the
deterministic policy-container DNS name. The root evaluator therefore cannot
reach IMDS, host-network services, or the public internet. The worker verifies
the network's ID, driver, scope, labels, and empty attachment set before use;
it then verifies removal on both success and failure. The zero-ingress
security group remains mandatory as a separate boundary.

This run does not stage the 259 GB DROID dataset. RoboLab base evaluation needs
only the released joint-position JAX manifest/provenance, its converted BF16
PyTorch checkpoint, the model source bundle, and the two images. The dedicated
spec pins all of those values literally and rejects any edit to the AMI,
driver, task list, seed, horizon, or `10 x 5` episode shape.

First commit the reviewed controller tree and create/publish its exact source
bundle with the create-once procedure in `repro/WORKER_RUNBOOK.md` section 1.
That controller commit is distinct from `model_source`, which remains
`229c08ea2a13a70cbbf1a9c8a1f31cb1ca674dee`. Both must use independently
versioned, fresh-clone/fsck-verified `-complete.bundle` objects; the rejected
legacy `HY8r1...` model bundle is not an admissible pin. Use a lowercase run ID
because the worker's S3 and container identity contract rejects uppercase
characters:

```bash
export ROBOLAB_RUN_ID="robolab-base-$(date -u +%Y%m%dt%H%M%Sz)-a1"
export CONTROLLER_COMMIT="$(git rev-parse HEAD)"
export CONTROLLER_BUNDLE_URI="s3://pi05-repro-752160877725-us-east-2/source/openpi-${CONTROLLER_COMMIT}-complete.bundle"
export CONTROLLER_BUNDLE_VERSION_ID="RECORDED_CREATE_ONCE_VERSION_ID"
export CONTROLLER_BUNDLE_SHA256="RECORDED_BUNDLE_SHA256"
export MODEL_SOURCE_COMMIT="229c08ea2a13a70cbbf1a9c8a1f31cb1ca674dee"
export MODEL_SOURCE_BUNDLE_URI="s3://pi05-repro-752160877725-us-east-2/source/openpi-${MODEL_SOURCE_COMMIT}-complete.bundle"
export MODEL_SOURCE_BUNDLE_VERSION_ID="CN9PJHZ3oHC3hb7lwDTH9p3JEAVQmUhh"
export MODEL_SOURCE_BUNDLE_SHA256="9be1f91dfec636d1cbb63ad87b166e301b98835b91a6212f73fa5b5350d0f7b5"
export ROBOLAB_SPEC="/tmp/${ROBOLAB_RUN_ID}.json"

python3 scripts/repro_robolab_worker.py make-spec \
  --run-id "$ROBOLAB_RUN_ID" \
  --source-s3-uri "$CONTROLLER_BUNDLE_URI" \
  --source-version-id "$CONTROLLER_BUNDLE_VERSION_ID" \
  --source-sha256 "$CONTROLLER_BUNDLE_SHA256" \
  --source-commit "$CONTROLLER_COMMIT" \
  --model-source-s3-uri "$MODEL_SOURCE_BUNDLE_URI" \
  --model-source-version-id "$MODEL_SOURCE_BUNDLE_VERSION_ID" \
  --model-source-sha256 "$MODEL_SOURCE_BUNDLE_SHA256" \
  --model-source-commit "$MODEL_SOURCE_COMMIT" \
  --output "$ROBOLAB_SPEC"

# Pure local schema/command plan: no AWS or Docker mutation.
python3 scripts/repro_robolab_worker.py run --spec "$ROBOLAB_SPEC"
```

Upload the already validated spec once, capture its returned VersionId, and
round-trip that exact version before rendering a command file. A conditional
put alone is insufficient in a versioned bucket because a key with a current
delete marker can still satisfy `If-None-Match`. Therefore, reject all prior
version/delete-marker history before the put and require one latest version
and no delete marker afterward:

```bash
set -euo pipefail
export ROBOLAB_SPEC_SHA256="$(sha256sum "$ROBOLAB_SPEC" | awk '{print $1}')"
export ROBOLAB_SPEC_KEY="specs/${ROBOLAB_RUN_ID}.json"
export ROBOLAB_SPEC_URI="s3://pi05-repro-752160877725-us-east-2/${ROBOLAB_SPEC_KEY}"
SPEC_HISTORY_BEFORE="$(aws s3api list-object-versions \
  --bucket pi05-repro-752160877725-us-east-2 \
  --prefix "$ROBOLAB_SPEC_KEY" --max-keys 10 \
  --region us-east-2 --expected-bucket-owner 752160877725 \
  --output json)"
jq -e --arg key "$ROBOLAB_SPEC_KEY" '
  (.IsTruncated == false) and
  ([.Versions[]? | select(.Key == $key)] | length == 0) and
  ([.DeleteMarkers[]? | select(.Key == $key)] | length == 0)
' <<<"$SPEC_HISTORY_BEFORE" >/dev/null

SPEC_PUT_JSON="$(aws s3api put-object \
  --bucket pi05-repro-752160877725-us-east-2 \
  --key "$ROBOLAB_SPEC_KEY" --body "$ROBOLAB_SPEC" \
  --region us-east-2 --expected-bucket-owner 752160877725 \
  --server-side-encryption AES256 --if-none-match '*' \
  --metadata "sha256=${ROBOLAB_SPEC_SHA256}" --output json)"
test "$(jq -er '.ServerSideEncryption' <<<"$SPEC_PUT_JSON")" = AES256
export ROBOLAB_SPEC_VERSION_ID="$(jq -er '.VersionId' <<<"$SPEC_PUT_JSON")"

SPEC_HISTORY_AFTER="$(aws s3api list-object-versions \
  --bucket pi05-repro-752160877725-us-east-2 \
  --prefix "$ROBOLAB_SPEC_KEY" --max-keys 10 \
  --region us-east-2 --expected-bucket-owner 752160877725 \
  --output json)"
jq -e --arg key "$ROBOLAB_SPEC_KEY" --arg version "$ROBOLAB_SPEC_VERSION_ID" '
  (.IsTruncated == false) and
  ([.Versions[]? | select(.Key == $key)] |
    length == 1 and .[0].VersionId == $version and .[0].IsLatest == true) and
  ([.DeleteMarkers[]? | select(.Key == $key)] | length == 0)
' <<<"$SPEC_HISTORY_AFTER" >/dev/null

export ROBOLAB_SPEC_BYTES="$(wc -c < "$ROBOLAB_SPEC" | tr -d ' ')"
SPEC_HEAD="$(aws s3api head-object \
  --bucket pi05-repro-752160877725-us-east-2 --key "$ROBOLAB_SPEC_KEY" \
  --version-id "$ROBOLAB_SPEC_VERSION_ID" \
  --region us-east-2 --expected-bucket-owner 752160877725 \
  --output json)"
jq -e --arg version "$ROBOLAB_SPEC_VERSION_ID" \
  --arg sha256 "$ROBOLAB_SPEC_SHA256" --argjson bytes "$ROBOLAB_SPEC_BYTES" '
  .VersionId == $version and .ServerSideEncryption == "AES256" and
  .ContentLength == $bytes and .Metadata.sha256 == $sha256
' <<<"$SPEC_HEAD" >/dev/null

SPEC_ROUNDTRIP="$(mktemp /tmp/robolab-spec-roundtrip.XXXXXX)"
aws s3api get-object \
  --bucket pi05-repro-752160877725-us-east-2 --key "$ROBOLAB_SPEC_KEY" \
  --version-id "$ROBOLAB_SPEC_VERSION_ID" \
  --region us-east-2 --expected-bucket-owner 752160877725 \
  "$SPEC_ROUNDTRIP" >/dev/null
test "$(sha256sum "$SPEC_ROUNDTRIP" | awk '{print $1}')" = \
  "$ROBOLAB_SPEC_SHA256"
rm -f -- "$SPEC_ROUNDTRIP"
```

There are two independent execution gates. Render the non-executing bootstrap
and run the launcher without `--execute` first. Review the selected evaluation
AMI, On-Demand purchase option, four-hour deadline, command hash, and projected
reservation. Four requested hours reserve 4.25 billed hours, or approximately
`$12.768` at the pinned `g6e.4xlarge` rate; the instance terminates as soon as
the worker finishes.

```bash
export ROBOLAB_COMMAND="/tmp/${ROBOLAB_RUN_ID}.command.sh"
python3 scripts/repro_robolab_worker.py render-bootstrap \
  --spec-s3-uri "$ROBOLAB_SPEC_URI" \
  --spec-version-id "$ROBOLAB_SPEC_VERSION_ID" \
  --spec-sha256 "$ROBOLAB_SPEC_SHA256" \
  > "$ROBOLAB_COMMAND"

python3 scripts/repro_aws_launch.py \
  --category evaluation --workload evaluation \
  --instance-type g6e.4xlarge --hours 4 \
  --label "$ROBOLAB_RUN_ID" \
  --scheduler-role-arn arn:aws:iam::752160877725:role/pi05-repro-scheduler-deadline-role \
  --command-file "$ROBOLAB_COMMAND"
```

Only after that nested dry plan is accepted, replace the command file with the
execution-enabled bootstrap. Dry-plan the launcher again with this exact final
file and review its new command hash before adding the launcher's separate
`--execute` flag. Do not edit or re-render the command file between the final
launcher plan and paid launch:

```bash
python3 scripts/repro_robolab_worker.py render-bootstrap \
  --spec-s3-uri "$ROBOLAB_SPEC_URI" \
  --spec-version-id "$ROBOLAB_SPEC_VERSION_ID" \
  --spec-sha256 "$ROBOLAB_SPEC_SHA256" --execute \
  > "$ROBOLAB_COMMAND"
export ROBOLAB_EXECUTE_COMMAND_SHA256="$(sha256sum "$ROBOLAB_COMMAND" | awk '{print $1}')"

# This second launcher plan covers the exact execution-enabled command file.
python3 scripts/repro_aws_launch.py \
  --category evaluation --workload evaluation \
  --instance-type g6e.4xlarge --hours 4 \
  --label "$ROBOLAB_RUN_ID" \
  --scheduler-role-arn arn:aws:iam::752160877725:role/pi05-repro-scheduler-deadline-role \
  --command-file "$ROBOLAB_COMMAND"

# Launch the unchanged command file only after recording that final plan.
test "$(sha256sum "$ROBOLAB_COMMAND" | awk '{print $1}')" = \
  "$ROBOLAB_EXECUTE_COMMAND_SHA256"
python3 scripts/repro_aws_launch.py \
  --category evaluation --workload evaluation \
  --instance-type g6e.4xlarge --hours 4 \
  --label "$ROBOLAB_RUN_ID" \
  --scheduler-role-arn arn:aws:iam::752160877725:role/pi05-repro-scheduler-deadline-role \
  --command-file "$ROBOLAB_COMMAND" --execute
```

Success produces `100` native episodes, the sealed `run-identity.json`, a
task-level success/motion summary, both exact input manifests, image/runtime
evidence, copied launch controls, container logs, a run manifest, and final
sync evidence under `s3://pi05-repro-752160877725-us-east-2/runs/$ROBOLAB_RUN_ID/`.
Payload receipts are complete before `run-manifest.json` is uploaded; final
sync evidence binds that manifest's S3 VersionId. Failures before scratch is
available use a separately prepared root-volume output manager so they still
produce a failed terminal manifest. During evaluation, the worker publishes
only complete ten-environment run batches as content-addressed, conditionally
created snapshots. Each snapshot is proven to have exactly one S3 version and
no delete marker and binds the evaluator image, policy-server image/source/
config/command, fixed evaluation contract, parent run ID, record count, and
canonical record hash.

Never reuse a failed run ID or its local/S3 output folder. A continuation is a
new run with a new spec that pins one exact parent snapshot by S3 URI,
VersionId, and SHA-256. RoboLab's pinned runner skips exactly the restored
complete `(task, run)` batches; the controller rejects incomplete or
out-of-order continuation state, so a partially written batch is rerun without
recounting its incomplete records:

```bash
export ROBOLAB_PARENT_RUN_ID="RECORDED_FAILED_RUN_ID"
export ROBOLAB_PARTIAL_S3_URI="RECORDED_PARTIAL_SNAPSHOT_S3_URI"
export ROBOLAB_PARTIAL_VERSION_ID="RECORDED_PARTIAL_SNAPSHOT_VERSION_ID"
export ROBOLAB_PARTIAL_SHA256="RECORDED_PARTIAL_SNAPSHOT_SHA256"
export ROBOLAB_RUN_ID="robolab-base-$(date -u +%Y%m%dt%H%M%Sz)-a2"
export ROBOLAB_SPEC="/tmp/${ROBOLAB_RUN_ID}.json"

python3 scripts/repro_robolab_worker.py make-spec \
  --run-id "$ROBOLAB_RUN_ID" \
  --source-s3-uri "$CONTROLLER_BUNDLE_URI" \
  --source-version-id "$CONTROLLER_BUNDLE_VERSION_ID" \
  --source-sha256 "$CONTROLLER_BUNDLE_SHA256" \
  --source-commit "$CONTROLLER_COMMIT" \
  --model-source-s3-uri "$MODEL_SOURCE_BUNDLE_URI" \
  --model-source-version-id "$MODEL_SOURCE_BUNDLE_VERSION_ID" \
  --model-source-sha256 "$MODEL_SOURCE_BUNDLE_SHA256" \
  --model-source-commit "$MODEL_SOURCE_COMMIT" \
  --continuation-parent-run-id "$ROBOLAB_PARENT_RUN_ID" \
  --continuation-s3-uri "$ROBOLAB_PARTIAL_S3_URI" \
  --continuation-version-id "$ROBOLAB_PARTIAL_VERSION_ID" \
  --continuation-sha256 "$ROBOLAB_PARTIAL_SHA256" \
  --output "$ROBOLAB_SPEC"
```

Run the same exact-history spec-publication and two dry-plan gates above for
the child. If ten environments OOM, preserve the failed prefix and add a
separately reviewed exact `5 x 10` contract; do not hand-edit the accepted
`10 x 5` spec or evidence.

## 1. Manually build the pinned evaluator image

Use the existing SSM-only `g6e.4xlarge` workbench with the 1 TiB encrypted gp3
volume. No inbound rule or published host port is needed because the evaluator
and server communicate only on a per-run internal Docker bridge.

```bash
export ROBOLAB_COMMIT=0aef241fb088ca21bb4ebd24448940ed56620d17
export ROBOLAB_CLIENT_COMMIT=aa6420561529593114160d05e5ad155792b272f3
export ROBOLAB_SRC=/mnt/openpi/src/robolab
export ROBOLAB_OUTPUT=/mnt/openpi/evidence/robolab
export POLICY_SEED=7003

git clone https://github.com/NVLabs/RoboLab.git "$ROBOLAB_SRC"
git -C "$ROBOLAB_SRC" checkout --detach "$ROBOLAB_COMMIT"
git -C "$ROBOLAB_SRC" lfs pull
git -C "$ROBOLAB_SRC" lfs fsck
test "$(git -C "$ROBOLAB_SRC" rev-parse HEAD)" = "$ROBOLAB_COMMIT"
test -z "$(git -C "$ROBOLAB_SRC" status --porcelain)"
mkdir -p "$ROBOLAB_OUTPUT"
```

Pull the already resolved base digest and build from that immutable identity.
The unusual-looking `2.2.0@sha256:...` build argument is intentional: the
pinned Dockerfile expands it after `nvcr.io/nvidia/isaac-lab:`.

```bash
export ISAACLAB_DIGEST=sha256:b4d8e96cbfb9a6c40067bec6cc5ee180e36d4c0164b25f7215c5f47e31897b94
export ISAACLAB_IMAGE="nvcr.io/nvidia/isaac-lab@$ISAACLAB_DIGEST"
docker pull "$ISAACLAB_IMAGE"
docker image inspect "$ISAACLAB_IMAGE" >/dev/null

docker build \
  --build-arg "ISAACLAB_TAG=2.2.0@$ISAACLAB_DIGEST" \
  --build-arg "ROBOLAB_COMMIT=$ROBOLAB_COMMIT" \
  --build-arg "OPENPI_COMMIT=$ROBOLAB_CLIENT_COMMIT" \
  --file repro/Dockerfile.robolab \
  --tag "robolab:$ROBOLAB_COMMIT" "$ROBOLAB_SRC"

export ROBOLAB_IMAGE="robolab:$ROBOLAB_COMMIT"
export ROBOLAB_IMAGE_DIGEST="$(docker image inspect --format '{{.Id}}' "$ROBOLAB_IMAGE")"
test "${ROBOLAB_IMAGE_DIGEST#sha256:}" != "$ROBOLAB_IMAGE_DIGEST"
```

Verify the GPU, the runtime's authoritative version files, task registry, and
one empty headless episode before connecting a model. The Isaac Lab release's
Python distribution has an independent internal version (`0.44.8` here), and
Isaac Sim is bundled as Kit without `isaacsim` package metadata, so neither is
a valid release gate. `--video-mode none` is used for all acceptance runs;
RoboLab still writes HDF5 trajectories needed for path and SPARC metrics.

```bash
nvidia-smi
test "$(docker run --rm --entrypoint sed "$ROBOLAB_IMAGE" \
  -n 1p /workspace/isaaclab/VERSION)" = "2.2.0"
docker run --rm --entrypoint sed "$ROBOLAB_IMAGE" \
  -n 1p /workspace/isaaclab/_isaac_sim/VERSION | grep '^5\.0\.0-'

docker run --rm --gpus all --network none --ipc host \
  --entrypoint /workspace/isaaclab/_isaac_sim/python.sh \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES \
  "$ROBOLAB_IMAGE" -m pytest -q \
  tests/test_isaaclab.py tests/test_registered_envs.py tests/test_tasks_valid.py tests/test_run_empty.py
```

The required release values are Isaac Lab `2.2.0` and an Isaac Sim build whose
version begins `5.0.0-`. Stop and investigate rather than sealing evidence if
either version file differs.

`repro/Dockerfile.robolab` deliberately restores `typing_extensions==4.12.2`
after RoboLab installation. Leaving the dependency unconstrained upgrades it
to 4.16.0, which is incompatible with the base image's Torch 2.7.0
`torch._dynamo` import path and aborts Isaac Lab startup.

## 2. Common server and evaluator commands

All accepted checkpoints have a `model.safetensors`; that exact file is what
the offline evaluator and RoboLab sealing command hash. Use the joint-position
released teacher, not the generic DROID checkpoint:

```bash
export BASE_CONFIG=pi05_droid_jointpos
export BASE_CHECKPOINT=/mnt/openpi/checkpoints/pi05_droid_jointpos_pytorch
export SHALLOW_CONFIG=pi05_droid_l09_distill
export SHALLOW_CHECKPOINT=/mnt/openpi/runs/pi05_droid_l09_distill/droid-shallow/10000
export FINAL_CONFIG=pi05_droid_l09_snapflow
export FINAL_CHECKPOINT=/mnt/openpi/runs/pi05_droid_l09_snapflow/droid-snapflow/5000
```

Replace the example numeric steps only with checkpoints accepted by the
offline gate. If expert-BC recovery was selected, set `SHALLOW_CONFIG` to the
actual `pi05_droid_l09_expert_bc_25` or `pi05_droid_l09_expert_bc_50` config
and `SHALLOW_CHECKPOINT` to that exact numeric checkpoint. It must be the same
model hash recorded as the SnapFlow run's initialization source and in the
offline SnapFlow report. For manual debugging, create one uniquely named
internal bridge and put the policy server in a non-root container on that
bridge. Never expose port 8000 on the host. The accepted ephemeral worker
performs these checks and cleanup automatically; the shell below mirrors its
network boundary.

```bash
export OPENPI_SOURCE="$(git rev-parse --show-toplevel)"
export POLICY_IMAGE="752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro@sha256:2afcc58cda27681892c7bbb9554e9603024c5b74f53358fad893ea876374803c"
export POLICY_SOURCE_COMMIT="$(git -C "$OPENPI_SOURCE" rev-parse HEAD)"
export POLICY_SOURCE_S3_URI="s3://pi05-repro-752160877725-us-east-2/source/openpi-${POLICY_SOURCE_COMMIT}-complete.bundle"
export POLICY_SOURCE_VERSION_ID="RECORDED_CREATE_ONCE_POLICY_SOURCE_VERSION_ID"
export POLICY_SOURCE_SHA256="RECORDED_POLICY_SOURCE_BUNDLE_SHA256"
export MANUAL_RUN_ID="robolab-manual-$(date -u +%Y%m%dt%H%M%Sz)"
export ROBOLAB_NETWORK="pi05-${MANUAL_RUN_ID}-network"
export POLICY_CONTAINER="pi05-${MANUAL_RUN_ID}-policy"

test -z "$(git -C "$OPENPI_SOURCE" status --porcelain --untracked-files=all)"
test "$(git -C "$OPENPI_SOURCE" rev-parse HEAD)" = "$POLICY_SOURCE_COMMIT"
test "$POLICY_SOURCE_S3_URI" = \
  "s3://pi05-repro-752160877725-us-east-2/source/openpi-${POLICY_SOURCE_COMMIT}-complete.bundle"
POLICY_SOURCE_BUNDLE="$(mktemp /tmp/robolab-policy-source.XXXXXX)"
aws s3api get-object \
  --bucket pi05-repro-752160877725-us-east-2 \
  --key "source/openpi-${POLICY_SOURCE_COMMIT}-complete.bundle" \
  --version-id "$POLICY_SOURCE_VERSION_ID" \
  --region us-east-2 --expected-bucket-owner 752160877725 \
  "$POLICY_SOURCE_BUNDLE" >/dev/null
test "$(sha256sum "$POLICY_SOURCE_BUNDLE" | awk '{print $1}')" = \
  "$POLICY_SOURCE_SHA256"
test "$(git bundle list-heads "$POLICY_SOURCE_BUNDLE" HEAD | \
  awk '$2 == "HEAD" {print $1}')" = "$POLICY_SOURCE_COMMIT"
rm -f -- "$POLICY_SOURCE_BUNDLE"
test -z "$(docker network ls --filter "name=^${ROBOLAB_NETWORK}$" --format '{{.Name}}')"
docker network create --driver bridge --internal \
  --label ai.openpi.project=pi05-aws-repro \
  --label "ai.openpi.run-id=${MANUAL_RUN_ID}" \
  "$ROBOLAB_NETWORK"
test "$(docker network inspect --format '{{.Internal}} {{.Driver}} {{.Scope}}' \
  "$ROBOLAB_NETWORK")" = "true bridge local"
```

Start each stage with the same seed so the inference-noise stream restarts for
paired evaluation. Change only the accepted config and checkpoint arguments:

```bash
start_policy () {
  local config="$1"
  local checkpoint="$2"
  test -z "$(docker container ls -a --filter "name=^${POLICY_CONTAINER}$" --format '{{.Names}}')"
  docker run --detach --name "$POLICY_CONTAINER" --gpus all \
    --network "$ROBOLAB_NETWORK" --network-alias "$POLICY_CONTAINER" \
    --ipc host --user 1000:1000 --workdir /workspace/openpi \
    --mount "type=bind,src=${OPENPI_SOURCE},dst=/workspace/openpi,readonly" \
    --mount type=bind,src=/mnt/openpi,dst=/mnt/openpi,readonly \
    --tmpfs /tmp:rw,exec,nosuid,size=16g \
    --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/cache \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env PYTHONPATH=/workspace/openpi/src:/workspace/openpi \
    --env CUDA_VISIBLE_DEVICES=0 --env XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
    "$POLICY_IMAGE" python scripts/serve_policy.py \
    --env DROID --port 8000 --seed "$POLICY_SEED" \
    policy:checkpoint --policy.config "$config" --policy.dir "$checkpoint"

  for _ in $(seq 1 120); do
    if docker exec "$POLICY_CONTAINER" python -c \
      "import urllib.request;r=urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=1);assert r.status==200 and r.read()==b'OK\\n'"; then
      return 0
    fi
    test "$(docker inspect --format '{{.State.Running}}' "$POLICY_CONTAINER")" = true
    sleep 5
  done
  return 1
}

stop_policy () {
  docker stop --time 30 "$POLICY_CONTAINER"
  docker logs --timestamps "$POLICY_CONTAINER" > \
    "$ROBOLAB_OUTPUT/${POLICY_CONTAINER}.log" 2>&1
  docker rm --force "$POLICY_CONTAINER"
}
```

The evaluator command is deliberately the unmodified public entry point. This
function varies only the output name and fixed run count:

```bash
run_robolab () {
  local output_name="$1"
  local num_runs="$2"
  docker run --rm --gpus all --network "$ROBOLAB_NETWORK" --ipc host \
    --entrypoint /workspace/isaaclab/_isaac_sim/python.sh \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES \
    -v "$ROBOLAB_OUTPUT:/workspace/robolab/output" \
    "$ROBOLAB_IMAGE" policies/pi0_family/run.py \
    --policy pi05 --remote-host "$POLICY_CONTAINER" --remote-port 8000 \
    --open-loop-horizon 15 \
    --task BananaInBowlTask Stack3RubiksCubeTask \
    --task-dirs benchmark --instruction-type default \
    --num-envs 10 --num-runs "$num_runs" \
    --renderer realtime --rendering-type balanced \
    --video-mode none --device cuda:0 --headless \
    --output-folder-name "$output_name"
}
```

Run these exact stage/count combinations, restarting the matching server before
each call:

```bash
# Base and Shallow intermediate pair: 50 episodes/task/stage.
start_policy "$BASE_CONFIG" "$BASE_CHECKPOINT"
run_robolab base-intermediate 5
stop_policy
start_policy "$SHALLOW_CONFIG" "$SHALLOW_CHECKPOINT"
run_robolab shallow-intermediate 5
stop_policy

# Accepted Shallow and SnapFlow intermediate pair. The prior Shallow result may
# be reused only when it is the exact checkpoint/seed used as SnapFlow teacher.
start_policy "$FINAL_CONFIG" "$FINAL_CHECKPOINT"
run_robolab snapflow-intermediate 5
stop_policy

# Fresh base and final pair: 200 episodes/task/stage.
start_policy "$BASE_CONFIG" "$BASE_CHECKPOINT"
run_robolab base-final 20
stop_policy
start_policy "$FINAL_CONFIG" "$FINAL_CHECKPOINT"
run_robolab final-final 20
stop_policy

test "$(docker network inspect --format '{{json .Containers}}' "$ROBOLAB_NETWORK")" = '{}'
docker network rm "$ROBOLAB_NETWORK"
test -z "$(docker network ls --filter "name=^${ROBOLAB_NETWORK}$" --format '{{.Name}}')"
```

Only a retained manual-debug directory may be resumed in place, and only with
the identical model, source, image, seed, command, and output folder; RoboLab
then skips complete episode identities. An ephemeral accepted run never
reuses or copies that mutable directory: use the immutable complete-batch
snapshot/new-run continuation contract above. Never reuse a folder for another
model, and never use `--disable-subtask`.

## 3. Seal native output before comparison

The result file itself does not contain a checkpoint hash. Seal it immediately
beside the JSONL. Sealing validates exact tasks, exact episode identities and
counts, `pi05`, default instructions, 15 Hz, and finite path/SPARC; it then
binds the result hash to the actual `model.safetensors` hash, evaluator image,
policy-server image, exact versioned source bundle, config, command hash, and
policy checkpoint hash. Any later byte or runtime-identity change invalidates
the sidecar and promotion report.

```bash
policy_command_sha256 () {
  python3 - "$1" "$2" "$POLICY_SEED" <<'PY'
import hashlib,json,sys
config,checkpoint,seed=sys.argv[1:]
argv=[
    "python","scripts/serve_policy.py","--env","DROID","--port","8000",
    "--seed",seed,"policy:checkpoint","--policy.config",config,
    "--policy.dir",checkpoint,
]
print(hashlib.sha256(json.dumps(argv,separators=(",",":"),sort_keys=True).encode()).hexdigest())
PY
}

seal_robolab () {
  local stage="$1"
  local mode="$2"
  local config="$3"
  local checkpoint="$4"
  local output_name="$5"
  local num_runs="$6"
  local policy_command_sha256_value
  policy_command_sha256_value="$(policy_command_sha256 "$config" "$checkpoint")"
  python scripts/repro_robolab_report.py seal \
    --stage "$stage" --mode "$mode" \
    --checkpoint-model "$checkpoint/model.safetensors" \
    --results "$ROBOLAB_OUTPUT/$output_name/episode_results.jsonl" \
    --num-envs 10 --num-runs "$num_runs" --policy-server-seed "$POLICY_SEED" \
    --image-digest "$ROBOLAB_IMAGE_DIGEST" --robolab-git-sha "$ROBOLAB_COMMIT" \
    --policy-image-digest "$POLICY_IMAGE" \
    --policy-source-s3-uri "$POLICY_SOURCE_S3_URI" \
    --policy-source-version-id "$POLICY_SOURCE_VERSION_ID" \
    --policy-source-sha256 "$POLICY_SOURCE_SHA256" \
    --policy-source-commit "$POLICY_SOURCE_COMMIT" \
    --policy-config "$config" \
    --policy-command-sha256 "$policy_command_sha256_value" \
    --output "$ROBOLAB_OUTPUT/$output_name/run-identity.json"
}

seal_robolab base intermediate "$BASE_CONFIG" "$BASE_CHECKPOINT" base-intermediate 5
seal_robolab shallow intermediate "$SHALLOW_CONFIG" "$SHALLOW_CHECKPOINT" shallow-intermediate 5
seal_robolab snapflow intermediate "$FINAL_CONFIG" "$FINAL_CHECKPOINT" snapflow-intermediate 5
seal_robolab base final "$BASE_CONFIG" "$BASE_CHECKPOINT" base-final 20
seal_robolab final final "$FINAL_CONFIG" "$FINAL_CHECKPOINT" final-final 20
```

## 4. Emit promotion evidence

For Shallow, the offline teacher hash must equal the sealed base hash. For
SnapFlow intermediate promotion, the offline teacher hash must equal the sealed
accepted-Shallow hash. The adapter copies the complete offline provenance into
its output, so `repro_promotion_report.py` accepts it without hand-written
quality JSON.

```bash
export SHALLOW_OFFLINE=/mnt/openpi/evidence/droid-shallow-10000.json
export SNAPFLOW_OFFLINE=/mnt/openpi/evidence/droid-snapflow-5000.json

python scripts/repro_robolab_report.py report \
  --reference-identity "$ROBOLAB_OUTPUT/base-intermediate/run-identity.json" \
  --candidate-identity "$ROBOLAB_OUTPUT/shallow-intermediate/run-identity.json" \
  --offline-report "$SHALLOW_OFFLINE" \
  --expected-reference-stage base --expected-candidate-stage shallow \
  --max-task-success-gap 0.05 \
  --output "$ROBOLAB_OUTPUT/shallow-intermediate/evidence.json"

python scripts/repro_robolab_report.py report \
  --reference-identity "$ROBOLAB_OUTPUT/shallow-intermediate/run-identity.json" \
  --candidate-identity "$ROBOLAB_OUTPUT/snapflow-intermediate/run-identity.json" \
  --offline-report "$SNAPFLOW_OFFLINE" \
  --expected-reference-stage shallow --expected-candidate-stage snapflow \
  --max-task-success-gap 0.05 --denoise-speedup 8.4 \
  --output "$ROBOLAB_OUTPUT/snapflow-intermediate/evidence.json"
```

Replace `8.4` only with the measured same-instance ten-step/one-step denoising
speedup report. The report command exits `2` when either task loses more than
five success points. A failed report retains the observations for diagnosis but
omits the `paired_rollout` field, so the existing promotion gate treats it as
missing required evidence. Invalid provenance, counts, hashes, tasks, episode
pairs, or motion metrics raise an error and produce no accepted evidence.

For the final 200/task comparison, the reference is the released base rather
than the immediate SnapFlow teacher. Make that exception explicit by supplying
the base file hash; the candidate must still equal the offline student hash:

```bash
export BASE_MODEL_SHA256="$(sha256sum "$BASE_CHECKPOINT/model.safetensors" | awk '{print $1}')"

python scripts/repro_robolab_report.py report \
  --reference-identity "$ROBOLAB_OUTPUT/base-final/run-identity.json" \
  --candidate-identity "$ROBOLAB_OUTPUT/final-final/run-identity.json" \
  --offline-report "$SNAPFLOW_OFFLINE" \
  --expected-reference-stage base --expected-candidate-stage final \
  --reference-model-sha256 "$BASE_MODEL_SHA256" \
  --max-task-success-gap 0.05 --denoise-speedup 8.4 \
  --output "$ROBOLAB_OUTPUT/final-final/evidence.json"
```

Use one RoboLab evidence file for a DROID checkpoint in the existing promotion
gate (intermediate while selecting a checkpoint, final when accepting the
finished checkpoint):

```bash
python scripts/repro_promotion_report.py --stage snapflow \
  --offline "$SNAPFLOW_OFFLINE" \
  --quality "$ROBOLAB_OUTPUT/final-final/evidence.json" \
  --max-rollout-gap 0.05 \
  --output "$ROBOLAB_OUTPUT/final-final/promotion.json"
```

The evidence preserves per-task success plus distribution summaries and paired
mean deltas for end-effector path length and SPARC. Inspect those motion
statistics even when success passes; they are useful regression signals but
are not silently promoted to thresholds after seeing the result.

## 5. What this proves, archive, and replay

This closed-loop run exercises the accepted PyTorch base/Shallow/SnapFlow
checkpoint through the exact public RoboLab client. BF16/FP8 ONNX and TensorRT
equivalence and latency remain the separately hashed gates in
`repro/EXPORT_RUNBOOK.md`. Until an engine-backed implementation of the same
OpenPI server API exists, do not claim that RoboLab directly exercised the
TensorRT plan; the combined evidence is checkpoint closed-loop quality plus
PyTorch-to-ONNX/FP8 numerical equivalence.

Upload each complete result directory (native JSONL, sidecar, HDF5, report,
logs, and optional videos) to the run's versioned S3 prefix. Record object
version IDs and SHA-256 values before ending the instance. A clean replay must:

1. start from the recorded RoboLab and base-image digests;
2. use a fresh server with seed `7003` for each stage;
3. use exactly the two named tasks and fixed count;
4. seal without a manual JSON edit; and
5. regenerate the same structural evidence contract (stochastic success need
   not be bit-identical across GPU-driver changes, which is why the image and
   driver details belong in the run manifest).

After two clean abbreviated replays (`10` episodes/task, marked smoke-only and
never passed to the acceptance adapter), translate the documented instance,
mount, image, logging, and shutdown mechanics into CloudFormation. Full
training or 50/200-episode evaluation is not repeated during that
CloudFormation replay.
