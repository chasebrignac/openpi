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

## 1. Manually build the pinned evaluator image

Use the existing SSM-only `g6e.4xlarge` workbench with the 1 TiB encrypted gp3
volume. No inbound rule is needed because the evaluator and server communicate
over host-local port 8000.

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

docker build --network host \
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

docker run --rm --gpus all --network host --ipc host \
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
offline gate. Start each stage in a fresh terminal (or a fresh tmux pane) and
wait until port 8000 is listening. Keeping the seed equal restarts the model's
inference-noise stream for paired evaluation.

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
  .venv/bin/python scripts/serve_policy.py \
  --env DROID --port 8000 --seed "$POLICY_SEED" \
  policy:checkpoint --policy.config "$BASE_CONFIG" --policy.dir "$BASE_CHECKPOINT"
```

For the Shallow or final SnapFlow stage, stop the previous server and change
only `--policy.config` and `--policy.dir`:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
  .venv/bin/python scripts/serve_policy.py \
  --env DROID --port 8000 --seed "$POLICY_SEED" \
  policy:checkpoint --policy.config "$SHALLOW_CONFIG" --policy.dir "$SHALLOW_CHECKPOINT"

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
  .venv/bin/python scripts/serve_policy.py \
  --env DROID --port 8000 --seed "$POLICY_SEED" \
  policy:checkpoint --policy.config "$FINAL_CONFIG" --policy.dir "$FINAL_CHECKPOINT"
```

The evaluator command is deliberately the unmodified public entry point. This
function varies only the output name and fixed run count:

```bash
run_robolab () {
  local output_name="$1"
  local num_runs="$2"
  docker run --rm --gpus all --network host --ipc host \
    --entrypoint /workspace/isaaclab/_isaac_sim/python.sh \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES \
    -v "$ROBOLAB_OUTPUT:/workspace/robolab/output" \
    "$ROBOLAB_IMAGE" policies/pi0_family/run.py \
    --policy pi05 --remote-host 127.0.0.1 --remote-port 8000 \
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
run_robolab base-intermediate 5
run_robolab shallow-intermediate 5

# Accepted Shallow and SnapFlow intermediate pair. The prior Shallow result may
# be reused only when it is the exact checkpoint/seed used as SnapFlow teacher.
run_robolab snapflow-intermediate 5

# Fresh base and final pair: 200 episodes/task/stage.
run_robolab base-final 20
run_robolab final-final 20
```

An interrupted run can be resumed with the identical output folder, model,
seed, and command. RoboLab skips completed episode identities. Never reuse a
folder for another model, and never use `--disable-subtask`.

## 3. Seal native output before comparison

The result file itself does not contain a checkpoint hash. Seal it immediately
beside the JSONL. Sealing validates exact tasks, exact episode identities and
counts, `pi05`, default instructions, 15 Hz, and finite path/SPARC; it then
binds the result hash to the actual `model.safetensors` hash and the evaluator
image digest. Any later byte change invalidates the sidecar.

```bash
python scripts/repro_robolab_report.py seal \
  --stage base --mode intermediate \
  --checkpoint-model "$BASE_CHECKPOINT/model.safetensors" \
  --results "$ROBOLAB_OUTPUT/base-intermediate/episode_results.jsonl" \
  --num-envs 10 --num-runs 5 --policy-server-seed "$POLICY_SEED" \
  --image-digest "$ROBOLAB_IMAGE_DIGEST" --robolab-git-sha "$ROBOLAB_COMMIT" \
  --output "$ROBOLAB_OUTPUT/base-intermediate/run-identity.json"

python scripts/repro_robolab_report.py seal \
  --stage shallow --mode intermediate \
  --checkpoint-model "$SHALLOW_CHECKPOINT/model.safetensors" \
  --results "$ROBOLAB_OUTPUT/shallow-intermediate/episode_results.jsonl" \
  --num-envs 10 --num-runs 5 --policy-server-seed "$POLICY_SEED" \
  --image-digest "$ROBOLAB_IMAGE_DIGEST" --robolab-git-sha "$ROBOLAB_COMMIT" \
  --output "$ROBOLAB_OUTPUT/shallow-intermediate/run-identity.json"

python scripts/repro_robolab_report.py seal \
  --stage snapflow --mode intermediate \
  --checkpoint-model "$FINAL_CHECKPOINT/model.safetensors" \
  --results "$ROBOLAB_OUTPUT/snapflow-intermediate/episode_results.jsonl" \
  --num-envs 10 --num-runs 5 --policy-server-seed "$POLICY_SEED" \
  --image-digest "$ROBOLAB_IMAGE_DIGEST" --robolab-git-sha "$ROBOLAB_COMMIT" \
  --output "$ROBOLAB_OUTPUT/snapflow-intermediate/run-identity.json"

python scripts/repro_robolab_report.py seal \
  --stage base --mode final \
  --checkpoint-model "$BASE_CHECKPOINT/model.safetensors" \
  --results "$ROBOLAB_OUTPUT/base-final/episode_results.jsonl" \
  --num-envs 10 --num-runs 20 --policy-server-seed "$POLICY_SEED" \
  --image-digest "$ROBOLAB_IMAGE_DIGEST" --robolab-git-sha "$ROBOLAB_COMMIT" \
  --output "$ROBOLAB_OUTPUT/base-final/run-identity.json"

python scripts/repro_robolab_report.py seal \
  --stage final --mode final \
  --checkpoint-model "$FINAL_CHECKPOINT/model.safetensors" \
  --results "$ROBOLAB_OUTPUT/final-final/episode_results.jsonl" \
  --num-envs 10 --num-runs 20 --policy-server-seed "$POLICY_SEED" \
  --image-digest "$ROBOLAB_IMAGE_DIGEST" --robolab-git-sha "$ROBOLAB_COMMIT" \
  --output "$ROBOLAB_OUTPUT/final-final/run-identity.json"
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
