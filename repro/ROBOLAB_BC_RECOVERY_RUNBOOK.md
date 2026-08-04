# Conditional RoboLab expert-BC recovery

This is the bounded recovery path for the DROID Shallow model. It is dormant
unless the accepted Shallow checkpoint trails the released teacher by more
than five success-rate points on `Stack3RubiksCubeTask`. It never launches AWS
capacity and must be run manually before it is represented in CloudFormation.

The public RoboLab example HDF5 at pinned commit
`0aef241fb088ca21bb4ebd24448940ed56620d17` contains actions and simulator
state but no policy observations. It cannot train a VLA. Recovery therefore
requires a new teacher collection with RoboLab's native
`--record-image-data` flag. The sealer accepts only these two complete native
layouts and rejects missing, mixed, or inferred fields:

- Current nested layout: `obs/image_obs/{over_shoulder_left_camera,wrist_cam}`
  and `obs/proprio_obs/{arm_joint_pos,gripper_pos}`.
- Pinned documented flat layout: `obs/{over_shoulder_left_camera,wrist_cam,
  arm_joint_pos,gripper_pos}`.

Both require HWC `uint8` images, seven joint positions, one gripper position,
and eight absolute joint-position-plus-gripper actions at a common 15 Hz frame
count. Do not substitute simulator `states` for the policy observations.

Record all manual corrections in the main run log. Replay this procedure
twice without undocumented edits before changing CloudFormation.

## 1. Prove that recovery is necessary

Start from the paired Shallow-vs-base evidence produced by
`repro/ROBOLAB_EVAL_RUNBOOK.md` and the exact accepted numeric checkpoint.

```bash
export RUNS=/mnt/openpi/runs
export EVIDENCE=/mnt/openpi/evidence
export ROBOLAB_OUTPUT=/mnt/openpi/evidence/robolab
export SHALLOW_REPORT="$ROBOLAB_OUTPUT/shallow-intermediate/evidence.json"
export ACCEPTED_SHALLOW="$RUNS/pi05_droid_l09_distill/droid-shallow/30000"
export DROID_TEACHER=/mnt/openpi/checkpoints/pi05_droid_jointpos_pytorch

python scripts/repro_robolab_bc.py check-trigger \
  --report "$SHALLOW_REPORT" \
  --output "$EVIDENCE/robolab-stack-recovery-trigger.json"
```

Exit `0` means `reference_success - candidate_success > 0.05` and permits
collection. Exit `3` means the trigger did not fire: stop here. A gap equal to
exactly `0.05` does not trigger recovery. Missing provenance, a different
RoboLab/client revision, or a checkpoint identity mismatch is an error, not a
trigger.

## 2. Collect at most 100 successful teacher trajectories

Use the already pinned evaluator image and released teacher from
`repro/ROBOLAB_EVAL_RUNBOOK.md`. Start a fresh teacher server with seed 7003:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
  .venv/bin/python scripts/serve_policy.py \
  --env DROID --port 8000 --seed 7003 \
  policy:checkpoint --policy.config pi05_droid_jointpos \
  --policy.dir "$DROID_TEACHER"
```

In another terminal, collect no more than 100 total episodes of the Stack task.
Only successful episodes are selected, so this yields at most 100 expert
trajectories without silently exceeding the cap. `--record-image-data` is
mandatory; `--video-mode none` does not disable the HDF5 observation recorder.

```bash
export EXPERT_ROOT=/mnt/openpi/datasets/robolab-stack-expert
mkdir -p /mnt/openpi/datasets

docker run --rm --gpus all --network host --ipc host \
  --entrypoint /workspace/isaaclab/_isaac_sim/python.sh \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES \
  -v /mnt/openpi/datasets:/workspace/robolab/output \
  "$ROBOLAB_IMAGE" policies/pi0_family/run.py \
  --policy pi05 --remote-host 127.0.0.1 --remote-port 8000 \
  --open-loop-horizon 15 --task Stack3RubiksCubeTask \
  --task-dirs benchmark --instruction-type default \
  --num-envs 10 --num-runs 10 \
  --renderer realtime --rendering-type balanced \
  --record-image-data --video-mode none --device cuda:0 --headless \
  --output-folder-name robolab-stack-expert
```

An interrupted collection may resume only with the identical output folder,
teacher, image, seed, and command. Do not append another policy's episodes.

## 3. Seal the native collection

Run the sealer in the pinned DROID training image, where `h5py` is supplied by
the locked LeRobot dependency. The manifest must live at the collection root;
all HDF5 references are safe relative paths so the directory can move as one
S3 artifact.

```bash
python scripts/repro_robolab_bc.py prepare \
  --robolab-output "$EXPERT_ROOT" \
  --trigger-report "$SHALLOW_REPORT" \
  --accepted-shallow-model "$ACCEPTED_SHALLOW/model.safetensors" \
  --teacher-model "$DROID_TEACHER/model.safetensors" \
  --source-sha "$(git rev-parse HEAD)" \
  --robolab-image-digest "$ROBOLAB_IMAGE_DIGEST" \
  --maximum-trajectories 100 \
  --output "$EXPERT_ROOT/expert_bc_manifest.json"
```

This command fails before writing a manifest if any of the following is true:

- the Stack trigger did not fire;
- either supplied model hash differs from the paired report;
- collection contains a non-Stack or non-`pi05` episode;
- more than 100 successful episodes exist;
- a selected JSONL success is not also marked successful in HDF5;
- an HDF5 file lacks recorded images/proprioception or uses another shape;
- frame counts, 15 Hz timing, simulator versions, or values are invalid.

The manifest hashes the trigger report, native JSONL, every HDF5 file, every
selected JSONL record, and both model files. Upload the entire directory and
verify its byte count before terminating the collection machine:

```bash
aws s3 sync "$EXPERT_ROOT/" \
  "s3://$REPRO_BUCKET/datasets/robolab-stack-expert/" --only-show-errors
aws s3 ls "s3://$REPRO_BUCKET/datasets/robolab-stack-expert/" --recursive --summarize
```

## 4. Run the single 25/75 recovery job

Stage the accepted Shallow checkpoint as a directory containing the exact
manifest-bound `model.safetensors`, and stage the expert directory at the
config's fixed path:

```bash
aws s3 sync "s3://$REPRO_BUCKET/datasets/robolab-stack-expert/" \
  /mnt/openpi/datasets/robolab-stack-expert/ --only-show-errors
test -f /mnt/openpi/datasets/robolab-stack-expert/expert_bc_manifest.json
test -f "$ACCEPTED_SHALLOW/model.safetensors"
export BC25_ATTEMPT_ID="droid-shallow-expert-bc25-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$RUNS/pi05_droid_l09_expert_bc_25/$BC25_ATTEMPT_ID"

torchrun --standalone --nproc-per-node=2 scripts/train_pytorch.py \
  pi05_droid_l09_expert_bc_25 \
  --exp-name "$BC25_ATTEMPT_ID" \
  --pytorch-weight-path "$ACCEPTED_SHALLOW" \
  --checkpoint-base-dir "$RUNS"
```

The config runs exactly 1,500 optimizer steps. Each rank-local batch has one
expert sample and three deterministically permuted MolmoAct2 DROID samples;
the global optimizer batch remains 64 through eight-step accumulation. The
loader disables random weighted sampling so every batch has the exact ratio.

Startup re-hashes the accepted source checkpoint and rejects a teacher path.
Training uses ordinary ground-truth flow matching on both sources. It does not
load the full teacher or compute KD. Checkpoint `metadata.pt` records the
manifest hash, trigger, selected trajectories, deterministic mix, source model
hash, and `teacher_checkpoint_resident=false`.

## 5. Evaluate and gate the optional 50/50 run

Evaluate the 1,500-step BC25 checkpoint for exactly 50 episodes on both tasks
using the intermediate command in `repro/ROBOLAB_EVAL_RUNBOOK.md`, then seal
it under a distinct stage:

```bash
export BC25="$RUNS/pi05_droid_l09_expert_bc_25/$BC25_ATTEMPT_ID/1500"

python scripts/repro_robolab_report.py seal \
  --stage shallow-bc25 --mode intermediate \
  --checkpoint-model "$BC25/model.safetensors" \
  --results "$ROBOLAB_OUTPUT/shallow-bc25-intermediate/episode_results.jsonl" \
  --num-envs 10 --num-runs 5 --policy-server-seed 7003 \
  --image-digest "$ROBOLAB_IMAGE_DIGEST" \
  --robolab-git-sha 0aef241fb088ca21bb4ebd24448940ed56620d17 \
  --output "$ROBOLAB_OUTPUT/shallow-bc25-intermediate/run-identity.json"

python scripts/repro_robolab_bc.py decide-50-50 \
  --before-identity "$ROBOLAB_OUTPUT/shallow-intermediate/run-identity.json" \
  --after-identity "$ROBOLAB_OUTPUT/shallow-bc25-intermediate/run-identity.json" \
  --expert-manifest "$EXPERT_ROOT/expert_bc_manifest.json" \
  --output "$EXPERT_ROOT/bc25_rerun_decision.json"
```

Exit `0` requires strictly higher Stack success and no Banana success drop.
Exit `3` means stop; do not launch another BC job. The decision requires
paired 50-episode identities, identical runtime/evaluation inputs, an accepted
Shallow baseline, and a distinct BC25 model hash.

If and only if approved, run one independent 1,500-step 50/50 job. It again
starts from the accepted pre-BC Shallow checkpoint, not from BC25, so the mix
comparison is interpretable and both candidates share initialization.

```bash
export BC50_ATTEMPT_ID="droid-shallow-expert-bc50-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$RUNS/pi05_droid_l09_expert_bc_50/$BC50_ATTEMPT_ID"

torchrun --standalone --nproc-per-node=2 scripts/train_pytorch.py \
  pi05_droid_l09_expert_bc_50 \
  --exp-name "$BC50_ATTEMPT_ID" \
  --pytorch-weight-path "$ACCEPTED_SHALLOW" \
  --checkpoint-base-dir "$RUNS"
```

Evaluate BC50 identically. Select a recovery checkpoint only if Stack returns
within five points of the base while Banana remains within five points, and
record the exact observed rates. Otherwise retain the original accepted
Shallow checkpoint and stop with the evidence. SnapFlow must receive the
selected numeric checkpoint through `--pytorch-weight-path`; no recovery
result is promoted merely because training completed.

## 6. Manual replay record

For each command, append source SHA, container digest, input/output S3 URIs,
instance type, start/end times, command, exit code, manifest/decision hash,
checkpoint hash, GPU-hours, and estimated On-Demand cost to the run log. A
clean abbreviated replay uses one successful trajectory and a short training
override, but it must exercise trigger, sealing, source-checkpoint validation,
mixing, checkpoint provenance, and the dormant 50/50 gate.
