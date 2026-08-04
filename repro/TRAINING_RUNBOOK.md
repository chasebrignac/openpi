# Shallow-pi and SnapFlow Training/Promotion Runbook

This runbook covers deterministic local diagnostics, bounded training, held-out
offline evaluation, and evidence-based promotion. It does not launch AWS
capacity. Public RoboLab evaluation and its promotion-evidence bridge are in
`repro/ROBOLAB_EVAL_RUNBOOK.md`.

## Paths and invariants

```bash
export RUNS=/mnt/openpi/runs
export EVIDENCE_ROOT=/mnt/openpi/evidence
# Set this once per fresh attempt. Reusing it intentionally fails instead of
# deleting an earlier run. This example format is sortable and accepted by the
# trainer's strict experiment-name validator.
export ATTEMPT_ID="${ATTEMPT_ID:?Set a unique attempt ID, for example 20260804T120000Z-a1}"
[[ "$ATTEMPT_ID" =~ ^[A-Za-z0-9]([A-Za-z0-9._-]{0,62}[A-Za-z0-9])?$ ]]
[[ "$ATTEMPT_ID" != *..* ]]
export EVIDENCE="$EVIDENCE_ROOT/$ATTEMPT_ID"
export REPRO_RUN_ID="pi05-aws-repro-$ATTEMPT_ID"
export TRAIN_SEED=42
export LIBERO_TEACHER=/mnt/openpi/checkpoints/pi05_libero_pytorch
export DROID_TEACHER=/mnt/openpi/checkpoints/pi05_droid_jointpos_pytorch
export LIBERO_SHALLOW_OVERFIT_EXP="libero-shallow-overfit-$ATTEMPT_ID"
export DROID_SHALLOW_OVERFIT_EXP="droid-shallow-overfit-$ATTEMPT_ID"
export LIBERO_SNAPFLOW_OVERFIT_EXP="libero-snapflow-overfit-$ATTEMPT_ID"
export DROID_SNAPFLOW_OVERFIT_EXP="droid-snapflow-overfit-$ATTEMPT_ID"
export LIBERO_SHALLOW_EXP="libero-shallow-$ATTEMPT_ID"
export DROID_SHALLOW_EXP="droid-shallow-$ATTEMPT_ID"
export LIBERO_SNAPFLOW_EXP="libero-snapflow-$ATTEMPT_ID"
export DROID_SNAPFLOW_EXP="droid-snapflow-$ATTEMPT_ID"
export LIBERO_SHALLOW_ROOT="$RUNS/pi05_libero_l09_distill/$LIBERO_SHALLOW_EXP"
export DROID_SHALLOW_ROOT="$RUNS/pi05_droid_l09_distill/$DROID_SHALLOW_EXP"
export LIBERO_SNAPFLOW_ROOT="$RUNS/pi05_libero_l09_snapflow/$LIBERO_SNAPFLOW_EXP"
export DROID_SNAPFLOW_ROOT="$RUNS/pi05_droid_l09_snapflow/$DROID_SNAPFLOW_EXP"
mkdir -p "$RUNS" "$EVIDENCE"
```

Fresh commands deliberately omit `--overwrite`. Generate a new `ATTEMPT_ID`
for a retry; use `--resume` only for a verified continuation of that exact
attempt.

The four reproduction configs use seed 42 to deterministically stratify by
task/site and reserve complete validation episodes covering at least 256
frames. Training excludes those episodes before delta/action queries; this is
an episode-level split, never a positional record slice. Golden-corpus
generation requires the same data-split seed and refuses to request more
samples than the reserved holdout. Keep the same dataset revision, split seed,
corpus-noise seed, corpus hash, action normalization, and camera ordering across
every checkpoint comparison.

## 1. Fixed held-out corpora

```bash
python scripts/repro_make_golden.py \
  --run-id "$REPRO_RUN_ID" \
  --config-name pi05_libero_l09_distill \
  --samples 64 --seed 7001 \
  --data-split-seed 42 \
  --dataset-revision a4336d589d589045d1c56423ffdf3b88a0e19b1f \
  --output "$EVIDENCE/libero-heldout.npz"

python scripts/repro_make_golden.py \
  --run-id "$REPRO_RUN_ID" \
  --config-name pi05_droid_l09_distill \
  --samples 64 --seed 7002 \
  --data-split-seed 42 \
  --dataset-revision e44d3138c64cfeb1c24fbbce087b475fb1233728 \
  --output "$EVIDENCE/droid-heldout.npz"
```

These are the same canonical files first created for teacher framework
equivalence in `RUNBOOK.md`, not a second generation step. If they already
exist, verify the adjacent JSON and recorded hashes and skip both generation
commands. Commit the adjacent JSON metadata and record both NPZ hashes in the
run manifest. Do not regenerate a corpus at any point in a checkpoint series;
all offline and export gates reuse these exact bytes.

## 2. Deterministic 300-step one-batch diagnostics

These runs repeat one fully materialized global batch. Each rank keeps its own
fixed shard and reuses identical image augmentation, noise, timestep, and
SnapFlow FM/shortcut mask on every microbatch. The reported loss is reduced
across all ranks before the first/last-window calculation. They prove optimizer
correctness only; never promote their checkpoints.

Normal training and resume use independent SHA256-derived seeds for every
model microstep and loader epoch. A resumed single- or multi-GPU job therefore
reconstructs the same shuffle, augmentation, noise, timestep, and SnapFlow mask
from its optimizer/data position rather than relying on process RNG history.

```bash
torchrun --standalone --nproc-per-node=2 scripts/train_pytorch.py pi05_libero_l09_distill \
  --exp-name "$LIBERO_SHALLOW_OVERFIT_EXP" --checkpoint-base-dir "$RUNS" \
  --teacher-pytorch-weight-path "$LIBERO_TEACHER" \
  --seed "$TRAIN_SEED" --num-train-steps 300 --save-interval 300 --log-interval 10 \
  --one-batch-overfit --one-batch-overfit-min-relative-decline 0.20 \
  --num-workers 0 --no-wandb-enabled

torchrun --standalone --nproc-per-node=2 scripts/train_pytorch.py pi05_droid_l09_distill \
  --exp-name "$DROID_SHALLOW_OVERFIT_EXP" --checkpoint-base-dir "$RUNS" \
  --teacher-pytorch-weight-path "$DROID_TEACHER" \
  --seed "$TRAIN_SEED" --num-train-steps 300 --save-interval 300 --log-interval 10 \
  --one-batch-overfit --one-batch-overfit-min-relative-decline 0.20 \
  --num-workers 0 --no-wandb-enabled
```

The trainer aborts coherently on every rank before publishing step 300 if any
loss, diagnostic, gradient, parameter, or optimizer tensor is non-finite, or if
the global mean loss over the last 20 optimizer steps is not at least 20% below
the global mean over the first 20. Runs shorter than 40 optimizer steps and
resumed one-batch runs are rejected. Logs are emitted every 10 optimizer steps.
Verify the complete checkpoint and its durable diagnostic report; the verifier
recomputes both means and the decline from the persisted complete loss sequence,
and enforces the exact 20-step/20% contract:

```bash
python scripts/repro_checkpoint.py "$RUNS/pi05_libero_l09_distill/$LIBERO_SHALLOW_OVERFIT_EXP" --step 300 --require-overfit-report
python scripts/repro_checkpoint.py "$RUNS/pi05_droid_l09_distill/$DROID_SHALLOW_OVERFIT_EXP" --step 300 --require-overfit-report
```

After selecting an accepted Shallow checkpoint, run the same 300-step
diagnostic for each SnapFlow track, supplying the numeric source directory:

```bash
: "${LIBERO_SNAPFLOW_SOURCE:?Set the accepted numeric LIBERO Shallow checkpoint}"
: "${DROID_SNAPFLOW_SOURCE:?Set the accepted numeric DROID Shallow or BC checkpoint}"
python scripts/train_pytorch.py pi05_libero_l09_snapflow \
  --exp-name "$LIBERO_SNAPFLOW_OVERFIT_EXP" --checkpoint-base-dir "$RUNS" \
  --pytorch-weight-path "$LIBERO_SNAPFLOW_SOURCE" \
  --seed "$TRAIN_SEED" --num-train-steps 300 --save-interval 300 --log-interval 10 \
  --one-batch-overfit --one-batch-overfit-min-relative-decline 0.20 \
  --num-workers 0 --no-wandb-enabled

python scripts/train_pytorch.py pi05_droid_l09_snapflow \
  --exp-name "$DROID_SNAPFLOW_OVERFIT_EXP" --checkpoint-base-dir "$RUNS" \
  --pytorch-weight-path "$DROID_SNAPFLOW_SOURCE" \
  --seed "$TRAIN_SEED" --num-train-steps 300 --save-interval 300 --log-interval 10 \
  --one-batch-overfit --one-batch-overfit-min-relative-decline 0.20 \
  --num-workers 0 --no-wandb-enabled
```

The current trainer rejects one-batch mode unless `num_workers` is exactly
zero. One-batch mode itself fixes the diagnostic to its single materialized
batch; disabled W&B removes unnecessary logging while enabled W&B would reuse
that same batch. Declare its numeric checkpoint as an ordinary retained
output, never a `publish_destination`; diagnostic weights are not eligible
SnapFlow inputs.

## 3. Shallow-pi pilots and bounded continuation

Run the 2k pilot, then resume the same experiment to each explicit target. A
target controls total optimizer steps, not additional steps.

Training randomness is derived from the seed, absolute optimizer step,
accumulation index, and global rank. Consequently a numeric resume continues
the same noise/time/augmentation schedule as an uninterrupted run; the fixed
one-batch mode deliberately omits step and accumulation position from that
derivation.

The commands below assume the prior numeric directory is already present in
the same writable `$RUNS` tree. On a fresh AWS worker, do not invoke bare
`--resume`. Use `resume_checkpoint` from `repro/WORKER_RUNBOOK.md`, copying the
prior run manifest's published input descriptor unchanged. That contract
hash-verifies and atomically restores the full model/optimizer/metadata plus
`resume-state.json` into the exact writable config/experiment directory before
the command starts.

```bash
# LIBERO: fresh 2k, then exact 5k/10k/20k/30k checkpoints.
torchrun --standalone --nproc-per-node=2 scripts/train_pytorch.py pi05_libero_l09_distill \
  --exp-name "$LIBERO_SHALLOW_EXP" --checkpoint-base-dir "$RUNS" \
  --teacher-pytorch-weight-path "$LIBERO_TEACHER" \
  --seed "$TRAIN_SEED" --num-train-steps 2000 --save-interval 2000
torchrun --standalone --nproc-per-node=2 scripts/train_pytorch.py pi05_libero_l09_distill \
  --exp-name "$LIBERO_SHALLOW_EXP" --checkpoint-base-dir "$RUNS" --resume --seed "$TRAIN_SEED" --num-train-steps 5000 --save-interval 5000
torchrun --standalone --nproc-per-node=2 scripts/train_pytorch.py pi05_libero_l09_distill \
  --exp-name "$LIBERO_SHALLOW_EXP" --checkpoint-base-dir "$RUNS" --resume --seed "$TRAIN_SEED" --num-train-steps 10000 --save-interval 5000
torchrun --standalone --nproc-per-node=2 scripts/train_pytorch.py pi05_libero_l09_distill \
  --exp-name "$LIBERO_SHALLOW_EXP" --checkpoint-base-dir "$RUNS" --resume --seed "$TRAIN_SEED" --num-train-steps 20000 --save-interval 5000
torchrun --standalone --nproc-per-node=2 scripts/train_pytorch.py pi05_libero_l09_distill \
  --exp-name "$LIBERO_SHALLOW_EXP" --checkpoint-base-dir "$RUNS" --resume --seed "$TRAIN_SEED" --num-train-steps 30000 --save-interval 5000

# DROID: the identical bounded sequence with the DROID config and teacher.
torchrun --standalone --nproc-per-node=2 scripts/train_pytorch.py pi05_droid_l09_distill \
  --exp-name "$DROID_SHALLOW_EXP" --checkpoint-base-dir "$RUNS" \
  --teacher-pytorch-weight-path "$DROID_TEACHER" \
  --seed "$TRAIN_SEED" --num-train-steps 2000 --save-interval 2000
torchrun --standalone --nproc-per-node=2 scripts/train_pytorch.py pi05_droid_l09_distill \
  --exp-name "$DROID_SHALLOW_EXP" --checkpoint-base-dir "$RUNS" --resume --seed "$TRAIN_SEED" --num-train-steps 5000 --save-interval 5000
torchrun --standalone --nproc-per-node=2 scripts/train_pytorch.py pi05_droid_l09_distill \
  --exp-name "$DROID_SHALLOW_EXP" --checkpoint-base-dir "$RUNS" --resume --seed "$TRAIN_SEED" --num-train-steps 10000 --save-interval 5000
torchrun --standalone --nproc-per-node=2 scripts/train_pytorch.py pi05_droid_l09_distill \
  --exp-name "$DROID_SHALLOW_EXP" --checkpoint-base-dir "$RUNS" --resume --seed "$TRAIN_SEED" --num-train-steps 20000 --save-interval 5000
torchrun --standalone --nproc-per-node=2 scripts/train_pytorch.py pi05_droid_l09_distill \
  --exp-name "$DROID_SHALLOW_EXP" --checkpoint-base-dir "$RUNS" --resume --seed "$TRAIN_SEED" --num-train-steps 30000 --save-interval 5000
```

Stop after any checkpoint whose complete promotion report passes; do not run
the later commands. Before launching SnapFlow, resolve the selected directory,
for example:

```bash
python scripts/repro_checkpoint.py "$LIBERO_SHALLOW_ROOT" --step 10000
```

## 4. SnapFlow pilots and bounded continuation

Resolve each accepted source to its exact numeric checkpoint directory. The
DROID source may be the accepted distillation checkpoint or the selected
25/75 or 50/50 expert-BC recovery checkpoint; do not relabel BC weights as the
distillation config. The initial SnapFlow pilot is 5k; later commands resume to
exact totals.

The initial 5k SnapFlow worker is a fresh run and points
`--pytorch-weight-path` at the staged accepted Shallow descriptor under
`/mnt/openpi/checkpoints`. Each later SnapFlow continuation is a full-state
`resume_checkpoint` handoff into `/mnt/openpi/runs`, exactly like Shallow; a
model-only `--pytorch-weight-path` is not a valid 5k-to-10k resume.

```bash
export LIBERO_SNAPFLOW_SOURCE="$LIBERO_SHALLOW_ROOT/10000"
export DROID_SNAPFLOW_SOURCE="$DROID_SHALLOW_ROOT/10000"
# If recovery was selected, replace both values below together, for example:
# export DROID_SNAPFLOW_SOURCE="$RUNS/pi05_droid_l09_expert_bc_25/EXPERIMENT/1500"
# export DROID_SNAPFLOW_TEACHER_CONFIG=pi05_droid_l09_expert_bc_25
export DROID_SNAPFLOW_TEACHER_CONFIG=pi05_droid_l09_distill
python scripts/repro_checkpoint.py "$LIBERO_SNAPFLOW_SOURCE"
python scripts/repro_checkpoint.py "$DROID_SNAPFLOW_SOURCE"

python scripts/train_pytorch.py pi05_libero_l09_snapflow \
  --exp-name "$LIBERO_SNAPFLOW_EXP" --checkpoint-base-dir "$RUNS" \
  --pytorch-weight-path "$LIBERO_SNAPFLOW_SOURCE" \
  --seed "$TRAIN_SEED" --num-train-steps 5000 --save-interval 5000
python scripts/train_pytorch.py pi05_libero_l09_snapflow \
  --exp-name "$LIBERO_SNAPFLOW_EXP" --checkpoint-base-dir "$RUNS" --resume --seed "$TRAIN_SEED" --num-train-steps 10000 --save-interval 5000
python scripts/train_pytorch.py pi05_libero_l09_snapflow \
  --exp-name "$LIBERO_SNAPFLOW_EXP" --checkpoint-base-dir "$RUNS" --resume --seed "$TRAIN_SEED" --num-train-steps 20000 --save-interval 5000
python scripts/train_pytorch.py pi05_libero_l09_snapflow \
  --exp-name "$LIBERO_SNAPFLOW_EXP" --checkpoint-base-dir "$RUNS" --resume --seed "$TRAIN_SEED" --num-train-steps 30000 --save-interval 5000

python scripts/train_pytorch.py pi05_droid_l09_snapflow \
  --exp-name "$DROID_SNAPFLOW_EXP" --checkpoint-base-dir "$RUNS" \
  --pytorch-weight-path "$DROID_SNAPFLOW_SOURCE" \
  --seed "$TRAIN_SEED" --num-train-steps 5000 --save-interval 5000 \
  --num-workers 0 --no-wandb-enabled
python scripts/train_pytorch.py pi05_droid_l09_snapflow \
  --exp-name "$DROID_SNAPFLOW_EXP" --checkpoint-base-dir "$RUNS" --resume \
  --seed "$TRAIN_SEED" --num-train-steps 10000 --save-interval 5000 \
  --num-workers 0 --no-wandb-enabled
python scripts/train_pytorch.py pi05_droid_l09_snapflow \
  --exp-name "$DROID_SNAPFLOW_EXP" --checkpoint-base-dir "$RUNS" --resume \
  --seed "$TRAIN_SEED" --num-train-steps 20000 --save-interval 5000 \
  --num-workers 0 --no-wandb-enabled
python scripts/train_pytorch.py pi05_droid_l09_snapflow \
  --exp-name "$DROID_SNAPFLOW_EXP" --checkpoint-base-dir "$RUNS" --resume \
  --seed "$TRAIN_SEED" --num-train-steps 30000 --save-interval 5000 \
  --num-workers 0 --no-wandb-enabled
```

The DROID commands intentionally keep `num_workers=0` and W&B disabled on the
64 GiB `g7e.2xlarge` host. This avoids duplicating the memory-mapped dataset in
loader processes until the first retained pilot records enough host-memory
headroom to justify a controlled change. Resume commands preserve both values
as worker-launch invariants; W&B is also embedded in the resume contract,
while `num_workers` is enforced by the reviewed worker command.

## 5. Offline evidence at every checkpoint

The evaluator emits JSON metrics plus an NPZ containing teacher/student action
chunks and, for Shallow, velocity arrays. It reports KD MSE/cosine, per-joint
NRMSE, action-chunk error, roughness, normalization saturation and normalized
range excursions. These are not physical joint-limit checks.
SnapFlow additionally measures how much one-step training closes the error gap
between naive one-step and accepted-Shallow ten-step inference.

```bash
# Shallow example; repeat with STEP=2000,5000,10000,20000,30000 as produced.
export STEP=5000
python scripts/repro_evaluate_distillation.py \
  --run-id "$REPRO_RUN_ID" \
  --student-config-name pi05_libero_l09_distill \
  --student-run-root "$LIBERO_SHALLOW_ROOT" --student-step "$STEP" \
  --teacher-config-name pi05_libero --teacher-checkpoint "$LIBERO_TEACHER" \
  --corpus "$EVIDENCE/libero-heldout.npz" \
  --normalization-low -1 --normalization-high 1 \
  --output "$EVIDENCE/libero-shallow-$STEP.json"

# SnapFlow example; the teacher path is the exact checkpoint that initialized
# this SnapFlow run. The evaluator checks the recorded initialization hash.
python scripts/repro_evaluate_distillation.py \
  --run-id "$REPRO_RUN_ID" \
  --student-config-name pi05_libero_l09_snapflow \
  --student-run-root "$LIBERO_SNAPFLOW_ROOT" --student-step 5000 \
  --teacher-config-name pi05_libero_l09_distill \
  --teacher-checkpoint "$LIBERO_SNAPFLOW_SOURCE" \
  --corpus "$EVIDENCE/libero-heldout.npz" \
  --normalization-low -1 --normalization-high 1 \
  --output "$EVIDENCE/libero-snapflow-5000.json"
```

The Shallow evaluator accepts only `pi05_libero` for the LIBERO distilled
student and `pi05_droid_jointpos` for the DROID distilled/BC students. For a
distilled checkpoint it also requires the selected teacher model hash to equal
the checkpoint's recorded `shallow_teacher_transplant` lineage; config names
alone are not sufficient.

Use the same form with `pi05_droid_l09_snapflow`,
`$DROID_SNAPFLOW_ROOT`, `$DROID_SNAPFLOW_TEACHER_CONFIG`,
`$DROID_SNAPFLOW_SOURCE`, and the DROID corpus. The evaluator accepts the
distilled or expert-BC nine-layer teacher config, verifies its numeric
checkpoint metadata, and requires its model hash to equal the SnapFlow
checkpoint's recorded `pytorch_source` lineage. The default saturation limits
are normalized `[-1, 1]`; override them only if the recorded normalization
manifest proves different limits. Offline evaluation of a BC25/BC50 student
reuses the immutable canonical DROID distillation corpus rather than attempting
to generate a zero-sample corpus from the recovery config. Treat that result as
a fixed paired diagnostic for recovery regression, not as a BC-held-out
generalization estimate.

## 6. Promotion and 30k stop decision

For LIBERO, convert the episode-level report produced by
`repro_quality_report.py`; do not hand-author promotion quality JSON. For the
DROID track, use the pinned public RoboLab procedure and
`repro_robolab_report.py` in `repro/ROBOLAB_EVAL_RUNBOOK.md`. The LIBERO episode
records and quality report must use model-bound stage names derived from the
matching offline report:

```bash
export OFFLINE="$EVIDENCE/libero-shallow-5000.json"
export REFERENCE_STAGE="teacher-sha256:$(jq -r '.provenance.teacher_checkpoint.model_sha256' "$OFFLINE")"
export STUDENT_STAGE="student-sha256:$(jq -r '.provenance.student_checkpoint.model_sha256' "$OFFLINE")"

# Generate paired episode records with REFERENCE_STAGE for the reference
# policy and STUDENT_STAGE for the student policy, then aggregate them.
python scripts/repro_quality_report.py "$EVIDENCE/libero-shallow-5000-paired.jsonl" \
  --mode intermediate --expected-pairs 400 \
  --base-stage "$REFERENCE_STAGE" --candidate-stage "$STUDENT_STAGE" \
  --output "$EVIDENCE/libero-shallow-5000-episode-report.json"

python scripts/repro_quality_evidence.py \
  --quality-report "$EVIDENCE/libero-shallow-5000-episode-report.json" \
  --offline-report "$OFFLINE" --required-pairs 400 \
  --output "$EVIDENCE/libero-shallow-rollout-5000.json"
```

For SnapFlow, use the same conversion and attach the measured denoising
speedup. The model hashes, step, run, configs, dataset and golden corpus must
all match or conversion fails:

```bash
python scripts/repro_quality_evidence.py \
  --quality-report "$EVIDENCE/libero-snapflow-5000-episode-report.json" \
  --offline-report "$EVIDENCE/libero-snapflow-5000.json" \
  --required-pairs 400 --denoise-speedup 8.4 \
  --output "$EVIDENCE/libero-snapflow-rollout-5000.json"
```

Freeze offline thresholds from the 2k pilot and paired-base variability before
looking at later checkpoints. Then run:

```bash
python scripts/repro_promotion_report.py --stage shallow \
  --offline "$EVIDENCE/libero-shallow-2000.json" \
  --offline "$EVIDENCE/libero-shallow-5000.json" \
  --quality "$EVIDENCE/libero-shallow-rollout-2000.json" \
  --quality "$EVIDENCE/libero-shallow-rollout-5000.json" \
  --max-rollout-gap 0.03 --min-kd-cosine 0.98 \
  --max-per-joint-nrmse 0.50 --max-action-chunk-rmse 0.25 \
  --output "$EVIDENCE/libero-shallow-promotion.json"

python scripts/repro_promotion_report.py --stage snapflow \
  --offline "$EVIDENCE/libero-snapflow-5000.json" \
  --quality "$EVIDENCE/libero-snapflow-rollout-5000.json" \
  --max-rollout-gap 0.03 \
  --output "$EVIDENCE/libero-snapflow-promotion.json"
```

Threshold numbers above are placeholders to freeze empirically, not published
claims. SnapFlow always requires at least 70% offline error-gap closure, 8x
denoising speedup, zero normalization-range excursions and the supplied
paired-rollout gap. Missing evidence is never treated as a pass. A failed or
incomplete promotion command exits nonzero; use `--report-only` only when an
exploratory report is intentionally not an acceptance decision.

At 30k, the report recommends an extension only when validation KD error is
still improving by at least 5% per 5k steps (interval-normalized) and paired
rollout success is also improving. Any extension still requires separate
approval; no command here exceeds 30k.
