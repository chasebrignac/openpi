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
- Every paid launch first calls `scripts/repro_cost_guard.py --reserve`.
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

1. `aws sts get-caller-identity` returns account `752160877725`.
2. `AWS_REGION` and CLI region are explicitly `us-east-2`.
3. Working tree SHA and container digest match the intended run manifest.
4. Dataset and checkpoint S3 object versions are resolved, never `latest`.
5. Projected hours include setup, checkpoint upload, and a shutdown margin.
6. The cost guard reservation passes both its category cap and the hard cap.
7. An instance-side shutdown deadline is installed before the workload starts.
8. SSM works; the security group has no inbound rules.

## Promotion order

1. Foundation, checkpoint conversion, golden vectors, baseline smoke and eager
   benchmark.
2. Shallow-pi: 300-step overfit, 2k pilot, then 5k/10k/20k/30k empirical gates.
3. Bounded RoboLab BC recovery only when the Stack3RubiksCube trigger fires.
4. SnapFlow: 5k pilot, then 10k/20k/30k only while gates require it.
5. BF16 ONNX validation, BF16 TensorRT, selective-MLP FP8 calibration.
6. Intermediate then paired final quality evaluation and fixed-shape latency.
7. Two clean abbreviated manual replays, then CloudFormation and Change Set.

## Manual execution log

### 2026-08-03 - source import and guardrails

- Imported upstream OpenPI commit `15a9616a00943ada6c20a0f158e3adb39df2ccac`
  on branch `codex/pi05-aws-repro`.
- Verified Ohio G/VT quota is 64 vCPUs: one `g7e.12xlarge` plus one
  `g6e.4xlarge` exactly. Verified G7e offerings in `us-east-2a` and `us-east-2b`.
- Captured current Linux On-Demand rates from AWS Pricing in
  `repro/reproduction.json`.
- No paid instance was launched and no cost reservation was made.
