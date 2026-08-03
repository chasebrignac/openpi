# Method-source ledger

Implementation decisions are anchored to the primary papers and vendor
documentation below. The blog-specific departures are kept explicit.

## Shallow-pi

Primary paper: `arXiv:2601.20262v1`.

- Uniformly subsample both the VLM and corresponding action-expert layers.
- Train velocity output with flow-matching and teacher-KD MSE.
- The paper uses 30,000 steps and batch size 64 for its LIBERO ablations.
- The blog drops attention distillation and selects a nine-layer student.
- This reproduction fixes the blog's nine-layer map to
  `[0, 2, 4, 6, 8, 11, 13, 15, 17]` and uses KD-only on filtered DROID.

## SnapFlow

Primary paper: `arXiv:2604.05656v1`, equations 9-13 and Appendix J.

- Shortcut midpoint: `x_0.5 = x_1 - 0.5 * stopgrad(F(x_1, 1, 1))`.
- Shortcut target: the average of stopped instantaneous velocities at
  `(x_1, t=1)` and `(x_0.5, t=0.5)`.
- One-step student predicts `F(x_1, s=0, t=1)`.
- Mix equal FM and consistency samples (`alpha=0.5`), consistency weight
  `lambda=0.1`, clamp predictions to `[-20, 20]`.
- AdamW, peak LR `2.5e-5`, 500-step linear warmup, gradient norm 1.0,
  batch size 4, BF16, VLM frozen, action expert and zero-init target-time MLP
  trainable.
- The paper reports loss plateauing near 3,500 steps in its 5,000-step study;
  30,000 is a convergence safety margin, not evidence that every run needs it.

## TensorRT and FP8

- NVIDIA TensorRT 11 uses strongly typed networks by default.
- FP8 is explicit quantization: the ONNX graph must contain Q/DQ operations.
- NVIDIA ModelOpt ONNX PTQ supports `quantize_mode="fp8"`, calibration data,
  and explicit `nodes_to_quantize` / `nodes_to_exclude` controls. We include
  only transformer MLP MatMul/Gemm nodes and retain norms, softmax, attention,
  inputs, and outputs in higher precision.
- Engines are hardware-specific and are built and benchmarked on the same
  `g7e.4xlarge`; they are never presented as Jetson Thor measurements.
