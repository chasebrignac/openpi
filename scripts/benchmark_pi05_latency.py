#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Benchmark fixed-shape pi0.5 prefix, denoising, and total policy latency."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
from typing import Any

import numpy as np

STAGES = ("base", "shallow", "snapflow", "tensorrt_bf16", "tensorrt_fp8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("torch", "tensorrt"))
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--artifact-dir", required=True, type=pathlib.Path)
    parser.add_argument("--config", help="required for the torch backend")
    parser.add_argument("--checkpoint", type=pathlib.Path, help="required for the torch backend")
    parser.add_argument("--num-denoise-steps", type=int)
    parser.add_argument("--warmups", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--allow-nonstandard-counts", action="store_true", help="smoke only; report is non-official")
    parser.add_argument("--track", required=True, choices=("libero", "droid"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--instance-type", default="g7e.4xlarge")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--cost-reservation", required=True)
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def _npz(path: pathlib.Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _weights_path(path: pathlib.Path) -> pathlib.Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint weights not found: {path}")
    return path


def _require_manifest_artifact(manifest: dict[str, Any], path: pathlib.Path) -> None:
    """Verify one relocated artifact against a stage manifest record."""

    from openpi.exporting.onnx_artifacts import file_identity

    matches = [
        record for record in manifest.get("artifacts", ()) if pathlib.Path(record.get("path", "")).name == path.name
    ]
    if len(matches) != 1:
        raise ValueError(f"Build manifest must identify exactly one {path.name!r} artifact")
    expected = {
        "name": pathlib.Path(matches[0]["path"]).name,
        "bytes": matches[0]["bytes"],
        "sha256": matches[0]["sha256"],
    }
    if file_identity(path) != expected:
        raise ValueError(f"Artifact no longer matches its build manifest: {path}")


class _TorchRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        import safetensors.torch
        import torch

        from openpi.exporting.pi05_onnx import encode_prefix_tensors
        from openpi.models import pi0_config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from openpi.models_pytorch.snapflow import SnapFlowPI0Pytorch
        from openpi.training import config as training_config

        if not args.config or args.checkpoint is None:
            raise ValueError("--config and --checkpoint are required for the torch backend")
        if args.stage not in {"base", "shallow", "snapflow"}:
            raise ValueError(f"Torch backend does not implement stage {args.stage!r}")
        self.torch = torch
        self.device = torch.device("cuda")
        self.config_name = args.config
        train_config = training_config.get_config(args.config)
        model_config = dataclasses.replace(train_config.model, pytorch_compile_mode=None)
        self.is_snapflow = isinstance(model_config, pi0_config.SnapFlowPi0Config)
        depth = int(getattr(model_config, "pytorch_gemma_depth", 18))
        if args.stage == "base" and (self.is_snapflow or depth != 18):
            raise ValueError("The base latency stage requires a full 18-layer non-SnapFlow config")
        if args.stage == "shallow" and (self.is_snapflow or depth != 9):
            raise ValueError("The shallow latency stage requires a nine-layer non-SnapFlow config")
        if args.stage == "snapflow" and (not self.is_snapflow or depth != 9):
            raise ValueError("The SnapFlow latency stage requires a nine-layer SnapFlow config")
        if self.is_snapflow:
            self.model = SnapFlowPI0Pytorch(model_config)
        else:
            self.model = PI0Pytorch(model_config)
        self.model.to(self.device)
        self.weights = _weights_path(args.checkpoint)
        missing, unexpected = safetensors.torch.load_model(
            self.model, self.weights, strict=True, device=str(self.device)
        )
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint did not load exactly: missing={missing}, unexpected={unexpected}")
        self.model.eval().requires_grad_(requires_grad=False)

        prefix_arrays = _npz(args.artifact_dir / "encode-inputs.npz")
        decoder_arrays = _npz(args.artifact_dir / "decode-inputs.npz")
        if prefix_arrays["image_0"].shape[0] != 1 or decoder_arrays["state"].shape[0] != 1:
            raise ValueError("Official latency benchmarking requires fixed batch one inputs")
        self.images = tuple(torch.from_numpy(prefix_arrays[f"image_{index}"]).to(self.device) for index in range(3))
        self.image_masks = tuple(
            torch.from_numpy(prefix_arrays[f"image_mask_{index}"]).to(self.device) for index in range(3)
        )
        self.lang_tokens = torch.from_numpy(prefix_arrays["lang_tokens"]).to(self.device)
        self.lang_mask = torch.from_numpy(prefix_arrays["lang_mask"]).to(self.device)
        self.state = torch.from_numpy(decoder_arrays["state"]).to(self.device)
        self.noise = torch.from_numpy(decoder_arrays["x_t"]).to(self.device)
        self.num_steps = args.num_denoise_steps or (1 if self.is_snapflow else 10)
        if self.num_steps < 1:
            raise ValueError("num-denoise-steps must be positive")
        if self.is_snapflow and self.num_steps != 1:
            raise ValueError("SnapFlow latency must use exactly one denoising step")
        self.timesteps = tuple(
            torch.full((1,), 1.0 - index / self.num_steps, dtype=torch.float32, device=self.device)
            for index in range(self.num_steps)
        )
        self.target_zero = torch.zeros((1,), dtype=torch.float32, device=self.device)
        self._encode_prefix_tensors = encode_prefix_tensors
        with torch.inference_mode():
            self.fixed_prefix_mask, self.fixed_cache = self._prefix()

    def _prefix(self):
        return self._encode_prefix_tensors(
            self.model,
            self.images,
            self.image_masks,
            self.lang_tokens,
            self.lang_mask,
        )

    def _denoise(self, prefix_mask, cache):
        x_t = self.noise
        delta = -1.0 / self.num_steps
        for timestep in self.timesteps:
            if self.is_snapflow:
                velocity = self.model.denoise_step(
                    self.state,
                    prefix_mask,
                    cache,
                    x_t,
                    timestep,
                    target_time=self.target_zero,
                )
            else:
                velocity = self.model.denoise_step(self.state, prefix_mask, cache, x_t, timestep)
            x_t = x_t + delta * velocity
        return x_t

    def prefix(self):
        return self._prefix()

    def denoise(self):
        return self._denoise(self.fixed_prefix_mask, self.fixed_cache)

    def total(self):
        prefix_mask, cache = self._prefix()
        return self._denoise(prefix_mask, cache)

    @property
    def artifacts(self) -> list[pathlib.Path]:
        return [self.weights]

    @property
    def details(self) -> dict[str, Any]:
        return {
            "backend": "torch-eager",
            "config": self.config_name,
            "num_denoise_steps": self.num_steps,
            "input_boundary": "preprocessed fixed tensors; policy transforms excluded equally for every stage",
        }


def _torch_dtype(trt, dtype):
    import torch

    pairs = (
        (trt.float32, torch.float32),
        (trt.float16, torch.float16),
        (trt.bfloat16, torch.bfloat16),
        (trt.int64, torch.int64),
        (trt.int32, torch.int32),
        (trt.int8, torch.int8),
        (trt.uint8, torch.uint8),
        (trt.bool, torch.bool),
    )
    for trt_dtype, torch_dtype in pairs:
        if dtype == trt_dtype:
            return torch_dtype
    raise TypeError(f"Unsupported TensorRT I/O dtype: {dtype}")


class _TensorRTEngine:
    def __init__(self, path: pathlib.Path) -> None:
        import tensorrt as trt
        import torch

        self.trt = trt
        self.torch = torch
        self.path = path
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"TensorRT could not deserialize engine: {path}")
        self.context = self.engine.create_execution_context()
        self.inputs: dict[str, Any] = {}
        self.outputs: dict[str, Any] = {}
        self.input_names: list[str] = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(int(dim) for dim in self.engine.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise ValueError(f"TensorRT engine is not fixed-shape: {name}={shape}")
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                tensor = torch.empty(shape, dtype=_torch_dtype(trt, self.engine.get_tensor_dtype(name)), device="cuda")
                self.outputs[name] = tensor
                self.context.set_tensor_address(name, tensor.data_ptr())
            else:
                self.input_names.append(name)

    def set_input_array(self, name: str, array: np.ndarray) -> None:
        tensor = self.torch.from_numpy(np.asarray(array)).to(
            device="cuda", dtype=_torch_dtype(self.trt, self.engine.get_tensor_dtype(name))
        )
        self.set_input_tensor(name, tensor)

    def set_input_tensor(self, name: str, tensor) -> None:
        expected_shape = tuple(int(dim) for dim in self.engine.get_tensor_shape(name))
        expected_dtype = _torch_dtype(self.trt, self.engine.get_tensor_dtype(name))
        if tuple(tensor.shape) != expected_shape or tensor.dtype != expected_dtype:
            raise ValueError(
                f"TensorRT input contract mismatch for {name}: got {tuple(tensor.shape)}/{tensor.dtype}, "
                f"expected {expected_shape}/{expected_dtype}"
            )
        self.inputs[name] = tensor
        self.context.set_tensor_address(name, tensor.data_ptr())

    def execute(self):
        stream = self.torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError(f"TensorRT execution failed: {self.path}")
        return self.outputs


class _TensorRTRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch

        if args.stage not in {"tensorrt_bf16", "tensorrt_fp8"}:
            raise ValueError(f"TensorRT backend does not implement stage {args.stage!r}")
        precision = "bf16" if args.stage == "tensorrt_bf16" else "fp8"
        self.precision = precision
        self.build_manifest_path = args.artifact_dir / f"tensorrt-manifest.{precision}.json"
        build_manifest = json.loads(self.build_manifest_path.read_text())
        expected_runtime = {
            "instance_type": args.instance_type,
            "instance_id": args.instance_id,
            "image_digest": args.image_digest,
        }
        if build_manifest.get("stage") != f"tensorrt-build-{precision}" or build_manifest.get("track") != args.track:
            raise ValueError("TensorRT build-manifest stage or track does not match the benchmark")
        if build_manifest.get("dataset") != {"name": args.dataset, "revision": args.dataset_revision}:
            raise ValueError("TensorRT build-manifest dataset does not match the benchmark")
        if build_manifest.get("runtime") != expected_runtime:
            raise ValueError("TensorRT engine must be built and benchmarked in the same pinned runtime")

        prefix_plan = args.artifact_dir / f"encode-prefix.{precision}.plan"
        decoder_plan = args.artifact_dir / f"decode-denoise.{precision}.plan"
        validation_name = pathlib.Path(build_manifest.get("details", {}).get("validation_report", "")).name
        if not validation_name:
            raise ValueError("TensorRT build manifest has no validation-report identity")
        self.validation_path = args.artifact_dir / validation_name
        for path in (prefix_plan, decoder_plan, self.validation_path):
            _require_manifest_artifact(build_manifest, path)
        validation = json.loads(self.validation_path.read_text())
        if validation.get("passes") is not True or validation.get("precision") != precision:
            raise ValueError("TensorRT engine build does not reference a passing matching-precision validation")
        action_gate = validation.get("end_to_end_actions", {})
        if not action_gate.get("action_limits_pass"):
            raise ValueError("TensorRT engine build does not reference a passing action-limit gate")
        self.action_low = np.asarray(action_gate["action_low"], dtype=np.float32)
        self.action_high = np.asarray(action_gate["action_high"], dtype=np.float32)
        self.action_mask = np.asarray(action_gate["action_mask"], dtype=np.bool_)

        self.prefix_engine = _TensorRTEngine(prefix_plan)
        self.decoder_engine = _TensorRTEngine(decoder_plan)
        prefix_arrays = _npz(args.artifact_dir / "encode-inputs.npz")
        decoder_arrays = _npz(args.artifact_dir / "decode-inputs.npz")
        if prefix_arrays["image_0"].shape[0] != 1 or decoder_arrays["state"].shape[0] != 1:
            raise ValueError("Official latency benchmarking requires fixed batch one inputs")

        for name in self.prefix_engine.input_names:
            self.prefix_engine.set_input_array(name, prefix_arrays[name])
        # Produce the reusable cache once, then bind its buffers directly to
        # the decoder so total timing has no host copies.
        prefix_outputs = self.prefix_engine.execute()
        torch.cuda.synchronize()
        for index in range(self.decoder_engine.engine.num_io_tensors):
            name = self.decoder_engine.engine.get_tensor_name(index)
            if self.decoder_engine.engine.get_tensor_mode(name) != self.decoder_engine.trt.TensorIOMode.INPUT:
                continue
            if name in prefix_outputs:
                self.decoder_engine.set_input_tensor(name, prefix_outputs[name])
            elif name in decoder_arrays:
                self.decoder_engine.set_input_array(name, decoder_arrays[name])
            else:
                raise KeyError(f"No fixed benchmark value for decoder input {name!r}")
        self.reference_actions = np.asarray(
            _npz(args.artifact_dir / "decode-reference.npz")["actions"], dtype=np.float32
        )

    def prefix(self):
        return self.prefix_engine.execute()

    def denoise(self):
        return self.decoder_engine.execute()

    def total(self):
        self.prefix_engine.execute()
        return self.decoder_engine.execute()

    def numerical_smoke(self) -> dict[str, Any]:
        import torch

        from openpi.exporting.numerics import action_diagnostics
        from openpi.exporting.numerics import compare_outputs

        candidate = self.total()["actions"]
        torch.cuda.synchronize()
        comparison = compare_outputs(
            {"actions": self.reference_actions},
            {"actions": candidate.float().cpu().numpy()},
            cosine_threshold=0.999 if self.precision == "bf16" else 0.995,
        )
        actions = action_diagnostics(
            self.reference_actions,
            candidate.float().cpu().numpy(),
            action_low=self.action_low,
            action_high=self.action_high,
            action_mask=self.action_mask,
            max_abs_joint_bias=0.01,
        )
        report = {
            "passes": comparison["passes"] and actions["bias_passes"] and actions["action_limits_pass"],
            "comparison": comparison,
            "actions": actions,
        }
        if not report["passes"]:
            raise RuntimeError(f"TensorRT numerical smoke failed: {report}")
        return report

    @property
    def artifacts(self) -> list[pathlib.Path]:
        return [
            self.prefix_engine.path,
            self.decoder_engine.path,
            self.validation_path,
            self.build_manifest_path,
        ]

    @property
    def details(self) -> dict[str, Any]:
        return {
            "backend": "tensorrt",
            "precision": self.precision,
            "num_denoise_steps": 1,
            "input_boundary": "preprocessed fixed tensors; policy transforms excluded equally for every stage",
        }


def main() -> int:
    args = _parse_args()
    if not args.allow_nonstandard_counts and (args.warmups, args.iterations) != (500, 10_000):
        raise ValueError("Official latency reports require exactly 500 warmups and 10,000 timed iterations")
    if args.instance_type != "g7e.4xlarge":
        raise ValueError("Stagewise acceptance latency must run on the same g7e.4xlarge instance type")
    expected_denoise_steps = 10 if args.stage in {"base", "shallow"} else 1
    if args.backend == "tensorrt" and args.num_denoise_steps not in {None, 1}:
        raise ValueError("The split TensorRT decoder implements exactly one denoising step")
    if (
        args.num_denoise_steps is not None
        and args.num_denoise_steps != expected_denoise_steps
        and not args.allow_nonstandard_counts
    ):
        raise ValueError(
            f"Official {args.stage} latency requires {expected_denoise_steps} denoising steps, "
            f"got {args.num_denoise_steps}"
        )

    from openpi.exporting.runtime_identity import query_gpu_inventory
    from openpi.exporting.runtime_identity import require_live_runtime_identity

    live_identity = require_live_runtime_identity(
        image_digest=args.image_digest,
        instance_type=args.instance_type,
        instance_id=args.instance_id,
    )

    from openpi.exporting.artifacts import require_absent_outputs
    from openpi.exporting.artifacts import require_clean_source_identity
    from openpi.exporting.artifacts import write_json_new

    require_clean_source_identity()
    output = args.output or args.artifact_dir / f"latency.{args.stage}.json"
    raw_samples = args.artifact_dir / f"latency-samples.{args.stage}.npz"
    manifest = args.artifact_dir / f"latency-manifest.{args.stage}.json"
    require_absent_outputs([output, raw_samples, manifest], stage=f"{args.stage} latency benchmark")
    gpu_inventory = query_gpu_inventory()

    import torch

    from openpi.exporting.artifacts import write_stage_manifest
    from openpi.exporting.benchmark import benchmark_cuda
    from openpi.exporting.onnx_artifacts import file_identity

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for latency benchmarking")
    runner = _TorchRunner(args) if args.backend == "torch" else _TensorRTRunner(args)
    if runner.details["num_denoise_steps"] != expected_denoise_steps and not args.allow_nonstandard_counts:
        raise ValueError(
            f"Official {args.stage} latency requires {expected_denoise_steps} denoising steps, "
            f"got {runner.details['num_denoise_steps']}"
        )
    numerical_smoke = runner.numerical_smoke() if isinstance(runner, _TensorRTRunner) else None
    timing_samples = {
        "prefix": benchmark_cuda(runner.prefix, warmups=args.warmups, iterations=args.iterations),
        "denoise": benchmark_cuda(runner.denoise, warmups=args.warmups, iterations=args.iterations),
        "total": benchmark_cuda(runner.total, warmups=args.warmups, iterations=args.iterations),
    }
    latency = {component: samples.report() for component, samples in timing_samples.items()}
    report = {
        "schema_version": 1,
        "stage": args.stage,
        "track": args.track,
        "official_protocol": (
            (args.warmups, args.iterations) == (500, 10_000)
            and runner.details["num_denoise_steps"] == expected_denoise_steps
        ),
        "batch_size": 1,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "latency": latency,
        "runner": runner.details,
        "numerical_smoke": numerical_smoke,
        "gpu_inventory": list(gpu_inventory),
        "dataset": {"name": args.dataset, "revision": args.dataset_revision},
        "benchmark_inputs": {
            "encode": file_identity(args.artifact_dir / "encode-inputs.npz"),
            "decode": file_identity(args.artifact_dir / "decode-inputs.npz"),
        },
        "source_artifacts": [file_identity(path) for path in runner.artifacts],
        "runtime": {
            "instance_type": args.instance_type,
            "instance_id": args.instance_id,
            "image_digest": args.image_digest,
            "instance_identity_source": live_identity.instance_identity_source,
        },
    }
    write_json_new(output, report)
    np.savez(
        raw_samples,
        **{
            f"{component}_{clock}": np.asarray(getattr(samples, clock), dtype=np.float32)
            for component, samples in timing_samples.items()
            for clock in ("cuda_event_ms", "wall_ms")
        },
    )
    write_stage_manifest(
        manifest,
        stage=f"latency-{args.stage}",
        track=args.track,
        command=sys.argv,
        image_digest=args.image_digest,
        dataset=args.dataset,
        dataset_revision=args.dataset_revision,
        instance_type=args.instance_type,
        instance_id=args.instance_id,
        cost_reservation=args.cost_reservation,
        artifacts=[*runner.artifacts, output, raw_samples],
        details=report,
        metrics={"latency": latency, "numerical_smoke": numerical_smoke},
    )
    print(json.dumps({"report": str(output), "manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
