#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Validate split ONNX graphs against their paired PyTorch golden outputs."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=pathlib.Path)
    parser.add_argument("--precision", required=True, choices=("bf16", "fp8"))
    parser.add_argument("--cosine-threshold", required=True, type=float)
    parser.add_argument("--track", required=True, choices=("libero", "droid"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--instance-type", default="g7e.4xlarge")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--cost-reservation", required=True)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--action-limits-npz", type=pathlib.Path)
    parser.add_argument("--max-abs-joint-bias", type=float, default=0.01)
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def _cast_input(array: np.ndarray, onnx_type: str) -> np.ndarray:
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(bool)": np.bool_,
    }
    if onnx_type == "tensor(bfloat16)":
        import ml_dtypes

        return np.asarray(array, dtype=ml_dtypes.bfloat16)
    if onnx_type not in mapping:
        raise TypeError(f"Unsupported ONNX Runtime input type: {onnx_type}")
    return np.asarray(array, dtype=mapping[onnx_type])


def _merge_candidate_prefix(
    decoder_inputs: dict[str, np.ndarray], prefix_outputs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Replace only the prefix mask/KV boundary with candidate graph outputs."""

    expected = {name for name in decoder_inputs if name == "prefix_pad_masks" or name.startswith("cache_")}
    if set(prefix_outputs) != expected:
        raise ValueError(
            f"Candidate prefix boundary differs from decoder cache inputs: candidate={sorted(prefix_outputs)}, "
            f"expected={sorted(expected)}"
        )
    return {**decoder_inputs, **prefix_outputs}


def _run_graph(
    model_path: pathlib.Path,
    input_path: pathlib.Path,
    provider: str,
    *,
    input_overrides: dict[str, np.ndarray] | None = None,
):
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    session_options = ort.SessionOptions()
    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider is required but unavailable; providers={sorted(available)}")
        session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        providers = ["CUDAExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), sess_options=session_options, providers=providers)
    active_providers = session.get_providers()
    if provider == "cuda" and active_providers != ["CUDAExecutionProvider"]:
        raise RuntimeError(
            "CUDA validation must run exclusively on CUDAExecutionProvider with CPU fallback disabled; "
            f"active providers={active_providers}"
        )
    with np.load(input_path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
        if input_overrides:
            values.update(input_overrides)
        missing = [metadata.name for metadata in session.get_inputs() if metadata.name not in values]
        if missing:
            raise KeyError(f"{input_path} has no ONNX inputs named {missing}")
        if provider == "cuda":
            outputs = _run_cuda_iobinding(session, values)
        else:
            feed = {
                metadata.name: _cast_input(values[metadata.name], metadata.type) for metadata in session.get_inputs()
            }
            result = session.run(None, feed)
            outputs = {
                metadata.name: np.asarray(value, dtype=np.float32)
                for metadata, value in zip(session.get_outputs(), result, strict=True)
            }
    return outputs, active_providers


def _run_cuda_iobinding(session, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Run fixed-shape graphs without routing BF16 tensors through NumPy."""

    import onnx
    import torch

    type_map = {
        "tensor(float)": (torch.float32, onnx.TensorProto.FLOAT),
        "tensor(float16)": (torch.float16, onnx.TensorProto.FLOAT16),
        "tensor(bfloat16)": (torch.bfloat16, onnx.TensorProto.BFLOAT16),
        "tensor(double)": (torch.float64, onnx.TensorProto.DOUBLE),
        "tensor(int64)": (torch.int64, onnx.TensorProto.INT64),
        "tensor(int32)": (torch.int32, onnx.TensorProto.INT32),
        "tensor(bool)": (torch.bool, onnx.TensorProto.BOOL),
    }

    def metadata_shape(metadata) -> tuple[int, ...]:
        if any(not isinstance(dimension, int) for dimension in metadata.shape):
            raise ValueError(f"CUDA validation requires fixed shape for {metadata.name}: {metadata.shape}")
        return tuple(int(dimension) for dimension in metadata.shape)

    binding = session.io_binding()
    input_tensors = []
    for metadata in session.get_inputs():
        if metadata.type not in type_map:
            raise TypeError(f"Unsupported ONNX Runtime CUDA input type: {metadata.type}")
        torch_dtype, onnx_dtype = type_map[metadata.type]
        tensor = torch.as_tensor(np.asarray(values[metadata.name])).to(device="cuda", dtype=torch_dtype).contiguous()
        expected_shape = metadata_shape(metadata)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Fixed CUDA input shape mismatch for {metadata.name}: {tuple(tensor.shape)} != {expected_shape}"
            )
        input_tensors.append(tensor)
        binding.bind_input(
            name=metadata.name,
            device_type="cuda",
            device_id=0,
            element_type=onnx_dtype,
            shape=expected_shape,
            buffer_ptr=tensor.data_ptr(),
        )

    output_tensors = {}
    for metadata in session.get_outputs():
        if metadata.type not in type_map:
            raise TypeError(f"Unsupported ONNX Runtime CUDA output type: {metadata.type}")
        torch_dtype, onnx_dtype = type_map[metadata.type]
        shape = metadata_shape(metadata)
        tensor = torch.empty(shape, dtype=torch_dtype, device="cuda")
        output_tensors[metadata.name] = tensor
        binding.bind_output(
            name=metadata.name,
            device_type="cuda",
            device_id=0,
            element_type=onnx_dtype,
            shape=shape,
            buffer_ptr=tensor.data_ptr(),
        )
    session.run_with_iobinding(binding)
    binding.synchronize_outputs()
    return {name: tensor.float().cpu().numpy() for name, tensor in output_tensors.items()}


def _reference(path: pathlib.Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name], dtype=np.float32) for name in archive.files}


def _external_onnx_data(model_path: pathlib.Path) -> list[pathlib.Path]:
    import onnx

    model = onnx.load(model_path, load_external_data=False)
    locations = {
        field.value
        for initializer in model.graph.initializer
        for field in initializer.external_data
        if field.key == "location"
    }
    files = [(model_path.parent / location).resolve() for location in sorted(locations)]
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"ONNX external data is missing: {missing}")
    return files


def _load_action_envelope(
    path: pathlib.Path,
    *,
    track: str,
    dataset: str,
    dataset_revision: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, pathlib.Path]:
    """Load a sealed corpus envelope and verify its non-safety semantics."""

    from openpi.exporting.artifacts import sha256_file

    sidecar_path = path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text())
    with np.load(path, allow_pickle=False) as limits:
        required = {"action_low", "action_high", "action_mask", "metadata_json"}
        missing = sorted(required.difference(limits.files))
        if missing:
            raise KeyError(f"Action envelope {path} is missing arrays: {missing}")
        action_low = np.asarray(limits["action_low"], dtype=np.float32)
        action_high = np.asarray(limits["action_high"], dtype=np.float32)
        action_mask = np.asarray(limits["action_mask"], dtype=np.bool_)
        metadata = json.loads(np.asarray(limits["metadata_json"], dtype=np.uint8).tobytes().decode("utf-8"))

        physical_fields = {
            "physical_low",
            "physical_high",
            "physical_mask",
            "physical_state_dependent_mask",
        }
        missing_physical = sorted(physical_fields.difference(limits.files))
        if missing_physical:
            raise KeyError(f"Action envelope {path} is missing physical semantics: {missing_physical}")
        physical_low = np.asarray(limits["physical_low"], dtype=np.float32)
        physical_high = np.asarray(limits["physical_high"], dtype=np.float32)
        physical_mask = np.asarray(limits["physical_mask"], dtype=np.bool_)
        state_dependent = np.asarray(limits["physical_state_dependent_mask"], dtype=np.bool_)

    artifact = sidecar.get("artifact", {})
    if artifact.get("bytes") != path.stat().st_size or artifact.get("sha256") != sha256_file(path):
        raise ValueError("Action-envelope bytes do not match their metadata sidecar")
    sidecar_metadata = {key: value for key, value in sidecar.items() if key != "artifact"}
    if sidecar_metadata != metadata:
        raise ValueError("Embedded action-envelope metadata differs from its sidecar")
    if metadata.get("schema_version") != 1 or metadata.get("gate_kind") != "corpus_envelope":
        raise ValueError("Action artifact is not a supported corpus-envelope schema")
    if metadata.get("hardware_safety_guarantee") is not False:
        raise ValueError("Action artifact must explicitly disclaim a hardware-safety guarantee")
    expected = {"track": track, "dataset": dataset, "dataset_revision": dataset_revision}
    observed = {key: metadata.get(key) for key in expected}
    if observed != expected:
        raise ValueError(f"Action-envelope identity mismatch: observed={observed}, expected={expected}")
    if not all(key in metadata.get("sources", {}) for key in ("calibration_manifest", "golden_corpus", "norm_stats")):
        raise ValueError("Action-envelope provenance is missing required corpus or normalization sources")
    if action_low.shape != action_high.shape or action_low.shape != action_mask.shape or action_low.ndim != 1:
        raise ValueError("Action-envelope low/high/mask arrays must be matching vectors")
    active_dim = 7 if track == "libero" else 8
    expected_action_mask = np.zeros(action_mask.shape, dtype=np.bool_)
    if action_mask.shape[0] < active_dim:
        raise ValueError(f"{track} action envelope has fewer than {active_dim} model dimensions")
    expected_action_mask[:active_dim] = True
    if not np.array_equal(action_mask, expected_action_mask):
        raise ValueError(f"{track} action envelope does not select exactly its {active_dim} robot dimensions")
    if physical_low.shape != physical_high.shape or physical_low.shape != physical_mask.shape:
        raise ValueError("Physical-envelope low/high/mask arrays must be matching vectors")
    if state_dependent.shape != physical_mask.shape:
        raise ValueError("Physical state-dependence mask does not match the physical action shape")
    if physical_mask.shape != (active_dim,):
        raise ValueError(f"{track} physical envelope must contain exactly {active_dim} robot dimensions")
    expected_state_dependent = np.zeros(active_dim, dtype=np.bool_)
    expected_physical_mask = np.ones(active_dim, dtype=np.bool_)
    if track == "droid":
        expected_state_dependent[:7] = True
        expected_physical_mask[:7] = False
    if not np.array_equal(state_dependent, expected_state_dependent) or not np.array_equal(
        physical_mask, expected_physical_mask
    ):
        raise ValueError(f"{track} physical action state-dependence semantics are invalid")
    if np.any(state_dependent & physical_mask):
        raise ValueError("State-dependent physical dimensions must not claim static bounds")
    if np.any(np.isfinite(physical_low[state_dependent])) or np.any(np.isfinite(physical_high[state_dependent])):
        raise ValueError("State-dependent physical bounds must remain unset")
    if (
        not np.all(np.isfinite(physical_low[physical_mask]))
        or not np.all(np.isfinite(physical_high[physical_mask]))
        or np.any(physical_low[physical_mask] >= physical_high[physical_mask])
    ):
        raise ValueError("State-independent physical corpus-envelope bounds must be finite and ordered")
    return action_low, action_high, action_mask, metadata, sidecar_path


def main() -> int:
    args = _parse_args()
    if args.action_limits_npz is None:
        raise ValueError("--action-limits-npz is required for the corpus-envelope regression gate")
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
    from openpi.exporting.artifacts import write_stage_manifest
    from openpi.exporting.numerics import action_diagnostics
    from openpi.exporting.numerics import compare_outputs
    from openpi.exporting.onnx_artifacts import onnx_model_identity

    output = args.output or args.artifact_dir / f"onnx-validation.{args.precision}.json"
    manifest = args.artifact_dir / f"onnx-validation-manifest.{args.precision}.json"
    require_absent_outputs([output, manifest], stage=f"ONNX {args.precision} validation")
    suffix = f"{args.precision}.onnx"
    prefix_model = args.artifact_dir / f"encode-prefix.{suffix}"
    decoder_model = args.artifact_dir / f"decode-denoise.{suffix}"
    models = {
        "encode-prefix": onnx_model_identity(prefix_model),
        "decode-denoise": onnx_model_identity(decoder_model),
    }
    prefix_candidate, prefix_providers = _run_graph(
        prefix_model,
        args.artifact_dir / "encode-inputs.npz",
        args.provider,
    )
    decoder_isolated_candidate, decoder_providers = _run_graph(
        decoder_model,
        args.artifact_dir / "decode-inputs.npz",
        args.provider,
    )
    with np.load(args.artifact_dir / "decode-inputs.npz", allow_pickle=False) as archive:
        decoder_saved_inputs = {name: np.asarray(archive[name]) for name in archive.files}
    end_to_end_inputs = _merge_candidate_prefix(decoder_saved_inputs, prefix_candidate)
    end_to_end_candidate, _ = _run_graph(
        decoder_model,
        args.artifact_dir / "decode-inputs.npz",
        args.provider,
        input_overrides=end_to_end_inputs,
    )
    prefix_reference = _reference(args.artifact_dir / "encode-reference.npz")
    decoder_reference = _reference(args.artifact_dir / "decode-reference.npz")
    prefix_report = compare_outputs(
        prefix_reference,
        prefix_candidate,
        cosine_threshold=args.cosine_threshold,
    )
    decoder_isolated_report = compare_outputs(
        decoder_reference,
        decoder_isolated_candidate,
        cosine_threshold=args.cosine_threshold,
    )
    end_to_end_report = compare_outputs(
        decoder_reference,
        end_to_end_candidate,
        cosine_threshold=args.cosine_threshold,
    )

    action_low, action_high, action_mask, action_envelope, action_envelope_sidecar = _load_action_envelope(
        args.action_limits_npz,
        track=args.track,
        dataset=args.dataset,
        dataset_revision=args.dataset_revision,
    )
    actions = action_diagnostics(
        decoder_reference["actions"],
        end_to_end_candidate["actions"],
        action_low=action_low,
        action_high=action_high,
        action_mask=action_mask,
        max_abs_joint_bias=args.max_abs_joint_bias,
    )
    passes = (
        prefix_report["passes"]
        and decoder_isolated_report["passes"]
        and end_to_end_report["passes"]
        and actions["bias_passes"]
    )
    passes &= actions["action_limits_pass"]
    report = {
        "schema_version": 1,
        "precision": args.precision,
        "cosine_threshold": args.cosine_threshold,
        "passes": passes,
        "prefix": prefix_report,
        "decoder_isolated": decoder_isolated_report,
        "end_to_end": end_to_end_report,
        "end_to_end_actions": actions,
        "providers": {"prefix": prefix_providers, "decoder": decoder_providers},
        "models": models,
        "action_envelope": action_envelope,
        "provenance": {
            "track": args.track,
            "dataset": args.dataset,
            "dataset_revision": args.dataset_revision,
            "image_digest": args.image_digest,
            "instance_type": args.instance_type,
            "instance_id": args.instance_id,
            "cost_reservation": args.cost_reservation,
            "instance_identity_source": live_identity.instance_identity_source,
        },
    }
    write_json_new(output, report)
    artifact_paths = [prefix_model, decoder_model, output, args.action_limits_npz, action_envelope_sidecar]
    artifact_paths.extend(_external_onnx_data(prefix_model))
    artifact_paths.extend(_external_onnx_data(decoder_model))
    write_stage_manifest(
        manifest,
        stage=f"onnx-validation-{args.precision}",
        track=args.track,
        command=sys.argv,
        image_digest=args.image_digest,
        dataset=args.dataset,
        dataset_revision=args.dataset_revision,
        instance_type=args.instance_type,
        instance_id=args.instance_id,
        cost_reservation=args.cost_reservation,
        artifacts=artifact_paths,
        details=report,
        metrics=report,
    )
    print(json.dumps({"report": str(output), "manifest": str(manifest), "passes": passes}, indent=2))
    return 0 if passes else 2


if __name__ == "__main__":
    raise SystemExit(main())
