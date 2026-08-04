#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Export fixed-shape BF16 pi0.5 prefix and one-step denoise ONNX graphs."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
from typing import Any

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="pi05_*_l09_snapflow training config")
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--calibration-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--track", required=True, choices=("libero", "droid"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--instance-type", default="g7e.4xlarge")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--cost-reservation", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--opset", type=int, default=20)
    parser.add_argument("--image-count", type=int, default=3)
    return parser.parse_args()


def _weights_path(path: pathlib.Path) -> pathlib.Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint weights not found: {path}")
    return path


def _checkpoint_assets(weights: pathlib.Path) -> tuple[pathlib.Path, tuple[pathlib.Path, ...]]:
    """Return a checkpoint root and its complete, local normalization asset set."""

    checkpoint_root = weights.parent.resolve()
    assets_root = checkpoint_root / "assets"
    if not assets_root.is_dir() or assets_root.is_symlink():
        raise FileNotFoundError(f"Checkpoint-local assets directory is required: {assets_root}")
    entries = tuple(assets_root.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("Checkpoint assets must not contain symlinks")
    assets = tuple(sorted(path for path in entries if path.is_file()))
    if not assets:
        raise FileNotFoundError(f"Checkpoint-local assets directory is empty: {assets_root}")
    norm_stats = [path for path in assets if path.name == "norm_stats.json"]
    if len(norm_stats) != 1:
        raise ValueError(f"Checkpoint must contain exactly one normalization asset, found {len(norm_stats)}")
    return checkpoint_root, assets


def _to_numpy(tensor: Any) -> np.ndarray:
    # NumPy has no native BF16 dtype.  Golden files retain the exact BF16
    # values represented as FP32; validation casts inputs back to the ONNX type.
    import torch

    value = tensor.detach().cpu()
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.numpy()


def _save_npz(path: pathlib.Path, names: tuple[str, ...], tensors: tuple[Any, ...]) -> None:
    if len(names) != len(tensors):
        raise ValueError(f"Cannot save {len(tensors)} tensors under {len(names)} names")
    np.savez(path, **{name: _to_numpy(tensor) for name, tensor in zip(names, tensors, strict=True)})


def _externalize_onnx(path: pathlib.Path) -> pathlib.Path:
    import onnx

    data_path = path.with_suffix(path.suffix + ".data")
    model = onnx.load(path, load_external_data=True)
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=0,
        convert_attribute=False,
    )
    metadata = onnx.load(path, load_external_data=False)
    inline = [initializer.name for initializer in metadata.graph.initializer if initializer.data_location != 1]
    if inline:
        raise RuntimeError(f"Expected every ONNX initializer to use external data; inline={inline[:5]}")
    if not data_path.is_file():
        raise FileNotFoundError(f"ONNX external weight file was not created: {data_path}")
    return data_path


def _torch_export(module, inputs, output, input_names, output_names, opset):
    import torch

    with torch.no_grad():
        torch.onnx.export(
            module,
            inputs,
            output,
            input_names=list(input_names),
            output_names=list(output_names),
            opset_version=opset,
            # The pinned OpenPI environment intentionally holds ml-dtypes at
            # 0.4.1 for JAX.  The legacy exporter avoids an incompatible
            # onnxscript/onnx-ir dependency while retaining external-data and
            # fixed-shape export support in PyTorch 2.7.1.
            dynamo=False,
            external_data=True,
            dynamic_shapes=None,
            do_constant_folding=True,
        )


def main() -> int:
    args = _parse_args()
    if args.opset < 20:
        raise ValueError("BF16 export requires opset 20+; ModelOpt will upgrade FP8 output to opset 21+")

    from openpi.exporting.runtime_identity import require_live_runtime_identity

    live_identity = require_live_runtime_identity(
        image_digest=args.image_digest,
        instance_type=args.instance_type,
        instance_id=args.instance_id,
    )

    from openpi.exporting.artifacts import prepare_fresh_output_directory
    from openpi.exporting.artifacts import require_clean_source_identity

    require_clean_source_identity()
    prepare_fresh_output_directory(args.output_dir, stage="ONNX export")

    import safetensors.torch
    import torch

    from openpi.exporting.artifacts import sha256_file
    from openpi.exporting.artifacts import write_stage_manifest
    from openpi.exporting.calibration import load_calibration_manifest
    from openpi.exporting.calibration import load_record_arrays
    from openpi.exporting.pi05_onnx import DecodeDenoiseWrapper
    from openpi.exporting.pi05_onnx import EncodePrefixWrapper
    from openpi.exporting.pi05_onnx import assert_fixed_input_contract
    from openpi.models import pi0_config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.models_pytorch.snapflow import SnapFlowPI0Pytorch
    from openpi.training import config as training_config

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the production BF16 export")
    device = torch.device(args.device)
    train_config = training_config.get_config(args.config)
    model_config = dataclasses.replace(train_config.model, pytorch_compile_mode=None)
    if isinstance(model_config, pi0_config.SnapFlowPi0Config):
        model = SnapFlowPI0Pytorch(model_config)
        use_target_time = True
    else:
        model = PI0Pytorch(model_config)
        use_target_time = False

    model.to(device)
    weights = _weights_path(args.checkpoint)
    checkpoint_root, checkpoint_assets = _checkpoint_assets(weights)
    missing, unexpected = safetensors.torch.load_model(model, weights, strict=True, device=str(device))
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint did not load exactly: missing={missing}, unexpected={unexpected}")
    model.eval().requires_grad_(requires_grad=False)

    records = load_calibration_manifest(args.calibration_manifest)
    first_record = records[0]
    if first_record.metadata.get("dataset") != args.dataset:
        raise ValueError(f"Calibration dataset mismatch: {first_record.metadata.get('dataset')!r} != {args.dataset!r}")
    if first_record.metadata.get("dataset_revision") != args.dataset_revision:
        raise ValueError(
            "Calibration dataset revision mismatch: "
            f"{first_record.metadata.get('dataset_revision')!r} != {args.dataset_revision!r}"
        )
    arrays = load_record_arrays(first_record, image_count=args.image_count)
    prefix_wrapper = EncodePrefixWrapper(model, image_count=args.image_count).to(device).eval()
    prefix_inputs = tuple(torch.from_numpy(arrays[name]).to(device) for name in prefix_wrapper.input_names)
    prefix_contract = assert_fixed_input_contract(prefix_wrapper.input_names, prefix_inputs)
    with torch.no_grad():
        prefix_outputs = tuple(prefix_wrapper(*prefix_inputs))
    cache_layers = (len(prefix_outputs) - 1) // 2
    prefix_output_names = prefix_wrapper.output_names(len(prefix_outputs) - 1)

    decoder_wrapper = (
        DecodeDenoiseWrapper(
            model,
            cache_layers=cache_layers,
            use_target_time=use_target_time,
        )
        .to(device)
        .eval()
    )
    batch_size = arrays["state"].shape[0]
    if batch_size != 1:
        raise ValueError(f"The fixed export contract requires batch one, got {batch_size}")
    decoder_inputs = (
        torch.from_numpy(arrays["state"]).to(device),
        torch.from_numpy(arrays["noise"]).to(device),
        torch.ones(batch_size, dtype=torch.float32, device=device),
        torch.zeros(batch_size, dtype=torch.float32, device=device),
        *prefix_outputs,
    )
    decoder_contract = assert_fixed_input_contract(decoder_wrapper.input_names, decoder_inputs)
    with torch.no_grad():
        decoder_outputs = (decoder_wrapper(*decoder_inputs),)

    prefix_path = args.output_dir / "encode-prefix.bf16.onnx"
    decoder_path = args.output_dir / "decode-denoise.bf16.onnx"
    _torch_export(
        prefix_wrapper,
        prefix_inputs,
        prefix_path,
        prefix_wrapper.input_names,
        prefix_output_names,
        args.opset,
    )
    prefix_data = _externalize_onnx(prefix_path)
    _torch_export(
        decoder_wrapper,
        decoder_inputs,
        decoder_path,
        decoder_wrapper.input_names,
        decoder_wrapper.output_names,
        args.opset,
    )
    decoder_data = _externalize_onnx(decoder_path)

    prefix_inputs_path = args.output_dir / "encode-inputs.npz"
    prefix_reference_path = args.output_dir / "encode-reference.npz"
    decoder_inputs_path = args.output_dir / "decode-inputs.npz"
    decoder_reference_path = args.output_dir / "decode-reference.npz"
    _save_npz(prefix_inputs_path, prefix_wrapper.input_names, prefix_inputs)
    _save_npz(prefix_reference_path, prefix_output_names, prefix_outputs)
    _save_npz(decoder_inputs_path, decoder_wrapper.input_names, decoder_inputs)
    _save_npz(decoder_reference_path, decoder_wrapper.output_names, decoder_outputs)

    artifacts = (
        weights,
        *checkpoint_assets,
        prefix_path,
        prefix_data,
        decoder_path,
        decoder_data,
        prefix_inputs_path,
        prefix_reference_path,
        decoder_inputs_path,
        decoder_reference_path,
        args.calibration_manifest.resolve(),
        first_record.path,
    )
    manifest_path = args.output_dir / "export-manifest.json"
    write_stage_manifest(
        manifest_path,
        stage="onnx-export-bf16",
        track=args.track,
        command=sys.argv,
        image_digest=args.image_digest,
        dataset=args.dataset,
        dataset_revision=args.dataset_revision,
        instance_type=args.instance_type,
        instance_id=args.instance_id,
        cost_reservation=args.cost_reservation,
        artifacts=artifacts,
        details={
            "config": args.config,
            "checkpoint": {
                "path": str(weights),
                "sha256": sha256_file(weights),
                "assets": [
                    {
                        "name": path.relative_to(checkpoint_root).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for path in checkpoint_assets
                ],
            },
            "opset": args.opset,
            "precision": "bfloat16",
            "batch_size": batch_size,
            "image_count": args.image_count,
            "cache_layers": cache_layers,
            "runtime_identity_source": live_identity.instance_identity_source,
            "golden_sample": {
                "calibration_manifest": str(args.calibration_manifest.resolve()),
                "path": str(first_record.path),
                "index": first_record.index,
                "stratum": first_record.stratum,
                "metadata": first_record.metadata,
            },
            "prefix_input_contract": prefix_contract,
            "decoder_input_contract": decoder_contract,
            "validation_status": "pending; run scripts/validate_pi05_onnx.py before TensorRT compilation",
        },
    )
    print(json.dumps({"manifest": str(manifest_path), "artifacts": [str(path) for path in artifacts]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
