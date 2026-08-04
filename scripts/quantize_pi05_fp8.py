#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Apply selective ModelOpt FP8 Q/DQ to pi0.5 transformer MLPs only."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import json
import pathlib
import re
import sys
import tempfile

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=pathlib.Path)
    parser.add_argument("--calibration-manifest", required=True, type=pathlib.Path)
    parser.add_argument(
        "--bf16-validation-report",
        type=pathlib.Path,
        help="defaults to ARTIFACT_DIR/onnx-validation.bf16.json",
    )
    parser.add_argument("--track", required=True, choices=("libero", "droid"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--instance-type", default="g7e.4xlarge")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--cost-reservation", required=True)
    parser.add_argument("--chunks", type=int, default=1024)
    parser.add_argument("--image-count", type=int, default=3)
    parser.add_argument("--include-regex", help="narrow manual node selector if exporter names differ")
    parser.add_argument("--keep-intermediate-files", action="store_true")
    return parser.parse_args()


def _cast(array: np.ndarray, type_name: str) -> np.ndarray:
    types = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(bool)": np.bool_,
    }
    if type_name == "tensor(bfloat16)":
        import ml_dtypes

        return np.asarray(array, dtype=ml_dtypes.bfloat16)
    if type_name not in types:
        raise TypeError(f"Unsupported ONNX calibration input type: {type_name}")
    return np.asarray(array, dtype=types[type_name])


def _cast_feed(session, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {item.name: _cast(values[item.name], item.type) for item in session.get_inputs()}


def _external_files(model_path: pathlib.Path) -> list[pathlib.Path]:
    import onnx

    model = onnx.load(model_path, load_external_data=False)
    locations = set()
    for initializer in model.graph.initializer:
        for field in initializer.external_data:
            if field.key == "location":
                locations.add(field.value)
    result = [(model_path.parent / location).resolve() for location in sorted(locations)]
    missing = [path for path in result if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Quantized ONNX external data missing: {missing}")
    return result


def main() -> int:
    args = _parse_args()
    if args.chunks != 1024:
        raise ValueError("This reproduction fixes calibration at exactly 1,024 chunks per track")

    from openpi.exporting.runtime_identity import require_live_runtime_identity

    live_identity = require_live_runtime_identity(
        image_digest=args.image_digest,
        instance_type=args.instance_type,
        instance_id=args.instance_id,
    )

    from openpi.exporting.artifacts import require_absent_outputs
    from openpi.exporting.artifacts import require_clean_source_identity

    require_clean_source_identity()

    from openpi.exporting.artifacts import write_stage_manifest
    from openpi.exporting.calibration import StreamingCalibrationReader
    from openpi.exporting.calibration import load_calibration_manifest
    from openpi.exporting.calibration import load_record_arrays
    from openpi.exporting.calibration import prefix_feed
    from openpi.exporting.calibration import select_stratified
    from openpi.exporting.calibration import stratum_counts
    from openpi.exporting.fp8 import prepare_fp32_calibration_model
    from openpi.exporting.fp8 import quantization_audit
    from openpi.exporting.fp8 import select_transformer_mlp_nodes
    from openpi.exporting.onnx_artifacts import require_validated_models

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    prefix_bf16 = args.artifact_dir / "encode-prefix.bf16.onnx"
    decoder_bf16 = args.artifact_dir / "decode-denoise.bf16.onnx"
    fp8_outputs = [
        args.artifact_dir / "encode-prefix.fp8.onnx",
        args.artifact_dir / "decode-denoise.fp8.onnx",
        args.artifact_dir / "modelopt-encode-prefix.log",
        args.artifact_dir / "modelopt-decode-denoise.log",
        args.artifact_dir / "fp8-manifest.json",
    ]
    require_absent_outputs(fp8_outputs, stage="ModelOpt FP8 quantization")
    bf16_models = {"encode-prefix": prefix_bf16, "decode-denoise": decoder_bf16}
    validation_path = args.bf16_validation_report or args.artifact_dir / "onnx-validation.bf16.json"
    validation = json.loads(validation_path.read_text())
    require_validated_models(
        validation,
        precision="bf16",
        models=bf16_models,
        provenance={
            "track": args.track,
            "dataset": args.dataset,
            "dataset_revision": args.dataset_revision,
            "image_digest": args.image_digest,
            "instance_type": args.instance_type,
            "instance_id": args.instance_id,
        },
    )

    from modelopt.onnx.quantization import quantize
    import onnx
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(f"CUDA ONNX Runtime is required for calibration; providers={sorted(available)}")
    records = select_stratified(load_calibration_manifest(args.calibration_manifest), args.chunks)
    record_datasets = {record.metadata.get("dataset") for record in records}
    record_revisions = {record.metadata.get("dataset_revision") for record in records}
    if record_datasets != {args.dataset} or record_revisions != {args.dataset_revision}:
        raise ValueError(
            f"Calibration corpus provenance mismatch: datasets={record_datasets}, revisions={record_revisions}"
        )
    prefix_fp8 = args.artifact_dir / "encode-prefix.fp8.onnx"
    decoder_fp8 = args.artifact_dir / "decode-denoise.fp8.onnx"
    audits = {}
    calibration_bridges = {}
    with tempfile.TemporaryDirectory(prefix="pi05-modelopt-", dir=args.artifact_dir) as temporary:
        temporary_dir = pathlib.Path(temporary)
        prefix_fp32 = temporary_dir / "encode-prefix.fp32-calibration.onnx"
        decoder_fp32 = temporary_dir / "decode-denoise.fp32-calibration.onnx"
        calibration_bridges["encode-prefix"] = prepare_fp32_calibration_model(prefix_bf16, prefix_fp32)
        calibration_bridges["decode-denoise"] = prepare_fp32_calibration_model(decoder_bf16, decoder_fp32)

        prefix_session = ort.InferenceSession(
            str(prefix_fp32), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        prefix_input_names = {item.name for item in prefix_session.get_inputs()}

        def prefix_feeds() -> Iterator[dict[str, np.ndarray]]:
            for record in records:
                arrays = load_record_arrays(record, image_count=args.image_count)
                values = prefix_feed(arrays, image_count=args.image_count)
                yield _cast_feed(prefix_session, {name: values[name] for name in prefix_input_names})

        decoder_session = ort.InferenceSession(
            str(decoder_fp32), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        def decoder_feeds() -> Iterator[dict[str, np.ndarray]]:
            for record in records:
                arrays = load_record_arrays(record, image_count=args.image_count)
                encoded = prefix_session.run(
                    None, _cast_feed(prefix_session, prefix_feed(arrays, image_count=args.image_count))
                )
                encoded_values = {
                    output.name: value for output, value in zip(prefix_session.get_outputs(), encoded, strict=True)
                }
                batch_size = arrays["state"].shape[0]
                values = {
                    "state": arrays["state"],
                    "x_t": arrays["noise"],
                    "timestep": np.ones((batch_size,), dtype=np.float32),
                    "target_time": np.zeros((batch_size,), dtype=np.float32),
                    **encoded_values,
                }
                yield _cast_feed(decoder_session, values)

        for source, output, reader_factory, graph_name in (
            (prefix_fp32, prefix_fp8, prefix_feeds, "encode-prefix"),
            (decoder_fp32, decoder_fp8, decoder_feeds, "decode-denoise"),
        ):
            source_model = onnx.load(source, load_external_data=False)
            selected = select_transformer_mlp_nodes(source_model.graph.node, include_regex=args.include_regex)
            quantize(
                onnx_path=str(source),
                quantize_mode="fp8",
                calibration_method="max",
                calibration_data_reader=StreamingCalibrationReader(reader_factory),
                calibration_eps=["cuda:0"],
                op_types_to_quantize=["MatMul", "Gemm"],
                nodes_to_quantize=[f"^{re.escape(name)}$" for name in selected],
                use_external_data_format=True,
                keep_intermediate_files=args.keep_intermediate_files,
                output_path=str(output),
                high_precision_dtype="bf16",
                mha_accumulation_dtype="fp32",
                disable_mha_qdq=True,
                log_file=str(args.artifact_dir / f"modelopt-{graph_name}.log"),
            )
            quantized_model = onnx.load(output, load_external_data=False)
            audits[graph_name] = quantization_audit(quantized_model, selected)

    artifacts = [prefix_fp8, decoder_fp8, args.calibration_manifest, validation_path]
    artifacts.extend(_external_files(prefix_fp8))
    artifacts.extend(_external_files(decoder_fp8))
    artifacts.extend(args.artifact_dir / f"modelopt-{name}.log" for name in ("encode-prefix", "decode-denoise"))
    manifest = args.artifact_dir / "fp8-manifest.json"
    write_stage_manifest(
        manifest,
        stage="modelopt-fp8-ptq",
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
            "quantize_mode": "fp8",
            "calibration_method": "max",
            "calibration_chunks": len(records),
            "stratum_counts": stratum_counts(records),
            "high_precision_dtype": "bf16",
            "mha_accumulation_dtype": "fp32",
            "calibration_bridge": {
                "reason": "ModelOpt 0.45 calibrates FLOAT/FLOAT16 activations, not BF16 activations",
                "source_precision": "bf16",
                "temporary_precision": "float32",
                "output_high_precision": "bf16",
                "conversion_counts": calibration_bridges,
            },
            "bf16_validation_report": str(validation_path.resolve()),
            "runtime_identity_source": live_identity.instance_identity_source,
            "audits": audits,
            "validation_status": "pending; run scripts/validate_pi05_onnx.py --precision fp8",
        },
        metrics={"calibration_chunks": len(records), "audits": audits},
    )
    print(json.dumps({"manifest": str(manifest), "audits": audits}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
