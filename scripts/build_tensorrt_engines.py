#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Build validated fixed-shape pi0.5 TensorRT engines with trtexec."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import pathlib
import re
import shutil
import subprocess
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=pathlib.Path)
    parser.add_argument("--precision", required=True, choices=("bf16", "fp8"))
    parser.add_argument("--validation-report", required=True, type=pathlib.Path)
    parser.add_argument("--export-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--export-image-digest", required=True, help="pinned policy image used for BF16 export")
    parser.add_argument("--fp8-manifest", type=pathlib.Path, help="required when --precision=fp8")
    parser.add_argument("--track", required=True, choices=("libero", "droid"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--instance-type", default="g7e.4xlarge")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--cost-reservation", required=True)
    parser.add_argument("--workspace-mib", type=int, default=8192)
    parser.add_argument("--trtexec", default="trtexec")
    parser.add_argument("--execute", action="store_true", help="build engines; otherwise print the exact commands")
    return parser.parse_args()


def _parse_major_version(text: str) -> int:
    dotted = re.search(r"TensorRT(?:\s+version)?[^0-9]*([0-9]+)\.[0-9]+", text, re.IGNORECASE)
    if dotted:
        return int(dotted.group(1))
    # Older trtexec banners use a compact build code, for example v100900 for
    # TensorRT 10.9 and v110000 for TensorRT 11.0.
    compact = re.search(r"TensorRT\s+v([0-9]{6})", text, re.IGNORECASE)
    if compact:
        code = compact.group(1)
        return int(code[:-4])
    raise RuntimeError(f"Could not parse TensorRT version from: {text.strip()}")


def _major_version(executable: str) -> tuple[int, str]:
    # TensorRT 11's trtexec prints a valid version banner for ``--version``
    # and then exits 1 because it still expects a model. ``--help`` emits the
    # same build banner and exits successfully, so it is the fail-closed probe.
    result = subprocess.run([executable, "--help"], check=True, text=True, capture_output=True)
    text = result.stdout + result.stderr
    return _parse_major_version(text), text.strip()


def _require_pinned_major(major: int) -> None:
    if major != 11:
        raise RuntimeError(f"This reproduction is pinned to TensorRT 11, found major version {major}")


def _command(executable, model, engine, timing_cache, layer_info, workspace_mib, major):
    command = [
        executable,
        f"--onnx={model}",
        f"--saveEngine={engine}",
        f"--timingCacheFile={timing_cache}",
        f"--memPoolSize=workspace:{workspace_mib}",
        "--builderOptimizationLevel=5",
        "--profilingVerbosity=detailed",
        f"--exportLayerInfo={layer_info}",
        "--skipInference",
        "--verbose",
    ]
    # TensorRT 11 is always strongly typed and removed all precision flags.
    if major < 11:
        command.append("--stronglyTyped")
    return command


def _record_for_name(manifest: Mapping, name: str) -> Mapping:
    records = [
        record
        for record in manifest.get("artifacts", ())
        if isinstance(record, Mapping) and pathlib.Path(str(record.get("path", ""))).name == name
    ]
    if len(records) != 1:
        raise ValueError(f"Manifest must identify exactly one artifact named {name!r}")
    record = records[0]
    if set(record) != {"path", "bytes", "sha256"}:
        raise ValueError(f"Artifact record for {name!r} has an invalid schema")
    return record


def _require_file_record(manifest: Mapping, path: pathlib.Path) -> None:
    from openpi.exporting.onnx_artifacts import file_identity

    record = _record_for_name(manifest, path.name)
    expected = {"name": path.name, "bytes": record["bytes"], "sha256": record["sha256"]}
    if file_identity(path) != expected:
        raise ValueError(f"Artifact no longer matches its source manifest: {path}")


def _require_model_records(manifest: Mapping, models: Mapping[str, Mapping]) -> None:
    if set(models) != {"encode-prefix", "decode-denoise"}:
        raise ValueError("Validation must identify the complete split ONNX model set")
    for graph_name, identity in models.items():
        if not isinstance(identity, Mapping) or not isinstance(identity.get("model"), Mapping):
            raise ValueError(f"Validation identity for {graph_name!r} is malformed")
        identities = (identity["model"], *identity.get("external_data", ()))
        for artifact in identities:
            name = artifact.get("name")
            if not isinstance(name, str) or pathlib.PurePath(name).name != name:
                raise ValueError(f"ONNX identity for {graph_name!r} has a non-portable artifact name")
            record = _record_for_name(manifest, name)
            if record.get("bytes") != artifact.get("bytes") or record.get("sha256") != artifact.get("sha256"):
                raise ValueError(f"{graph_name} identity does not match source-manifest artifact {name!r}")


def _require_stage_identity(
    manifest: Mapping,
    *,
    stage: str,
    track: str,
    dataset: str,
    dataset_revision: str,
    runtime: Mapping[str, str],
) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("stage") != stage or manifest.get("track") != track:
        raise ValueError(f"Unexpected {stage} source-manifest identity")
    if manifest.get("source", {}).get("dirty") is not False:
        raise ValueError(f"{stage} source manifest must come from a clean source tree")
    if manifest.get("dataset") != {"name": dataset, "revision": dataset_revision}:
        raise ValueError(f"{stage} source-manifest dataset identity differs")
    if manifest.get("runtime") != runtime:
        raise ValueError(f"{stage} source-manifest runtime identity differs")


def _validate_policy_provenance(
    *,
    precision: str,
    export_manifest_path: pathlib.Path,
    fp8_manifest_path: pathlib.Path | None,
    validation: Mapping,
    artifact_dir: pathlib.Path,
    track: str,
    dataset: str,
    dataset_revision: str,
    runtime: Mapping[str, str],
    export_runtime: Mapping[str, str],
) -> tuple[dict, list[pathlib.Path]]:
    """Validate and return the deployment identity sealed into the engine manifest."""

    if validation.get("passes") is not True or validation.get("precision") != precision:
        raise ValueError("Engine source must have a passing matching-precision validation report")
    expected_validation_provenance = {
        "track": track,
        "dataset": dataset,
        "dataset_revision": dataset_revision,
        **runtime,
    }
    for key, expected in expected_validation_provenance.items():
        if validation.get("provenance", {}).get(key) != expected:
            raise ValueError(f"{precision} validation provenance differs for {key}")

    if export_manifest_path.resolve().parent != artifact_dir.resolve():
        raise ValueError("The export manifest must be stored in the artifact directory for portable replay")
    export_manifest = json.loads(export_manifest_path.read_text())
    _require_stage_identity(
        export_manifest,
        stage="onnx-export-bf16",
        track=track,
        dataset=dataset,
        dataset_revision=dataset_revision,
        runtime=export_runtime,
    )
    export_details = export_manifest.get("details", {})
    config = export_details.get("config")
    checkpoint = export_details.get("checkpoint")
    if not isinstance(config, str) or config != f"pi05_{track}_l09_snapflow":
        raise ValueError(f"Export manifest has the wrong one-step policy config: {config!r}")
    if (
        not isinstance(checkpoint, Mapping)
        or set(checkpoint) != {"path", "sha256", "assets"}
        or not isinstance(checkpoint.get("path"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(checkpoint.get("sha256", "")))
        or not isinstance(checkpoint.get("assets"), list)
        or not checkpoint["assets"]
    ):
        raise ValueError("Export manifest checkpoint identity is malformed")
    weight_record = _record_for_name(export_manifest, pathlib.Path(checkpoint["path"]).name)
    if weight_record.get("sha256") != checkpoint["sha256"]:
        raise ValueError("Export manifest checkpoint hash differs from its artifact record")
    norm_stats = 0
    asset_names: set[str] = set()
    for asset in checkpoint["assets"]:
        if (
            not isinstance(asset, Mapping)
            or set(asset) != {"name", "bytes", "sha256"}
            or not isinstance(asset.get("name"), str)
            or pathlib.PurePosixPath(asset["name"]).is_absolute()
            or ".." in pathlib.PurePosixPath(asset["name"]).parts
            or not asset["name"].startswith("assets/")
            or asset["name"] in asset_names
        ):
            raise ValueError("Export manifest checkpoint asset identity is malformed")
        asset_names.add(asset["name"])
        norm_stats += pathlib.PurePosixPath(asset["name"]).name == "norm_stats.json"
        record = _record_for_name(export_manifest, pathlib.PurePosixPath(asset["name"]).name)
        if record.get("bytes") != asset.get("bytes") or record.get("sha256") != asset.get("sha256"):
            raise ValueError(f"Checkpoint asset differs from export-manifest record: {asset['name']}")
    if norm_stats != 1:
        raise ValueError("Export manifest must bind exactly one normalization asset")

    source_paths = [export_manifest_path]
    if precision == "bf16":
        _require_model_records(export_manifest, validation.get("models", {}))
    else:
        if fp8_manifest_path is None:
            raise ValueError("--fp8-manifest is required when --precision=fp8")
        if fp8_manifest_path.resolve().parent != artifact_dir.resolve():
            raise ValueError("The FP8 manifest must be stored in the artifact directory for portable replay")
        fp8_manifest = json.loads(fp8_manifest_path.read_text())
        _require_stage_identity(
            fp8_manifest,
            stage="modelopt-fp8-ptq",
            track=track,
            dataset=dataset,
            dataset_revision=dataset_revision,
            runtime=runtime,
        )
        _require_model_records(fp8_manifest, validation.get("models", {}))
        bf16_validation_name = pathlib.Path(str(fp8_manifest.get("details", {}).get("bf16_validation_report", ""))).name
        if not bf16_validation_name:
            raise ValueError("FP8 manifest does not identify its BF16 validation report")
        bf16_validation_path = artifact_dir / bf16_validation_name
        _require_file_record(fp8_manifest, bf16_validation_path)
        bf16_validation = json.loads(bf16_validation_path.read_text())
        if bf16_validation.get("passes") is not True or bf16_validation.get("precision") != "bf16":
            raise ValueError("FP8 source does not reference a passing BF16 validation report")
        expected_provenance = {
            "track": track,
            "dataset": dataset,
            "dataset_revision": dataset_revision,
            **runtime,
        }
        for key, expected in expected_provenance.items():
            if bf16_validation.get("provenance", {}).get(key) != expected:
                raise ValueError(f"BF16 validation provenance differs for {key}")
        _require_model_records(export_manifest, bf16_validation.get("models", {}))
        source_paths.extend((fp8_manifest_path, bf16_validation_path))

    if precision == "bf16" and fp8_manifest_path is not None:
        raise ValueError("--fp8-manifest is invalid when --precision=bf16")
    from openpi.exporting.onnx_artifacts import file_identity

    return (
        {
            "schema_version": 1,
            "protocol": "openpi-policy-websocket-v1",
            "config": config,
            "checkpoint": dict(checkpoint),
            "precision": precision,
            "num_denoise_steps": 1,
            "source_manifests": [file_identity(path) for path in source_paths],
            "export_runtime": dict(export_runtime),
        },
        source_paths,
    )


def main() -> int:
    args = _parse_args()
    from openpi.exporting.runtime_identity import query_gpu_inventory
    from openpi.exporting.runtime_identity import require_live_runtime_identity
    from openpi.exporting.runtime_identity import require_same_image_digest

    live_identity = require_live_runtime_identity(
        image_digest=args.image_digest,
        instance_type=args.instance_type,
        instance_id=args.instance_id,
    )
    require_same_image_digest(args.export_image_digest, args.image_digest)
    from openpi.exporting.artifacts import require_absent_outputs
    from openpi.exporting.artifacts import require_clean_source_identity

    require_clean_source_identity()
    if args.validation_report.resolve().parent != args.artifact_dir.resolve():
        raise ValueError("The validation report must be stored in the artifact directory for portable replay")
    validation = json.loads(args.validation_report.read_text())
    from openpi.exporting.onnx_artifacts import require_validated_models

    models = {
        graph: args.artifact_dir / f"{graph}.{args.precision}.onnx" for graph in ("encode-prefix", "decode-denoise")
    }
    require_validated_models(
        validation,
        precision=args.precision,
        models=models,
        provenance={
            "track": args.track,
            "dataset": args.dataset,
            "dataset_revision": args.dataset_revision,
            "instance_type": args.instance_type,
            "instance_id": args.instance_id,
        },
    )
    expected_runtime = {
        "image_digest": args.image_digest,
        "instance_type": args.instance_type,
        "instance_id": args.instance_id,
    }
    expected_export_runtime = {
        "image_digest": args.export_image_digest,
        "instance_type": args.instance_type,
        "instance_id": args.instance_id,
    }
    policy_contract, policy_source_paths = _validate_policy_provenance(
        precision=args.precision,
        export_manifest_path=args.export_manifest,
        fp8_manifest_path=args.fp8_manifest,
        validation=validation,
        artifact_dir=args.artifact_dir,
        track=args.track,
        dataset=args.dataset,
        dataset_revision=args.dataset_revision,
        runtime=expected_runtime,
        export_runtime=expected_export_runtime,
    )
    executable = shutil.which(args.trtexec)
    if executable is None:
        raise FileNotFoundError(
            "trtexec is not installed in this image; use the pinned NVIDIA TensorRT 11 build image or mount its bin directory"
        )
    major, version = _major_version(executable)
    _require_pinned_major(major)
    gpu_inventory = query_gpu_inventory()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    timing_cache = args.artifact_dir / f"tensorrt-{args.precision}.timing.cache"
    build_outputs = [
        timing_cache,
        args.artifact_dir / f"tensorrt-manifest.{args.precision}.json",
        *(
            args.artifact_dir / f"{graph}.{args.precision}.{suffix}"
            for graph in ("encode-prefix", "decode-denoise")
            for suffix in ("plan", "layers.json", "trtexec.log")
        ),
    ]
    require_absent_outputs(build_outputs, stage=f"TensorRT {args.precision} build")
    commands = []
    artifacts = []
    for graph in ("encode-prefix", "decode-denoise"):
        model = models[graph]
        engine = args.artifact_dir / f"{graph}.{args.precision}.plan"
        layer_info = args.artifact_dir / f"{graph}.{args.precision}.layers.json"
        log = args.artifact_dir / f"{graph}.{args.precision}.trtexec.log"
        command = _command(executable, model, engine, timing_cache, layer_info, args.workspace_mib, major)
        commands.append(command)
        if args.execute:
            with log.open("w") as stream:
                subprocess.run(command, check=True, text=True, stdout=stream, stderr=subprocess.STDOUT)
            artifacts.extend((engine, layer_info, log))
    if not args.execute:
        print(json.dumps({"tensorrt_version": version, "commands": commands}, indent=2))
        return 0

    from openpi.exporting.artifacts import write_stage_manifest

    artifacts.extend((timing_cache, args.validation_report, *policy_source_paths))
    manifest = args.artifact_dir / f"tensorrt-manifest.{args.precision}.json"
    write_stage_manifest(
        manifest,
        stage=f"tensorrt-build-{args.precision}",
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
            "tensorrt_version": version,
            "strongly_typed": True,
            "precision_source": "explicit ONNX tensor types and Q/DQ nodes",
            "commands": commands,
            "gpu_inventory": list(gpu_inventory),
            "runtime_identity_source": live_identity.instance_identity_source,
            "validation_report": str(args.validation_report.resolve()),
            "policy_contract": policy_contract,
        },
    )
    print(
        json.dumps(
            {"manifest": str(manifest), "engines": [str(path) for path in artifacts if path.suffix == ".plan"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
