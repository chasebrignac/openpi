#!/usr/bin/env python3
# ruff: noqa: PLR0912, PLR0915
"""Fail-closed validation and versioned S3 publication of TensorRT artifacts.

``plan`` and ``validate`` never call AWS.  S3 operations are possible only via
the exact spelling ``upload --execute``.  The payload revision is a digest of
the complete, manifest-covered directory and its build provenance; timestamps
and S3 locations therefore cannot change its content-addressed identity.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
import dataclasses
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import sys
import tempfile
from typing import Any
import urllib.parse

try:
    from scripts import repro_stage_data
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root.
    import repro_stage_data


DEFAULT_CONFIG = pathlib.Path("repro/reproduction.json")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
INSTANCE_ID_RE = re.compile(r"^i-(?:[0-9a-f]{8}|[0-9a-f]{17})$")
INSTANCE_TYPE_RE = re.compile(r"^[a-z0-9][a-z0-9.]{1,63}$")
PARTIAL_NAMES = re.compile(r"(?:^\.|\.partial$|\.part$|\.tmp$|~$)")
TRACKS = ("libero", "droid")
PRECISIONS = ("bf16", "fp8")

CommandRunner = Callable[[Sequence[str]], str]


@dataclasses.dataclass(frozen=True)
class DeclaredIdentity:
    source_commit: str
    image_digest: str
    track: str
    dataset: str
    dataset_revision: str
    precision: str
    instance_type: str
    instance_id: str
    gpu_inventory: tuple[str, ...]

    @property
    def runtime(self) -> dict[str, str]:
        return {
            "image_digest": self.image_digest,
            "instance_type": self.instance_type,
            "instance_id": self.instance_id,
        }


@dataclasses.dataclass(frozen=True)
class CompiledS3Target:
    bucket: str
    prefix: str
    payload_prefix: str
    payload_uri: str
    manifest_key: str
    manifest_uri: str


@dataclasses.dataclass(frozen=True)
class SealedArtifact:
    root: pathlib.Path
    revision: str
    files: tuple[dict[str, Any], ...]
    totals: dict[str, int]
    identity: DeclaredIdentity
    build_manifest: dict[str, Any]
    build_manifest_identity: dict[str, Any]


def _fail(message: str) -> None:
    raise repro_stage_data.StageError(message)


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise repro_stage_data.StageError(f"required artifact is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise repro_stage_data.StageError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"JSON artifact must contain an object: {path}")
    return value


def _file_identity(path: pathlib.Path, *, relative_to: pathlib.Path | None = None) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix() if relative_to is not None else path.name,
        "bytes": path.stat().st_size,
        "sha256": repro_stage_data.sha256_file(path),
    }


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def _payload_files(root: pathlib.Path) -> dict[str, pathlib.Path]:
    if not root.is_absolute():
        _fail(f"--artifact-dir must be absolute: {root}")
    if root.is_symlink() or not root.is_dir():
        _fail(f"compiled artifact directory is missing, not a directory, or a symlink: {root}")
    files: dict[str, pathlib.Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            _fail(f"compiled artifact directory contains a symlink: {path}")
        if path.is_dir():
            if path != root:
                _fail(f"compiled artifact directory must be flat; nested directory found: {path}")
            continue
        if not path.is_file():
            _fail(f"compiled artifact directory contains a non-regular entry: {path}")
        relative = path.relative_to(root).as_posix()
        if PARTIAL_NAMES.search(path.name):
            _fail(f"compiled artifact directory contains a partial/temporary file: {relative}")
        if path.stat().st_size <= 0:
            _fail(f"compiled artifact file is empty: {relative}")
        files[relative] = path
    if not files:
        _fail("compiled artifact directory is empty")
    return files


def _validate_declared_identity(identity: DeclaredIdentity) -> None:
    _validate_static_identity(identity)
    try:
        from openpi.exporting.runtime_identity import validate_gpu_inventory

        canonical = validate_gpu_inventory(identity.gpu_inventory)
    except (RuntimeError, ValueError) as exc:
        raise repro_stage_data.StageError(f"invalid live GPU inventory: {exc}") from exc
    if canonical != identity.gpu_inventory:
        _fail("GPU inventory must already be in canonical UUID, name, driver form")


def _validate_static_identity(identity: DeclaredIdentity) -> None:
    if COMMIT_RE.fullmatch(identity.source_commit) is None:
        _fail("--source-commit must be a full lowercase git commit")
    if IMAGE_DIGEST_RE.fullmatch(identity.image_digest) is None:
        _fail("--image-digest must be a sha256-pinned container digest")
    if identity.track not in TRACKS or identity.precision not in PRECISIONS:
        _fail("unsupported track or precision")
    if not identity.dataset or COMMIT_RE.fullmatch(identity.dataset_revision) is None:
        _fail("dataset name and full lowercase dataset revision are required")
    if (
        INSTANCE_TYPE_RE.fullmatch(identity.instance_type) is None
        or INSTANCE_ID_RE.fullmatch(identity.instance_id) is None
    ):
        _fail("build instance type or ID is malformed")


def verify_source_checkout(
    source_commit: str,
    *,
    runner: CommandRunner = repro_stage_data.run_command,
    environ: Mapping[str, str] | None = None,
) -> None:
    environment = os.environ if environ is None else environ
    protected = environment.get("PI05_SOURCE_SHA", "")
    if COMMIT_RE.fullmatch(protected) is None:
        _fail("protected PI05_SOURCE_SHA must be a full lowercase git commit")
    if protected != source_commit:
        _fail("protected PI05_SOURCE_SHA differs from --source-commit")
    head = runner(["git", "rev-parse", "HEAD"]).strip()
    if head != source_commit:
        _fail(f"executing Git HEAD differs from --source-commit: {head!r}")
    if runner(["git", "status", "--porcelain=v1", "--untracked-files=all"]).strip():
        _fail("executing source checkout is dirty; commit the reviewed tree before sealing or upload")


def _require_stage_identity(
    manifest: Mapping[str, Any],
    *,
    expected_stage: str,
    identity: DeclaredIdentity,
    path: pathlib.Path,
) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("stage") != expected_stage:
        _fail(f"unexpected stage identity in {path.name}; expected {expected_stage}")
    if manifest.get("track") != identity.track:
        _fail(f"track provenance mismatch in {path.name}")
    if manifest.get("source") != {"sha": identity.source_commit, "dirty": False}:
        _fail(f"source provenance mismatch or dirty source in {path.name}")
    if manifest.get("runtime") != identity.runtime:
        _fail(f"runtime provenance mismatch in {path.name}")
    if manifest.get("dataset") != {"name": identity.dataset, "revision": identity.dataset_revision}:
        _fail(f"dataset provenance mismatch in {path.name}")


def _record_map(manifest: Mapping[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    records = manifest.get("artifacts")
    if not isinstance(records, list) or not records:
        _fail(f"{label} has no artifact records")
    result: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            _fail(f"{label} artifact record {index} has an invalid schema")
        source_path = record.get("path")
        if not isinstance(source_path, str) or not pathlib.PurePosixPath(source_path).is_absolute():
            _fail(f"{label} artifact record {index} must use an absolute build path")
        name = pathlib.PurePosixPath(source_path).name
        if pathlib.PurePosixPath(source_path).name != name or not name:
            _fail(f"{label} artifact record {index} has an invalid path")
        size = record.get("bytes")
        digest = record.get("sha256")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0 or not isinstance(digest, str):
            _fail(f"{label} artifact record {index} has invalid size/hash fields")
        if SHA256_RE.fullmatch(digest) is None:
            _fail(f"{label} artifact record {index} has an invalid SHA-256")
        if name in result:
            _fail(f"{label} contains duplicate artifact basename {name!r}")
        result[name] = dict(record)
    return result


def _original_artifact_root(
    build_manifest: Mapping[str, Any], *, expected_validation_name: str
) -> pathlib.PurePosixPath:
    validation_path = build_manifest.get("details", {}).get("validation_report")
    if not isinstance(validation_path, str) or not pathlib.PurePosixPath(validation_path).is_absolute():
        _fail("TensorRT build manifest has no absolute validation-report provenance")
    path = pathlib.PurePosixPath(validation_path)
    if path.name != expected_validation_name:
        _fail("TensorRT build manifest identifies the wrong validation report")
    return path.parent


def _add_covered_record(
    covered: dict[str, dict[str, Any]],
    record: Mapping[str, Any],
    *,
    original_root: pathlib.PurePosixPath,
    label: str,
) -> None:
    path = pathlib.PurePosixPath(str(record["path"]))
    try:
        relative = path.relative_to(original_root)
    except ValueError:
        return  # Checkpoints, calibration corpora, and action envelopes are external evidence.
    if len(relative.parts) != 1:
        _fail(f"{label} covers nested compiled payload {relative}; the portable bundle must be flat")
    name = relative.as_posix()
    normalized = {"path": name, "bytes": record["bytes"], "sha256": record["sha256"]}
    existing = covered.get(name)
    if existing is not None and existing != normalized:
        _fail(f"stage manifests disagree about compiled artifact {name!r}")
    covered[name] = normalized


def _require_actual_identity(
    files: Mapping[str, pathlib.Path],
    name: str,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    path = files.get(name)
    if path is None:
        _fail(f"{label} requires missing compiled artifact {name!r}")
    actual = _file_identity(path)
    expected_local = {"path": name, "bytes": expected["bytes"], "sha256": expected["sha256"]}
    if actual != expected_local:
        _fail(f"compiled artifact differs from {label}: {name}")


def _validate_validation_report(
    report: Mapping[str, Any],
    *,
    precision: str,
    identity: DeclaredIdentity,
    files: Mapping[str, pathlib.Path],
) -> None:
    if report.get("schema_version") != 1 or report.get("precision") != precision or report.get("passes") is not True:
        _fail(f"{precision} ONNX validation report is absent, malformed, or failing")
    expected = {
        "track": identity.track,
        "dataset": identity.dataset,
        "dataset_revision": identity.dataset_revision,
        **identity.runtime,
    }
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        _fail(f"{precision} ONNX validation report has no provenance")
    for key, value in expected.items():
        if provenance.get(key) != value:
            _fail(f"{precision} ONNX validation provenance mismatch for {key}")
    models = report.get("models")
    if not isinstance(models, Mapping) or set(models) != {"encode-prefix", "decode-denoise"}:
        _fail(f"{precision} ONNX validation report does not bind both split graphs")
    for graph_name, model in models.items():
        if not isinstance(model, Mapping) or not isinstance(model.get("model"), Mapping):
            _fail(f"{precision} ONNX validation identity for {graph_name} is malformed")
        identities = (model["model"], *model.get("external_data", ()))
        for index, artifact in enumerate(identities):
            if not isinstance(artifact, Mapping) or set(artifact) != {"name", "bytes", "sha256"}:
                _fail(f"{precision} ONNX validation identity for {graph_name}[{index}] is malformed")
            name = artifact.get("name")
            if not isinstance(name, str) or pathlib.PurePosixPath(name).name != name:
                _fail(f"{precision} ONNX validation identity for {graph_name}[{index}] is not flat/portable")
            if index == 0 and name != f"{graph_name}.{precision}.onnx":
                _fail(f"{precision} ONNX validation identity names the wrong {graph_name} graph")
            _require_actual_identity(
                files,
                name,
                {"path": name, "bytes": artifact.get("bytes"), "sha256": artifact.get("sha256")},
                label=f"{precision} ONNX validation report",
            )
    actions = report.get("end_to_end_actions")
    if not isinstance(actions, Mapping):
        _fail(f"{precision} ONNX validation report has no end-to-end action checks")
    if actions.get("bias_passes") is not True or actions.get("action_limits_pass") is not True:
        _fail(f"{precision} ONNX validation report does not pass bias/action-limit gates")


def _validate_stage_manifest(
    path: pathlib.Path,
    *,
    expected_stage: str,
    identity: DeclaredIdentity,
    files: Mapping[str, pathlib.Path],
    original_root: pathlib.PurePosixPath,
    covered: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    manifest = _load_json(path)
    _require_stage_identity(manifest, expected_stage=expected_stage, identity=identity, path=path)
    records = _record_map(manifest, label=path.name)
    for record in records.values():
        _add_covered_record(covered, record, original_root=original_root, label=path.name)
    covered[path.name] = _file_identity(path)
    return manifest


def _validate_source_manifest_identity(
    source: Mapping[str, Any], *, files: Mapping[str, pathlib.Path], expected_name: str
) -> None:
    if set(source) != {"name", "bytes", "sha256"} or source.get("name") != expected_name:
        _fail(f"TensorRT policy contract has malformed source manifest {expected_name!r}")
    expected = {"path": expected_name, "bytes": source.get("bytes"), "sha256": source.get("sha256")}
    _require_actual_identity(files, expected_name, expected, label="TensorRT policy contract")


def _validate_build_manifest(
    path: pathlib.Path,
    *,
    precision: str,
    identity: DeclaredIdentity,
    files: Mapping[str, pathlib.Path],
    covered: dict[str, dict[str, Any]],
    primary: bool,
) -> tuple[dict[str, Any], pathlib.PurePosixPath]:
    manifest = _load_json(path)
    _require_stage_identity(
        manifest,
        expected_stage=f"tensorrt-build-{precision}",
        identity=identity,
        path=path,
    )
    validation_name = f"onnx-validation.{precision}.json"
    original_root = _original_artifact_root(manifest, expected_validation_name=validation_name)
    records = _record_map(manifest, label=path.name)
    for name, record in records.items():
        _add_covered_record(covered, record, original_root=original_root, label=path.name)
        _require_actual_identity(files, name, record, label=path.name)
    covered[path.name] = _file_identity(path)

    details = manifest.get("details")
    if not isinstance(details, Mapping):
        _fail(f"{path.name} has no build details")
    if details.get("strongly_typed") is not True:
        _fail(f"{path.name} was not built strongly typed")
    if details.get("precision_source") != "explicit ONNX tensor types and Q/DQ nodes":
        _fail(f"{path.name} has an unexpected precision contract")
    version = details.get("tensorrt_version")
    dotted = isinstance(version, str) and re.search(r"TensorRT(?:\s+version)?[^0-9]*11\.", version, re.IGNORECASE)
    compact = isinstance(version, str) and re.search(r"TensorRT\s+v11[0-9]{4}", version, re.IGNORECASE)
    if not dotted and not compact:
        _fail(f"{path.name} was not built with the pinned TensorRT 11 major")
    commands = details.get("commands")
    if (
        not isinstance(commands, list)
        or len(commands) != 2
        or not all(isinstance(item, list) and item for item in commands)
    ):
        _fail(f"{path.name} must bind exactly two non-empty trtexec commands")
    if details.get("gpu_inventory") != list(identity.gpu_inventory):
        _fail(f"{path.name} GPU/driver provenance differs from the live build GPU")

    contract = details.get("policy_contract")
    if not isinstance(contract, Mapping):
        _fail(f"{path.name} has no policy contract")
    expected_contract = {
        "schema_version": 1,
        "protocol": "openpi-policy-websocket-v1",
        "config": f"pi05_{identity.track}_l09_snapflow",
        "precision": precision,
        "num_denoise_steps": 1,
        "export_runtime": identity.runtime,
    }
    for key, value in expected_contract.items():
        if contract.get(key) != value:
            _fail(f"{path.name} policy contract mismatch for {key}")
    checkpoint = contract.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {"path", "sha256", "assets"}:
        _fail(f"{path.name} policy contract checkpoint identity is malformed")
    if SHA256_RE.fullmatch(str(checkpoint.get("sha256", ""))) is None or not checkpoint.get("assets"):
        _fail(f"{path.name} policy contract checkpoint is incomplete")
    source_manifests = contract.get("source_manifests")
    expected_sources = (
        ["export-manifest.json"]
        if precision == "bf16"
        else ["export-manifest.json", "fp8-manifest.json", "onnx-validation.bf16.json"]
    )
    if not isinstance(source_manifests, list) or len(source_manifests) != len(expected_sources):
        _fail(f"{path.name} policy contract has an incomplete source-manifest chain")
    for source, expected_name in zip(source_manifests, expected_sources, strict=True):
        if not isinstance(source, Mapping):
            _fail(f"{path.name} source-manifest identity is malformed")
        _validate_source_manifest_identity(source, files=files, expected_name=expected_name)

    required = {
        f"encode-prefix.{precision}.plan",
        f"decode-denoise.{precision}.plan",
        f"encode-prefix.{precision}.layers.json",
        f"decode-denoise.{precision}.layers.json",
        f"encode-prefix.{precision}.trtexec.log",
        f"decode-denoise.{precision}.trtexec.log",
        f"tensorrt-{precision}.timing.cache",
        validation_name,
        *expected_sources,
    }
    if not required.issubset(records):
        _fail(f"{path.name} omits required build artifacts: {sorted(required.difference(records))}")
    if primary and precision != identity.precision:
        _fail("selected TensorRT build precision differs from requested precision")
    return manifest, original_root


def validate_compiled_artifact(
    root: pathlib.Path,
    identity: DeclaredIdentity,
    *,
    source_runner: CommandRunner = repro_stage_data.run_command,
    environ: Mapping[str, str] | None = None,
) -> SealedArtifact:
    """Validate the complete directory and return its deterministic seal in memory."""

    _validate_declared_identity(identity)
    verify_source_checkout(identity.source_commit, runner=source_runner, environ=environ)
    files = _payload_files(root)
    build_name = f"tensorrt-manifest.{identity.precision}.json"
    build_path = files.get(build_name)
    if build_path is None:
        _fail(f"selected TensorRT build manifest is missing: {build_name}")

    covered: dict[str, dict[str, Any]] = {}
    build, original_root = _validate_build_manifest(
        build_path,
        precision=identity.precision,
        identity=identity,
        files=files,
        covered=covered,
        primary=True,
    )

    export = _validate_stage_manifest(
        files.get("export-manifest.json", root / "export-manifest.json"),
        expected_stage="onnx-export-bf16",
        identity=identity,
        files=files,
        original_root=original_root,
        covered=covered,
    )
    export_details = export.get("details", {})
    contract = build["details"]["policy_contract"]
    if export_details.get("config") != contract.get("config") or export_details.get("checkpoint") != contract.get(
        "checkpoint"
    ):
        _fail("export manifest policy/checkpoint identity differs from the TensorRT policy contract")
    export_records = _record_map(export, label="export-manifest.json")
    required_export = {
        "encode-prefix.bf16.onnx",
        "decode-denoise.bf16.onnx",
        "encode-inputs.npz",
        "decode-inputs.npz",
        "encode-reference.npz",
        "decode-reference.npz",
    }
    if not required_export.issubset(export_records):
        _fail(
            f"export manifest omits required portable artifacts: {sorted(required_export.difference(export_records))}"
        )

    validation_precisions = ["bf16"] if identity.precision == "bf16" else ["bf16", "fp8"]
    if identity.precision == "fp8":
        fp8 = _validate_stage_manifest(
            files.get("fp8-manifest.json", root / "fp8-manifest.json"),
            expected_stage="modelopt-fp8-ptq",
            identity=identity,
            files=files,
            original_root=original_root,
            covered=covered,
        )
        details = fp8.get("details", {})
        if details.get("quantize_mode") != "fp8" or details.get("calibration_chunks") != 1024:
            _fail("FP8 manifest does not bind the published 1,024-chunk FP8 recipe")
        if pathlib.PurePosixPath(str(details.get("bf16_validation_report", ""))).name != "onnx-validation.bf16.json":
            _fail("FP8 manifest does not bind the BF16 validation report")

    for precision in validation_precisions:
        report_name = f"onnx-validation.{precision}.json"
        report_path = files.get(report_name)
        if report_path is None:
            _fail(f"required validation report is missing: {report_name}")
        report = _load_json(report_path)
        _validate_validation_report(report, precision=precision, identity=identity, files=files)
        manifest_name = f"onnx-validation-manifest.{precision}.json"
        validation_manifest = _validate_stage_manifest(
            files.get(manifest_name, root / manifest_name),
            expected_stage=f"onnx-validation-{precision}",
            identity=identity,
            files=files,
            original_root=original_root,
            covered=covered,
        )
        if validation_manifest.get("details") != report or validation_manifest.get("metrics") != report:
            _fail(f"{manifest_name} does not exactly bind its validation report")

    auxiliary_bf16 = files.get("tensorrt-manifest.bf16.json")
    if identity.precision == "fp8" and auxiliary_bf16 is not None:
        _, auxiliary_root = _validate_build_manifest(
            auxiliary_bf16,
            precision="bf16",
            identity=identity,
            files=files,
            covered=covered,
            primary=False,
        )
        if auxiliary_root != original_root:
            _fail("BF16 and FP8 TensorRT manifests were not built from the same artifact directory")
    if identity.precision == "bf16" and "tensorrt-manifest.fp8.json" in files:
        _fail("BF16 publication directory contains an unselected FP8 build")

    for name, expected in covered.items():
        _require_actual_identity(files, name, expected, label="manifest closure")
    actual_names = set(files)
    covered_names = set(covered)
    if actual_names != covered_names:
        extras = sorted(actual_names - covered_names)
        missing = sorted(covered_names - actual_names)
        _fail(f"compiled artifact directory is not exactly manifest-covered; extras={extras}, missing={missing}")

    file_records = tuple(_file_identity(files[name], relative_to=root) for name in sorted(files))
    totals = {"files": len(file_records), "bytes": sum(record["bytes"] for record in file_records)}
    build_identity = _file_identity(build_path)
    revision_basis = {
        "schema_version": 1,
        "source_commit": identity.source_commit,
        "runtime": identity.runtime,
        "gpu_inventory": list(identity.gpu_inventory),
        "track": identity.track,
        "dataset": {"name": identity.dataset, "revision": identity.dataset_revision},
        "precision": identity.precision,
        "build_manifest": build_identity,
        "totals": totals,
        "files": file_records,
    }
    return SealedArtifact(
        root=root,
        revision=_canonical_hash(revision_basis),
        files=file_records,
        totals=totals,
        identity=identity,
        build_manifest=build,
        build_manifest_identity=build_identity,
    )


def parse_s3_target(s3_root: str, sealed: SealedArtifact) -> CompiledS3Target:
    parsed = urllib.parse.urlsplit(s3_root)
    if parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment:
        _fail(f"--s3-root must be an s3://bucket/prefix URI: {s3_root!r}")
    base = parsed.path.strip("/")
    prefix = "/".join(
        part for part in (base, sealed.identity.track, sealed.identity.precision, sealed.revision) if part
    )
    payload_prefix = f"{prefix}/artifact"
    manifest_key = f"{prefix}/manifest.sha256.json"
    return CompiledS3Target(
        bucket=parsed.netloc,
        prefix=prefix,
        payload_prefix=payload_prefix,
        payload_uri=f"s3://{parsed.netloc}/{payload_prefix}/",
        manifest_key=manifest_key,
        manifest_uri=f"s3://{parsed.netloc}/{manifest_key}",
    )


def _artifact_name(sealed: SealedArtifact) -> str:
    return f"{sealed.identity.track}_{sealed.identity.precision}_engines"


def _destination(sealed: SealedArtifact) -> str:
    return f"tensorrt/{sealed.identity.track}/{sealed.identity.precision}"


def build_publication_manifest(sealed: SealedArtifact, target: CompiledS3Target) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "source": {
            "provider": "pi05-compiled-tensorrt",
            "revision_kind": "compiled-artifact-content-and-provenance-sha256",
            "revision": sealed.revision,
            "source_commit": sealed.identity.source_commit,
        },
        "compiled": {
            "track": sealed.identity.track,
            "precision": sealed.identity.precision,
            "dataset": {"name": sealed.identity.dataset, "revision": sealed.identity.dataset_revision},
            "runtime": sealed.identity.runtime,
            "gpu_inventory": list(sealed.identity.gpu_inventory),
            "build_manifest": sealed.build_manifest_identity,
        },
        "artifact": {
            "name": _artifact_name(sealed),
            "kind": "artifact",
            "publish_destination": _destination(sealed),
            "payload_s3_uri": target.payload_uri,
        },
        "totals": sealed.totals,
        "files": list(sealed.files),
    }


def worker_artifact_descriptor(
    sealed: SealedArtifact,
    target: CompiledS3Target,
    *,
    manifest_version_id: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "name": _artifact_name(sealed),
        "kind": "asset",
        "revision": sealed.revision,
        "manifest": {
            "s3_uri": target.manifest_uri,
            "version_id": manifest_version_id,
            "sha256": manifest_sha256,
        },
        "payload_s3_uri": target.payload_uri,
        "destination": _destination(sealed),
    }


def _json_command(runner: CommandRunner, argv: Sequence[str]) -> Any:
    output = runner(argv)
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise repro_stage_data.StageError(f"command did not return JSON ({' '.join(argv)}): {output!r}") from exc


def _metadata(sealed: SealedArtifact, digest: str) -> str:
    return ",".join(
        (
            "source-provider=pi05-compiled-tensorrt",
            f"source-revision={sealed.revision}",
            f"source-commit={sealed.identity.source_commit}",
            f"image-digest={sealed.identity.image_digest}",
            f"file-sha256={digest}",
        )
    )


def _head_and_hash_version(
    *,
    runner: CommandRunner,
    account: str,
    region: str,
    target: CompiledS3Target,
    sealed: SealedArtifact,
    key: str,
    expected_path: pathlib.Path,
    expected_sha256: str,
    expected_bytes: int,
    expected_version_id: str,
) -> dict[str, Any]:
    common = [
        "--bucket",
        target.bucket,
        "--key",
        key,
        "--expected-bucket-owner",
        account,
        "--region",
        region,
    ]
    head = _json_command(
        runner, ["aws", "s3api", "head-object", *common, "--checksum-mode", "ENABLED", "--output", "json"]
    )
    if not isinstance(head, Mapping):
        _fail(f"S3 head-object returned a non-object for {key}")
    version_id = head.get("VersionId")
    if not isinstance(version_id, str) or not version_id:
        _fail(f"uploaded S3 object has no version ID: {key}")
    if version_id != expected_version_id:
        _fail(f"uploaded S3 object's current version differs from the conditional-write receipt: {key}")
    if head.get("ServerSideEncryption") != "AES256":
        _fail(f"uploaded S3 object is not encrypted with AES256: {key}")
    if head.get("ContentLength") != expected_bytes:
        _fail(f"uploaded S3 object has the wrong byte count: {key}")
    metadata = head.get("Metadata")
    if not isinstance(metadata, Mapping) or metadata.get("source-revision") != target.prefix.rsplit("/", 1)[-1]:
        _fail(f"uploaded S3 object has the wrong revision metadata: {key}")
    if metadata.get("source-commit") != sealed.identity.source_commit:
        _fail(f"uploaded S3 object has the wrong source-commit metadata: {key}")
    if metadata.get("image-digest") != sealed.identity.image_digest:
        _fail(f"uploaded S3 object has the wrong image-digest metadata: {key}")
    if metadata.get("file-sha256") != expected_sha256:
        _fail(f"uploaded S3 object has the wrong SHA-256 metadata: {key}")

    with tempfile.TemporaryDirectory(prefix="pi05-s3-verify-") as temporary:
        downloaded = pathlib.Path(temporary) / expected_path.name
        runner(["aws", "s3api", "get-object", *common, "--version-id", version_id, str(downloaded), "--output", "json"])
        if not downloaded.is_file():
            _fail(f"S3 get-object did not create verification payload for {key}")
        if downloaded.stat().st_size != expected_bytes or repro_stage_data.sha256_file(downloaded) != expected_sha256:
            _fail(f"version-pinned S3 payload hash verification failed: {key}")
    return {"key": key, "version_id": version_id, "bytes": expected_bytes, "sha256": expected_sha256}


def _put_file(
    *,
    runner: CommandRunner,
    account: str,
    region: str,
    target: CompiledS3Target,
    sealed: SealedArtifact,
    source: pathlib.Path,
    key: str,
    digest: str,
) -> dict[str, Any]:
    response = _json_command(
        runner,
        [
            "aws",
            "s3api",
            "put-object",
            "--bucket",
            target.bucket,
            "--key",
            key,
            "--body",
            str(source),
            "--expected-bucket-owner",
            account,
            "--region",
            region,
            "--server-side-encryption",
            "AES256",
            "--metadata",
            _metadata(sealed, digest),
            "--if-none-match",
            "*",
            "--output",
            "json",
        ],
    )
    if not isinstance(response, Mapping):
        _fail(f"S3 conditional put returned a non-object for {key}")
    version_id = response.get("VersionId")
    if not isinstance(version_id, str) or not version_id:
        _fail(f"S3 conditional put returned no version ID for {key}")
    if response.get("ServerSideEncryption") != "AES256":
        _fail(f"S3 conditional put did not acknowledge AES256 encryption for {key}")
    return _head_and_hash_version(
        runner=runner,
        account=account,
        region=region,
        target=target,
        sealed=sealed,
        key=key,
        expected_path=source,
        expected_sha256=digest,
        expected_bytes=source.stat().st_size,
        expected_version_id=version_id,
    )


def _require_empty_prefix_history(
    *,
    runner: CommandRunner,
    account: str,
    region: str,
    target: CompiledS3Target,
) -> None:
    prefix = f"{target.prefix}/"
    common = ["--bucket", target.bucket, "--prefix", prefix, "--expected-bucket-owner", account, "--region", region]
    objects = _json_command(runner, ["aws", "s3api", "list-objects-v2", *common, "--output", "json"])
    if not isinstance(objects, Mapping) or objects.get("IsTruncated"):
        _fail("S3 create-once object-history preflight is malformed or truncated")
    if objects.get("Contents"):
        _fail("content-addressed S3 publication prefix already contains objects")
    versions = _json_command(runner, ["aws", "s3api", "list-object-versions", *common, "--output", "json"])
    if not isinstance(versions, Mapping) or versions.get("IsTruncated"):
        _fail("S3 create-once version-history preflight is malformed or truncated")
    if versions.get("Versions") or versions.get("DeleteMarkers"):
        _fail("content-addressed S3 publication prefix has prior object-version or delete-marker history")


def _verify_list_receipts(
    *,
    runner: CommandRunner,
    account: str,
    region: str,
    target: CompiledS3Target,
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    common = [
        "--bucket",
        target.bucket,
        "--prefix",
        f"{target.prefix}/",
        "--expected-bucket-owner",
        account,
        "--region",
        region,
    ]
    objects = _json_command(runner, ["aws", "s3api", "list-objects-v2", *common, "--output", "json"])
    if objects.get("IsTruncated"):
        _fail("S3 list-objects-v2 receipt is truncated")
    observed = {item.get("Key"): item.get("Size") for item in objects.get("Contents", ()) if isinstance(item, Mapping)}
    expected = {item["key"]: item["bytes"] for item in receipts}
    if observed != expected:
        _fail(f"S3 object listing differs from sealed publication: observed={observed}, expected={expected}")

    versions = _json_command(runner, ["aws", "s3api", "list-object-versions", *common, "--output", "json"])
    if versions.get("IsTruncated"):
        _fail("S3 list-object-versions receipt is truncated")
    observed_versions: dict[str, str] = {}
    allowed_keys = set(expected)
    for item in versions.get("Versions", ()):
        if not isinstance(item, Mapping):
            _fail("S3 version listing contains a malformed version receipt")
        key = item.get("Key")
        if key not in allowed_keys:
            _fail(f"S3 version listing contains an unexpected key under the immutable prefix: {key!r}")
        if item.get("IsLatest") is not True or key in observed_versions:
            _fail(f"S3 immutable prefix contains prior or duplicate object-version history for {key}")
        observed_versions[str(key)] = str(item.get("VersionId", ""))
    expected_versions = {str(item["key"]): str(item["version_id"]) for item in receipts}
    if observed_versions != expected_versions:
        _fail(
            "S3 version receipts differ from the create-once uploaded objects: "
            f"observed={observed_versions}, expected={expected_versions}"
        )
    delete_markers = versions.get("DeleteMarkers", ())
    if delete_markers:
        _fail("S3 immutable publication prefix contains delete-marker history")
    return {"objects": observed, "versions": observed_versions}


def upload_compiled_artifact(
    config: Mapping[str, Any],
    sealed: SealedArtifact,
    s3_root: str,
    *,
    runner: CommandRunner = repro_stage_data.run_command,
    source_runner: CommandRunner = repro_stage_data.run_command,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    verify_source_checkout(sealed.identity.source_commit, runner=source_runner, environ=environ)
    target = parse_s3_target(s3_root, sealed)
    configured_bucket = config.get("aws", {}).get("artifact_bucket")
    if configured_bucket is not None and target.bucket != configured_bucket:
        _fail(f"compiled artifact bucket differs from reproduction config: {target.bucket!r} != {configured_bucket!r}")
    account, region = repro_stage_data.verify_aws_destination(config, target, runner=runner, environ=environ)
    _require_empty_prefix_history(
        runner=runner,
        account=account,
        region=region,
        target=target,
    )

    receipts: list[dict[str, Any]] = []
    for record in sealed.files:
        source = sealed.root / record["path"]
        if _file_identity(source, relative_to=sealed.root) != record:
            _fail(f"compiled artifact changed after sealing: {record['path']}")
        receipts.append(
            _put_file(
                runner=runner,
                account=account,
                region=region,
                target=target,
                sealed=sealed,
                source=source,
                key=f"{target.payload_prefix}/{record['path']}",
                digest=record["sha256"],
            )
        )

    publication = build_publication_manifest(sealed, target)
    with tempfile.TemporaryDirectory(prefix="pi05-publication-") as temporary:
        manifest_path = pathlib.Path(temporary) / "manifest.sha256.json"
        manifest_path.write_text(json.dumps(publication, indent=2, sort_keys=True) + "\n")
        manifest_sha256 = repro_stage_data.sha256_file(manifest_path)
        manifest_receipt = _put_file(
            runner=runner,
            account=account,
            region=region,
            target=target,
            sealed=sealed,
            source=manifest_path,
            key=target.manifest_key,
            digest=manifest_sha256,
        )
    receipts.append(manifest_receipt)
    for record in sealed.files:
        if _file_identity(sealed.root / record["path"], relative_to=sealed.root) != record:
            _fail(f"compiled artifact changed during upload: {record['path']}")
    list_receipts = _verify_list_receipts(
        runner=runner,
        account=account,
        region=region,
        target=target,
        receipts=receipts,
    )
    return {
        "source_revision": sealed.revision,
        "payload_s3_uri": target.payload_uri,
        "manifest_uri": target.manifest_uri,
        "object_receipts": receipts[:-1],
        "manifest_receipt": manifest_receipt,
        "list_receipts": list_receipts,
        "worker_artifact": worker_artifact_descriptor(
            sealed,
            target,
            manifest_version_id=manifest_receipt["version_id"],
            manifest_sha256=manifest_sha256,
        ),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="plan", choices=("plan", "validate", "upload"))
    parser.add_argument("--artifact-dir", required=True, type=pathlib.Path)
    parser.add_argument("--track", required=True, choices=TRACKS)
    parser.add_argument("--precision", required=True, choices=PRECISIONS)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--instance-type", default="g7e.4xlarge")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--s3-root")
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _declared_from_live(args: argparse.Namespace) -> DeclaredIdentity:
    from openpi.exporting.runtime_identity import query_gpu_inventory
    from openpi.exporting.runtime_identity import require_live_runtime_identity

    live = require_live_runtime_identity(
        image_digest=args.image_digest,
        instance_type=args.instance_type,
        instance_id=args.instance_id,
    )
    return DeclaredIdentity(
        source_commit=args.source_commit,
        image_digest=live.image_digest,
        track=args.track,
        dataset=args.dataset,
        dataset_revision=args.dataset_revision,
        precision=args.precision,
        instance_type=live.instance_type,
        instance_id=live.instance_id,
        gpu_inventory=query_gpu_inventory(),
    )


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        static_identity = DeclaredIdentity(
            source_commit=args.source_commit,
            image_digest=args.image_digest,
            track=args.track,
            dataset=args.dataset,
            dataset_revision=args.dataset_revision,
            precision=args.precision,
            instance_type=args.instance_type,
            instance_id=args.instance_id,
            gpu_inventory=(),
        )
        _validate_static_identity(static_identity)
        if not args.artifact_dir.is_absolute():
            _fail(f"--artifact-dir must be absolute: {args.artifact_dir}")
        if args.execute and args.action != "upload":
            _fail("--execute is valid only with the upload action")
        if args.action == "upload" and not args.s3_root:
            _fail("--s3-root is required for upload")
        if args.action == "plan":
            print(
                json.dumps(
                    {
                        "mode": "dry-run",
                        "requested_action": args.action,
                        "artifact_dir": str(args.artifact_dir),
                        "track": args.track,
                        "precision": args.precision,
                        "source_commit": args.source_commit,
                        "image_digest": args.image_digest,
                        "dataset": {"name": args.dataset, "revision": args.dataset_revision},
                        "runtime": {"instance_type": args.instance_type, "instance_id": args.instance_id},
                        "s3_root": args.s3_root,
                        "mutations_authorized": False,
                        "aws_calls_authorized": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        identity = _declared_from_live(args)
        sealed = validate_compiled_artifact(args.artifact_dir, identity)
        result: dict[str, Any] = {
            "mode": "validated" if args.action == "validate" else "dry-run",
            "source_revision": sealed.revision,
            "totals": sealed.totals,
            "build_manifest": sealed.build_manifest_identity,
            "mutations_authorized": False,
            "aws_calls_authorized": False,
        }
        if args.action == "upload":
            target = parse_s3_target(args.s3_root, sealed)
            result["target"] = dataclasses.asdict(target)
            if args.execute:
                config = repro_stage_data.load_json(args.config)
                result = upload_compiled_artifact(config, sealed, args.s3_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (repro_stage_data.StageError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"COMPILED ARTIFACT STAGING REJECTED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
