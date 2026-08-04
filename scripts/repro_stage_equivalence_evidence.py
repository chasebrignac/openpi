#!/usr/bin/env python3
"""Validate and immutably stage canonical JAX/PyTorch equivalence evidence.

``validate`` is entirely local and read-only.  ``upload`` performs the same
validation first, snapshots the validated bytes, and requires ``--execute``
before making any AWS call.  The four evidence files are written with S3
SHA-256 checksums to a content-addressed prefix; a manifest containing their
exact version IDs is written last.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import sys
import tempfile
from typing import Any

import numpy as np

try:
    from scripts import repro_stage_checkpoints
    from scripts import repro_stage_converted_checkpoints
    from scripts import repro_stage_data
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root, to sys.path.
    import repro_stage_checkpoints
    import repro_stage_converted_checkpoints
    import repro_stage_data


DEFAULT_CONFIG = pathlib.Path("repro/reproduction.json")
EQUIVALENCE_COSINE_MINIMUM = 0.999
SAMPLES = 64
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
BUCKET_RE = re.compile(r"^(?!-)(?!.*\.\.)(?!.*\.-)(?!.*-\.)(?!.*-$)[a-z0-9.-]{3,63}$")


@dataclasses.dataclass(frozen=True)
class EvidenceTrack:
    name: str
    checkpoint_key: str
    teacher_config: str
    golden_config: str
    golden_seed: int


TRACKS = {
    "libero": EvidenceTrack(
        name="libero",
        checkpoint_key="libero",
        teacher_config="pi05_libero",
        golden_config="pi05_libero_l09_distill",
        golden_seed=7001,
    ),
    "droid_jointpos": EvidenceTrack(
        name="droid_jointpos",
        checkpoint_key="droid_jointpos",
        teacher_config="pi05_droid_jointpos",
        golden_config="pi05_droid_l09_distill",
        golden_seed=7002,
    ),
}


@dataclasses.dataclass(frozen=True)
class EvidenceInputs:
    golden_npz: pathlib.Path
    golden_sidecar: pathlib.Path
    equivalence_report: pathlib.Path
    velocity_npz: pathlib.Path
    source_manifest: pathlib.Path
    converted_manifest: pathlib.Path


@dataclasses.dataclass(frozen=True)
class EvidenceFile:
    role: str
    canonical_name: str
    path: pathlib.Path
    sha256: str
    bytes: int


CommandRunner = Callable[[Sequence[str]], str]


def _json_object(path: pathlib.Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise repro_stage_data.StageError(f"cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise repro_stage_data.StageError(f"{label} JSON must be an object: {path}")
    return value


def _require_regular_file(path: pathlib.Path, *, label: str) -> pathlib.Path:
    path = path.expanduser()
    if path.is_symlink():
        raise repro_stage_data.StageError(f"{label} must not be a symlink: {path}")
    try:
        stat = path.stat()
    except OSError as exc:
        raise repro_stage_data.StageError(f"cannot stat {label} {path}: {exc}") from exc
    if not path.is_file() or stat.st_size <= 0:
        raise repro_stage_data.StageError(f"{label} must be a non-empty regular file: {path}")
    return path.resolve()


def _report_path_references(path_value: Any, expected: pathlib.Path) -> bool:
    """Accept lexical aliases only when they resolve to the validated file.

    Framework comparison runs inside a container, so its absolute provenance
    paths can traverse a bind mount such as ``/mnt/openpi``.  Host-side staging
    may see that same tree through a compatibility symlink such as
    ``/mnt/openpi -> /opt/pi05``.  Comparing the two path strings would reject
    byte-identical evidence even though both names resolve to the same file.
    """
    if not isinstance(path_value, str) or not path_value or not pathlib.Path(path_value).is_absolute():
        return False
    try:
        return pathlib.Path(path_value).expanduser().resolve(strict=True) == expected
    except (OSError, RuntimeError):
        return False


def _stable_sha256(path: pathlib.Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise repro_stage_data.StageError(f"input changed while hashing: {path}")
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any], *, pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _finite_number(value: Any, *, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise repro_stage_data.StageError(f"{label} must be a finite number")
    return float(value)


def _normalized_inputs(inputs: EvidenceInputs) -> EvidenceInputs:
    normalized = EvidenceInputs(
        golden_npz=_require_regular_file(inputs.golden_npz, label="golden NPZ"),
        golden_sidecar=_require_regular_file(inputs.golden_sidecar, label="golden sidecar"),
        equivalence_report=_require_regular_file(inputs.equivalence_report, label="equivalence report"),
        velocity_npz=_require_regular_file(inputs.velocity_npz, label="velocity NPZ"),
        source_manifest=_require_regular_file(inputs.source_manifest, label="source manifest"),
        converted_manifest=_require_regular_file(inputs.converted_manifest, label="converted manifest"),
    )
    paths = dataclasses.astuple(normalized)
    identities = [(path.stat().st_dev, path.stat().st_ino) for path in paths]
    if len(set(identities)) != len(identities):
        raise repro_stage_data.StageError("evidence inputs must be six distinct regular files")
    if normalized.golden_sidecar != normalized.golden_npz.with_suffix(".json"):
        raise repro_stage_data.StageError("golden sidecar must be the golden NPZ path with suffix .json")
    if normalized.velocity_npz != normalized.equivalence_report.with_suffix(".npz"):
        raise repro_stage_data.StageError("velocity NPZ must be the equivalence-report path with suffix .npz")
    return normalized


def _validate_split(split: Any) -> dict[str, Any]:
    if not isinstance(split, Mapping):
        raise repro_stage_data.StageError("golden data_split must be an object")
    episode_ids = split.get("validation_episode_ids")
    if (
        split.get("strategy") != "deterministic_whole_episode_stratified"
        or split.get("split") != "validation"
        or split.get("seed") != 42
        or not isinstance(episode_ids, list)
        or not episode_ids
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in episode_ids)
        or len(episode_ids) != len(set(episode_ids))
    ):
        raise repro_stage_data.StageError("golden validation split is not the canonical seed-42 whole-episode split")
    if "validation_episode_count" in split and split["validation_episode_count"] != len(episode_ids):
        raise repro_stage_data.StageError("golden validation episode count is inconsistent")
    if "selected_episode_count" in split and split["selected_episode_count"] != len(episode_ids):
        raise repro_stage_data.StageError("golden selected episode count is inconsistent")
    return dict(split)


def _validate_golden_archive(path: pathlib.Path, sidecar: Mapping[str, Any]) -> None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise repro_stage_data.StageError(f"golden NPZ is invalid: {exc}") from exc
    image_names = sidecar.get("image_names")
    if (
        not isinstance(image_names, list)
        or not image_names
        or any(not isinstance(name, str) or not name for name in image_names)
        or len(set(image_names)) != len(image_names)
    ):
        raise repro_stage_data.StageError("golden sidecar image_names must be a non-empty unique string list")
    required = {
        "state",
        "tokenized_prompt",
        "tokenized_prompt_mask",
        "actions",
        "noise",
        "time",
        *(f"image__{name}" for name in image_names),
        *(f"image_mask__{name}" for name in image_names),
    }
    if set(arrays) != required:
        raise repro_stage_data.StageError("golden NPZ array set differs from its canonical contract")
    if any(array.ndim == 0 or array.shape[0] != SAMPLES for array in arrays.values()):
        raise repro_stage_data.StageError("every golden NPZ array must contain exactly 64 samples")
    actions = arrays["actions"]
    if actions.ndim != 3 or arrays["noise"].shape != actions.shape:
        raise repro_stage_data.StageError("golden actions/noise must share a rank-three shape")
    if arrays["time"].shape != (SAMPLES,):
        raise repro_stage_data.StageError("golden time must have shape [64]")
    if sidecar.get("action_horizon") != actions.shape[1] or sidecar.get("action_dim") != actions.shape[2]:
        raise repro_stage_data.StageError("golden action shape differs from its sidecar")
    for name, array in arrays.items():
        if array.dtype.hasobject:
            raise repro_stage_data.StageError(f"golden array {name!r} has an object dtype")
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise repro_stage_data.StageError(f"golden array {name!r} contains NaN or Inf")


def _validate_golden(
    track: EvidenceTrack,
    inputs: EvidenceInputs,
    report_golden: Any,
) -> tuple[dict[str, Any], str, str]:
    sidecar = _json_object(inputs.golden_sidecar, label="golden sidecar")
    golden_sha = _stable_sha256(inputs.golden_npz)
    sidecar_sha = _stable_sha256(inputs.golden_sidecar)
    resolved = sidecar.get("resolved_config")
    dataset = sidecar.get("dataset")
    split = _validate_split(sidecar.get("data_split"))
    if (
        sidecar.get("schema_version") != 2
        or sidecar.get("sha256") != golden_sha
        or sidecar.get("config_name") != track.golden_config
        or sidecar.get("samples") != SAMPLES
        or sidecar.get("seed") != track.golden_seed
        or sidecar.get("data_split_seed") != 42
        or not isinstance(sidecar.get("run_id"), str)
        or not sidecar["run_id"].strip()
        or not isinstance(resolved, Mapping)
        or resolved.get("name") != track.golden_config
        or resolved.get("training_seed") != 42
        or not isinstance(resolved.get("fingerprint_sha256"), str)
        or SHA256_RE.fullmatch(resolved["fingerprint_sha256"]) is None
        or not isinstance(dataset, Mapping)
        or resolved.get("dataset") != dataset
        or sidecar.get("dataset_revision") != dataset.get("revision")
    ):
        raise repro_stage_data.StageError("golden corpus sidecar does not match the canonical track contract")
    _validate_golden_archive(inputs.golden_npz, sidecar)
    if not isinstance(report_golden, Mapping):
        raise repro_stage_data.StageError("equivalence report has no golden-corpus provenance")
    expected_report = {
        "sha256": golden_sha,
        "sidecar_sha256": sidecar_sha,
        "run_id": sidecar["run_id"],
        "config_name": track.golden_config,
        "config_fingerprint_sha256": resolved["fingerprint_sha256"],
        "dataset": dataset,
        "seed": track.golden_seed,
        "data_split_seed": 42,
        "data_split": split,
    }
    actual_report = dict(report_golden)
    report_path = actual_report.pop("path", None)
    report_sidecar_path = actual_report.pop("sidecar_path", None)
    if (
        not _report_path_references(report_path, inputs.golden_npz)
        or not _report_path_references(report_sidecar_path, inputs.golden_sidecar)
        or actual_report != expected_report
    ):
        raise repro_stage_data.StageError("equivalence report is not bound to the exact golden NPZ and sidecar")
    return sidecar, golden_sha, sidecar_sha


def _validate_manifests(
    track: EvidenceTrack,
    inputs: EvidenceInputs,
    report_provenance: Mapping[str, Any],
    *,
    source_commit: str,
    image_digest: str,
) -> dict[str, Any]:
    source = _json_object(inputs.source_manifest, label="source manifest")
    converted = _json_object(inputs.converted_manifest, label="converted manifest")
    source_sha = _stable_sha256(inputs.source_manifest)
    converted_sha = _stable_sha256(inputs.converted_manifest)
    source_info = source.get("source")
    source_checkpoint = source.get("checkpoint")
    converted_info = converted.get("source")
    converted_checkpoint = converted.get("checkpoint")
    conversion = converted.get("conversion")
    objects = source_info.get("objects") if isinstance(source_info, Mapping) else None
    if (
        source.get("schema_version") != 1
        or not isinstance(source_checkpoint, Mapping)
        or source_checkpoint.get("key") != track.checkpoint_key
        or not isinstance(source_info, Mapping)
        or not isinstance(objects, list)
        or not objects
        or source_info.get("revision") != repro_stage_checkpoints.inventory_revision(objects)
    ):
        raise repro_stage_data.StageError("source manifest is invalid or belongs to another track")
    source_revision = source_info["revision"]
    if SHA256_RE.fullmatch(str(source_revision)) is None:
        raise repro_stage_data.StageError("source manifest revision is not a SHA-256 identity")
    if (
        converted.get("schema_version") != 1
        or not isinstance(converted_checkpoint, Mapping)
        or converted_checkpoint.get("key") != track.checkpoint_key
        or not isinstance(converted_info, Mapping)
        or not isinstance(conversion, Mapping)
        or converted_info.get("upstream", {}).get("revision") != source_revision
        or conversion.get("config_name") != track.teacher_config
        or conversion.get("precision") != "bfloat16"
        or conversion.get("source_commit") != source_commit
        or conversion.get("image_digest") != image_digest
    ):
        raise repro_stage_data.StageError("converted manifest provenance does not match the requested track/build")
    converted_revision = converted_info.get("revision")
    try:
        expected_converted_revision = repro_stage_converted_checkpoints.conversion_revision(
            repro_stage_converted_checkpoints.manifest_identity(converted)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise repro_stage_data.StageError(f"converted manifest identity is malformed: {exc}") from exc
    if converted_revision != expected_converted_revision or SHA256_RE.fullmatch(str(converted_revision)) is None:
        raise repro_stage_data.StageError("converted manifest revision is inconsistent with its content")

    jax = report_provenance.get("jax_checkpoint")
    pytorch = report_provenance.get("pytorch_checkpoint")
    jax_manifest = jax.get("manifest") if isinstance(jax, Mapping) else None
    pytorch_manifest = pytorch.get("manifest") if isinstance(pytorch, Mapping) else None
    expected_jax = {"sha256": source_sha, "revision": source_revision}
    expected_pytorch = {
        "sha256": converted_sha,
        "revision": converted_revision,
        "source_commit": source_commit,
        "image_digest": image_digest,
    }
    actual_jax = dict(jax_manifest) if isinstance(jax_manifest, Mapping) else {}
    actual_jax_path = actual_jax.pop("path", None)
    actual_pytorch = dict(pytorch_manifest) if isinstance(pytorch_manifest, Mapping) else {}
    actual_pytorch_path = actual_pytorch.pop("path", None)
    if not _report_path_references(actual_jax_path, inputs.source_manifest) or actual_jax != expected_jax:
        raise repro_stage_data.StageError("equivalence report does not bind the exact source manifest")
    if (
        not _report_path_references(actual_pytorch_path, inputs.converted_manifest)
        or actual_pytorch != expected_pytorch
    ):
        raise repro_stage_data.StageError("equivalence report does not bind the exact converted manifest")
    pytorch_config = pytorch.get("config") if isinstance(pytorch, Mapping) else None
    if not isinstance(pytorch_config, Mapping) or pytorch_config.get("config_name") != track.teacher_config:
        raise repro_stage_data.StageError("equivalence report converted-checkpoint config is inconsistent")
    return {
        "source": {"sha256": source_sha, "revision": source_revision},
        "converted": {
            "sha256": converted_sha,
            "revision": converted_revision,
            "source_commit": source_commit,
            "image_digest": image_digest,
        },
    }


def _validate_velocities(path: pathlib.Path, report: Mapping[str, Any]) -> dict[str, float]:
    velocity_sha = _stable_sha256(path)
    velocities = report.get("velocities")
    actual_velocities = dict(velocities) if isinstance(velocities, Mapping) else {}
    velocity_report_path = actual_velocities.pop("path", None)
    if not _report_path_references(velocity_report_path, path) or actual_velocities != {"sha256": velocity_sha}:
        raise repro_stage_data.StageError("equivalence report is not bound to the exact velocity NPZ")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"jax", "pytorch"}:
                raise repro_stage_data.StageError("velocity NPZ must contain exactly jax and pytorch arrays")
            jax = archive["jax"]
            pytorch = archive["pytorch"]
    except (OSError, ValueError, KeyError) as exc:
        raise repro_stage_data.StageError(f"velocity NPZ is invalid: {exc}") from exc
    if jax.shape != pytorch.shape or jax.ndim != 3 or jax.shape[0] != SAMPLES:
        raise repro_stage_data.StageError("velocity arrays must share a [64, horizon, joints] shape")
    if not np.all(np.isfinite(jax)) or not np.all(np.isfinite(pytorch)):
        raise repro_stage_data.StageError("velocity arrays contain NaN or Inf")
    jax_flat = jax.reshape(SAMPLES, -1).astype(np.float64)
    pytorch_flat = pytorch.reshape(SAMPLES, -1).astype(np.float64)
    denominator = np.linalg.norm(jax_flat, axis=1) * np.linalg.norm(pytorch_flat, axis=1)
    cosine = np.sum(jax_flat * pytorch_flat, axis=1) / np.maximum(denominator, 1e-12)
    absolute = np.abs(pytorch.astype(np.float64) - jax.astype(np.float64))
    calculated = {
        "cosine_mean": float(np.mean(cosine)),
        "cosine_min": float(np.min(cosine)),
        "mse": float(np.mean(np.square(absolute))),
        "max_absolute_error": float(np.max(absolute)),
    }
    for key, expected in calculated.items():
        actual = _finite_number(report.get(key), label=f"equivalence report {key}")
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15):
            raise repro_stage_data.StageError(f"equivalence report {key} differs from the velocity NPZ")
    return calculated


def validate_evidence(
    track: EvidenceTrack,
    inputs: EvidenceInputs,
    *,
    source_commit: str,
    image_digest: str,
) -> dict[str, Any]:
    """Return a content-addressed manifest after validating all local bytes."""
    if COMMIT_RE.fullmatch(source_commit) is None:
        raise repro_stage_data.StageError("--source-commit must be a full lowercase git commit")
    if IMAGE_DIGEST_RE.fullmatch(image_digest) is None:
        raise repro_stage_data.StageError("--image-digest must be a sha256-pinned container digest")
    inputs = _normalized_inputs(inputs)
    report = _json_object(inputs.equivalence_report, label="equivalence report")
    cosine_min = _finite_number(report.get("cosine_min"), label="equivalence report cosine_min")
    if (
        report.get("schema_version") != 2
        or report.get("config_name") != track.teacher_config
        or report.get("samples") != SAMPLES
        or report.get("gate_cosine_minimum") != EQUIVALENCE_COSINE_MINIMUM
        or report.get("gate_pass") is not True
        or cosine_min < EQUIVALENCE_COSINE_MINIMUM
    ):
        raise repro_stage_data.StageError("equivalence report does not pass the exact 64-sample cosine gate")
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise repro_stage_data.StageError("equivalence report has no provenance")
    _, golden_sha, sidecar_sha = _validate_golden(track, inputs, provenance.get("golden_corpus"))
    checkpoint_identity = _validate_manifests(
        track,
        inputs,
        provenance,
        source_commit=source_commit,
        image_digest=image_digest,
    )
    metrics = _validate_velocities(inputs.velocity_npz, report)
    report_sha = _stable_sha256(inputs.equivalence_report)
    velocity_sha = _stable_sha256(inputs.velocity_npz)
    evidence_files = (
        EvidenceFile("golden_npz", "golden.npz", inputs.golden_npz, golden_sha, inputs.golden_npz.stat().st_size),
        EvidenceFile(
            "golden_sidecar",
            "golden.json",
            inputs.golden_sidecar,
            sidecar_sha,
            inputs.golden_sidecar.stat().st_size,
        ),
        EvidenceFile(
            "framework_equivalence",
            "framework-equivalence.json",
            inputs.equivalence_report,
            report_sha,
            inputs.equivalence_report.stat().st_size,
        ),
        EvidenceFile(
            "velocities",
            "velocities.npz",
            inputs.velocity_npz,
            velocity_sha,
            inputs.velocity_npz.stat().st_size,
        ),
    )
    content: dict[str, Any] = {
        "kind": "pi05-framework-equivalence-evidence",
        "track": track.name,
        "config_name": track.teacher_config,
        "golden_config_name": track.golden_config,
        "samples": SAMPLES,
        "gate": {
            "cosine_minimum": EQUIVALENCE_COSINE_MINIMUM,
            "cosine_min": metrics["cosine_min"],
            "pass": True,
        },
        "checkpoint_identity": checkpoint_identity,
        "files": [
            {
                "role": item.role,
                "name": item.canonical_name,
                "source_filename": item.path.name,
                "bytes": item.bytes,
                "sha256": item.sha256,
            }
            for item in evidence_files
        ],
    }
    return {
        "schema_version": 1,
        "evidence_revision": _sha256_json(content),
        "content": content,
        "_local_files": {item.role: str(item.path) for item in evidence_files},
    }


def _public_manifest(validated: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in validated.items() if key != "_local_files"}


def _normalize_destination(bucket: str, prefix: str, track: EvidenceTrack, revision: str) -> tuple[str, str]:
    if BUCKET_RE.fullmatch(bucket) is None:
        raise repro_stage_data.StageError(f"--bucket is not a valid S3 bucket name: {bucket!r}")
    if (
        not prefix
        or prefix != prefix.strip("/")
        or "//" in prefix
        or "\\" in prefix
        or any(part in {"", ".", ".."} for part in prefix.split("/"))
        or any(ord(char) < 32 or ord(char) == 127 for char in prefix)
    ):
        raise repro_stage_data.StageError("--prefix must be a non-empty normalized S3 key prefix")
    return bucket, f"{prefix}/{track.name}/{revision}"


def _json_command(runner: CommandRunner, argv: Sequence[str]) -> Any:
    output = runner(argv)
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise repro_stage_data.StageError(f"command did not return JSON ({' '.join(argv)}): {output!r}") from exc


def _ensure_empty_prefix(
    bucket: str,
    prefix: str,
    *,
    account: str,
    region: str,
    runner: CommandRunner,
) -> None:
    result = _json_command(
        runner,
        [
            "aws",
            "s3api",
            "list-object-versions",
            "--bucket",
            bucket,
            "--prefix",
            f"{prefix}/",
            "--expected-bucket-owner",
            account,
            "--region",
            region,
            "--output",
            "json",
        ],
    )
    if not isinstance(result, Mapping):
        raise repro_stage_data.StageError("S3 version listing returned a non-object")
    if result.get("Versions") or result.get("DeleteMarkers") or result.get("IsTruncated") is True:
        raise repro_stage_data.StageError("immutable evidence prefix already has object history; refusing overwrite")


def _put_verified_object(
    *,
    path: pathlib.Path,
    role: str,
    sha256: str,
    bucket: str,
    key: str,
    evidence_revision: str,
    source_commit: str,
    account: str,
    region: str,
    runner: CommandRunner,
) -> dict[str, Any]:
    if _stable_sha256(path) != sha256:
        raise repro_stage_data.StageError(f"staged {role} bytes changed before upload")
    checksum_base64 = __import__("base64").b64encode(bytes.fromhex(sha256)).decode()
    metadata = f"role={role},sha256={sha256},evidence-revision={evidence_revision},source-commit={source_commit}"
    response = _json_command(
        runner,
        [
            "aws",
            "s3api",
            "put-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--body",
            str(path),
            "--if-none-match",
            "*",
            "--expected-bucket-owner",
            account,
            "--region",
            region,
            "--server-side-encryption",
            "AES256",
            "--checksum-algorithm",
            "SHA256",
            "--checksum-sha256",
            checksum_base64,
            "--metadata",
            metadata,
            "--output",
            "json",
        ],
    )
    version_id = response.get("VersionId") if isinstance(response, Mapping) else None
    if not isinstance(version_id, str) or not version_id:
        raise repro_stage_data.StageError(f"uploaded {role} object has no S3 version ID")
    head = _json_command(
        runner,
        [
            "aws",
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--version-id",
            version_id,
            "--checksum-mode",
            "ENABLED",
            "--expected-bucket-owner",
            account,
            "--region",
            region,
            "--output",
            "json",
        ],
    )
    remote_metadata = head.get("Metadata") if isinstance(head, Mapping) else None
    if (
        not isinstance(head, Mapping)
        or head.get("VersionId") != version_id
        or head.get("ContentLength") != path.stat().st_size
        or head.get("ChecksumSHA256") != checksum_base64
        or head.get("ServerSideEncryption") != "AES256"
        or not isinstance(remote_metadata, Mapping)
        or remote_metadata.get("role") != role
        or remote_metadata.get("sha256") != sha256
        or remote_metadata.get("evidence-revision") != evidence_revision
        or remote_metadata.get("source-commit") != source_commit
    ):
        raise repro_stage_data.StageError(f"uploaded {role} object failed version/checksum/encryption verification")
    return {
        "role": role,
        "s3_uri": f"s3://{bucket}/{key}",
        "key": key,
        "version_id": version_id,
        "sha256": sha256,
        "bytes": path.stat().st_size,
        "storage": {
            "server_side_encryption": "AES256",
            "checksum_algorithm": "SHA256",
            "checksum_sha256_base64": checksum_base64,
        },
    }


def upload_evidence(
    config: Mapping[str, Any],
    track: EvidenceTrack,
    validated: Mapping[str, Any],
    *,
    bucket: str,
    prefix: str,
    runner: CommandRunner = repro_stage_data.run_command,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    revision = str(validated["evidence_revision"])
    bucket, immutable_prefix = _normalize_destination(bucket, prefix, track, revision)
    target = repro_stage_data.S3Target(
        bucket=bucket,
        prefix=immutable_prefix,
        snapshot_uri=f"s3://{bucket}/{immutable_prefix}/",
        manifest_uri=f"s3://{bucket}/{immutable_prefix}/manifest.sha256.json",
        manifest_key=f"{immutable_prefix}/manifest.sha256.json",
    )
    account, region = repro_stage_data.verify_aws_destination(
        config,
        target,
        runner=runner,
        environ=os.environ if environ is None else environ,
    )
    _ensure_empty_prefix(bucket, immutable_prefix, account=account, region=region, runner=runner)
    content_files = validated["content"]["files"]
    local_files = validated.get("_local_files")
    if not isinstance(content_files, list) or not isinstance(local_files, Mapping):
        raise repro_stage_data.StageError("validated evidence is missing its local upload bindings")
    source_commit = str(validated["content"]["checkpoint_identity"]["converted"]["source_commit"])
    uploaded: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="pi05-equivalence-evidence-") as temporary:
        staging = pathlib.Path(temporary)
        for item in content_files:
            if not isinstance(item, Mapping) or item.get("role") not in local_files:
                raise repro_stage_data.StageError("validated evidence file inventory is malformed")
            source = pathlib.Path(str(local_files[item["role"]]))
            if _stable_sha256(source) != item.get("sha256") or source.stat().st_size != item.get("bytes"):
                raise repro_stage_data.StageError(f"local {item.get('role')} evidence changed after validation")
            staged = staging / str(item["name"])
            shutil.copyfile(source, staged)
            if _stable_sha256(staged) != item["sha256"] or staged.stat().st_size != item["bytes"]:
                raise repro_stage_data.StageError(f"could not snapshot exact {item['role']} evidence bytes")
            uploaded.append(
                _put_verified_object(
                    path=staged,
                    role=str(item["role"]),
                    sha256=str(item["sha256"]),
                    bucket=bucket,
                    key=f"{immutable_prefix}/{item['name']}",
                    evidence_revision=revision,
                    source_commit=source_commit,
                    account=account,
                    region=region,
                    runner=runner,
                )
            )

        manifest = _public_manifest(validated) | {
            "storage": {
                "bucket": bucket,
                "region": region,
                "prefix": immutable_prefix,
                "objects": uploaded,
            }
        }
        manifest_path = staging / "manifest.sha256.json"
        manifest_path.write_bytes(_canonical_json_bytes(manifest, pretty=True))
        manifest_sha = _stable_sha256(manifest_path)
        manifest_upload = _put_verified_object(
            path=manifest_path,
            role="manifest",
            sha256=manifest_sha,
            bucket=bucket,
            key=target.manifest_key,
            evidence_revision=revision,
            source_commit=source_commit,
            account=account,
            region=region,
            runner=runner,
        )
    return {
        "evidence_revision": revision,
        "prefix_uri": target.snapshot_uri,
        "objects": uploaded,
        "manifest": manifest_upload,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "upload"))
    parser.add_argument("--track", required=True, choices=tuple(TRACKS))
    parser.add_argument("--golden-npz", required=True, type=pathlib.Path)
    parser.add_argument("--golden-sidecar", required=True, type=pathlib.Path)
    parser.add_argument("--equivalence-report", required=True, type=pathlib.Path)
    parser.add_argument("--velocity-npz", required=True, type=pathlib.Path)
    parser.add_argument("--source-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--converted-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--bucket")
    parser.add_argument("--prefix")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.action == "validate" and args.execute:
            raise repro_stage_data.StageError("--execute is only valid with upload")
        if args.action == "upload" and (not args.bucket or not args.prefix):
            raise repro_stage_data.StageError("upload requires explicit --bucket and --prefix")
        validated = validate_evidence(
            TRACKS[args.track],
            EvidenceInputs(
                golden_npz=args.golden_npz,
                golden_sidecar=args.golden_sidecar,
                equivalence_report=args.equivalence_report,
                velocity_npz=args.velocity_npz,
                source_manifest=args.source_manifest,
                converted_manifest=args.converted_manifest,
            ),
            source_commit=args.source_commit,
            image_digest=args.image_digest,
        )
        result: dict[str, Any] = {"validation": _public_manifest(validated)}
        if args.action == "upload" and not args.execute:
            result |= {
                "mode": "dry-run",
                "destination": {
                    "bucket": args.bucket,
                    "prefix": args.prefix,
                    "requires_execute": True,
                },
                "mutations_authorized": False,
            }
        elif args.action == "upload":
            result["s3"] = upload_evidence(
                repro_stage_data.load_json(args.config),
                TRACKS[args.track],
                validated,
                bucket=args.bucket,
                prefix=args.prefix,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (repro_stage_data.StageError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"EQUIVALENCE EVIDENCE STAGING REJECTED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
