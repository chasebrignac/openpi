#!/usr/bin/env python3
"""Stage the two pinned reproduction datasets without accidental large transfers.

The default action is a JSON plan.  Network writes require both an explicit
action and ``--execute``.  Dataset identities are read from
``repro/reproduction.json``; this command intentionally has no repo/revision
override flags.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
import concurrent.futures
import dataclasses
import datetime as dt
import hashlib
import importlib
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Any
import urllib.parse

DEFAULT_CONFIG = pathlib.Path("repro/reproduction.json")
SHA256_CHUNK_BYTES = 8 * 1024 * 1024
PINNED_REVISION = re.compile(r"^[0-9a-f]{40}$")
DROID_REQUIRED_VIDEO_FEATURES = (
    "observation.images.exterior_1_left",
    "observation.images.wrist_left",
)
DROID_REQUIRED_PARQUET_FEATURES = (
    "observation.state.joint_position",
    "observation.state.gripper_position",
    "action",
    "episode_index",
    "task_index",
)
DROID_EXPECTED_VIDEO_FILE_COUNTS = (
    ("observation.images.exterior_1_left", 518),
    ("observation.images.wrist_left", 316),
)


class StageError(RuntimeError):
    """Raised when staging would violate an integrity or AWS boundary."""


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    key: str
    repo_id: str
    revision: str
    codebase_version: str
    local_dirname: str
    required_features: tuple[str, ...]
    required_metadata: tuple[str, ...]
    expected_bytes: int | None = None
    expected_video_file_counts: tuple[tuple[str, int], ...] = ()


@dataclasses.dataclass(frozen=True)
class S3Target:
    bucket: str
    prefix: str
    snapshot_uri: str
    manifest_uri: str
    manifest_key: str


CommandRunner = Callable[[Sequence[str]], str]


def load_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise StageError(f"configuration does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise StageError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StageError(f"configuration must be a JSON object: {path}")
    return value


def dataset_spec(config: Mapping[str, Any], key: str) -> DatasetSpec:
    try:
        source = config["source"]
        if key == "libero":
            spec = DatasetSpec(
                key=key,
                repo_id=str(source["libero_repo"]),
                revision=str(source["libero_revision"]),
                codebase_version="v2.0",
                local_dirname="libero",
                required_features=("image", "wrist_image", "state", "actions"),
                required_metadata=("meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl"),
                expected_bytes=int(source["libero_expected_bytes"]),
            )
        elif key == "droid":
            spec = DatasetSpec(
                key=key,
                repo_id=str(source["molmoact2_droid_repo"]),
                revision=str(source["molmoact2_droid_revision"]),
                codebase_version="v3.0",
                local_dirname="molmoact2-droid",
                required_features=(*DROID_REQUIRED_VIDEO_FEATURES, *DROID_REQUIRED_PARQUET_FEATURES),
                required_metadata=(
                    "meta/info.json",
                    "meta/tasks.parquet",
                    "meta/tasks_annotated.parquet",
                    "meta/episodes",
                ),
                expected_bytes=int(source["molmoact2_droid_expected_bytes"]),
                expected_video_file_counts=DROID_EXPECTED_VIDEO_FILE_COUNTS,
            )
        else:
            raise StageError(f"unknown dataset: {key}")
    except (KeyError, TypeError, ValueError) as exc:
        raise StageError(f"missing or invalid source pin for {key}: {exc}") from exc

    if not spec.repo_id or spec.repo_id.count("/") != 1:
        raise StageError(f"invalid Hugging Face dataset repository in config: {spec.repo_id!r}")
    if PINNED_REVISION.fullmatch(spec.revision) is None:
        raise StageError(f"dataset revision must be a full 40-character commit: {spec.revision!r}")
    return spec


def resolve_paths(
    local_root: pathlib.Path, spec: DatasetSpec, manifest_output: pathlib.Path | None
) -> tuple[pathlib.Path, pathlib.Path]:
    if not local_root.is_absolute():
        raise StageError(f"--local-root must be an absolute path: {local_root}")
    snapshot_root = local_root / spec.local_dirname
    default_manifest = local_root / "_manifests" / f"{spec.local_dirname}-{spec.revision}.sha256.json"
    manifest_path = manifest_output or default_manifest
    if not manifest_path.is_absolute():
        raise StageError(f"--manifest-output must be an absolute path: {manifest_path}")
    if snapshot_root == manifest_path or snapshot_root in manifest_path.parents:
        raise StageError("the integrity manifest must be outside the dataset snapshot")
    return snapshot_root, manifest_path


def parse_s3_target(s3_root: str, spec: DatasetSpec) -> S3Target:
    parsed = urllib.parse.urlsplit(s3_root)
    if parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment:
        raise StageError(f"--s3-root must be an s3://bucket/prefix URI: {s3_root!r}")
    root_prefix = parsed.path.strip("/")
    immutable_prefix = "/".join(part for part in (root_prefix, spec.local_dirname, spec.revision) if part)
    snapshot_prefix = f"{immutable_prefix}/snapshot"
    manifest_key = f"{immutable_prefix}/manifest.sha256.json"
    return S3Target(
        bucket=parsed.netloc,
        prefix=immutable_prefix,
        snapshot_uri=f"s3://{parsed.netloc}/{snapshot_prefix}/",
        manifest_uri=f"s3://{parsed.netloc}/{manifest_key}",
        manifest_key=manifest_key,
    )


def render_plan(
    *,
    action: str,
    spec: DatasetSpec,
    snapshot_root: pathlib.Path,
    manifest_path: pathlib.Path,
    s3_target: S3Target | None,
) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "mode": "dry-run",
        "requested_action": action,
        "dataset": {
            "key": spec.key,
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "repo_type": "dataset",
            "expected_codebase_version": spec.codebase_version,
        },
        "local": {"snapshot_root": str(snapshot_root), "manifest": str(manifest_path)},
        "download_api": {
            "call": "huggingface_hub.snapshot_download",
            "kwargs": {
                "repo_id": spec.repo_id,
                "repo_type": "dataset",
                "revision": spec.revision,
                "local_dir": str(snapshot_root),
            },
        },
        "mutations_authorized": False,
    }
    if s3_target is not None:
        plan["s3"] = {
            "snapshot_uri": s3_target.snapshot_uri,
            "manifest_uri": s3_target.manifest_uri,
            "requires_account_and_region_preflight": True,
        }
    return plan


def ensure_download_capacity(local_root: pathlib.Path, spec: DatasetSpec) -> None:
    if spec.expected_bytes is None:
        return
    existing = local_root
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    free_bytes = shutil.disk_usage(existing).free
    required_bytes = int(spec.expected_bytes * 1.05)
    if free_bytes < required_bytes:
        raise StageError(
            f"insufficient free space for {spec.key}: {free_bytes} available, at least {required_bytes} required"
        )


def download_snapshot(
    spec: DatasetSpec,
    snapshot_root: pathlib.Path,
    *,
    snapshot_download_fn: Callable[..., str] | None = None,
) -> pathlib.Path:
    ensure_download_capacity(snapshot_root.parent, spec)
    snapshot_root.mkdir(parents=True, exist_ok=True)
    if snapshot_download_fn is None:
        try:
            snapshot_download_fn = importlib.import_module("huggingface_hub").snapshot_download
        except ImportError as exc:
            raise StageError("huggingface_hub is required for an executed download") from exc
    downloaded = pathlib.Path(
        snapshot_download_fn(
            repo_id=spec.repo_id,
            repo_type="dataset",
            revision=spec.revision,
            local_dir=str(snapshot_root),
        )
    )
    if downloaded.resolve() != snapshot_root.resolve():
        raise StageError(f"snapshot_download returned an unexpected directory: {downloaded}")
    return downloaded


def _read_json_object(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise StageError(f"required metadata is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise StageError(f"invalid JSON metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StageError(f"metadata must be a JSON object: {path}")
    return value


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError as exc:
        raise StageError(f"required metadata is missing: {path}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StageError(f"invalid JSONL metadata {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise StageError(f"JSONL record must be an object: {path}:{line_number}")
        records.append(value)
    return records


def _validate_libero_metadata(snapshot_root: pathlib.Path, info: Mapping[str, Any]) -> dict[str, Any]:
    tasks = _read_jsonl(snapshot_root / "meta/tasks.jsonl")
    episodes = _read_jsonl(snapshot_root / "meta/episodes.jsonl")
    total_tasks = int(info["total_tasks"])
    total_episodes = int(info["total_episodes"])
    if len(tasks) != total_tasks:
        raise StageError(f"LIBERO task count mismatch: info={total_tasks}, tasks.jsonl={len(tasks)}")
    if len(episodes) != total_episodes:
        raise StageError(f"LIBERO episode count mismatch: info={total_episodes}, episodes.jsonl={len(episodes)}")
    if any(not str(record.get("task", "")).strip() for record in tasks):
        raise StageError("LIBERO tasks.jsonl contains an empty task")
    episode_indices = {record.get("episode_index") for record in episodes}
    if episode_indices != set(range(total_episodes)):
        raise StageError("LIBERO episodes.jsonl does not contain each expected episode_index exactly once")
    data_files = sorted((snapshot_root / "data").glob("**/*.parquet"))
    if len(data_files) != total_episodes:
        raise StageError(f"LIBERO data file count mismatch: expected {total_episodes}, found {len(data_files)}")
    return {"task_records": len(tasks), "episode_records": len(episodes), "data_files": len(data_files)}


def _physical_files(snapshot_root: pathlib.Path, subtree: pathlib.Path) -> set[str]:
    root = snapshot_root / subtree
    if not root.is_dir():
        return set()
    return {
        path.relative_to(snapshot_root).as_posix() for path in root.rglob("*") if path.is_file() or path.is_symlink()
    }


def _require_exact_file_coverage(label: str, expected: set[str], actual: set[str]) -> None:
    if expected == actual:
        return
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    raise StageError(
        f"MolmoAct2 {label} coverage mismatch: expected {len(expected)} files, found {len(actual)}; "
        f"missing={missing[:5]}, unexpected={unexpected[:5]}"
    )


def _validate_droid_metadata(
    snapshot_root: pathlib.Path,
    info: Mapping[str, Any],
    expected_video_file_counts: tuple[tuple[str, int], ...],
) -> dict[str, Any]:
    try:
        pl = importlib.import_module("polars")
    except ImportError as exc:
        raise StageError("polars is required to validate MolmoAct2 parquet metadata") from exc

    annotation_path = snapshot_root / "meta/tasks_annotated.parquet"
    try:
        annotation_schema = set(pl.read_parquet_schema(annotation_path))
    except FileNotFoundError as exc:
        raise StageError(f"required metadata is missing: {annotation_path}") from exc
    required_annotation_columns = {"episode_index", "task"}
    missing_annotation_columns = required_annotation_columns - annotation_schema
    if missing_annotation_columns:
        raise StageError(f"MolmoAct2 annotations are missing columns: {sorted(missing_annotation_columns)}")
    annotations = pl.read_parquet(annotation_path, columns=sorted(required_annotation_columns))
    total_episodes = int(info["total_episodes"])
    if annotations.height != total_episodes:
        raise StageError(
            f"MolmoAct2 annotation row count mismatch: expected {total_episodes}, found {annotations.height}"
        )
    if annotations["episode_index"].null_count() or annotations["episode_index"].n_unique() != total_episodes:
        raise StageError("MolmoAct2 annotations must contain one non-null row per episode_index")
    indices = annotations["episode_index"]
    if indices.min() != 0 or indices.max() != total_episodes - 1:
        raise StageError("MolmoAct2 annotated episode_index range is incomplete")
    if annotations["task"].null_count() or any(not str(task).strip() for task in annotations["task"]):
        raise StageError("MolmoAct2 annotations contain an empty task")

    standard_tasks_path = snapshot_root / "meta/tasks.parquet"
    standard_task_schema = set(pl.read_parquet_schema(standard_tasks_path))
    task_text_column = "task" if "task" in standard_task_schema else "__index_level_0__"
    if "task_index" not in standard_task_schema or task_text_column not in standard_task_schema:
        raise StageError("MolmoAct2 standard task metadata must contain task_index and a task text column")
    standard_tasks = pl.read_parquet(standard_tasks_path, columns=["task_index", task_text_column])
    total_tasks = int(info["total_tasks"])
    task_indices = standard_tasks["task_index"]
    if (
        standard_tasks.height != total_tasks
        or task_indices.null_count()
        or task_indices.n_unique() != total_tasks
        or task_indices.min() != 0
        or task_indices.max() != total_tasks - 1
    ):
        raise StageError(
            "MolmoAct2 standard task metadata must contain each task_index exactly once: "
            f"expected {total_tasks} rows, found {standard_tasks.height}"
        )
    if standard_tasks[task_text_column].null_count() or any(
        not str(task).strip() for task in standard_tasks[task_text_column]
    ):
        raise StageError("MolmoAct2 standard task metadata contains an empty task")

    episode_files = sorted((snapshot_root / "meta/episodes").glob("**/*.parquet"))
    if not episode_files:
        raise StageError("MolmoAct2 snapshot contains no episode metadata parquet files")

    integer_reference_columns = ["episode_index", "data/chunk_index", "data/file_index"]
    timestamp_columns: list[str] = []
    for feature_name in DROID_REQUIRED_VIDEO_FEATURES:
        prefix = f"videos/{feature_name}"
        integer_reference_columns.extend([f"{prefix}/chunk_index", f"{prefix}/file_index"])
        timestamp_columns.extend([f"{prefix}/from_timestamp", f"{prefix}/to_timestamp"])
    required_episode_columns = [*integer_reference_columns, *timestamp_columns]
    episode_frames = []
    for path in episode_files:
        schema = pl.read_parquet_schema(path)
        missing_columns = set(required_episode_columns) - set(schema)
        if missing_columns:
            raise StageError(
                f"MolmoAct2 episode metadata {path.relative_to(snapshot_root)} is missing columns: "
                f"{sorted(missing_columns)}"
            )
        wrong_integer_types = [name for name in integer_reference_columns if not schema[name].is_integer()]
        wrong_timestamp_types = [name for name in timestamp_columns if not schema[name].is_float()]
        if wrong_integer_types or wrong_timestamp_types:
            raise StageError(
                f"MolmoAct2 episode metadata {path.relative_to(snapshot_root)} has invalid reference types: "
                f"integer={wrong_integer_types}, timestamp={wrong_timestamp_types}"
            )
        episode_frames.append(pl.read_parquet(path, columns=required_episode_columns))
    episode_metadata = pl.concat(episode_frames, how="vertical")
    episode_indices = episode_metadata["episode_index"]
    if (
        len(episode_indices) != total_episodes
        or episode_indices.null_count()
        or episode_indices.n_unique() != total_episodes
        or episode_indices.min() != 0
        or episode_indices.max() != total_episodes - 1
    ):
        raise StageError("MolmoAct2 episode metadata does not contain every episode_index exactly once")

    def referenced_pairs(chunk_column: str, file_column: str, label: str) -> set[tuple[int, int]]:
        references = episode_metadata.select([chunk_column, file_column])
        if references[chunk_column].null_count() or references[file_column].null_count():
            raise StageError(f"MolmoAct2 {label} references contain null chunk/file indices")
        pairs = {(int(chunk), int(file)) for chunk, file in references.iter_rows()}
        if any(chunk < 0 or file < 0 for chunk, file in pairs):
            raise StageError(f"MolmoAct2 {label} references contain negative chunk/file indices")
        return pairs

    data_pairs = referenced_pairs("data/chunk_index", "data/file_index", "data")
    expected_data_files = {f"data/chunk-{chunk:03d}/file-{file:03d}.parquet" for chunk, file in data_pairs}
    actual_data_files = _physical_files(snapshot_root, pathlib.Path("data"))
    _require_exact_file_coverage("data", expected_data_files, actual_data_files)

    data_files = sorted((snapshot_root / "data").glob("**/*.parquet"))
    for path in data_files:
        missing_physical = set(DROID_REQUIRED_PARQUET_FEATURES) - set(pl.read_parquet_schema(path))
        if missing_physical:
            raise StageError(
                f"MolmoAct2 data parquet {path.relative_to(snapshot_root)} is missing fields: "
                f"{sorted(missing_physical)}"
            )

    # LeRobot v3 stores video features as MP4 files, not columns in the data
    # parquet. Validate the released path contract and each required camera's
    # physical subtree instead of incorrectly requiring video feature names in
    # the parquet schema.
    expected_video_path = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    if info.get("video_path") != expected_video_path:
        raise StageError(
            f"unexpected MolmoAct2 video_path: expected {expected_video_path!r}, found {info.get('video_path')!r}"
        )
    expected_counts = dict(expected_video_file_counts)
    video_files_by_feature: dict[str, int] = {}
    video_timestamp_bounds_by_feature: dict[str, dict[str, float]] = {}
    for feature_name in DROID_REQUIRED_VIDEO_FEATURES:
        prefix = f"videos/{feature_name}"
        chunk_column = f"{prefix}/chunk_index"
        file_column = f"{prefix}/file_index"
        from_column = f"{prefix}/from_timestamp"
        to_column = f"{prefix}/to_timestamp"
        pairs = referenced_pairs(chunk_column, file_column, f"video feature {feature_name}")
        expected_count = expected_counts.get(feature_name)
        if expected_count is not None and len(pairs) != expected_count:
            raise StageError(
                f"MolmoAct2 video feature {feature_name} referenced file count mismatch: "
                f"expected {expected_count}, found {len(pairs)}"
            )
        timestamp_rows = episode_metadata.select([from_column, to_column])
        timestamps: list[tuple[float, float]] = []
        for from_timestamp, to_timestamp in timestamp_rows.iter_rows():
            if (
                from_timestamp is None
                or to_timestamp is None
                or not math.isfinite(float(from_timestamp))
                or not math.isfinite(float(to_timestamp))
                or not 0 <= float(from_timestamp) < float(to_timestamp)
            ):
                raise StageError(
                    f"MolmoAct2 video feature {feature_name} must have finite timestamps with "
                    "0 <= from_timestamp < to_timestamp for every episode"
                )
            timestamps.append((float(from_timestamp), float(to_timestamp)))
        expected_feature_files = {f"{prefix}/chunk-{chunk:03d}/file-{file:03d}.mp4" for chunk, file in pairs}
        actual_feature_files = _physical_files(snapshot_root, pathlib.Path(prefix))
        _require_exact_file_coverage(f"video feature {feature_name}", expected_feature_files, actual_feature_files)
        empty_files = [path for path in expected_feature_files if (snapshot_root / path).stat().st_size == 0]
        if empty_files:
            raise StageError(
                f"MolmoAct2 video feature {feature_name} contains empty MP4 files: {sorted(empty_files)[:5]}"
            )
        video_files_by_feature[feature_name] = len(expected_feature_files)
        video_timestamp_bounds_by_feature[feature_name] = {
            "min_from_timestamp": min(start for start, _ in timestamps),
            "min_duration_seconds": min(end - start for start, end in timestamps),
            "max_to_timestamp": max(end for _, end in timestamps),
        }
    return {
        "layout_contract": "molmoact2-v3-exact-media-references-v1",
        "annotation_records": annotations.height,
        "standard_task_records": standard_tasks.height,
        "episode_metadata_records": len(episode_indices),
        "data_files": len(data_files),
        "required_video_files": sum(video_files_by_feature.values()),
        "required_video_files_by_feature": video_files_by_feature,
        "expected_video_files_by_feature": expected_counts,
        "video_timestamp_bounds_by_feature": video_timestamp_bounds_by_feature,
    }


def payload_files(snapshot_root: pathlib.Path) -> list[pathlib.Path]:
    if not snapshot_root.is_dir():
        raise StageError(f"dataset snapshot does not exist: {snapshot_root}")
    result: list[pathlib.Path] = []
    for path in snapshot_root.rglob("*"):
        relative = path.relative_to(snapshot_root)
        if relative.parts and relative.parts[0] == ".cache":
            continue
        if path.is_symlink():
            raise StageError(f"dataset snapshot contains a symlink; restage with local_dir: {path}")
        if path.is_file():
            result.append(path)
    return sorted(result, key=lambda path: path.relative_to(snapshot_root).as_posix())


def validate_snapshot(spec: DatasetSpec, snapshot_root: pathlib.Path) -> dict[str, Any]:
    for relative in spec.required_metadata:
        if not (snapshot_root / relative).exists():
            raise StageError(f"required metadata is missing: {snapshot_root / relative}")
    info = _read_json_object(snapshot_root / "meta/info.json")
    if info.get("codebase_version") != spec.codebase_version:
        raise StageError(
            f"LeRobot codebase version mismatch for {spec.repo_id}: "
            f"expected {spec.codebase_version}, found {info.get('codebase_version')!r}"
        )
    features = info.get("features")
    if not isinstance(features, dict):
        raise StageError("meta/info.json features must be an object")
    missing_features = set(spec.required_features) - set(features)
    if missing_features:
        raise StageError(f"{spec.key} metadata is missing features: {sorted(missing_features)}")
    expected_data_path = {
        "libero": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "droid": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    }[spec.key]
    if info.get("data_path") != expected_data_path:
        raise StageError(
            f"unexpected {spec.key} data_path: expected {expected_data_path!r}, found {info.get('data_path')!r}"
        )
    if spec.key == "droid":
        expected_signatures = {
            "observation.images.exterior_1_left": ("video", [180, 320, 3]),
            "observation.images.wrist_left": ("video", [180, 320, 3]),
            "observation.state.joint_position": ("float32", [7]),
            "observation.state.gripper_position": ("float32", [1]),
            "action": ("float32", [8]),
        }
        for feature_name, (expected_dtype, expected_shape) in expected_signatures.items():
            feature = features[feature_name]
            if not isinstance(feature, dict) or (
                feature.get("dtype") != expected_dtype or feature.get("shape") != expected_shape
            ):
                raise StageError(
                    f"unexpected MolmoAct2 feature signature for {feature_name}: "
                    f"expected dtype={expected_dtype}, shape={expected_shape}, found {feature!r}"
                )
    for count_key in ("total_episodes", "total_frames", "total_tasks"):
        if not isinstance(info.get(count_key), int) or info[count_key] <= 0:
            raise StageError(f"meta/info.json {count_key} must be a positive integer")

    format_report = (
        _validate_libero_metadata(snapshot_root, info)
        if spec.key == "libero"
        else _validate_droid_metadata(snapshot_root, info, spec.expected_video_file_counts)
    )
    files = payload_files(snapshot_root)
    total_bytes = sum(path.stat().st_size for path in files)
    if spec.expected_bytes is not None and total_bytes < int(spec.expected_bytes * 0.85):
        raise StageError(
            f"snapshot is unexpectedly small: {total_bytes} bytes; expected approximately {spec.expected_bytes} bytes"
        )
    return {
        "codebase_version": info["codebase_version"],
        "total_episodes": info["total_episodes"],
        "total_frames": info["total_frames"],
        "total_tasks": info["total_tasks"],
        "payload_files": len(files),
        "payload_bytes": total_bytes,
        **format_report,
    }


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(SHA256_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    spec: DatasetSpec,
    snapshot_root: pathlib.Path,
    validation: Mapping[str, Any],
    *,
    hash_workers: int,
) -> dict[str, Any]:
    if hash_workers < 1:
        raise StageError("--hash-workers must be at least 1")
    paths = payload_files(snapshot_root)
    with concurrent.futures.ThreadPoolExecutor(max_workers=hash_workers) as executor:
        digests = list(executor.map(sha256_file, paths))
    files = [
        {
            "path": path.relative_to(snapshot_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
        for path, digest in zip(paths, digests, strict=True)
    ]
    return {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "source": {
            "provider": "huggingface",
            "repo_type": "dataset",
            "repo_id": spec.repo_id,
            "revision": spec.revision,
        },
        "dataset": {
            "key": spec.key,
            "codebase_version": spec.codebase_version,
            "local_dirname": spec.local_dirname,
        },
        "validation": dict(validation),
        "totals": {"files": len(files), "bytes": sum(item["bytes"] for item in files)},
        "files": files,
    }


def write_manifest(path: pathlib.Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_command(argv: Sequence[str]) -> str:
    completed = subprocess.run(list(argv), check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        raise StageError(f"command failed ({' '.join(argv)}): {message}")
    return completed.stdout.strip()


def _json_command(runner: CommandRunner, argv: Sequence[str]) -> Any:
    output = runner(argv)
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise StageError(f"command did not return JSON ({' '.join(argv)}): {output!r}") from exc


def verify_aws_destination(
    config: Mapping[str, Any],
    target: S3Target,
    *,
    runner: CommandRunner = run_command,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    try:
        account_id = str(config["aws"]["account_id"])
        expected_region = str(config["aws"]["region"])
    except (KeyError, TypeError) as exc:
        raise StageError(f"invalid AWS boundary in reproduction config: {exc}") from exc

    environment = os.environ if environ is None else environ
    configured_regions = {
        name: environment[name] for name in ("AWS_REGION", "AWS_DEFAULT_REGION") if environment.get(name)
    }
    if configured_regions:
        wrong = {name: region for name, region in configured_regions.items() if region != expected_region}
        if wrong:
            raise StageError(f"AWS environment region mismatch: expected {expected_region}, found {wrong}")
    else:
        cli_region = runner(["aws", "configure", "get", "region"]).strip()
        if cli_region != expected_region:
            raise StageError(f"AWS CLI region mismatch: expected {expected_region}, found {cli_region!r}")

    identity = _json_command(
        runner,
        ["aws", "sts", "get-caller-identity", "--region", expected_region, "--output", "json"],
    )
    if str(identity.get("Account")) != account_id:
        raise StageError(f"AWS account mismatch: expected {account_id}, found {identity.get('Account')!r}")

    common = ["--bucket", target.bucket, "--expected-bucket-owner", account_id, "--region", expected_region]
    location = _json_command(runner, ["aws", "s3api", "get-bucket-location", *common, "--output", "json"])
    bucket_region = location.get("LocationConstraint") or "us-east-1"
    if bucket_region != expected_region:
        raise StageError(f"S3 bucket region mismatch: expected {expected_region}, found {bucket_region}")
    versioning = _json_command(runner, ["aws", "s3api", "get-bucket-versioning", *common, "--output", "json"])
    if versioning.get("Status") != "Enabled":
        raise StageError(f"S3 bucket versioning is not enabled for {target.bucket}")
    encryption = _json_command(runner, ["aws", "s3api", "get-bucket-encryption", *common, "--output", "json"])
    if not encryption.get("ServerSideEncryptionConfiguration", {}).get("Rules"):
        raise StageError(f"S3 bucket default encryption is not configured for {target.bucket}")
    return account_id, expected_region


def _sync_argv(
    snapshot_root: pathlib.Path,
    target: S3Target,
    spec: DatasetSpec,
    *,
    account_id: str,
    region: str,
) -> list[str]:
    metadata = (
        f"source-provider=huggingface,source-repo={spec.repo_id},"
        f"source-revision={spec.revision},lerobot-version={spec.codebase_version}"
    )
    return [
        "aws",
        "s3",
        "sync",
        str(snapshot_root),
        target.snapshot_uri,
        "--region",
        region,
        "--exclude",
        ".cache/*",
        "--no-follow-symlinks",
        "--only-show-errors",
        "--no-progress",
        "--sse",
        "AES256",
        "--metadata",
        metadata,
    ]


def upload_snapshot(
    config: Mapping[str, Any],
    spec: DatasetSpec,
    snapshot_root: pathlib.Path,
    manifest_path: pathlib.Path,
    target: S3Target,
    *,
    runner: CommandRunner = run_command,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    account_id, region = verify_aws_destination(config, target, runner=runner, environ=environ)
    sync_argv = _sync_argv(snapshot_root, target, spec, account_id=account_id, region=region)
    runner(sync_argv)
    runner(
        [
            "aws",
            "s3api",
            "put-object",
            "--bucket",
            target.bucket,
            "--key",
            target.manifest_key,
            "--body",
            str(manifest_path),
            "--expected-bucket-owner",
            account_id,
            "--region",
            region,
            "--server-side-encryption",
            "AES256",
            "--metadata",
            f"source-repo={spec.repo_id},source-revision={spec.revision}",
        ]
    )
    verification_argv = [argument for argument in sync_argv if argument != "--only-show-errors"]
    dry_run_output = runner([*verification_argv, "--dryrun"])
    if dry_run_output.strip():
        raise StageError(f"S3 sync verification still reports pending changes:\n{dry_run_output}")
    head = _json_command(
        runner,
        [
            "aws",
            "s3api",
            "head-object",
            "--bucket",
            target.bucket,
            "--key",
            target.manifest_key,
            "--expected-bucket-owner",
            account_id,
            "--region",
            region,
            "--output",
            "json",
        ],
    )
    if int(head.get("ContentLength", -1)) != manifest_path.stat().st_size:
        raise StageError("uploaded integrity manifest has an unexpected content length")
    remote_metadata = head.get("Metadata", {})
    if remote_metadata.get("source-revision") != spec.revision:
        raise StageError("uploaded integrity manifest is missing its pinned source revision metadata")
    if not head.get("VersionId"):
        raise StageError("uploaded integrity manifest has no S3 version ID")
    return {
        "snapshot_uri": target.snapshot_uri,
        "manifest_uri": target.manifest_uri,
        "manifest_version_id": head.get("VersionId"),
        "worker_artifact": {
            "name": spec.key,
            "kind": "dataset",
            "revision": spec.revision,
            "manifest": {
                "s3_uri": target.manifest_uri,
                "version_id": head.get("VersionId"),
                "sha256": sha256_file(manifest_path),
            },
            "payload_s3_uri": target.snapshot_uri,
            "destination": spec.local_dirname,
        },
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", nargs="?", default="plan", choices=("plan", "download", "validate", "upload", "stage")
    )
    parser.add_argument("--dataset", required=True, choices=("libero", "droid"))
    parser.add_argument("--local-root", required=True, type=pathlib.Path)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest-output", type=pathlib.Path)
    parser.add_argument("--s3-root", help="base URI; dataset name and pinned revision are appended")
    parser.add_argument("--hash-workers", type=int, default=4)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="authorize the requested Hugging Face download and/or S3 upload",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_json(args.config)
        spec = dataset_spec(config, args.dataset)
        snapshot_root, manifest_path = resolve_paths(args.local_root, spec, args.manifest_output)
        target = parse_s3_target(args.s3_root, spec) if args.s3_root else None
        if args.action in {"upload", "stage"} and target is None:
            raise StageError(f"--s3-root is required for {args.action}")

        plan = render_plan(
            action=args.action,
            spec=spec,
            snapshot_root=snapshot_root,
            manifest_path=manifest_path,
            s3_target=target,
        )
        if args.action == "plan" or (args.action in {"download", "upload", "stage"} and not args.execute):
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0

        if args.action in {"download", "stage"}:
            download_snapshot(spec, snapshot_root)
        validation = validate_snapshot(spec, snapshot_root)
        manifest = build_manifest(spec, snapshot_root, validation, hash_workers=args.hash_workers)
        write_manifest(manifest_path, manifest)
        result: dict[str, Any] = {
            "dataset": spec.key,
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "validation": validation,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        }
        if args.action in {"upload", "stage"}:
            assert target is not None
            result["s3"] = upload_snapshot(config, spec, snapshot_root, manifest_path, target)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (StageError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"DATA STAGING REJECTED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
