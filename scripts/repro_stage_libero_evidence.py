#!/usr/bin/env python3
# ruff: noqa: SLF001
"""Validate and immutably publish one direct eager LIBERO runtime smoke.

Local validation is always performed first and obtains fresh host identity from
IMDSv2.  ``upload`` makes no AWS control-plane call unless the exact
``--execute`` flag is present.  Executing publication re-downloads every pinned
external S3 input by exact VersionId, uses a deterministic claim first, uploads
every snapshotted payload with create-once semantics, writes a durable receipt,
and writes the evidence manifest last.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
import dataclasses
import datetime as dt
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
import urllib.parse

try:
    from scripts import repro_stage_converted_checkpoints as converted
    from scripts import repro_stage_data
    from scripts import repro_worker
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root.
    import repro_stage_converted_checkpoints as converted
    import repro_stage_data
    import repro_worker


DEFAULT_CONFIG = pathlib.Path("repro/reproduction.json")
EXPECTED_ACCOUNT = "752160877725"
EXPECTED_REGION = "us-east-2"
EXPECTED_BUCKET = "pi05-repro-752160877725-us-east-2"
EXPECTED_PROJECT = "pi05-aws-repro"
EXPECTED_EVALUATOR_SOURCE_COMMIT = "e30480a6de404c74a996863c4fde89367350cf70"
EXPECTED_EVALUATOR_IMAGE_DIGEST = "sha256:51b352c1a7205d6bdae668f99060ebd05049042e1d89916993830acbdc63b374"
EXPECTED_PARENT_POLICY_SOURCE_COMMIT = "229c08ea2a13a70cbbf1a9c8a1f31cb1ca674dee"
EXPECTED_PARENT_POLICY_IMAGE_DIGEST = "sha256:d76e6d73fca409e998304a6a8997f80fab1252fe0301d667a072f99dd6624f24"
EXPECTED_MODEL_REVISION = "c73bb6ff5cbaa3c7bba5f03ea38c22bd95e8274308285e2f17b6ed2d73688dd0"
EXPECTED_CONVERTED_ARTIFACT_FILE_SHA256 = "df51b3779ce4c597eb634b2ce8b11bb6cd85401b7ed2e73b3b518c47b1333395"
EXPECTED_INSTANCE_TYPE = "g6e.4xlarge"
EXPECTED_POLICY_CONFIG = "pi05_libero"
EXPECTED_CHECKPOINT = "/mnt/openpi/checkpoints/pi05_libero"
EXPECTED_STAGE = "base"
EXPECTED_SEED = 7
EXPECTED_TRIALS_PER_TASK = 1
EXPECTED_RUN_IDS = ("libero-base-runtime-smoke-05", "libero-base-runtime-smoke-06")
EXPECTED_SIMULATOR = {
    "repository": "https://github.com/Lifelong-Robot-Learning/LIBERO.git",
    "revision": "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c",
}
EXPECTED_DEPENDENCIES = {
    "path": "repro/libero-evaluator-requirements.txt",
    "installed_path": "/opt/libero-evaluator-requirements.txt",
    "sha256": "124e74d09719941c9e3e75a61330808a8d32ae35a1ebee00c18e1222e966d0c8",
}
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MAX_EPISODE_STEPS = {
    "libero_spatial": 230,
    "libero_object": 290,
    "libero_goal": 310,
    "libero_10": 530,
}
EXPECTED_SUITE_SUCCESSES = {
    "libero_spatial": 9,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
}
EXPECTED_MANIFEST_KEYS = {
    "schema_version",
    "project",
    "kind",
    "run_id",
    "started_at",
    "finished_at",
    "source",
    "image",
    "dataset",
    "simulator",
    "dependencies",
    "policy",
    "evaluation",
    "command",
    "child_commands",
    "instance",
    "cost",
    "artifacts",
}
EXPECTED_EPISODE_KEYS = {
    "pair_id",
    "stage",
    "benchmark",
    "suite",
    "task",
    "task_id",
    "success",
    "seed",
    "init_index",
    "steps",
    "libero_revision",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
INSTANCE_ID_RE = re.compile(r"^i-(?:[0-9a-f]{8}|[0-9a-f]{17})$")

CommandRunner = Callable[[Sequence[str]], str]


@dataclasses.dataclass(frozen=True)
class SealedSmoke:
    root: pathlib.Path
    revision: str
    files: tuple[dict[str, Any], ...]
    content: dict[str, Any]
    checkpoint_root: pathlib.Path
    converted_manifest_path: pathlib.Path
    converted_checkpoint_artifact_path: pathlib.Path
    instance_identity: dict[str, Any]
    cost_ledger_path: pathlib.Path
    cost_ledger: dict[str, Any]


def _fail(message: str) -> None:
    raise repro_stage_data.StageError(message)


def _duplicate_safe_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            _fail(f"JSON object contains duplicate key: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    _fail(f"JSON contains non-finite constant: {value}")


def _load_json(path: pathlib.Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_duplicate_safe_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot read {label} JSON {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{label} JSON must be an object: {path}")
    return value


def _canonical_bytes(value: Mapping[str, Any], *, pretty: bool = False) -> bytes:
    try:
        if pretty:
            return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
        return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    except (TypeError, ValueError) as exc:
        _fail(f"evidence is not finite canonical JSON: {exc}")


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_stat_identity(path: pathlib.Path) -> tuple[int, ...]:
    stat = path.stat(follow_symlinks=False)
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_mode,
        stat.st_nlink,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _stable_file_record(path: pathlib.Path, *, root: pathlib.Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"smoke payload must be a regular non-symlink file: {path}")
    before = _file_stat_identity(path)
    if before[3] != 1:
        _fail(f"smoke payload must not be a hard link: {path}")
    digest = repro_stage_data.sha256_file(path)
    after = _file_stat_identity(path)
    if before != after:
        _fail(f"smoke payload changed while hashing: {path}")
    return {"path": path.relative_to(root).as_posix(), "bytes": before[4], "sha256": digest}


def _single_line(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or len(value) > 1024
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        _fail(f"{label} must be a non-empty single-line string")
    return value


def _validate_instance_identity(identity: Mapping[str, Any], *, instance_id: str) -> dict[str, Any]:
    expected = {
        "accountId": EXPECTED_ACCOUNT,
        "region": EXPECTED_REGION,
        "instanceType": EXPECTED_INSTANCE_TYPE,
        "instanceId": instance_id,
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        _fail("fresh IMDSv2 identity differs from the pinned account, region, instance type, or instance ID")
    return {
        "account_id": EXPECTED_ACCOUNT,
        "region": EXPECTED_REGION,
        "type": EXPECTED_INSTANCE_TYPE,
        "id": instance_id,
        "identity_recorded_by": "fresh host IMDSv2 at evidence seal time",
    }


def _validate_cost_ledger(
    *,
    s3_uri: str,
    version_id: str,
    sha256: str,
) -> dict[str, str]:
    parsed = urllib.parse.urlsplit(s3_uri)
    if (
        parsed.scheme != "s3"
        or parsed.netloc != EXPECTED_BUCKET
        or parsed.path != "/control/cost-ledger.json"
        or parsed.query
        or parsed.fragment
    ):
        _fail("--cost-ledger-s3-uri must name the project control/cost-ledger.json object")
    _single_line(version_id, label="--cost-ledger-version-id")
    if SHA256_RE.fullmatch(sha256) is None:
        _fail("--cost-ledger-sha256 must be a lowercase SHA-256")
    return {"s3_uri": s3_uri, "version_id": version_id, "sha256": sha256}


def _validate_cost_ledger_file(
    path: pathlib.Path,
    descriptor: Mapping[str, str],
    *,
    instance_id: str,
    timing: Mapping[str, Any],
    projected_cost: Any,
) -> dict[str, Any]:
    if not path.is_absolute():
        _fail("--cost-ledger-path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise repro_stage_data.StageError(f"--cost-ledger-path does not resolve: {path}: {exc}") from exc
    if path != resolved or path.is_symlink():
        _fail("--cost-ledger-path must be normalized and must not traverse a symlink")
    record = _stable_file_record(path, root=path.parent)
    if record["sha256"] != descriptor["sha256"]:
        _fail("local exact-version cost ledger differs from --cost-ledger-sha256")
    ledger = _load_json(path, label="cost ledger")
    if (
        set(ledger) != {"schema_version", "entries"}
        or not isinstance(ledger.get("schema_version"), int)
        or isinstance(ledger.get("schema_version"), bool)
        or ledger.get("schema_version") != 1
        or not isinstance(ledger.get("entries"), list)
    ):
        _fail("cost ledger has the wrong schema")
    if not isinstance(projected_cost, int | float) or isinstance(projected_cost, bool):
        _fail("evaluation projected cost is invalid")
    started = _parse_utc(timing.get("started_at"), label="smoke timing started_at")
    finished = _parse_utc(timing.get("finished_at"), label="smoke timing finished_at")
    covering_entries: list[str] = []
    for entry in ledger["entries"]:
        if not isinstance(entry, Mapping):
            continue
        instance_ids = entry.get("instance_ids")
        entry_id = entry.get("id")
        usd = entry.get("usd")
        if (
            entry.get("category") not in {"workbench_setup", "evaluation"}
            or entry.get("instance_type") != EXPECTED_INSTANCE_TYPE
            or not isinstance(instance_ids, list)
            or instance_id not in instance_ids
            or not isinstance(entry_id, str)
            or not entry_id
            or not isinstance(usd, int | float)
            or isinstance(usd, bool)
            or not math.isfinite(float(usd))
            or float(usd) < float(projected_cost)
        ):
            continue
        try:
            entry_started = _parse_utc(entry.get("created_at"), label=f"cost entry {entry_id} created_at")
            entry_deadline = _parse_utc(entry.get("deadline_utc"), label=f"cost entry {entry_id} deadline_utc")
        except repro_stage_data.StageError:
            continue
        if entry_started <= started <= finished <= entry_deadline:
            covering_entries.append(entry_id)
    if not covering_entries:
        _fail("cost ledger has no paid entry covering this exact instance, smoke interval, and projected cost")
    return {
        **descriptor,
        "local_bytes": record["bytes"],
        "covering_entry_ids": sorted(set(covering_entries)),
    }


def _safe_payload_path(value: Any, *, label: str) -> str:
    raw = _single_line(value, label=label)
    path = pathlib.PurePosixPath(raw)
    if path.is_absolute() or raw != path.as_posix() or any(part in {"", ".", ".."} for part in path.parts):
        _fail(f"{label} must be a normalized relative POSIX path")
    return raw


def _validate_converted_checkpoint(
    checkpoint_root: pathlib.Path,
    converted_manifest_path: pathlib.Path,
    converted_checkpoint_artifact_path: pathlib.Path,
    *,
    parent_policy_source_commit: str,
    parent_policy_image_digest: str,
    model_revision: str,
) -> dict[str, Any]:
    for value, label in (
        (checkpoint_root, "--checkpoint-root"),
        (converted_manifest_path, "--converted-manifest"),
        (converted_checkpoint_artifact_path, "--converted-checkpoint-artifact"),
    ):
        if not value.is_absolute():
            _fail(f"{label} must be absolute")
        try:
            resolved = value.resolve(strict=True)
        except OSError as exc:
            raise repro_stage_data.StageError(f"{label} does not resolve: {value}: {exc}") from exc
        if value != resolved or value.is_symlink():
            _fail(f"{label} must be normalized and must not traverse a symlink")

    artifact_record = _stable_file_record(
        converted_checkpoint_artifact_path,
        root=converted_checkpoint_artifact_path.parent,
    )
    if artifact_record["sha256"] != EXPECTED_CONVERTED_ARTIFACT_FILE_SHA256:
        _fail("converted checkpoint worker_artifact is not the exact tracked published descriptor")
    artifact = _load_json(converted_checkpoint_artifact_path, label="converted checkpoint worker_artifact")
    if set(artifact) != {
        "name",
        "kind",
        "revision",
        "manifest",
        "payload_s3_uri",
        "payload_objects",
        "destination",
    }:
        _fail("converted checkpoint worker_artifact has the wrong schema")

    spec = converted.CONVERTED_CHECKPOINTS["libero"]
    local_files = converted._validate_conversion_output(checkpoint_root, spec)
    manifest = _load_json(converted_manifest_path, label="converted checkpoint manifest")
    if set(manifest) != {"schema_version", "source", "conversion", "checkpoint", "totals", "files"}:
        _fail("converted checkpoint manifest has the wrong top-level schema")
    if (
        not isinstance(manifest.get("schema_version"), int)
        or isinstance(manifest.get("schema_version"), bool)
        or manifest.get("schema_version") != 1
    ):
        _fail("converted checkpoint manifest schema_version must be 1")
    source = manifest.get("source")
    conversion = manifest.get("conversion")
    checkpoint = manifest.get("checkpoint")
    totals = manifest.get("totals")
    manifest_files = manifest.get("files")
    if (
        not isinstance(source, Mapping)
        or set(source) != {"provider", "revision_kind", "revision", "upstream"}
        or source.get("provider") != "openpi-jax-to-pytorch"
        or source.get("revision_kind") != "converted-checkpoint-content-and-provenance-sha256"
        or source.get("revision") != model_revision
        or not isinstance(source.get("upstream"), Mapping)
    ):
        _fail("converted checkpoint manifest has the wrong immutable source identity")
    if (
        not isinstance(conversion, Mapping)
        or set(conversion) != {"source_commit", "image_digest", "converter", "config_name", "precision"}
        or conversion.get("source_commit") != parent_policy_source_commit
        or conversion.get("image_digest") != parent_policy_image_digest
        or conversion.get("converter") != "examples/convert_jax_model_to_pytorch.py"
        or conversion.get("config_name") != EXPECTED_POLICY_CONFIG
        or conversion.get("precision") != "bfloat16"
    ):
        _fail("converted checkpoint conversion identity differs from the parent policy boundary")
    if checkpoint != {
        "key": "libero",
        "local_dirname": "pi05_libero_pytorch",
        "format": "pytorch-safetensors",
    }:
        _fail("converted checkpoint manifest describes the wrong checkpoint")
    if not isinstance(manifest_files, list) or not manifest_files:
        _fail("converted checkpoint manifest files must be a non-empty list")

    rebuilt_files = [_stable_file_record(path, root=checkpoint_root) for path in local_files]
    if manifest_files != rebuilt_files:
        _fail("local converted checkpoint bytes differ from the immutable converted manifest")
    if totals != {
        "files": len(rebuilt_files),
        "bytes": sum(int(item["bytes"]) for item in rebuilt_files),
    }:
        _fail("converted checkpoint totals differ from the local byte inventory")
    try:
        expected_revision = converted.conversion_revision(converted.manifest_identity(manifest))
    except (KeyError, TypeError, ValueError) as exc:
        raise repro_stage_data.StageError(f"converted checkpoint manifest identity is malformed: {exc}") from exc
    if expected_revision != model_revision:
        _fail("converted checkpoint revision is inconsistent with its canonical manifest identity")

    manifest_record = _stable_file_record(converted_manifest_path, root=converted_manifest_path.parent)
    target = converted.converted_s3_target(
        f"s3://{EXPECTED_BUCKET}/checkpoints",
        spec,
        model_revision,
    )
    descriptor_manifest = artifact.get("manifest")
    if (
        artifact.get("name") != "libero_teacher_pytorch"
        or artifact.get("kind") != "checkpoint"
        or artifact.get("revision") != model_revision
        or artifact.get("destination") != "pi05_libero_pytorch"
        or artifact.get("payload_s3_uri") != target.snapshot_uri
        or not isinstance(descriptor_manifest, Mapping)
        or set(descriptor_manifest) != {"s3_uri", "version_id", "sha256"}
        or descriptor_manifest.get("s3_uri") != target.manifest_uri
        or descriptor_manifest.get("sha256") != manifest_record["sha256"]
        or SHA256_RE.fullmatch(str(descriptor_manifest.get("sha256", ""))) is None
    ):
        _fail("converted checkpoint worker_artifact does not bind the verified LIBERO checkpoint")
    _single_line(descriptor_manifest.get("version_id"), label="converted manifest version_id")

    payload_objects = artifact.get("payload_objects")
    if not isinstance(payload_objects, list) or len(payload_objects) != len(rebuilt_files):
        _fail("converted checkpoint worker_artifact payload inventory is incomplete")
    expected_by_path = {str(item["path"]): str(item["sha256"]) for item in rebuilt_files}
    observed: dict[str, str] = {}
    for index, item in enumerate(payload_objects):
        if not isinstance(item, Mapping) or set(item) != {"path", "version_id", "sha256"}:
            _fail(f"converted checkpoint payload object {index} has the wrong schema")
        path = _safe_payload_path(item.get("path"), label=f"converted payload object {index} path")
        version_id = _single_line(item.get("version_id"), label=f"converted payload object {index} version_id")
        digest = item.get("sha256")
        if path in observed or SHA256_RE.fullmatch(str(digest or "")) is None or expected_by_path.get(path) != digest:
            _fail(f"converted checkpoint payload object {index} differs from the verified local bytes")
        observed[path] = version_id
    if set(observed) != set(expected_by_path):
        _fail("converted checkpoint worker_artifact does not cover every verified local file")

    return {
        "artifact": artifact,
        "local_manifest": {
            "bytes": manifest_record["bytes"],
            "sha256": manifest_record["sha256"],
        },
        "descriptor_file": {
            "bytes": artifact_record["bytes"],
            "sha256": artifact_record["sha256"],
        },
        "local_bytes_verified": True,
    }


def _parse_utc(value: Any, *, label: str) -> dt.datetime:
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be a non-empty UTC timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.removesuffix("Z") + ("+00:00" if value.endswith("Z") else ""))
    except ValueError:
        _fail(f"{label} is not an ISO-8601 timestamp: {value!r}")
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        _fail(f"{label} must include an explicit UTC offset")
    return parsed


def _exact_paths(root: pathlib.Path) -> dict[str, pathlib.Path]:
    if not root.is_absolute():
        _fail(f"--output-root must be absolute: {root}")
    if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
        _fail(f"smoke output root is missing, not a directory, or traverses a symlink: {root}")
    expected_files = {
        *(f"artifacts/libero/base/{suite}.jsonl" for suite in SUITES),
        "artifacts/libero/base/episodes.jsonl",
        "manifests/libero-base.json",
        "replay.log",
        "timing.json",
    }
    expected_dirs = {
        "artifacts",
        "artifacts/libero",
        "artifacts/libero/base",
        "manifests",
    }
    observed_files: set[str] = set()
    observed_dirs: set[str] = set()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            _fail(f"smoke output contains a symlink: {path}")
        relative = path.relative_to(root).as_posix()
        if path.is_file():
            observed_files.add(relative)
        elif path.is_dir():
            observed_dirs.add(relative)
        else:
            _fail(f"smoke output contains a non-regular entry: {path}")
    if observed_files != expected_files:
        _fail(
            "smoke output file inventory differs: "
            f"missing={sorted(expected_files - observed_files)}, extras={sorted(observed_files - expected_files)}"
        )
    if observed_dirs != expected_dirs:
        _fail(
            "smoke output directory inventory differs: "
            f"missing={sorted(expected_dirs - observed_dirs)}, extras={sorted(observed_dirs - expected_dirs)}"
        )
    return {relative: root / relative for relative in sorted(expected_files)}


def _parse_jsonl(path: pathlib.Path, *, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text().splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        _fail(f"cannot read {label} JSONL {path}: {exc}")
    if not lines or any(not line for line in lines):
        _fail(f"{label} JSONL must contain only non-empty records")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(
                line,
                object_pairs_hook=_duplicate_safe_object,
                parse_constant=_reject_json_constant,
            )
        except json.JSONDecodeError as exc:
            _fail(f"invalid JSONL in {path}:{line_number}: {exc}")
        if not isinstance(record, dict):
            _fail(f"non-object JSONL record in {path}:{line_number}")
        records.append(record)
    return records


def _validate_episode_records(paths: Mapping[str, pathlib.Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_records: list[dict[str, Any]] = []
    pair_ids: set[str] = set()
    suite_metrics: dict[str, dict[str, Any]] = {}
    for suite in SUITES:
        relative = f"artifacts/libero/base/{suite}.jsonl"
        records = _parse_jsonl(paths[relative], label=suite)
        if len(records) != 10:
            _fail(f"{suite} runtime smoke must contain exactly 10 episodes, found {len(records)}")
        if [record.get("task_id") for record in records] != list(range(10)):
            _fail(f"{suite} must contain task IDs 0 through 9 exactly once and in order")
        for record in records:
            if set(record) != EXPECTED_EPISODE_KEYS:
                _fail(f"{suite} episode record has the wrong schema keys")
            task_id = record.get("task_id")
            if not isinstance(task_id, int) or isinstance(task_id, bool):
                _fail(f"{suite} episode task ID is not an integer")
            expected_pair_id = f"libero:{suite}:task-{task_id:03d}:init-000:seed-{EXPECTED_SEED}"
            pair_id = record.get("pair_id")
            if (
                record.get("stage") != EXPECTED_STAGE
                or record.get("benchmark") != "libero"
                or record.get("suite") != suite
                or record.get("seed") != EXPECTED_SEED
                or record.get("init_index") != 0
                or not isinstance(record.get("init_index"), int)
                or isinstance(record.get("init_index"), bool)
                or record.get("libero_revision") != EXPECTED_SIMULATOR["revision"]
                or pair_id != expected_pair_id
                or pair_id in pair_ids
                or not isinstance(record.get("task"), str)
                or not record["task"].strip()
                or not isinstance(record.get("success"), bool)
                or not isinstance(record.get("steps"), int)
                or isinstance(record.get("steps"), bool)
                or (record["success"] and not 11 <= record["steps"] <= MAX_EPISODE_STEPS[suite])
                or (not record["success"] and record["steps"] != MAX_EPISODE_STEPS[suite])
            ):
                _fail(f"{suite} episode identity, value, or rollout-step semantics are invalid: task {task_id!r}")
            pair_ids.add(pair_id)
        successes = sum(record["success"] for record in records)
        if successes != EXPECTED_SUITE_SUCCESSES[suite]:
            _fail(
                f"{suite} success count differs from accepted attempts 05/06: "
                f"expected {EXPECTED_SUITE_SUCCESSES[suite]}, found {successes}"
            )
        suite_metrics[suite] = {
            "episodes": 10,
            "successes": successes,
            "success_rate": successes / 10,
        }
        all_records.extend(records)

    combined_path = paths["artifacts/libero/base/episodes.jsonl"]
    combined_records = _parse_jsonl(combined_path, label="combined episodes")
    if combined_records != all_records:
        _fail("combined episodes JSONL does not equal the ordered four-suite records")
    expected_combined_bytes = b"".join(paths[f"artifacts/libero/base/{suite}.jsonl"].read_bytes() for suite in SUITES)
    if combined_path.read_bytes() != expected_combined_bytes:
        _fail("combined episodes JSONL is not the byte-exact ordered suite concatenation")
    successes = sum(record["success"] for record in all_records)
    metrics = {
        "episodes": 40,
        "successes": successes,
        "success_rate": successes / 40,
        "environment_steps": sum(record["steps"] for record in all_records),
        "infrastructure_errors": 0,
        "suites": suite_metrics,
    }
    return all_records, metrics


def _expected_command(model_revision: str, projected_cost_token: str) -> list[str]:
    return [
        "scripts/repro_libero_eval.py",
        "run",
        "--policy-config",
        EXPECTED_POLICY_CONFIG,
        "--checkpoint",
        EXPECTED_CHECKPOINT,
        "--model-revision",
        model_revision,
        "--stage",
        EXPECTED_STAGE,
        "--trials-per-task",
        str(EXPECTED_TRIALS_PER_TASK),
        "--seed",
        str(EXPECTED_SEED),
        "--instance-type",
        EXPECTED_INSTANCE_TYPE,
        "--projected-cost-usd",
        projected_cost_token,
        "--output-root",
        "/output",
    ]


def _validate_child_commands(value: Any) -> None:
    if not isinstance(value, list) or len(value) != 5 or any(not isinstance(item, list) for item in value):
        _fail("LIBERO manifest must contain the policy command and four suite commands")
    expected_server_tail = [
        "scripts/serve_policy.py",
        "--env",
        "LIBERO",
        "--port",
        "8000",
        "--seed",
        str(EXPECTED_SEED),
        "policy:checkpoint",
        "--policy.config",
        EXPECTED_POLICY_CONFIG,
        "--policy.dir",
        EXPECTED_CHECKPOINT,
    ]
    server = value[0]
    if len(server) != len(expected_server_tail) + 1 or not pathlib.PurePosixPath(str(server[0])).is_absolute():
        _fail("LIBERO policy child command has an invalid Python executable")
    if server[1:] != expected_server_tail:
        _fail("LIBERO policy child command differs from the reviewed eager server command")
    for suite, command in zip(SUITES, value[1:], strict=True):
        expected = [
            "/opt/libero-venv/bin/python",
            "examples/libero/main.py",
            "--args.host",
            "127.0.0.1",
            "--args.port",
            "8000",
            "--args.task-suite-name",
            suite,
            "--args.num-trials-per-task",
            str(EXPECTED_TRIALS_PER_TASK),
            "--args.stage",
            EXPECTED_STAGE,
            "--args.seed",
            str(EXPECTED_SEED),
            "--args.results-out-path",
            f"/output/artifacts/libero/base/{suite}.jsonl",
            "--args.runtime-contract-path",
            "/opt/libero-evaluator-contract.json",
            "--args.expected-libero-revision",
            EXPECTED_SIMULATOR["revision"],
            "--args.no-save-videos",
        ]
        if command != expected:
            _fail(f"LIBERO {suite} child command differs from the reviewed evaluator command")


def _validate_manifest(
    path: pathlib.Path,
    paths: Mapping[str, pathlib.Path],
    *,
    run_id: str,
    evaluator_source_commit: str,
    evaluator_image_digest: str,
    model_revision: str,
    instance_id: str,
    metrics: Mapping[str, Any],
    timing: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = _load_json(path, label="LIBERO evaluation manifest")
    if (
        set(manifest) != EXPECTED_MANIFEST_KEYS
        or not isinstance(manifest.get("schema_version"), int)
        or isinstance(manifest.get("schema_version"), bool)
        or manifest.get("schema_version") != 1
    ):
        _fail("LIBERO evaluation manifest has the wrong schema")
    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, Mapping) or set(evaluation) != {
        "stage",
        "seed",
        "suites",
        "trials_per_task",
        "metrics",
    }:
        _fail("LIBERO evaluation identity has the wrong schema")
    cost = manifest.get("cost")
    if not isinstance(cost, Mapping) or set(cost) != {"projected_usd", "actual_recorded_by"}:
        _fail("LIBERO evaluation cost has the wrong schema")
    projected_cost = cost.get("projected_usd")
    if (
        not isinstance(projected_cost, int | float)
        or isinstance(projected_cost, bool)
        or not math.isfinite(float(projected_cost))
        or float(projected_cost) <= 0
    ):
        _fail("LIBERO runtime smoke must record a finite positive projected cost allocation")
    command = manifest.get("command")
    if not isinstance(command, list):
        _fail("LIBERO evaluation command must be argv")
    try:
        projected_cost_token = command[command.index("--projected-cost-usd") + 1]
    except (ValueError, IndexError) as exc:
        raise repro_stage_data.StageError("LIBERO evaluation command has no projected-cost argument") from exc
    try:
        command_cost = float(projected_cost_token)
    except (TypeError, ValueError) as exc:
        raise repro_stage_data.StageError("LIBERO projected-cost command argument is not numeric") from exc
    if not math.isfinite(command_cost) or command_cost != float(projected_cost):
        _fail("LIBERO projected cost differs between command and manifest")
    if command != _expected_command(model_revision, str(projected_cost_token)):
        _fail("LIBERO evaluation command differs from the reviewed base-smoke command")

    expected_identity = {
        "project": EXPECTED_PROJECT,
        "kind": "libero-evaluation",
        "run_id": run_id,
        "source": {"commit": evaluator_source_commit},
        "image": {"digest": evaluator_image_digest},
        "dataset": {"name": "LIBERO fixed benchmark assets", "revision": EXPECTED_SIMULATOR["revision"]},
        "simulator": EXPECTED_SIMULATOR,
        "dependencies": EXPECTED_DEPENDENCIES,
        "policy": {
            "backend": "eager",
            "config": EXPECTED_POLICY_CONFIG,
            "checkpoint": EXPECTED_CHECKPOINT,
            "model_revision": model_revision,
        },
        "instance": {
            "type": EXPECTED_INSTANCE_TYPE,
            "id": instance_id,
            "identity_recorded_by": "worker run manifest",
        },
        "cost": {"projected_usd": projected_cost, "actual_recorded_by": "worker run manifest"},
    }
    for key, expected in expected_identity.items():
        if manifest.get(key) != expected:
            _fail(f"LIBERO evaluation manifest identity differs for {key}")
    expected_evaluation = {
        "stage": EXPECTED_STAGE,
        "seed": EXPECTED_SEED,
        "suites": list(SUITES),
        "trials_per_task": EXPECTED_TRIALS_PER_TASK,
        "metrics": dict(metrics),
    }
    if _canonical_bytes(evaluation) != _canonical_bytes(expected_evaluation):
        _fail("LIBERO evaluation metrics or rollout identity differ from recomputed values")
    _validate_child_commands(manifest.get("child_commands"))

    started = _parse_utc(manifest.get("started_at"), label="LIBERO manifest started_at")
    finished = _parse_utc(manifest.get("finished_at"), label="LIBERO manifest finished_at")
    timing_started = _parse_utc(timing.get("started_at"), label="smoke timing started_at")
    timing_finished = _parse_utc(timing.get("finished_at"), label="smoke timing finished_at")
    # The retained Bash wrapper uses GNU date's whole-second ISO format while
    # the evaluator manifest records microseconds.  Permit only that final
    # one-second truncation; the wrapper must still start before evaluation.
    if not timing_started <= started <= finished <= timing_finished + dt.timedelta(seconds=1):
        _fail("smoke timing does not enclose the complete LIBERO evaluation interval")

    expected_artifacts = [
        _stable_file_record(paths[f"artifacts/libero/base/{suite}.jsonl"], root=path.parents[1]) for suite in SUITES
    ]
    expected_artifacts.append(_stable_file_record(paths["artifacts/libero/base/episodes.jsonl"], root=path.parents[1]))
    if manifest.get("artifacts") != expected_artifacts:
        _fail("LIBERO manifest artifact hashes or sizes differ from the exact JSONL payloads")
    return manifest


def validate_smoke(
    output_root: pathlib.Path,
    *,
    run_id: str,
    evaluator_source_commit: str,
    evaluator_image_digest: str,
    parent_policy_source_commit: str,
    parent_policy_image_digest: str,
    model_revision: str,
    instance_id: str,
    instance_identity: Mapping[str, Any],
    checkpoint_root: pathlib.Path,
    converted_manifest_path: pathlib.Path,
    converted_checkpoint_artifact_path: pathlib.Path,
    cost_ledger_path: pathlib.Path,
    cost_ledger_s3_uri: str,
    cost_ledger_version_id: str,
    cost_ledger_sha256: str,
) -> SealedSmoke:
    if run_id not in EXPECTED_RUN_IDS:
        _fail(f"--run-id must identify one accepted clean replay: {EXPECTED_RUN_IDS}")
    if evaluator_source_commit != EXPECTED_EVALUATOR_SOURCE_COMMIT:
        _fail(f"base smoke evaluator source must be {EXPECTED_EVALUATOR_SOURCE_COMMIT}")
    if evaluator_image_digest != EXPECTED_EVALUATOR_IMAGE_DIGEST:
        _fail(f"base smoke evaluator image must be {EXPECTED_EVALUATOR_IMAGE_DIGEST}")
    if parent_policy_source_commit != EXPECTED_PARENT_POLICY_SOURCE_COMMIT:
        _fail(f"base smoke parent policy source must be {EXPECTED_PARENT_POLICY_SOURCE_COMMIT}")
    if parent_policy_image_digest != EXPECTED_PARENT_POLICY_IMAGE_DIGEST:
        _fail(f"base smoke parent policy image must be {EXPECTED_PARENT_POLICY_IMAGE_DIGEST}")
    if model_revision != EXPECTED_MODEL_REVISION or SHA256_RE.fullmatch(model_revision) is None:
        _fail(f"base smoke model revision must be {EXPECTED_MODEL_REVISION}")
    if INSTANCE_ID_RE.fullmatch(instance_id) is None:
        _fail("--instance-id must be a valid EC2 instance ID")
    validated_instance = _validate_instance_identity(instance_identity, instance_id=instance_id)
    cost_ledger_descriptor = _validate_cost_ledger(
        s3_uri=cost_ledger_s3_uri,
        version_id=cost_ledger_version_id,
        sha256=cost_ledger_sha256,
    )
    checkpoint_evidence = _validate_converted_checkpoint(
        checkpoint_root,
        converted_manifest_path,
        converted_checkpoint_artifact_path,
        parent_policy_source_commit=parent_policy_source_commit,
        parent_policy_image_digest=parent_policy_image_digest,
        model_revision=model_revision,
    )

    root = output_root.resolve(strict=True)
    if root != output_root:
        _fail("--output-root must be an already-normalized absolute path without symlink traversal")
    paths = _exact_paths(root)
    replay_log = paths["replay.log"]
    if replay_log.stat().st_size <= 0:
        _fail("replay.log must be non-empty")
    timing = _load_json(paths["timing.json"], label="smoke timing")
    if set(timing) != {"started_at", "finished_at", "exit_code"}:
        _fail("timing.json must contain exactly started_at, finished_at, and exit_code")
    if (
        not isinstance(timing.get("exit_code"), int)
        or isinstance(timing.get("exit_code"), bool)
        or timing.get("exit_code") != 0
    ):
        _fail("timing.json must record integer exit_code=0")
    timing_started = _parse_utc(timing.get("started_at"), label="smoke timing started_at")
    timing_finished = _parse_utc(timing.get("finished_at"), label="smoke timing finished_at")
    if timing_finished < timing_started:
        _fail("smoke timing finished_at precedes started_at")

    _records, metrics = _validate_episode_records(paths)
    manifest_path = paths["manifests/libero-base.json"]
    manifest = _validate_manifest(
        manifest_path,
        paths,
        run_id=run_id,
        evaluator_source_commit=evaluator_source_commit,
        evaluator_image_digest=evaluator_image_digest,
        model_revision=model_revision,
        instance_id=instance_id,
        metrics=metrics,
        timing=timing,
    )
    cost_ledger = _validate_cost_ledger_file(
        cost_ledger_path,
        cost_ledger_descriptor,
        instance_id=instance_id,
        timing=timing,
        projected_cost=manifest["cost"]["projected_usd"],
    )
    files = tuple(_stable_file_record(paths[relative], root=root) for relative in sorted(paths))
    content = {
        "kind": "pi05-libero-eager-base-runtime-smoke-evidence",
        "run_id": run_id,
        "evaluator": {
            "source_commit": evaluator_source_commit,
            "image_digest": evaluator_image_digest,
        },
        "parent_policy": {
            "source_commit": parent_policy_source_commit,
            "image_digest": parent_policy_image_digest,
        },
        "model": {"revision": model_revision, "converted_checkpoint": checkpoint_evidence},
        "instance": validated_instance,
        "timing": dict(timing),
        "evaluation": dict(manifest["evaluation"]),
        "cost": {"evaluation": dict(manifest["cost"]), "ledger": cost_ledger},
        "files": list(files),
    }
    return SealedSmoke(
        root=root,
        revision=_canonical_hash(content),
        files=files,
        content=content,
        checkpoint_root=checkpoint_root,
        converted_manifest_path=converted_manifest_path,
        converted_checkpoint_artifact_path=converted_checkpoint_artifact_path,
        instance_identity=dict(instance_identity),
        cost_ledger_path=cost_ledger_path,
        cost_ledger=cost_ledger,
    )


def _parse_target(s3_root: str, sealed: SealedSmoke) -> repro_stage_data.S3Target:
    parsed = urllib.parse.urlsplit(s3_root)
    if parsed.scheme != "s3" or parsed.netloc != EXPECTED_BUCKET or parsed.query or parsed.fragment:
        _fail(f"--s3-root must use the project bucket {EXPECTED_BUCKET}: {s3_root!r}")
    root = parsed.path.strip("/")
    if (
        not root
        or root != "manual-smoke/libero"
        or "//" in parsed.path
        or any(part in {"", ".", ".."} for part in root.split("/"))
    ):
        _fail("--s3-root must be exactly s3://<project-bucket>/manual-smoke/libero")
    prefix = f"{root}/{sealed.content['run_id']}/{sealed.revision}"
    manifest_key = f"{prefix}/manifest.sha256.json"
    return repro_stage_data.S3Target(
        bucket=parsed.netloc,
        prefix=prefix,
        snapshot_uri=f"s3://{parsed.netloc}/{prefix}/output/",
        manifest_uri=f"s3://{parsed.netloc}/{manifest_key}",
        manifest_key=manifest_key,
    )


def _snapshot_smoke(sealed: SealedSmoke, destination: pathlib.Path) -> SealedSmoke:
    snapshot_root = destination / "output"
    before = {item["path"]: _file_stat_identity(sealed.root / item["path"]) for item in sealed.files}
    for record in sealed.files:
        source = sealed.root / record["path"]
        target = snapshot_root / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as source_stream, target.open("xb") as target_stream:
            shutil.copyfileobj(source_stream, target_stream, length=8 * 1024 * 1024)
            target_stream.flush()
            os.fsync(target_stream.fileno())
    after = {item["path"]: _file_stat_identity(sealed.root / item["path"]) for item in sealed.files}
    if before != after:
        _fail("smoke output changed while creating its publication snapshot")
    snapshot = validate_smoke(
        snapshot_root,
        run_id=str(sealed.content["run_id"]),
        evaluator_source_commit=str(sealed.content["evaluator"]["source_commit"]),
        evaluator_image_digest=str(sealed.content["evaluator"]["image_digest"]),
        parent_policy_source_commit=str(sealed.content["parent_policy"]["source_commit"]),
        parent_policy_image_digest=str(sealed.content["parent_policy"]["image_digest"]),
        model_revision=str(sealed.content["model"]["revision"]),
        instance_id=str(sealed.content["instance"]["id"]),
        instance_identity=sealed.instance_identity,
        checkpoint_root=sealed.checkpoint_root,
        converted_manifest_path=sealed.converted_manifest_path,
        converted_checkpoint_artifact_path=sealed.converted_checkpoint_artifact_path,
        cost_ledger_path=sealed.cost_ledger_path,
        cost_ledger_s3_uri=sealed.cost_ledger["s3_uri"],
        cost_ledger_version_id=sealed.cost_ledger["version_id"],
        cost_ledger_sha256=sealed.cost_ledger["sha256"],
    )
    if snapshot.revision != sealed.revision or snapshot.content != sealed.content or snapshot.files != sealed.files:
        _fail("publication snapshot differs from the validated smoke evidence")
    return snapshot


def _assert_no_multipart_uploads(
    target: repro_stage_data.S3Target,
    *,
    account: str,
    region: str,
    runner: CommandRunner,
) -> None:
    output = runner(
        [
            "aws",
            "s3api",
            "list-multipart-uploads",
            "--bucket",
            target.bucket,
            "--prefix",
            f"{target.prefix}/",
            "--expected-bucket-owner",
            account,
            "--region",
            region,
            "--output",
            "json",
        ]
    )
    try:
        result = json.loads(output)
    except json.JSONDecodeError as exc:
        raise repro_stage_data.StageError("multipart-upload listing did not return JSON") from exc
    if not isinstance(result, Mapping) or result.get("IsTruncated") is True or result.get("Uploads"):
        _fail("immutable LIBERO evidence prefix has incomplete or truncated multipart-upload state")


def _download_exact_input(
    *,
    s3_uri: str,
    version_id: str,
    sha256: str,
    destination: pathlib.Path,
    account: str,
    region: str,
    runner: CommandRunner,
) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(s3_uri)
    key = parsed.path.lstrip("/")
    if (
        parsed.scheme != "s3"
        or parsed.netloc != EXPECTED_BUCKET
        or not key
        or parsed.query
        or parsed.fragment
        or "//" in key
        or any(part in {"", ".", ".."} for part in pathlib.PurePosixPath(key).parts)
    ):
        _fail(f"immutable input is not an exact project S3 object URI: {s3_uri!r}")
    _single_line(version_id, label=f"S3 VersionId for {s3_uri}")
    if SHA256_RE.fullmatch(sha256) is None:
        _fail(f"immutable input has an invalid SHA-256: {s3_uri}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = converted._json_command(
        runner,
        [
            "aws",
            "s3api",
            "get-object",
            "--bucket",
            parsed.netloc,
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
            str(destination),
        ],
    )
    if result.get("VersionId") != version_id:
        _fail(f"immutable input GET returned a different VersionId: {s3_uri}")
    record = _stable_file_record(destination, root=destination.parent)
    if record["sha256"] != sha256:
        _fail(f"immutable input exact-version bytes differ from the pinned SHA-256: {s3_uri}")
    receipt = {
        "s3_uri": s3_uri,
        "version_id": version_id,
        "sha256": sha256,
        "bytes": record["bytes"],
    }
    destination.unlink()
    return receipt


def _verify_external_versions(
    sealed: SealedSmoke,
    *,
    temporary: pathlib.Path,
    account: str,
    region: str,
    runner: CommandRunner,
) -> dict[str, Any]:
    checkpoint = sealed.content["model"]["converted_checkpoint"]["artifact"]
    manifest = checkpoint["manifest"]
    manifest_receipt = _download_exact_input(
        s3_uri=str(manifest["s3_uri"]),
        version_id=str(manifest["version_id"]),
        sha256=str(manifest["sha256"]),
        destination=temporary / "external-inputs" / "converted-manifest.json",
        account=account,
        region=region,
        runner=runner,
    )
    payload_receipts: list[dict[str, Any]] = []
    payload_root = str(checkpoint["payload_s3_uri"])
    for index, item in enumerate(checkpoint["payload_objects"]):
        payload_receipts.append(
            _download_exact_input(
                s3_uri=f"{payload_root}{item['path']}",
                version_id=str(item["version_id"]),
                sha256=str(item["sha256"]),
                destination=temporary / "external-inputs" / f"checkpoint-{index}",
                account=account,
                region=region,
                runner=runner,
            )
        )
    ledger = sealed.cost_ledger
    ledger_receipt = _download_exact_input(
        s3_uri=ledger["s3_uri"],
        version_id=ledger["version_id"],
        sha256=ledger["sha256"],
        destination=temporary / "external-inputs" / "cost-ledger.json",
        account=account,
        region=region,
        runner=runner,
    )
    return {
        "converted_checkpoint": {"manifest": manifest_receipt, "payload": payload_receipts},
        "cost_ledger": ledger_receipt,
    }


def upload_smoke(
    config: Mapping[str, Any],
    sealed: SealedSmoke,
    *,
    s3_root: str,
    runner: CommandRunner = repro_stage_data.run_command,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    target = _parse_target(s3_root, sealed)
    account, region = repro_stage_data.verify_aws_destination(
        config,
        target,
        runner=runner,
        environ=os.environ if environ is None else environ,
    )
    if account != EXPECTED_ACCOUNT or region != EXPECTED_REGION:
        _fail("AWS destination differs from the project account or region")

    with tempfile.TemporaryDirectory(prefix="pi05-libero-smoke-publication-") as temporary_name:
        temporary = pathlib.Path(temporary_name).resolve(strict=True)
        snapshot = _snapshot_smoke(sealed, temporary)
        claim = {
            "schema_version": 1,
            "kind": "pi05-libero-smoke-publication-claim",
            "evidence_revision": snapshot.revision,
            "run_id": snapshot.content["run_id"],
            "evaluator_source_commit": snapshot.content["evaluator"]["source_commit"],
            "evaluator_image_digest": snapshot.content["evaluator"]["image_digest"],
            "parent_policy_source_commit": snapshot.content["parent_policy"]["source_commit"],
            "parent_policy_image_digest": snapshot.content["parent_policy"]["image_digest"],
            "model_revision": snapshot.content["model"]["revision"],
            "payload": list(snapshot.files),
        }
        claim_path = temporary / "publication-claim.json"
        claim_path.write_bytes(_canonical_bytes(claim, pretty=True))
        claim_sha256 = repro_stage_data.sha256_file(claim_path)
        claim_key = f"{target.prefix}/publication-claim.json"
        receipt_key = f"{target.prefix}/publication-receipt.json"
        payload_keys = {f"{target.prefix}/output/{item['path']}" for item in snapshot.files}
        allowed_keys = {claim_key, receipt_key, target.manifest_key, *payload_keys}
        common_metadata = {
            "source-provider": "pi05-libero-direct-evaluation",
            "source-revision": snapshot.revision,
            "source-commit": str(snapshot.content["evaluator"]["source_commit"]),
            "parent-source-commit": str(snapshot.content["parent_policy"]["source_commit"]),
            "parent-image-digest": str(snapshot.content["parent_policy"]["image_digest"]),
            "run-id": str(snapshot.content["run_id"]),
        }

        initial_history = converted._list_prefix_history(target, account=account, region=region, runner=runner)
        claim_versions, claim_markers = converted._key_versions(initial_history, claim_key)
        if (
            not claim_versions
            and not claim_markers
            and (initial_history.get("Versions") or initial_history.get("DeleteMarkers"))
        ):
            _fail("LIBERO evidence prefix has history without the exact publication claim")
        converted._assert_known_create_once_history(initial_history, allowed_keys)
        initial_keys = {str(item.get("Key")) for item in initial_history.get("Versions", [])}
        premanifest_keys = {claim_key, receipt_key, *payload_keys}
        if target.manifest_key in initial_keys and initial_keys != allowed_keys:
            _fail("terminal LIBERO manifest exists before the complete exact prefix")
        if receipt_key in initial_keys and frozenset(initial_keys) not in {
            frozenset(premanifest_keys),
            frozenset(allowed_keys),
        }:
            _fail("LIBERO publication receipt exists without its complete exact payload prefix")
        _assert_no_multipart_uploads(target, account=account, region=region, runner=runner)
        input_verification = _verify_external_versions(
            sealed,
            temporary=temporary,
            account=account,
            region=region,
            runner=runner,
        )

        claim_receipt = converted._publish_exact_object(
            path=claim_path,
            sha256=claim_sha256,
            metadata=common_metadata | {"role": "publication-claim", "sha256": claim_sha256},
            target=target,
            key=claim_key,
            account=account,
            region=region,
            temporary=temporary,
            runner=runner,
        )
        payload_receipts: list[dict[str, Any]] = []
        for record in snapshot.files:
            source = snapshot.root / record["path"]
            payload_receipts.append(
                converted._publish_exact_object(
                    path=source,
                    sha256=str(record["sha256"]),
                    metadata=common_metadata | {"role": "smoke-payload", "sha256": str(record["sha256"])},
                    target=target,
                    key=f"{target.prefix}/output/{record['path']}",
                    account=account,
                    region=region,
                    temporary=temporary,
                    runner=runner,
                )
            )

        publication_receipt = {
            "schema_version": 1,
            "kind": "pi05-libero-smoke-publication-receipt",
            "evidence_revision": snapshot.revision,
            "claim": claim_receipt,
            "payload": payload_receipts,
        }
        receipt_path = temporary / "publication-receipt.json"
        receipt_path.write_bytes(_canonical_bytes(publication_receipt, pretty=True))
        receipt_sha256 = repro_stage_data.sha256_file(receipt_path)
        receipt_receipt = converted._publish_exact_object(
            path=receipt_path,
            sha256=receipt_sha256,
            metadata=common_metadata | {"role": "publication-receipt", "sha256": receipt_sha256},
            target=target,
            key=receipt_key,
            account=account,
            region=region,
            temporary=temporary,
            runner=runner,
        )
        public_manifest = {
            "schema_version": 1,
            "evidence_revision": snapshot.revision,
            "content": snapshot.content,
            "storage": {
                "bucket": target.bucket,
                "region": region,
                "prefix": target.prefix,
                "claim": claim_receipt,
                "payload": payload_receipts,
                "receipt": receipt_receipt,
                "verified_inputs": input_verification,
            },
        }
        manifest_path = temporary / "manifest.sha256.json"
        manifest_path.write_bytes(_canonical_bytes(public_manifest, pretty=True))
        manifest_sha256 = repro_stage_data.sha256_file(manifest_path)
        manifest_receipt = converted._publish_exact_object(
            path=manifest_path,
            sha256=manifest_sha256,
            metadata=common_metadata | {"role": "manifest", "sha256": manifest_sha256},
            target=target,
            key=target.manifest_key,
            account=account,
            region=region,
            temporary=temporary,
            runner=runner,
        )
        final_history = converted._list_prefix_history(target, account=account, region=region, runner=runner)
        converted._assert_known_create_once_history(final_history, allowed_keys)
        final_keys = {str(item.get("Key")) for item in final_history.get("Versions", [])}
        if final_keys != allowed_keys or len(final_keys) != 11:
            _fail("LIBERO evidence publication does not contain exactly the eleven expected sole-version objects")
        _assert_no_multipart_uploads(target, account=account, region=region, runner=runner)

        source_after = validate_smoke(
            sealed.root,
            run_id=str(sealed.content["run_id"]),
            evaluator_source_commit=str(sealed.content["evaluator"]["source_commit"]),
            evaluator_image_digest=str(sealed.content["evaluator"]["image_digest"]),
            parent_policy_source_commit=str(sealed.content["parent_policy"]["source_commit"]),
            parent_policy_image_digest=str(sealed.content["parent_policy"]["image_digest"]),
            model_revision=str(sealed.content["model"]["revision"]),
            instance_id=str(sealed.content["instance"]["id"]),
            instance_identity=sealed.instance_identity,
            checkpoint_root=sealed.checkpoint_root,
            converted_manifest_path=sealed.converted_manifest_path,
            converted_checkpoint_artifact_path=sealed.converted_checkpoint_artifact_path,
            cost_ledger_path=sealed.cost_ledger_path,
            cost_ledger_s3_uri=sealed.cost_ledger["s3_uri"],
            cost_ledger_version_id=sealed.cost_ledger["version_id"],
            cost_ledger_sha256=sealed.cost_ledger["sha256"],
        )
        if (
            source_after.revision != sealed.revision
            or source_after.content != sealed.content
            or source_after.files != sealed.files
        ):
            _fail("source smoke output changed during publication")

    return {
        "evidence_revision": sealed.revision,
        "prefix_uri": f"s3://{target.bucket}/{target.prefix}/",
        "manifest": manifest_receipt,
        "publication": {
            "claim": claim_receipt,
            "payload": payload_receipts,
            "receipt": receipt_receipt,
            "manifest": manifest_receipt,
        },
        "verified_inputs": input_verification,
    }


def _public_validation(sealed: SealedSmoke) -> dict[str, Any]:
    return {"schema_version": 1, "evidence_revision": sealed.revision, "content": sealed.content}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "upload"))
    parser.add_argument("--output-root", required=True, type=pathlib.Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evaluator-source-commit", required=True)
    parser.add_argument("--evaluator-image-digest", required=True)
    parser.add_argument("--parent-policy-source-commit", required=True)
    parser.add_argument("--parent-policy-image-digest", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--checkpoint-root", required=True, type=pathlib.Path)
    parser.add_argument("--converted-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--converted-checkpoint-artifact", required=True, type=pathlib.Path)
    parser.add_argument("--cost-ledger-path", required=True, type=pathlib.Path)
    parser.add_argument("--cost-ledger-s3-uri", required=True)
    parser.add_argument("--cost-ledger-version-id", required=True)
    parser.add_argument("--cost-ledger-sha256", required=True)
    parser.add_argument("--s3-root")
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.action == "validate" and (args.execute or args.s3_root):
            _fail("validate does not accept --execute or --s3-root")
        if args.action == "upload" and not args.s3_root:
            _fail("upload requires --s3-root")
        try:
            instance_identity = repro_worker.get_instance_identity()
        except repro_worker.WorkerError as exc:
            raise repro_stage_data.StageError(f"fresh host identity verification failed: {exc}") from exc
        sealed = validate_smoke(
            args.output_root,
            run_id=args.run_id,
            evaluator_source_commit=args.evaluator_source_commit,
            evaluator_image_digest=args.evaluator_image_digest,
            parent_policy_source_commit=args.parent_policy_source_commit,
            parent_policy_image_digest=args.parent_policy_image_digest,
            model_revision=args.model_revision,
            instance_id=args.instance_id,
            instance_identity=instance_identity,
            checkpoint_root=args.checkpoint_root,
            converted_manifest_path=args.converted_manifest,
            converted_checkpoint_artifact_path=args.converted_checkpoint_artifact,
            cost_ledger_path=args.cost_ledger_path,
            cost_ledger_s3_uri=args.cost_ledger_s3_uri,
            cost_ledger_version_id=args.cost_ledger_version_id,
            cost_ledger_sha256=args.cost_ledger_sha256,
        )
        result: dict[str, Any] = {"validation": _public_validation(sealed)}
        if args.action == "upload" and not args.execute:
            target = _parse_target(args.s3_root, sealed)
            result |= {
                "mode": "dry-run",
                "destination": {
                    "prefix_uri": f"s3://{target.bucket}/{target.prefix}/",
                    "manifest_uri": target.manifest_uri,
                    "requires_execute": True,
                },
                "mutations_authorized": False,
            }
        elif args.action == "upload":
            result["s3"] = upload_smoke(
                repro_stage_data.load_json(args.config),
                sealed,
                s3_root=args.s3_root,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (repro_stage_data.StageError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"LIBERO SMOKE EVIDENCE STAGING REJECTED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
