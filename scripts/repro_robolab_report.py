#!/usr/bin/env python3
"""Seal native RoboLab results and turn a paired run into promotion evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import statistics
from typing import Any
import urllib.parse

if __package__:
    from scripts.repro_promotion_report import validate_evidence_provenance
else:
    from repro_promotion_report import validate_evidence_provenance


ROBOLAB_GIT_SHA = "0aef241fb088ca21bb4ebd24448940ed56620d17"
ROBOLAB_OPENPI_CLIENT_GIT_SHA = "aa6420561529593114160d05e5ad155792b272f3"
ISAAC_SIM_VERSION = "5.0.0"
ISAAC_LAB_VERSION = "2.2.0"
POLICY = "pi05"
INSTRUCTION_TYPE = "default"
OPEN_LOOP_HORIZON = 15
ENVIRONMENT_SEED = 1
EXPECTED_DT = 1.0 / 15.0
TASKS = ("BananaInBowlTask", "Stack3RubiksCubeTask")
EPISODES_PER_TASK = {"intermediate": 50, "final": 200}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IMAGE_DIGEST_RE = re.compile(r"(?:.+@)?sha256:[0-9a-f]{64}")
_STAGE_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_CONFIG_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_stage(value: Any, label: str = "stage") -> str:
    if not isinstance(value, str) or _STAGE_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must match {_STAGE_RE.pattern!r}")
    return value


def stage_identity(stage: str, model_sha256: str) -> str:
    return f"{_require_stage(stage)}-sha256:{_require_sha256(model_sha256, 'model hash')}"


def load_results(path: pathlib.Path) -> list[dict[str, Any]]:
    """Load native ``episode_results.jsonl`` or its legacy JSON-list form strictly."""
    if not path.is_file():
        raise ValueError(f"RoboLab result file does not exist: {path}")
    if path.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed RoboLab JSONL line {line_number} in {path}") from error
            if not isinstance(record, dict):
                raise ValueError(f"RoboLab JSONL line {line_number} is not an object")
            records.append(record)
        return records

    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed RoboLab JSON file: {path}") from error
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("Legacy RoboLab result JSON must be a list of episode objects")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _validate_native_records(
    records: list[dict[str, Any]],
    *,
    mode: str,
    num_envs: int,
    num_runs: int,
    require_complete: bool,
) -> dict[tuple[str, int], dict[str, Any]]:
    if mode not in EPISODES_PER_TASK:
        raise ValueError(f"Unknown RoboLab evaluation mode: {mode}")
    if (
        isinstance(num_envs, bool)
        or isinstance(num_runs, bool)
        or not isinstance(num_envs, int)
        or not isinstance(num_runs, int)
        or num_envs <= 0
        or num_runs <= 0
    ):
        raise ValueError("num_envs and num_runs must be positive integers")
    episodes_per_task = EPISODES_PER_TASK[mode]
    if num_envs * num_runs != episodes_per_task:
        raise ValueError(
            f"{mode} requires exactly {episodes_per_task} episodes/task; num_envs*num_runs is {num_envs * num_runs}"
        )
    expected_total = episodes_per_task * len(TASKS)
    if require_complete and len(records) != expected_total:
        raise ValueError(f"RoboLab results contain {len(records)} episodes; exactly {expected_total} are required")
    if not require_complete and len(records) > expected_total:
        raise ValueError(f"RoboLab partial results exceed the exact {expected_total}-episode contract")

    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for record_index, record in enumerate(records):
        task = record.get("task_name")
        if task not in TASKS:
            raise ValueError(f"Episode {record_index} has unexpected task_name {task!r}")
        if record.get("env_name") != task:
            raise ValueError(f"Episode {record_index} env_name must equal pinned task_name {task!r}")
        if record.get("policy") != POLICY:
            raise ValueError(f"Episode {record_index} policy must be {POLICY!r}")
        if record.get("instruction_type") != INSTRUCTION_TYPE:
            raise ValueError(f"Episode {record_index} instruction_type must be {INSTRUCTION_TYPE!r}")
        if not isinstance(record.get("instruction"), str) or not record["instruction"].strip():
            raise ValueError(f"Episode {record_index} is missing its instruction")
        if not isinstance(record.get("success"), bool):
            raise ValueError(f"Episode {record_index} success must be a JSON boolean")

        episode = record.get("episode")
        run = record.get("run")
        env_id = record.get("env_id")
        for value, name in ((episode, "episode"), (run, "run"), (env_id, "env_id")):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Episode {record_index} {name} must be an integer")
        if not 0 <= run < num_runs or not 0 <= env_id < num_envs:
            raise ValueError(f"Episode {record_index} has out-of-range run/env_id")
        if episode != run * num_envs + env_id:
            raise ValueError(f"Episode {record_index} does not satisfy episode=run*num_envs+env_id")
        if record.get("run_name") != f"{task}_{run}":
            raise ValueError(f"Episode {record_index} has a non-default run_name")

        dt = _finite_number(record.get("dt"), f"episode {record_index} dt")
        if not math.isclose(dt, EXPECTED_DT, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"Episode {record_index} dt is {dt}, expected {EXPECTED_DT}")
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"Episode {record_index} has no metrics object")
        for metric in ("ee_path_length", "ee_sparc"):
            _finite_number(metrics.get(metric), f"episode {record_index} metrics.{metric}")

        key = (task, episode)
        if key in indexed:
            raise ValueError(f"Duplicate RoboLab episode identity: {key}")
        indexed[key] = record

    expected_keys = {(task, episode) for task in TASKS for episode in range(episodes_per_task)}
    missing = sorted(expected_keys - indexed.keys())
    extra = sorted(indexed.keys() - expected_keys)
    if extra or (require_complete and missing):
        raise ValueError(f"RoboLab episode identities are not exact: missing={missing[:5]}, extra={extra[:5]}")
    return indexed


def validate_native_results(
    records: list[dict[str, Any]],
    *,
    mode: str,
    num_envs: int,
    num_runs: int,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Validate the exact schema/count contract emitted by pinned RoboLab."""

    return _validate_native_records(
        records,
        mode=mode,
        num_envs=num_envs,
        num_runs=num_runs,
        require_complete=True,
    )


def complete_native_run_prefix(
    records: list[dict[str, Any]],
    *,
    mode: str,
    num_envs: int,
    num_runs: int,
) -> list[dict[str, Any]]:
    """Return only the strict prefix of complete RoboLab run batches.

    A process can be interrupted while its ten JSONL records are being
    appended. RoboLab reruns an incomplete batch, so only whole, ordered
    ``(task, run)`` groups are safe continuation state.
    """

    indexed = _validate_native_records(
        records,
        mode=mode,
        num_envs=num_envs,
        num_runs=num_runs,
        require_complete=False,
    )
    expected_groups = [(task, run) for task in TASKS for run in range(num_runs)]
    observed_by_group: dict[tuple[str, int], set[int]] = {}
    for (task, _episode), record in indexed.items():
        observed_by_group.setdefault((task, int(record["run"])), set()).add(int(record["env_id"]))
    expected_envs = set(range(num_envs))
    complete_groups = {group for group, env_ids in observed_by_group.items() if env_ids == expected_envs}
    prefix_length = 0
    while prefix_length < len(expected_groups) and expected_groups[prefix_length] in complete_groups:
        prefix_length += 1
    prefix_groups = set(expected_groups[:prefix_length])
    if complete_groups != prefix_groups:
        raise ValueError("RoboLab partial results contain a complete run beyond the resumable prefix")
    incomplete_groups = set(observed_by_group) - complete_groups
    permitted_incomplete = {expected_groups[prefix_length]} if prefix_length < len(expected_groups) else set()
    if incomplete_groups - permitted_incomplete:
        raise ValueError("RoboLab partial results contain out-of-order incomplete run batches")
    return [
        indexed[(task, run * num_envs + env_id)]
        for task, run in expected_groups[:prefix_length]
        for env_id in range(num_envs)
    ]


def validate_native_continuation(
    records: list[dict[str, Any]],
    *,
    mode: str,
    num_envs: int,
    num_runs: int,
) -> dict[tuple[str, int], dict[str, Any]]:
    complete = complete_native_run_prefix(records, mode=mode, num_envs=num_envs, num_runs=num_runs)
    if len(complete) != len(records):
        raise ValueError("RoboLab continuation contains an incomplete run batch")
    return {(str(record["task_name"]), int(record["episode"])): record for record in complete}


def create_run_identity(
    *,
    stage: str,
    mode: str,
    checkpoint_model: pathlib.Path,
    results: pathlib.Path,
    output: pathlib.Path,
    num_envs: int,
    num_runs: int,
    policy_server_seed: int,
    image_digest: str,
    robolab_git_sha: str,
    policy_image_digest: str,
    policy_source_s3_uri: str,
    policy_source_version_id: str,
    policy_source_sha256: str,
    policy_source_commit: str,
    policy_config: str,
    policy_command_sha256: str,
    checkpoint_model_identity_path: str | None = None,
) -> dict[str, Any]:
    """Create a portable sidecar that binds a model hash to one native result file."""
    stage = _require_stage(stage)
    if robolab_git_sha != ROBOLAB_GIT_SHA:
        raise ValueError(f"RoboLab must be pinned to {ROBOLAB_GIT_SHA}")
    if _IMAGE_DIGEST_RE.fullmatch(image_digest) is None:
        raise ValueError("image_digest must be an immutable sha256 digest or repository@digest")
    if _IMAGE_DIGEST_RE.fullmatch(policy_image_digest) is None:
        raise ValueError("policy_image_digest must be an immutable sha256 digest or repository@digest")
    if not isinstance(policy_source_commit, str) or _COMMIT_RE.fullmatch(policy_source_commit) is None:
        raise ValueError("policy_source_commit must be a full lowercase Git SHA")
    source_uri = urllib.parse.urlsplit(policy_source_s3_uri)
    if (
        source_uri.scheme != "s3"
        or not source_uri.netloc
        or source_uri.path.lstrip("/") != f"source/openpi-{policy_source_commit}-complete.bundle"
        or source_uri.query
        or source_uri.fragment
    ):
        raise ValueError("policy_source_s3_uri must identify the commit-qualified complete source bundle")
    if not policy_source_version_id or any(character in policy_source_version_id for character in "\x00\r\n"):
        raise ValueError("policy_source_version_id must be a non-empty immutable version")
    _require_sha256(policy_source_sha256, "policy source hash")
    if not isinstance(policy_config, str) or _CONFIG_RE.fullmatch(policy_config) is None:
        raise ValueError("policy_config is invalid")
    _require_sha256(policy_command_sha256, "policy command hash")
    if isinstance(policy_server_seed, bool) or not isinstance(policy_server_seed, int) or policy_server_seed < 0:
        raise ValueError("policy_server_seed must be a non-negative integer")
    checkpoint_model = checkpoint_model.expanduser().resolve()
    results = results.expanduser().resolve()
    output = output.expanduser().resolve()
    if not checkpoint_model.is_file():
        raise ValueError(f"Checkpoint model file does not exist: {checkpoint_model}")
    records = load_results(results)
    validate_native_results(records, mode=mode, num_envs=num_envs, num_runs=num_runs)
    model_hash = sha256_file(checkpoint_model)
    try:
        result_path = str(results.relative_to(output.parent))
    except ValueError:
        result_path = str(results)
    return {
        "schema_version": 1,
        "benchmark": "robolab",
        "stage": stage,
        "stage_identity": stage_identity(stage, model_hash),
        "checkpoint": {
            "model_path": checkpoint_model_identity_path or str(checkpoint_model),
            "model_sha256": model_hash,
        },
        "results": {"path": result_path, "sha256": sha256_file(results)},
        "runtime": {
            "image_digest": image_digest,
            "robolab_git_sha": ROBOLAB_GIT_SHA,
            "openpi_client_git_sha": ROBOLAB_OPENPI_CLIENT_GIT_SHA,
            "isaac_sim_version": ISAAC_SIM_VERSION,
            "isaac_lab_version": ISAAC_LAB_VERSION,
        },
        "policy_server": {
            "image_digest": policy_image_digest,
            "source": {
                "s3_uri": policy_source_s3_uri,
                "version_id": policy_source_version_id,
                "sha256": policy_source_sha256,
                "commit": policy_source_commit,
            },
            "config": policy_config,
            "command_sha256": policy_command_sha256,
            "checkpoint_model_sha256": model_hash,
        },
        "evaluation": {
            "mode": mode,
            "tasks": list(TASKS),
            "episodes_per_task": EPISODES_PER_TASK[mode],
            "num_envs": num_envs,
            "num_runs": num_runs,
            "policy": POLICY,
            "policy_server_seed": policy_server_seed,
            "environment_seed": ENVIRONMENT_SEED,
            "instruction_type": INSTRUCTION_TYPE,
            "open_loop_horizon": OPEN_LOOP_HORIZON,
        },
    }


def _load_identity(path: pathlib.Path) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    path = path.expanduser().resolve()
    try:
        identity = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read RoboLab identity: {path}") from error
    if not isinstance(identity, dict) or identity.get("schema_version") != 1 or identity.get("benchmark") != "robolab":
        raise ValueError(f"Invalid RoboLab identity schema: {path}")
    checkpoint = identity.get("checkpoint")
    results = identity.get("results")
    runtime = identity.get("runtime")
    policy_server = identity.get("policy_server")
    evaluation = identity.get("evaluation")
    if not all(isinstance(value, dict) for value in (checkpoint, results, runtime, policy_server, evaluation)):
        raise ValueError(f"RoboLab identity {path} is missing required objects")

    model_hash = _require_sha256(checkpoint.get("model_sha256"), "checkpoint model hash")
    stage = _require_stage(identity.get("stage"))
    if identity.get("stage_identity") != stage_identity(stage, model_hash):
        raise ValueError(f"RoboLab identity {path} has a mismatched stage_identity")
    expected_runtime = {
        "robolab_git_sha": ROBOLAB_GIT_SHA,
        "openpi_client_git_sha": ROBOLAB_OPENPI_CLIENT_GIT_SHA,
        "isaac_sim_version": ISAAC_SIM_VERSION,
        "isaac_lab_version": ISAAC_LAB_VERSION,
    }
    for key, value in expected_runtime.items():
        if runtime.get(key) != value:
            raise ValueError(f"RoboLab identity {path} has unpinned runtime {key}")
    if _IMAGE_DIGEST_RE.fullmatch(str(runtime.get("image_digest", ""))) is None:
        raise ValueError(f"RoboLab identity {path} has no immutable image digest")

    if set(policy_server) != {
        "image_digest",
        "source",
        "config",
        "command_sha256",
        "checkpoint_model_sha256",
    }:
        raise ValueError(f"RoboLab identity {path} has an incomplete policy-server identity")
    if _IMAGE_DIGEST_RE.fullmatch(str(policy_server.get("image_digest", ""))) is None:
        raise ValueError(f"RoboLab identity {path} has no immutable policy image digest")
    source = policy_server.get("source")
    if not isinstance(source, dict) or set(source) != {"s3_uri", "version_id", "sha256", "commit"}:
        raise ValueError(f"RoboLab identity {path} has an incomplete policy source identity")
    if not isinstance(source.get("commit"), str) or _COMMIT_RE.fullmatch(source["commit"]) is None:
        raise ValueError(f"RoboLab identity {path} has an invalid policy source commit")
    source_uri = urllib.parse.urlsplit(str(source.get("s3_uri", "")))
    if (
        source_uri.scheme != "s3"
        or not source_uri.netloc
        or source_uri.path.lstrip("/") != f"source/openpi-{source['commit']}-complete.bundle"
        or source_uri.query
        or source_uri.fragment
    ):
        raise ValueError(f"RoboLab identity {path} has an invalid policy source URI")
    if (
        not isinstance(source.get("version_id"), str)
        or not source["version_id"]
        or any(character in source["version_id"] for character in "\x00\r\n")
    ):
        raise ValueError(f"RoboLab identity {path} has an invalid policy source version")
    _require_sha256(source.get("sha256"), "policy source hash")
    if not isinstance(policy_server.get("config"), str) or _CONFIG_RE.fullmatch(policy_server["config"]) is None:
        raise ValueError(f"RoboLab identity {path} has an invalid policy config")
    _require_sha256(policy_server.get("command_sha256"), "policy command hash")
    if _require_sha256(policy_server.get("checkpoint_model_sha256"), "policy checkpoint hash") != model_hash:
        raise ValueError(f"RoboLab identity {path} policy checkpoint differs from its sealed model")

    mode = evaluation.get("mode")
    if mode not in EPISODES_PER_TASK:
        raise ValueError(f"RoboLab identity {path} has invalid evaluation mode")
    expected_evaluation = {
        "tasks": list(TASKS),
        "episodes_per_task": EPISODES_PER_TASK[mode],
        "policy": POLICY,
        "environment_seed": ENVIRONMENT_SEED,
        "instruction_type": INSTRUCTION_TYPE,
        "open_loop_horizon": OPEN_LOOP_HORIZON,
    }
    for key, value in expected_evaluation.items():
        if evaluation.get(key) != value:
            raise ValueError(f"RoboLab identity {path} has unexpected evaluation field {key}")
    policy_seed = evaluation.get("policy_server_seed")
    if isinstance(policy_seed, bool) or not isinstance(policy_seed, int) or policy_seed < 0:
        raise ValueError(f"RoboLab identity {path} has an invalid policy_server_seed")

    result_path = pathlib.Path(results.get("path", ""))
    if not result_path.is_absolute():
        result_path = path.parent / result_path
    result_path = result_path.resolve()
    if not result_path.is_file():
        raise ValueError(f"RoboLab result bound by {path} does not exist: {result_path}")
    if sha256_file(result_path) != _require_sha256(results.get("sha256"), "result hash"):
        raise ValueError(f"RoboLab result hash does not match identity: {result_path}")
    indexed = validate_native_results(
        load_results(result_path),
        mode=mode,
        num_envs=evaluation.get("num_envs"),
        num_runs=evaluation.get("num_runs"),
    )
    return identity, indexed


def _metric_summary(records: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = [float(record["metrics"][metric]) for record in records]
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "stddev": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def build_report(
    *,
    reference_identity_path: pathlib.Path,
    candidate_identity_path: pathlib.Path,
    offline_report: dict[str, Any],
    expected_reference_stage: str,
    expected_candidate_stage: str,
    reference_model_sha256: str | None = None,
    denoise_speedup: float | None = None,
    max_task_success_gap: float = 0.05,
) -> dict[str, Any]:
    """Build provenance-complete paired RoboLab evidence for the promotion gate."""
    validate_evidence_provenance([offline_report], [])
    reference_identity_path = reference_identity_path.expanduser().resolve()
    candidate_identity_path = candidate_identity_path.expanduser().resolve()
    reference, reference_records = _load_identity(reference_identity_path)
    candidate, candidate_records = _load_identity(candidate_identity_path)
    expected_reference_stage = _require_stage(expected_reference_stage, "expected reference stage")
    expected_candidate_stage = _require_stage(expected_candidate_stage, "expected candidate stage")
    if expected_reference_stage == expected_candidate_stage:
        raise ValueError("Reference and candidate RoboLab stages must be different")
    if reference["stage"] != expected_reference_stage or candidate["stage"] != expected_candidate_stage:
        raise ValueError(
            "RoboLab stage mismatch: "
            f"expected {expected_reference_stage!r}/{expected_candidate_stage!r}, "
            f"got {reference['stage']!r}/{candidate['stage']!r}"
        )

    provenance = offline_report["provenance"]
    expected_candidate_hash = provenance["student_checkpoint"]["model_sha256"]
    if candidate["checkpoint"]["model_sha256"] != expected_candidate_hash:
        raise ValueError("RoboLab candidate model does not match offline student checkpoint")
    if candidate["policy_server"]["config"] != provenance["student_config"]["name"]:
        raise ValueError("RoboLab candidate policy config does not match offline student config")
    if reference_model_sha256 is None:
        expected_reference_hash = provenance["teacher_checkpoint"]["model_sha256"]
        reference_source = "offline_teacher"
    else:
        expected_reference_hash = _require_sha256(reference_model_sha256, "reference model hash")
        reference_source = "explicit_base"
    if reference["checkpoint"]["model_sha256"] != expected_reference_hash:
        raise ValueError("RoboLab reference model does not match the required checkpoint")
    if reference_model_sha256 is None and reference["policy_server"]["config"] != provenance["teacher_config"]["name"]:
        raise ValueError("RoboLab reference policy config does not match offline teacher config")

    if reference["runtime"] != candidate["runtime"]:
        raise ValueError("Reference and candidate RoboLab runtime pins differ")
    if reference["evaluation"] != candidate["evaluation"]:
        raise ValueError("Reference and candidate RoboLab evaluation inputs differ")
    if reference_records.keys() != candidate_records.keys():
        raise ValueError("Reference and candidate RoboLab episode identities differ")
    if not math.isfinite(max_task_success_gap) or not 0 <= max_task_success_gap <= 1:
        raise ValueError("max_task_success_gap must be finite and in [0, 1]")
    if denoise_speedup is not None and (not math.isfinite(denoise_speedup) or denoise_speedup <= 0):
        raise ValueError("denoise_speedup must be finite and positive")

    task_evidence: dict[str, Any] = {}
    checks: list[dict[str, Any]] = []
    reference_successes = 0
    candidate_successes = 0
    paired_path_deltas: list[float] = []
    paired_sparc_deltas: list[float] = []
    for task in TASKS:
        keys = sorted((key for key in reference_records if key[0] == task), key=lambda key: key[1])
        reference_task = [reference_records[key] for key in keys]
        candidate_task = [candidate_records[key] for key in keys]
        for index, (reference_episode, candidate_episode) in enumerate(
            zip(reference_task, candidate_task, strict=True)
        ):
            for field in ("env_name", "task_name", "episode", "run", "env_id", "instruction", "dt"):
                if reference_episode.get(field) != candidate_episode.get(field):
                    raise ValueError(f"Paired RoboLab {task} episode {index} differs in {field}")
            paired_path_deltas.append(
                float(candidate_episode["metrics"]["ee_path_length"])
                - float(reference_episode["metrics"]["ee_path_length"])
            )
            paired_sparc_deltas.append(
                float(candidate_episode["metrics"]["ee_sparc"]) - float(reference_episode["metrics"]["ee_sparc"])
            )
        reference_task_successes = sum(record["success"] for record in reference_task)
        candidate_task_successes = sum(record["success"] for record in candidate_task)
        reference_successes += reference_task_successes
        candidate_successes += candidate_task_successes
        count = len(keys)
        reference_rate = reference_task_successes / count
        candidate_rate = candidate_task_successes / count
        success_gap = reference_rate - candidate_rate
        check = {
            "name": f"{task}_success_noninferiority",
            "value": success_gap,
            "maximum": max_task_success_gap,
            "passed": success_gap <= max_task_success_gap,
        }
        checks.append(check)
        task_evidence[task] = {
            "episodes": count,
            "reference_successes": reference_task_successes,
            "candidate_successes": candidate_task_successes,
            "reference_success": reference_rate,
            "candidate_success": candidate_rate,
            "success_gap": success_gap,
            "success_difference_points": (candidate_rate - reference_rate) * 100.0,
            "reference_ee_path_length": _metric_summary(reference_task, "ee_path_length"),
            "candidate_ee_path_length": _metric_summary(candidate_task, "ee_path_length"),
            "paired_ee_path_length_delta_mean": statistics.fmean(
                float(candidate_episode["metrics"]["ee_path_length"])
                - float(reference_episode["metrics"]["ee_path_length"])
                for reference_episode, candidate_episode in zip(reference_task, candidate_task, strict=True)
            ),
            "reference_ee_sparc": _metric_summary(reference_task, "ee_sparc"),
            "candidate_ee_sparc": _metric_summary(candidate_task, "ee_sparc"),
            "paired_ee_sparc_delta_mean": statistics.fmean(
                float(candidate_episode["metrics"]["ee_sparc"]) - float(reference_episode["metrics"]["ee_sparc"])
                for reference_episode, candidate_episode in zip(reference_task, candidate_task, strict=True)
            ),
        }

    complete_pairs = len(reference_records)
    aggregate_reference = reference_successes / complete_pairs
    aggregate_candidate = candidate_successes / complete_pairs
    gate_passed = all(check["passed"] for check in checks)
    observed_paired_rollout = {
        "student_success": aggregate_candidate,
        "reference_success": aggregate_reference,
        "complete_pairs": complete_pairs,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "robolab",
        "checkpoint_step": offline_report["student_step"],
        "provenance": provenance,
        "observed_paired_rollout": observed_paired_rollout,
        "model_identity": {
            "reference_stage": reference["stage_identity"],
            "candidate_stage": candidate["stage_identity"],
            "reference_source": reference_source,
        },
        "evaluation": reference["evaluation"],
        "runtime": reference["runtime"],
        "policy_server": {
            "reference": reference["policy_server"],
            "candidate": candidate["policy_server"],
        },
        "task_evidence": task_evidence,
        "aggregate_motion_evidence": {
            "paired_ee_path_length_delta_mean": statistics.fmean(paired_path_deltas),
            "paired_ee_sparc_delta_mean": statistics.fmean(paired_sparc_deltas),
        },
        "evaluation_gate": {
            "passed": gate_passed,
            "max_task_success_gap": max_task_success_gap,
            "checks": checks,
        },
        "sources": {
            "reference_identity": {
                "path": str(reference_identity_path),
                "sha256": sha256_file(reference_identity_path),
                "results": reference["results"],
            },
            "candidate_identity": {
                "path": str(candidate_identity_path),
                "sha256": sha256_file(candidate_identity_path),
                "results": candidate["results"],
            },
        },
    }
    # The existing promotion report treats a missing paired_rollout as missing
    # required evidence. Keep the observations above for diagnosis, but expose
    # the promotion field only after every per-task non-inferiority check passes.
    if gate_passed:
        report["paired_rollout"] = observed_paired_rollout
    if denoise_speedup is not None:
        report["denoise_speedup"] = denoise_speedup
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    seal = subparsers.add_parser("seal", help="bind one native result file to a checkpoint and runtime")
    seal.add_argument("--stage", required=True)
    seal.add_argument("--mode", choices=tuple(EPISODES_PER_TASK), required=True)
    seal.add_argument("--checkpoint-model", type=pathlib.Path, required=True)
    seal.add_argument("--results", type=pathlib.Path, required=True)
    seal.add_argument("--output", type=pathlib.Path, required=True)
    seal.add_argument("--num-envs", type=int, required=True)
    seal.add_argument("--num-runs", type=int, required=True)
    seal.add_argument("--policy-server-seed", type=int, required=True)
    seal.add_argument("--image-digest", required=True)
    seal.add_argument("--robolab-git-sha", required=True)
    seal.add_argument("--policy-image-digest", required=True)
    seal.add_argument("--policy-source-s3-uri", required=True)
    seal.add_argument("--policy-source-version-id", required=True)
    seal.add_argument("--policy-source-sha256", required=True)
    seal.add_argument("--policy-source-commit", required=True)
    seal.add_argument("--policy-config", required=True)
    seal.add_argument("--policy-command-sha256", required=True)

    report = subparsers.add_parser("report", help="compare two sealed runs and emit promotion evidence")
    report.add_argument("--reference-identity", type=pathlib.Path, required=True)
    report.add_argument("--candidate-identity", type=pathlib.Path, required=True)
    report.add_argument("--offline-report", type=pathlib.Path, required=True)
    report.add_argument("--expected-reference-stage", required=True)
    report.add_argument("--expected-candidate-stage", required=True)
    report.add_argument("--reference-model-sha256")
    report.add_argument("--denoise-speedup", type=float)
    report.add_argument("--max-task-success-gap", type=float, default=0.05)
    report.add_argument("--output", type=pathlib.Path, required=True)
    report.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "seal":
        identity = create_run_identity(
            stage=args.stage,
            mode=args.mode,
            checkpoint_model=args.checkpoint_model,
            results=args.results,
            output=args.output,
            num_envs=args.num_envs,
            num_runs=args.num_runs,
            policy_server_seed=args.policy_server_seed,
            image_digest=args.image_digest,
            robolab_git_sha=args.robolab_git_sha,
            policy_image_digest=args.policy_image_digest,
            policy_source_s3_uri=args.policy_source_s3_uri,
            policy_source_version_id=args.policy_source_version_id,
            policy_source_sha256=args.policy_source_sha256,
            policy_source_commit=args.policy_source_commit,
            policy_config=args.policy_config,
            policy_command_sha256=args.policy_command_sha256,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
        print(json.dumps(identity, indent=2, sort_keys=True))
        return 0

    offline_path = args.offline_report.expanduser().resolve()
    report = build_report(
        reference_identity_path=args.reference_identity,
        candidate_identity_path=args.candidate_identity,
        offline_report=json.loads(offline_path.read_text()),
        expected_reference_stage=args.expected_reference_stage,
        expected_candidate_stage=args.expected_candidate_stage,
        reference_model_sha256=args.reference_model_sha256,
        denoise_speedup=args.denoise_speedup,
        max_task_success_gap=args.max_task_success_gap,
    )
    report["sources"]["offline_report"] = {
        "path": str(offline_path),
        "sha256": sha256_file(offline_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if args.report_only or report["evaluation_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
