#!/usr/bin/env python3
"""Run one provenance-complete, two-container RoboLab base evaluation worker.

The generic reproduction worker intentionally owns a single, network-isolated
container.  RoboLab needs a policy-server container and a separately pinned
Isaac Sim evaluator container at the same time, so this module provides the
smallest dedicated orchestration boundary without weakening the generic one.

Planning is the default.  ``run --execute`` is the only path that stages S3
objects, initializes scratch storage, pulls images, starts containers, or
publishes evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
import contextlib
import copy
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

if __package__:
    from scripts import repro_checkout_permissions
    from scripts import repro_robolab_report
    from scripts import repro_worker
else:
    import repro_checkout_permissions
    import repro_robolab_report
    import repro_worker


UTC = getattr(dt, "UTC", dt.timezone.utc)  # noqa: UP017 -- supports the pinned Ubuntu 22.04 host Python.

PROJECT = "pi05-aws-repro"
ACCOUNT_ID = "752160877725"
REGION = "us-east-2"
BUCKET = "pi05-repro-752160877725-us-east-2"

MODEL_SOURCE_COMMIT = "229c08ea2a13a70cbbf1a9c8a1f31cb1ca674dee"
POLICY_IMAGE = {
    "uri": (
        "752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro"
        "@sha256:2afcc58cda27681892c7bbb9554e9603024c5b74f53358fad893ea876374803c"
    ),
    "digest": "sha256:2afcc58cda27681892c7bbb9554e9603024c5b74f53358fad893ea876374803c",
    "purpose": "policy",
    "lerobot_runtime": "v3",
    "lerobot_revision": "0b067df57d21d3a02d6c511f1609172fa39ac29b",
}

ROBOLAB_GIT_SHA = "0aef241fb088ca21bb4ebd24448940ed56620d17"
ROBOLAB_CLIENT_GIT_SHA = "aa6420561529593114160d05e5ad155792b272f3"
ISAACLAB_BASE_DIGEST = "sha256:b4d8e96cbfb9a6c40067bec6cc5ee180e36d4c0164b25f7215c5f47e31897b94"
EVALUATOR_IMAGE = {
    "uri": (
        "752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro"
        "@sha256:2d17c15e62887c9fc8b4c41b7ee3d39c4c187348eb55b4273fd24e785a3325e7"
    ),
    "digest": "sha256:2d17c15e62887c9fc8b4c41b7ee3d39c4c187348eb55b4273fd24e785a3325e7",
    "robolab_revision": ROBOLAB_GIT_SHA,
    "openpi_client_revision": ROBOLAB_CLIENT_GIT_SHA,
    "isaaclab_base_digest": ISAACLAB_BASE_DIGEST,
    "isaac_lab_version": "2.2.0",
    "isaac_sim_version_prefix": "5.0.0-",
    "typeguard_version": "4.4.2",
    "typing_extensions_version": "4.12.2",
}

EVALUATION_AMI_ID = "ami-06517bc7fad3c6a48"
EVALUATION_DRIVER_VERSION = "580.126.09"
INSTANCE_TYPE = "g6e.4xlarge"
POLICY_SERVER_SEED = 7003
ENVIRONMENT_SEED = 1
TASKS = ("BananaInBowlTask", "Stack3RubiksCubeTask")
NUM_ENVS = 10
NUM_RUNS = 5
EPISODES_PER_TASK = NUM_ENVS * NUM_RUNS
OUTPUT_FOLDER_NAME = "base-intermediate"

JAX_TEACHER = {
    "name": "droid_teacher_jax",
    "kind": "checkpoint",
    "revision": "6487c08461e26cac570a2781f477474e6573c7a6e0a4ba93a9f0efb146c2db5b",
    "manifest": {
        "s3_uri": (
            f"s3://{BUCKET}/checkpoints/pi05_droid_jointpos/"
            "6487c08461e26cac570a2781f477474e6573c7a6e0a4ba93a9f0efb146c2db5b/manifest.sha256.json"
        ),
        "version_id": "etUbiXvb8B6C7ltGXEfmrB96kJxI18HC",
        "sha256": "64e4082767ac652d35828f721ca0906bd9a97f78a769a4bf4f75b09837d5bf46",
    },
    "payload_s3_uri": (
        f"s3://{BUCKET}/checkpoints/pi05_droid_jointpos/"
        "6487c08461e26cac570a2781f477474e6573c7a6e0a4ba93a9f0efb146c2db5b/checkpoint/"
    ),
    "destination": "pi05_droid_jointpos",
}
PYTORCH_TEACHER = {
    "name": "droid_teacher_pytorch",
    "kind": "checkpoint",
    "revision": "b4e9dcd2767b497b707d912b708729a9edd5c91bcbf402f542cd682b32c943b7",
    "manifest": {
        "s3_uri": (
            f"s3://{BUCKET}/checkpoints/pi05_droid_jointpos_pytorch/"
            "b4e9dcd2767b497b707d912b708729a9edd5c91bcbf402f542cd682b32c943b7/manifest.sha256.json"
        ),
        "version_id": "xgmhHet70zLej9LpoJ9bWROwVYvFj5ow",
        "sha256": "ed611e4814897b84b0f138e76eed3b9caaa05ec108632a993652a69910d0b78f",
    },
    "payload_s3_uri": (
        f"s3://{BUCKET}/checkpoints/pi05_droid_jointpos_pytorch/"
        "b4e9dcd2767b497b707d912b708729a9edd5c91bcbf402f542cd682b32c943b7/checkpoint/"
    ),
    "payload_objects": [
        {
            "path": "assets/droid/norm_stats.json",
            "version_id": "fr9dSF.rlbM4swMPpwqzHNWOuWnQZIN_",
            "sha256": "57ce9956f9e07d65f8a8205aabec72d436a2c8927f53edb40c7a77b14a5a90c7",
        },
        {
            "path": "assets/physical-intelligence/droid.lock",
            "version_id": "OlJuBIA6Pv9y_lVWsI1aWaPXoF9uqhzN",
            "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        },
        {
            "path": "config.json",
            "version_id": "zG8hGH6Hy2jwU8F.9Ld2st5t76HOUELc",
            "sha256": "655293290a8d6921bfd83e0c8c4be093cbf9f9b05b4c3bc99a296a8af7ebdc6e",
        },
        {
            "path": "model.safetensors",
            "version_id": ".mJzDLYOwUQvdlE9ORGR5T8OEsPqMuKW",
            "sha256": "3212bbd9737caf175ba238193a9e1e3b7b16a4c5d1c4b586ad3d65d58deb5117",
        },
    ],
    "destination": "pi05_droid_jointpos_pytorch",
}

EXPECTED_OUTPUTS = [
    {"name": "robolab_evidence", "kind": "artifact", "path": "artifacts/robolab"},
    {"name": "policy_server_log", "kind": "log", "path": "logs/policy-server.log"},
]

_CONTROLLER_SOURCE_KEYS = {"s3_uri", "version_id", "sha256", "commit"}
_SPEC_KEYS = {
    "schema_version",
    "project",
    "run_id",
    "aws",
    "source",
    "model_source",
    "policy_image",
    "evaluator_image",
    "host",
    "artifacts",
    "evaluation",
    "continuation",
    "expected_outputs",
    "output",
    "timing",
    "scratch",
    "seed",
}
_CONTINUATION_KEYS = {"parent_run_id", "snapshot"}
_SNAPSHOT_PIN_KEYS = {"s3_uri", "version_id", "sha256"}
_PARTIAL_SNAPSHOT_NAME_RE = re.compile(r"snapshot-[0-9]{4}-[0-9a-f]{16}\.json")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise repro_worker.WorkerError(f"required JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise repro_worker.WorkerError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise repro_worker.WorkerError(f"JSON document must be an object: {path}")
    return value


def _require_exact(value: Any, expected: Any, context: str) -> None:
    if value != expected:
        raise repro_worker.WorkerError(f"{context} differs from the pinned RoboLab base-evaluation contract")


def _validate_complete_source(source: Any, context: str) -> dict[str, str]:
    if not isinstance(source, dict) or set(source) != _CONTROLLER_SOURCE_KEYS:
        raise repro_worker.WorkerError(f"{context} must contain only s3_uri/version_id/sha256/commit")
    commit = source.get("commit")
    digest = source.get("sha256")
    version_id = source.get("version_id")
    if not isinstance(commit, str) or repro_worker.COMMIT_RE.fullmatch(commit) is None:
        raise repro_worker.WorkerError(f"{context} commit must be a full lowercase Git SHA")
    if not isinstance(digest, str) or repro_worker.SHA256_RE.fullmatch(digest) is None:
        raise repro_worker.WorkerError(f"{context} sha256 must be a lowercase SHA-256")
    if not isinstance(version_id, str) or not version_id or any(character in version_id for character in "\x00\r\n"):
        raise repro_worker.WorkerError(f"{context} VersionId is invalid")
    location = repro_worker.parse_s3_uri(str(source.get("s3_uri", "")))
    if location.bucket != BUCKET or location.key != f"source/openpi-{commit}-complete.bundle":
        raise repro_worker.WorkerError(f"{context} must use its commit-qualified complete project bundle key")
    return {str(key): str(value) for key, value in source.items()}


def _validate_controller_source(source: Any) -> dict[str, str]:
    return _validate_complete_source(source, "controller source")


def _validate_model_source(source: Any) -> dict[str, str]:
    validated = _validate_complete_source(source, "model source")
    if validated["commit"] != MODEL_SOURCE_COMMIT:
        raise repro_worker.WorkerError("model source must remain the exact pinned 229c commit")
    return validated


def policy_server_argv() -> list[str]:
    return [
        "python",
        "scripts/serve_policy.py",
        "--env",
        "DROID",
        "--port",
        "8000",
        "--seed",
        str(POLICY_SERVER_SEED),
        "policy:checkpoint",
        "--policy.config",
        "pi05_droid_jointpos",
        "--policy.dir",
        "/mnt/openpi/checkpoints/pi05_droid_jointpos_pytorch",
    ]


def policy_server_identity(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "image_digest": POLICY_IMAGE["uri"],
        "source": dict(spec["model_source"]),
        "config": "pi05_droid_jointpos",
        "command_sha256": _canonical_sha256(policy_server_argv()),
        "checkpoint_model_sha256": PYTORCH_TEACHER["payload_objects"][-1]["sha256"],
    }


def _bounded_docker_resource_name(run_id: str, suffix: str) -> str:
    """Return a collision-resistant Docker/DNS name no longer than one label."""

    readable = "".join(
        character if character in "abcdefghijklmnopqrstuvwxyz0123456789-" else "-" for character in run_id
    )
    readable = readable.strip("-") or "run"
    digest = hashlib.sha256(run_id.encode()).hexdigest()[:12]
    ending = f"-{digest}-{suffix}"
    retained = 63 - len("pi05-") - len(ending)
    return f"pi05-{readable[:retained].rstrip('-')}{ending}"


def policy_container_name(spec: Mapping[str, Any]) -> str:
    return _bounded_docker_resource_name(str(spec["run_id"]), "policy")


def internal_network_name(spec: Mapping[str, Any]) -> str:
    return _bounded_docker_resource_name(str(spec["run_id"]), "network")


def evaluator_argv(spec: Mapping[str, Any]) -> list[str]:
    return [
        "policies/pi0_family/run.py",
        "--policy",
        "pi05",
        "--remote-host",
        policy_container_name(spec),
        "--remote-port",
        "8000",
        "--open-loop-horizon",
        "15",
        "--task",
        *TASKS,
        "--task-dirs",
        "benchmark",
        "--instruction-type",
        "default",
        "--num-envs",
        str(NUM_ENVS),
        "--num-runs",
        str(NUM_RUNS),
        "--renderer",
        "realtime",
        "--rendering-type",
        "balanced",
        "--video-mode",
        "none",
        "--device",
        "cuda:0",
        "--headless",
        "--output-folder-name",
        OUTPUT_FOLDER_NAME,
    ]


def seal_argv(spec: Mapping[str, Any]) -> list[str]:
    return [
        "python",
        "scripts/repro_robolab_report.py",
        "seal",
        "--stage",
        "base",
        "--mode",
        "intermediate",
        "--checkpoint-model",
        "/mnt/openpi/checkpoints/pi05_droid_jointpos_pytorch/model.safetensors",
        "--results",
        f"/output/{OUTPUT_FOLDER_NAME}/episode_results.jsonl",
        "--num-envs",
        str(NUM_ENVS),
        "--num-runs",
        str(NUM_RUNS),
        "--policy-server-seed",
        str(POLICY_SERVER_SEED),
        "--image-digest",
        EVALUATOR_IMAGE["uri"],
        "--robolab-git-sha",
        ROBOLAB_GIT_SHA,
        "--policy-image-digest",
        POLICY_IMAGE["uri"],
        "--policy-source-s3-uri",
        spec["model_source"]["s3_uri"],
        "--policy-source-version-id",
        spec["model_source"]["version_id"],
        "--policy-source-sha256",
        spec["model_source"]["sha256"],
        "--policy-source-commit",
        spec["model_source"]["commit"],
        "--policy-config",
        "pi05_droid_jointpos",
        "--policy-command-sha256",
        _canonical_sha256(policy_server_argv()),
        "--output",
        f"/output/{OUTPUT_FOLDER_NAME}/run-identity.json",
    ]


def _runtime_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Project the dedicated schema into the generic staging/image contract."""

    return {
        "schema_version": 1,
        "project": PROJECT,
        "run_id": spec["run_id"],
        "aws": copy.deepcopy(spec["aws"]),
        "controller_source": copy.deepcopy(spec["source"]),
        "source": copy.deepcopy(spec["model_source"]),
        "image": copy.deepcopy(spec["policy_image"]),
        "artifacts": copy.deepcopy(spec["artifacts"]),
        "container": {
            "command": policy_server_argv(),
            "environment": {
                "CUDA_VISIBLE_DEVICES": "0",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.45",
            },
            "shm_size_gib": 64,
        },
        "expected_outputs": copy.deepcopy(spec["expected_outputs"]),
        "output": copy.deepcopy(spec["output"]),
        "timing": copy.deepcopy(spec["timing"]),
        "scratch": copy.deepcopy(spec["scratch"]),
        "seed": spec["seed"],
    }


def validate_spec(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact, base-only RoboLab worker contract."""

    try:
        spec = json.loads(json.dumps(raw))
    except (TypeError, ValueError) as exc:
        raise repro_worker.WorkerError("RoboLab worker spec is not JSON serializable") from exc
    if not isinstance(spec, dict) or set(spec) != _SPEC_KEYS:
        actual = set(spec) if isinstance(spec, dict) else set()
        raise repro_worker.WorkerError(f"RoboLab worker spec has unexpected schema keys: {sorted(actual ^ _SPEC_KEYS)}")
    if spec.get("schema_version") != 1 or spec.get("project") != PROJECT:
        raise repro_worker.WorkerError("RoboLab worker schema/project mismatch")
    run_id = spec.get("run_id")
    if not isinstance(run_id, str) or repro_worker.RUN_ID_RE.fullmatch(run_id) is None:
        raise repro_worker.WorkerError("RoboLab run_id must satisfy the generic lowercase worker contract")
    _require_exact(
        spec.get("aws"),
        {"account_id": ACCOUNT_ID, "region": REGION, "artifact_bucket": BUCKET},
        "AWS boundary",
    )
    spec["source"] = _validate_controller_source(spec.get("source"))
    spec["model_source"] = _validate_model_source(spec.get("model_source"))
    _require_exact(spec.get("policy_image"), POLICY_IMAGE, "policy image")
    _require_exact(spec.get("evaluator_image"), EVALUATOR_IMAGE, "RoboLab evaluator image")
    _require_exact(
        spec.get("host"),
        {"instance_type": INSTANCE_TYPE, "ami_id": EVALUATION_AMI_ID, "driver_version": EVALUATION_DRIVER_VERSION},
        "R580 evaluation host",
    )
    _require_exact(spec.get("artifacts"), [JAX_TEACHER, PYTORCH_TEACHER], "teacher inputs")
    _require_exact(
        spec.get("evaluation"),
        {
            "stage": "base",
            "mode": "intermediate",
            "policy_server_seed": POLICY_SERVER_SEED,
            "environment_seed": ENVIRONMENT_SEED,
            "tasks": list(TASKS),
            "num_envs": NUM_ENVS,
            "num_runs": NUM_RUNS,
            "episodes_per_task": EPISODES_PER_TASK,
            "open_loop_horizon": 15,
            "instruction_type": "default",
        },
        "evaluation",
    )
    _require_exact(spec.get("expected_outputs"), EXPECTED_OUTPUTS, "expected outputs")
    continuation = spec.get("continuation")
    if continuation is not None:
        if not isinstance(continuation, dict) or set(continuation) != _CONTINUATION_KEYS:
            raise repro_worker.WorkerError("RoboLab continuation must contain only parent_run_id/snapshot")
        parent_run_id = continuation.get("parent_run_id")
        if (
            not isinstance(parent_run_id, str)
            or repro_worker.RUN_ID_RE.fullmatch(parent_run_id) is None
            or parent_run_id == run_id
        ):
            raise repro_worker.WorkerError("RoboLab continuation parent_run_id must be a distinct valid run ID")
        snapshot = continuation.get("snapshot")
        if not isinstance(snapshot, dict) or set(snapshot) != _SNAPSHOT_PIN_KEYS:
            raise repro_worker.WorkerError("RoboLab continuation snapshot pin is incomplete")
        location = repro_worker.parse_s3_uri(str(snapshot.get("s3_uri", "")))
        expected_prefix = f"runs/{parent_run_id}/artifacts/robolab-partials/"
        snapshot_name = location.key.removeprefix(expected_prefix)
        if (
            location.bucket != BUCKET
            or not location.key.startswith(expected_prefix)
            or _PARTIAL_SNAPSHOT_NAME_RE.fullmatch(snapshot_name) is None
        ):
            raise repro_worker.WorkerError("RoboLab continuation snapshot is outside its parent immutable prefix")
        version_id = snapshot.get("version_id")
        digest = snapshot.get("sha256")
        if (
            not isinstance(version_id, str)
            or not version_id
            or any(character in version_id for character in "\x00\r\n")
            or not isinstance(digest, str)
            or repro_worker.SHA256_RE.fullmatch(digest) is None
        ):
            raise repro_worker.WorkerError("RoboLab continuation snapshot lacks an immutable version/hash")
    if spec.get("seed") != POLICY_SERVER_SEED:
        raise repro_worker.WorkerError("worker seed must equal the fixed policy-server seed")
    output = spec.get("output")
    if output != {"s3_uri": f"s3://{BUCKET}/runs/{run_id}/"}:
        raise repro_worker.WorkerError("RoboLab output must use the unique run-id S3 prefix")

    # Reuse the mature artifact, output, timing, scratch, and policy-image
    # validators without permitting the generic worker to orchestrate this job.
    runtime_spec = repro_worker.validate_worker_spec(_runtime_spec(spec))
    spec["artifacts"] = runtime_spec["artifacts"]
    return spec


def validate_controller_source_evidence(spec: Mapping[str, Any], evidence: Mapping[str, Any]) -> None:
    expected = spec["source"]
    if evidence.get("schema_version") != 1:
        raise repro_worker.WorkerError("RoboLab controller source evidence has the wrong schema")
    for key in ("s3_uri", "version_id", "sha256", "commit"):
        if evidence.get("source", {}).get(key) != expected[key]:
            raise repro_worker.WorkerError(f"RoboLab controller source evidence differs for {key}")
    if (
        evidence.get("bundle_sha256_actual") != expected["sha256"]
        or evidence.get("head_commit") != expected["commit"]
        or evidence.get("source_clean") is not True
        or evidence.get("source_fsck_full") is not True
    ):
        raise repro_worker.WorkerError("RoboLab controller bundle/hash/checkout/fsck evidence is incomplete")


def make_spec(
    *,
    run_id: str,
    source: Mapping[str, Any],
    model_source: Mapping[str, Any],
    continuation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spec = {
        "schema_version": 1,
        "project": PROJECT,
        "run_id": run_id,
        "aws": {"account_id": ACCOUNT_ID, "region": REGION, "artifact_bucket": BUCKET},
        "source": dict(source),
        "model_source": dict(model_source),
        "policy_image": copy.deepcopy(POLICY_IMAGE),
        "evaluator_image": copy.deepcopy(EVALUATOR_IMAGE),
        "host": {
            "instance_type": INSTANCE_TYPE,
            "ami_id": EVALUATION_AMI_ID,
            "driver_version": EVALUATION_DRIVER_VERSION,
        },
        "artifacts": [copy.deepcopy(JAX_TEACHER), copy.deepcopy(PYTORCH_TEACHER)],
        "evaluation": {
            "stage": "base",
            "mode": "intermediate",
            "policy_server_seed": POLICY_SERVER_SEED,
            "environment_seed": ENVIRONMENT_SEED,
            "tasks": list(TASKS),
            "num_envs": NUM_ENVS,
            "num_runs": NUM_RUNS,
            "episodes_per_task": EPISODES_PER_TASK,
            "open_loop_horizon": 15,
            "instruction_type": "default",
        },
        "continuation": copy.deepcopy(continuation),
        "expected_outputs": copy.deepcopy(EXPECTED_OUTPUTS),
        "output": {"s3_uri": f"s3://{BUCKET}/runs/{run_id}/"},
        "timing": {"sync_interval_seconds": 60, "upload_buffer_seconds": 900, "stop_grace_seconds": 30},
        "scratch": {
            "model": repro_worker.INSTANCE_STORE_MODEL,
            "expected_count": 1,
            "ordinal": 0,
            "mount": "/mnt/openpi",
            "filesystem_label": repro_worker.SCRATCH_LABEL,
        },
        "seed": POLICY_SERVER_SEED,
    }
    return validate_spec(spec)


def validate_launch_and_host(
    spec: Mapping[str, Any],
    launch_metadata: Mapping[str, Any],
    identity: Mapping[str, Any],
    driver_output: str,
    *,
    now: dt.datetime | None = None,
    command_path: pathlib.Path | None = None,
) -> tuple[dt.datetime, dt.datetime, str, str]:
    hard_deadline, soft_deadline = repro_worker.validate_launch_metadata(
        spec,
        launch_metadata,
        now=now,
        command_path=command_path,
    )
    instance_id, instance_type = repro_worker.validate_instance_identity(spec, launch_metadata, identity)
    observed_drivers = sorted({line.strip() for line in driver_output.splitlines() if line.strip()})
    if (
        launch_metadata.get("category") != "evaluation"
        or launch_metadata.get("workload") != "evaluation"
        or launch_metadata.get("retain_after_command") is not False
        or instance_type != INSTANCE_TYPE
        or identity.get("imageId") != EVALUATION_AMI_ID
        or observed_drivers != [EVALUATION_DRIVER_VERSION]
    ):
        raise repro_worker.WorkerError("live AMI/driver/instance or launch workload differs from the R580 contract")
    return hard_deadline, soft_deadline, instance_id, instance_type


def validate_evaluator_image_identity(repo_digests: Any, labels: Any) -> list[str]:
    if not isinstance(repo_digests, list) or EVALUATOR_IMAGE["uri"] not in repo_digests:
        raise repro_worker.WorkerError("pulled RoboLab image does not expose the requested immutable digest")
    expected_labels = {
        "org.opencontainers.image.revision": ROBOLAB_GIT_SHA,
        "ai.openpi.client-revision": ROBOLAB_CLIENT_GIT_SHA,
        "ai.openpi.isaaclab-base-digest": ISAACLAB_BASE_DIGEST,
    }
    if not isinstance(labels, dict) or any(labels.get(key) != value for key, value in expected_labels.items()):
        raise repro_worker.WorkerError("pulled RoboLab image provenance labels differ from the pinned contract")
    return [str(value) for value in repo_digests]


def _json_command(runner: repro_worker.CommandRunner, argv: Sequence[str]) -> dict[str, Any]:
    try:
        value = json.loads(runner(argv) or "{}")
    except json.JSONDecodeError as exc:
        raise repro_worker.WorkerError(f"command returned invalid JSON: {shlex.join(argv)}") from exc
    if not isinstance(value, dict):
        raise repro_worker.WorkerError(f"command returned non-object JSON: {shlex.join(argv)}")
    return value


def pull_and_verify_evaluator_image(runner: repro_worker.CommandRunner) -> dict[str, Any]:
    uri = EVALUATOR_IMAGE["uri"]
    runner(["docker", "pull", uri])
    try:
        repo_digests = json.loads(runner(["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", uri]))
        labels = json.loads(runner(["docker", "image", "inspect", "--format", "{{json .Config.Labels}}", uri]))
    except json.JSONDecodeError as exc:
        raise repro_worker.WorkerError("RoboLab image inspection returned invalid JSON") from exc
    verified_digests = validate_evaluator_image_identity(repo_digests, labels)

    runtime_probe = (
        "import importlib.metadata as m,json,pathlib;"
        "lab=pathlib.Path('/workspace/isaaclab/VERSION').read_text().splitlines()[0];"
        "sim=pathlib.Path('/workspace/isaaclab/_isaac_sim/VERSION').read_text().splitlines()[0];"
        "observed={'isaac_lab':lab,'isaac_sim':sim,'typeguard':m.version('typeguard'),"
        "'typing_extensions':m.version('typing_extensions')};"
        "assert lab=='2.2.0';assert sim.startswith('5.0.0-');"
        "assert observed['typeguard']=='4.4.2';assert observed['typing_extensions']=='4.12.2';"
        "print(json.dumps(observed,sort_keys=True))"
    )
    runtime = _json_command(
        runner,
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "/workspace/isaaclab/_isaac_sim/python.sh",
            uri,
            "-c",
            runtime_probe,
        ],
    )
    expected_runtime = {
        "isaac_lab": EVALUATOR_IMAGE["isaac_lab_version"],
        "typeguard": EVALUATOR_IMAGE["typeguard_version"],
        "typing_extensions": EVALUATOR_IMAGE["typing_extensions_version"],
    }
    if any(runtime.get(key) != value for key, value in expected_runtime.items()) or not str(
        runtime.get("isaac_sim", "")
    ).startswith(str(EVALUATOR_IMAGE["isaac_sim_version_prefix"])):
        raise repro_worker.WorkerError("RoboLab evaluator runtime versions differ from the pinned contract")
    return {"repo_digests": verified_digests, "labels": labels, "runtime": runtime}


def stage_model_source(
    spec: Mapping[str, Any],
    runtime_spec: Mapping[str, Any],
    run_root: pathlib.Path,
    runner: repro_worker.CommandRunner,
) -> tuple[pathlib.Path, dict[str, Any]]:
    bundle = run_root / "model-source.bundle"
    checkout = run_root / "model-source"
    verify_repo = run_root / "model-source-verify.git"
    if bundle.exists() or checkout.exists() or verify_repo.exists():
        raise repro_worker.WorkerError("model source staging target already exists")
    repro_worker.download_versioned_object(spec["model_source"], bundle, runtime_spec, runner)
    runner(["git", "init", "--bare", str(verify_repo)])
    runner(["git", "-C", str(verify_repo), "bundle", "verify", str(bundle)])
    bundle_head = runner(["git", "bundle", "list-heads", str(bundle), "HEAD"]).split()
    if bundle_head != [spec["model_source"]["commit"], "HEAD"]:
        raise repro_worker.WorkerError("model source bundle HEAD differs from the pinned commit")
    runner(["git", "clone", "--no-checkout", str(bundle), str(checkout)])
    runner(["git", "-C", str(checkout), "checkout", "--detach", spec["model_source"]["commit"]])
    head = runner(["git", "-C", str(checkout), "rev-parse", "HEAD"]).strip()
    dirty = runner(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"]).strip()
    shallow = runner(["git", "-C", str(checkout), "rev-parse", "--is-shallow-repository"]).strip()
    if head != spec["model_source"]["commit"] or dirty or shallow != "false":
        raise repro_worker.WorkerError("staged model source checkout is not the exact clean complete commit")
    runner(["git", "-C", str(checkout), "fsck", "--full", "--no-dangling"])
    repro_checkout_permissions.secure_checkout(checkout, run_root, [bundle])
    evidence = {
        "schema_version": 1,
        "source": dict(spec["model_source"]),
        "bundle_sha256_actual": repro_worker.sha256_file(bundle),
        "head_commit": head,
        "source_clean": True,
        "source_fsck_full": True,
        "bundle_path": str(bundle),
        "checkout_path": str(checkout),
    }
    return checkout, evidence


def policy_container_command(spec: Mapping[str, Any], source_root: pathlib.Path, run_root: pathlib.Path) -> list[str]:
    name = policy_container_name(spec)
    return [
        "docker",
        "run",
        "--detach",
        "--name",
        name,
        "--gpus",
        "all",
        "--network",
        internal_network_name(spec),
        "--network-alias",
        name,
        "--ipc",
        "host",
        "--shm-size",
        "64g",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        "--user",
        "1000:1000",
        "--workdir",
        "/workspace/openpi",
        "--mount",
        f"type=bind,src={source_root},dst=/workspace/openpi,readonly",
        "--mount",
        f"type=bind,src={run_root / 'inputs'},dst=/mnt/openpi,readonly",
        "--mount",
        f"type=bind,src={run_root / 'tmp'},dst=/tmp",
        "--mount",
        f"type=bind,src={run_root / 'cache'},dst=/cache",
        "--env",
        "HOME=/tmp",
        "--env",
        "XDG_CACHE_HOME=/cache",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "PYTHONPATH=/workspace/openpi/src:/workspace/openpi",
        "--env",
        "CUDA_VISIBLE_DEVICES=0",
        "--env",
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.45",
        "--env",
        f"PI05_RUN_ID={spec['run_id']}",
        "--env",
        f"PI05_SOURCE_SHA={spec['model_source']['commit']}",
        "--env",
        f"PI05_IMAGE_DIGEST={POLICY_IMAGE['digest']}",
        "--env",
        f"PI05_SEED={POLICY_SERVER_SEED}",
        POLICY_IMAGE["uri"],
        *policy_server_argv(),
    ]


def evaluator_container_command(spec: Mapping[str, Any], run_root: pathlib.Path) -> list[str]:
    output = run_root / "output" / "artifacts" / "robolab"
    return [
        "docker",
        "run",
        "--name",
        f"pi05-{spec['run_id']}",
        "--gpus",
        "all",
        "--network",
        internal_network_name(spec),
        "--ipc",
        "host",
        "--ulimit",
        "core=0",
        "--env",
        "OMNI_KIT_ACCEPT_EULA=YES",
        "--env",
        "ACCEPT_EULA=Y",
        "--env",
        "PRIVACY_CONSENT=Y",
        "--mount",
        f"type=bind,src={output},dst=/workspace/robolab/output",
        "--entrypoint",
        "/workspace/isaaclab/_isaac_sim/python.sh",
        EVALUATOR_IMAGE["uri"],
        *evaluator_argv(spec),
    ]


def _network_names(name: str, runner: repro_worker.CommandRunner) -> list[str]:
    output = runner(["docker", "network", "ls", "--filter", f"name=^{name}$", "--format", "{{.Name}}"])
    names = [line.strip() for line in output.splitlines() if line.strip()]
    if any(observed != name for observed in names) or len(names) > 1:
        raise repro_worker.WorkerError(f"Docker returned ambiguous exact-network lookup for {name!r}: {names!r}")
    return names


def _expected_network_labels(spec: Mapping[str, Any]) -> dict[str, str]:
    return {"ai.openpi.project": PROJECT, "ai.openpi.run-id": str(spec["run_id"])}


def internal_network_create_argv(spec: Mapping[str, Any]) -> list[str]:
    labels = _expected_network_labels(spec)
    return [
        "docker",
        "network",
        "create",
        "--driver",
        "bridge",
        "--internal",
        "--label",
        f"ai.openpi.project={labels['ai.openpi.project']}",
        "--label",
        f"ai.openpi.run-id={labels['ai.openpi.run-id']}",
        internal_network_name(spec),
    ]


def assert_internal_network_absent(spec: Mapping[str, Any], runner: repro_worker.CommandRunner) -> None:
    if _network_names(internal_network_name(spec), runner):
        raise repro_worker.WorkerError("internal network name already exists; refusing stale reuse")


def inspect_internal_network(
    spec: Mapping[str, Any],
    runner: repro_worker.CommandRunner,
    *,
    require_empty: bool,
) -> dict[str, Any]:
    name = internal_network_name(spec)
    document = _json_command(
        runner,
        ["docker", "network", "inspect", "--format", "{{json .}}", name],
    )
    labels = document.get("Labels")
    containers = document.get("Containers")
    if (
        document.get("Name") != name
        or document.get("Driver") != "bridge"
        or document.get("Scope") != "local"
        or document.get("Internal") is not True
        or document.get("Attachable") is not False
        or document.get("Ingress") is not False
        or labels != _expected_network_labels(spec)
        or not isinstance(containers, dict)
        or (require_empty and containers)
    ):
        raise repro_worker.WorkerError("Docker network differs from the owned internal-bridge contract")
    network_id = document.get("Id")
    if not isinstance(network_id, str) or not network_id:
        raise repro_worker.WorkerError("Docker network inspection omitted its immutable ID")
    return {
        "schema_version": 1,
        "name": name,
        "id": network_id,
        "driver": "bridge",
        "scope": "local",
        "internal": True,
        "attachable": False,
        "ingress": False,
        "labels": dict(labels),
        "containers_at_inspection": sorted(containers),
        "policy_dns_name": policy_container_name(spec),
        "published_host_ports": [],
    }


def create_internal_network(
    spec: Mapping[str, Any],
    runner: repro_worker.CommandRunner,
    *,
    preflight: bool = True,
) -> dict[str, Any]:
    if preflight:
        assert_internal_network_absent(spec, runner)
    network_id = runner(internal_network_create_argv(spec)).strip()
    if not network_id:
        raise repro_worker.WorkerError("Docker network creation omitted its immutable ID")
    evidence = inspect_internal_network(spec, runner, require_empty=True)
    if evidence["id"] != network_id:
        raise repro_worker.WorkerError("created Docker network ID differs from inspected network ID")
    return evidence


def policy_server_is_ready(container_name: str, runner: repro_worker.CommandRunner) -> bool:
    probe = (
        "import urllib.request;"
        "response=urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=1);"
        "body=response.read();"
        "assert response.status==200 and body==b'OK\\n'"
    )
    try:
        runner(["docker", "exec", container_name, "python", "-c", probe])
    except repro_worker.CommandError:
        return False
    return True


def wait_for_policy_server(
    container_name: str,
    soft_deadline: dt.datetime,
    runner: repro_worker.CommandRunner,
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    port_probe: Callable[[], bool] | None = None,
    timeout_seconds: float = 600.0,
) -> None:
    started = monotonic()
    while dt.datetime.now(UTC) < soft_deadline and monotonic() - started < timeout_seconds:
        ready = port_probe() if port_probe is not None else policy_server_is_ready(container_name, runner)
        if ready:
            return
        state = runner(
            ["docker", "container", "inspect", "--format", "{{.State.Running}} {{.State.ExitCode}}", container_name]
        ).split()
        if not state or state[0] != "true":
            exit_code = state[1] if len(state) > 1 else "unknown"
            raise repro_worker.WorkerError(f"policy server exited before readiness with code {exit_code}")
        sleep(1.0)
    raise repro_worker.WorkerError("policy server did not become ready before its bounded startup deadline")


def _write_bytes_new(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise repro_worker.WorkerError(f"refusing to overwrite evidence file: {path}") from exc


def capture_docker_logs(container_name: str) -> bytes:
    completed = subprocess.run(
        ["docker", "logs", "--timestamps", container_name],
        check=False,
        capture_output=True,
    )
    payload = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise repro_worker.CommandError(
            ["docker", "logs", "--timestamps", container_name],
            completed.returncode,
            payload.decode(errors="replace"),
        )
    return payload


def cleanup_policy_container(
    container_name: str,
    log_path: pathlib.Path,
    stop_grace_seconds: int,
    runner: repro_worker.CommandRunner,
    *,
    log_capture: Callable[[str], bytes] = capture_docker_logs,
) -> None:
    errors: list[Exception] = []
    with contextlib.suppress(Exception):
        runner(["docker", "stop", "--time", str(stop_grace_seconds), container_name])
    try:
        _write_bytes_new(log_path, log_capture(container_name))
    except Exception as exc:
        errors.append(exc)
    try:
        runner(["docker", "rm", "--force", container_name])
    except Exception as exc:
        errors.append(exc)
    if errors:
        detail = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
        raise repro_worker.WorkerError(f"policy-container cleanup failed: {detail}")


def cleanup_internal_network(spec: Mapping[str, Any], runner: repro_worker.CommandRunner) -> None:
    name = internal_network_name(spec)
    if not _network_names(name, runner):
        return
    inspect_internal_network(spec, runner, require_empty=True)
    runner(["docker", "network", "rm", name])
    if _network_names(name, runner):
        raise repro_worker.WorkerError("owned internal Docker network remains after removal")


def cleanup_runtime_resources(
    *,
    spec: Mapping[str, Any],
    policy_container_owned: bool,
    network_owned: bool,
    policy_log: pathlib.Path,
    stop_grace_seconds: int,
    runner: repro_worker.CommandRunner,
    outcome: dict[str, bool] | None = None,
) -> None:
    outcome = outcome if outcome is not None else {}
    errors: list[Exception] = []
    if policy_container_owned:
        try:
            cleanup_policy_container(
                policy_container_name(spec),
                policy_log,
                stop_grace_seconds,
                runner,
            )
            outcome["policy_container_removed"] = True
        except Exception as exc:
            outcome["policy_container_removed"] = False
            errors.append(exc)
    if network_owned:
        try:
            cleanup_internal_network(spec, runner)
            outcome["internal_network_removed"] = True
        except Exception as exc:
            outcome["internal_network_removed"] = False
            errors.append(exc)
    if errors:
        detail = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
        raise repro_worker.WorkerError(f"runtime-resource cleanup failed: {detail}")


def continuation_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark": "robolab",
        "stage": "base",
        "output_folder_name": OUTPUT_FOLDER_NAME,
        "evaluation": dict(spec["evaluation"]),
        "evaluator_image": dict(spec["evaluator_image"]),
        "policy_server": policy_server_identity(spec),
    }


def _canonical_jsonl(records: Sequence[Mapping[str, Any]]) -> bytes:
    try:
        return b"".join(
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
            for record in records
        )
    except (TypeError, ValueError) as exc:
        raise repro_worker.WorkerError("RoboLab records are not finite canonical JSON") from exc


def _load_live_jsonl_prefix(path: pathlib.Path) -> list[dict[str, Any]]:
    payload = path.read_bytes()
    if not payload:
        return []
    if not payload.endswith(b"\n"):
        newline = payload.rfind(b"\n")
        if newline < 0:
            return []
        payload = payload[: newline + 1]
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise repro_worker.WorkerError(f"completed RoboLab JSONL line {line_number} is malformed") from exc
        if not isinstance(record, dict):
            raise repro_worker.WorkerError(f"completed RoboLab JSONL line {line_number} is not an object")
        records.append(record)
    return records


def _exact_object_history(
    *,
    bucket: str,
    key: str,
    runner: repro_worker.CommandRunner,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    history = _json_command(
        runner,
        [
            "aws",
            "s3api",
            "list-object-versions",
            "--bucket",
            bucket,
            "--prefix",
            key,
            "--max-keys",
            "10",
            "--expected-bucket-owner",
            ACCOUNT_ID,
            "--region",
            REGION,
            "--output",
            "json",
        ],
    )
    if history.get("IsTruncated") is not False:
        raise repro_worker.WorkerError("partial snapshot object history is truncated")
    raw_versions = history.get("Versions", [])
    raw_markers = history.get("DeleteMarkers", [])
    if not isinstance(raw_versions, list) or not isinstance(raw_markers, list):
        raise repro_worker.WorkerError("partial snapshot object history has invalid list fields")
    if any(not isinstance(item, dict) or not isinstance(item.get("Key"), str) for item in raw_versions):
        raise repro_worker.WorkerError("partial snapshot object history has a malformed version")
    if any(not isinstance(item, dict) or not isinstance(item.get("Key"), str) for item in raw_markers):
        raise repro_worker.WorkerError("partial snapshot object history has a malformed delete marker")
    versions = [item for item in raw_versions if item["Key"] == key]
    markers = [item for item in raw_markers if item["Key"] == key]
    for label, entries in (("version", versions), ("delete marker", markers)):
        if any(
            not isinstance(item.get("VersionId"), str)
            or not item["VersionId"]
            or not isinstance(item.get("IsLatest"), bool)
            for item in entries
        ):
            raise repro_worker.WorkerError(f"partial snapshot object history has a malformed exact {label}")
    return versions, markers


def publish_partial_snapshot_object(
    *,
    spec: Mapping[str, Any],
    output_root: pathlib.Path,
    path: pathlib.Path,
    runner: repro_worker.CommandRunner,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise repro_worker.WorkerError("partial snapshot must be a regular non-symlink file")
    try:
        relative = path.resolve().relative_to(output_root.resolve()).as_posix()
    except ValueError as exc:
        raise repro_worker.WorkerError("partial snapshot escaped the worker output root") from exc
    if not relative.startswith("artifacts/robolab-partials/"):
        raise repro_worker.WorkerError("partial snapshot path is outside its dedicated artifact root")
    key = f"runs/{spec['run_id']}/{relative}"
    digest = repro_worker.sha256_file(path)
    size = path.stat().st_size
    versions, markers = _exact_object_history(bucket=BUCKET, key=key, runner=runner)
    if versions or markers:
        raise repro_worker.WorkerError("partial snapshot key has prior object or delete-marker history")
    response = _json_command(
        runner,
        [
            "aws",
            "s3api",
            "put-object",
            "--bucket",
            BUCKET,
            "--key",
            key,
            "--body",
            str(path),
            "--expected-bucket-owner",
            ACCOUNT_ID,
            "--region",
            REGION,
            "--server-side-encryption",
            "AES256",
            "--metadata",
            f"sha256={digest},run-id={spec['run_id']},role=robolab-partial-continuation",
            "--if-none-match",
            "*",
            "--output",
            "json",
        ],
    )
    version_id = response.get("VersionId")
    if not isinstance(version_id, str) or not version_id or response.get("ServerSideEncryption") != "AES256":
        raise repro_worker.WorkerError("partial snapshot conditional put returned an incomplete receipt")
    versions, markers = _exact_object_history(bucket=BUCKET, key=key, runner=runner)
    if (
        markers
        or len(versions) != 1
        or versions[0].get("VersionId") != version_id
        or versions[0].get("IsLatest") is not True
    ):
        raise repro_worker.WorkerError("partial snapshot does not have exact singleton/no-delete history")
    head = _json_command(
        runner,
        [
            "aws",
            "s3api",
            "head-object",
            "--bucket",
            BUCKET,
            "--key",
            key,
            "--version-id",
            version_id,
            "--expected-bucket-owner",
            ACCOUNT_ID,
            "--region",
            REGION,
            "--output",
            "json",
        ],
    )
    metadata = head.get("Metadata", {})
    if (
        head.get("VersionId") != version_id
        or head.get("ServerSideEncryption") != "AES256"
        or head.get("ContentLength") != size
        or metadata.get("sha256") != digest
        or metadata.get("run-id") != spec["run_id"]
        or metadata.get("role") != "robolab-partial-continuation"
    ):
        raise repro_worker.WorkerError("partial snapshot version head differs from the local immutable object")
    with tempfile.TemporaryDirectory(prefix="pi05-robolab-partial-verify-") as temporary:
        downloaded = pathlib.Path(temporary) / "snapshot.json"
        repro_worker.download_versioned_object(
            {
                "s3_uri": f"s3://{BUCKET}/{key}",
                "version_id": version_id,
                "sha256": digest,
            },
            downloaded,
            _runtime_spec(spec),
            runner,
        )
    return {
        "s3_uri": f"s3://{BUCKET}/{key}",
        "version_id": version_id,
        "sha256": digest,
        "bytes": size,
    }


class PartialEpisodePublisher:
    def __init__(
        self,
        spec: Mapping[str, Any],
        results: pathlib.Path,
        manager: repro_worker.OutputManager,
        runner: repro_worker.CommandRunner,
        *,
        initial_records_sha256: str | None = None,
    ) -> None:
        self.spec = spec
        self.results = results
        self.manager = manager
        self.runner = runner
        self.last_records_sha256 = initial_records_sha256
        self.receipts: list[dict[str, Any]] = []

    def __call__(self) -> None:
        if not self.results.is_file():
            return
        records = _load_live_jsonl_prefix(self.results)
        complete = repro_robolab_report.complete_native_run_prefix(
            records,
            mode="intermediate",
            num_envs=NUM_ENVS,
            num_runs=NUM_RUNS,
        )
        if not complete:
            return
        canonical_records = _canonical_jsonl(complete)
        records_sha256 = hashlib.sha256(canonical_records).hexdigest()
        if records_sha256 == self.last_records_sha256:
            return
        document = {
            "schema_version": 1,
            "kind": "robolab-partial-continuation",
            "parent_run_id": self.spec["run_id"],
            "contract": continuation_contract(self.spec),
            "record_count": len(complete),
            "complete_run_groups": len(complete) // NUM_ENVS,
            "records_sha256": records_sha256,
            "records": complete,
        }
        destination = (
            self.manager.root
            / "artifacts"
            / "robolab-partials"
            / f"snapshot-{len(complete):04d}-{records_sha256[:16]}.json"
        )
        _write_json_new(destination, document)
        receipt = publish_partial_snapshot_object(
            spec=self.spec,
            output_root=self.manager.root,
            path=destination,
            runner=self.runner,
        )
        receipt_path = destination.with_name(destination.stem + ".receipt.json")
        _write_json_new(
            receipt_path,
            {
                "schema_version": 1,
                "kind": "robolab-partial-continuation-receipt",
                "parent_run_id": self.spec["run_id"],
                "record_count": len(complete),
                "records_sha256": records_sha256,
                "snapshot": receipt,
            },
        )
        self.manager.create_marker(
            "artifact",
            [destination, receipt_path],
            f"robolab-partial-{len(complete):04d}-{records_sha256[:16]}.ready.json",
        )
        self.receipts.append(receipt)
        self.last_records_sha256 = records_sha256


def restore_partial_snapshot(
    spec: Mapping[str, Any],
    runtime_spec: Mapping[str, Any],
    run_root: pathlib.Path,
    results: pathlib.Path,
    runner: repro_worker.CommandRunner,
) -> dict[str, Any] | None:
    continuation = spec.get("continuation")
    if continuation is None:
        return None
    snapshot_path = run_root / "continuation-snapshot.json"
    repro_worker.download_versioned_object(continuation["snapshot"], snapshot_path, runtime_spec, runner)
    document = _read_json(snapshot_path)
    records = document.get("records")
    if (
        set(document)
        != {
            "schema_version",
            "kind",
            "parent_run_id",
            "contract",
            "record_count",
            "complete_run_groups",
            "records_sha256",
            "records",
        }
        or document.get("schema_version") != 1
        or document.get("kind") != "robolab-partial-continuation"
        or document.get("parent_run_id") != continuation["parent_run_id"]
        or document.get("contract") != continuation_contract(spec)
        or not isinstance(records, list)
        or any(not isinstance(record, dict) for record in records)
    ):
        raise repro_worker.WorkerError("RoboLab continuation snapshot contract differs from this run")
    try:
        repro_robolab_report.validate_native_continuation(
            records,
            mode="intermediate",
            num_envs=NUM_ENVS,
            num_runs=NUM_RUNS,
        )
    except ValueError as exc:
        raise repro_worker.WorkerError("RoboLab continuation records are not an exact complete-run prefix") from exc
    canonical_records = _canonical_jsonl(records)
    records_sha256 = hashlib.sha256(canonical_records).hexdigest()
    if (
        not records
        or document.get("record_count") != len(records)
        or document.get("complete_run_groups") != len(records) // NUM_ENVS
        or document.get("records_sha256") != records_sha256
    ):
        raise repro_worker.WorkerError("RoboLab continuation snapshot counts or record hash differ")
    results.parent.mkdir(mode=0o755, parents=True)
    _write_bytes_new(results, canonical_records)
    return {
        "parent_run_id": continuation["parent_run_id"],
        "snapshot": dict(continuation["snapshot"]),
        "record_count": len(records),
        "complete_run_groups": len(records) // NUM_ENVS,
        "records_sha256": records_sha256,
    }


def summarize_results(spec: Mapping[str, Any], results: pathlib.Path, identity: pathlib.Path) -> dict[str, Any]:
    try:
        records = repro_robolab_report.load_results(results)
        repro_robolab_report.validate_native_results(
            records,
            mode="intermediate",
            num_envs=NUM_ENVS,
            num_runs=NUM_RUNS,
        )
    except (OSError, ValueError) as exc:
        raise repro_worker.WorkerError("sealed RoboLab results could not be summarized") from exc
    results_sha256 = repro_worker.sha256_file(results)
    model_sha256 = PYTORCH_TEACHER["payload_objects"][-1]["sha256"]
    expected_identity = {
        "schema_version": 1,
        "benchmark": "robolab",
        "stage": "base",
        "stage_identity": f"base-sha256:{model_sha256}",
        "checkpoint": {
            "model_path": "/mnt/openpi/checkpoints/pi05_droid_jointpos_pytorch/model.safetensors",
            "model_sha256": model_sha256,
        },
        "results": {"path": "episode_results.jsonl", "sha256": results_sha256},
        "runtime": {
            "image_digest": EVALUATOR_IMAGE["uri"],
            "robolab_git_sha": ROBOLAB_GIT_SHA,
            "openpi_client_git_sha": ROBOLAB_CLIENT_GIT_SHA,
            "isaac_sim_version": "5.0.0",
            "isaac_lab_version": EVALUATOR_IMAGE["isaac_lab_version"],
        },
        "policy_server": policy_server_identity(spec),
        "evaluation": {
            "mode": "intermediate",
            "tasks": list(TASKS),
            "episodes_per_task": EPISODES_PER_TASK,
            "num_envs": NUM_ENVS,
            "num_runs": NUM_RUNS,
            "policy": "pi05",
            "policy_server_seed": POLICY_SERVER_SEED,
            "environment_seed": ENVIRONMENT_SEED,
            "instruction_type": "default",
            "open_loop_horizon": 15,
        },
    }
    identity_document = _read_json(identity)
    if identity_document != expected_identity:
        raise repro_worker.WorkerError("sealed RoboLab run identity differs from the pinned base-evaluation contract")

    task_metrics: dict[str, Any] = {}
    for task in TASKS:
        selected = [record for record in records if record.get("task_name") == task]
        path_lengths = [float(record["metrics"]["ee_path_length"]) for record in selected]
        sparc = [float(record["metrics"]["ee_sparc"]) for record in selected]
        successes = sum(record["success"] for record in selected)
        task_metrics[task] = {
            "episodes": EPISODES_PER_TASK,
            "successes": successes,
            "success_rate": successes / EPISODES_PER_TASK,
            "ee_path_length_mean": statistics.fmean(path_lengths),
            "ee_sparc_mean": statistics.fmean(sparc),
        }
    return {
        "schema_version": 1,
        "benchmark": "robolab",
        "stage": "base",
        "mode": "intermediate",
        "tasks": task_metrics,
        "aggregate_success_rate": statistics.fmean(metric["success_rate"] for metric in task_metrics.values()),
        "run_identity_sha256": repro_worker.sha256_file(identity),
        "results_sha256": results_sha256,
        "model_sha256": model_sha256,
    }


def _copy_regular_file_new(source: pathlib.Path, destination: pathlib.Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise repro_worker.WorkerError(f"control evidence is not a regular file: {source}")
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=8 * 1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except FileExistsError as exc:
        raise repro_worker.WorkerError(f"refusing to overwrite control evidence: {destination}") from exc
    if destination.stat().st_size != source.stat().st_size or repro_worker.sha256_file(destination) != (
        repro_worker.sha256_file(source)
    ):
        raise repro_worker.WorkerError(f"control evidence copy failed validation: {source}")


def _write_json_new(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    # Use the generic worker's create-once, fsyncing writer so control evidence
    # and terminal manifests have the same durability semantics.
    repro_worker._write_json_new(path, value)  # noqa: SLF001


def seal_results(
    *,
    spec: Mapping[str, Any],
    checkpoint_model: pathlib.Path,
    results: pathlib.Path,
    output: pathlib.Path,
) -> dict[str, Any]:
    try:
        identity = repro_robolab_report.create_run_identity(
            stage="base",
            mode="intermediate",
            checkpoint_model=checkpoint_model,
            checkpoint_model_identity_path=("/mnt/openpi/checkpoints/pi05_droid_jointpos_pytorch/model.safetensors"),
            results=results,
            output=output,
            num_envs=NUM_ENVS,
            num_runs=NUM_RUNS,
            policy_server_seed=POLICY_SERVER_SEED,
            image_digest=EVALUATOR_IMAGE["uri"],
            robolab_git_sha=ROBOLAB_GIT_SHA,
            policy_image_digest=POLICY_IMAGE["uri"],
            policy_source_s3_uri=spec["model_source"]["s3_uri"],
            policy_source_version_id=spec["model_source"]["version_id"],
            policy_source_sha256=spec["model_source"]["sha256"],
            policy_source_commit=spec["model_source"]["commit"],
            policy_config="pi05_droid_jointpos",
            policy_command_sha256=_canonical_sha256(policy_server_argv()),
        )
    except (OSError, ValueError) as exc:
        raise repro_worker.WorkerError("RoboLab results could not be sealed by the controller") from exc
    _write_json_new(output, identity)
    return identity


def publish_terminal_manifests(
    manager: repro_worker.OutputManager,
    run_manifest: Mapping[str, Any],
    *,
    commit_expected_outputs: bool,
) -> None:
    """Publish payload receipts, then the run manifest, then final-sync evidence."""

    if commit_expected_outputs:
        manager.commit_expected_outputs()
    receipts = manager.sync_once()
    manifest_path = manager.root / "manifests" / "run-manifest.json"
    materialized = copy.deepcopy(dict(run_manifest))
    materialized["completed_receipts_before_manifest"] = receipts
    _write_json_new(manifest_path, materialized)
    manager.create_marker("manifest", [manifest_path], "run-manifest.ready.json")
    receipts = manager.sync_once()
    try:
        manifest_receipt = next(item for item in receipts if item["marker"] == "run-manifest.ready.json")
    except StopIteration as exc:
        raise repro_worker.WorkerError("run manifest was not durably uploaded") from exc
    final_path = manager.root / "manifests" / "final-sync-evidence.json"
    _write_json_new(
        final_path,
        {
            "schema_version": 1,
            "run_id": manager.spec["run_id"],
            "status": materialized["status"],
            "final_sync_succeeded": True,
            "recorded_at": dt.datetime.now(UTC).isoformat(),
            "run_manifest_receipt": manifest_receipt,
            "receipt_count": len(receipts),
        },
    )
    manager.create_marker("manifest", [final_path], "final-sync-evidence.ready.json")
    manager.sync_once()


def _check_before_soft_deadline(soft_deadline: dt.datetime, phase: str) -> None:
    if dt.datetime.now(UTC) >= soft_deadline:
        raise repro_worker.WorkerError(f"soft deadline reached before {phase}")


def _base_manifest(
    *,
    spec: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
    launch_metadata: Mapping[str, Any],
    identity: Mapping[str, Any],
    driver_version: str,
    started_at: dt.datetime,
    hard_deadline: dt.datetime,
    soft_deadline: dt.datetime,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "project": PROJECT,
        "run_id": spec["run_id"],
        "started_at": started_at.isoformat(),
        "hard_deadline_utc": hard_deadline.isoformat(),
        "soft_deadline_utc": soft_deadline.isoformat(),
        "source": dict(spec["source"]),
        "source_evidence": dict(source_evidence),
        "model_source": dict(spec["model_source"]),
        "images": {
            "policy": dict(spec["policy_image"]),
            "evaluator": dict(spec["evaluator_image"]),
        },
        "checkpoint": {
            "revision": PYTORCH_TEACHER["revision"],
            "model_sha256": PYTORCH_TEACHER["payload_objects"][-1]["sha256"],
            "config": "pi05_droid_jointpos",
        },
        "dataset": {
            "name": None,
            "revision": None,
            "reason": "RoboLab closed-loop base evaluation uses simulator tasks, not an offline DROID dataset",
        },
        "experiment": {"seed": POLICY_SERVER_SEED, "steps": None},
        "evaluation": dict(spec["evaluation"]),
        "network": {
            "name": internal_network_name(spec),
            "driver": "bridge",
            "internal": True,
            "policy_dns_name": policy_container_name(spec),
            "published_host_ports": [],
        },
        "commands": {
            "network_create": internal_network_create_argv(spec),
            "network_create_sha256": _canonical_sha256(internal_network_create_argv(spec)),
            "policy_server": policy_server_argv(),
            "policy_server_sha256": _canonical_sha256(policy_server_argv()),
            "evaluator": evaluator_argv(spec),
            "evaluator_sha256": _canonical_sha256(evaluator_argv(spec)),
            "seal_execution": "controller-direct",
            "seal_contract_argv": seal_argv(spec),
            "seal_contract_sha256": _canonical_sha256(seal_argv(spec)),
            "launcher_command_sha256": launch_metadata["command_sha256"],
        },
        "cost": repro_worker.worker_cost_record(launch_metadata),
        "instance": dict(identity),
        "driver_version": driver_version,
        "launch": dict(launch_metadata),
        "expected_outputs": list(spec["expected_outputs"]),
    }


def _execute_worker_with_scratch(
    spec: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
    launch_metadata: Mapping[str, Any],
    *,
    runner: repro_worker.CommandRunner | None = None,
    identity: Mapping[str, Any] | None = None,
    command_path: pathlib.Path = pathlib.Path("/opt/pi05/run-command.sh"),
) -> int:
    runner = runner or repro_worker.SubprocessRunner()
    runtime_spec = _runtime_spec(spec)
    validate_controller_source_evidence(spec, source_evidence)
    identity = dict(identity or repro_worker.get_instance_identity())
    driver_output = runner(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    hard_deadline, soft_deadline, instance_id, instance_type = validate_launch_and_host(
        spec,
        launch_metadata,
        identity,
        driver_output,
        command_path=command_path,
    )
    repro_worker.verify_aws_boundary(runtime_spec, runner)
    started_at = dt.datetime.now(UTC)
    selection = repro_worker.prepare_scratch(runtime_spec, runner)
    if selection.run_root is None:
        raise repro_worker.WorkerError("scratch preparation did not return a run workspace")
    run_root = pathlib.Path(selection.run_root)
    manager = repro_worker.OutputManager(runtime_spec, run_root / "output", runner)
    evidence_root = run_root / "output" / "artifacts" / "robolab"
    evidence_root.mkdir(mode=0o755)
    result_root = evidence_root / OUTPUT_FOLDER_NAME
    results = result_root / "episode_results.jsonl"

    model_source_evidence: dict[str, Any] | None = None
    staged_inputs: list[dict[str, Any]] = []
    image_evidence: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    failure: str | None = None
    evaluator_exit_code: int | None = None
    evaluator_termination: str | None = None
    policy_container_owned = False
    network_owned = False
    policy_name = policy_container_name(spec)
    policy_log = run_root / "output" / "logs" / "policy-server.log"
    model_source_root: pathlib.Path | None = None
    network_evidence: dict[str, Any] = {}
    cleanup_evidence: dict[str, bool] = {}
    continuation_evidence: dict[str, Any] | None = None
    partial_publisher: PartialEpisodePublisher | None = None

    previous_handlers: dict[int, Any] = {}

    def interrupt(signum: int, _frame: Any) -> None:
        raise InterruptedError(signal.Signals(signum).name)

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, interrupt)
    try:
        _check_before_soft_deadline(soft_deadline, "model-source staging")
        model_source_root, model_source_evidence = stage_model_source(spec, runtime_spec, run_root, runner)
        for artifact in runtime_spec["artifacts"]:
            _check_before_soft_deadline(soft_deadline, f"staging {artifact['name']}")
            staged_inputs.append(repro_worker.stage_artifact(artifact, runtime_spec, run_root, runner))
        manifests = {
            artifact["name"]: _read_json(run_root / "inputs" / ".manifests" / f"{artifact['name']}.json")
            for artifact in runtime_spec["artifacts"]
        }
        repro_worker.validate_input_cross_contracts(runtime_spec, manifests)
        repro_worker.make_staged_input_container_readable(run_root / "inputs")
        continuation_evidence = restore_partial_snapshot(spec, runtime_spec, run_root, results, runner)
        partial_publisher = PartialEpisodePublisher(
            spec,
            results,
            manager,
            runner,
            initial_records_sha256=(
                str(continuation_evidence["records_sha256"]) if continuation_evidence is not None else None
            ),
        )

        _check_before_soft_deadline(soft_deadline, "image verification")
        policy_repo_digests = repro_worker.verify_and_pull_image(runtime_spec, runner)
        evaluator_image_evidence = pull_and_verify_evaluator_image(runner)
        image_evidence = {
            "policy": {"repo_digests": policy_repo_digests},
            "evaluator": evaluator_image_evidence,
        }

        existing = runner(
            ["docker", "container", "ls", "-a", "--filter", f"name=^{policy_name}$", "--format", "{{.Names}}"]
        )
        if existing.strip():
            raise repro_worker.WorkerError("policy container name already exists; refusing stale reuse")
        assert_internal_network_absent(spec, runner)
        # The exact network name was proven absent, so this invocation owns
        # any network Docker may create even if ``network create`` returns an
        # error after the daemon has materialized it.
        network_owned = True
        network_evidence = create_internal_network(spec, runner, preflight=False)
        assert model_source_root is not None
        # The name was proven absent, so this invocation owns any container
        # Docker may create even if ``docker run`` itself returns an error.
        # Claim cleanup before launch to cover that partial-creation path.
        policy_container_owned = True
        runner(policy_container_command(spec, model_source_root, run_root))
        wait_for_policy_server(policy_name, soft_deadline, runner)

        _check_before_soft_deadline(soft_deadline, "RoboLab evaluation")
        evaluator_exit_code, evaluator_termination = repro_worker.run_container_until_deadline(
            evaluator_container_command(spec, run_root),
            manager,
            soft_deadline,
            runtime_spec,
            runner,
            periodic_callback=partial_publisher,
        )
        if evaluator_exit_code != 0 or evaluator_termination is not None:
            raise repro_worker.WorkerError(
                f"RoboLab evaluator failed: exit={evaluator_exit_code}, termination={evaluator_termination}"
            )
        if not policy_server_is_ready(policy_name, runner):
            raise repro_worker.WorkerError("policy server exited before RoboLab result sealing")
        run_identity = result_root / "run-identity.json"
        checkpoint_model = (
            run_root / "inputs" / repro_worker.artifact_relative_destination(PYTORCH_TEACHER) / "model.safetensors"
        )
        seal_results(spec=spec, checkpoint_model=checkpoint_model, results=results, output=run_identity)
        metrics = summarize_results(spec, results, run_identity)
        _write_json_new(result_root / "summary.json", metrics)

        control = evidence_root / "control"
        control.mkdir(mode=0o755)
        _copy_regular_file_new(command_path, control / "run-command.sh")
        _copy_regular_file_new(pathlib.Path("/opt/pi05/launch-metadata.json"), control / "launch-metadata.json")
        spec_path = pathlib.Path(str(source_evidence.get("worker_spec", {}).get("local_path", "")))
        if not spec_path.is_file():
            spec_path = pathlib.Path("/opt/pi05/robolab-worker-spec.json")
        _copy_regular_file_new(spec_path, control / "worker-spec.json")
        source_path = pathlib.Path(str(source_evidence.get("evidence_path", "")))
        if not source_path.is_file():
            source_path = pathlib.Path("/opt/pi05/source-evidence.json")
        _copy_regular_file_new(source_path, control / "source-evidence.json")
        _write_json_new(control / "model-source-evidence.json", model_source_evidence)
        _write_json_new(control / "image-evidence.json", image_evidence)
        _write_json_new(control / "network-evidence.json", network_evidence)
        if continuation_evidence is not None:
            _write_json_new(control / "continuation-evidence.json", continuation_evidence)
        _write_json_new(
            control / "runtime-identity.json",
            {
                "schema_version": 1,
                "instance_id": instance_id,
                "instance_type": instance_type,
                "ami_id": identity["imageId"],
                "driver_version": EVALUATION_DRIVER_VERSION,
            },
        )
        for artifact in runtime_spec["artifacts"]:
            _copy_regular_file_new(
                run_root / "inputs" / ".manifests" / f"{artifact['name']}.json",
                control / f"{artifact['name']}-manifest.json",
            )
    except Exception as exc:  # A terminal manifest records the exact bounded failure.
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            cleanup_runtime_resources(
                spec=spec,
                policy_container_owned=policy_container_owned,
                network_owned=network_owned,
                policy_log=policy_log,
                stop_grace_seconds=int(spec["timing"]["stop_grace_seconds"]),
                runner=runner,
                outcome=cleanup_evidence,
            )
        except Exception as exc:
            cleanup_failure = f"cleanup failed: {type(exc).__name__}: {exc}"
            failure = f"{failure}; {cleanup_failure}" if failure is not None else cleanup_failure
        for signum, handler in previous_handlers.items():
            with contextlib.suppress(Exception):
                signal.signal(signum, handler)

    status = "succeeded" if failure is None else "failed"
    manifest = _base_manifest(
        spec=spec,
        source_evidence=source_evidence,
        launch_metadata=launch_metadata,
        identity=identity,
        driver_version=EVALUATION_DRIVER_VERSION,
        started_at=started_at,
        hard_deadline=hard_deadline,
        soft_deadline=soft_deadline,
    )
    manifest.update(
        {
            "status": status,
            "finished_at": dt.datetime.now(UTC).isoformat(),
            "failure": failure,
            "model_source_evidence": model_source_evidence,
            "staged_inputs": staged_inputs,
            "image_evidence": image_evidence,
            "network_evidence": network_evidence,
            "cleanup_evidence": cleanup_evidence,
            "continuation_evidence": continuation_evidence,
            "partial_snapshot_receipts": partial_publisher.receipts if partial_publisher is not None else [],
            "metrics": metrics,
            "evaluator_exit_code": evaluator_exit_code,
            "evaluator_termination": evaluator_termination,
            "scratch": {
                "device": selection.path,
                "serial": selection.serial,
                "reused": selection.reuse,
                "mounted_at": selection.mounted_at,
                "filesystem": selection.filesystem,
                "run_root": str(run_root),
            },
        }
    )
    if failure is not None and policy_log.is_file():
        manager.create_marker("log", [policy_log], "policy-server-failure.ready.json")
    try:
        publish_terminal_manifests(manager, manifest, commit_expected_outputs=status == "succeeded")
    except Exception as exc:
        print(f"FINAL SYNC FAILED: {exc}", file=sys.stderr)
        return 3
    if failure is not None:
        print(f"ROBOLAB WORKER FAILED: {failure}", file=sys.stderr)
        return evaluator_exit_code if evaluator_exit_code not in (None, 0) else 3
    return 0


def _prepare_emergency_output_root(base: pathlib.Path, run_id: str) -> pathlib.Path:
    if base.is_symlink():
        raise repro_worker.WorkerError("emergency evidence root is a symlink")
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not base.is_dir():
        raise repro_worker.WorkerError("emergency evidence root is not a directory")
    run_root = base / run_id
    try:
        run_root.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise repro_worker.WorkerError("emergency evidence run root already exists") from exc
    output = run_root / "output"
    for relative in (
        ".ready",
        ".receipts",
        ".spool",
        ".active",
        "checkpoints",
        "logs",
        "manifests",
        "artifacts",
    ):
        (output / relative).mkdir(mode=0o700, parents=True)
    return output


def execute_worker(
    spec: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
    launch_metadata: Mapping[str, Any],
    *,
    runner: repro_worker.CommandRunner | None = None,
    identity: Mapping[str, Any] | None = None,
    command_path: pathlib.Path = pathlib.Path("/opt/pi05/run-command.sh"),
    emergency_root: pathlib.Path = pathlib.Path("/opt/pi05/emergency-runs"),
    startup_failure: Exception | None = None,
) -> int:
    """Run the worker and durably publish failures that precede scratch setup."""

    runner = runner or repro_worker.SubprocessRunner()
    started_at = dt.datetime.now(UTC)
    emergency_output = _prepare_emergency_output_root(emergency_root, str(spec["run_id"]))
    emergency_manager = repro_worker.OutputManager(_runtime_spec(spec), emergency_output, runner)
    try:
        if startup_failure is not None:
            raise startup_failure
        return _execute_worker_with_scratch(
            spec,
            source_evidence,
            launch_metadata,
            runner=runner,
            identity=identity,
            command_path=command_path,
        )
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        command_sha256 = None
        with contextlib.suppress(OSError):
            command_sha256 = repro_worker.sha256_file(command_path)
        manifest = {
            "schema_version": 1,
            "project": PROJECT,
            "run_id": spec["run_id"],
            "status": "failed",
            "failure_phase": (
                "startup-evidence-load" if startup_failure is not None else "before-dedicated-output-manager"
            ),
            "started_at": started_at.isoformat(),
            "finished_at": dt.datetime.now(UTC).isoformat(),
            "failure": failure,
            "source": dict(spec["source"]),
            "source_evidence": dict(source_evidence),
            "model_source": dict(spec["model_source"]),
            "images": {
                "policy": dict(spec["policy_image"]),
                "evaluator": dict(spec["evaluator_image"]),
            },
            "evaluation": dict(spec["evaluation"]),
            "continuation": copy.deepcopy(spec.get("continuation")),
            "launch": dict(launch_metadata),
            "executing_command_sha256": command_sha256,
            "instance": dict(identity) if identity is not None else None,
            "expected_outputs": list(spec["expected_outputs"]),
            "emergency_output_root": str(emergency_output),
        }
        try:
            publish_terminal_manifests(emergency_manager, manifest, commit_expected_outputs=False)
        except Exception as publish_exc:
            print(f"EARLY FAILURE SYNC FAILED: {publish_exc}", file=sys.stderr)
            return 3
        print(f"ROBOLAB WORKER FAILED BEFORE SCRATCH OUTPUT: {failure}", file=sys.stderr)
        return 3


def render_bootstrap_command(
    *,
    spec_s3_uri: str,
    spec_version_id: str,
    spec_sha256: str,
    execute: bool,
) -> str:
    location = repro_worker.parse_s3_uri(spec_s3_uri)
    if location.bucket != BUCKET:
        raise repro_worker.WorkerError("RoboLab worker spec must be in the pinned artifact bucket")
    if not spec_version_id or any(character in spec_version_id for character in "\x00\r\n"):
        raise repro_worker.WorkerError("RoboLab worker spec VersionId is invalid")
    if repro_worker.SHA256_RE.fullmatch(spec_sha256) is None:
        raise repro_worker.WorkerError("RoboLab worker spec SHA-256 is invalid")
    execute_flag = " --execute" if execute else ""
    spec_uri_literal = json.dumps(spec_s3_uri)
    quoted = {
        key: shlex.quote(value)
        for key, value in {
            "bucket": location.bucket,
            "key": location.key,
            "version": spec_version_id,
            "sha": spec_sha256,
        }.items()
    }
    return f"""#!/bin/bash
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
install -d -m 0700 /opt/pi05
for command in aws git python3 sha256sum; do command -v "$command" >/dev/null; done
test "$(aws sts get-caller-identity --region {REGION} --query Account --output text)" = {ACCOUNT_ID}
readonly spec_path=/opt/pi05/robolab-worker-spec.json
readonly source_bundle=/opt/pi05/controller-source.bundle
readonly checkout=/opt/pi05/controller-source
readonly evidence=/opt/pi05/source-evidence.json
aws s3api get-object \
  --bucket {quoted["bucket"]} --key {quoted["key"]} --version-id {quoted["version"]} \
  --expected-bucket-owner {ACCOUNT_ID} --region {REGION} "$spec_path" >/dev/null
printf '%s  %s\\n' {quoted["sha"]} "$spec_path" | sha256sum --check --status
mapfile -d '' -t source_fields < <(python3 - "$spec_path" <<'PY'
import json,pathlib,re,sys,urllib.parse
spec_path=pathlib.Path(sys.argv[1])
spec=json.loads(spec_path.read_text())
source=spec.get("source")
if not isinstance(source,dict) or set(source)!={{"s3_uri","version_id","sha256","commit"}}:
    raise SystemExit("invalid controller source object")
uri=urllib.parse.urlsplit(str(source["s3_uri"]))
key=uri.path.lstrip("/")
commit=str(source["commit"]); digest=str(source["sha256"]); version=str(source["version_id"])
if uri.scheme!="s3" or uri.netloc!="{BUCKET}" or key!=f"source/openpi-{{commit}}-complete.bundle":
    raise SystemExit("invalid controller source URI")
if re.fullmatch(r"[0-9a-f]{{40}}",commit) is None or re.fullmatch(r"[0-9a-f]{{64}}",digest) is None:
    raise SystemExit("invalid controller source identity")
if not version or any(c in version for c in "\\x00\\r\\n"):
    raise SystemExit("invalid controller source VersionId")
for value in (uri.netloc,key,version,digest,commit):
    sys.stdout.write(value+"\\0")
PY
)
test "${{#source_fields[@]}}" -eq 5
readonly source_bucket="${{source_fields[0]}}"
readonly source_key="${{source_fields[1]}}"
readonly source_version="${{source_fields[2]}}"
readonly source_sha="${{source_fields[3]}}"
readonly source_commit="${{source_fields[4]}}"
aws s3api get-object \
  --bucket "$source_bucket" --key "$source_key" --version-id "$source_version" \
  --expected-bucket-owner {ACCOUNT_ID} --region {REGION} "$source_bundle" >/dev/null
printf '%s  %s\\n' "$source_sha" "$source_bundle" | sha256sum --check --status
readonly verify_repo=/opt/pi05/controller-source-verify.git
git init --bare "$verify_repo" >/dev/null
git -C "$verify_repo" bundle verify "$source_bundle" >/dev/null
test "$(git bundle list-heads "$source_bundle" HEAD | awk '$2 == "HEAD" {{print $1}}')" = "$source_commit"
git clone --no-checkout "$source_bundle" "$checkout"
git -C "$checkout" checkout --detach "$source_commit"
test "$(git -C "$checkout" rev-parse HEAD)" = "$source_commit"
test -z "$(git -C "$checkout" status --porcelain --untracked-files=all)"
test "$(git -C "$checkout" rev-parse --is-shallow-repository)" = false
git -C "$checkout" fsck --full --no-dangling >/dev/null
python3 - "$evidence" "$spec_path" {quoted["version"]} {quoted["sha"]} \
  "$source_bucket" "$source_key" "$source_version" "$source_sha" "$source_commit" \
  "$source_bundle" "$checkout" <<'PY'
import json,os,sys
(output,spec_path,spec_version,spec_sha,bucket,key,source_version,source_sha,commit,bundle,checkout)=sys.argv[1:]
value={{
  "schema_version":1,
  "worker_spec":{{
    "s3_uri":{spec_uri_literal},
    "version_id":spec_version,
    "sha256":spec_sha,
    "local_path":spec_path,
  }},
  "source":{{"s3_uri":f"s3://{{bucket}}/{{key}}","version_id":source_version,"sha256":source_sha,"commit":commit}},
  "bundle_sha256_actual":source_sha,
  "head_commit":commit,
  "source_clean":True,
  "source_fsck_full":True,
  "bundle_path":bundle,
  "checkout_path":checkout,
  "evidence_path":output,
}}
temporary=output+".tmp"
with open(temporary,"x",encoding="utf-8") as stream:
    json.dump(value,stream,indent=2,sort_keys=True);stream.write("\\n");stream.flush();os.fsync(stream.fileno())
os.replace(temporary,output)
PY
python3 "$checkout/scripts/repro_checkout_permissions.py" \
  --checkout "$checkout" --control-root /opt/pi05 \
  --control-file "$spec_path" --control-file "$source_bundle" \
  --control-file "$evidence" --control-file /opt/pi05/launch-metadata.json \
  --control-file /opt/pi05/run-command.sh
test "$(git -C "$checkout" rev-parse HEAD)" = "$source_commit"
test -z "$(git -C "$checkout" status --porcelain --untracked-files=all)"
exec python3 "$checkout/scripts/repro_robolab_worker.py" run \
  --spec "$spec_path" --source-evidence "$evidence" \
  --launch-metadata /opt/pi05/launch-metadata.json{execute_flag}
"""


def render_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mode": "dry-run",
        "run_id": spec["run_id"],
        "controller_source": spec["source"],
        "model_source": spec["model_source"],
        "policy_image": spec["policy_image"],
        "evaluator_image": spec["evaluator_image"],
        "host": spec["host"],
        "artifacts": spec["artifacts"],
        "evaluation": spec["evaluation"],
        "continuation": spec["continuation"],
        "network": {
            "name": internal_network_name(spec),
            "driver": "bridge",
            "internal": True,
            "policy_dns_name": policy_container_name(spec),
            "published_host_ports": [],
        },
        "commands": {
            "network_create": internal_network_create_argv(spec),
            "policy_server": policy_server_argv(),
            "evaluator": evaluator_argv(spec),
            "seal_execution": "controller-direct",
            "seal_contract_argv": seal_argv(spec),
        },
        "expected_outputs": spec["expected_outputs"],
        "output": spec["output"],
        "mutations_authorized": False,
    }


def _write_spec(path: pathlib.Path, spec: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    except FileExistsError as exc:
        raise repro_worker.WorkerError(f"refusing to overwrite worker spec: {path}") from exc


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    make = subparsers.add_parser("make-spec", help="create the exact base-intermediate worker spec")
    make.add_argument("--run-id", required=True)
    make.add_argument("--source-s3-uri", required=True)
    make.add_argument("--source-version-id", required=True)
    make.add_argument("--source-sha256", required=True)
    make.add_argument("--source-commit", required=True)
    make.add_argument("--model-source-s3-uri", required=True)
    make.add_argument("--model-source-version-id", required=True)
    make.add_argument("--model-source-sha256", required=True)
    make.add_argument("--model-source-commit", required=True)
    make.add_argument("--continuation-parent-run-id")
    make.add_argument("--continuation-s3-uri")
    make.add_argument("--continuation-version-id")
    make.add_argument("--continuation-sha256")
    make.add_argument("--output", required=True, type=pathlib.Path)

    run = subparsers.add_parser("run", help="plan or execute one exact RoboLab worker")
    run.add_argument("--spec", required=True, type=pathlib.Path)
    run.add_argument("--source-evidence", type=pathlib.Path)
    run.add_argument("--launch-metadata", type=pathlib.Path, default=pathlib.Path("/opt/pi05/launch-metadata.json"))
    run.add_argument("--execute", action="store_true")

    render = subparsers.add_parser("render-bootstrap", help="render the exact EC2 command file")
    render.add_argument("--spec-s3-uri", required=True)
    render.add_argument("--spec-version-id", required=True)
    render.add_argument("--spec-sha256", required=True)
    render.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.action == "make-spec":
            continuation_values = (
                args.continuation_parent_run_id,
                args.continuation_s3_uri,
                args.continuation_version_id,
                args.continuation_sha256,
            )
            if any(value is not None for value in continuation_values) and not all(
                value is not None for value in continuation_values
            ):
                raise repro_worker.WorkerError("all four continuation arguments must be supplied together")
            continuation = None
            if all(value is not None for value in continuation_values):
                continuation = {
                    "parent_run_id": args.continuation_parent_run_id,
                    "snapshot": {
                        "s3_uri": args.continuation_s3_uri,
                        "version_id": args.continuation_version_id,
                        "sha256": args.continuation_sha256,
                    },
                }
            spec = make_spec(
                run_id=args.run_id,
                source={
                    "s3_uri": args.source_s3_uri,
                    "version_id": args.source_version_id,
                    "sha256": args.source_sha256,
                    "commit": args.source_commit,
                },
                model_source={
                    "s3_uri": args.model_source_s3_uri,
                    "version_id": args.model_source_version_id,
                    "sha256": args.model_source_sha256,
                    "commit": args.model_source_commit,
                },
                continuation=continuation,
            )
            _write_spec(args.output, spec)
            print(json.dumps({"path": str(args.output), "sha256": repro_worker.sha256_file(args.output)}, indent=2))
            return 0
        if args.action == "render-bootstrap":
            print(
                render_bootstrap_command(
                    spec_s3_uri=args.spec_s3_uri,
                    spec_version_id=args.spec_version_id,
                    spec_sha256=args.spec_sha256,
                    execute=args.execute,
                ),
                end="",
            )
            return 0

        spec = validate_spec(_read_json(args.spec))
        if not args.execute:
            print(json.dumps(render_plan(spec), indent=2, sort_keys=True))
            return 0
        if args.source_evidence is None:
            raise repro_worker.WorkerError("executed RoboLab worker requires --source-evidence")
        source_evidence: dict[str, Any] = {}
        launch_metadata: dict[str, Any] = {}
        try:
            source_evidence = _read_json(args.source_evidence)
            launch_metadata = _read_json(args.launch_metadata)
        except (OSError, ValueError, repro_worker.WorkerError) as exc:
            return execute_worker(spec, source_evidence, launch_metadata, startup_failure=exc)
        return execute_worker(spec, source_evidence, launch_metadata)
    except (OSError, ValueError, repro_worker.WorkerError) as exc:
        print(f"ROBOLAB WORKER REJECTED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
