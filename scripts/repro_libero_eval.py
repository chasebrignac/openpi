#!/usr/bin/env python3
"""Run or render one immutable, network-isolated LIBERO evaluation worker."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import math
import os
import pathlib
import re
import socket
import subprocess
import sys
import time
from typing import Any

PROJECT = "pi05-aws-repro"
ACCOUNT = "752160877725"
REGION = "us-east-2"
BUCKET = "pi05-repro-752160877725-us-east-2"
LIBERO_REPOSITORY = "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
LIBERO_REVISION = "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"
LEROBOT_V2_REVISION = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
LIBERO_REQUIREMENTS_SHA256 = "124e74d09719941c9e3e75a61330808a8d32ae35a1ebee00c18e1222e966d0c8"
DEFAULT_CONTRACT = pathlib.Path("/opt/libero-evaluator-contract.json")
DEFAULT_EVALUATOR_PYTHON = pathlib.Path("/opt/libero-venv/bin/python")
TENSORRT_POLICY_PYTHON = pathlib.Path("/opt/modelopt/bin/python")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
BACKENDS = ("eager", "tensorrt")
TENSORRT_PRECISIONS = ("bf16", "fp8")
TENSORRT_INSTANCE_TYPE = "g7e.4xlarge"
LIBERO_DATASET = "physical-intelligence/libero"
LIBERO_DATASET_REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"
TENSORRT_POLICY_CONFIG = "pi05_libero_l09_snapflow"
TENSORRT_TOOLCHAIN = {
    "tensorrt_version": "11.0.0.114",
    "cuda_version": "13.3.0",
    "modelopt_version": "0.45.0",
    "torch_version": "2.8.0",
    "onnx_version": "1.21.0",
    "onnxruntime_gpu_version": "1.24.2",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
INSTANCE_ID_RE = re.compile(r"^i-[0-9a-f]{8,17}$")
STAGE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
IMAGE_RE = re.compile(
    r"^752160877725\.dkr\.ecr\.us-east-2\.amazonaws\.com/" r"[a-z0-9._/-]+@(?P<digest>sha256:[0-9a-f]{64})$"
)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: pathlib.Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"{label} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def write_atomic_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_runtime_contract(path: pathlib.Path) -> dict[str, Any]:
    contract = read_json_object(path, label="LIBERO runtime contract")
    if contract.get("schema_version") != 1 or contract.get("kind") != "pi05-libero-evaluator":
        raise ValueError("LIBERO runtime contract schema/kind mismatch")
    simulator = contract.get("simulator")
    if simulator != {"repository": LIBERO_REPOSITORY, "revision": LIBERO_REVISION}:
        raise ValueError("LIBERO simulator identity differs from the pinned gitlink target")
    python = contract.get("python")
    if not isinstance(python, dict) or python.get("version") != "3.8.20":
        raise ValueError("LIBERO evaluator Python is not the pinned 3.8.20 runtime")
    requirements = contract.get("requirements")
    if not isinstance(requirements, dict):
        raise ValueError("LIBERO runtime contract has no dependency lock identity")
    lock_path = pathlib.Path(str(requirements.get("installed_path", "")))
    expected_hash = requirements.get("sha256")
    if not lock_path.is_absolute() or not isinstance(expected_hash, str) or SHA256_RE.fullmatch(expected_hash) is None:
        raise ValueError("LIBERO runtime contract dependency lock identity is invalid")
    if sha256_file(lock_path) != expected_hash:
        raise ValueError("installed LIBERO dependency lock hash differs from its runtime contract")
    if tuple(contract.get("suites", ())) != SUITES:
        raise ValueError("LIBERO runtime contract suite order differs from the official evaluation order")
    if contract.get("tasks_per_suite") != 10 or contract.get("minimum_fixed_init_states_per_task") != 50:
        raise ValueError("LIBERO runtime contract task/init-state cardinality mismatch")
    expected_backends = {
        "eager": {"instance_type": "g6e.4xlarge", "server": "scripts/serve_policy.py"},
        "tensorrt": {
            "instance_type": TENSORRT_INSTANCE_TYPE,
            "policy_python": str(TENSORRT_POLICY_PYTHON),
            "server": "scripts/serve_tensorrt_policy.py",
            "track": "libero",
            "dataset": LIBERO_DATASET,
            "dataset_revision": LIBERO_DATASET_REVISION,
            "precisions": list(TENSORRT_PRECISIONS),
            "placement": "exact-engine-build-instance",
        },
    }
    # Schema-v1 contracts produced before the compiled backend did not carry
    # this descriptive field. New images always embed it; if present it must
    # be exact, while the executable backend checks below remain authoritative.
    if contract.get("policy_backends", expected_backends) != expected_backends:
        raise ValueError("LIBERO runtime contract policy-backend identity mismatch")
    if contract.get("gpu_environment") != {
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": "0",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
        "PYOPENGL_PLATFORM": "egl",
    }:
        raise ValueError("LIBERO runtime contract GPU/EGL environment mismatch")
    return contract


def validate_run_identity(args: argparse.Namespace) -> tuple[str, str, str]:
    def environment_bound(argument: str, environment_name: str) -> str:
        environment_value = os.environ.get(environment_name, "")
        if argument and environment_value and argument != environment_value:
            raise ValueError(f"explicit identity differs from {environment_name}")
        return argument or environment_value

    source_commit = environment_bound(args.source_commit, "PI05_SOURCE_SHA")
    image_digest = environment_bound(args.image_digest, "PI05_IMAGE_DIGEST")
    run_id = environment_bound(args.run_id, "PI05_RUN_ID")
    if COMMIT_RE.fullmatch(source_commit) is None:
        raise ValueError("source commit must be a full lowercase git commit")
    if not image_digest.startswith("sha256:") or SHA256_RE.fullmatch(image_digest.removeprefix("sha256:")) is None:
        raise ValueError("image digest must be sha256 followed by 64 lowercase hex characters")
    if STAGE_RE.fullmatch(run_id) is None:
        raise ValueError("run id must use the worker-safe lowercase identifier format")
    if STAGE_RE.fullmatch(args.stage) is None:
        raise ValueError("stage must use lowercase letters, digits, dot, underscore, or hyphen")
    worker_seed = os.environ.get("PI05_SEED")
    if worker_seed is not None and int(worker_seed) != args.seed:
        raise ValueError("evaluation seed differs from the immutable worker spec seed")
    if not args.policy_config.strip() or not args.model_revision.strip():
        raise ValueError("policy config and model revision must be non-empty")
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", args.model_revision) is None:
        raise ValueError("model revision must be a pinned 40- or 64-character revision")
    if not args.checkpoint.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {args.checkpoint}")
    if not 1 <= args.port <= 65535:
        raise ValueError("policy-server port must be between 1 and 65535")
    backend = getattr(args, "backend", "eager")
    if backend not in BACKENDS:
        raise ValueError(f"unsupported policy backend: {backend!r}")
    if backend == "tensorrt":
        validate_tensorrt_run_identity(args, source_commit=source_commit, image_digest=image_digest)
    elif any(
        getattr(args, name, None)
        for name in (
            "compiled_artifact_dir",
            "compiled_artifact_revision",
            "compiled_manifest_s3_uri",
            "compiled_manifest_version_id",
            "compiled_manifest_sha256",
            "compiled_payload_s3_uri",
            "precision",
            "dataset",
            "dataset_revision",
            "build_instance_id",
            "build_run_id",
            "engine_build_manifest_sha256",
        )
    ):
        raise ValueError("TensorRT-only identity arguments are invalid with the eager backend")
    return source_commit, image_digest, run_id


def _required_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty")
    return value


def _immutable_compiled_input(args: argparse.Namespace) -> dict[str, Any]:
    revision = _required_string(getattr(args, "compiled_artifact_revision", ""), label="compiled revision")
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision) is None:
        raise ValueError("compiled artifact revision must be a pinned 40- or 64-character revision")
    manifest_uri = _required_string(
        getattr(args, "compiled_manifest_s3_uri", ""), label="compiled descriptor manifest URI"
    )
    payload_uri = _required_string(getattr(args, "compiled_payload_s3_uri", ""), label="compiled payload URI")
    required_prefix = f"s3://{BUCKET}/"
    if not manifest_uri.startswith(required_prefix) or not payload_uri.startswith(required_prefix):
        raise ValueError("compiled artifact descriptor and payload must be in the pinned artifact bucket")
    manifest_sha256 = _required_string(
        getattr(args, "compiled_manifest_sha256", ""), label="compiled descriptor manifest SHA-256"
    )
    if SHA256_RE.fullmatch(manifest_sha256) is None:
        raise ValueError("compiled descriptor manifest SHA-256 is invalid")
    return {
        "revision": revision,
        "manifest": {
            "s3_uri": manifest_uri,
            "version_id": _required_string(
                getattr(args, "compiled_manifest_version_id", ""),
                label="compiled descriptor manifest VersionId",
            ),
            "sha256": manifest_sha256,
        },
        "payload_s3_uri": payload_uri,
    }


def validate_tensorrt_run_identity(
    args: argparse.Namespace, *, source_commit: str, image_digest: str
) -> dict[str, Any]:
    """Bind serving to the immutable artifact and exact engine-build runtime."""

    if args.instance_type != TENSORRT_INSTANCE_TYPE:
        raise ValueError(f"TensorRT LIBERO evaluation requires {TENSORRT_INSTANCE_TYPE}")
    if args.policy_config != TENSORRT_POLICY_CONFIG:
        raise ValueError(f"TensorRT LIBERO evaluation requires policy config {TENSORRT_POLICY_CONFIG}")
    if getattr(args, "precision", None) not in TENSORRT_PRECISIONS:
        raise ValueError("TensorRT precision must be bf16 or fp8")
    if (
        getattr(args, "dataset", None) != LIBERO_DATASET
        or getattr(args, "dataset_revision", None) != LIBERO_DATASET_REVISION
    ):
        raise ValueError("TensorRT LIBERO dataset identity differs from the pinned benchmark")
    instance_id = getattr(args, "build_instance_id", "")
    if not isinstance(instance_id, str) or INSTANCE_ID_RE.fullmatch(instance_id) is None:
        raise ValueError("TensorRT evaluation requires the exact EC2 engine-build instance ID")
    worker_instance_id = os.environ.get("PI05_INSTANCE_ID", "")
    if INSTANCE_ID_RE.fullmatch(worker_instance_id) is None:
        raise ValueError("TensorRT evaluation requires PI05_INSTANCE_ID from independently observed EC2 metadata")
    if worker_instance_id != instance_id:
        raise ValueError("current EC2 instance differs from the exact TensorRT engine-build instance")
    build_run_id = getattr(args, "build_run_id", "")
    if not isinstance(build_run_id, str) or STAGE_RE.fullmatch(build_run_id) is None:
        raise ValueError("TensorRT evaluation requires the pinned engine-build run ID")
    build_manifest_sha256 = getattr(args, "engine_build_manifest_sha256", "")
    if not isinstance(build_manifest_sha256, str) or SHA256_RE.fullmatch(build_manifest_sha256) is None:
        raise ValueError("TensorRT engine-build manifest SHA-256 is invalid")
    artifact_dir = getattr(args, "compiled_artifact_dir", None)
    if not isinstance(artifact_dir, pathlib.Path) or artifact_dir.is_symlink() or not artifact_dir.is_dir():
        raise ValueError(f"compiled TensorRT artifact directory does not exist or is unsafe: {artifact_dir}")
    manifest_path = artifact_dir / f"tensorrt-manifest.{args.precision}.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"TensorRT engine-build manifest is missing: {manifest_path}")
    if sha256_file(manifest_path) != build_manifest_sha256:
        raise ValueError("TensorRT engine-build manifest differs from the supplied build-instance contract")
    manifest = read_json_object(manifest_path, label="TensorRT engine-build manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("stage") != f"tensorrt-build-{args.precision}"
        or manifest.get("track") != "libero"
        or manifest.get("source") != {"sha": source_commit, "dirty": False}
        or manifest.get("runtime")
        != {
            "image_digest": image_digest,
            "instance_type": TENSORRT_INSTANCE_TYPE,
            "instance_id": instance_id,
        }
        or manifest.get("dataset") != {"name": LIBERO_DATASET, "revision": LIBERO_DATASET_REVISION}
    ):
        raise ValueError("TensorRT engine-build manifest identity differs from this evaluation")
    immutable_input = _immutable_compiled_input(args)
    return {
        **immutable_input,
        "track": "libero",
        "dataset": {"name": LIBERO_DATASET, "revision": LIBERO_DATASET_REVISION},
        "precision": args.precision,
        "engine_build_manifest_sha256": build_manifest_sha256,
        "build_run_id": build_run_id,
        "build_runtime": dict(manifest["runtime"]),
    }


def server_command(args: argparse.Namespace) -> list[str]:
    if getattr(args, "backend", "eager") == "tensorrt":
        return [
            str(TENSORRT_POLICY_PYTHON),
            "scripts/serve_tensorrt_policy.py",
            "--artifact-dir",
            str(args.compiled_artifact_dir),
            "--checkpoint-dir",
            str(args.checkpoint),
            "--precision",
            args.precision,
            "--track",
            "libero",
            "--dataset",
            args.dataset,
            "--dataset-revision",
            args.dataset_revision,
            "--image-digest",
            args.image_digest or os.environ.get("PI05_IMAGE_DIGEST", ""),
            "--instance-type",
            args.instance_type,
            "--instance-id",
            args.build_instance_id,
            "--port",
            str(args.port),
            "--seed",
            str(args.seed),
        ]
    return [
        sys.executable,
        "scripts/serve_policy.py",
        "--env",
        "LIBERO",
        "--port",
        str(args.port),
        "--seed",
        str(args.seed),
        "policy:checkpoint",
        "--policy.config",
        args.policy_config,
        "--policy.dir",
        str(args.checkpoint),
    ]


def evaluator_command(args: argparse.Namespace, suite: str, result_path: pathlib.Path) -> list[str]:
    return [
        str(args.evaluator_python),
        "examples/libero/main.py",
        "--args.host",
        "127.0.0.1",
        "--args.port",
        str(args.port),
        "--args.task-suite-name",
        suite,
        "--args.num-trials-per-task",
        str(args.trials_per_task),
        "--args.stage",
        args.stage,
        "--args.seed",
        str(args.seed),
        "--args.results-out-path",
        str(result_path),
        "--args.runtime-contract-path",
        str(args.runtime_contract),
        "--args.expected-libero-revision",
        LIBERO_REVISION,
        "--args.no-save-videos",
    ]


def evaluator_environment() -> dict[str, str]:
    environment = os.environ.copy()
    roots = ["/workspace/openpi/packages/openpi-client/src", "/opt/libero"]
    if existing := environment.get("PYTHONPATH"):
        roots.append(existing)
    environment["PYTHONPATH"] = ":".join(roots)
    environment["LIBERO_CONFIG_PATH"] = "/opt/libero-config"
    environment["MUJOCO_GL"] = "egl"
    environment["MUJOCO_EGL_DEVICE_ID"] = "0"
    environment["NVIDIA_DRIVER_CAPABILITIES"] = "compute,utility,graphics"
    environment["PYOPENGL_PLATFORM"] = "egl"
    return environment


def wait_for_server(process: subprocess.Popen[Any], *, port: int, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"policy server exited before readiness with code {return_code}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"policy server did not listen on loopback port {port} within {timeout_seconds}s")


def stop_server(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def load_and_validate_results(
    paths: list[pathlib.Path], *, args: argparse.Namespace
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    pair_ids: set[str] = set()
    suite_metrics: dict[str, dict[str, Any]] = {}
    for suite, path in zip(SUITES, paths, strict=True):
        suite_records = []
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"non-object JSONL record in {path}:{line_number}")
            if (
                record.get("suite") != suite
                or record.get("stage") != args.stage
                or record.get("seed") != args.seed
                or record.get("libero_revision") != LIBERO_REVISION
            ):
                raise ValueError(f"result identity mismatch in {path}:{line_number}")
            if "error" in record:
                raise ValueError(f"infrastructure-error record is not valid quality evidence in {path}:{line_number}")
            pair_id = record.get("pair_id")
            if not isinstance(pair_id, str) or pair_id in pair_ids:
                raise ValueError(f"missing or duplicate pair_id in {path}:{line_number}")
            pair_ids.add(pair_id)
            suite_records.append(record)
        expected = 10 * args.trials_per_task
        if len(suite_records) != expected:
            raise ValueError(f"{suite} produced {len(suite_records)} episodes; expected exactly {expected}")
        successes = sum(bool(record.get("success")) for record in suite_records)
        suite_metrics[suite] = {
            "episodes": len(suite_records),
            "successes": successes,
            "success_rate": successes / len(suite_records),
        }
        records.extend(suite_records)
    successes = sum(bool(record.get("success")) for record in records)
    return records, {
        "episodes": len(records),
        "successes": successes,
        "success_rate": successes / len(records),
        "environment_steps": sum(int(record.get("steps", 0)) for record in records),
        "infrastructure_errors": 0,
        "suites": suite_metrics,
    }


def write_combined_jsonl(path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def run_evaluation(args: argparse.Namespace) -> pathlib.Path:
    source_commit, image_digest, run_id = validate_run_identity(args)
    contract = validate_runtime_contract(args.runtime_contract)
    if not args.evaluator_python.is_file():
        raise ValueError(f"pinned LIBERO Python does not exist: {args.evaluator_python}")
    if getattr(args, "backend", "eager") == "tensorrt" and not TENSORRT_POLICY_PYTHON.is_file():
        raise ValueError(f"pinned TensorRT policy Python does not exist: {TENSORRT_POLICY_PYTHON}")
    if args.trials_per_task not in {1, 10, 50}:
        raise ValueError("trials-per-task must be 1 (runtime smoke), 10 (intermediate), or 50 (official)")
    if not math.isfinite(args.projected_cost_usd) or args.projected_cost_usd < 0:
        raise ValueError("projected cost must be finite and non-negative")

    artifact_dir = args.output_root / "artifacts" / "libero" / args.stage
    manifest_path = args.output_root / "manifests" / f"libero-{args.stage}.json"
    suite_paths = [artifact_dir / f"{suite}.jsonl" for suite in SUITES]
    combined_path = artifact_dir / "episodes.jsonl"
    collisions = [path for path in [*suite_paths, combined_path, manifest_path] if path.exists()]
    if collisions:
        raise FileExistsError(f"evaluation outputs already exist: {', '.join(map(str, collisions))}")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    policy_command = server_command(args)
    child_commands: list[list[str]] = [policy_command]
    started_at = dt.datetime.now(dt.UTC)
    server = subprocess.Popen(policy_command, start_new_session=True)
    try:
        wait_for_server(server, port=args.port, timeout_seconds=args.server_start_timeout_seconds)
        environment = evaluator_environment()
        for suite, result_path in zip(SUITES, suite_paths, strict=True):
            command = evaluator_command(args, suite, result_path)
            child_commands.append(command)
            completed = subprocess.run(command, check=False, env=environment)
            if completed.returncode != 0:
                raise RuntimeError(f"LIBERO evaluator failed for {suite} with code {completed.returncode}")
            server_return_code = server.poll()
            if server_return_code is not None:
                raise RuntimeError(f"policy server exited during {suite} with code {server_return_code}")
    finally:
        stop_server(server)

    records, metrics = load_and_validate_results(suite_paths, args=args)
    write_combined_jsonl(combined_path, records)
    artifacts = [
        {
            "path": path.relative_to(args.output_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in [*suite_paths, combined_path]
    ]
    backend = getattr(args, "backend", "eager")
    policy_identity: dict[str, Any] = {
        "backend": backend,
        "config": args.policy_config,
        "checkpoint": str(args.checkpoint),
        "model_revision": args.model_revision,
    }
    if backend == "tensorrt":
        policy_identity["compiled_artifact"] = validate_tensorrt_run_identity(
            args, source_commit=source_commit, image_digest=image_digest
        )
    compiled_manual_session = backend == "tensorrt"
    manifest = {
        "schema_version": 1,
        "project": PROJECT,
        "kind": "libero-evaluation",
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "finished_at": dt.datetime.now(dt.UTC).isoformat(),
        "source": {"commit": source_commit},
        "image": {"digest": image_digest},
        "dataset": {"name": "LIBERO fixed benchmark assets", "revision": LIBERO_REVISION},
        "simulator": contract["simulator"],
        "dependencies": contract["requirements"],
        "policy": policy_identity,
        "evaluation": {
            "stage": args.stage,
            "seed": args.seed,
            "suites": list(SUITES),
            "trials_per_task": args.trials_per_task,
            "metrics": metrics,
        },
        "command": list(sys.argv),
        "child_commands": child_commands,
        "instance": {
            "type": args.instance_type,
            "id": os.environ.get("PI05_INSTANCE_ID") or None,
            "identity_recorded_by": (
                "retained-session IMDS evidence" if compiled_manual_session else "worker run manifest"
            ),
        },
        "cost": {
            "projected_usd": args.projected_cost_usd,
            "actual_recorded_by": "retained-session ledger" if compiled_manual_session else "worker run manifest",
        },
        "artifacts": artifacts,
    }
    write_atomic_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "metrics": metrics}, indent=2, sort_keys=True))
    return manifest_path


def worker_artifact(
    path: pathlib.Path, *, expected_kind: str = "checkpoint", label: str = "checkpoint"
) -> dict[str, Any]:
    value = read_json_object(path, label=f"{label} worker artifact")
    if "worker_artifact" in value:
        value = value["worker_artifact"]
    if not isinstance(value, dict) or value.get("kind") != expected_kind:
        raise ValueError(f"{label} artifact file must contain a worker {expected_kind} descriptor")
    return value


def safe_artifact_destination(artifact: dict[str, Any], *, root: str, label: str) -> pathlib.PurePosixPath:
    destination = pathlib.PurePosixPath(str(artifact.get("destination", "")))
    if destination.is_absolute() or not destination.parts or ".." in destination.parts:
        raise ValueError(f"{label} artifact destination is unsafe")
    return pathlib.PurePosixPath(root) / destination


def build_instance_contract(path: pathlib.Path, *, args: argparse.Namespace, image_digest: str) -> dict[str, Any]:
    contract = read_json_object(path, label="TensorRT build-instance contract")
    expected_keys = {
        "schema_version",
        "kind",
        "execution_constraint",
        "build_run_id",
        "source_commit",
        "image_digest",
        "instance_type",
        "instance_id",
        "track",
        "dataset",
        "precision",
        "engine_build_manifest_sha256",
    }
    if set(contract) != expected_keys or contract.get("schema_version") != 1:
        raise ValueError("TensorRT build-instance contract schema differs")
    if contract.get("kind") != "pi05-tensorrt-build-instance" or contract.get("execution_constraint") != (
        "evaluate-before-exact-build-instance-stop"
    ):
        raise ValueError("TensorRT build-instance execution constraint differs")
    if STAGE_RE.fullmatch(str(contract.get("build_run_id", ""))) is None:
        raise ValueError("TensorRT build-instance contract has an invalid build run ID")
    expected = {
        "source_commit": args.source_commit,
        "image_digest": image_digest,
        "instance_type": TENSORRT_INSTANCE_TYPE,
        "track": "libero",
        "dataset": {"name": args.dataset, "revision": args.dataset_revision},
        "precision": args.precision,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"TensorRT build-instance contract differs for {key}")
    if INSTANCE_ID_RE.fullmatch(str(contract.get("instance_id", ""))) is None:
        raise ValueError("TensorRT build-instance contract has an invalid EC2 instance ID")
    if SHA256_RE.fullmatch(str(contract.get("engine_build_manifest_sha256", ""))) is None:
        raise ValueError("TensorRT build-instance contract has an invalid manifest SHA-256")
    return contract


def render_worker_spec(args: argparse.Namespace) -> dict[str, Any]:
    if STAGE_RE.fullmatch(args.stage) is None:
        raise ValueError("stage must use the worker-safe lowercase identifier format")
    if args.trials_per_task not in {1, 10, 50}:
        raise ValueError("trials-per-task must be 1, 10, or 50")
    if not math.isfinite(args.projected_cost_usd) or args.projected_cost_usd < 0:
        raise ValueError("projected cost must be finite and non-negative")
    backend = getattr(args, "backend", "eager")
    if backend not in BACKENDS:
        raise ValueError(f"unsupported policy backend: {backend!r}")
    expected_instance_type = "g6e.4xlarge" if backend == "eager" else TENSORRT_INSTANCE_TYPE
    if args.instance_type != expected_instance_type:
        raise ValueError(f"{backend} LIBERO evaluation workers must use {expected_instance_type}")
    image_match = IMAGE_RE.fullmatch(args.image_uri)
    if image_match is None:
        raise ValueError("evaluator image must be an account-local ECR URI pinned by digest")
    parent_match = IMAGE_RE.fullmatch(args.parent_policy_image)
    if parent_match is None or args.parent_policy_image == args.image_uri:
        raise ValueError("parent policy image must be a distinct account-local ECR URI pinned by digest")
    parent_compiler_image = getattr(args, "parent_tensorrt_compiler_image", "")
    parent_compiler_source = getattr(args, "parent_tensorrt_compiler_source_revision", "")
    checkpoint = worker_artifact(args.checkpoint_artifact)
    checkpoint_path = safe_artifact_destination(checkpoint, root="/mnt/openpi/checkpoints", label="checkpoint")
    artifacts = [checkpoint]
    compiled: dict[str, Any] | None = None
    build_contract: dict[str, Any] | None = None
    if backend == "tensorrt":
        if args.policy_config != TENSORRT_POLICY_CONFIG:
            raise ValueError(f"TensorRT LIBERO evaluation requires policy config {TENSORRT_POLICY_CONFIG}")
        if args.precision not in TENSORRT_PRECISIONS:
            raise ValueError("TensorRT precision must be bf16 or fp8")
        if args.dataset != LIBERO_DATASET or args.dataset_revision != LIBERO_DATASET_REVISION:
            raise ValueError("TensorRT LIBERO dataset identity differs from the pinned benchmark")
        if args.compiled_artifact is None or args.build_instance_contract is None:
            raise ValueError(
                "TensorRT rendering requires a compiled artifact descriptor and same-running-build-instance contract"
            )
        compiler_match = IMAGE_RE.fullmatch(parent_compiler_image)
        if (
            compiler_match is None
            or parent_compiler_image in {args.image_uri, args.parent_policy_image}
            or parent_compiler_source != args.source_commit
        ):
            raise ValueError(
                "TensorRT LIBERO evaluation requires a distinct pinned parent compiler and matching source revision"
            )
        compiled = worker_artifact(args.compiled_artifact, expected_kind="asset", label="compiled TensorRT")
        compiled_path = safe_artifact_destination(compiled, root="/mnt/openpi/assets", label="compiled TensorRT")
        if compiled_path == checkpoint_path:
            raise ValueError("compiled TensorRT and checkpoint destinations overlap")
        build_contract = build_instance_contract(
            args.build_instance_contract,
            args=args,
            image_digest=image_match.group("digest"),
        )
        artifacts.append(compiled)
    elif any(
        getattr(args, name, None)
        for name in (
            "compiled_artifact",
            "build_instance_contract",
            "precision",
            "dataset",
            "dataset_revision",
            "parent_tensorrt_compiler_image",
            "parent_tensorrt_compiler_source_revision",
        )
    ):
        raise ValueError("TensorRT-only renderer arguments are invalid with the eager backend")
    artifact_root = f"artifacts/libero/{args.stage}"
    expected_outputs = [
        {"name": f"suite_{suite.removeprefix('libero_')}", "kind": "artifact", "path": f"{artifact_root}/{suite}.jsonl"}
        for suite in SUITES
    ]
    expected_outputs.extend(
        [
            {"name": "episodes", "kind": "artifact", "path": f"{artifact_root}/episodes.jsonl"},
            {"name": "evaluation_manifest", "kind": "manifest", "path": f"manifests/libero-{args.stage}.json"},
        ]
    )
    policy_python = str(TENSORRT_POLICY_PYTHON) if backend == "tensorrt" else "python"
    command = [
        policy_python,
        "scripts/repro_libero_eval.py",
        "run",
        "--backend",
        backend,
        "--policy-config",
        args.policy_config,
        "--checkpoint",
        str(checkpoint_path),
        "--model-revision",
        str(checkpoint["revision"]),
        "--stage",
        args.stage,
        "--trials-per-task",
        str(args.trials_per_task),
        "--seed",
        str(args.seed),
        "--instance-type",
        args.instance_type,
        "--projected-cost-usd",
        str(args.projected_cost_usd),
        "--output-root",
        "/output",
    ]
    if backend == "tensorrt":
        assert compiled is not None
        assert build_contract is not None
        command.extend(
            [
                "--compiled-artifact-dir",
                str(compiled_path),
                "--compiled-artifact-revision",
                str(compiled["revision"]),
                "--compiled-manifest-s3-uri",
                str(compiled["manifest"]["s3_uri"]),
                "--compiled-manifest-version-id",
                str(compiled["manifest"]["version_id"]),
                "--compiled-manifest-sha256",
                str(compiled["manifest"]["sha256"]),
                "--compiled-payload-s3-uri",
                str(compiled["payload_s3_uri"]),
                "--precision",
                args.precision,
                "--dataset",
                args.dataset,
                "--dataset-revision",
                args.dataset_revision,
                "--build-instance-id",
                str(build_contract["instance_id"]),
                "--build-run-id",
                str(build_contract["build_run_id"]),
                "--engine-build-manifest-sha256",
                str(build_contract["engine_build_manifest_sha256"]),
            ]
        )
    image_contract: dict[str, Any] = {
        "uri": args.image_uri,
        "digest": image_match.group("digest"),
        "purpose": "libero-evaluator",
        "policy_backend": backend,
        "lerobot_runtime": "v2",
        "lerobot_revision": LEROBOT_V2_REVISION,
        "libero_simulator_revision": LIBERO_REVISION,
        "libero_requirements_sha256": LIBERO_REQUIREMENTS_SHA256,
        "parent_policy_image": args.parent_policy_image,
    }
    if backend == "tensorrt":
        image_contract.update(
            {
                "parent_tensorrt_compiler_image": parent_compiler_image,
                "parent_tensorrt_compiler_source_revision": parent_compiler_source,
                "toolchain": dict(TENSORRT_TOOLCHAIN),
            }
        )
    spec = {
        "schema_version": 1,
        "project": PROJECT,
        "run_id": args.run_id,
        "aws": {"account_id": ACCOUNT, "region": REGION, "artifact_bucket": BUCKET},
        "controller_source": {
            "s3_uri": args.controller_source_s3_uri,
            "version_id": args.controller_source_version_id,
            "sha256": args.controller_source_sha256,
            "commit": args.controller_source_commit,
        },
        "source": {
            "s3_uri": args.source_s3_uri,
            "version_id": args.source_version_id,
            "sha256": args.source_sha256,
            "commit": args.source_commit,
        },
        "image": image_contract,
        "artifacts": artifacts,
        "container": {
            "command": command,
            "environment": {
                "MUJOCO_GL": "egl",
                "MUJOCO_EGL_DEVICE_ID": "0",
                "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
                "PYOPENGL_PLATFORM": "egl",
            },
            "shm_size_gib": 32,
        },
        "expected_outputs": expected_outputs,
        "output": {"s3_uri": f"s3://{BUCKET}/runs/{args.run_id}/"},
        "timing": {"sync_interval_seconds": 60, "upload_buffer_seconds": 900, "stop_grace_seconds": 30},
        "scratch": {
            "model": "Amazon EC2 NVMe Instance Storage",
            "expected_count": 1,
            "ordinal": 0,
            "mount": "/mnt/openpi",
            "filesystem_label": "PI05_SCRATCH",
        },
        "seed": args.seed,
    }
    if backend == "tensorrt":
        assert build_contract is not None
        spec["placement"] = {
            "mode": "exact-existing-instance",
            "instance_id": str(build_contract["instance_id"]),
        }
    try:
        repro_worker = importlib.import_module("scripts.repro_worker")
    except ModuleNotFoundError as exc:  # Executing directly can place scripts/ first on sys.path.
        if exc.name != "scripts":
            raise
        repro_worker = importlib.import_module("repro_worker")

    return repro_worker.validate_worker_spec(spec)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def add_run_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("run", help="run policy server and all four LIBERO suites in one container")
    parser.add_argument("--policy-config", required=True)
    parser.add_argument("--backend", choices=BACKENDS, default="eager")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--trials-per-task", type=positive_int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--server-start-timeout-seconds", type=positive_int, default=900)
    parser.add_argument("--runtime-contract", type=pathlib.Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--evaluator-python", type=pathlib.Path, default=DEFAULT_EVALUATOR_PYTHON)
    parser.add_argument("--output-root", type=pathlib.Path, default=pathlib.Path("/output"))
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--image-digest", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--instance-type", default="local")
    parser.add_argument("--projected-cost-usd", type=float, default=0.0)
    parser.add_argument("--compiled-artifact-dir", type=pathlib.Path)
    parser.add_argument("--compiled-artifact-revision", default="")
    parser.add_argument("--compiled-manifest-s3-uri", default="")
    parser.add_argument("--compiled-manifest-version-id", default="")
    parser.add_argument("--compiled-manifest-sha256", default="")
    parser.add_argument("--compiled-payload-s3-uri", default="")
    parser.add_argument("--precision", choices=TENSORRT_PRECISIONS)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--dataset-revision", default="")
    parser.add_argument("--build-instance-id", default="")
    parser.add_argument("--build-run-id", default="")
    parser.add_argument("--engine-build-manifest-sha256", default="")
    parser.set_defaults(handler=run_evaluation)


def add_render_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "render-worker-spec",
        help=(
            "render and validate a network-none worker JSON spec; TensorRT output is a future, "
            "non-launchable exact-instance contract"
        ),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--controller-source-s3-uri", required=True)
    parser.add_argument("--controller-source-version-id", required=True)
    parser.add_argument("--controller-source-sha256", required=True)
    parser.add_argument("--controller-source-commit", required=True)
    parser.add_argument("--source-s3-uri", required=True)
    parser.add_argument("--source-version-id", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--parent-policy-image", required=True)
    parser.add_argument("--backend", choices=BACKENDS, default="eager")
    parser.add_argument("--checkpoint-artifact", type=pathlib.Path, required=True)
    parser.add_argument("--policy-config", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--trials-per-task", type=positive_int, choices=(1, 10, 50), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--instance-type", default="g6e.4xlarge", choices=("g6e.4xlarge", TENSORRT_INSTANCE_TYPE))
    parser.add_argument("--projected-cost-usd", type=float, required=True)
    parser.add_argument("--compiled-artifact", type=pathlib.Path)
    parser.add_argument("--build-instance-contract", type=pathlib.Path)
    parser.add_argument("--parent-tensorrt-compiler-image", default="")
    parser.add_argument("--parent-tensorrt-compiler-source-revision", default="")
    parser.add_argument("--precision", choices=TENSORRT_PRECISIONS)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--dataset-revision", default="")
    parser.add_argument("--output", type=pathlib.Path)

    def handler(args: argparse.Namespace) -> None:
        spec = render_worker_spec(args)
        payload = json.dumps(spec, indent=2, sort_keys=True) + "\n"
        if args.output is None:
            print(payload, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(payload)
            os.replace(temporary, args.output)

    parser.set_defaults(handler=handler)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_run_parser(subparsers)
    add_render_parser(subparsers)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
