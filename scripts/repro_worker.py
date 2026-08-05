#!/usr/bin/env python3
"""Validate and run an immutable, deadline-aware AWS reproduction worker.

The command is deliberately dry-run by default.  ``run --execute`` is the
only mode that formats instance-store storage, reads or writes S3, pulls an
image, or starts a container.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
import contextlib
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, ClassVar
import urllib.parse
import urllib.request

EXPECTED_PROJECT = "pi05-aws-repro"
EXPECTED_ACCOUNT = "752160877725"
EXPECTED_REGION = "us-east-2"
# DLAMIs initialize and mount instance-store NVMe here.  Keep the host mount
# distinct from the stable path exposed to containers and referenced by the
# reproduction training configurations.
SCRATCH_ROOT = pathlib.Path("/opt/dlami/nvme")
CONTAINER_INPUT_ROOT = pathlib.PurePosixPath("/mnt/openpi")
MODEL_SOURCE_CHECKOUT = pathlib.PurePosixPath("/opt/pi05/model-source")
CONTROLLER_SOURCE_CHECKOUT = pathlib.PurePosixPath("/opt/pi05/controller-source")
RESERVED_INPUT_MOUNTPOINTS = ("runs", "evidence")
INSTANCE_STORE_MODEL = "Amazon EC2 NVMe Instance Storage"
SCRATCH_LABEL = "PI05_SCRATCH"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
REVISION_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
ENV_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
DOCKER_HOSTNAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
INSTANCE_ID_RE = re.compile(r"^i-[0-9a-f]{17}$")
INSTANCE_TYPE_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,63}$")
OUTPUT_KIND_ROOT = {
    "artifact": "artifacts",
    "checkpoint": "checkpoints",
    "log": "logs",
    "manifest": "manifests",
}
INPUT_KIND_ROOT = {"asset": "assets", "checkpoint": "checkpoints", "dataset": "datasets"}
IMAGE_RE = re.compile(
    r"^752160877725\.dkr\.ecr\.us-east-2\.amazonaws\.com/"
    r"[a-z0-9._/-]+@(?P<digest>sha256:[0-9a-f]{64})$"
)
LEROBOT_REVISIONS = {
    "v2": "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5",
    "v3": "0b067df57d21d3a02d6c511f1609172fa39ac29b",
}
POLICY_VIDEO_DECODER = "pyav"
POLICY_ONNXRUNTIME_GPU_VERSION = "1.26.0"
IMAGE_PURPOSE_POLICY = "policy"
IMAGE_PURPOSE_LIBERO_EVALUATOR = "libero-evaluator"
IMAGE_PURPOSE_TENSORRT_COMPILER = "tensorrt-compiler"
IMAGE_PURPOSE_TENSORRT_POLICY = "tensorrt-policy"
LIBERO_SIMULATOR_REVISION = "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"
LIBERO_REQUIREMENTS_SHA256 = "124e74d09719941c9e3e75a61330808a8d32ae35a1ebee00c18e1222e966d0c8"
LIBERO_EVALUATOR_ENVIRONMENT = {
    "MUJOCO_EGL_DEVICE_ID": "0",
    "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
}
WORKER_OWNED_ENVIRONMENT = {
    "HOME",
    "LOGNAME",
    "PATH",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONPATH",
    "USER",
    "XDG_CACHE_HOME",
}
TORCHRUN_LOOPBACK_ENVIRONMENT = {
    "NCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    # The pinned NCCL 2.26.2 runtime predates the 2.26.5 automatic fallback
    # for Docker/NUMA-incompatible cuMem allocations. The G7e two-GPU pilot
    # reached P2P/CUMEM communicator initialization and then faulted both ranks.
    # Keep PCIe P2P enabled while selecting NCCL's legacy allocation path.
    "NCCL_CUMEM_ENABLE": "0",
    "NCCL_CUMEM_HOST_ENABLE": "0",
}
DROID_LAYOUT_CONTRACT = "molmoact2-v3-exact-media-references-v1"
DROID_CAMERA_FILE_COUNTS = {
    "observation.images.exterior_1_left": 518,
    "observation.images.wrist_left": 316,
}
DATASET_TRACK_CONTRACTS = {
    "libero": {
        "repo_id": "physical-intelligence/libero",
        "codebase_version": "v2.0",
        "local_dirname": "libero",
        "lerobot_runtime": "v2",
    },
    "droid": {
        "repo_id": "allenai/MolmoAct2-DROID-Dataset",
        "codebase_version": "v3.0",
        "local_dirname": "molmoact2-droid",
        "lerobot_runtime": "v3",
    },
}
TEACHER_TRACK_CONTRACTS = {
    "libero": {
        "source_uri": "gs://openpi-assets/checkpoints/pi05_libero",
        "source_local_dirname": "pi05_libero",
        "converted_local_dirname": "pi05_libero_pytorch",
        "config_name": "pi05_libero",
        "lerobot_runtime": "v2",
    },
    "droid_jointpos": {
        "source_uri": "gs://openpi-assets-simeval/pi05_droid_jointpos",
        "source_local_dirname": "pi05_droid_jointpos",
        "converted_local_dirname": "pi05_droid_jointpos_pytorch",
        "config_name": "pi05_droid_jointpos",
        "lerobot_runtime": "v3",
    },
}
TENSORRT_COMPILER_TOOLCHAIN = {
    "tensorrt_version": "11.0.0.114",
    "cuda_version": "13.3.0",
    "modelopt_version": "0.45.0",
    "torch_version": "2.8.0",
    "onnx_version": "1.21.0",
    "onnxruntime_gpu_version": "1.24.2",
}
TENSORRT_TOOLCHAIN_LABELS = {
    "tensorrt_version": "ai.openpi.tensorrt-version",
    "cuda_version": "ai.openpi.cuda-version",
    "modelopt_version": "ai.openpi.modelopt-version",
    "torch_version": "ai.openpi.torch-version",
    "onnx_version": "ai.openpi.onnx-version",
    "onnxruntime_gpu_version": "ai.openpi.onnxruntime-gpu-version",
}
TENSORRT_POLICY_LABELS = {
    "ai.openpi.parent-image-purpose": IMAGE_PURPOSE_TENSORRT_COMPILER,
    "ai.openpi.policy-runtime": "openpi-transform-websocket",
    "ai.openpi.policy-python": "/opt/modelopt/bin/python",
    "ai.openpi.policy-protocol": "openpi-policy-websocket-v1",
}
COMPILED_PIPELINE_WRITE_OPTION = {
    "scripts/export_pi05_onnx.py": "--output-dir",
    "scripts/validate_pi05_onnx.py": "--output",
    "scripts/quantize_pi05_fp8.py": "--artifact-dir",
    "scripts/build_tensorrt_engines.py": "--artifact-dir",
    "scripts/benchmark_pi05_latency.py": "--output",
}
RESUME_STATE_KEYS = {
    "schema_version",
    "global_step",
    "config_name",
    "exp_name",
    "resume_contract",
    "resume_fingerprint_sha256",
    "initialization_lineage",
    "state_files",
}
RESUME_CONTRACT_KEYS = {
    "schema_version",
    "config_name",
    "exp_name",
    "seed",
    "batch_size",
    "gradient_accumulation_steps",
    "lr_schedule",
    "optimizer",
    "wandb_enabled",
    "model",
    "pytorch_training_precision",
    "dataset",
    "teacher",
    "data_split",
    "stochastic_schedule",
    "one_batch_overfit",
    "one_batch_overfit_min_relative_decline",
}
TRAINING_METRICS_FILENAME = "training-metrics.json"
TRAINING_METRICS_KEYS = {"schema_version", "config_name", "exp_name", "global_step", "metrics"}
STAGE_METRICS_MANIFEST_KEYS = {
    "schema_version",
    "created_at",
    "stage",
    "track",
    "source",
    "runtime",
    "dataset",
    "experiment",
    "cost",
    "command",
    "metrics",
    "artifacts",
    "details",
}
LIBERO_METRICS_MANIFEST_KEYS = {
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
LATENCY_REPORT_KEYS = {
    "schema_version",
    "stage",
    "track",
    "official_protocol",
    "batch_size",
    "warmups",
    "iterations",
    "latency",
    "runner",
    "numerical_smoke",
    "gpu_inventory",
    "dataset",
    "benchmark_inputs",
    "source_artifacts",
    "runtime",
}
DATASET_CONTRACT_KEYS = {
    "factory",
    "factory_config_sha256",
    "repo_id",
    "revision",
    "codebase_version",
    "episode_prompt_path",
    "episode_prompt_sha256",
    "action_sequence_keys",
    "asset_id",
    "use_quantile_norm",
    "prompt_from_task",
    "normalization_sha256",
    "recovery_provenance_sha256",
}
UTC = getattr(dt, "UTC", dt.timezone.utc)  # noqa: UP017 -- supports the system Python on older AMIs.


class WorkerError(RuntimeError):
    """Raised when a worker invariant is not satisfied."""


class CommandError(WorkerError):
    """Raised when a required local or AWS command fails."""

    def __init__(self, argv: Sequence[str], returncode: int, stderr: str):
        detail = " ".join(stderr.strip().split())[-1500:]
        super().__init__(f"command failed ({returncode}): {shlex.join(argv)}: {detail}")
        self.argv = tuple(argv)
        self.returncode = returncode


CommandRunner = Any


class SubprocessRunner:
    """Injectable command adapter used by execution and unit tests."""

    def __call__(self, argv: Sequence[str]) -> str:
        completed = subprocess.run(list(argv), check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise CommandError(argv, completed.returncode, completed.stderr or completed.stdout)
        return completed.stdout.strip()


@dataclasses.dataclass(frozen=True)
class S3Location:
    bucket: str
    key: str

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


@dataclasses.dataclass(frozen=True)
class ScratchSelection:
    path: str
    serial: str
    reuse: bool
    mounted_at: str | None = None
    filesystem: str | None = None
    run_root: str | None = None


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise WorkerError(f"required JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise WorkerError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkerError(f"JSON document must be an object: {path}")
    return value


def _only_keys(value: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unexpected = set(value) - allowed
    if unexpected:
        raise WorkerError(f"unexpected {context} keys: {sorted(unexpected)}")


def canonical_resume_fingerprint(value: Any) -> str:
    try:
        payload = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    except (TypeError, ValueError) as exc:
        raise WorkerError("resume contract is not canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def resume_identity_fingerprint(contract: Mapping[str, Any], lineage: Mapping[str, Any]) -> str:
    return canonical_resume_fingerprint(
        {
            "initialization_lineage": lineage,
            "resume_contract": contract,
        }
    )


def _required_string(value: Mapping[str, Any], key: str, context: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result or "\x00" in result or "\n" in result or "\r" in result:
        raise WorkerError(f"{context}.{key} must be a non-empty single-line string")
    return result


def parse_s3_uri(uri: str, *, prefix: bool = False) -> S3Location:
    parsed = urllib.parse.urlsplit(uri)
    key = parsed.path.lstrip("/")
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not key
        or parsed.query
        or parsed.fragment
        or "//" in key
        or any(part in {"", ".", ".."} for part in pathlib.PurePosixPath(key).parts)
    ):
        kind = "prefix" if prefix else "object"
        raise WorkerError(f"invalid S3 {kind} URI: {uri!r}")
    if not prefix and uri.endswith("/"):
        raise WorkerError(f"S3 object URI must not end in '/': {uri!r}")
    return S3Location(parsed.netloc, key.rstrip("/") if prefix else key)


def _safe_relative_path(raw: str, context: str) -> pathlib.PurePosixPath:
    path = pathlib.PurePosixPath(raw)
    if not raw or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise WorkerError(f"{context} must be a safe relative POSIX path: {raw!r}")
    return path


def artifact_relative_destination(artifact: Mapping[str, Any]) -> pathlib.PurePosixPath:
    """Return the stable config-compatible path below /mnt/openpi."""

    destination = _safe_relative_path(str(artifact["destination"]), "artifact destination")
    try:
        root = INPUT_KIND_ROOT[str(artifact["kind"])]
    except KeyError as exc:
        raise WorkerError(f"unsupported artifact kind: {artifact.get('kind')!r}") from exc
    return pathlib.PurePosixPath(root) / destination


def _single_command_option(command: Sequence[str], name: str, context: str) -> str:
    positions = [index for index, item in enumerate(command) if item == name]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise WorkerError(f"{context} must contain exactly one {name} VALUE")
    return command[positions[0] + 1]


def _torchrun_processes_per_node(command: Sequence[str]) -> int | None:
    """Return the explicit local process count for a torchrun training argv."""

    train_positions = [index for index, item in enumerate(command) if item == "scripts/train_pytorch.py"]
    if len(train_positions) != 1:
        return None
    launcher = command[: train_positions[0]]
    torchrun_positions = [
        index for index, item in enumerate(launcher) if pathlib.PurePosixPath(item).name == "torchrun"
    ]
    if not torchrun_positions:
        return None
    if len(torchrun_positions) != 1:
        raise WorkerError("training worker command must invoke torchrun exactly once")

    option_names = ("--nproc-per-node", "--nproc_per_node")
    values: list[str] = []
    index = torchrun_positions[0] + 1
    while index < len(launcher):
        item = launcher[index]
        if item in option_names:
            if index + 1 >= len(launcher):
                raise WorkerError("torchrun --nproc-per-node requires a value")
            values.append(launcher[index + 1])
            index += 2
            continue
        matching_names = [name for name in option_names if item.startswith(f"{name}=")]
        if matching_names:
            values.append(item.split("=", 1)[1])
        index += 1
    if not values:
        return 1
    if len(values) != 1 or re.fullmatch(r"[1-9][0-9]*", values[0]) is None:
        raise WorkerError("torchrun --nproc-per-node must be one explicit positive integer")
    return int(values[0])


def _uses_multi_process_torchrun(command: Sequence[str]) -> bool:
    processes = _torchrun_processes_per_node(command)
    if processes is None or processes == 1:
        return False
    train_index = command.index("scripts/train_pytorch.py")
    if command[:train_index].count("--standalone") != 1:
        raise WorkerError("multi-process torchrun under network none requires --standalone exactly once")
    return True


SNAPFLOW_SOURCE_CONFIGS = {
    "pi05_libero_l09_snapflow": {"pi05_libero_l09_distill"},
    "pi05_droid_l09_snapflow": {
        "pi05_droid_l09_distill",
        "pi05_droid_l09_expert_bc_25",
        "pi05_droid_l09_expert_bc_50",
    },
}

SHALLOW_RESUME_CONFIGS = {
    "pi05_libero_l09_distill",
    "pi05_droid_l09_distill",
}
SHALLOW_RESUME_TRANSITIONS = {
    (2_000, 5_000),
    (5_000, 10_000),
    (10_000, 20_000),
    (20_000, 30_000),
}
SHALLOW_RESUME_ALLOWED_OPTIONS = {
    "--checkpoint-base-dir",
    "--exp-name",
    "--log-interval",
    "--no-wandb-enabled",
    "--num-train-steps",
    "--resume",
    "--save-interval",
    "--seed",
}
SNAPFLOW_ALLOWED_OPTIONS = {
    "--batch-size",
    "--checkpoint-base-dir",
    "--exp-name",
    "--gradient-accumulation-steps",
    "--log-interval",
    "--no-wandb-enabled",
    "--num-train-steps",
    "--num-workers",
    "--one-batch-overfit",
    "--one-batch-overfit-min-relative-decline",
    "--pytorch-training-precision",
    "--pytorch-weight-path",
    "--resume",
    "--save-interval",
    "--seed",
    "--teacher-pytorch-weight-path",
}


def _validate_snapflow_common_command(command: Sequence[str], training_config: str, context: str) -> None:
    """Reject ambiguous argv and recipe changes for every SnapFlow training stage."""

    if training_config not in SNAPFLOW_SOURCE_CONFIGS:
        raise WorkerError(f"{context} uses an unapproved SnapFlow config: {training_config!r}")
    train_index = command.index("scripts/train_pytorch.py")
    if train_index != 1 or pathlib.PurePosixPath(command[0]).name not in {"python", "python3"}:
        raise WorkerError(f"{context} must use a direct single-process Python argv")
    unreviewed_options = [value for value in command if value.startswith("-") and value not in SNAPFLOW_ALLOWED_OPTIONS]
    if unreviewed_options:
        raise WorkerError(f"{context} forbids unreviewed or equals-form options: {unreviewed_options}")
    for flag in ("--no-wandb-enabled", "--one-batch-overfit", "--resume"):
        if command.count(flag) > 1:
            raise WorkerError(f"{context} must not repeat {flag}")

    recipe_overrides = {
        "--batch-size": "4",
        "--gradient-accumulation-steps": "1",
        "--pytorch-training-precision": "bfloat16",
    }
    for option, expected in recipe_overrides.items():
        positions = [index for index, value in enumerate(command) if value == option]
        if positions and (
            len(positions) != 1 or positions[0] + 1 >= len(command) or command[positions[0] + 1] != expected
        ):
            raise WorkerError(f"{context} {option} override must preserve the published value {expected}")
    log_positions = [index for index, value in enumerate(command) if value == "--log-interval"]
    if log_positions and (
        len(log_positions) != 1 or log_positions[0] + 1 >= len(command) or command[log_positions[0] + 1] != "10"
    ):
        raise WorkerError(f"{context} --log-interval override must be exactly 10")
    if training_config == "pi05_droid_l09_snapflow" and (
        _single_command_option(command, "--num-workers", context) != "0" or command.count("--no-wandb-enabled") != 1
    ):
        raise WorkerError(f"{context} DROID runs require num_workers=0 and disabled W&B on g7e.2xlarge")


def validate_shallow_resume_training_contract(
    command: Sequence[str],
    training_config: str,
    *,
    source_step: int,
    target_step: int,
) -> None:
    """Keep every Shallow resume on the reviewed checkpoint ladder.

    The single-GPU corrective path is intentionally narrower than an ordinary
    two-process Shallow worker. Its argv is matched byte-for-byte after
    substituting the already validated config, experiment, seed, and target
    step. DROID keeps disabled W&B from its accepted 2k checkpoint but drops
    the diagnostic ``--num-workers 0`` override so the reviewed config's four
    deterministic loader workers can overlap real-video decoding.
    """

    context = "Shallow resume training command"
    if training_config not in SHALLOW_RESUME_CONFIGS:
        raise WorkerError(f"{context} uses an unapproved config: {training_config!r}")
    if (source_step, target_step) not in SHALLOW_RESUME_TRANSITIONS:
        raise WorkerError(f"{context} must follow the reviewed 2k->5k->10k->20k->30k ladder")

    train_index = command.index("scripts/train_pytorch.py")
    training_arguments = command[train_index + 2 :]
    equals_options = [value for value in training_arguments if value.startswith("-") and "=" in value]
    if equals_options:
        raise WorkerError(f"{context} forbids equals-form options: {equals_options}")
    allowed_negated_options = {"--no-wandb-enabled"} if training_config == "pi05_droid_l09_distill" else set()
    negated_options = [
        value for value in training_arguments if value.startswith("--no-") and value not in allowed_negated_options
    ]
    if negated_options:
        raise WorkerError(f"{context} forbids negated recipe options: {negated_options}")
    unreviewed_options = [
        value for value in training_arguments if value.startswith("-") and value not in SHALLOW_RESUME_ALLOWED_OPTIONS
    ]
    if unreviewed_options:
        raise WorkerError(f"{context} forbids unreviewed options: {unreviewed_options}")
    for option in SHALLOW_RESUME_ALLOWED_OPTIONS:
        maximum = 1
        if command.count(option) > maximum:
            raise WorkerError(f"{context} must not repeat {option}")

    if _single_command_option(command, "--save-interval", context) != "5000":
        raise WorkerError(f"{context} must retain the exact 5000-step save interval")
    if "--log-interval" in command and _single_command_option(command, "--log-interval", context) != "10":
        raise WorkerError(f"{context} --log-interval override must be exactly 10")

    processes = _torchrun_processes_per_node(command)
    if processes is not None:
        if processes != 2:
            raise WorkerError(f"{context} torchrun path requires exactly two local processes")
        return

    experiment = _single_command_option(command, "--exp-name", context)
    seed = _single_command_option(command, "--seed", context)
    expected = [
        "python",
        "scripts/train_pytorch.py",
        training_config,
        "--exp-name",
        experiment,
        "--checkpoint-base-dir",
        str(CONTAINER_INPUT_ROOT / "runs"),
        "--resume",
        "--seed",
        seed,
        "--num-train-steps",
        str(target_step),
        "--save-interval",
        "5000",
        "--log-interval",
        "10",
    ]
    if training_config == "pi05_droid_l09_distill":
        expected.append("--no-wandb-enabled")
    if list(command) != expected:
        raise WorkerError(f"{context} single-GPU corrective path must use the exact reviewed direct-Python argv")


def validate_fresh_snapflow_training_contract(
    command: Sequence[str],
    training_config: str,
    artifacts: Sequence[Mapping[str, Any]],
    expected_outputs: Sequence[Mapping[str, Any]],
) -> None:
    """Bind a fresh single-GPU SnapFlow run to one exact accepted Shallow checkpoint."""

    context = "fresh SnapFlow training command"
    _validate_snapflow_common_command(command, training_config, context)
    source_path = pathlib.PurePosixPath(_single_command_option(command, "--pytorch-weight-path", context))
    checkpoint_root = pathlib.PurePosixPath("/mnt/openpi/checkpoints")
    try:
        source_destination = source_path.relative_to(checkpoint_root)
    except ValueError as exc:
        raise WorkerError(f"{context} --pytorch-weight-path must be below {checkpoint_root}") from exc
    if len(source_destination.parts) != 3 or not source_destination.parts[2].isdigit():
        raise WorkerError(f"{context} source must be an exact CONFIG/EXPERIMENT/POSITIVE_STEP checkpoint")
    if int(source_destination.parts[2]) <= 0:
        raise WorkerError(f"{context} source checkpoint step must be positive")

    if source_destination.parts[0] not in SNAPFLOW_SOURCE_CONFIGS[training_config]:
        raise WorkerError(f"{context} source config is not an accepted Shallow checkpoint for {training_config}")
    matching_sources = [
        artifact
        for artifact in artifacts
        if artifact["kind"] == "checkpoint" and artifact["destination"] == source_destination.as_posix()
    ]
    if len(matching_sources) != 1:
        raise WorkerError(f"{context} must select exactly one staged accepted Shallow checkpoint")
    if "--teacher-pytorch-weight-path" in command:
        raise WorkerError(f"{context} must not load an external distillation teacher")

    experiment = _single_command_option(command, "--exp-name", context)
    try:
        target_step = int(_single_command_option(command, "--num-train-steps", context))
    except ValueError as exc:
        raise WorkerError(f"{context} --num-train-steps must be an integer") from exc
    if target_step <= 0:
        raise WorkerError(f"{context} --num-train-steps must be positive")
    expected_path = f"checkpoints/{training_config}/{experiment}/{target_step}"
    expected_destination = f"{training_config}/{experiment}/{target_step}"
    matching_outputs = [
        output for output in expected_outputs if output["kind"] == "checkpoint" and output["path"] == expected_path
    ]
    if len(matching_outputs) != 1:
        raise WorkerError(f"{context} must declare its exact numeric checkpoint output")

    if "--one-batch-overfit" in command:
        if command.count("--one-batch-overfit") != 1 or target_step != 300:
            raise WorkerError(f"{context} one-batch diagnostic must run exactly 300 optimizer steps")
        if _single_command_option(command, "--num-workers", context) != "0" or command.count("--no-wandb-enabled") != 1:
            raise WorkerError(f"{context} one-batch diagnostic requires num_workers=0 and disabled W&B")
        try:
            minimum_decline = float(
                _single_command_option(command, "--one-batch-overfit-min-relative-decline", context)
            )
        except ValueError as exc:
            raise WorkerError(f"{context} one-batch decline gate must be numeric") from exc
        if minimum_decline != 0.20:
            raise WorkerError(f"{context} one-batch diagnostic must retain the 20% decline gate")
        if matching_outputs[0].get("publish_destination") is not None:
            raise WorkerError(f"{context} one-batch diagnostic checkpoint must not be published as a worker input")
    else:
        if "--one-batch-overfit-min-relative-decline" in command:
            raise WorkerError(f"{context} pilot must not carry a one-batch decline override")
        if target_step != 5000:
            raise WorkerError(f"{context} initial pilot must target exactly 5000 optimizer steps")
        if (
            training_config == "pi05_libero_l09_snapflow"
            and "--num-workers" in command
            and _single_command_option(command, "--num-workers", context) != "4"
        ):
            raise WorkerError(f"{context} LIBERO pilot must retain the configured four loader workers")
        if matching_outputs[0].get("publish_destination") != expected_destination:
            raise WorkerError(f"{context} pilot must publish its exact numeric checkpoint for continuation")
    if _single_command_option(command, "--save-interval", context) != str(target_step):
        raise WorkerError(f"{context} must save exactly at its terminal diagnostic or pilot step")


def validate_snapflow_resume_training_contract(
    command: Sequence[str],
    training_config: str,
    *,
    source_step: int,
    target_step: int,
) -> None:
    """Keep every continued SnapFlow run on the reviewed single-GPU stage ladder."""

    context = "SnapFlow resume training command"
    _validate_snapflow_common_command(command, training_config, context)
    if "--pytorch-weight-path" in command or "--teacher-pytorch-weight-path" in command:
        raise WorkerError(f"{context} must restore full state without a model-only or teacher override")
    if "--one-batch-overfit" in command or "--one-batch-overfit-min-relative-decline" in command:
        raise WorkerError(f"{context} cannot be a one-batch diagnostic")
    allowed_transitions = {(5000, 10000), (10000, 20000), (20000, 30000)}
    if (source_step, target_step) not in allowed_transitions:
        raise WorkerError(f"{context} must follow the reviewed 5k->10k->20k->30k continuation ladder")
    if (
        training_config == "pi05_libero_l09_snapflow"
        and "--num-workers" in command
        and _single_command_option(command, "--num-workers", context) != "4"
    ):
        raise WorkerError(f"{context} LIBERO continuation must retain the configured four loader workers")
    if _single_command_option(command, "--save-interval", context) != "5000":
        raise WorkerError(f"{context} must retain the 5000-step checkpoint interval")


def validate_compiled_pipeline_command(spec: Mapping[str, Any]) -> None:
    """Keep hardware-bound compiler writes on the worker-owned output mount."""
    command = spec["container"]["command"]
    positions = [(index, item) for index, item in enumerate(command) if item in COMPILED_PIPELINE_WRITE_OPTION]
    purpose = spec["image"]["purpose"]
    if purpose == IMAGE_PURPOSE_TENSORRT_COMPILER and len(positions) != 1:
        raise WorkerError("TensorRT compiler workers must invoke one reviewed compile-pipeline script")
    if not positions:
        return
    if len(positions) != 1:
        raise WorkerError("compiled workers must invoke exactly one reviewed compile-pipeline script")
    script_index, script = positions[0]
    if script_index != 1 or command[0] not in {"python", "python3", "/opt/modelopt/bin/python"}:
        raise WorkerError("compile-pipeline scripts require a direct Python argv command without a shell")
    if "--help" in command:
        raise WorkerError("compile-pipeline workers must execute a stage, not its help command")
    if script == "scripts/benchmark_pi05_latency.py":
        backend = _single_command_option(command, "--backend", "latency benchmark command")
        expected_purpose = IMAGE_PURPOSE_POLICY if backend == "torch" else IMAGE_PURPOSE_TENSORRT_POLICY
        if backend not in {"torch", "tensorrt"} or purpose != expected_purpose:
            raise WorkerError("latency benchmark backend requires its exact eager or TensorRT policy image")
    elif purpose not in {IMAGE_PURPOSE_TENSORRT_COMPILER, IMAGE_PURPOSE_TENSORRT_POLICY}:
        raise WorkerError("compile-pipeline scripts require a TensorRT compiler or combined policy image")
    if script == "scripts/export_pi05_onnx.py" and purpose != IMAGE_PURPOSE_TENSORRT_POLICY:
        raise WorkerError("ONNX export requires the combined TensorRT policy image with LeRobot and transforms")

    placement = spec.get("placement")
    if not isinstance(placement, Mapping):
        raise WorkerError("compile-pipeline workers require exact existing-instance placement")
    if _single_command_option(command, "--instance-id", "compile-pipeline command") != placement["instance_id"]:
        raise WorkerError("compile-pipeline --instance-id must equal placement.instance_id")
    if _single_command_option(command, "--image-digest", "compile-pipeline command") != spec["image"]["digest"]:
        raise WorkerError("compile-pipeline --image-digest must equal the executing image digest")

    write_option = COMPILED_PIPELINE_WRITE_OPTION[script]
    write_path = pathlib.PurePosixPath(_single_command_option(command, write_option, "compile-pipeline command"))
    output_root = pathlib.PurePosixPath("/output/artifacts")
    try:
        relative_output = write_path.relative_to(output_root)
    except ValueError as exc:
        raise WorkerError(
            f"compile-pipeline {write_option} must be below writable /output/artifacts, not read-only /mnt/openpi"
        ) from exc
    if not relative_output.parts:
        raise WorkerError(
            f"compile-pipeline {write_option} must select a stage-specific output below /output/artifacts"
        )
    declared = [pathlib.PurePosixPath(item["path"]) for item in spec.get("expected_outputs", [])]
    output_relative_to_worker = pathlib.PurePosixPath("artifacts") / relative_output
    if not any(
        candidate == output_relative_to_worker or output_relative_to_worker in candidate.parents
        for candidate in declared
    ):
        raise WorkerError("compile-pipeline writable path is not covered by expected_outputs")
    if script == "scripts/build_tensorrt_engines.py" and command.count("--execute") != 1:
        raise WorkerError("TensorRT engine build workers require --execute exactly once")


def validate_worker_spec(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete worker contract and return a detached copy."""

    spec = json.loads(json.dumps(raw))
    _only_keys(
        spec,
        {
            "schema_version",
            "project",
            "run_id",
            "aws",
            "controller_source",
            "source",
            "image",
            "artifacts",
            "container",
            "resume_checkpoint",
            "expected_outputs",
            "output",
            "timing",
            "scratch",
            "seed",
            "placement",
        },
        "worker spec",
    )
    if spec.get("schema_version") != 1 or spec.get("project") != EXPECTED_PROJECT:
        raise WorkerError("worker spec schema/project mismatch")
    run_id = _required_string(spec, "run_id", "worker spec")
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise WorkerError(f"invalid run_id: {run_id!r}")
    if not isinstance(spec.get("seed"), int) or isinstance(spec.get("seed"), bool) or spec["seed"] < 0:
        raise WorkerError("worker spec.seed must be a non-negative integer")

    aws = spec.get("aws")
    if not isinstance(aws, dict):
        raise WorkerError("worker spec.aws must be an object")
    _only_keys(aws, {"account_id", "region", "artifact_bucket"}, "aws")
    if aws.get("account_id") != EXPECTED_ACCOUNT or aws.get("region") != EXPECTED_REGION:
        raise WorkerError("worker AWS account/region differs from the reproduction boundary")
    bucket = _required_string(aws, "artifact_bucket", "aws")

    def validate_source_pin(name: str) -> dict[str, Any]:
        value = spec.get(name)
        if not isinstance(value, dict):
            raise WorkerError(f"worker spec.{name} must be an object")
        _only_keys(value, {"s3_uri", "version_id", "sha256", "commit"}, name)
        location = parse_s3_uri(_required_string(value, "s3_uri", name))
        if location.bucket != bucket:
            raise WorkerError(f"{name} bundle must be in the pinned artifact bucket")
        _required_string(value, "version_id", name)
        if SHA256_RE.fullmatch(_required_string(value, "sha256", name)) is None:
            raise WorkerError(f"{name}.sha256 must be a lowercase SHA-256")
        if COMMIT_RE.fullmatch(_required_string(value, "commit", name)) is None:
            raise WorkerError(f"{name}.commit must be a full lowercase git commit")
        return value

    validate_source_pin("controller_source")
    source = validate_source_pin("source")

    image = spec.get("image")
    if not isinstance(image, dict):
        raise WorkerError("worker spec.image must be an object")
    image_uri = _required_string(image, "uri", "image")
    match = IMAGE_RE.fullmatch(image_uri)
    if match is None or image.get("digest") != match.group("digest"):
        raise WorkerError("image must be the project ECR URI pinned by the matching sha256 digest")
    purpose = _required_string(image, "purpose", "image")
    common_image_keys = {"uri", "digest", "purpose"}
    if purpose == IMAGE_PURPOSE_POLICY:
        _only_keys(image, common_image_keys | {"lerobot_runtime", "lerobot_revision"}, "policy image")
        runtime = _required_string(image, "lerobot_runtime", "image")
        revision = _required_string(image, "lerobot_revision", "image")
        if LEROBOT_REVISIONS.get(runtime) != revision:
            raise WorkerError("policy image must pin the approved LeRobot v2 or v3 runtime revision")
    elif purpose == IMAGE_PURPOSE_LIBERO_EVALUATOR:
        evaluator_keys = common_image_keys | {
            "policy_backend",
            "lerobot_runtime",
            "lerobot_revision",
            "libero_simulator_revision",
            "libero_requirements_sha256",
            "parent_policy_image",
        }
        backend = _required_string(image, "policy_backend", "image")
        if backend == "eager":
            _only_keys(image, evaluator_keys, "eager LIBERO evaluator image")
        elif backend == "tensorrt":
            _only_keys(
                image,
                evaluator_keys
                | {
                    "parent_tensorrt_compiler_image",
                    "parent_tensorrt_compiler_source_revision",
                    "toolchain",
                },
                "TensorRT LIBERO evaluator image",
            )
            compiler_image = _required_string(image, "parent_tensorrt_compiler_image", "image")
            compiler_source = _required_string(image, "parent_tensorrt_compiler_source_revision", "image")
            toolchain = image.get("toolchain")
            if (
                IMAGE_RE.fullmatch(compiler_image) is None
                or compiler_image in {image_uri, image.get("parent_policy_image")}
                or compiler_source != source["commit"]
                or not isinstance(toolchain, dict)
                or toolchain != TENSORRT_COMPILER_TOOLCHAIN
            ):
                raise WorkerError(
                    "TensorRT LIBERO evaluator must pin its distinct compiler, matching source, and complete toolchain"
                )
        else:
            raise WorkerError("LIBERO evaluator image.policy_backend must be eager or tensorrt")
        if image.get("lerobot_runtime") != "v2" or image.get("lerobot_revision") != LEROBOT_REVISIONS["v2"]:
            raise WorkerError("LIBERO evaluator image must pin the approved LeRobot v2 runtime revision")
        if image.get("libero_simulator_revision") != LIBERO_SIMULATOR_REVISION:
            raise WorkerError("LIBERO evaluator image must pin the approved simulator revision")
        if image.get("libero_requirements_sha256") != LIBERO_REQUIREMENTS_SHA256:
            raise WorkerError("LIBERO evaluator image must pin the approved dependency lock SHA-256")
        parent_policy_image = _required_string(image, "parent_policy_image", "image")
        if IMAGE_RE.fullmatch(parent_policy_image) is None or parent_policy_image == image_uri:
            raise WorkerError("LIBERO evaluator parent must be a distinct account-local policy image pinned by digest")
    elif purpose == IMAGE_PURPOSE_TENSORRT_COMPILER:
        _only_keys(image, common_image_keys | {"toolchain"}, "TensorRT compiler image")
        toolchain = image.get("toolchain")
        if not isinstance(toolchain, dict):
            raise WorkerError("TensorRT compiler image.toolchain must be an object")
        _only_keys(toolchain, set(TENSORRT_COMPILER_TOOLCHAIN), "TensorRT compiler image.toolchain")
        if toolchain != TENSORRT_COMPILER_TOOLCHAIN:
            raise WorkerError("TensorRT compiler image must pin the approved complete toolchain")
    elif purpose == IMAGE_PURPOSE_TENSORRT_POLICY:
        _only_keys(
            image,
            common_image_keys
            | {
                "lerobot_runtime",
                "lerobot_revision",
                "parent_tensorrt_compiler_image",
                "parent_tensorrt_compiler_source_revision",
                "toolchain",
            },
            "TensorRT policy image",
        )
        runtime = _required_string(image, "lerobot_runtime", "image")
        revision = _required_string(image, "lerobot_revision", "image")
        compiler_image = _required_string(image, "parent_tensorrt_compiler_image", "image")
        compiler_source = _required_string(image, "parent_tensorrt_compiler_source_revision", "image")
        toolchain = image.get("toolchain")
        if LEROBOT_REVISIONS.get(runtime) != revision:
            raise WorkerError("TensorRT policy image must pin the exact approved LeRobot v2 or v3 revision")
        if (
            IMAGE_RE.fullmatch(compiler_image) is None
            or compiler_image == image_uri
            or compiler_source != source["commit"]
            or not isinstance(toolchain, dict)
            or toolchain != TENSORRT_COMPILER_TOOLCHAIN
        ):
            raise WorkerError("TensorRT policy image must pin its compiler digest/source and complete toolchain")
    else:
        raise WorkerError(
            "image.purpose must be one of "
            f"{IMAGE_PURPOSE_POLICY!r}, {IMAGE_PURPOSE_LIBERO_EVALUATOR!r}, "
            f"{IMAGE_PURPOSE_TENSORRT_COMPILER!r}, or {IMAGE_PURPOSE_TENSORRT_POLICY!r}"
        )

    placement = spec.get("placement")
    if placement is not None:
        if not isinstance(placement, dict):
            raise WorkerError("worker spec.placement must be an object")
        _only_keys(placement, {"mode", "instance_id"}, "placement")
        if (
            placement.get("mode") != "exact-existing-instance"
            or INSTANCE_ID_RE.fullmatch(str(placement.get("instance_id", ""))) is None
        ):
            raise WorkerError("placement must pin one exact-existing-instance EC2 instance_id")
    if purpose == IMAGE_PURPOSE_LIBERO_EVALUATOR and image.get("policy_backend") == "tensorrt" and placement is None:
        raise WorkerError("TensorRT LIBERO evaluation requires exact existing-instance placement")

    artifacts = spec.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise WorkerError("worker spec.artifacts must be a non-empty list")
    names: set[str] = set()
    destinations: set[str] = set()
    for index, artifact in enumerate(artifacts):
        context = f"artifacts[{index}]"
        if not isinstance(artifact, dict):
            raise WorkerError(f"{context} must be an object")
        _only_keys(
            artifact,
            {"name", "kind", "revision", "manifest", "payload_s3_uri", "payload_objects", "destination"},
            context,
        )
        name = _required_string(artifact, "name", context)
        if NAME_RE.fullmatch(name) is None or name in names:
            raise WorkerError(f"{context}.name is invalid or duplicated: {name!r}")
        names.add(name)
        kind = artifact.get("kind")
        if kind not in {"dataset", "checkpoint", "asset"}:
            raise WorkerError(f"{context}.kind must be dataset, checkpoint, or asset")
        revision = _required_string(artifact, "revision", context)
        if REVISION_RE.fullmatch(revision) is None or (kind == "dataset" and COMMIT_RE.fullmatch(revision) is None):
            raise WorkerError(f"{context}.revision is not a pinned revision")
        destination = _safe_relative_path(_required_string(artifact, "destination", context), f"{context}.destination")
        artifact["destination"] = destination.as_posix()
        logical_destination = pathlib.PurePosixPath(INPUT_KIND_ROOT[str(kind)]) / destination
        if any(
            logical_destination == existing
            or logical_destination in existing.parents
            or existing in logical_destination.parents
            for existing in map(pathlib.PurePosixPath, destinations)
        ):
            raise WorkerError(f"overlapping artifact destination: {logical_destination}")
        destinations.add(logical_destination.as_posix())
        manifest = artifact.get("manifest")
        if not isinstance(manifest, dict):
            raise WorkerError(f"{context}.manifest must be an object")
        _only_keys(manifest, {"s3_uri", "version_id", "sha256"}, f"{context}.manifest")
        manifest_location = parse_s3_uri(_required_string(manifest, "s3_uri", f"{context}.manifest"))
        if manifest_location.bucket != bucket:
            raise WorkerError(f"{context} manifest must be in the pinned artifact bucket")
        _required_string(manifest, "version_id", f"{context}.manifest")
        if SHA256_RE.fullmatch(_required_string(manifest, "sha256", f"{context}.manifest")) is None:
            raise WorkerError(f"{context}.manifest.sha256 must be a lowercase SHA-256")
        payload = parse_s3_uri(_required_string(artifact, "payload_s3_uri", context), prefix=True)
        if payload.bucket != bucket:
            raise WorkerError(f"{context} payload must be in the pinned artifact bucket")
        payload_objects = artifact.get("payload_objects")
        if payload_objects is not None:
            if not isinstance(payload_objects, list) or not payload_objects:
                raise WorkerError(f"{context}.payload_objects must be a non-empty list when present")
            normalized_payload_objects: list[dict[str, str]] = []
            payload_paths: set[str] = set()
            for object_index, payload_object in enumerate(payload_objects):
                object_context = f"{context}.payload_objects[{object_index}]"
                if not isinstance(payload_object, dict):
                    raise WorkerError(f"{object_context} must be an object")
                _only_keys(payload_object, {"path", "version_id", "sha256"}, object_context)
                object_path = _safe_relative_path(
                    _required_string(payload_object, "path", object_context), f"{object_context}.path"
                ).as_posix()
                if object_path in payload_paths:
                    raise WorkerError(f"{context}.payload_objects contains a duplicate path: {object_path}")
                payload_paths.add(object_path)
                version_id = _required_string(payload_object, "version_id", object_context)
                digest = _required_string(payload_object, "sha256", object_context)
                if SHA256_RE.fullmatch(digest) is None:
                    raise WorkerError(f"{object_context}.sha256 must be a lowercase SHA-256")
                normalized_payload_objects.append({"path": object_path, "version_id": version_id, "sha256": digest})
            artifact["payload_objects"] = sorted(normalized_payload_objects, key=lambda item: item["path"])

    image_runtime = image.get("lerobot_runtime")
    for artifact in artifacts:
        expected_runtime = None
        if artifact["kind"] == "dataset":
            dataset_runtime_by_destination = {"libero": "v2", "molmoact2-droid": "v3"}
            expected_runtime = dataset_runtime_by_destination.get(artifact["destination"])
            if expected_runtime is None:
                raise WorkerError(f"unsupported dataset destination: {artifact['destination']!r}")
        elif artifact["kind"] == "checkpoint":
            destination_head = pathlib.PurePosixPath(artifact["destination"]).parts[0]
            if destination_head.startswith("pi05_libero"):
                expected_runtime = "v2"
            elif destination_head.startswith("pi05_droid"):
                expected_runtime = "v3"
        if expected_runtime is not None and image_runtime != expected_runtime:
            raise WorkerError(
                f"artifact {artifact['name']!r} requires LeRobot {expected_runtime}, found {image_runtime!r}"
            )

    container = spec.get("container")
    if not isinstance(container, dict):
        raise WorkerError("worker spec.container must be an object")
    _only_keys(container, {"command", "environment", "shm_size_gib"}, "container")
    command = container.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(part, str) or not part or "\x00" in part for part in command)
    ):
        raise WorkerError("container.command must be a non-empty argv string list")
    uses_multi_process_torchrun = _uses_multi_process_torchrun(command)
    environment = container.get("environment", {})
    if not isinstance(environment, dict):
        raise WorkerError("container.environment must be an object")
    for key, value in environment.items():
        if (
            not isinstance(key, str)
            or ENV_RE.fullmatch(key) is None
            or key.startswith(("AWS_", "DOCKER_", "PI05_"))
            or key in WORKER_OWNED_ENVIRONMENT
            or (uses_multi_process_torchrun and key in TORCHRUN_LOOPBACK_ENVIRONMENT)
            or not isinstance(value, str)
            or "\x00" in value
        ):
            raise WorkerError(f"unsafe container environment entry: {key!r}")
    libero_eval_positions = [index for index, item in enumerate(command) if item == "scripts/repro_libero_eval.py"]
    if purpose == IMAGE_PURPOSE_LIBERO_EVALUATOR:
        if (
            len(libero_eval_positions) != 1
            or libero_eval_positions[0] + 1 >= len(command)
            or command[libero_eval_positions[0] + 1] != "run"
        ):
            raise WorkerError("LIBERO evaluator image must invoke scripts/repro_libero_eval.py run exactly once")
        for key, expected in LIBERO_EVALUATOR_ENVIRONMENT.items():
            if environment.get(key) != expected:
                raise WorkerError(f"LIBERO evaluator container.environment must set {key}={expected}")
        backend = _single_command_option(command, "--backend", "LIBERO evaluator command")
        if backend != image["policy_backend"]:
            raise WorkerError("LIBERO evaluator command backend differs from image.policy_backend")
        if _single_command_option(command, "--output-root", "LIBERO evaluator command") != "/output":
            raise WorkerError("LIBERO evaluator command must write through --output-root /output")
        if backend == "tensorrt":
            assert isinstance(placement, dict)
            if (
                _single_command_option(command, "--build-instance-id", "TensorRT LIBERO evaluator command")
                != placement["instance_id"]
            ):
                raise WorkerError("TensorRT LIBERO build instance must equal exact placement.instance_id")
    elif libero_eval_positions:
        raise WorkerError("LIBERO evaluation command requires the dedicated libero-evaluator image purpose")
    shm_size = container.get("shm_size_gib", 32)
    if not isinstance(shm_size, int) or isinstance(shm_size, bool) or not 1 <= shm_size <= 512:
        raise WorkerError("container.shm_size_gib must be an integer from 1 through 512")

    train_positions = [index for index, item in enumerate(command) if item == "scripts/train_pytorch.py"]
    if train_positions:
        if len(train_positions) != 1 or train_positions[0] + 1 >= len(command):
            raise WorkerError("training worker must invoke scripts/train_pytorch.py exactly once")
        training_config = command[train_positions[0] + 1]
        expected_runtime = "v2" if training_config.startswith("pi05_libero") else None
        if training_config.startswith("pi05_droid"):
            expected_runtime = "v3"
        if expected_runtime is not None and image_runtime != expected_runtime:
            raise WorkerError(
                f"training config {training_config!r} requires LeRobot {expected_runtime}, found {image_runtime!r}"
            )
        seed_positions = [index for index, item in enumerate(command) if item == "--seed"]
        if len(seed_positions) != 1 or seed_positions[0] + 1 >= len(command):
            raise WorkerError("training worker command must contain exactly one --seed VALUE")
        try:
            command_seed = int(command[seed_positions[0] + 1])
        except ValueError as exc:
            raise WorkerError("training worker --seed must be an integer") from exc
        if command_seed != spec["seed"]:
            raise WorkerError("training worker --seed must equal worker spec.seed")
        checkpoint_base_positions = [index for index, item in enumerate(command) if item == "--checkpoint-base-dir"]
        if len(checkpoint_base_positions) != 1 or checkpoint_base_positions[0] + 1 >= len(command):
            raise WorkerError("training worker command must contain exactly one --checkpoint-base-dir VALUE")
        if command[checkpoint_base_positions[0] + 1] != str(CONTAINER_INPUT_ROOT / "runs"):
            raise WorkerError("training worker --checkpoint-base-dir must be /mnt/openpi/runs")

    output = spec.get("output")
    if not isinstance(output, dict):
        raise WorkerError("worker spec.output must be an object")
    _only_keys(output, {"s3_uri"}, "output")
    output_location = parse_s3_uri(_required_string(output, "s3_uri", "output"), prefix=True)
    if output_location.bucket != bucket or output_location.key != f"runs/{run_id}":
        raise WorkerError(f"output.s3_uri must be s3://{bucket}/runs/{run_id}/")

    expected_outputs = spec.get("expected_outputs", [])
    if not isinstance(expected_outputs, list):
        raise WorkerError("worker spec.expected_outputs must be a list")
    output_names: set[str] = set()
    output_paths: set[str] = set()
    for index, expected in enumerate(expected_outputs):
        context = f"expected_outputs[{index}]"
        if not isinstance(expected, dict):
            raise WorkerError(f"{context} must be an object")
        _only_keys(expected, {"name", "kind", "path", "publish_destination"}, context)
        name = _required_string(expected, "name", context)
        if NAME_RE.fullmatch(name) is None or name in output_names:
            raise WorkerError(f"{context}.name is invalid or duplicated: {name!r}")
        output_names.add(name)
        kind = expected.get("kind")
        if kind not in OUTPUT_KIND_ROOT:
            raise WorkerError(f"{context}.kind must be one of {sorted(OUTPUT_KIND_ROOT)}")
        path = _safe_relative_path(_required_string(expected, "path", context), f"{context}.path")
        if len(path.parts) < 2 or path.parts[0] != OUTPUT_KIND_ROOT[kind]:
            raise WorkerError(f"{context}.path must be below {OUTPUT_KIND_ROOT[kind]}/")
        if path.as_posix() in output_paths:
            raise WorkerError(f"duplicate expected output path: {path}")
        output_paths.add(path.as_posix())
        expected["path"] = path.as_posix()
        publish_destination = expected.get("publish_destination")
        if publish_destination is not None:
            if kind not in {"checkpoint", "artifact"}:
                raise WorkerError(f"{context}.publish_destination is supported only for checkpoint/artifact")
            destination = _safe_relative_path(str(publish_destination), f"{context}.publish_destination")
            expected["publish_destination"] = destination.as_posix()
    spec["expected_outputs"] = expected_outputs
    validate_compiled_pipeline_command(spec)

    resume = spec.get("resume_checkpoint")
    if resume is None and train_positions and training_config.endswith("_snapflow"):
        validate_fresh_snapflow_training_contract(command, training_config, artifacts, expected_outputs)
    if resume is not None:
        if not isinstance(resume, dict):
            raise WorkerError("worker spec.resume_checkpoint must be an object")
        _only_keys(resume, {"artifact_name", "target"}, "resume_checkpoint")
        artifact_name = _required_string(resume, "artifact_name", "resume_checkpoint")
        matching_artifacts = [artifact for artifact in artifacts if artifact["name"] == artifact_name]
        if len(matching_artifacts) != 1 or matching_artifacts[0]["kind"] != "checkpoint":
            raise WorkerError("resume_checkpoint.artifact_name must select exactly one checkpoint input")
        target = _safe_relative_path(
            _required_string(resume, "target", "resume_checkpoint"), "resume_checkpoint.target"
        )
        if len(target.parts) != 3 or not target.parts[-1].isdigit() or int(target.parts[-1]) <= 0:
            raise WorkerError("resume_checkpoint.target must be CONFIG/EXPERIMENT/POSITIVE_STEP")
        if matching_artifacts[0]["destination"] != target.as_posix():
            raise WorkerError("resume checkpoint input destination must exactly match resume_checkpoint.target")
        resume["target"] = target.as_posix()

        def command_option(name: str) -> str:
            positions = [index for index, item in enumerate(command) if item == name]
            if len(positions) != 1 or positions[0] + 1 >= len(command):
                raise WorkerError(f"resume worker command must contain exactly one {name} VALUE")
            return command[positions[0] + 1]

        if len(train_positions) != 1 or train_positions[0] + 1 >= len(command):
            raise WorkerError("resume worker must invoke scripts/train_pytorch.py exactly once")
        if command[train_positions[0] + 1] != target.parts[0]:
            raise WorkerError("resume worker training config differs from resume_checkpoint.target")
        if command_option("--exp-name") != target.parts[1]:
            raise WorkerError("resume worker experiment differs from resume_checkpoint.target")
        if command_option("--checkpoint-base-dir") != str(CONTAINER_INPUT_ROOT / "runs"):
            raise WorkerError("resume worker checkpoint base must be /mnt/openpi/runs")
        if command.count("--resume") != 1 or "--overwrite" in command:
            raise WorkerError("resume worker command requires --resume exactly once and forbids --overwrite")
        try:
            target_step = int(command_option("--num-train-steps"))
        except ValueError as exc:
            raise WorkerError("resume worker --num-train-steps must be an integer") from exc
        source_step = int(target.parts[-1])
        if target_step <= source_step:
            raise WorkerError("resume worker target step must be greater than the restored source step")
        expected_path = f"checkpoints/{target.parts[0]}/{target.parts[1]}/{target_step}"
        expected_destination = f"{target.parts[0]}/{target.parts[1]}/{target_step}"
        matching_outputs = [
            item
            for item in expected_outputs
            if item["kind"] == "checkpoint"
            and item["path"] == expected_path
            and item.get("publish_destination") == expected_destination
        ]
        if len(matching_outputs) != 1:
            raise WorkerError(
                "resume worker must declare one published checkpoint output at its exact target config/experiment/step"
            )
        if training_config.endswith("_snapflow"):
            validate_snapflow_resume_training_contract(
                command,
                training_config,
                source_step=source_step,
                target_step=target_step,
            )
        elif training_config in SHALLOW_RESUME_CONFIGS:
            validate_shallow_resume_training_contract(
                command,
                training_config,
                source_step=source_step,
                target_step=target_step,
            )
    elif train_positions and ("--resume" in command or "--overwrite" in command):
        raise WorkerError("fresh training worker commands forbid --resume and --overwrite; use a unique experiment ID")

    timing = spec.get("timing")
    if not isinstance(timing, dict):
        raise WorkerError("worker spec.timing must be an object")
    _only_keys(timing, {"sync_interval_seconds", "upload_buffer_seconds", "stop_grace_seconds"}, "timing")
    sync_interval = timing.get("sync_interval_seconds")
    upload_buffer = timing.get("upload_buffer_seconds")
    stop_grace = timing.get("stop_grace_seconds")
    if not isinstance(sync_interval, int) or not 10 <= sync_interval <= 600:
        raise WorkerError("timing.sync_interval_seconds must be from 10 through 600")
    if not isinstance(upload_buffer, int) or not 300 <= upload_buffer <= 7200:
        raise WorkerError("timing.upload_buffer_seconds must be from 300 through 7200")
    if not isinstance(stop_grace, int) or not 5 <= stop_grace <= 120 or stop_grace >= upload_buffer:
        raise WorkerError("timing.stop_grace_seconds must be 5..120 and smaller than the upload buffer")

    scratch = spec.get("scratch")
    if not isinstance(scratch, dict):
        raise WorkerError("worker spec.scratch must be an object")
    _only_keys(scratch, {"model", "expected_count", "ordinal", "mount", "filesystem_label"}, "scratch")
    if scratch.get("model") != INSTANCE_STORE_MODEL:
        raise WorkerError(f"scratch.model must be exactly {INSTANCE_STORE_MODEL!r}")
    if scratch.get("mount") != str(CONTAINER_INPUT_ROOT) or scratch.get("filesystem_label") != SCRATCH_LABEL:
        raise WorkerError("scratch mount or filesystem label differs from the safety contract")
    expected_count = scratch.get("expected_count")
    ordinal = scratch.get("ordinal")
    if not isinstance(expected_count, int) or isinstance(expected_count, bool) or expected_count < 1:
        raise WorkerError("scratch.expected_count must be a positive integer")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or not 0 <= ordinal < expected_count:
        raise WorkerError("scratch.ordinal must select exactly one expected device")
    return spec


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_evidence(spec: Mapping[str, Any], evidence: Mapping[str, Any]) -> None:
    expected = spec["source"]
    expected_controller = spec["controller_source"]
    if evidence.get("schema_version") != 2:
        raise WorkerError("source verification evidence has the wrong schema")
    for key in ("s3_uri", "version_id", "sha256", "commit"):
        if evidence.get("source", {}).get(key) != expected[key]:
            raise WorkerError(f"source evidence does not match the worker spec for {key}")
        if evidence.get("controller_source", {}).get(key) != expected_controller[key]:
            raise WorkerError(f"controller source evidence does not match the worker spec for {key}")
    if (
        evidence.get("bundle_sha256_actual") != expected["sha256"]
        or evidence.get("head_commit") != expected["commit"]
        or evidence.get("source_clean") is not True
    ):
        raise WorkerError("source bundle hash or checked-out commit was not verified")
    if (
        evidence.get("controller_bundle_sha256_actual") != expected_controller["sha256"]
        or evidence.get("controller_head_commit") != expected_controller["commit"]
        or evidence.get("controller_source_clean") is not True
    ):
        raise WorkerError("controller bundle hash or checked-out commit was not verified")
    if evidence.get("source_fsck_full") is not True or evidence.get("controller_source_fsck_full") is not True:
        raise WorkerError("model and controller source checkouts must pass a full Git object-integrity check")
    if (
        pathlib.PurePosixPath(str(evidence.get("checkout_path", ""))) != MODEL_SOURCE_CHECKOUT
        or pathlib.PurePosixPath(str(evidence.get("controller_checkout_path", ""))) != CONTROLLER_SOURCE_CHECKOUT
    ):
        raise WorkerError("model and controller evidence must name the fixed distinct checkout paths")


def validate_launch_metadata(
    spec: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    now: dt.datetime | None = None,
    command_path: pathlib.Path | None = None,
) -> tuple[dt.datetime, dt.datetime]:
    if metadata.get("project") != EXPECTED_PROJECT:
        raise WorkerError("launch metadata project mismatch")
    try:
        hard_deadline = dt.datetime.fromisoformat(str(metadata["deadline_utc"]))
    except (KeyError, ValueError) as exc:
        raise WorkerError("launch metadata has no valid hard deadline") from exc
    if hard_deadline.tzinfo is None:
        raise WorkerError("launch hard deadline must be timezone-aware")
    hard_deadline = hard_deadline.astimezone(UTC)
    soft_deadline = hard_deadline - dt.timedelta(seconds=spec["timing"]["upload_buffer_seconds"])
    current = (now or dt.datetime.now(UTC)).astimezone(UTC)
    if current >= soft_deadline:
        raise WorkerError("worker cannot start at or beyond its upload-buffer deadline")
    command_hash = metadata.get("command_sha256")
    if not isinstance(command_hash, str) or SHA256_RE.fullmatch(command_hash) is None:
        raise WorkerError("launch metadata has no valid command hash")
    if command_path is not None and sha256_file(command_path) != command_hash:
        raise WorkerError("launch metadata command hash differs from the executing command file")
    if metadata.get("purchase_option") != "On-Demand" or metadata.get("instance_count") != 1:
        raise WorkerError("launch metadata does not prove one On-Demand instance")
    container = spec.get("container")
    command = container.get("command") if isinstance(container, Mapping) else None
    direct_single_gpu_shallow_resume = (
        spec.get("resume_checkpoint") is not None
        and isinstance(command, list)
        and len(command) >= 3
        and command[:2] == ["python", "scripts/train_pytorch.py"]
        and command[2] in SHALLOW_RESUME_CONFIGS
    )
    if direct_single_gpu_shallow_resume:
        expected_launch = {
            "category": "corrective_run",
            "workload": "shallow_training",
            "instance_type": "g7e.4xlarge",
        }
        if any(metadata.get(key) != value for key, value in expected_launch.items()):
            raise WorkerError("single-GPU Shallow resume requires corrective_run/shallow_training on g7e.4xlarge")
    reservation_id = metadata.get("reservation_id")
    if not isinstance(reservation_id, str) or UUID_RE.fullmatch(reservation_id) is None:
        raise WorkerError("launch metadata has no valid cost-ledger reservation ID")
    projected = metadata.get("projected_compute_usd")
    reserved_hours = metadata.get("reserved_hours")
    if (
        isinstance(projected, bool)
        or not isinstance(projected, int | float)
        or not math.isfinite(projected)
        or projected < 0
        or isinstance(reserved_hours, bool)
        or not isinstance(reserved_hours, int | float)
        or not math.isfinite(reserved_hours)
        or reserved_hours <= 0
    ):
        raise WorkerError("launch metadata has invalid projected cost or reserved hours")
    return hard_deadline, soft_deadline


def _flatten_devices(devices: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    def visit(device: Mapping[str, Any], top_path: str) -> None:
        copied = dict(device)
        copied["_top_path"] = top_path
        result.append(copied)
        for child in device.get("children") or []:
            visit(child, top_path)

    for device in devices:
        path = str(device.get("path") or f"/dev/{device.get('name', '')}")
        visit(device, path)
    return result


def select_instance_store_device(
    lsblk_document: Mapping[str, Any], root_source: str, *, expected_count: int, ordinal: int
) -> ScratchSelection:
    """Select one raw instance-store disk while proving it is not the root disk."""

    devices = lsblk_document.get("blockdevices")
    if not isinstance(devices, list):
        raise WorkerError("lsblk did not return a blockdevices list")
    flat = _flatten_devices(devices)
    # Ubuntu commonly reports the root source as /dev/root.  Prefer the block
    # device that lsblk itself reports mounted at / and fall back to matching
    # the supplied source only when mountpoint evidence is unavailable.
    root_matches = [device for device in flat if "/" in (device.get("mountpoints") or [])]
    root_basename = pathlib.PurePath(root_source).name
    source_matches = [
        device
        for device in flat
        if root_source in {str(device.get("path")), f"/dev/{device.get('name', '')}"}
        or root_basename in {str(device.get("name")), str(device.get("kname"))}
    ]
    if not root_matches:
        root_matches = source_matches
    if len(root_matches) != 1:
        raise WorkerError(f"could not resolve the root filesystem block device exactly once: {root_source!r}")
    root_top = root_matches[0]["_top_path"]

    candidates = []
    for device in flat:
        model = " ".join(str(device.get("model") or "").split())
        if device.get("type") == "disk" and model == INSTANCE_STORE_MODEL:
            candidates.append(device)
    if len(candidates) != expected_count:
        raise WorkerError(
            f"expected exactly {expected_count} instance-store NVMe disks, found {len(candidates)}; refusing to format"
        )
    serials = [str(device.get("serial") or "").strip() for device in candidates]
    if any(not serial for serial in serials) or len(serials) != len(set(serials)):
        raise WorkerError("instance-store devices do not have unique non-empty serial numbers")
    candidates.sort(key=lambda device: str(device["serial"]))
    selected = candidates[ordinal]
    path = str(selected.get("path") or "")
    if not path.startswith("/dev/") or selected["_top_path"] == root_top:
        raise WorkerError("selected scratch disk resolves to the root block-device tree")
    selected_tree = [device for device in flat if device["_top_path"] == selected["_top_path"]]
    mounted_nodes = [
        (device, [str(item) for item in (device.get("mountpoints") or []) if item])
        for device in selected_tree
        if any(device.get("mountpoints") or [])
    ]
    fstype = str(selected.get("fstype") or "")
    label = str(selected.get("label") or "")
    if mounted_nodes:
        if len(mounted_nodes) != 1:
            raise WorkerError("selected instance-store tree has multiple mounted filesystems")
        mounted_device, mountpoints = mounted_nodes[0]
        mounted_fstype = str(mounted_device.get("fstype") or "")
        mounted_path = str(mounted_device.get("path") or f"/dev/{mounted_device.get('name', '')}")
        direct_mount = mounted_device.get("path") == selected.get("path")
        descendants = [device for device in selected_tree if device.get("path") != selected.get("path")]
        direct_layout_ok = direct_mount and not descendants
        dlami_lvm_layout_ok = (
            not direct_mount
            and fstype == "LVM2_member"
            and len(descendants) == 1
            and mounted_device.get("type") == "lvm"
            and not mounted_device.get("children")
        )
        if (
            mountpoints != [str(SCRATCH_ROOT)]
            or mounted_fstype not in {"ext4", "xfs"}
            or not mounted_path.startswith("/dev/")
            or not (direct_layout_ok or dlami_lvm_layout_ok)
        ):
            raise WorkerError(
                "selected instance-store disk is mounted outside the verified DLAMI scratch contract: "
                f"mountpoints={mountpoints}, fstype={mounted_fstype!r}, device={mounted_path!r}"
            )
        # A mounted filesystem is never passed to mkfs or mount.  Its device,
        # filesystem, and target are verified again with findmnt before use.
        return ScratchSelection(
            path=mounted_path,
            serial=str(selected["serial"]),
            reuse=True,
            mounted_at=str(SCRATCH_ROOT),
            filesystem=mounted_fstype,
        )
    if selected.get("children"):
        raise WorkerError("selected scratch disk has unmounted partitions or child mappings")
    if not fstype and not label:
        reuse = False
    elif fstype == "ext4" and label == SCRATCH_LABEL:
        reuse = True
    else:
        raise WorkerError(
            f"selected instance-store disk is not blank or an owned scratch filesystem: fstype={fstype!r}, label={label!r}"
        )
    return ScratchSelection(
        path=path,
        serial=str(selected["serial"]),
        reuse=reuse,
        filesystem=fstype or None,
    )


def scratch_command_plan(selection: ScratchSelection) -> list[list[str]]:
    if selection.mounted_at is not None:
        return []
    commands: list[list[str]] = []
    if not selection.reuse:
        commands.append(["mkfs.ext4", "-q", "-L", SCRATCH_LABEL, selection.path])
    commands.append(["mount", "-o", "noatime,nosuid,nodev", selection.path, str(SCRATCH_ROOT)])
    return commands


def create_run_workspace(
    mount_root: pathlib.Path,
    run_id: str,
    *,
    expected_owner_uid: int = 0,
) -> pathlib.Path:
    """Create one fresh, container-isolated workspace on the verified scratch mount."""

    if RUN_ID_RE.fullmatch(run_id) is None:
        raise WorkerError(f"invalid run ID for scratch workspace: {run_id!r}")
    if mount_root.is_symlink() or not mount_root.is_dir():
        raise WorkerError(f"scratch mount root is missing, not a directory, or a symlink: {mount_root}")

    runs_root = mount_root / "pi05-runs"
    if runs_root.exists() or runs_root.is_symlink():
        if runs_root.is_symlink() or not runs_root.is_dir():
            raise WorkerError(f"scratch runs root is not a safe directory: {runs_root}")
        if runs_root.stat().st_uid != expected_owner_uid:
            raise WorkerError(f"scratch runs root is not owned by root: {runs_root}")
    else:
        runs_root.mkdir(mode=0o755)

    run_root = runs_root / run_id
    try:
        run_root.mkdir(mode=0o755)
    except FileExistsError as exc:
        raise WorkerError(f"scratch workspace already exists; refusing stale run reuse: {run_root}") from exc

    root_owned = (
        run_root / "inputs",
        run_root / "inputs/runs",
        run_root / "inputs/evidence",
        run_root / "output",
        run_root / "output/.ready",
        run_root / "output/.receipts",
        run_root / "output/.spool",
        run_root / "output/.active",
    )
    container_owned = (
        run_root / "output/checkpoints",
        run_root / "output/logs",
        run_root / "output/manifests",
        run_root / "output/artifacts",
        run_root / "tmp",
        run_root / "cache",
    )
    for directory in (*root_owned, *container_owned):
        directory.mkdir(mode=0o700 if directory in root_owned else 0o755, parents=True)
    for directory in root_owned:
        directory.chmod(0o700)
    run_root.chmod(0o755)
    for directory in container_owned:
        directory.chmod(0o755)
        os.chown(directory, 1000, 1000)
    return run_root


def prepare_scratch(spec: Mapping[str, Any], runner: CommandRunner) -> ScratchSelection:
    if os.geteuid() != 0:
        raise WorkerError("executed worker must run as root to initialize instance-store storage")
    root_source = runner(["findmnt", "-n", "-o", "SOURCE", "/"]).strip()
    try:
        lsblk = json.loads(
            runner(
                [
                    "lsblk",
                    "--json",
                    "--bytes",
                    "--output",
                    "NAME,KNAME,PATH,TYPE,MODEL,SERIAL,FSTYPE,LABEL,MOUNTPOINTS,PKNAME",
                ]
            )
        )
    except json.JSONDecodeError as exc:
        raise WorkerError("lsblk returned invalid JSON") from exc
    scratch = spec["scratch"]
    selection = select_instance_store_device(
        lsblk,
        root_source,
        expected_count=scratch["expected_count"],
        ordinal=scratch["ordinal"],
    )
    if SCRATCH_ROOT.is_symlink():
        raise WorkerError(f"DLAMI scratch root must not be a symlink: {SCRATCH_ROOT}")
    SCRATCH_ROOT.mkdir(mode=0o755, parents=True, exist_ok=True)
    for command in scratch_command_plan(selection):
        runner(command)
    mounted_source = runner(["findmnt", "-n", "-o", "SOURCE", "--target", str(SCRATCH_ROOT)]).strip()
    selected_realpath = runner(["readlink", "-f", selection.path]).strip()
    mounted_realpath = runner(["readlink", "-f", mounted_source]).strip()
    if (
        not selected_realpath.startswith("/dev/")
        or not mounted_realpath.startswith("/dev/")
        or mounted_realpath != selected_realpath
    ):
        raise WorkerError(f"scratch mount source mismatch: expected {selection.path}, found {mounted_source}")
    mounted_filesystem = runner(["findmnt", "-n", "-o", "FSTYPE", "--target", str(SCRATCH_ROOT)]).strip()
    if mounted_filesystem not in {"ext4", "xfs"}:
        raise WorkerError(f"scratch filesystem is not an approved local filesystem: {mounted_filesystem!r}")
    if selection.mounted_at is not None and mounted_filesystem != selection.filesystem:
        raise WorkerError(
            f"scratch filesystem changed during verification: expected {selection.filesystem}, found {mounted_filesystem}"
        )
    run_root = create_run_workspace(SCRATCH_ROOT, str(spec["run_id"]))
    return dataclasses.replace(
        selection,
        mounted_at=str(SCRATCH_ROOT),
        filesystem=mounted_filesystem,
        run_root=str(run_root),
    )


def _json_command(runner: CommandRunner, argv: Sequence[str]) -> dict[str, Any]:
    output = runner(argv)
    try:
        value = json.loads(output or "{}")
    except json.JSONDecodeError as exc:
        raise WorkerError(f"command returned invalid JSON: {shlex.join(argv)}") from exc
    if not isinstance(value, dict):
        raise WorkerError(f"command returned non-object JSON: {shlex.join(argv)}")
    return value


def verify_aws_boundary(spec: Mapping[str, Any], runner: CommandRunner) -> None:
    aws = spec["aws"]
    identity = _json_command(
        runner,
        ["aws", "sts", "get-caller-identity", "--region", EXPECTED_REGION, "--output", "json"],
    )
    if identity.get("Account") != EXPECTED_ACCOUNT:
        raise WorkerError(f"refusing AWS account {identity.get('Account')!r}")
    bucket = aws["artifact_bucket"]
    common = ["--bucket", bucket, "--expected-bucket-owner", EXPECTED_ACCOUNT, "--region", EXPECTED_REGION]
    location = _json_command(runner, ["aws", "s3api", "get-bucket-location", *common, "--output", "json"])
    if location.get("LocationConstraint") != EXPECTED_REGION:
        raise WorkerError("artifact bucket is not in the pinned region")
    versioning = _json_command(runner, ["aws", "s3api", "get-bucket-versioning", *common, "--output", "json"])
    if versioning.get("Status") != "Enabled":
        raise WorkerError("artifact bucket versioning is not enabled")
    encryption = _json_command(runner, ["aws", "s3api", "get-bucket-encryption", *common, "--output", "json"])
    if not encryption.get("ServerSideEncryptionConfiguration", {}).get("Rules"):
        raise WorkerError("artifact bucket default encryption is not enabled")


def download_versioned_object(
    pin: Mapping[str, Any], destination: pathlib.Path, spec: Mapping[str, Any], runner: CommandRunner
) -> None:
    location = parse_s3_uri(str(pin["s3_uri"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as stream:
        temporary = pathlib.Path(stream.name)
    try:
        runner(
            [
                "aws",
                "s3api",
                "get-object",
                "--bucket",
                location.bucket,
                "--key",
                location.key,
                "--version-id",
                str(pin["version_id"]),
                "--expected-bucket-owner",
                spec["aws"]["account_id"],
                "--region",
                spec["aws"]["region"],
                str(temporary),
            ]
        )
        actual = sha256_file(temporary)
        if actual != pin["sha256"]:
            raise WorkerError(f"versioned S3 object failed SHA-256 validation: {location.uri}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def validate_artifact_manifest(
    manifest: Mapping[str, Any], artifact: Mapping[str, Any], destination: pathlib.Path | None = None
) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != 1 or manifest.get("source", {}).get("revision") != artifact["revision"]:
        raise WorkerError(f"{artifact['name']} manifest does not match its pinned source revision")
    published = manifest.get("artifact")
    if published is not None:
        expected_kind = {"checkpoint": "checkpoint", "artifact": "asset"}
        if (
            not isinstance(published, dict)
            or published.get("name") != artifact["name"]
            or expected_kind.get(str(published.get("kind"))) != artifact["kind"]
            or published.get("publish_destination") != artifact["destination"]
            or published.get("payload_s3_uri") != artifact["payload_s3_uri"]
        ):
            raise WorkerError(f"{artifact['name']} manifest publication path, destination, or kind was changed")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise WorkerError(f"{artifact['name']} manifest has no files")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(files):
        if not isinstance(item, dict) or not {"path", "bytes", "sha256"} <= set(item):
            raise WorkerError(f"{artifact['name']} manifest file {index} is invalid")
        path = _safe_relative_path(str(item.get("path", "")), f"{artifact['name']} manifest path")
        size = item.get("bytes")
        digest = item.get("sha256")
        if path.as_posix() in seen or not isinstance(size, int) or size < 0 or not isinstance(digest, str):
            raise WorkerError(f"{artifact['name']} manifest file {index} is invalid or duplicated")
        if SHA256_RE.fullmatch(digest) is None:
            raise WorkerError(f"{artifact['name']} manifest file {index} has an invalid SHA-256")
        seen.add(path.as_posix())
        normalized.append({"path": path.as_posix(), "bytes": size, "sha256": digest})
    totals = manifest.get("totals", {})
    if totals.get("files") != len(normalized) or totals.get("bytes") != sum(item["bytes"] for item in normalized):
        raise WorkerError(f"{artifact['name']} manifest totals are inconsistent")
    if destination is not None:
        actual: dict[str, pathlib.Path] = {}
        for path in destination.rglob("*"):
            if path.is_symlink():
                raise WorkerError(f"staged artifact contains a symlink: {path}")
            if path.is_file():
                actual[path.relative_to(destination).as_posix()] = path
        if set(actual) != seen:
            raise WorkerError(
                f"staged {artifact['name']} object set differs from the versioned manifest; "
                f"missing={sorted(seen - set(actual))[:10]}, extra={sorted(set(actual) - seen)[:10]}"
            )
        for item in normalized:
            path = actual[item["path"]]
            if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
                raise WorkerError(f"staged {artifact['name']} file failed validation: {item['path']}")
    return normalized


def _require_track_runtime(spec: Mapping[str, Any], expected_runtime: str, context: str) -> None:
    actual_runtime = spec["image"].get("lerobot_runtime")
    if actual_runtime != expected_runtime:
        raise WorkerError(f"{context} requires the LeRobot {expected_runtime} image runtime, found {actual_runtime!r}")


def _track_from_worker_output_destination(artifact: Mapping[str, Any]) -> str | None:
    head = pathlib.PurePosixPath(str(artifact["destination"])).parts[0]
    if head.startswith("pi05_libero"):
        return "libero"
    if head.startswith("pi05_droid"):
        return "droid_jointpos"
    return None


def validate_artifact_track_contract(
    manifest: Mapping[str, Any], artifact: Mapping[str, Any], spec: Mapping[str, Any]
) -> None:
    """Reject weak or cross-track manifests before their payload can run."""

    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise WorkerError(f"{artifact['name']} manifest source provenance is missing")

    if artifact["kind"] == "dataset":
        dataset = manifest.get("dataset")
        if source.get("provider") != "huggingface" or not isinstance(dataset, Mapping):
            raise WorkerError(f"{artifact['name']} dataset manifest is not a validated Hugging Face snapshot")
        track = dataset.get("key")
        contract = DATASET_TRACK_CONTRACTS.get(str(track))
        if contract is None:
            raise WorkerError(f"{artifact['name']} dataset manifest has an unsupported track: {track!r}")
        if (
            source.get("repo_id") != contract["repo_id"]
            or dataset.get("codebase_version") != contract["codebase_version"]
            or dataset.get("local_dirname") != contract["local_dirname"]
            or artifact["destination"] != contract["local_dirname"]
        ):
            raise WorkerError(f"{artifact['name']} dataset manifest identity differs from the {track} contract")
        _require_track_runtime(spec, str(contract["lerobot_runtime"]), f"{track} dataset")
        if track == "droid":
            validation = manifest.get("validation")
            if not isinstance(validation, Mapping) or validation.get("layout_contract") != DROID_LAYOUT_CONTRACT:
                raise WorkerError("DROID dataset manifest lacks the exact MolmoAct2 v3 layout contract")
            for field in ("required_video_files_by_feature", "expected_video_files_by_feature"):
                if validation.get(field) != DROID_CAMERA_FILE_COUNTS:
                    raise WorkerError(f"DROID dataset manifest {field} does not contain the exact camera counts")
            if validation.get("required_video_files") != sum(DROID_CAMERA_FILE_COUNTS.values()):
                raise WorkerError("DROID dataset manifest total required camera files is inconsistent")
        return

    if artifact["kind"] != "checkpoint":
        return
    provider = source.get("provider")
    checkpoint = manifest.get("checkpoint")
    if provider == "pi05-worker-output":
        track = _track_from_worker_output_destination(artifact)
        if track is not None:
            _require_track_runtime(
                spec,
                str(TEACHER_TRACK_CONTRACTS[track]["lerobot_runtime"]),
                f"{track} worker checkpoint",
            )
        return
    if provider not in {"gcs", "openpi-jax-to-pytorch"} or not isinstance(checkpoint, Mapping):
        raise WorkerError(f"{artifact['name']} checkpoint manifest has unsupported provenance")
    track = checkpoint.get("key")
    contract = TEACHER_TRACK_CONTRACTS.get(str(track))
    if contract is None:
        raise WorkerError(f"{artifact['name']} checkpoint manifest has an unsupported teacher track: {track!r}")
    _require_track_runtime(spec, str(contract["lerobot_runtime"]), f"{track} teacher")
    expected_local_dirname = (
        contract["source_local_dirname"] if provider == "gcs" else contract["converted_local_dirname"]
    )
    if checkpoint.get("local_dirname") != expected_local_dirname or artifact["destination"] != expected_local_dirname:
        raise WorkerError(f"{artifact['name']} checkpoint destination differs from its {track} provenance")
    if provider == "gcs":
        if (
            source.get("uri") != contract["source_uri"]
            or not isinstance(source.get("objects"), list)
            or not source.get("objects")
        ):
            raise WorkerError(f"{artifact['name']} original JAX teacher provenance is incomplete")
        return
    conversion = manifest.get("conversion")
    upstream = source.get("upstream")
    if (
        not isinstance(conversion, Mapping)
        or conversion.get("config_name") != contract["config_name"]
        or not isinstance(upstream, Mapping)
        or upstream.get("provider") != "gcs"
        or upstream.get("uri") != contract["source_uri"]
        or SHA256_RE.fullmatch(str(upstream.get("revision", ""))) is None
    ):
        raise WorkerError(f"{artifact['name']} converted teacher config/upstream provenance is invalid")
    selected_originals = [
        candidate
        for candidate in spec["artifacts"]
        if candidate["kind"] == "checkpoint" and candidate["destination"] == contract["source_local_dirname"]
    ]
    if len(selected_originals) != 1:
        raise WorkerError(f"{artifact['name']} converted teacher has no selected original JAX teacher (must be unique)")
    if upstream.get("revision") != selected_originals[0]["revision"]:
        raise WorkerError(f"{artifact['name']} converted teacher upstream revision differs from selected JAX teacher")


def validate_input_cross_contracts(spec: Mapping[str, Any], manifests: Mapping[str, Mapping[str, Any]]) -> None:
    """Bind every converted teacher to the selected original JAX manifest."""

    artifacts_by_name = {str(artifact["name"]): artifact for artifact in spec["artifacts"]}
    if set(manifests) != set(artifacts_by_name):
        raise WorkerError("staged input manifests do not cover the worker spec exactly")
    originals: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    converted: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    for name, artifact in artifacts_by_name.items():
        manifest = manifests[name]
        validate_artifact_track_contract(manifest, artifact, spec)
        source = manifest.get("source", {})
        checkpoint = manifest.get("checkpoint", {})
        if artifact["kind"] != "checkpoint" or not isinstance(source, Mapping) or not isinstance(checkpoint, Mapping):
            continue
        track = str(checkpoint.get("key", ""))
        if source.get("provider") == "gcs":
            if track in originals:
                raise WorkerError(f"multiple original JAX teachers were selected for {track}")
            originals[track] = (artifact, manifest)
        elif source.get("provider") == "openpi-jax-to-pytorch":
            converted.append((track, artifact, manifest))
    for track, artifact, manifest in converted:
        original_pair = originals.get(track)
        if original_pair is None:
            raise WorkerError(f"{artifact['name']} converted teacher has no selected original JAX teacher")
        _, original = original_pair
        upstream = manifest["source"]["upstream"]
        if upstream.get("revision") != original["source"].get("revision"):
            raise WorkerError(
                f"{artifact['name']} converted teacher upstream revision differs from selected JAX teacher"
            )


def make_staged_input_container_readable(destination: pathlib.Path) -> None:
    """Expose one verified input tree read-only to the unprivileged container user."""

    if destination.is_symlink() or not destination.is_dir():
        raise WorkerError(f"staged input is not a safe directory: {destination}")
    for name in RESERVED_INPUT_MOUNTPOINTS:
        mountpoint = destination / name
        if (
            mountpoint.is_symlink()
            or not mountpoint.is_dir()
            or mountpoint.stat().st_uid != destination.stat().st_uid
            or any(mountpoint.iterdir())
        ):
            raise WorkerError(f"reserved input mountpoint is missing, nonempty, or unsafe: {mountpoint}")
    directories = [destination]
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise WorkerError(f"staged input contains a symlink: {path}")
        if path.is_dir():
            directories.append(path)
        elif path.is_file():
            path.chmod(0o444)
        else:
            raise WorkerError(f"staged input contains a non-regular entry: {path}")
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        directory.chmod(0o555)


def versioned_worker_output_pins(
    manifest: Mapping[str, Any], artifact: Mapping[str, Any]
) -> list[dict[str, str]] | None:
    """Resolve exact S3 object versions for an artifact published by another worker."""

    source = manifest.get("source", {})
    if not isinstance(source, dict) or source.get("provider") != "pi05-worker-output":
        return None
    files = manifest.get("files")
    objects = source.get("objects")
    if not isinstance(files, list) or not isinstance(objects, list):
        raise WorkerError(f"{artifact['name']} worker-output manifest is missing file/object pins")
    file_hashes = {str(item.get("path")): item.get("sha256") for item in files if isinstance(item, dict)}
    payload = parse_s3_uri(str(artifact["payload_s3_uri"]), prefix=True)
    pins: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in objects:
        if not isinstance(item, dict):
            raise WorkerError(f"{artifact['name']} worker-output object pin is invalid")
        path = _safe_relative_path(str(item.get("path", "")), f"{artifact['name']} worker-output object path")
        s3_key = item.get("s3_key")
        version_id = item.get("version_id")
        expected_key = f"{payload.key}/{path.as_posix()}"
        digest = file_hashes.get(path.as_posix())
        if (
            path.as_posix() in seen
            or s3_key != expected_key
            or not isinstance(version_id, str)
            or not version_id
            or not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
        ):
            raise WorkerError(f"{artifact['name']} worker-output object pin differs from its payload/files")
        seen.add(path.as_posix())
        pins.append(
            {
                "path": path.as_posix(),
                "s3_uri": f"s3://{payload.bucket}/{expected_key}",
                "version_id": version_id,
                "sha256": digest,
            }
        )
    if seen != set(file_hashes):
        raise WorkerError(f"{artifact['name']} worker-output object pins do not cover every file")
    return sorted(pins, key=lambda item: item["path"])


def exact_artifact_payload_pins(
    manifest: Mapping[str, Any], artifact: Mapping[str, Any]
) -> list[dict[str, str]] | None:
    """Resolve an artifact's explicit object versions, with legacy worker-output fallback."""

    explicit = artifact.get("payload_objects")
    if explicit is None:
        source = manifest.get("source", {})
        if isinstance(source, Mapping) and source.get("provider") == "openpi-jax-to-pytorch":
            raise WorkerError(
                f"{artifact['name']} converted teacher requires explicit versioned payload_objects from its uploader"
            )
        return versioned_worker_output_pins(manifest, artifact)
    if not isinstance(explicit, list) or not explicit:
        raise WorkerError(f"{artifact['name']} explicit payload object pins are missing")

    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, list):
        raise WorkerError(f"{artifact['name']} manifest has no file inventory for its explicit payload pins")
    file_hashes = {str(item.get("path")): item.get("sha256") for item in manifest_files if isinstance(item, Mapping)}
    payload = parse_s3_uri(str(artifact["payload_s3_uri"]), prefix=True)
    pins: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(explicit):
        if not isinstance(item, Mapping):
            raise WorkerError(f"{artifact['name']} explicit payload pin {index} is invalid")
        path = _safe_relative_path(str(item.get("path", "")), f"{artifact['name']} explicit payload path")
        relative = path.as_posix()
        version_id = item.get("version_id")
        digest = item.get("sha256")
        if (
            relative in seen
            or not isinstance(version_id, str)
            or not version_id
            or not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
            or file_hashes.get(relative) != digest
        ):
            raise WorkerError(f"{artifact['name']} explicit payload pins differ from its versioned manifest")
        seen.add(relative)
        pins.append(
            {
                "path": relative,
                "s3_uri": f"s3://{payload.bucket}/{payload.key}/{relative}",
                "version_id": version_id,
                "sha256": digest,
            }
        )
    if seen != set(file_hashes):
        raise WorkerError(f"{artifact['name']} explicit payload pins do not cover every manifest file")

    worker_pins = versioned_worker_output_pins(manifest, artifact)
    if worker_pins is not None and sorted(pins, key=lambda item: item["path"]) != worker_pins:
        raise WorkerError(f"{artifact['name']} explicit payload pins conflict with its worker-output manifest")
    return sorted(pins, key=lambda item: item["path"])


def stage_artifact(
    artifact: Mapping[str, Any], spec: Mapping[str, Any], root: pathlib.Path, runner: CommandRunner
) -> dict[str, Any]:
    destination = root / "inputs" / artifact_relative_destination(artifact)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "inputs" / ".manifests" / f"{artifact['name']}.json"
    download_versioned_object(artifact["manifest"], manifest_path, spec, runner)
    manifest = _read_json(manifest_path)
    files = validate_artifact_manifest(manifest, artifact)
    validate_artifact_track_contract(manifest, artifact, spec)
    versioned_pins = exact_artifact_payload_pins(manifest, artifact)
    if versioned_pins is None:
        runner(
            [
                "aws",
                "s3",
                "sync",
                artifact["payload_s3_uri"].rstrip("/") + "/",
                str(destination),
                "--region",
                spec["aws"]["region"],
                "--no-follow-symlinks",
                "--only-show-errors",
                "--no-progress",
            ]
        )
    else:
        for pin in versioned_pins:
            output = destination.joinpath(*pathlib.PurePosixPath(pin["path"]).parts)
            download_versioned_object(pin, output, spec, runner)
    validate_artifact_manifest(manifest, artifact, destination)
    return {
        "name": artifact["name"],
        "kind": artifact["kind"],
        "revision": artifact["revision"],
        "manifest": dict(artifact["manifest"]),
        "payload_s3_uri": artifact["payload_s3_uri"],
        "destination": str(destination),
        "files": len(files),
        "bytes": sum(item["bytes"] for item in files),
    }


def _ensure_owned_output_parents(root: pathlib.Path, relative_parent: pathlib.PurePosixPath) -> pathlib.Path:
    cursor = root
    for part in relative_parent.parts:
        cursor /= part
        if cursor.is_symlink():
            raise WorkerError(f"resume checkpoint target traverses a symlink: {cursor}")
        if cursor.exists() and not cursor.is_dir():
            raise WorkerError(f"resume checkpoint target parent is not a directory: {cursor}")
        if not cursor.exists():
            cursor.mkdir(mode=0o755)
        cursor.chmod(0o755)
        if os.geteuid() == 0:
            os.chown(cursor, 1000, 1000)
    return cursor


def restore_resume_checkpoint(spec: Mapping[str, Any], root: pathlib.Path) -> dict[str, Any] | None:
    """Copy one hash-verified worker checkpoint into the writable run tree."""

    resume = spec.get("resume_checkpoint")
    if resume is None:
        return None
    artifact = next(item for item in spec["artifacts"] if item["name"] == resume["artifact_name"])
    source = root / "inputs" / artifact_relative_destination(artifact)
    manifest_path = root / "inputs" / ".manifests" / f"{artifact['name']}.json"
    manifest = _read_json(manifest_path)
    files = validate_artifact_manifest(manifest, artifact, source)
    if versioned_worker_output_pins(manifest, artifact) is None:
        raise WorkerError("resume checkpoint must come from a version-pinned pi05 worker output manifest")

    target_relative = _safe_relative_path(str(resume["target"]), "resume_checkpoint.target")
    expected_published_path = f"checkpoints/{target_relative.as_posix()}"
    published = manifest.get("artifact", {})
    if published.get("path") != expected_published_path:
        raise WorkerError("resume checkpoint source output path differs from its exact writable target")
    by_path = {item["path"]: item for item in files}
    required_state = {
        "model.safetensors",
        "optimizer.pt",
        "metadata.pt",
        "resume-state.json",
        "wandb_id.txt",
    }
    if not required_state <= set(by_path):
        raise WorkerError(f"resume checkpoint is missing full training state: {sorted(required_state - set(by_path))}")
    state = _read_json(source / "resume-state.json")
    if set(state) != RESUME_STATE_KEYS:
        raise WorkerError(f"resume state keys differ from schema: {sorted(set(state) ^ RESUME_STATE_KEYS)}")
    contract = state.get("resume_contract")
    if not isinstance(contract, dict) or set(contract) != RESUME_CONTRACT_KEYS:
        actual = set(contract) if isinstance(contract, dict) else set()
        raise WorkerError(f"resume contract keys differ from schema: {sorted(actual ^ RESUME_CONTRACT_KEYS)}")
    fingerprint = state.get("resume_fingerprint_sha256")
    lineage = state.get("initialization_lineage")
    dataset = contract.get("dataset")
    expected_step = int(target_relative.parts[-1])
    if (
        state.get("schema_version") != 2
        or state.get("global_step") != expected_step
        or state.get("config_name") != target_relative.parts[0]
        or state.get("exp_name") != target_relative.parts[1]
        or contract.get("schema_version") != 1
        or contract.get("config_name") != state.get("config_name")
        or contract.get("exp_name") != state.get("exp_name")
        or contract.get("seed") != spec.get("seed")
        or not isinstance(contract.get("seed"), int)
        or isinstance(contract.get("seed"), bool)
        or not isinstance(contract.get("batch_size"), int)
        or contract["batch_size"] < 1
        or not isinstance(contract.get("gradient_accumulation_steps"), int)
        or contract["gradient_accumulation_steps"] < 1
        or not isinstance(contract.get("lr_schedule"), dict)
        or not isinstance(contract.get("optimizer"), dict)
        or not isinstance(contract.get("wandb_enabled"), bool)
        or not isinstance(contract.get("model"), dict)
        or set(contract["model"]) != {"class", "config"}
        or not isinstance(contract["model"].get("class"), str)
        or not contract["model"]["class"]
        or not isinstance(contract["model"].get("config"), dict)
        or not isinstance(dataset, dict)
        or set(dataset) != DATASET_CONTRACT_KEYS
        or not isinstance(dataset.get("factory"), str)
        or not dataset["factory"]
        or not isinstance(dataset.get("factory_config_sha256"), str)
        or SHA256_RE.fullmatch(dataset["factory_config_sha256"]) is None
        or not isinstance(dataset.get("repo_id"), str)
        or not dataset["repo_id"]
        or not (dataset.get("revision") is None or isinstance(dataset.get("revision"), str))
        or not (dataset.get("codebase_version") is None or isinstance(dataset.get("codebase_version"), str))
        or not (dataset.get("episode_prompt_path") is None or isinstance(dataset.get("episode_prompt_path"), str))
        or not (
            dataset.get("episode_prompt_sha256") is None
            or (
                isinstance(dataset.get("episode_prompt_sha256"), str)
                and SHA256_RE.fullmatch(dataset["episode_prompt_sha256"]) is not None
            )
        )
        or not isinstance(dataset.get("action_sequence_keys"), list)
        or not dataset["action_sequence_keys"]
        or any(not isinstance(key, str) or not key for key in dataset["action_sequence_keys"])
        or not (dataset.get("asset_id") is None or isinstance(dataset.get("asset_id"), str))
        or not isinstance(dataset.get("use_quantile_norm"), bool)
        or not isinstance(dataset.get("prompt_from_task"), bool)
        or not (
            dataset.get("normalization_sha256") is None
            or (
                isinstance(dataset.get("normalization_sha256"), str)
                and SHA256_RE.fullmatch(dataset["normalization_sha256"]) is not None
            )
        )
        or not (
            dataset.get("recovery_provenance_sha256") is None
            or (
                isinstance(dataset.get("recovery_provenance_sha256"), str)
                and SHA256_RE.fullmatch(dataset["recovery_provenance_sha256"]) is not None
            )
        )
        or contract.get("pytorch_training_precision") not in {"bfloat16", "float32"}
        or contract.get("stochastic_schedule") != "sha256-v2(model:seed,step,accumulation,rank;loader:seed,epoch)"
        or not isinstance(contract.get("one_batch_overfit"), bool)
        or contract.get("one_batch_overfit") is not False
        or not isinstance(contract.get("one_batch_overfit_min_relative_decline"), int | float)
        or isinstance(contract.get("one_batch_overfit_min_relative_decline"), bool)
        or not 0.0 < contract["one_batch_overfit_min_relative_decline"] < 1.0
        or not isinstance(fingerprint, str)
        or SHA256_RE.fullmatch(fingerprint) is None
        or fingerprint != resume_identity_fingerprint(contract, lineage)
        or not isinstance(lineage, dict)
        or set(lineage) != {"kind", "model_sha256"}
        or lineage.get("kind") not in {"shallow_teacher_transplant", "pytorch_source", "random_initialization"}
        or (
            lineage.get("kind") != "random_initialization"
            and (
                not isinstance(lineage.get("model_sha256"), str) or SHA256_RE.fullmatch(lineage["model_sha256"]) is None
            )
        )
        or (lineage.get("kind") == "random_initialization" and lineage.get("model_sha256") is not None)
        or state.get("state_files") != ["metadata.pt", "model.safetensors", "optimizer.pt", "wandb_id.txt"]
    ):
        raise WorkerError("resume-state.json does not match the declared config, experiment, step, or state files")
    if contract["config_name"].startswith(("pi05_libero_", "pi05_droid_")):
        revision = dataset.get("revision")
        normalization_sha256 = dataset.get("normalization_sha256")
        if (
            not isinstance(revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", revision) is None
            or not isinstance(normalization_sha256, str)
            or SHA256_RE.fullmatch(normalization_sha256) is None
        ):
            raise WorkerError("reproduction resume lacks immutable dataset and normalization identities")
    if contract["config_name"].startswith("pi05_droid_"):
        prompt_sha256 = dataset.get("episode_prompt_sha256")
        if not isinstance(prompt_sha256, str) or SHA256_RE.fullmatch(prompt_sha256) is None:
            raise WorkerError("DROID reproduction resume lacks its episode-prompt content identity")
    if contract["config_name"].endswith("_distill"):
        teacher = contract.get("teacher")
        if (
            not isinstance(teacher, dict)
            or set(teacher) != {"model_sha256"}
            or not isinstance(teacher.get("model_sha256"), str)
            or SHA256_RE.fullmatch(teacher["model_sha256"]) is None
            or lineage.get("kind") != "shallow_teacher_transplant"
            or lineage.get("model_sha256") != teacher["model_sha256"]
        ):
            raise WorkerError("distillation resume lineage differs from its teacher identity")
    elif contract.get("teacher") is not None:
        raise WorkerError("non-distillation resume unexpectedly declares a teacher")

    output_root = root / "output"
    checkpoint_root = output_root / "checkpoints"
    parent = _ensure_owned_output_parents(checkpoint_root, target_relative.parent)
    target = checkpoint_root.joinpath(*target_relative.parts)
    if target.exists() or target.is_symlink():
        raise WorkerError(f"resume checkpoint target already exists; refusing overwrite: {target}")
    temporary = pathlib.Path(tempfile.mkdtemp(prefix=f".resume-{spec['run_id']}.", dir=parent))
    try:
        for record in files:
            relative = _safe_relative_path(str(record["path"]), "resume checkpoint manifest path")
            source_path = source.joinpath(*relative.parts)
            destination = temporary.joinpath(*relative.parts)
            destination.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            with source_path.open("rb") as source_stream, destination.open("xb") as destination_stream:
                shutil.copyfileobj(source_stream, destination_stream, length=8 * 1024 * 1024)
                destination_stream.flush()
                os.fsync(destination_stream.fileno())
            destination.chmod(0o444)
            if destination.stat().st_size != record["bytes"] or sha256_file(destination) != record["sha256"]:
                raise WorkerError(f"restored checkpoint file failed hash verification: {relative}")
        for directory in sorted(
            [temporary, *(path for path in temporary.rglob("*") if path.is_dir())],
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
        os.replace(temporary, target)
        validate_artifact_manifest(manifest, artifact, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "artifact_name": artifact["name"],
        "artifact_revision": artifact["revision"],
        "manifest": dict(artifact["manifest"]),
        "target": f"/mnt/openpi/runs/{target_relative.as_posix()}",
        "global_step": expected_step,
        "files": len(files),
        "bytes": sum(item["bytes"] for item in files),
        "resume_state_sha256": by_path["resume-state.json"]["sha256"],
        "resume_fingerprint_sha256": fingerprint,
    }


def validate_image_identity(spec: Mapping[str, Any], repo_digests: Any, labels: Any) -> list[str]:
    """Prove that a pulled digest implements its declared immutable runtime contract."""

    uri = spec["image"]["uri"]
    if not isinstance(repo_digests, list) or uri not in repo_digests:
        raise WorkerError(f"pulled image does not expose the requested immutable digest: {uri}")
    if not isinstance(labels, dict):
        raise WorkerError("pulled image has no OCI revision labels")
    revision = labels.get("org.opencontainers.image.revision")
    if revision != spec["source"]["commit"]:
        raise WorkerError(
            "pulled image OCI revision does not match source.commit: "
            f"expected {spec['source']['commit']}, found {revision!r}"
        )
    purpose = spec["image"]["purpose"]
    if labels.get("ai.openpi.image-purpose") != purpose:
        raise WorkerError("pulled image purpose label does not match the worker spec")
    if purpose == IMAGE_PURPOSE_POLICY:
        if (
            labels.get("ai.openpi.lerobot-runtime") != spec["image"]["lerobot_runtime"]
            or labels.get("ai.openpi.lerobot-revision") != spec["image"]["lerobot_revision"]
        ):
            raise WorkerError("pulled policy image LeRobot runtime labels do not match the worker spec")
        if labels.get("ai.openpi.video-decoder") != POLICY_VIDEO_DECODER:
            raise WorkerError("pulled policy image video decoder label does not match the reproduction contract")
        if labels.get("ai.openpi.onnxruntime-gpu-version") != POLICY_ONNXRUNTIME_GPU_VERSION:
            raise WorkerError("pulled policy image ONNX Runtime label does not match the reproduction contract")
    elif purpose == IMAGE_PURPOSE_LIBERO_EVALUATOR:
        expected_labels = {
            "ai.openpi.policy-backend": spec["image"]["policy_backend"],
            "ai.openpi.lerobot-runtime": "v2",
            "ai.openpi.lerobot-revision": LEROBOT_REVISIONS["v2"],
            "ai.openpi.libero-simulator-revision": LIBERO_SIMULATOR_REVISION,
            "ai.openpi.libero-requirements-sha256": LIBERO_REQUIREMENTS_SHA256,
            "ai.openpi.parent-policy-image": spec["image"]["parent_policy_image"],
        }
        for label, expected in expected_labels.items():
            if labels.get(label) != expected:
                raise WorkerError(f"pulled LIBERO evaluator image label does not match: {label}")
        compiler_identity_labels = {
            "ai.openpi.parent-image-purpose",
            "ai.openpi.parent-tensorrt-compiler-image",
            "ai.openpi.parent-tensorrt-compiler-source-revision",
            *(label for key, label in TENSORRT_TOOLCHAIN_LABELS.items() if key != "onnxruntime_gpu_version"),
        }
        if spec["image"]["policy_backend"] == "eager":
            if labels.get("ai.openpi.onnxruntime-gpu-version") != POLICY_ONNXRUNTIME_GPU_VERSION:
                raise WorkerError("pulled eager LIBERO evaluator ONNX Runtime label does not match")
            if any(label in labels for label in compiler_identity_labels):
                raise WorkerError("pulled eager LIBERO evaluator must not claim a TensorRT compiler identity")
        else:
            tensorrt_labels = {
                **TENSORRT_POLICY_LABELS,
                "ai.openpi.parent-tensorrt-compiler-image": spec["image"]["parent_tensorrt_compiler_image"],
                "ai.openpi.parent-tensorrt-compiler-source-revision": spec["image"][
                    "parent_tensorrt_compiler_source_revision"
                ],
                **{label: spec["image"]["toolchain"][key] for key, label in TENSORRT_TOOLCHAIN_LABELS.items()},
            }
            for label, expected in tensorrt_labels.items():
                if labels.get(label) != expected:
                    raise WorkerError(f"pulled TensorRT LIBERO evaluator label does not match: {label}")
    elif purpose == IMAGE_PURPOSE_TENSORRT_COMPILER:
        if any(label in labels for label in ("ai.openpi.lerobot-runtime", "ai.openpi.lerobot-revision")):
            raise WorkerError("pulled TensorRT compiler image must not claim a LeRobot runtime")
        for key, label in TENSORRT_TOOLCHAIN_LABELS.items():
            if labels.get(label) != spec["image"]["toolchain"][key]:
                raise WorkerError(f"pulled TensorRT compiler image toolchain label does not match: {label}")
    elif purpose == IMAGE_PURPOSE_TENSORRT_POLICY:
        expected_labels = {
            "ai.openpi.lerobot-runtime": spec["image"]["lerobot_runtime"],
            "ai.openpi.lerobot-revision": spec["image"]["lerobot_revision"],
            "ai.openpi.parent-tensorrt-compiler-image": spec["image"]["parent_tensorrt_compiler_image"],
            "ai.openpi.parent-tensorrt-compiler-source-revision": spec["image"][
                "parent_tensorrt_compiler_source_revision"
            ],
            **TENSORRT_POLICY_LABELS,
            **{label: spec["image"]["toolchain"][key] for key, label in TENSORRT_TOOLCHAIN_LABELS.items()},
        }
        for label, expected in expected_labels.items():
            if labels.get(label) != expected:
                raise WorkerError(f"pulled TensorRT policy image label does not match: {label}")
    else:  # validate_worker_spec rejects this; retain a fail-closed boundary for direct callers.
        raise WorkerError(f"unsupported image purpose: {purpose!r}")
    return [str(item) for item in repo_digests]


def verify_and_pull_image(spec: Mapping[str, Any], runner: CommandRunner) -> list[str]:
    uri = spec["image"]["uri"]
    registry = uri.split("/", 1)[0]
    password = runner(["aws", "ecr", "get-login-password", "--region", EXPECTED_REGION])
    completed = subprocess.run(
        ["docker", "login", "--username", "AWS", "--password-stdin", registry],
        input=password,
        text=True,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise CommandError(
            ["docker", "login", "--username", "AWS", "--password-stdin", registry],
            completed.returncode,
            completed.stderr,
        )
    runner(["docker", "pull", uri])
    try:
        repo_digests = json.loads(runner(["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", uri]))
    except json.JSONDecodeError as exc:
        raise WorkerError("docker image inspection returned invalid RepoDigests JSON") from exc
    try:
        labels = json.loads(runner(["docker", "image", "inspect", "--format", "{{json .Config.Labels}}", uri]))
    except json.JSONDecodeError as exc:
        raise WorkerError("docker image inspection returned invalid labels JSON") from exc
    return validate_image_identity(spec, repo_digests, labels)


def worker_container_hostname(run_id: str) -> str:
    """Derive a stable DNS hostname without exposing run-ID punctuation to Docker."""

    if not isinstance(run_id, str) or RUN_ID_RE.fullmatch(run_id) is None:
        raise WorkerError(f"invalid run ID for Docker hostname: {run_id!r}")
    digest = hashlib.sha256(run_id.encode()).hexdigest()[:32]
    hostname = f"pi05-worker-{digest}"
    if len(hostname) > 63 or DOCKER_HOSTNAME_RE.fullmatch(hostname) is None:
        raise WorkerError("derived Docker hostname violates the DNS hostname contract")
    return hostname


def build_docker_command(
    spec: Mapping[str, Any],
    source_root: pathlib.Path,
    scratch_root: pathlib.Path,
    *,
    instance_id: str | None = None,
    instance_type: str | None = None,
) -> list[str]:
    run_id = spec["run_id"]
    hostname = worker_container_hostname(run_id)
    output = scratch_root / "output"
    command = [
        "docker",
        "run",
        "--name",
        f"pi05-{run_id}",
        "--hostname",
        hostname,
        "--add-host",
        f"{hostname}:127.0.0.1",
        "--gpus",
        "all",
        "--network",
        "none",
        "--ipc",
        "host",
        "--shm-size",
        f"{spec['container'].get('shm_size_gib', 32)}g",
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
        f"type=bind,src={scratch_root / 'inputs'},dst={CONTAINER_INPUT_ROOT},readonly",
        # Mount only the four declared payload roots. Host-owned ready markers,
        # receipts, upload spools, and active log state remain outside the
        # container namespace even though they share the same host parent.
        "--mount",
        f"type=bind,src={output / 'checkpoints'},dst=/output/checkpoints",
        "--mount",
        f"type=bind,src={output / 'logs'},dst=/output/logs",
        "--mount",
        f"type=bind,src={output / 'manifests'},dst=/output/manifests",
        "--mount",
        f"type=bind,src={output / 'artifacts'},dst=/output/artifacts",
        "--mount",
        f"type=bind,src={output / 'checkpoints'},dst={CONTAINER_INPUT_ROOT / 'runs'}",
        "--mount",
        f"type=bind,src={output / 'artifacts'},dst={CONTAINER_INPUT_ROOT / 'evidence'}",
        "--mount",
        f"type=bind,src={scratch_root / 'tmp'},dst=/tmp",
        "--mount",
        f"type=bind,src={scratch_root / 'cache'},dst=/cache",
        "--env",
        "HOME=/tmp",
        "--env",
        "USER=pi05",
        "--env",
        "LOGNAME=pi05",
        "--env",
        "XDG_CACHE_HOME=/cache",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "PYTHONPATH=/workspace/openpi/src:/workspace/openpi",
        "--env",
        f"PI05_RUN_ID={run_id}",
        "--env",
        f"PI05_SOURCE_SHA={spec['source']['commit']}",
        "--env",
        f"PI05_IMAGE_DIGEST={spec['image']['digest']}",
        "--env",
        f"PI05_SEED={spec['seed']}",
    ]
    if _uses_multi_process_torchrun(spec["container"]["command"]):
        # Network isolation leaves only loopback. Override any image-level NCCL
        # exclusions and bind optional Gloo control collectives to the same
        # deterministic interface. Standalone torchrun owns MASTER_ADDR itself.
        for key, value in TORCHRUN_LOOPBACK_ENVIRONMENT.items():
            command.extend(["--env", f"{key}={value}"])
    if (instance_id is None) != (instance_type is None):
        raise WorkerError("container instance ID and type must be supplied together from live IMDS")
    if instance_id is not None and instance_type is not None:
        if INSTANCE_ID_RE.fullmatch(instance_id) is None or INSTANCE_TYPE_RE.fullmatch(instance_type) is None:
            raise WorkerError("container instance identity must contain a valid EC2 instance ID and type")
        command.extend(
            [
                "--env",
                f"PI05_INSTANCE_ID={instance_id}",
                "--env",
                f"PI05_INSTANCE_TYPE={instance_type}",
            ]
        )
    for artifact in spec["artifacts"]:
        env_name = re.sub(r"[^A-Z0-9_]", "_", artifact["name"].upper())
        destination = CONTAINER_INPUT_ROOT / artifact_relative_destination(artifact)
        command.extend(["--env", f"PI05_INPUT_{env_name}={destination}"])
    if spec.get("resume_checkpoint") is not None:
        command.extend(
            [
                "--env",
                f"PI05_RESUME_CHECKPOINT={CONTAINER_INPUT_ROOT / 'runs' / spec['resume_checkpoint']['target']}",
            ]
        )
    for key, value in sorted(spec["container"].get("environment", {}).items()):
        command.extend(["--env", f"{key}={value}"])
    command.append(spec["image"]["uri"])
    command.extend(spec["container"]["command"])
    return command


class OutputManager:
    """Syncs only artifacts committed by an atomic ready marker."""

    ALLOWED_ROOTS: ClassVar[set[str]] = {"checkpoints", "logs", "manifests", "artifacts"}

    def __init__(self, spec: Mapping[str, Any], root: pathlib.Path, runner: CommandRunner):
        self.spec = spec
        self.root = root
        self.runner = runner
        self.ready_dir = root / ".ready"
        self.receipt_dir = root / ".receipts"
        self.spool_dir = root / ".spool"
        resume = spec.get("resume_checkpoint")
        self.restored_checkpoint_path = f"checkpoints/{resume['target']}" if isinstance(resume, Mapping) else None
        output = parse_s3_uri(spec["output"]["s3_uri"], prefix=True)
        self.bucket = output.bucket
        self.prefix = output.key

    def create_marker(self, kind: str, paths: Sequence[pathlib.Path], marker_name: str) -> pathlib.Path:
        if not marker_name.endswith(".ready.json") or pathlib.PurePath(marker_name).name != marker_name:
            raise WorkerError(f"invalid ready marker name: {marker_name!r}")
        records = []
        for path in sorted(paths):
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(self.root.resolve())
            except ValueError as exc:
                raise WorkerError(f"ready artifact is outside the output root: {path}") from exc
            if relative.parts[0] not in self.ALLOWED_ROOTS or path.is_symlink() or not path.is_file():
                raise WorkerError(f"ready artifact is not an allowed regular file: {path}")
            records.append({"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        if not records:
            raise WorkerError("ready marker must contain at least one artifact")
        marker = {"schema_version": 1, "kind": kind, "artifacts": records}
        destination = self.ready_dir / marker_name
        payload = json.dumps(marker, indent=2, sort_keys=True) + "\n"
        if destination.exists():
            if destination.read_text() != payload:
                raise WorkerError(f"ready marker collision: {destination}")
            return destination
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(payload)
        os.replace(temporary, destination)
        return destination

    def _require_safe_output_path(self, relative: pathlib.PurePosixPath) -> pathlib.Path:
        target = self.root.joinpath(*relative.parts)
        cursor = self.root
        for part in relative.parts:
            cursor /= part
            if cursor.is_symlink():
                raise WorkerError(f"expected output traverses a symlink: {cursor}")
        try:
            target.resolve().relative_to(self.root.resolve())
        except ValueError as exc:
            raise WorkerError(f"expected output is outside the output root: {target}") from exc
        return target

    def commit_expected_outputs(self) -> list[pathlib.Path]:
        """Atomically mark the declared successful-run outputs for upload."""

        markers: list[pathlib.Path] = []
        for expected in self.spec.get("expected_outputs", []):
            relative = _safe_relative_path(str(expected["path"]), "expected output path")
            target = self._require_safe_output_path(relative)
            if target.is_file():
                files = [target]
            elif target.is_dir():
                files = []
                for path in sorted(target.rglob("*")):
                    if path.is_symlink():
                        raise WorkerError(f"expected output contains a symlink: {path}")
                    if path.is_file():
                        files.append(path)
                    elif not path.is_dir():
                        raise WorkerError(f"expected output contains a non-regular entry: {path}")
            else:
                raise WorkerError(f"expected output does not exist: {target}")
            if not files:
                raise WorkerError(f"expected output contains no files: {target}")
            markers.append(
                self.create_marker(
                    str(expected["kind"]),
                    files,
                    f"expected-{expected['name']}.ready.json",
                )
            )
        return markers

    def _worker_input_manifest(
        self,
        expected: Mapping[str, Any],
        receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build a staging manifest whose paths are relative to one output directory."""

        root = pathlib.PurePosixPath(str(expected["path"]))
        local_root = self._require_safe_output_path(root)
        if not local_root.is_dir():
            raise WorkerError(f"published worker input must be a directory: {local_root}")
        records = receipt.get("artifacts")
        if not isinstance(records, list) or not records:
            raise WorkerError(f"expected output receipt has no artifacts: {expected['name']}")
        files: list[dict[str, Any]] = []
        objects: list[dict[str, Any]] = []
        for record in records:
            try:
                output_path = pathlib.PurePosixPath(str(record["path"]))
                relative = output_path.relative_to(root)
            except (KeyError, ValueError) as exc:
                raise WorkerError(f"expected output receipt escaped its declared root: {expected['name']}") from exc
            if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
                raise WorkerError(f"invalid published worker input path: {relative}")
            size = record.get("bytes")
            digest = record.get("sha256")
            version_id = record.get("version_id")
            s3_key = record.get("s3_key")
            if (
                not isinstance(size, int)
                or size < 0
                or not isinstance(digest, str)
                or SHA256_RE.fullmatch(digest) is None
                or not isinstance(version_id, str)
                or not version_id
                or not isinstance(s3_key, str)
                or not s3_key
            ):
                raise WorkerError(f"invalid uploaded output identity for {expected['name']}")
            files.append({"path": relative.as_posix(), "bytes": size, "sha256": digest})
            objects.append({"path": relative.as_posix(), "s3_key": s3_key, "version_id": version_id})
        files.sort(key=lambda item: item["path"])
        objects.sort(key=lambda item: item["path"])
        if len({item["path"] for item in files}) != len(files):
            raise WorkerError(f"duplicate paths in expected output receipt: {expected['name']}")
        identity = {
            "run_id": self.spec["run_id"],
            "source_commit": self.spec["source"]["commit"],
            "image_digest": self.spec["image"]["digest"],
            "seed": self.spec["seed"],
            "output": {
                "name": expected["name"],
                "kind": expected["kind"],
                "path": root.as_posix(),
                "publish_destination": expected["publish_destination"],
                "payload_s3_uri": f"s3://{self.bucket}/{self.prefix}/{root.as_posix()}/",
            },
            "files": files,
            "objects": objects,
        }
        revision = hashlib.sha256(json.dumps(identity, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
        return {
            "schema_version": 1,
            "created_at": dt.datetime.now(UTC).isoformat(),
            "source": {
                "provider": "pi05-worker-output",
                "revision_kind": "worker-output-content-and-provenance-sha256",
                "revision": revision,
                **{key: identity[key] for key in ("run_id", "source_commit", "image_digest", "seed")},
                "objects": objects,
            },
            "artifact": identity["output"],
            "totals": {"files": len(files), "bytes": sum(item["bytes"] for item in files)},
            "files": files,
        }

    def publish_expected_inputs(
        self, completed_receipts: Sequence[Mapping[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Publish worker-compatible manifests for declared cross-worker outputs."""

        receipt_by_marker = {str(item.get("marker")): item for item in completed_receipts}
        pending: list[tuple[Mapping[str, Any], dict[str, Any], pathlib.Path, str]] = []
        for expected in self.spec.get("expected_outputs", []):
            if "publish_destination" not in expected:
                continue
            output_marker = f"expected-{expected['name']}.ready.json"
            try:
                output_receipt = receipt_by_marker[output_marker]
            except KeyError as exc:
                raise WorkerError(f"expected output was not durably uploaded: {expected['name']}") from exc
            manifest = self._worker_input_manifest(expected, output_receipt)
            manifest_path = self.root / "manifests" / f"worker-input-{expected['name']}.sha256.json"
            _write_json_new(manifest_path, manifest)
            marker_name = f"worker-input-{expected['name']}.ready.json"
            self.create_marker("manifest", [manifest_path], marker_name)
            pending.append((expected, manifest, manifest_path, marker_name))

        if not pending:
            return [], list(completed_receipts)
        all_receipts = self.sync_once()
        receipt_by_marker = {str(item.get("marker")): item for item in all_receipts}
        published: list[dict[str, Any]] = []
        input_kind = {"checkpoint": "checkpoint", "artifact": "asset"}
        for expected, manifest, manifest_path, marker_name in pending:
            try:
                manifest_receipt = receipt_by_marker[marker_name]
                uploaded = manifest_receipt["artifacts"]
                if len(uploaded) != 1:
                    raise ValueError("manifest marker did not upload exactly one file")
                record = uploaded[0]
                manifest_key = str(record["s3_key"])
                version_id = str(record["version_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise WorkerError(f"worker-input manifest upload is incomplete: {expected['name']}") from exc
            published.append(
                {
                    "name": expected["name"],
                    "kind": input_kind[str(expected["kind"])],
                    "revision": manifest["source"]["revision"],
                    "manifest": {
                        "s3_uri": f"s3://{self.bucket}/{manifest_key}",
                        "version_id": version_id,
                        "sha256": sha256_file(manifest_path),
                    },
                    "payload_s3_uri": manifest["artifact"]["payload_s3_uri"],
                    "destination": expected["publish_destination"],
                }
            )
        return published, all_receipts

    def discover_atomic_checkpoints(self) -> None:
        checkpoint_root = self.root / "checkpoints"
        for directory in sorted(path for path in checkpoint_root.rglob("*") if path.is_dir() and path.name.isdigit()):
            relative = directory.relative_to(self.root).as_posix()
            if relative == self.restored_checkpoint_path:
                continue
            marker_name = f"checkpoint-{hashlib.sha256(relative.encode()).hexdigest()[:24]}.ready.json"
            if (self.ready_dir / marker_name).exists() or (self.receipt_dir / f"{marker_name}.receipt.json").exists():
                continue
            if directory.is_symlink() or any(
                part.startswith("tmp_") for part in directory.relative_to(checkpoint_root).parts
            ):
                continue
            files = [path for path in directory.rglob("*") if path.is_file()]
            if files:
                self.create_marker("checkpoint", files, marker_name)

    def _load_marker(self, path: pathlib.Path) -> tuple[str, dict[str, Any]]:
        if path.is_symlink() or not path.is_file() or not path.name.endswith(".ready.json"):
            raise WorkerError(f"invalid ready marker: {path}")
        marker_hash = sha256_file(path)
        marker = _read_json(path)
        if marker.get("schema_version") != 1 or marker.get("kind") not in {
            "checkpoint",
            "log",
            "manifest",
            "artifact",
        }:
            raise WorkerError(f"invalid ready marker schema: {path}")
        records = marker.get("artifacts")
        if not isinstance(records, list) or not records:
            raise WorkerError(f"ready marker has no artifacts: {path}")
        seen: set[str] = set()
        for record in records:
            if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
                raise WorkerError(f"ready marker artifact record is invalid: {path}")
            relative = _safe_relative_path(str(record["path"]), "ready artifact path")
            if relative.parts[0] not in self.ALLOWED_ROOTS or relative.as_posix() in seen:
                raise WorkerError(f"ready marker artifact path is unsafe or duplicated: {relative}")
            if (
                not isinstance(record["bytes"], int)
                or record["bytes"] < 0
                or not isinstance(record["sha256"], str)
                or SHA256_RE.fullmatch(record["sha256"]) is None
            ):
                raise WorkerError(f"ready marker artifact identity is invalid: {relative}")
            seen.add(relative.as_posix())
        return marker_hash, marker

    def _existing_versions(self, key: str) -> list[dict[str, Any]]:
        response = _json_command(
            self.runner,
            [
                "aws",
                "s3api",
                "list-object-versions",
                "--bucket",
                self.bucket,
                "--prefix",
                key,
                "--max-keys",
                "10",
                "--expected-bucket-owner",
                EXPECTED_ACCOUNT,
                "--region",
                EXPECTED_REGION,
                "--output",
                "json",
            ],
        )
        versions = [item for item in response.get("Versions", []) if item.get("Key") == key]
        delete_markers = [item for item in response.get("DeleteMarkers", []) if item.get("Key") == key]
        if delete_markers or len(versions) > 1:
            raise WorkerError(
                f"run output key has prior mutation history; refusing overwrite: s3://{self.bucket}/{key}"
            )
        return versions

    def _head_and_validate(self, key: str, record: Mapping[str, Any], version_id: str | None = None) -> dict[str, Any]:
        argv = [
            "aws",
            "s3api",
            "head-object",
            "--bucket",
            self.bucket,
            "--key",
            key,
            "--expected-bucket-owner",
            EXPECTED_ACCOUNT,
            "--region",
            EXPECTED_REGION,
            "--output",
            "json",
        ]
        if version_id:
            argv.extend(["--version-id", version_id])
        head = _json_command(self.runner, argv)
        metadata = head.get("Metadata", {})
        if (
            int(head.get("ContentLength", -1)) != record["bytes"]
            or metadata.get("sha256") != record["sha256"]
            or metadata.get("run-id") != self.spec["run_id"]
            or not head.get("VersionId")
        ):
            raise WorkerError(f"remote output identity mismatch: s3://{self.bucket}/{key}")
        return head

    def _upload_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        relative = _safe_relative_path(str(record["path"]), "ready artifact path")
        source = self._require_safe_output_path(relative)
        if source.is_symlink() or not source.is_file():
            raise WorkerError(f"ready artifact disappeared or became unsafe: {source}")
        if source.stat().st_size != record["bytes"] or sha256_file(source) != record["sha256"]:
            raise WorkerError(f"ready artifact changed after its marker was committed: {source}")
        key = f"{self.prefix}/{relative.as_posix()}"
        versions = self._existing_versions(key)
        if versions:
            version_id = str(versions[0].get("VersionId", ""))
            head = self._head_and_validate(key, record, version_id)
            return {"path": relative.as_posix(), "s3_key": key, "version_id": head["VersionId"], **dict(record)}

        spool = self.spool_dir / f"{record['sha256']}.upload"
        temporary = spool.with_suffix(".tmp")
        with source.open("rb") as source_stream, temporary.open("wb") as destination_stream:
            shutil.copyfileobj(source_stream, destination_stream, length=8 * 1024 * 1024)
            destination_stream.flush()
            os.fsync(destination_stream.fileno())
        os.replace(temporary, spool)
        if spool.stat().st_size != record["bytes"] or sha256_file(spool) != record["sha256"]:
            raise WorkerError(f"atomic upload snapshot failed validation: {source}")
        try:
            self.runner(
                [
                    "aws",
                    "s3",
                    "cp",
                    str(spool),
                    f"s3://{self.bucket}/{key}",
                    "--region",
                    EXPECTED_REGION,
                    "--only-show-errors",
                    "--no-progress",
                    "--sse",
                    "AES256",
                    "--metadata",
                    f"sha256={record['sha256']},run-id={self.spec['run_id']}",
                ]
            )
            head = self._head_and_validate(key, record)
        finally:
            spool.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)
        return {"path": relative.as_posix(), "s3_key": key, "version_id": head["VersionId"], **dict(record)}

    def sync_once(self) -> list[dict[str, Any]]:
        self.discover_atomic_checkpoints()
        completed: list[dict[str, Any]] = []
        for marker_path in sorted(self.ready_dir.glob("*.ready.json")):
            marker_hash, marker = self._load_marker(marker_path)
            receipt_path = self.receipt_dir / f"{marker_path.name}.receipt.json"
            if receipt_path.exists():
                receipt = _read_json(receipt_path)
                if receipt.get("marker_sha256") != marker_hash:
                    raise WorkerError(f"ready marker changed after upload: {marker_path}")
                completed.append(receipt)
                continue
            uploads = [self._upload_record(record) for record in marker["artifacts"]]
            receipt = {
                "schema_version": 1,
                "marker": marker_path.name,
                "marker_sha256": marker_hash,
                "kind": marker["kind"],
                "uploaded_at": dt.datetime.now(UTC).isoformat(),
                "artifacts": uploads,
            }
            temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
            temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, receipt_path)
            completed.append(receipt)
        return completed


def _require_finite_json_numbers(value: Any, *, context: str) -> None:
    """Reject JSON-compatible evidence containing NaN or infinity."""

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WorkerError(f"{context} contains a non-finite numeric value")
        return
    if isinstance(value, list):
        for item in value:
            _require_finite_json_numbers(item, context=context)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise WorkerError(f"{context} contains a non-string object key")
            _require_finite_json_numbers(item, context=context)
        return
    raise WorkerError(f"{context} contains a non-JSON value")


def _metric_command_argv(spec: Mapping[str, Any]) -> list[str]:
    command = list(spec["container"]["command"])
    training_positions = [index for index, value in enumerate(command) if value == "scripts/train_pytorch.py"]
    if len(training_positions) == 1:
        return command[training_positions[0] :]
    if len(command) < 2 or command[0] not in {"python", "python3", "/opt/modelopt/bin/python"}:
        raise WorkerError("metrics manifest requires a direct Python worker command")
    return command[1:]


def _optional_command_option(command: Sequence[str], option: str) -> str | None:
    positions = [index for index, value in enumerate(command) if value == option]
    if not positions:
        return None
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise WorkerError(f"metrics-producing command has an invalid {option} option")
    return command[positions[0] + 1]


def _is_declared_latency_report(spec: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Recognize only the benchmark report selected by its direct worker argv."""

    command = list(spec["container"]["command"])
    if (
        expected["kind"] != "artifact"
        or len(command) < 2
        or command[0] not in {"python", "python3", "/opt/modelopt/bin/python"}
        or command[1] != "scripts/benchmark_pi05_latency.py"
    ):
        return False
    output = _optional_command_option(command, "--output")
    prefix = "/output/"
    return isinstance(output, str) and output.startswith(prefix) and output.removeprefix(prefix) == expected["path"]


def _metric_candidates(
    spec: Mapping[str, Any], expected: Mapping[str, Any], target: pathlib.Path
) -> list[pathlib.Path]:
    """Select only deterministic, schema-bearing metrics sources."""

    kind = str(expected["kind"])
    if kind == "checkpoint":
        candidate = target / TRAINING_METRICS_FILENAME
        return [candidate] if candidate.is_file() else []
    if target.is_file():
        return (
            [target]
            if kind == "manifest" or "manifest" in target.name or _is_declared_latency_report(spec, expected)
            else []
        )
    if not target.is_dir() or kind not in {"artifact", "manifest"}:
        return []
    return sorted(
        path for path in target.rglob("*.json") if path.is_file() and not path.is_symlink() and "manifest" in path.name
    )


def _metric_runtime_identity(
    launch_metadata: Mapping[str, Any] | None, instance_identity: Mapping[str, Any] | None
) -> tuple[str, str, str]:
    if launch_metadata is None or instance_identity is None:
        raise WorkerError("stage metrics require launch and live instance identity")
    reservation_id = launch_metadata.get("reservation_id")
    instance_id = instance_identity.get("instanceId")
    instance_type = instance_identity.get("instanceType")
    if (
        not isinstance(reservation_id, str)
        or UUID_RE.fullmatch(reservation_id) is None
        or not isinstance(instance_id, str)
        or INSTANCE_ID_RE.fullmatch(instance_id) is None
        or not isinstance(instance_type, str)
        or INSTANCE_TYPE_RE.fullmatch(instance_type) is None
        or launch_metadata.get("instance_type") != instance_type
    ):
        raise WorkerError("stage metrics launch or instance identity is invalid")
    return reservation_id, instance_id, instance_type


def _require_exact_keys(value: Any, keys: set[str], *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise WorkerError(f"{context} has the wrong schema keys: {sorted(actual ^ keys)}")
    return value


def _command_dataset_identity(spec: Mapping[str, Any], command: Sequence[str]) -> tuple[str | None, str]:
    dataset_name = _optional_command_option(command, "--dataset")
    dataset_revision = _optional_command_option(command, "--dataset-revision")
    if dataset_revision is None:
        revisions = {str(artifact["revision"]) for artifact in spec["artifacts"] if artifact.get("kind") == "dataset"}
        if len(revisions) != 1:
            raise WorkerError("metrics manifest cannot resolve one dataset revision from the worker contract")
        dataset_revision = revisions.pop()
    return dataset_name, dataset_revision


def _validate_stage_metrics_manifest(
    document: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    candidate: pathlib.Path,
    launch_metadata: Mapping[str, Any] | None,
    instance_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if set(document) != STAGE_METRICS_MANIFEST_KEYS or document.get("schema_version") != 1:
        raise WorkerError(f"stage metrics manifest has the wrong schema: {candidate}")
    source = _require_exact_keys(document.get("source"), {"sha", "dirty"}, context="stage metrics source")
    runtime = _require_exact_keys(
        document.get("runtime"), {"image_digest", "instance_type", "instance_id"}, context="stage metrics runtime"
    )
    dataset = _require_exact_keys(document.get("dataset"), {"name", "revision"}, context="stage metrics dataset")
    experiment = _require_exact_keys(document.get("experiment"), {"seed", "steps"}, context="stage metrics experiment")
    cost = _require_exact_keys(document.get("cost"), {"reservation_id"}, context="stage metrics cost")
    manifest_command = _require_exact_keys(document.get("command"), {"argv", "shell"}, context="stage metrics command")
    reservation_id, instance_id, instance_type = _metric_runtime_identity(launch_metadata, instance_identity)
    expected_argv = _metric_command_argv(spec)
    dataset_name, dataset_revision = _command_dataset_identity(spec, expected_argv)
    if (
        source != {"sha": spec["source"]["commit"], "dirty": False}
        or runtime
        != {"image_digest": spec["image"]["digest"], "instance_type": instance_type, "instance_id": instance_id}
        or dataset.get("revision") != dataset_revision
        or (dataset_name is not None and dataset.get("name") != dataset_name)
        or cost.get("reservation_id") != reservation_id
        or manifest_command.get("argv") != expected_argv
        or manifest_command.get("shell") != shlex.join(expected_argv)
    ):
        raise WorkerError(f"stage metrics manifest identity differs from its worker run: {candidate}")
    for field in ("stage", "track"):
        command_value = _optional_command_option(expected_argv, f"--{field}")
        if command_value is not None and document.get(field) != command_value:
            raise WorkerError(f"stage metrics manifest {field} differs from its worker command")
    command_seed = _optional_command_option(expected_argv, "--seed")
    if experiment.get("seed") is not None and (
        experiment.get("seed") != spec["seed"] or (command_seed is not None and str(experiment["seed"]) != command_seed)
    ):
        raise WorkerError("stage metrics manifest seed differs from its worker command")
    command_steps = _optional_command_option(expected_argv, "--steps")
    if command_steps is None:
        if experiment.get("steps") is not None:
            raise WorkerError("stage metrics manifest reports steps not bound by its worker command")
    else:
        try:
            expected_steps = int(command_steps)
        except ValueError as exc:
            raise WorkerError("metrics-producing command has non-integer --steps") from exc
        if experiment.get("steps") != expected_steps:
            raise WorkerError("stage metrics manifest steps differ from its worker command")
    metrics = document.get("metrics")
    if not isinstance(metrics, dict):
        raise WorkerError(f"metrics source must contain an object: {candidate}")
    _require_finite_json_numbers(metrics, context=f"metrics source {candidate}")
    return json.loads(json.dumps(metrics, allow_nan=False, sort_keys=True))


def _validate_libero_metrics_manifest(
    document: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    candidate: pathlib.Path,
    launch_metadata: Mapping[str, Any] | None,
    instance_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if set(document) != LIBERO_METRICS_MANIFEST_KEYS or document.get("schema_version") != 1:
        raise WorkerError(f"LIBERO metrics manifest has the wrong schema: {candidate}")
    _reservation_id, instance_id, instance_type = _metric_runtime_identity(launch_metadata, instance_identity)
    command = _metric_command_argv(spec)
    evaluation = _require_exact_keys(
        document.get("evaluation"),
        {"stage", "seed", "suites", "trials_per_task", "metrics"},
        context="LIBERO evaluation",
    )
    source = _require_exact_keys(document.get("source"), {"commit"}, context="LIBERO metrics source")
    image = _require_exact_keys(document.get("image"), {"digest"}, context="LIBERO metrics image")
    dataset = _require_exact_keys(document.get("dataset"), {"name", "revision"}, context="LIBERO metrics dataset")
    runtime = _require_exact_keys(
        document.get("instance"), {"type", "id", "identity_recorded_by"}, context="LIBERO metrics instance"
    )
    _dataset_name, dataset_revision = _command_dataset_identity(spec, command)
    if (
        document.get("project") != EXPECTED_PROJECT
        or document.get("kind") != "libero-evaluation"
        or document.get("run_id") != spec["run_id"]
        or source.get("commit") != spec["source"]["commit"]
        or image.get("digest") != spec["image"]["digest"]
        or dataset.get("revision") != dataset_revision
        or runtime.get("type") != instance_type
        or runtime.get("id") != instance_id
        or document.get("command") != command
        or evaluation.get("seed") != spec["seed"]
        or evaluation.get("stage") != _optional_command_option(command, "--stage")
    ):
        raise WorkerError(f"LIBERO metrics manifest identity differs from its worker run: {candidate}")
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, dict):
        raise WorkerError(f"metrics source must contain an object: {candidate}")
    _require_finite_json_numbers(metrics, context=f"metrics source {candidate}")
    return json.loads(json.dumps(metrics, allow_nan=False, sort_keys=True))


def _validate_latency_report(
    document: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    candidate: pathlib.Path,
    expected: Mapping[str, Any],
    launch_metadata: Mapping[str, Any] | None,
    instance_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not _is_declared_latency_report(spec, expected):
        raise WorkerError(f"latency report is not the command-declared expected output: {candidate}")
    if set(document) != LATENCY_REPORT_KEYS or document.get("schema_version") != 1:
        raise WorkerError(f"latency metrics report has the wrong schema: {candidate}")
    _reservation_id, instance_id, instance_type = _metric_runtime_identity(launch_metadata, instance_identity)
    command = _metric_command_argv(spec)
    dataset = _require_exact_keys(document.get("dataset"), {"name", "revision"}, context="latency dataset")
    runtime = _require_exact_keys(
        document.get("runtime"),
        {"instance_type", "instance_id", "image_digest", "instance_identity_source"},
        context="latency runtime",
    )
    dataset_name, dataset_revision = _command_dataset_identity(spec, command)
    try:
        expected_warmups = int(_optional_command_option(command, "--warmups") or 500)
        expected_iterations = int(_optional_command_option(command, "--iterations") or 10_000)
    except ValueError as exc:
        raise WorkerError("latency command has non-integer timing counts") from exc
    stage = _optional_command_option(command, "--stage")
    expected_denoise_steps = 10 if stage in {"base", "shallow"} else 1
    runner = document.get("runner")
    expected_backend = "tensorrt" if _optional_command_option(command, "--backend") == "tensorrt" else "torch-eager"
    if (
        document.get("stage") != stage
        or document.get("track") != _optional_command_option(command, "--track")
        or dataset.get("revision") != dataset_revision
        or (dataset_name is not None and dataset.get("name") != dataset_name)
        or runtime.get("image_digest") != spec["image"]["digest"]
        or runtime.get("instance_type") != instance_type
        or runtime.get("instance_id") != instance_id
        or document.get("batch_size") != 1
        or document.get("warmups") != expected_warmups
        or document.get("iterations") != expected_iterations
        or not isinstance(runner, Mapping)
        or runner.get("backend") != expected_backend
        or runner.get("num_denoise_steps") != expected_denoise_steps
        or document.get("official_protocol") is not (expected_warmups == 500 and expected_iterations == 10_000)
    ):
        raise WorkerError(f"latency metrics report identity differs from its worker run: {candidate}")
    latency = document.get("latency")
    numerical_smoke = document.get("numerical_smoke")
    if (
        not isinstance(latency, dict)
        or not latency
        or not (numerical_smoke is None or isinstance(numerical_smoke, dict))
    ):
        raise WorkerError(f"latency metrics report has invalid metrics: {candidate}")
    metrics = {"latency": latency, "numerical_smoke": numerical_smoke}
    _require_finite_json_numbers(metrics, context=f"metrics source {candidate}")
    return json.loads(json.dumps(metrics, allow_nan=False, sort_keys=True))


def _extract_metrics_document(
    document: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    candidate: pathlib.Path,
    expected: Mapping[str, Any],
    launch_metadata: Mapping[str, Any] | None,
    instance_identity: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    expected_path = pathlib.PurePosixPath(str(expected["path"]))
    strict_metric_source = expected["kind"] == "manifest" or expected_path.suffix == ".json"
    expected_command = _metric_command_argv(spec)
    if candidate.name == TRAINING_METRICS_FILENAME:
        if set(document) != TRAINING_METRICS_KEYS or document.get("schema_version") != 1:
            raise WorkerError("training metrics sidecar has the wrong schema")
        target = pathlib.PurePosixPath(str(expected["path"]))
        try:
            expected_step = int(target.parts[-1])
            expected_config, expected_experiment = target.parts[-3:-1]
        except (ValueError, IndexError) as exc:
            raise WorkerError("training metrics checkpoint path has no config/experiment/step identity") from exc
        if (
            document.get("config_name") != expected_config
            or document.get("exp_name") != expected_experiment
            or document.get("global_step") != expected_step
        ):
            raise WorkerError("training metrics sidecar differs from its checkpoint config/experiment/step")
        metrics = document.get("metrics")
    elif (
        _is_declared_latency_report(spec, expected)
        and candidate.name == pathlib.PurePosixPath(str(expected["path"])).name
    ):
        return _validate_latency_report(
            document,
            spec=spec,
            candidate=candidate,
            expected=expected,
            launch_metadata=launch_metadata,
            instance_identity=instance_identity,
        )
    elif document.get("kind") == "libero-evaluation":
        if document.get("command") != expected_command and not strict_metric_source:
            return None
        return _validate_libero_metrics_manifest(
            document,
            spec=spec,
            candidate=candidate,
            launch_metadata=launch_metadata,
            instance_identity=instance_identity,
        )
    elif "metrics" in document:
        manifest_command = document.get("command")
        manifest_argv = manifest_command.get("argv") if isinstance(manifest_command, Mapping) else None
        if manifest_argv != expected_command and not strict_metric_source:
            return None
        return _validate_stage_metrics_manifest(
            document,
            spec=spec,
            candidate=candidate,
            launch_metadata=launch_metadata,
            instance_identity=instance_identity,
        )
    elif isinstance(document.get("evaluation"), Mapping) and "metrics" in document["evaluation"]:
        if strict_metric_source:
            raise WorkerError(f"metrics manifest is not a recognized schema: {candidate}")
        return None
    else:
        return None
    if metrics is None:
        return None
    if not isinstance(metrics, dict):
        raise WorkerError(f"metrics source must contain an object: {candidate}")
    _require_finite_json_numbers(metrics, context=f"metrics source {candidate}")
    return json.loads(json.dumps(metrics, allow_nan=False, sort_keys=True))


def collect_run_metrics(
    spec: Mapping[str, Any],
    root: pathlib.Path,
    committed_markers: Sequence[pathlib.Path],
    *,
    launch_metadata: Mapping[str, Any] | None = None,
    instance_identity: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Copy metrics only from files covered by immutable expected-output markers.

    Every checkpoint produced by a training command must contain its training
    sidecar. Its live hash, and every stage manifest hash, must still equal the
    marker created after the container exited.
    """

    marker_by_name = {path.name: path for path in committed_markers}
    collected: dict[str, Any] = {}
    provenance: list[dict[str, Any]] = []
    resolved_root = root.resolve()
    for expected in sorted(spec.get("expected_outputs", []), key=lambda item: str(item["name"])):
        marker_name = f"expected-{expected['name']}.ready.json"
        marker_path = marker_by_name.get(marker_name)
        if marker_path is None:
            raise WorkerError(f"metrics collection is missing committed marker for {expected['name']}")
        marker = _read_json(marker_path)
        marker_records = marker.get("artifacts")
        if marker.get("schema_version") != 1 or not isinstance(marker_records, list):
            raise WorkerError(f"expected-output marker is malformed: {marker_path}")
        covered = {
            str(record.get("path")): record
            for record in marker_records
            if isinstance(record, dict) and isinstance(record.get("path"), str)
        }
        relative_target = _safe_relative_path(str(expected["path"]), "metrics expected output")
        target = root.joinpath(*relative_target.parts)
        training_relative = (relative_target / TRAINING_METRICS_FILENAME).as_posix()
        training_output = (
            expected["kind"] == "checkpoint" and "scripts/train_pytorch.py" in spec["container"]["command"]
        )
        if training_output and training_relative not in covered:
            raise WorkerError("training checkpoint is missing its mandatory hash-covered metrics sidecar")
        if (
            expected["kind"] == "checkpoint"
            and training_relative in covered
            and not (target / TRAINING_METRICS_FILENAME).is_file()
        ):
            raise WorkerError("hash-covered training metrics sidecar disappeared after output commit")
        for candidate in _metric_candidates(spec, expected, target):
            try:
                relative = candidate.resolve().relative_to(resolved_root).as_posix()
            except ValueError as exc:
                raise WorkerError(f"metrics source escapes the output root: {candidate}") from exc
            record = covered.get(relative)
            if record is None:
                raise WorkerError(f"metrics source is not covered by its expected-output marker: {relative}")
            digest = record.get("sha256")
            if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None or sha256_file(candidate) != digest:
                raise WorkerError(f"metrics source changed after expected-output commit: {relative}")
            document = _read_json(candidate)
            metrics = _extract_metrics_document(
                document,
                spec=spec,
                candidate=candidate,
                expected=expected,
                launch_metadata=launch_metadata,
                instance_identity=instance_identity,
            )
            if metrics is None:
                continue
            if relative in collected:
                raise WorkerError(f"duplicate metrics source path: {relative}")
            collected[relative] = metrics
            provenance.append(
                {
                    "expected_output": expected["name"],
                    "path": relative,
                    "sha256": digest,
                }
            )
    return collected, provenance


def worker_cost_record(launch_metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Expose the authoritative reservation projection without guessing billing."""

    return {
        "reservation_id": launch_metadata["reservation_id"],
        "projected_usd": launch_metadata["projected_compute_usd"],
        "actual_usd": None,
        "actual_recorded_by": "versioned cost ledger after AWS billing reconciliation",
    }


class SegmentLog:
    def __init__(self, manager: OutputManager):
        self.manager = manager
        self.index = 0
        self.bytes_written = 0
        self.path: pathlib.Path | None = None
        self.stream: Any = None
        self._open()

    def _open(self) -> None:
        self.path = self.manager.root / ".active" / f"container-{self.index:06d}.partial"
        self.stream = self.path.open("wb")
        self.bytes_written = 0

    def write(self, data: bytes) -> None:
        self.stream.write(data)
        self.stream.flush()
        self.bytes_written += len(data)
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()

    def rotate(self) -> None:
        assert self.path is not None
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        if self.bytes_written:
            completed = self.manager.root / "logs" / f"container-{self.index:06d}.log"
            os.replace(self.path, completed)
            self.manager.create_marker("log", [completed], f"log-{self.index:06d}.ready.json")
            self.index += 1
        else:
            self.path.unlink(missing_ok=True)
        self._open()

    def close(self) -> None:
        self.rotate()
        self.stream.close()
        assert self.path is not None
        self.path.unlink(missing_ok=True)


def run_container_until_deadline(
    command: Sequence[str],
    manager: OutputManager,
    soft_deadline: dt.datetime,
    spec: Mapping[str, Any],
    runner: CommandRunner,
    *,
    periodic_callback: Callable[[], None] | None = None,
) -> tuple[int, str | None]:
    name = f"pi05-{spec['run_id']}"
    existing = runner(["docker", "container", "ls", "-a", "--filter", f"name=^{name}$", "--format", "{{.Names}}"])
    if existing.strip():
        raise WorkerError(f"container name already exists; refusing destructive reuse: {name}")
    process = subprocess.Popen(list(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert process.stdout is not None
    os.set_blocking(process.stdout.fileno(), False)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    log = SegmentLog(manager)
    requested_signal: str | None = None
    stop_requested = False
    previous_handlers: dict[int, Any] = {}

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal requested_signal, stop_requested
        requested_signal = signal.Signals(signum).name
        stop_requested = True

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, handle_signal)
    interval = spec["timing"]["sync_interval_seconds"]
    next_sync = time.monotonic() + interval
    try:
        while True:
            if dt.datetime.now(UTC) >= soft_deadline:
                requested_signal = requested_signal or "SOFT_DEADLINE"
                stop_requested = True
            if stop_requested and process.poll() is None:
                runner(["docker", "stop", "--time", str(spec["timing"]["stop_grace_seconds"]), name])
                stop_requested = False
            events = selector.select(timeout=1.0)
            for key, _ in events:
                try:
                    data = os.read(key.fd, 1024 * 1024)
                except BlockingIOError:
                    data = b""
                if data:
                    log.write(data)
            now_mono = time.monotonic()
            if now_mono >= next_sync:
                if periodic_callback is not None:
                    periodic_callback()
                log.rotate()
                manager.sync_once()
                next_sync = now_mono + interval
            if process.poll() is not None:
                while True:
                    try:
                        data = os.read(process.stdout.fileno(), 1024 * 1024)
                    except BlockingIOError:
                        continue
                    if not data:
                        break
                    log.write(data)
                break
        return process.wait(), requested_signal
    finally:
        cleanup_errors: list[Exception] = []
        propagating_exception = sys.exc_info()[0] is not None
        for signum, handler in previous_handlers.items():
            try:
                signal.signal(signum, handler)
            except Exception as exc:
                cleanup_errors.append(exc)
        if process.poll() is None:
            try:
                runner(["docker", "stop", "--time", str(spec["timing"]["stop_grace_seconds"]), name])
            except Exception as exc:
                cleanup_errors.append(exc)
                with contextlib.suppress(Exception):
                    runner(["docker", "kill", name])
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired as exc:
                cleanup_errors.append(exc)
                with contextlib.suppress(Exception):
                    runner(["docker", "kill", name])
                process.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=5)
        try:
            while True:
                try:
                    data = os.read(process.stdout.fileno(), 1024 * 1024)
                except (BlockingIOError, OSError):
                    break
                if not data:
                    break
                log.write(data)
        except Exception as exc:
            cleanup_errors.append(exc)
        try:
            selector.close()
        except Exception as exc:
            cleanup_errors.append(exc)
        try:
            log.close()
        except Exception as exc:
            cleanup_errors.append(exc)
        try:
            if periodic_callback is not None:
                periodic_callback()
        except Exception as exc:
            cleanup_errors.append(exc)
        try:
            manager.sync_once()
        except Exception as exc:
            cleanup_errors.append(exc)
        # This container name is owned by this invocation because preflight
        # rejected any pre-existing container.  Force removal is therefore a
        # final fail-safe against a paid worker surviving an error path.
        with contextlib.suppress(Exception):
            runner(["docker", "rm", "--force", name])
        if cleanup_errors:
            detail = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
            if not propagating_exception:
                raise WorkerError(f"container cleanup or final log sync failed: {detail}")
            print(f"CONTAINER CLEANUP WARNING: {detail}", file=sys.stderr)


def _write_json_new(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    """Atomically create a host-owned JSON artifact without following collisions."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise WorkerError(f"JSON destination parent is not a safe directory: {path.parent}")
    temporary: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".pi05-host-json-",
            delete=False,
        ) as stream:
            temporary = pathlib.Path(stream.name)
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise WorkerError(f"JSON destination already exists; refusing overwrite: {path}") from exc
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def get_instance_identity(opener: Any = urllib.request.urlopen) -> dict[str, Any]:
    token_request = urllib.request.Request(
        "http://169.254.169.254/latest/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
    )
    try:
        with opener(token_request, timeout=3) as response:
            token = response.read().decode()
        identity_request = urllib.request.Request(
            "http://169.254.169.254/latest/dynamic/instance-identity/document",
            headers={"X-aws-ec2-metadata-token": token},
        )
        with opener(identity_request, timeout=3) as response:
            identity = json.load(response)
    except Exception as exc:
        raise WorkerError(f"IMDSv2 instance identity failed: {exc}") from exc
    if not isinstance(identity, dict):
        raise WorkerError("IMDSv2 identity document was not an object")
    return identity


def validate_instance_identity(
    spec: Mapping[str, Any], launch_metadata: Mapping[str, Any], identity: Mapping[str, Any]
) -> tuple[str, str]:
    """Bind the executing worker, and optionally an engine, to live IMDSv2 identity."""
    instance_id = identity.get("instanceId")
    instance_type = identity.get("instanceType")
    if (
        str(identity.get("accountId")) != EXPECTED_ACCOUNT
        or identity.get("region") != EXPECTED_REGION
        or instance_type != launch_metadata.get("instance_type")
        or not isinstance(instance_id, str)
        or INSTANCE_ID_RE.fullmatch(instance_id) is None
        or not isinstance(instance_type, str)
        or INSTANCE_TYPE_RE.fullmatch(instance_type) is None
    ):
        raise WorkerError("IMDS identity differs from the pinned account, region, launcher type, or EC2 identity")
    placement = spec.get("placement")
    if isinstance(placement, Mapping) and instance_id != placement.get("instance_id"):
        raise WorkerError("live IMDS instanceId differs from exact existing-instance placement")
    return instance_id, instance_type


def execute_worker(
    spec: Mapping[str, Any], source_evidence: Mapping[str, Any], launch_metadata: Mapping[str, Any]
) -> int:
    runner = SubprocessRunner()
    validate_source_evidence(spec, source_evidence)
    hard_deadline, soft_deadline = validate_launch_metadata(
        spec, launch_metadata, command_path=pathlib.Path("/opt/pi05/run-command.sh")
    )
    identity = get_instance_identity()
    instance_id, instance_type = validate_instance_identity(spec, launch_metadata, identity)
    verify_aws_boundary(spec, runner)
    selection = prepare_scratch(spec, runner)
    if selection.run_root is None:
        raise WorkerError("scratch preparation did not return a run-specific workspace")
    run_root = pathlib.Path(selection.run_root)
    staged = []
    for artifact in spec["artifacts"]:
        if dt.datetime.now(UTC) >= soft_deadline:
            raise WorkerError("soft deadline reached while staging immutable inputs")
        staged.append(stage_artifact(artifact, spec, run_root, runner))
    staged_manifests = {
        artifact["name"]: _read_json(run_root / "inputs" / ".manifests" / f"{artifact['name']}.json")
        for artifact in spec["artifacts"]
    }
    validate_input_cross_contracts(spec, staged_manifests)
    # Artifact parent directories are created under the bootstrap's 077 umask.
    # Apply one permission handoff only after all siblings and manifests exist.
    make_staged_input_container_readable(run_root / "inputs")
    resume_checkpoint = restore_resume_checkpoint(spec, run_root)
    repo_digests = verify_and_pull_image(spec, runner)
    manager = OutputManager(spec, run_root / "output", runner)
    docker_command = build_docker_command(
        spec,
        pathlib.Path(source_evidence["checkout_path"]),
        run_root,
        instance_id=instance_id,
        instance_type=instance_type,
    )
    started_at = dt.datetime.now(UTC)
    exit_code: int | None = None
    termination: str | None = None
    failure: str | None = None
    receipts: list[dict[str, Any]] = []
    published_inputs: list[dict[str, Any]] = []
    expected_output_markers: list[pathlib.Path] = []
    run_metrics: dict[str, Any] = {}
    metrics_provenance: list[dict[str, Any]] = []
    try:
        exit_code, termination = run_container_until_deadline(docker_command, manager, soft_deadline, spec, runner)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    if exit_code == 0 and termination is None and failure is None:
        try:
            expected_output_markers = manager.commit_expected_outputs()
            run_metrics, metrics_provenance = collect_run_metrics(
                spec,
                manager.root,
                expected_output_markers,
                launch_metadata=launch_metadata,
                instance_identity=identity,
            )
        except Exception as exc:
            failure = f"expected output validation failed: {type(exc).__name__}: {exc}"
    try:
        receipts = manager.sync_once()
    except Exception as exc:
        failure = failure or f"final sync failed: {type(exc).__name__}: {exc}"
    if exit_code == 0 and termination is None and failure is None:
        try:
            published_inputs, receipts = manager.publish_expected_inputs(receipts)
        except Exception as exc:
            failure = f"worker-input publication failed: {type(exc).__name__}: {exc}"
    status = "succeeded" if exit_code == 0 and termination is None and failure is None else "failed"
    command_sha = hashlib.sha256(
        json.dumps(spec["container"]["command"], separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    run_manifest = {
        "schema_version": 1,
        "project": EXPECTED_PROJECT,
        "run_id": spec["run_id"],
        "status": status,
        "started_at": started_at.isoformat(),
        "finished_at": dt.datetime.now(UTC).isoformat(),
        "hard_deadline_utc": hard_deadline.isoformat(),
        "soft_deadline_utc": soft_deadline.isoformat(),
        "termination": termination,
        "exit_code": exit_code,
        "failure": failure,
        "controller_source": dict(spec["controller_source"]),
        "source": dict(spec["source"]),
        "source_evidence": dict(source_evidence),
        "image": {**dict(spec["image"]), "repo_digests": repo_digests},
        "datasets": {item["name"]: item["revision"] for item in staged if item["kind"] == "dataset"},
        "staged_inputs": staged,
        "resume_checkpoint": resume_checkpoint,
        "container": {
            "command": list(spec["container"]["command"]),
            "command_sha256": command_sha,
            "seed": spec["seed"],
        },
        "cost": worker_cost_record(launch_metadata),
        "metrics": run_metrics,
        "metrics_provenance": metrics_provenance,
        "expected_outputs": list(spec.get("expected_outputs", [])),
        "committed_expected_output_markers": [path.name for path in expected_output_markers],
        "published_inputs": published_inputs,
        "instance": identity,
        "launch": dict(launch_metadata),
        "scratch": {
            "device": selection.path,
            "serial": selection.serial,
            "reused": selection.reuse,
            "mounted_at": selection.mounted_at or str(SCRATCH_ROOT),
            "filesystem": selection.filesystem,
            "run_root": str(run_root),
        },
        "completed_receipts_before_manifest": receipts,
    }
    manifest_path = manager.root / "manifests" / "run-manifest.json"
    _write_json_new(manifest_path, run_manifest)
    manager.create_marker("manifest", [manifest_path], "run-manifest.ready.json")
    try:
        receipts = manager.sync_once()
        manifest_receipt = next(item for item in receipts if item["marker"] == "run-manifest.ready.json")
        evidence = {
            "schema_version": 1,
            "run_id": spec["run_id"],
            "status": status,
            "final_sync_succeeded": True,
            "recorded_at": dt.datetime.now(UTC).isoformat(),
            "run_manifest_receipt": manifest_receipt,
            "receipt_count": len(receipts),
        }
        evidence_path = manager.root / "manifests" / "final-sync-evidence.json"
        _write_json_new(evidence_path, evidence)
        manager.create_marker("manifest", [evidence_path], "final-sync-evidence.ready.json")
        manager.sync_once()
    except Exception as exc:
        print(f"FINAL SYNC FAILED: {exc}", file=sys.stderr)
        return 3
    return 0 if status == "succeeded" else (exit_code if exit_code not in (None, 0) else 3)


def render_bootstrap_command(args: argparse.Namespace) -> str:
    bootstrap = parse_s3_uri(args.bootstrap_s3_uri)
    spec = parse_s3_uri(args.spec_s3_uri)
    for label, value in (
        ("bootstrap version", args.bootstrap_version_id),
        ("spec version", args.spec_version_id),
    ):
        if not value or any(character in value for character in "\x00\r\n"):
            raise WorkerError(f"{label} must be a non-empty single-line S3 version ID")
    for label, value in (("bootstrap SHA-256", args.bootstrap_sha256), ("spec SHA-256", args.spec_sha256)):
        if SHA256_RE.fullmatch(value or "") is None:
            raise WorkerError(f"{label} is invalid")
    env = {
        "EXPECTED_ACCOUNT_ID": EXPECTED_ACCOUNT,
        "EXPECTED_AWS_REGION": EXPECTED_REGION,
        "WORKER_SPEC_S3_URI": spec.uri,
        "WORKER_SPEC_VERSION_ID": args.spec_version_id,
        "WORKER_SPEC_SHA256": args.spec_sha256,
        "WORKER_EXECUTE": "1" if args.execute else "0",
    }
    env_text = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    destination = "/opt/pi05/worker-bootstrap.sh"
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "umask 077",
        "install -d -m 0700 /opt/pi05",
        shlex.join(
            [
                "aws",
                "s3api",
                "get-object",
                "--bucket",
                bootstrap.bucket,
                "--key",
                bootstrap.key,
                "--version-id",
                args.bootstrap_version_id,
                "--expected-bucket-owner",
                EXPECTED_ACCOUNT,
                "--region",
                EXPECTED_REGION,
                destination,
            ]
        ),
        f"printf '%s  %s\\n' {shlex.quote(args.bootstrap_sha256)} {shlex.quote(destination)} | sha256sum --check --status",
        f"chmod 0500 {destination}",
        f"env {env_text} {destination}",
    ]
    return "\n".join(lines) + "\n"


def render_plan(spec: Mapping[str, Any], launch_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result = {
        "mode": "dry-run",
        "run_id": spec["run_id"],
        "controller_source": spec["controller_source"],
        "source": spec["source"],
        "image": spec["image"],
        "artifacts": [
            {
                "name": item["name"],
                "kind": item["kind"],
                "revision": item["revision"],
                "manifest": item["manifest"],
                "payload_s3_uri": item["payload_s3_uri"],
            }
            for item in spec["artifacts"]
        ],
        "expected_outputs": spec.get("expected_outputs", []),
        "resume_checkpoint": spec.get("resume_checkpoint"),
        "output": spec["output"],
        "scratch": spec["scratch"],
        "container_command": spec["container"]["command"],
        "mutations_authorized": False,
    }
    if launch_metadata is not None:
        hard, soft = validate_launch_metadata(spec, launch_metadata)
        result["hard_deadline_utc"] = hard.isoformat()
        result["soft_deadline_utc"] = soft.isoformat()
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action")
    worker = subparsers.add_parser("run", help="validate or execute a worker spec")
    worker.add_argument("--spec", required=True, type=pathlib.Path)
    worker.add_argument("--source-evidence", type=pathlib.Path)
    worker.add_argument("--launch-metadata", type=pathlib.Path, default=pathlib.Path("/opt/pi05/launch-metadata.json"))
    worker.add_argument("--execute", action="store_true", help="authorize storage, S3, Docker, and container mutations")
    bootstrap = subparsers.add_parser("render-bootstrap", help="render a launcher command file; dry-run by default")
    bootstrap.add_argument("--bootstrap-s3-uri", required=True)
    bootstrap.add_argument("--bootstrap-version-id", required=True)
    bootstrap.add_argument("--bootstrap-sha256", required=True)
    bootstrap.add_argument("--spec-s3-uri", required=True)
    bootstrap.add_argument("--spec-version-id", required=True)
    bootstrap.add_argument("--spec-sha256", required=True)
    bootstrap.add_argument(
        "--execute", action="store_true", help="render a bootstrap that executes rather than validates"
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.action is None:
        print("an action is required: run or render-bootstrap", file=sys.stderr)
        return 2
    try:
        if args.action == "render-bootstrap":
            print(render_bootstrap_command(args), end="")
            return 0
        spec = validate_worker_spec(_read_json(args.spec))
        if not args.execute:
            launch = _read_json(args.launch_metadata) if args.launch_metadata.exists() else None
            print(json.dumps(render_plan(spec, launch), indent=2, sort_keys=True))
            return 0
        if args.source_evidence is None:
            raise WorkerError("--source-evidence is required with --execute")
        source_evidence = _read_json(args.source_evidence)
        launch_metadata = _read_json(args.launch_metadata)
        return execute_worker(spec, source_evidence, launch_metadata)
    except (WorkerError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"WORKER REJECTED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
