#!/usr/bin/env python3
"""Resolve and validate a numeric PyTorch training checkpoint directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib

from safetensors import safe_open
import torch

CORE_STATE_FILES = ("model.safetensors", "optimizer.pt", "metadata.pt", "resume-state.json", "wandb_id.txt")
RESUME_STATE_FILES = ["metadata.pt", "model.safetensors", "optimizer.pt", "wandb_id.txt"]
SHA256_LENGTH = 64
OVERFIT_WINDOW = 20
OVERFIT_MINIMUM_DECLINE = 0.20
OVERFIT_REPORT_KEYS = {
    "schema_version",
    "gate_pass",
    "optimizer_steps",
    "window_optimizer_steps",
    "optimizer_step_losses",
    "initial_loss_mean",
    "final_loss_mean",
    "relative_loss_decline",
    "minimum_relative_loss_decline",
    "all_losses_finite",
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


def _read_json(path: pathlib.Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Required checkpoint file is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Checkpoint JSON is invalid: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint JSON must contain an object: {path}")
    return value


def _required_norm_stats(config_name: str) -> pathlib.Path | None:
    if config_name.startswith("pi05_libero_"):
        return pathlib.Path("assets/physical-intelligence/libero/norm_stats.json")
    if config_name.startswith("pi05_droid_"):
        return pathlib.Path("assets/droid/norm_stats.json")
    return None


def canonical_resume_fingerprint(value) -> str:
    payload = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def resume_identity_fingerprint(contract: dict, lineage: dict) -> str:
    return canonical_resume_fingerprint(
        {
            "initialization_lineage": lineage,
            "resume_contract": contract,
        }
    )


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _nonfinite_tensor_paths(value, path: str) -> list[str]:
    invalid: list[str] = []
    if isinstance(value, torch.Tensor):
        if (torch.is_floating_point(value) or torch.is_complex(value)) and not bool(torch.isfinite(value).all().item()):
            invalid.append(path)
    elif isinstance(value, dict):
        for key, item in value.items():
            invalid.extend(_nonfinite_tensor_paths(item, f"{path}.{key}"))
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            invalid.extend(_nonfinite_tensor_paths(item, f"{path}[{index}]"))
    return invalid


def _validate_model_file(path: pathlib.Path) -> None:
    try:
        with safe_open(path, framework="pt", device="cpu") as tensors:
            keys = list(tensors.keys())
            if not keys:
                raise ValueError("model.safetensors contains no tensors")
            invalid = []
            for key in keys:
                value = tensors.get_tensor(key)
                if (torch.is_floating_point(value) or torch.is_complex(value)) and not bool(
                    torch.isfinite(value).all().item()
                ):
                    invalid.append(key)
    except Exception as error:
        if isinstance(error, ValueError) and str(error).startswith("model.safetensors"):
            raise
        raise ValueError(f"model.safetensors is not a readable tensor archive: {path}") from error
    if invalid:
        raise FloatingPointError(f"model.safetensors contains non-finite tensors: {invalid[:10]}")


def _load_torch_mapping(path: pathlib.Path, *, label: str) -> dict:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"{label} is not a readable PyTorch checkpoint: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a mapping: {path}")
    invalid = _nonfinite_tensor_paths(value, label)
    if invalid:
        raise FloatingPointError(f"{label} contains non-finite tensors: {invalid[:10]}")
    return value


def _validate_resume_contract(state: dict, step: int) -> dict:
    if set(state) != RESUME_STATE_KEYS:
        raise ValueError(f"resume-state.json keys differ from the schema: {sorted(set(state) ^ RESUME_STATE_KEYS)}")
    contract = state.get("resume_contract")
    if not isinstance(contract, dict) or set(contract) != RESUME_CONTRACT_KEYS:
        actual = set(contract) if isinstance(contract, dict) else set()
        raise ValueError(f"resume contract keys differ from the schema: {sorted(actual ^ RESUME_CONTRACT_KEYS)}")
    lineage = state.get("initialization_lineage")
    fingerprint = state.get("resume_fingerprint_sha256")
    if not _is_sha256(fingerprint) or fingerprint != resume_identity_fingerprint(contract, lineage):
        raise ValueError("resume fingerprint does not match the canonical embedded contract and lineage")
    dataset = contract.get("dataset")
    if (
        state.get("schema_version") != 2
        or state.get("global_step") != step
        or state.get("config_name") != contract.get("config_name")
        or state.get("exp_name") != contract.get("exp_name")
        or state.get("state_files") != RESUME_STATE_FILES
        or contract.get("schema_version") != 1
        or not isinstance(contract.get("config_name"), str)
        or not contract["config_name"]
        or not isinstance(contract.get("exp_name"), str)
        or not contract["exp_name"]
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
        or not _is_sha256(dataset.get("factory_config_sha256"))
        or not isinstance(dataset.get("repo_id"), str)
        or not dataset["repo_id"]
        or not (dataset.get("codebase_version") is None or isinstance(dataset.get("codebase_version"), str))
        or not (dataset.get("episode_prompt_path") is None or isinstance(dataset.get("episode_prompt_path"), str))
        or not (dataset.get("episode_prompt_sha256") is None or _is_sha256(dataset.get("episode_prompt_sha256")))
        or not isinstance(dataset.get("action_sequence_keys"), list)
        or not dataset["action_sequence_keys"]
        or any(not isinstance(key, str) or not key for key in dataset["action_sequence_keys"])
        or not (dataset.get("asset_id") is None or isinstance(dataset.get("asset_id"), str))
        or not isinstance(dataset.get("use_quantile_norm"), bool)
        or not isinstance(dataset.get("prompt_from_task"), bool)
        or not (dataset.get("normalization_sha256") is None or _is_sha256(dataset.get("normalization_sha256")))
        or not (
            dataset.get("recovery_provenance_sha256") is None or _is_sha256(dataset.get("recovery_provenance_sha256"))
        )
        or contract.get("pytorch_training_precision") not in {"bfloat16", "float32"}
        or contract.get("stochastic_schedule") != "sha256-v2(model:seed,step,accumulation,rank;loader:seed,epoch)"
        or not isinstance(contract.get("one_batch_overfit"), bool)
        or not isinstance(contract.get("one_batch_overfit_min_relative_decline"), int | float)
        or isinstance(contract.get("one_batch_overfit_min_relative_decline"), bool)
        or not 0.0 < contract["one_batch_overfit_min_relative_decline"] < 1.0
        or not isinstance(lineage, dict)
        or set(lineage) != {"kind", "model_sha256"}
        or lineage.get("kind") not in {"shallow_teacher_transplant", "pytorch_source", "random_initialization"}
        or (lineage["kind"] != "random_initialization" and not _is_sha256(lineage.get("model_sha256")))
        or (lineage["kind"] == "random_initialization" and lineage.get("model_sha256") is not None)
    ):
        raise ValueError("resume-state.json does not satisfy the resumable training contract")

    config_name = contract["config_name"]
    dataset = contract["dataset"]
    if config_name.startswith(("pi05_libero_", "pi05_droid_")):
        if (
            not isinstance(dataset.get("revision"), str)
            or len(dataset["revision"]) != 40
            or any(character not in "0123456789abcdef" for character in dataset["revision"])
        ):
            raise ValueError("reproduction checkpoint dataset identity lacks an immutable revision")
        if not _is_sha256(dataset.get("normalization_sha256")):
            raise ValueError("reproduction checkpoint lacks a normalization-content fingerprint")
        if config_name.startswith("pi05_droid_") and not _is_sha256(dataset.get("episode_prompt_sha256")):
            raise ValueError("DROID reproduction checkpoint lacks its episode-prompt content fingerprint")
    teacher = contract.get("teacher")
    if config_name.endswith("_distill"):
        if (
            not isinstance(teacher, dict)
            or set(teacher) != {"model_sha256"}
            or not _is_sha256(teacher.get("model_sha256"))
        ):
            raise ValueError("distillation resume contract lacks the teacher model content hash")
        if lineage.get("kind") != "shallow_teacher_transplant" or lineage.get("model_sha256") != teacher.get(
            "model_sha256"
        ):
            raise ValueError("distillation initialization lineage differs from the teacher identity")
    elif teacher is not None:
        raise ValueError("non-distillation resume contract unexpectedly declares a teacher")
    return contract


def _validate_metadata(metadata: dict, contract: dict, step: int) -> None:
    config = metadata.get("config")
    if metadata.get("global_step") != step or metadata.get("data_split") != contract.get("data_split"):
        raise ValueError("metadata.pt step or data split differs from the resume contract")
    if not isinstance(config, dict):
        raise ValueError("metadata.pt does not contain the serialized training config")
    expected = {
        "name": contract["config_name"],
        "exp_name": contract["exp_name"],
        "seed": contract["seed"],
        "batch_size": contract["batch_size"],
        "gradient_accumulation_steps": contract["gradient_accumulation_steps"],
        "lr_schedule": contract["lr_schedule"],
        "optimizer": contract["optimizer"],
        "wandb_enabled": contract["wandb_enabled"],
        "model": contract["model"]["config"],
        "pytorch_training_precision": contract["pytorch_training_precision"],
        "one_batch_overfit": contract["one_batch_overfit"],
        "one_batch_overfit_min_relative_decline": contract["one_batch_overfit_min_relative_decline"],
    }
    actual = {key: config.get(key) for key in expected}
    if canonical_resume_fingerprint(actual) != canonical_resume_fingerprint(expected):
        differing = sorted(key for key in expected if actual.get(key) != expected[key])
        raise ValueError(f"metadata.pt training config differs from the resume contract: {differing}")
    if canonical_resume_fingerprint(config.get("data")) != contract["dataset"]["factory_config_sha256"]:
        raise ValueError("metadata.pt data factory differs from the resume contract")


def validate_training_checkpoint(checkpoint_dir: pathlib.Path, *, require_overfit_report: bool = False) -> dict:
    """Validate the complete resumable state and reproduction-track assets."""
    step = checkpoint_step(checkpoint_dir)
    for name in CORE_STATE_FILES:
        path = checkpoint_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required non-empty checkpoint file is missing: {path}")

    state = _read_json(checkpoint_dir / "resume-state.json")
    contract = _validate_resume_contract(state, step)

    _validate_model_file(checkpoint_dir / "model.safetensors")
    optimizer = _load_torch_mapping(checkpoint_dir / "optimizer.pt", label="optimizer.pt")
    if not isinstance(optimizer.get("state"), dict) or not isinstance(optimizer.get("param_groups"), list):
        raise ValueError("optimizer.pt does not contain optimizer state and parameter groups")
    metadata = _load_torch_mapping(checkpoint_dir / "metadata.pt", label="metadata.pt")
    _validate_metadata(metadata, contract, step)
    if not (checkpoint_dir / "wandb_id.txt").read_text().strip():
        raise ValueError("wandb_id.txt is empty")

    norm_stats = _required_norm_stats(contract["config_name"])
    if norm_stats is not None:
        norm_path = checkpoint_dir / norm_stats
        if not norm_path.is_file() or norm_path.stat().st_size == 0:
            raise FileNotFoundError(f"Required track normalization statistics are missing: {norm_path}")
        norm_payload = _read_json(norm_path)
        norm_stats = norm_payload.get("norm_stats")
        if (
            not isinstance(norm_stats, dict)
            or not norm_stats
            or canonical_resume_fingerprint(norm_stats) != contract["dataset"]["normalization_sha256"]
        ):
            raise ValueError(f"Normalization statistics differ from the resume contract: {norm_path}")

    require_overfit_report = require_overfit_report or contract["exp_name"].endswith("-overfit")
    if require_overfit_report:
        report = _read_json(checkpoint_dir / "overfit-diagnostic.json")
        losses = report.get("optimizer_step_losses")
        numeric_fields = (
            "initial_loss_mean",
            "final_loss_mean",
            "relative_loss_decline",
            "minimum_relative_loss_decline",
        )
        if (
            set(report) != OVERFIT_REPORT_KEYS
            or report.get("schema_version") != 1
            or report.get("gate_pass") is not True
            or report.get("all_losses_finite") is not True
            or report.get("optimizer_steps") != step
            or step < 2 * OVERFIT_WINDOW
            or report.get("window_optimizer_steps") != OVERFIT_WINDOW
            or not isinstance(losses, list)
            or len(losses) != step
            or any(
                not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value)
                for value in losses
            )
            or any(
                not isinstance(report.get(field), int | float)
                or isinstance(report.get(field), bool)
                or not math.isfinite(report[field])
                for field in numeric_fields
            )
            or report["initial_loss_mean"] <= 0.0
            or report["final_loss_mean"] < 0.0
            or not math.isclose(
                report["initial_loss_mean"],
                math.fsum(losses[:OVERFIT_WINDOW]) / OVERFIT_WINDOW,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                report["final_loss_mean"],
                math.fsum(losses[-OVERFIT_WINDOW:]) / OVERFIT_WINDOW,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or report["minimum_relative_loss_decline"] < OVERFIT_MINIMUM_DECLINE
            or not math.isclose(
                report["minimum_relative_loss_decline"],
                contract["one_batch_overfit_min_relative_decline"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or report["relative_loss_decline"] < report["minimum_relative_loss_decline"]
            or not math.isclose(
                report["relative_loss_decline"],
                (report["initial_loss_mean"] - report["final_loss_mean"]) / report["initial_loss_mean"],
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            or contract.get("one_batch_overfit") is not True
            or contract.get("one_batch_overfit_min_relative_decline", 0.0) < OVERFIT_MINIMUM_DECLINE
        ):
            raise ValueError("overfit-diagnostic.json does not contain a passing finite optimizer-smoke result")
    return state


def checkpoint_step(checkpoint_dir: pathlib.Path) -> int:
    try:
        return int(checkpoint_dir.name)
    except ValueError as error:
        raise ValueError(f"Checkpoint directory must have a numeric step name: {checkpoint_dir}") from error


def resolve_checkpoint(path: pathlib.Path | str, *, step: int | None = None) -> pathlib.Path:
    """Resolve ``RUN_ROOT/STEP`` and reject ambiguous run roots.

    A direct numeric checkpoint directory is accepted without ``step``. A run
    root always requires an explicit step, which prevents an evaluation or a
    SnapFlow initialization from silently selecting a different checkpoint.
    """
    path = pathlib.Path(path).expanduser().resolve()
    if path.is_file():
        if path.name != "model.safetensors":
            raise ValueError(f"Expected model.safetensors, got {path}")
        path = path.parent

    if step is not None:
        if step <= 0:
            raise ValueError("Checkpoint step must be positive")
        candidate = path if path.name == str(step) and (path / "model.safetensors").is_file() else path / str(step)
    else:
        candidate = path
        checkpoint_step(candidate)

    weights = candidate / "model.safetensors"
    if not weights.is_file():
        numeric_children = sorted(
            int(child.name)
            for child in path.glob("[0-9]*")
            if child.is_dir() and child.name.isdigit() and (child / "model.safetensors").is_file()
        )
        hint = f"; available steps: {numeric_children}" if numeric_children else ""
        raise FileNotFoundError(f"Checkpoint weights not found: {weights}{hint}")
    checkpoint_step(candidate)
    validate_training_checkpoint(candidate)
    return candidate


def resolve_weights_directory(path: pathlib.Path | str, *, step: int | None = None) -> pathlib.Path:
    """Resolve either a released artifact directory or a numeric training step."""
    path = pathlib.Path(path).expanduser().resolve()
    if step is not None:
        return resolve_checkpoint(path, step=step)
    if path.is_file():
        if path.name != "model.safetensors":
            raise ValueError(f"Expected model.safetensors, got {path}")
        path = path.parent
    if not (path / "model.safetensors").is_file():
        raise FileNotFoundError(f"Checkpoint weights not found: {path / 'model.safetensors'}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=pathlib.Path)
    parser.add_argument("--step", type=int)
    parser.add_argument("--require-overfit-report", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = resolve_checkpoint(args.path, step=args.step)
    if args.require_overfit_report:
        validate_training_checkpoint(checkpoint, require_overfit_report=True)
    print(checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
