#!/usr/bin/env python3
"""Evaluate Shallow-pi or SnapFlow on a deterministic held-out golden corpus."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
from typing import Any

import numpy as np
import torch

import openpi.models.model as _model
import openpi.models.pi0_config as _pi0_config
import openpi.training.config as _config

if __package__:
    from scripts.repro_checkpoint import checkpoint_step
    from scripts.repro_checkpoint import resolve_checkpoint
    from scripts.repro_checkpoint import resolve_weights_directory
    from scripts.repro_checkpoint import validate_training_checkpoint
    from scripts.repro_make_golden import canonical_sha256
    from scripts.repro_make_golden import config_provenance
    from scripts.repro_make_golden import sha256_file
    from scripts.repro_offline_metrics import compute_metrics
else:
    from repro_checkpoint import checkpoint_step
    from repro_checkpoint import resolve_checkpoint
    from repro_checkpoint import resolve_weights_directory
    from repro_checkpoint import validate_training_checkpoint
    from repro_make_golden import canonical_sha256
    from repro_make_golden import config_provenance
    from repro_make_golden import sha256_file
    from repro_offline_metrics import compute_metrics

TRAINING_IDENTITY_FIELDS = (
    "name",
    "project_name",
    "exp_name",
    "model",
    "pytorch_weight_path",
    "teacher_pytorch_weight_path",
    "pytorch_training_precision",
    "lr_schedule",
    "optimizer",
    "ema_decay",
    "seed",
    "batch_size",
    "gradient_accumulation_steps",
    "offline_holdout_samples",
)


def observation_from_arrays(arrays: dict[str, np.ndarray], metadata: dict[str, Any], device: torch.device):
    image_layout = metadata.get("image_layout")
    if image_layout not in {"BCHW", "BHWC"}:
        raise ValueError(f"Unsupported or missing image_layout: {image_layout!r}")

    def tensor(value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(value)).to(device)

    images = {}
    for name in metadata["image_names"]:
        image = arrays[f"image__{name}"]
        if image_layout == "BHWC":
            image = np.transpose(image, (0, 3, 1, 2))
        images[name] = tensor(image)
    return _model.Observation(
        images=images,
        image_masks={name: tensor(arrays[f"image_mask__{name}"]) for name in metadata["image_names"]},
        state=tensor(arrays["state"]),
        tokenized_prompt=tensor(arrays["tokenized_prompt"]),
        tokenized_prompt_mask=tensor(arrays["tokenized_prompt_mask"]),
        token_ar_mask=None,
        token_loss_mask=None,
    )


def slice_observation(observation: _model.Observation, start: int, stop: int) -> _model.Observation:
    return _model.Observation(
        images={name: value[start:stop] for name, value in observation.images.items()},
        image_masks={name: value[start:stop] for name, value in observation.image_masks.items()},
        state=observation.state[start:stop],
        tokenized_prompt=observation.tokenized_prompt[start:stop],
        tokenized_prompt_mask=observation.tokenized_prompt_mask[start:stop],
        token_ar_mask=None,
        token_loss_mask=None,
    )


def load_model(config_name: str, checkpoint_dir: pathlib.Path, device: torch.device):
    resolved_train_config = _config.get_config(config_name)
    model_config = dataclasses.replace(resolved_train_config.model, pytorch_compile_mode=None)
    runtime_train_config = dataclasses.replace(resolved_train_config, model=model_config)
    model = model_config.load_pytorch(runtime_train_config, str(checkpoint_dir / "model.safetensors"))
    return resolved_train_config, model.to(device).eval()


def checkpoint_training_provenance(
    checkpoint_dir: pathlib.Path,
    *,
    expected_config_name: str,
    expected_step: int,
    expected_model: dict[str, Any] | None = None,
    expected_holdout_samples: int | None = None,
) -> dict[str, Any]:
    metadata_path = checkpoint_dir / "metadata.pt"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Training checkpoint metadata not found: {metadata_path}")
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
    if metadata.get("global_step") != expected_step:
        raise ValueError("Checkpoint metadata global_step does not match its numeric directory")
    saved_config = metadata.get("config")
    if not isinstance(saved_config, dict) or saved_config.get("name") != expected_config_name:
        raise ValueError("Checkpoint metadata does not identify the requested student config")
    if expected_model is not None and saved_config.get("model") != expected_model:
        raise ValueError("Checkpoint model config does not match the resolved student config")
    if expected_holdout_samples is not None and saved_config.get("offline_holdout_samples") != expected_holdout_samples:
        raise ValueError("Checkpoint offline holdout does not match the resolved student config")
    identity = {field: saved_config.get(field) for field in TRAINING_IDENTITY_FIELDS}
    return {
        "training_fingerprint_sha256": canonical_sha256(identity),
        "metadata_sha256": sha256_file(metadata_path),
    }


def checkpoint_initialization_lineage(checkpoint_dir: pathlib.Path) -> dict[str, Any]:
    """Read the validated source-model identity recorded by a numeric checkpoint."""

    state_path = checkpoint_dir / "resume-state.json"
    try:
        state = json.loads(state_path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Training checkpoint resume state not found: {state_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Training checkpoint resume state is invalid JSON: {state_path}") from exc
    lineage = state.get("initialization_lineage")
    if not isinstance(lineage, dict) or set(lineage) != {"kind", "model_sha256"}:
        raise ValueError("Training checkpoint initialization lineage is missing or malformed")
    model_sha256 = lineage.get("model_sha256")
    if (
        lineage.get("kind") not in {"shallow_teacher_transplant", "pytorch_source", "random_initialization"}
        or (
            lineage.get("kind") != "random_initialization"
            and (
                not isinstance(model_sha256, str)
                or len(model_sha256) != 64
                or any(character not in "0123456789abcdef" for character in model_sha256)
            )
        )
        or (lineage.get("kind") == "random_initialization" and model_sha256 is not None)
    ):
        raise ValueError("Training checkpoint initialization lineage has an invalid source-model identity")
    return dict(lineage)


def validate_snapflow_teacher_lineage(
    student_checkpoint: pathlib.Path,
    teacher_checkpoint: pathlib.Path,
) -> dict[str, Any]:
    """Require evaluation against the exact numeric checkpoint that initialized SnapFlow."""

    lineage = checkpoint_initialization_lineage(student_checkpoint)
    teacher_model_sha256 = sha256_file(teacher_checkpoint / "model.safetensors")
    if lineage != {"kind": "pytorch_source", "model_sha256": teacher_model_sha256}:
        raise ValueError(
            "SnapFlow evaluation teacher differs from the checkpoint that initialized the student: "
            f"lineage={lineage}, teacher_model_sha256={teacher_model_sha256}"
        )
    return lineage


def validate_shallow_teacher_lineage(
    student_checkpoint: pathlib.Path,
    teacher_checkpoint: pathlib.Path,
) -> dict[str, Any]:
    """Require a distilled student to retain the selected full-depth teacher identity."""

    lineage = checkpoint_initialization_lineage(student_checkpoint)
    teacher_model_sha256 = sha256_file(teacher_checkpoint / "model.safetensors")
    if lineage != {"kind": "shallow_teacher_transplant", "model_sha256": teacher_model_sha256}:
        raise ValueError(
            "Shallow-pi evaluation teacher differs from the checkpoint that initialized the student: "
            f"lineage={lineage}, teacher_model_sha256={teacher_model_sha256}"
        )
    return lineage


SNAPFLOW_TEACHER_CONFIGS = {
    "pi05_libero_l09_snapflow": {"pi05_libero_l09_distill"},
    "pi05_droid_l09_snapflow": {
        "pi05_droid_l09_distill",
        "pi05_droid_l09_expert_bc_25",
        "pi05_droid_l09_expert_bc_50",
    },
}
SNAPFLOW_CANONICAL_GOLDEN_CONFIG = {
    "pi05_libero_l09_snapflow": "pi05_libero_l09_distill",
    "pi05_droid_l09_snapflow": "pi05_droid_l09_distill",
}
SHALLOW_TEACHER_CONFIGS = {
    "pi05_libero_l09_distill": "pi05_libero",
    "pi05_droid_l09_distill": "pi05_droid_jointpos",
    "pi05_droid_l09_expert_bc_25": "pi05_droid_jointpos",
    "pi05_droid_l09_expert_bc_50": "pi05_droid_jointpos",
}


def canonical_snapflow_golden_config(student_config_name: str, teacher_config_name: str) -> str:
    """Validate the exact per-track SnapFlow teacher and return its canonical golden source."""

    accepted_teachers = SNAPFLOW_TEACHER_CONFIGS.get(student_config_name)
    if accepted_teachers is None or teacher_config_name not in accepted_teachers:
        raise ValueError(
            "SnapFlow evaluation requires the exact accepted per-track Shallow/BC teacher config: "
            f"student={student_config_name!r}, teacher={teacher_config_name!r}"
        )
    return SNAPFLOW_CANONICAL_GOLDEN_CONFIG[student_config_name]


def validate_shallow_teacher_config(student_config_name: str, teacher_config_name: str) -> None:
    """Require the exact released full-depth teacher for a Shallow or BC student."""

    expected_teacher = SHALLOW_TEACHER_CONFIGS.get(student_config_name)
    if expected_teacher is None or teacher_config_name != expected_teacher:
        raise ValueError(
            "Shallow-pi evaluation requires the exact released per-track full-depth teacher config: "
            f"student={student_config_name!r}, teacher={teacher_config_name!r}, expected={expected_teacher!r}"
        )


def gap_closed_fraction(*, student_mse: float, naive_mse: float) -> float:
    """Fraction of the naive one-step error removed by SnapFlow."""
    if naive_mse < 0 or student_mse < 0:
        raise ValueError("MSE values must be non-negative")
    if naive_mse == 0:
        return 1.0 if student_mse == 0 else 0.0
    return 1.0 - student_mse / naive_mse


def validate_golden_provenance(
    metadata: dict[str, Any],
    *,
    run_id: str,
    actual_hash: str,
    student_provenance: dict[str, Any],
    teacher_provenance: dict[str, Any],
    additional_source_provenances: tuple[dict[str, Any], ...] = (),
) -> None:
    """Require the golden corpus to resolve to this run and model/data contract."""
    if metadata.get("schema_version") != 2:
        raise ValueError("Golden metadata must use provenance schema_version 2")
    if metadata.get("run_id") != run_id:
        raise ValueError(f"Golden run_id mismatch: {metadata.get('run_id')!r} != {run_id!r}")
    if metadata.get("sha256") != actual_hash:
        raise ValueError(f"Golden corpus hash mismatch: expected {metadata.get('sha256')}, got {actual_hash}")
    expected_dataset = student_provenance["dataset"]
    if metadata.get("dataset") != expected_dataset:
        raise ValueError("Golden dataset provenance does not match the approved evaluation source config")
    if metadata.get("dataset_revision") != expected_dataset["revision"]:
        raise ValueError("Golden dataset_revision does not match the approved evaluation source config")
    if metadata.get("action_horizon") != student_provenance["action_horizon"]:
        raise ValueError("Golden action horizon does not match the resolved student config")
    if metadata.get("action_dim") != student_provenance["action_dim"]:
        raise ValueError("Golden action dimension does not match the resolved student config")
    source = metadata.get("resolved_config")
    accepted = {
        provenance["name"]: provenance
        for provenance in (student_provenance, teacher_provenance, *additional_source_provenances)
    }
    if not isinstance(source, dict) or source.get("name") not in accepted:
        raise ValueError("Golden source config is neither the resolved student nor teacher config")
    if metadata.get("config_name") != source["name"]:
        raise ValueError("Golden config_name does not match its resolved config provenance")
    if source != accepted[source["name"]]:
        raise ValueError("Golden source config fingerprint does not match the resolved config")


@torch.inference_mode()
def evaluate(
    *,
    run_id: str,
    student_config_name: str,
    student_checkpoint: pathlib.Path,
    teacher_config_name: str,
    teacher_checkpoint: pathlib.Path,
    corpus_path: pathlib.Path,
    device_name: str,
    batch_size: int,
    normalization_low: float,
    normalization_high: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    named_snapflow = student_config_name in SNAPFLOW_TEACHER_CONFIGS
    if named_snapflow:
        canonical_distill_name = canonical_snapflow_golden_config(student_config_name, teacher_config_name)
    else:
        canonical_distill_name = None
        validate_shallow_teacher_config(student_config_name, teacher_config_name)
    device = torch.device(device_name)
    student_train_config, student = load_model(student_config_name, student_checkpoint, device)
    teacher_train_config, teacher = load_model(teacher_config_name, teacher_checkpoint, device)
    student_config = student_train_config.model
    teacher_config = teacher_train_config.model
    student_provenance = config_provenance(student_train_config)
    teacher_provenance = config_provenance(teacher_train_config, require_dataset=False)

    is_snapflow = isinstance(student_config, _pi0_config.SnapFlowPi0Config)
    if is_snapflow != named_snapflow:
        raise ValueError("Resolved student model type differs from its reviewed evaluation-stage config")
    if is_snapflow and not isinstance(teacher_config, _pi0_config.DistilledPi0Config | _pi0_config.ShallowPi0Config):
        raise ValueError("SnapFlow evaluation requires an accepted distilled or expert-BC Shallow-pi teacher")
    if not is_snapflow and not isinstance(
        student_config, _pi0_config.DistilledPi0Config | _pi0_config.ShallowPi0Config
    ):
        raise ValueError("Shallow-pi evaluation requires a reviewed distilled or expert-BC student config")
    if not is_snapflow and isinstance(
        teacher_config,
        _pi0_config.DistilledPi0Config | _pi0_config.ShallowPi0Config | _pi0_config.SnapFlowPi0Config,
    ):
        raise ValueError("Shallow-pi evaluation requires a full-depth teacher")
    if (student_config.action_horizon, student_config.action_dim) != (
        teacher_config.action_horizon,
        teacher_config.action_dim,
    ):
        raise ValueError("Student and teacher action shapes differ")

    snapflow_teacher_training = None
    student_initialization_lineage = None
    additional_golden_sources: tuple[dict[str, Any], ...] = ()
    if is_snapflow:
        assert canonical_distill_name is not None
        validate_training_checkpoint(teacher_checkpoint)
        teacher_step = checkpoint_step(teacher_checkpoint)
        snapflow_teacher_training = checkpoint_training_provenance(
            teacher_checkpoint,
            expected_config_name=teacher_config_name,
            expected_step=teacher_step,
            expected_model=dataclasses.asdict(teacher_train_config.model),
            expected_holdout_samples=teacher_train_config.offline_holdout_samples,
        )
        student_initialization_lineage = validate_snapflow_teacher_lineage(student_checkpoint, teacher_checkpoint)
        canonical_distill = config_provenance(_config.get_config(canonical_distill_name))
        additional_golden_sources = (canonical_distill,)
    elif isinstance(student_config, _pi0_config.DistilledPi0Config):
        student_initialization_lineage = validate_shallow_teacher_lineage(student_checkpoint, teacher_checkpoint)
    else:
        student_initialization_lineage = checkpoint_initialization_lineage(student_checkpoint)
        if student_initialization_lineage["kind"] != "pytorch_source":
            raise ValueError("Expert-BC evaluation requires a checkpoint initialized from an accepted Shallow model")
        canonical_distill = config_provenance(_config.get_config("pi05_droid_l09_distill"))
        additional_golden_sources = (canonical_distill,)

    metadata_path = corpus_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    actual_hash = sha256_file(corpus_path)
    validate_golden_provenance(
        metadata,
        run_id=run_id,
        actual_hash=actual_hash,
        student_provenance=student_provenance,
        teacher_provenance=teacher_provenance,
        additional_source_provenances=additional_golden_sources,
    )
    with np.load(corpus_path) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if arrays["actions"].shape != (
        metadata["samples"],
        student_config.action_horizon,
        student_config.action_dim,
    ):
        raise ValueError("Golden action array shape does not match its metadata and resolved config")
    observation = observation_from_arrays(arrays, metadata, device)
    actions = torch.from_numpy(np.ascontiguousarray(arrays["actions"])).to(device=device, dtype=torch.float32)
    noise = torch.from_numpy(np.ascontiguousarray(arrays["noise"])).to(device=device, dtype=torch.float32)
    time = torch.from_numpy(np.ascontiguousarray(arrays["time"])).to(device=device, dtype=torch.float32)

    student_chunks: list[np.ndarray] = []
    teacher_chunks: list[np.ndarray] = []
    naive_chunks: list[np.ndarray] = []
    student_velocities: list[np.ndarray] = []
    teacher_velocities: list[np.ndarray] = []
    for start in range(0, actions.shape[0], batch_size):
        stop = min(start + batch_size, actions.shape[0])
        obs_batch = slice_observation(observation, start, stop)
        noise_batch = noise[start:stop]
        teacher_chunk = teacher.sample_actions(device, obs_batch, noise=noise_batch, num_steps=10)
        student_chunk = student.sample_actions(
            device,
            obs_batch,
            noise=noise_batch,
            num_steps=1 if is_snapflow else 10,
        )
        teacher_chunks.append(teacher_chunk.float().cpu().numpy())
        student_chunks.append(student_chunk.float().cpu().numpy())
        if is_snapflow:
            naive = teacher.sample_actions(device, obs_batch, noise=noise_batch, num_steps=1)
            naive_chunks.append(naive.float().cpu().numpy())
        else:
            preprocessed = student.preprocess_observation(obs_batch, train=False)
            student_velocity = student.predict_velocity(
                preprocessed,
                actions[start:stop],
                noise_batch,
                time[start:stop],
                observation_is_preprocessed=True,
            )
            teacher_velocity = teacher.predict_velocity(
                preprocessed,
                actions[start:stop],
                noise_batch,
                time[start:stop],
                observation_is_preprocessed=True,
            )
            student_velocities.append(student_velocity.float().cpu().numpy())
            teacher_velocities.append(teacher_velocity.float().cpu().numpy())

    student_array = np.concatenate(student_chunks)
    teacher_array = np.concatenate(teacher_chunks)
    action_low_array = np.full(student_array.shape[-1], normalization_low, dtype=np.float32)
    action_high_array = np.full(student_array.shape[-1], normalization_high, dtype=np.float32)
    action_metrics = compute_metrics(
        student_array,
        teacher_array,
        ground_truth=arrays["actions"],
        action_low=action_low_array,
        action_high=action_high_array,
    )
    student_weights = student_checkpoint / "model.safetensors"
    teacher_weights = teacher_checkpoint / "model.safetensors"
    student_training = checkpoint_training_provenance(
        student_checkpoint,
        expected_config_name=student_config_name,
        expected_step=checkpoint_step(student_checkpoint),
        expected_model=dataclasses.asdict(student_train_config.model),
        expected_holdout_samples=student_train_config.offline_holdout_samples,
    )
    evidence_student_config = {
        **student_provenance,
        "training_fingerprint_sha256": student_training["training_fingerprint_sha256"],
    }
    provenance = {
        "schema_version": 1,
        "run_id": run_id,
        "student_config": evidence_student_config,
        "student_checkpoint": {
            "path": str(student_checkpoint),
            "step": checkpoint_step(student_checkpoint),
            "model_sha256": sha256_file(student_weights),
            "metadata_sha256": student_training["metadata_sha256"],
            "initialization_lineage": student_initialization_lineage,
        },
        "teacher_config": teacher_provenance,
        "teacher_checkpoint": {
            "path": str(teacher_checkpoint),
            "model_sha256": sha256_file(teacher_weights),
            **(
                {
                    "step": checkpoint_step(teacher_checkpoint),
                    "training_fingerprint_sha256": snapflow_teacher_training["training_fingerprint_sha256"],
                    "metadata_sha256": snapflow_teacher_training["metadata_sha256"],
                }
                if snapflow_teacher_training is not None
                else {}
            ),
        },
        "dataset": student_provenance["dataset"],
        "golden": {
            "path": str(corpus_path.resolve()),
            "sha256": actual_hash,
            "metadata_path": str(metadata_path.resolve()),
            "metadata_sha256": sha256_file(metadata_path),
            "source_config": metadata["resolved_config"],
        },
        "normalization_range": {"low": normalization_low, "high": normalization_high},
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "provenance": provenance,
        "stage": "snapflow" if is_snapflow else "shallow",
        "student_config": student_config_name,
        "student_checkpoint": str(student_checkpoint),
        "student_step": checkpoint_step(student_checkpoint),
        "teacher_config": teacher_config_name,
        "teacher_checkpoint": str(teacher_checkpoint),
        "golden_corpus": str(corpus_path.resolve()),
        "golden_corpus_sha256": actual_hash,
        "action_metrics": action_metrics,
    }
    output_arrays = {"student": student_array, "teacher": teacher_array, "ground_truth": arrays["actions"]}
    if is_snapflow:
        naive_array = np.concatenate(naive_chunks)
        naive_mse = float(np.mean(np.square(naive_array.astype(np.float64) - teacher_array)))
        student_mse = action_metrics["kd_mse"]
        report["snapflow_metrics"] = {
            "one_step_mse_to_ten_step_teacher": student_mse,
            "naive_one_step_mse_to_ten_step_teacher": naive_mse,
            "offline_error_gap_closed_fraction": gap_closed_fraction(
                student_mse=student_mse,
                naive_mse=naive_mse,
            ),
            "offline_error_gap_gate_minimum": 0.70,
            "offline_error_gap_gate_pass": gap_closed_fraction(
                student_mse=student_mse,
                naive_mse=naive_mse,
            )
            >= 0.70,
        }
        output_arrays["naive_one_step"] = naive_array
    else:
        student_velocity_array = np.concatenate(student_velocities)
        teacher_velocity_array = np.concatenate(teacher_velocities)
        report["velocity_metrics"] = compute_metrics(student_velocity_array, teacher_velocity_array)
        output_arrays["student_velocity"] = student_velocity_array
        output_arrays["teacher_velocity"] = teacher_velocity_array
    return report, output_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--student-config-name", required=True)
    parser.add_argument("--student-run-root", type=pathlib.Path, required=True)
    parser.add_argument("--student-step", type=int, required=True)
    parser.add_argument("--teacher-config-name", required=True)
    parser.add_argument("--teacher-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--teacher-step", type=int)
    parser.add_argument("--corpus", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--normalization-low", type=float, default=-1.0)
    parser.add_argument("--normalization-high", type=float, default=1.0)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.run_id.strip():
        raise ValueError("run-id must be non-empty")
    if not math.isfinite(args.normalization_low) or not math.isfinite(args.normalization_high):
        raise ValueError("normalization range must be finite")
    if args.normalization_low >= args.normalization_high:
        raise ValueError("normalization-low must be less than normalization-high")
    student_checkpoint = resolve_checkpoint(args.student_run_root, step=args.student_step)
    teacher_checkpoint = resolve_weights_directory(args.teacher_checkpoint, step=args.teacher_step)
    report, arrays = evaluate(
        run_id=args.run_id,
        student_config_name=args.student_config_name,
        student_checkpoint=student_checkpoint,
        teacher_config_name=args.teacher_config_name,
        teacher_checkpoint=teacher_checkpoint,
        corpus_path=args.corpus,
        device_name=args.device,
        batch_size=args.batch_size,
        normalization_low=args.normalization_low,
        normalization_high=args.normalization_high,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
