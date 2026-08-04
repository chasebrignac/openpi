import json
import pathlib

import pytest
import safetensors.torch
import torch

from scripts import repro_checkpoint


def make_checkpoint(
    root: pathlib.Path,
    step: int,
    *,
    config_name: str = "pi05_libero_l09_distill",
    exp_name: str = "libero-shallow",
) -> pathlib.Path:
    checkpoint = root / str(step)
    checkpoint.mkdir(parents=True)
    overfit = exp_name.endswith("-overfit")
    model_config = {"layers": 9, "pi05": True}
    factory_config = {"repo_id": "physical-intelligence/libero" if "libero" in config_name else "droid"}
    norm_stats = {"actions": {"mean": [0.0], "std": [1.0]}}
    dataset = {
        "factory": "test.Data",
        "factory_config_sha256": repro_checkpoint.canonical_resume_fingerprint(factory_config),
        "repo_id": "physical-intelligence/libero" if "libero" in config_name else "allenai/MolmoAct2-DROID-Dataset",
        "revision": "a" * 40,
        "codebase_version": "v2.0" if "libero" in config_name else "v3.0",
        "episode_prompt_path": None if "libero" in config_name else "meta/tasks_annotated.parquet",
        "episode_prompt_sha256": None if "libero" in config_name else "d" * 64,
        "action_sequence_keys": ["actions"],
        "asset_id": "physical-intelligence/libero" if "libero" in config_name else "droid",
        "use_quantile_norm": True,
        "prompt_from_task": "libero" in config_name,
        "normalization_sha256": repro_checkpoint.canonical_resume_fingerprint(norm_stats),
        "recovery_provenance_sha256": None,
    }
    contract = {
        "schema_version": 1,
        "config_name": config_name,
        "exp_name": exp_name,
        "seed": 42,
        "batch_size": 8,
        "gradient_accumulation_steps": 8,
        "lr_schedule": {"warmup_steps": 1000},
        "optimizer": {"b1": 0.9},
        "wandb_enabled": False,
        "model": {"class": "test.Model", "config": model_config},
        "pytorch_training_precision": "bfloat16",
        "dataset": dataset,
        "teacher": {"model_sha256": "b" * 64} if config_name.endswith("_distill") else None,
        "data_split": {"schema_version": 1, "validation_episode_ids": [3]},
        "stochastic_schedule": "sha256-v2(model:seed,step,accumulation,rank;loader:seed,epoch)",
        "one_batch_overfit": overfit,
        "one_batch_overfit_min_relative_decline": 0.2,
    }
    safetensors.torch.save_file({"weight": torch.ones(2)}, checkpoint / "model.safetensors")
    torch.save({"state": {}, "param_groups": [{"params": []}]}, checkpoint / "optimizer.pt")
    torch.save(
        {
            "global_step": step,
            "data_split": contract["data_split"],
            "config": {
                "name": config_name,
                "exp_name": exp_name,
                "seed": contract["seed"],
                "batch_size": contract["batch_size"],
                "gradient_accumulation_steps": contract["gradient_accumulation_steps"],
                "lr_schedule": contract["lr_schedule"],
                "optimizer": contract["optimizer"],
                "wandb_enabled": contract["wandb_enabled"],
                "model": model_config,
                "data": factory_config,
                "pytorch_training_precision": contract["pytorch_training_precision"],
                "one_batch_overfit": overfit,
                "one_batch_overfit_min_relative_decline": 0.2,
            },
        },
        checkpoint / "metadata.pt",
    )
    (checkpoint / "wandb_id.txt").write_text("disabled\n")
    (checkpoint / "resume-state.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "global_step": step,
                "config_name": config_name,
                "exp_name": exp_name,
                "resume_contract": contract,
                "resume_fingerprint_sha256": repro_checkpoint.resume_identity_fingerprint(
                    contract,
                    {"kind": "shallow_teacher_transplant", "model_sha256": "b" * 64},
                ),
                "initialization_lineage": {
                    "kind": "shallow_teacher_transplant",
                    "model_sha256": "b" * 64,
                },
                "state_files": ["metadata.pt", "model.safetensors", "optimizer.pt", "wandb_id.txt"],
            }
        )
    )
    if config_name.startswith("pi05_libero_"):
        norm = checkpoint / "assets/physical-intelligence/libero/norm_stats.json"
    else:
        norm = checkpoint / "assets/droid/norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text(json.dumps({"norm_stats": norm_stats}))
    return checkpoint


def test_resolves_explicit_step_from_run_root(tmp_path: pathlib.Path):
    expected = make_checkpoint(tmp_path, 5_000)
    make_checkpoint(tmp_path, 10_000)
    assert repro_checkpoint.resolve_checkpoint(tmp_path, step=5_000) == expected


def test_accepts_direct_numeric_checkpoint(tmp_path: pathlib.Path):
    checkpoint = make_checkpoint(tmp_path, 2_000)
    assert repro_checkpoint.resolve_checkpoint(checkpoint) == checkpoint
    assert repro_checkpoint.resolve_checkpoint(checkpoint / "model.safetensors") == checkpoint


def test_rejects_ambiguous_run_root_and_lists_available_steps(tmp_path: pathlib.Path):
    make_checkpoint(tmp_path, 2_000)
    with pytest.raises(ValueError, match="numeric step"):
        repro_checkpoint.resolve_checkpoint(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"available steps: \[2000\]"):
        repro_checkpoint.resolve_checkpoint(tmp_path, step=5_000)


def test_released_weight_directory_need_not_have_numeric_name(tmp_path: pathlib.Path):
    released = tmp_path / "released_teacher"
    released.mkdir()
    (released / "model.safetensors").touch()
    assert repro_checkpoint.resolve_weights_directory(released) == released


def test_rejects_incomplete_training_state_and_missing_track_assets(tmp_path: pathlib.Path):
    checkpoint = make_checkpoint(tmp_path, 2_000)
    (checkpoint / "optimizer.pt").unlink()
    with pytest.raises(FileNotFoundError, match=r"optimizer\.pt"):
        repro_checkpoint.resolve_checkpoint(checkpoint)

    torch.save({"state": {}, "param_groups": [{"params": []}]}, checkpoint / "optimizer.pt")
    (checkpoint / "assets/physical-intelligence/libero/norm_stats.json").unlink()
    with pytest.raises(FileNotFoundError, match="normalization"):
        repro_checkpoint.resolve_checkpoint(checkpoint)


def test_overfit_checkpoint_requires_passing_diagnostic(tmp_path: pathlib.Path):
    checkpoint = make_checkpoint(tmp_path, 300, exp_name="libero-shallow-overfit")
    with pytest.raises(FileNotFoundError, match="overfit-diagnostic"):
        repro_checkpoint.resolve_checkpoint(checkpoint)

    (checkpoint / "overfit-diagnostic.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "gate_pass": True,
                "optimizer_steps": 300,
                "window_optimizer_steps": 20,
                "optimizer_step_losses": [1.0] * 20 + [0.8] * 260 + [0.7] * 20,
                "initial_loss_mean": 1.0,
                "final_loss_mean": 0.7,
                "relative_loss_decline": 0.3,
                "minimum_relative_loss_decline": 0.2,
                "all_losses_finite": True,
            }
        )
    )
    assert repro_checkpoint.resolve_checkpoint(checkpoint) == checkpoint


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("window_optimizer_steps", 1),
        ("minimum_relative_loss_decline", 0.01),
        ("minimum_relative_loss_decline", 0.25),
        ("relative_loss_decline", 0.9),
    ],
)
def test_overfit_verifier_enforces_exact_window_threshold_and_arithmetic(tmp_path, field, value):
    checkpoint = make_checkpoint(tmp_path, 300, exp_name="libero-shallow-overfit")
    report = {
        "schema_version": 1,
        "gate_pass": True,
        "optimizer_steps": 300,
        "window_optimizer_steps": 20,
        "optimizer_step_losses": [1.0] * 20 + [0.8] * 260 + [0.7] * 20,
        "initial_loss_mean": 1.0,
        "final_loss_mean": 0.7,
        "relative_loss_decline": 0.3,
        "minimum_relative_loss_decline": 0.2,
        "all_losses_finite": True,
    }
    report[field] = value
    (checkpoint / "overfit-diagnostic.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="optimizer-smoke"):
        repro_checkpoint.resolve_checkpoint(checkpoint)


def test_rejects_corrupt_or_nonfinite_tensor_state(tmp_path):
    corrupt = make_checkpoint(tmp_path / "corrupt", 2000)
    (corrupt / "optimizer.pt").write_bytes(b"not a torch checkpoint")
    with pytest.raises(ValueError, match="readable PyTorch"):
        repro_checkpoint.resolve_checkpoint(corrupt)

    nonfinite = make_checkpoint(tmp_path / "nonfinite", 2000)
    safetensors.torch.save_file({"weight": torch.tensor([float("nan")])}, nonfinite / "model.safetensors")
    with pytest.raises(FloatingPointError, match="non-finite"):
        repro_checkpoint.resolve_checkpoint(nonfinite)


def test_rejects_metadata_or_resume_fingerprint_tampering(tmp_path):
    checkpoint = make_checkpoint(tmp_path, 2000)
    metadata = torch.load(checkpoint / "metadata.pt", weights_only=False)
    metadata["global_step"] = 1999
    torch.save(metadata, checkpoint / "metadata.pt")
    with pytest.raises(ValueError, match="step or data split"):
        repro_checkpoint.resolve_checkpoint(checkpoint)

    checkpoint = make_checkpoint(tmp_path / "fingerprint", 2000)
    state = json.loads((checkpoint / "resume-state.json").read_text())
    state["resume_contract"]["seed"] = 99
    (checkpoint / "resume-state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="fingerprint"):
        repro_checkpoint.resolve_checkpoint(checkpoint)
