import numpy as np
import pytest
import torch

from openpi.training import config as training_config
from scripts import repro_evaluate_distillation
from scripts import repro_make_golden


def test_gap_closed_fraction():
    assert repro_evaluate_distillation.gap_closed_fraction(student_mse=2.5, naive_mse=10.0) == pytest.approx(0.75)
    assert repro_evaluate_distillation.gap_closed_fraction(student_mse=0.0, naive_mse=0.0) == 1.0


def test_observation_conversion_transposes_bhwc_to_bchw():
    arrays = {
        "image__base": np.zeros((2, 8, 9, 3), dtype=np.float32),
        "image_mask__base": np.ones(2, dtype=bool),
        "state": np.zeros((2, 4), dtype=np.float32),
        "tokenized_prompt": np.zeros((2, 5), dtype=np.int32),
        "tokenized_prompt_mask": np.ones((2, 5), dtype=bool),
    }
    observation = repro_evaluate_distillation.observation_from_arrays(
        arrays,
        {"image_names": ["base"], "image_layout": "BHWC"},
        repro_evaluate_distillation.torch.device("cpu"),
    )
    assert observation.images["base"].shape == (2, 3, 8, 9)


def test_gap_closed_fraction_rejects_negative_errors():
    with pytest.raises(ValueError, match="non-negative"):
        repro_evaluate_distillation.gap_closed_fraction(student_mse=-1.0, naive_mse=1.0)


def golden_metadata(run_id: str = "run-001"):
    student = repro_make_golden.config_provenance(training_config.get_config("pi05_libero_l09_distill"))
    return {
        "schema_version": 2,
        "run_id": run_id,
        "sha256": "a" * 64,
        "action_horizon": 10,
        "action_dim": 32,
        "dataset": student["dataset"],
        "dataset_revision": student["dataset"]["revision"],
        "config_name": student["name"],
        "resolved_config": student,
    }


def test_golden_provenance_matches_resolved_config():
    student = repro_make_golden.config_provenance(training_config.get_config("pi05_libero_l09_distill"))
    teacher = repro_make_golden.config_provenance(training_config.get_config("pi05_libero"), require_dataset=False)
    repro_evaluate_distillation.validate_golden_provenance(
        golden_metadata(),
        run_id="run-001",
        actual_hash="a" * 64,
        student_provenance=student,
        teacher_provenance=teacher,
    )


def test_golden_provenance_rejects_wrong_run_dataset_and_config_fingerprint():
    student = repro_make_golden.config_provenance(training_config.get_config("pi05_libero_l09_distill"))
    teacher = repro_make_golden.config_provenance(training_config.get_config("pi05_libero"), require_dataset=False)
    with pytest.raises(ValueError, match="run_id mismatch"):
        repro_evaluate_distillation.validate_golden_provenance(
            golden_metadata("other-run"),
            run_id="run-001",
            actual_hash="a" * 64,
            student_provenance=student,
            teacher_provenance=teacher,
        )

    wrong_dataset = golden_metadata()
    wrong_dataset["dataset"] = {**wrong_dataset["dataset"], "revision": "moving-tag"}
    with pytest.raises(ValueError, match="dataset provenance"):
        repro_evaluate_distillation.validate_golden_provenance(
            wrong_dataset,
            run_id="run-001",
            actual_hash="a" * 64,
            student_provenance=student,
            teacher_provenance=teacher,
        )

    wrong_config = golden_metadata()
    wrong_config["resolved_config"] = {**student, "fingerprint_sha256": "f" * 64}
    with pytest.raises(ValueError, match="fingerprint"):
        repro_evaluate_distillation.validate_golden_provenance(
            wrong_config,
            run_id="run-001",
            actual_hash="a" * 64,
            student_provenance=student,
            teacher_provenance=teacher,
        )


def test_checkpoint_training_provenance_hashes_resolved_saved_config(tmp_path):
    checkpoint = tmp_path / "5000"
    checkpoint.mkdir()
    saved_config = dict.fromkeys(repro_evaluate_distillation.TRAINING_IDENTITY_FIELDS)
    saved_config |= {"name": "student", "batch_size": 8, "gradient_accumulation_steps": 8}
    torch.save({"global_step": 5_000, "config": saved_config}, checkpoint / "metadata.pt")

    provenance = repro_evaluate_distillation.checkpoint_training_provenance(
        checkpoint,
        expected_config_name="student",
        expected_step=5_000,
    )
    assert len(provenance["training_fingerprint_sha256"]) == 64
    assert len(provenance["metadata_sha256"]) == 64

    with pytest.raises(ValueError, match="global_step"):
        repro_evaluate_distillation.checkpoint_training_provenance(
            checkpoint,
            expected_config_name="student",
            expected_step=10_000,
        )
