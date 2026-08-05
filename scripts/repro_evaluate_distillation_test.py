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


def test_droid_action_dimension_contract_evaluates_first_eight_of_32():
    contract = repro_evaluate_distillation.action_dimension_contract("pi05_droid_l09_snapflow", 32)

    assert contract["track"] == "droid"
    assert contract["model_action_dimensions"] == 32
    assert contract["evaluated_action_dimensions"] == 8
    assert contract["evaluated_dimension_indices"] == list(range(8))
    assert contract["evidence_array_action_dimensions"] == 32


def test_libero_action_dimension_contract_evaluates_first_seven_of_32():
    contract = repro_evaluate_distillation.action_dimension_contract("pi05_libero_l09_distill", 32)

    assert contract["track"] == "libero"
    assert contract["evaluated_action_dimensions"] == 7
    assert contract["evaluated_dimension_indices"] == list(range(7))


def test_action_dimension_contract_fails_closed_for_unreviewed_mapping():
    with pytest.raises(ValueError, match="No reviewed active-action mapping"):
        repro_evaluate_distillation.action_dimension_contract("pi05_unreviewed_l09_snapflow", 32)


def test_action_and_snapflow_metrics_ignore_padded_dimensions():
    contract = repro_evaluate_distillation.action_dimension_contract("pi05_droid_l09_snapflow", 32)
    teacher = np.zeros((2, 3, 32), dtype=np.float32)
    student = np.ones_like(teacher)
    naive = np.full_like(teacher, 2.0)
    ground_truth = np.zeros_like(teacher)
    student[..., 8:] = 1_000.0
    naive[..., 8:] = -2_000.0
    ground_truth[..., 8:] = 3_000.0

    action_metrics = repro_evaluate_distillation.compute_active_action_metrics(
        student,
        teacher,
        ground_truth=ground_truth,
        contract=contract,
        normalization_low=-1.0,
        normalization_high=1.0,
    )
    snapflow_metrics = repro_evaluate_distillation.compute_snapflow_metrics(
        student,
        teacher,
        naive,
        contract=contract,
    )

    assert action_metrics["joints"] == 8
    assert action_metrics["kd_mse"] == pytest.approx(1.0)
    assert action_metrics["ground_truth_mse"] == pytest.approx(1.0)
    assert snapflow_metrics["one_step_mse_to_ten_step_teacher"] == pytest.approx(1.0)
    assert snapflow_metrics["naive_one_step_mse_to_ten_step_teacher"] == pytest.approx(4.0)
    assert snapflow_metrics["offline_error_gap_closed_fraction"] == pytest.approx(0.75)
    assert snapflow_metrics["offline_error_gap_gate_pass"] is True
    assert student.shape == teacher.shape == naive.shape == ground_truth.shape == (2, 3, 32)


def test_velocity_metrics_ignore_droid_padding_and_require_full_model_shape():
    contract = repro_evaluate_distillation.action_dimension_contract("pi05_droid_l09_distill", 32)
    teacher_velocity = np.zeros((2, 3, 32), dtype=np.float32)
    student_velocity = np.ones_like(teacher_velocity)
    student_velocity[..., 8:] = 1_000.0

    metrics = repro_evaluate_distillation.compute_active_velocity_metrics(
        student_velocity,
        teacher_velocity,
        contract=contract,
    )

    assert contract["evaluated_action_dimensions"] == 8
    assert metrics["joints"] == 8
    assert metrics["kd_mse"] == pytest.approx(1.0)
    assert metrics["action_chunk_rmse"] == pytest.approx(1.0)
    assert student_velocity.shape == teacher_velocity.shape == (2, 3, 32)

    with pytest.raises(ValueError, match="full-model shape"):
        repro_evaluate_distillation.compute_active_velocity_metrics(
            student_velocity[..., :8],
            teacher_velocity[..., :8],
            contract=contract,
        )


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


def test_golden_provenance_accepts_canonical_distill_source_for_bc_teacher():
    student = repro_make_golden.config_provenance(training_config.get_config("pi05_droid_l09_snapflow"))
    teacher = repro_make_golden.config_provenance(training_config.get_config("pi05_droid_l09_expert_bc_25"))
    canonical = repro_make_golden.config_provenance(training_config.get_config("pi05_droid_l09_distill"))
    metadata = {
        "schema_version": 2,
        "run_id": "run-001",
        "sha256": "a" * 64,
        "action_horizon": student["action_horizon"],
        "action_dim": student["action_dim"],
        "dataset": student["dataset"],
        "dataset_revision": student["dataset"]["revision"],
        "config_name": canonical["name"],
        "resolved_config": canonical,
    }

    repro_evaluate_distillation.validate_golden_provenance(
        metadata,
        run_id="run-001",
        actual_hash="a" * 64,
        student_provenance=student,
        teacher_provenance=teacher,
        additional_source_provenances=(canonical,),
    )


def test_golden_provenance_accepts_canonical_droid_corpus_for_bc_student():
    student = repro_make_golden.config_provenance(training_config.get_config("pi05_droid_l09_expert_bc_25"))
    teacher = repro_make_golden.config_provenance(
        training_config.get_config("pi05_droid_jointpos"), require_dataset=False
    )
    canonical = repro_make_golden.config_provenance(training_config.get_config("pi05_droid_l09_distill"))
    metadata = {
        "schema_version": 2,
        "run_id": "run-001",
        "sha256": "a" * 64,
        "action_horizon": student["action_horizon"],
        "action_dim": student["action_dim"],
        "dataset": canonical["dataset"],
        "dataset_revision": canonical["dataset"]["revision"],
        "config_name": canonical["name"],
        "resolved_config": canonical,
    }

    repro_evaluate_distillation.validate_golden_provenance(
        metadata,
        run_id="run-001",
        actual_hash="a" * 64,
        student_provenance=student,
        teacher_provenance=teacher,
        additional_source_provenances=(canonical,),
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


def test_snapflow_teacher_must_equal_recorded_initialization_checkpoint(tmp_path):
    student = tmp_path / "student" / "5000"
    teacher = tmp_path / "teacher" / "10000"
    student.mkdir(parents=True)
    teacher.mkdir(parents=True)
    teacher_weights = b"accepted shallow weights"
    (teacher / "model.safetensors").write_bytes(teacher_weights)
    teacher_sha256 = repro_make_golden.sha256_file(teacher / "model.safetensors")
    (student / "resume-state.json").write_text(
        '{"initialization_lineage":{"kind":"pytorch_source","model_sha256":"' + teacher_sha256 + '"}}\n'
    )

    lineage = repro_evaluate_distillation.validate_snapflow_teacher_lineage(student, teacher)
    assert lineage == {"kind": "pytorch_source", "model_sha256": teacher_sha256}

    (teacher / "model.safetensors").write_bytes(b"different shallow weights")
    with pytest.raises(ValueError, match="differs from the checkpoint that initialized"):
        repro_evaluate_distillation.validate_snapflow_teacher_lineage(student, teacher)


def test_shallow_teacher_must_equal_recorded_transplant_checkpoint(tmp_path):
    student = tmp_path / "student" / "5000"
    teacher = tmp_path / "teacher"
    student.mkdir(parents=True)
    teacher.mkdir(parents=True)
    (teacher / "model.safetensors").write_bytes(b"released full-depth teacher")
    teacher_sha256 = repro_make_golden.sha256_file(teacher / "model.safetensors")
    (student / "resume-state.json").write_text(
        '{"initialization_lineage":{"kind":"shallow_teacher_transplant","model_sha256":"' + teacher_sha256 + '"}}\n'
    )

    lineage = repro_evaluate_distillation.validate_shallow_teacher_lineage(student, teacher)
    assert lineage == {"kind": "shallow_teacher_transplant", "model_sha256": teacher_sha256}

    (teacher / "model.safetensors").write_bytes(b"different full-depth teacher")
    with pytest.raises(ValueError, match="differs from the checkpoint that initialized"):
        repro_evaluate_distillation.validate_shallow_teacher_lineage(student, teacher)


def test_snapflow_teacher_config_is_exactly_allowlisted_per_track():
    assert (
        repro_evaluate_distillation.canonical_snapflow_golden_config(
            "pi05_libero_l09_snapflow", "pi05_libero_l09_distill"
        )
        == "pi05_libero_l09_distill"
    )
    for teacher in (
        "pi05_droid_l09_distill",
        "pi05_droid_l09_expert_bc_25",
        "pi05_droid_l09_expert_bc_50",
    ):
        assert (
            repro_evaluate_distillation.canonical_snapflow_golden_config("pi05_droid_l09_snapflow", teacher)
            == "pi05_droid_l09_distill"
        )

    with pytest.raises(ValueError, match="exact accepted per-track"):
        repro_evaluate_distillation.canonical_snapflow_golden_config(
            "pi05_libero_l09_snapflow", "pi05_droid_l09_distill"
        )
    with pytest.raises(ValueError, match="exact accepted per-track"):
        repro_evaluate_distillation.canonical_snapflow_golden_config("unreviewed_snapflow", "pi05_droid_l09_distill")


def test_shallow_teacher_config_is_exactly_allowlisted_per_track():
    accepted = {
        "pi05_libero_l09_distill": "pi05_libero",
        "pi05_droid_l09_distill": "pi05_droid_jointpos",
        "pi05_droid_l09_expert_bc_25": "pi05_droid_jointpos",
        "pi05_droid_l09_expert_bc_50": "pi05_droid_jointpos",
    }
    for student, teacher in accepted.items():
        repro_evaluate_distillation.validate_shallow_teacher_config(student, teacher)

    with pytest.raises(ValueError, match="exact released per-track"):
        repro_evaluate_distillation.validate_shallow_teacher_config(
            "pi05_droid_l09_distill", "pi05_droid_l09_expert_bc_25"
        )
    with pytest.raises(ValueError, match="exact released per-track"):
        repro_evaluate_distillation.validate_shallow_teacher_config("unreviewed_student", "pi05_libero")
