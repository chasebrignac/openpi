import copy

import pytest

from scripts import repro_promotion_report


def provenance(step: int, stage: str) -> dict:
    return {
        "schema_version": 1,
        "run_id": "repro-run-001",
        "student_config": {
            "name": f"test_{stage}",
            "fingerprint_sha256": "a" * 64,
            "training_fingerprint_sha256": "1" * 64,
        },
        "student_checkpoint": {
            "path": f"/runs/{stage}/{step}",
            "step": step,
            "model_sha256": f"{step:064x}",
            "metadata_sha256": f"{step + 1:064x}",
        },
        "teacher_config": {"name": "teacher", "fingerprint_sha256": "b" * 64},
        "teacher_checkpoint": {"path": "/teacher", "model_sha256": "c" * 64},
        "dataset": {"repo_id": "example/dataset", "revision": "f" * 40},
        "golden": {"sha256": "d" * 64, "metadata_sha256": "e" * 64},
        "normalization_range": {"low": -1.0, "high": 1.0},
    }


def offline_report(step: int, *, kd_mse: float, stage: str = "shallow", gap_closed: float = 0.8):
    report = {
        "stage": stage,
        "student_config": f"test_{stage}",
        "student_step": step,
        "teacher_config": "teacher",
        "provenance": provenance(step, stage),
        "action_metrics": {
            "kd_mse": kd_mse,
            "kd_cosine_mean": 0.99,
            "per_joint_normalized_rmse": [0.1, 0.2],
            "action_chunk_rmse": kd_mse**0.5,
            "normalization_range_excursions": 0,
        },
        "velocity_metrics": {
            "kd_mse": kd_mse,
            "kd_cosine_mean": 0.99,
            "per_joint_normalized_rmse": [0.1, 0.2],
        },
    }
    if stage == "snapflow":
        report.pop("velocity_metrics")
        report["snapflow_metrics"] = {"offline_error_gap_closed_fraction": gap_closed}
    return report


def quality_report(offline: dict, student_success: float, *, speedup: float | None = None):
    report = {
        "checkpoint_step": offline["student_step"],
        "provenance": offline["provenance"],
        "paired_rollout": {"student_success": student_success, "reference_success": 0.8},
    }
    if speedup is not None:
        report["denoise_speedup"] = speedup
    return report


def test_promotes_shallow_when_all_required_gates_pass():
    offline = offline_report(5_000, kd_mse=0.01)
    report = repro_promotion_report.build_promotion_report(
        stage="shallow",
        offline_reports=[offline],
        quality_reports=[quality_report(offline, 0.78)],
        max_rollout_gap=0.05,
        min_kd_cosine=0.98,
    )
    assert report["promotion_ready"] is True
    assert report["recommendation"] == "promote"


def test_snapflow_missing_latency_is_not_silently_promoted():
    offline = offline_report(5_000, kd_mse=0.01, stage="snapflow")
    report = repro_promotion_report.build_promotion_report(
        stage="snapflow",
        offline_reports=[offline],
        quality_reports=[quality_report(offline, 0.78)],
        max_rollout_gap=0.03,
    )
    assert report["promotion_ready"] is False
    assert report["missing_required_gates"] == ["denoise_speedup"]


def test_30k_extension_requires_both_kd_and_rollout_improvement():
    previous = offline_report(20_000, kd_mse=1.0)
    current = offline_report(30_000, kd_mse=0.81)
    report = repro_promotion_report.build_promotion_report(
        stage="shallow",
        offline_reports=[previous, current],
        quality_reports=[quality_report(previous, 0.50), quality_report(current, 0.55)],
        max_rollout_gap=0.01,
    )
    assert report["trend"]["kd_improvement_per_5k"] == pytest.approx(0.1)
    assert report["recommendation"] == "extension_requires_separate_approval"


def test_30k_plateau_stops_without_extension_signal():
    previous = offline_report(20_000, kd_mse=1.0)
    current = offline_report(30_000, kd_mse=0.95)
    report = repro_promotion_report.build_promotion_report(
        stage="shallow",
        offline_reports=[previous, current],
        quality_reports=[quality_report(previous, 0.50), quality_report(current, 0.51)],
        max_rollout_gap=0.01,
    )
    assert report["trend"]["kd_improving_at_least_five_percent"] is False
    assert report["recommendation"] == "stop_at_cap_no_extension_signal"


def test_rejects_mixed_run_or_checkpoint_provenance():
    previous = offline_report(5_000, kd_mse=1.0)
    current = offline_report(10_000, kd_mse=0.9)
    current["provenance"]["run_id"] = "different-run"
    with pytest.raises(ValueError, match="diverges"):
        repro_promotion_report.build_promotion_report(
            stage="shallow",
            offline_reports=[previous, current],
            quality_reports=[],
            max_rollout_gap=0.03,
        )


@pytest.mark.parametrize("field", ["student_config", "teacher_checkpoint", "dataset", "golden"])
def test_rejects_mixed_config_teacher_dataset_or_golden_hash(field):
    previous = offline_report(5_000, kd_mse=1.0)
    current = offline_report(10_000, kd_mse=0.9)
    current["provenance"] = copy.deepcopy(current["provenance"])
    if field == "student_config":
        current["provenance"][field]["training_fingerprint_sha256"] = "9" * 64
    elif field == "teacher_checkpoint":
        current["provenance"][field]["model_sha256"] = "9" * 64
    elif field == "dataset":
        current["provenance"][field]["revision"] = "9" * 40
    else:
        current["provenance"][field]["sha256"] = "9" * 64
    with pytest.raises(ValueError, match="diverges"):
        repro_promotion_report.build_promotion_report(
            stage="shallow",
            offline_reports=[previous, current],
            quality_reports=[],
            max_rollout_gap=0.03,
        )


def test_rejects_quality_evidence_for_different_student_hash():
    offline = offline_report(5_000, kd_mse=0.01)
    quality = quality_report(offline, 0.78)
    quality["provenance"] = {
        **quality["provenance"],
        "student_checkpoint": {**quality["provenance"]["student_checkpoint"], "model_sha256": "f" * 64},
    }
    with pytest.raises(ValueError, match="does not match offline"):
        repro_promotion_report.build_promotion_report(
            stage="shallow",
            offline_reports=[offline],
            quality_reports=[quality],
            max_rollout_gap=0.03,
        )


def test_failed_or_incomplete_report_exits_nonzero_unless_report_only():
    failed = {"promotion_ready": False}
    assert repro_promotion_report.promotion_exit_code(failed, report_only=False) == 2
    assert repro_promotion_report.promotion_exit_code(failed, report_only=True) == 0
    assert repro_promotion_report.promotion_exit_code({"promotion_ready": True}, report_only=False) == 0
