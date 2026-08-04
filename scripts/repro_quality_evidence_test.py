import copy
import json
import pathlib
import sys

import pytest

from scripts import repro_promotion_report
from scripts import repro_quality_evidence


def offline_report(step: int = 5_000) -> dict:
    provenance = {
        "schema_version": 1,
        "run_id": "run-001",
        "student_config": {
            "name": "pi05_libero_l09_distill",
            "fingerprint_sha256": "a" * 64,
            "training_fingerprint_sha256": "b" * 64,
        },
        "student_checkpoint": {
            "path": f"/runs/student/{step}",
            "step": step,
            "model_sha256": "c" * 64,
            "metadata_sha256": "d" * 64,
        },
        "teacher_config": {"name": "pi05_libero", "fingerprint_sha256": "e" * 64},
        "teacher_checkpoint": {"path": "/teacher", "model_sha256": "f" * 64},
        "dataset": {"repo_id": "physical-intelligence/libero", "revision": "1" * 40},
        "golden": {"sha256": "2" * 64, "metadata_sha256": "3" * 64},
        "normalization_range": {"low": -1.0, "high": 1.0},
    }
    return {
        "stage": "shallow",
        "student_config": "pi05_libero_l09_distill",
        "student_step": step,
        "teacher_config": "pi05_libero",
        "provenance": provenance,
        "action_metrics": {
            "kd_mse": 0.01,
            "kd_cosine_mean": 0.99,
            "per_joint_normalized_rmse": [0.1],
            "action_chunk_rmse": 0.1,
            "normalization_range_excursions": 0,
        },
        "velocity_metrics": {
            "kd_mse": 0.01,
            "kd_cosine_mean": 0.99,
            "per_joint_normalized_rmse": [0.1],
        },
    }


def quality_report(offline: dict, *, counts: tuple[int, ...] = (200, 200)) -> dict:
    provenance = offline["provenance"]
    return {
        "base_stage": repro_quality_evidence.model_stage_identity(
            "teacher", provenance["teacher_checkpoint"]["model_sha256"]
        ),
        "candidate_stage": repro_quality_evidence.model_stage_identity(
            "student", provenance["student_checkpoint"]["model_sha256"]
        ),
        "groups": {
            f"libero/suite-{index}": {
                "episodes": count,
                "baseline_success": 0.8,
                "candidate_success": 0.75 + index * 0.05,
            }
            for index, count in enumerate(counts)
        },
        "evaluation_gate": {"mode": "intermediate", "passed": True},
    }


def build(quality: dict, offline: dict, *, required_pairs: int = 400, denoise_speedup: float | None = None):
    return repro_quality_evidence.build_quality_evidence(
        quality,
        offline,
        required_pairs=required_pairs,
        denoise_speedup=denoise_speedup,
        quality_report_sha256="4" * 64,
        offline_report_sha256="5" * 64,
        quality_report_path="/evidence/quality.json",
        offline_report_path="/evidence/offline.json",
    )


def test_emits_provenance_complete_weighted_quality_evidence():
    offline = offline_report()
    quality = quality_report(offline)
    assert quality["base_stage"] == "f" * 64
    assert quality["candidate_stage"] == "c" * 64
    result = build(quality, offline, denoise_speedup=8.5)
    assert result["checkpoint_step"] == 5_000
    assert result["provenance"] == offline["provenance"]
    assert result["paired_rollout"] == {
        "student_success": pytest.approx(0.775),
        "reference_success": pytest.approx(0.8),
        "complete_pairs": 400,
    }
    assert result["denoise_speedup"] == 8.5
    assert result["source_quality_report"]["sha256"] == "4" * 64
    promotion = repro_promotion_report.build_promotion_report(
        stage="shallow",
        offline_reports=[offline],
        quality_reports=[result],
        max_rollout_gap=0.03,
    )
    assert promotion["promotion_ready"] is True


def test_rejects_under_counted_or_failed_episode_report():
    offline = offline_report()
    with pytest.raises(ValueError, match="exactly 400"):
        build(quality_report(offline, counts=(100, 100)), offline)

    failed = quality_report(offline)
    failed["evaluation_gate"]["passed"] = False
    with pytest.raises(ValueError, match="did not pass"):
        build(failed, offline)


def test_rejects_unpaired_model_identity():
    offline = offline_report()
    quality = quality_report(offline)
    quality["candidate_stage"] = repro_quality_evidence.model_stage_identity("student", "9" * 64)
    with pytest.raises(ValueError, match="model identity mismatch"):
        build(quality, offline)

    prefixed = quality_report(offline)
    prefixed["candidate_stage"] = "student-sha256:" + "c" * 64
    with pytest.raises(ValueError, match="model identity mismatch"):
        build(prefixed, offline)


def test_model_stage_identity_is_exact_worker_safe_sha256():
    digest = "a" * 64
    assert repro_quality_evidence.model_stage_identity("teacher", digest) == digest
    assert repro_quality_evidence.model_stage_identity("student", digest) == digest
    with pytest.raises(ValueError, match="Unknown model identity role"):
        repro_quality_evidence.model_stage_identity("candidate", digest)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        repro_quality_evidence.model_stage_identity("teacher", "A" * 64)


def test_rejects_offline_provenance_that_does_not_match_its_step():
    offline = offline_report()
    offline["provenance"] = copy.deepcopy(offline["provenance"])
    offline["provenance"]["student_checkpoint"]["step"] = 10_000
    with pytest.raises(ValueError, match="step does not match"):
        build(quality_report(offline), offline)


def test_cli_hashes_both_input_files(tmp_path: pathlib.Path, monkeypatch):
    offline = offline_report()
    quality = quality_report(offline)
    quality_path = tmp_path / "quality.json"
    offline_path = tmp_path / "offline.json"
    output_path = tmp_path / "evidence.json"
    quality_path.write_text(json.dumps(quality))
    offline_path.write_text(json.dumps(offline))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "repro_quality_evidence.py",
            "--quality-report",
            str(quality_path),
            "--offline-report",
            str(offline_path),
            "--required-pairs",
            "400",
            "--output",
            str(output_path),
        ],
    )

    assert repro_quality_evidence.main() == 0
    result = json.loads(output_path.read_text())
    assert result["source_quality_report"]["sha256"] == repro_quality_evidence.sha256_file(quality_path)
    assert result["source_offline_report"]["sha256"] == repro_quality_evidence.sha256_file(offline_path)
