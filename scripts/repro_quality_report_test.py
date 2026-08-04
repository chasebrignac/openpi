import json
import sys

import pytest

from scripts import repro_quality_report


def test_summarizes_strict_pairs():
    records = []
    for index, (base, final) in enumerate([(1, 1), (1, 0), (0, 1), (1, 1)]):
        common = {"pair_id": index, "benchmark": "libero", "suite": "goal", "task": "t"}
        records.append(common | {"stage": "base", "success": base})
        records.append(common | {"stage": "final", "success": final})
    report = repro_quality_report.summarize(records, base_stage="base", candidate_stage="final")
    group = report["groups"]["libero/goal"]
    assert group["episodes"] == 4
    assert group["difference_points"] == pytest.approx(0)


def test_rejects_duplicate_pair_stage():
    record = {
        "pair_id": 0,
        "benchmark": "libero",
        "suite": "goal",
        "task": "t",
        "stage": "base",
        "success": 1,
    }
    with pytest.raises(ValueError, match="duplicate"):
        repro_quality_report.summarize([record, record], base_stage="base", candidate_stage="final")


def test_rejects_incomplete_pair_instead_of_silently_dropping_it():
    common = {"benchmark": "libero", "suite": "libero_goal", "task": "t", "success": True}
    records = [
        common | {"pair_id": "complete", "stage": "base"},
        common | {"pair_id": "complete", "stage": "final"},
        common | {"pair_id": "missing-final", "stage": "base"},
    ]

    with pytest.raises(ValueError, match="incomplete baseline/candidate pairs"):
        repro_quality_report.summarize(records, base_stage="base", candidate_stage="final")


def test_rejects_identical_stage_names():
    with pytest.raises(ValueError, match="must be different"):
        repro_quality_report.summarize([], base_stage="base", candidate_stage="base")


def _official_report(*, aggregate_drop: float = 0.0, suite_drop: float | None = None):
    groups = {}
    for index, suite in enumerate(repro_quality_report.OFFICIAL_LIBERO_SUITES):
        difference = suite_drop if index == 0 and suite_drop is not None else aggregate_drop
        groups[f"libero/{suite}"] = {
            "episodes": repro_quality_report.OFFICIAL_LIBERO_EPISODES_PER_SUITE,
            "baseline_success": 0.9,
            "candidate_success": 0.9 + difference / 100,
            "difference_points": difference,
            "difference_95ci_points": [difference, difference],
        }
    return {"base_stage": "base", "candidate_stage": "final", "groups": groups}


def test_official_final_gate_enforces_counts_and_quality():
    result = repro_quality_report.apply_evaluation_gate(
        _official_report(), mode="official-final", minimum_baseline_success=0.8
    )
    assert result["evaluation_gate"]["passed"] is True

    suite_failure = repro_quality_report.apply_evaluation_gate(
        _official_report(suite_drop=-3.1), mode="official-final", minimum_baseline_success=0.8
    )
    failed_names = {check["name"] for check in suite_failure["evaluation_gate"]["checks"] if not check["passed"]}
    assert "libero_spatial_noninferiority" in failed_names

    aggregate_failure = repro_quality_report.apply_evaluation_gate(
        _official_report(aggregate_drop=-2.1), mode="official-final", minimum_baseline_success=0.8
    )
    failed_names = {check["name"] for check in aggregate_failure["evaluation_gate"]["checks"] if not check["passed"]}
    assert "aggregate_noninferiority" in failed_names


def test_intermediate_mode_requires_explicit_count():
    report = _official_report()
    with pytest.raises(ValueError, match="requires a positive expected_pairs"):
        repro_quality_report.apply_evaluation_gate(report, mode="intermediate", minimum_baseline_success=0.8)

    result = repro_quality_report.apply_evaluation_gate(
        report, mode="intermediate", expected_pairs=400, minimum_baseline_success=0.8
    )
    assert result["evaluation_gate"]["passed"] is False


def test_relative_gate_rejects_equally_broken_zero_success_models():
    report = _official_report()
    for group in report["groups"].values():
        group["baseline_success"] = 0.0
        group["candidate_success"] = 0.0
    result = repro_quality_report.apply_evaluation_gate(report, mode="official-final", minimum_baseline_success=0.8)
    failed = {check["name"] for check in result["evaluation_gate"]["checks"] if not check["passed"]}
    assert "absolute_baseline_health" in failed


def test_cli_returns_nonzero_when_count_gate_fails(tmp_path, monkeypatch):
    records = tmp_path / "records.jsonl"
    output = tmp_path / "report.json"
    common = {"pair_id": "pair", "benchmark": "libero", "suite": "libero_goal", "task": "t"}
    records.write_text(
        "\n".join(
            [
                json.dumps(common | {"stage": "base", "success": True}),
                json.dumps(common | {"stage": "final", "success": True}),
            ]
        )
        + "\n"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "repro_quality_report.py",
            str(records),
            "--mode",
            "intermediate",
            "--expected-pairs",
            "2",
            "--minimum-baseline-success",
            "0.8",
            "--output",
            str(output),
        ],
    )

    assert repro_quality_report.main() == 2
    assert json.loads(output.read_text())["evaluation_gate"]["passed"] is False
