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
