import json

import pytest

from scripts import repro_cost_guard


@pytest.fixture
def config():
    return {
        "aws": {
            "hard_cap_usd": 100.0,
            "category_caps_usd": {"training": 60.0},
            "approved_instances": {"g.test": {"hourly_usd": 5.0}},
        }
    }


def test_projects_approved_run(config):
    result = repro_cost_guard.project_run(
        config, {"entries": []}, category="training", instance_type="g.test", instance_count=2, hours=3
    )
    assert result.projected_usd == 30
    assert result.remaining_after_usd == 70


def test_rejects_category_overage(config):
    ledger = {"entries": [{"category": "training", "usd": 40}]}
    with pytest.raises(repro_cost_guard.BudgetError, match="category cap"):
        repro_cost_guard.project_run(
            config, ledger, category="training", instance_type="g.test", instance_count=1, hours=5
        )


def test_cancelled_reservations_do_not_count(config):
    ledger = {"entries": [{"category": "training", "usd": 99, "state": "cancelled"}]}
    result = repro_cost_guard.project_run(
        config, ledger, category="training", instance_type="g.test", instance_count=1, hours=1
    )
    assert result.total_committed_usd == 0


def test_reservation_is_atomic(tmp_path, config):
    projection = repro_cost_guard.project_run(
        config, {"entries": []}, category="training", instance_type="g.test", instance_count=1, hours=1
    )
    path = tmp_path / "ledger.json"
    reservation_id = repro_cost_guard.reserve(path, projection, label="pilot", command="run")
    payload = json.loads(path.read_text())
    assert payload["entries"][0]["id"] == reservation_id
    assert payload["entries"][0]["usd"] == 5
