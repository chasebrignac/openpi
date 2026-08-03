import numpy as np
import pytest

from scripts import repro_offline_metrics


def test_identical_outputs_pass_exactly():
    actions = np.arange(48, dtype=np.float32).reshape(2, 3, 8)
    metrics = repro_offline_metrics.compute_metrics(actions, actions)
    assert metrics["kd_mse"] == 0
    assert metrics["kd_cosine_mean"] == pytest.approx(1)
    assert metrics["final_chunk_rmse"] == 0


def test_reports_limits_and_per_joint_error():
    teacher = np.ones((2, 4, 2), dtype=np.float32)
    teacher[:, :, 1] = np.arange(4)
    student = teacher.copy()
    student[0, -1, 0] = 3
    metrics = repro_offline_metrics.compute_metrics(
        student,
        teacher,
        action_low=np.array([-2, -2]),
        action_high=np.array([2, 4]),
    )
    assert metrics["action_limit_violations"] == 1
    assert len(metrics["per_joint_normalized_rmse"]) == 2


def test_rejects_non_finite_values():
    student = np.zeros((1, 2, 1))
    student[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        repro_offline_metrics.compute_metrics(student, np.zeros_like(student))
