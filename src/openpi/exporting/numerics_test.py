from __future__ import annotations

import numpy as np
import pytest

from openpi.exporting.numerics import action_diagnostics
from openpi.exporting.numerics import compare_outputs
from openpi.exporting.numerics import cosine_similarity


def test_cosine_handles_two_zero_tensors():
    assert cosine_similarity(np.zeros(2), np.zeros(2)) == 1.0
    assert cosine_similarity(np.zeros(2), np.ones(2)) == 0.0


def test_comparison_enforces_cosine_and_finiteness():
    passing = compare_outputs({"y": np.array([1.0, 2.0])}, {"y": np.array([1.0, 2.01])}, cosine_threshold=0.99)
    assert passing["passes"]
    failing = compare_outputs({"y": np.array([1.0])}, {"y": np.array([np.nan])}, cosine_threshold=0.99)
    assert not failing["passes"]


def test_action_diagnostics_counts_only_introduced_violations():
    reference = np.array([[[-1.1, 0.0]]])
    candidate = np.array([[[-1.1, 1.1]]])
    result = action_diagnostics(
        reference,
        candidate,
        action_low=np.array([-1.0, -1.0]),
        action_high=np.array([1.0, 1.0]),
    )
    assert result["reference_action_limit_violations"] == 1
    assert result["candidate_action_limit_violations"] == 2
    assert result["introduced_action_limit_violations"] == 1
    assert not result["action_limits_pass"]


def test_action_limit_gate_rejects_even_preexisting_candidate_violation():
    actions = np.array([[[-1.1, 0.0]]])
    result = action_diagnostics(
        actions,
        actions,
        action_low=np.array([-1.0, -1.0]),
        action_high=np.array([1.0, 1.0]),
    )
    assert result["introduced_action_limit_violations"] == 0
    assert not result["action_limits_pass"]


def test_action_limits_must_be_ordered_and_finite():
    actions = np.zeros((1, 1, 1))
    with pytest.raises(ValueError, match="less than"):
        action_diagnostics(actions, actions, action_low=np.array([1.0]), action_high=np.array([1.0]))
    with pytest.raises(ValueError, match="finite"):
        action_diagnostics(actions, actions, action_low=np.array([np.nan]), action_high=np.array([1.0]))


def test_action_envelope_mask_ignores_padded_model_dimensions():
    reference = np.array([[[0.0, 99.0]]])
    candidate = np.array([[[0.05, -99.0]]])
    result = action_diagnostics(
        reference,
        candidate,
        action_low=np.array([-1.0, 0.0]),
        action_high=np.array([1.0, 0.0]),
        action_mask=np.array([True, False]),
        max_abs_joint_bias=0.1,
    )
    assert result["action_limits_pass"]
    assert result["corpus_envelope_pass"]
    assert result["bias_passes"]
    assert result["max_abs_joint_bias"] == pytest.approx(0.05)
    assert result["active_action_dimensions"] == [0]


def test_action_envelope_mask_requires_at_least_one_dimension():
    actions = np.zeros((1, 1, 1))
    with pytest.raises(ValueError, match="at least one"):
        action_diagnostics(
            actions,
            actions,
            action_low=np.array([0.0]),
            action_high=np.array([0.0]),
            action_mask=np.array([False]),
        )


def test_compare_rejects_different_names():
    with pytest.raises(ValueError, match="names differ"):
        compare_outputs({"a": np.ones(1)}, {"b": np.ones(1)}, cosine_threshold=0.9)
