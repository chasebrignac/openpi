"""Numerical comparison gates shared by ONNX and TensorRT validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def cosine_similarity(reference: np.ndarray, candidate: np.ndarray, *, eps: float = 1e-30) -> float:
    left = np.asarray(reference, dtype=np.float64).reshape(-1)
    right = np.asarray(candidate, dtype=np.float64).reshape(-1)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= eps and right_norm <= eps:
        return 1.0
    if left_norm <= eps or right_norm <= eps:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def compare_outputs(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    *,
    cosine_threshold: float,
) -> dict[str, Any]:
    if not 0.0 <= cosine_threshold <= 1.0:
        raise ValueError("cosine_threshold must be in [0, 1]")
    if set(reference) != set(candidate):
        raise ValueError(
            f"Reference/candidate output names differ: reference={sorted(reference)}, candidate={sorted(candidate)}"
        )

    tensors: dict[str, Any] = {}
    passes = True
    for name in sorted(reference):
        left = np.asarray(reference[name])
        right = np.asarray(candidate[name])
        if left.shape != right.shape:
            raise ValueError(f"Output {name!r} shape differs: {left.shape} != {right.shape}")
        finite = bool(np.all(np.isfinite(left)) and np.all(np.isfinite(right)))
        difference = right.astype(np.float64) - left.astype(np.float64)
        cosine = cosine_similarity(left, right) if finite else float("nan")
        tensor_passes = finite and cosine >= cosine_threshold
        passes &= tensor_passes
        tensors[name] = {
            "shape": list(left.shape),
            "finite": finite,
            "cosine_similarity": cosine,
            "rmse": float(np.sqrt(np.mean(np.square(difference)))) if finite else float("nan"),
            "max_abs_error": float(np.max(np.abs(difference))) if finite else float("nan"),
            "passes": tensor_passes,
        }
    return {"cosine_threshold": cosine_threshold, "passes": passes, "tensors": tensors}


def action_diagnostics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
    action_mask: np.ndarray | None = None,
    max_abs_joint_bias: float | None = None,
) -> dict[str, Any]:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape or reference.ndim < 2:
        raise ValueError("Action tensors must have identical batched [..., joints] shapes")
    reduce_axes = tuple(range(candidate.ndim - 1))
    bias = np.mean(candidate.astype(np.float64) - reference.astype(np.float64), axis=reduce_axes)
    bias_mask = (
        np.ones(bias.shape, dtype=np.bool_)
        if action_mask is None
        else np.asarray(action_mask, dtype=np.bool_).reshape(-1)
    )
    if bias_mask.shape != bias.shape:
        raise ValueError("Action mask dimension does not match the action tensor")
    if not np.any(bias_mask):
        raise ValueError("Action envelope mask must select at least one dimension")
    evaluated_bias = bias[bias_mask]
    maximum_bias = float(np.max(np.abs(evaluated_bias)))
    result: dict[str, Any] = {
        "per_joint_bias": bias.tolist(),
        "max_abs_joint_bias": maximum_bias,
        "bias_limit": max_abs_joint_bias,
        "bias_passes": max_abs_joint_bias is None or maximum_bias <= max_abs_joint_bias,
        "bias_evaluated_dimensions": np.flatnonzero(bias_mask).tolist(),
        "action_limits_evaluated": action_low is not None,
        "action_gate_kind": "corpus_envelope_not_hardware_safety",
    }
    if (action_low is None) != (action_high is None):
        raise ValueError("action_low and action_high must be provided together")
    if action_low is None and action_mask is not None:
        raise ValueError("action_mask requires action_low and action_high")
    if action_low is not None and action_high is not None:
        action_low = np.asarray(action_low, dtype=np.float64).reshape(-1)
        action_high = np.asarray(action_high, dtype=np.float64).reshape(-1)
        action_mask = np.ones(action_low.shape, dtype=np.bool_) if action_mask is None else bias_mask
        if action_low.shape != action_high.shape or action_low.shape != action_mask.shape:
            raise ValueError("Action low/high/mask dimensions must match")
        if not np.any(action_mask):
            raise ValueError("Action envelope mask must select at least one dimension")
        if not np.all(np.isfinite(action_low[action_mask])) or not np.all(np.isfinite(action_high[action_mask])):
            raise ValueError("Active action envelope bounds must be finite")
        if np.any(action_low[action_mask] >= action_high[action_mask]):
            raise ValueError("Every active action lower bound must be less than its upper bound")
        low = action_low.reshape((1,) * (candidate.ndim - 1) + (-1,))
        high = action_high.reshape((1,) * (candidate.ndim - 1) + (-1,))
        mask = action_mask.reshape((1,) * (candidate.ndim - 1) + (-1,))
        if low.shape[-1] != candidate.shape[-1] or high.shape[-1] != candidate.shape[-1]:
            raise ValueError("Action envelope dimension does not match the action tensor")
        reference_violation = ((reference < low) | (reference > high)) & mask
        candidate_violation = ((candidate < low) | (candidate > high)) & mask
        introduced = candidate_violation & ~reference_violation
        corpus_envelope_pass = not bool(np.any(candidate_violation))
        result.update(
            {
                "reference_action_limit_violations": int(np.sum(reference_violation)),
                "candidate_action_limit_violations": int(np.sum(candidate_violation)),
                "introduced_action_limit_violations": int(np.sum(introduced)),
                # Keep the original field for report consumers while naming
                # the actual semantics explicitly. This is an empirical
                # corpus-envelope regression gate, not a hardware-safety gate.
                "action_limits_pass": corpus_envelope_pass,
                "corpus_envelope_pass": corpus_envelope_pass,
                "action_low": action_low.tolist(),
                "action_high": action_high.tolist(),
                "action_mask": action_mask.tolist(),
                "active_action_dimensions": np.flatnonzero(action_mask).tolist(),
            }
        )
    return result
