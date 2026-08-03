#!/usr/bin/env python3
"""Compute the offline promotion metrics for a fixed golden action corpus.

Input is an ``npz`` containing ``student`` and ``teacher`` arrays shaped
``[samples, horizon, joints]``. Optional keys are ``ground_truth``,
``action_low`` and ``action_high``. The command emits stable JSON suitable for
the run manifest and promotion-gate tooling.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np


def cosine_similarity(left: np.ndarray, right: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    left_flat = left.reshape(left.shape[0], -1).astype(np.float64)
    right_flat = right.reshape(right.shape[0], -1).astype(np.float64)
    numerator = np.sum(left_flat * right_flat, axis=1)
    denominator = np.linalg.norm(left_flat, axis=1) * np.linalg.norm(right_flat, axis=1)
    return numerator / np.maximum(denominator, eps)


def normalized_rmse(prediction: np.ndarray, reference: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    rmse = np.sqrt(np.mean(np.square(prediction - reference), axis=(0, 1)))
    scale = np.std(reference, axis=(0, 1))
    return rmse / np.maximum(scale, eps)


def trajectory_roughness(actions: np.ndarray) -> np.ndarray:
    """RMS second difference per sample; zero for horizons shorter than three."""
    if actions.shape[1] < 3:
        return np.zeros(actions.shape[0], dtype=np.float64)
    acceleration = np.diff(actions.astype(np.float64), n=2, axis=1)
    return np.sqrt(np.mean(np.square(acceleration), axis=(1, 2)))


def saturation_rate(actions: np.ndarray, low: np.ndarray, high: np.ndarray, margin: float = 0.005) -> float:
    low = np.asarray(low).reshape((1,) * (actions.ndim - 1) + (-1,))
    high = np.asarray(high).reshape((1,) * (actions.ndim - 1) + (-1,))
    width = high - low
    saturated = (actions <= low + margin * width) | (actions >= high - margin * width)
    return float(np.mean(saturated))


def compute_metrics(
    student: np.ndarray,
    teacher: np.ndarray,
    *,
    ground_truth: np.ndarray | None = None,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
) -> dict[str, Any]:
    if student.shape != teacher.shape or student.ndim != 3:
        raise ValueError("student and teacher must have identical [samples, horizon, joints] shapes")
    if not np.all(np.isfinite(student)) or not np.all(np.isfinite(teacher)):
        raise ValueError("student and teacher arrays must be finite")
    if ground_truth is not None and ground_truth.shape != student.shape:
        raise ValueError("ground_truth must have the same shape as student")

    difference = student.astype(np.float64) - teacher.astype(np.float64)
    cosine = cosine_similarity(student, teacher)
    result: dict[str, Any] = {
        "samples": student.shape[0],
        "horizon": student.shape[1],
        "joints": student.shape[2],
        "kd_mse": float(np.mean(np.square(difference))),
        "kd_cosine_mean": float(np.mean(cosine)),
        "kd_cosine_p05": float(np.quantile(cosine, 0.05)),
        "per_joint_normalized_rmse": normalized_rmse(student, teacher).tolist(),
        "final_chunk_rmse": float(np.sqrt(np.mean(np.square(difference[:, -1, :])))),
        "student_roughness_mean": float(np.mean(trajectory_roughness(student))),
        "teacher_roughness_mean": float(np.mean(trajectory_roughness(teacher))),
    }
    if ground_truth is not None:
        result["ground_truth_mse"] = float(np.mean(np.square(student.astype(np.float64) - ground_truth)))
    if (action_low is None) != (action_high is None):
        raise ValueError("action_low and action_high must be supplied together")
    if action_low is not None and action_high is not None:
        result["student_saturation_rate"] = saturation_rate(student, action_low, action_high)
        result["teacher_saturation_rate"] = saturation_rate(teacher, action_low, action_high)
        result["action_limit_violations"] = int(np.sum((student < action_low) | (student > action_high)))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with np.load(args.corpus) as corpus:
        metrics = compute_metrics(
            corpus["student"],
            corpus["teacher"],
            ground_truth=corpus["ground_truth"] if "ground_truth" in corpus else None,
            action_low=corpus["action_low"] if "action_low" in corpus else None,
            action_high=corpus["action_high"] if "action_high" in corpus else None,
        )
    payload = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
