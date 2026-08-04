#!/usr/bin/env python3
"""Combine offline and paired-rollout evidence into a stage promotion report."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any


def relative_improvement_per_5k(previous: float, current: float, step_delta: int) -> float:
    """Return multiplicative error reduction normalized to a 5k-step interval."""
    if previous <= 0 or current < 0:
        raise ValueError("Error metrics require previous > 0 and current >= 0")
    if step_delta <= 0:
        raise ValueError("step_delta must be positive")
    if current == 0:
        return 1.0
    return 1.0 - (current / previous) ** (5_000 / step_delta)


def _gate(value: float | int | None, *, threshold: float | int | None, comparison: str) -> dict[str, Any]:
    available = value is not None and threshold is not None
    if not available:
        return {"available": False, "value": value, "threshold": threshold, "pass": None}
    if comparison == "min":
        passed = value >= threshold
    elif comparison == "max":
        passed = value <= threshold
    else:
        raise ValueError(f"Unknown comparison {comparison}")
    return {"available": True, "value": value, "threshold": threshold, "comparison": comparison, "pass": passed}


def _kd_mse(report: dict[str, Any]) -> float:
    container = report.get("velocity_metrics", report["action_metrics"])
    return float(container["kd_mse"])


def _quality_by_step(quality_reports: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result = {}
    for report in quality_reports:
        step = int(report["checkpoint_step"])
        if step in result:
            raise ValueError(f"Duplicate quality report for step {step}")
        result[step] = report
    return result


def _constant_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in provenance.items() if key != "student_checkpoint"}


def _require_sha256(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _require_pinned_revision(value: Any) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("Dataset revision must be a pinned lowercase 40-character Git hash")


def validate_evidence_provenance(
    offline_reports: list[dict[str, Any]],
    quality_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Require one evidence chain while allowing the student checkpoint to advance."""
    if not offline_reports:
        raise ValueError("At least one offline report is required")
    required = {
        "run_id",
        "student_config",
        "student_checkpoint",
        "teacher_config",
        "teacher_checkpoint",
        "dataset",
        "golden",
        "normalization_range",
    }
    by_step: dict[int, dict[str, Any]] = {}
    common = None
    for report in offline_reports:
        provenance = report.get("provenance")
        if not isinstance(provenance, dict) or not required.issubset(provenance):
            raise ValueError("Offline report is missing required provenance fields")
        if (
            provenance.get("schema_version") != 1
            or not isinstance(provenance["run_id"], str)
            or not provenance["run_id"].strip()
        ):
            raise ValueError("Offline report has an invalid provenance schema or run_id")
        for field in (
            "student_config",
            "student_checkpoint",
            "teacher_config",
            "teacher_checkpoint",
            "dataset",
            "golden",
        ):
            if not isinstance(provenance[field], dict):
                raise ValueError(f"Offline report provenance field {field!r} must be an object")
        step = int(report["student_step"])
        if provenance["student_checkpoint"].get("step") != step:
            raise ValueError(f"Student checkpoint provenance step does not match report step {step}")
        if provenance["student_config"].get("name") != report.get("student_config"):
            raise ValueError(f"Student config provenance does not match report step {step}")
        if provenance["teacher_config"].get("name") != report.get("teacher_config"):
            raise ValueError(f"Teacher config provenance does not match report step {step}")
        _require_sha256(provenance["student_config"].get("fingerprint_sha256"), "student config fingerprint")
        _require_sha256(
            provenance["student_config"].get("training_fingerprint_sha256"),
            "student training config fingerprint",
        )
        _require_sha256(provenance["teacher_config"].get("fingerprint_sha256"), "teacher config fingerprint")
        _require_sha256(provenance["student_checkpoint"].get("model_sha256"), "student checkpoint hash")
        _require_sha256(provenance["student_checkpoint"].get("metadata_sha256"), "student checkpoint metadata hash")
        _require_sha256(provenance["teacher_checkpoint"].get("model_sha256"), "teacher checkpoint hash")
        _require_sha256(provenance["golden"].get("sha256"), "golden corpus hash")
        _require_sha256(provenance["golden"].get("metadata_sha256"), "golden metadata hash")
        if not provenance["dataset"].get("repo_id") or not provenance["dataset"].get("revision"):
            raise ValueError("Dataset provenance requires repo_id and revision")
        _require_pinned_revision(provenance["dataset"]["revision"])
        normalization_range = provenance["normalization_range"]
        if not isinstance(normalization_range, dict):
            raise ValueError("Normalization range provenance must be an object")
        low = normalization_range.get("low")
        high = normalization_range.get("high")
        if not isinstance(low, int | float) or not isinstance(high, int | float):
            raise ValueError("Normalization range provenance must contain numeric low/high")
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError("Normalization range provenance must be finite and ordered")
        current_common = _constant_provenance(provenance)
        if common is None:
            common = current_common
        elif current_common != common:
            raise ValueError(f"Offline evidence provenance diverges at step {step}")
        by_step[step] = provenance

    for quality in quality_reports:
        step = int(quality["checkpoint_step"])
        if step not in by_step:
            raise ValueError(f"Quality evidence at step {step} has no matching offline report")
        if quality.get("provenance") != by_step[step]:
            raise ValueError(f"Quality evidence provenance does not match offline evidence at step {step}")

    assert common is not None
    return {
        **common,
        "student_checkpoints": {str(step): provenance["student_checkpoint"] for step, provenance in by_step.items()},
    }


def build_promotion_report(
    *,
    stage: str,
    offline_reports: list[dict[str, Any]],
    quality_reports: list[dict[str, Any]],
    max_rollout_gap: float | None,
    min_kd_cosine: float | None = None,
    max_kd_mse: float | None = None,
    max_per_joint_nrmse: float | None = None,
    max_action_chunk_rmse: float | None = None,
    max_steps: int = 30_000,
) -> dict[str, Any]:
    if stage not in {"shallow", "snapflow"}:
        raise ValueError(f"Unknown stage: {stage}")
    if not offline_reports:
        raise ValueError("At least one offline report is required")
    evidence_provenance = validate_evidence_provenance(offline_reports, quality_reports)
    reports = sorted(offline_reports, key=lambda item: int(item["student_step"]))
    if any(report.get("stage") != stage for report in reports):
        raise ValueError(f"All offline reports must be for stage {stage!r}")
    steps = [int(report["student_step"]) for report in reports]
    if len(set(steps)) != len(steps):
        raise ValueError(f"Offline reports contain duplicate steps: {steps}")
    latest = reports[-1]
    if latest.get("stage") != stage:
        raise ValueError(f"Latest report is for {latest.get('stage')!r}, not {stage!r}")
    latest_step = steps[-1]
    action_metrics = latest["action_metrics"]
    kd_metrics = latest.get("velocity_metrics", action_metrics)
    quality_map = _quality_by_step(quality_reports)
    quality = quality_map.get(latest_step)
    rollout = None if quality is None else quality.get("paired_rollout")

    gates = {
        "normalization_range_excursions": _gate(
            action_metrics.get("normalization_range_excursions"), threshold=0, comparison="max"
        ),
        "kd_cosine": _gate(kd_metrics.get("kd_cosine_mean"), threshold=min_kd_cosine, comparison="min"),
        "kd_mse": _gate(kd_metrics.get("kd_mse"), threshold=max_kd_mse, comparison="max"),
        "per_joint_nrmse": _gate(
            max(kd_metrics["per_joint_normalized_rmse"]),
            threshold=max_per_joint_nrmse,
            comparison="max",
        ),
        "action_chunk_rmse": _gate(
            action_metrics.get("action_chunk_rmse"),
            threshold=max_action_chunk_rmse,
            comparison="max",
        ),
        "paired_rollout_gap": _gate(
            None if rollout is None else float(rollout["reference_success"]) - float(rollout["student_success"]),
            threshold=max_rollout_gap,
            comparison="max",
        ),
    }
    required = ["normalization_range_excursions", "paired_rollout_gap"]
    required.extend(
        name
        for name in ("kd_cosine", "kd_mse", "per_joint_nrmse", "action_chunk_rmse")
        if gates[name]["threshold"] is not None
    )
    if stage == "snapflow":
        snapflow_metrics = latest["snapflow_metrics"]
        gates["offline_error_gap_closed"] = _gate(
            snapflow_metrics["offline_error_gap_closed_fraction"],
            threshold=0.70,
            comparison="min",
        )
        denoise_speedup = None if quality is None else quality.get("denoise_speedup")
        gates["denoise_speedup"] = _gate(denoise_speedup, threshold=8.0, comparison="min")
        required += ["offline_error_gap_closed", "denoise_speedup"]

    missing_required = [name for name in required if not gates[name]["available"]]
    failed_required = [name for name in required if gates[name]["pass"] is False]
    promotion_ready = not missing_required and not failed_required

    trend: dict[str, Any] = {"available": False}
    if len(reports) >= 2:
        previous = reports[-2]
        kd_improvement = relative_improvement_per_5k(
            _kd_mse(previous),
            _kd_mse(latest),
            latest_step - int(previous["student_step"]),
        )
        previous_quality = quality_map.get(int(previous["student_step"]))
        previous_rollout = None if previous_quality is None else previous_quality.get("paired_rollout")
        rollout_improving = None
        if rollout is not None and previous_rollout is not None:
            rollout_improving = float(rollout["student_success"]) > float(previous_rollout["student_success"])
        trend = {
            "available": True,
            "previous_step": int(previous["student_step"]),
            "current_step": latest_step,
            "kd_improvement_per_5k": kd_improvement,
            "kd_improving_at_least_five_percent": kd_improvement >= 0.05,
            "rollout_improving": rollout_improving,
        }

    if promotion_ready:
        recommendation = "promote"
    elif latest_step < max_steps:
        recommendation = "continue_to_next_checkpoint"
    elif trend.get("kd_improving_at_least_five_percent") is True and trend.get("rollout_improving") is True:
        recommendation = "extension_requires_separate_approval"
    elif missing_required:
        recommendation = "stop_at_cap_pending_missing_evidence"
    else:
        recommendation = "stop_at_cap_no_extension_signal"

    finite = all(
        math.isfinite(float(value))
        for value in (
            action_metrics["kd_mse"],
            action_metrics["kd_cosine_mean"],
            action_metrics["action_chunk_rmse"],
            _kd_mse(latest),
        )
    )
    return {
        "schema_version": 1,
        "provenance": evidence_provenance,
        "stage": stage,
        "student_config": latest["student_config"],
        "checkpoint_step": latest_step,
        "evaluated_steps": steps,
        "metrics_finite": finite,
        "gates": gates,
        "required_gates": required,
        "missing_required_gates": missing_required,
        "failed_required_gates": failed_required,
        "promotion_ready": promotion_ready and finite,
        "trend": trend,
        "recommendation": recommendation if finite else "stop_non_finite_metrics",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("shallow", "snapflow"), required=True)
    parser.add_argument("--offline", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--quality", type=pathlib.Path, action="append", default=[])
    parser.add_argument("--max-rollout-gap", type=float, required=True)
    parser.add_argument("--min-kd-cosine", type=float)
    parser.add_argument("--max-kd-mse", type=float)
    parser.add_argument("--max-per-joint-nrmse", type=float)
    parser.add_argument("--max-action-chunk-rmse", type=float)
    parser.add_argument("--max-steps", type=int, default=30_000)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Write incomplete/failed evidence reports without returning a failing exit status.",
    )
    return parser.parse_args()


def promotion_exit_code(report: dict[str, Any], *, report_only: bool) -> int:
    return 0 if report_only or report.get("promotion_ready") is True else 2


def main() -> int:
    args = parse_args()
    report = build_promotion_report(
        stage=args.stage,
        offline_reports=[json.loads(path.read_text()) for path in args.offline],
        quality_reports=[json.loads(path.read_text()) for path in args.quality],
        max_rollout_gap=args.max_rollout_gap,
        min_kd_cosine=args.min_kd_cosine,
        max_kd_mse=args.max_kd_mse,
        max_per_joint_nrmse=args.max_per_joint_nrmse,
        max_action_chunk_rmse=args.max_action_chunk_rmse,
        max_steps=args.max_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return promotion_exit_code(report, report_only=args.report_only)


if __name__ == "__main__":
    raise SystemExit(main())
