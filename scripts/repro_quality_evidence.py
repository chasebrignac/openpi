#!/usr/bin/env python3
"""Bind a paired episode quality report to one offline checkpoint report."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any

if __package__:
    from scripts.repro_make_golden import sha256_file
    from scripts.repro_promotion_report import validate_evidence_provenance
else:
    from repro_make_golden import sha256_file
    from repro_promotion_report import validate_evidence_provenance


def model_stage_identity(role: str, model_sha256: str) -> str:
    """Return the full model digest in the evaluator's worker-safe stage syntax."""

    if role not in {"teacher", "student"}:
        raise ValueError(f"Unknown model identity role: {role}")
    if len(model_sha256) != 64 or any(character not in "0123456789abcdef" for character in model_sha256):
        raise ValueError("Model identity requires a lowercase SHA-256 digest")
    # repro_libero_eval intentionally permits at most 64 characters and no
    # colon in a stage identifier. The complete digest is already globally
    # role-independent and collision-resistant; the quality report fields
    # (`base_stage` and `candidate_stage`) supply the role.
    return model_sha256


def aggregate_success(groups: dict[str, Any]) -> tuple[int, float, float]:
    total_pairs = 0
    weighted_reference = 0.0
    weighted_student = 0.0
    if not groups:
        raise ValueError("Quality report has no episode groups")
    for name, group in groups.items():
        episodes = group.get("episodes")
        reference = group.get("baseline_success")
        student = group.get("candidate_success")
        if not isinstance(episodes, int) or isinstance(episodes, bool) or episodes <= 0:
            raise ValueError(f"Quality group {name!r} has invalid episode count")
        if not isinstance(reference, int | float) or not isinstance(student, int | float):
            raise ValueError(f"Quality group {name!r} is missing numeric paired success")
        if (
            not math.isfinite(reference)
            or not math.isfinite(student)
            or not 0 <= reference <= 1
            or not 0 <= student <= 1
        ):
            raise ValueError(f"Quality group {name!r} success must be finite and in [0, 1]")
        total_pairs += episodes
        weighted_reference += episodes * reference
        weighted_student += episodes * student
    return total_pairs, weighted_reference / total_pairs, weighted_student / total_pairs


def build_quality_evidence(
    quality_report: dict[str, Any],
    offline_report: dict[str, Any],
    *,
    required_pairs: int,
    quality_report_sha256: str,
    offline_report_sha256: str,
    quality_report_path: str,
    offline_report_path: str,
    denoise_speedup: float | None = None,
) -> dict[str, Any]:
    if required_pairs <= 0:
        raise ValueError("required_pairs must be positive")
    validate_evidence_provenance([offline_report], [])
    provenance = offline_report["provenance"]
    expected_reference = model_stage_identity(
        "teacher",
        provenance["teacher_checkpoint"]["model_sha256"],
    )
    expected_student = model_stage_identity(
        "student",
        provenance["student_checkpoint"]["model_sha256"],
    )
    actual_reference = quality_report.get("base_stage")
    actual_student = quality_report.get("candidate_stage")
    if actual_reference != expected_reference or actual_student != expected_student:
        raise ValueError(
            "Paired quality model identity mismatch: "
            f"expected {expected_reference!r}/{expected_student!r}, "
            f"got {actual_reference!r}/{actual_student!r}"
        )
    evaluation_gate = quality_report.get("evaluation_gate")
    if not isinstance(evaluation_gate, dict) or evaluation_gate.get("passed") is not True:
        raise ValueError("Source quality report did not pass its evaluation gate")
    total_pairs, reference_success, student_success = aggregate_success(quality_report.get("groups", {}))
    if total_pairs != required_pairs:
        raise ValueError(f"Quality report has {total_pairs} complete pairs; exactly {required_pairs} are required")
    if denoise_speedup is not None and (not math.isfinite(denoise_speedup) or denoise_speedup <= 0):
        raise ValueError("denoise_speedup must be finite and positive")

    result: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint_step": offline_report["student_step"],
        "provenance": provenance,
        "paired_rollout": {
            "student_success": student_success,
            "reference_success": reference_success,
            "complete_pairs": total_pairs,
        },
        "model_identity": {
            "reference_stage": expected_reference,
            "student_stage": expected_student,
        },
        "source_quality_report": {
            "path": quality_report_path,
            "sha256": quality_report_sha256,
            "evaluation_mode": evaluation_gate.get("mode"),
        },
        "source_offline_report": {
            "path": offline_report_path,
            "sha256": offline_report_sha256,
        },
    }
    if denoise_speedup is not None:
        result["denoise_speedup"] = denoise_speedup
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-report", type=pathlib.Path, required=True)
    parser.add_argument("--offline-report", type=pathlib.Path, required=True)
    parser.add_argument("--required-pairs", type=int, required=True)
    parser.add_argument("--denoise-speedup", type=float)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    quality_path = args.quality_report.expanduser().resolve()
    offline_path = args.offline_report.expanduser().resolve()
    result = build_quality_evidence(
        json.loads(quality_path.read_text()),
        json.loads(offline_path.read_text()),
        required_pairs=args.required_pairs,
        denoise_speedup=args.denoise_speedup,
        quality_report_sha256=sha256_file(quality_path),
        offline_report_sha256=sha256_file(offline_path),
        quality_report_path=str(quality_path),
        offline_report_path=str(offline_path),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
