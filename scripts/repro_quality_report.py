#!/usr/bin/env python3
"""Aggregate paired closed-loop evaluation JSONL and apply quality gates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import pathlib
from typing import Any

import numpy as np

OFFICIAL_LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
OFFICIAL_LIBERO_EPISODES_PER_SUITE = 500
OFFICIAL_LIBERO_EPISODES = len(OFFICIAL_LIBERO_SUITES) * OFFICIAL_LIBERO_EPISODES_PER_SUITE


def paired_bootstrap_interval(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    samples: int = 20_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if baseline.shape != candidate.shape or baseline.ndim != 1:
        raise ValueError("paired success arrays must be one-dimensional and equal length")
    if baseline.size == 0:
        raise ValueError("paired success arrays cannot be empty")
    rng = np.random.default_rng(seed)
    differences = candidate.astype(np.float64) - baseline.astype(np.float64)
    draws = rng.integers(0, differences.size, size=(samples, differences.size))
    means = differences[draws].mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def load_records(path: pathlib.Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    required = {"pair_id", "stage", "benchmark", "suite", "task", "success"}
    for index, record in enumerate(records, start=1):
        missing = required - record.keys()
        if missing:
            raise ValueError(f"record {index} is missing keys: {sorted(missing)}")
        if "error" in record:
            raise ValueError(f"record {index} contains an infrastructure error and is not quality evidence")
    return records


def summarize(records: list[dict[str, Any]], *, base_stage: str, candidate_stage: str) -> dict[str, Any]:
    if base_stage == candidate_stage:
        raise ValueError("base_stage and candidate_stage must be different")
    expected_stages = {base_stage, candidate_stage}
    indexed: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        if record["stage"] not in expected_stages:
            continue
        key = (record["benchmark"], str(record["pair_id"]))
        if record["stage"] in indexed[key]:
            raise ValueError(f"duplicate stage/pair record: {key} {record['stage']}")
        indexed[key][record["stage"]] = record

    if not indexed:
        raise ValueError("no complete baseline/candidate pairs")
    incomplete = {
        key: sorted(expected_stages - value.keys()) for key, value in indexed.items() if value.keys() != expected_stages
    }
    if incomplete:
        examples = list(incomplete.items())[:5]
        raise ValueError(f"incomplete baseline/candidate pairs ({len(incomplete)}): {examples}")

    pairs = list(indexed.values())

    groups: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair in pairs:
        base = pair[base_stage]
        candidate = pair[candidate_stage]
        for field in ("benchmark", "suite", "task"):
            if base[field] != candidate[field]:
                raise ValueError(
                    f"paired metadata mismatch for {base['pair_id']}: {field}={base[field]!r}/{candidate[field]!r}"
                )
        groups[(base["benchmark"], base["suite"])].append((base, candidate))

    summaries: dict[str, Any] = {}
    for (benchmark, suite), group in sorted(groups.items()):
        baseline = np.asarray([item[0]["success"] for item in group], dtype=np.float64)
        candidate = np.asarray([item[1]["success"] for item in group], dtype=np.float64)
        low, high = paired_bootstrap_interval(baseline, candidate)
        metrics: dict[str, Any] = {
            "episodes": len(group),
            "baseline_success": float(baseline.mean()),
            "candidate_success": float(candidate.mean()),
            "difference_points": float(100 * (candidate.mean() - baseline.mean())),
            "difference_95ci_points": [100 * low, 100 * high],
        }
        for name in ("path_length", "sparc"):
            available = all(name in item for pair in group for item in pair)
            if available:
                metrics[f"baseline_{name}"] = float(np.mean([item[0][name] for item in group]))
                metrics[f"candidate_{name}"] = float(np.mean([item[1][name] for item in group]))
        summaries[f"{benchmark}/{suite}"] = metrics

    return {"base_stage": base_stage, "candidate_stage": candidate_stage, "groups": summaries}


def _gate_check(name: str, *, passed: bool, actual: Any, requirement: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "actual": actual, "requirement": requirement}


def apply_evaluation_gate(
    report: dict[str, Any],
    *,
    mode: str,
    expected_pairs: int | None = None,
    minimum_baseline_success: float | None = None,
) -> dict[str, Any]:
    """Attach count/quality checks and return a report with an overall pass bit."""
    if mode not in {"official-final", "intermediate", "report-only"}:
        raise ValueError(f"unknown evaluation mode: {mode}")
    if mode == "official-final" and expected_pairs is not None:
        raise ValueError("official-final always requires exactly 2,000 pairs; do not set expected_pairs")
    if mode == "intermediate" and (expected_pairs is None or expected_pairs <= 0):
        raise ValueError("intermediate mode requires a positive expected_pairs count")
    if mode == "report-only" and expected_pairs is not None and expected_pairs <= 0:
        raise ValueError("expected_pairs must be positive")
    if mode != "report-only" and (minimum_baseline_success is None or not 0.0 < minimum_baseline_success <= 1.0):
        raise ValueError("official and intermediate gates require a minimum_baseline_success in (0, 1]")
    if minimum_baseline_success is not None and not 0.0 < minimum_baseline_success <= 1.0:
        raise ValueError("minimum_baseline_success must be in (0, 1]")

    groups = report["groups"]
    libero_groups = {key: value for key, value in groups.items() if key.startswith("libero/")}
    total_pairs = sum(group["episodes"] for group in groups.values())
    checks = []

    count_requirement = OFFICIAL_LIBERO_EPISODES if mode == "official-final" else expected_pairs
    if count_requirement is not None:
        checks.append(
            _gate_check(
                "complete_pair_count",
                passed=total_pairs == count_requirement,
                actual=total_pairs,
                requirement=f"exactly {count_requirement} complete pairs",
            )
        )

    if minimum_baseline_success is not None:
        baseline_successes = [
            (group["baseline_success"], group["episodes"]) for group in groups.values() if group.get("episodes", 0) > 0
        ]
        baseline_pair_count = sum(episodes for _, episodes in baseline_successes)
        aggregate_baseline = (
            sum(success * episodes for success, episodes in baseline_successes) / baseline_pair_count
            if baseline_pair_count
            else None
        )
        checks.append(
            _gate_check(
                "absolute_baseline_health",
                passed=aggregate_baseline is not None and aggregate_baseline >= minimum_baseline_success,
                actual=aggregate_baseline,
                requirement=f"aggregate base success is at least {minimum_baseline_success:.3f}",
            )
        )

    if mode == "official-final":
        actual_suites = sorted(key.removeprefix("libero/") for key in libero_groups)
        checks.append(
            _gate_check(
                "libero_only",
                passed=len(libero_groups) == len(groups),
                actual=sorted(groups),
                requirement="all groups use benchmark='libero'",
            )
        )
        checks.append(
            _gate_check(
                "official_suite_set",
                passed=set(actual_suites) == set(OFFICIAL_LIBERO_SUITES),
                actual=actual_suites,
                requirement=f"exact suites {list(OFFICIAL_LIBERO_SUITES)}",
            )
        )
        for suite in OFFICIAL_LIBERO_SUITES:
            group = libero_groups.get(f"libero/{suite}")
            episodes = 0 if group is None else group["episodes"]
            checks.append(
                _gate_check(
                    f"{suite}_pair_count",
                    passed=episodes == OFFICIAL_LIBERO_EPISODES_PER_SUITE,
                    actual=episodes,
                    requirement=f"exactly {OFFICIAL_LIBERO_EPISODES_PER_SUITE} complete pairs",
                )
            )

        official_groups = [libero_groups.get(f"libero/{suite}") for suite in OFFICIAL_LIBERO_SUITES]
        if all(group is not None for group in official_groups):
            official_pair_count = sum(group["episodes"] for group in official_groups)
            aggregate_difference = (
                sum(group["difference_points"] * group["episodes"] for group in official_groups) / official_pair_count
            )
        else:
            aggregate_difference = None
        checks.append(
            _gate_check(
                "aggregate_noninferiority",
                passed=aggregate_difference is not None and aggregate_difference >= -2.0 - 1e-9,
                actual=aggregate_difference,
                requirement="candidate success is no more than 2 percentage points below base",
            )
        )
        for suite in OFFICIAL_LIBERO_SUITES:
            group = libero_groups.get(f"libero/{suite}")
            difference = None if group is None else group["difference_points"]
            checks.append(
                _gate_check(
                    f"{suite}_noninferiority",
                    passed=difference is not None and difference >= -3.0 - 1e-9,
                    actual=difference,
                    requirement="candidate success is no more than 3 percentage points below base",
                )
            )

    gated_report = dict(report)
    gated_report["evaluation_gate"] = {
        "mode": mode,
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }
    return gated_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=pathlib.Path)
    parser.add_argument("--base-stage", default="base")
    parser.add_argument("--candidate-stage", default="final")
    parser.add_argument("--mode", choices=("official-final", "intermediate", "report-only"), default="official-final")
    parser.add_argument(
        "--expected-pairs",
        type=int,
        help="Required for intermediate mode; optional count check for report-only mode.",
    )
    parser.add_argument(
        "--minimum-baseline-success",
        type=float,
        help="Reviewed absolute base-policy floor; required for intermediate and official-final gates.",
    )
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = summarize(load_records(args.records), base_stage=args.base_stage, candidate_stage=args.candidate_stage)
    report = apply_evaluation_gate(
        report,
        mode=args.mode,
        expected_pairs=args.expected_pairs,
        minimum_baseline_success=args.minimum_baseline_success,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end="")
    return 0 if report["evaluation_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
