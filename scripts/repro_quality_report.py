#!/usr/bin/env python3
"""Aggregate paired closed-loop evaluation JSONL and apply quality gates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import pathlib
from typing import Any

import numpy as np


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
    return records


def summarize(records: list[dict[str, Any]], *, base_stage: str, candidate_stage: str) -> dict[str, Any]:
    indexed: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = (record["benchmark"], record["suite"], record["task"], str(record["pair_id"]))
        if record["stage"] in indexed[key]:
            raise ValueError(f"duplicate stage/pair record: {key} {record['stage']}")
        indexed[key][record["stage"]] = record

    pairs = [value for value in indexed.values() if base_stage in value and candidate_stage in value]
    if not pairs:
        raise ValueError("no complete baseline/candidate pairs")

    groups: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair in pairs:
        base = pair[base_stage]
        candidate = pair[candidate_stage]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=pathlib.Path)
    parser.add_argument("--base-stage", default="base")
    parser.add_argument("--candidate-stage", default="final")
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = summarize(load_records(args.records), base_stage=args.base_stage, candidate_stage=args.candidate_stage)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
