#!/usr/bin/env python3
"""Create a reproducible run manifest and hash the produced artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import shlex
import subprocess
from typing import Any


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def sha256_file(path: pathlib.Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: pathlib.Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def optional_cost(value: float | None, *, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return float(value)


def parse_metrics(raw: str) -> dict[str, Any]:
    try:
        metrics = json.loads(raw)
        # Round-trip with strict JSON so NaN/Infinity can never enter durable
        # run evidence through Python's permissive decoder.
        json.dumps(metrics, allow_nan=False)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("metrics JSON must be a finite JSON object") from exc
    if not isinstance(metrics, dict):
        raise ValueError("metrics JSON must be a finite JSON object")
    return metrics


def create_manifest(args: argparse.Namespace) -> dict[str, Any]:
    artifacts = [artifact_record(path) for path in args.artifact]
    projected_cost = optional_cost(getattr(args, "projected_cost_usd", None), label="projected cost")
    actual_cost = optional_cost(args.actual_cost_usd, label="actual cost")
    return {
        "schema_version": 1,
        "run_id": args.run_id,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "source": {
            "git_sha": git_output("rev-parse", "HEAD"),
            "dirty": bool(git_output("status", "--porcelain")),
        },
        "runtime": {
            "image": args.image,
            "image_digest": args.image_digest,
            "instance_type": args.instance_type,
            "instance_id": args.instance_id,
            "region": args.region,
        },
        "dataset": {"name": args.dataset, "revision": args.dataset_revision},
        "experiment": {
            "config": args.training_config,
            "seed": args.seed,
            "steps": args.steps,
            "command_argv": shlex.split(args.command),
            "command": args.command,
        },
        "cost": {
            "reservation_id": args.cost_reservation,
            "projected_usd": projected_cost,
            "actual_usd": actual_cost,
        },
        "metrics": parse_metrics(args.metrics_json),
        "artifacts": artifacts,
        "environment": {"aws_execution_env": os.environ.get("AWS_EXECUTION_ENV")},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--instance-type", required=True)
    parser.add_argument("--instance-id", default="unknown")
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--cost-reservation", required=True)
    parser.add_argument("--projected-cost-usd", type=float)
    parser.add_argument("--actual-cost-usd", type=float)
    parser.add_argument("--metrics-json", default="{}")
    parser.add_argument("--artifact", type=pathlib.Path, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = create_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
