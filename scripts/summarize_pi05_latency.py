#!/usr/bin/env python3
"""Combine the five G7e latency reports and evaluate stagewise speedup gates."""

from __future__ import annotations

import argparse
import json
import pathlib

from openpi.exporting.benchmark import stage_speedups


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=pathlib.Path)
    parser.add_argument("--shallow", required=True, type=pathlib.Path)
    parser.add_argument("--snapflow", required=True, type=pathlib.Path)
    parser.add_argument("--tensorrt-bf16", required=True, type=pathlib.Path)
    parser.add_argument("--tensorrt-fp8", required=True, type=pathlib.Path)
    parser.add_argument("--reproduction-config", type=pathlib.Path, default=pathlib.Path("repro/reproduction.json"))
    parser.add_argument("--output", required=True, type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = {
        "base": args.base,
        "shallow": args.shallow,
        "snapflow": args.snapflow,
        "tensorrt_bf16": args.tensorrt_bf16,
        "tensorrt_fp8": args.tensorrt_fp8,
    }
    reports = {stage: json.loads(path.read_text()) for stage, path in paths.items()}
    for expected_stage, report in reports.items():
        if report.get("stage") != expected_stage:
            raise ValueError(f"Expected {expected_stage!r} report, got {report.get('stage')!r}")
    if any(not report.get("official_protocol") for report in reports.values()):
        raise ValueError("Every aggregate input must use the official 500-warmup/10,000-iteration protocol")
    instance_ids = {report.get("runtime", {}).get("instance_id") for report in reports.values()}
    instance_types = {report.get("runtime", {}).get("instance_type") for report in reports.values()}
    tracks = {report.get("track") for report in reports.values()}
    datasets = {json.dumps(report.get("dataset"), sort_keys=True) for report in reports.values()}
    benchmark_inputs = {json.dumps(report.get("benchmark_inputs"), sort_keys=True) for report in reports.values()}
    gpu_inventories = {tuple(report.get("gpu_inventory", ())) for report in reports.values()}
    if len(instance_ids) != 1 or None in instance_ids:
        raise ValueError(f"All latency stages must run on the same EC2 instance ID: {instance_ids}")
    if instance_types != {"g7e.4xlarge"}:
        raise ValueError(f"All latency stages must run on g7e.4xlarge: {instance_types}")
    if len(tracks) != 1 or None in tracks:
        raise ValueError(f"Latency tracks differ: {tracks}")
    if len(datasets) != 1 or "null" in datasets:
        raise ValueError("All latency stages must use the same dataset and revision")
    if len(benchmark_inputs) != 1 or "null" in benchmark_inputs:
        raise ValueError("All latency stages must use byte-identical fixed benchmark inputs")
    if len(gpu_inventories) != 1 or not next(iter(gpu_inventories)):
        raise ValueError("All latency stages must report the same non-empty GPU inventory")
    torch_images = {reports[stage].get("runtime", {}).get("image_digest") for stage in ("base", "shallow", "snapflow")}
    tensorrt_images = {
        reports[stage].get("runtime", {}).get("image_digest") for stage in ("tensorrt_bf16", "tensorrt_fp8")
    }
    if len(torch_images) != 1 or None in torch_images:
        raise ValueError(f"Eager stages must use one pinned runtime image: {torch_images}")
    if len(tensorrt_images) != 1 or None in tensorrt_images:
        raise ValueError(f"TensorRT stages must use one pinned runtime image: {tensorrt_images}")
    speedups = stage_speedups(reports)
    config = json.loads(args.reproduction_config.read_text())["benchmark"]
    thresholds = {
        "shallow_vs_base_total": float(config["shallow_speedup_min"]),
        "snapflow_vs_shallow_denoise": float(config["snapflow_denoise_speedup_min"]),
        "snapflow_vs_shallow_total": float(config["snapflow_total_speedup_min"]),
        "tensorrt_vs_eager_snapflow_total": float(config["tensorrt_speedup_min"]),
        "fp8_vs_bf16_tensorrt_total": float(config["fp8_speedup_min"]),
        "cumulative_fp8_vs_base_total": float(config["cumulative_aws_speedup_min"]),
    }
    gates = {
        name: {"observed": speedups[name], "minimum": minimum, "passes": speedups[name] >= minimum}
        for name, minimum in thresholds.items()
    }
    payload = {
        "schema_version": 1,
        "scope": "AWS g7e.4xlarge relative latency; not a Jetson Thor claim",
        "speedups": speedups,
        "gates": gates,
        "passes": all(gate["passes"] for gate in gates.values()),
        "source_reports": {stage: str(path.resolve()) for stage, path in paths.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
