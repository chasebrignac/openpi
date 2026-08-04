from __future__ import annotations

import json
import sys

import pytest

from scripts import summarize_pi05_latency


def _report(stage, total, denoise):
    component = lambda value: {"cuda_event_ms": {"mean": value}}  # noqa: E731
    return {
        "stage": stage,
        "track": "libero",
        "official_protocol": True,
        "runtime": {
            "instance_id": "i-test",
            "instance_type": "g7e.4xlarge",
            "image_digest": "sha256:torch" if stage in {"base", "shallow", "snapflow"} else "sha256:trt",
        },
        "dataset": {"name": "physical-intelligence/libero", "revision": "revision"},
        "benchmark_inputs": {
            "encode": {"name": "encode-inputs.npz", "bytes": 1, "sha256": "encode"},
            "decode": {"name": "decode-inputs.npz", "bytes": 1, "sha256": "decode"},
        },
        "gpu_inventory": ["GPU-test, NVIDIA test, driver"],
        "latency": {"prefix": component(1), "denoise": component(denoise), "total": component(total)},
    }


def test_summary_applies_all_acceptance_gates(tmp_path, monkeypatch):
    values = {
        "base": _report("base", 120, 100),
        "shallow": _report("shallow", 60, 50),
        "snapflow": _report("snapflow", 20, 5),
        "tensorrt_bf16": _report("tensorrt_bf16", 15, 4),
        "tensorrt_fp8": _report("tensorrt_fp8", 10, 2),
    }
    paths = {}
    for name, value in values.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(value))
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "benchmark": {
                    "shallow_speedup_min": 1.7,
                    "snapflow_denoise_speedup_min": 8.0,
                    "snapflow_total_speedup_min": 1.8,
                    "tensorrt_speedup_min": 1.1,
                    "fp8_speedup_min": 1.25,
                    "cumulative_aws_speedup_min": 6.0,
                }
            }
        )
    )
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize",
            "--base",
            str(paths["base"]),
            "--shallow",
            str(paths["shallow"]),
            "--snapflow",
            str(paths["snapflow"]),
            "--tensorrt-bf16",
            str(paths["tensorrt_bf16"]),
            "--tensorrt-fp8",
            str(paths["tensorrt_fp8"]),
            "--reproduction-config",
            str(config),
            "--output",
            str(output),
        ],
    )
    assert summarize_pi05_latency.main() == 0
    assert json.loads(output.read_text())["passes"]


def test_summary_rejects_different_fixed_inputs(tmp_path, monkeypatch):
    values = {
        "base": _report("base", 120, 100),
        "shallow": _report("shallow", 60, 50),
        "snapflow": _report("snapflow", 20, 5),
        "tensorrt_bf16": _report("tensorrt_bf16", 15, 4),
        "tensorrt_fp8": _report("tensorrt_fp8", 10, 2),
    }
    values["tensorrt_fp8"]["benchmark_inputs"]["decode"]["sha256"] = "different"
    paths = {}
    for name, value in values.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(value))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"benchmark": {}}))
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize",
            "--base",
            str(paths["base"]),
            "--shallow",
            str(paths["shallow"]),
            "--snapflow",
            str(paths["snapflow"]),
            "--tensorrt-bf16",
            str(paths["tensorrt_bf16"]),
            "--tensorrt-fp8",
            str(paths["tensorrt_fp8"]),
            "--reproduction-config",
            str(config),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="byte-identical"):
        summarize_pi05_latency.main()
