from __future__ import annotations

import pytest

from openpi.exporting.benchmark import stage_speedups
from openpi.exporting.benchmark import summarize_ms


def _report(prefix, denoise, total):
    def component(value):
        return {"cuda_event_ms": {"mean": value}}

    return {"latency": {"prefix": component(prefix), "denoise": component(denoise), "total": component(total)}}


def test_summary_has_required_percentiles():
    result = summarize_ms([1.0, 2.0, 3.0, 4.0])
    assert result["mean"] == 2.5
    assert result["p50"] == 2.5
    assert result["p95"] == pytest.approx(3.85)
    assert result["p99"] == pytest.approx(3.97)


def test_stage_speedups_use_cuda_event_means():
    reports = {
        "base": _report(4, 96, 100),
        "shallow": _report(3, 47, 50),
        "snapflow": _report(3, 2, 5),
        "tensorrt_bf16": _report(2, 2, 4),
        "tensorrt_fp8": _report(1, 1, 2),
    }
    result = stage_speedups(reports)
    assert result == {
        "shallow_vs_base_total": 2.0,
        "snapflow_vs_shallow_denoise": 23.5,
        "snapflow_vs_shallow_total": 10.0,
        "tensorrt_vs_eager_snapflow_total": 1.25,
        "fp8_vs_bf16_tensorrt_total": 2.0,
        "cumulative_fp8_vs_base_total": 50.0,
    }


def test_summary_rejects_invalid_values():
    with pytest.raises(ValueError, match="non-empty"):
        summarize_ms([])
    with pytest.raises(ValueError, match="non-negative"):
        summarize_ms([-1.0])
