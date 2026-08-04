# ruff: noqa: PLC0415
"""Latency statistics and the fixed CUDA timing protocol."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import time
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class TimingSamples:
    cuda_event_ms: tuple[float, ...]
    wall_ms: tuple[float, ...]

    def report(self) -> dict[str, Any]:
        return {
            "samples": len(self.cuda_event_ms),
            "cuda_event_ms": summarize_ms(self.cuda_event_ms),
            "wall_ms": summarize_ms(self.wall_ms),
        }


def summarize_ms(samples: Sequence[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Timing samples must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Timing samples must be finite and non-negative")
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def benchmark_cuda(
    operation: Callable[[], Any],
    *,
    warmups: int,
    iterations: int,
) -> TimingSamples:
    """Measure one callable with paired CUDA-event and synchronized wall time."""

    import torch

    if warmups < 0 or iterations < 1:
        raise ValueError("warmups must be non-negative and iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA timing requires a CUDA device")
    with torch.inference_mode():
        for _ in range(warmups):
            operation()
        torch.cuda.synchronize()

        cuda_values: list[float] = []
        wall_values: list[float] = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(iterations):
            wall_start = time.perf_counter_ns()
            start.record()
            operation()
            end.record()
            end.synchronize()
            wall_end = time.perf_counter_ns()
            cuda_values.append(float(start.elapsed_time(end)))
            wall_values.append((wall_end - wall_start) / 1_000_000.0)
    return TimingSamples(tuple(cuda_values), tuple(wall_values))


def stage_speedups(reports: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Compute the reproduction's named mean-latency speedups."""

    def mean(stage: str, component: str) -> float:
        try:
            return float(reports[stage]["latency"][component]["cuda_event_ms"]["mean"])
        except KeyError as exc:
            raise ValueError(f"Missing {stage}.{component} latency") from exc

    result: dict[str, float] = {}
    if {"base", "shallow"}.issubset(reports):
        result["shallow_vs_base_total"] = mean("base", "total") / mean("shallow", "total")
    if {"shallow", "snapflow"}.issubset(reports):
        result["snapflow_vs_shallow_denoise"] = mean("shallow", "denoise") / mean("snapflow", "denoise")
        result["snapflow_vs_shallow_total"] = mean("shallow", "total") / mean("snapflow", "total")
    if {"snapflow", "tensorrt_bf16"}.issubset(reports):
        result["tensorrt_vs_eager_snapflow_total"] = mean("snapflow", "total") / mean("tensorrt_bf16", "total")
    if {"tensorrt_bf16", "tensorrt_fp8"}.issubset(reports):
        result["fp8_vs_bf16_tensorrt_total"] = mean("tensorrt_bf16", "total") / mean("tensorrt_fp8", "total")
    if {"base", "tensorrt_fp8"}.issubset(reports):
        result["cumulative_fp8_vs_base_total"] = mean("base", "total") / mean("tensorrt_fp8", "total")
    return result
