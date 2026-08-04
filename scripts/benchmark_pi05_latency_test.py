from __future__ import annotations

import sys

import pytest

from scripts import benchmark_pi05_latency
from scripts.benchmark_pi05_latency import _require_manifest_artifact

IMAGE = "sha256:" + "1" * 64
INSTANCE_ID = "i-0123456789abcdef0"


def _argv(*extra: str) -> list[str]:
    return [
        "benchmark",
        "--backend",
        "torch",
        "--stage",
        "base",
        "--artifact-dir",
        "/tmp/artifacts",
        "--track",
        "libero",
        "--dataset",
        "physical-intelligence/libero",
        "--dataset-revision",
        "revision",
        "--image-digest",
        IMAGE,
        "--instance-id",
        INSTANCE_ID,
        "--cost-reservation",
        "reservation",
        *extra,
    ]


def test_official_base_benchmark_cannot_use_one_denoising_step(monkeypatch):
    monkeypatch.setattr(sys, "argv", _argv("--num-denoise-steps", "1"))
    with pytest.raises(ValueError, match="requires 10 denoising steps"):
        benchmark_pi05_latency.main()


def test_tensorrt_cannot_claim_a_different_step_count_even_for_smoke(monkeypatch):
    argv = _argv("--num-denoise-steps", "10", "--allow-nonstandard-counts")
    argv[argv.index("torch")] = "tensorrt"
    argv[argv.index("base")] = "tensorrt_bf16"
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="exactly one"):
        benchmark_pi05_latency.main()


def test_benchmark_rejects_declared_image_before_loading_gpu_runtime(monkeypatch):
    monkeypatch.setattr(sys, "argv", _argv())
    monkeypatch.setenv("PI05_IMAGE_DIGEST", "sha256:" + "2" * 64)
    monkeypatch.setenv("PI05_INSTANCE_ID", INSTANCE_ID)
    monkeypatch.setenv("PI05_INSTANCE_TYPE", "g7e.4xlarge")
    with pytest.raises(ValueError, match="image digest differs"):
        benchmark_pi05_latency.main()


def test_build_manifest_artifact_check_is_relocatable_and_detects_changes(tmp_path):
    artifact = tmp_path / "decode-denoise.bf16.plan"
    artifact.write_bytes(b"engine")
    manifest = {
        "artifacts": [
            {
                "path": "/original/host/decode-denoise.bf16.plan",
                "bytes": 6,
                "sha256": "ed9f6f25068608efd412958da4dfc19328ca3511251fa6d5f9c42baf230e32f8",
            }
        ]
    }
    _require_manifest_artifact(manifest, artifact)
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="no longer matches"):
        _require_manifest_artifact(manifest, artifact)
