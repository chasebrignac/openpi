from __future__ import annotations

import sys

import pytest

from scripts import quantize_pi05_fp8

IMAGE = "sha256:" + "1" * 64
INSTANCE_ID = "i-0123456789abcdef0"


def test_fp8_rejects_cross_image_bf16_inputs_before_calibration(monkeypatch, tmp_path):
    monkeypatch.setenv("PI05_IMAGE_DIGEST", "sha256:" + "2" * 64)
    monkeypatch.setenv("PI05_INSTANCE_ID", INSTANCE_ID)
    monkeypatch.setenv("PI05_INSTANCE_TYPE", "g7e.4xlarge")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quantize",
            "--artifact-dir",
            str(tmp_path / "artifacts"),
            "--calibration-manifest",
            str(tmp_path / "calibration.jsonl"),
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
        ],
    )
    with pytest.raises(ValueError, match="image digest differs"):
        quantize_pi05_fp8.main()
