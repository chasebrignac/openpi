from __future__ import annotations

import pathlib
import sys

import pytest

from scripts import export_pi05_onnx
from scripts.export_pi05_onnx import _checkpoint_assets

IMAGE = "sha256:" + "1" * 64
INSTANCE_ID = "i-0123456789abcdef0"


def test_checkpoint_assets_are_complete_and_relocatable(tmp_path: pathlib.Path):
    weights = tmp_path / "model.safetensors"
    norm_stats = tmp_path / "assets" / "physical-intelligence" / "libero" / "norm_stats.json"
    metadata = norm_stats.parent / "metadata.json"
    norm_stats.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    norm_stats.write_text("{}")
    metadata.write_text("{}")

    root, assets = _checkpoint_assets(weights)

    assert root == tmp_path.resolve()
    assert [path.relative_to(root).as_posix() for path in assets] == [
        "assets/physical-intelligence/libero/metadata.json",
        "assets/physical-intelligence/libero/norm_stats.json",
    ]


def test_checkpoint_assets_require_exactly_one_norm_stats_file(tmp_path: pathlib.Path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    (tmp_path / "assets").mkdir()

    with pytest.raises(FileNotFoundError, match="is empty"):
        _checkpoint_assets(weights)

    for asset_id in ("a", "b"):
        path = tmp_path / "assets" / asset_id / "norm_stats.json"
        path.parent.mkdir()
        path.write_text("{}")
    with pytest.raises(ValueError, match="exactly one normalization asset"):
        _checkpoint_assets(weights)


def test_checkpoint_assets_reject_symlinks(tmp_path: pathlib.Path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    real = tmp_path / "real.json"
    real.write_text("{}")
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "norm_stats.json").symlink_to(real)

    with pytest.raises(ValueError, match="must not contain symlinks"):
        _checkpoint_assets(weights)


def test_export_rejects_a_declared_image_that_differs_from_worker(monkeypatch, tmp_path):
    monkeypatch.setenv("PI05_IMAGE_DIGEST", "sha256:" + "2" * 64)
    monkeypatch.setenv("PI05_INSTANCE_ID", INSTANCE_ID)
    monkeypatch.setenv("PI05_INSTANCE_TYPE", "g7e.4xlarge")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export",
            "--config",
            "pi05_libero_l09_snapflow",
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--calibration-manifest",
            str(tmp_path / "calibration.jsonl"),
            "--output-dir",
            str(tmp_path / "artifacts"),
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
        export_pi05_onnx.main()
