from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from scripts import validate_pi05_onnx
from scripts.repro_make_action_limits import write_artifact
from scripts.validate_pi05_onnx import _merge_candidate_prefix

IMAGE = "sha256:" + "1" * 64
INSTANCE_ID = "i-0123456789abcdef0"


def test_end_to_end_validation_replaces_saved_prefix_boundary():
    saved = {
        "state": np.array([[1.0]]),
        "x_t": np.array([[[2.0]]]),
        "timestep": np.array([1.0]),
        "target_time": np.array([0.0]),
        "prefix_pad_masks": np.array([[True]]),
        "cache_key_00": np.array([1.0]),
        "cache_value_00": np.array([2.0]),
    }
    candidate = {
        "prefix_pad_masks": np.array([[False]]),
        "cache_key_00": np.array([3.0]),
        "cache_value_00": np.array([4.0]),
    }
    merged = _merge_candidate_prefix(saved, candidate)
    assert merged["state"] is saved["state"]
    assert merged["cache_key_00"] is candidate["cache_key_00"]
    assert merged["prefix_pad_masks"] is candidate["prefix_pad_masks"]


def test_end_to_end_validation_requires_exact_cache_interface():
    with pytest.raises(ValueError, match="boundary differs"):
        _merge_candidate_prefix(
            {"prefix_pad_masks": np.ones((1, 1)), "cache_key_00": np.ones(1)},
            {"prefix_pad_masks": np.ones((1, 1))},
        )


def test_overall_gate_includes_compounded_prefix_decoder_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PI05_IMAGE_DIGEST", IMAGE)
    monkeypatch.setenv("PI05_INSTANCE_ID", INSTANCE_ID)
    monkeypatch.setenv("PI05_INSTANCE_TYPE", "g7e.4xlarge")
    monkeypatch.setattr("openpi.exporting.artifacts.require_clean_source_identity", lambda: {"sha": "a" * 40})
    for name in ("encode-prefix.bf16.onnx", "decode-denoise.bf16.onnx"):
        (tmp_path / name).write_bytes(b"onnx")
    np.savez(
        tmp_path / "encode-reference.npz",
        prefix_pad_masks=np.array([[True]]),
        cache_key_00=np.array([1.0]),
        cache_value_00=np.array([1.0]),
    )
    np.savez(tmp_path / "decode-reference.npz", actions=np.ones((1, 1, 7)))
    np.savez(
        tmp_path / "decode-inputs.npz",
        state=np.array([[0.0]]),
        x_t=np.array([[[0.0]]]),
        timestep=np.array([1.0]),
        target_time=np.array([0.0]),
        prefix_pad_masks=np.array([[True]]),
        cache_key_00=np.array([1.0]),
        cache_value_00=np.array([1.0]),
    )
    (tmp_path / "encode-inputs.npz").write_bytes(b"unused by fake runner")
    limits_path = tmp_path / "action-limits.npz"
    write_artifact(
        limits_path,
        arrays={
            "action_low": np.full(7, -2.0),
            "action_high": np.full(7, 2.0),
            "action_mask": np.ones(7, dtype=bool),
            "physical_low": np.full(7, -2.0),
            "physical_high": np.full(7, 2.0),
            "physical_mask": np.ones(7, dtype=bool),
            "physical_state_dependent_mask": np.zeros(7, dtype=bool),
        },
        metadata={
            "schema_version": 1,
            "gate_kind": "corpus_envelope",
            "hardware_safety_guarantee": False,
            "track": "libero",
            "dataset": "physical-intelligence/libero",
            "dataset_revision": "revision",
            "sources": {
                "calibration_manifest": {"sha256": "a" * 64},
                "golden_corpus": {"sha256": "b" * 64},
                "norm_stats": {"sha256": "c" * 64},
            },
        },
    )
    prefix_candidate = {
        "prefix_pad_masks": np.array([[True]]),
        "cache_key_00": np.array([1.0001]),
        "cache_value_00": np.array([1.0]),
    }
    calls = 0

    def fake_run(model_path, input_path, provider, *, input_overrides=None):
        nonlocal calls
        del model_path, input_path, provider
        calls += 1
        if calls == 1:
            return prefix_candidate, ["CPUExecutionProvider"]
        if calls == 2:
            assert input_overrides is None
            return {"actions": np.ones((1, 1, 7))}, ["CPUExecutionProvider"]
        assert input_overrides["cache_key_00"] is prefix_candidate["cache_key_00"]
        return {"actions": -np.ones((1, 1, 7))}, ["CPUExecutionProvider"]

    monkeypatch.setattr(validate_pi05_onnx, "_run_graph", fake_run)
    monkeypatch.setattr(validate_pi05_onnx, "_external_onnx_data", lambda path: [])
    monkeypatch.setattr(
        "openpi.exporting.onnx_artifacts.onnx_model_identity",
        lambda path: {"model": {"name": path.name, "bytes": 4, "sha256": "test"}},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate",
            "--artifact-dir",
            str(tmp_path),
            "--precision",
            "bf16",
            "--cosine-threshold",
            "0.999",
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
            "--provider",
            "cpu",
            "--action-limits-npz",
            str(limits_path),
        ],
    )
    assert validate_pi05_onnx.main() == 2
    report = json.loads((tmp_path / "onnx-validation.bf16.json").read_text())
    assert report["decoder_isolated"]["passes"]
    assert not report["end_to_end"]["passes"]
    assert not report["passes"]
