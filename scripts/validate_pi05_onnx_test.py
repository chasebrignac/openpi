from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from scripts import validate_pi05_onnx
from scripts.repro_make_action_limits import write_artifact
from scripts.validate_pi05_onnx import _merge_candidate_prefix
from scripts.validate_pi05_onnx import _run_graph

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


def test_cuda_graph_validation_disables_cpu_provider_fallback(tmp_path, monkeypatch):
    input_path = tmp_path / "inputs.npz"
    np.savez(input_path, input=np.ones((1, 2), dtype=np.float32))
    observed = {}

    class FakeSessionOptions:
        def add_session_config_entry(self, key, value):
            observed["session_config"] = (key, value)

    class FakeSession:
        def __init__(self, model_path, *, sess_options, providers):
            observed["model_path"] = model_path
            observed["session_options"] = sess_options
            observed["providers"] = providers

        def get_inputs(self):
            return [type("Input", (), {"name": "input", "type": "tensor(float)"})()]

        def get_providers(self):
            return ["CUDAExecutionProvider"]

    fake_ort = type(
        "FakeOrt",
        (),
        {
            "SessionOptions": FakeSessionOptions,
            "InferenceSession": FakeSession,
            "get_available_providers": staticmethod(lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]),
        },
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(
        validate_pi05_onnx,
        "_run_cuda_iobinding",
        lambda session, values: {"output": values["input"]},
    )

    outputs, providers = _run_graph(tmp_path / "model.onnx", input_path, "cuda")

    assert outputs["output"].shape == (1, 2)
    assert providers == ["CUDAExecutionProvider"]
    assert observed["providers"] == ["CUDAExecutionProvider"]
    assert observed["session_config"] == ("session.disable_cpu_ep_fallback", "1")


def test_cuda_graph_validation_rejects_an_active_cpu_provider(tmp_path, monkeypatch):
    input_path = tmp_path / "inputs.npz"
    np.savez(input_path, input=np.ones((1, 2), dtype=np.float32))

    class FakeSessionOptions:
        def add_session_config_entry(self, key, value):
            del key, value

    class FakeSession:
        def __init__(self, model_path, *, sess_options, providers):
            del model_path, sess_options, providers

        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    fake_ort = type(
        "FakeOrt",
        (),
        {
            "SessionOptions": FakeSessionOptions,
            "InferenceSession": FakeSession,
            "get_available_providers": staticmethod(lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]),
        },
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    with pytest.raises(RuntimeError, match="exclusively on CUDAExecutionProvider"):
        _run_graph(tmp_path / "model.onnx", input_path, "cuda")


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
