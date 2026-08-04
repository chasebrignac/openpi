from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib

import numpy as np
import pytest

from openpi.exporting import tensorrt_policy as tp

RUNTIME = tp.RuntimeIdentity(
    image_digest="sha256:" + "1" * 64,
    instance_type="g7e.4xlarge",
    instance_id="i-build",
    gpu_inventory=("GPU-11111111-1111-1111-1111-111111111111, NVIDIA L40S, 595.45",),
    tensorrt_version="11.0.0.114",
)


def _record(path: pathlib.Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _artifact_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    precision = "fp8"
    prefix = tmp_path / f"encode-prefix.{precision}.plan"
    decoder = tmp_path / f"decode-denoise.{precision}.plan"
    validation = tmp_path / f"onnx-validation.{precision}.json"
    export = tmp_path / "export-manifest.json"
    fp8 = tmp_path / "fp8-manifest.json"
    bf16_validation = tmp_path / "onnx-validation.bf16.json"
    for path in (prefix, decoder):
        path.write_bytes(path.name.encode())
    for path in (export, fp8, bf16_validation):
        path.write_text(json.dumps({"name": path.name}))
    validation.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "precision": precision,
                "passes": True,
                "provenance": {
                    "track": "libero",
                    "dataset": "dataset",
                    "dataset_revision": "revision",
                    **RUNTIME.manifest_runtime,
                },
                "end_to_end_actions": {
                    "bias_passes": True,
                    "action_limits_pass": True,
                    "action_gate_kind": "corpus_envelope_not_hardware_safety",
                },
            }
        )
    )
    sources = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in (export, fp8, bf16_validation)
    ]
    manifest = {
        "schema_version": 1,
        "stage": "tensorrt-build-fp8",
        "track": "libero",
        "source": {"sha": "a" * 40, "dirty": False},
        "dataset": {"name": "dataset", "revision": "revision"},
        "runtime": RUNTIME.manifest_runtime,
        "artifacts": [_record(path) for path in (prefix, decoder, validation, export, fp8, bf16_validation)],
        "details": {
            "strongly_typed": True,
            "precision_source": "explicit ONNX tensor types and Q/DQ nodes",
            "gpu_inventory": list(RUNTIME.gpu_inventory),
            "tensorrt_version": "TensorRT v110000",
            "validation_report": str(validation),
            "policy_contract": {
                "schema_version": 1,
                "protocol": "openpi-policy-websocket-v1",
                "config": "pi05_libero_l09_snapflow",
                "checkpoint": {"path": "/source/model.safetensors", "sha256": "b" * 64, "assets": [{}]},
                "precision": "fp8",
                "num_denoise_steps": 1,
                "source_manifests": sources,
                "export_runtime": RUNTIME.manifest_runtime,
            },
        },
    }
    (tmp_path / "tensorrt-manifest.fp8.json").write_text(json.dumps(manifest))
    return tmp_path


def test_artifact_bundle_verifies_runtime_hashes_and_validation(tmp_path: pathlib.Path):
    root = _artifact_dir(tmp_path)

    bundle = tp.load_artifact_bundle(
        root,
        precision="fp8",
        track="libero",
        dataset="dataset",
        dataset_revision="revision",
        runtime=RUNTIME,
    )

    assert bundle.config == "pi05_libero_l09_snapflow"
    assert bundle.prefix_plan.name == "encode-prefix.fp8.plan"


@pytest.mark.parametrize("mutation", ["plan", "gpu", "instance", "validation", "export_image"])
def test_artifact_bundle_fails_closed_on_identity_drift(tmp_path: pathlib.Path, mutation: str):
    root = _artifact_dir(tmp_path)
    runtime = RUNTIME
    if mutation == "plan":
        (root / "encode-prefix.fp8.plan").write_bytes(b"tampered")
    elif mutation == "gpu":
        runtime = dataclasses.replace(RUNTIME, gpu_inventory=("different",))
    elif mutation == "instance":
        runtime = dataclasses.replace(RUNTIME, instance_id="i-other")
    elif mutation == "validation":
        validation = root / "onnx-validation.fp8.json"
        payload = json.loads(validation.read_text())
        payload["passes"] = False
        validation.write_text(json.dumps(payload))
    elif mutation == "export_image":
        manifest_path = root / "tensorrt-manifest.fp8.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["details"]["policy_contract"]["export_runtime"]["image_digest"] = "sha256:" + "2" * 64
        manifest_path.write_text(json.dumps(manifest))

    with pytest.raises((ValueError, RuntimeError)):
        tp.load_artifact_bundle(
            root,
            precision="fp8",
            track="libero",
            dataset="dataset",
            dataset_revision="revision",
            runtime=runtime,
        )


class FakeEngine:
    def __init__(self, inputs: dict[str, tp.TensorSpec], outputs: dict[str, tp.TensorSpec], result: dict):
        self.inputs = inputs
        self.outputs = outputs
        self.result = result
        self.calls = []

    def execute(self, values):
        assert set(values) == set(self.inputs)
        self.calls.append(values)
        return self.result


def _pipeline() -> tuple[tp.TensorRTPipeline, FakeEngine, FakeEngine]:
    f32 = "float32"
    prefix_inputs = {
        **{f"image_{index}": tp.TensorSpec((1, 3, 224, 224), f32) for index in range(3)},
        **{f"image_mask_{index}": tp.TensorSpec((1,), "bool") for index in range(3)},
        "lang_tokens": tp.TensorSpec((1, 4), "int64"),
        "lang_mask": tp.TensorSpec((1, 4), "bool"),
    }
    boundary = {
        "prefix_pad_masks": tp.TensorSpec((1, 2), "bool"),
        "cache_key_00": tp.TensorSpec((1, 1, 2, 2), "bfloat16"),
        "cache_value_00": tp.TensorSpec((1, 1, 2, 2), "bfloat16"),
    }
    prefix_result = {name: np.ones(spec.shape) for name, spec in boundary.items()}
    prefix = FakeEngine(prefix_inputs, boundary, prefix_result)
    decoder_inputs = {
        "state": tp.TensorSpec((1, 3), f32),
        "x_t": tp.TensorSpec((1, 2, 3), f32),
        "timestep": tp.TensorSpec((1,), f32),
        "target_time": tp.TensorSpec((1,), f32),
        **boundary,
    }
    decoder = FakeEngine(
        decoder_inputs,
        {"actions": tp.TensorSpec((1, 2, 3), f32)},
        {"actions": np.full((1, 2, 3), 0.25, dtype=np.float32)},
    )
    return tp.TensorRTPipeline(prefix, decoder, shape=tp.PolicyShape(2, 3, 4)), prefix, decoder


def test_policy_preserves_transforms_noise_and_split_engine_boundary():
    pipeline, prefix, decoder = _pipeline()
    policy = tp.TensorRTPolicy(
        pipeline,
        input_transform=lambda value: value,
        output_transform=lambda value: {"actions": value["actions"][:, :2]},
        metadata={"reset_pose": [0]},
    )
    obs = {
        "image": {name: np.zeros((224, 224, 3), dtype=np.uint8) for name in tp.IMAGE_KEYS},
        "image_mask": dict.fromkeys(tp.IMAGE_KEYS, np.True_),
        "tokenized_prompt": np.arange(4),
        "tokenized_prompt_mask": np.ones(4, dtype=bool),
        "state": np.arange(3, dtype=np.float32),
    }
    noise = np.ones((2, 3), dtype=np.float32)

    result = policy.infer(obs, noise=noise)

    assert result["actions"].shape == (2, 2)
    assert "policy_timing" in result
    assert len(prefix.calls) == len(decoder.calls) == 1
    assert decoder.calls[0]["cache_key_00"] is prefix.result["cache_key_00"]
    np.testing.assert_array_equal(decoder.calls[0]["x_t"], noise[None, ...])
    assert policy.metadata == {"reset_pose": [0]}


def test_pipeline_rejects_wrong_action_shape():
    pipeline, _, decoder = _pipeline()
    decoder.outputs["actions"] = tp.TensorSpec((1, 3, 3), "float32")
    with pytest.raises(ValueError, match="action shape"):
        tp.TensorRTPipeline(pipeline.prefix, decoder, shape=tp.PolicyShape(2, 3, 4))
