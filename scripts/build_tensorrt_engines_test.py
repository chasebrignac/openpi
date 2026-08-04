from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

import pytest

from scripts import build_tensorrt_engines
from scripts.build_tensorrt_engines import _command
from scripts.build_tensorrt_engines import _major_version
from scripts.build_tensorrt_engines import _parse_major_version
from scripts.build_tensorrt_engines import _require_pinned_major
from scripts.build_tensorrt_engines import _validate_policy_provenance

RUNTIME = {
    "image_digest": "sha256:" + "1" * 64,
    "instance_type": "g7e.4xlarge",
    "instance_id": "i-build",
}


def _identity(path: pathlib.Path) -> dict:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _model_identity(model: pathlib.Path, weights: pathlib.Path) -> dict:
    return {
        "model": {"name": model.name, **{key: value for key, value in _identity(model).items() if key != "path"}},
        "external_data": [
            {"name": weights.name, **{key: value for key, value in _identity(weights).items() if key != "path"}}
        ],
        "opsets": {"ai.onnx": 20},
        "inputs": {},
        "outputs": {},
    }


def _stage(stage: str, track: str, artifacts: list[pathlib.Path], *, details: dict | None = None) -> dict:
    return {
        "schema_version": 1,
        "stage": stage,
        "track": track,
        "source": {"sha": "a" * 40, "dirty": False},
        "dataset": {"name": "dataset", "revision": "revision"},
        "runtime": RUNTIME,
        "artifacts": [_identity(path) for path in artifacts],
        "details": details or {},
    }


def _validation(precision: str, models: dict) -> dict:
    if set(models) == {"encode-prefix"}:
        models = {**models, "decode-denoise": models["encode-prefix"]}
    return {
        "schema_version": 1,
        "precision": precision,
        "passes": True,
        "models": models,
        "provenance": {
            "track": "libero",
            "dataset": "dataset",
            "dataset_revision": "revision",
            **RUNTIME,
        },
    }


def _checkpoint(tmp_path: pathlib.Path) -> tuple[dict, list[pathlib.Path]]:
    weights = tmp_path / "model.safetensors"
    norm_stats = tmp_path / "norm_stats.json"
    weights.write_bytes(b"checkpoint")
    norm_stats.write_text("{}")
    return (
        {
            "path": str(weights),
            "sha256": _identity(weights)["sha256"],
            "assets": [
                {
                    "name": "assets/physical-intelligence/libero/norm_stats.json",
                    "bytes": _identity(norm_stats)["bytes"],
                    "sha256": _identity(norm_stats)["sha256"],
                }
            ],
        },
        [weights, norm_stats],
    )


def test_tensorrt_11_command_uses_graph_precision_without_removed_flags():
    command = _command("trtexec", "m.onnx", "m.plan", "t.cache", "l.json", 1024, 11)
    assert "--stronglyTyped" not in command
    assert not any(flag in command for flag in ("--bf16", "--fp8", "--fp16"))


def test_tensorrt_10_command_requests_strong_typing():
    command = _command("trtexec", "m.onnx", "m.plan", "t.cache", "l.json", 1024, 10)
    assert "--stronglyTyped" in command


def test_parses_dotted_and_compact_tensorrt_versions():
    assert _parse_major_version("TensorRT version: 11.1.0") == 11
    assert _parse_major_version("[TensorRT v100900]") == 10
    assert _parse_major_version("[TensorRT v110000]") == 11


def test_runtime_version_probe_uses_successful_help_banner(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="[TensorRT v110000] [b114]", stderr="")

    monkeypatch.setattr("scripts.build_tensorrt_engines.subprocess.run", fake_run)

    assert _major_version("/opt/tensorrt/bin/trtexec") == (11, "[TensorRT v110000] [b114]")
    assert calls == [
        (
            ["/opt/tensorrt/bin/trtexec", "--help"],
            {"check": True, "text": True, "capture_output": True},
        )
    ]


def test_build_runtime_is_pinned_to_tensorrt_11():
    _require_pinned_major(11)
    with pytest.raises(RuntimeError, match="pinned to TensorRT 11"):
        _require_pinned_major(10)


def test_build_rejects_export_from_another_image_before_reading_artifacts(monkeypatch, tmp_path):
    image = "sha256:" + "1" * 64
    instance_id = "i-0123456789abcdef0"
    monkeypatch.setenv("PI05_IMAGE_DIGEST", image)
    monkeypatch.setenv("PI05_INSTANCE_ID", instance_id)
    monkeypatch.setenv("PI05_INSTANCE_TYPE", "g7e.4xlarge")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build",
            "--artifact-dir",
            str(tmp_path),
            "--precision",
            "bf16",
            "--validation-report",
            str(tmp_path / "onnx-validation.bf16.json"),
            "--export-manifest",
            str(tmp_path / "export-manifest.json"),
            "--export-image-digest",
            "sha256:" + "2" * 64,
            "--track",
            "libero",
            "--dataset",
            "physical-intelligence/libero",
            "--dataset-revision",
            "revision",
            "--image-digest",
            image,
            "--instance-id",
            instance_id,
            "--cost-reservation",
            "reservation",
        ],
    )

    with pytest.raises(ValueError, match="must use one image digest"):
        build_tensorrt_engines.main()


def test_policy_provenance_binds_export_config_checkpoint_and_models(tmp_path):
    model = tmp_path / "encode-prefix.bf16.onnx"
    weights = tmp_path / "encode-prefix.bf16.onnx.data"
    model.write_bytes(b"model")
    weights.write_bytes(b"weights")
    models = {"encode-prefix": _model_identity(model, weights)}
    checkpoint, checkpoint_artifacts = _checkpoint(tmp_path)
    export = _stage(
        "onnx-export-bf16",
        "libero",
        [model, weights, *checkpoint_artifacts],
        details={
            "config": "pi05_libero_l09_snapflow",
            "checkpoint": checkpoint,
        },
    )
    export_path = tmp_path / "export-manifest.json"
    export_path.write_text(json.dumps(export))

    contract, sources = _validate_policy_provenance(
        precision="bf16",
        export_manifest_path=export_path,
        fp8_manifest_path=None,
        validation=_validation("bf16", models),
        artifact_dir=tmp_path,
        track="libero",
        dataset="dataset",
        dataset_revision="revision",
        runtime=RUNTIME,
        export_runtime=RUNTIME,
    )

    assert contract["config"] == "pi05_libero_l09_snapflow"
    assert contract["checkpoint"]["sha256"] == checkpoint["sha256"]
    assert contract["source_manifests"][0]["sha256"] == hashlib.sha256(export_path.read_bytes()).hexdigest()
    assert sources == [export_path]


@pytest.mark.parametrize("mutation", ["config", "checkpoint", "dirty", "model"])
def test_policy_provenance_rejects_broken_export_chain(tmp_path, mutation):
    model = tmp_path / "encode-prefix.bf16.onnx"
    weights = tmp_path / "encode-prefix.bf16.onnx.data"
    model.write_bytes(b"model")
    weights.write_bytes(b"weights")
    models = {"encode-prefix": _model_identity(model, weights)}
    checkpoint, checkpoint_artifacts = _checkpoint(tmp_path)
    export = _stage(
        "onnx-export-bf16",
        "libero",
        [model, weights, *checkpoint_artifacts],
        details={
            "config": "pi05_libero_l09_snapflow",
            "checkpoint": checkpoint,
        },
    )
    if mutation == "config":
        export["details"]["config"] = "pi05_libero"
    elif mutation == "checkpoint":
        export["details"]["checkpoint"]["sha256"] = "not-a-digest"
    elif mutation == "dirty":
        export["source"]["dirty"] = True
    elif mutation == "model":
        export["artifacts"][0]["sha256"] = "c" * 64
    export_path = tmp_path / "export-manifest.json"
    export_path.write_text(json.dumps(export))

    expected = {
        "config": "wrong one-step policy config",
        "checkpoint": "checkpoint identity is malformed",
        "dirty": "clean source tree",
        "model": "identity does not match",
    }
    with pytest.raises(ValueError, match=expected[mutation]):
        _validate_policy_provenance(
            precision="bf16",
            export_manifest_path=export_path,
            fp8_manifest_path=None,
            validation=_validation("bf16", models),
            artifact_dir=tmp_path,
            track="libero",
            dataset="dataset",
            dataset_revision="revision",
            runtime=RUNTIME,
            export_runtime=RUNTIME,
        )


def test_fp8_policy_provenance_requires_quantization_and_bf16_validation_chain(tmp_path):
    bf16_model = tmp_path / "encode-prefix.bf16.onnx"
    bf16_weights = tmp_path / "encode-prefix.bf16.onnx.data"
    fp8_model = tmp_path / "encode-prefix.fp8.onnx"
    fp8_weights = tmp_path / "encode-prefix.fp8.onnx.data"
    for path in (bf16_model, bf16_weights, fp8_model, fp8_weights):
        path.write_bytes(path.name.encode())
    bf16_models = {"encode-prefix": _model_identity(bf16_model, bf16_weights)}
    fp8_models = {"encode-prefix": _model_identity(fp8_model, fp8_weights)}
    checkpoint, checkpoint_artifacts = _checkpoint(tmp_path)
    export_path = tmp_path / "export-manifest.json"
    export_path.write_text(
        json.dumps(
            _stage(
                "onnx-export-bf16",
                "libero",
                [bf16_model, bf16_weights, *checkpoint_artifacts],
                details={
                    "config": "pi05_libero_l09_snapflow",
                    "checkpoint": checkpoint,
                },
            )
        )
    )
    bf16_validation_path = tmp_path / "onnx-validation.bf16.json"
    bf16_validation_path.write_text(json.dumps(_validation("bf16", bf16_models)))
    fp8_manifest_path = tmp_path / "fp8-manifest.json"
    fp8_manifest_path.write_text(
        json.dumps(
            _stage(
                "modelopt-fp8-ptq",
                "libero",
                [fp8_model, fp8_weights, bf16_validation_path],
                details={"bf16_validation_report": str(bf16_validation_path)},
            )
        )
    )

    contract, sources = _validate_policy_provenance(
        precision="fp8",
        export_manifest_path=export_path,
        fp8_manifest_path=fp8_manifest_path,
        validation=_validation("fp8", fp8_models),
        artifact_dir=tmp_path,
        track="libero",
        dataset="dataset",
        dataset_revision="revision",
        runtime=RUNTIME,
        export_runtime=RUNTIME,
    )

    assert contract["precision"] == "fp8"
    assert sources == [export_path, fp8_manifest_path, bf16_validation_path]
