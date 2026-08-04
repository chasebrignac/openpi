import base64
import hashlib
import json
import math
import pathlib

import numpy as np
import pytest

from scripts import repro_stage_checkpoints
from scripts import repro_stage_converted_checkpoints
from scripts import repro_stage_data
from scripts import repro_stage_equivalence_evidence as evidence

SOURCE_COMMIT = "a" * 40
IMAGE_DIGEST = "sha256:" + "b" * 64


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fixture(tmp_path: pathlib.Path, *, track_name: str = "libero"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    track = evidence.TRACKS[track_name]
    golden = tmp_path / "golden.npz"
    sidecar_path = tmp_path / "golden.json"
    report_path = tmp_path / "framework-equivalence.json"
    velocity_path = tmp_path / "framework-equivalence.npz"
    source_path = tmp_path / "source-manifest.json"
    converted_path = tmp_path / "converted-manifest.json"

    rng = np.random.default_rng(7)
    arrays = {
        "state": rng.standard_normal((64, 8), dtype=np.float32),
        "tokenized_prompt": np.ones((64, 5), dtype=np.int32),
        "tokenized_prompt_mask": np.ones((64, 5), dtype=np.bool_),
        "actions": rng.standard_normal((64, 4, 8), dtype=np.float32),
        "noise": rng.standard_normal((64, 4, 8), dtype=np.float32),
        "time": np.linspace(0.01, 1.0, 64, dtype=np.float32),
        "image__base_0_rgb": np.zeros((64, 3, 4, 4), dtype=np.uint8),
        "image_mask__base_0_rgb": np.ones(64, dtype=np.bool_),
    }
    np.savez_compressed(golden, **arrays)
    split = {
        "schema_version": 1,
        "strategy": "deterministic_whole_episode_stratified",
        "split": "validation",
        "seed": 42,
        "validation_episode_ids": [3, 9],
        "validation_episode_count": 2,
        "selected_episode_count": 2,
    }
    dataset = {
        "repo_id": "test/example",
        "revision": "c" * 40,
        "codebase_version": "v2.0" if track_name == "libero" else "v3.0",
    }
    sidecar = {
        "schema_version": 2,
        "sha256": _sha256(golden),
        "run_id": f"{track_name}-golden",
        "config_name": track.golden_config,
        "resolved_config": {
            "name": track.golden_config,
            "training_seed": 42,
            "fingerprint_sha256": "d" * 64,
            "dataset": dataset,
        },
        "dataset": dataset,
        "dataset_revision": dataset["revision"],
        "seed": track.golden_seed,
        "data_split_seed": 42,
        "data_split": split,
        "samples": 64,
        "action_horizon": 4,
        "action_dim": 8,
        "image_names": ["base_0_rgb"],
        "image_layout": "BCHW",
    }
    _write_json(sidecar_path, sidecar)

    inventory = [
        {
            "name": "checkpoints/params/chunk",
            "generation": "123",
            "bytes": 10,
            "md5_base64": "AAAAAA==",
            "crc32c_base64": "BBBBBB==",
        }
    ]
    source_revision = repro_stage_checkpoints.inventory_revision(inventory)
    source = {
        "schema_version": 1,
        "source": {
            "provider": "gcs",
            "uri": "gs://openpi-assets/checkpoints/test",
            "revision": source_revision,
            "objects": inventory,
        },
        "checkpoint": {"key": track.checkpoint_key, "local_dirname": f"{track_name}_jax"},
        "totals": {"files": 1, "bytes": 10},
        "files": [{"path": "params/chunk", "bytes": 10, "sha256": "e" * 64}],
    }
    _write_json(source_path, source)

    converted = {
        "schema_version": 1,
        "source": {
            "provider": "openpi-jax-to-pytorch",
            "revision": "",
            "upstream": {
                "provider": "gcs",
                "uri": source["source"]["uri"],
                "revision": source_revision,
            },
        },
        "conversion": {
            "source_commit": SOURCE_COMMIT,
            "image_digest": IMAGE_DIGEST,
            "converter": "examples/convert_jax_model_to_pytorch.py",
            "config_name": track.teacher_config,
            "precision": "bfloat16",
        },
        "checkpoint": {
            "key": track.checkpoint_key,
            "local_dirname": f"{track_name}_pytorch",
            "format": "pytorch-safetensors",
        },
        "totals": {"files": 1, "bytes": 12},
        "files": [{"path": "model.safetensors", "bytes": 12, "sha256": "f" * 64}],
    }
    converted["source"]["revision"] = repro_stage_converted_checkpoints.conversion_revision(
        repro_stage_converted_checkpoints.manifest_identity(converted)
    )
    _write_json(converted_path, converted)

    jax = rng.standard_normal((64, 4, 8), dtype=np.float32)
    pytorch = jax.copy()
    np.savez_compressed(velocity_path, jax=jax, pytorch=pytorch)
    cosine = np.ones(64)
    report = {
        "schema_version": 2,
        "config_name": track.teacher_config,
        "samples": 64,
        "provenance": {
            "golden_corpus": {
                "path": str(golden.resolve()),
                "sha256": _sha256(golden),
                "sidecar_path": str(sidecar_path.resolve()),
                "sidecar_sha256": _sha256(sidecar_path),
                "run_id": sidecar["run_id"],
                "config_name": track.golden_config,
                "config_fingerprint_sha256": sidecar["resolved_config"]["fingerprint_sha256"],
                "dataset": dataset,
                "seed": track.golden_seed,
                "data_split_seed": 42,
                "data_split": split,
            },
            "jax_checkpoint": {
                "path": str(tmp_path / "jax"),
                "manifest": {
                    "path": str(source_path.resolve()),
                    "sha256": _sha256(source_path),
                    "revision": source_revision,
                },
            },
            "pytorch_checkpoint": {
                "path": str(tmp_path / "pytorch"),
                "config": {"config_name": track.teacher_config},
                "manifest": {
                    "path": str(converted_path.resolve()),
                    "sha256": _sha256(converted_path),
                    "revision": converted["source"]["revision"],
                    "source_commit": SOURCE_COMMIT,
                    "image_digest": IMAGE_DIGEST,
                },
            },
        },
        "cosine_mean": float(np.mean(cosine)),
        "cosine_min": float(np.min(cosine)),
        "mse": 0.0,
        "max_absolute_error": 0.0,
        "gate_cosine_minimum": 0.999,
        "gate_pass": True,
        "velocities": {"path": str(velocity_path.resolve()), "sha256": _sha256(velocity_path)},
    }
    _write_json(report_path, report)
    inputs = evidence.EvidenceInputs(
        golden_npz=golden,
        golden_sidecar=sidecar_path,
        equivalence_report=report_path,
        velocity_npz=velocity_path,
        source_manifest=source_path,
        converted_manifest=converted_path,
    )
    return track, inputs, sidecar, report, source, converted


def _validate(track, inputs):
    return evidence.validate_evidence(
        track,
        inputs,
        source_commit=SOURCE_COMMIT,
        image_digest=IMAGE_DIGEST,
    )


def _cli_args(inputs, *, action="validate"):
    return [
        action,
        "--track",
        "libero",
        "--golden-npz",
        str(inputs.golden_npz),
        "--golden-sidecar",
        str(inputs.golden_sidecar),
        "--equivalence-report",
        str(inputs.equivalence_report),
        "--velocity-npz",
        str(inputs.velocity_npz),
        "--source-manifest",
        str(inputs.source_manifest),
        "--converted-manifest",
        str(inputs.converted_manifest),
        "--source-commit",
        SOURCE_COMMIT,
        "--image-digest",
        IMAGE_DIGEST,
    ]


def test_validate_builds_content_addressed_four_file_manifest_without_aws(tmp_path, monkeypatch):
    track, inputs, *_ = _fixture(tmp_path)
    monkeypatch.setattr(
        repro_stage_data,
        "run_command",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("validate called AWS")),
    )
    validated = _validate(track, inputs)
    public = {key: value for key, value in validated.items() if key != "_local_files"}
    content_bytes = json.dumps(validated["content"], separators=(",", ":"), sort_keys=True).encode()
    assert validated["evidence_revision"] == hashlib.sha256(content_bytes).hexdigest()
    assert [item["role"] for item in public["content"]["files"]] == [
        "golden_npz",
        "golden_sidecar",
        "framework_equivalence",
        "velocities",
    ]
    assert "_local_files" not in public
    assert public["content"]["gate"]["cosine_minimum"] == 0.999
    assert public["content"]["gate"]["cosine_min"] == pytest.approx(1.0)
    assert public["content"]["gate"]["pass"] is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("gate_pass", False, "does not pass"),
        ("samples", 63, "does not pass"),
        ("config_name", "pi05_droid_jointpos", "does not pass"),
        ("cosine_min", 0.998, "does not pass"),
    ],
)
def test_rejects_noncanonical_report_gate(tmp_path, field, value, message):
    track, inputs, _, report, *_ = _fixture(tmp_path)
    report[field] = value
    _write_json(inputs.equivalence_report, report)
    with pytest.raises(repro_stage_data.StageError, match=message):
        _validate(track, inputs)


def test_rejects_changed_golden_bytes_and_sidecar(tmp_path):
    track, inputs, sidecar, *_ = _fixture(tmp_path)
    with inputs.golden_npz.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(repro_stage_data.StageError, match="golden corpus sidecar"):
        _validate(track, inputs)

    track, inputs, sidecar, *_ = _fixture(tmp_path / "second")
    sidecar["data_split"]["seed"] = 41
    _write_json(inputs.golden_sidecar, sidecar)
    with pytest.raises(repro_stage_data.StageError, match="canonical seed-42"):
        _validate(track, inputs)


def test_rejects_changed_velocity_bytes_even_when_report_hash_is_rewritten(tmp_path):
    track, inputs, _, report, *_ = _fixture(tmp_path)
    with np.load(inputs.velocity_npz) as archive:
        jax = archive["jax"]
        pytorch = archive["pytorch"].copy()
    pytorch[0] *= -1
    np.savez_compressed(inputs.velocity_npz, jax=jax, pytorch=pytorch)
    report["velocities"]["sha256"] = _sha256(inputs.velocity_npz)
    _write_json(inputs.equivalence_report, report)
    with pytest.raises(repro_stage_data.StageError, match="differs from the velocity NPZ"):
        _validate(track, inputs)


@pytest.mark.parametrize("which", ["source", "converted"])
def test_rejects_manifest_corruption_even_if_report_hash_is_rewritten(tmp_path, which):
    track, inputs, _, report, source, converted = _fixture(tmp_path)
    if which == "source":
        source["source"]["objects"][0]["generation"] = "changed"
        _write_json(inputs.source_manifest, source)
        report["provenance"]["jax_checkpoint"]["manifest"]["sha256"] = _sha256(inputs.source_manifest)
        expected = "source manifest"
    else:
        converted["conversion"]["image_digest"] = "sha256:" + "9" * 64
        converted["source"]["revision"] = repro_stage_converted_checkpoints.conversion_revision(
            repro_stage_converted_checkpoints.manifest_identity(converted)
        )
        _write_json(inputs.converted_manifest, converted)
        converted_provenance = report["provenance"]["pytorch_checkpoint"]["manifest"]
        converted_provenance["sha256"] = _sha256(inputs.converted_manifest)
        converted_provenance["revision"] = converted["source"]["revision"]
        converted_provenance["image_digest"] = converted["conversion"]["image_digest"]
        expected = "converted manifest provenance"
    _write_json(inputs.equivalence_report, report)
    with pytest.raises(repro_stage_data.StageError, match=expected):
        _validate(track, inputs)


def test_rejects_symlink_and_path_contract_aliases(tmp_path):
    track, inputs, *_ = _fixture(tmp_path)
    real_golden = tmp_path / "real-golden.npz"
    inputs.golden_npz.rename(real_golden)
    inputs.golden_npz.symlink_to(real_golden)
    with pytest.raises(repro_stage_data.StageError, match="must not be a symlink"):
        _validate(track, inputs)


def test_accepts_report_paths_through_parent_symlink(tmp_path):
    real_root = tmp_path / "real"
    alias_root = tmp_path / "alias"
    track, inputs, _, report, *_ = _fixture(real_root)
    alias_root.symlink_to(real_root, target_is_directory=True)

    report["provenance"]["golden_corpus"]["path"] = str(alias_root / inputs.golden_npz.name)
    report["provenance"]["golden_corpus"]["sidecar_path"] = str(alias_root / inputs.golden_sidecar.name)
    report["provenance"]["jax_checkpoint"]["manifest"]["path"] = str(alias_root / inputs.source_manifest.name)
    report["provenance"]["pytorch_checkpoint"]["manifest"]["path"] = str(alias_root / inputs.converted_manifest.name)
    report["velocities"]["path"] = str(alias_root / inputs.velocity_npz.name)
    _write_json(inputs.equivalence_report, report)

    assert _validate(track, inputs)["content"]["gate"]["pass"] is True


def test_rejects_report_path_that_does_not_resolve_to_input(tmp_path):
    track, inputs, _, report, *_ = _fixture(tmp_path)
    report["provenance"]["golden_corpus"]["path"] = str(tmp_path / "missing.npz")
    _write_json(inputs.equivalence_report, report)

    with pytest.raises(repro_stage_data.StageError, match="exact golden NPZ"):
        _validate(track, inputs)


def _mock_s3_runner(*, existing=False, corrupt_head=False):
    calls = []
    stored = {}

    def option(argv, name):
        return argv[argv.index(name) + 1]

    def runner(argv):
        argv = list(argv)
        calls.append(argv)
        operation = tuple(argv[1:3])
        if operation == ("s3api", "list-object-versions"):
            return json.dumps({"Versions": [{"Key": "existing"}]} if existing else {})
        if operation == ("s3api", "put-object"):
            assert option(argv, "--if-none-match") == "*"
            assert option(argv, "--server-side-encryption") == "AES256"
            assert option(argv, "--checksum-algorithm") == "SHA256"
            key = option(argv, "--key")
            version = f"version-{len(stored) + 1}"
            stored[key] = {
                "version": version,
                "body": pathlib.Path(option(argv, "--body")).read_bytes(),
                "checksum": option(argv, "--checksum-sha256"),
                "metadata": dict(part.split("=", 1) for part in option(argv, "--metadata").split(",")),
            }
            return json.dumps({"VersionId": version, "ChecksumSHA256": stored[key]["checksum"]})
        if operation == ("s3api", "head-object"):
            key = option(argv, "--key")
            item = stored[key]
            return json.dumps(
                {
                    "VersionId": item["version"],
                    "ContentLength": len(item["body"]),
                    "ChecksumSHA256": "corrupt" if corrupt_head else item["checksum"],
                    "ServerSideEncryption": "AES256",
                    "Metadata": item["metadata"],
                }
            )
        raise AssertionError(argv)

    return runner, calls, stored


def test_upload_records_all_version_hash_size_and_encryption_metadata(tmp_path, monkeypatch):
    track, inputs, *_ = _fixture(tmp_path)
    validated = _validate(track, inputs)
    runner, calls, stored = _mock_s3_runner()
    monkeypatch.setattr(
        repro_stage_data,
        "verify_aws_destination",
        lambda *_args, **_kwargs: ("752160877725", "us-east-2"),
    )
    result = evidence.upload_evidence(
        {"aws": {"account_id": "752160877725", "region": "us-east-2"}},
        track,
        validated,
        bucket="pi05-repro-752160877725-us-east-2",
        prefix="evidence/framework-equivalence",
        runner=runner,
        environ={"AWS_REGION": "us-east-2"},
    )
    assert len(result["objects"]) == 4
    assert result["manifest"]["version_id"] == "version-5"
    assert all(item["version_id"] and item["bytes"] > 0 for item in [*result["objects"], result["manifest"]])
    assert all(
        item["storage"]["server_side_encryption"] == "AES256" and item["storage"]["checksum_algorithm"] == "SHA256"
        for item in [*result["objects"], result["manifest"]]
    )
    manifest_key = next(key for key in stored if key.endswith("manifest.sha256.json"))
    uploaded_manifest = json.loads(stored[manifest_key]["body"])
    assert [item["version_id"] for item in uploaded_manifest["storage"]["objects"]] == [
        "version-1",
        "version-2",
        "version-3",
        "version-4",
    ]
    assert all(
        base64.b64decode(item["storage"]["checksum_sha256_base64"]).hex() == item["sha256"]
        for item in uploaded_manifest["storage"]["objects"]
    )
    assert len([call for call in calls if call[1:3] == ["s3api", "put-object"]]) == 5


def test_upload_refuses_any_existing_object_history_before_put(tmp_path, monkeypatch):
    track, inputs, *_ = _fixture(tmp_path)
    runner, calls, _ = _mock_s3_runner(existing=True)
    monkeypatch.setattr(
        repro_stage_data,
        "verify_aws_destination",
        lambda *_args, **_kwargs: ("752160877725", "us-east-2"),
    )
    with pytest.raises(repro_stage_data.StageError, match="refusing overwrite"):
        evidence.upload_evidence(
            {},
            track,
            _validate(track, inputs),
            bucket="pi05-repro-752160877725-us-east-2",
            prefix="evidence/framework-equivalence",
            runner=runner,
            environ={"AWS_REGION": "us-east-2"},
        )
    assert not any(call[1:3] == ["s3api", "put-object"] for call in calls)


def test_upload_fails_closed_on_remote_checksum_mismatch(tmp_path, monkeypatch):
    track, inputs, *_ = _fixture(tmp_path)
    runner, _, _ = _mock_s3_runner(corrupt_head=True)
    monkeypatch.setattr(
        repro_stage_data,
        "verify_aws_destination",
        lambda *_args, **_kwargs: ("752160877725", "us-east-2"),
    )
    with pytest.raises(repro_stage_data.StageError, match="checksum/encryption"):
        evidence.upload_evidence(
            {},
            track,
            _validate(track, inputs),
            bucket="pi05-repro-752160877725-us-east-2",
            prefix="evidence/framework-equivalence",
            runner=runner,
            environ={"AWS_REGION": "us-east-2"},
        )


def test_cli_validate_and_upload_dry_run_never_call_aws(tmp_path, monkeypatch, capsys):
    _, inputs, *_ = _fixture(tmp_path)
    monkeypatch.setattr(
        evidence,
        "upload_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dry-run called AWS uploader")),
    )
    assert evidence.main(_cli_args(inputs)) == 0
    assert json.loads(capsys.readouterr().out)["validation"]["content"]["samples"] == 64
    assert (
        evidence.main(
            [
                *_cli_args(inputs, action="upload"),
                "--bucket",
                "pi05-repro-752160877725-us-east-2",
                "--prefix",
                "evidence/framework-equivalence",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "dry-run"
    assert output["mutations_authorized"] is False


def test_cli_upload_requires_explicit_bucket_and_prefix(tmp_path, capsys):
    _, inputs, *_ = _fixture(tmp_path)
    assert evidence.main(_cli_args(inputs, action="upload")) == 2
    assert "requires explicit --bucket and --prefix" in capsys.readouterr().err


def test_metric_recomputation_handles_nonzero_roundoff(tmp_path):
    track, inputs, _, report, *_ = _fixture(tmp_path)
    with np.load(inputs.velocity_npz) as archive:
        jax = archive["jax"]
    pytorch = jax + np.float32(1e-5)
    np.savez_compressed(inputs.velocity_npz, jax=jax, pytorch=pytorch)
    jax_flat = jax.reshape(64, -1).astype(np.float64)
    pytorch_flat = pytorch.reshape(64, -1).astype(np.float64)
    cosine = np.sum(jax_flat * pytorch_flat, axis=1) / (
        np.linalg.norm(jax_flat, axis=1) * np.linalg.norm(pytorch_flat, axis=1)
    )
    absolute = np.abs(pytorch.astype(np.float64) - jax.astype(np.float64))
    report |= {
        "cosine_mean": float(np.mean(cosine)),
        "cosine_min": float(np.min(cosine)),
        "mse": float(np.mean(np.square(absolute))),
        "max_absolute_error": float(np.max(absolute)),
    }
    report["velocities"]["sha256"] = _sha256(inputs.velocity_npz)
    _write_json(inputs.equivalence_report, report)
    assert math.isclose(_validate(track, inputs)["content"]["gate"]["cosine_min"], float(np.min(cosine)))
