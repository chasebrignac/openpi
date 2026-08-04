from __future__ import annotations

import hashlib
import json

import pytest

from openpi.exporting import artifacts
from openpi.exporting.artifacts import artifact_record


def test_artifact_record_hashes_content(tmp_path):
    artifact = tmp_path / "model.plan"
    artifact.write_bytes(b"engine")
    assert artifact_record(artifact) == {
        "path": str(artifact.resolve()),
        "bytes": 6,
        "sha256": hashlib.sha256(b"engine").hexdigest(),
    }


def test_artifact_record_rejects_symlink(tmp_path):
    target = tmp_path / "target"
    target.write_text("payload")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="non-symlink"):
        artifact_record(link)


def test_require_clean_source_identity_binds_worker_sha(monkeypatch):
    monkeypatch.setattr(artifacts, "git_state", lambda: {"sha": "a" * 40, "dirty": False})
    assert artifacts.require_clean_source_identity(environ={"PI05_SOURCE_SHA": "a" * 40}) == {
        "sha": "a" * 40,
        "dirty": False,
    }
    with pytest.raises(RuntimeError, match="differs"):
        artifacts.require_clean_source_identity(environ={"PI05_SOURCE_SHA": "b" * 40})
    monkeypatch.setattr(artifacts, "git_state", lambda: {"sha": "a" * 40, "dirty": True})
    with pytest.raises(RuntimeError, match="clean"):
        artifacts.require_clean_source_identity(environ={"PI05_SOURCE_SHA": "a" * 40})


def test_write_stage_manifest_is_atomic_and_refuses_overwrite(tmp_path, monkeypatch):
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"payload")
    output = tmp_path / "manifest.json"
    monkeypatch.setattr(artifacts, "git_state", lambda: {"sha": "a" * 40, "dirty": False})
    monkeypatch.setenv("PI05_SOURCE_SHA", "a" * 40)
    kwargs = {
        "stage": "export",
        "track": "libero",
        "command": ["python", "export.py"],
        "image_digest": f"sha256:{'b' * 64}",
        "dataset": "physical-intelligence/libero",
        "dataset_revision": "c" * 40,
        "instance_type": "g7e.4xlarge",
        "instance_id": "i-0123456789abcdef0",
        "cost_reservation": "reservation",
        "artifacts": [source],
        "details": {"metric": 1.0},
    }
    value = artifacts.write_stage_manifest(output, **kwargs)
    assert json.loads(output.read_text()) == value
    assert not list(tmp_path.glob(".*.tmp"))
    with pytest.raises(FileExistsError, match="already exists"):
        artifacts.write_stage_manifest(output, **kwargs)


def test_write_stage_manifest_rejects_duplicate_artifacts(tmp_path, monkeypatch):
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"payload")
    monkeypatch.setattr(artifacts, "git_state", lambda: {"sha": "a" * 40, "dirty": False})
    monkeypatch.setenv("PI05_SOURCE_SHA", "a" * 40)
    with pytest.raises(ValueError, match="duplicate"):
        artifacts.write_stage_manifest(
            tmp_path / "manifest.json",
            stage="export",
            track="droid",
            command=["export"],
            image_digest=f"sha256:{'b' * 64}",
            dataset="allenai/MolmoAct2-DROID-Dataset",
            dataset_revision="c" * 40,
            instance_type="g7e.4xlarge",
            instance_id="i-0123456789abcdef0",
            cost_reservation="reservation",
            artifacts=[source, source],
            details={},
        )


def test_output_preflight_never_overwrites_prior_evidence(tmp_path):
    output_dir = tmp_path / "fresh"
    artifacts.prepare_fresh_output_directory(output_dir, stage="export")
    assert output_dir.is_dir()
    prior = output_dir / "prior.json"
    artifacts.write_json_new(prior, {"finite": 1.0})
    with pytest.raises(FileExistsError, match="not empty"):
        artifacts.prepare_fresh_output_directory(output_dir, stage="export")
    with pytest.raises(FileExistsError, match="fresh output"):
        artifacts.require_absent_outputs([prior], stage="validation")
    with pytest.raises(FileExistsError, match="already exists"):
        artifacts.write_json_new(prior, {"replacement": True})
