import base64
import hashlib
import json

import pytest

from scripts import repro_stage_checkpoints
from scripts import repro_stage_converted_checkpoints as converted
from scripts import repro_stage_data
from scripts import repro_worker


def _source_fixture(tmp_path, key="libero"):
    spec = converted.CONVERTED_CHECKPOINTS[key]
    source_root, source_manifest_path = repro_stage_checkpoints.checkpoint_paths(tmp_path, spec.source)
    payloads = {
        f"assets/{spec.asset_id}/norm_stats.json": b'{"stats": true}\n',
        "params/chunk": b"jax weights",
    }
    inventory = []
    for index, (relative, payload) in enumerate(payloads.items(), start=1):
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        inventory.append(
            {
                "name": f"{spec.source.object_prefix}{relative}",
                "generation": str(index),
                "bytes": len(payload),
                "md5_base64": base64.b64encode(hashlib.md5(payload, usedforsecurity=False).digest()).decode(),
                "crc32c_base64": "AAAAAA==",
                "updated": "2026-01-01T00:00:00Z",
            }
        )
    source_manifest = repro_stage_checkpoints.build_checkpoint_manifest(
        spec.source, source_root, inventory, hash_workers=1
    )
    repro_stage_data.write_manifest(source_manifest_path, source_manifest)
    output = tmp_path / spec.local_dirname
    (output / "assets" / spec.asset_id).mkdir(parents=True)
    (output / "model.safetensors").write_bytes(b"pytorch weights")
    (output / "assets" / spec.asset_id / "norm_stats.json").write_bytes(
        payloads[f"assets/{spec.asset_id}/norm_stats.json"]
    )
    (output / "config.json").write_text(
        json.dumps({"config_name": spec.config_name, "precision": "bfloat16", "pi05": True})
    )
    return spec, source_root, source_manifest_path, output


def test_manifest_binds_converted_bytes_upstream_revision_and_source_commit(tmp_path):
    spec, source_root, source_manifest_path, output = _source_fixture(tmp_path)
    commit = "a" * 40
    manifest = converted.build_converted_manifest(
        spec,
        source_root,
        source_manifest_path,
        output,
        source_commit=commit,
        image_digest="sha256:" + "1" * 64,
        hash_workers=2,
    )
    assert manifest["conversion"]["source_commit"] == commit
    assert manifest["source"]["upstream"]["revision"] == repro_stage_checkpoints.inventory_revision(
        repro_stage_checkpoints.load_checkpoint_manifest(source_manifest_path, spec.source)["source"]["objects"]
    )
    assert manifest["source"]["revision"] == converted.conversion_revision(converted.manifest_identity(manifest))
    assert {item["path"] for item in manifest["files"]} >= {
        "model.safetensors",
        "config.json",
        "assets/physical-intelligence/libero/norm_stats.json",
    }
    target = converted.converted_s3_target("s3://bucket/checkpoints", spec, manifest["source"]["revision"])
    assert f"/{spec.local_dirname}/{manifest['source']['revision']}/checkpoint/" in target.snapshot_uri


def test_validation_rejects_wrong_config_and_missing_asset(tmp_path):
    spec, source_root, source_manifest_path, output = _source_fixture(tmp_path, "droid_jointpos")
    (output / "config.json").write_text(
        json.dumps({"config_name": "pi05_droid", "precision": "bfloat16", "pi05": True})
    )
    with pytest.raises(repro_stage_data.StageError, match="config mismatch"):
        converted.build_converted_manifest(
            spec,
            source_root,
            source_manifest_path,
            output,
            source_commit="b" * 40,
            image_digest="sha256:" + "2" * 64,
            hash_workers=1,
        )

    (output / "config.json").write_text(
        json.dumps({"config_name": spec.config_name, "precision": "bfloat16", "pi05": True})
    )
    (output / "assets" / spec.asset_id / "norm_stats.json").unlink()
    with pytest.raises(repro_stage_data.StageError, match="missing or empty"):
        converted.build_converted_manifest(
            spec,
            source_root,
            source_manifest_path,
            output,
            source_commit="b" * 40,
            image_digest="sha256:" + "2" * 64,
            hash_workers=1,
        )


def test_saved_manifest_detects_changed_converted_bytes(tmp_path):
    spec, source_root, source_manifest_path, output = _source_fixture(tmp_path)
    saved = converted.build_converted_manifest(
        spec,
        source_root,
        source_manifest_path,
        output,
        source_commit="c" * 40,
        image_digest="sha256:" + "3" * 64,
        hash_workers=1,
    )
    (output / "model.safetensors").write_bytes(b"changed")
    rebuilt = converted.build_converted_manifest(
        spec,
        source_root,
        source_manifest_path,
        output,
        source_commit="c" * 40,
        image_digest="sha256:" + "3" * 64,
        hash_workers=1,
    )
    with pytest.raises(repro_stage_data.StageError, match="verification failed"):
        converted.validate_saved_manifest(saved, rebuilt)


def test_upload_dry_run_performs_no_git_or_aws_calls(tmp_path, monkeypatch, capsys):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("dry run performed an external action")

    monkeypatch.setattr(converted, "verify_source_checkout", unexpected)
    monkeypatch.setattr(converted, "upload_converted_checkpoint", unexpected)
    result = converted.main(
        [
            "upload",
            "--checkpoint",
            "libero",
            "--local-root",
            str(tmp_path),
            "--source-commit",
            "d" * 40,
            "--image-digest",
            "sha256:" + "4" * 64,
            "--s3-root",
            "s3://bucket/checkpoints",
            "--equivalence-report",
            str(tmp_path / "equivalence.json"),
        ]
    )
    assert result == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "dry-run"
    assert plan["mutations_authorized"] is False


def test_source_checkout_must_match_and_be_clean():
    calls = []

    def runner(argv):
        calls.append(argv)
        return "e" * 40 if argv[1:3] == ["rev-parse", "HEAD"] else ""

    converted.verify_source_checkout("e" * 40, runner=runner)
    assert calls == [["git", "rev-parse", "HEAD"], ["git", "status", "--porcelain"]]

    with pytest.raises(repro_stage_data.StageError, match="dirty"):
        converted.verify_source_checkout(
            "e" * 40,
            runner=lambda argv: "e" * 40 if argv[1] == "rev-parse" else " M converter.py",
        )


def test_equivalence_gate_binds_exact_manifests_and_velocity_evidence(tmp_path):
    spec, source_root, source_manifest_path, output = _source_fixture(tmp_path)
    manifest = converted.build_converted_manifest(
        spec,
        source_root,
        source_manifest_path,
        output,
        source_commit="a" * 40,
        image_digest="sha256:" + "1" * 64,
        hash_workers=1,
    )
    manifest_path = tmp_path / "converted-manifest.json"
    repro_stage_data.write_manifest(manifest_path, manifest)
    report_path = tmp_path / "framework-equivalence.json"
    velocity_path = report_path.with_suffix(".npz")
    velocity_path.write_bytes(b"fixed velocities")
    report = {
        "schema_version": 2,
        "config_name": spec.config_name,
        "samples": 64,
        "gate_pass": True,
        "cosine_min": 0.9995,
        "provenance": {
            "golden_corpus": {
                "config_name": "pi05_libero_l09_distill",
                "seed": 7001,
                "data_split_seed": 42,
                "data_split": {
                    "strategy": "deterministic_whole_episode_stratified",
                    "split": "validation",
                    "seed": 42,
                    "validation_episode_ids": [7],
                },
                "sha256": "2" * 64,
                "sidecar_sha256": "3" * 64,
            },
            "jax_checkpoint": {
                "manifest": {
                    "sha256": repro_stage_data.sha256_file(source_manifest_path),
                    "revision": manifest["source"]["upstream"]["revision"],
                }
            },
            "pytorch_checkpoint": {
                "manifest": {
                    "sha256": repro_stage_data.sha256_file(manifest_path),
                    "revision": manifest["source"]["revision"],
                    "source_commit": manifest["conversion"]["source_commit"],
                    "image_digest": manifest["conversion"]["image_digest"],
                }
            },
        },
        "velocities": {
            "path": str(velocity_path.resolve()),
            "sha256": repro_stage_data.sha256_file(velocity_path),
        },
    }
    report_path.write_text(json.dumps(report))

    evidence = converted.validate_equivalence_report(report_path, spec, source_manifest_path, manifest_path, manifest)
    assert evidence["gate_pass"] is True
    assert evidence["report"]["sha256"] == repro_stage_data.sha256_file(report_path)

    report["gate_pass"] = False
    report_path.write_text(json.dumps(report))
    with pytest.raises(repro_stage_data.StageError, match="does not pass"):
        converted.validate_equivalence_report(report_path, spec, source_manifest_path, manifest_path, manifest)


def test_upload_emits_a_copy_ready_worker_artifact(tmp_path):
    spec, source_root, source_manifest_path, output = _source_fixture(tmp_path)
    manifest = converted.build_converted_manifest(
        spec,
        source_root,
        source_manifest_path,
        output,
        source_commit="9" * 40,
        image_digest="sha256:" + "8" * 64,
        hash_workers=1,
    )
    manifest_path = tmp_path / "converted-manifest.json"
    repro_stage_data.write_manifest(manifest_path, manifest)
    calls = []

    def runner(argv):
        argv = list(argv)
        calls.append(argv)
        operation = tuple(argv[1:3])
        if operation == ("sts", "get-caller-identity"):
            return json.dumps({"Account": "752160877725"})
        if operation == ("s3api", "get-bucket-location"):
            return json.dumps({"LocationConstraint": "us-east-2"})
        if operation == ("s3api", "get-bucket-versioning"):
            return json.dumps({"Status": "Enabled"})
        if operation == ("s3api", "get-bucket-encryption"):
            return json.dumps({"ServerSideEncryptionConfiguration": {"Rules": [{}]}})
        if operation == ("s3api", "head-object"):
            return json.dumps(
                {
                    "ContentLength": manifest_path.stat().st_size,
                    "Metadata": {"source-revision": manifest["source"]["revision"]},
                    "VersionId": "converted-v1",
                }
            )
        if operation in {("s3", "sync"), ("s3api", "put-object")}:
            return ""
        raise AssertionError(argv)

    result = converted.upload_converted_checkpoint(
        {"aws": {"account_id": "752160877725", "region": "us-east-2"}},
        spec,
        output,
        manifest_path,
        manifest,
        "s3://pi05-repro-752160877725-us-east-2/checkpoints",
        equivalence_report_sha256="a" * 64,
        runner=runner,
        environ={"AWS_REGION": "us-east-2"},
    )
    artifact = result["worker_artifact"]
    next_spec = {
        "schema_version": 1,
        "project": "pi05-aws-repro",
        "run_id": "converted-smoke",
        "aws": {
            "account_id": "752160877725",
            "region": "us-east-2",
            "artifact_bucket": "pi05-repro-752160877725-us-east-2",
        },
        "source": {
            "s3_uri": "s3://pi05-repro-752160877725-us-east-2/source/openpi.bundle",
            "version_id": "source-v1",
            "sha256": "7" * 64,
            "commit": "9" * 40,
        },
        "image": {
            "uri": "752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro@sha256:" + "8" * 64,
            "digest": "sha256:" + "8" * 64,
            "purpose": "policy",
            "lerobot_runtime": "v2",
            "lerobot_revision": "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5",
        },
        "artifacts": [artifact],
        "container": {"command": ["python", "-V"], "environment": {}, "shm_size_gib": 1},
        "output": {"s3_uri": "s3://pi05-repro-752160877725-us-east-2/runs/converted-smoke/"},
        "timing": {"sync_interval_seconds": 60, "upload_buffer_seconds": 300, "stop_grace_seconds": 30},
        "scratch": {
            "model": "Amazon EC2 NVMe Instance Storage",
            "expected_count": 1,
            "ordinal": 0,
            "mount": "/mnt/openpi",
            "filesystem_label": "PI05_SCRATCH",
        },
        "seed": 0,
    }
    assert repro_worker.validate_worker_spec(next_spec)["artifacts"] == [artifact]
    assert any(call[1:3] == ["s3api", "put-object"] for call in calls)
