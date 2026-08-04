import base64
import hashlib
import json
import pathlib

import pytest

from scripts import repro_stage_checkpoints
from scripts import repro_stage_converted_checkpoints as converted
from scripts import repro_stage_data
from scripts import repro_worker


class FakeVersionedS3:
    """Small command-runner fake for the converted publisher's S3 protocol."""

    def __init__(self):
        self.objects = {}
        self.multipart = {}
        self.calls = []
        self.mutations = []
        self.fail_put_key_once = None
        self.mutate_source_on_upload_key = None
        self.mutate_source_path = None
        self._version = 0
        self._upload = 0

    @staticmethod
    def _arg(argv, name):
        return argv[argv.index(name) + 1]

    def _new_version(self):
        self._version += 1
        return f"version-{self._version}"

    def _store(self, key, data, metadata):
        version = self._new_version()
        entry = {
            "version_id": version,
            "data": data,
            "metadata": metadata,
            "etag": f'"{hashlib.md5(data, usedforsecurity=False).hexdigest()}"',
            "checksum": base64.b64encode(hashlib.sha256(data).digest()).decode(),
        }
        self.objects.setdefault(key, []).append(entry)
        self.mutations.append(("store", key))
        return entry

    def seed_object(self, key, data=b"seed"):
        return self._store(key, data, {})

    def __call__(self, raw_argv):
        argv = list(raw_argv)
        self.calls.append(argv)
        operation = tuple(argv[1:3])
        if operation == ("sts", "get-caller-identity"):
            return json.dumps({"Account": "752160877725"})
        if operation == ("s3api", "get-bucket-location"):
            return json.dumps({"LocationConstraint": "us-east-2"})
        if operation == ("s3api", "get-bucket-versioning"):
            return json.dumps({"Status": "Enabled"})
        if operation == ("s3api", "get-bucket-encryption"):
            return json.dumps({"ServerSideEncryptionConfiguration": {"Rules": [{}]}})
        if operation == ("s3api", "list-object-versions"):
            prefix = self._arg(argv, "--prefix")
            versions = []
            for key, entries in self.objects.items():
                if key.startswith(prefix):
                    versions.extend(
                        {
                            "Key": key,
                            "VersionId": entry["version_id"],
                            "IsLatest": index == len(entries) - 1,
                        }
                        for index, entry in enumerate(entries)
                    )
            return json.dumps({"Versions": versions, "DeleteMarkers": [], "IsTruncated": False})
        if operation == ("s3api", "put-object"):
            key = self._arg(argv, "--key")
            if key == self.fail_put_key_once:
                self.fail_put_key_once = None
                raise repro_stage_data.StageError("injected interrupted publication")
            if "--if-none-match" in argv and self.objects.get(key):
                raise repro_stage_data.StageError("conditional request failed")
            data = pathlib.Path(self._arg(argv, "--body")).read_bytes()
            metadata = json.loads(self._arg(argv, "--metadata"))
            entry = self._store(key, data, metadata)
            return json.dumps({"VersionId": entry["version_id"], "ETag": entry["etag"]})
        if operation == ("s3api", "head-object"):
            key = self._arg(argv, "--key")
            version = self._arg(argv, "--version-id")
            entry = next(item for item in self.objects[key] if item["version_id"] == version)
            return json.dumps(
                {
                    "VersionId": version,
                    "ContentLength": len(entry["data"]),
                    "Metadata": entry["metadata"],
                    "ServerSideEncryption": "AES256",
                    "ChecksumSHA256": entry["checksum"],
                    "ETag": entry["etag"],
                }
            )
        if operation == ("s3api", "get-object"):
            key = self._arg(argv, "--key")
            version = self._arg(argv, "--version-id")
            entry = next(item for item in self.objects[key] if item["version_id"] == version)
            pathlib.Path(argv[-1]).write_bytes(entry["data"])
            return json.dumps({"VersionId": version, "ChecksumSHA256": entry["checksum"]})
        if operation == ("s3api", "create-multipart-upload"):
            assert self._arg(argv, "--checksum-algorithm") == "SHA256"
            assert self._arg(argv, "--checksum-type") == "COMPOSITE"
            self._upload += 1
            upload_id = f"upload-{self._upload}"
            self.multipart[upload_id] = {
                "key": self._arg(argv, "--key"),
                "metadata": json.loads(self._arg(argv, "--metadata")),
                "parts": {},
            }
            return json.dumps({"UploadId": upload_id})
        if operation == ("s3api", "upload-part"):
            upload_id = self._arg(argv, "--upload-id")
            part_number = int(self._arg(argv, "--part-number"))
            data = pathlib.Path(self._arg(argv, "--body")).read_bytes()
            checksum = base64.b64encode(hashlib.sha256(data).digest()).decode()
            assert self._arg(argv, "--checksum-algorithm") == "SHA256"
            assert self._arg(argv, "--checksum-sha256") == checksum
            self.multipart[upload_id]["parts"][part_number] = {
                "data": data,
                "etag": f'"part-{part_number}"',
                "checksum": checksum,
            }
            if self._arg(argv, "--key") == self.mutate_source_on_upload_key and self.mutate_source_path is not None:
                self.mutate_source_path.write_bytes(b"x" * self.mutate_source_path.stat().st_size)
                self.mutate_source_on_upload_key = None
            return json.dumps({"ETag": f'"part-{part_number}"', "ChecksumSHA256": checksum})
        if operation == ("s3api", "complete-multipart-upload"):
            upload_id = self._arg(argv, "--upload-id")
            pending = self.multipart.pop(upload_id)
            key = pending["key"]
            if "--if-none-match" in argv and self.objects.get(key):
                raise repro_stage_data.StageError("conditional multipart completion failed")
            assert self._arg(argv, "--checksum-type") == "COMPOSITE"
            description = json.loads(
                pathlib.Path(self._arg(argv, "--multipart-upload").removeprefix("file://")).read_text()
            )
            for part in description["Parts"]:
                stored = pending["parts"][part["PartNumber"]]
                assert part["ETag"] == stored["etag"]
                assert part["ChecksumSHA256"] == stored["checksum"]
            data = b"".join(pending["parts"][number]["data"] for number in sorted(pending["parts"]))
            entry = self._store(key, data, pending["metadata"])
            return json.dumps({"VersionId": entry["version_id"], "ETag": entry["etag"]})
        if operation == ("s3api", "abort-multipart-upload"):
            self.multipart.pop(self._arg(argv, "--upload-id"), None)
            return ""
        raise AssertionError(argv)


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


def test_converted_manifest_is_deterministic_for_identical_conversion_bytes(tmp_path):
    spec, source_root, source_manifest_path, output = _source_fixture(tmp_path)
    kwargs = {
        "source_commit": "a" * 40,
        "image_digest": "sha256:" + "1" * 64,
        "hash_workers": 1,
    }

    first = converted.build_converted_manifest(spec, source_root, source_manifest_path, output, **kwargs)
    second = converted.build_converted_manifest(spec, source_root, source_manifest_path, output, **kwargs)

    assert first == second
    assert "created_at" not in first
    assert json.dumps(first, indent=2, sort_keys=True) == json.dumps(second, indent=2, sort_keys=True)

    noncanonical = json.loads(json.dumps(first))
    noncanonical["created_at"] = "2026-08-04T00:00:00+00:00"
    with pytest.raises(repro_stage_data.StageError, match="exact canonical manifest"):
        converted.validate_saved_manifest(noncanonical, second)


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

    alias_root = tmp_path.parent / f"{tmp_path.name}-alias"
    alias_root.symlink_to(tmp_path, target_is_directory=True)
    report["velocities"]["path"] = str(alias_root / velocity_path.name)
    report_path.write_text(json.dumps(report))
    assert (
        converted.validate_equivalence_report(report_path, spec, source_manifest_path, manifest_path, manifest)[
            "gate_pass"
        ]
        is True
    )

    report["velocities"]["path"] = str(alias_root / "missing.npz")
    report_path.write_text(json.dumps(report))
    with pytest.raises(repro_stage_data.StageError, match="velocity evidence"):
        converted.validate_equivalence_report(report_path, spec, source_manifest_path, manifest_path, manifest)

    report["velocities"]["path"] = str(alias_root / velocity_path.name)
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
    runner = FakeVersionedS3()

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
    put_calls = [call for call in runner.calls if call[1:3] == ["s3api", "put-object"]]
    assert put_calls
    assert all(call[call.index("--if-none-match") + 1] == "*" for call in put_calls)
    assert not any(call[1:3] == ["s3", "sync"] for call in runner.calls)
    assert runner.mutations[-1][1].endswith("/manifest.sha256.json")
    assert all(len(versions) == 1 for versions in runner.objects.values())
    assert set(result["publication"]) == {"claim", "payload", "receipt", "manifest"}
    assert artifact["payload_objects"] == [
        {"path": item["path"], "version_id": receipt["version_id"], "sha256": item["sha256"]}
        for item, receipt in zip(manifest["files"], result["publication"]["payload"], strict=True)
    ]
    pins = repro_worker.exact_artifact_payload_pins(manifest, artifact)
    assert pins is not None
    assert [pin["version_id"] for pin in pins] == [item["version_id"] for item in artifact["payload_objects"]]


def test_upload_resumes_an_exact_interrupted_publication_without_new_versions(tmp_path):
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
    target = converted.converted_s3_target(
        "s3://pi05-repro-752160877725-us-east-2/checkpoints", spec, manifest["source"]["revision"]
    )
    runner = FakeVersionedS3()
    runner.fail_put_key_once = f"{target.prefix}/checkpoint/config.json"
    kwargs = {
        "config": {"aws": {"account_id": "752160877725", "region": "us-east-2"}},
        "spec": spec,
        "converted_root": output,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "s3_root": "s3://pi05-repro-752160877725-us-east-2/checkpoints",
        "equivalence_report_sha256": "a" * 64,
        "runner": runner,
        "environ": {"AWS_REGION": "us-east-2"},
    }

    with pytest.raises(repro_stage_data.StageError, match="injected interrupted"):
        converted.upload_converted_checkpoint(**kwargs)
    versions_before_resume = {key: entries[0]["version_id"] for key, entries in runner.objects.items()}

    result = converted.upload_converted_checkpoint(**kwargs)

    assert result["manifest_version_id"]
    assert all(len(entries) == 1 for entries in runner.objects.values())
    assert all(runner.objects[key][0]["version_id"] == version for key, version in versions_before_resume.items())
    assert runner.mutations[-1][1] == target.manifest_key


def test_upload_rejects_prefix_history_without_the_exact_claim(tmp_path):
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
    target = converted.converted_s3_target(
        "s3://pi05-repro-752160877725-us-east-2/checkpoints", spec, manifest["source"]["revision"]
    )
    runner = FakeVersionedS3()
    runner.seed_object(f"{target.prefix}/unexpected", b"contamination")

    with pytest.raises(repro_stage_data.StageError, match="without the exact publication claim"):
        converted.upload_converted_checkpoint(
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


def test_large_object_path_uses_conditional_multipart_completion(tmp_path, monkeypatch):
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
    runner = FakeVersionedS3()
    monkeypatch.setattr(converted, "SINGLE_PUT_LIMIT_BYTES", 1)

    converted.upload_converted_checkpoint(
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

    completes = [call for call in runner.calls if call[1:3] == ["s3api", "complete-multipart-upload"]]
    assert completes
    assert all(call[call.index("--if-none-match") + 1] == "*" for call in completes)
    assert all(len(entries) == 1 for entries in runner.objects.values())


def test_multipart_source_mutation_aborts_before_create_once_completion(tmp_path, monkeypatch):
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
    target = converted.converted_s3_target(
        "s3://pi05-repro-752160877725-us-east-2/checkpoints", spec, manifest["source"]["revision"]
    )
    model_key = f"{target.prefix}/checkpoint/model.safetensors"
    runner = FakeVersionedS3()
    runner.mutate_source_on_upload_key = model_key
    runner.mutate_source_path = output / "model.safetensors"
    monkeypatch.setattr(converted, "SINGLE_PUT_LIMIT_BYTES", 1)

    with pytest.raises(repro_stage_data.StageError, match="multipart source changed"):
        converted.upload_converted_checkpoint(
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

    assert model_key not in runner.objects
    assert not runner.multipart
    assert not any(
        call[1:3] == ["s3api", "complete-multipart-upload"] and call[call.index("--key") + 1] == model_key
        for call in runner.calls
    )
