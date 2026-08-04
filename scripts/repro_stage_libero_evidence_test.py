# ruff: noqa: SLF001

import base64
import hashlib
import json
import pathlib

import pytest

from scripts import repro_stage_data
from scripts import repro_stage_libero_evidence as evidence

EVALUATOR_SOURCE_COMMIT = evidence.EXPECTED_EVALUATOR_SOURCE_COMMIT
EVALUATOR_IMAGE_DIGEST = evidence.EXPECTED_EVALUATOR_IMAGE_DIGEST
PARENT_POLICY_SOURCE_COMMIT = evidence.EXPECTED_PARENT_POLICY_SOURCE_COMMIT
PARENT_POLICY_IMAGE_DIGEST = evidence.EXPECTED_PARENT_POLICY_IMAGE_DIGEST
MODEL_REVISION = evidence.EXPECTED_MODEL_REVISION
INSTANCE_ID = "i-0123456789abcdef0"
RUN_ID = "libero-base-runtime-smoke-05"
COST_LEDGER_S3_URI = f"s3://{evidence.EXPECTED_BUCKET}/control/cost-ledger.json"
COST_LEDGER_VERSION_ID = "ledger-version-1"
INSTANCE_IDENTITY = {
    "accountId": evidence.EXPECTED_ACCOUNT,
    "region": evidence.EXPECTED_REGION,
    "instanceType": evidence.EXPECTED_INSTANCE_TYPE,
    "instanceId": INSTANCE_ID,
}
TRACKED_DESCRIPTOR = pathlib.Path(__file__).parents[1] / "repro/libero-teacher-pytorch.worker-artifact.json"
TRACKED_DESCRIPTOR_SHA256 = evidence.EXPECTED_CONVERTED_ARTIFACT_FILE_SHA256


@pytest.fixture(autouse=True)
def _fixed_production_conversion_revision(monkeypatch, tmp_path):
    # The real 7.5 GB manifest hashes to EXPECTED_MODEL_REVISION.  Tiny test
    # bytes cannot reproduce that preimage, so retain every other check while
    # pinning the already-unit-tested canonical revision helper's result.
    monkeypatch.setattr(evidence.converted, "conversion_revision", lambda _identity: MODEL_REVISION)
    _, _, artifact_path = _converted_fixture(tmp_path / "expected-artifact")
    monkeypatch.setattr(
        evidence,
        "EXPECTED_CONVERTED_ARTIFACT_FILE_SHA256",
        hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
    )


def _write_json(path: pathlib.Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def test_tracked_converted_teacher_descriptor_is_the_exact_published_artifact():
    assert hashlib.sha256(TRACKED_DESCRIPTOR.read_bytes()).hexdigest() == TRACKED_DESCRIPTOR_SHA256
    assert json.loads(TRACKED_DESCRIPTOR.read_text()) == {
        "name": "libero_teacher_pytorch",
        "kind": "checkpoint",
        "revision": MODEL_REVISION,
        "manifest": {
            "s3_uri": (
                f"s3://{evidence.EXPECTED_BUCKET}/checkpoints/pi05_libero_pytorch/{MODEL_REVISION}/manifest.sha256.json"
            ),
            "version_id": "2lhXK.lU9urPfUKPftPS._nqx_fFyTZa",
            "sha256": "b1eb42ac73351d749587e3c3fd1667bc140610819e73408099d39b262cd08daa",
        },
        "payload_s3_uri": (
            f"s3://{evidence.EXPECTED_BUCKET}/checkpoints/pi05_libero_pytorch/{MODEL_REVISION}/checkpoint/"
        ),
        "payload_objects": [
            {
                "path": "assets/physical-intelligence/libero/norm_stats.json",
                "version_id": "lovWxnfPjGXaumqqymRFSooyROWg0QqA",
                "sha256": "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84",
            },
            {
                "path": "config.json",
                "version_id": "xkFsFoARHCXyKpuweYp7sSckoQcN24Ef",
                "sha256": "272039e8c92478c0cfce91491224d54b20d57225525099a368a6cc625b3e9ec6",
            },
            {
                "path": "model.safetensors",
                "version_id": "CSsniON0z0hrMv7LrnLO0v9qnPlxYWGw",
                "sha256": "c1efa01d7e5b97edd7880acacd85885dbd38d97cf826717d089b29d02184f44a",
            },
        ],
        "destination": "pi05_libero_pytorch",
    }


def _artifact_record(path: pathlib.Path, root: pathlib.Path):
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _command(projected_cost="1.0"):
    return evidence._expected_command(MODEL_REVISION, projected_cost)


def _child_commands():
    commands = [
        [
            "/usr/local/bin/python",
            "scripts/serve_policy.py",
            "--env",
            "LIBERO",
            "--port",
            "8000",
            "--seed",
            "7",
            "policy:checkpoint",
            "--policy.config",
            "pi05_libero",
            "--policy.dir",
            "/mnt/openpi/checkpoints/pi05_libero",
        ]
    ]
    commands.extend(
        [
            "/opt/libero-venv/bin/python",
            "examples/libero/main.py",
            "--args.host",
            "127.0.0.1",
            "--args.port",
            "8000",
            "--args.task-suite-name",
            suite,
            "--args.num-trials-per-task",
            "1",
            "--args.stage",
            "base",
            "--args.seed",
            "7",
            "--args.results-out-path",
            f"/output/artifacts/libero/base/{suite}.jsonl",
            "--args.runtime-contract-path",
            "/opt/libero-evaluator-contract.json",
            "--args.expected-libero-revision",
            evidence.EXPECTED_SIMULATOR["revision"],
            "--args.no-save-videos",
        ]
        for suite in evidence.SUITES
    )
    return commands


def _converted_fixture(tmp_path: pathlib.Path):
    checkpoint_root = tmp_path / "pi05_libero_pytorch"
    (checkpoint_root / "assets/physical-intelligence/libero").mkdir(parents=True)
    (checkpoint_root / "model.safetensors").write_bytes(b"small-test-model")
    _write_json(
        checkpoint_root / "config.json",
        {"config_name": "pi05_libero", "precision": "bfloat16", "pi05": True},
    )
    _write_json(
        checkpoint_root / "assets/physical-intelligence/libero/norm_stats.json",
        {"mean": [0.0], "std": [1.0]},
    )
    files = [_artifact_record(path, checkpoint_root) for path in sorted(checkpoint_root.rglob("*")) if path.is_file()]
    converted_manifest = {
        "schema_version": 1,
        "source": {
            "provider": "openpi-jax-to-pytorch",
            "revision_kind": "converted-checkpoint-content-and-provenance-sha256",
            "revision": MODEL_REVISION,
            "upstream": {
                "provider": "gcs",
                "uri": "gs://openpi-assets/checkpoints/pi05_libero",
                "revision": "d" * 64,
            },
        },
        "conversion": {
            "source_commit": PARENT_POLICY_SOURCE_COMMIT,
            "image_digest": PARENT_POLICY_IMAGE_DIGEST,
            "converter": "examples/convert_jax_model_to_pytorch.py",
            "config_name": "pi05_libero",
            "precision": "bfloat16",
        },
        "checkpoint": {
            "key": "libero",
            "local_dirname": "pi05_libero_pytorch",
            "format": "pytorch-safetensors",
        },
        "totals": {"files": len(files), "bytes": sum(item["bytes"] for item in files)},
        "files": files,
    }
    converted_manifest_path = tmp_path / "pi05_libero_pytorch.converted-manifest.json"
    _write_json(converted_manifest_path, converted_manifest)
    target = evidence.converted.converted_s3_target(
        f"s3://{evidence.EXPECTED_BUCKET}/checkpoints",
        evidence.converted.CONVERTED_CHECKPOINTS["libero"],
        MODEL_REVISION,
    )
    artifact = {
        "name": "libero_teacher_pytorch",
        "kind": "checkpoint",
        "revision": MODEL_REVISION,
        "manifest": {
            "s3_uri": target.manifest_uri,
            "version_id": "manifest-version-1",
            "sha256": hashlib.sha256(converted_manifest_path.read_bytes()).hexdigest(),
        },
        "payload_s3_uri": target.snapshot_uri,
        "payload_objects": [
            {"path": item["path"], "version_id": f"payload-version-{index}", "sha256": item["sha256"]}
            for index, item in enumerate(files)
        ],
        "destination": "pi05_libero_pytorch",
    }
    artifact_path = tmp_path / "libero-teacher-worker-artifact.json"
    _write_json(artifact_path, artifact)
    return checkpoint_root, converted_manifest_path, artifact_path


def _cost_ledger_fixture(tmp_path: pathlib.Path):
    path = tmp_path / "cost-ledger.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "entries": [
                {
                    "id": "cost-entry-1",
                    "category": "workbench_setup",
                    "instance_type": evidence.EXPECTED_INSTANCE_TYPE,
                    "instance_ids": [INSTANCE_ID],
                    "created_at": "2026-08-04T09:00:00Z",
                    "deadline_utc": "2026-08-04T11:00:00Z",
                    "state": "launched",
                    "usd": 2.0,
                }
            ],
        },
    )
    return path


def _fixture(tmp_path: pathlib.Path) -> pathlib.Path:
    _converted_fixture(tmp_path)
    _cost_ledger_fixture(tmp_path)
    root = tmp_path / "smoke"
    artifact_root = root / "artifacts/libero/base"
    artifact_root.mkdir(parents=True)
    all_records = []
    suite_metrics = {}
    for suite in evidence.SUITES:
        records = []
        for task_id in range(10):
            success = suite != "libero_spatial" or task_id != 0
            records.append(
                {
                    "pair_id": f"libero:{suite}:task-{task_id:03d}:init-000:seed-7",
                    "stage": "base",
                    "benchmark": "libero",
                    "suite": suite,
                    "task": f"task {task_id}",
                    "task_id": task_id,
                    "success": success,
                    "seed": 7,
                    "init_index": 0,
                    "steps": 20 + task_id if success else evidence.MAX_EPISODE_STEPS[suite],
                    "libero_revision": evidence.EXPECTED_SIMULATOR["revision"],
                }
            )
        suite_path = artifact_root / f"{suite}.jsonl"
        suite_path.write_text(
            "".join(json.dumps(item, separators=(",", ":"), sort_keys=True) + "\n" for item in records)
        )
        all_records.extend(records)
        successes = sum(item["success"] for item in records)
        suite_metrics[suite] = {"episodes": 10, "successes": successes, "success_rate": successes / 10}
    combined_path = artifact_root / "episodes.jsonl"
    combined_path.write_bytes(b"".join((artifact_root / f"{suite}.jsonl").read_bytes() for suite in evidence.SUITES))
    successes = sum(item["success"] for item in all_records)
    metrics = {
        "episodes": 40,
        "successes": successes,
        "success_rate": successes / 40,
        "environment_steps": sum(item["steps"] for item in all_records),
        "infrastructure_errors": 0,
        "suites": suite_metrics,
    }
    artifacts = [_artifact_record(artifact_root / f"{suite}.jsonl", root) for suite in evidence.SUITES]
    artifacts.append(_artifact_record(combined_path, root))
    manifest = {
        "schema_version": 1,
        "project": "pi05-aws-repro",
        "kind": "libero-evaluation",
        "run_id": RUN_ID,
        "started_at": "2026-08-04T10:00:01+00:00",
        "finished_at": "2026-08-04T10:00:02+00:00",
        "source": {"commit": EVALUATOR_SOURCE_COMMIT},
        "image": {"digest": EVALUATOR_IMAGE_DIGEST},
        "dataset": {
            "name": "LIBERO fixed benchmark assets",
            "revision": evidence.EXPECTED_SIMULATOR["revision"],
        },
        "simulator": evidence.EXPECTED_SIMULATOR,
        "dependencies": evidence.EXPECTED_DEPENDENCIES,
        "policy": {
            "backend": "eager",
            "config": "pi05_libero",
            "checkpoint": "/mnt/openpi/checkpoints/pi05_libero",
            "model_revision": MODEL_REVISION,
        },
        "evaluation": {
            "stage": "base",
            "seed": 7,
            "suites": list(evidence.SUITES),
            "trials_per_task": 1,
            "metrics": metrics,
        },
        "command": _command(),
        "child_commands": _child_commands(),
        "instance": {
            "type": "g6e.4xlarge",
            "id": INSTANCE_ID,
            "identity_recorded_by": "worker run manifest",
        },
        "cost": {"projected_usd": 1.0, "actual_recorded_by": "worker run manifest"},
        "artifacts": artifacts,
    }
    _write_json(root / "manifests/libero-base.json", manifest)
    (root / "replay.log").write_text("policy server ready\n40 episodes complete\n")
    _write_json(
        root / "timing.json",
        {"started_at": "2026-08-04T10:00:00Z", "finished_at": "2026-08-04T10:00:03Z", "exit_code": 0},
    )
    return root


def _validate(root, **overrides):
    kwargs = {
        "run_id": RUN_ID,
        "evaluator_source_commit": EVALUATOR_SOURCE_COMMIT,
        "evaluator_image_digest": EVALUATOR_IMAGE_DIGEST,
        "parent_policy_source_commit": PARENT_POLICY_SOURCE_COMMIT,
        "parent_policy_image_digest": PARENT_POLICY_IMAGE_DIGEST,
        "model_revision": MODEL_REVISION,
        "instance_id": INSTANCE_ID,
        "instance_identity": INSTANCE_IDENTITY,
        "checkpoint_root": root.parent / "pi05_libero_pytorch",
        "converted_manifest_path": root.parent / "pi05_libero_pytorch.converted-manifest.json",
        "converted_checkpoint_artifact_path": root.parent / "libero-teacher-worker-artifact.json",
        "cost_ledger_path": root.parent / "cost-ledger.json",
        "cost_ledger_s3_uri": COST_LEDGER_S3_URI,
        "cost_ledger_version_id": COST_LEDGER_VERSION_ID,
        "cost_ledger_sha256": hashlib.sha256((root.parent / "cost-ledger.json").read_bytes()).hexdigest(),
    }
    kwargs.update(overrides)
    return evidence.validate_smoke(root, **kwargs)


class FakeVersionedS3:
    def __init__(self):
        self.objects = {}
        self.delete_markers = []
        self.calls = []
        self._version = 0
        self.fail_put_key_once = None

    @staticmethod
    def _arg(argv, name):
        return argv[argv.index(name) + 1]

    def seed(self, key, payload=b"unrelated", *, version=None):
        self._version += 1
        self.objects.setdefault(key, []).append(
            {
                "version": version or f"version-{self._version}",
                "data": payload,
                "metadata": {},
                "checksum": base64.b64encode(hashlib.sha256(payload).digest()).decode(),
            }
        )

    def __call__(self, raw_argv):
        argv = list(raw_argv)
        self.calls.append(argv)
        operation = tuple(argv[1:3])
        if operation == ("sts", "get-caller-identity"):
            return json.dumps({"Account": evidence.EXPECTED_ACCOUNT})
        if operation == ("s3api", "get-bucket-location"):
            return json.dumps({"LocationConstraint": evidence.EXPECTED_REGION})
        if operation == ("s3api", "get-bucket-versioning"):
            return json.dumps({"Status": "Enabled"})
        if operation == ("s3api", "get-bucket-encryption"):
            return json.dumps({"ServerSideEncryptionConfiguration": {"Rules": [{}]}})
        if operation == ("s3api", "list-multipart-uploads"):
            return json.dumps({"Uploads": [], "IsTruncated": False})
        if operation == ("s3api", "list-object-versions"):
            prefix = self._arg(argv, "--prefix")
            versions = []
            for key, entries in self.objects.items():
                if key.startswith(prefix):
                    versions.extend(
                        {
                            "Key": key,
                            "VersionId": entry["version"],
                            "IsLatest": index == len(entries) - 1,
                        }
                        for index, entry in enumerate(entries)
                    )
            markers = [item for item in self.delete_markers if item["Key"].startswith(prefix)]
            return json.dumps({"Versions": versions, "DeleteMarkers": markers, "IsTruncated": False})
        if operation == ("s3api", "put-object"):
            assert self._arg(argv, "--if-none-match") == "*"
            assert self._arg(argv, "--server-side-encryption") == "AES256"
            assert self._arg(argv, "--checksum-algorithm") == "SHA256"
            key = self._arg(argv, "--key")
            if key == self.fail_put_key_once:
                self.fail_put_key_once = None
                raise repro_stage_data.StageError(f"injected PUT interruption for {key}")
            if self.objects.get(key):
                raise repro_stage_data.StageError("conditional put failed")
            payload = pathlib.Path(self._arg(argv, "--body")).read_bytes()
            assert self._arg(argv, "--checksum-sha256") == base64.b64encode(hashlib.sha256(payload).digest()).decode()
            metadata = json.loads(self._arg(argv, "--metadata"))
            self._version += 1
            version = f"version-{self._version}"
            self.objects.setdefault(key, []).append(
                {
                    "version": version,
                    "data": payload,
                    "metadata": metadata,
                    "checksum": base64.b64encode(hashlib.sha256(payload).digest()).decode(),
                }
            )
            return json.dumps({"VersionId": version})
        if operation == ("s3api", "head-object"):
            key = self._arg(argv, "--key")
            version = self._arg(argv, "--version-id")
            try:
                item = next(item for item in self.objects[key] if item["version"] == version)
            except (KeyError, StopIteration) as exc:
                raise repro_stage_data.StageError(f"missing exact version {version} for {key}") from exc
            return json.dumps(
                {
                    "VersionId": version,
                    "ContentLength": len(item["data"]),
                    "ServerSideEncryption": "AES256",
                    "Metadata": item["metadata"],
                    "ChecksumSHA256": item["checksum"],
                    "ETag": f'"{hashlib.md5(item["data"], usedforsecurity=False).hexdigest()}"',
                }
            )
        if operation == ("s3api", "get-object"):
            key = self._arg(argv, "--key")
            version = self._arg(argv, "--version-id")
            try:
                item = next(item for item in self.objects[key] if item["version"] == version)
            except (KeyError, StopIteration) as exc:
                raise repro_stage_data.StageError(f"missing exact version {version} for {key}") from exc
            pathlib.Path(argv[-1]).write_bytes(item["data"])
            return json.dumps({"VersionId": version, "ChecksumSHA256": item["checksum"]})
        raise AssertionError(argv)


def _s3_key(uri):
    return uri.split("/", 3)[3]


def _seed_external_inputs(fake, sealed):
    artifact = sealed.content["model"]["converted_checkpoint"]["artifact"]
    manifest = artifact["manifest"]
    fake.seed(
        _s3_key(manifest["s3_uri"]),
        sealed.converted_manifest_path.read_bytes(),
        version=manifest["version_id"],
    )
    for item in artifact["payload_objects"]:
        fake.seed(
            _s3_key(f"{artifact['payload_s3_uri']}{item['path']}"),
            (sealed.checkpoint_root / item["path"]).read_bytes(),
            version=item["version_id"],
        )
    fake.seed(
        _s3_key(sealed.cost_ledger["s3_uri"]),
        sealed.cost_ledger_path.read_bytes(),
        version=sealed.cost_ledger["version_id"],
    )


def _prefix_objects(fake, target):
    return {key: value for key, value in fake.objects.items() if key.startswith(f"{target.prefix}/")}


def test_valid_smoke_has_deterministic_revision_and_exact_inventory(tmp_path):
    root = _fixture(tmp_path)
    first = _validate(root)
    second = _validate(root)

    assert first.revision == second.revision
    assert len(first.files) == 8
    assert [item["path"] for item in first.files] == sorted(
        [
            *(f"artifacts/libero/base/{suite}.jsonl" for suite in evidence.SUITES),
            "artifacts/libero/base/episodes.jsonl",
            "manifests/libero-base.json",
            "replay.log",
            "timing.json",
        ]
    )
    assert first.content["evaluation"]["metrics"]["episodes"] == 40
    assert first.content["instance"]["account_id"] == evidence.EXPECTED_ACCOUNT
    assert first.content["model"]["converted_checkpoint"]["local_bytes_verified"] is True
    assert first.content["cost"]["ledger"]["version_id"] == COST_LEDGER_VERSION_ID


def test_rejects_wrong_live_instance_identity(tmp_path):
    root = _fixture(tmp_path)
    identity = dict(INSTANCE_IDENTITY, accountId="000000000000")

    with pytest.raises(repro_stage_data.StageError, match="fresh IMDSv2 identity"):
        _validate(root, instance_identity=identity)


def test_rejects_failed_or_unaccepted_attempt_id(tmp_path):
    root = _fixture(tmp_path)

    with pytest.raises(repro_stage_data.StageError, match="accepted clean replay"):
        _validate(root, run_id="libero-base-runtime-smoke-01")


def test_rejects_unreviewed_evaluator_source(tmp_path):
    root = _fixture(tmp_path)

    with pytest.raises(repro_stage_data.StageError, match="base smoke evaluator source"):
        _validate(root, evaluator_source_commit="f" * 40)


def test_rejects_unreviewed_evaluator_image(tmp_path):
    root = _fixture(tmp_path)

    with pytest.raises(repro_stage_data.StageError, match="base smoke evaluator image"):
        _validate(root, evaluator_image_digest=f"sha256:{'f' * 64}")


def test_rejects_changed_local_converted_checkpoint_bytes(tmp_path):
    root = _fixture(tmp_path)
    (root.parent / "pi05_libero_pytorch/model.safetensors").write_bytes(b"changed-after-conversion")

    with pytest.raises(repro_stage_data.StageError, match="local converted checkpoint bytes"):
        _validate(root)


def test_rejects_nonexact_converted_checkpoint_descriptor_file(tmp_path):
    root = _fixture(tmp_path)
    artifact_path = root.parent / "libero-teacher-worker-artifact.json"
    artifact_path.write_text(artifact_path.read_text() + "\n")

    with pytest.raises(repro_stage_data.StageError, match="exact tracked published descriptor"):
        _validate(root)


def test_rejects_converted_input_path_through_an_ancestor_symlink(tmp_path):
    root = _fixture(tmp_path)
    alias = root.parent / "host-alias"
    alias.symlink_to(root.parent, target_is_directory=True)
    aliased_manifest = alias / "pi05_libero_pytorch.converted-manifest.json"
    assert not aliased_manifest.is_symlink()

    with pytest.raises(repro_stage_data.StageError, match="must be normalized"):
        _validate(root, converted_manifest_path=aliased_manifest)


def test_rejects_unpinned_or_wrong_cost_ledger(tmp_path):
    root = _fixture(tmp_path)

    with pytest.raises(repro_stage_data.StageError, match="control/cost-ledger.json"):
        _validate(root, cost_ledger_s3_uri=f"s3://{evidence.EXPECTED_BUCKET}/control/other.json")


def test_rejects_cost_ledger_without_covering_paid_instance_entry(tmp_path):
    root = _fixture(tmp_path)
    ledger_path = root.parent / "cost-ledger.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["entries"][0]["instance_ids"] = ["i-fffffffffffffffff"]
    _write_json(ledger_path, ledger)

    with pytest.raises(repro_stage_data.StageError, match="no paid entry covering"):
        _validate(root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda root: (root / "replay.log").write_bytes(b""), "replay.log must be non-empty"),
        (
            lambda root: _write_json(
                root / "timing.json",
                {"started_at": "2026-08-04T10:00:00Z", "finished_at": "2026-08-04T10:00:03Z", "exit_code": 1},
            ),
            "exit_code=0",
        ),
        (
            lambda root: _write_json(
                root / "timing.json",
                {"started_at": "2026-08-04T10:00:00Z", "finished_at": "2026-08-04T10:00:03Z", "exit_code": 0.0},
            ),
            "exit_code=0",
        ),
        (
            lambda root: _write_json(
                root / "timing.json",
                {"started_at": "2026-08-04T10:00:02Z", "finished_at": "2026-08-04T10:00:03Z", "exit_code": 0},
            ),
            "does not enclose",
        ),
    ],
)
def test_rejects_missing_successful_replay_evidence(tmp_path, mutation, message):
    root = _fixture(tmp_path)
    mutation(root)
    with pytest.raises(repro_stage_data.StageError, match=message):
        _validate(root)


def test_rejects_episode_or_manifest_hash_tampering(tmp_path):
    root = _fixture(tmp_path)
    suite_path = root / "artifacts/libero/base/libero_spatial.jsonl"
    records = [json.loads(line) for line in suite_path.read_text().splitlines()]
    records[0]["task"] += " tampered"
    suite_path.write_text("".join(json.dumps(item, separators=(",", ":"), sort_keys=True) + "\n" for item in records))

    with pytest.raises(repro_stage_data.StageError, match="combined episodes"):
        _validate(root)


def test_rejects_fully_rederived_results_with_unobserved_suite_metrics(tmp_path):
    root = _fixture(tmp_path)
    artifact_root = root / "artifacts/libero/base"
    spatial_path = artifact_root / "libero_spatial.jsonl"
    spatial = [json.loads(line) for line in spatial_path.read_text().splitlines()]
    spatial[1]["success"] = False
    spatial[1]["steps"] = evidence.MAX_EPISODE_STEPS["libero_spatial"]
    spatial_path.write_text("".join(json.dumps(item, separators=(",", ":"), sort_keys=True) + "\n" for item in spatial))
    combined_path = artifact_root / "episodes.jsonl"
    combined_path.write_bytes(b"".join((artifact_root / f"{suite}.jsonl").read_bytes() for suite in evidence.SUITES))

    all_records = []
    suite_metrics = {}
    for suite in evidence.SUITES:
        records = [json.loads(line) for line in (artifact_root / f"{suite}.jsonl").read_text().splitlines()]
        all_records.extend(records)
        successes = sum(item["success"] for item in records)
        suite_metrics[suite] = {"episodes": 10, "successes": successes, "success_rate": successes / 10}
    successes = sum(item["success"] for item in all_records)
    metrics = {
        "episodes": 40,
        "successes": successes,
        "success_rate": successes / 40,
        "environment_steps": sum(item["steps"] for item in all_records),
        "infrastructure_errors": 0,
        "suites": suite_metrics,
    }
    manifest_path = root / "manifests/libero-base.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evaluation"]["metrics"] = metrics
    manifest["artifacts"] = [_artifact_record(artifact_root / f"{suite}.jsonl", root) for suite in evidence.SUITES]
    manifest["artifacts"].append(_artifact_record(combined_path, root))
    _write_json(manifest_path, manifest)

    with pytest.raises(repro_stage_data.StageError, match="success count differs from accepted attempts"):
        _validate(root)


@pytest.mark.parametrize(
    ("success", "steps"),
    [
        (True, 10),
        (False, evidence.MAX_EPISODE_STEPS["libero_spatial"] - 1),
    ],
)
def test_rejects_impossible_success_or_failure_step_counts(tmp_path, success, steps):
    root = _fixture(tmp_path)
    suite_path = root / "artifacts/libero/base/libero_spatial.jsonl"
    records = [json.loads(line) for line in suite_path.read_text().splitlines()]
    records[0]["success"] = success
    records[0]["steps"] = steps
    suite_path.write_text("".join(json.dumps(item, separators=(",", ":"), sort_keys=True) + "\n" for item in records))
    combined = root / "artifacts/libero/base/episodes.jsonl"
    combined.write_bytes(
        b"".join((root / f"artifacts/libero/base/{suite}.jsonl").read_bytes() for suite in evidence.SUITES)
    )

    with pytest.raises(repro_stage_data.StageError, match="rollout-step semantics"):
        _validate(root)


def test_claim_first_manifest_last_publication_is_version_pinned_and_idempotent(tmp_path):
    sealed = _validate(_fixture(tmp_path))
    fake = FakeVersionedS3()
    _seed_external_inputs(fake, sealed)
    config = {"aws": {"account_id": evidence.EXPECTED_ACCOUNT, "region": evidence.EXPECTED_REGION}}
    kwargs = {
        "s3_root": f"s3://{evidence.EXPECTED_BUCKET}/manual-smoke/libero",
        "runner": fake,
        "environ": {"AWS_REGION": evidence.EXPECTED_REGION},
    }

    target = evidence._parse_target(kwargs["s3_root"], sealed)
    first = evidence.upload_smoke(config, sealed, **kwargs)
    stored_keys = list(_prefix_objects(fake, target))
    assert len(stored_keys) == 11
    assert stored_keys[0].endswith("/publication-claim.json")
    assert stored_keys[-1].endswith("/manifest.sha256.json")
    assert len(first["publication"]["payload"]) == 8
    assert len(first["verified_inputs"]["converted_checkpoint"]["payload"]) == 3
    assert all(len(versions) == 1 for versions in fake.objects.values())
    get_calls = [call for call in fake.calls if tuple(call[1:3]) == ("s3api", "get-object")]
    assert len(get_calls) == 16
    assert all("--version-id" in call for call in get_calls)

    second = evidence.upload_smoke(config, sealed, **kwargs)
    assert second["manifest"]["version_id"] == first["manifest"]["version_id"]
    assert all(len(versions) == 1 for versions in fake.objects.values())


def test_publication_rejects_history_without_exact_claim(tmp_path):
    sealed = _validate(_fixture(tmp_path))
    fake = FakeVersionedS3()
    _seed_external_inputs(fake, sealed)
    target = evidence._parse_target(f"s3://{evidence.EXPECTED_BUCKET}/manual-smoke/libero", sealed)
    fake.seed(f"{target.prefix}/output/unexpected")
    config = {"aws": {"account_id": evidence.EXPECTED_ACCOUNT, "region": evidence.EXPECTED_REGION}}

    with pytest.raises(repro_stage_data.StageError, match="without the exact publication claim"):
        evidence.upload_smoke(
            config,
            sealed,
            s3_root=f"s3://{evidence.EXPECTED_BUCKET}/manual-smoke/libero",
            runner=fake,
            environ={"AWS_REGION": evidence.EXPECTED_REGION},
        )


def test_publication_rejects_nonexistent_converted_checkpoint_version(tmp_path):
    sealed = _validate(_fixture(tmp_path))
    fake = FakeVersionedS3()
    _seed_external_inputs(fake, sealed)
    manifest = sealed.content["model"]["converted_checkpoint"]["artifact"]["manifest"]
    fake.objects[_s3_key(manifest["s3_uri"])][0]["version"] = "different-version"
    config = {"aws": {"account_id": evidence.EXPECTED_ACCOUNT, "region": evidence.EXPECTED_REGION}}

    with pytest.raises(repro_stage_data.StageError, match="missing exact version"):
        evidence.upload_smoke(
            config,
            sealed,
            s3_root=f"s3://{evidence.EXPECTED_BUCKET}/manual-smoke/libero",
            runner=fake,
            environ={"AWS_REGION": evidence.EXPECTED_REGION},
        )


def test_publication_never_repairs_payloads_after_a_terminal_object(tmp_path):
    sealed = _validate(_fixture(tmp_path))
    fake = FakeVersionedS3()
    _seed_external_inputs(fake, sealed)
    config = {"aws": {"account_id": evidence.EXPECTED_ACCOUNT, "region": evidence.EXPECTED_REGION}}
    kwargs = {
        "s3_root": f"s3://{evidence.EXPECTED_BUCKET}/manual-smoke/libero",
        "runner": fake,
        "environ": {"AWS_REGION": evidence.EXPECTED_REGION},
    }
    result = evidence.upload_smoke(config, sealed, **kwargs)
    missing_key = result["publication"]["payload"][0]["key"]
    del fake.objects[missing_key]

    with pytest.raises(repro_stage_data.StageError, match="terminal LIBERO manifest"):
        evidence.upload_smoke(config, sealed, **kwargs)
    assert missing_key not in fake.objects


def test_publication_resumes_after_receipt_before_manifest_without_new_versions(tmp_path):
    sealed = _validate(_fixture(tmp_path))
    fake = FakeVersionedS3()
    _seed_external_inputs(fake, sealed)
    config = {"aws": {"account_id": evidence.EXPECTED_ACCOUNT, "region": evidence.EXPECTED_REGION}}
    kwargs = {
        "s3_root": f"s3://{evidence.EXPECTED_BUCKET}/manual-smoke/libero",
        "runner": fake,
        "environ": {"AWS_REGION": evidence.EXPECTED_REGION},
    }
    target = evidence._parse_target(kwargs["s3_root"], sealed)
    fake.fail_put_key_once = target.manifest_key

    with pytest.raises(repro_stage_data.StageError, match="injected PUT interruption"):
        evidence.upload_smoke(config, sealed, **kwargs)
    assert len(_prefix_objects(fake, target)) == 10
    assert any(key.endswith("/publication-receipt.json") for key in _prefix_objects(fake, target))
    assert target.manifest_key not in fake.objects

    result = evidence.upload_smoke(config, sealed, **kwargs)
    assert result["manifest"]["key"] == target.manifest_key
    assert len(_prefix_objects(fake, target)) == 11
    assert all(len(versions) == 1 for versions in fake.objects.values())


def test_upload_without_execute_is_a_local_only_plan(tmp_path, monkeypatch, capsys):
    root = _fixture(tmp_path)
    monkeypatch.setattr(evidence, "upload_smoke", lambda *_args, **_kwargs: pytest.fail("AWS path was invoked"))
    monkeypatch.setattr(evidence.repro_worker, "get_instance_identity", lambda: INSTANCE_IDENTITY)

    assert (
        evidence.main(
            [
                "upload",
                "--output-root",
                str(root),
                "--run-id",
                RUN_ID,
                "--evaluator-source-commit",
                EVALUATOR_SOURCE_COMMIT,
                "--evaluator-image-digest",
                EVALUATOR_IMAGE_DIGEST,
                "--parent-policy-source-commit",
                PARENT_POLICY_SOURCE_COMMIT,
                "--parent-policy-image-digest",
                PARENT_POLICY_IMAGE_DIGEST,
                "--model-revision",
                MODEL_REVISION,
                "--instance-id",
                INSTANCE_ID,
                "--checkpoint-root",
                str(root.parent / "pi05_libero_pytorch"),
                "--converted-manifest",
                str(root.parent / "pi05_libero_pytorch.converted-manifest.json"),
                "--converted-checkpoint-artifact",
                str(root.parent / "libero-teacher-worker-artifact.json"),
                "--cost-ledger-path",
                str(root.parent / "cost-ledger.json"),
                "--cost-ledger-s3-uri",
                COST_LEDGER_S3_URI,
                "--cost-ledger-version-id",
                COST_LEDGER_VERSION_ID,
                "--cost-ledger-sha256",
                hashlib.sha256((root.parent / "cost-ledger.json").read_bytes()).hexdigest(),
                "--s3-root",
                f"s3://{evidence.EXPECTED_BUCKET}/manual-smoke/libero",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "dry-run"
    assert result["mutations_authorized"] is False
    assert result["destination"]["requires_execute"] is True
