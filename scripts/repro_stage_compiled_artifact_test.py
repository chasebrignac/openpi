import copy
import dataclasses
import hashlib
import json
import pathlib

import pytest

from scripts import repro_stage_compiled_artifact as compiled
from scripts import repro_stage_data
from scripts import repro_worker

ACCOUNT = "752160877725"
REGION = "us-east-2"
BUCKET = "pi05-repro-752160877725-us-east-2"
SOURCE = "a" * 40
DATASET_REVISION = "b" * 40
IMAGE = "sha256:" + "c" * 64
INSTANCE = "i-0123456789abcdef0"
GPU = "GPU-01234567-89ab-cdef-0123-456789abcdef, NVIDIA L40S, 595.71.05"
ORIGINAL_ROOT = pathlib.PurePosixPath("/mnt/openpi/artifacts/libero")


def _source_runner(commit: str = SOURCE, *, dirty: bool = False):
    def runner(argv):
        if list(argv) == ["git", "rev-parse", "HEAD"]:
            return commit
        if list(argv) == ["git", "status", "--porcelain=v1", "--untracked-files=all"]:
            return " M changed.py" if dirty else ""
        raise AssertionError(argv)

    return runner


def _seal(root: pathlib.Path, identity: compiled.DeclaredIdentity):
    return compiled.validate_compiled_artifact(
        root,
        identity,
        source_runner=_source_runner(identity.source_commit),
        environ={"PI05_SOURCE_SHA": identity.source_commit},
    )


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _record(root: pathlib.Path, name: str, *, original: pathlib.PurePosixPath = ORIGINAL_ROOT):
    path = root / name
    return {"path": str(original / name), "bytes": path.stat().st_size, "sha256": repro_stage_data.sha256_file(path)}


def _external_record(name: str, payload: bytes):
    return {
        "path": f"/mnt/openpi/external/{name}",
        "bytes": len(payload),
        "sha256": _sha(payload),
    }


def _source_identity(path: pathlib.Path):
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": repro_stage_data.sha256_file(path)}


def _stage(*, stage: str, artifacts: list[dict], details: dict, metrics: dict | None = None):
    return {
        "schema_version": 1,
        "created_at": "2026-08-03T00:00:00+00:00",
        "stage": stage,
        "track": "libero",
        "source": {"sha": SOURCE, "dirty": False},
        "runtime": {"image_digest": IMAGE, "instance_type": "g7e.4xlarge", "instance_id": INSTANCE},
        "dataset": {"name": "physical-intelligence/libero", "revision": DATASET_REVISION},
        "experiment": {"seed": None, "steps": None},
        "cost": {"reservation_id": "reservation"},
        "command": {"argv": ["command"], "shell": "command"},
        "metrics": metrics or {},
        "artifacts": artifacts,
        "details": details,
    }


def _write_json(path: pathlib.Path, value: dict):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _compiled_fixture(tmp_path: pathlib.Path):
    root = (tmp_path / "compiled").resolve()
    root.mkdir()
    payloads = {
        "encode-prefix.bf16.onnx": b"prefix graph",
        "encode-prefix.bf16.data": b"prefix weights",
        "decode-denoise.bf16.onnx": b"decoder graph",
        "decode-denoise.bf16.data": b"decoder weights",
        "encode-inputs.npz": b"encode inputs",
        "decode-inputs.npz": b"decode inputs",
        "encode-reference.npz": b"encode golden",
        "decode-reference.npz": b"decode golden",
        "encode-prefix.bf16.plan": b"prefix engine",
        "decode-denoise.bf16.plan": b"decoder engine",
        "encode-prefix.bf16.layers.json": b'{"layers": ["prefix"]}\n',
        "decode-denoise.bf16.layers.json": b'{"layers": ["decoder"]}\n',
        "encode-prefix.bf16.trtexec.log": b"prefix build log\n",
        "decode-denoise.bf16.trtexec.log": b"decoder build log\n",
        "tensorrt-bf16.timing.cache": b"timing cache",
    }
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)

    checkpoint = {
        "path": "/mnt/openpi/checkpoints/pi05_libero/model.safetensors",
        "sha256": _sha(b"checkpoint"),
        "assets": [
            {"name": "assets/physical-intelligence/libero/norm_stats.json", "bytes": 5, "sha256": _sha(b"stats")}
        ],
    }
    export_artifacts = [
        _record(root, name)
        for name in (
            "encode-prefix.bf16.onnx",
            "encode-prefix.bf16.data",
            "decode-denoise.bf16.onnx",
            "decode-denoise.bf16.data",
            "encode-inputs.npz",
            "decode-inputs.npz",
            "encode-reference.npz",
            "decode-reference.npz",
        )
    ]
    export_artifacts.extend(
        (
            _external_record("model.safetensors", b"checkpoint"),
            _external_record("norm_stats.json", b"stats"),
        )
    )
    export = _stage(
        stage="onnx-export-bf16",
        artifacts=export_artifacts,
        details={"config": "pi05_libero_l09_snapflow", "checkpoint": checkpoint},
    )
    _write_json(root / "export-manifest.json", export)

    report = {
        "schema_version": 1,
        "precision": "bf16",
        "passes": True,
        "models": {
            "encode-prefix": {
                "model": _source_identity(root / "encode-prefix.bf16.onnx"),
                "external_data": [_source_identity(root / "encode-prefix.bf16.data")],
            },
            "decode-denoise": {
                "model": _source_identity(root / "decode-denoise.bf16.onnx"),
                "external_data": [_source_identity(root / "decode-denoise.bf16.data")],
            },
        },
        "end_to_end_actions": {"bias_passes": True, "action_limits_pass": True},
        "provenance": {
            "track": "libero",
            "dataset": "physical-intelligence/libero",
            "dataset_revision": DATASET_REVISION,
            "image_digest": IMAGE,
            "instance_type": "g7e.4xlarge",
            "instance_id": INSTANCE,
            "cost_reservation": "reservation",
        },
    }
    _write_json(root / "onnx-validation.bf16.json", report)
    validation_artifacts = [
        _record(root, name)
        for name in (
            "encode-prefix.bf16.onnx",
            "encode-prefix.bf16.data",
            "decode-denoise.bf16.onnx",
            "decode-denoise.bf16.data",
            "onnx-validation.bf16.json",
        )
    ]
    validation_artifacts.extend(
        (
            _external_record("action-limits.npz", b"limits"),
            _external_record("action-limits.json", b"limits metadata"),
        )
    )
    validation_manifest = _stage(
        stage="onnx-validation-bf16",
        artifacts=validation_artifacts,
        details=report,
        metrics=report,
    )
    _write_json(root / "onnx-validation-manifest.bf16.json", validation_manifest)

    build_artifacts = [
        _record(root, name)
        for name in (
            "encode-prefix.bf16.plan",
            "decode-denoise.bf16.plan",
            "encode-prefix.bf16.layers.json",
            "decode-denoise.bf16.layers.json",
            "encode-prefix.bf16.trtexec.log",
            "decode-denoise.bf16.trtexec.log",
            "tensorrt-bf16.timing.cache",
            "onnx-validation.bf16.json",
            "export-manifest.json",
        )
    ]
    build = _stage(
        stage="tensorrt-build-bf16",
        artifacts=build_artifacts,
        details={
            "tensorrt_version": "&&&& RUNNING TensorRT.trtexec [TensorRT v110000]",
            "strongly_typed": True,
            "precision_source": "explicit ONNX tensor types and Q/DQ nodes",
            "commands": [["trtexec", "--onnx=prefix"], ["trtexec", "--onnx=decoder"]],
            "gpu_inventory": [GPU],
            "runtime_identity_source": "worker-environment",
            "validation_report": str(ORIGINAL_ROOT / "onnx-validation.bf16.json"),
            "policy_contract": {
                "schema_version": 1,
                "protocol": "openpi-policy-websocket-v1",
                "config": "pi05_libero_l09_snapflow",
                "checkpoint": checkpoint,
                "precision": "bf16",
                "num_denoise_steps": 1,
                "source_manifests": [_source_identity(root / "export-manifest.json")],
                "export_runtime": {
                    "image_digest": IMAGE,
                    "instance_type": "g7e.4xlarge",
                    "instance_id": INSTANCE,
                },
            },
        },
    )
    _write_json(root / "tensorrt-manifest.bf16.json", build)
    identity = compiled.DeclaredIdentity(
        source_commit=SOURCE,
        image_digest=IMAGE,
        track="libero",
        dataset="physical-intelligence/libero",
        dataset_revision=DATASET_REVISION,
        precision="bf16",
        instance_type="g7e.4xlarge",
        instance_id=INSTANCE,
        gpu_inventory=(GPU,),
    )
    return root, identity


def _fp8_fixture(tmp_path: pathlib.Path):
    root, identity = _compiled_fixture(tmp_path)
    for name, payload in {
        "encode-prefix.fp8.onnx": b"fp8 prefix graph",
        "encode-prefix.fp8.data": b"fp8 prefix weights",
        "decode-denoise.fp8.onnx": b"fp8 decoder graph",
        "decode-denoise.fp8.data": b"fp8 decoder weights",
        "modelopt-encode-prefix.log": b"prefix calibration log\n",
        "modelopt-decode-denoise.log": b"decoder calibration log\n",
        "encode-prefix.fp8.plan": b"fp8 prefix engine",
        "decode-denoise.fp8.plan": b"fp8 decoder engine",
        "encode-prefix.fp8.layers.json": b'{"layers": ["fp8-prefix"]}\n',
        "decode-denoise.fp8.layers.json": b'{"layers": ["fp8-decoder"]}\n',
        "encode-prefix.fp8.trtexec.log": b"fp8 prefix build log\n",
        "decode-denoise.fp8.trtexec.log": b"fp8 decoder build log\n",
        "tensorrt-fp8.timing.cache": b"fp8 timing cache",
    }.items():
        (root / name).write_bytes(payload)

    fp8_manifest = _stage(
        stage="modelopt-fp8-ptq",
        artifacts=[
            *[
                _record(root, name)
                for name in (
                    "encode-prefix.fp8.onnx",
                    "encode-prefix.fp8.data",
                    "decode-denoise.fp8.onnx",
                    "decode-denoise.fp8.data",
                    "modelopt-encode-prefix.log",
                    "modelopt-decode-denoise.log",
                    "onnx-validation.bf16.json",
                )
            ],
            _external_record("calibration-manifest.json", b"calibration"),
        ],
        details={
            "quantize_mode": "fp8",
            "calibration_chunks": 1024,
            "bf16_validation_report": str(ORIGINAL_ROOT / "onnx-validation.bf16.json"),
        },
    )
    _write_json(root / "fp8-manifest.json", fp8_manifest)

    report = {
        "schema_version": 1,
        "precision": "fp8",
        "passes": True,
        "models": {
            "encode-prefix": {
                "model": _source_identity(root / "encode-prefix.fp8.onnx"),
                "external_data": [_source_identity(root / "encode-prefix.fp8.data")],
            },
            "decode-denoise": {
                "model": _source_identity(root / "decode-denoise.fp8.onnx"),
                "external_data": [_source_identity(root / "decode-denoise.fp8.data")],
            },
        },
        "end_to_end_actions": {"bias_passes": True, "action_limits_pass": True},
        "provenance": {
            "track": "libero",
            "dataset": "physical-intelligence/libero",
            "dataset_revision": DATASET_REVISION,
            "image_digest": IMAGE,
            "instance_type": "g7e.4xlarge",
            "instance_id": INSTANCE,
        },
    }
    _write_json(root / "onnx-validation.fp8.json", report)
    validation_manifest = _stage(
        stage="onnx-validation-fp8",
        artifacts=[
            *[
                _record(root, name)
                for name in (
                    "encode-prefix.fp8.onnx",
                    "encode-prefix.fp8.data",
                    "decode-denoise.fp8.onnx",
                    "decode-denoise.fp8.data",
                    "onnx-validation.fp8.json",
                )
            ],
            _external_record("action-limits.npz", b"limits"),
            _external_record("action-limits.json", b"limits metadata"),
        ],
        details=report,
        metrics=report,
    )
    _write_json(root / "onnx-validation-manifest.fp8.json", validation_manifest)

    bf16_build = json.loads((root / "tensorrt-manifest.bf16.json").read_text())
    checkpoint = bf16_build["details"]["policy_contract"]["checkpoint"]
    build = _stage(
        stage="tensorrt-build-fp8",
        artifacts=[
            _record(root, name)
            for name in (
                "encode-prefix.fp8.plan",
                "decode-denoise.fp8.plan",
                "encode-prefix.fp8.layers.json",
                "decode-denoise.fp8.layers.json",
                "encode-prefix.fp8.trtexec.log",
                "decode-denoise.fp8.trtexec.log",
                "tensorrt-fp8.timing.cache",
                "onnx-validation.fp8.json",
                "export-manifest.json",
                "fp8-manifest.json",
                "onnx-validation.bf16.json",
            )
        ],
        details={
            "tensorrt_version": "TensorRT version 11.0.0.114",
            "strongly_typed": True,
            "precision_source": "explicit ONNX tensor types and Q/DQ nodes",
            "commands": [["trtexec", "--onnx=prefix-fp8"], ["trtexec", "--onnx=decoder-fp8"]],
            "gpu_inventory": [GPU],
            "validation_report": str(ORIGINAL_ROOT / "onnx-validation.fp8.json"),
            "policy_contract": {
                "schema_version": 1,
                "protocol": "openpi-policy-websocket-v1",
                "config": "pi05_libero_l09_snapflow",
                "checkpoint": checkpoint,
                "precision": "fp8",
                "num_denoise_steps": 1,
                "source_manifests": [
                    _source_identity(root / "export-manifest.json"),
                    _source_identity(root / "fp8-manifest.json"),
                    _source_identity(root / "onnx-validation.bf16.json"),
                ],
                "export_runtime": identity.runtime,
            },
        },
    )
    _write_json(root / "tensorrt-manifest.fp8.json", build)
    return root, dataclasses.replace(identity, precision="fp8")


def test_seal_binds_complete_manifest_closure_and_is_worker_compatible(tmp_path):
    root, identity = _compiled_fixture(tmp_path)
    sealed = _seal(root, identity)
    assert len(sealed.revision) == 64
    assert sealed.totals == {
        "files": len(list(root.iterdir())),
        "bytes": sum(path.stat().st_size for path in root.iterdir()),
    }
    target = compiled.parse_s3_target(f"s3://{BUCKET}/compiled", sealed)
    manifest = compiled.build_publication_manifest(sealed, target)
    descriptor = compiled.worker_artifact_descriptor(
        sealed,
        target,
        manifest_version_id="version-1",
        manifest_sha256="d" * 64,
    )
    assert descriptor == {
        "name": "libero_bf16_engines",
        "kind": "asset",
        "revision": sealed.revision,
        "manifest": {"s3_uri": target.manifest_uri, "version_id": "version-1", "sha256": "d" * 64},
        "payload_s3_uri": target.payload_uri,
        "destination": "tensorrt/libero/bf16",
    }
    assert repro_worker.validate_artifact_manifest(manifest, descriptor) == list(sealed.files)


def test_fp8_seal_validates_quantization_and_bf16_build_chain(tmp_path):
    root, identity = _fp8_fixture(tmp_path)
    sealed = _seal(root, identity)
    assert sealed.build_manifest["stage"] == "tensorrt-build-fp8"
    assert {item["path"] for item in sealed.files} == {path.name for path in root.iterdir()}


def test_fp8_seal_rejects_wrong_calibration_recipe(tmp_path):
    root, identity = _fp8_fixture(tmp_path)
    manifest_path = root / "fp8-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["details"]["calibration_chunks"] = 1023
    _write_json(manifest_path, manifest)
    # Rebind the deliberately changed source manifest in the selected build so
    # the rejection proves the recipe gate rather than stopping at its hash.
    build_path = root / "tensorrt-manifest.fp8.json"
    build = json.loads(build_path.read_text())
    source = _source_identity(manifest_path)
    build["details"]["policy_contract"]["source_manifests"][1] = source
    for record in build["artifacts"]:
        if pathlib.PurePosixPath(record["path"]).name == manifest_path.name:
            record.update({"bytes": source["bytes"], "sha256": source["sha256"]})
    _write_json(build_path, build)
    with pytest.raises(repro_stage_data.StageError, match="1,024-chunk"):
        _seal(root, identity)


@pytest.mark.parametrize("bad_name", ["surprise.txt", "decode-denoise.bf16.plan.partial", ".upload-state"])
def test_seal_rejects_extras_and_partial_files(tmp_path, bad_name):
    root, identity = _compiled_fixture(tmp_path)
    (root / bad_name).write_bytes(b"unexpected")
    with pytest.raises(repro_stage_data.StageError, match="partial|exactly manifest-covered"):
        _seal(root, identity)


def test_seal_rejects_symlinks_nested_directories_and_tampering(tmp_path):
    root, identity = _compiled_fixture(tmp_path)
    (root / "link").symlink_to(root / "encode-prefix.bf16.plan")
    with pytest.raises(repro_stage_data.StageError, match="symlink"):
        _seal(root, identity)
    (root / "link").unlink()

    nested = root / "scratch"
    nested.mkdir()
    (nested / "result").write_bytes(b"stale")
    with pytest.raises(repro_stage_data.StageError, match="flat"):
        _seal(root, identity)
    (nested / "result").unlink()
    nested.rmdir()

    (root / "decode-denoise.bf16.plan").write_bytes(b"tampered engine")
    with pytest.raises(repro_stage_data.StageError, match="differs"):
        _seal(root, identity)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_commit", "d" * 40, "source provenance"),
        ("image_digest", "sha256:" + "d" * 64, "runtime provenance"),
        ("dataset_revision", "d" * 40, "dataset provenance"),
        ("instance_id", "i-fedcba98765432100", "runtime provenance"),
        (
            "gpu_inventory",
            ("GPU-01234567-89ab-cdef-0123-456789abcdef, NVIDIA L40S, 999.1",),
            "GPU/driver provenance",
        ),
    ],
)
def test_seal_rejects_exact_provenance_mismatch(tmp_path, field, value, message):
    root, identity = _compiled_fixture(tmp_path)
    bad = dataclass_replace(identity, **{field: value})
    with pytest.raises(repro_stage_data.StageError, match=message):
        _seal(root, bad)


def dataclass_replace(identity, **changes):
    return dataclasses.replace(identity, **changes)


def test_source_checkout_rejects_unset_protected_source_sha():
    with pytest.raises(repro_stage_data.StageError, match="protected PI05_SOURCE_SHA"):
        compiled.verify_source_checkout(
            SOURCE,
            runner=lambda _argv: (_ for _ in ()).throw(AssertionError("git must not run")),
            environ={},
        )


def test_source_checkout_rejects_protected_source_sha_mismatch():
    with pytest.raises(repro_stage_data.StageError, match="differs from --source-commit"):
        compiled.verify_source_checkout(
            SOURCE,
            runner=lambda _argv: (_ for _ in ()).throw(AssertionError("git must not run")),
            environ={"PI05_SOURCE_SHA": "d" * 40},
        )


def test_source_checkout_rejects_executing_head_mismatch():
    with pytest.raises(repro_stage_data.StageError, match="Git HEAD differs"):
        compiled.verify_source_checkout(
            SOURCE,
            runner=_source_runner("d" * 40),
            environ={"PI05_SOURCE_SHA": SOURCE},
        )


def test_source_checkout_rejects_dirty_checkout():
    with pytest.raises(repro_stage_data.StageError, match="checkout is dirty"):
        compiled.verify_source_checkout(
            SOURCE,
            runner=_source_runner(dirty=True),
            environ={"PI05_SOURCE_SHA": SOURCE},
        )


def test_upload_dry_run_never_calls_aws(tmp_path, monkeypatch, capsys):
    root, identity = _compiled_fixture(tmp_path)
    monkeypatch.setattr(compiled, "_declared_from_live", lambda _args: identity)
    monkeypatch.setattr(compiled, "verify_source_checkout", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        compiled,
        "upload_compiled_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("AWS upload called during dry run")),
    )
    result = compiled.main(
        [
            "upload",
            "--artifact-dir",
            str(root),
            "--track",
            "libero",
            "--precision",
            "bf16",
            "--source-commit",
            SOURCE,
            "--image-digest",
            IMAGE,
            "--dataset",
            "physical-intelligence/libero",
            "--dataset-revision",
            DATASET_REVISION,
            "--instance-id",
            INSTANCE,
            "--s3-root",
            f"s3://{BUCKET}/compiled",
        ]
    )
    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["aws_calls_authorized"] is False
    assert output["mutations_authorized"] is False


class _S3Runner:
    def __init__(
        self,
        *,
        bad_remote_hash: bool = False,
        preexisting_history: str | None = None,
        conditional_conflict: bool = False,
        postflight_history: str | None = None,
    ):
        self.calls: list[list[str]] = []
        self.objects: dict[str, dict] = {}
        self.bad_remote_hash = bad_remote_hash
        self.preexisting_history = preexisting_history
        self.conditional_conflict = conditional_conflict
        self.postflight_history = postflight_history
        self.version_list_calls = 0

    @staticmethod
    def _arg(argv, option):
        return argv[argv.index(option) + 1]

    def __call__(self, argv):
        argv = list(argv)
        self.calls.append(argv)
        operation = tuple(argv[1:3])
        if operation == ("sts", "get-caller-identity"):
            return json.dumps({"Account": ACCOUNT})
        if operation == ("s3api", "get-bucket-location"):
            return json.dumps({"LocationConstraint": REGION})
        if operation == ("s3api", "get-bucket-versioning"):
            return json.dumps({"Status": "Enabled"})
        if operation == ("s3api", "get-bucket-encryption"):
            return json.dumps({"ServerSideEncryptionConfiguration": {"Rules": [{}]}})
        if operation == ("s3api", "put-object"):
            if self.conditional_conflict:
                raise repro_stage_data.StageError("command failed: PreconditionFailed (HTTP 412)")
            source = pathlib.Path(self._arg(argv, "--body"))
            key = self._arg(argv, "--key")
            metadata = dict(item.split("=", 1) for item in self._arg(argv, "--metadata").split(","))
            self.objects[key] = {
                "payload": source.read_bytes(),
                "metadata": metadata,
                "version": f"v-{len(self.objects) + 1}",
            }
            return json.dumps(
                {"VersionId": self.objects[key]["version"], "ServerSideEncryption": "AES256", "ETag": "etag"}
            )
        if operation == ("s3api", "head-object"):
            item = self.objects[self._arg(argv, "--key")]
            return json.dumps(
                {
                    "ContentLength": len(item["payload"]),
                    "ServerSideEncryption": "AES256",
                    "Metadata": item["metadata"],
                    "VersionId": item["version"],
                }
            )
        if operation == ("s3api", "get-object"):
            key = self._arg(argv, "--key")
            output = pathlib.Path(argv[argv.index("--version-id") + 2])
            payload = self.objects[key]["payload"]
            if self.bad_remote_hash and key.endswith("decode-denoise.bf16.plan"):
                payload = b"same-sized-corrupt"[: len(payload)].ljust(len(payload), b"x")
            output.write_bytes(payload)
            return json.dumps({"VersionId": self.objects[key]["version"]})
        if operation == ("s3api", "list-objects-v2"):
            prefix = self._arg(argv, "--prefix")
            if not self.objects and self.preexisting_history == "object":
                return json.dumps({"IsTruncated": False, "Contents": [{"Key": f"{prefix}old", "Size": 3}]})
            return json.dumps(
                {
                    "IsTruncated": False,
                    "Contents": [
                        {"Key": key, "Size": len(item["payload"])}
                        for key, item in sorted(self.objects.items())
                        if key.startswith(prefix)
                    ],
                }
            )
        if operation == ("s3api", "list-object-versions"):
            self.version_list_calls += 1
            prefix = self._arg(argv, "--prefix")
            if self.version_list_calls == 1 and self.preexisting_history == "version":
                return json.dumps(
                    {
                        "IsTruncated": False,
                        "Versions": [{"Key": f"{prefix}old", "VersionId": "old-v1", "IsLatest": True}],
                        "DeleteMarkers": [],
                    }
                )
            if self.version_list_calls == 1 and self.preexisting_history == "delete-marker":
                return json.dumps(
                    {
                        "IsTruncated": False,
                        "Versions": [],
                        "DeleteMarkers": [{"Key": f"{prefix}old", "VersionId": "old-d1", "IsLatest": True}],
                    }
                )
            versions = [
                {"Key": key, "VersionId": item["version"], "IsLatest": True}
                for key, item in sorted(self.objects.items())
                if key.startswith(prefix)
            ]
            delete_markers = []
            if self.version_list_calls > 1 and self.postflight_history == "prior-version" and versions:
                versions.append({"Key": versions[0]["Key"], "VersionId": "old-v0", "IsLatest": False})
            if self.version_list_calls > 1 and self.postflight_history == "delete-marker" and versions:
                delete_markers.append({"Key": versions[0]["Key"], "VersionId": "deleted-v0", "IsLatest": False})
            return json.dumps(
                {
                    "IsTruncated": False,
                    "Versions": versions,
                    "DeleteMarkers": delete_markers,
                }
            )
        raise AssertionError(argv)


def test_execute_upload_rechecks_clean_source_before_any_aws_call(tmp_path):
    root, identity = _compiled_fixture(tmp_path)
    sealed = _seal(root, identity)
    runner = _S3Runner()
    with pytest.raises(repro_stage_data.StageError, match="checkout is dirty"):
        compiled.upload_compiled_artifact(
            {"aws": {"account_id": ACCOUNT, "region": REGION, "artifact_bucket": BUCKET}},
            sealed,
            f"s3://{BUCKET}/compiled",
            runner=runner,
            source_runner=_source_runner(dirty=True),
            environ={"AWS_REGION": REGION, "PI05_SOURCE_SHA": SOURCE},
        )
    assert runner.calls == []


def test_execute_upload_returns_versioned_hash_verified_asset_descriptor(tmp_path):
    root, identity = _compiled_fixture(tmp_path)
    sealed = _seal(root, identity)
    runner = _S3Runner()
    result = compiled.upload_compiled_artifact(
        {"aws": {"account_id": ACCOUNT, "region": REGION, "artifact_bucket": BUCKET}},
        sealed,
        f"s3://{BUCKET}/compiled",
        runner=runner,
        source_runner=_source_runner(),
        environ={"AWS_REGION": REGION, "PI05_SOURCE_SHA": SOURCE},
    )
    artifact = result["worker_artifact"]
    assert artifact["kind"] == "asset"
    assert artifact["revision"] == sealed.revision
    assert artifact["manifest"]["version_id"] == result["manifest_receipt"]["version_id"]
    manifest_key = result["manifest_receipt"]["key"]
    manifest = json.loads(runner.objects[manifest_key]["payload"])
    assert repro_worker.validate_artifact_manifest(manifest, artifact) == list(sealed.files)
    assert len(result["object_receipts"]) == sealed.totals["files"]
    assert any(call[1:3] == ["s3api", "list-object-versions"] for call in runner.calls)
    assert sum(call[1:3] == ["s3api", "get-object"] for call in runner.calls) == sealed.totals["files"] + 1
    put_calls = [call for call in runner.calls if call[1:3] == ["s3api", "put-object"]]
    assert len(put_calls) == sealed.totals["files"] + 1
    assert not any(call[1:3] == ["s3", "cp"] for call in runner.calls)
    for call in put_calls:
        assert call[call.index("--if-none-match") + 1] == "*"
        assert call[call.index("--expected-bucket-owner") + 1] == ACCOUNT
        assert call[call.index("--server-side-encryption") + 1] == "AES256"
        assert "--body" in call
        assert call[0:3] == ["aws", "s3api", "put-object"]


@pytest.mark.parametrize(
    ("history", "message"),
    [
        ("object", "already contains objects"),
        ("version", "prior object-version or delete-marker history"),
        ("delete-marker", "prior object-version or delete-marker history"),
    ],
)
def test_execute_upload_rejects_preexisting_object_version_or_delete_marker_history(tmp_path, history, message):
    root, identity = _compiled_fixture(tmp_path)
    sealed = _seal(root, identity)
    runner = _S3Runner(preexisting_history=history)
    with pytest.raises(repro_stage_data.StageError, match=message):
        compiled.upload_compiled_artifact(
            {"aws": {"account_id": ACCOUNT, "region": REGION, "artifact_bucket": BUCKET}},
            sealed,
            f"s3://{BUCKET}/compiled",
            runner=runner,
            source_runner=_source_runner(),
            environ={"AWS_REGION": REGION, "PI05_SOURCE_SHA": SOURCE},
        )
    assert not any(call[1:3] == ["s3api", "put-object"] for call in runner.calls)


def test_execute_upload_rejects_conditional_create_race(tmp_path):
    root, identity = _compiled_fixture(tmp_path)
    sealed = _seal(root, identity)
    runner = _S3Runner(conditional_conflict=True)
    with pytest.raises(repro_stage_data.StageError, match="PreconditionFailed"):
        compiled.upload_compiled_artifact(
            {"aws": {"account_id": ACCOUNT, "region": REGION, "artifact_bucket": BUCKET}},
            sealed,
            f"s3://{BUCKET}/compiled",
            runner=runner,
            source_runner=_source_runner(),
            environ={"AWS_REGION": REGION, "PI05_SOURCE_SHA": SOURCE},
        )
    put = next(call for call in runner.calls if call[1:3] == ["s3api", "put-object"])
    assert put[put.index("--if-none-match") + 1] == "*"


@pytest.mark.parametrize(
    ("history", "message"),
    [("prior-version", "prior or duplicate"), ("delete-marker", "delete-marker history")],
)
def test_execute_upload_rejects_any_postflight_history(tmp_path, history, message):
    root, identity = _compiled_fixture(tmp_path)
    sealed = _seal(root, identity)
    with pytest.raises(repro_stage_data.StageError, match=message):
        compiled.upload_compiled_artifact(
            {"aws": {"account_id": ACCOUNT, "region": REGION, "artifact_bucket": BUCKET}},
            sealed,
            f"s3://{BUCKET}/compiled",
            runner=_S3Runner(postflight_history=history),
            source_runner=_source_runner(),
            environ={"AWS_REGION": REGION, "PI05_SOURCE_SHA": SOURCE},
        )


def test_execute_upload_rejects_remote_content_hash_mismatch(tmp_path):
    root, identity = _compiled_fixture(tmp_path)
    sealed = _seal(root, identity)
    with pytest.raises(repro_stage_data.StageError, match="hash verification failed"):
        compiled.upload_compiled_artifact(
            {"aws": {"account_id": ACCOUNT, "region": REGION, "artifact_bucket": BUCKET}},
            sealed,
            f"s3://{BUCKET}/compiled",
            runner=_S3Runner(bad_remote_hash=True),
            source_runner=_source_runner(),
            environ={"AWS_REGION": REGION, "PI05_SOURCE_SHA": SOURCE},
        )


def test_publication_manifest_tampering_is_rejected_by_worker(tmp_path):
    root, identity = _compiled_fixture(tmp_path)
    sealed = _seal(root, identity)
    target = compiled.parse_s3_target(f"s3://{BUCKET}/compiled", sealed)
    manifest = compiled.build_publication_manifest(sealed, target)
    artifact = compiled.worker_artifact_descriptor(
        sealed,
        target,
        manifest_version_id="v1",
        manifest_sha256="e" * 64,
    )
    tampered = copy.deepcopy(manifest)
    tampered["artifact"]["payload_s3_uri"] = f"s3://{BUCKET}/other/"
    with pytest.raises(repro_worker.WorkerError, match="publication path"):
        repro_worker.validate_artifact_manifest(tampered, artifact)
