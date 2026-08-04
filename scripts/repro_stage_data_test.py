import dataclasses
import hashlib
import json

import polars as pl
import pytest

from scripts import repro_stage_data


@pytest.fixture
def config():
    return {
        "source": {
            "libero_repo": "physical-intelligence/libero",
            "libero_revision": "a4336d589d589045d1c56423ffdf3b88a0e19b1f",
            "libero_expected_bytes": 34_938_927_454,
            "molmoact2_droid_repo": "allenai/MolmoAct2-DROID-Dataset",
            "molmoact2_droid_revision": "e44d3138c64cfeb1c24fbbce087b475fb1233728",
            "molmoact2_droid_expected_bytes": 259_000_000_000,
        },
        "aws": {"account_id": "752160877725", "region": "us-east-2"},
    }


def write_info(root, *, version, features, episodes=1, frames=1, tasks=1):
    metadata = root / "meta"
    metadata.mkdir(parents=True)
    feature_metadata = {name: {"dtype": "float32", "shape": [1]} for name in features}
    if version == "v3.0":
        feature_metadata.update(
            {
                "observation.images.exterior_1_left": {"dtype": "video", "shape": [180, 320, 3]},
                "observation.images.wrist_left": {"dtype": "video", "shape": [180, 320, 3]},
                "observation.state.joint_position": {"dtype": "float32", "shape": [7]},
                "observation.state.gripper_position": {"dtype": "float32", "shape": [1]},
                "action": {"dtype": "float32", "shape": [8]},
            }
        )
    data_path = (
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        if version == "v2.0"
        else "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    )
    info = {
        "codebase_version": version,
        "total_episodes": episodes,
        "total_frames": frames,
        "total_tasks": tasks,
        "data_path": data_path,
        "features": feature_metadata,
    }
    if version == "v3.0":
        info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    (metadata / "info.json").write_text(json.dumps(info))


def fixture_droid_spec(config):
    return dataclasses.replace(
        repro_stage_data.dataset_spec(config, "droid"),
        expected_bytes=None,
        expected_video_file_counts=(),
    )


def make_libero(root, spec):
    write_info(root, version="v2.0", features=spec.required_features)
    (root / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "put object in bowl"}\n')
    (root / "meta/episodes.jsonl").write_text('{"episode_index": 0, "tasks": ["put object in bowl"], "length": 1}\n')
    data = root / "data/chunk-000"
    data.mkdir(parents=True)
    (data / "episode_000000.parquet").write_bytes(b"fixture")


def make_droid(root, spec):
    write_info(root, version="v3.0", features=spec.required_features, episodes=2, frames=2, tasks=2)
    metadata = root / "meta"
    pl.DataFrame({"episode_index": [0, 1], "task": ["pick cup", "place cup"]}).write_parquet(
        metadata / "tasks_annotated.parquet"
    )
    pl.DataFrame({"task_index": [0, 1], "__index_level_0__": ["pick cup", "place cup"]}).write_parquet(
        metadata / "tasks.parquet"
    )
    (metadata / "episodes").mkdir()
    pl.DataFrame(
        {
            "episode_index": [0, 1],
            "data/chunk_index": [0, 0],
            "data/file_index": [0, 0],
            "videos/observation.images.exterior_1_left/chunk_index": [0, 0],
            "videos/observation.images.exterior_1_left/file_index": [0, 1],
            "videos/observation.images.exterior_1_left/from_timestamp": [0.0, 0.0],
            "videos/observation.images.exterior_1_left/to_timestamp": [1.0, 1.0],
            "videos/observation.images.wrist_left/chunk_index": [0, 0],
            "videos/observation.images.wrist_left/file_index": [0, 0],
            "videos/observation.images.wrist_left/from_timestamp": [0.0, 1.0],
            "videos/observation.images.wrist_left/to_timestamp": [1.0, 2.0],
        }
    ).write_parquet(metadata / "episodes/chunk.parquet")
    fields = {
        "observation.state.joint_position": [0, 1],
        "observation.state.gripper_position": [0, 1],
        "action": [0, 1],
        "episode_index": [0, 1],
        "task_index": [0, 1],
    }
    data = root / "data/chunk-000"
    data.mkdir(parents=True)
    pl.DataFrame(fields).write_parquet(data / "file-000.parquet")
    exterior_videos = root / "videos/observation.images.exterior_1_left/chunk-000"
    exterior_videos.mkdir(parents=True)
    (exterior_videos / "file-000.mp4").write_bytes(b"video")
    (exterior_videos / "file-001.mp4").write_bytes(b"video")
    wrist_videos = root / "videos/observation.images.wrist_left/chunk-000"
    wrist_videos.mkdir(parents=True)
    (wrist_videos / "file-000.mp4").write_bytes(b"video")


def test_specs_are_exactly_config_pinned(config):
    libero = repro_stage_data.dataset_spec(config, "libero")
    droid = repro_stage_data.dataset_spec(config, "droid")
    assert (libero.repo_id, libero.revision, libero.codebase_version) == (
        "physical-intelligence/libero",
        "a4336d589d589045d1c56423ffdf3b88a0e19b1f",
        "v2.0",
    )
    assert (droid.repo_id, droid.revision, droid.codebase_version) == (
        "allenai/MolmoAct2-DROID-Dataset",
        "e44d3138c64cfeb1c24fbbce087b475fb1233728",
        "v3.0",
    )
    assert dict(droid.expected_video_file_counts) == {
        "observation.images.exterior_1_left": 518,
        "observation.images.wrist_left": 316,
    }
    with pytest.raises(repro_stage_data.StageError, match="full 40-character commit"):
        repro_stage_data.dataset_spec({**config, "source": {**config["source"], "libero_revision": "main"}}, "libero")


def test_download_passes_only_the_pinned_snapshot_arguments(tmp_path, config):
    spec = repro_stage_data.dataset_spec(config, "libero")
    destination = tmp_path / "libero"
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return kwargs["local_dir"]

    repro_stage_data.download_snapshot(spec, destination, snapshot_download_fn=fake_download)
    assert calls == [
        {
            "repo_id": "physical-intelligence/libero",
            "repo_type": "dataset",
            "revision": "a4336d589d589045d1c56423ffdf3b88a0e19b1f",
            "local_dir": str(destination),
        }
    ]


def test_dry_run_does_not_download_or_contact_aws(tmp_path, config, monkeypatch, capsys):
    config_path = tmp_path / "reproduction.json"
    config_path.write_text(json.dumps(config))

    def unexpected(*_args, **_kwargs):
        raise AssertionError("dry-run performed an external action")

    monkeypatch.setattr(repro_stage_data, "download_snapshot", unexpected)
    monkeypatch.setattr(repro_stage_data, "upload_snapshot", unexpected)
    result = repro_stage_data.main(
        [
            "stage",
            "--dataset",
            "droid",
            "--local-root",
            str(tmp_path / "datasets"),
            "--s3-root",
            "s3://example/datasets",
            "--config",
            str(config_path),
        ]
    )
    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "dry-run"
    assert output["mutations_authorized"] is False
    assert output["download_api"]["kwargs"]["revision"] == config["source"]["molmoact2_droid_revision"]


def test_validates_libero_v2_and_droid_v3_schema(tmp_path, config):
    libero = dataclasses.replace(repro_stage_data.dataset_spec(config, "libero"), expected_bytes=None)
    libero_root = tmp_path / "libero"
    make_libero(libero_root, libero)
    libero_result = repro_stage_data.validate_snapshot(libero, libero_root)
    assert libero_result["codebase_version"] == "v2.0"
    assert libero_result["data_files"] == 1

    droid = fixture_droid_spec(config)
    droid_root = tmp_path / "droid"
    make_droid(droid_root, droid)
    droid_result = repro_stage_data.validate_snapshot(droid, droid_root)
    assert droid_result["codebase_version"] == "v3.0"
    assert droid_result["layout_contract"] == "molmoact2-v3-exact-media-references-v1"
    assert droid_result["annotation_records"] == 2
    assert droid_result["required_video_files"] == 3
    assert droid_result["required_video_files_by_feature"] == {
        "observation.images.exterior_1_left": 2,
        "observation.images.wrist_left": 1,
    }
    assert droid_result["video_timestamp_bounds_by_feature"] == {
        "observation.images.exterior_1_left": {
            "min_from_timestamp": 0.0,
            "min_duration_seconds": 1.0,
            "max_to_timestamp": 1.0,
        },
        "observation.images.wrist_left": {
            "min_from_timestamp": 0.0,
            "min_duration_seconds": 1.0,
            "max_to_timestamp": 2.0,
        },
    }


def test_rejects_droid_camera_count_that_differs_from_pinned_revision(tmp_path, config):
    droid = dataclasses.replace(
        fixture_droid_spec(config),
        expected_video_file_counts=(("observation.images.exterior_1_left", 3),),
    )
    root = tmp_path / "droid"
    make_droid(root, droid)
    with pytest.raises(repro_stage_data.StageError, match="referenced file count mismatch"):
        repro_stage_data.validate_snapshot(droid, root)


def test_rejects_wrong_version_missing_droid_field_and_bad_annotations(tmp_path, config):
    droid = fixture_droid_spec(config)
    root = tmp_path / "droid"
    make_droid(root, droid)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["codebase_version"] = "v2.0"
    info_path.write_text(json.dumps(info))
    with pytest.raises(repro_stage_data.StageError, match="codebase version mismatch"):
        repro_stage_data.validate_snapshot(droid, root)

    info["codebase_version"] = "v3.0"
    del info["features"]["action"]
    info_path.write_text(json.dumps(info))
    with pytest.raises(repro_stage_data.StageError, match="missing features"):
        repro_stage_data.validate_snapshot(droid, root)

    info["features"]["action"] = {"dtype": "float32", "shape": [8]}
    info_path.write_text(json.dumps(info))
    pl.DataFrame({"episode_index": [0, 0], "task": ["pick", "place"]}).write_parquet(
        root / "meta/tasks_annotated.parquet"
    )
    with pytest.raises(repro_stage_data.StageError, match="one non-null row per episode_index"):
        repro_stage_data.validate_snapshot(droid, root)


def test_rejects_missing_droid_video_subtree_and_wrong_video_path(tmp_path, config):
    droid = fixture_droid_spec(config)
    root = tmp_path / "droid"
    make_droid(root, droid)
    wrist_video = root / "videos/observation.images.wrist_left/chunk-000/file-000.mp4"
    wrist_video.unlink()
    with pytest.raises(repro_stage_data.StageError, match="video feature .* coverage mismatch"):
        repro_stage_data.validate_snapshot(droid, root)

    wrist_video.write_bytes(b"video")
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["video_path"] = "videos/{video_key}/{episode_index}.mp4"
    info_path.write_text(json.dumps(info))
    with pytest.raises(repro_stage_data.StageError, match="unexpected MolmoAct2 video_path"):
        repro_stage_data.validate_snapshot(droid, root)


def test_rejects_droid_orphan_video_and_empty_referenced_video(tmp_path, config):
    droid = fixture_droid_spec(config)
    root = tmp_path / "droid"
    make_droid(root, droid)
    exterior = root / "videos/observation.images.exterior_1_left/chunk-000"
    orphan = exterior / "file-002.mp4"
    orphan.write_bytes(b"video")
    with pytest.raises(repro_stage_data.StageError, match="video feature .* coverage mismatch"):
        repro_stage_data.validate_snapshot(droid, root)

    orphan.unlink()
    (exterior / "file-001.mp4").write_bytes(b"")
    with pytest.raises(repro_stage_data.StageError, match="contains empty MP4 files"):
        repro_stage_data.validate_snapshot(droid, root)


def test_rejects_droid_null_media_reference_and_invalid_timestamps(tmp_path, config):
    droid = fixture_droid_spec(config)
    root = tmp_path / "droid"
    make_droid(root, droid)
    episodes_path = root / "meta/episodes/chunk.parquet"
    episodes = pl.read_parquet(episodes_path)

    episodes.with_columns(
        pl.when(pl.col("episode_index") == 1)
        .then(None)
        .otherwise(pl.col("videos/observation.images.wrist_left/file_index"))
        .alias("videos/observation.images.wrist_left/file_index")
    ).write_parquet(episodes_path)
    with pytest.raises(repro_stage_data.StageError, match="references contain null chunk/file indices"):
        repro_stage_data.validate_snapshot(droid, root)

    episodes.with_columns(
        pl.when(pl.col("episode_index") == 1)
        .then(pl.col("videos/observation.images.wrist_left/from_timestamp"))
        .otherwise(pl.col("videos/observation.images.wrist_left/to_timestamp"))
        .alias("videos/observation.images.wrist_left/to_timestamp")
    ).write_parquet(episodes_path)
    with pytest.raises(repro_stage_data.StageError, match="must have finite timestamps"):
        repro_stage_data.validate_snapshot(droid, root)


def test_validates_every_referenced_droid_data_parquet_schema(tmp_path, config):
    droid = fixture_droid_spec(config)
    root = tmp_path / "droid"
    make_droid(root, droid)
    episodes_path = root / "meta/episodes/chunk.parquet"
    episodes = pl.read_parquet(episodes_path).with_columns(
        pl.when(pl.col("episode_index") == 1).then(1).otherwise(pl.col("data/file_index")).alias("data/file_index")
    )
    episodes.write_parquet(episodes_path)
    pl.DataFrame({"episode_index": [1]}).write_parquet(root / "data/chunk-000/file-001.parquet")
    with pytest.raises(repro_stage_data.StageError, match="file-001.parquet is missing fields"):
        repro_stage_data.validate_snapshot(droid, root)


def test_manifest_hashes_payload_and_excludes_hf_cache(tmp_path, config):
    spec = dataclasses.replace(repro_stage_data.dataset_spec(config, "libero"), expected_bytes=None)
    root = tmp_path / "libero"
    make_libero(root, spec)
    cache = root / ".cache/huggingface"
    cache.mkdir(parents=True)
    (cache / "download.lock").write_text("not payload")
    validation = repro_stage_data.validate_snapshot(spec, root)
    manifest = repro_stage_data.build_manifest(spec, root, validation, hash_workers=2)
    paths = {item["path"] for item in manifest["files"]}
    assert ".cache/huggingface/download.lock" not in paths
    assert manifest["source"]["revision"] == spec.revision
    info_record = next(item for item in manifest["files"] if item["path"] == "meta/info.json")
    assert info_record["sha256"] == hashlib.sha256((root / "meta/info.json").read_bytes()).hexdigest()
    assert manifest["totals"]["files"] == len(manifest["files"])


class FakeAwsRunner:
    def __init__(self, *, account="752160877725", manifest_bytes=0):
        self.account = account
        self.manifest_bytes = manifest_bytes
        self.calls = []

    def __call__(self, argv):
        argv = list(argv)
        self.calls.append(argv)
        operation = tuple(argv[1:3])
        if operation == ("sts", "get-caller-identity"):
            return json.dumps({"Account": self.account})
        if operation == ("s3api", "get-bucket-location"):
            return json.dumps({"LocationConstraint": "us-east-2"})
        if operation == ("s3api", "get-bucket-versioning"):
            return json.dumps({"Status": "Enabled"})
        if operation == ("s3api", "get-bucket-encryption"):
            return json.dumps({"ServerSideEncryptionConfiguration": {"Rules": [{}]}})
        if operation == ("s3api", "head-object"):
            return json.dumps(
                {
                    "ContentLength": self.manifest_bytes,
                    "Metadata": {"source-revision": "a4336d589d589045d1c56423ffdf3b88a0e19b1f"},
                    "VersionId": "version-1",
                }
            )
        if operation in {("s3", "sync"), ("s3api", "put-object")}:
            return ""
        raise AssertionError(f"unexpected command: {argv}")


def test_upload_checks_aws_boundary_and_uses_immutable_revision_prefix(tmp_path, config):
    spec = repro_stage_data.dataset_spec(config, "libero")
    root = tmp_path / "libero"
    make_libero(root, spec)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n")
    target = repro_stage_data.parse_s3_target("s3://bucket/datasets", spec)
    runner = FakeAwsRunner(manifest_bytes=manifest_path.stat().st_size)

    result = repro_stage_data.upload_snapshot(
        config,
        spec,
        root,
        manifest_path,
        target,
        runner=runner,
        environ={"AWS_REGION": "us-east-2"},
    )
    assert spec.revision in result["snapshot_uri"]
    assert result["worker_artifact"] == {
        "name": "libero",
        "kind": "dataset",
        "revision": spec.revision,
        "manifest": {
            "s3_uri": target.manifest_uri,
            "version_id": "version-1",
            "sha256": repro_stage_data.sha256_file(manifest_path),
        },
        "payload_s3_uri": target.snapshot_uri,
        "destination": "libero",
    }
    sync = next(call for call in runner.calls if call[1:3] == ["s3", "sync"] and "--dryrun" not in call)
    assert sync[0:3] == ["aws", "s3", "sync"]
    assert "--no-follow-symlinks" in sync
    assert "--expected-bucket-owner" not in sync
    manifest_upload = next(call for call in runner.calls if call[1:3] == ["s3api", "put-object"])
    assert manifest_upload[manifest_upload.index("--expected-bucket-owner") + 1] == "752160877725"
    assert all(isinstance(argument, str) for call in runner.calls for argument in call)
    assert runner.calls.index(sync) > next(
        index for index, call in enumerate(runner.calls) if call[1:3] == ["sts", "get-caller-identity"]
    )


def test_upload_rejects_wrong_account_before_sync(tmp_path, config):
    spec = repro_stage_data.dataset_spec(config, "libero")
    target = repro_stage_data.parse_s3_target("s3://bucket/datasets", spec)
    runner = FakeAwsRunner(account="000000000000")
    with pytest.raises(repro_stage_data.StageError, match="AWS account mismatch"):
        repro_stage_data.verify_aws_destination(
            config,
            target,
            runner=runner,
            environ={"AWS_DEFAULT_REGION": "us-east-2"},
        )
    assert not any(call[1:3] == ["s3", "sync"] for call in runner.calls)
