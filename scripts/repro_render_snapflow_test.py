import copy
import hashlib

import pytest

from scripts import repro_render_snapflow
from scripts import repro_worker


def artifact(name: str, kind: str, destination: str) -> dict:
    return {
        "name": name,
        "kind": kind,
        "revision": "d" * (40 if kind == "dataset" else 64),
        "manifest": {
            "s3_uri": f"s3://pi05-repro-752160877725-us-east-2/inputs/{name}/manifest.json",
            "version_id": f"{name}-v1",
            "sha256": hashlib.sha256(name.encode()).hexdigest(),
        },
        "payload_s3_uri": f"s3://pi05-repro-752160877725-us-east-2/inputs/{name}/payload/",
        "destination": destination,
    }


def base_spec(track: str) -> dict:
    runtime = "v2" if track == "libero" else "v3"
    dataset = artifact("libero", "dataset", "libero")
    jax_teacher = artifact("libero_teacher_jax", "checkpoint", "pi05_libero")
    pytorch_teacher = artifact("libero_teacher_pytorch", "checkpoint", "pi05_libero_pytorch")
    if track == "droid":
        dataset = artifact("droid", "dataset", "molmoact2-droid")
        jax_teacher = artifact("droid_jointpos_teacher_jax", "checkpoint", "pi05_droid_jointpos")
        pytorch_teacher = artifact("droid_jointpos_teacher_pytorch", "checkpoint", "pi05_droid_jointpos_pytorch")
    return {
        "schema_version": 1,
        "project": "pi05-aws-repro",
        "run_id": "source-shallow",
        "aws": {
            "account_id": "752160877725",
            "region": "us-east-2",
            "artifact_bucket": "pi05-repro-752160877725-us-east-2",
        },
        "controller_source": {
            "s3_uri": "s3://pi05-repro-752160877725-us-east-2/source/controller.bundle",
            "version_id": "controller-v1",
            "sha256": "9" * 64,
            "commit": "8" * 40,
        },
        "source": {
            "s3_uri": "s3://pi05-repro-752160877725-us-east-2/source/openpi.bundle",
            "version_id": "source-v1",
            "sha256": "a" * 64,
            "commit": "b" * 40,
        },
        "image": {
            "uri": "752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro@sha256:" + "c" * 64,
            "digest": "sha256:" + "c" * 64,
            "purpose": "policy",
            "lerobot_runtime": runtime,
            "lerobot_revision": repro_worker.LEROBOT_REVISIONS[runtime],
        },
        "artifacts": [dataset, jax_teacher, pytorch_teacher],
        "container": {
            "command": ["python", "scripts/train_pytorch.py", f"pi05_{track}_l09_distill"],
            "environment": {"WANDB_MODE": "offline"},
            "shm_size_gib": 64,
        },
        "output": {"s3_uri": "s3://pi05-repro-752160877725-us-east-2/runs/source-shallow/"},
        "timing": {"sync_interval_seconds": 60, "upload_buffer_seconds": 900, "stop_grace_seconds": 30},
        "scratch": {
            "model": "Amazon EC2 NVMe Instance Storage",
            "expected_count": 1,
            "ordinal": 0,
            "mount": "/mnt/openpi",
            "filesystem_label": "PI05_SCRATCH",
        },
        "seed": 42,
    }


def accepted_checkpoint(track: str, step: int) -> dict:
    config = f"pi05_{track}_l09_distill"
    descriptor = artifact("accepted_shallow", "checkpoint", f"{config}/{track}-shallow/{step}")
    descriptor["revision"] = "e" * 64
    return descriptor


@pytest.mark.parametrize(("track", "source_step"), [("libero", 5000), ("droid", 2000)])
def test_rendered_fast_lane_spec_is_worker_valid(track, source_step):
    rendered = repro_render_snapflow.render_snapflow_spec(
        base_spec(track),
        accepted_checkpoint(track, source_step),
        track=track,
        run_id=f"{track}-snapflow-fast-5000",
        experiment=f"{track}-snapflow-fast",
    )

    config = f"pi05_{track}_l09_snapflow"
    command = rendered["container"]["command"]
    assert command[:3] == ["python", "scripts/train_pytorch.py", config]
    assert command[command.index("--num-train-steps") + 1] == "5000"
    source_destination = f"pi05_{track}_l09_distill/{track}-shallow/{source_step}"
    assert rendered["artifacts"][2]["destination"] == source_destination
    assert command[command.index("--pytorch-weight-path") + 1] == f"/mnt/openpi/checkpoints/{source_destination}"
    assert [item["name"] for item in rendered["artifacts"]] == [
        track,
        "libero_teacher_jax" if track == "libero" else "droid_jointpos_teacher_jax",
        "accepted_shallow",
    ]
    if track == "droid":
        assert command[-3:] == ["--num-workers", "0", "--no-wandb-enabled"]
        assert command.count("--num-workers") == 1
    else:
        assert "--num-workers" not in command


def test_renderer_rejects_wrong_source_config_and_missing_jax():
    base = base_spec("libero")
    wrong_source = accepted_checkpoint("droid", 2000)
    with pytest.raises(repro_render_snapflow.RenderError, match="accepted source must be"):
        repro_render_snapflow.render_snapflow_spec(
            base,
            wrong_source,
            track="libero",
            run_id="libero-snapflow-fast-5000",
            experiment="libero-snapflow-fast",
        )

    missing = copy.deepcopy(base)
    missing["artifacts"] = [item for item in missing["artifacts"] if item["name"] != "libero_teacher_jax"]
    with pytest.raises(repro_render_snapflow.RenderError, match="libero_teacher_jax"):
        repro_render_snapflow.render_snapflow_spec(
            missing,
            accepted_checkpoint("libero", 5000),
            track="libero",
            run_id="libero-snapflow-fast-5000",
            experiment="libero-snapflow-fast",
        )
