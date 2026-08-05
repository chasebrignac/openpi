import argparse
import datetime as dt
import hashlib
import json
import pathlib
import re
import shlex
import subprocess

import pytest

from scripts import repro_worker


def make_spec(tmp_path: pathlib.Path) -> dict:
    manifest_bytes = b"manifest"
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    return {
        "schema_version": 1,
        "project": "pi05-aws-repro",
        "run_id": "shallow-libero-pilot-01",
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
            "lerobot_runtime": "v2",
            "lerobot_revision": "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5",
        },
        "artifacts": [
            {
                "name": "libero",
                "kind": "dataset",
                "revision": "d" * 40,
                "manifest": {
                    "s3_uri": "s3://pi05-repro-752160877725-us-east-2/datasets/libero/manifest.json",
                    "version_id": "manifest-v1",
                    "sha256": manifest_sha,
                },
                "payload_s3_uri": "s3://pi05-repro-752160877725-us-east-2/datasets/libero/snapshot/",
                "destination": "libero",
            }
        ],
        "container": {
            "command": [
                "python",
                "scripts/train_pytorch.py",
                "pi05_libero_l09_distill",
                "--checkpoint-base-dir",
                "/mnt/openpi/runs",
                "--seed",
                "7",
            ],
            "environment": {"WANDB_MODE": "offline"},
            "shm_size_gib": 64,
        },
        "output": {"s3_uri": "s3://pi05-repro-752160877725-us-east-2/runs/shallow-libero-pilot-01/"},
        "timing": {"sync_interval_seconds": 60, "upload_buffer_seconds": 900, "stop_grace_seconds": 30},
        "scratch": {
            "model": "Amazon EC2 NVMe Instance Storage",
            "expected_count": 1,
            "ordinal": 0,
            "mount": "/mnt/openpi",
            "filesystem_label": "PI05_SCRATCH",
        },
        "seed": 7,
    }


def make_resume_spec(tmp_path: pathlib.Path) -> dict:
    spec = make_spec(tmp_path)
    resume_artifact = {
        "name": "shallow_checkpoint",
        "kind": "checkpoint",
        "revision": "e" * 64,
        "manifest": {
            "s3_uri": "s3://pi05-repro-752160877725-us-east-2/runs/source/manifests/worker-input.json",
            "version_id": "manifest-v2",
            "sha256": "f" * 64,
        },
        "payload_s3_uri": (
            "s3://pi05-repro-752160877725-us-east-2/runs/source/"
            "checkpoints/pi05_libero_l09_distill/libero-shallow/2000/"
        ),
        "destination": "pi05_libero_l09_distill/libero-shallow/2000",
    }
    spec["artifacts"].append(resume_artifact)
    spec["container"]["command"] = [
        "torchrun",
        "--standalone",
        "--nproc-per-node=2",
        "scripts/train_pytorch.py",
        "pi05_libero_l09_distill",
        "--exp-name",
        "libero-shallow",
        "--checkpoint-base-dir",
        "/mnt/openpi/runs",
        "--resume",
        "--num-train-steps",
        "5000",
        "--save-interval",
        "5000",
        "--seed",
        "7",
    ]
    spec["resume_checkpoint"] = {
        "artifact_name": "shallow_checkpoint",
        "target": "pi05_libero_l09_distill/libero-shallow/2000",
    }
    spec["expected_outputs"] = [
        {
            "name": "continued_checkpoint",
            "kind": "checkpoint",
            "path": "checkpoints/pi05_libero_l09_distill/libero-shallow/5000",
            "publish_destination": "pi05_libero_l09_distill/libero-shallow/5000",
        }
    ]
    return spec


def make_single_gpu_shallow_resume_spec(
    tmp_path: pathlib.Path,
    *,
    source_step: int = 2_000,
    target_step: int = 5_000,
    track: str = "libero",
) -> dict:
    if track not in {"libero", "droid"}:
        raise ValueError(f"unsupported Shallow resume test track: {track}")
    spec = make_resume_spec(tmp_path)
    config = f"pi05_{track}_l09_distill"
    experiment = f"{track}-shallow"
    spec["run_id"] = f"{track}-shallow-continue-{source_step}-to-{target_step}"
    spec["output"]["s3_uri"] = f"s3://pi05-repro-752160877725-us-east-2/runs/{spec['run_id']}/"
    if track == "droid":
        spec["image"]["lerobot_runtime"] = "v3"
        spec["image"]["lerobot_revision"] = repro_worker.LEROBOT_REVISIONS["v3"]
        spec["artifacts"][0].update(
            {
                "name": "droid",
                "destination": "molmoact2-droid",
                "payload_s3_uri": "s3://pi05-repro-752160877725-us-east-2/datasets/droid/snapshot/",
            }
        )
    resume_artifact = spec["artifacts"][1]
    resume_artifact["destination"] = f"{config}/{experiment}/{source_step}"
    resume_artifact["payload_s3_uri"] = (
        f"s3://pi05-repro-752160877725-us-east-2/runs/source/checkpoints/{config}/{experiment}/{source_step}/"
    )
    spec["resume_checkpoint"]["target"] = f"{config}/{experiment}/{source_step}"
    spec["container"]["command"] = [
        "python",
        "scripts/train_pytorch.py",
        config,
        "--exp-name",
        experiment,
        "--checkpoint-base-dir",
        "/mnt/openpi/runs",
        "--resume",
        "--seed",
        "7",
        "--num-train-steps",
        str(target_step),
        "--save-interval",
        "5000",
        "--log-interval",
        "10",
    ]
    if track == "droid":
        spec["container"]["command"].append("--no-wandb-enabled")
    spec["expected_outputs"] = [
        {
            "name": "continued_checkpoint",
            "kind": "checkpoint",
            "path": f"checkpoints/{config}/{experiment}/{target_step}",
            "publish_destination": f"{config}/{experiment}/{target_step}",
        }
    ]
    return spec


def make_snapflow_spec(tmp_path: pathlib.Path, *, diagnostic: bool = False) -> dict:
    spec = make_spec(tmp_path)
    source = {
        "name": "accepted_shallow_checkpoint",
        "kind": "checkpoint",
        "revision": "e" * 64,
        "manifest": {
            "s3_uri": "s3://pi05-repro-752160877725-us-east-2/runs/shallow/manifests/worker-input.json",
            "version_id": "manifest-v2",
            "sha256": "f" * 64,
        },
        "payload_s3_uri": (
            "s3://pi05-repro-752160877725-us-east-2/runs/shallow/"
            "checkpoints/pi05_libero_l09_distill/libero-shallow/5000/"
        ),
        "destination": "pi05_libero_l09_distill/libero-shallow/5000",
    }
    spec["artifacts"].append(source)
    experiment = "libero-snapflow-overfit" if diagnostic else "libero-snapflow"
    target_step = 300 if diagnostic else 5000
    spec["container"]["command"] = [
        "python",
        "scripts/train_pytorch.py",
        "pi05_libero_l09_snapflow",
        "--exp-name",
        experiment,
        "--checkpoint-base-dir",
        "/mnt/openpi/runs",
        "--pytorch-weight-path",
        "/mnt/openpi/checkpoints/pi05_libero_l09_distill/libero-shallow/5000",
        "--seed",
        "7",
        "--num-train-steps",
        str(target_step),
        "--save-interval",
        str(target_step),
    ]
    if diagnostic:
        spec["container"]["command"].extend(
            [
                "--one-batch-overfit",
                "--one-batch-overfit-min-relative-decline",
                "0.20",
                "--num-workers",
                "0",
                "--no-wandb-enabled",
            ]
        )
    output = {
        "name": "snapflow_checkpoint",
        "kind": "checkpoint",
        "path": f"checkpoints/pi05_libero_l09_snapflow/{experiment}/{target_step}",
    }
    if not diagnostic:
        output["publish_destination"] = f"pi05_libero_l09_snapflow/{experiment}/{target_step}"
    spec["expected_outputs"] = [output]
    return spec


def make_droid_snapflow_resume_spec(tmp_path: pathlib.Path) -> dict:
    spec = make_resume_spec(tmp_path)
    spec["run_id"] = "droid-snapflow-continue-10000"
    spec["output"]["s3_uri"] = "s3://pi05-repro-752160877725-us-east-2/runs/droid-snapflow-continue-10000/"
    spec["image"]["lerobot_runtime"] = "v3"
    spec["image"]["lerobot_revision"] = repro_worker.LEROBOT_REVISIONS["v3"]
    spec["artifacts"][0].update({"name": "droid", "destination": "molmoact2-droid"})
    resume_artifact = spec["artifacts"][1]
    resume_artifact["destination"] = "pi05_droid_l09_snapflow/droid-snapflow/5000"
    resume_artifact["payload_s3_uri"] = (
        "s3://pi05-repro-752160877725-us-east-2/runs/source/checkpoints/pi05_droid_l09_snapflow/droid-snapflow/5000/"
    )
    spec["container"]["command"] = [
        "python",
        "scripts/train_pytorch.py",
        "pi05_droid_l09_snapflow",
        "--exp-name",
        "droid-snapflow",
        "--checkpoint-base-dir",
        "/mnt/openpi/runs",
        "--resume",
        "--num-train-steps",
        "10000",
        "--save-interval",
        "5000",
        "--seed",
        "7",
        "--num-workers",
        "0",
        "--no-wandb-enabled",
    ]
    spec["resume_checkpoint"]["target"] = "pi05_droid_l09_snapflow/droid-snapflow/5000"
    spec["expected_outputs"] = [
        {
            "name": "continued_checkpoint",
            "kind": "checkpoint",
            "path": "checkpoints/pi05_droid_l09_snapflow/droid-snapflow/10000",
            "publish_destination": "pi05_droid_l09_snapflow/droid-snapflow/10000",
        }
    ]
    return spec


def make_launch_metadata(deadline: str) -> dict:
    return {
        "project": "pi05-aws-repro",
        "deadline_utc": deadline,
        "command_sha256": "e" * 64,
        "purchase_option": "On-Demand",
        "instance_count": 1,
        "reservation_id": "12345678-1234-4123-8123-123456789abc",
        "projected_compute_usd": 18.0,
        "reserved_hours": 2.25,
    }


def make_tensorrt_spec(tmp_path: pathlib.Path) -> dict:
    spec = make_spec(tmp_path)
    spec["image"] = {
        "uri": "752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro-tensorrt@sha256:" + "1" * 64,
        "digest": "sha256:" + "1" * 64,
        "purpose": "tensorrt-compiler",
        "toolchain": dict(repro_worker.TENSORRT_COMPILER_TOOLCHAIN),
    }
    spec["artifacts"] = [
        {
            **spec["artifacts"][0],
            "name": "onnx_graphs",
            "kind": "asset",
            "revision": "3" * 64,
            "destination": "onnx/libero",
        }
    ]
    spec["placement"] = {"mode": "exact-existing-instance", "instance_id": "i-0123456789abcdef0"}
    spec["expected_outputs"] = [
        {"name": "engines", "kind": "artifact", "path": "artifacts/libero"},
    ]
    spec["container"]["command"] = [
        "python",
        "scripts/build_tensorrt_engines.py",
        "--artifact-dir",
        "/output/artifacts/libero",
        "--image-digest",
        spec["image"]["digest"],
        "--instance-id",
        spec["placement"]["instance_id"],
        "--execute",
    ]
    return spec


def make_libero_evaluator_spec(tmp_path: pathlib.Path) -> dict:
    spec = make_spec(tmp_path)
    spec["image"] = {
        "uri": "752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro@sha256:" + "2" * 64,
        "digest": "sha256:" + "2" * 64,
        "purpose": "libero-evaluator",
        "policy_backend": "eager",
        "lerobot_runtime": "v2",
        "lerobot_revision": "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5",
        "libero_simulator_revision": repro_worker.LIBERO_SIMULATOR_REVISION,
        "libero_requirements_sha256": repro_worker.LIBERO_REQUIREMENTS_SHA256,
        "parent_policy_image": ("752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro@sha256:" + "c" * 64),
    }
    spec["container"] = {
        "command": [
            "python",
            "scripts/repro_libero_eval.py",
            "run",
            "--backend",
            "eager",
            "--output-root",
            "/output",
        ],
        "environment": {
            **repro_worker.LIBERO_EVALUATOR_ENVIRONMENT,
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
        },
        "shm_size_gib": 32,
    }
    return spec


def make_tensorrt_policy_spec(tmp_path: pathlib.Path) -> dict:
    spec = make_spec(tmp_path)
    spec["image"] = {
        "uri": "752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro@sha256:" + "4" * 64,
        "digest": "sha256:" + "4" * 64,
        "purpose": "tensorrt-policy",
        "lerobot_runtime": "v2",
        "lerobot_revision": repro_worker.LEROBOT_REVISIONS["v2"],
        "parent_tensorrt_compiler_image": (
            "752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro@sha256:" + "1" * 64
        ),
        "parent_tensorrt_compiler_source_revision": spec["source"]["commit"],
        "toolchain": dict(repro_worker.TENSORRT_COMPILER_TOOLCHAIN),
    }
    return spec


def make_tensorrt_libero_evaluator_spec(tmp_path: pathlib.Path) -> dict:
    spec = make_libero_evaluator_spec(tmp_path)
    spec["image"].update(
        {
            "policy_backend": "tensorrt",
            "parent_tensorrt_compiler_image": (
                "752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro@sha256:" + "1" * 64
            ),
            "parent_tensorrt_compiler_source_revision": spec["source"]["commit"],
            "toolchain": dict(repro_worker.TENSORRT_COMPILER_TOOLCHAIN),
        }
    )
    instance_id = "i-0123456789abcdef0"
    spec["placement"] = {"mode": "exact-existing-instance", "instance_id": instance_id}
    command = spec["container"]["command"]
    command[command.index("eager")] = "tensorrt"
    command.extend(["--build-instance-id", instance_id])
    return spec


def make_droid_dataset_contract(tmp_path: pathlib.Path) -> tuple[dict, dict, dict]:
    spec = make_spec(tmp_path)
    spec["image"]["lerobot_runtime"] = "v3"
    spec["image"]["lerobot_revision"] = repro_worker.LEROBOT_REVISIONS["v3"]
    artifact = spec["artifacts"][0]
    artifact.update(
        {
            "name": "droid",
            "revision": "e" * 40,
            "destination": "molmoact2-droid",
        }
    )
    spec["container"]["command"][2] = "pi05_droid_l09_distill"
    manifest = {
        "schema_version": 1,
        "source": {
            "provider": "huggingface",
            "repo_id": "allenai/MolmoAct2-DROID-Dataset",
            "revision": artifact["revision"],
        },
        "dataset": {"key": "droid", "codebase_version": "v3.0", "local_dirname": "molmoact2-droid"},
        "validation": {
            "layout_contract": repro_worker.DROID_LAYOUT_CONTRACT,
            "required_video_files": sum(repro_worker.DROID_CAMERA_FILE_COUNTS.values()),
            "required_video_files_by_feature": dict(repro_worker.DROID_CAMERA_FILE_COUNTS),
            "expected_video_files_by_feature": dict(repro_worker.DROID_CAMERA_FILE_COUNTS),
        },
    }
    return spec, artifact, manifest


def make_teacher_cross_contract(tmp_path: pathlib.Path, track: str = "libero") -> tuple[dict, dict[str, dict]]:
    contract = repro_worker.TEACHER_TRACK_CONTRACTS[track]
    spec = make_spec(tmp_path)
    runtime = contract["lerobot_runtime"]
    spec["image"]["lerobot_runtime"] = runtime
    spec["image"]["lerobot_revision"] = repro_worker.LEROBOT_REVISIONS[runtime]
    spec["container"]["command"][2] = f"pi05_{'libero' if track == 'libero' else 'droid'}_l09_distill"
    original_revision = "8" * 64
    converted_revision = "9" * 64
    original = {
        "name": f"{track}_teacher_jax",
        "kind": "checkpoint",
        "revision": original_revision,
        "manifest": {
            "s3_uri": f"s3://pi05-repro-752160877725-us-east-2/checkpoints/{track}-jax.json",
            "version_id": "jax-v1",
            "sha256": "a" * 64,
        },
        "payload_s3_uri": f"s3://pi05-repro-752160877725-us-east-2/checkpoints/{track}-jax/",
        "destination": contract["source_local_dirname"],
    }
    converted = {
        "name": f"{track}_teacher_pytorch",
        "kind": "checkpoint",
        "revision": converted_revision,
        "manifest": {
            "s3_uri": f"s3://pi05-repro-752160877725-us-east-2/checkpoints/{track}-pytorch.json",
            "version_id": "pytorch-v1",
            "sha256": "b" * 64,
        },
        "payload_s3_uri": f"s3://pi05-repro-752160877725-us-east-2/checkpoints/{track}-pytorch/",
        "destination": contract["converted_local_dirname"],
    }
    spec["artifacts"] = [original, converted]
    manifests = {
        original["name"]: {
            "source": {
                "provider": "gcs",
                "uri": contract["source_uri"],
                "revision": original_revision,
                "objects": [{"name": "weights"}],
            },
            "checkpoint": {"key": track, "local_dirname": contract["source_local_dirname"]},
        },
        converted["name"]: {
            "source": {
                "provider": "openpi-jax-to-pytorch",
                "revision": converted_revision,
                "upstream": {
                    "provider": "gcs",
                    "uri": contract["source_uri"],
                    "revision": original_revision,
                },
            },
            "conversion": {"config_name": contract["config_name"]},
            "checkpoint": {"key": track, "local_dirname": contract["converted_local_dirname"]},
        },
    }
    return spec, manifests


def test_spec_requires_digest_pins_revisions_and_exact_run_prefix(tmp_path):
    spec = make_spec(tmp_path)
    assert repro_worker.validate_worker_spec(spec)["seed"] == 7

    missing_controller = json.loads(json.dumps(spec))
    del missing_controller["controller_source"]
    with pytest.raises(repro_worker.WorkerError, match=r"controller_source must be an object"):
        repro_worker.validate_worker_spec(missing_controller)

    unpinned_controller = json.loads(json.dumps(spec))
    unpinned_controller["controller_source"]["commit"] = "main"
    with pytest.raises(repro_worker.WorkerError, match=r"controller_source\.commit"):
        repro_worker.validate_worker_spec(unpinned_controller)

    tagged = json.loads(json.dumps(spec))
    tagged["image"]["uri"] = "752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro:latest"
    with pytest.raises(repro_worker.WorkerError, match="pinned"):
        repro_worker.validate_worker_spec(tagged)

    mutable_dataset = json.loads(json.dumps(spec))
    mutable_dataset["artifacts"][0]["revision"] = "main"
    with pytest.raises(repro_worker.WorkerError, match="pinned revision"):
        repro_worker.validate_worker_spec(mutable_dataset)

    crossed_run = json.loads(json.dumps(spec))
    crossed_run["output"]["s3_uri"] = crossed_run["output"]["s3_uri"].replace("pilot-01", "pilot-02")
    with pytest.raises(repro_worker.WorkerError, match=r"output\.s3_uri"):
        repro_worker.validate_worker_spec(crossed_run)

    mismatched_seed = json.loads(json.dumps(spec))
    mismatched_seed["container"]["command"][-1] = "42"
    with pytest.raises(repro_worker.WorkerError, match=r"--seed must equal"):
        repro_worker.validate_worker_spec(mismatched_seed)

    overwrite = json.loads(json.dumps(spec))
    overwrite["container"]["command"].append("--overwrite")
    with pytest.raises(repro_worker.WorkerError, match="unique experiment ID"):
        repro_worker.validate_worker_spec(overwrite)


def test_image_purpose_contracts_are_disjoint_complete_and_fail_closed(tmp_path):
    policy = make_spec(tmp_path)
    assert repro_worker.validate_worker_spec(policy)["image"]["purpose"] == "policy"

    missing_purpose = json.loads(json.dumps(policy))
    del missing_purpose["image"]["purpose"]
    with pytest.raises(repro_worker.WorkerError, match=r"image\.purpose"):
        repro_worker.validate_worker_spec(missing_purpose)

    compiler = make_tensorrt_spec(tmp_path)
    assert repro_worker.validate_worker_spec(compiler)["image"]["purpose"] == "tensorrt-compiler"

    combined = make_tensorrt_policy_spec(tmp_path)
    assert repro_worker.validate_worker_spec(combined)["image"]["purpose"] == "tensorrt-policy"

    evaluator = make_libero_evaluator_spec(tmp_path)
    assert repro_worker.validate_worker_spec(evaluator)["image"]["purpose"] == "libero-evaluator"

    compiler_claiming_lerobot = json.loads(json.dumps(compiler))
    compiler_claiming_lerobot["image"]["lerobot_runtime"] = "v2"
    with pytest.raises(repro_worker.WorkerError, match="unexpected TensorRT compiler image keys"):
        repro_worker.validate_worker_spec(compiler_claiming_lerobot)

    incomplete_compiler = json.loads(json.dumps(compiler))
    del incomplete_compiler["image"]["toolchain"]["modelopt_version"]
    with pytest.raises(repro_worker.WorkerError, match="complete toolchain"):
        repro_worker.validate_worker_spec(incomplete_compiler)

    policy_with_toolchain = json.loads(json.dumps(policy))
    policy_with_toolchain["image"]["toolchain"] = dict(repro_worker.TENSORRT_COMPILER_TOOLCHAIN)
    with pytest.raises(repro_worker.WorkerError, match="unexpected policy image keys"):
        repro_worker.validate_worker_spec(policy_with_toolchain)

    policy_running_evaluator = json.loads(json.dumps(policy))
    policy_running_evaluator["container"]["command"] = ["python", "scripts/repro_libero_eval.py", "run"]
    with pytest.raises(repro_worker.WorkerError, match="dedicated libero-evaluator"):
        repro_worker.validate_worker_spec(policy_running_evaluator)

    evaluator_running_training = json.loads(json.dumps(evaluator))
    evaluator_running_training["container"]["command"] = ["python", "scripts/train_pytorch.py", "--seed", "7"]
    with pytest.raises(repro_worker.WorkerError, match=r"must invoke scripts/repro_libero_eval\.py run"):
        repro_worker.validate_worker_spec(evaluator_running_training)

    for key in repro_worker.LIBERO_EVALUATOR_ENVIRONMENT:
        missing_environment = json.loads(json.dumps(evaluator))
        del missing_environment["container"]["environment"][key]
        with pytest.raises(repro_worker.WorkerError, match=key):
            repro_worker.validate_worker_spec(missing_environment)

    incomplete_evaluator = json.loads(json.dumps(evaluator))
    del incomplete_evaluator["image"]["libero_requirements_sha256"]
    with pytest.raises(repro_worker.WorkerError, match="dependency lock"):
        repro_worker.validate_worker_spec(incomplete_evaluator)


def test_tensorrt_policy_schema_pins_exact_runtime_compiler_source_and_six_toolchain_fields(tmp_path):
    v2 = make_tensorrt_policy_spec(tmp_path)
    assert repro_worker.validate_worker_spec(v2)["image"]["lerobot_runtime"] == "v2"

    v3 = json.loads(json.dumps(v2))
    v3["image"]["lerobot_runtime"] = "v3"
    v3["image"]["lerobot_revision"] = repro_worker.LEROBOT_REVISIONS["v3"]
    v3["artifacts"][0]["destination"] = "molmoact2-droid"
    command = v3["container"]["command"]
    command[command.index("pi05_libero_l09_distill")] = "pi05_droid_l09_distill"
    assert repro_worker.validate_worker_spec(v3)["image"]["lerobot_runtime"] == "v3"

    wrong_revision = json.loads(json.dumps(v2))
    wrong_revision["image"]["lerobot_revision"] = repro_worker.LEROBOT_REVISIONS["v3"]
    with pytest.raises(repro_worker.WorkerError, match="exact approved LeRobot"):
        repro_worker.validate_worker_spec(wrong_revision)

    wrong_source = json.loads(json.dumps(v2))
    wrong_source["image"]["parent_tensorrt_compiler_source_revision"] = "0" * 40
    with pytest.raises(repro_worker.WorkerError, match="compiler digest/source"):
        repro_worker.validate_worker_spec(wrong_source)

    for key in repro_worker.TENSORRT_COMPILER_TOOLCHAIN:
        incomplete = json.loads(json.dumps(v2))
        del incomplete["image"]["toolchain"][key]
        with pytest.raises(repro_worker.WorkerError, match="complete toolchain"):
            repro_worker.validate_worker_spec(incomplete)


def test_libero_backend_schemas_are_disjoint_and_tensorrt_requires_exact_placement(tmp_path):
    eager = make_libero_evaluator_spec(tmp_path)
    assert repro_worker.validate_worker_spec(eager)["image"]["policy_backend"] == "eager"
    eager_with_compiler = json.loads(json.dumps(eager))
    eager_with_compiler["image"]["toolchain"] = dict(repro_worker.TENSORRT_COMPILER_TOOLCHAIN)
    with pytest.raises(repro_worker.WorkerError, match="unexpected eager LIBERO evaluator image keys"):
        repro_worker.validate_worker_spec(eager_with_compiler)

    compiled = make_tensorrt_libero_evaluator_spec(tmp_path)
    assert repro_worker.validate_worker_spec(compiled)["placement"]["instance_id"] == "i-0123456789abcdef0"

    no_placement = json.loads(json.dumps(compiled))
    del no_placement["placement"]
    with pytest.raises(repro_worker.WorkerError, match="exact existing-instance placement"):
        repro_worker.validate_worker_spec(no_placement)

    wrong_command_instance = json.loads(json.dumps(compiled))
    position = wrong_command_instance["container"]["command"].index("--build-instance-id")
    wrong_command_instance["container"]["command"][position + 1] = "i-0fedcba9876543210"
    with pytest.raises(repro_worker.WorkerError, match="build instance"):
        repro_worker.validate_worker_spec(wrong_command_instance)


def test_compiled_pipeline_writes_only_declared_output_and_binds_instance_and_image(tmp_path):
    valid = make_tensorrt_spec(tmp_path)
    repro_worker.validate_worker_spec(valid)

    read_only = json.loads(json.dumps(valid))
    position = read_only["container"]["command"].index("--artifact-dir")
    read_only["container"]["command"][position + 1] = "/mnt/openpi/assets/onnx/libero"
    with pytest.raises(repro_worker.WorkerError, match="writable /output/artifacts"):
        repro_worker.validate_worker_spec(read_only)

    wrong_instance = json.loads(json.dumps(valid))
    position = wrong_instance["container"]["command"].index("--instance-id")
    wrong_instance["container"]["command"][position + 1] = "i-0fedcba9876543210"
    with pytest.raises(repro_worker.WorkerError, match=r"placement\.instance_id"):
        repro_worker.validate_worker_spec(wrong_instance)

    shell = json.loads(json.dumps(valid))
    shell["container"]["command"] = ["bash", "-lc", "scripts/build_tensorrt_engines.py"]
    with pytest.raises(repro_worker.WorkerError, match="direct Python argv"):
        repro_worker.validate_worker_spec(shell)

    benchmark = make_tensorrt_policy_spec(tmp_path)
    benchmark["placement"] = {"mode": "exact-existing-instance", "instance_id": "i-0123456789abcdef0"}
    benchmark["expected_outputs"] = [{"name": "latency", "kind": "artifact", "path": "artifacts/latency.json"}]
    benchmark["container"]["command"] = [
        "python",
        "scripts/benchmark_pi05_latency.py",
        "--backend",
        "tensorrt",
        "--output",
        "/output/artifacts/latency.json",
        "--image-digest",
        benchmark["image"]["digest"],
        "--instance-id",
        benchmark["placement"]["instance_id"],
    ]
    repro_worker.validate_worker_spec(benchmark)
    benchmark["container"]["command"][benchmark["container"]["command"].index("--output") + 1] = (
        "/mnt/openpi/evidence/latency.json"
    )
    with pytest.raises(repro_worker.WorkerError, match="writable /output/artifacts"):
        repro_worker.validate_worker_spec(benchmark)

    mutable_parent = json.loads(json.dumps(make_libero_evaluator_spec(tmp_path)))
    mutable_parent["image"]["parent_policy_image"] = "752160877725.dkr.ecr.us-east-2.amazonaws.com/pi05-repro:libero-v2"
    with pytest.raises(repro_worker.WorkerError, match="parent must be a distinct account-local policy image"):
        repro_worker.validate_worker_spec(mutable_parent)


def test_worker_owned_environment_cannot_be_overridden(tmp_path):
    for key in (
        "PI05_SOURCE_SHA",
        "PI05_IMAGE_DIGEST",
        "PI05_SEED",
        "PI05_INPUT_LIBERO",
        "PI05_RESUME_CHECKPOINT",
        "HOME",
        "PATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPATH",
        "XDG_CACHE_HOME",
    ):
        spec = make_spec(tmp_path)
        spec["container"]["environment"][key] = "override"
        with pytest.raises(repro_worker.WorkerError, match="unsafe container environment"):
            repro_worker.validate_worker_spec(spec)


def test_droid_dataset_requires_strong_layout_camera_counts_and_v3_runtime(tmp_path):
    spec, artifact, manifest = make_droid_dataset_contract(tmp_path)
    assert repro_worker.validate_worker_spec(spec)["image"]["lerobot_runtime"] == "v3"
    repro_worker.validate_artifact_track_contract(manifest, artifact, spec)

    weak = json.loads(json.dumps(manifest))
    del weak["validation"]["layout_contract"]
    with pytest.raises(repro_worker.WorkerError, match="exact MolmoAct2 v3 layout contract"):
        repro_worker.validate_artifact_track_contract(weak, artifact, spec)

    for field in ("required_video_files_by_feature", "expected_video_files_by_feature"):
        wrong_counts = json.loads(json.dumps(manifest))
        wrong_counts["validation"][field]["observation.images.wrist_left"] = 315
        with pytest.raises(repro_worker.WorkerError, match="exact camera counts"):
            repro_worker.validate_artifact_track_contract(wrong_counts, artifact, spec)

    wrong_runtime = json.loads(json.dumps(spec))
    wrong_runtime["image"]["lerobot_runtime"] = "v2"
    wrong_runtime["image"]["lerobot_revision"] = repro_worker.LEROBOT_REVISIONS["v2"]
    with pytest.raises(repro_worker.WorkerError, match="artifact 'droid' requires LeRobot v3"):
        repro_worker.validate_worker_spec(wrong_runtime)
    with pytest.raises(repro_worker.WorkerError, match="requires the LeRobot v3"):
        repro_worker.validate_artifact_track_contract(manifest, artifact, wrong_runtime)

    wrong_training_track = json.loads(json.dumps(spec))
    wrong_training_track["container"]["command"][2] = "pi05_libero_l09_distill"
    with pytest.raises(repro_worker.WorkerError, match=r"training config.*requires LeRobot v2"):
        repro_worker.validate_worker_spec(wrong_training_track)


def test_converted_teacher_binds_selected_jax_revision_config_and_track_runtime(tmp_path):
    spec, manifests = make_teacher_cross_contract(tmp_path)
    repro_worker.validate_worker_spec(spec)
    repro_worker.validate_input_cross_contracts(spec, manifests)

    converted_name = "libero_teacher_pytorch"
    wrong_revision = json.loads(json.dumps(manifests))
    wrong_revision[converted_name]["source"]["upstream"]["revision"] = "7" * 64
    with pytest.raises(repro_worker.WorkerError, match="upstream revision differs"):
        repro_worker.validate_input_cross_contracts(spec, wrong_revision)

    wrong_config = json.loads(json.dumps(manifests))
    wrong_config[converted_name]["conversion"]["config_name"] = "pi05_droid_jointpos"
    with pytest.raises(repro_worker.WorkerError, match="config/upstream provenance"):
        repro_worker.validate_input_cross_contracts(spec, wrong_config)

    converted_only_spec = json.loads(json.dumps(spec))
    converted_only_spec["artifacts"] = [converted_only_spec["artifacts"][1]]
    with pytest.raises(repro_worker.WorkerError, match="no selected original JAX teacher"):
        repro_worker.validate_input_cross_contracts(converted_only_spec, {converted_name: manifests[converted_name]})

    droid_spec, droid_manifests = make_teacher_cross_contract(tmp_path, "droid_jointpos")
    repro_worker.validate_worker_spec(droid_spec)
    droid_spec["image"]["lerobot_runtime"] = "v2"
    droid_spec["image"]["lerobot_revision"] = repro_worker.LEROBOT_REVISIONS["v2"]
    with pytest.raises(repro_worker.WorkerError, match="requires the LeRobot v3"):
        repro_worker.validate_input_cross_contracts(droid_spec, droid_manifests)


def lsblk_fixture(
    *,
    second_instance_store=False,
    ebs_model=False,
    scratch_fstype=None,
    scratch_label=None,
    scratch_mountpoints=None,
):
    devices = [
        {
            "name": "nvme0n1",
            "kname": "nvme0n1",
            "path": "/dev/nvme0n1",
            "type": "disk",
            "model": "Amazon Elastic Block Store",
            "serial": "vol-root",
            "fstype": None,
            "label": None,
            "mountpoints": [None],
            "children": [
                {
                    "name": "nvme0n1p1",
                    "kname": "nvme0n1p1",
                    "path": "/dev/nvme0n1p1",
                    "type": "part",
                    "mountpoints": ["/"],
                }
            ],
        },
        {
            "name": "nvme1n1",
            "kname": "nvme1n1",
            "path": "/dev/nvme1n1",
            "type": "disk",
            "model": "Amazon Elastic Block Store" if ebs_model else "Amazon EC2 NVMe Instance Storage",
            "serial": "AWS-local-a",
            "fstype": scratch_fstype,
            "label": scratch_label,
            "mountpoints": [None] if scratch_mountpoints is None else scratch_mountpoints,
        },
    ]
    if second_instance_store:
        devices.append(
            {
                "name": "nvme2n1",
                "kname": "nvme2n1",
                "path": "/dev/nvme2n1",
                "type": "disk",
                "model": "Amazon EC2 NVMe Instance Storage",
                "serial": "AWS-local-b",
                "fstype": None,
                "label": None,
                "mountpoints": [None],
            }
        )
    return {"blockdevices": devices}


def test_instance_store_selection_proves_non_root_model_count_and_serial():
    selected = repro_worker.select_instance_store_device(lsblk_fixture(), "/dev/nvme0n1p1", expected_count=1, ordinal=0)
    assert selected == repro_worker.ScratchSelection(path="/dev/nvme1n1", serial="AWS-local-a", reuse=False)
    assert repro_worker.scratch_command_plan(selected)[0][:3] == ["mkfs.ext4", "-q", "-L"]

    with pytest.raises(repro_worker.WorkerError, match="expected exactly 1"):
        repro_worker.select_instance_store_device(
            lsblk_fixture(second_instance_store=True), "/dev/nvme0n1p1", expected_count=1, ordinal=0
        )
    with pytest.raises(repro_worker.WorkerError, match="found 0"):
        repro_worker.select_instance_store_device(
            lsblk_fixture(ebs_model=True), "/dev/nvme0n1p1", expected_count=1, ordinal=0
        )


def test_owned_ext4_can_be_reused_but_unknown_filesystem_fails_closed():
    selected = repro_worker.select_instance_store_device(
        lsblk_fixture(scratch_fstype="ext4", scratch_label="PI05_SCRATCH"),
        "/dev/nvme0n1p1",
        expected_count=1,
        ordinal=0,
    )
    assert selected.reuse
    assert repro_worker.scratch_command_plan(selected) == [
        ["mount", "-o", "noatime,nosuid,nodev", "/dev/nvme1n1", "/opt/dlami/nvme"]
    ]
    with pytest.raises(repro_worker.WorkerError, match="not blank"):
        repro_worker.select_instance_store_device(
            lsblk_fixture(scratch_fstype="xfs", scratch_label="someone_else"),
            "/dev/nvme0n1p1",
            expected_count=1,
            ordinal=0,
        )


def test_verified_dlami_mount_is_reused_without_format_or_mount_commands():
    selected = repro_worker.select_instance_store_device(
        lsblk_fixture(scratch_fstype="xfs", scratch_mountpoints=["/opt/dlami/nvme"]),
        "/dev/root",
        expected_count=1,
        ordinal=0,
    )
    assert selected == repro_worker.ScratchSelection(
        path="/dev/nvme1n1",
        serial="AWS-local-a",
        reuse=True,
        mounted_at="/opt/dlami/nvme",
        filesystem="xfs",
    )
    assert repro_worker.scratch_command_plan(selected) == []

    with pytest.raises(repro_worker.WorkerError, match="mounted outside"):
        repro_worker.select_instance_store_device(
            lsblk_fixture(scratch_fstype="xfs", scratch_mountpoints=["/data"]),
            "/dev/root",
            expected_count=1,
            ordinal=0,
        )


def test_verified_dlami_lvm_mount_is_reused_without_touching_the_raw_disk():
    document = lsblk_fixture(scratch_fstype="LVM2_member")
    document["blockdevices"][1]["children"] = [
        {
            "name": "vg.01-lv_ephemeral",
            "kname": "dm-0",
            "path": "/dev/mapper/vg.01-lv_ephemeral",
            "type": "lvm",
            "fstype": "ext4",
            "label": None,
            "mountpoints": ["/opt/dlami/nvme"],
        }
    ]
    selected = repro_worker.select_instance_store_device(
        document,
        "/dev/root",
        expected_count=1,
        ordinal=0,
    )
    assert selected == repro_worker.ScratchSelection(
        path="/dev/mapper/vg.01-lv_ephemeral",
        serial="AWS-local-a",
        reuse=True,
        mounted_at="/opt/dlami/nvme",
        filesystem="ext4",
    )
    assert repro_worker.scratch_command_plan(selected) == []


def test_manifest_revision_paths_totals_and_local_hashes_are_verified(tmp_path):
    artifact = make_spec(tmp_path)["artifacts"][0]
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "data.bin").write_bytes(b"exact")
    digest = hashlib.sha256(b"exact").hexdigest()
    manifest = {
        "schema_version": 1,
        "source": {"revision": artifact["revision"]},
        "files": [{"path": "data.bin", "bytes": 5, "sha256": digest}],
        "totals": {"files": 1, "bytes": 5},
    }
    assert repro_worker.validate_artifact_manifest(manifest, artifact, payload)[0]["sha256"] == digest
    manifest["files"][0]["path"] = "../root"
    with pytest.raises(repro_worker.WorkerError, match="safe relative"):
        repro_worker.validate_artifact_manifest(manifest, artifact)


def test_explicit_payload_versions_are_validated_and_staged_without_unversioned_sync(tmp_path):
    raw = make_spec(tmp_path)
    content = b"exact-versioned-payload"
    digest = hashlib.sha256(content).hexdigest()
    manifest = {
        "schema_version": 1,
        "source": {
            "provider": "huggingface",
            "repo_id": "physical-intelligence/libero",
            "revision": raw["artifacts"][0]["revision"],
        },
        "dataset": {"key": "libero", "codebase_version": "v2.0", "local_dirname": "libero"},
        "files": [{"path": "data.bin", "bytes": len(content), "sha256": digest}],
        "totals": {"files": 1, "bytes": len(content)},
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    artifact = raw["artifacts"][0]
    artifact["manifest"]["sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    artifact["payload_objects"] = [{"path": "data.bin", "version_id": "payload-v7", "sha256": digest}]
    spec = repro_worker.validate_worker_spec(raw)
    artifact = spec["artifacts"][0]
    manifest_location = repro_worker.parse_s3_uri(artifact["manifest"]["s3_uri"])
    payload_location = repro_worker.parse_s3_uri(artifact["payload_s3_uri"], prefix=True)
    objects = {
        (manifest_location.key, artifact["manifest"]["version_id"]): manifest_bytes,
        (f"{payload_location.key}/data.bin", "payload-v7"): content,
    }
    calls = []

    def runner(argv):
        argv = list(argv)
        calls.append(argv)
        assert argv[:3] == ["aws", "s3api", "get-object"]
        key = argv[argv.index("--key") + 1]
        version = argv[argv.index("--version-id") + 1]
        pathlib.Path(argv[-1]).write_bytes(objects[(key, version)])
        return json.dumps({"VersionId": version})

    staged = repro_worker.stage_artifact(artifact, spec, tmp_path / "scratch", runner)

    assert staged["files"] == 1
    assert pathlib.Path(staged["destination"], "data.bin").read_bytes() == content
    assert len(calls) == 2
    assert all(call[:3] == ["aws", "s3api", "get-object"] for call in calls)

    incomplete = {**artifact, "payload_objects": []}
    with pytest.raises(repro_worker.WorkerError, match="non-empty list"):
        repro_worker.validate_worker_spec({**raw, "artifacts": [incomplete]})


def test_converted_teacher_cannot_fall_back_to_unversioned_payload_sync():
    artifact = {
        "name": "libero_teacher_pytorch",
        "kind": "checkpoint",
        "revision": "9" * 64,
        "manifest": {"s3_uri": "s3://bucket/manifest", "version_id": "v1", "sha256": "a" * 64},
        "payload_s3_uri": "s3://bucket/checkpoint/",
        "destination": "pi05_libero_pytorch",
    }
    manifest = {
        "source": {"provider": "openpi-jax-to-pytorch", "revision": artifact["revision"]},
        "files": [{"path": "model.safetensors", "bytes": 5, "sha256": "b" * 64}],
    }

    with pytest.raises(repro_worker.WorkerError, match="requires explicit versioned payload_objects"):
        repro_worker.exact_artifact_payload_pins(manifest, artifact)


def test_complete_staged_input_tree_is_readable_and_nonwritable_for_container_uid(tmp_path):
    root = tmp_path / "inputs"
    payload = root / "checkpoints" / "model" / "model.safetensors"
    manifest = root / ".manifests" / "model.json"
    reserved = [root / name for name in repro_worker.RESERVED_INPUT_MOUNTPOINTS]
    payload.parent.mkdir(parents=True, mode=0o700)
    manifest.parent.mkdir(parents=True, mode=0o700)
    for path in reserved:
        path.mkdir(mode=0o700)
    payload.write_bytes(b"model")
    manifest.write_text("{}")

    repro_worker.make_staged_input_container_readable(root)

    for path in (root, root / "checkpoints", payload.parent, manifest.parent, *reserved):
        assert path.stat().st_mode & 0o777 == 0o555
    for path in (payload, manifest):
        assert path.stat().st_mode & 0o777 == 0o444

    # Restore owner directory permissions so pytest can remove its tree.
    for path in (root, root / "checkpoints", payload.parent, manifest.parent, *reserved):
        path.chmod(0o755)


def test_staged_input_handoff_rejects_missing_or_nonempty_reserved_mountpoint(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    for name in repro_worker.RESERVED_INPUT_MOUNTPOINTS:
        (root / name).mkdir()
    (root / "evidence/payload").write_text("collision")
    with pytest.raises(repro_worker.WorkerError, match="reserved input mountpoint"):
        repro_worker.make_staged_input_container_readable(root)


def test_source_evidence_and_deadline_reserve_are_checked(tmp_path):
    spec = make_spec(tmp_path)
    evidence = {
        "schema_version": 2,
        "source": dict(spec["source"]),
        "controller_source": dict(spec["controller_source"]),
        "bundle_sha256_actual": spec["source"]["sha256"],
        "head_commit": spec["source"]["commit"],
        "source_clean": True,
        "source_fsck_full": True,
        "controller_bundle_sha256_actual": spec["controller_source"]["sha256"],
        "controller_head_commit": spec["controller_source"]["commit"],
        "controller_source_clean": True,
        "controller_source_fsck_full": True,
        "checkout_path": "/opt/pi05/model-source",
        "controller_checkout_path": "/opt/pi05/controller-source",
    }
    repro_worker.validate_source_evidence(spec, evidence)
    evidence["head_commit"] = "0" * 40
    with pytest.raises(repro_worker.WorkerError, match="checked-out"):
        repro_worker.validate_source_evidence(spec, evidence)
    evidence["head_commit"] = spec["source"]["commit"]
    evidence["source_clean"] = False
    with pytest.raises(repro_worker.WorkerError, match="checked-out"):
        repro_worker.validate_source_evidence(spec, evidence)
    evidence["source_clean"] = True
    evidence["controller_head_commit"] = "0" * 40
    with pytest.raises(repro_worker.WorkerError, match="controller bundle"):
        repro_worker.validate_source_evidence(spec, evidence)
    evidence["controller_head_commit"] = spec["controller_source"]["commit"]
    evidence["source_fsck_full"] = False
    with pytest.raises(repro_worker.WorkerError, match="full Git object-integrity"):
        repro_worker.validate_source_evidence(spec, evidence)
    evidence["source_fsck_full"] = True
    evidence["controller_checkout_path"] = evidence["checkout_path"]
    with pytest.raises(repro_worker.WorkerError, match="fixed distinct checkout"):
        repro_worker.validate_source_evidence(spec, evidence)

    now = dt.datetime(2026, 8, 3, 12, 0, tzinfo=dt.UTC)
    hard, soft = repro_worker.validate_launch_metadata(
        spec,
        make_launch_metadata("2026-08-03T14:00:00+00:00"),
        now=now,
    )
    assert hard - soft == dt.timedelta(seconds=900)
    with pytest.raises(repro_worker.WorkerError, match="upload-buffer"):
        repro_worker.validate_launch_metadata(
            spec,
            make_launch_metadata("2026-08-03T12:10:00+00:00"),
            now=now,
        )

    spot = make_launch_metadata("2026-08-03T14:00:00+00:00")
    spot["purchase_option"] = "Spot"
    with pytest.raises(repro_worker.WorkerError, match="On-Demand"):
        repro_worker.validate_launch_metadata(spec, spot, now=now)

    command_path = tmp_path / "run-command.sh"
    command_path.write_text("#!/bin/bash\ntrue\n")
    pinned_command = make_launch_metadata("2026-08-03T14:00:00+00:00")
    pinned_command["command_sha256"] = repro_worker.sha256_file(command_path)
    repro_worker.validate_launch_metadata(spec, pinned_command, now=now, command_path=command_path)
    command_path.write_text("#!/bin/bash\nfalse\n")
    with pytest.raises(repro_worker.WorkerError, match="executing command file"):
        repro_worker.validate_launch_metadata(spec, pinned_command, now=now, command_path=command_path)


def test_run_workspace_is_fresh_and_keeps_host_control_state_out_of_container_ownership(tmp_path, monkeypatch):
    chowned: list[pathlib.Path] = []
    monkeypatch.setattr(repro_worker.os, "chown", lambda path, _uid, _gid: chowned.append(pathlib.Path(path)))

    run_root = repro_worker.create_run_workspace(
        tmp_path,
        "libero-shallow-2k-01",
        expected_owner_uid=repro_worker.os.getuid(),
    )

    for relative in (
        "inputs/runs",
        "inputs/evidence",
        "output/.ready",
        "output/.receipts",
        "output/.spool",
        "output/.active",
    ):
        assert (run_root / relative).stat().st_mode & 0o777 == 0o700
        assert run_root / relative not in chowned
    for relative in ("output/checkpoints", "output/logs", "output/manifests", "output/artifacts", "tmp", "cache"):
        assert run_root / relative in chowned
    with pytest.raises(repro_worker.WorkerError, match="stale run reuse"):
        repro_worker.create_run_workspace(
            tmp_path,
            "libero-shallow-2k-01",
            expected_owner_uid=repro_worker.os.getuid(),
        )


def test_docker_command_is_argv_digest_pinned_and_read_only(tmp_path):
    spec = repro_worker.validate_worker_spec(make_spec(tmp_path))
    command = repro_worker.build_docker_command(
        spec,
        pathlib.Path("/opt/pi05/repo"),
        pathlib.Path("/opt/dlami/nvme"),
    )
    hostname = "pi05-worker-a4b12b31c35ecbae6dedcf62f39d8446"
    assert command[:12] == [
        "docker",
        "run",
        "--name",
        "pi05-shallow-libero-pilot-01",
        "--hostname",
        hostname,
        "--add-host",
        f"{hostname}:127.0.0.1",
        "--gpus",
        "all",
        "--network",
        "none",
    ]
    assert command.count("--hostname") == command.count("--add-host") == 1
    assert "type=bind,src=/opt/pi05/repo,dst=/workspace/openpi,readonly" in command
    assert "type=bind,src=/opt/dlami/nvme/inputs,dst=/mnt/openpi,readonly" in command
    assert "type=bind,src=/opt/dlami/nvme/output,dst=/output" not in command
    for relative in ("checkpoints", "logs", "manifests", "artifacts"):
        assert f"type=bind,src=/opt/dlami/nvme/output/{relative},dst=/output/{relative}" in command
    assert not any(
        "/.ready" in part or "/.receipts" in part or "/.spool" in part or "/.active" in part for part in command
    )
    assert "type=bind,src=/opt/dlami/nvme/output/checkpoints,dst=/mnt/openpi/runs" in command
    assert not any(part.endswith("dst=/workspace/openpi/checkpoints") for part in command)
    assert "type=bind,src=/opt/dlami/nvme/output/artifacts,dst=/mnt/openpi/evidence" in command
    assert "PYTHONPATH=/workspace/openpi/src:/workspace/openpi" in command
    assert "USER=pi05" in command
    assert "LOGNAME=pi05" in command
    assert "PI05_INPUT_LIBERO=/mnt/openpi/datasets/libero" in command
    assert spec["image"]["uri"] in command
    image_index = command.index(spec["image"]["uri"])
    assert command[image_index + 1 :] == spec["container"]["command"]
    assert not any(part in {"sh", "bash", "-c"} for part in command[image_index + 1 :])


def test_worker_container_hostname_is_deterministic_dns_safe_and_validated():
    run_id = "a.with_docker-unsafe.hostname_punctuation"
    first = repro_worker.worker_container_hostname(run_id)
    assert first == repro_worker.worker_container_hostname(run_id)
    assert len(first) <= 63
    assert repro_worker.DOCKER_HOSTNAME_RE.fullmatch(first)
    assert "." not in first
    assert "_" not in first

    for invalid in ("Uppercase", "-leading", "unsafe/name", "a" * 65, 7):
        with pytest.raises(repro_worker.WorkerError, match="invalid run ID for Docker hostname"):
            repro_worker.worker_container_hostname(invalid)


def test_multi_process_torchrun_forces_loopback_transports_without_weakening_network_none(tmp_path):
    spec = repro_worker.validate_worker_spec(make_resume_spec(tmp_path))
    command = repro_worker.build_docker_command(spec, tmp_path / "source", tmp_path / "scratch")
    seed_index = command.index("PI05_SEED=7")
    assert command[seed_index + 1 : seed_index + 9] == [
        "--env",
        "NCCL_SOCKET_IFNAME=lo",
        "--env",
        "GLOO_SOCKET_IFNAME=lo",
        "--env",
        "NCCL_CUMEM_ENABLE=0",
        "--env",
        "NCCL_CUMEM_HOST_ENABLE=0",
    ]
    assert command[command.index("--network") + 1] == "none"
    assert not any(item.startswith("MASTER_ADDR=") for item in command)

    single_process = repro_worker.validate_worker_spec(make_spec(tmp_path))
    single_command = repro_worker.build_docker_command(single_process, tmp_path / "source", tmp_path / "scratch")
    assert "NCCL_SOCKET_IFNAME=lo" not in single_command
    assert "GLOO_SOCKET_IFNAME=lo" not in single_command
    assert "NCCL_CUMEM_ENABLE=0" not in single_command
    assert "NCCL_CUMEM_HOST_ENABLE=0" not in single_command


def test_multi_process_torchrun_loopback_contract_is_fail_closed(tmp_path):
    for key in repro_worker.TORCHRUN_LOOPBACK_ENVIRONMENT:
        override = make_resume_spec(tmp_path)
        override["container"]["environment"][key] = "^docker0,lo"
        with pytest.raises(repro_worker.WorkerError, match="unsafe container environment"):
            repro_worker.validate_worker_spec(override)

    for key in ("USER", "LOGNAME"):
        override = make_spec(tmp_path)
        override["container"]["environment"][key] = "attacker-controlled"
        with pytest.raises(repro_worker.WorkerError, match="unsafe container environment"):
            repro_worker.validate_worker_spec(override)

    symbolic_process_count = make_resume_spec(tmp_path)
    option_index = symbolic_process_count["container"]["command"].index("--nproc-per-node=2")
    symbolic_process_count["container"]["command"][option_index] = "--nproc-per-node=gpu"
    with pytest.raises(repro_worker.WorkerError, match="explicit positive integer"):
        repro_worker.validate_worker_spec(symbolic_process_count)

    non_standalone = make_resume_spec(tmp_path)
    non_standalone["container"]["command"].remove("--standalone")
    with pytest.raises(repro_worker.WorkerError, match="network none requires --standalone"):
        repro_worker.validate_worker_spec(non_standalone)


def test_all_nested_mnt_openpi_mount_destinations_exist_before_readonly_handoff(tmp_path, monkeypatch):
    monkeypatch.setattr(repro_worker.os, "chown", lambda *_args: None)
    run_root = repro_worker.create_run_workspace(
        tmp_path,
        "nested-mount-topology",
        expected_owner_uid=tmp_path.stat().st_uid,
    )
    inputs = run_root / "inputs"
    repro_worker.make_staged_input_container_readable(inputs)
    spec = repro_worker.validate_worker_spec(make_spec(tmp_path))
    command = repro_worker.build_docker_command(spec, tmp_path / "source", run_root)
    mounts = [command[index + 1] for index, item in enumerate(command) if item == "--mount"]
    nested_destinations = {
        option.removeprefix("dst=")
        for mount in mounts
        for option in mount.split(",")
        if option.startswith("dst=/mnt/openpi/")
    }
    assert nested_destinations == {"/mnt/openpi/runs", "/mnt/openpi/evidence"}
    for destination in nested_destinations:
        mountpoint = inputs / pathlib.PurePosixPath(destination).relative_to("/mnt/openpi")
        assert mountpoint.is_dir()
        assert not any(mountpoint.iterdir())
        assert mountpoint.stat().st_mode & 0o777 == 0o555


def test_training_worker_requires_the_single_writable_checkpoint_base(tmp_path):
    missing = make_spec(tmp_path)
    option = missing["container"]["command"].index("--checkpoint-base-dir")
    del missing["container"]["command"][option : option + 2]
    with pytest.raises(repro_worker.WorkerError, match="exactly one --checkpoint-base-dir"):
        repro_worker.validate_worker_spec(missing)

    wrong = make_spec(tmp_path)
    option = wrong["container"]["command"].index("--checkpoint-base-dir")
    wrong["container"]["command"][option + 1] = "/workspace/openpi/checkpoints"
    with pytest.raises(repro_worker.WorkerError, match="must be /mnt/openpi/runs"):
        repro_worker.validate_worker_spec(wrong)

    duplicated = make_spec(tmp_path)
    duplicated["container"]["command"].extend(["--checkpoint-base-dir", "/mnt/openpi/runs"])
    with pytest.raises(repro_worker.WorkerError, match="exactly one --checkpoint-base-dir"):
        repro_worker.validate_worker_spec(duplicated)


def test_host_json_publication_is_create_once_and_does_not_follow_collision(tmp_path):
    destination = tmp_path / "manifests/run.json"
    repro_worker._write_json_new(destination, {"status": "first"})  # noqa: SLF001
    assert json.loads(destination.read_text()) == {"status": "first"}
    with pytest.raises(repro_worker.WorkerError, match="refusing overwrite"):
        repro_worker._write_json_new(destination, {"status": "second"})  # noqa: SLF001
    assert json.loads(destination.read_text()) == {"status": "first"}

    victim = tmp_path / "victim.json"
    victim.write_text("preserve\n")
    collision = tmp_path / "manifests/collision.json"
    collision.symlink_to(victim)
    with pytest.raises(repro_worker.WorkerError, match="refusing overwrite"):
        repro_worker._write_json_new(collision, {"status": "attack"})  # noqa: SLF001
    assert victim.read_text() == "preserve\n"


def test_resume_contract_binds_input_target_training_command_and_output(tmp_path):
    spec = repro_worker.validate_worker_spec(make_resume_spec(tmp_path))
    command = repro_worker.build_docker_command(spec, tmp_path / "source", tmp_path / "scratch")
    assert "PI05_RESUME_CHECKPOINT=/mnt/openpi/runs/pi05_libero_l09_distill/libero-shallow/2000" in command

    traversal = make_resume_spec(tmp_path)
    traversal["resume_checkpoint"]["target"] = "../escape/2000"
    with pytest.raises(repro_worker.WorkerError, match="safe relative"):
        repro_worker.validate_worker_spec(traversal)

    wrong_command = make_resume_spec(tmp_path)
    wrong_command["container"]["command"].remove("--resume")
    with pytest.raises(repro_worker.WorkerError, match="requires --resume"):
        repro_worker.validate_worker_spec(wrong_command)

    unreported_target = make_resume_spec(tmp_path)
    unreported_target["expected_outputs"] = []
    with pytest.raises(repro_worker.WorkerError, match="declare one published checkpoint"):
        repro_worker.validate_worker_spec(unreported_target)


def test_single_gpu_shallow_resume_uses_exact_reviewed_ladder_and_argv(tmp_path):
    for source_step, target_step in ((2_000, 5_000), (5_000, 10_000), (10_000, 20_000), (20_000, 30_000)):
        validated = repro_worker.validate_worker_spec(
            make_single_gpu_shallow_resume_spec(
                tmp_path,
                source_step=source_step,
                target_step=target_step,
            )
        )
        assert validated["container"]["command"][0] == "python"

    skipped_checkpoint = make_single_gpu_shallow_resume_spec(tmp_path, source_step=2_000, target_step=10_000)
    with pytest.raises(repro_worker.WorkerError, match="reviewed 2k->5k->10k->20k->30k ladder"):
        repro_worker.validate_worker_spec(skipped_checkpoint)

    equals_override = make_single_gpu_shallow_resume_spec(tmp_path)
    equals_override["container"]["command"].append("--batch-size=8")
    with pytest.raises(repro_worker.WorkerError, match="equals-form"):
        repro_worker.validate_worker_spec(equals_override)

    negated_override = make_single_gpu_shallow_resume_spec(tmp_path)
    negated_override["container"]["command"].append("--no-wandb-enabled")
    with pytest.raises(repro_worker.WorkerError, match="negated recipe options"):
        repro_worker.validate_worker_spec(negated_override)

    arbitrary_override = make_single_gpu_shallow_resume_spec(tmp_path)
    arbitrary_override["container"]["command"].extend(["--batch-size", "8"])
    with pytest.raises(repro_worker.WorkerError, match="unreviewed options"):
        repro_worker.validate_worker_spec(arbitrary_override)

    wrong_save_interval = make_single_gpu_shallow_resume_spec(tmp_path)
    option = wrong_save_interval["container"]["command"].index("--save-interval")
    wrong_save_interval["container"]["command"][option + 1] = "3000"
    with pytest.raises(repro_worker.WorkerError, match="exact 5000-step save interval"):
        repro_worker.validate_worker_spec(wrong_save_interval)

    reordered = make_single_gpu_shallow_resume_spec(tmp_path)
    seed_index = reordered["container"]["command"].index("--seed")
    seed_pair = reordered["container"]["command"][seed_index : seed_index + 2]
    del reordered["container"]["command"][seed_index : seed_index + 2]
    reordered["container"]["command"].extend(seed_pair)
    with pytest.raises(repro_worker.WorkerError, match="exact reviewed direct-Python argv"):
        repro_worker.validate_worker_spec(reordered)


def test_single_gpu_droid_shallow_resume_restores_reviewed_parallel_loader(tmp_path):
    spec = make_single_gpu_shallow_resume_spec(tmp_path, track="droid")
    validated = repro_worker.validate_worker_spec(spec)
    command = validated["container"]["command"]
    assert command[:3] == ["python", "scripts/train_pytorch.py", "pi05_droid_l09_distill"]
    assert command.count("--no-wandb-enabled") == 1
    assert "--num-workers" not in command

    wandb_enabled = make_single_gpu_shallow_resume_spec(tmp_path, track="droid")
    wandb_enabled["container"]["command"].remove("--no-wandb-enabled")
    with pytest.raises(repro_worker.WorkerError, match="exact reviewed direct-Python argv"):
        repro_worker.validate_worker_spec(wandb_enabled)

    serialized_loader = make_single_gpu_shallow_resume_spec(tmp_path, track="droid")
    serialized_loader["container"]["command"].extend(["--num-workers", "0"])
    with pytest.raises(repro_worker.WorkerError, match="unreviewed options"):
        repro_worker.validate_worker_spec(serialized_loader)


def test_single_gpu_shallow_resume_requires_corrective_g7e_launch_metadata(tmp_path):
    now = dt.datetime.now(dt.UTC)
    metadata = {
        **make_launch_metadata((now + dt.timedelta(hours=6)).isoformat()),
        "category": "corrective_run",
        "workload": "shallow_training",
        "instance_type": "g7e.4xlarge",
    }
    for track in ("libero", "droid"):
        spec = repro_worker.validate_worker_spec(make_single_gpu_shallow_resume_spec(tmp_path, track=track))
        repro_worker.validate_launch_metadata(spec, metadata, now=now)

        for field, value in (
            ("category", "shallow_training"),
            ("workload", "evaluation"),
            ("instance_type", "g7e.12xlarge"),
        ):
            wrong = {**metadata, field: value}
            with pytest.raises(repro_worker.WorkerError, match="corrective_run/shallow_training on g7e.4xlarge"):
                repro_worker.validate_launch_metadata(spec, wrong, now=now)


def test_fresh_snapflow_contract_binds_accepted_source_recipe_and_output(tmp_path):
    spec = repro_worker.validate_worker_spec(make_snapflow_spec(tmp_path))
    assert spec["container"]["command"][0] == "python"

    missing_source = make_snapflow_spec(tmp_path)
    option = missing_source["container"]["command"].index("--pytorch-weight-path")
    del missing_source["container"]["command"][option : option + 2]
    with pytest.raises(repro_worker.WorkerError, match="--pytorch-weight-path"):
        repro_worker.validate_worker_spec(missing_source)

    wrong_source = make_snapflow_spec(tmp_path)
    option = wrong_source["container"]["command"].index("--pytorch-weight-path")
    wrong_source["container"]["command"][option + 1] = (
        "/mnt/openpi/checkpoints/pi05_libero_l09_distill/other-shallow/5000"
    )
    with pytest.raises(repro_worker.WorkerError, match="exactly one staged accepted Shallow"):
        repro_worker.validate_worker_spec(wrong_source)

    wrong_batch = make_snapflow_spec(tmp_path)
    wrong_batch["container"]["command"].extend(["--batch-size", "8"])
    with pytest.raises(repro_worker.WorkerError, match="published value 4"):
        repro_worker.validate_worker_spec(wrong_batch)

    unpublishable = make_snapflow_spec(tmp_path)
    unpublishable["expected_outputs"][0].pop("publish_destination")
    with pytest.raises(repro_worker.WorkerError, match="publish its exact numeric checkpoint"):
        repro_worker.validate_worker_spec(unpublishable)

    unbounded_fresh = make_snapflow_spec(tmp_path)
    option = unbounded_fresh["container"]["command"].index("--num-train-steps")
    unbounded_fresh["container"]["command"][option + 1] = "30000"
    unbounded_fresh["expected_outputs"][0]["path"] = "checkpoints/pi05_libero_l09_snapflow/libero-snapflow/30000"
    unbounded_fresh["expected_outputs"][0]["publish_destination"] = "pi05_libero_l09_snapflow/libero-snapflow/30000"
    with pytest.raises(repro_worker.WorkerError, match="initial pilot must target exactly 5000"):
        repro_worker.validate_worker_spec(unbounded_fresh)

    equals_override = make_snapflow_spec(tmp_path)
    equals_override["container"]["command"].append("--batch-size=8")
    with pytest.raises(repro_worker.WorkerError, match="equals-form"):
        repro_worker.validate_worker_spec(equals_override)

    unreviewed_model_override = make_snapflow_spec(tmp_path)
    unreviewed_model_override["container"]["command"].extend(["--model.snapflow-alpha", "1.0"])
    with pytest.raises(repro_worker.WorkerError, match="unreviewed"):
        repro_worker.validate_worker_spec(unreviewed_model_override)

    wrong_save_interval = make_snapflow_spec(tmp_path)
    option = wrong_save_interval["container"]["command"].index("--save-interval")
    wrong_save_interval["container"]["command"][option + 1] = "1000"
    with pytest.raises(repro_worker.WorkerError, match="save exactly"):
        repro_worker.validate_worker_spec(wrong_save_interval)


def test_snapflow_one_batch_contract_requires_safe_loader_and_nonpromotable_output(tmp_path):
    repro_worker.validate_worker_spec(make_snapflow_spec(tmp_path, diagnostic=True))

    unsafe_loader = make_snapflow_spec(tmp_path, diagnostic=True)
    option = unsafe_loader["container"]["command"].index("--num-workers")
    unsafe_loader["container"]["command"][option + 1] = "4"
    with pytest.raises(repro_worker.WorkerError, match="num_workers=0"):
        repro_worker.validate_worker_spec(unsafe_loader)

    promoted_diagnostic = make_snapflow_spec(tmp_path, diagnostic=True)
    promoted_diagnostic["expected_outputs"][0]["publish_destination"] = (
        "pi05_libero_l09_snapflow/libero-snapflow-overfit/300"
    )
    with pytest.raises(repro_worker.WorkerError, match="must not be published"):
        repro_worker.validate_worker_spec(promoted_diagnostic)

    negated_diagnostic = make_snapflow_spec(tmp_path, diagnostic=True)
    negated_diagnostic["container"]["command"].append("--no-one-batch-overfit")
    with pytest.raises(repro_worker.WorkerError, match="unreviewed"):
        repro_worker.validate_worker_spec(negated_diagnostic)


def test_droid_snapflow_contract_accepts_exact_expert_bc_source():
    source = "pi05_droid_l09_expert_bc_25/droid-bc25/1500"
    command = [
        "python",
        "scripts/train_pytorch.py",
        "pi05_droid_l09_snapflow",
        "--exp-name",
        "droid-snapflow",
        "--checkpoint-base-dir",
        "/mnt/openpi/runs",
        "--pytorch-weight-path",
        f"/mnt/openpi/checkpoints/{source}",
        "--seed",
        "42",
        "--num-train-steps",
        "5000",
        "--save-interval",
        "5000",
        "--num-workers",
        "0",
        "--no-wandb-enabled",
    ]
    artifacts = [{"name": "bc25", "kind": "checkpoint", "destination": source}]
    outputs = [
        {
            "name": "snapflow",
            "kind": "checkpoint",
            "path": "checkpoints/pi05_droid_l09_snapflow/droid-snapflow/5000",
            "publish_destination": "pi05_droid_l09_snapflow/droid-snapflow/5000",
        }
    ]

    repro_worker.validate_fresh_snapflow_training_contract(
        command,
        "pi05_droid_l09_snapflow",
        artifacts,
        outputs,
    )

    unsafe_loader = command.copy()
    option = unsafe_loader.index("--num-workers")
    del unsafe_loader[option : option + 2]
    with pytest.raises(repro_worker.WorkerError, match="--num-workers"):
        repro_worker.validate_fresh_snapflow_training_contract(
            unsafe_loader,
            "pi05_droid_l09_snapflow",
            artifacts,
            outputs,
        )


def test_snapflow_resume_contract_enforces_single_gpu_recipe_and_reviewed_ladder(tmp_path):
    repro_worker.validate_worker_spec(make_droid_snapflow_resume_spec(tmp_path))

    unsafe_loader = make_droid_snapflow_resume_spec(tmp_path)
    option = unsafe_loader["container"]["command"].index("--num-workers")
    del unsafe_loader["container"]["command"][option : option + 2]
    with pytest.raises(repro_worker.WorkerError, match="--num-workers"):
        repro_worker.validate_worker_spec(unsafe_loader)

    wandb_enabled = make_droid_snapflow_resume_spec(tmp_path)
    wandb_enabled["container"]["command"].remove("--no-wandb-enabled")
    with pytest.raises(repro_worker.WorkerError, match="disabled W&B"):
        repro_worker.validate_worker_spec(wandb_enabled)

    multi_gpu = make_droid_snapflow_resume_spec(tmp_path)
    multi_gpu["container"]["command"][:1] = ["torchrun", "--standalone", "--nproc-per-node=2"]
    with pytest.raises(repro_worker.WorkerError, match="direct single-process Python"):
        repro_worker.validate_worker_spec(multi_gpu)

    skipped_gate = make_droid_snapflow_resume_spec(tmp_path)
    option = skipped_gate["container"]["command"].index("--num-train-steps")
    skipped_gate["container"]["command"][option + 1] = "20000"
    skipped_gate["expected_outputs"][0]["path"] = "checkpoints/pi05_droid_l09_snapflow/droid-snapflow/20000"
    skipped_gate["expected_outputs"][0]["publish_destination"] = "pi05_droid_l09_snapflow/droid-snapflow/20000"
    with pytest.raises(repro_worker.WorkerError, match="continuation ladder"):
        repro_worker.validate_worker_spec(skipped_gate)

    equals_override = make_droid_snapflow_resume_spec(tmp_path)
    equals_override["container"]["command"].append("--gradient-accumulation-steps=9")
    with pytest.raises(repro_worker.WorkerError, match="equals-form"):
        repro_worker.validate_worker_spec(equals_override)

    negated_resume = make_droid_snapflow_resume_spec(tmp_path)
    negated_resume["container"]["command"].append("--no-resume")
    with pytest.raises(repro_worker.WorkerError, match="unreviewed"):
        repro_worker.validate_worker_spec(negated_resume)

    model_only_override = make_droid_snapflow_resume_spec(tmp_path)
    model_only_override["container"]["command"].extend(
        ["--pytorch-weight-path", "/mnt/openpi/checkpoints/pi05_droid_l09_distill/other/5000"]
    )
    with pytest.raises(repro_worker.WorkerError, match="full state"):
        repro_worker.validate_worker_spec(model_only_override)


def test_shallow_worker_stages_every_config_path_at_the_expected_container_location(tmp_path):
    raw = make_spec(tmp_path)
    manifest_pin = {
        "s3_uri": "s3://pi05-repro-752160877725-us-east-2/checkpoints/manifest.json",
        "version_id": "v1",
        "sha256": "e" * 64,
    }
    raw["artifacts"].extend(
        [
            {
                "name": "libero_teacher_jax",
                "kind": "checkpoint",
                "revision": "f" * 64,
                "manifest": manifest_pin,
                "payload_s3_uri": "s3://pi05-repro-752160877725-us-east-2/checkpoints/pi05_libero/checkpoint/",
                "destination": "pi05_libero",
            },
            {
                "name": "libero_teacher_pytorch",
                "kind": "checkpoint",
                "revision": "1" * 64,
                "manifest": {**manifest_pin, "s3_uri": manifest_pin["s3_uri"].replace("manifest", "converted")},
                "payload_s3_uri": "s3://pi05-repro-752160877725-us-east-2/checkpoints/pi05_libero_pytorch/checkpoint/",
                "destination": "pi05_libero_pytorch",
            },
        ]
    )
    spec = repro_worker.validate_worker_spec(raw)
    command = repro_worker.build_docker_command(spec, tmp_path / "source", tmp_path / "scratch")
    assert "PI05_INPUT_LIBERO=/mnt/openpi/datasets/libero" in command
    assert "PI05_INPUT_LIBERO_TEACHER_JAX=/mnt/openpi/checkpoints/pi05_libero" in command
    assert "PI05_INPUT_LIBERO_TEACHER_PYTORCH=/mnt/openpi/checkpoints/pi05_libero_pytorch" in command


def test_image_digest_must_carry_the_pinned_source_revision(tmp_path):
    spec = repro_worker.validate_worker_spec(make_spec(tmp_path))
    labels = {
        "org.opencontainers.image.revision": spec["source"]["commit"],
        "ai.openpi.image-purpose": "policy",
        "ai.openpi.lerobot-runtime": "v2",
        "ai.openpi.lerobot-revision": "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5",
        "ai.openpi.video-decoder": "pyav",
        "ai.openpi.onnxruntime-gpu-version": "1.26.0",
    }
    assert repro_worker.validate_image_identity(spec, [spec["image"]["uri"]], labels) == [spec["image"]["uri"]]

    with pytest.raises(repro_worker.WorkerError, match=r"source\.commit"):
        repro_worker.validate_image_identity(
            spec,
            [spec["image"]["uri"]],
            {"org.opencontainers.image.revision": "0" * 40},
        )
    with pytest.raises(repro_worker.WorkerError, match="requested immutable digest"):
        repro_worker.validate_image_identity(spec, [], labels)

    wrong_runtime = {**labels, "ai.openpi.lerobot-runtime": "v3"}
    with pytest.raises(repro_worker.WorkerError, match="LeRobot runtime"):
        repro_worker.validate_image_identity(spec, [spec["image"]["uri"]], wrong_runtime)

    wrong_decoder = {**labels, "ai.openpi.video-decoder": "torchcodec"}
    with pytest.raises(repro_worker.WorkerError, match="video decoder"):
        repro_worker.validate_image_identity(spec, [spec["image"]["uri"]], wrong_decoder)

    wrong_ort = {**labels, "ai.openpi.onnxruntime-gpu-version": "1.28.0"}
    with pytest.raises(repro_worker.WorkerError, match="ONNX Runtime"):
        repro_worker.validate_image_identity(spec, [spec["image"]["uri"]], wrong_ort)

    wrong_purpose = {**labels, "ai.openpi.image-purpose": "tensorrt-compiler"}
    with pytest.raises(repro_worker.WorkerError, match="purpose label"):
        repro_worker.validate_image_identity(spec, [spec["image"]["uri"]], wrong_purpose)


def test_tensorrt_image_identity_requires_exact_toolchain_and_no_lerobot_claim(tmp_path):
    spec = repro_worker.validate_worker_spec(make_tensorrt_spec(tmp_path))
    labels = {
        "org.opencontainers.image.revision": spec["source"]["commit"],
        "ai.openpi.image-purpose": "tensorrt-compiler",
        **{label: spec["image"]["toolchain"][key] for key, label in repro_worker.TENSORRT_TOOLCHAIN_LABELS.items()},
    }
    assert repro_worker.validate_image_identity(spec, [spec["image"]["uri"]], labels) == [spec["image"]["uri"]]

    wrong_toolchain = {**labels, "ai.openpi.modelopt-version": "0.44.0"}
    with pytest.raises(repro_worker.WorkerError, match="toolchain label"):
        repro_worker.validate_image_identity(spec, [spec["image"]["uri"]], wrong_toolchain)

    false_lerobot_claim = {**labels, "ai.openpi.lerobot-runtime": "v2"}
    with pytest.raises(repro_worker.WorkerError, match="must not claim"):
        repro_worker.validate_image_identity(spec, [spec["image"]["uri"]], false_lerobot_claim)


def test_tensorrt_policy_image_identity_verifies_compiler_source_runtime_and_toolchain(tmp_path):
    spec = repro_worker.validate_worker_spec(make_tensorrt_policy_spec(tmp_path))
    labels = {
        "org.opencontainers.image.revision": spec["source"]["commit"],
        "ai.openpi.image-purpose": "tensorrt-policy",
        "ai.openpi.lerobot-runtime": "v2",
        "ai.openpi.lerobot-revision": repro_worker.LEROBOT_REVISIONS["v2"],
        "ai.openpi.parent-tensorrt-compiler-image": spec["image"]["parent_tensorrt_compiler_image"],
        "ai.openpi.parent-tensorrt-compiler-source-revision": spec["image"]["parent_tensorrt_compiler_source_revision"],
        **repro_worker.TENSORRT_POLICY_LABELS,
        **{label: spec["image"]["toolchain"][key] for key, label in repro_worker.TENSORRT_TOOLCHAIN_LABELS.items()},
    }
    assert repro_worker.validate_image_identity(spec, [spec["image"]["uri"]], labels) == [spec["image"]["uri"]]

    for label in (
        "ai.openpi.parent-tensorrt-compiler-image",
        "ai.openpi.parent-tensorrt-compiler-source-revision",
        "ai.openpi.tensorrt-version",
        "ai.openpi.policy-runtime",
    ):
        wrong = {**labels, label: "wrong"}
        with pytest.raises(repro_worker.WorkerError, match=re.escape(label)):
            repro_worker.validate_image_identity(spec, [spec["image"]["uri"]], wrong)


def test_libero_evaluator_identity_requires_dedicated_exact_image_contract(tmp_path):
    spec = repro_worker.validate_worker_spec(make_libero_evaluator_spec(tmp_path))
    labels = {
        "org.opencontainers.image.revision": spec["source"]["commit"],
        "ai.openpi.image-purpose": "libero-evaluator",
        "ai.openpi.policy-backend": "eager",
        "ai.openpi.lerobot-runtime": "v2",
        "ai.openpi.lerobot-revision": repro_worker.LEROBOT_REVISIONS["v2"],
        "ai.openpi.libero-simulator-revision": repro_worker.LIBERO_SIMULATOR_REVISION,
        "ai.openpi.libero-requirements-sha256": repro_worker.LIBERO_REQUIREMENTS_SHA256,
        "ai.openpi.parent-policy-image": spec["image"]["parent_policy_image"],
        "ai.openpi.onnxruntime-gpu-version": repro_worker.POLICY_ONNXRUNTIME_GPU_VERSION,
    }
    assert repro_worker.validate_image_identity(spec, [spec["image"]["uri"]], labels) == [spec["image"]["uri"]]

    normal_policy_labels = {
        "org.opencontainers.image.revision": spec["source"]["commit"],
        "ai.openpi.image-purpose": "policy",
        "ai.openpi.lerobot-runtime": "v2",
        "ai.openpi.lerobot-revision": repro_worker.LEROBOT_REVISIONS["v2"],
    }
    with pytest.raises(repro_worker.WorkerError, match="purpose label"):
        repro_worker.validate_image_identity(spec, [spec["image"]["uri"]], normal_policy_labels)

    for label in (
        "ai.openpi.policy-backend",
        "ai.openpi.lerobot-runtime",
        "ai.openpi.lerobot-revision",
        "ai.openpi.libero-simulator-revision",
        "ai.openpi.libero-requirements-sha256",
        "ai.openpi.parent-policy-image",
    ):
        wrong = {**labels, label: "wrong"}
        with pytest.raises(repro_worker.WorkerError, match=re.escape(label)):
            repro_worker.validate_image_identity(spec, [spec["image"]["uri"]], wrong)

    wrong_ort = {**labels, "ai.openpi.onnxruntime-gpu-version": "wrong"}
    with pytest.raises(repro_worker.WorkerError, match="ONNX Runtime"):
        repro_worker.validate_image_identity(spec, [spec["image"]["uri"]], wrong_ort)

    compiler_claim = {**labels, "ai.openpi.tensorrt-version": "11.0.0.114"}
    with pytest.raises(repro_worker.WorkerError, match="TensorRT compiler identity"):
        repro_worker.validate_image_identity(spec, [spec["image"]["uri"]], compiler_claim)


def test_tensorrt_libero_identity_includes_combined_parent_compiler_source_and_toolchain(tmp_path):
    spec = repro_worker.validate_worker_spec(make_tensorrt_libero_evaluator_spec(tmp_path))
    labels = {
        "org.opencontainers.image.revision": spec["source"]["commit"],
        "ai.openpi.image-purpose": "libero-evaluator",
        "ai.openpi.policy-backend": "tensorrt",
        "ai.openpi.lerobot-runtime": "v2",
        "ai.openpi.lerobot-revision": repro_worker.LEROBOT_REVISIONS["v2"],
        "ai.openpi.libero-simulator-revision": repro_worker.LIBERO_SIMULATOR_REVISION,
        "ai.openpi.libero-requirements-sha256": repro_worker.LIBERO_REQUIREMENTS_SHA256,
        "ai.openpi.parent-policy-image": spec["image"]["parent_policy_image"],
        "ai.openpi.parent-tensorrt-compiler-image": spec["image"]["parent_tensorrt_compiler_image"],
        "ai.openpi.parent-tensorrt-compiler-source-revision": spec["image"]["parent_tensorrt_compiler_source_revision"],
        **repro_worker.TENSORRT_POLICY_LABELS,
        **{label: spec["image"]["toolchain"][key] for key, label in repro_worker.TENSORRT_TOOLCHAIN_LABELS.items()},
    }
    repro_worker.validate_image_identity(spec, [spec["image"]["uri"]], labels)

    wrong_parent = {**labels, "ai.openpi.parent-policy-image": "wrong"}
    with pytest.raises(repro_worker.WorkerError, match="parent-policy-image"):
        repro_worker.validate_image_identity(spec, [spec["image"]["uri"]], wrong_parent)
    wrong_toolchain = {**labels, "ai.openpi.cuda-version": "12.0"}
    with pytest.raises(repro_worker.WorkerError, match="cuda-version"):
        repro_worker.validate_image_identity(spec, [spec["image"]["uri"]], wrong_toolchain)


def test_libero_evaluator_docker_command_renders_headless_gpu_contract(tmp_path):
    spec = repro_worker.validate_worker_spec(make_libero_evaluator_spec(tmp_path))
    command = repro_worker.build_docker_command(spec, tmp_path / "source", tmp_path / "scratch")
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics" in command
    assert "MUJOCO_EGL_DEVICE_ID=0" in command
    assert "NCCL_SOCKET_IFNAME=lo" not in command
    assert "GLOO_SOCKET_IFNAME=lo" not in command
    image_index = command.index(spec["image"]["uri"])
    assert command[image_index + 1 :] == spec["container"]["command"]


def test_exact_placement_compares_live_imds_and_worker_injects_instance_identity(tmp_path):
    spec = repro_worker.validate_worker_spec(make_tensorrt_libero_evaluator_spec(tmp_path))
    launch = {"instance_type": "g7e.4xlarge"}
    identity = {
        "accountId": repro_worker.EXPECTED_ACCOUNT,
        "region": repro_worker.EXPECTED_REGION,
        "instanceType": "g7e.4xlarge",
        "instanceId": spec["placement"]["instance_id"],
    }
    instance_id, instance_type = repro_worker.validate_instance_identity(spec, launch, identity)
    command = repro_worker.build_docker_command(
        spec,
        tmp_path / "source",
        tmp_path / "scratch",
        instance_id=instance_id,
        instance_type=instance_type,
    )
    assert f"PI05_INSTANCE_ID={instance_id}" in command
    assert f"PI05_INSTANCE_TYPE={instance_type}" in command

    replacement = {**identity, "instanceId": "i-0fedcba9876543210"}
    with pytest.raises(repro_worker.WorkerError, match="exact existing-instance placement"):
        repro_worker.validate_instance_identity(spec, launch, replacement)

    with pytest.raises(repro_worker.WorkerError, match="supplied together from live IMDS"):
        repro_worker.build_docker_command(
            spec,
            tmp_path / "source",
            tmp_path / "scratch",
            instance_id=instance_id,
        )


def test_expected_outputs_are_rooted_and_committed_only_when_present(tmp_path):
    raw = make_spec(tmp_path)
    raw["expected_outputs"] = [
        {"name": "metrics", "kind": "artifact", "path": "artifacts/final"},
        {"name": "weights", "kind": "checkpoint", "path": "checkpoints/final/model.safetensors"},
    ]
    spec = repro_worker.validate_worker_spec(raw)
    root = tmp_path / "output"
    for directory in (".ready", ".receipts", ".spool", "artifacts/final", "checkpoints/final"):
        (root / directory).mkdir(parents=True)
    (root / "artifacts/final/metrics.json").write_text('{"loss": 1}\n')
    (root / "checkpoints/final/model.safetensors").write_bytes(b"weights")
    manager = repro_worker.OutputManager(spec, root, OutputRunner())
    markers = manager.commit_expected_outputs()
    assert {path.name for path in markers} == {
        "expected-metrics.ready.json",
        "expected-weights.ready.json",
    }

    (root / "checkpoints/final/model.safetensors").unlink()
    with pytest.raises(repro_worker.WorkerError, match="does not exist"):
        manager.commit_expected_outputs()

    wrong_root = make_spec(tmp_path)
    wrong_root["expected_outputs"] = [{"name": "metrics", "kind": "artifact", "path": "checkpoints/metrics.json"}]
    with pytest.raises(repro_worker.WorkerError, match="below artifacts"):
        repro_worker.validate_worker_spec(wrong_root)

    publish_log = make_spec(tmp_path)
    publish_log["expected_outputs"] = [
        {
            "name": "log",
            "kind": "log",
            "path": "logs/container.log",
            "publish_destination": "logs",
        }
    ]
    with pytest.raises(repro_worker.WorkerError, match="supported only"):
        repro_worker.validate_worker_spec(publish_log)


class OutputRunner:
    def __init__(self):
        self.calls = []
        self.objects = {}

    def __call__(self, argv):
        argv = list(argv)
        self.calls.append(argv)
        if argv[:3] == ["aws", "s3api", "list-object-versions"]:
            key = argv[argv.index("--prefix") + 1]
            record = self.objects.get(key)
            return json.dumps({"Versions": [] if record is None else [{"Key": key, "VersionId": record["VersionId"]}]})
        if argv[:3] == ["aws", "s3", "cp"]:
            key = argv[4].split("/", 3)[-1]
            source = pathlib.Path(argv[3])
            metadata = dict(item.split("=", 1) for item in argv[argv.index("--metadata") + 1].split(","))
            self.objects[key] = {
                "VersionId": "output-v1",
                "ContentLength": source.stat().st_size,
                "Metadata": metadata,
            }
            return ""
        if argv[:3] == ["aws", "s3api", "head-object"]:
            return json.dumps(self.objects[argv[argv.index("--key") + 1]])
        raise AssertionError(argv)


def test_output_sync_requires_atomic_marker_and_records_s3_version(tmp_path):
    spec = repro_worker.validate_worker_spec(make_spec(tmp_path))
    root = tmp_path / "output"
    for directory in (".ready", ".receipts", ".spool", "artifacts", "checkpoints"):
        (root / directory).mkdir(parents=True)
    artifact = root / "artifacts" / "metrics.json"
    artifact.write_text('{"loss": 1}\n')
    runner = OutputRunner()
    manager = repro_worker.OutputManager(spec, root, runner)
    assert manager.sync_once() == []
    assert not runner.calls
    manager.create_marker("artifact", [artifact], "metrics.ready.json")
    receipts = manager.sync_once()
    assert receipts[0]["artifacts"][0]["version_id"] == "output-v1"
    assert any(call[:3] == ["aws", "s3", "cp"] for call in runner.calls)
    call_count = len(runner.calls)
    manager.sync_once()
    assert len(runner.calls) == call_count


def test_atomic_numeric_checkpoint_is_discovered(tmp_path):
    spec = repro_worker.validate_worker_spec(make_spec(tmp_path))
    root = tmp_path / "output"
    for directory in (".ready", ".receipts", ".spool", "artifacts", "checkpoints/run/5000"):
        (root / directory).mkdir(parents=True)
    (root / "checkpoints/run/5000/model.safetensors").write_bytes(b"weights")
    manager = repro_worker.OutputManager(spec, root, OutputRunner())
    manager.discover_atomic_checkpoints()
    markers = list((root / ".ready").glob("checkpoint-*.ready.json"))
    assert len(markers) == 1
    assert json.loads(markers[0].read_text())["kind"] == "checkpoint"


def test_expected_checkpoint_publishes_worker_compatible_input_manifest(tmp_path):
    raw = make_spec(tmp_path)
    raw["expected_outputs"] = [
        {
            "name": "shallow_checkpoint",
            "kind": "checkpoint",
            "path": "checkpoints/pi05_libero_l09_distill/libero-shallow/2000",
            "publish_destination": "pi05_libero_l09_distill/libero-shallow/2000",
        }
    ]
    spec = repro_worker.validate_worker_spec(raw)
    root = tmp_path / "output"
    for directory in (
        ".ready",
        ".receipts",
        ".spool",
        "artifacts",
        "manifests",
        "checkpoints/pi05_libero_l09_distill/libero-shallow/2000/assets/libero",
    ):
        (root / directory).mkdir(parents=True)
    checkpoint = root / "checkpoints/pi05_libero_l09_distill/libero-shallow/2000"
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "assets/libero/norm_stats.json").write_text('{"stats": true}\n')

    runner = OutputRunner()
    manager = repro_worker.OutputManager(spec, root, runner)
    manager.commit_expected_outputs()
    first_receipts = manager.sync_once()
    published, receipts = manager.publish_expected_inputs(first_receipts)

    assert len(published) == 1
    artifact = published[0]
    assert artifact["kind"] == "checkpoint"
    assert artifact["destination"] == "pi05_libero_l09_distill/libero-shallow/2000"
    assert artifact["payload_s3_uri"].endswith("/checkpoints/pi05_libero_l09_distill/libero-shallow/2000/")
    assert artifact["manifest"]["version_id"] == "output-v1"
    assert len(artifact["revision"]) == 64
    manifest_path = root / "manifests/worker-input-shallow_checkpoint.sha256.json"
    manifest = json.loads(manifest_path.read_text())
    assert [item["path"] for item in manifest["files"]] == [
        "assets/libero/norm_stats.json",
        "model.safetensors",
    ]
    repro_worker.validate_artifact_manifest(manifest, artifact, checkpoint)
    changed_destination = {**artifact, "destination": "different/step"}
    with pytest.raises(repro_worker.WorkerError, match="path, destination, or kind was changed"):
        repro_worker.validate_artifact_manifest(manifest, changed_destination)
    next_spec = make_spec(tmp_path)
    next_spec["artifacts"] = [artifact]
    assert repro_worker.validate_worker_spec(next_spec)["artifacts"][0] == artifact
    assert any(receipt["marker"] == "worker-input-shallow_checkpoint.ready.json" for receipt in receipts)

    pins = repro_worker.versioned_worker_output_pins(manifest, artifact)
    assert pins is not None
    assert [pin["path"] for pin in pins] == ["assets/libero/norm_stats.json", "model.safetensors"]
    assert all(pin["version_id"] == "output-v1" for pin in pins)

    changed_payload = {**artifact, "payload_s3_uri": artifact["payload_s3_uri"].replace("/2000/", "/5000/")}
    with pytest.raises(repro_worker.WorkerError, match="publication path"):
        repro_worker.validate_artifact_manifest(manifest, changed_payload)


def _training_metrics_output(tmp_path, metrics_document=None):
    raw = make_spec(tmp_path)
    raw["container"]["command"] = [
        "torchrun",
        "--standalone",
        "--nproc-per-node=2",
        "scripts/train_pytorch.py",
        "pi05_libero_l09_distill",
        "--checkpoint-base-dir",
        "/mnt/openpi/runs",
        "--seed",
        "7",
    ]
    relative = "checkpoints/pi05_libero_l09_distill/libero-shallow/2000"
    raw["expected_outputs"] = [{"name": "pilot_checkpoint", "kind": "checkpoint", "path": relative}]
    spec = repro_worker.validate_worker_spec(raw)
    root = tmp_path / "output"
    for directory in (".ready", ".receipts", ".spool", relative):
        (root / directory).mkdir(parents=True)
    checkpoint = root / relative
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    if metrics_document is not None:
        (checkpoint / repro_worker.TRAINING_METRICS_FILENAME).write_text(json.dumps(metrics_document))
    manager = repro_worker.OutputManager(spec, root, OutputRunner())
    markers = manager.commit_expected_outputs()
    return spec, root, checkpoint, markers


def _metrics_runtime_context(instance_type="g6e.4xlarge"):
    launch = make_launch_metadata("2026-08-04T12:00:00+00:00")
    launch["instance_type"] = instance_type
    identity = {
        "accountId": repro_worker.EXPECTED_ACCOUNT,
        "region": repro_worker.EXPECTED_REGION,
        "instanceId": "i-0123456789abcdef0",
        "instanceType": instance_type,
    }
    return launch, identity


def _write_expected_json(tmp_path, spec, relative, document):
    root = tmp_path / "output"
    for directory in (".ready", ".receipts", ".spool", pathlib.PurePosixPath(relative).parent.as_posix()):
        (root / directory).mkdir(parents=True, exist_ok=True)
    source = root / relative
    source.write_text(json.dumps(document))
    manager = repro_worker.OutputManager(spec, root, OutputRunner())
    return root, source, manager.commit_expected_outputs()


def test_run_metrics_are_finite_hash_covered_and_training_sidecar_is_mandatory(tmp_path):
    document = {
        "schema_version": 1,
        "config_name": "pi05_libero_l09_distill",
        "exp_name": "libero-shallow",
        "global_step": 2000,
        "metrics": {"loss": 0.25, "kd_cosine": 0.999},
    }
    spec, root, _checkpoint, markers = _training_metrics_output(tmp_path / "current", document)
    metrics, provenance = repro_worker.collect_run_metrics(spec, root, markers)
    path = "checkpoints/pi05_libero_l09_distill/libero-shallow/2000/training-metrics.json"
    assert metrics == {path: document["metrics"]}
    assert provenance == [
        {
            "expected_output": "pilot_checkpoint",
            "path": path,
            "sha256": repro_worker.sha256_file(root / path),
        }
    ]

    old_spec, old_root, _old_checkpoint, old_markers = _training_metrics_output(tmp_path / "old")
    with pytest.raises(repro_worker.WorkerError, match="mandatory hash-covered metrics sidecar"):
        repro_worker.collect_run_metrics(old_spec, old_root, old_markers)


def test_run_metrics_ingest_declared_evaluation_manifest(tmp_path):
    raw = make_libero_evaluator_spec(tmp_path)
    raw["container"]["command"].extend(["--stage", "final", "--seed", "7"])
    relative = "manifests/libero-final.json"
    raw["expected_outputs"] = [{"name": "evaluation_manifest", "kind": "manifest", "path": relative}]
    spec = repro_worker.validate_worker_spec(raw)
    command = spec["container"]["command"][1:]
    launch, identity = _metrics_runtime_context()
    document = {
        "schema_version": 1,
        "project": repro_worker.EXPECTED_PROJECT,
        "kind": "libero-evaluation",
        "run_id": spec["run_id"],
        "started_at": "2026-08-04T10:00:00+00:00",
        "finished_at": "2026-08-04T10:01:00+00:00",
        "source": {"commit": spec["source"]["commit"]},
        "image": {"digest": spec["image"]["digest"]},
        "dataset": {
            "name": "LIBERO fixed benchmark assets",
            "revision": spec["image"]["libero_simulator_revision"],
        },
        "simulator": {},
        "dependencies": {},
        "policy": {},
        "evaluation": {
            "stage": "final",
            "seed": spec["seed"],
            "suites": ["libero_spatial"],
            "trials_per_task": 50,
            "metrics": {"success": 0.97},
        },
        "command": command,
        "child_commands": [],
        "instance": {
            "type": identity["instanceType"],
            "id": identity["instanceId"],
            "identity_recorded_by": "worker run manifest",
        },
        "cost": {"projected_usd": 18.0, "actual_recorded_by": "worker run manifest"},
        "artifacts": [],
    }
    root, source, markers = _write_expected_json(tmp_path, spec, relative, document)

    metrics, provenance = repro_worker.collect_run_metrics(
        spec, root, markers, launch_metadata=launch, instance_identity=identity
    )

    assert metrics == {relative: {"success": 0.97}}
    assert provenance[0]["sha256"] == repro_worker.sha256_file(source)


@pytest.mark.parametrize(
    ("dataset_name", "dataset_revision"),
    [
        ("wrong benchmark", repro_worker.LIBERO_SIMULATOR_REVISION),
        ("LIBERO fixed benchmark assets", "f" * 40),
    ],
)
def test_run_metrics_reject_libero_dataset_identity_mismatch(tmp_path, dataset_name, dataset_revision):
    raw = make_libero_evaluator_spec(tmp_path)
    raw["container"]["command"].extend(["--stage", "base", "--seed", "7"])
    relative = "manifests/libero-base.json"
    raw["expected_outputs"] = [{"name": "evaluation_manifest", "kind": "manifest", "path": relative}]
    spec = repro_worker.validate_worker_spec(raw)
    command = spec["container"]["command"][1:]
    launch, identity = _metrics_runtime_context()
    document = {
        "schema_version": 1,
        "project": repro_worker.EXPECTED_PROJECT,
        "kind": "libero-evaluation",
        "run_id": spec["run_id"],
        "started_at": "2026-08-04T10:00:00+00:00",
        "finished_at": "2026-08-04T10:01:00+00:00",
        "source": {"commit": spec["source"]["commit"]},
        "image": {"digest": spec["image"]["digest"]},
        "dataset": {"name": dataset_name, "revision": dataset_revision},
        "simulator": {},
        "dependencies": {},
        "policy": {},
        "evaluation": {
            "stage": "base",
            "seed": spec["seed"],
            "suites": ["libero_spatial"],
            "trials_per_task": 50,
            "metrics": {"success": 0.97},
        },
        "command": command,
        "child_commands": [],
        "instance": {
            "type": identity["instanceType"],
            "id": identity["instanceId"],
            "identity_recorded_by": "worker run manifest",
        },
        "cost": {"projected_usd": 18.0, "actual_recorded_by": "worker run manifest"},
        "artifacts": [],
    }
    root, _source, markers = _write_expected_json(tmp_path, spec, relative, document)

    with pytest.raises(repro_worker.WorkerError, match="identity differs"):
        repro_worker.collect_run_metrics(spec, root, markers, launch_metadata=launch, instance_identity=identity)


def test_run_metrics_reject_unrecognized_or_mismatched_evaluation_manifest(tmp_path):
    launch, identity = _metrics_runtime_context()
    for suffix, document in (
        ("unknown", {"schema_version": 1, "evaluation": {"metrics": {"success": 0.97}}}),
        (
            "wrong-schema",
            {
                "schema_version": 999,
                "kind": "libero-evaluation",
                "evaluation": {"metrics": {"success": 0.97}},
            },
        ),
    ):
        raw = make_spec(tmp_path / suffix)
        relative = f"manifests/{suffix}.json"
        raw["expected_outputs"] = [{"name": "evaluation_manifest", "kind": "manifest", "path": relative}]
        spec = repro_worker.validate_worker_spec(raw)
        root, _source, markers = _write_expected_json(tmp_path / suffix, spec, relative, document)
        with pytest.raises(repro_worker.WorkerError, match=r"recognized schema|wrong schema"):
            repro_worker.collect_run_metrics(spec, root, markers, launch_metadata=launch, instance_identity=identity)


def _latency_metrics_output(tmp_path, *, schema_version=1, image_digest=None):
    raw = make_tensorrt_policy_spec(tmp_path)
    instance_id = "i-0123456789abcdef0"
    raw["placement"] = {"mode": "exact-existing-instance", "instance_id": instance_id}
    relative = "artifacts/latency.json"
    raw["expected_outputs"] = [{"name": "latency", "kind": "artifact", "path": relative}]
    raw["container"]["command"] = [
        "python",
        "scripts/benchmark_pi05_latency.py",
        "--backend",
        "tensorrt",
        "--stage",
        "tensorrt_fp8",
        "--artifact-dir",
        "/mnt/openpi/artifacts",
        "--track",
        "libero",
        "--dataset",
        "physical-intelligence/libero",
        "--dataset-revision",
        raw["artifacts"][0]["revision"],
        "--image-digest",
        raw["image"]["digest"],
        "--instance-id",
        instance_id,
        "--cost-reservation",
        "12345678-1234-4123-8123-123456789abc",
        "--output",
        "/output/artifacts/latency.json",
    ]
    spec = repro_worker.validate_worker_spec(raw)
    launch, identity = _metrics_runtime_context("g7e.4xlarge")
    document = {
        "schema_version": schema_version,
        "stage": "tensorrt_fp8",
        "track": "libero",
        "official_protocol": True,
        "batch_size": 1,
        "warmups": 500,
        "iterations": 10_000,
        "latency": {"total": {"cuda_event_ms": {"mean": 12.5}}},
        "runner": {"backend": "tensorrt", "num_denoise_steps": 1},
        "numerical_smoke": {"cosine_similarity": 0.999},
        "gpu_inventory": [],
        "dataset": {"name": "physical-intelligence/libero", "revision": spec["artifacts"][0]["revision"]},
        "benchmark_inputs": {},
        "source_artifacts": [],
        "runtime": {
            "instance_type": identity["instanceType"],
            "instance_id": identity["instanceId"],
            "image_digest": image_digest or spec["image"]["digest"],
            "instance_identity_source": "IMDSv2",
        },
    }
    root, source, markers = _write_expected_json(tmp_path, spec, relative, document)
    return spec, root, source, markers, launch, identity


def test_run_metrics_ingest_command_bound_latency_report(tmp_path):
    spec, root, source, markers, launch, identity = _latency_metrics_output(tmp_path)
    metrics, provenance = repro_worker.collect_run_metrics(
        spec, root, markers, launch_metadata=launch, instance_identity=identity
    )
    relative = "artifacts/latency.json"
    assert metrics == {
        relative: {
            "latency": {"total": {"cuda_event_ms": {"mean": 12.5}}},
            "numerical_smoke": {"cosine_similarity": 0.999},
        }
    }
    assert provenance[0]["sha256"] == repro_worker.sha256_file(source)


def test_run_metrics_ignores_unrelated_manifests_in_composite_artifact(tmp_path):
    raw = make_spec(tmp_path)
    relative = "artifacts/composite"
    raw["expected_outputs"] = [{"name": "composite", "kind": "artifact", "path": relative}]
    dataset = raw["artifacts"][0]
    raw["container"]["command"] = [
        "python",
        "scripts/export_pi05_onnx.py",
        "--stage",
        "current",
        "--track",
        "libero",
        "--dataset",
        "physical-intelligence/libero",
        "--dataset-revision",
        dataset["revision"],
    ]
    spec = raw
    launch, identity = _metrics_runtime_context()
    command = spec["container"]["command"][1:]

    def stage_manifest(stage, argv, value):
        return {
            "schema_version": 1,
            "created_at": "2026-08-04T10:00:00+00:00",
            "stage": stage,
            "track": "libero",
            "source": {"sha": spec["source"]["commit"], "dirty": False},
            "runtime": {
                "image_digest": spec["image"]["digest"],
                "instance_type": identity["instanceType"],
                "instance_id": identity["instanceId"],
            },
            "dataset": {"name": "physical-intelligence/libero", "revision": dataset["revision"]},
            "experiment": {"seed": None, "steps": None},
            "cost": {"reservation_id": launch["reservation_id"]},
            "command": {"argv": argv, "shell": shlex.join(argv)},
            "metrics": {"value": value},
            "artifacts": [],
            "details": {},
        }

    root = tmp_path / "output"
    target = root / relative
    for directory in (root / ".ready", root / ".receipts", root / ".spool", target):
        directory.mkdir(parents=True, exist_ok=True)
    (target / "current-manifest.json").write_text(json.dumps(stage_manifest("current", command, 1.0)))
    old_command = [*command]
    old_command[old_command.index("current")] = "old"
    (target / "old-manifest.json").write_text(json.dumps(stage_manifest("old", old_command, 99.0)))
    markers = repro_worker.OutputManager(spec, root, OutputRunner()).commit_expected_outputs()

    metrics, provenance = repro_worker.collect_run_metrics(
        spec, root, markers, launch_metadata=launch, instance_identity=identity
    )

    assert metrics == {f"{relative}/current-manifest.json": {"value": 1.0}}
    assert [item["path"] for item in provenance] == [f"{relative}/current-manifest.json"]


@pytest.mark.parametrize(
    ("schema_version", "image_digest", "message"),
    [(999, None, "wrong schema"), (1, "sha256:" + "9" * 64, "identity differs")],
)
def test_run_metrics_reject_latency_schema_or_identity(tmp_path, schema_version, image_digest, message):
    spec, root, _source, markers, launch, identity = _latency_metrics_output(
        tmp_path, schema_version=schema_version, image_digest=image_digest
    )
    with pytest.raises(repro_worker.WorkerError, match=message):
        repro_worker.collect_run_metrics(spec, root, markers, launch_metadata=launch, instance_identity=identity)


def test_run_metrics_reject_identity_nonfinite_and_post_commit_tampering(tmp_path):
    base = {
        "schema_version": 1,
        "config_name": "pi05_libero_l09_distill",
        "exp_name": "libero-shallow",
        "global_step": 2000,
        "metrics": {"loss": 0.25},
    }
    wrong_step = {**base, "global_step": 5000}
    spec, root, _checkpoint, markers = _training_metrics_output(tmp_path / "identity", wrong_step)
    with pytest.raises(repro_worker.WorkerError, match="config/experiment/step"):
        repro_worker.collect_run_metrics(spec, root, markers)

    nonfinite = {**base, "metrics": {"loss": float("nan")}}
    spec, root, _checkpoint, markers = _training_metrics_output(tmp_path / "nonfinite", nonfinite)
    with pytest.raises(repro_worker.WorkerError, match="non-finite"):
        repro_worker.collect_run_metrics(spec, root, markers)

    spec, root, checkpoint, markers = _training_metrics_output(tmp_path / "tampered", base)
    (checkpoint / repro_worker.TRAINING_METRICS_FILENAME).write_text(json.dumps({**base, "metrics": {"loss": 9}}))
    with pytest.raises(repro_worker.WorkerError, match="changed after expected-output commit"):
        repro_worker.collect_run_metrics(spec, root, markers)


def test_worker_cost_exposes_projection_but_never_infers_actual():
    metadata = make_launch_metadata("2026-08-04T12:00:00+00:00")
    assert repro_worker.worker_cost_record(metadata) == {
        "reservation_id": metadata["reservation_id"],
        "projected_usd": 18.0,
        "actual_usd": None,
        "actual_recorded_by": "versioned cost ledger after AWS billing reconciliation",
    }


def test_resume_checkpoint_is_hash_verified_restored_and_not_reuploaded(tmp_path):
    spec = repro_worker.validate_worker_spec(make_resume_spec(tmp_path))
    artifact = next(item for item in spec["artifacts"] if item["name"] == "shallow_checkpoint")
    root = tmp_path / "scratch"
    source = root / "inputs/checkpoints/pi05_libero_l09_distill/libero-shallow/2000"
    source.mkdir(parents=True)
    contract = {
        "schema_version": 1,
        "config_name": "pi05_libero_l09_distill",
        "exp_name": "libero-shallow",
        "seed": 7,
        "batch_size": 8,
        "gradient_accumulation_steps": 8,
        "lr_schedule": {"warmup_steps": 1000},
        "optimizer": {"b1": 0.9},
        "wandb_enabled": True,
        "model": {"class": "test.Model", "config": {"layers": 9}},
        "pytorch_training_precision": "bfloat16",
        "dataset": {
            "factory": "test.Factory",
            "factory_config_sha256": "f" * 64,
            "repo_id": "physical-intelligence/libero",
            "revision": "a" * 40,
            "codebase_version": "v2.0",
            "episode_prompt_path": None,
            "episode_prompt_sha256": None,
            "action_sequence_keys": ["actions"],
            "asset_id": "physical-intelligence/libero",
            "use_quantile_norm": True,
            "prompt_from_task": True,
            "normalization_sha256": "c" * 64,
            "recovery_provenance_sha256": None,
        },
        "teacher": {"model_sha256": "b" * 64},
        "data_split": {"schema_version": 1, "validation_episode_ids": [3]},
        "stochastic_schedule": "sha256-v2(model:seed,step,accumulation,rank;loader:seed,epoch)",
        "one_batch_overfit": False,
        "one_batch_overfit_min_relative_decline": 0.2,
    }
    state = {
        "schema_version": 2,
        "global_step": 2000,
        "config_name": contract["config_name"],
        "exp_name": contract["exp_name"],
        "resume_contract": contract,
        "resume_fingerprint_sha256": repro_worker.resume_identity_fingerprint(
            contract, {"kind": "shallow_teacher_transplant", "model_sha256": "b" * 64}
        ),
        "initialization_lineage": {"kind": "shallow_teacher_transplant", "model_sha256": "b" * 64},
        "state_files": ["metadata.pt", "model.safetensors", "optimizer.pt", "wandb_id.txt"],
    }
    payloads = {
        "model.safetensors": b"model",
        "optimizer.pt": b"optimizer",
        "metadata.pt": b"metadata",
        "wandb_id.txt": b"offline-run-id\n",
        "resume-state.json": (json.dumps(state, indent=2, sort_keys=True) + "\n").encode(),
    }
    files = []
    objects = []
    payload_prefix = repro_worker.parse_s3_uri(artifact["payload_s3_uri"], prefix=True).key
    for relative, content in payloads.items():
        (source / relative).write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        files.append({"path": relative, "bytes": len(content), "sha256": digest})
        objects.append(
            {
                "path": relative,
                "s3_key": f"{payload_prefix}/{relative}",
                "version_id": f"version-{relative}",
            }
        )
    manifest = {
        "schema_version": 1,
        "source": {
            "provider": "pi05-worker-output",
            "revision": artifact["revision"],
            "objects": objects,
        },
        "artifact": {
            "name": artifact["name"],
            "kind": "checkpoint",
            "path": "checkpoints/pi05_libero_l09_distill/libero-shallow/2000",
            "publish_destination": artifact["destination"],
            "payload_s3_uri": artifact["payload_s3_uri"],
        },
        "totals": {"files": len(files), "bytes": sum(item["bytes"] for item in files)},
        "files": files,
    }
    manifest_path = root / "inputs/.manifests/shallow_checkpoint.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest))
    for relative in ("output/checkpoints", "output/.ready", "output/.receipts", "output/.spool"):
        (root / relative).mkdir(parents=True)

    evidence = repro_worker.restore_resume_checkpoint(spec, root)
    target = root / "output/checkpoints/pi05_libero_l09_distill/libero-shallow/2000"
    assert evidence["global_step"] == 2000
    assert evidence["resume_state_sha256"] == next(
        item["sha256"] for item in files if item["path"] == "resume-state.json"
    )
    assert evidence["resume_fingerprint_sha256"] == state["resume_fingerprint_sha256"]
    assert (target / "optimizer.pt").read_bytes() == b"optimizer"
    repro_worker.validate_artifact_manifest(manifest, artifact, target)

    manager = repro_worker.OutputManager(spec, root / "output", OutputRunner())
    manager.discover_atomic_checkpoints()
    assert list((root / "output/.ready").glob("checkpoint-*.ready.json")) == []
    with pytest.raises(repro_worker.WorkerError, match="already exists"):
        repro_worker.restore_resume_checkpoint(spec, root)


def test_rendered_bootstrap_is_versioned_hashed_and_dry_by_default():
    args = argparse.Namespace(
        bootstrap_s3_uri="s3://pi05-repro-752160877725-us-east-2/bootstrap/worker-bootstrap.sh",
        bootstrap_version_id="bootstrap-v1",
        bootstrap_sha256="a" * 64,
        spec_s3_uri="s3://pi05-repro-752160877725-us-east-2/specs/run.json",
        spec_version_id="spec-v1",
        spec_sha256="b" * 64,
        execute=False,
    )
    command = repro_worker.render_bootstrap_command(args)
    assert "--version-id bootstrap-v1" in command
    assert "sha256sum --check --status" in command
    assert "WORKER_SPEC_VERSION_ID=spec-v1" in command
    assert "WORKER_EXECUTE=0" in command
    assert "latest" not in command


def test_controller_v2_bootstrap_independently_materializes_model_and_controller_sources():
    bootstrap = pathlib.Path(__file__).parents[1] / "repro/worker-bootstrap-controller-v2.sh"
    subprocess.run(["bash", "-n", str(bootstrap)], check=True)
    text = bootstrap.read_text()
    assert "read_source_pin source" in text
    assert "read_source_pin controller_source" in text
    assert "source_checkout=/opt/pi05/model-source" in text
    assert "controller_checkout=/opt/pi05/controller-source" in text
    assert 'exec python3 "${controller_checkout}/scripts/repro_worker.py"' in text
    assert '"checkout_path": source_checkout' in text
    assert '"controller_checkout_path": controller_checkout' in text
    assert 'git -C "${checkout}" fsck --full --no-dangling' in text
    assert '"source_fsck_full": True' in text
    assert '"controller_source_fsck_full": True' in text
    assert 'schema_version": 2' in text
