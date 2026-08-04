from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess

import pytest

from examples.libero import main as libero_main
from scripts import repro_libero_eval
from scripts import repro_worker

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _runtime_contract(tmp_path: pathlib.Path) -> pathlib.Path:
    lock = tmp_path / "requirements.txt"
    lock.write_text("numpy==1.22.4\n")
    contract = {
        "schema_version": 1,
        "kind": "pi05-libero-evaluator",
        "simulator": {
            "repository": repro_libero_eval.LIBERO_REPOSITORY,
            "revision": repro_libero_eval.LIBERO_REVISION,
        },
        "python": {"implementation": "CPython", "version": "3.8.20", "environment": "/opt/libero-venv"},
        "requirements": {
            "path": "repro/libero-evaluator-requirements.txt",
            "installed_path": str(lock),
            "sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        },
        "protocol": {"transport": "loopback-websocket"},
        "gpu_environment": {
            "MUJOCO_GL": "egl",
            "MUJOCO_EGL_DEVICE_ID": "0",
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
            "PYOPENGL_PLATFORM": "egl",
        },
        "suites": list(repro_libero_eval.SUITES),
        "tasks_per_suite": 10,
        "minimum_fixed_init_states_per_task": 50,
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    return path


def _checkpoint_artifact(tmp_path: pathlib.Path) -> pathlib.Path:
    artifact = {
        "name": "libero_checkpoint",
        "kind": "checkpoint",
        "revision": "a" * 64,
        "manifest": {
            "s3_uri": f"s3://{repro_libero_eval.BUCKET}/checkpoints/libero/manifest.sha256.json",
            "version_id": "version-1",
            "sha256": "b" * 64,
        },
        "payload_s3_uri": f"s3://{repro_libero_eval.BUCKET}/checkpoints/libero/checkpoint/",
        "destination": "pi05_libero",
    }
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps({"worker_artifact": artifact}))
    return path


def _checkpoint_provenance_artifact(tmp_path: pathlib.Path) -> pathlib.Path:
    artifact = {
        "name": "libero_jax_checkpoint",
        "kind": "checkpoint",
        "revision": "1" * 64,
        "manifest": {
            "s3_uri": f"s3://{repro_libero_eval.BUCKET}/checkpoints/libero-jax/manifest.sha256.json",
            "version_id": "version-jax-1",
            "sha256": "2" * 64,
        },
        "payload_s3_uri": f"s3://{repro_libero_eval.BUCKET}/checkpoints/libero-jax/checkpoint/",
        "destination": "pi05_libero",
    }
    path = tmp_path / "provenance-artifact.json"
    path.write_text(json.dumps({"worker_artifact": artifact}))
    return path


def _render_args(tmp_path: pathlib.Path) -> argparse.Namespace:
    return argparse.Namespace(
        run_id="libero-base-smoke-01",
        controller_source_s3_uri=f"s3://{repro_libero_eval.BUCKET}/source/controller-complete.bundle",
        controller_source_version_id="controller-version-1",
        controller_source_sha256="9" * 64,
        controller_source_commit="8" * 40,
        source_s3_uri=f"s3://{repro_libero_eval.BUCKET}/source/openpi.bundle",
        source_version_id="version-1",
        source_sha256="c" * 64,
        source_commit="d" * 40,
        image_uri=(
            f"{repro_libero_eval.ACCOUNT}.dkr.ecr.{repro_libero_eval.REGION}.amazonaws.com/pi05-repro@sha256:{'e' * 64}"
        ),
        parent_policy_image=(
            f"{repro_libero_eval.ACCOUNT}.dkr.ecr.{repro_libero_eval.REGION}.amazonaws.com/pi05-repro@sha256:{'f' * 64}"
        ),
        checkpoint_artifact=_checkpoint_artifact(tmp_path),
        policy_config="pi05_libero",
        stage="base",
        trials_per_task=10,
        seed=7,
        instance_type="g6e.4xlarge",
        projected_cost_usd=12.5,
    )


def test_source_gitlink_and_docker_contract_are_exact():
    gitlink = subprocess.run(
        ["git", "ls-tree", "HEAD", "third_party/libero"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert gitlink == f"160000 commit {repro_libero_eval.LIBERO_REVISION}\tthird_party/libero"

    dockerfile = (REPO_ROOT / "repro/Dockerfile.libero").read_text()
    assert "ARG POLICY_BASE_IMAGE" in dockerfile
    assert "FROM ${POLICY_BASE_IMAGE}" in dockerfile
    assert f"ARG LIBERO_SHA={repro_libero_eval.LIBERO_REVISION}" in dockerfile
    assert 'ai.openpi.image-purpose="libero-evaluator"' in dockerfile
    assert "ai.openpi.libero-simulator-revision=${LIBERO_SHA}" in dockerfile
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics" in dockerfile
    assert "MUJOCO_EGL_DEVICE_ID=0" in dockerfile
    assert 'git -C "${LIBERO_ROOT}" fetch --depth=1 origin "${LIBERO_SHA}"' in dockerfile
    assert "COPY ./third_party/libero" not in dockerfile
    assert "COPY third_party/libero" not in dockerfile


def test_checked_in_dependency_lock_matches_contract_and_dockerfile():
    contract = json.loads((REPO_ROOT / "repro/libero-evaluator-contract.json").read_text())
    lock = REPO_ROOT / contract["requirements"]["path"]
    lock_hash = hashlib.sha256(lock.read_bytes()).hexdigest()
    assert contract["requirements"]["sha256"] == lock_hash
    assert f"ARG LIBERO_REQUIREMENTS_SHA256={lock_hash}" in (REPO_ROOT / "repro/Dockerfile.libero").read_text()


def test_runtime_contract_validation_fails_after_lock_tampering(tmp_path):
    contract_path = _runtime_contract(tmp_path)
    contract = repro_libero_eval.validate_runtime_contract(contract_path)
    assert contract["simulator"]["revision"] == repro_libero_eval.LIBERO_REVISION
    libero_main.validate_runtime_contract(
        contract_path,
        expected_libero_revision=repro_libero_eval.LIBERO_REVISION,
    )

    pathlib.Path(contract["requirements"]["installed_path"]).write_text("numpy==9.9.9\n")
    with pytest.raises(ValueError, match="lock hash"):
        repro_libero_eval.validate_runtime_contract(contract_path)
    with pytest.raises(ValueError, match="lock hash"):
        libero_main.validate_runtime_contract(
            contract_path,
            expected_libero_revision=repro_libero_eval.LIBERO_REVISION,
        )


def test_rendered_worker_spec_uses_one_network_none_policy_container(tmp_path):
    spec = repro_libero_eval.render_worker_spec(_render_args(tmp_path))
    assert spec["controller_source"] == {
        "s3_uri": f"s3://{repro_libero_eval.BUCKET}/source/controller-complete.bundle",
        "version_id": "controller-version-1",
        "sha256": "9" * 64,
        "commit": "8" * 40,
    }
    assert spec["image"] == {
        "uri": (
            f"{repro_libero_eval.ACCOUNT}.dkr.ecr.{repro_libero_eval.REGION}.amazonaws.com/pi05-repro@sha256:{'e' * 64}"
        ),
        "digest": f"sha256:{'e' * 64}",
        "purpose": "libero-evaluator",
        "policy_backend": "eager",
        "lerobot_runtime": "v2",
        "lerobot_revision": repro_libero_eval.LEROBOT_V2_REVISION,
        "libero_simulator_revision": repro_libero_eval.LIBERO_REVISION,
        "libero_requirements_sha256": repro_libero_eval.LIBERO_REQUIREMENTS_SHA256,
        "parent_policy_image": (
            f"{repro_libero_eval.ACCOUNT}.dkr.ecr.{repro_libero_eval.REGION}.amazonaws.com/pi05-repro@sha256:{'f' * 64}"
        ),
    }
    assert spec["container"]["environment"] == {
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": "0",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
        "PYOPENGL_PLATFORM": "egl",
    }
    assert spec["container"]["command"][:3] == ["python", "scripts/repro_libero_eval.py", "run"]
    assert len(spec["expected_outputs"]) == 6
    assert {item["name"] for item in spec["expected_outputs"]} == {
        "suite_spatial",
        "suite_object",
        "suite_goal",
        "suite_10",
        "episodes",
        "evaluation_manifest",
    }
    docker_command = repro_worker.build_docker_command(
        spec,
        pathlib.Path("/verified/source"),
        pathlib.Path("/verified/scratch"),
    )
    assert docker_command[docker_command.index("--network") + 1] == "none"
    assert docker_command.count(spec["image"]["uri"]) == 1


def test_rendered_worker_spec_can_stage_distinct_checkpoint_provenance(tmp_path):
    args = _render_args(tmp_path)
    checkpoint_payload = json.loads(args.checkpoint_artifact.read_text())
    checkpoint_payload["worker_artifact"]["destination"] = "pi05_libero_pytorch"
    args.checkpoint_artifact.write_text(json.dumps(checkpoint_payload))
    args.checkpoint_provenance_artifact = _checkpoint_provenance_artifact(tmp_path)
    spec = repro_libero_eval.render_worker_spec(args)
    assert [artifact["destination"] for artifact in spec["artifacts"]] == [
        "pi05_libero",
        "pi05_libero_pytorch",
    ]
    assert spec["container"]["command"][spec["container"]["command"].index("--checkpoint") + 1] == (
        "/mnt/openpi/checkpoints/pi05_libero_pytorch"
    )

    overlapping = _checkpoint_provenance_artifact(tmp_path)
    payload = json.loads(overlapping.read_text())
    payload["worker_artifact"]["destination"] = "pi05_libero_pytorch"
    overlapping.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checkpoint and checkpoint-provenance destinations overlap"):
        repro_libero_eval.render_worker_spec(args)


def test_run_identity_cannot_override_worker_environment(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    args = argparse.Namespace(
        source_commit="1" * 40,
        image_digest=f"sha256:{'2' * 64}",
        run_id="libero-smoke",
        stage="base",
        seed=7,
        policy_config="pi05_libero",
        model_revision="3" * 64,
        checkpoint=checkpoint,
        port=8000,
    )
    monkeypatch.setenv("PI05_SOURCE_SHA", "4" * 40)
    monkeypatch.setenv("PI05_IMAGE_DIGEST", args.image_digest)
    monkeypatch.setenv("PI05_RUN_ID", args.run_id)
    monkeypatch.setenv("PI05_SEED", "7")
    with pytest.raises(ValueError, match="PI05_SOURCE_SHA"):
        repro_libero_eval.validate_run_identity(args)


def test_result_validation_requires_exact_suite_identity_and_cardinality(tmp_path):
    args = argparse.Namespace(stage="final", seed=7, trials_per_task=1)
    paths = []
    for suite in repro_libero_eval.SUITES:
        path = tmp_path / f"{suite}.jsonl"
        records = [
            libero_main.make_episode_record(
                suite=suite,
                task=f"task {task_id}",
                task_id=task_id,
                init_index=0,
                seed=7,
                stage="final",
                success=task_id % 2 == 0,
                steps=10,
                libero_revision=repro_libero_eval.LIBERO_REVISION,
            )
            for task_id in range(10)
        ]
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        paths.append(path)

    records, metrics = repro_libero_eval.load_and_validate_results(paths, args=args)
    assert len(records) == 40
    assert metrics == {
        "episodes": 40,
        "successes": 20,
        "success_rate": 0.5,
        "environment_steps": 400,
        "infrastructure_errors": 0,
        "suites": {suite: {"episodes": 10, "successes": 5, "success_rate": 0.5} for suite in repro_libero_eval.SUITES},
    }

    first = json.loads(paths[0].read_text().splitlines()[0])
    first["libero_revision"] = "0" * 40
    lines = paths[0].read_text().splitlines()
    lines[0] = json.dumps(first)
    paths[0].write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        repro_libero_eval.load_and_validate_results(paths, args=args)

    first["libero_revision"] = repro_libero_eval.LIBERO_REVISION
    first["error"] = "ConnectionError: policy server unavailable"
    lines[0] = json.dumps(first)
    paths[0].write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="infrastructure-error"):
        repro_libero_eval.load_and_validate_results(paths, args=args)
