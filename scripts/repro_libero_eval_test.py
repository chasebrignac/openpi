import argparse
import hashlib
import json
import pathlib
import types

import numpy as np
import pytest

from examples.libero import main as libero_eval
from scripts import repro_libero_eval

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class _FakeTaskSuite:
    n_tasks = 1

    def __init__(self, initial_state_count: int = 2):
        self.initial_states = list(range(initial_state_count))

    def get_task(self, task_id: int):
        assert task_id == 0
        return types.SimpleNamespace(language="put the red block in the bowl")

    def get_task_init_states(self, task_id: int):
        assert task_id == 0
        return self.initial_states


def _observation() -> dict[str, np.ndarray]:
    return {
        "agentview_image": np.zeros((4, 4, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((4, 4, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }


class _FakeEnv:
    def __init__(self):
        self.init_indices = []
        self.closed = False

    def reset(self):
        return None

    def set_init_state(self, init_index: int):
        self.init_indices.append(init_index)
        return _observation()

    def step(self, action):
        assert len(action) == 7
        return _observation(), 0.0, True, {}

    def close(self):
        self.closed = True


class _FakeClient:
    def __init__(self):
        self.calls = 0
        self.closed = False

    def infer(self, element):
        assert element["prompt"] == "put the red block in the bowl"
        self.calls += 1
        return {"actions": np.zeros((1, 7), dtype=np.float32)}

    def close(self):
        self.closed = True


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_episode_records_pair_across_stages():
    common = {
        "suite": "libero_goal",
        "task": "open the drawer",
        "task_id": 3,
        "init_index": 17,
        "seed": 7,
        "success": True,
        "steps": 41,
    }
    baseline = libero_eval.make_episode_record(stage="base", **common)
    final = libero_eval.make_episode_record(stage="final", **common)

    assert baseline["pair_id"] == final["pair_id"]
    assert baseline == {
        "pair_id": "libero:libero_goal:task-003:init-017:seed-7",
        "stage": "base",
        "benchmark": "libero",
        "suite": "libero_goal",
        "task": "open the drawer",
        "task_id": 3,
        "success": True,
        "seed": 7,
        "init_index": 17,
        "steps": 41,
    }


def test_trial_selection_is_exact():
    assert list(libero_eval.select_init_indices(requested=2, available=2, suite="libero_10", task_id=0)) == [0, 1]
    with pytest.raises(ValueError, match="must be positive"):
        libero_eval.select_init_indices(requested=0, available=50, suite="libero_10", task_id=0)
    with pytest.raises(ValueError, match="has 2 fixed initial states"):
        libero_eval.select_init_indices(requested=3, available=2, suite="libero_10", task_id=0)


def test_result_writer_refuses_to_mix_runs_without_explicit_overwrite(tmp_path):
    output = tmp_path / "episodes.jsonl"
    with libero_eval.EpisodeResultWriter(output, overwrite=False) as writer:
        writer.write({"pair_id": "one"})
        assert writer.count == 1

    with pytest.raises(FileExistsError):
        libero_eval.EpisodeResultWriter(output, overwrite=False)

    with libero_eval.EpisodeResultWriter(output, overwrite=True) as writer:
        writer.write({"pair_id": "two"})
    assert _read_jsonl(output) == [{"pair_id": "two"}]


def test_eval_writes_exact_deterministic_pairs_without_simulator(tmp_path, monkeypatch):
    suite = _FakeTaskSuite()
    environments = []
    clients = []

    monkeypatch.setattr(libero_eval, "_get_task_suite", lambda _: suite)
    monkeypatch.setattr(libero_eval.tqdm, "tqdm", lambda iterable: iterable)

    def make_env(task, resolution, seed):
        assert (resolution, seed) == (libero_eval.LIBERO_ENV_RESOLUTION, 19)
        env = _FakeEnv()
        environments.append(env)
        return env, task.language

    def make_client(host, port):
        assert (host, port) == ("127.0.0.1", 8123)
        client = _FakeClient()
        clients.append(client)
        return client

    monkeypatch.setattr(libero_eval, "_get_libero_env", make_env)
    monkeypatch.setattr(libero_eval, "_make_client", make_client)

    outputs = {}
    for stage in ("base", "final"):
        output = tmp_path / f"{stage}.jsonl"
        libero_eval.eval_libero(
            libero_eval.Args(
                host="127.0.0.1",
                port=8123,
                resize_size=4,
                replan_steps=1,
                task_suite_name="libero_spatial",
                num_steps_wait=0,
                num_trials_per_task=2,
                save_videos=False,
                results_out_path=str(output),
                stage=stage,
                seed=19,
            )
        )
        outputs[stage] = _read_jsonl(output)

    assert [record["pair_id"] for record in outputs["base"]] == [record["pair_id"] for record in outputs["final"]]
    assert [record["init_index"] for record in outputs["base"]] == [0, 1]
    assert all(record["success"] for records in outputs.values() for record in records)
    assert all(env.init_indices == [0, 1] and env.closed for env in environments)
    assert all(client.calls == 2 and client.closed for client in clients)


def test_eval_rejects_short_init_state_set_before_creating_output(tmp_path, monkeypatch):
    output = tmp_path / "episodes.jsonl"
    monkeypatch.setattr(libero_eval, "_get_task_suite", lambda _: _FakeTaskSuite(initial_state_count=1))

    with pytest.raises(ValueError, match="but 2 trials were requested"):
        libero_eval.eval_libero(
            libero_eval.Args(
                task_suite_name="libero_spatial",
                num_trials_per_task=2,
                save_videos=False,
                results_out_path=str(output),
            )
        )

    assert not output.exists()


def test_policy_transport_failure_aborts_instead_of_becoming_a_failed_episode(tmp_path, monkeypatch):
    output = tmp_path / "episodes.jsonl"
    environment = _FakeEnv()
    monkeypatch.setattr(libero_eval, "_get_task_suite", lambda _: _FakeTaskSuite(initial_state_count=1))
    monkeypatch.setattr(libero_eval.tqdm, "tqdm", lambda iterable: iterable)
    monkeypatch.setattr(
        libero_eval,
        "_get_libero_env",
        lambda *_args: (environment, "put the red block in the bowl"),
    )

    class BrokenClient:
        def infer(self, _element):
            raise ConnectionError("policy server unavailable")

    monkeypatch.setattr(libero_eval, "_make_client", lambda *_args: BrokenClient())
    with pytest.raises(ConnectionError, match="policy server unavailable"):
        libero_eval.eval_libero(
            libero_eval.Args(
                task_suite_name="libero_spatial",
                num_steps_wait=0,
                num_trials_per_task=1,
                save_videos=False,
                results_out_path=str(output),
            )
        )
    assert environment.closed
    assert output.read_text() == ""


def _compiled_runtime_args(tmp_path: pathlib.Path) -> argparse.Namespace:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    artifacts = tmp_path / "compiled"
    artifacts.mkdir()
    manifest = {
        "schema_version": 1,
        "stage": "tensorrt-build-fp8",
        "track": "libero",
        "source": {"sha": "a" * 40, "dirty": False},
        "runtime": {
            "image_digest": f"sha256:{'b' * 64}",
            "instance_type": repro_libero_eval.TENSORRT_INSTANCE_TYPE,
            "instance_id": "i-0123456789abcdef0",
        },
        "dataset": {
            "name": repro_libero_eval.LIBERO_DATASET,
            "revision": repro_libero_eval.LIBERO_DATASET_REVISION,
        },
    }
    manifest_path = artifacts / "tensorrt-manifest.fp8.json"
    manifest_path.write_text(json.dumps(manifest))
    return argparse.Namespace(
        backend="tensorrt",
        source_commit="a" * 40,
        image_digest=f"sha256:{'b' * 64}",
        run_id="libero-final-fp8",
        stage="final",
        seed=7,
        policy_config=repro_libero_eval.TENSORRT_POLICY_CONFIG,
        model_revision="c" * 64,
        checkpoint=checkpoint,
        port=8000,
        instance_type=repro_libero_eval.TENSORRT_INSTANCE_TYPE,
        precision="fp8",
        dataset=repro_libero_eval.LIBERO_DATASET,
        dataset_revision=repro_libero_eval.LIBERO_DATASET_REVISION,
        build_instance_id="i-0123456789abcdef0",
        build_run_id="libero-fp8-build-01",
        compiled_artifact_dir=artifacts,
        engine_build_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        compiled_artifact_revision="d" * 64,
        compiled_manifest_s3_uri=(
            f"s3://{repro_libero_eval.BUCKET}/runs/build/manifests/worker-input-compiled.sha256.json"
        ),
        compiled_manifest_version_id="version-1",
        compiled_manifest_sha256="e" * 64,
        compiled_payload_s3_uri=f"s3://{repro_libero_eval.BUCKET}/runs/build/artifacts/tensorrt/libero/fp8/",
    )


def test_tensorrt_run_binds_artifact_source_image_dataset_and_exact_build_instance(tmp_path, monkeypatch):
    args = _compiled_runtime_args(tmp_path)
    monkeypatch.delenv("PI05_SOURCE_SHA", raising=False)
    monkeypatch.delenv("PI05_IMAGE_DIGEST", raising=False)
    monkeypatch.delenv("PI05_RUN_ID", raising=False)
    monkeypatch.delenv("PI05_SEED", raising=False)
    monkeypatch.setenv("PI05_INSTANCE_ID", args.build_instance_id)

    assert repro_libero_eval.validate_run_identity(args) == (
        args.source_commit,
        args.image_digest,
        args.run_id,
    )
    identity = repro_libero_eval.validate_tensorrt_run_identity(
        args,
        source_commit=args.source_commit,
        image_digest=args.image_digest,
    )
    assert identity["build_runtime"]["instance_id"] == args.build_instance_id
    assert identity["precision"] == "fp8"
    assert identity["revision"] == args.compiled_artifact_revision

    args.instance_type = "g6e.4xlarge"
    with pytest.raises(ValueError, match=r"requires g7e\.4xlarge"):
        repro_libero_eval.validate_run_identity(args)


def test_tensorrt_run_rejects_fresh_instance_artifact_identity(tmp_path, monkeypatch):
    args = _compiled_runtime_args(tmp_path)
    monkeypatch.delenv("PI05_SOURCE_SHA", raising=False)
    monkeypatch.delenv("PI05_IMAGE_DIGEST", raising=False)
    monkeypatch.delenv("PI05_RUN_ID", raising=False)
    monkeypatch.delenv("PI05_SEED", raising=False)
    monkeypatch.setenv("PI05_INSTANCE_ID", "i-fedcba98765432100")
    args.build_instance_id = "i-fedcba98765432100"
    with pytest.raises(ValueError, match="engine-build manifest identity differs"):
        repro_libero_eval.validate_run_identity(args)


def test_tensorrt_run_rejects_when_worker_instance_is_not_build_instance(tmp_path, monkeypatch):
    args = _compiled_runtime_args(tmp_path)
    monkeypatch.setenv("PI05_INSTANCE_ID", "i-fedcba98765432100")
    with pytest.raises(ValueError, match="current EC2 instance differs"):
        repro_libero_eval.validate_run_identity(args)


def test_tensorrt_server_preserves_websocket_protocol_and_passes_exact_runtime(tmp_path):
    args = _compiled_runtime_args(tmp_path)
    command = repro_libero_eval.server_command(args)
    assert command[:2] == [str(repro_libero_eval.TENSORRT_POLICY_PYTHON), "scripts/serve_tensorrt_policy.py"]
    artifact_position = command.index("--artifact-dir")
    assert command[artifact_position : artifact_position + 3] == [
        "--artifact-dir",
        str(args.compiled_artifact_dir),
        "--checkpoint-dir",
    ]
    assert command[command.index("--checkpoint-dir") + 1] == str(args.checkpoint)
    assert command[command.index("--track") + 1] == "libero"
    assert command[command.index("--instance-id") + 1] == args.build_instance_id
    assert command[command.index("--image-digest") + 1] == args.image_digest


def _worker_artifact(path: pathlib.Path, *, kind: str, destination: str, name: str) -> pathlib.Path:
    descriptor = {
        "name": name,
        "kind": kind,
        "revision": "1" * 64,
        "manifest": {
            "s3_uri": f"s3://{repro_libero_eval.BUCKET}/runs/build/manifests/{name}.json",
            "version_id": "version-1",
            "sha256": "2" * 64,
        },
        "payload_s3_uri": f"s3://{repro_libero_eval.BUCKET}/runs/build/artifacts/{name}/",
        "destination": destination,
    }
    path.write_text(json.dumps({"worker_artifact": descriptor}))
    return path


def _compiled_render_args(tmp_path: pathlib.Path) -> argparse.Namespace:
    image_digest = f"sha256:{'3' * 64}"
    contract = {
        "schema_version": 1,
        "kind": "pi05-tensorrt-build-instance",
        "execution_constraint": "evaluate-before-exact-build-instance-stop",
        "build_run_id": "libero-fp8-build-01",
        "source_commit": "4" * 40,
        "image_digest": image_digest,
        "instance_type": repro_libero_eval.TENSORRT_INSTANCE_TYPE,
        "instance_id": "i-0123456789abcdef0",
        "track": "libero",
        "dataset": {
            "name": repro_libero_eval.LIBERO_DATASET,
            "revision": repro_libero_eval.LIBERO_DATASET_REVISION,
        },
        "precision": "fp8",
        "engine_build_manifest_sha256": "5" * 64,
    }
    contract_path = tmp_path / "build-instance.json"
    contract_path.write_text(json.dumps(contract))
    return argparse.Namespace(
        run_id="libero-final-fp8",
        source_s3_uri=f"s3://{repro_libero_eval.BUCKET}/source/openpi.bundle",
        source_version_id="source-version",
        source_sha256="6" * 64,
        source_commit="4" * 40,
        image_uri=(
            f"{repro_libero_eval.ACCOUNT}.dkr.ecr.{repro_libero_eval.REGION}.amazonaws.com/pi05-repro@{image_digest}"
        ),
        parent_policy_image=(
            f"{repro_libero_eval.ACCOUNT}.dkr.ecr.{repro_libero_eval.REGION}.amazonaws.com/pi05-repro@sha256:{'7' * 64}"
        ),
        parent_tensorrt_compiler_image=(
            f"{repro_libero_eval.ACCOUNT}.dkr.ecr.{repro_libero_eval.REGION}.amazonaws.com/pi05-repro@sha256:{'8' * 64}"
        ),
        parent_tensorrt_compiler_source_revision="4" * 40,
        backend="tensorrt",
        checkpoint_artifact=_worker_artifact(
            tmp_path / "checkpoint.json",
            kind="checkpoint",
            destination="pi05_libero_l09_snapflow/final/30000",
            name="libero_checkpoint",
        ),
        compiled_artifact=_worker_artifact(
            tmp_path / "compiled.json",
            kind="asset",
            destination="tensorrt/libero/fp8",
            name="libero_fp8_engines",
        ),
        build_instance_contract=contract_path,
        policy_config=repro_libero_eval.TENSORRT_POLICY_CONFIG,
        precision="fp8",
        dataset=repro_libero_eval.LIBERO_DATASET,
        dataset_revision=repro_libero_eval.LIBERO_DATASET_REVISION,
        stage="final",
        trials_per_task=50,
        seed=7,
        instance_type=repro_libero_eval.TENSORRT_INSTANCE_TYPE,
        projected_cost_usd=25.0,
    )


def test_rendered_tensorrt_spec_stages_complete_tree_and_binds_build_contract(tmp_path):
    args = _compiled_render_args(tmp_path)
    spec = repro_libero_eval.render_worker_spec(args)
    assert [item["kind"] for item in spec["artifacts"]] == ["checkpoint", "asset"]
    command = spec["container"]["command"]
    assert command[0] == str(repro_libero_eval.TENSORRT_POLICY_PYTHON)
    assert command[command.index("--backend") + 1] == "tensorrt"
    assert command[command.index("--compiled-artifact-dir") + 1] == "/mnt/openpi/assets/tensorrt/libero/fp8"
    assert command[command.index("--build-instance-id") + 1] == "i-0123456789abcdef0"
    assert command[command.index("--build-run-id") + 1] == "libero-fp8-build-01"
    assert command[command.index("--engine-build-manifest-sha256") + 1] == "5" * 64
    assert spec["image"]["digest"] == f"sha256:{'3' * 64}"
    assert spec["image"]["policy_backend"] == "tensorrt"
    assert spec["image"]["parent_tensorrt_compiler_image"].endswith(f"sha256:{'8' * 64}")
    assert spec["image"]["parent_tensorrt_compiler_source_revision"] == "4" * 40
    assert spec["image"]["toolchain"] == repro_libero_eval.TENSORRT_TOOLCHAIN
    assert spec["placement"] == {
        "mode": "exact-existing-instance",
        "instance_id": "i-0123456789abcdef0",
    }
    assert "PI05_INSTANCE_ID" not in spec["container"]["environment"]


def test_rendered_tensorrt_spec_requires_explicit_same_instance_contract(tmp_path):
    args = _compiled_render_args(tmp_path)
    args.build_instance_contract = None
    with pytest.raises(ValueError, match="same-running-build-instance contract"):
        repro_libero_eval.render_worker_spec(args)


def test_libero_image_and_runtime_contract_declare_both_backends():
    dockerfile = (REPO_ROOT / "repro/Dockerfile.libero").read_text()
    assert "ARG POLICY_BACKEND=eager" in dockerfile
    assert "ai.openpi.policy-backend=${POLICY_BACKEND}" in dockerfile
    assert "test -x /opt/modelopt/bin/python" in dockerfile
    assert "/opt/modelopt/bin/python -c 'import tensorrt as trt" in dockerfile

    contract = json.loads((REPO_ROOT / "repro/libero-evaluator-contract.json").read_text())
    assert contract["policy_backends"]["eager"]["instance_type"] == "g6e.4xlarge"
    assert contract["policy_backends"]["tensorrt"] == {
        "instance_type": repro_libero_eval.TENSORRT_INSTANCE_TYPE,
        "policy_python": str(repro_libero_eval.TENSORRT_POLICY_PYTHON),
        "server": "scripts/serve_tensorrt_policy.py",
        "track": "libero",
        "dataset": repro_libero_eval.LIBERO_DATASET,
        "dataset_revision": repro_libero_eval.LIBERO_DATASET_REVISION,
        "precisions": list(repro_libero_eval.TENSORRT_PRECISIONS),
        "placement": "exact-engine-build-instance",
    }
    assert contract["compiled_orchestration"] == {
        "manual_replays": "direct-final-evaluator-image-on-build-instance",
        "rendered_exact_instance_spec": "future-non-launchable",
    }


def test_export_and_eval_runbooks_use_final_evaluator_digest_on_one_g7e():
    export_runbook = (REPO_ROOT / "repro/EXPORT_RUNBOOK.md").read_text()
    eval_runbook = (REPO_ROOT / "repro/LIBERO_EVAL_RUNBOOK.md").read_text()

    assert "--build-arg POLICY_BACKEND=tensorrt" in export_runbook
    assert 'export LIBERO_RUNTIME_IMAGE="$ECR_REPOSITORY@$LIBERO_EVALUATOR_DIGEST"' in export_runbook
    assert 'export IMAGE_DIGEST="${LIBERO_RUNTIME_IMAGE##*@}"' in export_runbook
    assert "sections 0 through 7" in export_runbook
    assert "normal EC2 launcher creates fresh capacity" in export_runbook
    assert "/output/artifacts/tensorrt/libero/bf16" in export_runbook
    assert "/output/artifacts/tensorrt/libero/fp8" in export_runbook
    assert "--retain-after-command" in export_runbook
    assert "not a completed ephemeral worker" in export_runbook
    assert "export, build, serve, and evaluate" in export_runbook
    assert "non-launchable" in export_runbook
    assert "not its parent digest" in eval_runbook
    assert "same still-running `g7e.4xlarge`" in eval_runbook
    assert "placement.mode=exact-existing-instance" in eval_runbook
    assert "PI05_INSTANCE_ID" in eval_runbook
    assert "PI05_RETAINED_EVAL_PROJECTED_COST_USD" in eval_runbook
    assert "non-overlapping part" in eval_runbook
    assert "--projected-cost-usd 0" not in eval_runbook
    assert "validation-only, non-launchable" in eval_runbook
    assert "Do not pass it to `scripts/repro_worker.py --execute`" in eval_runbook
    assert "Let the worker finish publication" not in eval_runbook
