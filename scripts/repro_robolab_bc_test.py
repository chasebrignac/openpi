import copy
import json
import pathlib

import h5py
import numpy as np
import pytest

from openpi.training import robolab_expert_dataset
from scripts import repro_robolab_bc
from scripts import repro_robolab_report


def _write_trigger_report(
    path: pathlib.Path, *, student_hash: str, teacher_hash: str, success_gap: float = 0.06
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark": "robolab",
                "checkpoint_step": 30_000,
                "runtime": {
                    "robolab_git_sha": robolab_expert_dataset.ROBOLAB_GIT_SHA,
                    "openpi_client_git_sha": robolab_expert_dataset.ROBOLAB_OPENPI_CLIENT_GIT_SHA,
                },
                "provenance": {
                    "student_config": {"name": "pi05_droid_l09_distill"},
                    "student_checkpoint": {"model_sha256": student_hash},
                    "teacher_config": {"name": "pi05_droid_jointpos"},
                    "teacher_checkpoint": {"model_sha256": teacher_hash},
                },
                "model_identity": {
                    "candidate_stage": f"shallow-sha256:{student_hash}",
                    "reference_stage": f"base-sha256:{teacher_hash}",
                },
                "task_evidence": {"Stack3RubiksCubeTask": {"success_gap": success_gap}},
            }
        )
    )


def _native_record(*, success: bool = True, frames: int = 3) -> dict:
    return {
        "env_name": "Stack3RubiksCubeTask",
        "task_name": "Stack3RubiksCubeTask",
        "run_name": "Stack3RubiksCubeTask_0",
        "run": 0,
        "episode": 0,
        "env_id": 0,
        "policy": "pi05",
        "instruction": "Stack the rubiks cubes in a tower",
        "instruction_type": "default",
        "success": success,
        "episode_step": frames,
        "dt": 1 / 15,
    }


def _write_native_hdf5(path: pathlib.Path, *, include_images: bool = True, flat: bool = False, frames: int = 3):
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        root = handle.create_group("data")
        root.attrs["env_args"] = json.dumps({"env_name": "Stack3RubiksCubeTask", "type": 2})
        root.attrs["policy"] = "pi05"
        root.attrs["isaaclab_version"] = "2.2.0"
        root.attrs["isaacsim_version"] = "5.0.0.0"
        demo = root.create_group("demo_0")
        demo.attrs["success"] = True
        demo.attrs["num_samples"] = frames
        actions = np.arange(frames * 8, dtype=np.float32).reshape(frames, 8) / 100
        actions[:, -1] = np.arange(frames) % 2
        demo.create_dataset("actions", data=actions)
        if not include_images:
            return
        if flat:
            obs = demo.create_group("obs")
            image_group = obs
            proprio_group = obs
        else:
            obs = demo.create_group("obs")
            image_group = obs.create_group("image_obs")
            proprio_group = obs.create_group("proprio_obs")
        image_group.create_dataset("over_shoulder_left_camera", data=np.zeros((frames, 4, 5, 3), dtype=np.uint8))
        image_group.create_dataset("wrist_cam", data=np.ones((frames, 2, 3, 3), dtype=np.uint8))
        proprio_group.create_dataset("arm_joint_pos", data=np.zeros((frames, 7), dtype=np.float32))
        proprio_group.create_dataset("gripper_pos", data=np.zeros((frames, 1), dtype=np.float32))


def _build_fixture(tmp_path: pathlib.Path, *, flat: bool = False) -> tuple[dict, pathlib.Path, pathlib.Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    student = tmp_path / "student.safetensors"
    teacher = tmp_path / "teacher.safetensors"
    trigger = tmp_path / "trigger.json"
    student.write_bytes(b"accepted shallow")
    teacher.write_bytes(b"released teacher")
    _write_trigger_report(
        trigger,
        student_hash=robolab_expert_dataset.sha256_file(student),
        teacher_hash=robolab_expert_dataset.sha256_file(teacher),
    )
    root = tmp_path / "collection"
    root.mkdir()
    (root / "episode_results.jsonl").write_text(json.dumps(_native_record()) + "\n")
    _write_native_hdf5(root / "Stack3RubiksCubeTask" / "run_0.hdf5", flat=flat)
    manifest = repro_robolab_bc.build_manifest(
        robolab_output=root,
        trigger_report=trigger,
        accepted_shallow_model=student,
        teacher_model=teacher,
        openpi_source_sha="c" * 40,
        robolab_image_digest="sha256:" + "d" * 64,
    )
    manifest_path = root / "expert_bc_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest, manifest_path, student


@pytest.mark.parametrize(("flat", "layout"), [(False, "nested_obs_v1"), (True, "flat_obs_v1")])
def test_builds_native_manifest_and_reads_padded_action_chunks(tmp_path: pathlib.Path, flat, layout: str):
    manifest, manifest_path, _ = _build_fixture(tmp_path, flat=flat)

    assert manifest["selection"] == {
        "task": "Stack3RubiksCubeTask",
        "success_only": True,
        "ordering": "run_env_episode_ascending",
        "maximum_trajectories": 100,
        "selected_trajectories": 1,
        "selected_frames": 3,
    }
    assert manifest["episodes"][0]["layout"] == layout
    dataset = robolab_expert_dataset.RoboLabExpertDataset(manifest_path, action_horizon=3)
    sample = dataset[2]
    assert sample["observation.images.exterior_1_left"].shape == (4, 5, 3)
    assert sample["observation.state.joint_position"].shape == (7,)
    assert sample["action"].shape == (3, 8)
    np.testing.assert_array_equal(sample["action"][0], sample["action"][1])
    np.testing.assert_array_equal(sample["action"][1], sample["action"][2])
    assert sample["prompt"] == "Stack the rubiks cubes in a tower"
    dataset.close()


def test_fails_closed_when_public_style_hdf5_has_no_recorded_observations(tmp_path: pathlib.Path):
    student = tmp_path / "student.safetensors"
    teacher = tmp_path / "teacher.safetensors"
    trigger = tmp_path / "trigger.json"
    student.write_bytes(b"student")
    teacher.write_bytes(b"teacher")
    _write_trigger_report(
        trigger,
        student_hash=robolab_expert_dataset.sha256_file(student),
        teacher_hash=robolab_expert_dataset.sha256_file(teacher),
    )
    root = tmp_path / "collection"
    root.mkdir()
    (root / "episode_results.jsonl").write_text(json.dumps(_native_record()) + "\n")
    _write_native_hdf5(root / "Stack3RubiksCubeTask" / "run_0.hdf5", include_images=False)

    with pytest.raises(ValueError, match="--record-image-data"):
        repro_robolab_bc.build_manifest(
            robolab_output=root,
            trigger_report=trigger,
            accepted_shallow_model=student,
            teacher_model=teacher,
            openpi_source_sha="c" * 40,
            robolab_image_digest="sha256:" + "d" * 64,
        )


def test_recovery_remains_dormant_at_exactly_five_points(tmp_path: pathlib.Path):
    report = tmp_path / "trigger.json"
    _write_trigger_report(report, student_hash="a" * 64, teacher_hash="b" * 64, success_gap=0.05)
    result = repro_robolab_bc.validate_trigger_report(report)
    assert result["fired"] is False


def test_manifest_binds_training_to_accepted_shallow_without_teacher(tmp_path: pathlib.Path):
    _, manifest_path, student = _build_fixture(tmp_path)
    checkpoint = tmp_path / "accepted"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(student.read_bytes())

    initialization = robolab_expert_dataset.validate_recovery_source_checkpoint(
        manifest_path, checkpoint, teacher_checkpoint_path=None
    )
    assert initialization["teacher_checkpoint_resident"] is False
    assert initialization["loss"] == "ground_truth_flow_matching"
    with pytest.raises(ValueError, match="forbids a teacher"):
        robolab_expert_dataset.validate_recovery_source_checkpoint(
            manifest_path, checkpoint, teacher_checkpoint_path=tmp_path / "teacher"
        )
    (checkpoint / "model.safetensors").write_bytes(b"different")
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        robolab_expert_dataset.validate_recovery_source_checkpoint(
            manifest_path, checkpoint, teacher_checkpoint_path=None
        )


@pytest.mark.parametrize(("fraction", "cycle", "expert_per_batch"), [(0.25, 4, 1), (0.5, 2, 2)])
def test_deterministic_mix_is_exact_in_every_rank_local_batch(fraction: float, cycle: int, expert_per_batch: int):
    first = robolab_expert_dataset.DeterministicMixtureDataset(
        list(range(24)), ["expert-a", "expert-b", "expert-c"], expert_fraction=fraction, seed=17, num_replicas=2
    )
    second = robolab_expert_dataset.DeterministicMixtureDataset(
        list(range(24)), ["expert-a", "expert-b", "expert-c"], expert_fraction=fraction, seed=17, num_replicas=2
    )
    assert first.denominator == cycle
    for rank in range(2):
        for batch_index in range(3):
            indices = [rank + 2 * (batch_index * 4 + offset) for offset in range(4)]
            sources = [first.source_for_index(index) for index in indices]
            assert sources.count("expert") == expert_per_batch
            assert [first[index] for index in indices] == [second[index] for index in indices]


def _evaluation_records(*, stack_successes: int, banana_successes: int) -> list[dict]:
    records = []
    successes = {"Stack3RubiksCubeTask": stack_successes, "BananaInBowlTask": banana_successes}
    for task in repro_robolab_report.TASKS:
        for run in range(5):
            for env_id in range(10):
                episode = run * 10 + env_id
                records.append(
                    {
                        "env_name": task,
                        "task_name": task,
                        "run_name": f"{task}_{run}",
                        "run": run,
                        "episode": episode,
                        "env_id": env_id,
                        "policy": "pi05",
                        "instruction": f"instruction for {task}",
                        "instruction_type": "default",
                        "success": episode < successes[task],
                        "episode_step": 20,
                        "dt": 1 / 15,
                        "metrics": {"ee_path_length": 1.0, "ee_sparc": -2.0},
                    }
                )
    return records


def _identity(
    root: pathlib.Path, *, stage: str, model: pathlib.Path, stack_successes: int, banana_successes: int
) -> pathlib.Path:
    directory = root / stage
    directory.mkdir()
    model_copy = directory / "model.safetensors"
    model_copy.write_bytes(model.read_bytes())
    results = directory / "episode_results.jsonl"
    results.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in _evaluation_records(stack_successes=stack_successes, banana_successes=banana_successes)
        )
    )
    path = directory / "identity.json"
    identity = repro_robolab_report.create_run_identity(
        stage=stage,
        mode="intermediate",
        checkpoint_model=model_copy,
        results=results,
        output=path,
        num_envs=10,
        num_runs=5,
        policy_server_seed=7,
        image_digest="sha256:" + "9" * 64,
        robolab_git_sha=robolab_expert_dataset.ROBOLAB_GIT_SHA,
    )
    path.write_text(json.dumps(identity))
    return path


def test_optional_50_50_run_requires_stack_improvement_without_banana_degradation(tmp_path: pathlib.Path):
    manifest, manifest_path, student = _build_fixture(tmp_path / "fixture")
    before = _identity(tmp_path, stage="shallow", model=student, stack_successes=30, banana_successes=42)
    bc25_model = tmp_path / "bc25.safetensors"
    bc25_model.write_bytes(b"bc25")
    after = _identity(tmp_path, stage="shallow-bc25", model=bc25_model, stack_successes=32, banana_successes=42)

    decision = repro_robolab_bc.build_rerun_decision(
        before_identity_path=before, after_identity_path=after, expert_manifest_path=manifest_path
    )
    assert decision["approved"] is True
    assert decision["expert_manifest_sha256"] == manifest["manifest_sha256"]
    robolab_expert_dataset.validate_rerun_decision(
        _write_decision(tmp_path / "decision.json", decision), manifest_sha256=manifest["manifest_sha256"]
    )

    degraded = copy.deepcopy(decision)
    degraded["checks"]["banana_not_degraded"] = False
    degraded["approved"] = False
    degraded.pop("decision_sha256")
    degraded["decision_sha256"] = robolab_expert_dataset.canonical_sha256(degraded)
    with pytest.raises(ValueError, match="was not approved"):
        robolab_expert_dataset.validate_rerun_decision(
            _write_decision(tmp_path / "degraded.json", degraded), manifest_sha256=manifest["manifest_sha256"]
        )


def _write_decision(path: pathlib.Path, decision: dict) -> pathlib.Path:
    path.write_text(json.dumps(decision))
    return path
