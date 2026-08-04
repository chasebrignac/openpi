import copy
import json
import pathlib

import pytest

from scripts import repro_promotion_report
from scripts import repro_robolab_report


def _offline_report(*, student_hash: str, teacher_hash: str, step: int = 5_000) -> dict:
    provenance = {
        "schema_version": 1,
        "run_id": "droid-run-001",
        "student_config": {
            "name": "pi05_droid_l09_distill",
            "fingerprint_sha256": "a" * 64,
            "training_fingerprint_sha256": "b" * 64,
        },
        "student_checkpoint": {
            "path": f"/runs/student/{step}",
            "step": step,
            "model_sha256": student_hash,
            "metadata_sha256": "d" * 64,
        },
        "teacher_config": {"name": "pi05_droid_jointpos", "fingerprint_sha256": "e" * 64},
        "teacher_checkpoint": {"path": "/teacher", "model_sha256": teacher_hash},
        "dataset": {"repo_id": "allenai/MolmoAct2-DROID-Dataset", "revision": "1" * 40},
        "golden": {"sha256": "2" * 64, "metadata_sha256": "3" * 64},
        "normalization_range": {"low": -1.0, "high": 1.0},
    }
    return {
        "stage": "shallow",
        "student_config": "pi05_droid_l09_distill",
        "student_step": step,
        "teacher_config": "pi05_droid_jointpos",
        "provenance": provenance,
        "action_metrics": {
            "kd_mse": 0.01,
            "kd_cosine_mean": 0.99,
            "per_joint_normalized_rmse": [0.1],
            "action_chunk_rmse": 0.1,
            "normalization_range_excursions": 0,
        },
    }


def _records(*, successes: dict[str, int], num_envs: int = 10, num_runs: int = 5) -> list[dict]:
    records = []
    for task_index, task in enumerate(repro_robolab_report.TASKS):
        for run in range(num_runs):
            for env_id in range(num_envs):
                episode = run * num_envs + env_id
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
                        "duration": 20 / 15,
                        "dt": 1 / 15,
                        "metrics": {
                            "ee_path_length": 1.0 + task_index + episode / 100,
                            "ee_sparc": -2.0 - task_index - episode / 100,
                        },
                        "events": {},
                    }
                )
    return records


def _write_jsonl(path: pathlib.Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _policy_identity_kwargs(*, config: str) -> dict:
    return {
        "policy_image_digest": "sha256:" + "8" * 64,
        "policy_source_s3_uri": "s3://pi05-test/source/openpi-" + "7" * 40 + "-complete.bundle",
        "policy_source_version_id": "source-version",
        "policy_source_sha256": "6" * 64,
        "policy_source_commit": "7" * 40,
        "policy_config": config,
        "policy_command_sha256": "5" * 64,
    }


def _seal(
    root: pathlib.Path,
    *,
    stage: str,
    model_bytes: bytes,
    successes: dict[str, int],
) -> tuple[pathlib.Path, str, pathlib.Path]:
    directory = root / stage
    directory.mkdir()
    model = directory / "model.safetensors"
    results = directory / "episode_results.jsonl"
    identity_path = directory / "run-identity.json"
    model.write_bytes(model_bytes)
    _write_jsonl(results, _records(successes=successes))
    identity = repro_robolab_report.create_run_identity(
        stage=stage,
        mode="intermediate",
        checkpoint_model=model,
        results=results,
        output=identity_path,
        num_envs=10,
        num_runs=5,
        policy_server_seed=7003,
        image_digest="sha256:" + "9" * 64,
        robolab_git_sha=repro_robolab_report.ROBOLAB_GIT_SHA,
        **_policy_identity_kwargs(config="pi05_droid_jointpos" if stage == "base" else "pi05_droid_l09_distill"),
    )
    identity_path.write_text(json.dumps(identity))
    return identity_path, repro_robolab_report.sha256_file(model), results


def test_seals_and_emits_promotion_ready_success_path_and_sparc(tmp_path: pathlib.Path):
    reference_identity, teacher_hash, _ = _seal(
        tmp_path,
        stage="base",
        model_bytes=b"teacher",
        successes={"BananaInBowlTask": 45, "Stack3RubiksCubeTask": 40},
    )
    candidate_identity, student_hash, _ = _seal(
        tmp_path,
        stage="shallow",
        model_bytes=b"student",
        successes={"BananaInBowlTask": 43, "Stack3RubiksCubeTask": 39},
    )
    offline = _offline_report(student_hash=student_hash, teacher_hash=teacher_hash)

    report = repro_robolab_report.build_report(
        reference_identity_path=reference_identity,
        candidate_identity_path=candidate_identity,
        offline_report=offline,
        expected_reference_stage="base",
        expected_candidate_stage="shallow",
    )

    assert report["checkpoint_step"] == 5_000
    assert report["provenance"] == offline["provenance"]
    assert report["paired_rollout"] == {
        "student_success": pytest.approx(0.82),
        "reference_success": pytest.approx(0.85),
        "complete_pairs": 100,
    }
    assert report["evaluation_gate"]["passed"] is True
    banana = report["task_evidence"]["BananaInBowlTask"]
    assert banana["episodes"] == 50
    assert banana["reference_ee_path_length"]["count"] == 50
    assert banana["candidate_ee_sparc"]["count"] == 50
    promotion = repro_promotion_report.build_promotion_report(
        stage="shallow",
        offline_reports=[offline],
        quality_reports=[report],
        max_rollout_gap=0.05,
    )
    assert promotion["promotion_ready"] is True


def test_rejects_missing_or_extra_episode_instead_of_shrinking_count(tmp_path: pathlib.Path):
    model = tmp_path / "model.safetensors"
    results = tmp_path / "episode_results.jsonl"
    model.write_bytes(b"model")
    records = _records(successes=dict.fromkeys(repro_robolab_report.TASKS, 50))
    _write_jsonl(results, records[:-1])

    with pytest.raises(ValueError, match="exactly 100"):
        repro_robolab_report.create_run_identity(
            stage="base",
            mode="intermediate",
            checkpoint_model=model,
            results=results,
            output=tmp_path / "identity.json",
            num_envs=10,
            num_runs=5,
            policy_server_seed=7,
            image_digest="sha256:" + "9" * 64,
            robolab_git_sha=repro_robolab_report.ROBOLAB_GIT_SHA,
            **_policy_identity_kwargs(config="pi05_droid_jointpos"),
        )


def test_rejects_wrong_task_policy_and_nonfinite_motion_metrics():
    records = _records(successes=dict.fromkeys(repro_robolab_report.TASKS, 50))
    bad_task = copy.deepcopy(records)
    bad_task[0]["task_name"] = "BananaInBowlTableTask"
    with pytest.raises(ValueError, match="unexpected task_name"):
        repro_robolab_report.validate_native_results(bad_task, mode="intermediate", num_envs=10, num_runs=5)

    bad_policy = copy.deepcopy(records)
    bad_policy[0]["policy"] = "pi0"
    with pytest.raises(ValueError, match="policy must be"):
        repro_robolab_report.validate_native_results(bad_policy, mode="intermediate", num_envs=10, num_runs=5)

    bad_metric = copy.deepcopy(records)
    bad_metric[0]["metrics"]["ee_sparc"] = float("nan")
    with pytest.raises(ValueError, match="ee_sparc must be a finite number"):
        repro_robolab_report.validate_native_results(bad_metric, mode="intermediate", num_envs=10, num_runs=5)


def test_final_mode_requires_exactly_200_episodes_per_task():
    records = _records(
        successes=dict.fromkeys(repro_robolab_report.TASKS, 200),
        num_envs=10,
        num_runs=20,
    )
    indexed = repro_robolab_report.validate_native_results(records, mode="final", num_envs=10, num_runs=20)
    assert len(indexed) == 400
    with pytest.raises(ValueError, match="exactly 200 episodes/task"):
        repro_robolab_report.validate_native_results(records, mode="final", num_envs=10, num_runs=5)


def test_partial_continuation_keeps_only_complete_ordered_run_batches():
    records = _records(successes=dict.fromkeys(repro_robolab_report.TASKS, 50))

    assert (
        len(repro_robolab_report.complete_native_run_prefix(records[:27], mode="intermediate", num_envs=10, num_runs=5))
        == 20
    )
    complete = records[:30]
    assert (
        len(repro_robolab_report.validate_native_continuation(complete, mode="intermediate", num_envs=10, num_runs=5))
        == 30
    )
    with pytest.raises(ValueError, match="incomplete run batch"):
        repro_robolab_report.validate_native_continuation(records[:27], mode="intermediate", num_envs=10, num_runs=5)


def test_partial_continuation_rejects_out_of_order_complete_batches():
    records = _records(successes=dict.fromkeys(repro_robolab_report.TASKS, 50))
    out_of_order = records[10:20]

    with pytest.raises(ValueError, match="beyond the resumable prefix"):
        repro_robolab_report.complete_native_run_prefix(out_of_order, mode="intermediate", num_envs=10, num_runs=5)


def test_rejects_tampered_results_and_wrong_stage_identity(tmp_path: pathlib.Path):
    identity_path, _, results = _seal(
        tmp_path,
        stage="base",
        model_bytes=b"teacher",
        successes=dict.fromkeys(repro_robolab_report.TASKS, 50),
    )
    results.write_text(results.read_text() + "\n")
    with pytest.raises(ValueError, match="result hash"):
        repro_robolab_report._load_identity(identity_path)  # noqa: SLF001

    identity = json.loads(identity_path.read_text())
    identity["stage_identity"] = "base-sha256:" + "f" * 64
    identity_path.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="stage_identity"):
        repro_robolab_report._load_identity(identity_path)  # noqa: SLF001


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("policy_server", "image_digest"), "not-a-digest", "policy image digest"),
        (("policy_server", "command_sha256"), "f" * 63, "policy command hash"),
        (("policy_server", "checkpoint_model_sha256"), "f" * 64, "policy checkpoint"),
        (("policy_server", "source", "commit"), "f" * 39, "policy source commit"),
        (("policy_server", "source", "version_id"), "", "policy source version"),
    ],
)
def test_identity_rejects_tampered_policy_server_pins(tmp_path, path, replacement, message):
    identity_path, _, _ = _seal(
        tmp_path,
        stage="base",
        model_bytes=b"teacher",
        successes=dict.fromkeys(repro_robolab_report.TASKS, 50),
    )
    document = json.loads(identity_path.read_text())
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    identity_path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match=message):
        repro_robolab_report._load_identity(identity_path)  # noqa: SLF001


def test_rejects_candidate_not_bound_to_offline_checkpoint(tmp_path: pathlib.Path):
    reference_identity, teacher_hash, _ = _seal(
        tmp_path,
        stage="base",
        model_bytes=b"teacher",
        successes=dict.fromkeys(repro_robolab_report.TASKS, 50),
    )
    candidate_identity, _, _ = _seal(
        tmp_path,
        stage="shallow",
        model_bytes=b"student",
        successes=dict.fromkeys(repro_robolab_report.TASKS, 50),
    )
    offline = _offline_report(student_hash="f" * 64, teacher_hash=teacher_hash)
    with pytest.raises(ValueError, match="offline student checkpoint"):
        repro_robolab_report.build_report(
            reference_identity_path=reference_identity,
            candidate_identity_path=candidate_identity,
            offline_report=offline,
            expected_reference_stage="base",
            expected_candidate_stage="shallow",
        )


def test_task_noninferiority_failure_is_reported_not_hidden(tmp_path: pathlib.Path):
    reference_identity, teacher_hash, _ = _seal(
        tmp_path,
        stage="base",
        model_bytes=b"teacher",
        successes=dict.fromkeys(repro_robolab_report.TASKS, 50),
    )
    candidate_identity, student_hash, _ = _seal(
        tmp_path,
        stage="final",
        model_bytes=b"student",
        successes={"BananaInBowlTask": 49, "Stack3RubiksCubeTask": 40},
    )
    report = repro_robolab_report.build_report(
        reference_identity_path=reference_identity,
        candidate_identity_path=candidate_identity,
        offline_report=_offline_report(student_hash=student_hash, teacher_hash=teacher_hash),
        expected_reference_stage="base",
        expected_candidate_stage="final",
    )
    assert report["evaluation_gate"]["passed"] is False
    assert "paired_rollout" not in report
    assert report["observed_paired_rollout"]["complete_pairs"] == 100
    failed = {check["name"] for check in report["evaluation_gate"]["checks"] if not check["passed"]}
    assert "Stack3RubiksCubeTask_success_noninferiority" in failed
    promotion = repro_promotion_report.build_promotion_report(
        stage="shallow",
        offline_reports=[_offline_report(student_hash=student_hash, teacher_hash=teacher_hash)],
        quality_reports=[report],
        max_rollout_gap=0.05,
    )
    assert promotion["promotion_ready"] is False
    assert "paired_rollout_gap" in promotion["missing_required_gates"]
