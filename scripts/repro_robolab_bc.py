#!/usr/bin/env python3
"""Gate, seal, and audit the conditional RoboLab expert-BC recovery path."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
from typing import Any

import numpy as np

if __package__:
    from scripts import repro_robolab_report
else:
    import repro_robolab_report

from openpi.training import robolab_expert_dataset as _expert

_STAGE_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]*-sha256:([0-9a-f]{64})")
_EXPECTED_DT = 1.0 / 15.0
_HDF5_ISAAC_SIM_VERSIONS = {"5.0.0", "5.0.0.0"}


def _json_object(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _stage_model_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or (match := _STAGE_ID_RE.fullmatch(value)) is None:
        raise ValueError(f"{label} is not a stage identity")
    return match.group(1)


def validate_trigger_report(path: pathlib.Path) -> dict[str, Any]:
    """Validate a Shallow-vs-base report and return the strict Stack trigger."""
    path = path.expanduser().resolve()
    report = _json_object(path, "RoboLab Shallow report")
    if report.get("schema_version") != 1 or report.get("benchmark") != "robolab":
        raise ValueError("Stack recovery requires a schema-v1 RoboLab report")
    runtime = report.get("runtime")
    provenance = report.get("provenance")
    identity = report.get("model_identity")
    evidence = report.get("task_evidence")
    if not all(isinstance(value, dict) for value in (runtime, provenance, identity, evidence)):
        raise ValueError("RoboLab trigger report is missing runtime, provenance, identity, or task evidence")
    if runtime.get("robolab_git_sha") != _expert.ROBOLAB_GIT_SHA:
        raise ValueError(f"Trigger report must use pinned RoboLab {_expert.ROBOLAB_GIT_SHA}")
    if runtime.get("openpi_client_git_sha") != _expert.ROBOLAB_OPENPI_CLIENT_GIT_SHA:
        raise ValueError(f"Trigger report must use pinned OpenPI client {_expert.ROBOLAB_OPENPI_CLIENT_GIT_SHA}")
    student = provenance.get("student_checkpoint")
    teacher = provenance.get("teacher_checkpoint")
    student_config = provenance.get("student_config")
    teacher_config = provenance.get("teacher_config")
    if not all(isinstance(value, dict) for value in (student, teacher, student_config, teacher_config)):
        raise ValueError("Trigger report is missing checkpoint provenance")
    if student_config.get("name") != "pi05_droid_l09_distill":
        raise ValueError("Stack recovery may only start from pi05_droid_l09_distill")
    if teacher_config.get("name") != "pi05_droid_jointpos":
        raise ValueError("Stack recovery must use the released pi05_droid_jointpos reference")
    candidate_hash = _stage_model_hash(identity.get("candidate_stage"), "candidate_stage")
    reference_hash = _stage_model_hash(identity.get("reference_stage"), "reference_stage")
    if student.get("model_sha256") != candidate_hash:
        raise ValueError("Trigger report candidate identity does not match its Shallow checkpoint")
    if teacher.get("model_sha256") != reference_hash:
        raise ValueError("Trigger report reference identity does not match its teacher checkpoint")
    stack = evidence.get(_expert.STACK_TASK)
    if not isinstance(stack, dict):
        raise ValueError(f"Trigger report has no {_expert.STACK_TASK} evidence")
    success_gap = stack.get("success_gap")
    if isinstance(success_gap, bool) or not isinstance(success_gap, int | float) or not math.isfinite(success_gap):
        raise ValueError("Stack success_gap must be a finite number")
    fired = success_gap > _expert.TRIGGER_GAP
    return {
        "task": _expert.STACK_TASK,
        "success_gap": float(success_gap),
        "threshold": _expert.TRIGGER_GAP,
        "comparison": "strictly_greater_than",
        "fired": fired,
        "accepted_shallow_model_sha256": candidate_hash,
        "teacher_model_sha256": reference_hash,
        "checkpoint_step": report.get("checkpoint_step"),
        "report_sha256": _expert.sha256_file(path),
    }


def _load_native_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.suffix != ".jsonl":
        raise ValueError(f"Expected native RoboLab episode_results.jsonl: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Malformed RoboLab JSONL line {line_number}: {path}") from error
        if not isinstance(record, dict):
            raise ValueError(f"RoboLab JSONL line {line_number} is not an object")
        records.append(record)
    if not records:
        raise ValueError(f"RoboLab JSONL has no episodes: {path}")
    return records


def _validated_collection_records(path: pathlib.Path) -> list[dict[str, Any]]:
    records = _load_native_jsonl(path)
    seen: set[tuple[int, int, int]] = set()
    successes: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if record.get("task_name") != _expert.STACK_TASK or record.get("env_name") != _expert.STACK_TASK:
            raise ValueError(f"Expert collection episode {index} is not the pinned Stack3RubiksCubeTask")
        if record.get("policy") != "pi05" or record.get("instruction_type") != "default":
            raise ValueError(f"Expert collection episode {index} does not use the pinned pi05/default contract")
        if not isinstance(record.get("instruction"), str) or not record["instruction"].strip():
            raise ValueError(f"Expert collection episode {index} has no instruction")
        if not isinstance(record.get("success"), bool):
            raise ValueError(f"Expert collection episode {index} success must be a JSON boolean")
        run = record.get("run")
        env_id = record.get("env_id")
        episode = record.get("episode")
        episode_step = record.get("episode_step")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (run, env_id, episode)):
            raise ValueError(f"Expert collection episode {index} has invalid run/env_id/episode")
        if isinstance(episode_step, bool) or not isinstance(episode_step, int) or episode_step <= 0:
            raise ValueError(f"Expert collection episode {index} has invalid episode_step")
        identity = (run, env_id, episode)
        if identity in seen:
            raise ValueError(f"Duplicate expert collection identity: {identity}")
        seen.add(identity)
        dt = record.get("dt")
        if (
            isinstance(dt, bool)
            or not isinstance(dt, int | float)
            or not math.isclose(dt, _EXPECTED_DT, rel_tol=0.0, abs_tol=1e-9)
        ):
            raise ValueError(f"Expert collection episode {index} has dt={dt!r}; expected {_EXPECTED_DT}")
        if record["success"]:
            successes.append(record)
    if not successes:
        raise ValueError("Expert collection has no successful Stack3RubiksCubeTask trajectories")
    if len(successes) > _expert.MAX_EXPERT_TRAJECTORIES:
        raise ValueError(
            f"Expert collection has {len(successes)} successful trajectories; the hard maximum is "
            f"{_expert.MAX_EXPERT_TRAJECTORIES}"
        )
    return sorted(successes, key=lambda record: (record["run"], record["env_id"], record["episode"]))


def _import_h5py():
    try:
        import h5py
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Preparing RoboLab expert data requires h5py; run this inside the pinned RoboLab image"
        ) from error
    return h5py


def _validate_array(dataset, *, shape: tuple[int | None, ...], dtype_kind: str, label: str) -> None:
    if dataset.ndim != len(shape) or any(
        expected is not None and dataset.shape[i] != expected for i, expected in enumerate(shape)
    ):
        raise ValueError(f"{label} has shape {dataset.shape}; expected {shape}")
    if dtype_kind == "uint8" and dataset.dtype != np.uint8:
        raise ValueError(f"{label} has dtype {dataset.dtype}; expected uint8")
    if dtype_kind == "float" and not np.issubdtype(dataset.dtype, np.floating):
        raise ValueError(f"{label} has dtype {dataset.dtype}; expected floating point")


def _validate_demo(hdf5_path: pathlib.Path, record: dict[str, Any]) -> tuple[str, int, str]:
    h5 = _import_h5py()
    group_path = f"data/demo_{record['env_id']}"
    with h5.File(hdf5_path, "r", swmr=True) as handle:
        if "data" not in handle or group_path not in handle:
            raise ValueError(f"RoboLab HDF5 lacks {group_path}: {hdf5_path}")
        root = handle["data"]
        try:
            env_args = json.loads(root.attrs["env_args"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"RoboLab HDF5 has invalid data.env_args: {hdf5_path}") from error
        if env_args.get("env_name") != _expert.STACK_TASK or root.attrs.get("policy") != "pi05":
            raise ValueError(f"RoboLab HDF5 is not a pi05 Stack collection: {hdf5_path}")
        if root.attrs.get("isaaclab_version") != "2.2.0":
            raise ValueError(f"RoboLab HDF5 does not use Isaac Lab 2.2.0: {hdf5_path}")
        if root.attrs.get("isaacsim_version") not in _HDF5_ISAAC_SIM_VERSIONS:
            raise ValueError(f"RoboLab HDF5 does not use Isaac Sim 5.0.0: {hdf5_path}")
        demo = handle[group_path]
        success = demo.attrs.get("success")
        if not isinstance(success, bool | np.bool_) or not bool(success):
            raise ValueError(f"RoboLab HDF5 {group_path} is not marked successful")
        recognized = [
            layout for layout, paths in _expert.LAYOUT_PATHS.items() if all(path in demo for path in paths.values())
        ]
        if not recognized:
            alternatives = [", ".join(paths.values()) for paths in _expert.LAYOUT_PATHS.values()]
            raise ValueError(
                f"RoboLab HDF5 {group_path} has no trainable observation payload. "
                f"Collection must use --record-image-data and contain one complete layout: {alternatives}"
            )
        if len(recognized) != 1:
            raise ValueError(f"RoboLab HDF5 {group_path} ambiguously contains multiple observation layouts")
        layout = recognized[0]
        paths = _expert.LAYOUT_PATHS[layout]
        actions = demo[paths["actions"]]
        frames = actions.shape[0]
        _validate_array(actions, shape=(frames, 8), dtype_kind="float", label=f"{group_path}/actions")
        _validate_array(
            demo[paths["exterior"]],
            shape=(frames, None, None, 3),
            dtype_kind="uint8",
            label=f"{group_path}/{paths['exterior']}",
        )
        _validate_array(
            demo[paths["wrist"]],
            shape=(frames, None, None, 3),
            dtype_kind="uint8",
            label=f"{group_path}/{paths['wrist']}",
        )
        _validate_array(
            demo[paths["joints"]],
            shape=(frames, 7),
            dtype_kind="float",
            label=f"{group_path}/{paths['joints']}",
        )
        _validate_array(
            demo[paths["gripper"]],
            shape=(frames, 1),
            dtype_kind="float",
            label=f"{group_path}/{paths['gripper']}",
        )
        if frames <= 0 or int(demo.attrs.get("num_samples", -1)) != frames or record["episode_step"] != frames:
            raise ValueError(f"RoboLab HDF5/JSONL frame count mismatch for {group_path}")
        numeric = {
            "actions": np.asarray(actions),
            "joint positions": np.asarray(demo[paths["joints"]]),
            "gripper positions": np.asarray(demo[paths["gripper"]]),
        }
        for label, values in numeric.items():
            if not np.isfinite(values).all():
                raise ValueError(f"RoboLab HDF5 {group_path} contains non-finite {label}")
        if np.any(numeric["gripper positions"] < 0) or np.any(numeric["gripper positions"] > 1):
            raise ValueError(f"RoboLab HDF5 {group_path} gripper observations leave [0, 1]")
        if np.any(numeric["actions"][:, -1] < 0) or np.any(numeric["actions"][:, -1] > 1):
            raise ValueError(f"RoboLab HDF5 {group_path} gripper actions leave [0, 1]")
        return layout, frames, group_path


def build_manifest(
    *,
    robolab_output: pathlib.Path,
    trigger_report: pathlib.Path,
    accepted_shallow_model: pathlib.Path,
    teacher_model: pathlib.Path,
    openpi_source_sha: str,
    robolab_image_digest: str,
    maximum_trajectories: int = _expert.MAX_EXPERT_TRAJECTORIES,
) -> dict[str, Any]:
    """Build a deterministic manifest from native successful RoboLab trajectories."""
    robolab_output = robolab_output.expanduser().resolve()
    results_path = robolab_output / "episode_results.jsonl"
    if (
        isinstance(maximum_trajectories, bool)
        or not isinstance(maximum_trajectories, int)
        or not 1 <= maximum_trajectories <= 100
    ):
        raise ValueError("maximum_trajectories must be an integer in [1, 100]")
    if re.fullmatch(r"[0-9a-f]{40}", openpi_source_sha) is None:
        raise ValueError("openpi_source_sha must be an exact lowercase 40-character Git SHA")
    if re.fullmatch(r"(?:.+@)?sha256:[0-9a-f]{64}", robolab_image_digest) is None:
        raise ValueError("robolab_image_digest must be an immutable sha256 digest")
    trigger = validate_trigger_report(trigger_report)
    if not trigger["fired"]:
        raise ValueError(
            f"Stack recovery is dormant: success gap {trigger['success_gap']:.6f} is not strictly greater than 0.05"
        )
    accepted_shallow_model = accepted_shallow_model.expanduser().resolve()
    teacher_model = teacher_model.expanduser().resolve()
    if not accepted_shallow_model.is_file() or not teacher_model.is_file():
        raise ValueError("Accepted Shallow and teacher model.safetensors files must both exist")
    accepted_hash = _expert.sha256_file(accepted_shallow_model)
    teacher_hash = _expert.sha256_file(teacher_model)
    if accepted_hash != trigger["accepted_shallow_model_sha256"]:
        raise ValueError("Accepted Shallow model does not match the checkpoint that fired the trigger")
    if teacher_hash != trigger["teacher_model_sha256"]:
        raise ValueError("Expert teacher model does not match the trigger report reference")

    successes = _validated_collection_records(results_path)[:maximum_trajectories]
    file_records: list[dict[str, Any]] = []
    file_indices: dict[str, int] = {}
    episode_records: list[dict[str, Any]] = []
    selected_frames = 0
    for record in successes:
        hdf5_path = robolab_output / record["env_name"] / f"run_{record['run']}.hdf5"
        if not hdf5_path.is_file():
            raise ValueError(f"Native RoboLab trajectory file does not exist: {hdf5_path}")
        relative_path = hdf5_path.relative_to(robolab_output).as_posix()
        if relative_path not in file_indices:
            file_indices[relative_path] = len(file_records)
            file_records.append(
                {
                    "path": relative_path,
                    "size_bytes": hdf5_path.stat().st_size,
                    "sha256": _expert.sha256_file(hdf5_path),
                }
            )
        layout, frames, group_path = _validate_demo(hdf5_path, record)
        selected_frames += frames
        episode_records.append(
            {
                "trajectory_id": f"run-{record['run']:04d}-env-{record['env_id']:03d}-episode-{record['episode']:05d}",
                "file_index": file_indices[relative_path],
                "group": group_path,
                "layout": layout,
                "frames": frames,
                "run": record["run"],
                "env_id": record["env_id"],
                "episode": record["episode"],
                "instruction": record["instruction"].strip(),
                "result_record_sha256": _expert.canonical_sha256(record),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset": _expert.MANIFEST_DATASET,
        "provenance": {
            "robolab_git_sha": _expert.ROBOLAB_GIT_SHA,
            "openpi_client_git_sha": _expert.ROBOLAB_OPENPI_CLIENT_GIT_SHA,
            "openpi_source_sha": openpi_source_sha,
            "robolab_image_digest": robolab_image_digest,
            "record_image_data_required": True,
            "native_results": {
                "path": "episode_results.jsonl",
                "sha256": _expert.sha256_file(results_path),
            },
            "trigger_report_sha256": trigger["report_sha256"],
            "trigger": {
                "task": trigger["task"],
                "success_gap": trigger["success_gap"],
                "threshold": trigger["threshold"],
                "comparison": trigger["comparison"],
                "fired": True,
            },
            "accepted_shallow_checkpoint": {
                "model_sha256": accepted_hash,
                "config": "pi05_droid_l09_distill",
                "step": trigger["checkpoint_step"],
            },
            "teacher_checkpoint": {
                "model_sha256": teacher_hash,
                "config": "pi05_droid_jointpos",
                "used_for_collection_only": True,
                "resident_during_bc": False,
            },
        },
        "selection": {
            "task": _expert.STACK_TASK,
            "success_only": True,
            "ordering": "run_env_episode_ascending",
            "maximum_trajectories": maximum_trajectories,
            "selected_trajectories": len(episode_records),
            "selected_frames": selected_frames,
        },
        "contract": {
            "frequency_hz": 15,
            "action_space": "absolute_7_joint_position_plus_binary_gripper",
            "image_dtype": "uint8_hwc",
            "joint_shape": [7],
            "gripper_shape": [1],
            "action_shape": [8],
            "supported_layouts": sorted(_expert.LAYOUT_PATHS),
        },
        "files": file_records,
        "episodes": episode_records,
    }
    manifest["manifest_sha256"] = _expert.canonical_sha256(manifest)
    return manifest


def build_rerun_decision(
    *, before_identity_path: pathlib.Path, after_identity_path: pathlib.Path, expert_manifest_path: pathlib.Path
) -> dict[str, Any]:
    """Permit 50/50 only when BC25 improves Stack and does not degrade Banana."""
    manifest = _expert.load_expert_manifest(expert_manifest_path, verify_files=False)
    before, before_records = repro_robolab_report._load_identity(before_identity_path)  # noqa: SLF001
    after, after_records = repro_robolab_report._load_identity(after_identity_path)  # noqa: SLF001
    if before.get("stage") != "shallow" or after.get("stage") != "shallow-bc25":
        raise ValueError("BC rerun decision requires shallow and shallow-bc25 stage identities")
    if before["evaluation"].get("mode") != "intermediate" or after["evaluation"].get("mode") != "intermediate":
        raise ValueError("BC rerun decision requires 50-episode intermediate evaluations")
    if before["runtime"] != after["runtime"] or before["evaluation"] != after["evaluation"]:
        raise ValueError("Before/after BC evaluations must use identical pinned runtime and evaluation inputs")
    if before_records.keys() != after_records.keys():
        raise ValueError("Before/after BC episode identities differ")
    for key in before_records:
        for field in ("env_name", "task_name", "episode", "run", "env_id", "instruction", "dt"):
            if before_records[key].get(field) != after_records[key].get(field):
                raise ValueError(f"Before/after BC episode {key} differs in {field}")
    accepted_hash = manifest["provenance"]["accepted_shallow_checkpoint"]["model_sha256"]
    if before["checkpoint"]["model_sha256"] != accepted_hash:
        raise ValueError("BC rerun baseline is not the accepted Shallow checkpoint")
    if after["checkpoint"]["model_sha256"] == accepted_hash:
        raise ValueError("BC25 evaluation did not use a new checkpoint")
    rates: dict[str, dict[str, float]] = {}
    for task in repro_robolab_report.TASKS:
        keys = [key for key in before_records if key[0] == task]
        before_rate = sum(bool(before_records[key]["success"]) for key in keys) / len(keys)
        after_rate = sum(bool(after_records[key]["success"]) for key in keys) / len(keys)
        rates[task] = {"before": before_rate, "after": after_rate, "delta": after_rate - before_rate}
    checks = {
        "stack_improved": rates[_expert.STACK_TASK]["delta"] > 0,
        "banana_not_degraded": rates["BananaInBowlTask"]["delta"] >= 0,
    }
    decision: dict[str, Any] = {
        "schema_version": 1,
        "decision": "robolab_bc_50_50",
        "expert_manifest_sha256": manifest["manifest_sha256"],
        "before_identity_sha256": _expert.sha256_file(before_identity_path.expanduser().resolve()),
        "after_identity_sha256": _expert.sha256_file(after_identity_path.expanduser().resolve()),
        "accepted_shallow_model_sha256": accepted_hash,
        "bc25_model_sha256": after["checkpoint"]["model_sha256"],
        "rates": rates,
        "checks": checks,
        "approved": all(checks.values()),
    }
    decision["decision_sha256"] = _expert.canonical_sha256(decision)
    return decision


def _write_json_atomic(path: pathlib.Path, value: dict[str, Any], *, overwrite: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise ValueError(f"Refusing to overwrite existing output without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    trigger = subparsers.add_parser("check-trigger", help="report whether the >5-point Stack trigger fired")
    trigger.add_argument("--report", required=True, type=pathlib.Path)
    trigger.add_argument("--output", type=pathlib.Path)
    trigger.add_argument("--overwrite", action="store_true")

    prepare = subparsers.add_parser("prepare", help="seal successful native RoboLab trajectories")
    prepare.add_argument("--robolab-output", required=True, type=pathlib.Path)
    prepare.add_argument("--trigger-report", required=True, type=pathlib.Path)
    prepare.add_argument("--accepted-shallow-model", required=True, type=pathlib.Path)
    prepare.add_argument("--teacher-model", required=True, type=pathlib.Path)
    prepare.add_argument("--source-sha", required=True)
    prepare.add_argument("--robolab-image-digest", required=True)
    prepare.add_argument("--maximum-trajectories", type=int, default=100)
    prepare.add_argument("--output", required=True, type=pathlib.Path)
    prepare.add_argument("--overwrite", action="store_true")

    rerun = subparsers.add_parser("decide-50-50", help="gate the optional second 1,500-step run")
    rerun.add_argument("--before-identity", required=True, type=pathlib.Path)
    rerun.add_argument("--after-identity", required=True, type=pathlib.Path)
    rerun.add_argument("--expert-manifest", required=True, type=pathlib.Path)
    rerun.add_argument("--output", required=True, type=pathlib.Path)
    rerun.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "check-trigger":
        result = validate_trigger_report(args.report)
        if args.output is not None:
            _write_json_atomic(args.output, result, overwrite=args.overwrite)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["fired"] else 3
    if args.command == "prepare":
        output = args.output.expanduser().resolve()
        root = args.robolab_output.expanduser().resolve()
        if output.parent != root:
            raise ValueError("The expert manifest must live at the native RoboLab output root for portable paths")
        manifest = build_manifest(
            robolab_output=root,
            trigger_report=args.trigger_report,
            accepted_shallow_model=args.accepted_shallow_model,
            teacher_model=args.teacher_model,
            openpi_source_sha=args.source_sha,
            robolab_image_digest=args.robolab_image_digest,
            maximum_trajectories=args.maximum_trajectories,
        )
        _write_json_atomic(output, manifest, overwrite=args.overwrite)
        print(
            json.dumps({"output": str(output), **manifest["selection"], "manifest_sha256": manifest["manifest_sha256"]})
        )
        return 0
    decision = build_rerun_decision(
        before_identity_path=args.before_identity,
        after_identity_path=args.after_identity,
        expert_manifest_path=args.expert_manifest,
    )
    _write_json_atomic(args.output, decision, overwrite=args.overwrite)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0 if decision["approved"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
