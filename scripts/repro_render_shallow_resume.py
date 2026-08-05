#!/usr/bin/env python3
"""Render the next fail-closed Shallow checkpoint-ladder worker."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
import json
import pathlib
from typing import Any

try:
    from scripts import repro_worker
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    import repro_worker


TRACKS: dict[str, dict[str, str]] = {
    "libero": {
        "dataset_name": "libero",
        "jax_name": "libero_teacher_jax",
        "pytorch_name": "libero_teacher_pytorch",
        "config": "pi05_libero_l09_distill",
    },
    "droid": {
        "dataset_name": "droid",
        "jax_name": "droid_jointpos_teacher_jax",
        "pytorch_name": "droid_jointpos_teacher_pytorch",
        "config": "pi05_droid_l09_distill",
    },
}
TRANSITIONS = {(2_000, 5_000), (5_000, 10_000), (10_000, 20_000), (20_000, 30_000)}
EXECUTION_PROFILE_SINGLE_GPU = "single-gpu"
EXECUTION_PROFILE_LIBERO_G7E48_8GPU = "libero-g7e48-8gpu"
EXECUTION_PROFILES = {
    EXECUTION_PROFILE_SINGLE_GPU,
    EXECUTION_PROFILE_LIBERO_G7E48_8GPU,
}


class RenderError(ValueError):
    """Raised when exact source evidence cannot produce a resume worker."""


def _select_artifact(artifacts: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [artifact for artifact in artifacts if artifact.get("name") == name]
    if len(matches) != 1:
        raise RenderError(f"base spec must contain exactly one {name!r} artifact")
    return copy.deepcopy(matches[0])


def _source_identity(descriptor: Mapping[str, Any], config: str) -> tuple[str, int]:
    if descriptor.get("kind") != "checkpoint":
        raise RenderError("accepted source descriptor must be a checkpoint")
    destination = descriptor.get("destination")
    if not isinstance(destination, str):
        raise RenderError("accepted source descriptor has no destination")
    parts = pathlib.PurePosixPath(destination).parts
    if len(parts) != 3 or parts[0] != config or not parts[2].isdigit():
        raise RenderError(f"accepted source must be {config}/EXPERIMENT/POSITIVE_STEP")
    step = int(parts[2])
    if step <= 0:
        raise RenderError("accepted source step must be positive")
    return parts[1], step


def render_shallow_resume_spec(
    base_spec: Mapping[str, Any],
    checkpoint_descriptor: Mapping[str, Any],
    controller_source: Mapping[str, Any],
    *,
    track: str,
    target_step: int,
    run_id: str,
    execution_profile: str = EXECUTION_PROFILE_SINGLE_GPU,
) -> dict[str, Any]:
    """Build and validate one exact Shallow continuation."""

    if track not in TRACKS:
        raise RenderError(f"unsupported track: {track!r}")
    if execution_profile not in EXECUTION_PROFILES:
        raise RenderError(f"unsupported execution profile: {execution_profile!r}")
    if not run_id or "/" in run_id:
        raise RenderError("run ID must be one non-empty path component")
    track_config = TRACKS[track]
    accepted = copy.deepcopy(dict(checkpoint_descriptor))
    experiment, source_step = _source_identity(accepted, track_config["config"])
    if (source_step, target_step) not in TRANSITIONS:
        raise RenderError("Shallow continuation must follow 2k->5k->10k->20k->30k")
    eight_gpu_libero = execution_profile == EXECUTION_PROFILE_LIBERO_G7E48_8GPU
    if eight_gpu_libero and (track, source_step, target_step) != ("libero", 20_000, 30_000):
        raise RenderError("eight-GPU Shallow is restricted to the LIBERO 20k->30k continuation")

    base = copy.deepcopy(dict(base_spec))
    artifacts = base.get("artifacts")
    if not isinstance(artifacts, list) or not all(isinstance(item, dict) for item in artifacts):
        raise RenderError("base spec artifacts must be an array of objects")
    required = [
        _select_artifact(artifacts, track_config["dataset_name"]),
        _select_artifact(artifacts, track_config["jax_name"]),
        _select_artifact(artifacts, track_config["pytorch_name"]),
    ]
    if accepted.get("name") in {item["name"] for item in required}:
        raise RenderError("accepted checkpoint name collides with a required artifact")

    seed = base.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RenderError("base spec must contain a non-negative integer seed")
    config = track_config["config"]
    command = []
    if eight_gpu_libero:
        command.extend(["torchrun", "--standalone", "--nnodes=1", "--nproc-per-node=8"])
    else:
        command.append("python")
    command.extend(
        [
            "scripts/train_pytorch.py",
            config,
            "--exp-name",
            experiment,
            "--checkpoint-base-dir",
            "/mnt/openpi/runs",
            "--resume",
            "--seed",
            str(seed),
            "--num-train-steps",
            str(target_step),
            "--save-interval",
            "5000",
            "--log-interval",
            "10",
        ]
    )
    if eight_gpu_libero:
        # These explicit values bind eight ranks to local microbatch one while
        # preserving the accepted optimizer batch of 8 * 8 = 64.
        command.extend(["--batch-size", "8", "--gradient-accumulation-steps", "8"])
    if track == "droid":
        command.append("--no-wandb-enabled")

    base["run_id"] = run_id
    base["controller_source"] = copy.deepcopy(dict(controller_source))
    base["artifacts"] = [*required, accepted]
    base["resume_checkpoint"] = {"artifact_name": accepted["name"], "target": accepted["destination"]}
    base["container"] = {
        "command": command,
        "environment": {"WANDB_MODE": "offline"},
        "shm_size_gib": 64,
    }
    if eight_gpu_libero:
        scratch = base.get("scratch")
        if not isinstance(scratch, dict):
            raise RenderError("eight-GPU base spec must contain a scratch contract")
        scratch["expected_count"] = 4
        scratch["ordinal"] = 0
    publish_destination = f"{config}/{experiment}/{target_step}"
    base["expected_outputs"] = [
        {
            "name": f"checkpoint_{target_step}",
            "kind": "checkpoint",
            "path": f"checkpoints/{publish_destination}",
            "publish_destination": publish_destination,
        }
    ]
    bucket = base.get("aws", {}).get("artifact_bucket")
    base["output"] = {"s3_uri": f"s3://{bucket}/runs/{run_id}/"}
    return repro_worker.validate_worker_spec(base)


def _load_object(path: pathlib.Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RenderError(f"{label} must be a JSON object")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=sorted(TRACKS), required=True)
    parser.add_argument("--base-spec", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint-descriptor", type=pathlib.Path, required=True)
    parser.add_argument("--controller-source", type=pathlib.Path, required=True)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--execution-profile",
        choices=sorted(EXECUTION_PROFILES),
        default=EXECUTION_PROFILE_SINGLE_GPU,
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise RenderError(f"refusing to overwrite existing output: {args.output}")
    rendered = render_shallow_resume_spec(
        _load_object(args.base_spec, "base spec"),
        _load_object(args.checkpoint_descriptor, "checkpoint descriptor"),
        _load_object(args.controller_source, "controller source"),
        track=args.track,
        target_step=args.target_step,
        run_id=args.run_id,
        execution_profile=args.execution_profile,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rendered, indent=2, sort_keys=True) + "\n")
    args.output.chmod(0o400)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
