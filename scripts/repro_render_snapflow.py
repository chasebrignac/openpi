#!/usr/bin/env python3
"""Render a fresh 5k SnapFlow worker from an accepted Shallow checkpoint."""

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
        "source_config": "pi05_libero_l09_distill",
        "snapflow_config": "pi05_libero_l09_snapflow",
    },
    "droid": {
        "dataset_name": "droid",
        "jax_name": "droid_jointpos_teacher_jax",
        "source_config": "pi05_droid_l09_distill",
        "snapflow_config": "pi05_droid_l09_snapflow",
    },
}


class RenderError(ValueError):
    """Raised when the source evidence cannot produce an exact worker spec."""


def _select_artifact(artifacts: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [artifact for artifact in artifacts if artifact.get("name") == name]
    if len(matches) != 1:
        raise RenderError(f"base spec must contain exactly one {name!r} artifact")
    return copy.deepcopy(matches[0])


def _checkpoint_step(descriptor: Mapping[str, Any], expected_config: str) -> int:
    if descriptor.get("kind") != "checkpoint":
        raise RenderError("accepted source descriptor must be a checkpoint")
    destination = descriptor.get("destination")
    if not isinstance(destination, str):
        raise RenderError("accepted source descriptor has no destination")
    parts = pathlib.PurePosixPath(destination).parts
    if len(parts) != 3 or parts[0] != expected_config or not parts[2].isdigit():
        raise RenderError(f"accepted source must be {expected_config}/EXPERIMENT/POSITIVE_STEP")
    step = int(parts[2])
    if step <= 0:
        raise RenderError("accepted source step must be positive")
    return step


def render_snapflow_spec(
    base_spec: Mapping[str, Any],
    checkpoint_descriptor: Mapping[str, Any],
    *,
    track: str,
    run_id: str,
    experiment: str,
) -> dict[str, Any]:
    """Build and validate one immutable, direct-Python 5k SnapFlow pilot."""

    if track not in TRACKS:
        raise RenderError(f"unsupported track: {track!r}")
    if not run_id or "/" in run_id or not experiment or "/" in experiment:
        raise RenderError("run ID and experiment must be non-empty path components")
    track_config = TRACKS[track]
    accepted = copy.deepcopy(dict(checkpoint_descriptor))
    _checkpoint_step(accepted, track_config["source_config"])

    base = copy.deepcopy(dict(base_spec))
    artifacts = base.get("artifacts")
    if not isinstance(artifacts, list) or not all(isinstance(item, dict) for item in artifacts):
        raise RenderError("base spec artifacts must be an array of objects")
    dataset = _select_artifact(artifacts, track_config["dataset_name"])
    jax_teacher = _select_artifact(artifacts, track_config["jax_name"])
    if accepted.get("name") in {dataset["name"], jax_teacher["name"]}:
        raise RenderError("accepted checkpoint name collides with a required artifact")

    seed = base.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RenderError("base spec must contain a non-negative integer seed")
    snapflow_config = track_config["snapflow_config"]
    target_step = 5_000
    source_destination = str(accepted["destination"])
    command = [
        "python",
        "scripts/train_pytorch.py",
        snapflow_config,
        "--exp-name",
        experiment,
        "--checkpoint-base-dir",
        "/mnt/openpi/runs",
        "--pytorch-weight-path",
        f"/mnt/openpi/checkpoints/{source_destination}",
        "--seed",
        str(seed),
        "--num-train-steps",
        str(target_step),
        "--save-interval",
        str(target_step),
    ]
    if track == "droid":
        command.extend(["--num-workers", "0", "--no-wandb-enabled"])

    base["run_id"] = run_id
    base["artifacts"] = [dataset, jax_teacher, accepted]
    base.pop("resume_checkpoint", None)
    base["container"] = {
        "command": command,
        "environment": {"WANDB_MODE": "offline"},
        "shm_size_gib": 64,
    }
    publish_destination = f"{snapflow_config}/{experiment}/{target_step}"
    base["expected_outputs"] = [
        {
            "name": f"snapflow_checkpoint_{target_step}",
            "kind": "checkpoint",
            "path": f"checkpoints/{publish_destination}",
            "publish_destination": publish_destination,
        }
    ]
    bucket = base.get("aws", {}).get("artifact_bucket")
    base["output"] = {"s3_uri": f"s3://{bucket}/runs/{run_id}/"}

    # The accepted source and step are already bound three ways: the exact
    # staged artifact destination, --pytorch-weight-path, and the validated
    # numeric destination parsed above. Keep the worker schema closed rather
    # than adding advisory metadata that execution would ignore.
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
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise RenderError(f"refusing to overwrite existing output: {args.output}")
    rendered = render_snapflow_spec(
        _load_object(args.base_spec, "base spec"),
        _load_object(args.checkpoint_descriptor, "checkpoint descriptor"),
        track=args.track,
        run_id=args.run_id,
        experiment=args.experiment,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rendered, indent=2, sort_keys=True) + "\n")
    args.output.chmod(0o400)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
