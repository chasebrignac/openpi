#!/usr/bin/env python3
"""Create fixed-observation/fixed-noise golden vectors from a configured dataset."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import dataclasses
import hashlib
import json
import pathlib
from typing import Any

import numpy as np

import openpi.training.config as _config
import openpi.training.data_loader as _data


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_new_corpus_paths(output: pathlib.Path) -> pathlib.Path:
    """Refuse to replace either half of the canonical corpus/sidecar pair."""
    sidecar = output.with_suffix(".json")
    collisions = [path for path in (output, sidecar) if path.exists()]
    if collisions:
        rendered = ", ".join(str(path) for path in collisions)
        raise FileExistsError(f"golden corpus output already exists: {rendered}")
    return sidecar


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def dataset_provenance(config: _config.TrainConfig) -> dict[str, str]:
    base_config = getattr(config.data, "base_config", None)
    repo_id = getattr(config.data, "repo_id", None)
    revision = getattr(base_config, "lerobot_revision", None)
    codebase_version = getattr(base_config, "lerobot_codebase_version", None)
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError(f"Config {config.name!r} does not resolve a concrete dataset repo")
    if not isinstance(revision, str) or not revision:
        raise ValueError(f"Config {config.name!r} does not pin a dataset revision")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError(f"Config {config.name!r} dataset revision is not a pinned lowercase Git hash")
    if not isinstance(codebase_version, str) or not codebase_version:
        raise ValueError(f"Config {config.name!r} does not pin a LeRobot codebase version")
    return {"repo_id": repo_id, "revision": revision, "codebase_version": codebase_version}


def config_provenance(config: _config.TrainConfig, *, require_dataset: bool = True) -> dict[str, Any]:
    try:
        dataset = dataset_provenance(config)
    except ValueError:
        if require_dataset:
            raise
        dataset = None
    fingerprint_payload = {
        "name": config.name,
        "model_class": type(config.model).__qualname__,
        "model": dataclasses.asdict(config.model),
        "data_factory_class": type(config.data).__qualname__,
        "dataset": dataset,
        "offline_holdout_samples": config.offline_holdout_samples,
        "training_seed": config.seed,
    }
    return {
        "name": config.name,
        "fingerprint_sha256": canonical_sha256(fingerprint_payload),
        "model_class": type(config.model).__qualname__,
        "action_horizon": config.model.action_horizon,
        "action_dim": config.model.action_dim,
        "offline_holdout_samples": config.offline_holdout_samples,
        "training_seed": config.seed,
        "dataset": dataset,
    }


def validate_validation_split_metadata(metadata: Any, config: _config.TrainConfig) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError("Golden corpus loader did not expose whole-episode split provenance")
    episode_ids = metadata.get("validation_episode_ids")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("strategy") != "deterministic_whole_episode_stratified"
        or metadata.get("split") != "validation"
        or metadata.get("seed") != config.seed
        or metadata.get("requested_holdout_samples") != config.offline_holdout_samples
        or not isinstance(episode_ids, list)
        or not episode_ids
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in episode_ids)
        or len(episode_ids) != len(set(episode_ids))
        or metadata.get("validation_episode_count") != len(episode_ids)
        or metadata.get("selected_episode_count") != len(episode_ids)
    ):
        raise ValueError("Golden corpus whole-episode split provenance is invalid or differs from training")
    return dict(metadata)


def validate_requested_dataset_revision(config: _config.TrainConfig, requested_revision: str) -> dict[str, str]:
    dataset = dataset_provenance(config)
    if requested_revision != dataset["revision"]:
        raise ValueError(
            f"Dataset revision does not match resolved config {config.name!r}: "
            f"requested={requested_revision}, configured={dataset['revision']}"
        )
    return dataset


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def corpus_arrays(observation, actions, *, seed: int) -> tuple[dict[str, np.ndarray], dict]:
    actions_array = as_numpy(actions).astype(np.float32)
    rng = np.random.default_rng(seed)
    arrays: dict[str, np.ndarray] = {
        "state": as_numpy(observation.state),
        "tokenized_prompt": as_numpy(observation.tokenized_prompt),
        "tokenized_prompt_mask": as_numpy(observation.tokenized_prompt_mask),
        "actions": actions_array,
        "noise": rng.standard_normal(actions_array.shape, dtype=np.float32),
        # Match PI0.sample_time exactly: beta samples are scaled into
        # [0.001, 1.0] rather than clipping the lower tail.
        "time": (rng.beta(1.5, 1.0, size=actions_array.shape[0]) * 0.999 + 0.001).astype(np.float32),
    }
    image_names = list(observation.images)
    for name in image_names:
        image = as_numpy(observation.images[name])
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Golden corpus expects model-ready BCHW image {name!r}, got {image.shape}")
        arrays[f"image__{name}"] = image
        arrays[f"image_mask__{name}"] = as_numpy(observation.image_masks[name])
    metadata = {
        "schema_version": 1,
        "seed": seed,
        "samples": int(actions_array.shape[0]),
        "action_horizon": int(actions_array.shape[1]),
        "action_dim": int(actions_array.shape[2]),
        "image_names": image_names,
        # The PyTorch data loader returns model-ready BCHW tensors. The
        # framework comparison transposes these to BHWC only for JAX.
        "image_layout": "BCHW",
    }
    return arrays, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-split-seed", type=int, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.run_id.strip():
        raise ValueError("run-id must be non-empty")
    sidecar_path = require_new_corpus_paths(args.output)
    base_config = _config.get_config(args.config_name)
    if args.data_split_seed != base_config.seed:
        raise ValueError(
            f"data-split seed must match training config seed {base_config.seed}, got {args.data_split_seed}"
        )
    config = dataclasses.replace(
        base_config,
        batch_size=args.samples,
        num_workers=0,
        seed=args.data_split_seed,
    )
    dataset = validate_requested_dataset_revision(config, args.dataset_revision)
    if config.offline_holdout_samples < args.samples:
        raise ValueError(
            f"Config reserves {config.offline_holdout_samples} offline samples, fewer than requested {args.samples}"
        )
    loader = _data.create_data_loader(
        config,
        framework="pytorch",
        shuffle=False,
        num_batches=1,
        split="validation",
    )
    observation, actions = next(iter(loader))
    split_metadata = validate_validation_split_metadata(loader.split_metadata(), config)
    arrays, metadata = corpus_arrays(observation, actions, seed=args.seed)
    metadata |= {
        "schema_version": 2,
        "run_id": args.run_id,
        "config_name": args.config_name,
        "resolved_config": config_provenance(config),
        "dataset": dataset,
        "dataset_revision": dataset["revision"],
        "data_split_seed": args.data_split_seed,
        "data_split": split_metadata,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    digest = sha256_file(args.output)
    sidecar_path.write_text(json.dumps(metadata | {"sha256": digest}, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
