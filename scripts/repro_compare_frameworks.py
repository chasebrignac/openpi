#!/usr/bin/env python3
"""Validate a converted PyTorch teacher against JAX on fixed golden vectors."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import dataclasses
import gc
import json
import os
import pathlib
from typing import Any

# JAX and Torch must coexist during this check. Avoid JAX's default allocator
# retaining nearly all device memory after the first forward pass.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import safetensors.torch
import torch

import openpi.models.model as _model
import openpi.models.pi0_config as _pi0_config
import openpi.models_pytorch.pi0_pytorch as _pi0_pytorch
import openpi.training.config as _config

if __package__:
    from scripts.repro_make_golden import config_provenance
    from scripts.repro_make_golden import sha256_file
    from scripts.repro_make_golden import validate_validation_split_metadata
    from scripts.repro_offline_metrics import cosine_similarity
else:
    from repro_make_golden import config_provenance
    from repro_make_golden import sha256_file
    from repro_make_golden import validate_validation_split_metadata
    from repro_offline_metrics import cosine_similarity


GATE_COSINE_MINIMUM = 0.999


def _read_json_object(path: pathlib.Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON must be an object: {path}")
    return value


def _json_round_trip(value: Any) -> Any:
    """Normalize tuples and other JSON-compatible containers for comparison."""
    return json.loads(json.dumps(value, sort_keys=True))


def load_golden_corpus(
    corpus_path: pathlib.Path,
    *,
    teacher_config_name: str,
    model_config: _pi0_config.Pi0Config,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load a corpus only after validating its bytes, config, and tensor contract."""
    corpus_path = corpus_path.expanduser().resolve()
    metadata_path = corpus_path.with_suffix(".json")
    if not corpus_path.is_file():
        raise ValueError(f"Golden corpus does not exist: {corpus_path}")
    metadata = _read_json_object(metadata_path, label="golden-corpus sidecar")
    if metadata.get("schema_version") != 2:
        raise ValueError("Golden corpus must use schema_version 2")
    actual_sha256 = sha256_file(corpus_path)
    if metadata.get("sha256") != actual_sha256:
        raise ValueError("Golden corpus SHA-256 does not match its sidecar")

    corpus_config_name = metadata.get("config_name")
    if not isinstance(corpus_config_name, str) or not corpus_config_name:
        raise ValueError("Golden corpus does not identify its resolved training config")
    corpus_config = _config.get_config(corpus_config_name)
    corpus_model_config = corpus_config.model
    if not isinstance(corpus_model_config, _pi0_config.DistilledPi0Config):
        raise ValueError("Golden corpus must be generated from a Shallow-pi distillation config")
    if corpus_model_config.teacher_config != teacher_config_name:
        raise ValueError(
            f"Golden corpus teacher mismatch: expected {teacher_config_name!r}, "
            f"found {corpus_model_config.teacher_config!r}"
        )
    expected_provenance = _json_round_trip(config_provenance(corpus_config))
    if metadata.get("resolved_config") != expected_provenance:
        raise ValueError("Golden corpus resolved-config fingerprint/provenance does not match this source checkout")
    if metadata.get("dataset") != expected_provenance["dataset"]:
        raise ValueError("Golden corpus dataset provenance does not match its resolved config")
    if metadata.get("dataset_revision") != expected_provenance["dataset"]["revision"]:
        raise ValueError("Golden corpus dataset revision does not match its resolved config")
    if metadata.get("data_split_seed") != corpus_config.seed:
        raise ValueError("Golden corpus data-split seed differs from the training seed")
    validate_validation_split_metadata(metadata.get("data_split"), corpus_config)
    if (
        corpus_model_config.action_horizon != model_config.action_horizon
        or corpus_model_config.action_dim != model_config.action_dim
    ):
        raise ValueError("Golden corpus and teacher model action shapes differ")

    try:
        with np.load(corpus_path, allow_pickle=False) as loaded:
            corpus = {key: loaded[key] for key in loaded.files}
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read golden corpus {corpus_path}: {exc}") from exc

    image_names = metadata.get("image_names")
    if (
        not isinstance(image_names, list)
        or not image_names
        or any(not isinstance(name, str) or not name for name in image_names)
        or len(image_names) != len(set(image_names))
    ):
        raise ValueError("Golden corpus image_names must be a non-empty unique string list")
    if set(image_names) != set(_model.IMAGE_KEYS):
        raise ValueError(f"Golden corpus image keys must be exactly {sorted(_model.IMAGE_KEYS)}")
    required_keys = {
        "state",
        "tokenized_prompt",
        "tokenized_prompt_mask",
        "actions",
        "noise",
        "time",
        *(f"image__{name}" for name in image_names),
        *(f"image_mask__{name}" for name in image_names),
    }
    if set(corpus) != required_keys:
        raise ValueError(
            f"Golden corpus array set mismatch: missing={sorted(required_keys - set(corpus))}, "
            f"unexpected={sorted(set(corpus) - required_keys)}"
        )

    samples = metadata.get("samples")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError("Golden corpus samples must be a positive integer")
    expected_action_shape = (samples, model_config.action_horizon, model_config.action_dim)
    if corpus["actions"].shape != expected_action_shape or corpus["noise"].shape != expected_action_shape:
        raise ValueError(f"Golden actions/noise must have shape {expected_action_shape}")
    if (
        metadata.get("action_horizon") != model_config.action_horizon
        or metadata.get("action_dim") != model_config.action_dim
    ):
        raise ValueError("Golden corpus metadata action shape does not match the teacher config")
    if corpus["time"].shape != (samples,):
        raise ValueError(f"Golden time must have shape {(samples,)}")
    if corpus["state"].shape != (samples, model_config.action_dim):
        raise ValueError(f"Golden state must have shape {(samples, model_config.action_dim)}")
    if corpus["tokenized_prompt"].ndim != 2 or corpus["tokenized_prompt"].shape[0] != samples:
        raise ValueError("Golden tokenized_prompt must be a batched rank-two array")
    if corpus["tokenized_prompt_mask"].shape != corpus["tokenized_prompt"].shape:
        raise ValueError("Golden tokenized_prompt_mask shape does not match tokenized_prompt")
    if not np.issubdtype(corpus["tokenized_prompt"].dtype, np.integer):
        raise ValueError("Golden tokenized_prompt must have an integer dtype")
    if corpus["tokenized_prompt_mask"].dtype != np.bool_:
        raise ValueError("Golden tokenized_prompt_mask must have boolean dtype")

    image_layout = metadata.get("image_layout")
    if image_layout not in {"BCHW", "BHWC"}:
        raise ValueError(f"Unsupported golden-corpus image layout: {image_layout!r}")
    for name in image_names:
        image = corpus[f"image__{name}"]
        if image.ndim != 4:
            raise ValueError(f"Golden image {name!r} must be batched rank four, got {image.shape}")
        channel_count = image.shape[1] if image_layout == "BCHW" else image.shape[-1]
        if image.shape[0] != samples or channel_count != 3:
            raise ValueError(
                f"Golden image {name!r} does not match {image_layout} batch/channel contract: {image.shape}"
            )
        if corpus[f"image_mask__{name}"].shape != (samples,):
            raise ValueError(f"Golden image mask {name!r} must have shape {(samples,)}")
        if corpus[f"image_mask__{name}"].dtype != np.bool_:
            raise ValueError(f"Golden image mask {name!r} must have boolean dtype")

    numeric_keys = ["state", "actions", "noise", "time", *(f"image__{name}" for name in image_names)]
    for key in numeric_keys:
        if not np.issubdtype(corpus[key].dtype, np.number) or not np.all(np.isfinite(corpus[key])):
            raise ValueError(f"Golden corpus array {key!r} must be finite and numeric")
    if np.any(corpus["time"] < 0.001) or np.any(corpus["time"] > 1.0):
        raise ValueError("Golden corpus time values must be in [0.001, 1.0]")
    return metadata | {"sha256": actual_sha256}, corpus


def validate_converted_checkpoint(
    checkpoint: pathlib.Path,
    *,
    config_name: str,
    model_config: _pi0_config.Pi0Config,
) -> tuple[pathlib.Path, pathlib.Path, dict[str, Any], str, str]:
    """Validate the self-describing converted tree and return its payload hashes."""
    checkpoint = checkpoint.expanduser().resolve()
    root = checkpoint if checkpoint.is_dir() else checkpoint.parent
    weight_path = root / "model.safetensors" if checkpoint.is_dir() else checkpoint
    config_path = root / "config.json"
    if not weight_path.is_file() or weight_path.stat().st_size == 0:
        raise ValueError(f"Converted model.safetensors is missing or empty: {weight_path}")
    conversion = _read_json_object(config_path, label="converted-checkpoint config")
    expected = {
        "schema_version": 1,
        "config_name": config_name,
        "pi05": True,
        "action_dim": model_config.action_dim,
        "action_horizon": model_config.action_horizon,
        "paligemma_variant": model_config.paligemma_variant,
        "action_expert_variant": model_config.action_expert_variant,
        "precision": "bfloat16",
    }
    for key, value in expected.items():
        if conversion.get(key) != value:
            raise ValueError(
                f"Converted-checkpoint config mismatch for {key!r}: expected {value!r}, found {conversion.get(key)!r}"
            )
    return weight_path, config_path, conversion, sha256_file(weight_path), sha256_file(config_path)


def validate_checkpoint_manifests(
    *,
    source_manifest_path: pathlib.Path,
    converted_manifest_path: pathlib.Path,
    jax_checkpoint: pathlib.Path,
    pytorch_root: pathlib.Path,
    config_name: str,
    model_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    """Bind the numerical result to the two manifests emitted by staging."""
    source_manifest_path = source_manifest_path.expanduser().resolve()
    converted_manifest_path = converted_manifest_path.expanduser().resolve()
    source = _read_json_object(source_manifest_path, label="source-checkpoint manifest")
    converted = _read_json_object(converted_manifest_path, label="converted-checkpoint manifest")
    if source.get("schema_version") != 1 or converted.get("schema_version") != 1:
        raise ValueError("Checkpoint manifests must use schema_version 1")
    source_checkpoint = source.get("checkpoint", {})
    converted_checkpoint = converted.get("checkpoint", {})
    if source_checkpoint.get("local_dirname") != jax_checkpoint.name:
        raise ValueError("Source-checkpoint manifest does not describe the selected JAX checkpoint")
    if converted_checkpoint.get("local_dirname") != pytorch_root.name:
        raise ValueError("Converted-checkpoint manifest does not describe the selected PyTorch checkpoint")
    if source_checkpoint.get("key") != converted_checkpoint.get("key"):
        raise ValueError("Source and converted checkpoint manifests identify different teachers")
    conversion = converted.get("conversion", {})
    if conversion.get("config_name") != config_name or conversion.get("precision") != "bfloat16":
        raise ValueError("Converted-checkpoint manifest has the wrong config or precision")
    source_revision = source.get("source", {}).get("revision")
    if converted.get("source", {}).get("upstream", {}).get("revision") != source_revision:
        raise ValueError("Converted-checkpoint manifest is not bound to the selected source manifest")
    converted_files = {
        item.get("path"): item
        for item in converted.get("files", [])
        if isinstance(item, Mapping) and isinstance(item.get("path"), str)
    }
    if converted_files.get("model.safetensors", {}).get("sha256") != model_sha256:
        raise ValueError("Converted model bytes do not match the converted-checkpoint manifest")
    if converted_files.get("config.json", {}).get("sha256") != config_sha256:
        raise ValueError("Converted config bytes do not match the converted-checkpoint manifest")
    return {
        "source": {
            "path": str(source_manifest_path),
            "sha256": sha256_file(source_manifest_path),
            "revision": source_revision,
        },
        "converted": {
            "path": str(converted_manifest_path),
            "sha256": sha256_file(converted_manifest_path),
            "revision": converted.get("source", {}).get("revision"),
            "source_commit": conversion.get("source_commit"),
            "image_digest": conversion.get("image_digest"),
        },
    }


def observation_from_corpus(corpus, image_names: list[str], convert, *, image_layout: str, expects_bhwc: bool):
    if image_layout not in {"BCHW", "BHWC"}:
        raise ValueError(f"Unsupported golden-corpus image layout: {image_layout!r}")

    def image_for_framework(value):
        image = np.asarray(value)
        if image.ndim != 4:
            raise ValueError(f"Golden image must be batched rank four, got {image.shape}")
        if expects_bhwc and image_layout == "BCHW":
            image = np.transpose(image, (0, 2, 3, 1))
        elif not expects_bhwc and image_layout == "BHWC":
            image = np.transpose(image, (0, 3, 1, 2))
        return convert(np.ascontiguousarray(image))

    return _model.Observation(
        images={name: image_for_framework(corpus[f"image__{name}"]) for name in image_names},
        image_masks={name: convert(corpus[f"image_mask__{name}"]) for name in image_names},
        state=convert(corpus["state"]),
        tokenized_prompt=convert(corpus["tokenized_prompt"]),
        tokenized_prompt_mask=convert(corpus["tokenized_prompt_mask"]),
        token_ar_mask=None,
        token_loss_mask=None,
    )


def velocity_report(pytorch_velocity: np.ndarray, jax_velocity: np.ndarray) -> dict:
    """Summarize equivalence and require every sample to pass the cosine gate."""
    if pytorch_velocity.shape != jax_velocity.shape or pytorch_velocity.ndim != 3:
        raise ValueError("JAX and PyTorch velocities must share a [samples, horizon, joints] shape")
    if not np.all(np.isfinite(pytorch_velocity)) or not np.all(np.isfinite(jax_velocity)):
        raise ValueError("JAX and PyTorch velocities must be finite")
    cosine = cosine_similarity(pytorch_velocity, jax_velocity)
    absolute = np.abs(pytorch_velocity.astype(np.float64) - jax_velocity.astype(np.float64))
    return {
        "cosine_mean": float(np.mean(cosine)),
        "cosine_min": float(np.min(cosine)),
        "mse": float(np.mean(np.square(absolute))),
        "max_absolute_error": float(np.max(absolute)),
        "gate_cosine_minimum": GATE_COSINE_MINIMUM,
        "gate_pass": bool(np.min(cosine) >= GATE_COSINE_MINIMUM),
    }


def compare(
    *,
    config_name: str,
    jax_checkpoint: pathlib.Path,
    pytorch_checkpoint: pathlib.Path,
    source_manifest: pathlib.Path,
    converted_manifest: pathlib.Path,
    corpus_path: pathlib.Path,
    device: str,
) -> tuple[dict, dict[str, np.ndarray]]:
    train_config = _config.get_config(config_name)
    model_config = train_config.model
    if not isinstance(model_config, _pi0_config.Pi0Config) or isinstance(model_config, _pi0_config.DistilledPi0Config):
        raise ValueError("framework comparison requires a full-depth Pi0Config")
    if not model_config.pi05:
        raise ValueError("framework comparison requires a pi0.5 teacher config")

    jax_checkpoint = jax_checkpoint.expanduser().resolve()
    params_path = jax_checkpoint / "params"
    if not jax_checkpoint.is_dir() or not params_path.is_dir():
        raise ValueError(f"JAX checkpoint must contain a params directory: {jax_checkpoint}")
    metadata, corpus = load_golden_corpus(
        corpus_path,
        teacher_config_name=config_name,
        model_config=model_config,
    )
    weight_path, conversion_config_path, conversion_config, model_sha256, config_sha256 = validate_converted_checkpoint(
        pytorch_checkpoint,
        config_name=config_name,
        model_config=model_config,
    )
    manifest_provenance = validate_checkpoint_manifests(
        source_manifest_path=source_manifest,
        converted_manifest_path=converted_manifest,
        jax_checkpoint=jax_checkpoint,
        pytorch_root=weight_path.parent,
        config_name=config_name,
        model_sha256=model_sha256,
        config_sha256=config_sha256,
    )

    actions_np = corpus["actions"].astype(np.float32)
    noise_np = corpus["noise"].astype(np.float32)
    time_np = corpus["time"].astype(np.float32)

    params = _model.restore_params(params_path)
    jax_model = model_config.load(params)
    image_layout = metadata["image_layout"]
    jax_observation = observation_from_corpus(
        corpus,
        metadata["image_names"],
        jnp.asarray,
        image_layout=image_layout,
        expects_bhwc=True,
    )
    jax_observation = _model.preprocess_observation(None, jax_observation, train=False)
    jax_velocity = np.asarray(
        jax_model.predict_velocity(
            jax_observation,
            jnp.asarray(actions_np),
            jnp.asarray(noise_np),
            jnp.asarray(time_np),
            observation_is_preprocessed=True,
        )
    ).astype(np.float32)

    del params, jax_model, jax_observation
    gc.collect()
    jax.clear_caches()
    gc.collect()

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"Requested CUDA comparison device is unavailable: {device}")
    comparison_model_config = dataclasses.replace(model_config, pytorch_compile_mode=None)
    pytorch_model = _pi0_pytorch.PI0Pytorch(comparison_model_config).to(torch_device).eval()
    safetensors.torch.load_model(pytorch_model, weight_path, device=str(torch_device))
    to_torch = lambda value: torch.from_numpy(np.asarray(value)).to(torch_device)  # noqa: E731
    pytorch_observation = observation_from_corpus(
        corpus,
        metadata["image_names"],
        to_torch,
        image_layout=image_layout,
        expects_bhwc=False,
    )
    pytorch_observation = pytorch_model.preprocess_observation(pytorch_observation, train=False)
    with torch.inference_mode():
        pytorch_velocity = (
            pytorch_model.predict_velocity(
                pytorch_observation,
                to_torch(actions_np),
                to_torch(noise_np),
                to_torch(time_np),
                observation_is_preprocessed=True,
            )
            .float()
            .cpu()
            .numpy()
        )

    report = {
        "schema_version": 2,
        "config_name": config_name,
        "samples": int(actions_np.shape[0]),
        "provenance": {
            "golden_corpus": {
                "path": str(corpus_path.expanduser().resolve()),
                "sha256": metadata["sha256"],
                "sidecar_path": str(corpus_path.expanduser().resolve().with_suffix(".json")),
                "sidecar_sha256": sha256_file(corpus_path.expanduser().resolve().with_suffix(".json")),
                "run_id": metadata["run_id"],
                "config_name": metadata["config_name"],
                "config_fingerprint_sha256": metadata["resolved_config"]["fingerprint_sha256"],
                "dataset": metadata["dataset"],
                "seed": metadata["seed"],
                "data_split_seed": metadata["data_split_seed"],
                "data_split": metadata["data_split"],
            },
            "jax_checkpoint": {
                "path": str(jax_checkpoint),
                "manifest": manifest_provenance["source"],
            },
            "pytorch_checkpoint": {
                "path": str(weight_path.parent),
                "model_sha256": model_sha256,
                "config_path": str(conversion_config_path),
                "config_sha256": config_sha256,
                "config": conversion_config,
                "manifest": manifest_provenance["converted"],
            },
        },
        **velocity_report(pytorch_velocity, jax_velocity),
    }
    return report, {"jax": jax_velocity, "pytorch": pytorch_velocity}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--jax-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--pytorch-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--source-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--converted-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--corpus", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def require_new_output_paths(output: pathlib.Path) -> pathlib.Path:
    """Refuse to replace either half of the report/velocity evidence pair."""
    velocity_path = output.with_suffix(".npz")
    collisions = [path for path in (output, velocity_path) if path.exists() or path.is_symlink()]
    if collisions:
        rendered = ", ".join(str(path) for path in collisions)
        raise FileExistsError(f"framework-equivalence output already exists: {rendered}")
    return velocity_path


def main() -> int:
    args = parse_args()
    # Check before loading either full-depth model. This makes a retry preserve
    # the original diagnostic evidence instead of spending GPU time and then
    # silently replacing it.
    velocity_path = require_new_output_paths(args.output)
    report, velocities = compare(
        config_name=args.config_name,
        jax_checkpoint=args.jax_checkpoint,
        pytorch_checkpoint=args.pytorch_checkpoint,
        source_manifest=args.source_manifest,
        converted_manifest=args.converted_manifest,
        corpus_path=args.corpus,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also closes the race between the early guard and the
    # final writes. If publication is interrupted after the velocity archive,
    # that file remains as explicit partial evidence and blocks a blind retry.
    with velocity_path.open("xb") as stream:
        np.savez_compressed(stream, **velocities)
    report["velocities"] = {"path": str(velocity_path.resolve()), "sha256": sha256_file(velocity_path)}
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
