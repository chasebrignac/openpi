#!/usr/bin/env python3
"""Create and validate the fixed 1,024-sample pi0.5 calibration corpus.

This command only reads a locally staged dataset and writes local files.  It
does not call AWS or upload artifacts.  Run it in the track's pinned LeRobot
container, then copy the validated directory to versioned S3 with the normal
artifact workflow.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable
import dataclasses
import hashlib
import json
import math
import pathlib
from typing import Any

import numpy as np

CALIBRATION_SAMPLES = 1_024
IMAGE_COUNT = 3


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_image_batch(value: Any, *, name: str) -> np.ndarray:
    image = _as_numpy(value)
    if image.ndim != 4:
        raise ValueError(f"{name} must be a batched image tensor, got {image.shape}")
    if image.shape[1] == 3:
        pass
    elif image.shape[-1] == 3:
        image = np.transpose(image, (0, 3, 1, 2))
    else:
        raise ValueError(f"{name} must be BCHW or BHWC RGB, got {image.shape}")
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0 * 2.0 - 1.0
    else:
        image = image.astype(np.float32, copy=False)
    if not np.isfinite(image).all():
        raise ValueError(f"{name} contains NaN or Inf")
    if image.size and (float(image.min()) < -1.001 or float(image.max()) > 1.001):
        raise ValueError(f"{name} is outside the model's normalized [-1, 1] image range")
    return np.ascontiguousarray(image)


def _prompt_digest(tokens: np.ndarray, mask: np.ndarray) -> str:
    tokens = np.asarray(tokens)
    mask = np.asarray(mask, dtype=np.bool_)
    if tokens.ndim != 1 or mask.shape != tokens.shape:
        raise ValueError(f"Prompt tokens and mask must be matching vectors, got {tokens.shape}/{mask.shape}")
    canonical = np.asarray(tokens[mask], dtype="<i8").tobytes()
    return hashlib.sha256(canonical).hexdigest()


def _sample_noise(shape: tuple[int, ...], *, seed: int, ordinal: int) -> np.ndarray:
    # A per-sample SeedSequence makes the corpus invariant to loader/chunk size.
    rng = np.random.default_rng(np.random.SeedSequence([seed, ordinal]))
    return rng.standard_normal(shape, dtype=np.float32)


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    """Choose one deterministic midpoint from each of ``count`` dataset bins."""

    if count < 1:
        raise ValueError("count must be positive")
    if total < count:
        raise ValueError(f"Dataset has {total} samples; cannot select {count} unique calibration samples")
    starts = np.arange(count, dtype=np.int64) * total // count
    ends = (np.arange(1, count + 1, dtype=np.int64) * total // count) - 1
    indices = ((starts + ends) // 2).tolist()
    if len(set(indices)) != count:
        raise RuntimeError("Evenly spaced calibration selection unexpectedly produced duplicate indices")
    return indices


def calibration_batch(
    observation: Any,
    actions: Any,
    *,
    seed: int,
    start_ordinal: int,
) -> tuple[dict[str, np.ndarray], list[str], list[str], list[str]]:
    """Convert one configured-loader batch to the export calibration schema."""

    state = _as_numpy(observation.state).astype(np.float32, copy=False)
    action_array = _as_numpy(actions).astype(np.float32, copy=False)
    tokens = _as_numpy(observation.tokenized_prompt).astype(np.int64, copy=False)
    token_mask = _as_numpy(observation.tokenized_prompt_mask).astype(np.bool_, copy=False)
    if state.ndim != 2 or action_array.ndim != 3 or tokens.ndim != 2 or token_mask.shape != tokens.shape:
        raise ValueError(
            "Expected state [B,S], actions [B,H,D], and matching prompt tensors [B,L]; "
            f"got {state.shape}, {action_array.shape}, {tokens.shape}, {token_mask.shape}"
        )
    batch_size = state.shape[0]
    if action_array.shape[0] != batch_size or tokens.shape[0] != batch_size:
        raise ValueError("Calibration batch fields have different batch dimensions")
    if not np.isfinite(state).all():
        raise ValueError("state contains NaN or Inf")
    if not np.isfinite(action_array).all():
        raise ValueError("actions contain NaN or Inf")

    image_names = list(observation.images)
    if len(image_names) != IMAGE_COUNT or len(set(image_names)) != IMAGE_COUNT:
        raise ValueError(f"pi0.5 calibration requires exactly {IMAGE_COUNT} uniquely named cameras: {image_names}")
    if set(observation.image_masks) != set(image_names):
        raise ValueError("Image and image-mask camera names differ")

    arrays: dict[str, np.ndarray] = {}
    for index, image_name in enumerate(image_names):
        image = _canonical_image_batch(observation.images[image_name], name=image_name)
        mask = _as_numpy(observation.image_masks[image_name]).astype(np.bool_, copy=False)
        if image.shape[0] != batch_size or mask.shape != (batch_size,):
            raise ValueError(
                f"Camera {image_name} has inconsistent batch shapes: image={image.shape}, mask={mask.shape}, "
                f"expected batch={batch_size}"
            )
        arrays[f"image_{index}"] = image
        arrays[f"image_mask_{index}"] = np.ascontiguousarray(mask)

    prompt_digests = [_prompt_digest(tokens[index], token_mask[index]) for index in range(batch_size)]
    strata = [f"prompt:{digest[:16]}" for digest in prompt_digests]
    noise = np.stack(
        [
            _sample_noise(tuple(action_array.shape[1:]), seed=seed, ordinal=start_ordinal + index)
            for index in range(batch_size)
        ],
        axis=0,
    )
    arrays.update(
        {
            "lang_tokens": np.ascontiguousarray(tokens),
            "lang_mask": np.ascontiguousarray(token_mask),
            "state": np.ascontiguousarray(state),
            # Retain the configured loader's normalized/internal action targets.
            # Export does not feed these to either graph, but the separately
            # sealed action-envelope artifact derives its bounds from these
            # exact calibration records instead of an invented static range.
            "actions": np.ascontiguousarray(action_array),
            "noise": noise,
        }
    )
    return arrays, image_names, strata, prompt_digests


def _slice_batch(arrays: dict[str, np.ndarray], count: int) -> dict[str, np.ndarray]:
    return {name: np.ascontiguousarray(value[:count]) for name, value in arrays.items()}


def _write_manifest(path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    path.write_text(payload)


def write_corpus(
    batches: Iterable[tuple[Any, Any]],
    *,
    output_dir: pathlib.Path,
    seed: int,
    config_name: str,
    dataset: str,
    dataset_revision: str,
    expected_samples: int = CALIBRATION_SAMPLES,
    source_indices: list[int] | None = None,
) -> pathlib.Path:
    """Write loader batches and return a fully validated manifest path."""

    if expected_samples < 1:
        raise ValueError("expected_samples must be positive")
    if source_indices is not None and (
        len(source_indices) != expected_samples or len(set(source_indices)) != expected_samples
    ):
        raise ValueError("source_indices must contain exactly one unique dataset index per calibration sample")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty calibration directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    corpus_image_names: list[str] | None = None
    ordinal = 0
    for chunk_index, (observation, actions) in enumerate(batches):
        if ordinal >= expected_samples:
            break
        arrays, image_names, strata, prompt_digests = calibration_batch(
            observation,
            actions,
            seed=seed,
            start_ordinal=ordinal,
        )
        if corpus_image_names is None:
            corpus_image_names = image_names
        elif image_names != corpus_image_names:
            raise ValueError(f"Camera order changed between chunks: {corpus_image_names} -> {image_names}")

        count = min(expected_samples - ordinal, arrays["state"].shape[0])
        if count == 0:
            raise ValueError("Configured loader emitted an empty batch")
        arrays = _slice_batch(arrays, count)
        chunk_path = output_dir / f"chunk-{chunk_index:05d}.npz"
        np.savez_compressed(chunk_path, **arrays)
        chunk_sha256 = _sha256_file(chunk_path)
        for index in range(count):
            sample_ordinal = ordinal + index
            record = {
                "path": chunk_path.name,
                "index": index,
                "stratum": strata[index],
                "sample_ordinal": sample_ordinal,
                "prompt_sha256": prompt_digests[index],
                "noise_seed": seed,
                "chunk_sha256": chunk_sha256,
                "image_names": image_names,
                "config_name": config_name,
                "dataset": dataset,
                "dataset_revision": dataset_revision,
            }
            if source_indices is not None:
                record["dataset_index"] = source_indices[sample_ordinal]
            records.append(record)
        ordinal += count

    if ordinal != expected_samples:
        raise RuntimeError(f"Loader ended after {ordinal} samples; exactly {expected_samples} are required")
    manifest_path = output_dir / "manifest.jsonl"
    _write_manifest(manifest_path, records)
    summary = validate_corpus(manifest_path, expected_samples=expected_samples)
    summary.update(
        {
            "config_name": config_name,
            "dataset": dataset,
            "dataset_revision": dataset_revision,
            "seed": seed,
            "image_names": corpus_image_names,
            "manifest": manifest_path.name,
            "manifest_sha256": _sha256_file(manifest_path),
        }
    )
    (output_dir / "corpus.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return manifest_path


def _validate_chunk(records, *, image_count: int) -> None:
    # Keep the CLI's lightweight helpers importable without loading the full
    # model/runtime dependency graph (useful for manifest-only validation).
    from openpi.exporting.calibration import load_record_arrays

    path = records[0].path
    digest = _sha256_file(path)
    indices = sorted(record.index for record in records)
    if indices != list(range(len(records))):
        raise ValueError(f"Manifest indices for {path} are not one contiguous complete batch: {indices}")
    if any(record.metadata.get("chunk_sha256") != digest for record in records):
        raise ValueError(f"Manifest chunk hash does not match {path}")

    # Exercise the exact reader used by export/quantization, then validate the
    # whole chunk once rather than decompressing it separately for every record.
    load_record_arrays(records[0], image_count=image_count)
    required = {
        *(f"image_{index}" for index in range(image_count)),
        *(f"image_mask_{index}" for index in range(image_count)),
        "lang_tokens",
        "lang_mask",
        "state",
        "actions",
        "noise",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"Calibration NPZ {path} is missing keys: {missing}")
        batch_size = len(records)
        expected_ranks = {
            **{f"image_{index}": 4 for index in range(image_count)},
            **{f"image_mask_{index}": 1 for index in range(image_count)},
            "lang_tokens": 2,
            "lang_mask": 2,
            "state": 2,
            "actions": 3,
            "noise": 3,
        }
        for name, rank in expected_ranks.items():
            value = archive[name]
            if value.ndim != rank or value.shape[0] != batch_size:
                raise ValueError(f"{path}:{name} has invalid shape {value.shape}; batch={batch_size}, rank={rank}")
        for index in range(image_count):
            image = archive[f"image_{index}"]
            if image.dtype != np.float32 or image.shape[1] != 3 or not np.isfinite(image).all():
                raise ValueError(f"{path}:image_{index} is not finite BCHW float32 RGB")
        if archive["state"].dtype != np.float32 or not np.isfinite(archive["state"]).all():
            raise ValueError(f"{path}:state is not finite float32")
        if archive["actions"].dtype != np.float32 or not np.isfinite(archive["actions"]).all():
            raise ValueError(f"{path}:actions is not finite float32")
        if archive["noise"].dtype != np.float32 or not np.isfinite(archive["noise"]).all():
            raise ValueError(f"{path}:noise is not finite float32")
        if archive["actions"].shape != archive["noise"].shape:
            raise ValueError(
                f"{path}:actions/noise shapes differ: {archive['actions'].shape} != {archive['noise'].shape}"
            )

        by_index = {record.index: record for record in records}
        for index in range(batch_size):
            record = by_index[index]
            prompt_digest = _prompt_digest(archive["lang_tokens"][index], archive["lang_mask"][index])
            if record.stratum != f"prompt:{prompt_digest[:16]}":
                raise ValueError(f"Manifest stratum does not match prompt tokens for {path} index {index}")
            if record.metadata.get("prompt_sha256") != prompt_digest:
                raise ValueError(f"Manifest prompt hash does not match {path} index {index}")
            ordinal = int(record.metadata.get("sample_ordinal", -1))
            seed = int(record.metadata.get("noise_seed", -1))
            expected_noise = _sample_noise(tuple(archive["noise"].shape[1:]), seed=seed, ordinal=ordinal)
            np.testing.assert_array_equal(archive["noise"][index], expected_noise)


def validate_corpus(
    manifest_path: pathlib.Path,
    *,
    expected_samples: int = CALIBRATION_SAMPLES,
    image_count: int = IMAGE_COUNT,
) -> dict[str, Any]:
    """Apply the final exact-count, uniqueness, schema, and strata gate."""

    from openpi.exporting.calibration import load_calibration_manifest
    from openpi.exporting.calibration import select_stratified
    from openpi.exporting.calibration import stratum_counts

    records = load_calibration_manifest(manifest_path)
    if len(records) != expected_samples:
        raise ValueError(f"Calibration corpus has {len(records)} records; exactly {expected_samples} are required")
    # This also rejects fewer than two non-empty strata.
    select_stratified(records, expected_samples)

    ordinals = [int(record.metadata.get("sample_ordinal", -1)) for record in records]
    if sorted(ordinals) != list(range(expected_samples)):
        raise ValueError("Calibration sample ordinals must be unique and exactly cover 0..N-1")
    dataset_indices = [record.metadata.get("dataset_index") for record in records]
    if any(index is not None for index in dataset_indices) and (
        any(index is None for index in dataset_indices)
        or len({int(index) for index in dataset_indices}) != expected_samples
    ):
        raise ValueError("Calibration dataset indices must be present and unique for every record")
    grouped = defaultdict(list)
    for record in records:
        grouped[record.path].append(record)
    for path_records in grouped.values():
        _validate_chunk(path_records, image_count=image_count)

    return {
        "schema_version": 1,
        "sample_count": len(records),
        "unique_record_count": len({(record.path, record.index) for record in records}),
        "chunk_count": len(grouped),
        "stratum_counts": stratum_counts(records),
        "passes": True,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="read a configured local dataset and write the corpus")
    generate.add_argument("--config-name", required=True)
    generate.add_argument("--dataset-revision", required=True)
    generate.add_argument("--output-dir", required=True, type=pathlib.Path)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--chunk-size", type=int, default=32)
    generate.add_argument("--num-workers", type=int, default=0)

    validate = subparsers.add_parser("validate", help="replay the final local corpus gate")
    validate.add_argument("--manifest", required=True, type=pathlib.Path)
    return parser.parse_args()


def _generate(args: argparse.Namespace) -> pathlib.Path:
    # Generation needs the heavy training stack; validation above does not.
    import openpi.training.config as training_config
    import openpi.training.data_loader as data_loader

    if args.chunk_size < 1 or args.chunk_size > CALIBRATION_SAMPLES or CALIBRATION_SAMPLES % args.chunk_size != 0:
        raise ValueError(f"chunk-size must be a positive divisor of {CALIBRATION_SAMPLES}")
    config = dataclasses.replace(
        training_config.get_config(args.config_name),
        batch_size=args.chunk_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    configured_data = config.data.create(config.assets_dirs, config.model)
    if configured_data.lerobot_revision != args.dataset_revision:
        raise ValueError(
            f"Dataset revision mismatch: config={configured_data.lerobot_revision!r}, command={args.dataset_revision!r}"
        )
    if configured_data.repo_id is None:
        raise ValueError("Configured calibration dataset has no repository identity")
    dataset = data_loader.create_torch_dataset(configured_data, config.model.action_horizon, config.model)
    source_indices = evenly_spaced_indices(len(dataset), CALIBRATION_SAMPLES)

    import torch

    dataset = torch.utils.data.Subset(dataset, source_indices)
    dataset = data_loader.transform_dataset(dataset, configured_data)
    loader = data_loader.DataLoaderImpl(
        configured_data,
        data_loader.TorchDataLoader(
            dataset,
            local_batch_size=args.chunk_size,
            shuffle=False,
            num_batches=math.ceil(CALIBRATION_SAMPLES / args.chunk_size),
            num_workers=args.num_workers,
            seed=args.seed,
            framework="pytorch",
        ),
    )
    return write_corpus(
        loader,
        output_dir=args.output_dir,
        seed=args.seed,
        config_name=args.config_name,
        dataset=configured_data.repo_id,
        dataset_revision=args.dataset_revision,
        source_indices=source_indices,
    )


def main() -> int:
    args = _parse_args()
    if args.command == "generate":
        manifest = _generate(args)
        summary = json.loads((manifest.parent / "corpus.json").read_text())
    else:
        manifest = args.manifest.resolve()
        summary = validate_corpus(manifest)
    print(json.dumps({"manifest": str(manifest), **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
