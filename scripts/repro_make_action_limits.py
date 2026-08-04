#!/usr/bin/env python3
"""Seal corpus-derived action envelopes for ONNX numerical validation.

The emitted limits are an empirical regression gate.  They are deliberately
not described as robot joint limits: neither LIBERO nor DROID data contains a
certified hardware-safety specification.  DROID joint-position outputs are
also state-dependent, so this command leaves their static physical bounds
unset and records that rollout-time environment checks remain required.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import pathlib
from typing import Any

import numpy as np

CALIBRATION_SAMPLES = 1_024
DEFAULT_INTERNAL_MARGIN = 0.01
TRACK_SPECS = {
    "libero": {
        "active_dim": 7,
        "physical_representation": "delta end-effector command plus gripper command",
        "state_dependent_dimensions": [],
    },
    "droid": {
        "active_dim": 8,
        "physical_representation": "absolute 7-DoF joint-position target plus absolute gripper command",
        "state_dependent_dimensions": list(range(7)),
    },
}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def artifact_record(path: pathlib.Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _read_npz(path: pathlib.Path, *keys: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(keys).difference(archive.files))
        if missing:
            raise KeyError(f"{path} is missing arrays: {missing}")
        result = {key: np.asarray(archive[key], dtype=np.float32) for key in keys}
    for key, value in result.items():
        if not np.isfinite(value).all():
            raise ValueError(f"{path}:{key} contains NaN or Inf")
    return result


def _selected_calibration_arrays(
    manifest_path: pathlib.Path,
    *,
    expected_samples: int = CALIBRATION_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    # Reuse the exact corpus and manifest gates used by export.
    from openpi.exporting.calibration import load_calibration_manifest

    if __package__:
        from scripts.repro_make_calibration import validate_corpus
    else:
        from repro_make_calibration import validate_corpus

    validate_corpus(manifest_path, expected_samples=expected_samples)
    records = load_calibration_manifest(manifest_path)
    identities = {
        (
            record.metadata.get("config_name"),
            record.metadata.get("dataset"),
            record.metadata.get("dataset_revision"),
        )
        for record in records
    }
    if len(identities) != 1:
        raise ValueError(f"Calibration records have mixed config/dataset identities: {identities!r}")
    config_name, dataset, dataset_revision = identities.pop()
    if not all(isinstance(value, str) and value for value in (config_name, dataset, dataset_revision)):
        raise ValueError("Calibration records do not contain a complete config/dataset identity")

    grouped: dict[pathlib.Path, list[Any]] = defaultdict(list)
    for record in records:
        grouped[record.path].append(record)
    action_samples: list[np.ndarray] = []
    state_samples: list[np.ndarray] = []
    chunk_sources: list[dict[str, Any]] = []
    for path in sorted(grouped, key=str):
        path_records = sorted(grouped[path], key=lambda item: item.index)
        arrays = _read_npz(path, "actions", "state")
        actions = arrays["actions"]
        states = arrays["state"]
        if actions.ndim != 3 or states.ndim != 2 or actions.shape[0] != states.shape[0]:
            raise ValueError(f"Calibration action/state shapes are invalid in {path}: {actions.shape}/{states.shape}")
        for record in path_records:
            if record.index >= actions.shape[0]:
                raise IndexError(f"Calibration index {record.index} is outside {path} batch {actions.shape[0]}")
            action_samples.append(actions[record.index])
            state_samples.append(states[record.index])
        chunk_sources.append(artifact_record(path))
    return (
        np.stack(action_samples),
        np.stack(state_samples),
        {"config_name": config_name, "dataset": dataset, "dataset_revision": dataset_revision},
        chunk_sources,
    )


def _golden_arrays(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any], pathlib.Path]:
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    actual_hash = sha256_file(path)
    if metadata.get("sha256") != actual_hash:
        raise ValueError(f"Golden corpus hash mismatch: metadata={metadata.get('sha256')!r}, actual={actual_hash}")
    dataset = metadata.get("dataset")
    if not isinstance(dataset, dict) or not dataset.get("repo_id") or not dataset.get("revision"):
        raise ValueError("Golden corpus metadata has no pinned dataset identity")
    arrays = _read_npz(path, "actions", "state")
    return arrays["actions"], arrays["state"], metadata, metadata_path


def _verified_export_arrays(
    artifact_dir: pathlib.Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], pathlib.Path, pathlib.Path, pathlib.Path]:
    manifest_path = artifact_dir / "export-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    actions_path = (artifact_dir / "decode-reference.npz").resolve()
    inputs_path = (artifact_dir / "decode-inputs.npz").resolve()
    by_path = {pathlib.Path(record["path"]).resolve(): record for record in manifest.get("artifacts", [])}
    for path in (actions_path, inputs_path):
        record = by_path.get(path)
        if record is None:
            raise ValueError(f"Export manifest does not seal {path}")
        actual = artifact_record(path)
        if record.get("bytes") != actual["bytes"] or record.get("sha256") != actual["sha256"]:
            raise ValueError(f"Export artifact changed after manifest creation: {path}")
    actions = _read_npz(actions_path, "actions")["actions"]
    state = _read_npz(inputs_path, "state")["state"]
    return actions, state, manifest, manifest_path, actions_path, inputs_path


def _norm_entry(payload: dict[str, Any], key: str) -> dict[str, np.ndarray]:
    raw = payload.get("norm_stats", payload).get(key)
    if not isinstance(raw, dict):
        raise ValueError(f"Normalization stats have no {key!r} entry")
    result: dict[str, np.ndarray] = {}
    for field in ("mean", "std", "q01", "q99"):
        if raw.get(field) is not None:
            value = np.asarray(raw[field], dtype=np.float64).reshape(-1)
            if not np.isfinite(value).all():
                raise ValueError(f"Normalization {key}.{field} contains NaN or Inf")
            result[field] = value
    return result


def _unnormalize(values: np.ndarray, stats: dict[str, np.ndarray], *, normalization: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    dim = values.shape[-1]
    if normalization == "quantile":
        if "q01" not in stats or "q99" not in stats:
            raise ValueError("Quantile unnormalization requires q01 and q99")
        low, high = stats["q01"][:dim], stats["q99"][:dim]
        if low.shape[0] != dim or high.shape[0] != dim or np.any(low >= high):
            raise ValueError(f"Quantile stats cannot unnormalize {dim} dimensions")
        return (values + 1.0) / 2.0 * (high - low + 1e-6) + low
    if normalization == "zscore":
        if "mean" not in stats or "std" not in stats:
            raise ValueError("Z-score unnormalization requires mean and std")
        mean, std = stats["mean"][:dim], stats["std"][:dim]
        if mean.shape[0] != dim or std.shape[0] != dim or np.any(std < 0):
            raise ValueError(f"Z-score stats cannot unnormalize {dim} dimensions")
        return values * (std + 1e-6) + mean
    raise ValueError(f"Unsupported normalization: {normalization}")


def _outward_envelope(values: np.ndarray, *, margin: float) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("internal margin must be finite and non-negative")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 2 or not np.isfinite(values).all():
        raise ValueError("Envelope values must be a finite [..., actions] array")
    low = np.min(values, axis=tuple(range(values.ndim - 1))) - margin
    high = np.max(values, axis=tuple(range(values.ndim - 1))) + margin
    # With a zero configured margin, move one representable float outward so a
    # constant source dimension still has a valid closed interval.
    low = np.nextafter(low.astype(np.float32), np.float32(-np.inf))
    high = np.nextafter(high.astype(np.float32), np.float32(np.inf))
    return low, high


def derive_envelopes(
    *,
    track: str,
    action_sources: list[np.ndarray],
    state_sources: list[np.ndarray],
    action_stats: dict[str, np.ndarray],
    state_stats: dict[str, np.ndarray],
    normalization: str,
    internal_margin: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Derive internal and state-aware physical corpus envelopes."""

    if track not in TRACK_SPECS:
        raise ValueError(f"Unsupported track: {track}")
    if len(action_sources) != len(state_sources) or not action_sources:
        raise ValueError("Every non-empty action source must have a matching state source")
    action_dim = action_sources[0].shape[-1]
    horizon = action_sources[0].shape[-2]
    active_dim = int(TRACK_SPECS[track]["active_dim"])
    if action_dim < active_dim:
        raise ValueError(f"{track} requires {active_dim} active action dimensions, got {action_dim}")
    normalized_actions: list[np.ndarray] = []
    physical_actions: list[np.ndarray] = []
    for action_source, state_source in zip(action_sources, state_sources, strict=True):
        actions = np.asarray(action_source, dtype=np.float32)
        states = np.asarray(state_source, dtype=np.float32)
        if actions.ndim != 3 or states.ndim != 2 or actions.shape[0] != states.shape[0]:
            raise ValueError(f"Action/state source shapes are invalid: {actions.shape}/{states.shape}")
        if actions.shape[1:] != (horizon, action_dim):
            raise ValueError("Action sources do not share one fixed graph shape")
        if states.shape[-1] < 7 or not np.isfinite(actions).all() or not np.isfinite(states).all():
            raise ValueError("Action/state sources must be finite and contain the robot state dimensions")
        active_actions = actions[..., :active_dim]
        normalized_actions.append(active_actions)
        physical = _unnormalize(active_actions, action_stats, normalization=normalization)
        if track == "droid":
            physical_state = _unnormalize(states[..., :7], state_stats, normalization=normalization)
            physical = physical.copy()
            physical[..., :7] += physical_state[:, None, :]
        physical_actions.append(physical)

    combined_normalized = np.concatenate(normalized_actions, axis=0)
    active_low, active_high = _outward_envelope(combined_normalized, margin=internal_margin)
    action_low = np.zeros(action_dim, dtype=np.float32)
    action_high = np.zeros(action_dim, dtype=np.float32)
    action_mask = np.zeros(action_dim, dtype=np.bool_)
    action_low[:active_dim] = active_low
    action_high[:active_dim] = active_high
    action_mask[:active_dim] = True

    combined_physical = np.concatenate(physical_actions, axis=0)
    observed_low, observed_high = _outward_envelope(combined_physical, margin=0.0)
    physical_low = np.full(active_dim, np.nan, dtype=np.float32)
    physical_high = np.full(active_dim, np.nan, dtype=np.float32)
    physical_mask = np.ones(active_dim, dtype=np.bool_)
    state_dependent = np.zeros(active_dim, dtype=np.bool_)
    if track == "droid":
        state_dependent[:7] = True
        physical_mask[:7] = False
    physical_low[physical_mask] = observed_low[physical_mask]
    physical_high[physical_mask] = observed_high[physical_mask]

    arrays = {
        "action_low": action_low,
        "action_high": action_high,
        "action_mask": action_mask,
        "physical_low": physical_low,
        "physical_high": physical_high,
        "physical_mask": physical_mask,
        "physical_state_dependent_mask": state_dependent,
    }
    description = {
        "active_action_dimensions": active_dim,
        "model_action_dimensions": action_dim,
        "action_horizon": horizon,
        "internal_margin": internal_margin,
        "internal_units": "normalized model graph boundary",
        "physical_representation": TRACK_SPECS[track]["physical_representation"],
        "physical_bound_dimensions": np.flatnonzero(physical_mask).tolist(),
        "physical_state_dependent_dimensions": np.flatnonzero(state_dependent).tolist(),
    }
    return arrays, description


def write_artifact(
    output: pathlib.Path,
    *,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> pathlib.Path:
    sidecar = output.with_suffix(".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"Refusing to overwrite action-envelope artifact: {output} / {sidecar}")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_bytes = np.frombuffer(_canonical_json(metadata).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(output, **arrays, metadata_json=metadata_bytes)
    sidecar.write_text(json.dumps({**metadata, "artifact": artifact_record(output)}, indent=2, sort_keys=True) + "\n")
    return sidecar


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", required=True, choices=tuple(TRACK_SPECS))
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--calibration-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--golden-corpus", required=True, type=pathlib.Path)
    parser.add_argument("--artifact-dir", required=True, type=pathlib.Path)
    parser.add_argument("--norm-stats-json", required=True, type=pathlib.Path)
    parser.add_argument("--normalization", choices=("quantile", "zscore"), default="quantile")
    parser.add_argument("--internal-margin", type=float, default=DEFAULT_INTERNAL_MARGIN)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    calibration_actions, calibration_states, identity, chunk_sources = _selected_calibration_arrays(
        args.calibration_manifest.resolve()
    )
    if identity["config_name"] != args.config_name:
        raise ValueError(f"Calibration config mismatch: {identity['config_name']!r} != requested {args.config_name!r}")
    golden_actions, golden_states, golden_metadata, golden_metadata_path = _golden_arrays(args.golden_corpus.resolve())
    golden_dataset = golden_metadata["dataset"]
    if golden_dataset["repo_id"] != identity["dataset"] or golden_dataset["revision"] != identity["dataset_revision"]:
        raise ValueError("Golden and calibration dataset identities differ")

    reference_actions, reference_state, export_manifest, export_manifest_path, reference_path, inputs_path = (
        _verified_export_arrays(args.artifact_dir.resolve())
    )
    if export_manifest.get("track") != args.track:
        raise ValueError(f"Export track mismatch: {export_manifest.get('track')!r} != {args.track!r}")
    export_dataset = export_manifest.get("dataset", {})
    if (
        export_dataset.get("name") != identity["dataset"]
        or export_dataset.get("revision") != identity["dataset_revision"]
    ):
        raise ValueError("Export and calibration dataset identities differ")
    if export_manifest.get("details", {}).get("config") != args.config_name:
        raise ValueError("Export and calibration config names differ")

    norm_payload = json.loads(args.norm_stats_json.read_text())
    arrays, envelope = derive_envelopes(
        track=args.track,
        action_sources=[calibration_actions, golden_actions, reference_actions],
        state_sources=[calibration_states, golden_states, reference_state],
        action_stats=_norm_entry(norm_payload, "actions"),
        state_stats=_norm_entry(norm_payload, "state"),
        normalization=args.normalization,
        internal_margin=args.internal_margin,
    )
    calibration_summary = args.calibration_manifest.resolve().parent / "corpus.json"
    if not calibration_summary.is_file():
        raise FileNotFoundError(f"Calibration summary is required: {calibration_summary}")
    summary_payload = json.loads(calibration_summary.read_text())
    if summary_payload.get("manifest_sha256") != sha256_file(args.calibration_manifest):
        raise ValueError("Calibration summary does not seal the selected manifest")
    for key, expected in (
        ("config_name", args.config_name),
        ("dataset", identity["dataset"]),
        ("dataset_revision", identity["dataset_revision"]),
    ):
        if summary_payload.get(key) != expected:
            raise ValueError(f"Calibration summary {key} mismatch: {summary_payload.get(key)!r} != {expected!r}")
    metadata = {
        "schema_version": 1,
        "gate_kind": "corpus_envelope",
        "hardware_safety_guarantee": False,
        "track": args.track,
        "config_name": args.config_name,
        "dataset": identity["dataset"],
        "dataset_revision": identity["dataset_revision"],
        "normalization": args.normalization,
        "envelope": {
            **envelope,
            "source_sample_counts": {
                "calibration": int(calibration_actions.shape[0]),
                "golden": int(golden_actions.shape[0]),
                "export_reference": int(reference_actions.shape[0]),
            },
        },
        "physical_gate": {
            "kind": "corpus_envelope_only",
            "hardware_limits_present": False,
            "rollout_environment_limit_check_required": True,
            "note": (
                "Physical arrays are observed corpus envelopes only. DROID joints 0-6 are unset because "
                "absolute joint targets depend on the current unnormalized state."
            ),
        },
        "sources": {
            "calibration_manifest": artifact_record(args.calibration_manifest),
            "calibration_summary": artifact_record(calibration_summary),
            "calibration_chunks": chunk_sources,
            "golden_corpus": artifact_record(args.golden_corpus),
            "golden_metadata": artifact_record(golden_metadata_path),
            "norm_stats": artifact_record(args.norm_stats_json),
            "export_manifest": artifact_record(export_manifest_path),
            "export_reference": artifact_record(reference_path),
            "export_inputs": artifact_record(inputs_path),
        },
    }
    sidecar = write_artifact(args.output.resolve(), arrays=arrays, metadata=metadata)
    print(
        json.dumps(
            {
                "artifact": artifact_record(args.output.resolve()),
                "metadata": artifact_record(sidecar),
                "gate_kind": metadata["gate_kind"],
                "hardware_safety_guarantee": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
