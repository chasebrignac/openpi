"""Fail-closed RoboLab expert dataset and deterministic DROID mixing.

The expert data remains in RoboLab's native HDF5 files.  A sealed manifest
binds every selected successful trajectory to the Shallow evaluation trigger,
the accepted student checkpoint, and the pinned RoboLab revision.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping
import hashlib
import json
import math
import pathlib
import re
from typing import Any, SupportsIndex

import numpy as np

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - exercised only outside the recovery image
    h5py = None


ROBOLAB_GIT_SHA = "0aef241fb088ca21bb4ebd24448940ed56620d17"
ROBOLAB_OPENPI_CLIENT_GIT_SHA = "aa6420561529593114160d05e5ad155792b272f3"
STACK_TASK = "Stack3RubiksCubeTask"
TRIGGER_GAP = 0.05
MAX_EXPERT_TRAJECTORIES = 100
MANIFEST_DATASET = "robolab_stack_success_bc"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_IMAGE_DIGEST_RE = re.compile(r"(?:.+@)?sha256:[0-9a-f]{64}")
LAYOUT_PATHS = {
    "nested_obs_v1": {
        "exterior": "obs/image_obs/over_shoulder_left_camera",
        "wrist": "obs/image_obs/wrist_cam",
        "joints": "obs/proprio_obs/arm_joint_pos",
        "gripper": "obs/proprio_obs/gripper_pos",
        "actions": "actions",
    },
    "flat_obs_v1": {
        "exterior": "obs/over_shoulder_left_camera",
        "wrist": "obs/wrist_cam",
        "joints": "obs/arm_joint_pos",
        "gripper": "obs/gripper_pos",
        "actions": "actions",
    },
}
_VERIFIED_FILE_HASHES: dict[tuple[pathlib.Path, int, int], str] = {}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _load_json_object(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _safe_relative_path(root: pathlib.Path, raw_path: Any, label: str) -> pathlib.Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} must be a non-empty relative path")
    relative = pathlib.PurePosixPath(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must stay inside the manifest directory: {raw_path!r}")
    resolved = (root / pathlib.Path(*relative.parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the manifest directory: {raw_path!r}") from error
    return resolved


def _manifest_without_hash(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "manifest_sha256"}


def load_expert_manifest(path: pathlib.Path, *, verify_files: bool = True) -> dict[str, Any]:
    """Load a sealed recovery manifest and validate all provenance and files."""
    path = path.expanduser().resolve()
    manifest = _load_json_object(path, "RoboLab expert manifest")
    if manifest.get("schema_version") != 1 or manifest.get("dataset") != MANIFEST_DATASET:
        raise ValueError(f"Unsupported RoboLab expert manifest schema: {path}")
    expected_manifest_hash = _require_sha256(manifest.get("manifest_sha256"), "manifest_sha256")
    actual_manifest_hash = canonical_sha256(_manifest_without_hash(manifest))
    if actual_manifest_hash != expected_manifest_hash:
        raise ValueError(
            f"RoboLab expert manifest hash mismatch: expected {expected_manifest_hash}, got {actual_manifest_hash}"
        )

    provenance = manifest.get("provenance")
    selection = manifest.get("selection")
    files = manifest.get("files")
    episodes = manifest.get("episodes")
    if not isinstance(provenance, dict) or not isinstance(selection, dict):
        raise ValueError("RoboLab expert manifest is missing provenance or selection")
    if not isinstance(files, list) or not isinstance(episodes, list):
        raise ValueError("RoboLab expert manifest files and episodes must be lists")
    if provenance.get("robolab_git_sha") != ROBOLAB_GIT_SHA:
        raise ValueError(f"Expert data must come from pinned RoboLab {ROBOLAB_GIT_SHA}")
    if provenance.get("openpi_client_git_sha") != ROBOLAB_OPENPI_CLIENT_GIT_SHA:
        raise ValueError(f"Expert data must use pinned OpenPI client {ROBOLAB_OPENPI_CLIENT_GIT_SHA}")
    if (
        not isinstance(provenance.get("openpi_source_sha"), str)
        or _GIT_SHA_RE.fullmatch(provenance["openpi_source_sha"]) is None
    ):
        raise ValueError("Expert manifest must include the exact 40-character OpenPI source SHA")
    if (
        not isinstance(provenance.get("robolab_image_digest"), str)
        or _IMAGE_DIGEST_RE.fullmatch(provenance["robolab_image_digest"]) is None
    ):
        raise ValueError("Expert manifest must include an immutable RoboLab image digest")
    if provenance.get("record_image_data_required") is not True:
        raise ValueError("Expert manifest must prove that --record-image-data was required")
    trigger = provenance.get("trigger")
    if not isinstance(trigger, dict) or trigger.get("task") != STACK_TASK or trigger.get("fired") is not True:
        raise ValueError("Expert recovery is dormant unless the Stack3RubiksCubeTask trigger fired")
    success_gap = trigger.get("success_gap")
    threshold = trigger.get("threshold")
    if (
        isinstance(success_gap, bool)
        or not isinstance(success_gap, int | float)
        or not math.isfinite(success_gap)
        or threshold != TRIGGER_GAP
        or success_gap <= TRIGGER_GAP
    ):
        raise ValueError("Stack recovery requires a finite success gap strictly greater than 0.05")
    accepted = provenance.get("accepted_shallow_checkpoint")
    teacher = provenance.get("teacher_checkpoint")
    if not isinstance(accepted, dict) or not isinstance(teacher, dict):
        raise ValueError("Expert manifest must bind accepted Shallow and teacher checkpoints")
    _require_sha256(accepted.get("model_sha256"), "accepted Shallow model hash")
    _require_sha256(teacher.get("model_sha256"), "teacher model hash")

    count = selection.get("selected_trajectories")
    maximum = selection.get("maximum_trajectories")
    if selection.get("task") != STACK_TASK or selection.get("success_only") is not True:
        raise ValueError("Expert manifest may contain successful Stack3RubiksCubeTask trajectories only")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or count > MAX_EXPERT_TRAJECTORIES
        or isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or not 1 <= maximum <= MAX_EXPERT_TRAJECTORIES
        or count > maximum
        or len(episodes) != count
    ):
        raise ValueError("Expert manifest must select between 1 and 100 trajectories within its declared cap")

    root = path.parent.resolve()
    resolved_files: list[pathlib.Path] = []
    seen_file_paths: set[str] = set()
    for index, file_record in enumerate(files):
        if not isinstance(file_record, dict):
            raise ValueError(f"Expert file record {index} must be an object")
        raw_path = file_record.get("path")
        if raw_path in seen_file_paths:
            raise ValueError(f"Duplicate expert HDF5 path in manifest: {raw_path!r}")
        seen_file_paths.add(raw_path)
        resolved = _safe_relative_path(root, raw_path, f"files[{index}].path")
        expected_size = file_record.get("size_bytes")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size <= 0:
            raise ValueError(f"files[{index}].size_bytes must be a positive integer")
        expected_hash = _require_sha256(file_record.get("sha256"), f"files[{index}].sha256")
        if verify_files:
            if not resolved.is_file():
                raise ValueError(f"Expert HDF5 file does not exist: {resolved}")
            if resolved.stat().st_size != expected_size:
                raise ValueError(f"Expert HDF5 size mismatch: {resolved}")
            stat = resolved.stat()
            cache_key = (resolved, stat.st_size, stat.st_mtime_ns)
            actual_hash = _VERIFIED_FILE_HASHES.get(cache_key)
            if actual_hash is None:
                actual_hash = sha256_file(resolved)
                _VERIFIED_FILE_HASHES[cache_key] = actual_hash
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Expert HDF5 hash mismatch for {resolved}: expected {expected_hash}, got {actual_hash}"
                )
        resolved_files.append(resolved)

    seen_trajectories: set[str] = set()
    total_frames = 0
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise ValueError(f"Expert episode record {index} must be an object")
        trajectory_id = episode.get("trajectory_id")
        if not isinstance(trajectory_id, str) or not trajectory_id or trajectory_id in seen_trajectories:
            raise ValueError(f"Expert episode {index} has an invalid or duplicate trajectory_id")
        seen_trajectories.add(trajectory_id)
        file_index = episode.get("file_index")
        frames = episode.get("frames")
        if isinstance(file_index, bool) or not isinstance(file_index, int) or not 0 <= file_index < len(files):
            raise ValueError(f"Expert episode {index} has an invalid file_index")
        if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0:
            raise ValueError(f"Expert episode {index} has an invalid frame count")
        if episode.get("layout") not in LAYOUT_PATHS:
            raise ValueError(f"Expert episode {index} has an unsupported observation layout")
        if not isinstance(episode.get("instruction"), str) or not episode["instruction"].strip():
            raise ValueError(f"Expert episode {index} has no instruction")
        if not isinstance(episode.get("group"), str) or re.fullmatch(r"data/demo_[0-9]+", episode["group"]) is None:
            raise ValueError(f"Expert episode {index} has an invalid HDF5 group")
        total_frames += frames
    if total_frames != selection.get("selected_frames"):
        raise ValueError("Expert manifest selected_frames does not match its episode records")

    return manifest


def validate_rerun_decision(path: pathlib.Path, *, manifest_sha256: str) -> dict[str, Any]:
    """Validate the evidence gate that permits the optional 50/50 run."""
    path = path.expanduser().resolve()
    decision = _load_json_object(path, "RoboLab BC rerun decision")
    if decision.get("schema_version") != 1 or decision.get("decision") != "robolab_bc_50_50":
        raise ValueError(f"Unsupported RoboLab BC rerun decision schema: {path}")
    expected_hash = _require_sha256(decision.get("decision_sha256"), "decision_sha256")
    actual_hash = canonical_sha256({key: value for key, value in decision.items() if key != "decision_sha256"})
    if actual_hash != expected_hash:
        raise ValueError(f"RoboLab BC rerun decision hash mismatch: expected {expected_hash}, got {actual_hash}")
    if decision.get("expert_manifest_sha256") != manifest_sha256:
        raise ValueError("RoboLab BC rerun decision is bound to a different expert manifest")
    if decision.get("approved") is not True:
        raise ValueError("The optional 50/50 BC run was not approved by its paired rollout evidence")
    checks = decision.get("checks")
    if (
        not isinstance(checks, dict)
        or checks.get("stack_improved") is not True
        or checks.get("banana_not_degraded") is not True
    ):
        raise ValueError("The optional 50/50 BC decision is missing its required quality checks")
    return decision


def validate_recovery_source_checkpoint(
    manifest_path: pathlib.Path,
    checkpoint_path: str | pathlib.Path | None,
    *,
    teacher_checkpoint_path: str | pathlib.Path | None,
) -> dict[str, Any]:
    """Ensure BC starts from the accepted Shallow checkpoint without a resident teacher."""
    if teacher_checkpoint_path is not None:
        raise ValueError("RoboLab expert BC uses ground-truth flow matching and forbids a teacher checkpoint")
    if checkpoint_path is None:
        raise ValueError("RoboLab expert BC requires an accepted Shallow --pytorch-weight-path")
    manifest = load_expert_manifest(manifest_path, verify_files=False)
    checkpoint_path = pathlib.Path(checkpoint_path).expanduser().resolve()
    model_path = (
        checkpoint_path if checkpoint_path.name == "model.safetensors" else checkpoint_path / "model.safetensors"
    )
    if not model_path.is_file():
        raise ValueError(f"Accepted Shallow checkpoint model does not exist: {model_path}")
    expected = manifest["provenance"]["accepted_shallow_checkpoint"]["model_sha256"]
    actual = sha256_file(model_path)
    if actual != expected:
        raise ValueError(
            f"BC initialization checkpoint hash mismatch: expected accepted Shallow {expected}, got {actual}"
        )
    return {
        "initial_checkpoint_model": str(model_path),
        "initial_checkpoint_sha256": actual,
        "teacher_checkpoint_resident": False,
        "loss": "ground_truth_flow_matching",
    }


class RoboLabExpertDataset:
    """Random-access view over selected native RoboLab HDF5 trajectories."""

    def __init__(self, manifest_path: str | pathlib.Path, *, action_horizon: int, verify_hashes: bool = True):
        if h5py is None:
            raise ModuleNotFoundError(
                "RoboLab expert BC requires h5py; use the pinned DROID reproduction image, which installs it"
            )
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        self._handles: dict[pathlib.Path, Any] = {}
        self.manifest_path = pathlib.Path(manifest_path).expanduser().resolve()
        self.manifest = load_expert_manifest(self.manifest_path, verify_files=verify_hashes)
        self._root = self.manifest_path.parent
        self._action_horizon = action_horizon
        self._files = tuple(
            _safe_relative_path(self._root, record["path"], "expert HDF5 path") for record in self.manifest["files"]
        )
        self._episodes = tuple(self.manifest["episodes"])
        self._cumulative_ends: list[int] = []
        cumulative = 0
        for episode in self._episodes:
            cumulative += episode["frames"]
            self._cumulative_ends.append(cumulative)

    def __len__(self) -> int:
        return self._cumulative_ends[-1] if self._cumulative_ends else 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self):  # pragma: no cover - interpreter shutdown is implementation-dependent
        self.close()

    def _handle(self, file_index: int):
        path = self._files[file_index]
        handle = self._handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r", swmr=True)
            self._handles[path] = handle
        return handle

    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]:
        frame_index = index.__index__()
        if frame_index < 0:
            frame_index += len(self)
        if frame_index < 0 or frame_index >= len(self):
            raise IndexError(frame_index)
        episode_index = bisect.bisect_right(self._cumulative_ends, frame_index)
        previous_end = self._cumulative_ends[episode_index - 1] if episode_index else 0
        local_index = frame_index - previous_end
        episode = self._episodes[episode_index]
        demo = self._handle(episode["file_index"])[episode["group"]]
        paths = LAYOUT_PATHS[episode["layout"]]

        exterior = np.asarray(demo[paths["exterior"]][local_index])
        wrist = np.asarray(demo[paths["wrist"]][local_index])
        joints = np.asarray(demo[paths["joints"]][local_index], dtype=np.float32)
        gripper = np.asarray(demo[paths["gripper"]][local_index], dtype=np.float32)
        stop = min(episode["frames"], local_index + self._action_horizon)
        actions = np.asarray(demo[paths["actions"]][local_index:stop], dtype=np.float32)
        if len(actions) < self._action_horizon:
            actions = np.concatenate(
                [actions, np.repeat(actions[-1:], self._action_horizon - len(actions), axis=0)], axis=0
            )
        if not np.isfinite(joints).all() or not np.isfinite(gripper).all() or not np.isfinite(actions).all():
            raise ValueError(f"Non-finite expert sample in trajectory {episode['trajectory_id']} frame {local_index}")
        return {
            "observation.images.exterior_1_left": exterior,
            "observation.images.wrist_left": wrist,
            "observation.state.joint_position": joints,
            "observation.state.gripper_position": gripper,
            "action": actions,
            "prompt": episode["instruction"],
        }


def _affine_parameters(length: int, seed: int, namespace: str) -> tuple[int, int]:
    if length <= 1:
        return 1, 0
    digest = hashlib.sha256(f"{seed}\0{namespace}".encode()).digest()
    multiplier = int.from_bytes(digest[:8], "big") % length
    if multiplier == 0:
        multiplier = 1
    while math.gcd(multiplier, length) != 1:
        multiplier = (multiplier + 1) % length
        if multiplier == 0:
            multiplier = 1
    offset = int.from_bytes(digest[8:16], "big") % length
    return multiplier, offset


class DeterministicMixtureDataset:
    """Exact per-rank 25/75 or 50/50 expert/base sampling without RNG state."""

    def __init__(
        self,
        base_dataset,
        expert_dataset,
        *,
        expert_fraction: float,
        seed: int,
        num_replicas: int = 1,
    ):
        ratios = {0.25: (1, 4), 0.5: (1, 2)}
        if expert_fraction not in ratios:
            raise ValueError("RoboLab expert_fraction must be exactly 0.25 or 0.5")
        if len(base_dataset) <= 0 or len(expert_dataset) <= 0:
            raise ValueError("Both DROID and RoboLab expert datasets must be non-empty")
        if isinstance(num_replicas, bool) or not isinstance(num_replicas, int) or num_replicas <= 0:
            raise ValueError("num_replicas must be a positive integer")
        self.base_dataset = base_dataset
        self.expert_dataset = expert_dataset
        self.expert_fraction = expert_fraction
        self.seed = seed
        self.num_replicas = num_replicas
        self.expert_per_cycle, self.denominator = ratios[expert_fraction]
        self.base_per_cycle = self.denominator - self.expert_per_cycle
        rotation = int.from_bytes(hashlib.sha256(f"{seed}\0mix-pattern".encode()).digest()[:8], "big")
        rotation %= self.denominator
        self._expert_slots = tuple(
            sorted((rotation + offset) % self.denominator for offset in range(self.expert_per_cycle))
        )
        self._base_slots = tuple(slot for slot in range(self.denominator) if slot not in self._expert_slots)
        cycles = math.ceil(len(base_dataset) / (num_replicas * self.base_per_cycle))
        self._length = cycles * self.denominator * num_replicas
        self._base_affine = _affine_parameters(len(base_dataset), seed, "droid")
        self._expert_affine = _affine_parameters(len(expert_dataset), seed, "robolab-expert")

    def __len__(self) -> int:
        return self._length

    @staticmethod
    def _permute(ordinal: int, length: int, parameters: tuple[int, int]) -> int:
        multiplier, offset = parameters
        return (multiplier * (ordinal % length) + offset) % length

    def source_for_index(self, index: SupportsIndex) -> str:
        mixed_index = index.__index__()
        if mixed_index < 0:
            mixed_index += len(self)
        if mixed_index < 0 or mixed_index >= len(self):
            raise IndexError(mixed_index)
        local_ordinal = mixed_index // self.num_replicas
        slot = local_ordinal % self.denominator
        return "expert" if slot in self._expert_slots else "droid"

    def __getitem__(self, index: SupportsIndex):
        mixed_index = index.__index__()
        if mixed_index < 0:
            mixed_index += len(self)
        if mixed_index < 0 or mixed_index >= len(self):
            raise IndexError(mixed_index)
        rank = mixed_index % self.num_replicas
        local_ordinal = mixed_index // self.num_replicas
        cycle, slot = divmod(local_ordinal, self.denominator)
        if slot in self._expert_slots:
            within_cycle = self._expert_slots.index(slot)
            ordinal = (cycle * self.num_replicas + rank) * self.expert_per_cycle + within_cycle
            expert_index = self._permute(ordinal, len(self.expert_dataset), self._expert_affine)
            return self.expert_dataset[expert_index]
        within_cycle = self._base_slots.index(slot)
        ordinal = (cycle * self.num_replicas + rank) * self.base_per_cycle + within_cycle
        base_index = self._permute(ordinal, len(self.base_dataset), self._base_affine)
        return self.base_dataset[base_index]

    def provenance(self) -> dict[str, Any]:
        return {
            "strategy": "deterministic_rank_local_exact_cycle",
            "expert_fraction": self.expert_fraction,
            "droid_fraction": 1.0 - self.expert_fraction,
            "seed": self.seed,
            "cycle_length": self.denominator,
            "expert_slots": list(self._expert_slots),
            "num_replicas": self.num_replicas,
            "virtual_samples": len(self),
            "droid_samples": len(self.base_dataset),
            "expert_frames": len(self.expert_dataset),
        }
