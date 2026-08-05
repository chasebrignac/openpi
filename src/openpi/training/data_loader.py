import bisect
from collections.abc import Iterator, Mapping, Sequence
import dataclasses
import hashlib
import logging
import multiprocessing
import os
import pathlib
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp

try:
    # LeRobot >=0.4 / dataset codebase v3.0.
    import lerobot.datasets.lerobot_dataset as lerobot_dataset
except ModuleNotFoundError:  # pragma: no cover - legacy conversion environment only
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import polars as pl
import torch

import openpi.models.model as _model
from openpi.training import robolab_expert_dataset as _robolab_expert
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)
LEROBOT_VIDEO_BACKEND = "pyav"


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def split_metadata(self) -> dict | None:
        """Get deterministic episode-split provenance, when configured."""
        raise NotImplementedError("Subclasses of DataLoader should implement split_metadata.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class EpisodePromptTransform:
    """Prefer episode-level annotations and fall back to LeRobot task metadata."""

    def __init__(self, episode_tasks: dict[int, str], standard_tasks: dict[int, str]):
        self._episode_tasks = episode_tasks
        self._standard_tasks = standard_tasks

    def __call__(self, sample: dict) -> dict:
        if "episode_index" not in sample:
            raise ValueError('Cannot resolve an episode prompt without "episode_index"')
        episode_index = int(np.asarray(sample["episode_index"]).reshape(-1)[0])
        prompt = self._episode_tasks.get(episode_index)
        if not prompt:
            if "task_index" not in sample:
                raise ValueError(f"Episode {episode_index} has no annotation and no task_index fallback")
            task_index = int(np.asarray(sample["task_index"]).reshape(-1)[0])
            prompt = self._standard_tasks.get(task_index)
        if not prompt:
            raise ValueError(f"No annotated or standard task prompt for episode {episode_index}")
        return {**sample, "prompt": prompt}


def load_episode_prompts(path: pathlib.Path) -> dict[int, str]:
    """Load and validate MolmoAct2's episode-indexed language annotations."""
    if not path.is_file():
        raise FileNotFoundError(f"Episode prompt parquet not found: {path}")
    frame = pl.read_parquet(path, columns=["episode_index", "task"])
    result: dict[int, str] = {}
    for raw_episode_index, raw_task in frame.iter_rows():
        episode_index = int(raw_episode_index)
        task = str(raw_task).strip() if raw_task is not None else ""
        if not task:
            continue
        if episode_index in result and result[episode_index] != task:
            raise ValueError(f"Conflicting annotations for episode_index={episode_index}")
        result[episode_index] = task
    if not result:
        raise ValueError(f"No non-empty episode annotations found in {path}")
    return result


def normalize_lerobot_tasks(tasks) -> dict[int, str]:
    """Normalize LeRobot v2 dictionaries and v3 task DataFrames."""
    if isinstance(tasks, Mapping):
        result = {int(index): str(task).strip() for index, task in tasks.items()}
    elif hasattr(tasks, "iterrows"):
        result = {}
        for task_text, row in tasks.iterrows():
            task_index = int(row["task_index"])
            task = str(task_text).strip()
            if task_index in result and result[task_index] != task:
                raise ValueError(f"Conflicting LeRobot task text for task_index={task_index}")
            result[task_index] = task
    else:
        raise TypeError(f"Unsupported LeRobot task metadata type: {type(tasks).__name__}")
    if not result or any(not task for task in result.values()):
        raise ValueError("LeRobot task metadata contains no usable non-empty task text")
    return result


@dataclasses.dataclass(frozen=True)
class EpisodeRecord:
    """Minimal episode metadata required for a leakage-free holdout."""

    episode_id: int
    frames: int
    start_index: int = 0
    task: str | None = None
    site: str | None = None


@dataclasses.dataclass(frozen=True)
class OfflineEpisodeSplit:
    """A deterministic whole-episode split selected before delta/action queries."""

    seed: int
    requested_holdout_samples: int
    validation_frames: int
    train_episode_ids: tuple[int, ...]
    validation_episode_ids: tuple[int, ...]
    stratified_by_task: bool
    stratified_by_site: bool

    def selected_episode_ids(self, split: Literal["train", "validation"]) -> tuple[int, ...]:
        if split == "train":
            return self.train_episode_ids
        if split == "validation":
            return self.validation_episode_ids
        raise ValueError(f"Unknown dataset split: {split}")

    def metadata(self, split: Literal["train", "validation"]) -> dict:
        """Return JSON-serializable provenance without enumerating the full training set."""
        selected = self.selected_episode_ids(split)
        return {
            "schema_version": 1,
            "strategy": "deterministic_whole_episode_stratified",
            "seed": self.seed,
            "split": split,
            "requested_holdout_samples": self.requested_holdout_samples,
            "validation_frames": self.validation_frames,
            "validation_episode_ids": list(self.validation_episode_ids),
            "validation_episode_count": len(self.validation_episode_ids),
            "train_episode_count": len(self.train_episode_ids),
            "selected_episode_count": len(selected),
            "stratified_by_task": self.stratified_by_task,
            "stratified_by_site": self.stratified_by_site,
        }


def _scalar_int(value, *, field: str) -> int:
    values = np.asarray(value).reshape(-1)
    if len(values) != 1:
        raise ValueError(f"Episode metadata field {field!r} must be scalar, got {value!r}")
    return int(values[0])


def _metadata_label(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        label = value.strip()
        return label or None
    if isinstance(value, Mapping):
        values = [_metadata_label(item) for item in value.values()]
    elif isinstance(value, Sequence | np.ndarray):
        values = [_metadata_label(item) for item in value]
    else:
        label = str(value).strip()
        return label or None
    labels = sorted({label for label in values if label})
    return " | ".join(labels) if labels else None


def normalize_episode_records(
    episodes,
    *,
    standard_tasks: Mapping[int, str] | None = None,
    episode_tasks: Mapping[int, str] | None = None,
) -> tuple[EpisodeRecord, ...]:
    """Normalize LeRobot v2 dictionaries and v3 episode datasets."""
    standard_tasks = standard_tasks or {}
    episode_tasks = episode_tasks or {}
    raw_records = episodes.items() if isinstance(episodes, Mapping) else enumerate(episodes)

    records: list[EpisodeRecord] = []
    seen_ids: set[int] = set()
    next_start_index = 0
    for fallback_id, raw_record in raw_records:
        if not isinstance(raw_record, Mapping):
            raise TypeError(f"Unsupported LeRobot episode metadata row: {type(raw_record).__name__}")
        episode_id = _scalar_int(raw_record.get("episode_index", fallback_id), field="episode_index")
        if episode_id in seen_ids:
            raise ValueError(f"Duplicate episode_index={episode_id}")
        seen_ids.add(episode_id)

        if "length" in raw_record:
            frames = _scalar_int(raw_record["length"], field="length")
        elif "num_frames" in raw_record:
            frames = _scalar_int(raw_record["num_frames"], field="num_frames")
        elif "dataset_from_index" in raw_record and "dataset_to_index" in raw_record:
            frames = _scalar_int(raw_record["dataset_to_index"], field="dataset_to_index") - _scalar_int(
                raw_record["dataset_from_index"], field="dataset_from_index"
            )
        else:
            raise ValueError(f"Episode {episode_id} metadata does not include a frame count")
        if frames <= 0:
            raise ValueError(f"Episode {episode_id} has invalid frame count {frames}")
        start_index = (
            _scalar_int(raw_record["dataset_from_index"], field="dataset_from_index")
            if "dataset_from_index" in raw_record
            else next_start_index
        )
        next_start_index = max(next_start_index, start_index + frames)

        task = _metadata_label(episode_tasks.get(episode_id))
        if task is None:
            task = _metadata_label(raw_record.get("task", raw_record.get("tasks")))
        if task is None:
            raw_task_indices = raw_record.get("task_index", raw_record.get("task_indices"))
            if raw_task_indices is not None:
                task_indices = np.asarray(raw_task_indices).reshape(-1)
                task = _metadata_label([standard_tasks.get(int(index)) for index in task_indices])

        site = None
        for key in ("site", "site_id", "location", "location_id", "scene", "scene_id", "lab", "lab_id"):
            site = _metadata_label(raw_record.get(key))
            if site is not None:
                break
        records.append(
            EpisodeRecord(episode_id=episode_id, frames=frames, start_index=start_index, task=task, site=site)
        )

    if len(records) < 2:
        raise ValueError("A whole-episode holdout requires at least two episodes")
    return tuple(sorted(records, key=lambda record: record.episode_id))


class EpisodeFilteredDataset(Dataset[T_co]):
    """Compact frame-index view over selected whole episodes of a full dataset."""

    def __init__(self, dataset: Dataset[T_co], records: Sequence[EpisodeRecord], episode_ids: Sequence[int]):
        selected_ids = set(episode_ids)
        selected = sorted(
            (record for record in records if record.episode_id in selected_ids), key=lambda x: x.start_index
        )
        if len(selected) != len(selected_ids):
            missing = selected_ids - {record.episode_id for record in selected}
            raise ValueError(f"Missing episode metadata for IDs {sorted(missing)}")
        self._dataset = dataset
        self._ranges = tuple((record.start_index, record.start_index + record.frames) for record in selected)
        cumulative = 0
        self._cumulative_ends = []
        for start, end in self._ranges:
            if start < 0 or end > len(dataset):
                raise ValueError(f"Episode frame range [{start}, {end}) is outside dataset length {len(dataset)}")
            cumulative += end - start
            self._cumulative_ends.append(cumulative)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        relative_index = index.__index__()
        if relative_index < 0:
            relative_index += len(self)
        if relative_index < 0 or relative_index >= len(self):
            raise IndexError(relative_index)
        range_index = bisect.bisect_right(self._cumulative_ends, relative_index)
        previous_end = self._cumulative_ends[range_index - 1] if range_index else 0
        start, _ = self._ranges[range_index]
        return self._dataset[start + relative_index - previous_end]

    def __len__(self) -> int:
        return self._cumulative_ends[-1] if self._cumulative_ends else 0


def _stable_split_key(seed: int, namespace: str, value: object) -> bytes:
    return hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode()).digest()


def select_offline_episode_split(
    records: Sequence[EpisodeRecord], *, holdout_samples: int, seed: int
) -> OfflineEpisodeSplit:
    """Select whole validation episodes with deterministic task/site round-robin stratification."""
    if holdout_samples <= 0:
        raise ValueError("holdout_samples must be positive")
    if len(records) < 2:
        raise ValueError("A whole-episode holdout requires at least two episodes")
    episode_ids = [record.episode_id for record in records]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("Episode IDs must be unique")

    strata: dict[tuple[str, str], list[EpisodeRecord]] = {}
    for record in records:
        stratum = (record.site or "<unknown-site>", record.task or "<unknown-task>")
        strata.setdefault(stratum, []).append(record)
    ordered_strata = sorted(strata, key=lambda value: _stable_split_key(seed, "stratum", value))
    for stratum, stratum_records in strata.items():
        stratum_records.sort(key=lambda record: _stable_split_key(seed, repr(stratum), record.episode_id))

    # Interleave strata so that task/site diversity is represented before a
    # second episode is taken from any stratum.
    ordered_records: list[EpisodeRecord] = []
    offset = 0
    while len(ordered_records) < len(records):
        for stratum in ordered_strata:
            stratum_records = strata[stratum]
            if offset < len(stratum_records):
                ordered_records.append(stratum_records[offset])
        offset += 1

    # Reserve a deterministic shortest episode as a training anchor. This
    # guarantees that any feasible requested frame count can be satisfied
    # while retaining at least one training episode.
    training_anchor = min(
        records,
        key=lambda record: (record.frames, _stable_split_key(seed, "training-anchor", record.episode_id)),
    )
    validation: list[EpisodeRecord] = []
    validation_frames = 0
    for record in ordered_records:
        if record.episode_id == training_anchor.episode_id:
            continue
        validation.append(record)
        validation_frames += record.frames
        if validation_frames >= holdout_samples:
            break
    if validation_frames < holdout_samples:
        available = sum(record.frames for record in records) - min(record.frames for record in records)
        raise ValueError(
            f"Cannot reserve {holdout_samples} validation frames while leaving a training episode; "
            f"at most {available} frames are available"
        )

    validation_ids = {record.episode_id for record in validation}
    return OfflineEpisodeSplit(
        seed=seed,
        requested_holdout_samples=holdout_samples,
        validation_frames=validation_frames,
        train_episode_ids=tuple(record.episode_id for record in records if record.episode_id not in validation_ids),
        validation_episode_ids=tuple(record.episode_id for record in validation),
        stratified_by_task=any(record.task is not None for record in records),
        stratified_by_site=any(record.site is not None for record in records),
    )


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def _lerobot_root(data_config: _config.DataConfig) -> pathlib.Path | None:
    return pathlib.Path(data_config.lerobot_root) if data_config.lerobot_root is not None else None


def _episode_prompt_path(data_config: _config.DataConfig, root: pathlib.Path | None) -> pathlib.Path | None:
    if data_config.episode_prompt_path is None:
        return None
    prompt_path = pathlib.Path(data_config.episode_prompt_path)
    if not prompt_path.is_absolute():
        if root is None:
            raise ValueError("A relative episode_prompt_path requires lerobot_root")
        prompt_path = root / prompt_path
    return prompt_path


def _load_lerobot_metadata(data_config: _config.DataConfig):
    if data_config.repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(
        data_config.repo_id,
        root=_lerobot_root(data_config),
        revision=data_config.lerobot_revision,
    )
    if data_config.lerobot_codebase_version is not None:
        actual_version = dataset_meta.info.get("codebase_version")
        if actual_version != data_config.lerobot_codebase_version:
            raise ValueError(
                f"LeRobot codebase version mismatch for {data_config.repo_id}: "
                f"expected {data_config.lerobot_codebase_version}, got {actual_version}"
            )
    return dataset_meta


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    episodes: Sequence[int] | None = None,
    episode_records: Sequence[EpisodeRecord] | None = None,
    dataset_meta=None,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        dataset = FakeDataset(model_config, num_samples=1024)
        return torch.utils.data.Subset(dataset, list(episodes)) if episodes is not None else dataset

    root = _lerobot_root(data_config)
    dataset_meta = dataset_meta or _load_lerobot_metadata(data_config)
    codebase_version = str(dataset_meta.info.get("codebase_version", ""))
    # The pinned v2 loader indexes action deltas by absolute episode ID after
    # loading a subset, so it always needs a full-dataset view. The pinned v3
    # constructor is safe for a small subset, but eagerly materializes the
    # filtered Arrow table and a Python absolute-to-relative entry for every
    # selected frame. A training split that is most of the dataset is therefore
    # cheaper as a memory-mapped full dataset plus this module's compact episode
    # range view. Both paths preserve absolute indices and whole-episode bounds.
    use_episode_view = codebase_version.startswith("v2.")
    if codebase_version.startswith("v3.") and episodes is not None and episode_records is not None:
        selected_ids = set(episodes)
        selected_frames = sum(record.frames for record in episode_records if record.episode_id in selected_ids)
        total_frames = sum(record.frames for record in episode_records)
        use_episode_view = selected_frames * 2 > total_frames
    constructor_episodes = None if use_episode_view else episodes
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        root=root,
        episodes=list(constructor_episodes) if constructor_episodes is not None else None,
        revision=data_config.lerobot_revision,
        # The pinned AWS DLC uses a static CPython 3.12 build, while the
        # lock-pinned TorchCodec wheel links libpython3.12.so. Both pinned
        # LeRobot revisions support PyAV explicitly, so select it rather than
        # allowing presence of an unusable TorchCodec package to choose the
        # default backend.
        video_backend=LEROBOT_VIDEO_BACKEND,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )
    if episodes is not None and constructor_episodes is None:
        if episode_records is None:
            raise ValueError("A whole-episode dataset view requires normalized episode metadata")
        dataset = EpisodeFilteredDataset(dataset, episode_records, episodes)

    standard_tasks = normalize_lerobot_tasks(dataset_meta.tasks)
    if data_config.episode_prompt_path is not None:
        prompt_path = _episode_prompt_path(data_config, root)
        assert prompt_path is not None
        dataset = TransformedDataset(
            dataset,
            [EpisodePromptTransform(load_episode_prompts(prompt_path), standard_tasks)],
        )
    elif data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(standard_tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
    split: Literal["train", "validation"] = "train",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
        split: Select the training records or the deterministic offline holdout.
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        if config.offline_holdout_samples:
            raise ValueError("offline_holdout_samples is not supported by the iterable RLDS loader")
        if split != "train":
            raise ValueError("The iterable RLDS loader does not provide a deterministic validation split")
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        holdout_samples=config.offline_holdout_samples,
        split=split,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    holdout_samples: int = 0,
    split: Literal["train", "validation"] = "train",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    if split not in {"train", "validation"}:
        raise ValueError(f"Unknown dataset split: {split}")
    if holdout_samples < 0:
        raise ValueError("holdout_samples must be non-negative")
    if split == "validation" and holdout_samples == 0:
        raise ValueError("validation split requires offline_holdout_samples > 0")
    recovery_enabled = data_config.robolab_expert_manifest_path is not None
    if recovery_enabled and framework != "pytorch":
        raise ValueError("RoboLab expert BC is supported by the PyTorch training path only")
    if recovery_enabled and (split != "train" or holdout_samples):
        raise ValueError("RoboLab expert BC cannot be combined with an offline holdout or validation loader")
    if recovery_enabled and data_config.robolab_expert_fraction not in {0.25, 0.5}:
        raise ValueError("RoboLab expert BC requires an exact 0.25 or 0.5 expert fraction")
    if not recovery_enabled and (
        data_config.robolab_expert_fraction is not None or data_config.robolab_rerun_decision_path is not None
    ):
        raise ValueError("RoboLab expert mix settings require robolab_expert_manifest_path")

    dataset_meta = None
    episode_split = None
    episode_records = None
    selected_episodes = None
    if holdout_samples:
        if data_config.repo_id == "fake":
            episode_records = tuple(EpisodeRecord(episode_id=index, frames=1) for index in range(1024))
        else:
            dataset_meta = _load_lerobot_metadata(data_config)
            standard_tasks = normalize_lerobot_tasks(dataset_meta.tasks)
            prompt_path = _episode_prompt_path(data_config, _lerobot_root(data_config))
            episode_tasks = load_episode_prompts(prompt_path) if prompt_path is not None else None
            episode_records = normalize_episode_records(
                dataset_meta.episodes,
                standard_tasks=standard_tasks,
                episode_tasks=episode_tasks,
            )
        episode_split = select_offline_episode_split(episode_records, holdout_samples=holdout_samples, seed=seed)
        selected_episodes = episode_split.selected_episode_ids(split)
        logging.info(
            "Offline split %s: held out %d whole episodes (%d frames), episode IDs=%s",
            split,
            len(episode_split.validation_episode_ids),
            episode_split.validation_frames,
            list(episode_split.validation_episode_ids),
        )

    # Episode filtering is applied to the raw LeRobot dataset (natively in v3,
    # through a whole-episode index view in v2) before model/data transforms.
    # Delta queries remain bounded to the selected sample's episode, so no
    # training chunk can reference a held-out episode.
    dataset = create_torch_dataset(
        data_config,
        action_horizon,
        model_config,
        episodes=selected_episodes,
        episode_records=episode_records if holdout_samples else None,
        dataset_meta=dataset_meta,
    )
    mixed_dataset = None
    if recovery_enabled:
        assert data_config.robolab_expert_manifest_path is not None
        assert data_config.robolab_expert_fraction is not None
        num_replicas = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        expert_dataset = _robolab_expert.RoboLabExpertDataset(
            data_config.robolab_expert_manifest_path,
            action_horizon=action_horizon,
        )
        manifest = expert_dataset.manifest
        decision = None
        if data_config.robolab_expert_fraction == 0.5:
            if data_config.robolab_rerun_decision_path is None:
                raise ValueError("The optional 50/50 expert-BC run requires a rerun decision")
            decision = _robolab_expert.validate_rerun_decision(
                pathlib.Path(data_config.robolab_rerun_decision_path),
                manifest_sha256=manifest["manifest_sha256"],
            )
        elif data_config.robolab_rerun_decision_path is not None:
            raise ValueError("The initial 25/75 expert-BC run must not use a rerun decision")
        mixed_dataset = _robolab_expert.DeterministicMixtureDataset(
            dataset,
            expert_dataset,
            expert_fraction=data_config.robolab_expert_fraction,
            seed=data_config.robolab_expert_seed,
            num_replicas=num_replicas,
        )
        dataset = mixed_dataset
        recovery_provenance = {
            "schema_version": 1,
            "kind": "conditional_robolab_expert_bc",
            "manifest_path": str(pathlib.Path(data_config.robolab_expert_manifest_path).expanduser().resolve()),
            "manifest_sha256": manifest["manifest_sha256"],
            "trigger": manifest["provenance"]["trigger"],
            "accepted_shallow_checkpoint": manifest["provenance"]["accepted_shallow_checkpoint"],
            "teacher_checkpoint_resident": False,
            "loss": "ground_truth_flow_matching",
            "selection": manifest["selection"],
            "mix": mixed_dataset.provenance(),
        }
        if decision is not None:
            recovery_provenance["rerun_decision"] = {
                "path": str(pathlib.Path(data_config.robolab_rerun_decision_path).expanduser().resolve()),
                "decision_sha256": decision["decision_sha256"],
                "approved": True,
            }
        data_config = dataclasses.replace(data_config, recovery_provenance=recovery_provenance)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle and mixed_dataset is None,
                drop_last=True,
                seed=seed,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    if mixed_dataset is not None and local_batch_size % mixed_dataset.denominator != 0:
        raise ValueError(
            f"Each rank's expert-BC batch ({local_batch_size}) must be divisible by the exact mix cycle "
            f"({mixed_dataset.denominator})"
        )

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        # The mixture dataset applies a seeded bijection to each source. Keeping
        # virtual indices ordered preserves an exact ratio in every rank-local
        # batch instead of relying on probabilistic weighted sampling.
        shuffle=(sampler is None and shuffle and mixed_dataset is None),
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    split_metadata = episode_split.metadata(split) if episode_split is not None else None
    return DataLoaderImpl(data_config, data_loader, split_metadata=split_metadata)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")
        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with shuffle")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches
        self._epoch = 0
        self._seed = seed
        self._next_batch_offset = 0

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._generator = generator
        if sampler is None:
            sampler = (
                torch.utils.data.RandomSampler(dataset, generator=generator)
                if shuffle
                else torch.utils.data.SequentialSampler(dataset)
            )
        self._sampler = sampler
        self._batch_sampler = _BatchOffsetSampler(
            torch.utils.data.BatchSampler(sampler, batch_size=local_batch_size, drop_last=True)
        )
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_sampler=self._batch_sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            if isinstance(self._sampler, torch.utils.data.distributed.DistributedSampler):
                self._sampler.set_epoch(self._epoch)
            # Make each epoch independently reproducible. In particular, a
            # single-GPU RandomSampler resumed directly at epoch N must not
            # depend on advancing a newly created generator through epochs
            # 0..N-1 first.
            epoch_seed = int.from_bytes(
                hashlib.sha256(f"openpi-loader-epoch-v1\0{self._seed}\0{self._epoch}".encode()).digest()[:8],
                "big",
            ) % (2**63 - 1)
            self._generator.manual_seed(epoch_seed)
            self._batch_sampler.set_batch_offset(self._next_batch_offset)
            self._next_batch_offset = 0
            try:
                data_iter = iter(self._data_loader)
            finally:
                # _BatchOffsetSampler snapshots the offset when DataLoader
                # constructs its sampler iterator, before workers are queued.
                self._batch_sampler.set_batch_offset(0)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)
            self._epoch += 1

    def __len__(self) -> int:
        if self._num_batches is not None:
            return self._num_batches
        return len(self._data_loader)

    def set_epoch(self, epoch: int, *, batch_offset: int = 0) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if batch_offset < 0 or batch_offset > len(self._batch_sampler):
            raise ValueError(f"batch_offset must be in [0, {len(self._batch_sampler)}]")
        self._epoch = epoch
        self._next_batch_offset = batch_offset


class _BatchOffsetSampler(torch.utils.data.Sampler[list[int]]):
    """Skip complete sampler batches before DataLoader dispatches indices to workers."""

    def __init__(self, batch_sampler: torch.utils.data.BatchSampler):
        self._batch_sampler = batch_sampler
        self._batch_offset = 0

    def __iter__(self):
        # Snapshot the offset now, but defer consuming the sampler until
        # DataLoader asks for its first index batch. This preserves DataLoader's
        # generator draw for worker base seeds before RandomSampler draws its
        # permutation, exactly matching an uninterrupted iterator.
        return _BatchOffsetIterator(iter(self._batch_sampler), self._batch_offset)

    def __len__(self) -> int:
        # Report the complete epoch length. The one-time resume offset changes
        # only the first iterator, not the training schedule's epoch geometry.
        return len(self._batch_sampler)

    def set_batch_offset(self, batch_offset: int) -> None:
        if batch_offset < 0 or batch_offset > len(self):
            raise ValueError(f"batch_offset must be in [0, {len(self)}]")
        self._batch_offset = batch_offset


class _BatchOffsetIterator(Iterator[list[int]]):
    def __init__(self, batch_iterator: Iterator[list[int]], batch_offset: int):
        self._batch_iterator = batch_iterator
        self._remaining_offset = batch_offset

    def __iter__(self):
        return self

    def __next__(self) -> list[int]:
        while self._remaining_offset:
            next(self._batch_iterator)
            self._remaining_offset -= 1
        return next(self._batch_iterator)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(
        self,
        data_config: _config.DataConfig,
        data_loader: TorchDataLoader | RLDSDataLoader,
        *,
        split_metadata: dict | None = None,
    ):
        self._data_config = data_config
        self._data_loader = data_loader
        self._split_metadata = split_metadata

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def split_metadata(self) -> dict | None:
        return dict(self._split_metadata) if self._split_metadata is not None else None

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]

    def __len__(self) -> int:
        return len(self._data_loader)

    def set_epoch(self, epoch: int, *, batch_offset: int = 0) -> None:
        if not hasattr(self._data_loader, "set_epoch"):
            raise TypeError(f"{type(self._data_loader).__name__} does not support epoch control")
        self._data_loader.set_epoch(epoch, batch_offset=batch_offset)
