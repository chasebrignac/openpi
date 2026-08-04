import dataclasses

import jax
import numpy as np
import polars as pl
import pytest
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_episode_prompt_prefers_annotation_and_falls_back():
    transform = _data_loader.EpisodePromptTransform({7: "annotated task"}, {3: "standard task"})
    annotated = transform({"episode_index": np.array([7]), "task_index": np.array([3])})
    fallback = transform({"episode_index": np.array([8]), "task_index": np.array([3])})
    assert annotated["prompt"] == "annotated task"
    assert fallback["prompt"] == "standard task"


def test_load_episode_prompts_rejects_conflicts(tmp_path):
    path = tmp_path / "tasks.parquet"
    pl.DataFrame({"episode_index": [1, 1], "task": ["one", "different"]}).write_parquet(path)
    with pytest.raises(ValueError, match="Conflicting"):
        _data_loader.load_episode_prompts(path)


def test_normalize_lerobot_tasks_supports_v2_mapping_and_v3_dataframe():
    assert _data_loader.normalize_lerobot_tasks({3: "standard task"}) == {3: "standard task"}
    frame = pl.DataFrame({"task": ["pick banana", "stack cubes"], "task_index": [7, 9]}).to_pandas()
    frame = frame.set_index("task")
    assert _data_loader.normalize_lerobot_tasks(frame) == {7: "pick banana", 9: "stack cubes"}


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_advances_distributed_sampler_epoch(monkeypatch):
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
    )
    epochs = []
    original_set_epoch = sampler.set_epoch

    def record_epoch(epoch):
        epochs.append(epoch)
        original_set_epoch(epoch)

    monkeypatch.setattr(sampler, "set_epoch", record_epoch)
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=2,
        sampler=sampler,
        num_batches=5,
        framework="pytorch",
    )
    loader.set_epoch(7)

    assert len(loader) == 5
    assert len(list(loader)) == 5
    assert epochs == [7, 8, 9]


def test_torch_data_loader_resume_reproduces_single_gpu_shuffle_epoch():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    uninterrupted = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        shuffle=True,
        seed=73,
        framework="pytorch",
    )
    uninterrupted_iter = iter(uninterrupted)
    uninterrupted_batches = [next(uninterrupted_iter) for _ in range(8)]

    resumed = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        shuffle=True,
        seed=73,
        framework="pytorch",
    )
    resumed.set_epoch(1)
    resumed_iter = iter(resumed)
    resumed_batches = [next(resumed_iter) for _ in range(4)]

    for expected, actual in zip(uninterrupted_batches[4:], resumed_batches, strict=True):
        assert all(
            torch.equal(left, right)
            for left, right in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True)
        )


def test_normalize_episode_records_supports_lerobot_v2_and_v3():
    v2 = {
        7: {"episode_index": 7, "length": 4, "tasks": ["pick banana"], "site": "lab-a"},
        9: {"episode_index": 9, "length": 5, "task_index": 3, "site": "lab-b"},
    }
    v3 = [
        {"episode_index": 7, "dataset_from_index": 0, "dataset_to_index": 4, "tasks": ["pick banana"]},
        {"episode_index": 9, "dataset_from_index": 4, "dataset_to_index": 9, "task_index": 3},
    ]

    v2_records = _data_loader.normalize_episode_records(v2, standard_tasks={3: "stack cubes"})
    v3_records = _data_loader.normalize_episode_records(v3, standard_tasks={3: "stack cubes"})

    assert [(record.episode_id, record.frames, record.task) for record in v2_records] == [
        (7, 4, "pick banana"),
        (9, 5, "stack cubes"),
    ]
    assert [(record.episode_id, record.frames, record.task) for record in v3_records] == [
        (7, 4, "pick banana"),
        (9, 5, "stack cubes"),
    ]
    assert [record.site for record in v2_records] == ["lab-a", "lab-b"]


def test_offline_holdout_is_deterministic_stratified_and_episode_disjoint():
    records = tuple(
        _data_loader.EpisodeRecord(episode_id=index, frames=3, task=f"task-{index % 3}", site=f"site-{index % 2}")
        for index in range(12)
    )

    first = _data_loader.select_offline_episode_split(records, holdout_samples=8, seed=42)
    second = _data_loader.select_offline_episode_split(records, holdout_samples=8, seed=42)

    assert first == second
    assert first.validation_frames >= 8
    assert set(first.train_episode_ids).isdisjoint(first.validation_episode_ids)
    assert set(first.train_episode_ids) | set(first.validation_episode_ids) == set(range(12))
    validation_strata = {
        (records[episode_id].task, records[episode_id].site) for episode_id in first.validation_episode_ids
    }
    assert len(validation_strata) == len(first.validation_episode_ids)
    assert first.metadata("validation")["validation_episode_ids"] == list(first.validation_episode_ids)


def test_offline_holdout_rejects_nonpositive_or_oversized_split():
    records = (
        _data_loader.EpisodeRecord(episode_id=0, frames=4),
        _data_loader.EpisodeRecord(episode_id=1, frames=4),
    )
    with pytest.raises(ValueError, match="must be positive"):
        _data_loader.select_offline_episode_split(records, holdout_samples=0, seed=0)
    with pytest.raises(ValueError, match="Cannot reserve"):
        _data_loader.select_offline_episode_split(records, holdout_samples=5, seed=0)


def test_create_torch_dataset_filters_episodes_before_action_deltas(monkeypatch):
    captured = {}

    class FakeMetadata:
        def __init__(self):
            self.info = {"codebase_version": "v3.0"}
            self.fps = 10
            self.tasks = {0: "pick banana"}

    class FakeLeRobotDataset:
        def __init__(self, _repo_id, **kwargs):
            captured.update(kwargs)

        def __len__(self):
            return 0

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", FakeLeRobotDataset)
    data_config = _config.DataConfig(repo_id="test/repo", prompt_from_task=False)
    model_config = pi0_config.Pi0Config(action_dim=8, action_horizon=3, max_token_len=16)

    _data_loader.create_torch_dataset(
        data_config,
        action_horizon=3,
        model_config=model_config,
        episodes=(5, 11),
        dataset_meta=FakeMetadata(),
    )

    assert captured["episodes"] == [5, 11]
    assert captured["video_backend"] == "pyav"
    assert captured["delta_timestamps"] == {"actions": [0.0, 0.1, 0.2]}


def test_lerobot_v2_uses_whole_episode_view_without_noncontiguous_constructor_ids(monkeypatch):
    captured = {}

    class FakeMetadata:
        def __init__(self):
            self.info = {"codebase_version": "v2.0"}
            self.fps = 10
            self.tasks = {0: "pick banana"}

    class FakeLeRobotDataset:
        def __init__(self, _repo_id, **kwargs):
            captured.update(kwargs)

        def __getitem__(self, index):
            return index

        def __len__(self):
            return 9

    records = (
        _data_loader.EpisodeRecord(episode_id=5, frames=3, start_index=0),
        _data_loader.EpisodeRecord(episode_id=11, frames=2, start_index=3),
        _data_loader.EpisodeRecord(episode_id=19, frames=4, start_index=5),
    )
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", FakeLeRobotDataset)

    dataset = _data_loader.create_torch_dataset(
        _config.DataConfig(repo_id="test/repo", prompt_from_task=False),
        action_horizon=3,
        model_config=pi0_config.Pi0Config(action_dim=8, action_horizon=3, max_token_len=16),
        episodes=(5, 19),
        episode_records=records,
        dataset_meta=FakeMetadata(),
    )

    assert captured["episodes"] is None
    assert captured["video_backend"] == "pyav"
    assert [dataset[index] for index in range(len(dataset))] == [0, 1, 2, 5, 6, 7, 8]


def test_create_torch_data_loader_passes_seed_to_distributed_sampler(monkeypatch):
    captured = {}
    sampler_class = torch.utils.data.distributed.DistributedSampler

    class RecordingDistributedSampler(sampler_class):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.utils.data.distributed, "DistributedSampler", RecordingDistributedSampler)

    _data_loader.create_torch_data_loader(
        _config.DataConfig(repo_id="fake"),
        pi0_config.Pi0Config(action_dim=8, action_horizon=3, max_token_len=16),
        action_horizon=3,
        batch_size=4,
        framework="pytorch",
        skip_norm_stats=True,
        seed=917,
    )

    assert captured["seed"] == 917


def test_fake_offline_holdout_exposes_episode_provenance():
    loader = _data_loader.create_torch_data_loader(
        _config.DataConfig(repo_id="fake"),
        pi0_config.Pi0Config(action_dim=8, action_horizon=3, max_token_len=16),
        action_horizon=3,
        batch_size=2,
        framework="pytorch",
        skip_norm_stats=True,
        holdout_samples=3,
        split="validation",
        seed=23,
    )

    metadata = loader.split_metadata()
    assert metadata is not None
    assert metadata["strategy"] == "deterministic_whole_episode_stratified"
    assert metadata["validation_episode_count"] == 3
    assert metadata["selected_episode_count"] == 3


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_fake_dataset_has_bchw_images_for_pytorch():
    config = dataclasses.replace(_config.get_config("debug"), batch_size=2, num_workers=0)
    loader = _data_loader.create_data_loader(
        config,
        framework="pytorch",
        skip_norm_stats=True,
        num_batches=1,
        shuffle=False,
    )
    observation, _actions = next(iter(loader))
    assert all(image.shape[:2] == (2, 3) for image in observation.images.values())


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
