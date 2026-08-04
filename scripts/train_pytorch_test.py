import dataclasses
import json
import types

import numpy as np
import pytest
import torch

from openpi.training import config as training_config
from openpi.training import data_loader as training_data_loader
from scripts import train_pytorch


@dataclasses.dataclass(frozen=True)
class Schedule:
    warmup_steps: int = 500
    peak_lr: float = 2.5e-5


@dataclasses.dataclass(frozen=True)
class Optimizer:
    b1: float = 0.9


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    layers: int = 9
    dtype: str = "bfloat16"


@dataclasses.dataclass(frozen=True)
class FactorySpec:
    repo_id: str = "physical-intelligence/libero"


@dataclasses.dataclass(frozen=True)
class SaveConfig:
    checkpoint_dir: object
    name: str = "debug"
    exp_name: str = "immutable"
    seed: int = 42
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    lr_schedule: Schedule = dataclasses.field(default_factory=Schedule)
    optimizer: Optimizer = dataclasses.field(default_factory=Optimizer)
    wandb_enabled: bool = False
    num_train_steps: int = 1
    save_interval: int = 1
    one_batch_overfit: bool = False


def _config(checkpoint_base_dir, *, exp_name="immutable", resume=False, overwrite=False):
    return training_config.TrainConfig(
        name="debug",
        exp_name=exp_name,
        checkpoint_base_dir=str(checkpoint_base_dir),
        resume=resume,
        overwrite=overwrite,
        wandb_enabled=False,
    )


def test_setup_ddp_binds_cuda_device_before_nccl_initialization(monkeypatch):
    events = []
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.delenv("TORCH_DISTRIBUTED_DEBUG", raising=False)
    monkeypatch.setattr(train_pytorch.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train_pytorch.torch.cuda, "set_device", lambda device: events.append(("set_device", device)))
    monkeypatch.setattr(train_pytorch.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        train_pytorch.torch.distributed,
        "init_process_group",
        lambda **kwargs: events.append(("init_process_group", kwargs)),
    )

    use_ddp, local_rank, device = train_pytorch.setup_ddp()

    assert use_ddp is True
    assert local_rank == 1
    assert device == torch.device("cuda:1")
    assert events == [
        ("set_device", torch.device("cuda:1")),
        (
            "init_process_group",
            {"backend": "nccl", "init_method": "env://", "device_id": torch.device("cuda:1")},
        ),
    ]
    assert train_pytorch.os.environ["TORCH_DISTRIBUTED_DEBUG"] == "INFO"


def test_ddp_barrier_passes_current_cuda_device(monkeypatch):
    calls = []
    monkeypatch.setattr(train_pytorch.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train_pytorch.torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(train_pytorch.torch.distributed, "barrier", lambda **kwargs: calls.append(kwargs))

    train_pytorch.ddp_barrier()

    assert calls == [{"device_ids": [1]}]


def test_checkpoint_overwrite_is_owned_by_rank_zero_and_barriered(tmp_path, monkeypatch):
    checkpoint_base_dir = tmp_path / "runs"
    checkpoint_dir = checkpoint_base_dir / "debug" / "immutable"
    checkpoint_dir.mkdir(parents=True)
    stale = checkpoint_dir / "stale"
    stale.write_text("old")
    barriers = []
    monkeypatch.setattr(train_pytorch.dist, "barrier", lambda: barriers.append(True))

    resuming = train_pytorch.prepare_checkpoint_directory(
        _config(checkpoint_base_dir, overwrite=True),
        is_main=True,
        use_ddp=True,
    )

    assert not resuming
    assert checkpoint_dir.is_dir()
    assert not stale.exists()
    assert barriers == [True]


def test_fresh_run_refuses_existing_directory_without_overwrite(tmp_path):
    checkpoint_base_dir = tmp_path / "runs"
    checkpoint_dir = checkpoint_base_dir / "debug" / "immutable"
    checkpoint_dir.mkdir(parents=True)
    sentinel = checkpoint_dir / "sentinel"
    sentinel.write_text("keep")

    with pytest.raises(FileExistsError, match="unique --exp-name"):
        train_pytorch.prepare_checkpoint_directory(
            _config(checkpoint_base_dir),
            is_main=True,
            use_ddp=False,
        )

    assert sentinel.read_text() == "keep"


def test_non_main_rank_never_mutates_checkpoint_directory(tmp_path, monkeypatch):
    checkpoint_base_dir = tmp_path / "runs"
    checkpoint_dir = checkpoint_base_dir / "debug" / "immutable"
    checkpoint_dir.mkdir(parents=True)
    sentinel = checkpoint_dir / "sentinel"
    sentinel.write_text("keep")
    barriers = []
    monkeypatch.setattr(train_pytorch.dist, "barrier", lambda: barriers.append(True))

    resuming = train_pytorch.prepare_checkpoint_directory(
        _config(checkpoint_base_dir, overwrite=True),
        is_main=False,
        use_ddp=True,
    )

    assert not resuming
    assert sentinel.read_text() == "keep"
    assert barriers == [True]


def test_resume_validates_numeric_checkpoint_without_mutating(tmp_path):
    checkpoint_base_dir = tmp_path / "runs"
    checkpoint_dir = checkpoint_base_dir / "debug" / "immutable"
    numeric_checkpoint = checkpoint_dir / "2000"
    numeric_checkpoint.mkdir(parents=True)
    sentinel = numeric_checkpoint / "model.safetensors"
    sentinel.write_text("keep")

    resuming = train_pytorch.prepare_checkpoint_directory(
        _config(checkpoint_base_dir, resume=True),
        is_main=True,
        use_ddp=False,
    )

    assert resuming
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize("exp_name", ["/tmp/escape", "../escape", "nested/name", "..", "run..escape"])
def test_checkpoint_dir_rejects_unsafe_experiment_names(tmp_path, exp_name):
    config = _config(tmp_path / "runs", exp_name=exp_name)

    with pytest.raises(ValueError, match="--exp-name"):
        _ = config.checkpoint_dir


@pytest.mark.parametrize("exp_name", ["run-01", "Run_20260804T120000Z-a1", "pilot.v2"])
def test_checkpoint_dir_accepts_safe_experiment_names_as_strict_descendants(tmp_path, exp_name):
    checkpoint_base_dir = tmp_path / "runs"
    config = _config(checkpoint_base_dir, exp_name=exp_name)

    assert config.checkpoint_dir == checkpoint_base_dir.resolve() / "debug" / exp_name
    assert config.checkpoint_dir.parent == checkpoint_base_dir.resolve() / "debug"


@pytest.mark.parametrize("link_location", ["base", "config", "experiment"])
def test_checkpoint_overwrite_rejects_symlink_escape_without_deleting_target(tmp_path, link_location):
    checkpoint_base_dir = tmp_path / "runs"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep"
    sentinel.write_text("safe")

    config_dir = checkpoint_base_dir / "debug"
    if link_location == "base":
        checkpoint_base_dir.symlink_to(outside, target_is_directory=True)
    elif link_location == "experiment":
        config_dir.mkdir(parents=True)
        (config_dir / "immutable").symlink_to(outside, target_is_directory=True)
    else:
        checkpoint_base_dir.mkdir()
        config_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        train_pytorch.prepare_checkpoint_directory(
            _config(checkpoint_base_dir, overwrite=True),
            is_main=True,
            use_ddp=False,
        )

    assert sentinel.read_text() == "safe"


def test_resume_skips_source_checkpoint_initialization(monkeypatch):
    calls = []
    monkeypatch.setattr(train_pytorch.safetensors.torch, "load_model", lambda *args, **kwargs: calls.append(kwargs))

    loaded = train_pytorch.load_initial_pytorch_weights(
        torch.nn.Linear(2, 2),
        object(),
        "/source/run-root-without-model-file",
        resuming=True,
    )

    assert not loaded
    assert calls == []


def test_resume_state_binds_step_scheduler_optimizer_and_batch_contract(tmp_path):
    checkpoint = tmp_path / "2000"
    checkpoint.mkdir()
    contract = {
        "schema_version": 1,
        "config_name": "pi05_libero_l09_distill",
        "exp_name": "libero-shallow",
        "seed": 7,
        "batch_size": 8,
        "gradient_accumulation_steps": 8,
        "lr_schedule": dataclasses.asdict(Schedule()),
        "optimizer": dataclasses.asdict(Optimizer()),
        "wandb_enabled": True,
        "model": {"class": "test.Model", "config": {"layers": 9}},
        "pytorch_training_precision": "bfloat16",
        "dataset": {
            "factory": "test.Factory",
            "factory_config_sha256": "f" * 64,
            "repo_id": "physical-intelligence/libero",
            "revision": "a" * 40,
            "codebase_version": "v2.0",
            "episode_prompt_path": None,
            "episode_prompt_sha256": None,
            "action_sequence_keys": ["actions"],
            "asset_id": "physical-intelligence/libero",
            "use_quantile_norm": True,
            "prompt_from_task": True,
            "normalization_sha256": "c" * 64,
            "recovery_provenance_sha256": None,
        },
        "teacher": {"model_sha256": "b" * 64},
        "data_split": {"seed": 7, "validation_episode_ids": [3]},
        "stochastic_schedule": "sha256-v2(model:seed,step,accumulation,rank;loader:seed,epoch)",
        "one_batch_overfit": False,
        "one_batch_overfit_min_relative_decline": 0.2,
    }
    state = {
        "schema_version": 2,
        "global_step": 2000,
        "config_name": contract["config_name"],
        "exp_name": contract["exp_name"],
        "resume_contract": contract,
        "resume_fingerprint_sha256": train_pytorch.resume_identity_sha256(
            contract, {"kind": "shallow_teacher_transplant", "model_sha256": "b" * 64}
        ),
        "initialization_lineage": {"kind": "shallow_teacher_transplant", "model_sha256": "b" * 64},
        "state_files": ["metadata.pt", "model.safetensors", "optimizer.pt", "wandb_id.txt"],
    }
    (checkpoint / "resume-state.json").write_text(json.dumps(state))
    assert train_pytorch.validate_resume_state(checkpoint, contract, 2000) == state

    changed_contract = {**contract, "lr_schedule": {"warmup_steps": 0, "peak_lr": 1.0}}
    with pytest.raises(ValueError, match="lr_schedule"):
        train_pytorch.validate_resume_state(checkpoint, changed_contract, 2000)

    tampered = {**state, "resume_contract": {**contract, "seed": 8}}
    (checkpoint / "resume-state.json").write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="fingerprint"):
        train_pytorch.validate_resume_state(checkpoint, contract, 2000)


def test_resume_state_rejects_malformed_initialization_lineage(tmp_path):
    checkpoint = tmp_path / "2000"
    checkpoint.mkdir()
    contract = {"config_name": "debug", "exp_name": "resume"}
    state = {
        "schema_version": 2,
        "global_step": 2000,
        "config_name": "debug",
        "exp_name": "resume",
        "resume_contract": contract,
        "resume_fingerprint_sha256": train_pytorch.resume_identity_sha256(
            contract, {"kind": "pytorch_source", "model_sha256": None}
        ),
        "initialization_lineage": {"kind": "pytorch_source", "model_sha256": None},
        "state_files": ["metadata.pt", "model.safetensors", "optimizer.pt", "wandb_id.txt"],
    }
    (checkpoint / "resume-state.json").write_text(json.dumps(state))

    with pytest.raises(ValueError, match="lineage"):
        train_pytorch.validate_resume_state(checkpoint, contract, 2000)


def test_resume_data_position_skips_consumed_microbatches_within_epoch():
    assert train_pytorch.resume_data_position(2000, 8, 10_000) == (1, 6000)


def test_nonfinite_training_values_fail_before_backward():
    train_pytorch.require_finite_training_values(torch.tensor(1.0), {"kd_cosine": 0.999})
    with pytest.raises(FloatingPointError, match="loss"):
        train_pytorch.require_finite_training_values(torch.tensor(float("nan")))
    with pytest.raises(FloatingPointError, match="diagnostics"):
        train_pytorch.require_finite_training_values(torch.tensor(1.0), {"kd_cosine": float("inf")})


def test_one_batch_overfit_gate_is_explicit_and_fail_closed():
    report = train_pytorch.evaluate_one_batch_overfit(
        [1.0] * 20 + [0.7] * 20,
        minimum_relative_decline=0.2,
    )
    assert report["gate_pass"]
    assert report["relative_loss_decline"] == pytest.approx(0.3)
    assert report["optimizer_step_losses"] == [1.0] * 20 + [0.7] * 20
    with pytest.raises(RuntimeError, match="below required"):
        train_pytorch.evaluate_one_batch_overfit(
            [1.0] * 20 + [0.9] * 20,
            minimum_relative_decline=0.2,
        )
    with pytest.raises(ValueError, match="at least 40"):
        train_pytorch.evaluate_one_batch_overfit([1.0, 0.7], minimum_relative_decline=0.2)


def test_one_batch_mode_requires_explicit_zero_loader_workers():
    config = types.SimpleNamespace(
        one_batch_overfit=True,
        resume=False,
        num_train_steps=300,
        one_batch_overfit_min_relative_decline=0.2,
        num_workers=0,
    )
    train_pytorch.validate_training_mode(config)

    config.num_workers = 4
    with pytest.raises(ValueError, match="--num-workers 0"):
        train_pytorch.validate_training_mode(config)


def test_disabled_wandb_never_constructs_a_sample_loader(monkeypatch):
    config = types.SimpleNamespace(wandb_enabled=False, one_batch_overfit=True)
    monkeypatch.setattr(
        training_data_loader,
        "create_data_loader",
        lambda *_args, **_kwargs: pytest.fail("disabled W&B constructed a sample loader"),
    )

    assert (
        train_pytorch.materialize_wandb_sample_batch(
            config,
            is_main=True,
            resuming=False,
            overfit_batch=object(),
        )
        is None
    )


def test_one_batch_wandb_reuses_the_materialized_training_batch(monkeypatch):
    config = types.SimpleNamespace(wandb_enabled=True, one_batch_overfit=True)
    batch = object()
    monkeypatch.setattr(
        training_data_loader,
        "create_data_loader",
        lambda *_args, **_kwargs: pytest.fail("one-batch W&B constructed a second loader"),
    )

    assert (
        train_pytorch.materialize_wandb_sample_batch(
            config,
            is_main=True,
            resuming=False,
            overfit_batch=batch,
        )
        is batch
    )


def test_counter_seed_is_resume_stable_and_overfit_locks_every_microstep():
    uninterrupted = train_pytorch.training_microstep_seed(42, 2000, 3, 1, one_batch_overfit=False)
    resumed = train_pytorch.training_microstep_seed(42, 2000, 3, 1, one_batch_overfit=False)
    assert uninterrupted == resumed
    assert uninterrupted != train_pytorch.training_microstep_seed(42, 2001, 3, 1, one_batch_overfit=False)
    assert uninterrupted != train_pytorch.training_microstep_seed(42, 2000, 3, 0, one_batch_overfit=False)
    assert train_pytorch.training_microstep_seed(42, 0, 0, 1, one_batch_overfit=True) == (
        train_pytorch.training_microstep_seed(42, 299, 7, 1, one_batch_overfit=True)
    )


def test_counter_seeded_rng_repeats_augmentation_randomness_without_leaking_state():
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    with train_pytorch.counter_seeded_rng(99, torch.device("cpu")):
        first = torch.rand(5)
    assert torch.equal(torch.random.get_rng_state(), before)
    with train_pytorch.counter_seeded_rng(99, torch.device("cpu")):
        second = torch.rand(5)
    assert torch.equal(first, second)


def test_resume_fingerprint_binds_teacher_model_dataset_precision_and_normalization():
    config = types.SimpleNamespace(
        name="pi05_libero_l09_distill",
        exp_name="libero-shallow",
        seed=42,
        batch_size=8,
        gradient_accumulation_steps=8,
        lr_schedule=Schedule(),
        optimizer=Optimizer(),
        wandb_enabled=False,
        pytorch_training_precision="bfloat16",
        one_batch_overfit=False,
        one_batch_overfit_min_relative_decline=0.2,
        data=FactorySpec(),
    )
    data_config = types.SimpleNamespace(
        repo_id="physical-intelligence/libero",
        lerobot_revision="a" * 40,
        lerobot_codebase_version="v2.0",
        episode_prompt_path=None,
        action_sequence_keys=("actions",),
        asset_id="physical-intelligence/libero",
        use_quantile_norm=True,
        prompt_from_task=True,
        norm_stats={"actions": {"mean": np.array([0.0]), "std": np.array([1.0])}},
        recovery_provenance=None,
    )
    contract = train_pytorch.build_resume_contract(
        config,
        ModelSpec(),
        data_config,
        {"validation_episode_ids": [3]},
        teacher_model_sha256="b" * 64,
    )
    baseline = train_pytorch.canonical_sha256(contract)
    for changed in (
        {**contract, "teacher": {"model_sha256": "c" * 64}},
        {**contract, "pytorch_training_precision": "float32"},
        {**contract, "model": {"class": "test.Model", "config": {"layers": 8}}},
        {**contract, "dataset": {**contract["dataset"], "revision": "d" * 40}},
        {**contract, "dataset": {**contract["dataset"], "normalization_sha256": "e" * 64}},
    ):
        assert train_pytorch.canonical_sha256(changed) != baseline


def test_checkpoint_publication_is_immutable_and_contains_resume_contract(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    config = SaveConfig(checkpoint_dir=tmp_path)
    data_config = types.SimpleNamespace(recovery_provenance=None, norm_stats=None, asset_id=None)
    contract = {"schema_version": 1, "config_name": config.name, "exp_name": config.exp_name}
    lineage = {"kind": "random_initialization", "model_sha256": None}

    train_pytorch.save_checkpoint(
        model=model,
        optimizer=optimizer,
        global_step=1,
        config=config,
        is_main=True,
        data_config=data_config,
        data_split_metadata=None,
        resume_contract=contract,
        initialization_lineage=lineage,
        training_metrics={"loss": 0.25, "kd_cosine": 0.999},
    )
    checkpoint = tmp_path / "1"
    assert checkpoint.is_dir()
    assert not (tmp_path / "tmp_1").exists()
    state = json.loads((checkpoint / "resume-state.json").read_text())
    assert state["resume_contract"] == contract
    assert state["resume_fingerprint_sha256"] == train_pytorch.resume_identity_sha256(contract, lineage)
    metrics = json.loads((checkpoint / "training-metrics.json").read_text())
    assert metrics == {
        "schema_version": 1,
        "config_name": config.name,
        "exp_name": config.exp_name,
        "global_step": 1,
        "metrics": {"kd_cosine": 0.999, "loss": 0.25},
    }

    with pytest.raises(FileExistsError, match="immutable"):
        train_pytorch.save_checkpoint(
            model=model,
            optimizer=optimizer,
            global_step=1,
            config=config,
            is_main=True,
            data_config=data_config,
            data_split_metadata=None,
            resume_contract=contract,
            initialization_lineage=lineage,
        )


def test_checkpoint_refuses_non_finite_training_metrics(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    config = SaveConfig(checkpoint_dir=tmp_path)
    data_config = types.SimpleNamespace(recovery_provenance=None, norm_stats=None, asset_id=None)

    with pytest.raises(ValueError, match="finite JSON values"):
        train_pytorch.save_checkpoint(
            model=model,
            optimizer=optimizer,
            global_step=1,
            config=config,
            is_main=True,
            data_config=data_config,
            data_split_metadata=None,
            resume_contract={"schema_version": 1},
            initialization_lineage={"kind": "random_initialization"},
            training_metrics={"loss": float("nan")},
        )
    assert not (tmp_path / "1").exists()


def test_training_split_metadata_binds_seed_and_complete_validation_episodes():
    metadata = {
        "schema_version": 1,
        "strategy": "deterministic_whole_episode_stratified",
        "split": "train",
        "seed": 42,
        "requested_holdout_samples": 256,
        "validation_episode_ids": [3, 9],
        "validation_episode_count": 2,
    }
    loader = types.SimpleNamespace(split_metadata=lambda: metadata)
    config = types.SimpleNamespace(offline_holdout_samples=256, seed=42)
    assert train_pytorch.validate_training_split_metadata(config, loader) == metadata

    loader = types.SimpleNamespace(split_metadata=lambda: {**metadata, "seed": 7})
    with pytest.raises(ValueError, match="whole-episode"):
        train_pytorch.validate_training_split_metadata(config, loader)
