"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp-name <run_name> --save-interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp-name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp-name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc-per-node=<num_gpus> scripts/train_pytorch.py <config_name> --exp-name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc-per-node=2 scripts/train_pytorch.py pi0_aloha_sim --exp-name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc-per-node=2 scripts/train_pytorch.py pi0_aloha_sim --exp-name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc-per-node=<gpus_per_node> --node-rank=<rank_of_node> \
    --master-addr=<master_ip> --master-port=<port> \
    scripts/train_pytorch.py <config_name> --exp-name=<run_name> --save-interval <interval>

"""

from collections.abc import Iterator, Mapping
import contextlib
import dataclasses
import enum
import gc
import hashlib
import itertools
import json
import logging
import math
import os
import pathlib
import platform
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.models_pytorch.shallow_pi
import openpi.models_pytorch.snapflow
import openpi.shared.normalize as _normalize
from openpi.training import robolab_expert_dataset as _robolab_expert
import openpi.training.config as _config
import openpi.training.data_loader as _data


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id_path = ckpt_dir / "wandb_id.txt"
        if not run_id_path.exists():
            numeric = sorted(
                (path for path in ckpt_dir.iterdir() if path.is_dir() and path.name.isdigit()),
                key=lambda path: int(path.name),
            )
            if not numeric or not (numeric[-1] / "wandb_id.txt").is_file():
                raise FileNotFoundError("Resume checkpoint has no persisted W&B run ID")
            run_id_path.write_text((numeric[-1] / "wandb_id.txt").read_text())
        run_id = run_id_path.read_text().strip()
        if not run_id:
            raise ValueError("Resume checkpoint W&B run ID is empty")
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Bind the process to its CUDA device before NCCL creates any communicator.
    # Otherwise ProcessGroupNCCL has to guess the rank-to-device mapping during
    # its first barrier and can initialize collectives on the wrong device.
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        # This must be set before process-group construction to affect NCCL's
        # diagnostic setup rather than only later collectives.
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
        kwargs = {"backend": backend, "init_method": "env://"}
        if backend == "nccl":
            kwargs["device_id"] = device
        torch.distributed.init_process_group(**kwargs)
    return use_ddp, local_rank, device


def ddp_barrier():
    if torch.cuda.is_available():
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    else:
        torch.distributed.barrier()


def cleanup_ddp():
    if torch.distributed.is_initialized():
        ddp_barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(
        config,
        framework="pytorch",
        shuffle=not config.one_batch_overfit,
        num_batches=1 if config.one_batch_overfit else None,
    )
    return data_loader, data_loader.data_config()


def validate_training_mode(config: _config.TrainConfig) -> None:
    """Fail before any checkpoint or external-logging mutation for an unsafe diagnostic config."""
    if config.one_batch_overfit and config.resume:
        raise ValueError("one-batch diagnostics cannot resume because their complete loss history is part of the gate")
    if config.one_batch_overfit and config.num_train_steps < 40:
        raise ValueError("one-batch diagnostics require at least 40 optimizer steps for first/last-20 windows")
    if config.one_batch_overfit and config.one_batch_overfit_min_relative_decline < 0.20:
        raise ValueError("one-batch diagnostics require at least a 20% relative loss-decline gate")
    if config.one_batch_overfit and config.num_workers != 0:
        raise ValueError("one-batch diagnostics require explicit --num-workers 0 to avoid dataset worker duplication")


def materialize_wandb_sample_batch(
    config: _config.TrainConfig,
    *,
    is_main: bool,
    resuming: bool,
    overfit_batch,
):
    """Return a W&B image batch without duplicating a one-batch diagnostic loader."""
    if not is_main or not config.wandb_enabled or resuming:
        return None
    if config.one_batch_overfit:
        if overfit_batch is None:
            raise ValueError("one-batch W&B sampling requires the already materialized training batch")
        return overfit_batch
    sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
    return next(iter(sample_data_loader))


def validate_training_split_metadata(config: _config.TrainConfig, loader) -> dict | None:
    metadata = loader.split_metadata()
    if not config.offline_holdout_samples:
        if metadata is not None:
            raise ValueError("training loader unexpectedly exposed an offline split")
        return None
    episode_ids = metadata.get("validation_episode_ids") if isinstance(metadata, dict) else None
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version") != 1
        or metadata.get("strategy") != "deterministic_whole_episode_stratified"
        or metadata.get("split") != "train"
        or metadata.get("seed") != config.seed
        or metadata.get("requested_holdout_samples") != config.offline_holdout_samples
        or not isinstance(episode_ids, list)
        or not episode_ids
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in episode_ids)
        or len(episode_ids) != len(set(episode_ids))
        or metadata.get("validation_episode_count") != len(episode_ids)
    ):
        raise ValueError("training loader whole-episode split metadata is invalid or differs from the config")
    return metadata


def deterministic_overfit_inputs(actions: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create fixed stochastic inputs for a repeat-one-batch optimizer smoke test."""
    generator = torch.Generator(device=actions.device)
    generator.manual_seed(seed)
    noise = torch.randn(actions.shape, dtype=torch.float32, device=actions.device, generator=generator)
    time = torch.linspace(0.05, 0.95, actions.shape[0], dtype=torch.float32, device=actions.device)
    flow_matching_mask = torch.arange(actions.shape[0], device=actions.device) % 2 == 0
    return noise, time, flow_matching_mask


def training_microstep_seed(
    seed: int,
    global_step: int,
    accumulation_index: int,
    rank: int,
    *,
    one_batch_overfit: bool,
) -> int:
    """Derive stochastic state from the resumable optimizer position.

    One-batch diagnostics intentionally omit the step and accumulation index so
    preprocessing augmentation is identical on every repeated microbatch.
    Normal training includes both values, making an interrupted run use the
    same stochastic inputs as an uninterrupted run without unsafe RNG pickles.
    """
    if seed < 0 or global_step < 0 or accumulation_index < 0 or rank < 0:
        raise ValueError("counter-derived training seeds require non-negative inputs")
    position = "fixed" if one_batch_overfit else f"{global_step}:{accumulation_index}"
    payload = f"openpi-pytorch-microstep-v1\0{seed}\0{position}\0{rank}".encode()
    # Torch accepts signed 64-bit seeds. Keep the high endpoint out so this is
    # portable across CPU and CUDA generators.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


@contextlib.contextmanager
def counter_seeded_rng(seed: int, device: torch.device) -> Iterator[None]:
    """Temporarily seed CPU and the current CUDA generator, restoring both."""
    cuda_devices = [device.index] if device.type == "cuda" and device.index is not None else []
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.random.default_generator.manual_seed(seed)
        if cuda_devices:
            torch.cuda.manual_seed(seed)
        yield


def require_finite_training_values(loss: torch.Tensor, metrics: dict[str, float] | None = None) -> None:
    """Abort every rank before backward when a loss or diagnostic is non-finite."""
    invalid_metrics = sorted(name for name, value in (metrics or {}).items() if not math.isfinite(float(value)))
    finite = torch.isfinite(loss.detach()).all() & torch.tensor(
        not invalid_metrics,
        dtype=torch.bool,
        device=loss.device,
    )
    if torch.distributed.is_initialized():
        finite_flag = finite.to(dtype=torch.int32)
        torch.distributed.all_reduce(finite_flag, op=torch.distributed.ReduceOp.MIN)
        finite = finite_flag.bool()
    if not bool(finite.item()):
        local_detail = f"; local invalid diagnostics={invalid_metrics}" if invalid_metrics else ""
        raise FloatingPointError(f"training loss or diagnostics are NaN or infinite on at least one rank{local_detail}")


def distributed_mean_loss(loss: torch.Tensor) -> float:
    """Return the equal-local-batch world mean used by the overfit gate."""
    value = loss.detach().float().mean()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
        value /= torch.distributed.get_world_size()
    return float(value.item())


def evaluate_one_batch_overfit(
    optimizer_step_losses: list[float],
    *,
    minimum_relative_decline: float,
    window: int = 20,
) -> dict:
    """Apply the objective optimizer-smoke gate and return its durable report."""
    if not 0.0 < minimum_relative_decline < 1.0:
        raise ValueError("minimum_relative_decline must be between zero and one")
    if window < 1:
        raise ValueError("one-batch diagnostic window must be positive")
    if len(optimizer_step_losses) < 2 * window:
        raise ValueError(f"one-batch diagnostic requires at least {2 * window} optimizer-step losses")
    if any(not math.isfinite(value) for value in optimizer_step_losses):
        raise FloatingPointError("one-batch diagnostic contains a non-finite loss")
    initial_mean = math.fsum(optimizer_step_losses[:window]) / window
    final_mean = math.fsum(optimizer_step_losses[-window:]) / window
    if initial_mean <= 0.0 or final_mean < 0.0:
        raise ValueError("one-batch MSE losses must have a positive initial mean and non-negative final mean")
    relative_decline = (initial_mean - final_mean) / initial_mean
    report = {
        "schema_version": 1,
        "gate_pass": relative_decline >= minimum_relative_decline,
        "optimizer_steps": len(optimizer_step_losses),
        "window_optimizer_steps": window,
        # The complete sequence is deliberately small for this bounded
        # diagnostic and lets the artifact verifier recompute the exact first
        # and last windows instead of trusting reported aggregate numbers.
        "optimizer_step_losses": list(optimizer_step_losses),
        "initial_loss_mean": initial_mean,
        "final_loss_mean": final_mean,
        "relative_loss_decline": relative_decline,
        "minimum_relative_loss_decline": minimum_relative_decline,
        "all_losses_finite": True,
    }
    if not report["gate_pass"]:
        raise RuntimeError(
            "one-batch optimizer diagnostic failed: "
            f"loss declined {relative_decline:.2%}, below required {minimum_relative_decline:.2%}"
        )
    return report


def _canonical_json_value(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonical_json_value(dataclasses.asdict(value))
    if isinstance(value, enum.Enum):
        return _canonical_json_value(value.value)
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _canonical_json_value(item) for key, item in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, list | tuple):
        return [_canonical_json_value(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"Cannot encode {type(value).__name__} in the resume contract")


def canonical_sha256(value) -> str:
    payload = json.dumps(
        _canonical_json_value(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def resume_identity_sha256(resume_contract: dict, initialization_lineage: dict) -> str:
    """Fingerprint every immutable input needed to interpret a continuation."""
    return canonical_sha256(
        {
            "initialization_lineage": initialization_lineage,
            "resume_contract": resume_contract,
        }
    )


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def dataset_sidecar_sha256(data_config, relative_or_absolute_path: str | None) -> str | None:
    if relative_or_absolute_path is None:
        return None
    path = pathlib.Path(relative_or_absolute_path)
    if not path.is_absolute():
        if data_config.lerobot_root is None:
            raise ValueError("A relative dataset sidecar path requires lerobot_root")
        path = pathlib.Path(data_config.lerobot_root) / path
    return sha256_file(path.expanduser().resolve())


def resolve_model_weights(path: str | pathlib.Path) -> pathlib.Path:
    result = pathlib.Path(path).expanduser().resolve()
    if result.is_dir():
        result = result / "model.safetensors"
    if not result.is_file():
        raise FileNotFoundError(f"Model weights are missing: {result}")
    return result


def build_resume_contract(
    config,
    model_cfg,
    data_config,
    data_split_metadata: dict | None,
    *,
    teacher_model_sha256: str | None,
) -> dict:
    """Build the canonical compatibility identity for a numeric checkpoint."""
    dataset_identity = {
        "factory": f"{type(config.data).__module__}.{type(config.data).__qualname__}",
        "factory_config_sha256": canonical_sha256(config.data),
        "repo_id": data_config.repo_id,
        "revision": data_config.lerobot_revision,
        "codebase_version": data_config.lerobot_codebase_version,
        "episode_prompt_path": data_config.episode_prompt_path,
        "episode_prompt_sha256": dataset_sidecar_sha256(data_config, data_config.episode_prompt_path),
        "action_sequence_keys": list(data_config.action_sequence_keys),
        "asset_id": data_config.asset_id,
        "use_quantile_norm": data_config.use_quantile_norm,
        "prompt_from_task": data_config.prompt_from_task,
        "normalization_sha256": (
            canonical_sha256(data_config.norm_stats) if data_config.norm_stats is not None else None
        ),
        "recovery_provenance_sha256": (
            canonical_sha256(data_config.recovery_provenance) if data_config.recovery_provenance is not None else None
        ),
    }
    contract = {
        "schema_version": 1,
        "config_name": config.name,
        "exp_name": config.exp_name,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "lr_schedule": dataclasses.asdict(config.lr_schedule),
        "optimizer": dataclasses.asdict(config.optimizer),
        "wandb_enabled": config.wandb_enabled,
        "model": {
            "class": f"{type(model_cfg).__module__}.{type(model_cfg).__qualname__}",
            "config": dataclasses.asdict(model_cfg),
        },
        "pytorch_training_precision": config.pytorch_training_precision,
        "dataset": dataset_identity,
        "teacher": ({"model_sha256": teacher_model_sha256} if teacher_model_sha256 is not None else None),
        "data_split": data_split_metadata,
        "stochastic_schedule": "sha256-v2(model:seed,step,accumulation,rank;loader:seed,epoch)",
        "one_batch_overfit": config.one_batch_overfit,
        "one_batch_overfit_min_relative_decline": config.one_batch_overfit_min_relative_decline,
    }
    reproduction_config = config.name.startswith(("pi05_libero_", "pi05_droid_"))
    if reproduction_config and (
        not isinstance(dataset_identity["revision"], str)
        or len(dataset_identity["revision"]) != 40
        or any(character not in "0123456789abcdef" for character in dataset_identity["revision"])
    ):
        raise ValueError("Reproduction training requires an immutable 40-character dataset revision")
    if reproduction_config and not _is_sha256(dataset_identity["normalization_sha256"]):
        raise ValueError("Reproduction training requires content-addressed normalization statistics")
    if config.name.endswith("_distill") and not _is_sha256(teacher_model_sha256):
        raise ValueError("Distillation training requires a SHA256-identified teacher")
    if not config.name.endswith("_distill") and teacher_model_sha256 is not None:
        raise ValueError("Non-distillation training must not declare a teacher")
    # Fail immediately if a supposedly canonical field is not JSON-safe.
    canonical_sha256(contract)
    return contract


def build_initialization_lineage(
    config,
    model_cfg,
    *,
    teacher_model_sha256: str | None,
) -> dict:
    if isinstance(model_cfg, openpi.models.pi0_config.DistilledPi0Config):
        assert teacher_model_sha256 is not None
        return {"kind": "shallow_teacher_transplant", "model_sha256": teacher_model_sha256}
    if config.pytorch_weight_path is not None:
        weights = resolve_model_weights(config.pytorch_weight_path)
        return {"kind": "pytorch_source", "model_sha256": sha256_file(weights)}
    return {"kind": "random_initialization", "model_sha256": None}


def validate_initialization_lineage(lineage) -> dict:
    """Validate the non-pickle identity of the weights that began a run."""
    if not isinstance(lineage, dict) or set(lineage) != {"kind", "model_sha256"}:
        raise ValueError("Resume checkpoint initialization lineage has unexpected fields")
    kind = lineage.get("kind")
    model_sha256 = lineage.get("model_sha256")
    if kind not in {"shallow_teacher_transplant", "pytorch_source", "random_initialization"}:
        raise ValueError("Resume checkpoint initialization lineage has an unknown kind")
    if kind == "random_initialization":
        if model_sha256 is not None:
            raise ValueError("Random initialization lineage must not name source weights")
    elif (
        not isinstance(model_sha256, str)
        or len(model_sha256) != 64
        or any(character not in "0123456789abcdef" for character in model_sha256)
    ):
        raise ValueError("Resume checkpoint initialization lineage lacks a valid model SHA256")
    return lineage


def require_finite_checkpoint_state(model, optimizer) -> None:
    """Reject non-finite parameters, buffers, or optimizer tensors before publication."""
    model_state = get_model_state_dict(model)
    invalid_model = [
        name
        for name, value in model_state.items()
        if isinstance(value, torch.Tensor)
        and (torch.is_floating_point(value) or torch.is_complex(value))
        and not bool(torch.isfinite(value.detach()).all().item())
    ]

    invalid_optimizer: list[str] = []

    def visit(value, path: str) -> None:
        if isinstance(value, torch.Tensor):
            if (torch.is_floating_point(value) or torch.is_complex(value)) and not bool(
                torch.isfinite(value.detach()).all().item()
            ):
                invalid_optimizer.append(path)
        elif isinstance(value, Mapping):
            for key, item in value.items():
                visit(item, f"{path}.{key}")
        elif isinstance(value, list | tuple):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(optimizer.state_dict(), "optimizer")
    if invalid_model or invalid_optimizer:
        raise FloatingPointError(
            "checkpoint state contains NaN or infinite tensors: "
            f"model={invalid_model[:10]}, optimizer={invalid_optimizer[:10]}"
        )


def _fsync_path(path: pathlib.Path, *, directory: bool = False) -> None:
    flags = os.O_RDONLY | (getattr(os, "O_DIRECTORY", 0) if directory else 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_tree(root: pathlib.Path) -> None:
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        _fsync_path(path)
    directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_path(path, directory=True)


def checkpoint_due(config, global_step: int) -> bool:
    return global_step >= config.num_train_steps or (
        not config.one_batch_overfit and global_step > 0 and global_step % config.save_interval == 0
    )


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(
    model,
    optimizer,
    global_step,
    config,
    is_main,
    data_config,
    data_split_metadata,
    resume_contract,
    initialization_lineage,
    *,
    overfit_diagnostic: dict | None = None,
    training_metrics: dict | None = None,
):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # Diagnostic checkpoints are evidence only when the final gate has passed;
    # never expose ungated intermediate numeric directories.
    should_save = checkpoint_due(config, global_step)
    if should_save:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        if final_ckpt_dir.exists() or final_ckpt_dir.is_symlink():
            raise FileExistsError(f"Checkpoint steps are immutable; refusing to replace {final_ckpt_dir}")
        if config.one_batch_overfit and global_step >= config.num_train_steps and overfit_diagnostic is None:
            raise RuntimeError("Refusing to publish a final one-batch checkpoint without its passing diagnostic")

        require_finite_checkpoint_state(model, optimizer)

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "data_split": data_split_metadata,
            "timestamp": time.time(),
        }
        if data_config.recovery_provenance is not None:
            metadata["data_provenance"] = data_config.recovery_provenance
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        wandb_id = config.checkpoint_dir / "wandb_id.txt"
        if config.wandb_enabled:
            if not wandb_id.is_file() or not wandb_id.read_text().strip():
                raise FileNotFoundError(f"Checkpoint cannot persist a missing W&B run ID: {wandb_id}")
            shutil.copyfile(wandb_id, tmp_ckpt_dir / "wandb_id.txt")
        else:
            (tmp_ckpt_dir / "wandb_id.txt").write_text("disabled\n")

        # The learning-rate schedule is deterministic from this step; this
        # non-pickle sidecar lets an ephemeral worker verify the full resume
        # contract before copying any state into its writable run directory.
        resume_state = {
            "schema_version": 2,
            "global_step": global_step,
            "config_name": config.name,
            "exp_name": config.exp_name,
            "resume_contract": resume_contract,
            "resume_fingerprint_sha256": resume_identity_sha256(resume_contract, initialization_lineage),
            "initialization_lineage": initialization_lineage,
            "state_files": ["metadata.pt", "model.safetensors", "optimizer.pt", "wandb_id.txt"],
        }
        (tmp_ckpt_dir / "resume-state.json").write_text(json.dumps(resume_state, indent=2, sort_keys=True) + "\n")

        if training_metrics is not None:
            metrics = _canonical_json_value(training_metrics)
            if not isinstance(metrics, dict) or not metrics:
                raise ValueError("training checkpoint metrics must be a non-empty object")
            # ``allow_nan=False`` makes the durable sidecar fail closed even if
            # a newly added diagnostic bypasses the tensor-level finite gate.
            metrics_payload = {
                "schema_version": 1,
                "config_name": config.name,
                "exp_name": config.exp_name,
                "global_step": global_step,
                "metrics": metrics,
            }
            try:
                serialized_metrics = json.dumps(metrics_payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
            except (TypeError, ValueError) as exc:
                raise ValueError("training checkpoint metrics must contain only finite JSON values") from exc
            (tmp_ckpt_dir / "training-metrics.json").write_text(serialized_metrics)

        if overfit_diagnostic is not None:
            (tmp_ckpt_dir / "overfit-diagnostic.json").write_text(
                json.dumps(overfit_diagnostic, indent=2, sort_keys=True) + "\n"
            )

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Flush the complete hidden tree before its single immutable namespace
        # publication, then flush the parent directory containing the rename.
        fsync_tree(tmp_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)
        _fsync_path(config.checkpoint_dir, directory=True)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def coordinated_save_checkpoint(*args, device: torch.device, **kwargs) -> None:
    """Make a rank-zero publication error terminate every DDP rank coherently."""
    error: Exception | None = None
    try:
        save_checkpoint(*args, **kwargs)
    except Exception as exc:  # Synchronize before preserving the original rank-zero exception.
        error = exc
    if torch.distributed.is_initialized():
        succeeded = torch.tensor(error is None, dtype=torch.int32, device=device)
        torch.distributed.all_reduce(succeeded, op=torch.distributed.ReduceOp.MIN)
        if not bool(succeeded.item()):
            if error is not None:
                raise error
            raise RuntimeError("Checkpoint publication failed on rank zero")
    if error is not None:
        raise error


def validate_resume_state(ckpt_dir: pathlib.Path, expected_contract: dict, latest_step: int) -> dict:
    path = ckpt_dir / "resume-state.json"
    try:
        state = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Resume checkpoint has no fail-closed state sidecar: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Resume checkpoint state sidecar is invalid JSON: {path}") from exc
    expected_keys = {
        "schema_version",
        "global_step",
        "config_name",
        "exp_name",
        "resume_contract",
        "resume_fingerprint_sha256",
        "initialization_lineage",
        "state_files",
    }
    if set(state) != expected_keys:
        raise ValueError(f"Resume checkpoint state has unexpected keys: {sorted(set(state) ^ expected_keys)}")
    actual_contract = state.get("resume_contract")
    actual_fingerprint = state.get("resume_fingerprint_sha256")
    actual_lineage = state.get("initialization_lineage")
    if actual_fingerprint != resume_identity_sha256(actual_contract, actual_lineage):
        raise ValueError("Resume checkpoint fingerprint does not match its embedded contract and lineage")
    canonical_expected = _canonical_json_value(expected_contract)
    canonical_actual = _canonical_json_value(actual_contract)
    if canonical_actual != canonical_expected:
        actual = canonical_actual if isinstance(canonical_actual, dict) else {}
        differing = sorted(
            key for key in set(actual) | set(canonical_expected) if actual.get(key) != canonical_expected.get(key)
        )
        raise ValueError(f"Resume checkpoint contract differs from this training invocation: {differing}")
    if (
        state.get("schema_version") != 2
        or state.get("global_step") != latest_step
        or state.get("config_name") != expected_contract["config_name"]
        or state.get("exp_name") != expected_contract["exp_name"]
        or state.get("state_files") != ["metadata.pt", "model.safetensors", "optimizer.pt", "wandb_id.txt"]
    ):
        raise ValueError("Resume checkpoint state does not match its step, contract identity, lineage, or state files")
    lineage = validate_initialization_lineage(actual_lineage)
    teacher = expected_contract.get("teacher")
    if expected_contract["config_name"].endswith("_distill") and (
        not isinstance(teacher, dict)
        or lineage["kind"] != "shallow_teacher_transplant"
        or lineage["model_sha256"] != teacher.get("model_sha256")
    ):
        raise ValueError("Resume checkpoint initialization lineage differs from its teacher identity")
    return state


def load_checkpoint(model, optimizer, checkpoint_dir, device, expected_contract):
    """Load the latest checkpoint and return its step and immutable lineage."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"
    resume_state = validate_resume_state(ckpt_dir, expected_contract, latest_step)

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step")
        if global_step != latest_step:
            raise ValueError(
                f"Checkpoint metadata step {global_step!r} does not match numeric checkpoint directory {latest_step}"
            )
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step, resume_state["initialization_lineage"]

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def resume_data_position(global_step: int, gradient_accumulation_steps: int, batches_per_epoch: int) -> tuple[int, int]:
    if global_step < 0 or gradient_accumulation_steps < 1 or batches_per_epoch < 1:
        raise ValueError("resume data position requires a non-negative step and positive accumulation/epoch length")
    consumed_microbatches = global_step * gradient_accumulation_steps
    return divmod(consumed_microbatches, batches_per_epoch)


def prepare_checkpoint_directory(config: _config.TrainConfig, *, is_main: bool, use_ddp: bool) -> bool:
    """Validate resume state and let rank zero exclusively mutate the checkpoint directory."""
    checkpoint_dir = _config.resolve_checkpoint_dir(config.checkpoint_base_dir, config.name, config.exp_name)
    resuming = False
    if config.resume:
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Experiment checkpoint directory {checkpoint_dir} does not exist for resume")
        latest_step = get_latest_checkpoint_step(checkpoint_dir)
        if latest_step is None:
            raise FileNotFoundError(f"No valid checkpoints found in {checkpoint_dir} for resume")
        resuming = True
        logging.info(f"Resuming from experiment checkpoint directory: {checkpoint_dir} at step {latest_step}")
    elif is_main:
        # Re-resolve immediately before the only destructive operation. This
        # rejects a target or config directory replaced by a symlink after the
        # initial validation.
        checkpoint_dir = _config.resolve_checkpoint_dir(config.checkpoint_base_dir, config.name, config.exp_name)
        if checkpoint_dir.exists():
            if not config.overwrite:
                raise FileExistsError(
                    f"Fresh experiment checkpoint directory already exists: {checkpoint_dir}; "
                    "choose a unique --exp-name or explicitly request --overwrite"
                )
            if not checkpoint_dir.is_dir() or checkpoint_dir.is_symlink():
                raise ValueError(f"Refusing to overwrite a non-directory or symlink checkpoint path: {checkpoint_dir}")
            if not shutil.rmtree.avoids_symlink_attacks:
                raise RuntimeError("Checkpoint overwrite requires a symlink-attack-resistant shutil.rmtree")
            checkpoint_dir = _config.resolve_checkpoint_dir(config.checkpoint_base_dir, config.name, config.exp_name)
            shutil.rmtree(checkpoint_dir)
            logging.info(f"Overwriting checkpoint directory: {checkpoint_dir}")
        checkpoint_dir = _config.resolve_checkpoint_dir(config.checkpoint_base_dir, config.name, config.exp_name)
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        logging.info(f"Created experiment checkpoint directory: {checkpoint_dir}")

    # Non-main ranks must not inspect or create a fresh directory until rank
    # zero has completed any requested removal and recreation.
    if use_ddp:
        ddp_barrier()

    checkpoint_dir = _config.resolve_checkpoint_dir(config.checkpoint_base_dir, config.name, config.exp_name)
    if resuming:
        logging.info(f"Using existing experiment checkpoint directory: {checkpoint_dir}")
    elif not checkpoint_dir.exists():
        raise FileNotFoundError(f"Rank zero did not create checkpoint directory {checkpoint_dir}")
    return resuming


def load_initial_pytorch_weights(model, model_cfg, weight_path: str | None, *, resuming: bool) -> bool:
    """Load a source checkpoint only for a fresh run, never before a numeric resume checkpoint."""
    if resuming:
        if weight_path is not None:
            logging.info("Skipping source checkpoint initialization while resuming from a numeric checkpoint")
        return False
    if weight_path is None:
        return False

    logging.info(f"Loading weights from: {weight_path}")
    model_path = os.path.join(weight_path, "model.safetensors")
    model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    missing, unexpected = safetensors.torch.load_model(
        model_to_load,
        model_path,
        strict=not isinstance(model_cfg, openpi.models.pi0_config.SnapFlowPi0Config),
    )
    if isinstance(model_cfg, openpi.models.pi0_config.SnapFlowPi0Config):
        allowed_missing = {
            "target_time_projection.input.weight",
            "target_time_projection.input.bias",
            "target_time_projection.output.weight",
            "target_time_projection.output.bias",
        }
        if set(missing) != allowed_missing or unexpected:
            raise ValueError(
                "SnapFlow initialization expected only a missing zero-init target-time projection; "
                f"missing={missing}, unexpected={unexpected}"
            )
    logging.info(f"Loaded PyTorch weights from {weight_path}")
    return True


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def train_loop(config: _config.TrainConfig):
    validate_training_mode(config)
    use_ddp, _local_rank, device = setup_ddp()
    process_rank = dist.get_rank() if use_ddp else 0
    is_main = process_rank == 0
    set_seed(config.seed, process_rank)

    # Rank zero exclusively owns destructive checkpoint-directory setup.
    resuming = prepare_checkpoint_directory(config, is_main=is_main, use_ddp=use_ddp)

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Build data loader using the unified data loader
    # ``batch_size`` is one global microbatch. DDP divides it between ranks;
    # accumulation then determines the batch represented by an optimizer step.
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    if config.batch_size % world_size != 0:
        raise ValueError(f"Global microbatch {config.batch_size} must be divisible by world size {world_size}")
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least one")
    local_batch_size = config.batch_size // world_size
    optimizer_batch_size = config.batch_size * config.gradient_accumulation_steps
    logging.info(
        "Using local microbatch %d across %d GPUs (global microbatch %d, accumulation %d, optimizer batch %d)",
        local_batch_size,
        world_size,
        config.batch_size,
        config.gradient_accumulation_steps,
        optimizer_batch_size,
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)
    data_split_metadata = validate_training_split_metadata(config, loader)
    if data_config.robolab_expert_manifest_path is not None:
        initialization = _robolab_expert.validate_recovery_source_checkpoint(
            pathlib.Path(data_config.robolab_expert_manifest_path),
            config.pytorch_weight_path,
            teacher_checkpoint_path=config.teacher_pytorch_weight_path,
        )
        data_config = dataclasses.replace(
            data_config,
            recovery_provenance={**(data_config.recovery_provenance or {}), "initialization": initialization},
        )
    overfit_batch = next(iter(loader)) if config.one_batch_overfit else None
    overfit_inputs = None
    if is_main and config.one_batch_overfit:
        logging.info("One-batch overfit mode: repeating one materialized batch with fixed noise/time/mask")

    # Log sample images to W&B on the first batch. One-batch diagnostics reuse
    # their sole materialized batch, and disabled W&B never constructs a loader.
    sample_batch = materialize_wandb_sample_batch(
        config,
        is_main=is_main,
        resuming=resuming,
        overfit_batch=overfit_batch,
    )
    if sample_batch is not None:
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared W&B sample batch from memory")

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    if isinstance(model_cfg, openpi.models.pi0_config.SnapFlowPi0Config):
        model = openpi.models_pytorch.snapflow.SnapFlowPI0Pytorch(model_cfg).to(device)
    else:
        model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    teacher = None
    teacher_model_sha256 = None

    if isinstance(model_cfg, openpi.models.pi0_config.DistilledPi0Config):
        if config.teacher_pytorch_weight_path is None:
            raise ValueError("Shallow-pi training requires --teacher-pytorch-weight-path")
        if not resuming and config.pytorch_weight_path is not None:
            raise ValueError("Use teacher_pytorch_weight_path, not pytorch_weight_path, for a fresh Shallow-pi run")

        teacher_weights = resolve_model_weights(config.teacher_pytorch_weight_path)
        teacher_model_sha256 = sha256_file(teacher_weights)
        teacher = openpi.models_pytorch.shallow_pi.load_frozen_teacher(
            model_cfg,
            config.teacher_pytorch_weight_path,
            device,
        )
        if not resuming:
            transplant_report = openpi.models_pytorch.shallow_pi.transplant_shallow_pi_weights(
                model,
                teacher,
                layer_map=model_cfg.pytorch_layer_map,
            )
            logging.info(
                "Initialized Shallow-pi from teacher layers %s (%d state keys; mapped VLM=%d, expert=%d)",
                transplant_report.layer_map,
                transplant_report.copied_keys,
                transplant_report.mapped_layer_keys["paligemma_with_expert.paligemma.model.language_model.layers"],
                transplant_report.mapped_layer_keys["paligemma_with_expert.gemma_expert.model.layers"],
            )

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=world_size >= 8,  # Enable for 8+ GPUs
        )

    resume_contract = build_resume_contract(
        config,
        model_cfg,
        data_config,
        data_split_metadata,
        teacher_model_sha256=teacher_model_sha256,
    )

    # Source weights initialize fresh runs only. Resumes restore the numeric
    # model+optimizer checkpoint below without touching the source directory.
    load_initial_pytorch_weights(
        model,
        model_cfg,
        config.pytorch_weight_path,
        resuming=resuming,
    )
    initialization_lineage = (
        None
        if resuming
        else build_initialization_lineage(
            config,
            model_cfg,
            teacher_model_sha256=teacher_model_sha256,
        )
    )

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step, initialization_lineage = load_checkpoint(
            model,
            optim,
            config.checkpoint_dir,
            device,
            resume_contract,
        )
        logging.info(f"Resumed training from step {global_step}")
    assert initialization_lineage is not None

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    optim.zero_grad(set_to_none=True)
    start_time = time.time()
    infos = []  # Collect stats over log interval
    optimizer_step_losses: list[float] = []
    accumulation_losses: list[float] = []
    micro_step = 0
    batches_per_epoch = max(1, len(loader))
    data_epoch, resume_batches_to_skip = resume_data_position(
        global_step, config.gradient_accumulation_steps, batches_per_epoch
    )
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            "Training config: global_microbatch=%d, local_microbatch=%d, accumulation=%d, "
            "optimizer_batch=%d, num_train_steps=%d",
            config.batch_size,
            local_batch_size,
            config.gradient_accumulation_steps,
            optimizer_batch_size,
            config.num_train_steps,
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )
    latest_optimizer_metrics = None

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(data_epoch)

        if overfit_batch is not None:
            batch_iterator = itertools.repeat(overfit_batch)
        else:
            batch_iterator = iter(loader)
            if resume_batches_to_skip:
                batch_iterator = itertools.islice(batch_iterator, resume_batches_to_skip, None)
                resume_batches_to_skip = 0
        for observation, actions in batch_iterator:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            if config.one_batch_overfit and overfit_inputs is None:
                overfit_inputs = deterministic_overfit_inputs(actions, config.seed + process_rank)
            fixed_noise, fixed_time, fixed_flow_matching_mask = (
                overfit_inputs if overfit_inputs is not None else (None, None, None)
            )

            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # DDP synchronizes only the final microbatch in an accumulation
            # window. The context must wrap both forward and backward.
            accumulation_index = micro_step % config.gradient_accumulation_steps
            should_step = accumulation_index + 1 == config.gradient_accumulation_steps
            stochastic_seed = training_microstep_seed(
                config.seed,
                global_step,
                accumulation_index,
                process_rank,
                one_batch_overfit=config.one_batch_overfit,
            )
            sync_context = (
                contextlib.nullcontext()
                if should_step or not isinstance(model, torch.nn.parallel.DistributedDataParallel)
                else model.no_sync()
            )
            with counter_seeded_rng(stochastic_seed, device), sync_context:
                distillation_metrics = None
                if isinstance(model_cfg, openpi.models.pi0_config.DistilledPi0Config):
                    assert teacher is not None
                    distillation = openpi.models_pytorch.shallow_pi.compute_distillation_loss(
                        model,
                        teacher,
                        observation,
                        actions,
                        fm_loss_weight=model_cfg.fm_loss_weight,
                        kd_loss_weight=model_cfg.kd_loss_weight,
                        noise=fixed_noise,
                        time=fixed_time,
                    )
                    loss = distillation.loss
                    distillation_metrics = {
                        "fm_mse": distillation.fm_mse.item(),
                        "kd_mse": distillation.kd_mse.item(),
                        "kd_cosine": distillation.kd_cosine.item(),
                        "per_joint_nrmse_mean": distillation.per_joint_nrmse.mean().item(),
                        "per_joint_nrmse_max": distillation.per_joint_nrmse.max().item(),
                    }
                else:
                    if isinstance(model_cfg, openpi.models.pi0_config.SnapFlowPi0Config):
                        losses = model(
                            observation,
                            actions,
                            noise=fixed_noise,
                            time=fixed_time,
                            flow_matching_mask=fixed_flow_matching_mask,
                        )
                    else:
                        losses = model(observation, actions, noise=fixed_noise, time=fixed_time)
                    # Ensure losses is a tensor and handle different return types
                    if isinstance(losses, list | tuple):
                        losses = torch.stack(losses)
                    elif not isinstance(losses, torch.Tensor):
                        losses = torch.tensor(losses, device=device, dtype=torch.float32)

                    loss = losses.mean()
                    if isinstance(model_cfg, openpi.models.pi0_config.SnapFlowPi0Config):
                        snapflow_model = (
                            model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                        )
                        distillation_metrics = {
                            "snapflow_total": snapflow_model.last_snapflow_metrics["snapflow/total"],
                            "snapflow_fm_mse": snapflow_model.last_snapflow_metrics["snapflow/flow_matching"],
                            "snapflow_shortcut_mse": snapflow_model.last_snapflow_metrics["snapflow/shortcut"],
                        }

                require_finite_training_values(loss, distillation_metrics)
                if config.one_batch_overfit:
                    accumulation_losses.append(distributed_mean_loss(loss))

                (loss / config.gradient_accumulation_steps).backward()

            micro_step += 1

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Preserve per-microbatch diagnostics, but do not update model
            # state or the optimizer-step counter until the window is full.
            if is_main:
                info = {
                    "loss": loss.item(),
                    "learning_rate": optim.param_groups[0]["lr"],
                }
                if distillation_metrics is not None:
                    info.update(distillation_metrics)
                infos.append(info)
            if not should_step:
                continue

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.optimizer.clip_gradient_norm,
                error_if_nonfinite=True,
            )

            # Optimizer step
            optim.step()
            optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            if is_main:
                infos[-1]["grad_norm"] = float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
                accumulation_window = infos[-config.gradient_accumulation_steps :]
                latest_optimizer_metrics = {
                    "measurement_scope": "rank-zero local microbatches for final optimizer step",
                    "loss": math.fsum(float(info["loss"]) for info in accumulation_window) / len(accumulation_window),
                    "learning_rate": math.fsum(float(info["learning_rate"]) for info in accumulation_window)
                    / len(accumulation_window),
                    "grad_norm": float(infos[-1]["grad_norm"]),
                }
                for metric_name in (
                    "fm_mse",
                    "kd_mse",
                    "kd_cosine",
                    "per_joint_nrmse_mean",
                    "per_joint_nrmse_max",
                    "snapflow_total",
                    "snapflow_fm_mse",
                    "snapflow_shortcut_mse",
                ):
                    if all(metric_name in info for info in accumulation_window):
                        latest_optimizer_metrics[metric_name] = math.fsum(
                            float(info[metric_name]) for info in accumulation_window
                        ) / len(accumulation_window)
            if config.one_batch_overfit:
                if len(accumulation_losses) != config.gradient_accumulation_steps:
                    raise RuntimeError("one-batch diagnostic lost an accumulated microbatch loss")
                optimizer_step_losses.append(math.fsum(accumulation_losses) / len(accumulation_losses))
                accumulation_losses = []

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)
                averaged_distillation_metrics = {
                    key: sum(info[key] for info in infos) / len(infos)
                    for key in (
                        "fm_mse",
                        "kd_mse",
                        "kd_cosine",
                        "per_joint_nrmse_mean",
                        "per_joint_nrmse_max",
                        "snapflow_total",
                        "snapflow_fm_mse",
                        "snapflow_shortcut_mse",
                    )
                    if all(key in info for info in infos)
                }

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )
                if "fm_mse" in averaged_distillation_metrics:
                    logging.info(
                        "distill step=%d fm_mse=%.5f kd_mse=%.5f kd_cosine=%.5f joint_nrmse_mean=%.5f "
                        "joint_nrmse_max=%.5f",
                        global_step,
                        averaged_distillation_metrics["fm_mse"],
                        averaged_distillation_metrics["kd_mse"],
                        averaged_distillation_metrics["kd_cosine"],
                        averaged_distillation_metrics["per_joint_nrmse_mean"],
                        averaged_distillation_metrics["per_joint_nrmse_max"],
                    )
                if "snapflow_total" in averaged_distillation_metrics:
                    logging.info(
                        "snapflow step=%d total=%.5f fm_mse=%.5f shortcut_mse=%.5f",
                        global_step,
                        averaged_distillation_metrics["snapflow_total"],
                        averaged_distillation_metrics["snapflow_fm_mse"],
                        averaged_distillation_metrics["snapflow_shortcut_mse"],
                    )

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    log_payload.update(averaged_distillation_metrics)
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            overfit_diagnostic = None
            if config.one_batch_overfit and global_step >= config.num_train_steps:
                overfit_diagnostic = evaluate_one_batch_overfit(
                    optimizer_step_losses,
                    minimum_relative_decline=config.one_batch_overfit_min_relative_decline,
                )
                if is_main:
                    logging.info(
                        "One-batch diagnostic passed: first-%d mean %.6g, last-%d mean %.6g, decline %.2f%%",
                        overfit_diagnostic["window_optimizer_steps"],
                        overfit_diagnostic["initial_loss_mean"],
                        overfit_diagnostic["window_optimizer_steps"],
                        overfit_diagnostic["final_loss_mean"],
                        100.0 * overfit_diagnostic["relative_loss_decline"],
                    )
            # Save checkpoint using the new mechanism
            if checkpoint_due(config, global_step):
                checkpoint_metrics = latest_optimizer_metrics
                if checkpoint_metrics is not None and overfit_diagnostic is not None:
                    checkpoint_metrics = {
                        **checkpoint_metrics,
                        "one_batch_overfit": overfit_diagnostic,
                    }
                coordinated_save_checkpoint(
                    model,
                    optim,
                    global_step,
                    config,
                    is_main,
                    data_config,
                    data_split_metadata,
                    resume_contract,
                    initialization_lineage,
                    overfit_diagnostic=overfit_diagnostic,
                    training_metrics=checkpoint_metrics,
                    device=device,
                )

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )
        data_epoch += 1

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
