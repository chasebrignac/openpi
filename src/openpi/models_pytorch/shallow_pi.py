"""Shallow-pi initialization and velocity distillation helpers."""

from collections.abc import Mapping, Sequence
import dataclasses
import pathlib

import safetensors.torch
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.models import pi0_config
from openpi.models_pytorch import pi0_pytorch

SHALLOW_LAYER_STACKS = (
    "paligemma_with_expert.paligemma.model.language_model.layers",
    "paligemma_with_expert.gemma_expert.model.layers",
)


@dataclasses.dataclass(frozen=True)
class LayerTransplantReport:
    """Summary of an exact teacher-to-student state transplant."""

    layer_map: tuple[int, ...]
    copied_keys: int
    mapped_layer_keys: dict[str, int]


@dataclasses.dataclass(frozen=True)
class DistillationLoss:
    """Scalar objective and detached diagnostics for one Shallow-pi batch."""

    loss: torch.Tensor
    fm_mse: torch.Tensor
    kd_mse: torch.Tensor
    kd_cosine: torch.Tensor
    per_joint_nrmse: torch.Tensor


def teacher_config_from_student(config: pi0_config.DistilledPi0Config) -> pi0_config.Pi0Config:
    """Construct the full-depth teacher config matching a distilled student."""
    kwargs = {field.name: getattr(config, field.name) for field in dataclasses.fields(pi0_config.Pi0Config)}
    kwargs["pytorch_compile_mode"] = None
    return pi0_config.Pi0Config(**kwargs)


def _teacher_key_for_student_key(
    student_key: str,
    layer_map: Sequence[int],
    layer_stacks: Sequence[str],
) -> tuple[str, str | None]:
    for stack in layer_stacks:
        marker = f"{stack}."
        if not student_key.startswith(marker):
            continue
        layer_text, separator, suffix = student_key.removeprefix(marker).partition(".")
        if not separator or not layer_text.isdigit():
            raise ValueError(f"Malformed transformer layer key: {student_key}")
        student_layer = int(layer_text)
        if student_layer >= len(layer_map):
            raise ValueError(f"Student layer {student_layer} has no entry in layer map {tuple(layer_map)}")
        return f"{marker}{layer_map[student_layer]}.{suffix}", stack
    return student_key, None


def build_shallow_state_dict(
    student_state: Mapping[str, torch.Tensor],
    teacher_state: Mapping[str, torch.Tensor],
    *,
    layer_map: Sequence[int] = pi0_config.SHALLOW_PI_LAYER_MAP,
    layer_stacks: Sequence[str] = SHALLOW_LAYER_STACKS,
) -> tuple[dict[str, torch.Tensor], LayerTransplantReport]:
    """Map both student transformer stacks to exact teacher layers.

    Parameters outside the VLM and action-expert transformer stacks retain
    their names and are copied from the teacher unchanged.
    """
    layer_map = tuple(layer_map)
    if layer_map != pi0_config.SHALLOW_PI_LAYER_MAP:
        raise ValueError(f"Expected the validated 18-to-9 layer map {pi0_config.SHALLOW_PI_LAYER_MAP}, got {layer_map}")

    mapped_counts = dict.fromkeys(layer_stacks, 0)
    result: dict[str, torch.Tensor] = {}
    for student_key, student_value in student_state.items():
        teacher_key, mapped_stack = _teacher_key_for_student_key(student_key, layer_map, layer_stacks)
        if teacher_key not in teacher_state:
            raise KeyError(f"Teacher checkpoint is missing {teacher_key!r}, required for {student_key!r}")
        teacher_value = teacher_state[teacher_key]
        if student_value.shape != teacher_value.shape:
            raise ValueError(
                f"Shape mismatch for {student_key!r} <- {teacher_key!r}: "
                f"student={tuple(student_value.shape)}, teacher={tuple(teacher_value.shape)}"
            )
        result[student_key] = teacher_value.detach()
        if mapped_stack is not None:
            mapped_counts[mapped_stack] += 1

    empty_stacks = [stack for stack, count in mapped_counts.items() if count == 0]
    if empty_stacks:
        raise ValueError(f"No mapped layer parameters found for: {', '.join(empty_stacks)}")

    return result, LayerTransplantReport(
        layer_map=layer_map,
        copied_keys=len(result),
        mapped_layer_keys=mapped_counts,
    )


def transplant_shallow_pi_weights(
    student: nn.Module,
    teacher: nn.Module,
    *,
    layer_map: Sequence[int] = pi0_config.SHALLOW_PI_LAYER_MAP,
) -> LayerTransplantReport:
    """Initialize a nine-layer student from the selected full teacher layers."""
    state, report = build_shallow_state_dict(student.state_dict(), teacher.state_dict(), layer_map=layer_map)
    student.load_state_dict(state, strict=True)
    return report


def resolve_safetensors_path(path: str | pathlib.Path) -> pathlib.Path:
    """Accept either a converted checkpoint directory or its weights file."""
    result = pathlib.Path(path).expanduser()
    if result.is_dir():
        result = result / "model.safetensors"
    if not result.is_file():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {result}")
    return result


def load_frozen_teacher(
    student_config: pi0_config.DistilledPi0Config,
    checkpoint_path: str | pathlib.Path,
    device: torch.device,
) -> pi0_pytorch.PI0Pytorch:
    """Load the released full-depth teacher and make its mode immutable by convention."""
    teacher = pi0_pytorch.PI0Pytorch(teacher_config_from_student(student_config)).to(device)
    weight_path = resolve_safetensors_path(checkpoint_path)
    safetensors.torch.load_model(teacher, weight_path, device=str(device))
    teacher.requires_grad_(requires_grad=False)
    teacher.eval()
    return teacher


def unwrap_student(model: nn.Module) -> pi0_pytorch.PI0Pytorch:
    """Return the PI0 module under DDP without changing the DDP forward path."""
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def compute_distillation_loss(
    student: nn.Module,
    teacher: pi0_pytorch.PI0Pytorch,
    observation,
    actions: torch.Tensor,
    *,
    fm_loss_weight: float,
    kd_loss_weight: float,
    noise: torch.Tensor | None = None,
    time: torch.Tensor | None = None,
) -> DistillationLoss:
    """Compute FM/KD losses using one augmentation, one noise draw, and one timestep draw."""
    student_core = unwrap_student(student)
    preprocessed_observation = student_core.preprocess_observation(observation, train=True)
    if noise is None:
        noise = student_core.sample_noise(actions.shape, actions.device)
    if time is None:
        time = student_core.sample_time(actions.shape[0], actions.device)

    student_velocity = student(
        preprocessed_observation,
        actions,
        noise=noise,
        time=time,
        observation_is_preprocessed=True,
        return_velocity=True,
    )

    teacher.eval()
    with torch.no_grad():
        teacher_velocity = teacher.predict_velocity(
            preprocessed_observation,
            actions,
            noise,
            time,
            observation_is_preprocessed=True,
        )

    target_velocity = noise - actions
    fm_elementwise = F.mse_loss(student_velocity, target_velocity, reduction="none")
    kd_elementwise = F.mse_loss(student_velocity, teacher_velocity, reduction="none")
    loss = (fm_loss_weight * fm_elementwise + kd_loss_weight * kd_elementwise).mean()

    with torch.no_grad():
        flattened_student = student_velocity.float().flatten(start_dim=1)
        flattened_teacher = teacher_velocity.float().flatten(start_dim=1)
        kd_cosine = F.cosine_similarity(flattened_student, flattened_teacher, dim=1).mean()
        per_joint_rmse = kd_elementwise.float().mean(dim=(0, 1)).sqrt()
        teacher_scale = teacher_velocity.float().std(dim=(0, 1)).clamp_min(1e-6)
        per_joint_nrmse = per_joint_rmse / teacher_scale

    return DistillationLoss(
        loss=loss,
        fm_mse=fm_elementwise.detach().mean(),
        kd_mse=kd_elementwise.detach().mean(),
        kd_cosine=kd_cosine.detach(),
        per_joint_nrmse=per_joint_nrmse.detach(),
    )
