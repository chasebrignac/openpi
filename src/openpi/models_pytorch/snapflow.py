"""SnapFlow self-distillation for the PyTorch pi0.5 implementation.

This module implements the training objective and the one-NFE inference path
from Luan et al., *SnapFlow: One-Step Action Generation for Flow-Matching
VLAs via Progressive Self-Distillation* (arXiv:2604.05656).  The implementation
uses OpenPI's convention: ``t=1`` is Gaussian noise and ``t=0`` is an action.

The VLM prefix is evaluated once, without gradients, and its KV cache is shared
by every action-expert evaluation in a training step.  Only the action expert,
the existing action/time projections, and the new target-time projection are
trainable.
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
from typing import Any

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812
from typing_extensions import override

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

VelocityFn = Callable[[Tensor, Tensor, Tensor], Tensor]
IndexedVelocityFn = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]


@dataclasses.dataclass(frozen=True)
class SnapFlowHyperparameters:
    """Published SnapFlow defaults for pi0.5."""

    alpha: float = 0.5
    consistency_weight: float = 0.1
    prediction_clamp: float = 20.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.consistency_weight < 0.0:
            raise ValueError("consistency_weight must be non-negative")
        if self.prediction_clamp <= 0.0:
            raise ValueError("prediction_clamp must be positive")


@dataclasses.dataclass(frozen=True)
class SnapFlowLosses:
    """Unreduced losses so the existing PyTorch trainer can call ``mean``."""

    total: Tensor
    flow_matching: Tensor
    shortcut: Tensor
    flow_matching_mask: Tensor

    def detached_means(self) -> dict[str, float]:
        per_sample_fm = self.flow_matching.detach().flatten(start_dim=1).mean(dim=1)
        per_sample_shortcut = self.shortcut.detach().flatten(start_dim=1).mean(dim=1)
        fm_mask = self.flow_matching_mask
        shortcut_mask = ~fm_mask
        return {
            "snapflow/total": float(self.total.detach().mean()),
            "snapflow/flow_matching": float(per_sample_fm[fm_mask].mean()) if torch.any(fm_mask) else 0.0,
            "snapflow/shortcut": (
                float(per_sample_shortcut[shortcut_mask].mean()) if torch.any(shortcut_mask) else 0.0
            ),
        }


@dataclasses.dataclass(frozen=True)
class SnapFlowPrefixContext:
    """Frozen prefix state shared by all expert calls in one optimizer step."""

    state: Tensor
    prefix_pad_masks: Tensor
    past_key_values: Any


def _select_prefix_context(
    context: SnapFlowPrefixContext,
    sample_indices: Tensor,
) -> SnapFlowPrefixContext:
    """Select a batch subset after the full-batch student query has run.

    Transformers' ``DynamicCache`` performs this selection in place. The
    caller must therefore finish every full-batch use of ``context`` before
    calling this helper.
    """

    if sample_indices.ndim != 1 or sample_indices.dtype != torch.long:
        raise ValueError("sample_indices must be a one-dimensional int64 tensor")

    past_key_values = context.past_key_values
    if not hasattr(past_key_values, "batch_select_indices"):
        raise TypeError(
            "SnapFlow consistency-subset execution requires a cache with "
            "batch_select_indices (for example transformers.DynamicCache)"
        )
    past_key_values.batch_select_indices(sample_indices)
    return SnapFlowPrefixContext(
        state=context.state.index_select(0, sample_indices),
        prefix_pad_masks=context.prefix_pad_masks.index_select(0, sample_indices),
        past_key_values=past_key_values,
    )


class TargetTimeProjection(nn.Module):
    """Zero-output-initialized two-layer target-time projection.

    Initializing the output layer to zero makes a newly constructed SnapFlow
    model exactly equivalent to its pre-SnapFlow checkpoint for every target
    time.  The first layer uses the normal PyTorch initialization so gradients
    can reach it as soon as the output layer moves away from zero.
    """

    def __init__(
        self,
        width: int,
        *,
        hidden_width: int | None = None,
        min_period: float = 4e-3,
        max_period: float = 4.0,
    ) -> None:
        super().__init__()
        if width <= 0 or width % 2 != 0:
            raise ValueError(f"width must be a positive even number, got {width}")
        if min_period <= 0 or max_period <= min_period:
            raise ValueError("target-time periods must satisfy 0 < min_period < max_period")

        hidden_width = width if hidden_width is None else hidden_width
        if hidden_width <= 0:
            raise ValueError("hidden_width must be positive")

        self.width = width
        self.min_period = min_period
        self.max_period = max_period
        self.input = nn.Linear(width, hidden_width)
        self.output = nn.Linear(hidden_width, width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, target_time: Tensor) -> Tensor:
        if target_time.ndim == 0:
            target_time = target_time[None]
        if target_time.ndim != 1:
            raise ValueError(f"target_time must have shape (batch,), got {tuple(target_time.shape)}")

        embedding = create_sinusoidal_pos_embedding(
            target_time,
            self.width,
            min_period=self.min_period,
            max_period=self.max_period,
            device=target_time.device,
        )
        embedding = embedding.to(dtype=self.input.weight.dtype)
        return self.output(F.silu(self.input(embedding)))


def add_target_time_condition(
    base_condition: Tensor,
    projection: TargetTimeProjection,
    target_time: Tensor | None,
) -> Tensor:
    """Add ``phi_s`` to the time condition consumed by every expert block."""

    if target_time is None:
        return base_condition
    correction = projection(target_time).to(dtype=base_condition.dtype)
    return base_condition + correction


def _clamp_velocity(velocity: Tensor, limit: float) -> Tensor:
    return torch.clamp(velocity, min=-limit, max=limit)


def two_step_euler_shortcut(
    velocity_fn: VelocityFn,
    noise: Tensor,
    *,
    prediction_clamp: float = 20.0,
) -> Tensor:
    """Compute the stop-gradient two-step shortcut target (paper Eq. 9-10).

    ``velocity_fn(x_t, t, s)`` predicts average velocity from current time
    ``t`` to target time ``s``.  Both target evaluations are local velocity
    queries (``s=t``); the returned trapezoidal average is detached.
    """

    if noise.ndim < 2:
        raise ValueError(f"noise must be batched, got shape {tuple(noise.shape)}")
    batch_size = noise.shape[0]
    time_one = torch.ones(batch_size, dtype=torch.float32, device=noise.device)
    time_half = torch.full_like(time_one, 0.5)

    with torch.no_grad():
        velocity_one = _clamp_velocity(velocity_fn(noise, time_one, time_one), prediction_clamp)
        midpoint = noise - 0.5 * velocity_one
        velocity_half = _clamp_velocity(velocity_fn(midpoint, time_half, time_half), prediction_clamp)
        return 0.5 * (velocity_one + velocity_half)


def snapflow_objective(
    velocity_fn: VelocityFn,
    *,
    actions: Tensor,
    noise: Tensor,
    time: Tensor,
    shortcut_velocity_fn: IndexedVelocityFn | None = None,
    hyperparameters: SnapFlowHyperparameters | None = None,
    flow_matching_mask: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> SnapFlowLosses:
    """Compute the published mixed FM/shortcut objective.

    The caller supplies the observation-conditioned velocity closure. This
    makes the shared tensors explicit: the same action batch, Gaussian noise,
    and frozen prefix context are used by all local and global predictions.

    As described in the paper appendix, an ``alpha`` fraction of each batch is
    assigned to flow matching and the remainder to shortcut consistency. The
    two student queries are combined into one mixed batch. The two
    stop-gradient target evaluations run only on consistency samples, while
    retaining the same three-call objective.
    """

    if hyperparameters is None:
        hyperparameters = SnapFlowHyperparameters()
    if actions.shape != noise.shape:
        raise ValueError(f"actions and noise must have identical shapes: {actions.shape} != {noise.shape}")
    if time.shape != (actions.shape[0],):
        raise ValueError(f"time must have shape ({actions.shape[0]},), got {tuple(time.shape)}")
    if not torch.is_floating_point(actions) or not torch.is_floating_point(noise):
        raise TypeError("actions and noise must be floating-point tensors")

    batch_size = actions.shape[0]
    if flow_matching_mask is None:
        if hyperparameters.alpha == 0.0:
            flow_matching_count = 0
        elif hyperparameters.alpha == 1.0:
            flow_matching_count = batch_size
        elif batch_size == 1:
            flow_matching_count = int(
                torch.rand((), device=actions.device, generator=generator) < hyperparameters.alpha
            )
        else:
            flow_matching_count = round(hyperparameters.alpha * batch_size)
            flow_matching_count = min(max(flow_matching_count, 1), batch_size - 1)
        permutation = torch.randperm(batch_size, device=actions.device, generator=generator)
        flow_matching_mask = torch.zeros(batch_size, dtype=torch.bool, device=actions.device)
        flow_matching_mask[permutation[:flow_matching_count]] = True
    elif flow_matching_mask.shape != (batch_size,) or flow_matching_mask.dtype != torch.bool:
        raise ValueError(f"flow_matching_mask must be bool ({batch_size},), got {flow_matching_mask}")
    else:
        flow_matching_mask = flow_matching_mask.to(device=actions.device)

    time_expanded = time
    while time_expanded.ndim < actions.ndim:
        time_expanded = time_expanded.unsqueeze(-1)
    noisy_actions = time_expanded * noise + (1.0 - time_expanded) * actions
    ground_truth_velocity = noise - actions

    time_one = torch.ones(batch_size, dtype=torch.float32, device=actions.device)
    target_zero = torch.zeros_like(time_one)
    broadcast_mask = flow_matching_mask
    while broadcast_mask.ndim < actions.ndim:
        broadcast_mask = broadcast_mask.unsqueeze(-1)
    mixed_input = torch.where(broadcast_mask, noisy_actions, noise)
    mixed_time = torch.where(flow_matching_mask, time, time_one)
    mixed_target_time = torch.where(flow_matching_mask, time, target_zero)
    mixed_velocity = _clamp_velocity(
        velocity_fn(mixed_input, mixed_time, mixed_target_time),
        hyperparameters.prediction_clamp,
    )
    flow_matching_loss = F.mse_loss(mixed_velocity, ground_truth_velocity, reduction="none")

    shortcut_indices = torch.nonzero(~flow_matching_mask, as_tuple=False).flatten()
    shortcut_loss = torch.zeros_like(flow_matching_loss)
    if shortcut_indices.numel() > 0:
        shortcut_noise = noise.index_select(0, shortcut_indices)

        def target_velocity_fn(x_t: Tensor, timestep: Tensor, target_time: Tensor) -> Tensor:
            if shortcut_velocity_fn is None:
                return velocity_fn(x_t, timestep, target_time)
            return shortcut_velocity_fn(x_t, timestep, target_time, shortcut_indices)

        shortcut_target = two_step_euler_shortcut(
            target_velocity_fn,
            shortcut_noise,
            prediction_clamp=hyperparameters.prediction_clamp,
        )
        shortcut_loss_subset = F.mse_loss(
            mixed_velocity.index_select(0, shortcut_indices),
            shortcut_target,
            reduction="none",
        )
        shortcut_loss = shortcut_loss.index_copy(0, shortcut_indices, shortcut_loss_subset)

    # The empirical batch fractions supply alpha and (1-alpha), so applying
    # those coefficients again would square the intended mixture weights.
    total = torch.where(
        broadcast_mask,
        flow_matching_loss,
        hyperparameters.consistency_weight * shortcut_loss,
    )
    return SnapFlowLosses(
        total=total,
        flow_matching=flow_matching_loss,
        shortcut=shortcut_loss,
        flow_matching_mask=flow_matching_mask,
    )


class SnapFlowPI0Pytorch(PI0Pytorch):
    """pi0.5 with SnapFlow training and one-NFE inference.

    This class deliberately preserves the policy-server interface:
    ``sample_actions(device, observation, noise=None, num_steps=1)`` returns an
    action chunk just like ``PI0Pytorch.sample_actions``.
    """

    def __init__(
        self,
        config,
        *,
        hyperparameters: SnapFlowHyperparameters | None = None,
    ) -> None:
        super().__init__(config)
        if not self.pi05:
            raise ValueError("SnapFlowPI0Pytorch currently supports pi0.5 only")

        width = self.action_in_proj.out_features
        self.target_time_projection = TargetTimeProjection(width)
        self.hyperparameters = hyperparameters or SnapFlowHyperparameters(
            alpha=getattr(config, "snapflow_alpha", 0.5),
            consistency_weight=getattr(config, "snapflow_consistency_weight", 0.1),
            prediction_clamp=getattr(config, "snapflow_prediction_clamp", 20.0),
        )
        self.last_snapflow_metrics: dict[str, float] = {}
        self.freeze_vlm()

    def freeze_vlm(self) -> None:
        """Freeze the complete PaliGemma prefix while leaving the expert trainable."""

        for parameter in self.parameters():
            parameter.requires_grad_(requires_grad=True)
        for parameter in self.paligemma_with_expert.paligemma.parameters():
            parameter.requires_grad_(requires_grad=False)
        self.paligemma_with_expert.paligemma.eval()

    @override
    def train(self, mode: bool = True) -> SnapFlowPI0Pytorch:
        super().train(mode)
        # ``model.train()`` must never re-enable dropout or other training
        # behavior in the frozen prefix.
        self.paligemma_with_expert.paligemma.eval()
        return self

    def embed_suffix(self, state, noisy_actions, timestep, target_time=None):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = super().embed_suffix(
            state,
            noisy_actions,
            timestep,
        )
        if adarms_cond is None:
            raise RuntimeError("SnapFlow target-time conditioning requires pi0.5 AdaRMS conditioning")
        adarms_cond = add_target_time_condition(adarms_cond, self.target_time_projection, target_time)
        return suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond

    def encode_prefix_context(
        self,
        observation,
        *,
        train: bool,
        observation_is_preprocessed: bool = False,
    ) -> SnapFlowPrefixContext:
        """Encode the frozen VLM once and return the reusable KV cache."""

        if observation_is_preprocessed:
            images, img_masks, lang_tokens, lang_masks, state = self._observation_components(observation)
        else:
            images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
                observation,
                train=train,
            )

        with torch.no_grad():
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
            self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

        return SnapFlowPrefixContext(
            state=state,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
        )

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        target_time=None,
    ):
        """Evaluate the action expert with an explicit target time ``s``."""

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state,
            x_t,
            timestep,
            target_time=target_time,
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1][:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out).to(dtype=torch.float32)

    def predict_velocity_from_context(
        self,
        context: SnapFlowPrefixContext,
        noisy_actions: Tensor,
        timestep: Tensor,
        target_time: Tensor,
    ) -> Tensor:
        return self.denoise_step(
            context.state,
            context.prefix_pad_masks,
            context.past_key_values,
            noisy_actions,
            timestep,
            target_time=target_time,
        )

    def compute_snapflow_losses(
        self,
        observation,
        actions: Tensor,
        *,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        flow_matching_mask: Tensor | None = None,
    ) -> SnapFlowLosses:
        """Preprocess/encode once, then run the published self-distillation loss."""

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            # SnapFlow uses Uniform(0, 1), unlike OpenPI's Beta FM sampler.
            time = torch.rand(actions.shape[0], dtype=torch.float32, device=actions.device)

        preprocessed = self.preprocess_observation(observation, train=True)
        context = self.encode_prefix_context(
            preprocessed,
            train=True,
            observation_is_preprocessed=True,
        )

        def velocity_fn(noisy_actions: Tensor, timestep: Tensor, target_time: Tensor) -> Tensor:
            return self.predict_velocity_from_context(context, noisy_actions, timestep, target_time)

        shortcut_context: SnapFlowPrefixContext | None = None
        shortcut_sample_indices: Tensor | None = None

        def shortcut_velocity_fn(
            noisy_actions: Tensor,
            timestep: Tensor,
            target_time: Tensor,
            sample_indices: Tensor,
        ) -> Tensor:
            nonlocal shortcut_context, shortcut_sample_indices
            if shortcut_context is None:
                shortcut_context = _select_prefix_context(context, sample_indices)
                shortcut_sample_indices = sample_indices.detach().clone()
            elif not torch.equal(shortcut_sample_indices, sample_indices):
                raise RuntimeError("SnapFlow shortcut sample indices changed within one objective evaluation")
            return self.predict_velocity_from_context(
                shortcut_context,
                noisy_actions,
                timestep,
                target_time,
            )

        losses = snapflow_objective(
            velocity_fn,
            actions=actions,
            noise=noise,
            time=time,
            shortcut_velocity_fn=shortcut_velocity_fn,
            hyperparameters=self.hyperparameters,
            flow_matching_mask=flow_matching_mask,
        )
        self.last_snapflow_metrics = losses.detached_means()
        return losses

    def forward(self, observation, actions, noise=None, time=None, flow_matching_mask=None) -> Tensor:
        return self.compute_snapflow_losses(
            observation,
            actions,
            noise=noise,
            time=time,
            flow_matching_mask=flow_matching_mask,
        ).total

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=1) -> Tensor:
        """Generate the action chunk with one global ``s=0, t=1`` jump."""

        if num_steps != 1:
            raise ValueError(f"A SnapFlow checkpoint must run with num_steps=1, got {num_steps}")
        batch_size = observation.state.shape[0]
        if noise is None:
            shape = (batch_size, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(shape, device)

        context = self.encode_prefix_context(observation, train=False)
        time_one = torch.ones(batch_size, dtype=torch.float32, device=device)
        target_zero = torch.zeros_like(time_one)
        velocity = self.predict_velocity_from_context(context, noise, time_one, target_zero)
        return noise - velocity
