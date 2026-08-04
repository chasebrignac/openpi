import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import safetensors.torch
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


SHALLOW_PI_LAYER_MAP = (0, 2, 4, 6, 8, 11, 13, 15, 17)


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


@dataclasses.dataclass(frozen=True)
class DistilledPi0Config(Pi0Config):
    """PyTorch-only configuration for a Shallow-pi student.

    The JAX model intentionally remains unchanged.  ``pytorch_gemma_depth`` is
    consumed by :class:`PI0Pytorch` to construct both the VLM and action expert
    at the requested depth.
    """

    teacher_config: str = "pi05_libero"
    pytorch_gemma_depth: int = 9
    pytorch_layer_map: tuple[int, ...] = SHALLOW_PI_LAYER_MAP
    fm_loss_weight: float = 1.0
    kd_loss_weight: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05:
            raise ValueError("Shallow-pi distillation currently supports pi0.5 only")
        if self.pytorch_gemma_depth != 9:
            raise ValueError("The released Shallow-pi reproduction supports the validated 18-to-9 mapping only")
        if self.pytorch_layer_map != SHALLOW_PI_LAYER_MAP:
            raise ValueError(f"Expected the validated Shallow-pi layer map {SHALLOW_PI_LAYER_MAP}")
        if len(self.pytorch_layer_map) != self.pytorch_gemma_depth:
            raise ValueError("The Shallow-pi layer map length must equal the student depth")
        if self.fm_loss_weight < 0 or self.kd_loss_weight < 0:
            raise ValueError("Distillation loss weights must be non-negative")
        if self.fm_loss_weight == 0 and self.kd_loss_weight == 0:
            raise ValueError("At least one distillation loss must be enabled")

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        del rng
        raise NotImplementedError("DistilledPi0Config is supported by the PyTorch training path only")


@dataclasses.dataclass(frozen=True)
class ShallowPi0Config(Pi0Config):
    """PyTorch-only nine-layer π0.5 config for post-distillation fine-tuning.

    Unlike :class:`DistilledPi0Config`, this config has no teacher or KD loss.
    It is used by the bounded expert-BC recovery run, which must initialize
    from an already accepted Shallow checkpoint and use ground-truth flow
    matching only.
    """

    pytorch_gemma_depth: int = 9
    source_layer_map: tuple[int, ...] = SHALLOW_PI_LAYER_MAP

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05:
            raise ValueError("ShallowPi0Config currently supports pi0.5 only")
        if self.pytorch_gemma_depth != 9 or self.source_layer_map != SHALLOW_PI_LAYER_MAP:
            raise ValueError("Shallow BC must use the validated 18-to-9 Shallow-pi architecture")

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        del rng
        raise NotImplementedError("ShallowPi0Config is supported by the PyTorch training path only")


@dataclasses.dataclass(frozen=True)
class SnapFlowPi0Config(Pi0Config):
    """PyTorch-only configuration for SnapFlow on an accepted Shallow-pi checkpoint."""

    pytorch_gemma_depth: int = 9
    source_layer_map: tuple[int, ...] = SHALLOW_PI_LAYER_MAP
    snapflow_alpha: float = 0.5
    snapflow_consistency_weight: float = 0.1
    snapflow_prediction_clamp: float = 20.0

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05:
            raise ValueError("SnapFlowPi0Config currently supports pi0.5 only")
        if self.pytorch_gemma_depth != 9 or self.source_layer_map != SHALLOW_PI_LAYER_MAP:
            raise ValueError("SnapFlow must start from the validated nine-layer Shallow-pi architecture")
        if not 0.0 <= self.snapflow_alpha <= 1.0:
            raise ValueError("snapflow_alpha must be in [0, 1]")
        if self.snapflow_consistency_weight < 0.0:
            raise ValueError("snapflow_consistency_weight must be non-negative")
        if self.snapflow_prediction_clamp <= 0.0:
            raise ValueError("snapflow_prediction_clamp must be positive")

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        del rng
        raise NotImplementedError("SnapFlowPi0Config is supported by the PyTorch training path only")

    @override
    def load_pytorch(self, train_config, weight_path: str):
        """Load the one-step implementation through the unchanged policy API."""
        from openpi.models_pytorch.snapflow import SnapFlowPI0Pytorch

        if train_config.model is not self:
            raise ValueError("SnapFlow loader received a train config with a different model config")
        model = SnapFlowPI0Pytorch(self)
        missing, unexpected = safetensors.torch.load_model(model, weight_path, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"SnapFlow checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        return model
