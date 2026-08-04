# ruff: noqa: PLC0415
"""ONNX-friendly, fixed-shape boundaries for split pi0.5 inference.

The policy-server API remains unchanged.  These wrappers are deployment-only:
``EncodePrefixWrapper`` converts the three camera views and prompt into a
prefix mask plus a flattened transformer KV cache, while
``DecodeDenoiseWrapper`` consumes that cache and performs the SnapFlow one-step
Euler update.  ONNX input shapes are intentionally static (batch one in the
reproduction) because TensorRT engines are built for exactly those shapes.

The cache crosses the graph boundary as ``key_0, value_0, ...``.  It is rebuilt
as a Transformers ``DynamicCache`` inside the decoder.  This keeps the engine
interfaces tensor-only without changing OpenPI's policy server or model code.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor
from torch import nn

from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

CacheFactory = Callable[[Sequence[tuple[Tensor, Tensor]]], Any]


def _cache_pairs(cache: Any) -> list[tuple[Tensor, Tensor]]:
    """Normalize supported Transformers cache representations to tensor pairs."""

    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        keys = list(cache.key_cache)
        values = list(cache.value_cache)
        if len(keys) != len(values):
            raise ValueError(f"KV cache key/value layer count differs: {len(keys)} != {len(values)}")
        return list(zip(keys, values, strict=True))

    # Transformers 4.56+ stores cache tensors on per-layer objects.
    if hasattr(cache, "layers"):
        pairs = []
        for index, layer in enumerate(cache.layers):
            key = getattr(layer, "keys", getattr(layer, "key_cache", None))
            value = getattr(layer, "values", getattr(layer, "value_cache", None))
            if key is None or value is None:
                raise TypeError(f"Cache layer {index} does not expose key/value tensors")
            pairs.append((key, value))
        return pairs

    if isinstance(cache, tuple | list):
        pairs = []
        for index, layer in enumerate(cache):
            if not isinstance(layer, tuple | list) or len(layer) != 2:
                raise TypeError(f"Legacy cache layer {index} is not a (key, value) pair")
            key, value = layer
            if not isinstance(key, Tensor) or not isinstance(value, Tensor):
                raise TypeError(f"Legacy cache layer {index} contains non-tensor values")
            pairs.append((key, value))
        return pairs

    raise TypeError(f"Unsupported KV cache representation: {type(cache)!r}")


def flatten_kv_cache(cache: Any) -> tuple[Tensor, ...]:
    """Flatten a Transformers cache in deterministic layer/key/value order."""

    flat: list[Tensor] = []
    for key, value in _cache_pairs(cache):
        flat.extend((key, value))
    if not flat:
        raise ValueError("The prefix encoder produced an empty KV cache")
    return tuple(flat)


def cache_pairs_from_flat(flat_cache: Sequence[Tensor]) -> tuple[tuple[Tensor, Tensor], ...]:
    if not flat_cache or len(flat_cache) % 2:
        raise ValueError(f"Expected a non-empty even number of KV tensors, got {len(flat_cache)}")
    return tuple((flat_cache[index], flat_cache[index + 1]) for index in range(0, len(flat_cache), 2))


def rebuild_transformers_cache(pairs: Sequence[tuple[Tensor, Tensor]]) -> Any:
    """Rebuild the cache type accepted by the pinned Transformers Gemma model."""

    from transformers.cache_utils import DynamicCache

    legacy = tuple(pairs)
    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(legacy)

    cache = DynamicCache()
    for layer_index, (key, value) in enumerate(legacy):
        cache.update(key, value, layer_index)
    return cache


class EncodePrefixWrapper(nn.Module):
    """Tensor-only wrapper around the frozen image/language prefix encoder."""

    def __init__(self, model: nn.Module, *, image_count: int = 3) -> None:
        super().__init__()
        if image_count < 1:
            raise ValueError("image_count must be positive")
        self.model = model
        self.image_count = image_count

    @property
    def input_names(self) -> tuple[str, ...]:
        images = tuple(f"image_{index}" for index in range(self.image_count))
        masks = tuple(f"image_mask_{index}" for index in range(self.image_count))
        return (*images, *masks, "lang_tokens", "lang_mask")

    def output_names(self, cache_tensor_count: int) -> tuple[str, ...]:
        if cache_tensor_count < 2 or cache_tensor_count % 2:
            raise ValueError("cache_tensor_count must be a positive key/value multiple")
        cache_names = []
        for layer in range(cache_tensor_count // 2):
            cache_names.extend((f"cache_key_{layer:02d}", f"cache_value_{layer:02d}"))
        return ("prefix_pad_masks", *cache_names)

    def forward(self, *flat_inputs: Tensor) -> tuple[Tensor, ...]:
        expected = 2 * self.image_count + 2
        if len(flat_inputs) != expected:
            raise ValueError(f"Expected {expected} prefix tensors, got {len(flat_inputs)}")

        images = list(flat_inputs[: self.image_count])
        image_masks = list(flat_inputs[self.image_count : 2 * self.image_count])
        lang_tokens, lang_mask = flat_inputs[-2:]
        prefix_pad_masks, cache = encode_prefix_tensors(
            self.model,
            images,
            image_masks,
            lang_tokens,
            lang_mask,
        )
        return (prefix_pad_masks, *flatten_kv_cache(cache))


def encode_prefix_tensors(
    model: nn.Module,
    images: Sequence[Tensor],
    image_masks: Sequence[Tensor],
    lang_tokens: Tensor,
    lang_mask: Tensor,
) -> tuple[Tensor, Any]:
    """Native eager prefix path used by export and apples-to-apples timing."""

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        list(images),
        list(image_masks),
        lang_tokens,
        lang_mask,
    )
    prefix_attention = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_attention_4d = model._prepare_attention_masks_4d(prefix_attention)  # noqa: SLF001

    language_model = model.paligemma_with_expert.paligemma.language_model
    language_model.config._attn_implementation = "eager"  # noqa: SLF001
    _, cache = model.paligemma_with_expert.forward(
        attention_mask=prefix_attention_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    return prefix_pad_masks, cache


class DecodeDenoiseWrapper(nn.Module):
    """Consume a flat KV cache and return the final one-step action chunk."""

    def __init__(
        self,
        model: nn.Module,
        *,
        cache_layers: int,
        use_target_time: bool = True,
        cache_factory: CacheFactory = rebuild_transformers_cache,
        integration_delta: float = -1.0,
    ) -> None:
        super().__init__()
        if cache_layers < 1:
            raise ValueError("cache_layers must be positive")
        if integration_delta == 0:
            raise ValueError("integration_delta must be nonzero")
        self.model = model
        self.cache_layers = cache_layers
        self.use_target_time = use_target_time
        self.cache_factory = cache_factory
        self.integration_delta = integration_delta

    @property
    def input_names(self) -> tuple[str, ...]:
        cache_names = []
        for layer in range(self.cache_layers):
            cache_names.extend((f"cache_key_{layer:02d}", f"cache_value_{layer:02d}"))
        return (
            "state",
            "x_t",
            "timestep",
            "target_time",
            "prefix_pad_masks",
            *cache_names,
        )

    @property
    def output_names(self) -> tuple[str, ...]:
        return ("actions",)

    def forward(
        self,
        state: Tensor,
        x_t: Tensor,
        timestep: Tensor,
        target_time: Tensor,
        prefix_pad_masks: Tensor,
        *flat_cache: Tensor,
    ) -> Tensor:
        if len(flat_cache) != self.cache_layers * 2:
            raise ValueError(f"Expected {self.cache_layers * 2} KV tensors, got {len(flat_cache)}")
        cache = self.cache_factory(cache_pairs_from_flat(flat_cache))
        if self.use_target_time:
            velocity = self.model.denoise_step(
                state,
                prefix_pad_masks,
                cache,
                x_t,
                timestep,
                target_time=target_time,
            )
        else:
            velocity = self.model.denoise_step(
                state,
                prefix_pad_masks,
                cache,
                x_t,
                timestep,
            )
        return x_t + self.integration_delta * velocity


def assert_fixed_input_contract(
    names: Sequence[str],
    inputs: Sequence[Tensor],
    expected_shapes: dict[str, Sequence[int]] | None = None,
) -> dict[str, tuple[int, ...]]:
    """Validate and serialize a static ONNX input contract."""

    if len(names) != len(inputs):
        raise ValueError(f"Input name/tensor count differs: {len(names)} != {len(inputs)}")
    shapes = {name: tuple(int(dim) for dim in tensor.shape) for name, tensor in zip(names, inputs, strict=True)}
    if expected_shapes is not None:
        normalized = {name: tuple(int(dim) for dim in shape) for name, shape in expected_shapes.items()}
        if shapes != normalized:
            raise ValueError(f"Fixed input contract mismatch: observed={shapes}, expected={normalized}")
    return shapes
