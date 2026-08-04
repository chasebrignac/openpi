from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from openpi.exporting.pi05_onnx import DecodeDenoiseWrapper  # noqa: E402
from openpi.exporting.pi05_onnx import EncodePrefixWrapper  # noqa: E402
from openpi.exporting.pi05_onnx import assert_fixed_input_contract  # noqa: E402
from openpi.exporting.pi05_onnx import cache_pairs_from_flat  # noqa: E402
from openpi.exporting.pi05_onnx import flatten_kv_cache  # noqa: E402


class _Config:
    _attn_implementation = "sdpa"


class _LanguageModel:
    config = _Config()


class _PaliGemma:
    language_model = _LanguageModel()


class _Cache:
    def __init__(self, pairs):
        self.key_cache = [pair[0] for pair in pairs]
        self.value_cache = [pair[1] for pair in pairs]


class _Combined(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma = _PaliGemma()

    def forward(self, **kwargs):
        del kwargs
        key_0 = torch.ones(1, 1, 4, 2)
        value_0 = torch.full_like(key_0, 2)
        key_1 = torch.full_like(key_0, 3)
        value_1 = torch.full_like(key_0, 4)
        return [None, None], _Cache(((key_0, value_0), (key_1, value_1)))


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = _Combined()

    def embed_prefix(self, images, masks, lang_tokens, lang_mask):
        del lang_tokens
        embeddings = torch.stack([image.mean(dim=(2, 3)) for image in images], dim=1)
        pad = torch.cat([mask[:, None] for mask in masks] + [lang_mask], dim=1)
        attention = torch.zeros_like(pad)
        return embeddings, pad, attention

    def _prepare_attention_masks_4d(self, mask):
        return mask[:, None].float()

    def denoise_step(self, state, prefix_mask, cache, x_t, timestep, target_time=None):
        del state, prefix_mask, timestep
        cache_sum = sum(key.mean() + value.mean() for key, value in cache)
        target = 0 if target_time is None else target_time[:, None, None]
        return torch.ones_like(x_t) * cache_sum + target


def test_prefix_wrapper_flattens_cache_and_names():
    model = _DummyModel()
    wrapper = EncodePrefixWrapper(model, image_count=2)
    inputs = (
        torch.ones(1, 3, 2, 2),
        torch.ones(1, 3, 2, 2),
        torch.ones(1, dtype=torch.bool),
        torch.ones(1, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.long),
        torch.ones(1, 2, dtype=torch.bool),
    )
    outputs = wrapper(*inputs)
    assert len(outputs) == 5
    assert wrapper.output_names(4) == (
        "prefix_pad_masks",
        "cache_key_00",
        "cache_value_00",
        "cache_key_01",
        "cache_value_01",
    )


def test_decoder_rebuilds_cache_and_applies_one_step():
    model = _DummyModel()
    wrapper = DecodeDenoiseWrapper(model, cache_layers=2, cache_factory=tuple)
    state = torch.zeros(1, 2)
    noise = torch.full((1, 3, 2), 20.0)
    timestep = torch.ones(1)
    target = torch.zeros(1)
    mask = torch.ones(1, 4, dtype=torch.bool)
    cache = tuple(torch.full((1, 1, 4, 2), value) for value in (1.0, 2.0, 3.0, 4.0))
    actions = wrapper(state, noise, timestep, target, mask, *cache)
    assert torch.allclose(actions, torch.full_like(noise, 10.0))


def test_cache_helpers_reject_odd_cache():
    with pytest.raises(ValueError, match="even"):
        cache_pairs_from_flat((torch.ones(1),))
    with pytest.raises(ValueError, match="empty"):
        flatten_kv_cache(_Cache(()))


def test_fixed_contract_detects_shape_change():
    tensor = torch.ones(1, 2)
    assert assert_fixed_input_contract(("x",), (tensor,)) == {"x": (1, 2)}
    with pytest.raises(ValueError, match="mismatch"):
        assert_fixed_input_contract(("x",), (tensor,), {"x": (2, 2)})
