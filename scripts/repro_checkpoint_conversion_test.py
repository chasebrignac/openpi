from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples import convert_jax_model_to_pytorch as conversion


def test_validate_converted_state_keys_allows_only_real_tied_head_paths():
    conversion.validate_converted_state_keys(
        SimpleNamespace(
            missing_keys=[
                "paligemma_with_expert.paligemma.lm_head.weight",
                "paligemma_with_expert.gemma_expert.lm_head.weight",
            ],
            unexpected_keys=[],
        )
    )

    with pytest.raises(ValueError, match=r"language_model\.lm_head"):
        conversion.validate_converted_state_keys(
            SimpleNamespace(
                missing_keys=["paligemma_with_expert.paligemma.language_model.lm_head.weight"],
                unexpected_keys=[],
            )
        )


def test_unconverted_heads_keep_tied_embedding_and_zero_unused_expert():
    token_embedding = torch.nn.Embedding(4, 3)
    paligemma_head = torch.nn.Linear(3, 4, bias=False)
    paligemma_head.weight = token_embedding.weight
    expert_head = torch.nn.Linear(3, 4, bias=False)
    with torch.no_grad():
        expert_head.weight.fill_(7)
    model = SimpleNamespace(
        paligemma_with_expert=SimpleNamespace(
            paligemma=SimpleNamespace(
                model=SimpleNamespace(language_model=SimpleNamespace(embed_tokens=token_embedding)),
                lm_head=paligemma_head,
            ),
            gemma_expert=SimpleNamespace(lm_head=expert_head),
        )
    )

    conversion.initialize_unconverted_heads(model)

    assert paligemma_head.weight is token_embedding.weight
    assert torch.count_nonzero(expert_head.weight) == 0


def test_copy_checkpoint_assets_preserves_nested_asset_id(tmp_path):
    checkpoint = tmp_path / "pi05_libero"
    norm_stats = checkpoint / "assets" / "libero" / "norm_stats.json"
    norm_stats.parent.mkdir(parents=True)
    norm_stats.write_text('{"norm": "pinned"}\n')
    output = tmp_path / "converted"
    output.mkdir()

    conversion.copy_checkpoint_assets(str(checkpoint), str(output))

    assert (output / "assets" / "libero" / "norm_stats.json").read_text() == '{"norm": "pinned"}\n'


def test_copy_checkpoint_assets_requires_local_staging(tmp_path):
    with pytest.raises(ValueError, match="Stage the GCS checkpoint locally"):
        conversion.copy_checkpoint_assets("gs://openpi-assets/checkpoints/pi05_libero", str(tmp_path))


def test_conversion_output_must_be_fresh_and_disjoint(tmp_path):
    source = tmp_path / "teacher"
    source.mkdir()
    output = tmp_path / "teacher-pytorch"
    assert conversion.prepare_conversion_output(str(source), str(output)) == output

    (output / "stale.bin").write_bytes(b"old")
    with pytest.raises(FileExistsError, match="non-empty"):
        conversion.prepare_conversion_output(str(source), str(output))
    with pytest.raises(ValueError, match="overlap"):
        conversion.prepare_conversion_output(str(source), str(source / "converted"))


def test_bfloat16_conversion_preserves_model_mixed_precision():
    model = torch.nn.Module()
    model.bf16_weight = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    model.sensitive_weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))

    converted = conversion.apply_conversion_precision(model, SimpleNamespace(dtype="bfloat16"), "bfloat16")

    assert converted.bf16_weight.dtype == torch.bfloat16
    assert converted.sensitive_weight.dtype == torch.float32


def test_float32_conversion_casts_the_complete_model():
    model = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    converted = conversion.apply_conversion_precision(model, SimpleNamespace(dtype="bfloat16"), "float32")
    assert {parameter.dtype for parameter in converted.parameters()} == {torch.float32}


def test_slice_gemma_uses_explicit_pi05_model_config():
    """The tensor layout is selected from the config, not a staging path."""

    config = SimpleNamespace(
        vocab_size=10,
        hidden_size=2,
        num_hidden_layers=1,
        num_attention_heads=1,
        head_dim=2,
    )
    state = {
        "llm/layers/attn/attn_vec_einsum_1/w": np.zeros((1, 1, 2, 2), dtype=np.float32),
        "llm/layers/attn/kv_einsum_1/w": np.zeros((1, 2, 1, 2, 2), dtype=np.float32),
        "llm/layers/attn/q_einsum_1/w": np.zeros((1, 2, 1, 2), dtype=np.float32),
        "llm/layers/mlp_1/gating_einsum": np.zeros((1, 2, 2, 2), dtype=np.float32),
        "llm/layers/mlp_1/linear": np.zeros((1, 2, 2), dtype=np.float32),
        "llm/layers/pre_attention_norm_1/Dense_0/bias": np.zeros((1, 2), dtype=np.float32),
        "llm/layers/pre_attention_norm_1/Dense_0/kernel": np.zeros((1, 2, 2), dtype=np.float32),
        "llm/layers/pre_ffw_norm_1/Dense_0/bias": np.zeros((1, 2), dtype=np.float32),
        "llm/layers/pre_ffw_norm_1/Dense_0/kernel": np.zeros((1, 2, 2), dtype=np.float32),
        "llm/final_norm_1/Dense_0/bias": np.zeros((2,), dtype=np.float32),
        "llm/final_norm_1/Dense_0/kernel": np.zeros((2, 2), dtype=np.float32),
    }

    result = conversion.slice_gemma_state_dict(
        state,
        config,
        num_expert=1,
        pi05=True,
    )

    assert "paligemma_with_expert.gemma_expert.model.norm.dense.weight" in result
