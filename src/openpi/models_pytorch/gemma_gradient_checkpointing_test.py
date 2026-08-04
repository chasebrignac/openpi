import torch
from transformers import GemmaConfig
from transformers.models.gemma import modeling_gemma


def test_suffix_decoder_layer_is_recomputed_during_backward(monkeypatch):
    config = GemmaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        use_cache=False,
        use_adarms=True,
        adarms_cond_dim=8,
    )
    model = modeling_gemma.GemmaModel(config).train()
    model.gradient_checkpointing = True

    layer = model.layers[0]
    original_forward = layer.forward
    invocations = 0

    def counted_forward(*args, **kwargs):
        nonlocal invocations
        invocations += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(layer, "forward", counted_forward)
    inputs = torch.randn(2, 3, config.hidden_size, requires_grad=True)
    condition = torch.randn(2, config.adarms_cond_dim, requires_grad=True)

    output = model(inputs_embeds=inputs, use_cache=False, adarms_cond=condition).last_hidden_state
    assert invocations == 1

    output.square().mean().backward()

    assert invocations == 2
    assert inputs.grad is not None
    assert condition.grad is not None
