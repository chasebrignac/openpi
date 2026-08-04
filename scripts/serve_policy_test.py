import random

import jax
import numpy as np
import torch

from openpi.policies import policy as policy_module
from scripts import serve_policy


def test_seed_process_repeats_python_numpy_and_torch_sequences():
    serve_policy.seed_process(1234)
    first = (random.random(), np.random.random(), torch.rand(3))

    serve_policy.seed_process(1234)
    second = (random.random(), np.random.random(), torch.rand(3))

    assert first[:2] == second[:2]
    torch.testing.assert_close(first[2], second[2])


def test_checkpoint_policy_receives_cli_seed_as_jax_key(monkeypatch):
    captured = {}
    sentinel_policy = object()
    monkeypatch.setattr(serve_policy._config, "get_config", lambda name: f"config:{name}")  # noqa: SLF001

    def create_trained_policy(config, directory, **kwargs):
        captured.update(config=config, directory=directory, **kwargs)
        return sentinel_policy

    monkeypatch.setattr(
        serve_policy._policy_config,  # noqa: SLF001
        "create_trained_policy",
        create_trained_policy,
    )
    args = serve_policy.Args(
        seed=91,
        default_prompt="prompt",
        policy=serve_policy.Checkpoint(config="pi05_libero", dir="/checkpoint"),
    )

    assert serve_policy.create_policy(args) is sentinel_policy
    assert captured["config"] == "config:pi05_libero"
    assert captured["directory"] == "/checkpoint"
    assert captured["default_prompt"] == "prompt"
    assert np.array_equal(jax.random.key_data(captured["rng"]), jax.random.key_data(jax.random.key(91)))


def test_policy_accepts_explicit_typed_jax_key(monkeypatch):
    class FakeModel:
        def sample_actions(self):
            raise AssertionError("not called")

    monkeypatch.setattr(policy_module.nnx_utils, "module_jit", lambda function: function)
    key = jax.random.key(23)
    policy = policy_module.Policy(FakeModel(), rng=key)

    assert np.array_equal(jax.random.key_data(policy._rng), jax.random.key_data(key))  # noqa: SLF001
