from __future__ import annotations

import argparse

import pytest
import torch

from scripts import repro_ddp_smoke

# The unit tests intentionally exercise the fail-closed pure helpers directly.
# ruff: noqa: SLF001


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "expected_world_size": 2,
        "iterations": 20,
        "batch_size": 32,
        "run_id": "ddp-two-node-20260805t223500z",
        "script_sha256": "a" * 64,
        "image_digest": "sha256:" + "b" * 64,
        "expected_private_ip": "172.31.1.10",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_validate_args_accepts_bounded_two_rank_contract() -> None:
    repro_ddp_smoke._validate_args(_args())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_world_size", 1),
        ("iterations", 0),
        ("iterations", 101),
        ("batch_size", 0),
        ("run_id", "Bad Run"),
        ("script_sha256", "ABC"),
        ("image_digest", "latest"),
        ("expected_private_ip", "8.8.8.8"),
    ],
)
def test_validate_args_rejects_unbounded_or_ambiguous_inputs(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=r".+"):
        repro_ddp_smoke._validate_args(_args(**{field: value}))


def test_tensor_state_hash_is_order_independent_and_value_sensitive() -> None:
    first = {
        "weight": torch.tensor([[1.0, 2.0]]),
        "bias": torch.tensor([3.0]),
    }
    reordered = {
        "bias": torch.tensor([3.0]),
        "weight": torch.tensor([[1.0, 2.0]]),
    }
    changed = {
        "weight": torch.tensor([[1.0, 2.0]]),
        "bias": torch.tensor([4.0]),
    }
    assert repro_ddp_smoke._tensor_state_sha256(first) == repro_ddp_smoke._tensor_state_sha256(reordered)
    assert repro_ddp_smoke._tensor_state_sha256(first) != repro_ddp_smoke._tensor_state_sha256(changed)


def test_all_tensors_finite_recurses_through_optimizer_shapes() -> None:
    assert repro_ddp_smoke._all_tensors_finite({"state": [{"momentum": torch.tensor([1.0])}]})
    assert not repro_ddp_smoke._all_tensors_finite({"state": [{"momentum": torch.tensor([float("nan")])}]})
