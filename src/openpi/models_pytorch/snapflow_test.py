from types import SimpleNamespace

import pytest
import safetensors.torch
import torch

from openpi.models import pi0_config
from openpi.models_pytorch import snapflow
from openpi.models_pytorch.snapflow import SnapFlowHyperparameters
from openpi.models_pytorch.snapflow import SnapFlowPI0Pytorch
from openpi.models_pytorch.snapflow import TargetTimeProjection
from openpi.models_pytorch.snapflow import add_target_time_condition
from openpi.models_pytorch.snapflow import snapflow_objective
from openpi.models_pytorch.snapflow import two_step_euler_shortcut


def test_policy_loader_constructs_one_step_snapflow_model(monkeypatch):
    config = pi0_config.SnapFlowPi0Config(pi05=True, action_horizon=10, pytorch_compile_mode=None)
    sentinel = object()
    loaded = {}
    monkeypatch.setattr(snapflow, "SnapFlowPI0Pytorch", lambda model_config: sentinel)

    def fake_load_model(model, path, *, strict):
        loaded.update(model=model, path=path, strict=strict)
        return set(), set()

    monkeypatch.setattr(safetensors.torch, "load_model", fake_load_model)

    result = config.load_pytorch(SimpleNamespace(model=config), "/checkpoint/model.safetensors")

    assert result is sentinel
    assert loaded == {"model": sentinel, "path": "/checkpoint/model.safetensors", "strict": True}


def test_target_time_projection_is_exactly_zero_at_initialization():
    projection = TargetTimeProjection(8)
    base = torch.randn(3, 8)
    target_time = torch.tensor([0.0, 0.5, 1.0])

    correction = projection(target_time)
    conditioned = add_target_time_condition(base, projection, target_time)

    assert torch.count_nonzero(correction) == 0
    assert torch.equal(conditioned, base)
    assert add_target_time_condition(base, projection, None) is base


def test_freeze_vlm_leaves_only_expert_and_projection_trainable():
    model = SnapFlowPI0Pytorch.__new__(SnapFlowPI0Pytorch)
    torch.nn.Module.__init__(model)
    model.paligemma_with_expert = torch.nn.Module()
    model.paligemma_with_expert.paligemma = torch.nn.Linear(2, 2)
    model.paligemma_with_expert.gemma_expert = torch.nn.Linear(2, 2)
    model.target_time_projection = TargetTimeProjection(2)

    model.freeze_vlm()
    model.train()

    assert not model.paligemma_with_expert.paligemma.training
    assert not any(parameter.requires_grad for parameter in model.paligemma_with_expert.paligemma.parameters())
    assert all(parameter.requires_grad for parameter in model.paligemma_with_expert.gemma_expert.parameters())
    assert all(parameter.requires_grad for parameter in model.target_time_projection.parameters())


def test_two_step_shortcut_uses_local_target_times_and_stop_gradient():
    scale = torch.nn.Parameter(torch.tensor(2.0))
    calls = []

    def velocity_fn(x, time, target_time):
        calls.append((x.clone(), time.clone(), target_time.clone()))
        return scale * torch.ones_like(x)

    noise = torch.ones(2, 3, 4)
    shortcut = two_step_euler_shortcut(velocity_fn, noise)

    assert torch.equal(shortcut, torch.full_like(shortcut, 2.0))
    assert not shortcut.requires_grad
    assert len(calls) == 2
    assert torch.equal(calls[0][1], torch.ones(2))
    assert torch.equal(calls[0][2], torch.ones(2))
    assert torch.equal(calls[1][1], torch.full((2,), 0.5))
    assert torch.equal(calls[1][2], torch.full((2,), 0.5))


def test_objective_uses_openpi_time_direction_and_published_weights():
    calls = []

    def velocity_fn(x, time, target_time):
        calls.append((x.clone(), time.clone(), target_time.clone()))
        expand = target_time[:, None, None]
        return expand.expand_as(x)

    actions = torch.zeros(2, 2, 2)
    noise = torch.ones_like(actions)
    time = torch.tensor([0.25, 0.25])
    flow_matching_mask = torch.tensor([True, False])
    losses = snapflow_objective(
        velocity_fn,
        actions=actions,
        noise=noise,
        time=time,
        hyperparameters=SnapFlowHyperparameters(alpha=0.5, consistency_weight=0.1),
        flow_matching_mask=flow_matching_mask,
    )

    # FM: (0.25 - 1)^2 = 0.5625. Shortcut target=(1+0.5)/2=0.75,
    # global prediction at s=0 is zero, so shortcut MSE is also 0.5625.
    assert torch.allclose(losses.flow_matching[0], torch.full_like(losses.flow_matching[0], 0.5625))
    assert torch.allclose(losses.shortcut[1], torch.full_like(losses.shortcut[1], 0.5625))
    assert torch.allclose(losses.total[0], torch.full_like(losses.total[0], 0.5625))
    assert torch.allclose(losses.total[1], torch.full_like(losses.total[1], 0.05625))
    assert losses.total.mean() == pytest.approx(0.309375)
    assert torch.equal(losses.flow_matching_mask, flow_matching_mask)

    assert len(calls) == 3
    # The first expert pass mixes an FM sample (x_t, t, t) and consistency
    # sample (noise, 1, 0); the next two are local shortcut targets.
    assert torch.equal(calls[0][0][0], torch.full_like(noise[0], 0.25))
    assert torch.equal(calls[0][0][1], noise[1])
    assert torch.equal(calls[0][1], torch.tensor([0.25, 1.0]))
    assert torch.equal(calls[0][2], torch.tensor([0.25, 0.0]))
    assert torch.equal(calls[1][0], noise[1:2])
    assert torch.equal(calls[1][1], torch.ones(1))
    assert torch.equal(calls[1][2], torch.ones(1))
    assert torch.equal(calls[2][1], torch.full((1,), 0.5))
    assert torch.equal(calls[2][2], torch.full((1,), 0.5))


def test_consistency_subset_matches_full_batch_reference_and_reduces_target_batches():
    actions = torch.tensor(
        [
            [[-0.2, 0.1]],
            [[0.4, -0.3]],
            [[0.2, 0.7]],
            [[-0.5, 0.8]],
        ]
    )
    noise = torch.tensor(
        [
            [[0.9, -0.4]],
            [[-0.6, 0.2]],
            [[0.3, 0.5]],
            [[0.1, -0.7]],
        ]
    )
    time = torch.tensor([0.2, 0.4, 0.6, 0.8])
    flow_matching_mask = torch.tensor([True, False, True, False])
    hyperparameters = SnapFlowHyperparameters(alpha=0.5, consistency_weight=0.1)

    reference_scale = torch.nn.Parameter(torch.tensor(0.7))
    optimized_scale = torch.nn.Parameter(reference_scale.detach().clone())
    reference_batch_sizes = []
    optimized_batch_sizes = []

    def reference_velocity_fn(x, current_time, target_time):
        reference_batch_sizes.append(x.shape[0])
        time_term = (0.3 * current_time - 0.2 * target_time)[:, None, None]
        return reference_scale * x + time_term

    def optimized_velocity_fn(x, current_time, target_time):
        optimized_batch_sizes.append(x.shape[0])
        time_term = (0.3 * current_time - 0.2 * target_time)[:, None, None]
        return optimized_scale * x + time_term

    time_expanded = time[:, None, None]
    noisy_actions = time_expanded * noise + (1.0 - time_expanded) * actions
    ground_truth_velocity = noise - actions
    broadcast_mask = flow_matching_mask[:, None, None]
    mixed_input = torch.where(broadcast_mask, noisy_actions, noise)
    mixed_time = torch.where(flow_matching_mask, time, torch.ones_like(time))
    mixed_target_time = torch.where(flow_matching_mask, time, torch.zeros_like(time))
    reference_mixed_velocity = reference_velocity_fn(mixed_input, mixed_time, mixed_target_time)
    reference_flow_matching = torch.nn.functional.mse_loss(
        reference_mixed_velocity,
        ground_truth_velocity,
        reduction="none",
    )
    reference_shortcut_target = two_step_euler_shortcut(reference_velocity_fn, noise)
    reference_shortcut = torch.nn.functional.mse_loss(
        reference_mixed_velocity,
        reference_shortcut_target,
        reduction="none",
    )
    reference_total = torch.where(
        broadcast_mask,
        reference_flow_matching,
        hyperparameters.consistency_weight * reference_shortcut,
    )

    losses = snapflow_objective(
        optimized_velocity_fn,
        actions=actions,
        noise=noise,
        time=time,
        hyperparameters=hyperparameters,
        flow_matching_mask=flow_matching_mask,
    )

    assert torch.allclose(losses.total, reference_total)
    assert torch.allclose(losses.flow_matching, reference_flow_matching)
    assert torch.allclose(losses.shortcut[~flow_matching_mask], reference_shortcut[~flow_matching_mask])
    assert torch.count_nonzero(losses.shortcut[flow_matching_mask]) == 0
    assert reference_batch_sizes == [4, 4, 4]
    assert optimized_batch_sizes == [4, 2, 2]

    reference_total.mean().backward()
    losses.total.mean().backward()
    assert torch.allclose(optimized_scale.grad, reference_scale.grad)


def test_all_flow_matching_batch_skips_shortcut_target_calls():
    batch_sizes = []

    def velocity_fn(x, _time, _target_time):
        batch_sizes.append(x.shape[0])
        return torch.zeros_like(x)

    losses = snapflow_objective(
        velocity_fn,
        actions=torch.zeros(3, 2, 2),
        noise=torch.ones(3, 2, 2),
        time=torch.full((3,), 0.5),
        flow_matching_mask=torch.ones(3, dtype=torch.bool),
    )

    assert batch_sizes == [3]
    assert torch.count_nonzero(losses.shortcut) == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": -0.1}, "alpha"),
        ({"alpha": 1.1}, "alpha"),
        ({"consistency_weight": -1.0}, "consistency_weight"),
        ({"prediction_clamp": 0.0}, "prediction_clamp"),
    ],
)
def test_hyperparameters_reject_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SnapFlowHyperparameters(**kwargs)
