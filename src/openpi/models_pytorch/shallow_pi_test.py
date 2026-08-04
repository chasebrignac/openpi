import dataclasses

import numpy as np
import pytest
import torch
from torch import nn

from openpi import transforms
from openpi.models import pi0_config
from openpi.models_pytorch import shallow_pi
from openpi.policies import droid_policy
from openpi.training import config as training_config


def _toy_state(depth: int) -> dict[str, torch.Tensor]:
    state = {"head.weight": torch.tensor([-1.0])}
    for stack_index, stack in enumerate(("vlm.layers", "expert.layers")):
        for layer in range(depth):
            state[f"{stack}.{layer}.weight"] = torch.tensor([stack_index * 100 + layer], dtype=torch.float32)
    return state


def test_build_shallow_state_dict_maps_both_transformer_stacks():
    student_state = _toy_state(9)
    teacher_state = _toy_state(18)

    mapped, report = shallow_pi.build_shallow_state_dict(
        student_state,
        teacher_state,
        layer_stacks=("vlm.layers", "expert.layers"),
    )

    for student_layer, teacher_layer in enumerate(pi0_config.SHALLOW_PI_LAYER_MAP):
        assert mapped[f"vlm.layers.{student_layer}.weight"].item() == teacher_layer
        assert mapped[f"expert.layers.{student_layer}.weight"].item() == 100 + teacher_layer
    assert mapped["head.weight"].item() == -1
    assert report.mapped_layer_keys == {"vlm.layers": 9, "expert.layers": 9}


def test_build_shallow_state_dict_rejects_a_different_map():
    with pytest.raises(ValueError, match="validated 18-to-9"):
        shallow_pi.build_shallow_state_dict(
            _toy_state(9),
            _toy_state(18),
            layer_map=(0, 2, 4, 6, 8, 10, 12, 14, 17),
            layer_stacks=("vlm.layers", "expert.layers"),
        )


class _FakeStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.velocity = nn.Parameter(torch.tensor(0.5))
        self.preprocessed = object()

    def preprocess_observation(self, observation, *, train):
        assert observation == "raw"
        assert train
        return self.preprocessed

    def sample_noise(self, shape, device):
        return torch.ones(shape, device=device)

    def sample_time(self, batch_size, device):
        return torch.full((batch_size,), 0.25, device=device)

    def forward(
        self,
        observation,
        actions,
        noise,
        time,
        *,
        observation_is_preprocessed,
        return_velocity,
    ):
        self.call = (observation, noise, time)
        assert observation_is_preprocessed
        assert return_velocity
        return self.velocity.expand_as(actions)


class _FakeTeacher(nn.Module):
    def predict_velocity(
        self,
        observation,
        actions,
        noise,
        time,
        *,
        observation_is_preprocessed,
    ):
        self.call = (observation, noise, time)
        self.grad_was_enabled = torch.is_grad_enabled()
        assert observation_is_preprocessed
        return torch.full_like(actions, 2.0)


def test_distillation_reuses_preprocessing_noise_and_time_and_freezes_teacher_graph():
    student = _FakeStudent()
    teacher = _FakeTeacher()
    teacher.train()
    actions = torch.zeros((2, 3, 2))

    output = shallow_pi.compute_distillation_loss(
        student,
        teacher,
        "raw",
        actions,
        fm_loss_weight=1.0,
        kd_loss_weight=1.0,
    )

    assert output.loss.item() == pytest.approx(2.5)
    assert output.fm_mse.item() == pytest.approx(0.25)
    assert output.kd_mse.item() == pytest.approx(2.25)
    assert output.kd_cosine.item() == pytest.approx(1.0)
    assert student.call[0] is teacher.call[0] is student.preprocessed
    assert student.call[1] is teacher.call[1]
    assert student.call[2] is teacher.call[2]
    assert not teacher.grad_was_enabled
    assert not teacher.training
    output.loss.backward()
    assert student.velocity.grad is not None


def test_distillation_configs_encode_loss_and_dataset_contracts():
    libero = training_config.get_config("pi05_libero_l09_distill")
    droid = training_config.get_config("pi05_droid_l09_distill")
    libero_snapflow = training_config.get_config("pi05_libero_l09_snapflow")
    droid_snapflow = training_config.get_config("pi05_droid_l09_snapflow")

    assert isinstance(libero.model, pi0_config.DistilledPi0Config)
    assert libero.model.pytorch_layer_map == pi0_config.SHALLOW_PI_LAYER_MAP
    assert (libero.model.fm_loss_weight, libero.model.kd_loss_weight) == (1.0, 1.0)
    assert (droid.model.fm_loss_weight, droid.model.kd_loss_weight) == (0.0, 1.0)
    assert droid.data.repo_id == "allenai/MolmoAct2-DROID-Dataset"
    assert isinstance(droid.data, training_config.MolmoAct2DROIDDataConfig)
    assert droid.model.teacher_config == "pi05_droid_jointpos"
    assert droid.teacher_pytorch_weight_path == "/mnt/openpi/checkpoints/pi05_droid_jointpos_pytorch"
    assert droid.data.assets.assets_dir == "/mnt/openpi/checkpoints/pi05_droid_jointpos/assets"
    assert libero.data.base_config.lerobot_revision == training_config.LIBERO_REVISION
    assert libero.data.base_config.lerobot_codebase_version == "v2.0"
    assert droid.data.base_config.lerobot_revision == training_config.MOLMOACT2_DROID_REVISION
    assert droid.data.base_config.lerobot_codebase_version == "v3.0"
    assert droid.data.base_config.episode_prompt_path == "meta/tasks_annotated.parquet"
    assert libero.num_train_steps == droid.num_train_steps == 30_000
    assert libero.batch_size * libero.gradient_accumulation_steps == 64
    assert droid.batch_size * droid.gradient_accumulation_steps == 64
    assert isinstance(libero_snapflow.model, pi0_config.SnapFlowPi0Config)
    assert isinstance(droid_snapflow.model, pi0_config.SnapFlowPi0Config)
    assert libero_snapflow.batch_size == droid_snapflow.batch_size == 4
    assert libero_snapflow.gradient_accumulation_steps == droid_snapflow.gradient_accumulation_steps == 1
    assert libero_snapflow.pytorch_training_precision == droid_snapflow.pytorch_training_precision == "bfloat16"
    assert libero_snapflow.lr_schedule.warmup_steps == droid_snapflow.lr_schedule.warmup_steps == 500
    assert libero_snapflow.lr_schedule.peak_lr == droid_snapflow.lr_schedule.peak_lr == 2.5e-5
    assert isinstance(droid_snapflow.data, training_config.MolmoAct2DROIDDataConfig)
    assert droid_snapflow.data.assets.assets_dir == "/mnt/openpi/checkpoints/pi05_droid_jointpos/assets"
    assert droid_snapflow.data.base_config.lerobot_revision == training_config.MOLMOACT2_DROID_REVISION
    assert droid_snapflow.data.base_config.episode_prompt_path == "meta/tasks_annotated.parquet"


def test_robolab_jointpos_inference_config_matches_pinned_reference():
    config = training_config.get_config("pi05_droid_jointpos")
    data_transforms = config.data.data_transforms(config.model)

    assert training_config.PI05_DROID_JOINTPOS_CHECKPOINT == "gs://openpi-assets-simeval/pi05_droid_jointpos"
    assert config.model.pi05
    assert config.model.action_horizon == 15
    assert config.data.assets.asset_id == "droid"
    assert [type(transform) for transform in data_transforms.inputs] == [droid_policy.DroidInputs]
    assert [type(transform) for transform in data_transforms.outputs] == [
        transforms.AbsoluteActions,
        droid_policy.DroidOutputs,
    ]
    assert tuple(data_transforms.outputs[0].mask) == (True,) * 7 + (False,)


def test_molmoact2_absolute_joint_actions_round_trip_through_delta_contract(tmp_path):
    config = training_config.get_config("pi05_droid_l09_distill")
    data_factory = dataclasses.replace(
        config.data,
        assets=training_config.AssetsConfig(assets_dir=str(tmp_path), asset_id="droid"),
    )
    data_config = data_factory.create(tmp_path, config.model)

    assert [type(transform) for transform in data_config.data_transforms.inputs] == [
        droid_policy.DroidInputs,
        transforms.DeltaActions,
    ]
    assert [type(transform) for transform in data_config.data_transforms.outputs] == [
        transforms.AbsoluteActions,
        droid_policy.DroidOutputs,
    ]
    assert tuple(data_config.data_transforms.inputs[-1].mask) == (True,) * 7 + (False,)
    assert tuple(data_config.data_transforms.outputs[0].mask) == (True,) * 7 + (False,)

    joint_state = np.linspace(-0.6, 0.6, 7, dtype=np.float32)
    absolute_actions = np.stack(
        [
            np.concatenate([joint_state + 0.1, np.array([0.2], dtype=np.float32)]),
            np.concatenate([joint_state - 0.2, np.array([0.8], dtype=np.float32)]),
        ]
    )
    raw = {
        "observation.images.exterior_1_left": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation.images.wrist_left": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation.state.joint_position": joint_state,
        "observation.state.gripper_position": np.array([0.4], dtype=np.float32),
        "action": absolute_actions.copy(),
        "prompt": "put the banana in the bowl",
    }
    repacked = transforms.compose(data_config.repack_transforms.inputs)(raw)
    model_data = transforms.compose(data_config.data_transforms.inputs)(repacked)

    np.testing.assert_allclose(model_data["actions"][..., :7], absolute_actions[..., :7] - joint_state)
    np.testing.assert_allclose(model_data["actions"][..., 7], absolute_actions[..., 7])

    restored = transforms.compose(data_config.data_transforms.outputs)(
        {"state": model_data["state"].copy(), "actions": model_data["actions"].copy()}
    )
    np.testing.assert_allclose(restored["actions"], absolute_actions)


def test_jointpos_changes_do_not_change_libero_action_semantics():
    libero = training_config.get_config("pi05_libero_l09_distill")
    libero_snapflow = training_config.get_config("pi05_libero_l09_snapflow")

    assert libero.model.teacher_config == "pi05_libero"
    assert not libero.data.extra_delta_transform
    assert not libero_snapflow.data.extra_delta_transform
    assert libero.data.assets.assets_dir == libero_snapflow.data.assets.assets_dir


def test_teacher_config_restores_full_depth_model_contract():
    student_config = pi0_config.DistilledPi0Config(pi05=True, action_horizon=10)
    teacher_config = shallow_pi.teacher_config_from_student(student_config)

    assert type(teacher_config) is pi0_config.Pi0Config
    assert teacher_config.action_horizon == student_config.action_horizon
    assert teacher_config.pytorch_compile_mode is None
