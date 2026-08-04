import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


@pytest.mark.parametrize(
    ("name", "fraction"), [("pi05_droid_l09_expert_bc_25", 0.25), ("pi05_droid_l09_expert_bc_50", 0.5)]
)
def test_expert_bc_configs_are_bounded_shallow_flow_matching_runs(name: str, fraction: float):
    config = _config.get_config(name)
    assert isinstance(config.model, pi0_config.ShallowPi0Config)
    assert config.model.pytorch_gemma_depth == 9
    assert config.model.source_layer_map == pi0_config.SHALLOW_PI_LAYER_MAP
    assert config.pytorch_weight_path == "/mnt/openpi/checkpoints/pi05_droid_l09_distill"
    assert config.teacher_pytorch_weight_path is None
    assert config.num_train_steps == 1_500
    assert config.data.expert_fraction == fraction
    assert config.offline_holdout_samples == 0


def test_50_50_factory_fails_without_a_paired_rerun_decision():
    factory = _config.MolmoAct2DROIDExpertBCDataConfig(
        repo_id="allenai/MolmoAct2-DROID-Dataset",
        expert_manifest_path="manifest.json",
        expert_fraction=0.5,
    )
    with pytest.raises(ValueError, match="requires an approved rerun decision"):
        factory.create(
            _config.get_config("pi05_droid_l09_expert_bc_50").assets_dirs,
            pi0_config.ShallowPi0Config(pi05=True),
        )


def test_shallow_bc_architecture_rejects_unvalidated_depth():
    with pytest.raises(ValueError, match="validated 18-to-9"):
        pi0_config.ShallowPi0Config(pi05=True, pytorch_gemma_depth=8)


def test_torch_loader_integrates_exact_expert_mix_and_records_provenance(monkeypatch):
    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=8,
        action_horizon=4,
        max_token_len=16,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )

    class FakeExpertDataset:
        def __init__(self, _manifest_path, *, action_horizon, verify_hashes=True):
            assert action_horizon == 4
            assert verify_hashes is True
            self.manifest = {
                "manifest_sha256": "a" * 64,
                "provenance": {
                    "trigger": {"task": "Stack3RubiksCubeTask", "success_gap": 0.06, "fired": True},
                    "accepted_shallow_checkpoint": {"model_sha256": "b" * 64},
                },
                "selection": {"selected_trajectories": 2, "selected_frames": 8},
            }
            self._dataset = _data_loader.FakeDataset(model_config, 8)

        def __len__(self):
            return len(self._dataset)

        def __getitem__(self, index):
            return self._dataset[index]

    monkeypatch.setattr(_data_loader._robolab_expert, "RoboLabExpertDataset", FakeExpertDataset)  # noqa: SLF001
    data_config = _config.DataConfig(
        repo_id="fake",
        robolab_expert_manifest_path="manifest.json",
        robolab_expert_fraction=0.25,
        robolab_expert_seed=7,
    )
    loader = _data_loader.create_torch_data_loader(
        data_config,
        model_config,
        action_horizon=4,
        batch_size=4,
        framework="pytorch",
        shuffle=True,
        num_batches=2,
        skip_norm_stats=True,
    )

    assert len(list(loader)) == 2
    provenance = loader.data_config().recovery_provenance
    assert provenance is not None
    assert provenance["manifest_sha256"] == "a" * 64
    assert provenance["mix"]["expert_fraction"] == 0.25
    assert provenance["teacher_checkpoint_resident"] is False
