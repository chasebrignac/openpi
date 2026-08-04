from types import SimpleNamespace

import numpy as np
import pytest

from openpi.training import config as training_config
from scripts import repro_make_golden


def test_corpus_noise_is_reproducible():
    observation = SimpleNamespace(
        images={"base": np.zeros((2, 3, 4, 4), dtype=np.float32)},
        image_masks={"base": np.ones(2, dtype=bool)},
        state=np.zeros((2, 8), dtype=np.float32),
        tokenized_prompt=np.ones((2, 5), dtype=np.int32),
        tokenized_prompt_mask=np.ones((2, 5), dtype=bool),
    )
    actions = np.zeros((2, 3, 8), dtype=np.float32)
    first, metadata = repro_make_golden.corpus_arrays(observation, actions, seed=7)
    second, _ = repro_make_golden.corpus_arrays(observation, actions, seed=7)
    np.testing.assert_array_equal(first["noise"], second["noise"])
    np.testing.assert_array_equal(first["time"], second["time"])
    rng = np.random.default_rng(7)
    rng.standard_normal(actions.shape, dtype=np.float32)
    expected_time = (rng.beta(1.5, 1.0, size=2) * 0.999 + 0.001).astype(np.float32)
    np.testing.assert_array_equal(first["time"], expected_time)
    assert metadata["image_names"] == ["base"]
    assert metadata["image_layout"] == "BCHW"


def test_corpus_rejects_ambiguous_non_model_image_layout():
    observation = SimpleNamespace(
        images={"base": np.zeros((1, 4, 4, 3), dtype=np.float32)},
        image_masks={"base": np.ones(1, dtype=bool)},
        state=np.zeros((1, 8), dtype=np.float32),
        tokenized_prompt=np.ones((1, 5), dtype=np.int32),
        tokenized_prompt_mask=np.ones((1, 5), dtype=bool),
    )
    actions = np.zeros((1, 3, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="BCHW"):
        repro_make_golden.corpus_arrays(observation, actions, seed=0)


def test_resolved_config_provenance_pins_dataset_and_model_shape():
    config = training_config.get_config("pi05_libero_l09_distill")
    provenance = repro_make_golden.config_provenance(config)
    assert provenance["dataset"] == {
        "repo_id": "physical-intelligence/libero",
        "revision": "a4336d589d589045d1c56423ffdf3b88a0e19b1f",
        "codebase_version": "v2.0",
    }
    assert provenance["action_horizon"] == 10
    assert provenance["action_dim"] == 32
    assert len(provenance["fingerprint_sha256"]) == 64


def test_rejects_dataset_revision_not_pinned_by_config():
    config = training_config.get_config("pi05_libero_l09_distill")
    with pytest.raises(ValueError, match="does not match resolved config"):
        repro_make_golden.validate_requested_dataset_revision(config, "moving-tag")


def test_canonical_corpus_refuses_to_replace_either_output(tmp_path):
    for index, existing in enumerate(("corpus.npz", "corpus.json")):
        case = tmp_path / str(index)
        case.mkdir()
        (case / existing).write_bytes(b"keep")
        with pytest.raises(FileExistsError, match="already exists"):
            repro_make_golden.require_new_corpus_paths(case / "corpus.npz")


def test_validation_split_provenance_must_use_training_seed_and_whole_episodes():
    config = training_config.get_config("pi05_libero_l09_distill")
    metadata = {
        "schema_version": 1,
        "strategy": "deterministic_whole_episode_stratified",
        "seed": config.seed,
        "split": "validation",
        "requested_holdout_samples": config.offline_holdout_samples,
        "validation_frames": 300,
        "validation_episode_ids": [3, 11],
        "validation_episode_count": 2,
        "train_episode_count": 80,
        "selected_episode_count": 2,
        "stratified_by_task": True,
        "stratified_by_site": False,
    }
    assert repro_make_golden.validate_validation_split_metadata(metadata, config)["validation_episode_ids"] == [3, 11]
    metadata["seed"] = 7001
    with pytest.raises(ValueError, match="differs from training"):
        repro_make_golden.validate_validation_split_metadata(metadata, config)
