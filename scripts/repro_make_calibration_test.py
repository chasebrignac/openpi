from types import SimpleNamespace

import numpy as np
import pytest

from openpi.exporting.calibration import load_calibration_manifest
from openpi.exporting.calibration import load_record_arrays
from scripts import repro_make_calibration


def _batch(prompts: list[int], *, image_offset: int = 0):
    batch = len(prompts)
    observation = SimpleNamespace(
        images={
            "base_0_rgb": np.full((batch, 2, 2, 3), image_offset, dtype=np.uint8),
            "left_wrist_0_rgb": np.full((batch, 3, 2, 2), -0.5, dtype=np.float32),
            "right_wrist_0_rgb": np.zeros((batch, 2, 2, 3), dtype=np.uint8),
        },
        image_masks={
            "base_0_rgb": np.ones(batch, dtype=bool),
            "left_wrist_0_rgb": np.ones(batch, dtype=bool),
            "right_wrist_0_rgb": np.zeros(batch, dtype=bool),
        },
        state=np.zeros((batch, 8), dtype=np.float32),
        tokenized_prompt=np.asarray([[prompt, 0, 0] for prompt in prompts], dtype=np.int32),
        tokenized_prompt_mask=np.asarray([[True, False, False] for _ in prompts]),
    )
    actions = np.zeros((batch, 3, 8), dtype=np.float32)
    return observation, actions


def test_calibration_batch_matches_export_schema_and_is_chunk_invariant():
    observation, actions = _batch([11, 12])
    arrays, image_names, strata, _ = repro_make_calibration.calibration_batch(
        observation, actions, seed=7, start_ordinal=3
    )
    assert image_names == ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    assert arrays["image_0"].shape == (2, 3, 2, 2)
    assert arrays["image_0"].dtype == np.float32
    assert arrays["lang_tokens"].dtype == np.int64
    assert arrays["noise"].shape == (2, 3, 8)
    assert arrays["actions"].shape == (2, 3, 8)
    assert arrays["actions"].dtype == np.float32
    assert len(set(strata)) == 2

    single_observation, single_actions = _batch([12])
    single, _, _, _ = repro_make_calibration.calibration_batch(
        single_observation, single_actions, seed=7, start_ordinal=4
    )
    np.testing.assert_array_equal(arrays["noise"][1], single["noise"][0])


def test_evenly_spaced_indices_are_unique_and_cover_the_dataset():
    indices = repro_make_calibration.evenly_spaced_indices(17_758_044, 1_024)
    assert len(indices) == len(set(indices)) == 1_024
    assert indices == sorted(indices)
    assert indices[0] < 17_758_044 // 1_024
    assert indices[-1] >= 17_758_044 - 17_758_044 // 1_024


def test_write_and_validate_corpus_reuses_export_loader(tmp_path):
    manifest = repro_make_calibration.write_corpus(
        [_batch([1, 2]), _batch([1, 2], image_offset=1)],
        output_dir=tmp_path / "calibration",
        seed=9,
        config_name="test",
        dataset="test/dataset",
        dataset_revision="a" * 40,
        expected_samples=4,
        source_indices=[3, 7, 11, 15],
    )
    summary = repro_make_calibration.validate_corpus(manifest, expected_samples=4)
    assert summary["sample_count"] == 4
    assert summary["unique_record_count"] == 4
    assert len(summary["stratum_counts"]) == 2
    assert [record.metadata["dataset_index"] for record in load_calibration_manifest(manifest)] == [3, 7, 11, 15]

    record = load_calibration_manifest(manifest)[0]
    arrays = load_record_arrays(record)
    assert arrays["image_0"].shape == (1, 3, 2, 2)
    assert arrays["noise"].shape == (1, 3, 8)


def test_calibration_batch_rejects_nonfinite_action_targets():
    observation, actions = _batch([11])
    actions[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="actions contain"):
        repro_make_calibration.calibration_batch(observation, actions, seed=7, start_ordinal=0)


def test_validation_rejects_one_stratum(tmp_path):
    with pytest.raises(ValueError, match="at least two"):
        repro_make_calibration.write_corpus(
            [_batch([1, 1, 1, 1])],
            output_dir=tmp_path / "one-stratum",
            seed=0,
            config_name="test",
            dataset="test/dataset",
            dataset_revision="b" * 40,
            expected_samples=4,
        )


def test_writer_refuses_to_overwrite_existing_corpus(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("owned by an earlier run")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        repro_make_calibration.write_corpus(
            [_batch([1, 2])],
            output_dir=output,
            seed=0,
            config_name="test",
            dataset="test/dataset",
            dataset_revision="c" * 40,
            expected_samples=2,
        )
