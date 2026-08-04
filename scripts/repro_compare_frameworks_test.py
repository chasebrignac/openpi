import json

import numpy as np
import pytest

import openpi.models.model as model
import openpi.training.config as training_config
from scripts import repro_compare_frameworks
from scripts import repro_make_golden


def _corpus(image: np.ndarray) -> dict[str, np.ndarray]:
    batch = image.shape[0]
    return {
        "image__camera": image,
        "image_mask__camera": np.ones(batch, dtype=bool),
        "state": np.zeros((batch, 2), dtype=np.float32),
        "tokenized_prompt": np.ones((batch, 3), dtype=np.int32),
        "tokenized_prompt_mask": np.ones((batch, 3), dtype=bool),
    }


def test_observation_transposes_bchw_only_for_jax():
    image = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    jax_observation = repro_compare_frameworks.observation_from_corpus(
        _corpus(image),
        ["camera"],
        np.asarray,
        image_layout="BCHW",
        expects_bhwc=True,
    )
    torch_observation = repro_compare_frameworks.observation_from_corpus(
        _corpus(image),
        ["camera"],
        np.asarray,
        image_layout="BCHW",
        expects_bhwc=False,
    )
    assert jax_observation.images["camera"].shape == (2, 4, 5, 3)
    assert torch_observation.images["camera"].shape == (2, 3, 4, 5)
    np.testing.assert_array_equal(jax_observation.images["camera"][..., 0], image[:, 0])


def test_velocity_gate_uses_worst_sample_not_mean():
    reference = np.asarray([[[1.0, 0.0]], [[1.0, 0.0]]], dtype=np.float32)
    candidate = np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32)
    report = repro_compare_frameworks.velocity_report(candidate, reference)
    assert report["cosine_mean"] == 0.5
    assert report["cosine_min"] == 0.0
    assert report["gate_pass"] is False


def _golden_fixture(tmp_path):
    samples = 2
    config = training_config.get_config("pi05_libero_l09_distill")
    arrays = {
        "state": np.zeros((samples, 32), dtype=np.float32),
        "tokenized_prompt": np.ones((samples, 5), dtype=np.int32),
        "tokenized_prompt_mask": np.ones((samples, 5), dtype=bool),
        "actions": np.zeros((samples, 10, 32), dtype=np.float32),
        "noise": np.ones((samples, 10, 32), dtype=np.float32),
        "time": np.asarray([0.1, 0.9], dtype=np.float32),
    }
    for name in model.IMAGE_KEYS:
        arrays[f"image__{name}"] = np.zeros((samples, 3, 2, 2), dtype=np.float32)
        arrays[f"image_mask__{name}"] = np.ones(samples, dtype=bool)
    path = tmp_path / "golden.npz"
    np.savez_compressed(path, **arrays)
    metadata = {
        "schema_version": 2,
        "run_id": "unit-golden",
        "config_name": config.name,
        "resolved_config": json.loads(json.dumps(repro_make_golden.config_provenance(config))),
        "dataset": repro_make_golden.dataset_provenance(config),
        "dataset_revision": repro_make_golden.dataset_provenance(config)["revision"],
        "data_split_seed": config.seed,
        "data_split": {
            "schema_version": 1,
            "strategy": "deterministic_whole_episode_stratified",
            "seed": config.seed,
            "split": "validation",
            "requested_holdout_samples": config.offline_holdout_samples,
            "validation_frames": config.offline_holdout_samples,
            "validation_episode_ids": [7],
            "validation_episode_count": 1,
            "train_episode_count": 10,
            "selected_episode_count": 1,
            "stratified_by_task": True,
            "stratified_by_site": False,
        },
        "seed": 7,
        "samples": samples,
        "action_horizon": 10,
        "action_dim": 32,
        "image_names": list(model.IMAGE_KEYS),
        "image_layout": "BCHW",
        "sha256": repro_make_golden.sha256_file(path),
    }
    path.with_suffix(".json").write_text(json.dumps(metadata))
    return path


def test_golden_corpus_is_bound_to_bytes_config_and_teacher(tmp_path):
    path = _golden_fixture(tmp_path)
    teacher = training_config.get_config("pi05_libero").model

    metadata, corpus = repro_compare_frameworks.load_golden_corpus(
        path,
        teacher_config_name="pi05_libero",
        model_config=teacher,
    )

    assert metadata["sha256"] == repro_make_golden.sha256_file(path)
    assert corpus["actions"].shape == (2, 10, 32)

    sidecar_path = path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["sha256"] = "0" * 64
    sidecar_path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="SHA-256"):
        repro_compare_frameworks.load_golden_corpus(
            path,
            teacher_config_name="pi05_libero",
            model_config=teacher,
        )


def test_golden_corpus_rejects_a_different_teacher(tmp_path):
    path = _golden_fixture(tmp_path)
    teacher = training_config.get_config("pi05_droid_jointpos").model
    with pytest.raises(ValueError, match="teacher mismatch"):
        repro_compare_frameworks.load_golden_corpus(
            path,
            teacher_config_name="pi05_droid_jointpos",
            model_config=teacher,
        )


def test_checkpoint_manifests_bind_actual_converted_bytes(tmp_path):
    teacher = training_config.get_config("pi05_libero").model
    source_root = tmp_path / "pi05_libero"
    (source_root / "params").mkdir(parents=True)
    pytorch_root = tmp_path / "pi05_libero_pytorch"
    pytorch_root.mkdir()
    weight_path = pytorch_root / "model.safetensors"
    weight_path.write_bytes(b"converted model")
    conversion_config = {
        "schema_version": 1,
        "config_name": "pi05_libero",
        "pi05": True,
        "action_dim": 32,
        "action_horizon": 10,
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "precision": "bfloat16",
    }
    config_path = pytorch_root / "config.json"
    config_path.write_text(json.dumps(conversion_config))
    _, _, _, model_sha256, config_sha256 = repro_compare_frameworks.validate_converted_checkpoint(
        pytorch_root,
        config_name="pi05_libero",
        model_config=teacher,
    )

    source_manifest_path = tmp_path / "source.json"
    source_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": {"revision": "source-revision"},
                "checkpoint": {"key": "libero", "local_dirname": source_root.name},
            }
        )
    )
    converted_manifest = {
        "schema_version": 1,
        "source": {"revision": "converted-revision", "upstream": {"revision": "source-revision"}},
        "conversion": {
            "config_name": "pi05_libero",
            "precision": "bfloat16",
            "source_commit": "a" * 40,
            "image_digest": "sha256:" + "b" * 64,
        },
        "checkpoint": {"key": "libero", "local_dirname": pytorch_root.name},
        "files": [
            {"path": "model.safetensors", "sha256": model_sha256},
            {"path": "config.json", "sha256": config_sha256},
        ],
    }
    converted_manifest_path = tmp_path / "converted.json"
    converted_manifest_path.write_text(json.dumps(converted_manifest))

    provenance = repro_compare_frameworks.validate_checkpoint_manifests(
        source_manifest_path=source_manifest_path,
        converted_manifest_path=converted_manifest_path,
        jax_checkpoint=source_root,
        pytorch_root=pytorch_root,
        config_name="pi05_libero",
        model_sha256=model_sha256,
        config_sha256=config_sha256,
    )
    assert provenance["source"]["revision"] == "source-revision"
    assert provenance["converted"]["revision"] == "converted-revision"

    converted_manifest["source"]["upstream"]["revision"] = "wrong-source"
    converted_manifest_path.write_text(json.dumps(converted_manifest))
    with pytest.raises(ValueError, match="not bound"):
        repro_compare_frameworks.validate_checkpoint_manifests(
            source_manifest_path=source_manifest_path,
            converted_manifest_path=converted_manifest_path,
            jax_checkpoint=source_root,
            pytorch_root=pytorch_root,
            config_name="pi05_libero",
            model_sha256=model_sha256,
            config_sha256=config_sha256,
        )
