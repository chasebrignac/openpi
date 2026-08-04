from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import repro_make_action_limits
from scripts import validate_pi05_onnx


def _stats(dim: int):
    return {
        "mean": np.ones(dim),
        "std": np.ones(dim),
        "q01": np.zeros(dim),
        "q99": np.full(dim, 2.0),
    }


def test_libero_envelope_masks_padding_and_derives_state_independent_physical_values():
    actions = np.zeros((2, 3, 8), dtype=np.float32)
    actions[0, :, :7] = -1.0
    actions[1, :, :7] = 1.0
    states = np.zeros((2, 8), dtype=np.float32)
    arrays, description = repro_make_action_limits.derive_envelopes(
        track="libero",
        action_sources=[actions],
        state_sources=[states],
        action_stats=_stats(7),
        state_stats=_stats(8),
        normalization="quantile",
        internal_margin=0.01,
    )

    np.testing.assert_array_equal(arrays["action_mask"], [True] * 7 + [False])
    assert np.all(arrays["action_low"][:7] < -1.01)
    assert np.all(arrays["action_high"][:7] > 1.01)
    np.testing.assert_array_equal(arrays["physical_mask"], np.ones(7, dtype=bool))
    assert not np.any(arrays["physical_state_dependent_mask"])
    np.testing.assert_allclose(arrays["physical_low"], 0.0, atol=2e-6)
    np.testing.assert_allclose(arrays["physical_high"], 2.0, atol=2e-6)
    assert description["internal_margin"] == 0.01


def test_droid_joint_physical_bounds_remain_unset_because_conversion_uses_state():
    actions = np.zeros((2, 3, 8), dtype=np.float32)
    states = np.zeros((2, 8), dtype=np.float32)
    arrays, description = repro_make_action_limits.derive_envelopes(
        track="droid",
        action_sources=[actions],
        state_sources=[states],
        action_stats=_stats(8),
        state_stats=_stats(8),
        normalization="quantile",
        internal_margin=0.01,
    )

    np.testing.assert_array_equal(arrays["physical_state_dependent_mask"], [True] * 7 + [False])
    np.testing.assert_array_equal(arrays["physical_mask"], [False] * 7 + [True])
    assert np.isnan(arrays["physical_low"][:7]).all()
    assert np.isnan(arrays["physical_high"][:7]).all()
    assert np.isfinite(arrays["physical_low"][7:]).all()
    assert description["physical_state_dependent_dimensions"] == list(range(7))


def test_action_artifact_seals_metadata_and_validator_rejects_tampering(tmp_path):
    path = tmp_path / "action-limits.normalized.npz"
    arrays = {
        "action_low": np.array([-1.0] * 7 + [0.0]),
        "action_high": np.array([1.0] * 7 + [0.0]),
        "action_mask": np.array([True] * 7 + [False]),
        "physical_low": np.full(7, -1.0),
        "physical_high": np.full(7, 1.0),
        "physical_mask": np.ones(7, dtype=bool),
        "physical_state_dependent_mask": np.zeros(7, dtype=bool),
    }
    metadata = {
        "schema_version": 1,
        "gate_kind": "corpus_envelope",
        "hardware_safety_guarantee": False,
        "track": "libero",
        "dataset": "dataset",
        "dataset_revision": "a" * 40,
        "sources": {
            "calibration_manifest": {"sha256": "b" * 64},
            "golden_corpus": {"sha256": "c" * 64},
            "norm_stats": {"sha256": "d" * 64},
        },
    }
    sidecar = repro_make_action_limits.write_artifact(path, arrays=arrays, metadata=metadata)
    low, high, mask, observed, observed_sidecar = validate_pi05_onnx._load_action_envelope(  # noqa: SLF001
        path,
        track="libero",
        dataset="dataset",
        dataset_revision="a" * 40,
    )
    np.testing.assert_array_equal(low, arrays["action_low"])
    np.testing.assert_array_equal(high, arrays["action_high"])
    np.testing.assert_array_equal(mask, arrays["action_mask"])
    assert observed == metadata
    assert observed_sidecar == sidecar

    payload = json.loads(sidecar.read_text())
    payload["track"] = "droid"
    sidecar.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=r"Embedded.*differs"):
        validate_pi05_onnx._load_action_envelope(  # noqa: SLF001
            path,
            track="libero",
            dataset="dataset",
            dataset_revision="a" * 40,
        )


def test_envelope_rejects_negative_margin():
    with pytest.raises(ValueError, match="non-negative"):
        repro_make_action_limits.derive_envelopes(
            track="libero",
            action_sources=[np.zeros((1, 1, 7))],
            state_sources=[np.zeros((1, 7))],
            action_stats=_stats(7),
            state_stats=_stats(7),
            normalization="quantile",
            internal_margin=-0.01,
        )
