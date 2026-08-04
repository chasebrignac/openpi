from __future__ import annotations

import json

import numpy as np
import pytest

from openpi.exporting.calibration import CalibrationRecord
from openpi.exporting.calibration import StreamingCalibrationReader
from openpi.exporting.calibration import load_calibration_manifest
from openpi.exporting.calibration import load_record_arrays
from openpi.exporting.calibration import select_stratified


def _write_npz(path, *, batch=1):
    values = {
        **{f"image_{index}": np.zeros((batch, 3, 2, 2), np.float32) for index in range(3)},
        **{f"image_mask_{index}": np.ones((batch,), bool) for index in range(3)},
        "lang_tokens": np.zeros((batch, 4), np.int64),
        "lang_mask": np.ones((batch, 4), bool),
        "state": np.zeros((batch, 2), np.float32),
        "noise": np.zeros((batch, 3, 2), np.float32),
    }
    np.savez(path, **values)


def test_manifest_and_npz_loading(tmp_path):
    npz = tmp_path / "chunk.npz"
    _write_npz(npz, batch=2)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"path": npz.name, "stratum": "task-a", "index": 1}) + "\n")
    record = load_calibration_manifest(manifest)[0]
    arrays = load_record_arrays(record)
    assert arrays["image_0"].shape == (1, 3, 2, 2)
    assert arrays["noise"].shape == (1, 3, 2)


def test_stratified_selection_round_robins():
    records = [
        CalibrationRecord(path=__file__, stratum=stratum, index=index)
        for index, stratum in enumerate(("b", "a", "b", "a", "b", "a"))
    ]
    selected = select_stratified(records, 4)
    assert [record.stratum for record in selected] == ["a", "b", "a", "b"]
    assert [record.index for record in selected] == [1, 0, 3, 2]


def test_stratification_requires_two_groups():
    records = [CalibrationRecord(path=__file__, stratum="only", index=index) for index in range(2)]
    with pytest.raises(ValueError, match="at least two"):
        select_stratified(records, 2)


def test_reader_rewinds_callable_source():
    calls = 0

    def feeds():
        nonlocal calls
        calls += 1
        yield {"x": np.array([calls])}

    reader = StreamingCalibrationReader(feeds)
    assert reader.get_next()["x"].item() == 1
    assert reader.get_next() is None
    reader.rewind()
    assert reader.get_next()["x"].item() == 2
