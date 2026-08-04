"""Deterministic, stratified calibration-corpus loading.

The corpus is a JSONL manifest.  Each non-empty line contains at least
``{"path": "chunk.npz", "stratum": "task-or-scene"}``; ``index`` selects a
sample when the NPZ arrays are batched.  Required NPZ keys are ``image_0`` ..
``image_N``, matching image masks, ``lang_tokens``, ``lang_mask``, ``state``
and ``noise``.  Images are channel-first and all graph inputs include a batch
dimension after loading.
"""

from __future__ import annotations

from collections import defaultdict
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
import dataclasses
import json
import pathlib
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class CalibrationRecord:
    path: pathlib.Path
    stratum: str
    index: int = 0
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


def load_calibration_manifest(path: pathlib.Path) -> list[CalibrationRecord]:
    records: list[CalibrationRecord] = []
    seen: set[tuple[pathlib.Path, int]] = set()
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        try:
            payload = json.loads(raw_line)
            record_path = pathlib.Path(payload["path"])
            stratum = str(payload["stratum"])
            index = int(payload.get("index", 0))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid calibration record at {path}:{line_number}: {exc}") from exc
        if not record_path.is_absolute():
            record_path = path.parent / record_path
        record_path = record_path.resolve()
        if not record_path.is_file():
            raise FileNotFoundError(f"Calibration NPZ not found at {path}:{line_number}: {record_path}")
        if not stratum:
            raise ValueError(f"Empty calibration stratum at {path}:{line_number}")
        if index < 0:
            raise ValueError(f"Negative calibration index at {path}:{line_number}")
        identity = (record_path, index)
        if identity in seen:
            raise ValueError(f"Duplicate calibration sample at {path}:{line_number}: {identity}")
        seen.add(identity)
        records.append(
            CalibrationRecord(
                path=record_path,
                stratum=stratum,
                index=index,
                metadata={key: value for key, value in payload.items() if key not in {"path", "stratum", "index"}},
            )
        )
    if not records:
        raise ValueError(f"Calibration manifest is empty: {path}")
    return records


def select_stratified(records: Sequence[CalibrationRecord], count: int) -> list[CalibrationRecord]:
    """Round-robin across strata while preserving order within each stratum."""

    if count < 1:
        raise ValueError("Calibration count must be positive")
    if len(records) < count:
        raise ValueError(f"Calibration manifest has {len(records)} samples; {count} are required")
    groups: dict[str, deque[CalibrationRecord]] = defaultdict(deque)
    for record in records:
        groups[record.stratum].append(record)
    if len(groups) < 2:
        raise ValueError("Stratified calibration requires at least two non-empty strata")

    selected: list[CalibrationRecord] = []
    strata = sorted(groups)
    while len(selected) < count:
        progressed = False
        for stratum in strata:
            if groups[stratum] and len(selected) < count:
                selected.append(groups[stratum].popleft())
                progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError(f"Could select only {len(selected)} of {count} requested calibration samples")
    return selected


def stratum_counts(records: Iterable[CalibrationRecord]) -> dict[str, int]:
    result: dict[str, int] = defaultdict(int)
    for record in records:
        result[record.stratum] += 1
    return dict(sorted(result.items()))


def _batched(array: np.ndarray, *, index: int, unbatched_rank: int, key: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == unbatched_rank:
        if index != 0:
            raise IndexError(f"{key} is unbatched but manifest requested index {index}")
        return array[None, ...]
    if array.ndim != unbatched_rank + 1:
        raise ValueError(
            f"{key} must have rank {unbatched_rank} (one sample) or {unbatched_rank + 1} (batched), got {array.shape}"
        )
    if index >= array.shape[0]:
        raise IndexError(f"{key} batch has {array.shape[0]} samples, index {index} was requested")
    return array[index : index + 1]


def load_record_arrays(record: CalibrationRecord, *, image_count: int = 3) -> dict[str, np.ndarray]:
    """Load one manifest record into the two graph's canonical input names."""

    with np.load(record.path, allow_pickle=False) as archive:
        required = {
            *(f"image_{index}" for index in range(image_count)),
            *(f"image_mask_{index}" for index in range(image_count)),
            "lang_tokens",
            "lang_mask",
            "state",
            "noise",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"Calibration NPZ {record.path} is missing keys: {missing}")

        result: dict[str, np.ndarray] = {}
        for index in range(image_count):
            image_key = f"image_{index}"
            mask_key = f"image_mask_{index}"
            result[image_key] = _batched(
                archive[image_key], index=record.index, unbatched_rank=3, key=image_key
            ).astype(np.float32, copy=False)
            result[mask_key] = _batched(archive[mask_key], index=record.index, unbatched_rank=0, key=mask_key).astype(
                np.bool_, copy=False
            )
        result["lang_tokens"] = _batched(
            archive["lang_tokens"], index=record.index, unbatched_rank=1, key="lang_tokens"
        ).astype(np.int64, copy=False)
        result["lang_mask"] = _batched(
            archive["lang_mask"], index=record.index, unbatched_rank=1, key="lang_mask"
        ).astype(np.bool_, copy=False)
        result["state"] = _batched(archive["state"], index=record.index, unbatched_rank=1, key="state").astype(
            np.float32, copy=False
        )
        result["noise"] = _batched(archive["noise"], index=record.index, unbatched_rank=2, key="noise").astype(
            np.float32, copy=False
        )
        for optional_key in ("action_low", "action_high"):
            if optional_key in archive:
                result[optional_key] = np.asarray(archive[optional_key], dtype=np.float32)
    return result


def prefix_feed(arrays: dict[str, np.ndarray], *, image_count: int = 3) -> dict[str, np.ndarray]:
    names = [*(f"image_{index}" for index in range(image_count))]
    names.extend(f"image_mask_{index}" for index in range(image_count))
    names.extend(("lang_tokens", "lang_mask"))
    return {name: arrays[name] for name in names}


FeedIterable = Iterable[dict[str, np.ndarray]]
FeedFactory = Callable[[], FeedIterable]


class StreamingCalibrationReader:
    """ModelOpt/ONNX Runtime compatible reader with deterministic rewind."""

    def __init__(self, feeds: FeedIterable | FeedFactory) -> None:
        self._feeds = feeds
        self._iterator: Iterator[dict[str, np.ndarray]] = iter(())
        self.rewind()

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._iterator, None)

    def rewind(self) -> None:
        source = self._feeds() if callable(self._feeds) else self._feeds
        self._iterator = iter(source)
