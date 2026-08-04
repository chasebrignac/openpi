#!/usr/bin/env python3
"""Exercise the explicit PyAV LeRobot decoder on a generated local video."""

from __future__ import annotations

import importlib
import pathlib
import tempfile

import av
import numpy as np
import torch


def _video_utils_module():
    try:
        return importlib.import_module("lerobot.datasets.video_utils")
    except ModuleNotFoundError:
        return importlib.import_module("lerobot.common.datasets.video_utils")


def smoke_pyav_decoder() -> None:
    with tempfile.TemporaryDirectory(prefix="pi05-pyav-smoke-") as temporary:
        path = pathlib.Path(temporary) / "sample.mp4"
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("mpeg4", rate=10)
            stream.width = 32
            stream.height = 24
            stream.pix_fmt = "yuv420p"
            for index in range(4):
                pixels = np.full((24, 32, 3), index * 48, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

        decode_video_frames = _video_utils_module().decode_video_frames
        frames = decode_video_frames(path, [0.0, 0.1], tolerance_s=0.11, backend="pyav")
        assert frames.shape == (2, 3, 24, 32), frames.shape
        assert frames.dtype == torch.float32, frames.dtype
        assert torch.isfinite(frames).all()
        assert float(frames.min()) >= 0.0
        assert float(frames.max()) <= 1.0
        print(f"pyav-decoder-smoke=passed shape={tuple(frames.shape)} av={av.__version__}")


if __name__ == "__main__":
    smoke_pyav_decoder()
