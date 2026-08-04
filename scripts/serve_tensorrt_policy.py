#!/usr/bin/env python3
"""Serve a validated split TensorRT pi0.5 policy over the existing WebSocket protocol."""

from __future__ import annotations

import dataclasses
import enum
import logging
import pathlib
import random
import socket

import numpy as np
import torch
import tyro

from openpi.exporting import tensorrt_policy
from openpi.policies import policy as policy_module
from openpi.serving import websocket_policy_server


class Precision(enum.Enum):
    BF16 = "bf16"
    FP8 = "fp8"


class Track(enum.Enum):
    LIBERO = "libero"
    DROID = "droid"


@dataclasses.dataclass(frozen=True)
class Args:
    artifact_dir: pathlib.Path
    checkpoint_dir: pathlib.Path
    precision: Precision
    track: Track
    dataset: str
    dataset_revision: str
    image_digest: str
    instance_type: str
    instance_id: str
    port: int = 8000
    seed: int = 0
    default_prompt: str | None = None
    record: bool = False


def seed_process(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_policy(args: Args):
    import tensorrt as trt

    runtime = tensorrt_policy.RuntimeIdentity(
        image_digest=args.image_digest,
        instance_type=args.instance_type,
        instance_id=args.instance_id,
        gpu_inventory=tensorrt_policy.gpu_inventory(),
        tensorrt_version=trt.__version__,
    )
    bundle = tensorrt_policy.load_artifact_bundle(
        args.artifact_dir,
        precision=args.precision.value,
        track=args.track.value,
        dataset=args.dataset,
        dataset_revision=args.dataset_revision,
        runtime=runtime,
    )
    return tensorrt_policy.create_policy(
        bundle,
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
    )


def main(args: Args) -> None:
    from openpi.exporting.runtime_identity import require_live_runtime_identity

    require_live_runtime_identity(
        image_digest=args.image_digest,
        instance_type=args.instance_type,
        instance_id=args.instance_id,
    )
    from openpi.exporting.artifacts import require_clean_source_identity

    require_clean_source_identity()
    seed_process(args.seed)
    policy = create_policy(args)
    metadata = policy.metadata
    if args.record:
        policy = policy_module.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    logging.info("Creating TensorRT policy server (host: %s, ip: %s)", hostname, socket.gethostbyname(hostname))
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
