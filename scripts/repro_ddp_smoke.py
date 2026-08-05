#!/usr/bin/env python3
"""Run a bounded CUDA/NCCL DistributedDataParallel smoke test.

The script is intentionally dataset-independent.  It proves that every rank can
join the intended process group, exchange CUDA tensors, synchronize gradients,
take identical optimizer steps, and reload a rank-local checkpoint.  It emits a
small JSON report and checkpoint per rank for external provenance capture.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import ipaddress
import json
import math
import os
import pathlib
import platform
import re
import socket
import sys
import time
import traceback
from typing import Any

import torch
import torch.distributed as dist

UTC = getattr(dt, "UTC", dt.timezone.utc)  # noqa: UP017 -- supports the pinned Ubuntu host Python.
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,126}[a-z0-9]$")


def _utc_now() -> str:
    return dt.datetime.now(UTC).isoformat()


def _required_rank_environment() -> tuple[int, int, int]:
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except (KeyError, ValueError) as exc:
        raise RuntimeError("RANK, LOCAL_RANK and WORLD_SIZE must be explicit integers") from exc
    if rank < 0 or local_rank < 0 or world_size < 1 or rank >= world_size:
        raise RuntimeError("distributed rank environment is out of range")
    return rank, local_rank, world_size


def _tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _all_gather_object(value: Any, world_size: int) -> list[Any]:
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, value)
    return gathered


def _all_tensors_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value.detach()).all().item())
    if isinstance(value, dict):
        return all(_all_tensors_finite(item) for item in value.values())
    if isinstance(value, list | tuple):
        return all(_all_tensors_finite(item) for item in value)
    return True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--expected-world-size", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--expected-private-ip", required=True)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.expected_world_size != 2:
        raise ValueError("this bounded smoke test requires exactly two ranks")
    if not 1 <= args.iterations <= 100:
        raise ValueError("iterations must be between one and 100")
    if not 1 <= args.batch_size <= 256:
        raise ValueError("batch size must be between one and 256")
    if SHA256_RE.fullmatch(args.script_sha256) is None:
        raise ValueError("script SHA-256 must be lowercase hexadecimal")
    if RUN_ID_RE.fullmatch(args.run_id) is None:
        raise ValueError("run id has an invalid shape")
    if not args.image_digest.startswith("sha256:") or SHA256_RE.fullmatch(args.image_digest[7:]) is None:
        raise ValueError("image digest must be an exact sha256 digest")
    try:
        address = ipaddress.ip_address(args.expected_private_ip)
    except ValueError as exc:
        raise ValueError("expected private IP must be a valid address") from exc
    if address.version != 4 or not address.is_private:
        raise ValueError("expected private IP must be a private IPv4 address")


def _write_report(output_dir: pathlib.Path, rank: int, report: dict[str, Any]) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"rank-{rank}.json"
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def _run(args: argparse.Namespace) -> dict[str, Any]:
    rank, local_rank, world_size = _required_rank_environment()
    if world_size != args.expected_world_size:
        raise RuntimeError(f"expected world size {args.expected_world_size}, got {world_size}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError("LOCAL_RANK exceeds the visible CUDA device count")
    actual_script_sha256 = _file_sha256(pathlib.Path(__file__).resolve())
    if actual_script_sha256 != args.script_sha256:
        raise RuntimeError("the executing smoke script differs from its exact SHA-256 binding")

    torch.cuda.set_device(local_rank)
    init_started = time.perf_counter()
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=dt.timedelta(minutes=10),
        device_id=torch.device("cuda", local_rank),
    )
    init_seconds = time.perf_counter() - init_started

    try:
        dist.barrier(device_ids=[local_rank])
        identity = {
            "rank": rank,
            "hostname": socket.gethostname(),
            "private_ip": args.expected_private_ip,
            "gpu_name": torch.cuda.get_device_name(local_rank),
            "gpu_total_memory": torch.cuda.get_device_properties(local_rank).total_memory,
        }
        peers = _all_gather_object(identity, world_size)
        if len({peer["hostname"] for peer in peers}) != world_size:
            raise RuntimeError("the two ranks did not originate on distinct hosts")

        reduced = torch.tensor([float(rank + 1)], device="cuda")
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        expected_sum = world_size * (world_size + 1) / 2
        if not bool(torch.isfinite(reduced).all()) or reduced.item() != expected_sum:
            raise RuntimeError(f"NCCL all-reduce returned {reduced.item()}, expected {expected_sum}")

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model = torch.nn.Sequential(
            torch.nn.Linear(256, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 64),
        ).cuda(local_rank)
        ddp = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.01, momentum=0.9)

        losses: list[float] = []
        torch.cuda.synchronize(local_rank)
        train_started = time.perf_counter()
        for step in range(args.iterations):
            generator = torch.Generator(device=f"cuda:{local_rank}")
            generator.manual_seed(args.seed + step * world_size + rank)
            inputs = torch.randn(args.batch_size, 256, generator=generator, device=f"cuda:{local_rank}")
            targets = torch.randn(args.batch_size, 64, generator=generator, device=f"cuda:{local_rank}")
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(ddp(inputs), targets)
            finite_loss = torch.isfinite(loss.detach()).to(dtype=torch.int32)
            dist.all_reduce(finite_loss, op=dist.ReduceOp.MIN)
            if not bool(finite_loss.item()):
                raise RuntimeError("at least one DDP rank produced a non-finite loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        torch.cuda.synchronize(local_rank)
        train_seconds = time.perf_counter() - train_started
        if not all(math.isfinite(value) for value in losses):
            raise RuntimeError("the recorded loss sequence contains a non-finite value")
        if not _all_tensors_finite(ddp.module.state_dict()):
            raise RuntimeError("the synchronized model state contains a non-finite tensor")
        if not _all_tensors_finite(optimizer.state_dict()):
            raise RuntimeError("the synchronized optimizer state contains a non-finite tensor")

        state_sha256 = _tensor_state_sha256(ddp.module.state_dict())
        gathered_state_hashes = _all_gather_object(state_sha256, world_size)
        if len(set(gathered_state_hashes)) != 1:
            raise RuntimeError(f"DDP model states diverged: {gathered_state_hashes}")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = args.output_dir / f"checkpoint-rank-{rank}.pt"
        torch.save(
            {
                "model": ddp.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iterations": args.iterations,
                "rank": rank,
                "run_id": args.run_id,
            },
            checkpoint,
        )

        reloaded = torch.nn.Sequential(
            torch.nn.Linear(256, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 64),
        ).cuda(local_rank)
        reloaded_optimizer = torch.optim.SGD(reloaded.parameters(), lr=0.01, momentum=0.9)
        payload = torch.load(checkpoint, map_location=f"cuda:{local_rank}", weights_only=False)
        reloaded.load_state_dict(payload["model"])
        reloaded_optimizer.load_state_dict(payload["optimizer"])
        reloaded_sha256 = _tensor_state_sha256(reloaded.state_dict())
        if reloaded_sha256 != state_sha256:
            raise RuntimeError("rank-local checkpoint reload changed the model state")

        dist.barrier(device_ids=[local_rank])
        return {
            "schema_version": 1,
            "status": "succeeded",
            "finished_at": _utc_now(),
            "run_id": args.run_id,
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "backend": dist.get_backend(),
            "script_sha256": args.script_sha256,
            "actual_script_sha256": actual_script_sha256,
            "image_digest": args.image_digest,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "nccl_version": list(torch.cuda.nccl.version()),
            "python_version": platform.python_version(),
            "identity": identity,
            "peers": peers,
            "init_seconds": init_seconds,
            "iterations": args.iterations,
            "batch_size_per_rank": args.batch_size,
            "train_seconds": train_seconds,
            "iterations_per_second": args.iterations / train_seconds,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "all_reduce_sum": reduced.item(),
            "expected_all_reduce_sum": expected_sum,
            "model_state_sha256": state_sha256,
            "all_model_state_sha256": gathered_state_hashes,
            "checkpoint_path": checkpoint.name,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": _file_sha256(checkpoint),
            "checkpoint_reload_state_sha256": reloaded_sha256,
            "environment": {
                key: os.environ.get(key)
                for key in (
                    "MASTER_ADDR",
                    "MASTER_PORT",
                    "NCCL_SOCKET_IFNAME",
                    "GLOO_SOCKET_IFNAME",
                    "NCCL_IB_DISABLE",
                    "NCCL_CUMEM_ENABLE",
                    "NCCL_CUMEM_HOST_ENABLE",
                )
            },
        }
    finally:
        with contextlib.suppress(Exception):
            dist.destroy_process_group()


def main() -> int:
    args = _parser().parse_args()
    _validate_args(args)
    rank = int(os.environ.get("RANK", "-1"))
    report: dict[str, Any]
    try:
        report = _run(args)
    except Exception as exc:
        report = {
            "schema_version": 1,
            "status": "failed",
            "finished_at": _utc_now(),
            "run_id": args.run_id,
            "rank": rank,
            "script_sha256": args.script_sha256,
            "image_digest": args.image_digest,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_report(args.output_dir, rank, report)
        raise
    _write_report(args.output_dir, rank, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
