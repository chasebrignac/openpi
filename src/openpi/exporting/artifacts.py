"""Small, dependency-free artifact and stage-manifest helpers."""

from __future__ import annotations

from collections.abc import Iterable
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import shlex
import subprocess
from typing import Any

COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def sha256_file(path: pathlib.Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: pathlib.Path) -> dict[str, Any]:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"stage artifact must be one regular non-symlink file: {path}")
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def git_state() -> dict[str, Any]:
    return {
        "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
    }


def require_clean_source_identity(*, environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Bind a stage to the worker-protected source SHA before expensive work."""

    environment = os.environ if environ is None else environ
    expected = environment.get("PI05_SOURCE_SHA", "")
    if COMMIT_RE.fullmatch(expected) is None:
        raise RuntimeError("PI05_SOURCE_SHA must be the worker-owned full source commit")
    state = git_state()
    if state["sha"] != expected:
        raise RuntimeError("checked-out source commit differs from PI05_SOURCE_SHA")
    if state["dirty"]:
        raise RuntimeError("compiled-policy stages require a clean source checkout")
    return state


def _atomic_write_new(path: pathlib.Path, payload: bytes) -> None:
    """Publish a manifest once and durably; never replace prior evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"stage manifest already exists; use a fresh output directory: {path}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link gives create-if-absent semantics even if another process
        # races this writer. It also avoids replacing immutable evidence.
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_new(path: pathlib.Path, value: Any) -> None:
    """Write canonical JSON once without replacing existing evidence."""

    payload = (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
    _atomic_write_new(path, payload)


def require_absent_outputs(paths: Iterable[pathlib.Path], *, stage: str) -> None:
    """Reject a stage before it can overwrite any prior local artifact."""

    collisions = sorted(str(path) for path in paths if path.exists() or path.is_symlink())
    if collisions:
        raise FileExistsError(f"{stage} requires fresh output paths; existing={collisions}")


def prepare_fresh_output_directory(path: pathlib.Path, *, stage: str) -> None:
    """Create an output directory or require an existing one to be empty."""

    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise ValueError(f"{stage} output must be a non-symlink directory: {path}")
    if path.is_dir() and any(path.iterdir()):
        raise FileExistsError(f"{stage} output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def write_stage_manifest(
    output: pathlib.Path,
    *,
    stage: str,
    track: str,
    command: Iterable[str],
    image_digest: str,
    dataset: str,
    dataset_revision: str,
    instance_type: str,
    instance_id: str,
    cost_reservation: str,
    artifacts: Iterable[pathlib.Path],
    details: dict[str, Any],
    seed: int | None = None,
    steps: int | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    argv = list(command)
    artifact_paths = list(artifacts)
    resolved_paths = [path.resolve() for path in artifact_paths]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("stage manifest artifacts must not contain duplicate files")
    source_state = require_clean_source_identity()
    payload = {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "stage": stage,
        "track": track,
        "source": source_state,
        "runtime": {
            "image_digest": image_digest,
            "instance_type": instance_type,
            "instance_id": instance_id,
        },
        "dataset": {"name": dataset, "revision": dataset_revision},
        "experiment": {"seed": seed, "steps": steps},
        "cost": {"reservation_id": cost_reservation},
        "command": {"argv": argv, "shell": shlex.join(argv)},
        "metrics": metrics or {},
        "artifacts": [artifact_record(path) for path in artifact_paths],
        "details": details,
    }
    write_json_new(output, payload)
    return payload
