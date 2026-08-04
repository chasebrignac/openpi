#!/usr/bin/env python3
"""Validate and stage converted PyTorch teacher checkpoints.

The released JAX checkpoint and the converted ``model.safetensors`` tree are
different immutable inputs.  This utility binds the converted bytes to the
exact GCS inventory revision and conversion source commit, then emits the
manifest shape consumed by ``repro_worker.py``.  S3 writes are dry-run unless
``upload --execute`` is used.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
import concurrent.futures
import dataclasses
import datetime as dt
import hashlib
import json
import math
import pathlib
import re
import sys
from typing import Any
import urllib.parse

try:
    from scripts import repro_stage_checkpoints
    from scripts import repro_stage_data
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root, to sys.path.
    import repro_stage_checkpoints
    import repro_stage_data


DEFAULT_CONFIG = pathlib.Path("repro/reproduction.json")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EQUIVALENCE_COSINE_MINIMUM = 0.999
GOLDEN_CONTRACTS = {
    "libero": {"config_name": "pi05_libero_l09_distill", "seed": 7001},
    "droid_jointpos": {"config_name": "pi05_droid_l09_distill", "seed": 7002},
}


@dataclasses.dataclass(frozen=True)
class ConvertedCheckpointSpec:
    key: str
    source_key: str
    config_name: str
    local_dirname: str
    asset_id: str

    @property
    def source(self) -> repro_stage_checkpoints.CheckpointSpec:
        return repro_stage_checkpoints.CHECKPOINTS[self.source_key]


CONVERTED_CHECKPOINTS = {
    "libero": ConvertedCheckpointSpec(
        key="libero",
        source_key="libero",
        config_name="pi05_libero",
        local_dirname="pi05_libero_pytorch",
        asset_id="physical-intelligence/libero",
    ),
    "droid_jointpos": ConvertedCheckpointSpec(
        key="droid_jointpos",
        source_key="droid_jointpos",
        config_name="pi05_droid_jointpos",
        local_dirname="pi05_droid_jointpos_pytorch",
        asset_id="droid",
    ),
}


CommandRunner = Callable[[Sequence[str]], str]


def converted_paths(
    local_root: pathlib.Path, spec: ConvertedCheckpointSpec
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    if not local_root.is_absolute():
        raise repro_stage_data.StageError(f"--local-root must be absolute: {local_root}")
    source_root, source_manifest = repro_stage_checkpoints.checkpoint_paths(local_root, spec.source)
    converted_root = local_root / spec.local_dirname
    manifest_path = local_root / "_manifests" / f"{spec.local_dirname}.converted-manifest.json"
    if converted_root == manifest_path or converted_root in manifest_path.parents:
        raise repro_stage_data.StageError("converted manifest must be outside the checkpoint payload")
    return source_root, source_manifest, manifest_path


def _converted_root(local_root: pathlib.Path, spec: ConvertedCheckpointSpec) -> pathlib.Path:
    return local_root / spec.local_dirname


def verify_source_checkout(source_commit: str, *, runner: CommandRunner = repro_stage_data.run_command) -> None:
    if COMMIT_RE.fullmatch(source_commit) is None:
        raise repro_stage_data.StageError("--source-commit must be a full lowercase git commit")
    actual = runner(["git", "rev-parse", "HEAD"]).strip()
    if actual != source_commit:
        raise repro_stage_data.StageError(
            f"conversion source commit mismatch: requested {source_commit}, checked out {actual!r}"
        )
    if runner(["git", "status", "--porcelain"]).strip():
        raise repro_stage_data.StageError("conversion source checkout is dirty; commit the reviewed tree first")


def _payload_files(root: pathlib.Path) -> list[pathlib.Path]:
    if not root.is_dir():
        raise repro_stage_data.StageError(f"converted checkpoint does not exist: {root}")
    files: list[pathlib.Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise repro_stage_data.StageError(f"converted checkpoint contains a symlink: {path}")
        if path.is_file():
            files.append(path)
        elif not path.is_dir():
            raise repro_stage_data.StageError(f"converted checkpoint contains a non-regular entry: {path}")
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _validate_conversion_output(root: pathlib.Path, spec: ConvertedCheckpointSpec) -> list[pathlib.Path]:
    files = _payload_files(root)
    required = (
        root / "model.safetensors",
        root / "config.json",
        root / "assets" / spec.asset_id / "norm_stats.json",
    )
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise repro_stage_data.StageError(f"required converted checkpoint file is missing or empty: {path}")
    try:
        conversion_config = json.loads((root / "config.json").read_text())
    except json.JSONDecodeError as exc:
        raise repro_stage_data.StageError(f"invalid converted config.json: {exc}") from exc
    if not isinstance(conversion_config, dict):
        raise repro_stage_data.StageError("converted config.json must be an object")
    if conversion_config.get("config_name") != spec.config_name:
        raise repro_stage_data.StageError(
            "converted checkpoint config mismatch: "
            f"expected {spec.config_name!r}, found {conversion_config.get('config_name')!r}"
        )
    if conversion_config.get("precision") != "bfloat16" or conversion_config.get("pi05") is not True:
        raise repro_stage_data.StageError("converted checkpoint must identify a bfloat16 pi0.5 model")
    return files


def manifest_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "upstream": manifest["source"]["upstream"],
        "conversion": manifest["conversion"],
        "totals": manifest["totals"],
        "files": manifest["files"],
    }


def conversion_revision(identity: Mapping[str, Any]) -> str:
    payload = json.dumps(identity, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_source_checkpoint(
    spec: ConvertedCheckpointSpec,
    source_root: pathlib.Path,
    source_manifest_path: pathlib.Path,
    *,
    hash_workers: int,
) -> dict[str, Any]:
    source_manifest = repro_stage_checkpoints.load_checkpoint_manifest(source_manifest_path, spec.source)
    rebuilt = repro_stage_checkpoints.build_checkpoint_manifest(
        spec.source,
        source_root,
        source_manifest["source"]["objects"],
        hash_workers=hash_workers,
    )
    if rebuilt["files"] != source_manifest.get("files") or rebuilt["totals"] != source_manifest.get("totals"):
        raise repro_stage_data.StageError("source JAX checkpoint no longer matches its SHA-256 manifest")
    return source_manifest


def build_converted_manifest(
    spec: ConvertedCheckpointSpec,
    source_root: pathlib.Path,
    source_manifest_path: pathlib.Path,
    converted_root: pathlib.Path,
    *,
    source_commit: str,
    image_digest: str,
    hash_workers: int,
) -> dict[str, Any]:
    if hash_workers < 1:
        raise repro_stage_data.StageError("--hash-workers must be at least 1")
    if IMAGE_DIGEST_RE.fullmatch(image_digest) is None:
        raise repro_stage_data.StageError("--image-digest must be a sha256-pinned container digest")
    source_manifest = _validate_source_checkpoint(
        spec,
        source_root,
        source_manifest_path,
        hash_workers=hash_workers,
    )
    paths = _validate_conversion_output(converted_root, spec)
    with concurrent.futures.ThreadPoolExecutor(max_workers=hash_workers) as executor:
        hashes = list(executor.map(repro_stage_data.sha256_file, paths))
    files = [
        {
            "path": path.relative_to(converted_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
        for path, digest in zip(paths, hashes, strict=True)
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "source": {
            "provider": "openpi-jax-to-pytorch",
            "revision_kind": "converted-checkpoint-content-and-provenance-sha256",
            "revision": "",
            "upstream": {
                "provider": "gcs",
                "uri": source_manifest["source"]["uri"],
                "revision": source_manifest["source"]["revision"],
            },
        },
        "conversion": {
            "source_commit": source_commit,
            "image_digest": image_digest,
            "converter": "examples/convert_jax_model_to_pytorch.py",
            "config_name": spec.config_name,
            "precision": "bfloat16",
        },
        "checkpoint": {
            "key": spec.key,
            "local_dirname": spec.local_dirname,
            "format": "pytorch-safetensors",
        },
        "totals": {"files": len(files), "bytes": sum(item["bytes"] for item in files)},
        "files": files,
    }
    manifest["source"]["revision"] = conversion_revision(manifest_identity(manifest))
    return manifest


def validate_saved_manifest(saved: Mapping[str, Any], rebuilt: Mapping[str, Any]) -> None:
    if saved.get("schema_version") != 1:
        raise repro_stage_data.StageError("converted checkpoint manifest schema mismatch")
    if saved.get("checkpoint") != rebuilt.get("checkpoint"):
        raise repro_stage_data.StageError("converted checkpoint manifest describes a different checkpoint")
    if manifest_identity(saved) != manifest_identity(rebuilt):
        raise repro_stage_data.StageError("converted checkpoint SHA-256 manifest verification failed")
    expected_revision = conversion_revision(manifest_identity(saved))
    if saved.get("source", {}).get("revision") != expected_revision:
        raise repro_stage_data.StageError("converted checkpoint revision is inconsistent with its manifest")


def validate_equivalence_report(
    report_path: pathlib.Path,
    spec: ConvertedCheckpointSpec,
    source_manifest_path: pathlib.Path,
    converted_manifest_path: pathlib.Path,
    converted_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind upload authorization to a passing report for these exact bytes."""
    report = repro_stage_data.load_json(report_path)
    cosine_min = report.get("cosine_min")
    if (
        report.get("schema_version") != 2
        or report.get("config_name") != spec.config_name
        or report.get("samples") != 64
        or report.get("gate_pass") is not True
        or not isinstance(cosine_min, int | float)
        or isinstance(cosine_min, bool)
        or not math.isfinite(float(cosine_min))
        or float(cosine_min) < EQUIVALENCE_COSINE_MINIMUM
    ):
        raise repro_stage_data.StageError("framework-equivalence report does not pass the exact 64-sample gate")

    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise repro_stage_data.StageError("framework-equivalence report has no provenance")
    golden = provenance.get("golden_corpus")
    expected_golden = GOLDEN_CONTRACTS[spec.key]
    golden_split = golden.get("data_split") if isinstance(golden, Mapping) else None
    validation_episode_ids = golden_split.get("validation_episode_ids") if isinstance(golden_split, Mapping) else None
    if (
        not isinstance(golden, Mapping)
        or golden.get("config_name") != expected_golden["config_name"]
        or golden.get("seed") != expected_golden["seed"]
        or golden.get("data_split_seed") != 42
        or SHA256_RE.fullmatch(str(golden.get("sha256", ""))) is None
        or SHA256_RE.fullmatch(str(golden.get("sidecar_sha256", ""))) is None
        or not isinstance(golden_split, Mapping)
        or golden_split.get("strategy") != "deterministic_whole_episode_stratified"
        or golden_split.get("split") != "validation"
        or golden_split.get("seed") != 42
        or not isinstance(validation_episode_ids, list)
        or not validation_episode_ids
    ):
        raise repro_stage_data.StageError("framework-equivalence report uses the wrong canonical golden corpus")

    jax_checkpoint = provenance.get("jax_checkpoint")
    pytorch_checkpoint = provenance.get("pytorch_checkpoint")
    jax_manifest = jax_checkpoint.get("manifest", {}) if isinstance(jax_checkpoint, Mapping) else {}
    pytorch_manifest = pytorch_checkpoint.get("manifest", {}) if isinstance(pytorch_checkpoint, Mapping) else {}
    upstream_revision = converted_manifest["source"]["upstream"]["revision"]
    converted_revision = converted_manifest["source"]["revision"]
    conversion = converted_manifest["conversion"]
    if (
        not isinstance(jax_manifest, Mapping)
        or jax_manifest.get("sha256") != repro_stage_data.sha256_file(source_manifest_path)
        or jax_manifest.get("revision") != upstream_revision
    ):
        raise repro_stage_data.StageError("framework-equivalence report does not bind the source JAX manifest")
    if (
        not isinstance(pytorch_manifest, Mapping)
        or pytorch_manifest.get("sha256") != repro_stage_data.sha256_file(converted_manifest_path)
        or pytorch_manifest.get("revision") != converted_revision
        or pytorch_manifest.get("source_commit") != conversion["source_commit"]
        or pytorch_manifest.get("image_digest") != conversion["image_digest"]
    ):
        raise repro_stage_data.StageError("framework-equivalence report does not bind the converted checkpoint")

    velocities = report.get("velocities")
    velocity_path = report_path.with_suffix(".npz").resolve()
    if (
        not isinstance(velocities, Mapping)
        or velocities.get("path") != str(velocity_path)
        or not velocity_path.is_file()
        or velocities.get("sha256") != repro_stage_data.sha256_file(velocity_path)
    ):
        raise repro_stage_data.StageError("framework-equivalence velocity evidence is missing or changed")
    return {
        "report": {"path": str(report_path.resolve()), "sha256": repro_stage_data.sha256_file(report_path)},
        "velocities": {"path": str(velocity_path), "sha256": velocities["sha256"]},
        "golden_corpus": {
            "sha256": golden["sha256"],
            "sidecar_sha256": golden["sidecar_sha256"],
            "data_split_seed": golden["data_split_seed"],
            "validation_episode_ids": validation_episode_ids,
            **expected_golden,
        },
        "cosine_min": float(cosine_min),
        "gate_pass": True,
    }


def converted_s3_target(s3_root: str, spec: ConvertedCheckpointSpec, revision: str) -> repro_stage_data.S3Target:
    parsed = urllib.parse.urlsplit(s3_root)
    if parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment:
        raise repro_stage_data.StageError(f"--s3-root must be an s3://bucket/prefix URI: {s3_root!r}")
    root = parsed.path.strip("/")
    prefix = "/".join(part for part in (root, spec.local_dirname, revision) if part)
    manifest_key = f"{prefix}/manifest.sha256.json"
    return repro_stage_data.S3Target(
        bucket=parsed.netloc,
        prefix=prefix,
        snapshot_uri=f"s3://{parsed.netloc}/{prefix}/checkpoint/",
        manifest_uri=f"s3://{parsed.netloc}/{manifest_key}",
        manifest_key=manifest_key,
    )


def upload_converted_checkpoint(
    config: Mapping[str, Any],
    spec: ConvertedCheckpointSpec,
    converted_root: pathlib.Path,
    manifest_path: pathlib.Path,
    manifest: Mapping[str, Any],
    s3_root: str,
    *,
    equivalence_report_sha256: str,
    runner: CommandRunner = repro_stage_data.run_command,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if SHA256_RE.fullmatch(equivalence_report_sha256) is None:
        raise repro_stage_data.StageError("equivalence report SHA-256 is invalid")
    revision = str(manifest["source"]["revision"])
    target = converted_s3_target(s3_root, spec, revision)
    account, region = repro_stage_data.verify_aws_destination(config, target, runner=runner, environ=environ)
    upstream = str(manifest["source"]["upstream"]["revision"])
    metadata = (
        f"source-provider=openpi-conversion,source-revision={revision},"
        f"upstream-revision={upstream},source-commit={manifest['conversion']['source_commit']},"
        f"image-digest={manifest['conversion']['image_digest']},"
        f"equivalence-report-sha256={equivalence_report_sha256}"
    )
    sync = [
        "aws",
        "s3",
        "sync",
        str(converted_root),
        target.snapshot_uri,
        "--region",
        region,
        "--no-follow-symlinks",
        "--only-show-errors",
        "--no-progress",
        "--sse",
        "AES256",
        "--metadata",
        metadata,
    ]
    runner(sync)
    runner(
        [
            "aws",
            "s3api",
            "put-object",
            "--bucket",
            target.bucket,
            "--key",
            target.manifest_key,
            "--body",
            str(manifest_path),
            "--expected-bucket-owner",
            account,
            "--region",
            region,
            "--server-side-encryption",
            "AES256",
            "--metadata",
            metadata,
        ]
    )
    verification_sync = [argument for argument in sync if argument != "--only-show-errors"]
    if runner([*verification_sync, "--dryrun"]).strip():
        raise repro_stage_data.StageError("S3 converted-checkpoint sync verification reports pending changes")
    head = json.loads(
        runner(
            [
                "aws",
                "s3api",
                "head-object",
                "--bucket",
                target.bucket,
                "--key",
                target.manifest_key,
                "--expected-bucket-owner",
                account,
                "--region",
                region,
                "--output",
                "json",
            ]
        )
    )
    if int(head.get("ContentLength", -1)) != manifest_path.stat().st_size:
        raise repro_stage_data.StageError("uploaded converted manifest has an unexpected content length")
    if head.get("Metadata", {}).get("source-revision") != revision:
        raise repro_stage_data.StageError("uploaded converted manifest has the wrong source revision")
    version_id = head.get("VersionId")
    if not version_id:
        raise repro_stage_data.StageError("uploaded converted manifest has no S3 version ID")
    return {
        "source_revision": revision,
        "checkpoint_uri": target.snapshot_uri,
        "manifest_uri": target.manifest_uri,
        "manifest_version_id": version_id,
        "worker_artifact": {
            "name": f"{spec.key}_teacher_pytorch",
            "kind": "checkpoint",
            "revision": revision,
            "manifest": {
                "s3_uri": target.manifest_uri,
                "version_id": version_id,
                "sha256": repro_stage_data.sha256_file(manifest_path),
            },
            "payload_s3_uri": target.snapshot_uri,
            "destination": spec.local_dirname,
        },
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="plan", choices=("plan", "validate", "upload"))
    parser.add_argument("--checkpoint", required=True, choices=tuple(CONVERTED_CHECKPOINTS))
    parser.add_argument("--local-root", required=True, type=pathlib.Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--s3-root")
    parser.add_argument("--equivalence-report", type=pathlib.Path)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--hash-workers", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        spec = CONVERTED_CHECKPOINTS[args.checkpoint]
        if COMMIT_RE.fullmatch(args.source_commit) is None:
            raise repro_stage_data.StageError("--source-commit must be a full lowercase git commit")
        if IMAGE_DIGEST_RE.fullmatch(args.image_digest) is None:
            raise repro_stage_data.StageError("--image-digest must be a sha256-pinned container digest")
        source_root, source_manifest_path, manifest_path = converted_paths(args.local_root, spec)
        converted_root = _converted_root(args.local_root, spec)
        if args.action == "upload" and not args.s3_root:
            raise repro_stage_data.StageError("--s3-root is required for upload")
        if args.action == "upload" and args.equivalence_report is None:
            raise repro_stage_data.StageError("--equivalence-report is required for upload")
        if args.action == "plan" or (args.action == "upload" and not args.execute):
            print(
                json.dumps(
                    {
                        "mode": "dry-run",
                        "requested_action": args.action,
                        "checkpoint": spec.key,
                        "source_checkpoint": str(source_root),
                        "source_manifest": str(source_manifest_path),
                        "source_commit": args.source_commit,
                        "image_digest": args.image_digest,
                        "converted_checkpoint": str(converted_root),
                        "converted_manifest": str(manifest_path),
                        "s3_root": args.s3_root,
                        "equivalence_report": str(args.equivalence_report) if args.equivalence_report else None,
                        "mutations_authorized": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        verify_source_checkout(args.source_commit)
        rebuilt = build_converted_manifest(
            spec,
            source_root,
            source_manifest_path,
            converted_root,
            source_commit=args.source_commit,
            image_digest=args.image_digest,
            hash_workers=args.hash_workers,
        )
        if args.action == "validate":
            if manifest_path.exists():
                manifest = repro_stage_data.load_json(manifest_path)
                validate_saved_manifest(manifest, rebuilt)
            else:
                repro_stage_data.write_manifest(manifest_path, rebuilt)
                manifest = rebuilt
        else:
            manifest = repro_stage_data.load_json(manifest_path)
            validate_saved_manifest(manifest, rebuilt)

        equivalence = None
        if args.action == "upload":
            equivalence = validate_equivalence_report(
                args.equivalence_report,
                spec,
                source_manifest_path,
                manifest_path,
                manifest,
            )

        result: dict[str, Any] = {
            "checkpoint": spec.key,
            "source_checkpoint_revision": manifest["source"]["upstream"]["revision"],
            "source_commit": manifest["conversion"]["source_commit"],
            "converted_revision": manifest["source"]["revision"],
            "manifest": str(manifest_path),
            "manifest_sha256": repro_stage_data.sha256_file(manifest_path),
            "totals": manifest["totals"],
        }
        if equivalence is not None:
            result["framework_equivalence"] = equivalence
        if args.action == "upload":
            config = repro_stage_data.load_json(args.config)
            result["s3"] = upload_converted_checkpoint(
                config,
                spec,
                converted_root,
                manifest_path,
                manifest,
                args.s3_root,
                equivalence_report_sha256=equivalence["report"]["sha256"],
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (repro_stage_data.StageError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"CONVERTED CHECKPOINT STAGING REJECTED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
