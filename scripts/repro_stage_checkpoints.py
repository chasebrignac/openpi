#!/usr/bin/env python3
"""Stage released pi0.5 teachers with a GCS generation inventory.

The public checkpoint names are mutable GCS prefixes.  This utility treats the
SHA-256 of the exact object-name/generation/checksum inventory as their source
revision.  Downloads and S3 uploads are dry-run unless ``--execute`` is given.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable, Iterable, Mapping, Sequence
import concurrent.futures
import dataclasses
import datetime as dt
import hashlib
import json
import pathlib
import shutil
import sys
from typing import Any
import urllib.parse
import urllib.request

try:
    from scripts import repro_stage_data
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root, to sys.path.
    import repro_stage_data


DEFAULT_CONFIG = pathlib.Path("repro/reproduction.json")
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024


@dataclasses.dataclass(frozen=True)
class CheckpointSpec:
    key: str
    source_uri: str
    bucket: str
    object_prefix: str
    local_dirname: str


CHECKPOINTS = {
    "libero": CheckpointSpec(
        key="libero",
        source_uri="gs://openpi-assets/checkpoints/pi05_libero",
        bucket="openpi-assets",
        object_prefix="checkpoints/pi05_libero/",
        local_dirname="pi05_libero",
    ),
    "droid": CheckpointSpec(
        key="droid",
        source_uri="gs://openpi-assets/checkpoints/pi05_droid",
        bucket="openpi-assets",
        object_prefix="checkpoints/pi05_droid/",
        local_dirname="pi05_droid",
    ),
    "droid_jointpos": CheckpointSpec(
        key="droid_jointpos",
        source_uri="gs://openpi-assets-simeval/pi05_droid_jointpos",
        bucket="openpi-assets-simeval",
        object_prefix="pi05_droid_jointpos/",
        local_dirname="pi05_droid_jointpos",
    ),
}


UrlOpen = Callable[..., Any]


def _read_json_response(opener: UrlOpen, url: str) -> dict[str, Any]:
    try:
        with opener(url) as response:
            value = json.load(response)
    except Exception as exc:
        raise repro_stage_data.StageError(f"failed to read GCS inventory: {url}: {exc}") from exc
    if not isinstance(value, dict):
        raise repro_stage_data.StageError("GCS inventory response must be a JSON object")
    return value


def list_gcs_inventory(spec: CheckpointSpec, *, opener: UrlOpen = urllib.request.urlopen) -> list[dict[str, Any]]:
    fields = "nextPageToken,items(name,size,md5Hash,crc32c,generation,updated)"
    page_token: str | None = None
    inventory: list[dict[str, Any]] = []
    while True:
        query = {
            "prefix": spec.object_prefix,
            "maxResults": "1000",
            "fields": fields,
        }
        if page_token is not None:
            query["pageToken"] = page_token
        url = f"https://storage.googleapis.com/storage/v1/b/{spec.bucket}/o?{urllib.parse.urlencode(query)}"
        page = _read_json_response(opener, url)
        for raw_item in page.get("items", []):
            try:
                item = {
                    "name": str(raw_item["name"]),
                    "generation": str(raw_item["generation"]),
                    "bytes": int(raw_item["size"]),
                    "md5_base64": raw_item.get("md5Hash"),
                    "crc32c_base64": raw_item.get("crc32c"),
                    "updated": str(raw_item["updated"]),
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise repro_stage_data.StageError(f"invalid GCS object inventory record: {raw_item}") from exc
            if not item["name"].startswith(spec.object_prefix) or item["bytes"] < 0:
                raise repro_stage_data.StageError(f"GCS returned an invalid object for {spec.source_uri}: {item}")
            relative = pathlib.PurePosixPath(item["name"].removeprefix(spec.object_prefix))
            if not relative.parts or ".." in relative.parts or relative.is_absolute():
                raise repro_stage_data.StageError(f"unsafe GCS object path: {item['name']}")
            inventory.append(item)
        page_token = page.get("nextPageToken")
        if not page_token:
            break
    if not inventory:
        raise repro_stage_data.StageError(f"no checkpoint objects found at {spec.source_uri}")
    names = [item["name"] for item in inventory]
    if len(names) != len(set(names)):
        raise repro_stage_data.StageError(f"duplicate object names in GCS inventory for {spec.source_uri}")
    return sorted(inventory, key=lambda item: item["name"])


def inventory_revision(inventory: Sequence[Mapping[str, Any]]) -> str:
    identity = sorted(
        [
            {key: item.get(key) for key in ("name", "generation", "bytes", "md5_base64", "crc32c_base64")}
            for item in inventory
        ],
        key=lambda item: str(item["name"]),
    )
    payload = json.dumps(identity, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def checkpoint_paths(local_root: pathlib.Path, spec: CheckpointSpec) -> tuple[pathlib.Path, pathlib.Path]:
    if not local_root.is_absolute():
        raise repro_stage_data.StageError(f"--local-root must be absolute: {local_root}")
    checkpoint_root = local_root / spec.local_dirname
    manifest_path = local_root / "_manifests" / f"{spec.local_dirname}.source-manifest.json"
    return checkpoint_root, manifest_path


def _object_url(spec: CheckpointSpec, item: Mapping[str, Any]) -> str:
    encoded_name = urllib.parse.quote(str(item["name"]), safe="")
    query = urllib.parse.urlencode({"alt": "media", "generation": item["generation"]})
    return f"https://storage.googleapis.com/download/storage/v1/b/{spec.bucket}/o/{encoded_name}?{query}"


def _md5_base64(path: pathlib.Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(DOWNLOAD_CHUNK_BYTES):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode()


def _validate_downloaded_object(path: pathlib.Path, item: Mapping[str, Any]) -> bool:
    if not path.is_file() or path.stat().st_size != int(item["bytes"]):
        return False
    expected_md5 = item.get("md5_base64")
    return expected_md5 is None or _md5_base64(path) == expected_md5


def download_gcs_object(
    spec: CheckpointSpec,
    checkpoint_root: pathlib.Path,
    item: Mapping[str, Any],
    *,
    opener: UrlOpen = urllib.request.urlopen,
) -> pathlib.Path:
    relative = pathlib.PurePosixPath(str(item["name"]).removeprefix(spec.object_prefix))
    destination = checkpoint_root.joinpath(*relative.parts)
    if _validate_downloaded_object(destination, item):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists() and partial.stat().st_size > int(item["bytes"]):
        raise repro_stage_data.StageError(f"partial checkpoint object is larger than its source: {partial}")
    if _validate_downloaded_object(partial, item):
        partial.replace(destination)
        return destination
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(_object_url(spec, item))
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    try:
        with opener(request) as response:
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode()
            append = offset > 0 and status == 206
            mode = "ab" if append else "wb"
            with partial.open(mode) as stream:
                shutil.copyfileobj(response, stream, length=DOWNLOAD_CHUNK_BYTES)
    except Exception as exc:
        raise repro_stage_data.StageError(
            f"failed to download {item['name']} at generation {item['generation']}: {exc}"
        ) from exc
    if not _validate_downloaded_object(partial, item):
        raise repro_stage_data.StageError(f"downloaded checkpoint object failed size/MD5 validation: {item['name']}")
    partial.replace(destination)
    return destination


def download_checkpoint(
    spec: CheckpointSpec,
    checkpoint_root: pathlib.Path,
    inventory: Sequence[Mapping[str, Any]],
    *,
    workers: int,
    opener: UrlOpen = urllib.request.urlopen,
) -> None:
    if workers < 1:
        raise repro_stage_data.StageError("--workers must be at least 1")
    total_bytes = sum(int(item["bytes"]) for item in inventory)
    existing = checkpoint_root.parent
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    if shutil.disk_usage(existing).free < int(total_bytes * 1.05):
        raise repro_stage_data.StageError(f"insufficient disk for {spec.key} teacher ({total_bytes} source bytes)")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(download_gcs_object, spec, checkpoint_root, item, opener=opener) for item in inventory
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def validate_checkpoint_files(
    spec: CheckpointSpec,
    checkpoint_root: pathlib.Path,
    inventory: Sequence[Mapping[str, Any]],
) -> list[pathlib.Path]:
    expected: dict[str, Mapping[str, Any]] = {
        str(item["name"]).removeprefix(spec.object_prefix): item for item in inventory
    }
    actual: dict[str, pathlib.Path] = {}
    for path in checkpoint_root.rglob("*"):
        if path.is_symlink():
            raise repro_stage_data.StageError(f"checkpoint contains a symlink: {path}")
        if path.is_file():
            actual[path.relative_to(checkpoint_root).as_posix()] = path
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise repro_stage_data.StageError(f"checkpoint object set mismatch; missing={missing}, extra={extra}")
    for relative, path in actual.items():
        if not _validate_downloaded_object(path, expected[relative]):
            raise repro_stage_data.StageError(f"checkpoint object failed size/MD5 validation: {relative}")
    return [actual[name] for name in sorted(actual)]


def build_checkpoint_manifest(
    spec: CheckpointSpec,
    checkpoint_root: pathlib.Path,
    inventory: Sequence[Mapping[str, Any]],
    *,
    hash_workers: int,
) -> dict[str, Any]:
    paths = validate_checkpoint_files(spec, checkpoint_root, inventory)
    if hash_workers < 1:
        raise repro_stage_data.StageError("--hash-workers must be at least 1")
    with concurrent.futures.ThreadPoolExecutor(max_workers=hash_workers) as executor:
        hashes = list(executor.map(repro_stage_data.sha256_file, paths))
    revision = inventory_revision(inventory)
    return {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "source": {
            "provider": "gcs",
            "uri": spec.source_uri,
            "revision_kind": "gcs-generation-inventory-sha256",
            "revision": revision,
            "objects": [dict(item) for item in inventory],
        },
        "checkpoint": {"key": spec.key, "local_dirname": spec.local_dirname},
        "totals": {"files": len(paths), "bytes": sum(path.stat().st_size for path in paths)},
        "files": [
            {
                "path": path.relative_to(checkpoint_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
            for path, digest in zip(paths, hashes, strict=True)
        ],
    }


def load_checkpoint_manifest(path: pathlib.Path, spec: CheckpointSpec) -> dict[str, Any]:
    manifest = repro_stage_data.load_json(path)
    if manifest.get("checkpoint", {}).get("key") != spec.key:
        raise repro_stage_data.StageError(f"checkpoint manifest does not describe {spec.key}: {path}")
    source = manifest.get("source", {})
    objects = source.get("objects")
    if not isinstance(objects, list) or not objects:
        raise repro_stage_data.StageError(f"checkpoint source inventory is missing: {path}")
    if source.get("uri") != spec.source_uri or source.get("revision") != inventory_revision(objects):
        raise repro_stage_data.StageError(f"checkpoint source inventory is inconsistent: {path}")
    return manifest


def checkpoint_s3_target(s3_root: str, spec: CheckpointSpec, revision: str) -> repro_stage_data.S3Target:
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


def upload_checkpoint(
    config: Mapping[str, Any],
    spec: CheckpointSpec,
    checkpoint_root: pathlib.Path,
    manifest_path: pathlib.Path,
    manifest: Mapping[str, Any],
    s3_root: str,
    *,
    runner: repro_stage_data.CommandRunner = repro_stage_data.run_command,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    revision = str(manifest["source"]["revision"])
    target = checkpoint_s3_target(s3_root, spec, revision)
    account, region = repro_stage_data.verify_aws_destination(config, target, runner=runner, environ=environ)
    metadata = f"source-provider=gcs,source-uri={spec.source_uri},source-revision={revision}"
    sync = [
        "aws",
        "s3",
        "sync",
        str(checkpoint_root),
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
        raise repro_stage_data.StageError("S3 checkpoint sync verification reports pending changes")
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
        raise repro_stage_data.StageError("uploaded checkpoint manifest has an unexpected content length")
    if head.get("Metadata", {}).get("source-revision") != revision:
        raise repro_stage_data.StageError("uploaded checkpoint manifest has the wrong source revision")
    if not head.get("VersionId"):
        raise repro_stage_data.StageError("uploaded checkpoint manifest has no S3 version ID")
    return {
        "source_revision": revision,
        "checkpoint_uri": target.snapshot_uri,
        "manifest_uri": target.manifest_uri,
        "manifest_version_id": head.get("VersionId"),
        "worker_artifact": {
            "name": f"{spec.key}_teacher_jax",
            "kind": "checkpoint",
            "revision": revision,
            "manifest": {
                "s3_uri": target.manifest_uri,
                "version_id": head.get("VersionId"),
                "sha256": repro_stage_data.sha256_file(manifest_path),
            },
            "payload_s3_uri": target.snapshot_uri,
            "destination": spec.local_dirname,
        },
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", nargs="?", default="plan", choices=("plan", "download", "validate", "upload", "stage")
    )
    parser.add_argument("--checkpoint", required=True, choices=tuple(CHECKPOINTS))
    parser.add_argument("--local-root", required=True, type=pathlib.Path)
    parser.add_argument("--s3-root")
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hash-workers", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        spec = CHECKPOINTS[args.checkpoint]
        config = repro_stage_data.load_json(args.config)
        checkpoint_root, manifest_path = checkpoint_paths(args.local_root, spec)
        if args.action in {"upload", "stage"} and not args.s3_root:
            raise repro_stage_data.StageError(f"--s3-root is required for {args.action}")
        if args.action == "plan" or (args.action in {"download", "upload", "stage"} and not args.execute):
            print(
                json.dumps(
                    {
                        "mode": "dry-run",
                        "requested_action": args.action,
                        "checkpoint": spec.key,
                        "source_uri": spec.source_uri,
                        "source_revision": "resolved from the exact GCS object-generation inventory on execute",
                        "local_root": str(checkpoint_root),
                        "manifest": str(manifest_path),
                        "s3_root": args.s3_root,
                        "mutations_authorized": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.action in {"download", "stage"}:
            inventory = list_gcs_inventory(spec)
            download_checkpoint(spec, checkpoint_root, inventory, workers=args.workers)
            if list_gcs_inventory(spec) != inventory:
                raise repro_stage_data.StageError(
                    "GCS checkpoint inventory changed during download; no manifest accepted"
                )
            manifest = build_checkpoint_manifest(spec, checkpoint_root, inventory, hash_workers=args.hash_workers)
            repro_stage_data.write_manifest(manifest_path, manifest)
        else:
            manifest = load_checkpoint_manifest(manifest_path, spec)
            rebuilt = build_checkpoint_manifest(
                spec,
                checkpoint_root,
                manifest["source"]["objects"],
                hash_workers=args.hash_workers,
            )
            if rebuilt["files"] != manifest.get("files") or rebuilt["totals"] != manifest.get("totals"):
                raise repro_stage_data.StageError("local checkpoint SHA-256 manifest verification failed")

        result: dict[str, Any] = {
            "checkpoint": spec.key,
            "source_uri": spec.source_uri,
            "source_revision": manifest["source"]["revision"],
            "manifest": str(manifest_path),
            "manifest_sha256": repro_stage_data.sha256_file(manifest_path),
            "totals": manifest["totals"],
        }
        if args.action in {"upload", "stage"}:
            result["s3"] = upload_checkpoint(
                config,
                spec,
                checkpoint_root,
                manifest_path,
                manifest,
                args.s3_root,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (repro_stage_data.StageError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"CHECKPOINT STAGING REJECTED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
