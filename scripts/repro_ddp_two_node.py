#!/usr/bin/env python3
"""Run and publish a fail-closed two-node OpenPI DDP validation.

This program is downloaded by exact S3 VersionId and runs on each of two
``g7e.2xlarge`` hosts.  A controller-issued, create-once GO document assigns
the ranks after both hosts have published their IMDS identities.  The program
then runs both a synthetic NCCL/DDP test and OpenPI's ten-step ``debug``
trainer in the same pinned container used by the reproduction.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import ipaddress
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time
import traceback
from typing import Any

UTC = getattr(dt, "UTC", dt.timezone.utc)  # noqa: UP017 -- Ubuntu host compatibility.
EXPECTED_ACCOUNT = "752160877725"
EXPECTED_REGION = "us-east-2"
EXPECTED_BUCKET = "pi05-repro-752160877725-us-east-2"
EXPECTED_INSTANCE_TYPE = "g7e.2xlarge"
EXPECTED_REPOSITORY = f"{EXPECTED_ACCOUNT}.dkr.ecr.{EXPECTED_REGION}.amazonaws.com/pi05-repro"
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,125}[a-z0-9]$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
INSTANCE_ID_RE = re.compile(r"^i-[0-9a-f]{17}$")
VERSION_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,1024}$")
RECOVERY_ATTEMPT_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")
CONTAINER_UID = 1000
CONTAINER_GID = 1000
CONTAINER_USER = "pi05"
CONTAINER_HOME = "/output/.home"
PASSWD_CONTENT = (
    f"{CONTAINER_USER}:x:{CONTAINER_UID}:{CONTAINER_GID}:pi05 distributed validation:"
    f"{CONTAINER_HOME}:/usr/sbin/nologin\n"
)
GROUP_CONTENT = f"{CONTAINER_USER}:x:{CONTAINER_GID}:\n"
SOURCE_UID_FAILURE = "KeyError: 'getpwuid(): uid not found: 1000'"
DEBUG_PALIGEMMA_VARIANT = "gemma_2b"
DEBUG_ACTION_EXPERT_VARIANT = "dummy"
DEBUG_PYTORCH_COMPILE_MODE = "None"


class ValidationError(RuntimeError):
    """Raised when a validation contract or runtime gate fails."""


def _utc_now() -> str:
    return dt.datetime.now(UTC).isoformat()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recovery-attempt",
        help="fresh same-node attempt that reuses the accepted synthetic stage and reruns only OpenPI",
    )
    return parser


def _validate_recovery_attempt(value: str | None) -> str | None:
    if value is None:
        return None
    if RECOVERY_ATTEMPT_RE.fullmatch(value) is None:
        raise ValidationError("recovery attempt must be a lowercase hyphenated slug")
    return value


def _recovery_port(attempt: str) -> int:
    """Return a deterministic fresh port outside the source run's 29400/29401 pair."""

    return 29500 + int(hashlib.sha256(attempt.encode()).hexdigest()[:8], 16) % 1000


def _recovery_prefix(run_id: str, attempt: str) -> str:
    _validate_recovery_attempt(attempt)
    return f"diagnostics/two-node-ddp/{run_id}/recoveries/{attempt}"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    arguments: list[str],
    *,
    input_bytes: bytes | None = None,
    timeout: int = 120,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        arguments,
        input=input_bytes,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace")[-2000:].strip()
        raise ValidationError(f"command failed ({completed.returncode}): {arguments[:3]}: {detail}")
    return completed


def _aws(arguments: list[str], *, timeout: int = 120) -> dict[str, Any]:
    completed = _run(
        ["aws", "--region", EXPECTED_REGION, "--no-cli-pager", *arguments, "--output", "json"],
        timeout=timeout,
    )
    try:
        value = json.loads(completed.stdout or b"{}")
    except json.JSONDecodeError as exc:
        raise ValidationError(f"AWS returned invalid JSON for {arguments[:2]}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"AWS returned a non-object for {arguments[:2]}")
    return value


def _required_environment(*, recovery_attempt: str | None = None) -> dict[str, str]:
    names = [
        "PI05_RUN_ID",
        "PI05_ORCHESTRATOR_SHA256",
        "PI05_SMOKE_KEY",
        "PI05_SMOKE_VERSION_ID",
        "PI05_SMOKE_SHA256",
        "PI05_IMAGE_URI",
        "PI05_IMAGE_DIGEST",
        "PI05_IMAGE_SOURCE_SHA",
    ]
    if recovery_attempt is not None:
        names.append("PI05_SOURCE_ORCHESTRATOR_SHA256")
    values = {name: os.environ.get(name, "") for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValidationError(f"missing required environment: {', '.join(missing)}")
    run_id = values["PI05_RUN_ID"]
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise ValidationError("PI05_RUN_ID has an invalid shape")
    for name in ("PI05_ORCHESTRATOR_SHA256", "PI05_SMOKE_SHA256"):
        if SHA256_RE.fullmatch(values[name]) is None:
            raise ValidationError(f"{name} must be a lowercase SHA-256")
    if recovery_attempt is not None and SHA256_RE.fullmatch(values["PI05_SOURCE_ORCHESTRATOR_SHA256"]) is None:
        raise ValidationError("PI05_SOURCE_ORCHESTRATOR_SHA256 must be a lowercase SHA-256")
    if GIT_SHA_RE.fullmatch(values["PI05_IMAGE_SOURCE_SHA"]) is None:
        raise ValidationError("PI05_IMAGE_SOURCE_SHA must be an exact 40-character Git commit")
    digest = values["PI05_IMAGE_DIGEST"]
    if not digest.startswith("sha256:") or SHA256_RE.fullmatch(digest[7:]) is None:
        raise ValidationError("PI05_IMAGE_DIGEST must be an exact sha256 digest")
    expected_uri = f"{EXPECTED_REPOSITORY}@{digest}"
    if values["PI05_IMAGE_URI"] != expected_uri:
        raise ValidationError("PI05_IMAGE_URI is not the exact approved ECR digest URI")
    if VERSION_ID_RE.fullmatch(values["PI05_SMOKE_VERSION_ID"]) is None:
        raise ValidationError("PI05_SMOKE_VERSION_ID has an invalid shape")
    expected_key = f"control/ddp-validation/{run_id}/repro_ddp_smoke.py"
    if values["PI05_SMOKE_KEY"] != expected_key:
        raise ValidationError("PI05_SMOKE_KEY is outside the run-scoped control prefix")
    actual_sha = _sha256(pathlib.Path(__file__).resolve())
    if actual_sha != values["PI05_ORCHESTRATOR_SHA256"]:
        raise ValidationError("executing orchestrator does not match its exact SHA-256 binding")
    return values


def _metadata(path: str, token: str) -> str:
    completed = _run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--header",
            f"X-aws-ec2-metadata-token: {token}",
            f"http://169.254.169.254/latest/meta-data/{path}",
        ]
    )
    return completed.stdout.decode().strip()


def _instance_identity() -> dict[str, str]:
    token = _run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--request",
            "PUT",
            "--header",
            "X-aws-ec2-metadata-token-ttl-seconds: 21600",
            "http://169.254.169.254/latest/api/token",
        ]
    ).stdout.decode()
    identity = {
        "instance_id": _metadata("instance-id", token),
        "instance_type": _metadata("instance-type", token),
        "private_ip": _metadata("local-ipv4", token),
        "availability_zone": _metadata("placement/availability-zone", token),
    }
    if INSTANCE_ID_RE.fullmatch(identity["instance_id"]) is None:
        raise ValidationError("IMDS returned an invalid instance ID")
    if identity["instance_type"] != EXPECTED_INSTANCE_TYPE:
        raise ValidationError(f"expected {EXPECTED_INSTANCE_TYPE}, got {identity['instance_type']}")
    try:
        address = ipaddress.ip_address(identity["private_ip"])
    except ValueError as exc:
        raise ValidationError("IMDS returned an invalid private IP") from exc
    if address.version != 4 or not address.is_private:
        raise ValidationError("IMDS did not return a private IPv4 address")
    if identity["availability_zone"] != "us-east-2a":
        raise ValidationError("two-node diagnostic must remain in the reviewed us-east-2a subnet")
    return identity


def _load_launch_metadata() -> dict[str, Any]:
    path = pathlib.Path("/opt/pi05/launch-metadata.json")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("launch metadata is missing or invalid") from exc
    expected = {
        "category": "corrective_run",
        "workload": "distributed_validation",
        "instance_type": EXPECTED_INSTANCE_TYPE,
        "instance_count": 2,
        "purchase_option": "On-Demand",
        "retain_after_command": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValidationError(f"launch metadata {key} mismatch")
    command_sha = value.get("command_sha256")
    reservation_id = value.get("reservation_id")
    if not isinstance(command_sha, str) or SHA256_RE.fullmatch(command_sha) is None:
        raise ValidationError("launch metadata command SHA-256 is invalid")
    if not isinstance(reservation_id, str) or not reservation_id:
        raise ValidationError("launch metadata reservation ID is invalid")
    try:
        deadline = dt.datetime.fromisoformat(str(value["deadline_utc"]))
    except (KeyError, ValueError) as exc:
        raise ValidationError("launch metadata deadline is invalid") from exc
    if deadline.tzinfo is None or deadline.astimezone(UTC) <= dt.datetime.now(UTC) + dt.timedelta(minutes=30):
        raise ValidationError("fewer than 30 minutes remain before the independent deadline")
    return value


def _write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _put_create_once(key: str, path: pathlib.Path, sha256: str) -> dict[str, Any]:
    if _sha256(path) != sha256:
        raise ValidationError(f"refusing to publish mutated local evidence: {path}")
    result = _aws(
        [
            "s3api",
            "put-object",
            "--bucket",
            EXPECTED_BUCKET,
            "--key",
            key,
            "--body",
            str(path),
            "--metadata",
            f"sha256={sha256}",
            "--server-side-encryption",
            "AES256",
            "--if-none-match",
            "*",
            "--expected-bucket-owner",
            EXPECTED_ACCOUNT,
        ]
    )
    version_id = result.get("VersionId")
    if not isinstance(version_id, str) or not version_id or version_id == "null":
        raise ValidationError("create-once S3 publication returned no VersionId")
    return {"key": key, "version_id": version_id, "sha256": sha256, "etag": result.get("ETag")}


def _download_exact(key: str, version_id: str, destination: pathlib.Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # ``get-object`` requires its positional output path after global CLI
    # options, so it cannot use the generic JSON wrapper above.
    completed = _run(
        [
            "aws",
            "--region",
            EXPECTED_REGION,
            "--no-cli-pager",
            "s3api",
            "get-object",
            "--bucket",
            EXPECTED_BUCKET,
            "--key",
            key,
            "--version-id",
            version_id,
            "--expected-bucket-owner",
            EXPECTED_ACCOUNT,
            "--output",
            "json",
            str(destination),
        ],
    )
    try:
        result = json.loads(completed.stdout or b"{}")
    except json.JSONDecodeError as exc:
        raise ValidationError(f"AWS returned invalid JSON while downloading {key}") from exc
    if result.get("VersionId") != version_id or _sha256(destination) != expected_sha256:
        raise ValidationError(f"exact S3 round trip failed for {key}")


def _wait_for_singleton(key: str, destination: pathlib.Path, timeout_seconds: int) -> tuple[dict[str, Any], str]:
    deadline = time.monotonic() + timeout_seconds
    head: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        completed = _run(
            [
                "aws",
                "--region",
                EXPECTED_REGION,
                "--no-cli-pager",
                "s3api",
                "head-object",
                "--bucket",
                EXPECTED_BUCKET,
                "--key",
                key,
                "--expected-bucket-owner",
                EXPECTED_ACCOUNT,
                "--output",
                "json",
            ],
            check=False,
        )
        if completed.returncode == 0:
            value = json.loads(completed.stdout or b"{}")
            if isinstance(value, dict):
                head = value
                break
        time.sleep(10)
    if head is None:
        raise ValidationError(f"timed out waiting for controller marker {key}")
    version_id = head.get("VersionId")
    metadata = head.get("Metadata", {})
    expected_sha = metadata.get("sha256") if isinstance(metadata, dict) else None
    if (
        not isinstance(version_id, str)
        or not isinstance(expected_sha, str)
        or SHA256_RE.fullmatch(expected_sha) is None
    ):
        raise ValidationError("controller marker lacks exact VersionId/SHA-256 identity")
    history = _aws(
        [
            "s3api",
            "list-object-versions",
            "--bucket",
            EXPECTED_BUCKET,
            "--prefix",
            key,
            "--expected-bucket-owner",
            EXPECTED_ACCOUNT,
        ]
    )
    versions = [item for item in history.get("Versions", []) if item.get("Key") == key]
    deletes = [item for item in history.get("DeleteMarkers", []) if item.get("Key") == key]
    if len(versions) != 1 or deletes or versions[0].get("VersionId") != version_id:
        raise ValidationError("controller marker is not singleton/no-delete")
    _download_exact(key, version_id, destination, expected_sha)
    value = json.loads(destination.read_text())
    if not isinstance(value, dict):
        raise ValidationError("controller marker body is not an object")
    return value, version_id


def _validate_go(
    go: dict[str, Any],
    values: dict[str, str],
    identity: dict[str, str],
    launch: dict[str, Any],
    *,
    expected_orchestrator_sha256: str | None = None,
) -> tuple[int, str, str]:
    expected = {
        "schema_version": 1,
        "run_id": values["PI05_RUN_ID"],
        "account_id": EXPECTED_ACCOUNT,
        "region": EXPECTED_REGION,
        "instance_type": EXPECTED_INSTANCE_TYPE,
        "world_size": 2,
        "image_uri": values["PI05_IMAGE_URI"],
        "image_digest": values["PI05_IMAGE_DIGEST"],
        "image_source_sha": values["PI05_IMAGE_SOURCE_SHA"],
        "orchestrator_sha256": expected_orchestrator_sha256 or values["PI05_ORCHESTRATOR_SHA256"],
        "smoke_key": values["PI05_SMOKE_KEY"],
        "smoke_version_id": values["PI05_SMOKE_VERSION_ID"],
        "smoke_sha256": values["PI05_SMOKE_SHA256"],
        "command_sha256": launch["command_sha256"],
        "reservation_id": launch["reservation_id"],
    }
    for key, expected_value in expected.items():
        if go.get(key) != expected_value:
            raise ValidationError(f"controller GO {key} mismatch")
    nodes = go.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 2:
        raise ValidationError("controller GO must contain exactly two nodes")
    ranks = {node.get("rank") for node in nodes if isinstance(node, dict)}
    instance_ids = {node.get("instance_id") for node in nodes if isinstance(node, dict)}
    private_ips = {node.get("private_ip") for node in nodes if isinstance(node, dict)}
    if ranks != {0, 1} or len(instance_ids) != 2 or len(private_ips) != 2:
        raise ValidationError("controller GO node identities are not unique ranks/hosts")
    matches = [node for node in nodes if node.get("instance_id") == identity["instance_id"]]
    if len(matches) != 1 or matches[0].get("private_ip") != identity["private_ip"]:
        raise ValidationError("controller GO does not bind this host's exact IMDS identity")
    master = [node for node in nodes if node.get("rank") == 0]
    if len(master) != 1:
        raise ValidationError("controller GO has no unique rank zero")
    return int(matches[0]["rank"]), str(master[0]["private_ip"]), str(nodes[1 - int(matches[0]["rank"])]["private_ip"])


def _network_interface(peer_ip: str) -> str:
    result = _run(["ip", "-json", "route", "get", peer_ip])
    try:
        routes = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValidationError("ip route returned invalid JSON") from exc
    if not isinstance(routes, list) or len(routes) != 1 or not isinstance(routes[0].get("dev"), str):
        raise ValidationError("could not resolve the private interface for the peer")
    interface = routes[0]["dev"]
    if re.fullmatch(r"[A-Za-z0-9_.-]{1,32}", interface) is None or interface == "lo":
        raise ValidationError("resolved an unsafe peer interface")
    return interface


def _prepare_image(values: dict[str, str]) -> dict[str, Any]:
    _run(["systemctl", "start", "docker"], timeout=180)
    password = _run(["aws", "--region", EXPECTED_REGION, "ecr", "get-login-password"], timeout=120).stdout
    _run(
        ["docker", "login", "--username", "AWS", "--password-stdin", EXPECTED_REPOSITORY.split("/", 1)[0]],
        input_bytes=password,
        timeout=120,
    )
    _run(["docker", "pull", values["PI05_IMAGE_URI"]], timeout=1200)
    inspection = json.loads(_run(["docker", "image", "inspect", values["PI05_IMAGE_URI"]]).stdout)
    if not isinstance(inspection, list) or len(inspection) != 1:
        raise ValidationError("docker did not return exactly one pinned image")
    image = inspection[0]
    if values["PI05_IMAGE_URI"] not in image.get("RepoDigests", []):
        raise ValidationError("local Docker image lacks the exact approved RepoDigest")
    labels = image.get("Config", {}).get("Labels", {})
    if labels.get("org.opencontainers.image.revision") != values["PI05_IMAGE_SOURCE_SHA"]:
        raise ValidationError("image source-revision label mismatch")
    if labels.get("ai.openpi.image-purpose") != "policy":
        raise ValidationError("image purpose label is not policy")
    return {
        "image_id": image.get("Id"),
        "repo_digests": image.get("RepoDigests"),
        "source_revision": labels.get("org.opencontainers.image.revision"),
        "purpose": labels.get("ai.openpi.image-purpose"),
    }


def _prepare_nonroot_identity(root: pathlib.Path) -> dict[str, Any]:
    identity_root = root / "identity"
    identity_root.mkdir(parents=True, exist_ok=False)
    passwd_path = identity_root / "passwd"
    group_path = identity_root / "group"
    with passwd_path.open("x") as stream:
        stream.write(PASSWD_CONTENT)
    with group_path.open("x") as stream:
        stream.write(GROUP_CONTENT)
    passwd_path.chmod(0o444)
    group_path.chmod(0o444)
    return {
        "passwd_path": passwd_path,
        "passwd_sha256": _sha256(passwd_path),
        "group_path": group_path,
        "group_sha256": _sha256(group_path),
        "uid": CONTAINER_UID,
        "gid": CONTAINER_GID,
        "user": CONTAINER_USER,
    }


def _validate_nonroot_identity(identity: dict[str, Any]) -> None:
    expected = (("passwd_path", PASSWD_CONTENT), ("group_path", GROUP_CONTENT))
    for key, content in expected:
        path = identity.get(key)
        if not isinstance(path, pathlib.Path) or path.is_symlink() or not path.is_file():
            raise ValidationError(f"non-root identity {key} is not a regular file")
        if path.read_text() != content or path.stat().st_mode & 0o777 != 0o444:
            raise ValidationError(f"non-root identity {key} differs from the reviewed read-only contract")
        if identity.get(key.replace("_path", "_sha256")) != _sha256(path):
            raise ValidationError(f"non-root identity {key} hash changed")


def _container_identity_arguments(identity: dict[str, Any]) -> list[str]:
    _validate_nonroot_identity(identity)
    return [
        "--user",
        f"{CONTAINER_UID}:{CONTAINER_GID}",
        "--env",
        f"HOME={CONTAINER_HOME}",
        "--env",
        f"USER={CONTAINER_USER}",
        "--env",
        f"LOGNAME={CONTAINER_USER}",
        "--env",
        "XDG_CACHE_HOME=/output/.cache",
        "--env",
        "TORCHINDUCTOR_CACHE_DIR=/output/.torchinductor-cache",
        "--mount",
        f"type=bind,src={identity['passwd_path']},dst=/etc/passwd,readonly",
        "--mount",
        f"type=bind,src={identity['group_path']},dst=/etc/group,readonly",
    ]


def _docker_base(
    values: dict[str, str],
    *,
    name: str,
    rank: int,
    master_ip: str,
    interface: str,
    output_dir: pathlib.Path,
    smoke_path: pathlib.Path,
    port: int,
    nonroot_identity: dict[str, Any],
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--gpus",
        "all",
        "--network",
        "host",
        "--ipc",
        "host",
        "--shm-size",
        "8g",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        *_container_identity_arguments(nonroot_identity),
        "--workdir",
        "/opt/openpi",
        "--env",
        f"MASTER_ADDR={master_ip}",
        "--env",
        f"MASTER_PORT={port}",
        "--env",
        f"NCCL_SOCKET_IFNAME={interface}",
        "--env",
        f"GLOO_SOCKET_IFNAME={interface}",
        "--env",
        "NCCL_IB_DISABLE=1",
        "--env",
        "NCCL_CUMEM_ENABLE=0",
        "--env",
        "NCCL_CUMEM_HOST_ENABLE=0",
        "--env",
        "NCCL_DEBUG=INFO",
        "--env",
        "TORCH_DISTRIBUTED_DEBUG=INFO",
        "--mount",
        f"type=bind,src={output_dir},dst=/output",
        "--mount",
        f"type=bind,src={smoke_path},dst=/opt/pi05/repro_ddp_smoke.py,readonly",
        values["PI05_IMAGE_URI"],
        "torchrun",
        "--nnodes=2",
        "--nproc-per-node=1",
        f"--node-rank={rank}",
        f"--master-addr={master_ip}",
        f"--master-port={port}",
    ]


def _run_container(arguments: list[str], name: str, log_path: pathlib.Path, timeout_seconds: int) -> float:
    started = time.monotonic()
    with log_path.open("wb") as stream:
        process = subprocess.Popen(arguments, stdout=stream, stderr=subprocess.STDOUT)
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _run(["docker", "rm", "--force", name], check=False, timeout=120)
            process.wait(timeout=60)
            raise ValidationError(f"container {name} exceeded {timeout_seconds} seconds") from exc
    if returncode != 0:
        raise ValidationError(f"container {name} failed with exit code {returncode}; see {log_path}")
    return time.monotonic() - started


def _identity_preflight_command(
    values: dict[str, str],
    *,
    name: str,
    output_dir: pathlib.Path,
    nonroot_identity: dict[str, Any],
) -> list[str]:
    script = """
set -eu
test "$(id -u)" = 1000
test "$(id -g)" = 1000
test "$(getent passwd 1000)" = 'pi05:x:1000:1000:pi05 distributed validation:/output/.home:/usr/sbin/nologin'
test "$(getent group 1000)" = 'pi05:x:1000:'
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR"
test -w "$HOME"
test -w "$XDG_CACHE_HOME"
test -w "$TORCHINDUCTOR_CACHE_DIR"
python3 -c 'import os, pwd; record = pwd.getpwuid(os.getuid()); assert record.pw_name == "pi05"; assert record.pw_uid == 1000; assert record.pw_gid == 1000'
printf 'nonroot_identity_preflight=PASS\\n'
""".strip()
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "none",
        *_container_identity_arguments(nonroot_identity),
        "--mount",
        f"type=bind,src={output_dir},dst=/output",
        values["PI05_IMAGE_URI"],
        "/bin/sh",
        "-c",
        script,
    ]


def _run_identity_preflight(
    values: dict[str, str],
    *,
    name: str,
    output_dir: pathlib.Path,
    nonroot_identity: dict[str, Any],
    log_path: pathlib.Path,
) -> dict[str, Any]:
    seconds = _run_container(
        _identity_preflight_command(
            values,
            name=name,
            output_dir=output_dir,
            nonroot_identity=nonroot_identity,
        ),
        name,
        log_path,
        timeout_seconds=120,
    )
    if log_path.read_text() != "nonroot_identity_preflight=PASS\n":
        raise ValidationError("non-root identity preflight did not emit the exact success marker")
    return {
        "seconds": seconds,
        "log_bytes": log_path.stat().st_size,
        "log_sha256": _sha256(log_path),
        "passwd_sha256": nonroot_identity["passwd_sha256"],
        "group_sha256": nonroot_identity["group_sha256"],
        "uid": CONTAINER_UID,
        "gid": CONTAINER_GID,
        "user": CONTAINER_USER,
    }


def _inventory(root: pathlib.Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValidationError(f"output contains a symlink: {path}")
        if path.is_file():
            result.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    if not result or len(result) > 256:
        raise ValidationError("output inventory is empty or unexpectedly large")
    return result


def _load_json_object(path: pathlib.Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValidationError(f"required JSON evidence is not a regular file: {path}")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValidationError(f"required JSON evidence is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"required JSON evidence is not an object: {path}")
    return value


def _validate_source_synthetic(
    source_root: pathlib.Path,
    values: dict[str, str],
    identity: dict[str, str],
    go: dict[str, Any],
    rank: int,
) -> dict[str, Any]:
    synthetic_root = source_root / "output" / "synthetic"
    report_path = synthetic_root / f"rank-{rank}.json"
    checkpoint_path = synthetic_root / f"checkpoint-rank-{rank}.pt"
    synthetic_log = source_root / "synthetic.log"
    failed_log = source_root / "openpi-debug.log"
    report = _load_json_object(report_path)
    expected = {
        "schema_version": 1,
        "status": "succeeded",
        "run_id": values["PI05_RUN_ID"],
        "rank": rank,
        "world_size": 2,
        "backend": "nccl",
        "script_sha256": values["PI05_SMOKE_SHA256"],
        "actual_script_sha256": values["PI05_SMOKE_SHA256"],
        "image_digest": values["PI05_IMAGE_DIGEST"],
        "iterations": 20,
        "all_reduce_sum": 3.0,
        "expected_all_reduce_sum": 3.0,
    }
    for key, expected_value in expected.items():
        if report.get(key) != expected_value:
            raise ValidationError(f"source synthetic report {key} mismatch")
    local_identity = report.get("identity")
    peers = report.get("peers")
    if not isinstance(local_identity, dict) or local_identity.get("private_ip") != identity["private_ip"]:
        raise ValidationError("source synthetic report has the wrong local identity")
    expected_peer_ips = {node.get("private_ip") for node in go.get("nodes", []) if isinstance(node, dict)}
    if (
        not isinstance(peers, list)
        or len(peers) != 2
        or {peer.get("private_ip") for peer in peers if isinstance(peer, dict)} != expected_peer_ips
    ):
        raise ValidationError("source synthetic report does not bind both GO peers")
    state_sha = report.get("model_state_sha256")
    state_hashes = report.get("all_model_state_sha256")
    if (
        not isinstance(state_sha, str)
        or SHA256_RE.fullmatch(state_sha) is None
        or state_hashes != [state_sha, state_sha]
        or report.get("checkpoint_reload_state_sha256") != state_sha
    ):
        raise ValidationError("source synthetic DDP/checkpoint state hashes are inconsistent")
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise ValidationError("source synthetic checkpoint is missing or unsafe")
    if report.get("checkpoint_bytes") != checkpoint_path.stat().st_size:
        raise ValidationError("source synthetic checkpoint size mismatch")
    if report.get("checkpoint_sha256") != _sha256(checkpoint_path):
        raise ValidationError("source synthetic checkpoint SHA-256 mismatch")
    for key in ("init_seconds", "train_seconds", "iterations_per_second", "loss_first", "loss_last"):
        value = report.get(key)
        if not isinstance(value, int | float) or not math.isfinite(value):
            raise ValidationError(f"source synthetic report {key} is not finite")
    for path in (synthetic_log, failed_log):
        if path.is_symlink() or not path.is_file():
            raise ValidationError(f"source run log is missing or unsafe: {path}")
    if SOURCE_UID_FAILURE not in failed_log.read_text(errors="replace"):
        raise ValidationError("source failure is not the reviewed missing UID 1000 identity failure")
    return {
        "report_path": str(report_path.relative_to(source_root)),
        "report_bytes": report_path.stat().st_size,
        "report_sha256": _sha256(report_path),
        "checkpoint_path": str(checkpoint_path.relative_to(source_root)),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "synthetic_log_sha256": _sha256(synthetic_log),
        "failed_openpi_log_sha256": _sha256(failed_log),
        "model_state_sha256": state_sha,
        "synthetic_reused_not_rerun": True,
    }


def _wait_for_results(prefix: str, work_root: pathlib.Path, timeout_seconds: int) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        listing = _aws(
            [
                "s3api",
                "list-objects-v2",
                "--bucket",
                EXPECTED_BUCKET,
                "--prefix",
                f"{prefix}/results/",
                "--expected-bucket-owner",
                EXPECTED_ACCOUNT,
            ]
        )
        keys = sorted(
            item.get("Key")
            for item in listing.get("Contents", [])
            if isinstance(item, dict) and isinstance(item.get("Key"), str) and item["Key"].endswith(".json")
        )
        if len(keys) == 2:
            reports: list[dict[str, Any]] = []
            for index, key in enumerate(keys):
                local = work_root / f"peer-result-{index}.json"
                head = _aws(
                    [
                        "s3api",
                        "head-object",
                        "--bucket",
                        EXPECTED_BUCKET,
                        "--key",
                        key,
                        "--expected-bucket-owner",
                        EXPECTED_ACCOUNT,
                    ]
                )
                sha = head.get("Metadata", {}).get("sha256")
                version = head.get("VersionId")
                if not isinstance(sha, str) or not isinstance(version, str):
                    raise ValidationError("peer result lacks exact S3 identity")
                _download_exact(key, version, local, sha)
                value = json.loads(local.read_text())
                if not isinstance(value, dict):
                    raise ValidationError("peer result is not an object")
                value["s3_version_id"] = version
                reports.append(value)
            return reports
        if len(keys) > 2:
            raise ValidationError("result prefix contains more than two reports")
        time.sleep(10)
    raise ValidationError("timed out waiting for both host result reports")


def _run_openpi_debug(
    values: dict[str, str],
    *,
    rank: int,
    master_ip: str,
    interface: str,
    actual_root: pathlib.Path,
    smoke_path: pathlib.Path,
    port: int,
    nonroot_identity: dict[str, Any],
    experiment: str,
    container_label: str,
    log_path: pathlib.Path,
) -> float:
    actual_root.mkdir(mode=0o777, parents=True, exist_ok=True)
    os.chmod(actual_root, 0o777)
    if rank != 0:
        rank_local_dir = actual_root / "debug" / experiment
        rank_local_dir.mkdir(parents=True, exist_ok=False)
        rank_local_dir.chmod(0o777)
        rank_local_dir.parent.chmod(0o777)
    name = f"pi05-ddp-openpi-{container_label}-rank-{rank}"
    command = [
        *_docker_base(
            values,
            name=name,
            rank=rank,
            master_ip=master_ip,
            interface=interface,
            output_dir=actual_root,
            smoke_path=smoke_path,
            port=port,
            nonroot_identity=nonroot_identity,
        ),
        "scripts/train_pytorch.py",
        "debug",
        "--exp-name",
        experiment,
        "--checkpoint-base-dir",
        "/output",
        "--num-train-steps",
        "10",
        "--save-interval",
        "10",
        "--log-interval",
        "1",
        "--model.paligemma-variant",
        DEBUG_PALIGEMMA_VARIANT,
        "--model.action-expert-variant",
        DEBUG_ACTION_EXPERT_VARIANT,
        "--model.pytorch-compile-mode",
        DEBUG_PYTORCH_COMPILE_MODE,
    ]
    return _run_container(command, name, log_path, timeout_seconds=1200)


def _publish_two_host_success(
    *,
    prefix: str,
    work_root: pathlib.Path,
    report: dict[str, Any],
    rank: int,
    aggregate_claims: dict[str, Any],
) -> dict[str, Any]:
    report_path = work_root / "result.json"
    _write_json(report_path, report)
    result_receipt = _put_create_once(
        f"{prefix}/results/{report['instance_id']}.json", report_path, _sha256(report_path)
    )
    if rank == 0:
        peers = _wait_for_results(prefix, work_root, timeout_seconds=600)
        if {item.get("rank") for item in peers} != {0, 1} or any(item.get("status") != "succeeded" for item in peers):
            raise ValidationError("two host reports do not form a successful rank-zero/rank-one pair")
        singleton_fields = ("command_sha256", "mode", "recovery_attempt")
        if any(len({item.get(field) for item in peers}) != 1 for field in singleton_fields):
            raise ValidationError("host reports disagree on the recovery identity")
        aggregate = {
            "schema_version": 1,
            "status": "succeeded",
            "finished_at": _utc_now(),
            "run_id": report["run_id"],
            "reservation_id": report["reservation_id"],
            "command_sha256": report["command_sha256"],
            "mode": report["mode"],
            "recovery_attempt": report.get("recovery_attempt"),
            **aggregate_claims,
            "hosts": peers,
        }
        aggregate_path = work_root / "aggregate.json"
        _write_json(aggregate_path, aggregate)
        aggregate_receipt = _put_create_once(f"{prefix}/aggregate.json", aggregate_path, _sha256(aggregate_path))
        marker = {
            "schema_version": 1,
            "status": "succeeded",
            "run_id": report["run_id"],
            "mode": report["mode"],
            "recovery_attempt": report.get("recovery_attempt"),
            "aggregate": aggregate_receipt,
        }
        marker_path = work_root / "expected-output.json"
        _write_json(marker_path, marker)
        _put_create_once(f"{prefix}/expected-output.json", marker_path, _sha256(marker_path))
    else:
        marker_path = work_root / "expected-output.json"
        marker, _ = _wait_for_singleton(f"{prefix}/expected-output.json", marker_path, timeout_seconds=600)
        if (
            marker.get("status") != "succeeded"
            or marker.get("run_id") != report["run_id"]
            or marker.get("mode") != report["mode"]
            or marker.get("recovery_attempt") != report.get("recovery_attempt")
        ):
            raise ValidationError("rank zero published an invalid expected-output marker")
    return result_receipt


def _run_validation(values: dict[str, str]) -> dict[str, Any]:
    run_id = values["PI05_RUN_ID"]
    prefix = f"diagnostics/two-node-ddp/{run_id}"
    work_root = pathlib.Path("/opt/pi05/ddp-validation") / run_id
    work_root.mkdir(parents=True, exist_ok=True)
    os.chmod(work_root, 0o777)
    identity = _instance_identity()
    launch = _load_launch_metadata()
    descriptor = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "run_id": run_id,
        **identity,
        "account_id": EXPECTED_ACCOUNT,
        "region": EXPECTED_REGION,
        "reservation_id": launch["reservation_id"],
        "command_sha256": launch["command_sha256"],
        "orchestrator_sha256": values["PI05_ORCHESTRATOR_SHA256"],
    }
    descriptor_path = work_root / "descriptor.json"
    _write_json(descriptor_path, descriptor)
    descriptor_sha = _sha256(descriptor_path)
    descriptor_receipt = _put_create_once(
        f"{prefix}/rendezvous/{identity['instance_id']}.json", descriptor_path, descriptor_sha
    )

    go_path = work_root / "go.json"
    go, go_version = _wait_for_singleton(f"{prefix}/go.json", go_path, timeout_seconds=1200)
    rank, master_ip, peer_ip = _validate_go(go, values, identity, launch)
    interface = _network_interface(peer_ip)

    smoke_path = work_root / "repro_ddp_smoke.py"
    _download_exact(
        values["PI05_SMOKE_KEY"],
        values["PI05_SMOKE_VERSION_ID"],
        smoke_path,
        values["PI05_SMOKE_SHA256"],
    )
    os.chmod(smoke_path, 0o444)
    image_identity = _prepare_image(values)
    nonroot_identity = _prepare_nonroot_identity(work_root)

    output_dir = work_root / "output"
    output_dir.mkdir(mode=0o777, exist_ok=True)
    os.chmod(output_dir, 0o777)
    identity_preflight = _run_identity_preflight(
        values,
        name=f"pi05-ddp-identity-rank-{rank}",
        output_dir=output_dir,
        nonroot_identity=nonroot_identity,
        log_path=work_root / "identity-preflight.log",
    )
    synthetic_name = f"pi05-ddp-synthetic-rank-{rank}"
    synthetic = [
        *_docker_base(
            values,
            name=synthetic_name,
            rank=rank,
            master_ip=master_ip,
            interface=interface,
            output_dir=output_dir,
            smoke_path=smoke_path,
            port=29400,
            nonroot_identity=nonroot_identity,
        ),
        "/opt/pi05/repro_ddp_smoke.py",
        "--output-dir",
        "/output/synthetic",
        "--expected-world-size",
        "2",
        "--iterations",
        "20",
        "--batch-size",
        "32",
        "--seed",
        "42",
        "--run-id",
        run_id,
        "--script-sha256",
        values["PI05_SMOKE_SHA256"],
        "--image-digest",
        values["PI05_IMAGE_DIGEST"],
        "--expected-private-ip",
        identity["private_ip"],
    ]
    synthetic_seconds = _run_container(synthetic, synthetic_name, work_root / "synthetic.log", timeout_seconds=900)

    actual_root = output_dir / "actual"
    actual_seconds = _run_openpi_debug(
        values,
        rank=rank,
        master_ip=master_ip,
        interface=interface,
        actual_root=actual_root,
        smoke_path=smoke_path,
        port=29401,
        nonroot_identity=nonroot_identity,
        experiment=f"{run_id}-debug",
        container_label="base",
        log_path=work_root / "openpi-debug.log",
    )

    inventory = _inventory(work_root)
    report = {
        "schema_version": 1,
        "status": "succeeded",
        "finished_at": _utc_now(),
        "run_id": run_id,
        "rank": rank,
        "world_size": 2,
        **identity,
        "master_private_ip": master_ip,
        "peer_private_ip": peer_ip,
        "network_interface": interface,
        "reservation_id": launch["reservation_id"],
        "command_sha256": launch["command_sha256"],
        "descriptor_receipt": descriptor_receipt,
        "go_version_id": go_version,
        "image": image_identity,
        "identity_preflight": identity_preflight,
        "synthetic_seconds": synthetic_seconds,
        "openpi_debug_seconds": actual_seconds,
        "openpi_debug_steps": 10,
        "openpi_debug_model": {
            "paligemma_variant": DEBUG_PALIGEMMA_VARIANT,
            "action_expert_variant": DEBUG_ACTION_EXPERT_VARIANT,
            "pytorch_compile_mode": None,
        },
        "inventory": inventory,
    }
    report_path = work_root / "result.json"
    _write_json(report_path, report)
    result_receipt = _put_create_once(
        f"{prefix}/results/{identity['instance_id']}.json", report_path, _sha256(report_path)
    )

    if rank == 0:
        peers = _wait_for_results(prefix, work_root, timeout_seconds=600)
        if {item.get("rank") for item in peers} != {0, 1} or any(item.get("status") != "succeeded" for item in peers):
            raise ValidationError("two host reports do not form a successful rank-zero/rank-one pair")
        if len({item.get("command_sha256") for item in peers}) != 1:
            raise ValidationError("host reports disagree on the launch command")
        aggregate = {
            "schema_version": 1,
            "status": "succeeded",
            "finished_at": _utc_now(),
            "run_id": run_id,
            "reservation_id": launch["reservation_id"],
            "command_sha256": launch["command_sha256"],
            "go_version_id": go_version,
            "synthetic_nccl_ddp_pass": True,
            "openpi_debug_two_node_ten_step_pass": True,
            "hosts": peers,
        }
        aggregate_path = work_root / "aggregate.json"
        _write_json(aggregate_path, aggregate)
        aggregate_receipt = _put_create_once(f"{prefix}/aggregate.json", aggregate_path, _sha256(aggregate_path))
        marker = {
            "schema_version": 1,
            "status": "succeeded",
            "run_id": run_id,
            "aggregate": aggregate_receipt,
        }
        marker_path = work_root / "expected-output.json"
        _write_json(marker_path, marker)
        _put_create_once(f"{prefix}/expected-output.json", marker_path, _sha256(marker_path))
    else:
        marker_path = work_root / "expected-output.json"
        marker, _ = _wait_for_singleton(f"{prefix}/expected-output.json", marker_path, timeout_seconds=600)
        if marker.get("status") != "succeeded" or marker.get("run_id") != run_id:
            raise ValidationError("rank zero published an invalid expected-output marker")
    return {**report, "result_receipt": result_receipt}


def _run_openpi_recovery(values: dict[str, str], attempt: str) -> dict[str, Any]:
    run_id = values["PI05_RUN_ID"]
    source_prefix = f"diagnostics/two-node-ddp/{run_id}"
    prefix = _recovery_prefix(run_id, attempt)
    source_root = pathlib.Path("/opt/pi05/ddp-validation") / run_id
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValidationError("source two-node validation root is missing or unsafe")
    recovery_root = source_root / "recoveries" / attempt
    recovery_root.mkdir(parents=True, exist_ok=False)
    os.chmod(recovery_root, 0o777)

    identity = _instance_identity()
    launch = _load_launch_metadata()
    go_path = recovery_root / "source-go.json"
    go, go_version = _wait_for_singleton(f"{source_prefix}/go.json", go_path, timeout_seconds=120)
    rank, master_ip, peer_ip = _validate_go(
        go,
        values,
        identity,
        launch,
        expected_orchestrator_sha256=values["PI05_SOURCE_ORCHESTRATOR_SHA256"],
    )
    interface = _network_interface(peer_ip)

    failure_path = recovery_root / "source-failure.json"
    source_failure, failure_version = _wait_for_singleton(
        f"{source_prefix}/failures/{identity['instance_id']}.json",
        failure_path,
        timeout_seconds=120,
    )
    if (
        source_failure.get("schema_version") != 1
        or source_failure.get("status") != "failed"
        or source_failure.get("run_id") != run_id
        or source_failure.get("instance_id") != identity["instance_id"]
    ):
        raise ValidationError("source failure object does not bind this exact host/run")
    source_synthetic = _validate_source_synthetic(source_root, values, identity, go, rank)

    smoke_path = recovery_root / "repro_ddp_smoke.py"
    _download_exact(
        values["PI05_SMOKE_KEY"],
        values["PI05_SMOKE_VERSION_ID"],
        smoke_path,
        values["PI05_SMOKE_SHA256"],
    )
    smoke_path.chmod(0o444)
    image_identity = _prepare_image(values)
    nonroot_identity = _prepare_nonroot_identity(recovery_root)

    actual_root = recovery_root / "output" / "actual"
    actual_root.mkdir(mode=0o777, parents=True, exist_ok=False)
    os.chmod(actual_root, 0o777)
    identity_preflight = _run_identity_preflight(
        values,
        name=f"pi05-ddp-identity-{attempt}-rank-{rank}",
        output_dir=actual_root,
        nonroot_identity=nonroot_identity,
        log_path=recovery_root / "identity-preflight.log",
    )
    port = _recovery_port(attempt)
    actual_seconds = _run_openpi_debug(
        values,
        rank=rank,
        master_ip=master_ip,
        interface=interface,
        actual_root=actual_root,
        smoke_path=smoke_path,
        port=port,
        nonroot_identity=nonroot_identity,
        experiment=f"{run_id}-{attempt}-debug",
        container_label=attempt,
        log_path=recovery_root / "openpi-debug.log",
    )
    inventory = _inventory(recovery_root)
    report = {
        "schema_version": 1,
        "status": "succeeded",
        "finished_at": _utc_now(),
        "mode": "openpi_same_node_recovery",
        "recovery_attempt": attempt,
        "run_id": run_id,
        "rank": rank,
        "world_size": 2,
        **identity,
        "master_private_ip": master_ip,
        "peer_private_ip": peer_ip,
        "network_interface": interface,
        "rendezvous_port": port,
        "reservation_id": launch["reservation_id"],
        "command_sha256": launch["command_sha256"],
        "source_orchestrator_sha256": values["PI05_SOURCE_ORCHESTRATOR_SHA256"],
        "recovery_orchestrator_sha256": values["PI05_ORCHESTRATOR_SHA256"],
        "go_version_id": go_version,
        "source_failure_version_id": failure_version,
        "source_failure_sha256": _sha256(failure_path),
        "source_synthetic": source_synthetic,
        "synthetic_reused_not_rerun": True,
        "image": image_identity,
        "identity_preflight": identity_preflight,
        "openpi_debug_seconds": actual_seconds,
        "openpi_debug_steps": 10,
        "openpi_debug_model": {
            "paligemma_variant": DEBUG_PALIGEMMA_VARIANT,
            "action_expert_variant": DEBUG_ACTION_EXPERT_VARIANT,
            "pytorch_compile_mode": None,
        },
        "inventory": inventory,
    }
    result_receipt = _publish_two_host_success(
        prefix=prefix,
        work_root=recovery_root,
        report=report,
        rank=rank,
        aggregate_claims={
            "source_synthetic_nccl_ddp_pass": True,
            "synthetic_reused_not_rerun": True,
            "nonroot_identity_preflight_pass": True,
            "openpi_debug_two_node_ten_step_pass": True,
        },
    )
    return {**report, "result_receipt": result_receipt}


def main() -> int:
    args = _parser().parse_args()
    recovery_attempt = _validate_recovery_attempt(args.recovery_attempt)
    values: dict[str, str] = {}
    identity: dict[str, str] = {}
    try:
        values = _required_environment(recovery_attempt=recovery_attempt)
        identity = _instance_identity()
        result = (
            _run_openpi_recovery(values, recovery_attempt) if recovery_attempt is not None else _run_validation(values)
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "finished_at": _utc_now(),
            "run_id": values.get("PI05_RUN_ID"),
            "instance_id": identity.get("instance_id"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            if values.get("PI05_RUN_ID") and identity.get("instance_id"):
                root = pathlib.Path("/opt/pi05/ddp-validation") / values["PI05_RUN_ID"]
                if recovery_attempt is not None:
                    root = root / "recoveries" / recovery_attempt
                path = root / "failure.json"
                _write_json(path, failure)
                failure_prefix = (
                    _recovery_prefix(values["PI05_RUN_ID"], recovery_attempt)
                    if recovery_attempt is not None
                    else f"diagnostics/two-node-ddp/{values['PI05_RUN_ID']}"
                )
                _put_create_once(
                    f"{failure_prefix}/failures/{identity['instance_id']}.json",
                    path,
                    _sha256(path),
                )
        except Exception as publish_exc:  # Keep the original diagnostic error authoritative.
            failure["failure_publish_error"] = str(publish_exc)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
