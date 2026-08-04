"""Fail-closed AWS runtime identity checks for compiled-policy stages."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import json
import os
import re
import subprocess
from typing import Any
import urllib.request

EXPECTED_AWS_ACCOUNT = "752160877725"
EXPECTED_AWS_REGION = "us-east-2"
IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
INSTANCE_ID_RE = re.compile(r"^i-(?:[0-9a-f]{8}|[0-9a-f]{17})$")
INSTANCE_TYPE_RE = re.compile(r"^[a-z0-9][a-z0-9.]{1,63}$")
GPU_UUID_RE = re.compile(r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
DRIVER_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")


@dataclasses.dataclass(frozen=True)
class LiveRuntimeIdentity:
    image_digest: str
    instance_type: str
    instance_id: str
    instance_identity_source: str

    @property
    def manifest_runtime(self) -> dict[str, str]:
        return {
            "image_digest": self.image_digest,
            "instance_type": self.instance_type,
            "instance_id": self.instance_id,
        }


def _required_match(value: str, pattern: re.Pattern[str], *, label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{label} has an invalid immutable identity: {value!r}")
    return value


def _imds_identity(
    opener: Callable[..., Any],
    *,
    timeout_seconds: float,
) -> Mapping[str, Any]:
    token_request = urllib.request.Request(
        "http://169.254.169.254/latest/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
    )
    try:
        with opener(token_request, timeout=timeout_seconds) as response:
            token = response.read().decode()
        if not token:
            raise RuntimeError("IMDSv2 returned an empty token")
        identity_request = urllib.request.Request(
            "http://169.254.169.254/latest/dynamic/instance-identity/document",
            headers={"X-aws-ec2-metadata-token": token},
        )
        with opener(identity_request, timeout=timeout_seconds) as response:
            identity = json.load(response)
    except Exception as exc:
        raise RuntimeError(
            "live EC2 identity is unavailable; a network-isolated worker must inject PI05_INSTANCE_ID"
        ) from exc
    if not isinstance(identity, Mapping):
        raise RuntimeError("IMDSv2 instance identity document is not an object")
    return identity


def require_live_runtime_identity(
    *,
    image_digest: str,
    instance_type: str,
    instance_id: str,
    environ: Mapping[str, str] | None = None,
    imds_opener: Callable[..., Any] = urllib.request.urlopen,
    imds_timeout_seconds: float = 1.0,
) -> LiveRuntimeIdentity:
    """Bind declared stage identity to worker-owned environment or live IMDSv2.

    Docker workers run with ``--network none``, so the host worker must inject
    the instance ID it already obtained from IMDSv2. Direct host execution has
    no such environment and therefore proves the same identity through IMDSv2.
    The image digest has no trustworthy in-container discovery mechanism and is
    consequently always required from the worker-owned environment.
    """

    image_digest = _required_match(image_digest, IMAGE_DIGEST_RE, label="declared image digest")
    instance_id = _required_match(instance_id, INSTANCE_ID_RE, label="declared EC2 instance ID")
    instance_type = _required_match(instance_type, INSTANCE_TYPE_RE, label="declared EC2 instance type")
    environment = os.environ if environ is None else environ

    worker_image_digest = environment.get("PI05_IMAGE_DIGEST", "")
    if not worker_image_digest:
        raise RuntimeError("PI05_IMAGE_DIGEST is required to bind the declared runtime to the pulled worker image")
    _required_match(worker_image_digest, IMAGE_DIGEST_RE, label="worker image digest")
    if worker_image_digest != image_digest:
        raise ValueError("declared image digest differs from worker-owned PI05_IMAGE_DIGEST")

    worker_instance_id = environment.get("PI05_INSTANCE_ID", "")
    worker_instance_type = environment.get("PI05_INSTANCE_TYPE", "")
    if bool(worker_instance_id) != bool(worker_instance_type):
        raise RuntimeError("PI05_INSTANCE_ID and PI05_INSTANCE_TYPE must be injected together")
    if worker_instance_id:
        _required_match(worker_instance_id, INSTANCE_ID_RE, label="worker EC2 instance ID")
        if worker_instance_id != instance_id:
            raise ValueError("declared EC2 instance ID differs from worker-owned PI05_INSTANCE_ID")
        _required_match(worker_instance_type, INSTANCE_TYPE_RE, label="worker EC2 instance type")
        if worker_instance_type != instance_type:
            raise ValueError("declared EC2 instance type differs from worker-owned PI05_INSTANCE_TYPE")
        return LiveRuntimeIdentity(image_digest, instance_type, instance_id, "worker-environment")

    identity = _imds_identity(imds_opener, timeout_seconds=imds_timeout_seconds)
    if str(identity.get("accountId")) != EXPECTED_AWS_ACCOUNT or identity.get("region") != EXPECTED_AWS_REGION:
        raise ValueError("live IMDS identity differs from the pinned AWS account or region")
    live_instance_id = _required_match(
        str(identity.get("instanceId", "")), INSTANCE_ID_RE, label="live EC2 instance ID"
    )
    live_instance_type = _required_match(
        str(identity.get("instanceType", "")), INSTANCE_TYPE_RE, label="live EC2 instance type"
    )
    if live_instance_id != instance_id:
        raise ValueError("declared EC2 instance ID differs from live IMDS identity")
    if live_instance_type != instance_type:
        raise ValueError("declared EC2 instance type differs from live IMDS identity")
    return LiveRuntimeIdentity(image_digest, instance_type, instance_id, "imds-v2")


def require_same_image_digest(*digests: str) -> str:
    """Require every export-chain phase to name one exact container digest."""

    if not digests:
        raise ValueError("at least one image digest is required")
    for digest in digests:
        _required_match(digest, IMAGE_DIGEST_RE, label="export-chain image digest")
    if len(set(digests)) != 1:
        raise ValueError("export, quantization, validation, build, benchmark, and serving must use one image digest")
    return digests[0]


def validate_gpu_inventory(rows: Sequence[str]) -> tuple[str, ...]:
    """Validate and canonicalize ``nvidia-smi`` UUID/name/driver rows."""

    if not rows:
        raise RuntimeError("nvidia-smi returned an empty GPU inventory")
    canonical: list[str] = []
    uuids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, str):
            raise RuntimeError(f"nvidia-smi GPU inventory row {index} is not text")
        fields = tuple(field.strip() for field in row.split(","))
        if len(fields) != 3:
            raise RuntimeError(f"nvidia-smi GPU inventory row {index} must contain UUID, name, and driver")
        uuid, name, driver = fields
        if GPU_UUID_RE.fullmatch(uuid) is None or not name or DRIVER_VERSION_RE.fullmatch(driver) is None:
            raise RuntimeError(f"nvidia-smi GPU inventory row {index} is malformed: {row!r}")
        if uuid in uuids:
            raise RuntimeError(f"nvidia-smi GPU inventory contains duplicate UUID: {uuid}")
        uuids.add(uuid)
        canonical.append(", ".join((uuid, name, driver)))
    return tuple(canonical)


def query_gpu_inventory(
    runner: Callable[..., str] = subprocess.check_output,
) -> tuple[str, ...]:
    """Read a required, well-formed live GPU/driver inventory."""

    command = ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"]
    try:
        output = runner(command, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("nvidia-smi GPU identity is required for the compiled-policy pipeline") from exc
    return validate_gpu_inventory(tuple(line.strip() for line in output.splitlines() if line.strip()))
