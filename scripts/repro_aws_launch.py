#!/usr/bin/env python3
"""Plan or launch a budget-guarded On-Demand EC2 reproduction worker.

Planning is the default and performs read-only AWS preflight checks.  A paid
launch requires the explicit ``--execute`` flag.  The launcher intentionally
uses the AWS CLI instead of boto3 so it works on the manual workbench without
adding another unpinned Python dependency.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping
import contextlib
import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import math
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Any
import uuid

try:
    from scripts import repro_cost_guard
except ModuleNotFoundError:  # Support `python scripts/repro_aws_launch.py ...`.
    import repro_cost_guard

EXPECTED_ACCOUNT = "752160877725"
EXPECTED_REGION = "us-east-2"
EXPECTED_PROJECT = "pi05-aws-repro"
EXPECTED_ARTIFACT_BUCKET = "pi05-repro-752160877725-us-east-2"
EXPECTED_AMI_OWNER_ID = "898082745236"
S3_LEDGER_KEY = "control/cost-ledger.json"
DEFAULT_CONFIG = pathlib.Path("repro/reproduction.json")
DEFAULT_FOUNDATION = pathlib.Path("repro/aws-foundation.json")
DEFAULT_LEDGER = repro_cost_guard.DEFAULT_LEDGER
BOOT_AND_SHUTDOWN_RESERVE_HOURS = 0.25
UTC = getattr(dt, "UTC", dt.timezone.utc)  # noqa: UP017 -- direct-script compatibility with macOS Python 3.9.
SCHEDULER_ROLE_RE = re.compile(r"^arn:aws:iam::752160877725:role/[A-Za-z0-9+=,.@_/-]{1,512}$")
DEFAULT_AMI_KEY = "base"
WORKLOAD_AMI_KEYS = {"evaluation": "evaluation"}

# A workload may only launch the machines assigned to that stage in the plan.
# Spend categories are separate: ``corrective_run`` can fund a bounded retry,
# but it may change hardware only through an explicit fallback below.
WORKLOAD_MATRIX: dict[str, dict[str, int]] = {
    "workbench_setup": {"g6e.4xlarge": 1},
    # Every documented Shallow command is one-node, two-process DDP.  Keep the
    # launch guard aligned with that contract: g7e.4xlarge has only one GPU and
    # this launcher does not implement a multi-node rendezvous.  A separately
    # budgeted single-process fallback is admitted below only as a corrective
    # run after a failed two-GPU runtime gate.
    "shallow_training": {"g7e.12xlarge": 1},
    "snapflow_bc": {"g7e.2xlarge": 2},
    "export_compile_quantize": {"g7e.4xlarge": 1},
    "evaluation": {"g6e.4xlarge": 4},
}

# Corrective capacity may narrow a workload's hardware shape when the manual
# runbook has established that the primary shape cannot pass its runtime gate.
# Keeping this outside WORKLOAD_MATRIX prevents an ordinary Shallow launch from
# silently selecting one GPU and preserves the explicit corrective-run audit
# trail in both the cost ledger and run manifest.
CORRECTIVE_WORKLOAD_FALLBACKS: dict[str, dict[str, int]] = {
    "shallow_training": {"g7e.4xlarge": 1},
}

# The category selects a spend cap. Normal runs use the same-named workload;
# corrective runs must declare one underlying workload and match either its
# primary hardware matrix or one explicit corrective fallback.
LAUNCH_MATRIX: dict[str, dict[str, int]] = {
    **WORKLOAD_MATRIX,
    "corrective_run": {
        "g6e.4xlarge": 2,
        "g7e.2xlarge": 2,
        "g7e.4xlarge": 2,
        "g7e.12xlarge": 1,
    },
}


class LaunchError(RuntimeError):
    """Raised when a launch violates policy or AWS preflight fails."""


class AwsCliError(LaunchError):
    """Raised for a failed AWS CLI request."""

    def __init__(self, operation: str, returncode: int, stderr: str):
        detail = " ".join(stderr.strip().split())[-1000:]
        super().__init__(f"AWS CLI {operation} failed ({returncode}): {detail}")
        self.operation = operation
        self.returncode = returncode


class AwsCli:
    """Small JSON-only AWS CLI adapter, kept injectable for unit tests."""

    def __init__(self, region: str):
        self.region = region

    def json(self, arguments: list[str]) -> dict[str, Any]:
        arguments = list(arguments)
        outfile: str | None = None
        if arguments[:2] == ["s3api", "get-object"]:
            # The AWS CLI requires get-object's positional outfile to be the
            # final token, after even global options such as --output.
            outfile = arguments.pop()
        command = [
            "aws",
            "--region",
            self.region,
            "--no-cli-pager",
            *arguments,
            "--output",
            "json",
        ]
        if outfile is not None:
            command.append(outfile)
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
        except OSError as exc:
            raise AwsCliError(" ".join(arguments[:2]), -1, str(exc)) from exc
        if completed.returncode != 0:
            operation = " ".join(arguments[:2])
            raise AwsCliError(operation, completed.returncode, completed.stderr)
        try:
            result = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise LaunchError(f"AWS CLI returned invalid JSON for {' '.join(arguments[:2])}") from exc
        if not isinstance(result, dict):
            raise LaunchError(f"AWS CLI returned a non-object for {' '.join(arguments[:2])}")
        return result

    def dry_run(self, arguments: list[str]) -> None:
        """Have EC2 validate a mutating request without creating resources."""

        command = [
            "aws",
            "--region",
            self.region,
            "--no-cli-pager",
            *arguments,
            "--dry-run",
            "--output",
            "json",
        ]
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
        except OSError as exc:
            raise AwsCliError(" ".join(arguments[:2]) + " --dry-run", -1, str(exc)) from exc
        # EC2 intentionally returns a nonzero DryRunOperation response when
        # permissions and request syntax are valid.
        if completed.returncode != 0 and "DryRunOperation" in completed.stderr:
            return
        detail = completed.stderr or completed.stdout or "unexpected successful dry-run response"
        raise AwsCliError(" ".join(arguments[:2]) + " --dry-run", completed.returncode, detail)


@dataclasses.dataclass(frozen=True)
class StaticInputs:
    config: dict[str, Any]
    foundation: dict[str, Any]
    workload: str
    subnet_id: str
    availability_zone: str
    security_group_id: str
    instance_profile_name: str
    instance_role_name: str
    artifact_bucket: str
    ami_id: str
    ami_name: str
    ami_owner: str
    ami_owner_id: str
    ami_architecture: str
    ami_platform_details: str
    ami_virtualization_type: str
    ami_root_device_name: str


@dataclasses.dataclass(frozen=True)
class LaunchPlan:
    category: str
    workload: str
    instance_type: str
    instance_count: int
    max_runtime_hours: float
    reserved_hours: float
    projected_usd: float
    non_compute_reserved_usd: float
    remaining_after_usd: float
    label: str
    command: str
    command_sha256: str
    subnet_id: str
    availability_zone: str
    security_group_id: str
    instance_profile_name: str
    ami_id: str
    root_device_name: str
    root_volume_gib: int
    shutdown_behavior: str
    retain_after_command: bool
    deadline_utc: str
    scheduler_role_arn: str | None


def _read_required_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise LaunchError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LaunchError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LaunchError(f"expected a JSON object in {path}")
    return value


def resolve_workload(category: str, workload: str | None) -> str:
    if category == "corrective_run":
        if workload is None:
            raise LaunchError("corrective_run requires an explicit underlying --workload")
    elif workload is None:
        workload = category
    elif workload != category:
        raise LaunchError(f"budget category {category} cannot declare workload {workload}")
    if workload not in WORKLOAD_MATRIX:
        raise LaunchError(f"unsupported workload: {workload}")
    return workload


def validate_launch_policy(
    category: str,
    instance_type: str,
    instance_count: int,
    *,
    workload: str | None = None,
) -> str:
    allowed = LAUNCH_MATRIX.get(category)
    if allowed is None:
        raise LaunchError(f"budget category cannot launch compute: {category}")
    maximum = allowed.get(instance_type)
    if maximum is None:
        raise LaunchError(f"{instance_type} is not approved for category {category}")
    if instance_count != 1:
        raise LaunchError(f"instance count must be exactly 1 per guarded launch, got {instance_count}")
    resolved_workload = resolve_workload(category, workload)
    workload_maximum = WORKLOAD_MATRIX[resolved_workload].get(instance_type)
    if workload_maximum is None and category == "corrective_run":
        workload_maximum = CORRECTIVE_WORKLOAD_FALLBACKS.get(resolved_workload, {}).get(instance_type)
    if workload_maximum is None:
        raise LaunchError(f"{instance_type} is not approved for workload {resolved_workload}")
    if instance_count > workload_maximum:
        raise LaunchError(
            f"instance count {instance_count} exceeds the maximum {workload_maximum} for "
            f"{resolved_workload} on {instance_type}"
        )
    return resolved_workload


def _select_pinned_ami(
    config_aws: Mapping[str, Any],
    foundation: Mapping[str, Any],
    workload: str,
) -> Mapping[str, Any]:
    """Select the one foundation-pinned AMI assigned to this workload."""

    ami_key = WORKLOAD_AMI_KEYS.get(workload, DEFAULT_AMI_KEY)
    launch_amis = foundation.get("launch_amis", {})
    if not isinstance(launch_amis, Mapping):
        raise LaunchError("foundation launch_amis must be an object")
    selected = launch_amis.get(ami_key)
    if not isinstance(selected, Mapping):
        raise LaunchError(f"foundation has no pinned {ami_key} AMI")

    config_key = f"{ami_key}_ami"
    recorded = config_aws.get(config_key)
    if not isinstance(recorded, Mapping):
        raise LaunchError(f"config has no pinned {config_key}")

    identity_fields = (
        "id",
        "name",
        "owner",
        "owner_id",
        "architecture",
        "platform_details",
        "virtualization_type",
        "root_device_name",
    )
    for field in identity_fields:
        value = selected.get(field)
        if not isinstance(value, str) or not value:
            raise LaunchError(f"foundation {ami_key} AMI has no {field}")
        if recorded.get(field) != value:
            raise LaunchError(f"config and foundation {ami_key} AMI {field} differ")
    if selected["owner"] != "amazon":
        raise LaunchError(f"foundation {ami_key} AMI owner must be 'amazon'")
    if selected["owner_id"] != EXPECTED_AMI_OWNER_ID:
        raise LaunchError(f"foundation {ami_key} AMI owner id must be {EXPECTED_AMI_OWNER_ID}")
    if selected["architecture"] != "x86_64":
        raise LaunchError(f"foundation {ami_key} AMI architecture must be 'x86_64'")
    if selected["platform_details"] != "Linux/UNIX":
        raise LaunchError(f"foundation {ami_key} AMI platform must be 'Linux/UNIX'")
    if selected["virtualization_type"] != "hvm":
        raise LaunchError(f"foundation {ami_key} AMI virtualization type must be 'hvm'")
    if selected["root_device_name"] != "/dev/sda1":
        raise LaunchError(f"foundation {ami_key} AMI root device must be '/dev/sda1'")
    if re.fullmatch(r"ami-[0-9a-f]{17}", selected["id"]) is None:
        raise LaunchError(f"foundation {ami_key} AMI id is invalid")
    return selected


def load_static_inputs(
    config_path: pathlib.Path,
    foundation_path: pathlib.Path,
    *,
    subnet_id: str | None,
    category: str,
    instance_type: str,
    instance_count: int,
    workload: str | None = None,
) -> StaticInputs:
    config = _read_required_json(config_path)
    foundation = _read_required_json(foundation_path)
    aws = config.get("aws", {})

    expected_pairs = {
        "config account": (aws.get("account_id"), EXPECTED_ACCOUNT),
        "foundation account": (foundation.get("account_id"), EXPECTED_ACCOUNT),
        "config region": (aws.get("region"), EXPECTED_REGION),
        "foundation region": (foundation.get("region"), EXPECTED_REGION),
        "config project": (config.get("project"), EXPECTED_PROJECT),
        "foundation project": (foundation.get("project"), EXPECTED_PROJECT),
        "purchase option": (aws.get("purchase_option"), "On-Demand"),
    }
    for field, (actual, expected) in expected_pairs.items():
        if actual != expected:
            raise LaunchError(f"{field} mismatch: expected {expected!r}, got {actual!r}")

    resolved_workload = validate_launch_policy(
        category,
        instance_type,
        instance_count,
        workload=workload,
    )
    if instance_type not in aws.get("approved_instances", {}):
        raise LaunchError(f"instance type missing from approved_instances: {instance_type}")

    network = foundation.get("network", {})
    subnets = network.get("subnets", [])
    by_id = {item.get("subnet_id"): item for item in subnets}
    if not by_id:
        raise LaunchError("foundation contains no approved subnets")
    selected_id = subnet_id or next(iter(by_id))
    if selected_id not in by_id:
        raise LaunchError(f"subnet is not pinned in the foundation: {selected_id}")
    selected = by_id[selected_id]
    availability_zone = str(selected.get("availability_zone", ""))
    if not availability_zone.startswith(EXPECTED_REGION):
        raise LaunchError(f"subnet has unexpected availability zone: {availability_zone}")

    resources = foundation.get("resources", {})
    iam = resources.get("iam", {})
    artifact_bucket = str(resources.get("s3", {}).get("bucket", ""))
    if artifact_bucket != EXPECTED_ARTIFACT_BUCKET:
        raise LaunchError(f"foundation artifact bucket mismatch: {artifact_bucket!r}")
    security_group = network.get("security_group", {})
    ami = _select_pinned_ami(aws, foundation, resolved_workload)
    if security_group.get("ingress_rule_count") != 0:
        raise LaunchError("foundation does not assert a zero-ingress security group")

    return StaticInputs(
        config=config,
        foundation=foundation,
        workload=resolved_workload,
        subnet_id=selected_id,
        availability_zone=availability_zone,
        security_group_id=str(security_group.get("id", "")),
        instance_profile_name=str(iam.get("instance_profile_name", "")),
        instance_role_name=str(iam.get("role_name", "")),
        artifact_bucket=artifact_bucket,
        ami_id=str(ami["id"]),
        ami_name=str(ami["name"]),
        ami_owner=str(ami["owner"]),
        ami_owner_id=str(ami["owner_id"]),
        ami_architecture=str(ami["architecture"]),
        ami_platform_details=str(ami["platform_details"]),
        ami_virtualization_type=str(ami["virtualization_type"]),
        ami_root_device_name=str(ami["root_device_name"]),
    )


def verify_live_environment(aws: AwsCli, inputs: StaticInputs, instance_type: str) -> str:
    identity = aws.json(["sts", "get-caller-identity"])
    if identity.get("Account") != EXPECTED_ACCOUNT:
        raise LaunchError(f"refusing AWS account {identity.get('Account')!r}; expected {EXPECTED_ACCOUNT}")

    profile_result = aws.json(["iam", "get-instance-profile", "--instance-profile-name", inputs.instance_profile_name])
    profile = profile_result.get("InstanceProfile", {})
    role_names = {role.get("RoleName") for role in profile.get("Roles", [])}
    if profile.get("InstanceProfileName") != inputs.instance_profile_name or role_names != {inputs.instance_role_name}:
        raise LaunchError("live instance profile does not contain exactly the pinned SSM role")

    groups = aws.json(["ec2", "describe-security-groups", "--group-ids", inputs.security_group_id]).get(
        "SecurityGroups", []
    )
    if len(groups) != 1:
        raise LaunchError("pinned security group was not returned exactly once")
    group = groups[0]
    if group.get("GroupId") != inputs.security_group_id:
        raise LaunchError("AWS returned the wrong security group")
    if group.get("VpcId") != inputs.foundation["network"]["vpc_id"]:
        raise LaunchError("security group VPC differs from the pinned foundation VPC")
    if group.get("IpPermissions"):
        raise LaunchError("pinned security group now has inbound rules; refusing launch")
    if not group.get("IpPermissionsEgress"):
        raise LaunchError("pinned security group has no egress for SSM")

    subnets = aws.json(["ec2", "describe-subnets", "--subnet-ids", inputs.subnet_id]).get("Subnets", [])
    if len(subnets) != 1:
        raise LaunchError("pinned subnet was not returned exactly once")
    subnet = subnets[0]
    if (
        subnet.get("SubnetId") != inputs.subnet_id
        or subnet.get("VpcId") != inputs.foundation["network"]["vpc_id"]
        or subnet.get("AvailabilityZone") != inputs.availability_zone
        or subnet.get("State") != "available"
    ):
        raise LaunchError("live subnet no longer matches the pinned foundation")

    images = aws.json(
        [
            "ec2",
            "describe-images",
            "--owners",
            inputs.ami_owner,
            "--image-ids",
            inputs.ami_id,
        ]
    ).get("Images", [])
    if len(images) != 1:
        raise LaunchError("pinned AMI was not returned exactly once")
    image = images[0]
    if image.get("ImageId") != inputs.ami_id or image.get("Name") != inputs.ami_name:
        raise LaunchError("live AMI id or name differs from the pinned identity")
    owner_alias = image.get("ImageOwnerAlias")
    if owner_alias is not None and owner_alias != inputs.ami_owner:
        raise LaunchError("live AMI owner alias differs from the pinned amazon owner")
    if image.get("OwnerId") != inputs.ami_owner_id:
        raise LaunchError("live AMI owner id differs from the pinned amazon owner")
    if image.get("State") != "available":
        raise LaunchError("pinned AMI is not available")
    if image.get("Architecture") != inputs.ami_architecture:
        raise LaunchError("live AMI architecture differs from the pinned x86_64 architecture")
    if image.get("PlatformDetails") != inputs.ami_platform_details:
        raise LaunchError("live AMI platform differs from the pinned Linux/UNIX platform")
    if image.get("VirtualizationType") != inputs.ami_virtualization_type:
        raise LaunchError("live AMI virtualization type differs from the pinned hvm type")
    root_device_name = str(image.get("RootDeviceName", ""))
    if root_device_name != inputs.ami_root_device_name:
        raise LaunchError("live AMI root device differs from the pinned root device")

    offerings = aws.json(
        [
            "ec2",
            "describe-instance-type-offerings",
            "--location-type",
            "availability-zone",
            "--filters",
            f"Name=instance-type,Values={instance_type}",
            f"Name=location,Values={inputs.availability_zone}",
        ]
    ).get("InstanceTypeOfferings", [])
    if not any(
        item.get("InstanceType") == instance_type and item.get("Location") == inputs.availability_zone
        for item in offerings
    ):
        raise LaunchError(f"{instance_type} is not offered in {inputs.availability_zone}")
    return root_device_name


def make_plan(
    inputs: StaticInputs,
    ledger_path: pathlib.Path,
    *,
    category: str,
    instance_type: str,
    instance_count: int,
    max_runtime_hours: float,
    label: str,
    command: str,
    root_device_name: str,
    root_volume_gib: int | None,
    workload: str | None = None,
    retain_after_command: bool = False,
    scheduler_role_arn: str | None = None,
    now: dt.datetime | None = None,
) -> LaunchPlan:
    if not math.isfinite(max_runtime_hours) or max_runtime_hours <= 0:
        raise LaunchError("hours must be a finite number greater than zero")
    if not label.strip():
        raise LaunchError("label must not be empty")
    if category != "workbench_setup" and not command.strip():
        raise LaunchError("non-workbench launches require a command so paid GPUs do not start idle")
    if scheduler_role_arn is not None and SCHEDULER_ROLE_RE.fullmatch(scheduler_role_arn) is None:
        raise LaunchError("scheduler role ARN is not a valid role in the pinned AWS account")
    resolved_workload = validate_launch_policy(
        category,
        instance_type,
        instance_count,
        workload=workload,
    )
    if inputs.workload != resolved_workload:
        raise LaunchError(f"static inputs were selected for workload {inputs.workload}, not {resolved_workload}")
    retained_corrective_shallow = (
        category == "corrective_run"
        and resolved_workload == "shallow_training"
        and instance_type == "g7e.4xlarge"
        and instance_count == 1
    )
    if retain_after_command and resolved_workload != "export_compile_quantize" and not retained_corrective_shallow:
        raise LaunchError(
            "--retain-after-command is allowed only for export/compile or the one-GPU corrective Shallow fallback"
        )

    volume_gib = root_volume_gib if root_volume_gib is not None else (1024 if category == "workbench_setup" else 256)
    if not 100 <= volume_gib <= 2048:
        raise LaunchError("root volume must be between 100 and 2048 GiB")

    reserved_hours = max_runtime_hours + BOOT_AND_SHUTDOWN_RESERVE_HOURS
    ledger = repro_cost_guard.load_json(ledger_path, {"schema_version": 1, "entries": []})
    projection = repro_cost_guard.project_run(
        inputs.config,
        ledger,
        category=category,
        instance_type=instance_type,
        instance_count=instance_count,
        hours=reserved_hours,
    )
    current_time = now or dt.datetime.now(UTC)
    if current_time.tzinfo is None:
        raise LaunchError("internal deadline time must be timezone-aware")
    deadline = current_time.astimezone(UTC) + dt.timedelta(hours=max_runtime_hours)

    return LaunchPlan(
        category=category,
        workload=resolved_workload,
        instance_type=instance_type,
        instance_count=instance_count,
        max_runtime_hours=max_runtime_hours,
        reserved_hours=reserved_hours,
        projected_usd=round(projection.projected_usd, 6),
        non_compute_reserved_usd=round(projection.non_compute_reserved_usd, 6),
        remaining_after_usd=round(projection.remaining_after_usd, 6),
        label=label.strip(),
        command=command,
        command_sha256=hashlib.sha256(command.encode()).hexdigest(),
        subnet_id=inputs.subnet_id,
        availability_zone=inputs.availability_zone,
        security_group_id=inputs.security_group_id,
        instance_profile_name=inputs.instance_profile_name,
        ami_id=inputs.ami_id,
        root_device_name=root_device_name,
        root_volume_gib=volume_gib,
        shutdown_behavior="stop" if category == "workbench_setup" else "terminate",
        retain_after_command=retain_after_command,
        deadline_utc=deadline.isoformat(),
        scheduler_role_arn=scheduler_role_arn,
    )


def build_user_data(plan: LaunchPlan, reservation_id: str) -> str:
    command_b64 = base64.b64encode(plan.command.encode()).decode()
    metadata = {
        "category": plan.category,
        "workload": plan.workload,
        "command_sha256": plan.command_sha256,
        "deadline_utc": plan.deadline_utc,
        "instance_type": plan.instance_type,
        "instance_count": plan.instance_count,
        "label": plan.label,
        "max_runtime_hours": plan.max_runtime_hours,
        "project": EXPECTED_PROJECT,
        "projected_compute_usd": plan.projected_usd,
        "purchase_option": "On-Demand",
        "retain_after_command": plan.retain_after_command,
        "reservation_id": reservation_id,
        "reserved_hours": plan.reserved_hours,
    }
    metadata_b64 = base64.b64encode(json.dumps(metadata, sort_keys=True).encode()).decode()
    on_calendar = dt.datetime.fromisoformat(plan.deadline_utc).astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    job_section = ""
    if plan.command:
        # Successful disposable jobs shut down immediately, but a failed job
        # stays reachable through SSM until the independently scheduled hard
        # deadline.  This preserves the journal and scratch state needed for a
        # bounded same-instance diagnosis instead of deleting the only failure
        # evidence as soon as ExecStart returns nonzero.
        stop_after_job = (
            ""
            if plan.retain_after_command
            else 'ExecStopPost=/bin/sh -c \'if [ "$SERVICE_RESULT" = success ]; then '
            "/usr/bin/systemctl --no-block poweroff; fi'"
        )
        job_section = f"""
printf '%s' '{command_b64}' | base64 --decode > /opt/pi05/run-command.sh
chmod 0700 /opt/pi05/run-command.sh
cat > /etc/systemd/system/pi05-job.service <<'PI05_JOB_SERVICE'
[Unit]
Description=pi05 reproduction job
After=network-online.target dlami-nvme.service docker.service
Wants=network-online.target docker.service

[Service]
Type=simple
WorkingDirectory=/opt/pi05
ExecStart=/bin/bash /opt/pi05/run-command.sh
KillSignal=SIGTERM
{stop_after_job}
StandardOutput=journal
StandardError=journal
PI05_JOB_SERVICE
"""

    return f"""#!/bin/bash
set -euo pipefail
install -d -m 0755 /opt/pi05
printf '%s' '{metadata_b64}' | base64 --decode > /opt/pi05/launch-metadata.json
chmod 0600 /opt/pi05/launch-metadata.json

cat > /etc/systemd/system/pi05-hard-deadline.service <<'PI05_DEADLINE_SERVICE'
[Unit]
Description=Stop pi05 instance at its prepaid deadline

[Service]
Type=oneshot
ExecStart=/usr/sbin/shutdown -h now
PI05_DEADLINE_SERVICE

cat > /etc/systemd/system/pi05-hard-deadline.timer <<'PI05_DEADLINE_TIMER'
[Unit]
Description=Hard prepaid deadline for pi05 instance

[Timer]
OnCalendar={on_calendar}
AccuracySec=1s
Persistent=true
Unit=pi05-hard-deadline.service

[Install]
WantedBy=timers.target
PI05_DEADLINE_TIMER
{job_section}
systemctl daemon-reload
systemctl enable --now pi05-hard-deadline.timer
systemctl enable --now amazon-ssm-agent.service || systemctl restart amazon-ssm-agent.service || true
if test -f /etc/systemd/system/pi05-job.service; then
  systemctl start pi05-job.service
fi
"""


def _tag_specifications(plan: LaunchPlan, reservation_id: str) -> list[dict[str, Any]]:
    safe_label = plan.label[:200]
    tags = [
        {"Key": "Project", "Value": EXPECTED_PROJECT},
        {"Key": "Reproduction", "Value": "pi05-repro"},
        {"Key": "ManagedBy", "Value": "repro-aws-launch"},
        {"Key": "Category", "Value": plan.category},
        {"Key": "Workload", "Value": plan.workload},
        {"Key": "RunId", "Value": reservation_id},
        {"Key": "CommandSha256", "Value": plan.command_sha256},
        {"Key": "Name", "Value": safe_label},
    ]
    return [
        {"ResourceType": "instance", "Tags": tags},
        {"ResourceType": "volume", "Tags": tags},
    ]


def build_run_instances_arguments(plan: LaunchPlan, reservation_id: str) -> list[str]:
    network_interfaces = [
        {
            "AssociatePublicIpAddress": True,
            "DeleteOnTermination": True,
            "DeviceIndex": 0,
            "Groups": [plan.security_group_id],
            "SubnetId": plan.subnet_id,
        }
    ]
    block_devices = [
        {
            "DeviceName": plan.root_device_name,
            "Ebs": {
                "DeleteOnTermination": True,
                "Encrypted": True,
                "VolumeSize": plan.root_volume_gib,
                "VolumeType": "gp3",
            },
        }
    ]
    return [
        "ec2",
        "run-instances",
        "--image-id",
        plan.ami_id,
        "--instance-type",
        plan.instance_type,
        "--count",
        str(plan.instance_count),
        "--client-token",
        reservation_id,
        "--iam-instance-profile",
        json.dumps({"Name": plan.instance_profile_name}, separators=(",", ":")),
        "--network-interfaces",
        json.dumps(network_interfaces, separators=(",", ":")),
        "--block-device-mappings",
        json.dumps(block_devices, separators=(",", ":")),
        "--metadata-options",
        "HttpEndpoint=enabled,HttpTokens=required,HttpPutResponseHopLimit=1,InstanceMetadataTags=enabled",
        "--capacity-reservation-specification",
        "CapacityReservationPreference=none",
        "--placement",
        "Tenancy=default",
        "--instance-initiated-shutdown-behavior",
        plan.shutdown_behavior,
        "--tag-specifications",
        json.dumps(_tag_specifications(plan, reservation_id), separators=(",", ":")),
        "--user-data",
        build_user_data(plan, reservation_id),
    ]


@contextlib.contextmanager
def _locked_ledger(ledger_path: pathlib.Path):
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ledger_path.with_name(f"{ledger_path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@dataclasses.dataclass(frozen=True)
class LedgerSnapshot:
    document: dict[str, Any]
    etag: str
    version_id: str


def _validate_ledger_document(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != 1 or not isinstance(value.get("entries"), list):
        raise LaunchError(f"{context} is not a schema-version-1 cost ledger")
    ids: set[str] = set()
    for entry in value["entries"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"]:
            raise LaunchError(f"{context} contains an invalid reservation entry")
        if entry["id"] in ids:
            raise LaunchError(f"{context} contains duplicate reservation id {entry['id']}")
        ids.add(entry["id"])
    try:
        repro_cost_guard.validate_ledger_entries(value)
    except repro_cost_guard.BudgetError as exc:
        raise LaunchError(f"{context} contains invalid budget data: {exc}") from exc
    return value


def _atomic_write_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = pathlib.Path(stream.name)
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _is_missing_s3_object(error: AwsCliError) -> bool:
    detail = str(error).lower()
    return "nosuchkey" in detail or "(404)" in detail or "not found" in detail


def _is_precondition_failure(error: AwsCliError) -> bool:
    detail = str(error).lower()
    return (
        "preconditionfailed" in detail
        or "conditionalrequestconflict" in detail
        or "(409)" in detail
        or "(412)" in detail
    )


class S3CostLedger:
    """Versioned S3 ledger with optimistic concurrency; the local file is only a cache."""

    MAX_BYTES = 16 * 1024 * 1024
    MAX_CAS_ATTEMPTS = 5

    def __init__(self, aws: AwsCli, bucket: str, local_path: pathlib.Path):
        if bucket != EXPECTED_ARTIFACT_BUCKET:
            raise LaunchError(f"refusing cost ledger bucket {bucket!r}")
        self.aws = aws
        self.bucket = bucket
        self.local_path = local_path
        if local_path.exists():
            try:
                local = json.loads(local_path.read_text())
            except json.JSONDecodeError as exc:
                raise LaunchError(f"invalid local cost-ledger cache: {local_path}") from exc
            self.bootstrap_document = _validate_ledger_document(local, "local cost-ledger cache")
        else:
            self.bootstrap_document = {"schema_version": 1, "entries": []}

    @property
    def _common_object_arguments(self) -> list[str]:
        return [
            "--bucket",
            self.bucket,
            "--key",
            S3_LEDGER_KEY,
            "--expected-bucket-owner",
            EXPECTED_ACCOUNT,
        ]

    def read(self) -> LedgerSnapshot | None:
        try:
            head = self.aws.json(["s3api", "head-object", *self._common_object_arguments])
        except AwsCliError as exc:
            if _is_missing_s3_object(exc):
                self._reject_missing_current_with_history()
                return None
            raise
        etag = head.get("ETag")
        version_id = head.get("VersionId")
        content_length = head.get("ContentLength")
        if (
            not isinstance(etag, str)
            or not etag
            or not isinstance(version_id, str)
            or not version_id
            or version_id == "null"
            or isinstance(content_length, bool)
            or not isinstance(content_length, int)
            or not 0 <= content_length <= self.MAX_BYTES
        ):
            raise LaunchError("remote cost ledger has invalid identity, versioning, or size metadata")

        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        temporary: pathlib.Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.local_path.parent,
                prefix=".cost-ledger-download.",
                delete=False,
            ) as stream:
                temporary = pathlib.Path(stream.name)
            response = self.aws.json(
                [
                    "s3api",
                    "get-object",
                    *self._common_object_arguments,
                    "--version-id",
                    version_id,
                    str(temporary),
                ]
            )
            if response.get("ETag") != etag or response.get("VersionId") != version_id:
                raise LaunchError("remote cost-ledger get did not match its pinned head version")
            if temporary.stat().st_size != content_length:
                raise LaunchError("remote cost-ledger body size differs from head metadata")
            try:
                document = json.loads(temporary.read_text())
            except json.JSONDecodeError as exc:
                raise LaunchError("remote cost ledger is not valid JSON") from exc
            document = _validate_ledger_document(document, "remote cost ledger")
            # Cache the authoritative remote state while retaining unique
            # pre-migration IDs until a conditional write has moved them into
            # S3. Matching IDs always use the remote entry.
            _atomic_write_json(self.local_path, self._merge_legacy_cache(document))
            return LedgerSnapshot(document=document, etag=etag, version_id=version_id)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _reject_missing_current_with_history(self) -> None:
        """Do not let a delete marker reset already committed spend.

        In a versioned bucket, ``If-None-Match: *`` may create a new current
        object when the current version is a delete marker.  Before treating a
        404 as first use, require that this exact key has no object history.
        """

        result = self.aws.json(
            [
                "s3api",
                "list-object-versions",
                "--bucket",
                self.bucket,
                "--prefix",
                S3_LEDGER_KEY,
                "--expected-bucket-owner",
                EXPECTED_ACCOUNT,
            ]
        )
        versions = result.get("Versions", [])
        delete_markers = result.get("DeleteMarkers", [])
        if not isinstance(versions, list) or not isinstance(delete_markers, list):
            raise LaunchError("S3 returned invalid cost-ledger version history")
        history = [*versions, *delete_markers]
        if not all(isinstance(item, dict) for item in history):
            raise LaunchError("S3 returned invalid cost-ledger version history")
        if any(item.get("Key") == S3_LEDGER_KEY for item in history):
            raise LaunchError(
                "remote cost ledger is missing but prior versions or delete markers exist; "
                "restore and reconcile the ledger before launching"
            )

    def _merge_legacy_cache(self, document: dict[str, Any]) -> dict[str, Any]:
        merged = json.loads(json.dumps(document))
        remote_ids = {entry["id"] for entry in merged["entries"]}
        for entry in self.bootstrap_document["entries"]:
            if entry["id"] not in remote_ids:
                merged["entries"].append(json.loads(json.dumps(entry)))
        return merged

    def conditional_write(self, document: dict[str, Any], previous_etag: str | None) -> LedgerSnapshot:
        document = _validate_ledger_document(document, "candidate cost ledger")
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        temporary: pathlib.Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.local_path.parent,
                prefix=".cost-ledger-upload.",
                delete=False,
            ) as stream:
                temporary = pathlib.Path(stream.name)
                stream.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
            condition = ["--if-none-match", "*"] if previous_etag is None else ["--if-match", previous_etag]
            response = self.aws.json(
                [
                    "s3api",
                    "put-object",
                    *self._common_object_arguments,
                    "--body",
                    str(temporary),
                    "--content-type",
                    "application/json",
                    "--server-side-encryption",
                    "AES256",
                    "--checksum-algorithm",
                    "SHA256",
                    *condition,
                ]
            )
            etag = response.get("ETag")
            version_id = response.get("VersionId")
            if (
                not isinstance(etag, str)
                or not etag
                or not isinstance(version_id, str)
                or not version_id
                or version_id == "null"
            ):
                raise LaunchError("conditional cost-ledger write returned no versioned object identity")
            _atomic_write_json(self.local_path, document)
            return LedgerSnapshot(document=document, etag=etag, version_id=version_id)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def mutate(self, mutation: Any) -> Any:
        for _attempt in range(self.MAX_CAS_ATTEMPTS):
            snapshot = self.read()
            base = snapshot.document if snapshot is not None else self.bootstrap_document
            document = self._merge_legacy_cache(base)
            result = mutation(document)
            try:
                self.conditional_write(document, snapshot.etag if snapshot is not None else None)
            except AwsCliError as exc:
                if _is_precondition_failure(exc):
                    continue
                raise
            return result
        raise LaunchError("remote cost ledger changed repeatedly; conditional mutation was not committed")

    def refresh_cache(self) -> None:
        self.read()

    def reserve(self, config: dict[str, Any], plan: LaunchPlan) -> str:
        reservation_id = str(uuid.uuid4())
        created_at = dt.datetime.now(UTC).isoformat()

        def append_reservation(ledger: dict[str, Any]) -> None:
            if any(entry.get("id") == reservation_id for entry in ledger["entries"]):
                raise LaunchError(f"duplicate generated reservation id: {reservation_id}")
            duplicates = [
                entry
                for entry in ledger["entries"]
                if entry.get("state", "reserved") not in repro_cost_guard.NON_COMMITTED_STATES
                and entry.get("category") == plan.category
                and entry.get("label") == plan.label
            ]
            if duplicates:
                raise LaunchError(
                    f"unresolved reservation already uses category/label {plan.category}/{plan.label}: "
                    f"{duplicates[0]['id']}"
                )
            projection = repro_cost_guard.project_run(
                config,
                ledger,
                category=plan.category,
                instance_type=plan.instance_type,
                instance_count=plan.instance_count,
                hours=plan.reserved_hours,
            )
            ledger["entries"].append(
                {
                    "id": reservation_id,
                    "created_at": created_at,
                    "state": "launching",
                    "label": plan.label,
                    "command": plan.command,
                    "command_sha256": plan.command_sha256,
                    "category": projection.category,
                    "workload": plan.workload,
                    "instance_type": projection.instance_type,
                    "instance_count": projection.instance_count,
                    "hours": projection.hours,
                    "hourly_usd": projection.hourly_usd,
                    "usd": projection.projected_usd,
                    "deadline_utc": plan.deadline_utc,
                    "max_runtime_hours": plan.max_runtime_hours,
                    "subnet_id": plan.subnet_id,
                    "shutdown_behavior": plan.shutdown_behavior,
                    "retain_after_command": plan.retain_after_command,
                }
            )

        self.mutate(append_reservation)
        return reservation_id

    def update(self, reservation_id: str, **changes: Any) -> None:
        def update_reservation(ledger: dict[str, Any]) -> None:
            matches = [entry for entry in ledger["entries"] if entry.get("id") == reservation_id]
            if len(matches) != 1:
                raise LaunchError(f"ledger reservation {reservation_id} was not found exactly once")
            matches[0].update(changes)

        self.mutate(update_reservation)


def _instances_for_client_token(aws: AwsCli, reservation_id: str) -> list[str]:
    result = aws.json(
        [
            "ec2",
            "describe-instances",
            "--filters",
            f"Name=client-token,Values={reservation_id}",
        ]
    )
    return [
        instance["InstanceId"]
        for reservation in result.get("Reservations", [])
        for instance in reservation.get("Instances", [])
        if instance.get("InstanceId")
    ]


def _active_workbench_ids(aws: AwsCli) -> list[str]:
    result = aws.json(
        [
            "ec2",
            "describe-instances",
            "--filters",
            f"Name=tag:Project,Values={EXPECTED_PROJECT}",
            "Name=tag:Category,Values=workbench_setup",
            "Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down",
        ]
    )
    return sorted(
        {
            str(instance["InstanceId"])
            for reservation in result.get("Reservations", [])
            for instance in reservation.get("Instances", [])
            if instance.get("InstanceId")
        }
    )


def _active_duplicate_job_ids(aws: AwsCli, plan: LaunchPlan) -> list[str]:
    """Find an active managed worker with the same category/label run identity."""

    result = aws.json(
        [
            "ec2",
            "describe-instances",
            "--filters",
            f"Name=tag:Project,Values={EXPECTED_PROJECT}",
            "Name=tag:ManagedBy,Values=repro-aws-launch",
            f"Name=tag:Category,Values={plan.category}",
            "Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down",
        ]
    )
    safe_label = plan.label[:200]
    duplicate_ids: set[str] = set()
    for reservation in result.get("Reservations", []):
        for instance in reservation.get("Instances", []):
            tags = instance.get("Tags", [])
            if not isinstance(tags, list):
                raise LaunchError("EC2 returned invalid tags while checking for duplicate jobs")
            by_key = {
                tag.get("Key"): tag.get("Value")
                for tag in tags
                if isinstance(tag, dict) and isinstance(tag.get("Key"), str)
            }
            instance_id = instance.get("InstanceId")
            if by_key.get("Name") == safe_label and isinstance(instance_id, str) and instance_id:
                duplicate_ids.add(instance_id)
    return sorted(duplicate_ids)


def _create_deadline_schedule(
    aws: AwsCli,
    plan: LaunchPlan,
    reservation_id: str,
    instance_ids: list[str],
) -> tuple[str, str]:
    role_arn = plan.scheduler_role_arn
    if role_arn is None or SCHEDULER_ROLE_RE.fullmatch(role_arn) is None:
        raise LaunchError("--scheduler-role-arn is required for every executed launch")
    try:
        deadline = dt.datetime.fromisoformat(plan.deadline_utc)
    except ValueError as exc:
        raise LaunchError("launch plan has an invalid scheduler deadline") from exc
    if deadline.tzinfo is None:
        raise LaunchError("launch plan scheduler deadline must be timezone-aware")
    action = "stopInstances" if plan.shutdown_behavior == "stop" else "terminateInstances"
    name = f"pi05-deadline-{reservation_id}"
    target = {
        "Arn": f"arn:aws:scheduler:::aws-sdk:ec2:{action}",
        "RoleArn": role_arn,
        "Input": json.dumps({"InstanceIds": instance_ids}, separators=(",", ":")),
    }
    result = aws.json(
        [
            "scheduler",
            "create-schedule",
            "--name",
            name,
            "--action-after-completion",
            "DELETE",
            "--schedule-expression",
            f"at({deadline.astimezone(UTC).strftime('%Y-%m-%dT%H:%M:%S')})",
            "--schedule-expression-timezone",
            "UTC",
            "--flexible-time-window",
            json.dumps({"Mode": "OFF"}, separators=(",", ":")),
            "--target",
            json.dumps(target, separators=(",", ":")),
            "--client-token",
            reservation_id,
            "--state",
            "ENABLED",
        ]
    )
    schedule_arn = result.get("ScheduleArn")
    if not isinstance(schedule_arn, str) or not schedule_arn:
        raise LaunchError("Scheduler accepted no verifiable schedule ARN")
    return name, schedule_arn


def _cleanup_instances_after_schedule_failure(aws: AwsCli, instance_ids: list[str]) -> None:
    aws.json(["ec2", "terminate-instances", "--instance-ids", *instance_ids])


def execute_launch(
    aws: AwsCli,
    inputs: StaticInputs,
    plan: LaunchPlan,
    ledger_path: pathlib.Path,
) -> dict[str, Any]:
    if plan.scheduler_role_arn is None or SCHEDULER_ROLE_RE.fullmatch(plan.scheduler_role_arn) is None:
        raise LaunchError("--scheduler-role-arn is required for every executed launch")
    if plan.category == "workbench_setup":
        existing_workbenches = _active_workbench_ids(aws)
        if existing_workbenches:
            raise LaunchError(f"active workbench already exists: {', '.join(existing_workbenches)}")
    else:
        duplicate_jobs = _active_duplicate_job_ids(aws, plan)
        if duplicate_jobs:
            raise LaunchError(f"active job already uses this category/label: {', '.join(duplicate_jobs)}")

    # Re-project and reserve in the versioned S3 ledger before RunInstances.
    # The file lock serializes local callers; S3 If-Match/If-None-Match is the
    # cross-controller authority.
    ledger_store = S3CostLedger(aws, inputs.artifact_bucket, ledger_path)
    with _locked_ledger(ledger_path):
        reservation_id = ledger_store.reserve(inputs.config, plan)

    arguments = build_run_instances_arguments(plan, reservation_id)
    recovered = False
    launch_error_text: str | None = None
    try:
        result = aws.json(arguments)
        instances = result.get("Instances")
        if not isinstance(instances, list) or not all(isinstance(instance, dict) for instance in instances):
            raise LaunchError("RunInstances returned an invalid Instances collection")
        instance_ids = [instance.get("InstanceId") for instance in instances]
        instance_ids = [value for value in instance_ids if isinstance(value, str) and value]
        if len(instance_ids) != plan.instance_count:
            raise LaunchError(
                f"RunInstances returned {len(instance_ids)} of {plan.instance_count} expected instance ids"
            )
    except LaunchError as launch_error:
        # A client token makes eventual reconciliation idempotent.  An immediate
        # empty query is not conclusive because EC2 visibility can lag.
        try:
            recovered_ids = _instances_for_client_token(aws, reservation_id)
        except LaunchError as recovery_error:
            with _locked_ledger(ledger_path):
                ledger_store.update(
                    reservation_id,
                    state="launch_unknown",
                    launch_error=str(launch_error),
                    recovery_error=str(recovery_error),
                    updated_at=dt.datetime.now(UTC).isoformat(),
                )
            raise LaunchError(
                f"launch result is unknown; reservation {reservation_id} remains committed. "
                "Reconcile by client token before retrying."
            ) from launch_error
        if recovered_ids:
            if len(recovered_ids) != plan.instance_count:
                with _locked_ledger(ledger_path):
                    ledger_store.update(
                        reservation_id,
                        state="launch_unknown",
                        response_instance_ids=recovered_ids,
                        launch_error=str(launch_error),
                        updated_at=dt.datetime.now(UTC).isoformat(),
                    )
                raise LaunchError(
                    f"recovery found {len(recovered_ids)} instances for reservation {reservation_id}; "
                    "reservation remains committed"
                ) from launch_error
            instance_ids = recovered_ids
            recovered = True
            launch_error_text = str(launch_error)
        else:
            with _locked_ledger(ledger_path):
                ledger_store.update(
                    reservation_id,
                    state="launch_unknown",
                    launch_error=str(launch_error),
                    recovery_observation="no instances visible on the first immediate query",
                    updated_at=dt.datetime.now(UTC).isoformat(),
                )
            raise LaunchError(
                f"launch result is unknown; reservation {reservation_id} remains committed even though the first "
                "query was empty. Reconcile later by client token before retrying."
            ) from launch_error

    try:
        schedule_name, schedule_arn = _create_deadline_schedule(aws, plan, reservation_id, instance_ids)
    except LaunchError as schedule_error:
        cleanup_error: LaunchError | None = None
        try:
            _cleanup_instances_after_schedule_failure(aws, instance_ids)
        except LaunchError as exc:
            cleanup_error = exc
        with _locked_ledger(ledger_path):
            ledger_store.update(
                reservation_id,
                state="cleanup_unknown" if cleanup_error else "schedule_failed_cleanup_requested",
                instance_ids=instance_ids,
                schedule_error=str(schedule_error),
                cleanup_error=str(cleanup_error) if cleanup_error else None,
                updated_at=dt.datetime.now(UTC).isoformat(),
            )
        detail = "cleanup result is unknown" if cleanup_error else "immediate termination was requested"
        raise LaunchError(f"deadline schedule failed for reservation {reservation_id}; {detail}") from schedule_error

    with _locked_ledger(ledger_path):
        ledger_store.update(
            reservation_id,
            state="launched_recovered" if recovered else "launched",
            instance_ids=instance_ids,
            launch_error=launch_error_text,
            schedule_name=schedule_name,
            schedule_arn=schedule_arn,
            updated_at=dt.datetime.now(UTC).isoformat(),
        )
    return {
        "reservation_id": reservation_id,
        "instance_ids": instance_ids,
        "recovered": recovered,
        "schedule_name": schedule_name,
        "schedule_arn": schedule_arn,
    }


def public_plan(plan: LaunchPlan) -> dict[str, Any]:
    result = dataclasses.asdict(plan)
    result.pop("command")
    result["mode"] = "plan"
    result["purchase_option"] = "On-Demand"
    result["capacity_reservation_preference"] = "none"
    result["metadata_tokens"] = "required"
    result["authoritative_cost_ledger"] = f"s3://{EXPECTED_ARTIFACT_BUCKET}/{S3_LEDGER_KEY}"
    result["root_volume"] = {
        "delete_on_termination": True,
        "encrypted": True,
        "size_gib": plan.root_volume_gib,
        "type": "gp3",
    }
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--foundation", type=pathlib.Path, default=DEFAULT_FOUNDATION)
    parser.add_argument("--ledger", type=pathlib.Path, default=DEFAULT_LEDGER)
    parser.add_argument("--category", required=True)
    parser.add_argument(
        "--workload",
        required=True,
        choices=tuple(WORKLOAD_MATRIX),
        help="underlying stage identity; corrective_run may use only an explicitly declared hardware fallback",
    )
    parser.add_argument("--instance-type", required=True)
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--hours", type=float, required=True, help="hard maximum runtime before stop/terminate")
    parser.add_argument("--label", required=True)
    parser.add_argument("--subnet-id", help="must be one of the subnets pinned in aws-foundation.json")
    parser.add_argument("--root-volume-gib", type=int)
    parser.add_argument(
        "--scheduler-role-arn",
        help="EventBridge Scheduler execution role; required with --execute",
    )
    parser.add_argument(
        "--retain-after-command",
        action="store_true",
        help=(
            "keep an export_compile_quantize instance running after its bootstrap command so reviewed "
            "same-instance phases can be replayed over SSM; the absolute guest/external deadline still terminates it"
        ),
    )
    command_group = parser.add_mutually_exclusive_group()
    command_group.add_argument("--command", default="")
    command_group.add_argument("--command-file", type=pathlib.Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="reserve spend and call EC2 RunInstances; omission is read-only plan mode",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        command = args.command_file.read_text() if args.command_file else args.command
        inputs = load_static_inputs(
            args.config,
            args.foundation,
            subnet_id=args.subnet_id,
            category=args.category,
            instance_type=args.instance_type,
            instance_count=args.instance_count,
            workload=args.workload,
        )
        aws = AwsCli(EXPECTED_REGION)
        root_device_name = verify_live_environment(aws, inputs, args.instance_type)
        S3CostLedger(aws, inputs.artifact_bucket, args.ledger).refresh_cache()
        plan = make_plan(
            inputs,
            args.ledger,
            category=args.category,
            instance_type=args.instance_type,
            instance_count=args.instance_count,
            max_runtime_hours=args.hours,
            label=args.label,
            command=command,
            root_device_name=root_device_name,
            root_volume_gib=args.root_volume_gib,
            workload=args.workload,
            retain_after_command=args.retain_after_command,
            scheduler_role_arn=args.scheduler_role_arn,
        )
        dry_run_id = f"dryrun-{plan.command_sha256[:32]}"
        aws.dry_run(build_run_instances_arguments(plan, dry_run_id))
        output = public_plan(plan)
        if args.execute:
            launch = execute_launch(aws, inputs, plan, args.ledger)
            output.update(launch)
            output["mode"] = "executed"
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    except (LaunchError, repro_cost_guard.BudgetError, OSError, KeyError, ValueError) as exc:
        print(f"LAUNCH REJECTED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
