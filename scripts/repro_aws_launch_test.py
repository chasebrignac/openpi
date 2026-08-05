import base64
import datetime as dt
import json
import subprocess

import pytest

from scripts import repro_aws_launch

BASE_AMI = {
    "id": "ami-01901bc01d5d9bb55",
    "name": "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) 20260724",
    "owner": "amazon",
    "owner_id": "898082745236",
    "architecture": "x86_64",
    "platform_details": "Linux/UNIX",
    "virtualization_type": "hvm",
    "root_device_name": "/dev/sda1",
}
EVALUATION_AMI = {
    "id": "ami-06517bc7fad3c6a48",
    "name": "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20260403",
    "owner": "amazon",
    "owner_id": "898082745236",
    "architecture": "x86_64",
    "platform_details": "Linux/UNIX",
    "virtualization_type": "hvm",
    "root_device_name": "/dev/sda1",
}


def test_aws_cli_get_object_keeps_required_outfile_last(monkeypatch):
    captured = []

    def fake_run(command, **_kwargs):
        captured.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    monkeypatch.setattr(repro_aws_launch.subprocess, "run", fake_run)
    repro_aws_launch.AwsCli("us-east-2").json(
        ["s3api", "get-object", "--bucket", "example", "--key", "ledger.json", "/tmp/ledger.json"]
    )
    assert captured[0][-1] == "/tmp/ledger.json"
    assert captured[0][-3:-1] == ["--output", "json"]


def test_aws_cli_wraps_process_start_failure(monkeypatch):
    def fail_to_start(*_args, **_kwargs):
        raise OSError("cannot execute aws")

    monkeypatch.setattr(repro_aws_launch.subprocess, "run", fail_to_start)
    with pytest.raises(repro_aws_launch.AwsCliError, match="cannot execute aws"):
        repro_aws_launch.AwsCli("us-east-2").json(["ec2", "run-instances"])


@pytest.fixture
def config():
    return {
        "project": "pi05-aws-repro",
        "aws": {
            "account_id": "752160877725",
            "region": "us-east-2",
            "purchase_option": "On-Demand",
            "hard_cap_usd": 3000,
            "category_caps_usd": {
                "workbench_setup": 350,
                "shallow_training": 1200,
                "snapflow_bc": 250,
                "export_compile_quantize": 200,
                "evaluation": 250,
                "storage_logs": 150,
                "corrective_run": 500,
                "headroom": 100,
            },
            "approved_instances": {
                "g6e.4xlarge": {"hourly_usd": 3.00424},
                "g7e.2xlarge": {"hourly_usd": 3.36312},
                "g7e.4xlarge": {"hourly_usd": 3.99816},
                "g7e.12xlarge": {"hourly_usd": 8.28608},
                "g7e.48xlarge": {"hourly_usd": 33.14432},
            },
            "base_ami": dict(BASE_AMI),
            "evaluation_ami": dict(EVALUATION_AMI),
        },
    }


@pytest.fixture
def foundation():
    return {
        "project": "pi05-aws-repro",
        "account_id": "752160877725",
        "region": "us-east-2",
        "launch_amis": {
            "base": dict(BASE_AMI),
            "evaluation": dict(EVALUATION_AMI),
        },
        "network": {
            "vpc_id": "vpc-test",
            "subnets": [
                {"subnet_id": "subnet-a", "availability_zone": "us-east-2a"},
                {"subnet_id": "subnet-b", "availability_zone": "us-east-2b"},
            ],
            "security_group": {"id": "sg-test", "ingress_rule_count": 0},
            "distributed_security_group": {
                "id": "sg-distributed",
                "ingress_rule_count": 1,
                "ingress_contract": "self_referencing_tcp_all_ports",
            },
        },
        "resources": {
            "s3": {"bucket": "pi05-repro-752160877725-us-east-2"},
            "iam": {
                "instance_profile_name": "profile-test",
                "role_name": "role-test",
            },
        },
    }


def write_inputs(tmp_path, config, foundation):
    config_path = tmp_path / "config.json"
    foundation_path = tmp_path / "foundation.json"
    config_path.write_text(json.dumps(config))
    foundation_path.write_text(json.dumps(foundation))
    return config_path, foundation_path


def make_inputs(tmp_path, config, foundation, **overrides):
    config_path, foundation_path = write_inputs(tmp_path, config, foundation)
    values = {
        "subnet_id": None,
        "category": "export_compile_quantize",
        "instance_type": "g7e.4xlarge",
        "instance_count": 1,
    }
    values.update(overrides)
    return repro_aws_launch.load_static_inputs(config_path, foundation_path, **values)


def make_plan(tmp_path, inputs, **overrides):
    values = {
        "category": "export_compile_quantize",
        "instance_type": "g7e.4xlarge",
        "instance_count": 1,
        "max_runtime_hours": 2,
        "label": "trt-pilot",
        "command": "python scripts/export_pi05_onnx.py",
        "root_device_name": "/dev/sda1",
        "root_volume_gib": None,
        "scheduler_role_arn": "arn:aws:iam::752160877725:role/pi05-repro-scheduler",
        "now": dt.datetime(2026, 8, 3, 20, 0, tzinfo=dt.UTC),
    }
    values.update(overrides)
    return repro_aws_launch.make_plan(inputs, tmp_path / "ledger.json", **values)


def test_rejects_unassigned_instance_and_unpinned_subnet(tmp_path, config, foundation):
    config_path, foundation_path = write_inputs(tmp_path, config, foundation)
    with pytest.raises(repro_aws_launch.LaunchError, match="not approved for category"):
        repro_aws_launch.load_static_inputs(
            config_path,
            foundation_path,
            subnet_id=None,
            category="evaluation",
            instance_type="g7e.4xlarge",
            instance_count=1,
        )
    with pytest.raises(repro_aws_launch.LaunchError, match="exactly 1"):
        repro_aws_launch.load_static_inputs(
            config_path,
            foundation_path,
            subnet_id=None,
            category="evaluation",
            instance_type="g6e.4xlarge",
            instance_count=2,
        )
    with pytest.raises(repro_aws_launch.LaunchError, match="not pinned"):
        repro_aws_launch.load_static_inputs(
            config_path,
            foundation_path,
            subnet_id="subnet-other",
            category="evaluation",
            instance_type="g6e.4xlarge",
            instance_count=1,
        )


def test_distributed_validation_requires_exactly_two_corrective_g7e_nodes(tmp_path, config, foundation):
    config_path, foundation_path = write_inputs(tmp_path, config, foundation)
    inputs = repro_aws_launch.load_static_inputs(
        config_path,
        foundation_path,
        subnet_id=None,
        category="corrective_run",
        workload="distributed_validation",
        instance_type="g7e.2xlarge",
        instance_count=2,
    )
    assert inputs.workload == "distributed_validation"
    assert inputs.security_group_id == "sg-distributed"
    assert inputs.security_group_contract == "self_referencing_tcp_all_ports"
    plan = make_plan(
        tmp_path,
        inputs,
        category="corrective_run",
        workload="distributed_validation",
        instance_type="g7e.2xlarge",
        instance_count=2,
        max_runtime_hours=1,
        label="two-node-ddp-smoke",
        command="python scripts/repro_ddp_smoke.py",
    )
    arguments = repro_aws_launch.build_run_instances_arguments(plan, "reservation-ddp")
    assert arguments[arguments.index("--count") + 1] == "2"
    interfaces = json.loads(arguments[arguments.index("--network-interfaces") + 1])
    assert interfaces[0]["Groups"] == ["sg-distributed"]

    for count in (1, 3):
        with pytest.raises(repro_aws_launch.LaunchError, match="exactly 2 for distributed_validation"):
            repro_aws_launch.load_static_inputs(
                config_path,
                foundation_path,
                subnet_id=None,
                category="corrective_run",
                workload="distributed_validation",
                instance_type="g7e.2xlarge",
                instance_count=count,
            )
    with pytest.raises(repro_aws_launch.LaunchError, match="must use the corrective_run"):
        repro_aws_launch.load_static_inputs(
            config_path,
            foundation_path,
            subnet_id=None,
            category="distributed_validation",
            instance_type="g7e.2xlarge",
            instance_count=2,
        )


def test_distributed_validation_requires_pinned_self_only_contract(tmp_path, config, foundation):
    for missing_or_wrong in (None, {}, {"id": "sg-distributed", "ingress_rule_count": 0}):
        candidate = json.loads(json.dumps(foundation))
        if missing_or_wrong is None:
            candidate["network"].pop("distributed_security_group")
        else:
            candidate["network"]["distributed_security_group"] = missing_or_wrong
        config_path, foundation_path = write_inputs(tmp_path, config, candidate)
        with pytest.raises(repro_aws_launch.LaunchError, match="self-referencing TCP all-ports"):
            repro_aws_launch.load_static_inputs(
                config_path,
                foundation_path,
                subnet_id=None,
                category="corrective_run",
                workload="distributed_validation",
                instance_type="g7e.2xlarge",
                instance_count=2,
            )


def test_shallow_training_admits_one_guarded_g7e48_and_rejects_single_gpu_fallback(tmp_path, config, foundation):
    config_path, foundation_path = write_inputs(tmp_path, config, foundation)

    primary = repro_aws_launch.load_static_inputs(
        config_path,
        foundation_path,
        subnet_id=None,
        category="shallow_training",
        instance_type="g7e.12xlarge",
        instance_count=1,
    )
    assert primary.availability_zone == "us-east-2a"

    eight_gpu = repro_aws_launch.load_static_inputs(
        config_path,
        foundation_path,
        subnet_id=None,
        category="shallow_training",
        instance_type="g7e.48xlarge",
        instance_count=1,
    )
    assert eight_gpu.workload == "shallow_training"

    with pytest.raises(repro_aws_launch.LaunchError, match="instance count must be exactly 1"):
        repro_aws_launch.load_static_inputs(
            config_path,
            foundation_path,
            subnet_id=None,
            category="shallow_training",
            instance_type="g7e.48xlarge",
            instance_count=2,
        )

    with pytest.raises(repro_aws_launch.LaunchError, match="not approved for category shallow_training"):
        repro_aws_launch.load_static_inputs(
            config_path,
            foundation_path,
            subnet_id=None,
            category="shallow_training",
            instance_type="g7e.4xlarge",
            instance_count=1,
        )

    with pytest.raises(repro_aws_launch.LaunchError, match="not approved for category corrective_run"):
        repro_aws_launch.load_static_inputs(
            config_path,
            foundation_path,
            subnet_id=None,
            category="corrective_run",
            workload="shallow_training",
            instance_type="g7e.48xlarge",
            instance_count=1,
        )


def test_corrective_budget_allows_only_declared_workload_fallbacks(tmp_path, config, foundation):
    config_path, foundation_path = write_inputs(tmp_path, config, foundation)

    with pytest.raises(repro_aws_launch.LaunchError, match="explicit underlying --workload"):
        repro_aws_launch.load_static_inputs(
            config_path,
            foundation_path,
            subnet_id=None,
            category="corrective_run",
            instance_type="g7e.4xlarge",
            instance_count=1,
        )

    with pytest.raises(repro_aws_launch.LaunchError, match="not approved for workload shallow_training"):
        repro_aws_launch.load_static_inputs(
            config_path,
            foundation_path,
            subnet_id=None,
            category="corrective_run",
            workload="shallow_training",
            instance_type="g7e.2xlarge",
            instance_count=1,
        )

    with pytest.raises(repro_aws_launch.LaunchError, match="not approved for workload evaluation"):
        repro_aws_launch.load_static_inputs(
            config_path,
            foundation_path,
            subnet_id=None,
            category="corrective_run",
            workload="evaluation",
            instance_type="g7e.4xlarge",
            instance_count=1,
        )

    shallow_retry = repro_aws_launch.load_static_inputs(
        config_path,
        foundation_path,
        subnet_id=None,
        category="corrective_run",
        workload="shallow_training",
        instance_type="g7e.12xlarge",
        instance_count=1,
    )
    assert shallow_retry.workload == "shallow_training"

    single_gpu_shallow_retry = repro_aws_launch.load_static_inputs(
        config_path,
        foundation_path,
        subnet_id=None,
        category="corrective_run",
        workload="shallow_training",
        instance_type="g7e.4xlarge",
        instance_count=1,
    )
    assert single_gpu_shallow_retry.workload == "shallow_training"

    export_retry = repro_aws_launch.load_static_inputs(
        config_path,
        foundation_path,
        subnet_id=None,
        category="corrective_run",
        workload="export_compile_quantize",
        instance_type="g7e.4xlarge",
        instance_count=1,
    )
    assert export_retry.workload == "export_compile_quantize"
    retry_plan = make_plan(
        tmp_path,
        export_retry,
        category="corrective_run",
        workload="export_compile_quantize",
        retain_after_command=True,
    )
    assert (retry_plan.category, retry_plan.workload) == ("corrective_run", "export_compile_quantize")
    assert retry_plan.retain_after_command is True


def test_corrective_evaluation_uses_workload_pinned_ami(tmp_path, config, foundation):
    inputs = make_inputs(
        tmp_path,
        config,
        foundation,
        category="corrective_run",
        workload="evaluation",
        instance_type="g6e.4xlarge",
    )
    assert inputs.workload == "evaluation"
    assert inputs.ami_id == EVALUATION_AMI["id"]


def test_noncorrective_category_cannot_impersonate_another_workload(tmp_path, config, foundation):
    config_path, foundation_path = write_inputs(tmp_path, config, foundation)
    with pytest.raises(repro_aws_launch.LaunchError, match="cannot declare workload"):
        repro_aws_launch.load_static_inputs(
            config_path,
            foundation_path,
            subnet_id=None,
            category="export_compile_quantize",
            workload="shallow_training",
            instance_type="g7e.4xlarge",
            instance_count=1,
        )


def test_category_deterministically_selects_pinned_evaluation_ami(tmp_path, config, foundation):
    base_inputs = make_inputs(tmp_path, config, foundation)
    assert base_inputs.ami_id == BASE_AMI["id"]

    evaluation_inputs = make_inputs(
        tmp_path,
        config,
        foundation,
        category="evaluation",
        instance_type="g6e.4xlarge",
    )
    assert evaluation_inputs.ami_id == EVALUATION_AMI["id"]
    assert evaluation_inputs.ami_name == EVALUATION_AMI["name"]
    assert evaluation_inputs.ami_owner == "amazon"
    assert evaluation_inputs.ami_architecture == "x86_64"

    plan = make_plan(
        tmp_path,
        evaluation_inputs,
        category="evaluation",
        instance_type="g6e.4xlarge",
        label="robolab-camera-smoke",
        command="bash repro/robolab-smoke-worker.sh",
    )
    assert plan.ami_id == EVALUATION_AMI["id"]
    assert repro_aws_launch.public_plan(plan)["ami_id"] == EVALUATION_AMI["id"]
    arguments = repro_aws_launch.build_run_instances_arguments(plan, "reservation-eval")
    assert arguments[arguments.index("--image-id") + 1] == EVALUATION_AMI["id"]


def test_pinned_ami_must_match_source_config(tmp_path, config, foundation):
    foundation["launch_amis"]["evaluation"]["name"] = "wrong image"
    with pytest.raises(repro_aws_launch.LaunchError, match="config and foundation evaluation AMI name differ"):
        make_inputs(
            tmp_path,
            config,
            foundation,
            category="evaluation",
            instance_type="g6e.4xlarge",
        )


def test_cli_has_no_arbitrary_ami_override():
    with pytest.raises(SystemExit):
        repro_aws_launch.parse_args(
            [
                "--category",
                "evaluation",
                "--workload",
                "evaluation",
                "--instance-type",
                "g6e.4xlarge",
                "--hours",
                "1",
                "--label",
                "robolab-camera-smoke",
                "--ami-id",
                EVALUATION_AMI["id"],
            ]
        )


def test_plan_reserves_boot_margin_and_worker_must_have_command(tmp_path, config, foundation):
    inputs = make_inputs(tmp_path, config, foundation)
    plan = make_plan(tmp_path, inputs)
    assert plan.reserved_hours == 2.25
    assert plan.projected_usd == pytest.approx(3.99816 * 2.25)
    assert plan.non_compute_reserved_usd == 0
    assert plan.shutdown_behavior == "terminate"
    assert plan.root_volume_gib == 256
    assert plan.deadline_utc == "2026-08-03T22:00:00+00:00"
    assert repro_aws_launch.public_plan(plan)["authoritative_cost_ledger"] == (
        "s3://pi05-repro-752160877725-us-east-2/control/cost-ledger.json"
    )
    assert repro_aws_launch.public_plan(plan)["workload"] == "export_compile_quantize"

    with pytest.raises(repro_aws_launch.LaunchError, match="require a command"):
        make_plan(tmp_path, inputs, command="")

    shallow_inputs = make_inputs(
        tmp_path,
        config,
        foundation,
        category="shallow_training",
        workload="shallow_training",
        instance_type="g7e.12xlarge",
    )
    with pytest.raises(repro_aws_launch.LaunchError, match="allowed only for export/compile"):
        make_plan(
            tmp_path,
            shallow_inputs,
            category="shallow_training",
            workload="shallow_training",
            instance_type="g7e.12xlarge",
            retain_after_command=True,
        )


def test_g7e48_shallow_plan_uses_live_ohio_on_demand_rate(tmp_path, config, foundation):
    inputs = make_inputs(
        tmp_path,
        config,
        foundation,
        category="shallow_training",
        workload="shallow_training",
        instance_type="g7e.48xlarge",
    )
    plan = make_plan(
        tmp_path,
        inputs,
        category="shallow_training",
        workload="shallow_training",
        instance_type="g7e.48xlarge",
        max_runtime_hours=12,
    )

    assert plan.instance_count == 1
    assert plan.reserved_hours == 12.25
    assert plan.projected_usd == pytest.approx(406.01792)
    assert plan.shutdown_behavior == "terminate"


def test_run_request_is_on_demand_hardened_and_tagged(tmp_path, config, foundation):
    inputs = make_inputs(tmp_path, config, foundation)
    plan = make_plan(tmp_path, inputs)
    arguments = repro_aws_launch.build_run_instances_arguments(plan, "reservation-1")

    assert arguments[:2] == ["ec2", "run-instances"]
    assert arguments[arguments.index("--count") + 1] == "1"
    assert "--min-count" not in arguments
    assert "--max-count" not in arguments
    assert "--instance-market-options" not in arguments
    assert arguments[arguments.index("--capacity-reservation-specification") + 1] == (
        "CapacityReservationPreference=none"
    )
    assert arguments[arguments.index("--placement") + 1] == "Tenancy=default"
    assert "HttpTokens=required" in arguments[arguments.index("--metadata-options") + 1]
    assert "HttpPutResponseHopLimit=1" in arguments[arguments.index("--metadata-options") + 1]

    interfaces = json.loads(arguments[arguments.index("--network-interfaces") + 1])
    assert interfaces == [
        {
            "AssociatePublicIpAddress": True,
            "DeleteOnTermination": True,
            "DeviceIndex": 0,
            "Groups": ["sg-test"],
            "SubnetId": "subnet-a",
        }
    ]
    devices = json.loads(arguments[arguments.index("--block-device-mappings") + 1])
    assert devices[0]["Ebs"] == {
        "DeleteOnTermination": True,
        "Encrypted": True,
        "VolumeSize": 256,
        "VolumeType": "gp3",
    }
    tag_specs = json.loads(arguments[arguments.index("--tag-specifications") + 1])
    assert {item["ResourceType"] for item in tag_specs} == {"instance", "volume"}
    for specification in tag_specs:
        tags = {tag["Key"]: tag["Value"] for tag in specification["Tags"]}
        assert tags["Project"] == "pi05-aws-repro"
        assert tags["Workload"] == "export_compile_quantize"
        assert tags["CommandSha256"] == plan.command_sha256
    user_data = arguments[arguments.index("--user-data") + 1]
    subprocess.run(["bash", "-n"], input=user_data, text=True, check=True)
    assert "pi05-hard-deadline.timer" in user_data
    assert "OnCalendar=2026-08-03 22:00:00 UTC" in user_data
    assert "amazon-ssm-agent.service" in user_data
    assert "systemctl mask --now apt-daily.timer apt-daily-upgrade.timer" in user_data
    assert "systemctl mask apt-daily.service apt-daily-upgrade.service" in user_data
    assert "systemctl reset-failed apt-daily.timer apt-daily-upgrade.timer" in user_data
    assert "while systemctl is-active --quiet apt-daily.service" in user_data
    assert "After=network-online.target dlami-nvme.service docker.service" in user_data
    assert "KillSignal=SIGTERM" in user_data
    assert (
        'ExecStopPost=/bin/sh -c \'if [ "$SERVICE_RESULT" = success ]; then '
        "/usr/bin/systemctl --no-block poweroff; fi'"
    ) in user_data
    metadata_line = next(line for line in user_data.splitlines() if line.endswith("> /opt/pi05/launch-metadata.json"))
    metadata = json.loads(base64.b64decode(metadata_line.split("'")[3]))
    assert metadata["reservation_id"] == "reservation-1"
    assert metadata["workload"] == "export_compile_quantize"
    assert metadata["purchase_option"] == "On-Demand"
    assert metadata["projected_compute_usd"] == plan.projected_usd
    assert metadata["reserved_hours"] == plan.reserved_hours
    assert "chmod 0600 /opt/pi05/launch-metadata.json" in user_data


def test_empty_workbench_preserves_manual_lifecycle(tmp_path, config, foundation):
    inputs = make_inputs(
        tmp_path,
        config,
        foundation,
        category="workbench_setup",
        instance_type="g6e.4xlarge",
    )
    plan = make_plan(
        tmp_path,
        inputs,
        category="workbench_setup",
        instance_type="g6e.4xlarge",
        label="manual-workbench",
        command="",
    )
    user_data = repro_aws_launch.build_user_data(plan, "manual-workbench-preview")
    assert "cat > /etc/systemd/system/pi05-job.service" not in user_data
    assert "ExecStopPost" not in user_data
    assert "apt-daily.timer" not in user_data
    assert plan.shutdown_behavior == "stop"


def test_export_session_can_retain_instance_until_absolute_termination_deadline(tmp_path, config, foundation):
    inputs = make_inputs(tmp_path, config, foundation)
    plan = make_plan(tmp_path, inputs, retain_after_command=True)
    user_data = repro_aws_launch.build_user_data(plan, "retained-export-preview")

    assert plan.retain_after_command is True
    assert plan.shutdown_behavior == "terminate"
    assert "cat > /etc/systemd/system/pi05-job.service" in user_data
    assert "ExecStopPost=" not in user_data
    assert "OnCalendar=2026-08-03 22:00:00 UTC" in user_data
    metadata_line = next(line for line in user_data.splitlines() if line.endswith("> /opt/pi05/launch-metadata.json"))
    metadata = json.loads(base64.b64decode(metadata_line.split("'")[3]))
    assert metadata["retain_after_command"] is True


def test_corrective_single_gpu_shallow_can_retain_instance_for_same_node_repair(tmp_path, config, foundation):
    inputs = make_inputs(
        tmp_path,
        config,
        foundation,
        category="corrective_run",
        workload="shallow_training",
        instance_type="g7e.4xlarge",
    )
    plan = make_plan(
        tmp_path,
        inputs,
        category="corrective_run",
        workload="shallow_training",
        instance_type="g7e.4xlarge",
        retain_after_command=True,
    )
    user_data = repro_aws_launch.build_user_data(plan, "retained-corrective-shallow-preview")

    assert plan.retain_after_command is True
    assert plan.shutdown_behavior == "terminate"
    assert "ExecStopPost=" not in user_data
    assert "OnCalendar=2026-08-03 22:00:00 UTC" in user_data


def test_ordinary_or_two_gpu_shallow_cannot_retain_instance(tmp_path, config, foundation):
    inputs = make_inputs(
        tmp_path,
        config,
        foundation,
        category="corrective_run",
        workload="shallow_training",
        instance_type="g7e.12xlarge",
    )

    with pytest.raises(repro_aws_launch.LaunchError, match="one-GPU corrective Shallow fallback"):
        make_plan(
            tmp_path,
            inputs,
            category="corrective_run",
            workload="shallow_training",
            instance_type="g7e.12xlarge",
            retain_after_command=True,
        )


class LedgerAwsMixin:
    def _init_ledger(self):
        self.remote_ledger_bytes = None
        self.remote_etag = None
        self.remote_version_id = None
        self.remote_version = 0
        self.remote_has_history = False

    def seed_remote_ledger(self, ledger):
        self.remote_ledger_bytes = (json.dumps(ledger, indent=2, sort_keys=True) + "\n").encode()
        self.remote_version += 1
        self.remote_etag = f'"etag-{self.remote_version}"'
        self.remote_version_id = f"version-{self.remote_version}"
        self.remote_has_history = True

    def _ledger_json(self, arguments):
        assert arguments[arguments.index("--expected-bucket-owner") + 1] == "752160877725"
        operation = arguments[:2]
        if operation == ["s3api", "list-object-versions"]:
            assert arguments[arguments.index("--prefix") + 1] == repro_aws_launch.S3_LEDGER_KEY
            if not self.remote_has_history:
                return {"Versions": [], "DeleteMarkers": []}
            if self.remote_ledger_bytes is None:
                return {"Versions": [], "DeleteMarkers": [{"Key": repro_aws_launch.S3_LEDGER_KEY}]}
            return {"Versions": [{"Key": repro_aws_launch.S3_LEDGER_KEY}], "DeleteMarkers": []}
        if operation == ["s3api", "head-object"]:
            if self.remote_ledger_bytes is None:
                raise repro_aws_launch.AwsCliError("s3api head-object", 255, "(404) Not Found")
            return {
                "ETag": self.remote_etag,
                "VersionId": self.remote_version_id,
                "ContentLength": len(self.remote_ledger_bytes),
            }
        if operation == ["s3api", "get-object"]:
            assert arguments[arguments.index("--version-id") + 1] == self.remote_version_id
            destination = arguments[-1]
            with open(destination, "wb") as stream:
                stream.write(self.remote_ledger_bytes)
            return {"ETag": self.remote_etag, "VersionId": self.remote_version_id}
        assert operation == ["s3api", "put-object"]
        if "--if-none-match" in arguments and self.remote_ledger_bytes is not None:
            raise repro_aws_launch.AwsCliError("s3api put-object", 255, "(412) PreconditionFailed")
        if "--if-match" in arguments:
            expected = arguments[arguments.index("--if-match") + 1]
            if expected != self.remote_etag:
                raise repro_aws_launch.AwsCliError("s3api put-object", 255, "(412) PreconditionFailed")
        body = arguments[arguments.index("--body") + 1]
        with open(body, "rb") as stream:
            self.remote_ledger_bytes = stream.read()
        self.remote_version += 1
        self.remote_etag = f'"etag-{self.remote_version}"'
        self.remote_version_id = f"version-{self.remote_version}"
        self.remote_has_history = True
        return {"ETag": self.remote_etag, "VersionId": self.remote_version_id}

    @property
    def remote_ledger(self):
        return json.loads(self.remote_ledger_bytes)


class SuccessfulAws(LedgerAwsMixin):
    def __init__(self, ledger_path):
        self.ledger_path = ledger_path
        self.calls = []
        self._init_ledger()

    def json(self, arguments):
        self.calls.append(arguments)
        if arguments[0] == "s3api":
            return self._ledger_json(arguments)
        if arguments[:2] == ["ec2", "describe-instances"]:
            return {"Reservations": []}
        if arguments[:2] == ["ec2", "run-instances"]:
            ledger = json.loads(self.ledger_path.read_text())
            assert ledger["entries"][0]["state"] == "launching"
            return {"Instances": [{"InstanceId": "i-123"}]}
        assert arguments[:2] == ["scheduler", "create-schedule"]
        return {"ScheduleArn": "arn:aws:scheduler:us-east-2:752160877725:schedule/default/test"}


class FailedAws(LedgerAwsMixin):
    def __init__(self, *, recovery_fails=False, recovered=False):
        self.recovery_fails = recovery_fails
        self.recovered = recovered
        self.calls = []
        self._init_ledger()

    def json(self, arguments):
        self.calls.append(arguments)
        if arguments[0] == "s3api":
            return self._ledger_json(arguments)
        if arguments[:2] == ["ec2", "run-instances"]:
            raise repro_aws_launch.AwsCliError("ec2 run-instances", 255, "InsufficientInstanceCapacity")
        assert arguments[:2] == ["ec2", "describe-instances"]
        is_recovery = any(str(value).startswith("Name=client-token,Values=") for value in arguments)
        if not is_recovery:
            return {"Reservations": []}
        if self.recovery_fails:
            raise repro_aws_launch.AwsCliError("ec2 describe-instances", 255, "network timeout")
        if self.recovered:
            return {"Reservations": [{"Instances": [{"InstanceId": "i-recovered"}]}]}
        return {"Reservations": []}


def test_execute_reserves_before_launch_and_records_instance(tmp_path, config, foundation):
    inputs = make_inputs(tmp_path, config, foundation)
    plan = make_plan(tmp_path, inputs)
    ledger_path = tmp_path / "ledger.json"
    aws = SuccessfulAws(ledger_path)
    result = repro_aws_launch.execute_launch(aws, inputs, plan, ledger_path)
    assert result["instance_ids"] == ["i-123"]
    schedule_call = next(call for call in aws.calls if call[:2] == ["scheduler", "create-schedule"])
    assert schedule_call[schedule_call.index("--action-after-completion") + 1] == "DELETE"
    assert schedule_call[schedule_call.index("--schedule-expression") + 1] == "at(2026-08-03T22:00:00)"
    target = json.loads(schedule_call[schedule_call.index("--target") + 1])
    assert target["Arn"] == "arn:aws:scheduler:::aws-sdk:ec2:terminateInstances"
    assert json.loads(target["Input"]) == {"InstanceIds": ["i-123"]}
    entry = json.loads(ledger_path.read_text())["entries"][0]
    assert entry["state"] == "launched"
    assert entry["workload"] == "export_compile_quantize"
    assert entry["schedule_arn"] == result["schedule_arn"]
    assert entry["usd"] == pytest.approx(plan.projected_usd)
    ledger_puts = [call for call in aws.calls if call[:2] == ["s3api", "put-object"]]
    assert "--if-none-match" in ledger_puts[0]
    assert all("--if-match" in call for call in ledger_puts[1:])
    assert aws.remote_ledger == json.loads(ledger_path.read_text())


def test_remote_ledger_is_authoritative_for_the_budget_gate(tmp_path, config, foundation):
    inputs = make_inputs(tmp_path, config, foundation)
    plan = make_plan(tmp_path, inputs)
    ledger_path = tmp_path / "ledger.json"
    aws = SuccessfulAws(ledger_path)
    aws.seed_remote_ledger(
        {
            "schema_version": 1,
            "entries": [
                {
                    "id": "prior-remote-run",
                    "state": "launched",
                    "category": "export_compile_quantize",
                    "usd": 195,
                }
            ],
        }
    )
    with pytest.raises(repro_aws_launch.repro_cost_guard.BudgetError, match="category cap exceeded"):
        repro_aws_launch.execute_launch(aws, inputs, plan, ledger_path)
    assert not any(call[:2] == ["ec2", "run-instances"] for call in aws.calls)
    assert json.loads(ledger_path.read_text()) == aws.remote_ledger


def test_deleted_remote_ledger_cannot_reset_committed_spend(tmp_path, config, foundation):
    inputs = make_inputs(tmp_path, config, foundation)
    plan = make_plan(tmp_path, inputs)
    ledger_path = tmp_path / "ledger.json"
    aws = SuccessfulAws(ledger_path)
    aws.seed_remote_ledger(
        {
            "schema_version": 1,
            "entries": [
                {
                    "id": "prior-run",
                    "state": "launched",
                    "category": "evaluation",
                    "usd": 1,
                }
            ],
        }
    )
    aws.remote_ledger_bytes = None  # Simulate a current delete marker with retained history.
    store = repro_aws_launch.S3CostLedger(aws, inputs.artifact_bucket, ledger_path)
    with pytest.raises(repro_aws_launch.LaunchError, match="prior versions or delete markers"):
        store.reserve(inputs.config, plan)
    assert not any(call[:2] == ["s3api", "put-object"] for call in aws.calls)


def test_malformed_remote_ledger_cost_fails_before_launch(tmp_path, config, foundation):
    inputs = make_inputs(tmp_path, config, foundation)
    plan = make_plan(tmp_path, inputs)
    ledger_path = tmp_path / "ledger.json"
    aws = SuccessfulAws(ledger_path)
    aws.seed_remote_ledger(
        {
            "schema_version": 1,
            "entries": [
                {
                    "id": "poisoned-run",
                    "state": "launched",
                    "category": "evaluation",
                    "usd": float("nan"),
                }
            ],
        }
    )
    with pytest.raises(repro_aws_launch.LaunchError, match="invalid budget data"):
        repro_aws_launch.execute_launch(aws, inputs, plan, ledger_path)
    assert not any(call[:2] == ["ec2", "run-instances"] for call in aws.calls)


class ConflictOnceAws(SuccessfulAws):
    def __init__(self, ledger_path):
        super().__init__(ledger_path)
        self.conflict_once = True

    def json(self, arguments):
        if arguments[:2] == ["s3api", "put-object"] and self.conflict_once:
            self.calls.append(arguments)
            self.conflict_once = False
            self.seed_remote_ledger(
                {
                    "schema_version": 1,
                    "entries": [
                        {
                            "id": "concurrent-run",
                            "state": "launched",
                            "category": "evaluation",
                            "usd": 1,
                        }
                    ],
                }
            )
            return self._ledger_json(arguments)
        return super().json(arguments)


def test_remote_ledger_cas_retries_without_losing_concurrent_reservation(tmp_path, config, foundation):
    inputs = make_inputs(tmp_path, config, foundation)
    plan = make_plan(tmp_path, inputs)
    ledger_path = tmp_path / "ledger.json"
    aws = ConflictOnceAws(ledger_path)
    store = repro_aws_launch.S3CostLedger(aws, inputs.artifact_bucket, ledger_path)
    reservation_id = store.reserve(inputs.config, plan)
    assert {entry["id"] for entry in aws.remote_ledger["entries"]} == {"concurrent-run", reservation_id}
    puts = [call for call in aws.calls if call[:2] == ["s3api", "put-object"]]
    assert "--if-none-match" in puts[0]
    assert "--if-match" in puts[1]


def test_first_remote_mutation_migrates_pre_s3_local_reservations(tmp_path, config, foundation):
    inputs = make_inputs(tmp_path, config, foundation)
    plan = make_plan(tmp_path, inputs)
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "id": "legacy-local-run",
                        "state": "launched",
                        "category": "workbench_setup",
                        "usd": 18.7765,
                    }
                ],
            }
        )
    )
    aws = SuccessfulAws(ledger_path)
    aws.seed_remote_ledger(
        {
            "schema_version": 1,
            "entries": [
                {
                    "id": "remote-run",
                    "state": "launched",
                    "category": "evaluation",
                    "usd": 1,
                }
            ],
        }
    )
    store = repro_aws_launch.S3CostLedger(aws, inputs.artifact_bucket, ledger_path)
    reservation_id = store.reserve(inputs.config, plan)
    assert {entry["id"] for entry in aws.remote_ledger["entries"]} == {
        "legacy-local-run",
        "remote-run",
        reservation_id,
    }
    assert json.loads(ledger_path.read_text()) == aws.remote_ledger


def test_execute_requires_scheduler_role_before_reserving_spend(tmp_path, config, foundation):
    inputs = make_inputs(tmp_path, config, foundation)
    plan = make_plan(tmp_path, inputs, scheduler_role_arn=None)
    ledger_path = tmp_path / "ledger.json"
    with pytest.raises(repro_aws_launch.LaunchError, match="scheduler-role-arn"):
        repro_aws_launch.execute_launch(SuccessfulAws(ledger_path), inputs, plan, ledger_path)
    assert not ledger_path.exists()


class ScheduleFailureAws(LedgerAwsMixin):
    def __init__(self):
        self.calls = []
        self._init_ledger()

    def json(self, arguments):
        self.calls.append(arguments)
        if arguments[0] == "s3api":
            return self._ledger_json(arguments)
        if arguments[:2] == ["ec2", "describe-instances"]:
            return {"Reservations": []}
        if arguments[:2] == ["ec2", "run-instances"]:
            return {"Instances": [{"InstanceId": "i-unscheduled"}]}
        if arguments[:2] == ["scheduler", "create-schedule"]:
            raise repro_aws_launch.AwsCliError("scheduler create-schedule", 255, "AccessDenied")
        assert arguments[:2] == ["ec2", "terminate-instances"]
        return {"TerminatingInstances": [{"InstanceId": "i-unscheduled"}]}


def test_schedule_failure_immediately_terminates_new_instance(tmp_path, config, foundation):
    inputs = make_inputs(tmp_path, config, foundation)
    plan = make_plan(tmp_path, inputs)
    ledger_path = tmp_path / "ledger.json"
    aws = ScheduleFailureAws()
    with pytest.raises(repro_aws_launch.LaunchError, match="immediate termination"):
        repro_aws_launch.execute_launch(aws, inputs, plan, ledger_path)
    assert ["ec2", "terminate-instances", "--instance-ids", "i-unscheduled"] in aws.calls
    entry = json.loads(ledger_path.read_text())["entries"][0]
    assert entry["state"] == "schedule_failed_cleanup_requested"


class ExistingWorkbenchAws:
    def json(self, arguments):
        assert arguments[:2] == ["ec2", "describe-instances"]
        return {"Reservations": [{"Instances": [{"InstanceId": "i-existing"}]}]}


def test_duplicate_active_workbench_is_rejected_before_reservation(tmp_path, config, foundation):
    inputs = make_inputs(
        tmp_path,
        config,
        foundation,
        category="workbench_setup",
        instance_type="g6e.4xlarge",
    )
    plan = make_plan(
        tmp_path,
        inputs,
        category="workbench_setup",
        instance_type="g6e.4xlarge",
        label="manual-workbench",
        command="",
    )
    ledger_path = tmp_path / "ledger.json"
    with pytest.raises(repro_aws_launch.LaunchError, match="active workbench already exists"):
        repro_aws_launch.execute_launch(ExistingWorkbenchAws(), inputs, plan, ledger_path)
    assert not ledger_path.exists()


class ActiveDuplicateJobAws(SuccessfulAws):
    def __init__(self, ledger_path, label):
        super().__init__(ledger_path)
        self.label = label

    def json(self, arguments):
        if arguments[:2] == ["ec2", "describe-instances"]:
            self.calls.append(arguments)
            return {
                "Reservations": [
                    {
                        "Instances": [
                            {
                                "InstanceId": "i-duplicate",
                                "Tags": [{"Key": "Name", "Value": self.label}],
                            }
                        ]
                    }
                ]
            }
        return super().json(arguments)


def test_duplicate_active_job_is_rejected_before_reservation(tmp_path, config, foundation):
    inputs = make_inputs(tmp_path, config, foundation)
    plan = make_plan(tmp_path, inputs)
    ledger_path = tmp_path / "ledger.json"
    aws = ActiveDuplicateJobAws(ledger_path, plan.label)
    with pytest.raises(repro_aws_launch.LaunchError, match="active job already uses"):
        repro_aws_launch.execute_launch(aws, inputs, plan, ledger_path)
    assert not ledger_path.exists()
    assert not any(call[:2] == ["ec2", "run-instances"] for call in aws.calls)


def test_duplicate_ledger_label_is_rejected_across_controllers(tmp_path, config, foundation):
    inputs = make_inputs(tmp_path, config, foundation)
    plan = make_plan(tmp_path, inputs)
    ledger_path = tmp_path / "ledger.json"
    aws = SuccessfulAws(ledger_path)
    aws.seed_remote_ledger(
        {
            "schema_version": 1,
            "entries": [
                {
                    "id": "other-controller-run",
                    "state": "launching",
                    "label": plan.label,
                    "category": plan.category,
                    "usd": 1,
                }
            ],
        }
    )
    with pytest.raises(repro_aws_launch.LaunchError, match="unresolved reservation already uses"):
        repro_aws_launch.execute_launch(aws, inputs, plan, ledger_path)
    assert not any(call[:2] == ["s3api", "put-object"] for call in aws.calls)
    assert not any(call[:2] == ["ec2", "run-instances"] for call in aws.calls)


def test_empty_immediate_recovery_query_keeps_reservation_committed(tmp_path, config, foundation):
    inputs = make_inputs(tmp_path, config, foundation)
    plan = make_plan(tmp_path, inputs)
    ledger_path = tmp_path / "ledger.json"
    with pytest.raises(repro_aws_launch.LaunchError, match="first query was empty"):
        repro_aws_launch.execute_launch(FailedAws(), inputs, plan, ledger_path)
    entry = json.loads(ledger_path.read_text())["entries"][0]
    assert entry["state"] == "launch_unknown"
    assert entry["recovery_observation"] == "no instances visible on the first immediate query"


def test_ambiguous_launch_failure_keeps_budget_committed(tmp_path, config, foundation):
    inputs = make_inputs(tmp_path, config, foundation)
    plan = make_plan(tmp_path, inputs)
    ledger_path = tmp_path / "ledger.json"
    with pytest.raises(repro_aws_launch.LaunchError, match="remains committed"):
        repro_aws_launch.execute_launch(FailedAws(recovery_fails=True), inputs, plan, ledger_path)
    entry = json.loads(ledger_path.read_text())["entries"][0]
    assert entry["state"] == "launch_unknown"


class MalformedRunResponseAws(SuccessfulAws):
    def json(self, arguments):
        if arguments[:2] == ["ec2", "describe-instances"]:
            self.calls.append(arguments)
            is_recovery = any(str(value).startswith("Name=client-token,Values=") for value in arguments)
            if is_recovery:
                return {"Reservations": [{"Instances": [{"InstanceId": "i-recovered"}]}]}
            return {"Reservations": []}
        if arguments[:2] == ["ec2", "run-instances"]:
            self.calls.append(arguments)
            raise repro_aws_launch.LaunchError("AWS CLI returned invalid JSON for ec2 run-instances")
        return super().json(arguments)


def test_malformed_run_response_is_reconciled_by_client_token(tmp_path, config, foundation):
    inputs = make_inputs(tmp_path, config, foundation)
    plan = make_plan(tmp_path, inputs)
    ledger_path = tmp_path / "ledger.json"
    aws = MalformedRunResponseAws(ledger_path)
    result = repro_aws_launch.execute_launch(aws, inputs, plan, ledger_path)
    assert result["instance_ids"] == ["i-recovered"]
    assert result["recovered"] is True
    entry = json.loads(ledger_path.read_text())["entries"][0]
    assert entry["state"] == "launched_recovered"
    assert "invalid JSON" in entry["launch_error"]


class InvalidInstancesResponseAws(MalformedRunResponseAws):
    def json(self, arguments):
        if arguments[:2] == ["ec2", "run-instances"]:
            self.calls.append(arguments)
            return {"Instances": None}
        return super().json(arguments)


def test_invalid_instances_collection_is_reconciled_by_client_token(tmp_path, config, foundation):
    inputs = make_inputs(tmp_path, config, foundation)
    plan = make_plan(tmp_path, inputs)
    ledger_path = tmp_path / "ledger.json"
    aws = InvalidInstancesResponseAws(ledger_path)
    result = repro_aws_launch.execute_launch(aws, inputs, plan, ledger_path)
    assert result["instance_ids"] == ["i-recovered"]
    entry = json.loads(ledger_path.read_text())["entries"][0]
    assert entry["state"] == "launched_recovered"
    assert "invalid Instances collection" in entry["launch_error"]


class PreflightAws:
    def __init__(
        self,
        *,
        account="752160877725",
        ingress=False,
        image_id=BASE_AMI["id"],
        image_name=BASE_AMI["name"],
        image_owner_alias="amazon",
        image_owner_id="898082745236",
        image_state="available",
        image_architecture="x86_64",
        image_platform_details="Linux/UNIX",
        image_virtualization_type="hvm",
        image_root_device_name="/dev/sda1",
        instance_type="g7e.4xlarge",
        security_group_id="sg-test",
        ingress_permissions=None,
    ):
        self.account = account
        self.ingress = ingress
        self.image_id = image_id
        self.image_name = image_name
        self.image_owner_alias = image_owner_alias
        self.image_owner_id = image_owner_id
        self.image_state = image_state
        self.image_architecture = image_architecture
        self.image_platform_details = image_platform_details
        self.image_virtualization_type = image_virtualization_type
        self.image_root_device_name = image_root_device_name
        self.instance_type = instance_type
        self.security_group_id = security_group_id
        self.ingress_permissions = ingress_permissions
        self.calls = []

    def json(self, arguments):
        self.calls.append(arguments)
        operation = tuple(arguments[:2])
        responses = {
            ("sts", "get-caller-identity"): {"Account": self.account},
            ("iam", "get-instance-profile"): {
                "InstanceProfile": {
                    "InstanceProfileName": "profile-test",
                    "Roles": [{"RoleName": "role-test"}],
                }
            },
            ("ec2", "describe-security-groups"): {
                "SecurityGroups": [
                    {
                        "GroupId": self.security_group_id,
                        "VpcId": "vpc-test",
                        "IpPermissions": (
                            self.ingress_permissions
                            if self.ingress_permissions is not None
                            else ([{}] if self.ingress else [])
                        ),
                        "IpPermissionsEgress": [{}],
                    }
                ]
            },
            ("ec2", "describe-subnets"): {
                "Subnets": [
                    {
                        "SubnetId": "subnet-a",
                        "VpcId": "vpc-test",
                        "AvailabilityZone": "us-east-2a",
                        "State": "available",
                    }
                ]
            },
            ("ec2", "describe-images"): {
                "Images": [
                    {
                        "ImageId": self.image_id,
                        "Name": self.image_name,
                        "ImageOwnerAlias": self.image_owner_alias,
                        "OwnerId": self.image_owner_id,
                        "State": self.image_state,
                        "Architecture": self.image_architecture,
                        "PlatformDetails": self.image_platform_details,
                        "VirtualizationType": self.image_virtualization_type,
                        "RootDeviceName": self.image_root_device_name,
                    }
                ]
            },
            ("ec2", "describe-instance-type-offerings"): {
                "InstanceTypeOfferings": [{"InstanceType": self.instance_type, "Location": "us-east-2a"}]
            },
        }
        return responses[operation]


def test_preflight_rejects_wrong_account_and_live_ingress(tmp_path, config, foundation):
    inputs = make_inputs(tmp_path, config, foundation)
    with pytest.raises(repro_aws_launch.LaunchError, match="refusing AWS account"):
        repro_aws_launch.verify_live_environment(PreflightAws(account="000000000000"), inputs, "g7e.4xlarge")
    with pytest.raises(repro_aws_launch.LaunchError, match="inbound rules"):
        repro_aws_launch.verify_live_environment(PreflightAws(ingress=True), inputs, "g7e.4xlarge")
    aws = PreflightAws()
    assert repro_aws_launch.verify_live_environment(aws, inputs, "g7e.4xlarge") == "/dev/sda1"
    image_call = next(call for call in aws.calls if call[:2] == ["ec2", "describe-images"])
    assert image_call == [
        "ec2",
        "describe-images",
        "--owners",
        "amazon",
        "--image-ids",
        BASE_AMI["id"],
    ]


def distributed_self_only_permission():
    return {
        "IpProtocol": "tcp",
        "FromPort": 0,
        "ToPort": 65535,
        "IpRanges": [],
        "Ipv6Ranges": [],
        "PrefixListIds": [],
        "UserIdGroupPairs": [
            {
                "GroupId": "sg-distributed",
                "UserId": "752160877725",
            }
        ],
    }


def test_distributed_preflight_accepts_only_same_group_tcp_ingress(tmp_path, config, foundation):
    inputs = make_inputs(
        tmp_path,
        config,
        foundation,
        category="corrective_run",
        workload="distributed_validation",
        instance_type="g7e.2xlarge",
        instance_count=2,
    )
    aws = PreflightAws(
        instance_type="g7e.2xlarge",
        security_group_id="sg-distributed",
        ingress_permissions=[distributed_self_only_permission()],
    )
    assert repro_aws_launch.verify_live_environment(aws, inputs, "g7e.2xlarge") == "/dev/sda1"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rule: rule.update(IpProtocol="-1"), "TCP ports"),
        (lambda rule: rule.update(FromPort=1), "TCP ports"),
        (lambda rule: rule["IpRanges"].append({"CidrIp": "172.31.0.0/16"}), "non-group source"),
        (
            lambda rule: rule["UserIdGroupPairs"][0].update(GroupId="sg-other"),
            "account-local self reference",
        ),
        (
            lambda rule: rule["UserIdGroupPairs"][0].update(UserId="000000000000"),
            "account-local self reference",
        ),
        (lambda rule: rule.update(UserIdGroupPairs=[]), "exactly one group source"),
    ],
)
def test_distributed_preflight_rejects_broader_or_nonself_ingress(tmp_path, config, foundation, mutate, message):
    inputs = make_inputs(
        tmp_path,
        config,
        foundation,
        category="corrective_run",
        workload="distributed_validation",
        instance_type="g7e.2xlarge",
        instance_count=2,
    )
    rule = distributed_self_only_permission()
    mutate(rule)
    with pytest.raises(repro_aws_launch.LaunchError, match=message):
        repro_aws_launch.verify_live_environment(
            PreflightAws(
                instance_type="g7e.2xlarge",
                security_group_id="sg-distributed",
                ingress_permissions=[rule],
            ),
            inputs,
            "g7e.2xlarge",
        )


def test_evaluation_preflight_verifies_exact_ami_identity_and_owner(tmp_path, config, foundation):
    inputs = make_inputs(
        tmp_path,
        config,
        foundation,
        category="evaluation",
        instance_type="g6e.4xlarge",
    )
    aws = PreflightAws(
        image_id=EVALUATION_AMI["id"],
        image_name=EVALUATION_AMI["name"],
        instance_type="g6e.4xlarge",
    )
    assert repro_aws_launch.verify_live_environment(aws, inputs, "g6e.4xlarge") == "/dev/sda1"

    with pytest.raises(repro_aws_launch.LaunchError, match="id or name"):
        repro_aws_launch.verify_live_environment(
            PreflightAws(
                image_id=EVALUATION_AMI["id"],
                image_name="wrong image name",
                instance_type="g6e.4xlarge",
            ),
            inputs,
            "g6e.4xlarge",
        )
    with pytest.raises(repro_aws_launch.LaunchError, match="owner alias"):
        repro_aws_launch.verify_live_environment(
            PreflightAws(
                image_id=EVALUATION_AMI["id"],
                image_name=EVALUATION_AMI["name"],
                image_owner_alias="self",
                instance_type="g6e.4xlarge",
            ),
            inputs,
            "g6e.4xlarge",
        )
    with pytest.raises(repro_aws_launch.LaunchError, match="owner id"):
        repro_aws_launch.verify_live_environment(
            PreflightAws(
                image_id=EVALUATION_AMI["id"],
                image_name=EVALUATION_AMI["name"],
                image_owner_id="000000000000",
                instance_type="g6e.4xlarge",
            ),
            inputs,
            "g6e.4xlarge",
        )


@pytest.mark.parametrize(
    (
        "image_state",
        "image_architecture",
        "image_platform_details",
        "image_virtualization_type",
        "image_root_device_name",
        "message",
    ),
    [
        ("pending", "x86_64", "Linux/UNIX", "hvm", "/dev/sda1", "not available"),
        ("available", "arm64", "Linux/UNIX", "hvm", "/dev/sda1", "architecture"),
        ("available", "x86_64", "Windows", "hvm", "/dev/sda1", "platform"),
        ("available", "x86_64", "Linux/UNIX", "paravirtual", "/dev/sda1", "virtualization"),
        ("available", "x86_64", "Linux/UNIX", "hvm", "/dev/xvda", "root device"),
    ],
)
def test_evaluation_preflight_rejects_unusable_ami(
    tmp_path,
    config,
    foundation,
    image_state,
    image_architecture,
    image_platform_details,
    image_virtualization_type,
    image_root_device_name,
    message,
):
    inputs = make_inputs(
        tmp_path,
        config,
        foundation,
        category="evaluation",
        instance_type="g6e.4xlarge",
    )
    with pytest.raises(repro_aws_launch.LaunchError, match=message):
        repro_aws_launch.verify_live_environment(
            PreflightAws(
                image_id=EVALUATION_AMI["id"],
                image_name=EVALUATION_AMI["name"],
                image_state=image_state,
                image_architecture=image_architecture,
                image_platform_details=image_platform_details,
                image_virtualization_type=image_virtualization_type,
                image_root_device_name=image_root_device_name,
                instance_type="g6e.4xlarge",
            ),
            inputs,
            "g6e.4xlarge",
        )
