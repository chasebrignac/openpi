from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import subprocess

import pytest

from scripts import repro_robolab_report
from scripts import repro_robolab_worker as robolab_worker
from scripts import repro_worker


def controller_source() -> dict[str, str]:
    commit = "a" * 40
    return {
        "s3_uri": f"s3://{robolab_worker.BUCKET}/source/openpi-{commit}-complete.bundle",
        "version_id": "controller-version",
        "sha256": "b" * 64,
        "commit": commit,
    }


def model_source() -> dict[str, str]:
    return {
        "s3_uri": (f"s3://{robolab_worker.BUCKET}/source/openpi-{robolab_worker.MODEL_SOURCE_COMMIT}-complete.bundle"),
        "version_id": "CN9PJHZ3oHC3hb7lwDTH9p3JEAVQmUhh",
        "sha256": "9be1f91dfec636d1cbb63ad87b166e301b98835b91a6212f73fa5b5350d0f7b5",
        "commit": robolab_worker.MODEL_SOURCE_COMMIT,
    }


def make_spec() -> dict:
    return robolab_worker.make_spec(
        run_id="robolab-base-20260804t120000z-a1",
        source=controller_source(),
        model_source=model_source(),
    )


def test_exact_base_spec_reuses_the_versioned_worker_contract():
    spec = make_spec()

    assert robolab_worker.validate_spec(spec) == spec
    assert spec["model_source"] == model_source()
    assert spec["policy_image"] == robolab_worker.POLICY_IMAGE
    assert spec["evaluator_image"] == robolab_worker.EVALUATOR_IMAGE
    assert spec["artifacts"][1]["payload_objects"][-1] == {
        "path": "model.safetensors",
        "version_id": ".mJzDLYOwUQvdlE9ORGR5T8OEsPqMuKW",
        "sha256": "3212bbd9737caf175ba238193a9e1e3b7b16a4c5d1c4b586ad3d65d58deb5117",
    }
    assert spec["output"]["s3_uri"].endswith(f"runs/{spec['run_id']}/")


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("model_source", "commit"), "c" * 40),
        (("policy_image", "digest"), "sha256:" + "d" * 64),
        (("evaluator_image", "robolab_revision"), "e" * 40),
        (("host", "ami_id"), "ami-00000000000000000"),
        (("host", "driver_version"), "595.71.05"),
        (("artifacts", 1, "revision"), "f" * 64),
        (("evaluation", "num_runs"), 10),
    ],
)
def test_spec_rejects_any_pinned_identity_change(path, replacement):
    spec = make_spec()
    target = spec
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = replacement

    with pytest.raises(repro_worker.WorkerError, match=r"pinned|contract|inputs|evaluation|host|source"):
        robolab_worker.validate_spec(spec)


def test_spec_rejects_legacy_shallow_model_bundle_pin():
    legacy = model_source()
    legacy.update(
        {
            "s3_uri": (f"s3://{robolab_worker.BUCKET}/source/openpi-{robolab_worker.MODEL_SOURCE_COMMIT}.bundle"),
            "version_id": "HY8r1VZTuShbxIAknhQgVyM6pVnWm9uk",
            "sha256": "bb5a5efa2d914de5ac223a9bf251082f7de03fd2c973d19117e92d708fb854be",
        }
    )

    with pytest.raises(repro_worker.WorkerError, match="complete project bundle key"):
        robolab_worker.make_spec(
            run_id="robolab-base-20260804t120000z-a1",
            source=controller_source(),
            model_source=legacy,
        )


def test_spec_rejects_continuation_receipt_instead_of_snapshot_payload():
    parent_run_id = "robolab-base-20260804t110000z-a0"
    with pytest.raises(repro_worker.WorkerError, match="immutable prefix"):
        robolab_worker.make_spec(
            run_id="robolab-base-20260804t120000z-a1",
            source=controller_source(),
            model_source=model_source(),
            continuation={
                "parent_run_id": parent_run_id,
                "snapshot": {
                    "s3_uri": (
                        f"s3://{robolab_worker.BUCKET}/runs/{parent_run_id}/artifacts/"
                        "robolab-partials/snapshot-0020-0123456789abcdef.receipt.json"
                    ),
                    "version_id": "snapshot-version",
                    "sha256": "1" * 64,
                },
            },
        )


def test_commands_are_the_public_protocol_and_fixed_acceptance_count():
    spec = make_spec()
    policy = robolab_worker.policy_server_argv()
    evaluator = robolab_worker.evaluator_argv(spec)
    seal = robolab_worker.seal_argv(spec)

    assert policy == [
        "python",
        "scripts/serve_policy.py",
        "--env",
        "DROID",
        "--port",
        "8000",
        "--seed",
        "7003",
        "policy:checkpoint",
        "--policy.config",
        "pi05_droid_jointpos",
        "--policy.dir",
        "/mnt/openpi/checkpoints/pi05_droid_jointpos_pytorch",
    ]
    assert evaluator[evaluator.index("--num-envs") + 1] == "10"
    assert evaluator[evaluator.index("--num-runs") + 1] == "5"
    assert evaluator[evaluator.index("--remote-host") + 1] == robolab_worker.policy_container_name(spec)
    assert evaluator[evaluator.index("--task") + 1 : evaluator.index("--task-dirs")] == list(robolab_worker.TASKS)
    assert "--disable-subtask" not in evaluator
    assert seal[seal.index("--mode") + 1] == "intermediate"
    assert seal[seal.index("--image-digest") + 1] == robolab_worker.EVALUATOR_IMAGE["uri"]
    assert seal[seal.index("--policy-image-digest") + 1] == robolab_worker.POLICY_IMAGE["uri"]
    assert seal[seal.index("--policy-source-commit") + 1] == spec["model_source"]["commit"]
    assert seal[seal.index("--policy-command-sha256") + 1] == robolab_worker._canonical_sha256(policy)  # noqa: SLF001


def test_two_container_commands_use_exact_images_and_internal_dns_protocol(tmp_path):
    source = tmp_path / "model-source"
    run_root = tmp_path / "run"
    source.mkdir()
    (run_root / "inputs").mkdir(parents=True)
    (run_root / "tmp").mkdir()
    (run_root / "cache").mkdir()
    (run_root / "output" / "artifacts" / "robolab").mkdir(parents=True)

    spec = make_spec()
    policy = robolab_worker.policy_container_command(spec, source, run_root)
    evaluator = robolab_worker.evaluator_container_command(spec, run_root)
    network = robolab_worker.internal_network_name(spec)
    policy_name = robolab_worker.policy_container_name(spec)

    assert policy[policy.index("--network") + 1] == network
    assert policy[policy.index("--network-alias") + 1] == policy_name
    assert policy[policy.index("--user") + 1] == "1000:1000"
    assert robolab_worker.POLICY_IMAGE["uri"] in policy
    assert evaluator[evaluator.index("--network") + 1] == network
    assert evaluator[evaluator.index("--remote-host") + 1] == policy_name
    assert robolab_worker.EVALUATOR_IMAGE["uri"] in evaluator
    assert evaluator[evaluator.index("--entrypoint") + 1] == "/workspace/isaaclab/_isaac_sim/python.sh"
    assert not any("dst=/output" in argument for argument in policy)
    for command in (policy, evaluator):
        assert command[command.index("--network") + 1] != "host"
        assert "--publish" not in command
        assert "--publish-all" not in command
        assert "-p" not in command


def test_policy_dns_name_is_bounded_dns_safe_and_collision_resistant():
    dotted = {"run_id": "a._" * 21 + "a"}
    dashed = {"run_id": "a--" * 21 + "a"}
    dotted_name = robolab_worker.policy_container_name(dotted)
    dashed_name = robolab_worker.policy_container_name(dashed)

    assert len(dotted_name) <= 63
    assert set(dotted_name) <= set("abcdefghijklmnopqrstuvwxyz0123456789-")
    assert dotted_name != dashed_name


def _network_document(spec, *, internal=True, containers=None, labels=None):
    return {
        "Name": robolab_worker.internal_network_name(spec),
        "Id": "a" * 64,
        "Driver": "bridge",
        "Scope": "local",
        "Internal": internal,
        "Attachable": False,
        "Ingress": False,
        "Labels": labels if labels is not None else robolab_worker._expected_network_labels(spec),  # noqa: SLF001
        "Containers": containers if containers is not None else {},
    }


def test_internal_network_creation_is_inspected_and_evidenced():
    spec = make_spec()
    calls = []

    def runner(argv):
        calls.append(list(argv))
        if list(argv[1:3]) == ["network", "ls"]:
            return ""
        if list(argv[1:3]) == ["network", "create"]:
            return "a" * 64
        if list(argv[1:3]) == ["network", "inspect"]:
            return json.dumps(_network_document(spec))
        raise AssertionError(argv)

    evidence = robolab_worker.create_internal_network(spec, runner)

    create = next(call for call in calls if call[1:3] == ["network", "create"])
    assert "--internal" in create
    assert create[create.index("--driver") + 1] == "bridge"
    assert evidence["internal"] is True
    assert evidence["policy_dns_name"] == robolab_worker.policy_container_name(spec)
    assert evidence["published_host_ports"] == []


@pytest.mark.parametrize(
    "mutation",
    [
        {"Internal": False},
        {"Driver": "host"},
        {"Scope": "swarm"},
        {"Labels": {"ai.openpi.project": "other", "ai.openpi.run-id": "wrong"}},
        {"Containers": {"unexpected": {}}},
    ],
)
def test_internal_network_inspection_fails_closed(mutation):
    spec = make_spec()
    document = _network_document(spec)
    document.update(mutation)

    with pytest.raises(repro_worker.WorkerError, match="internal-bridge contract"):
        robolab_worker.inspect_internal_network(
            spec,
            lambda _argv: json.dumps(document),
            require_empty=True,
        )


def test_internal_network_cleanup_inspects_removes_and_verifies_absence():
    spec = make_spec()
    name = robolab_worker.internal_network_name(spec)
    calls = []
    removed = False

    def runner(argv):
        nonlocal removed
        calls.append(list(argv))
        if list(argv[1:3]) == ["network", "ls"]:
            return "" if removed else name
        if list(argv[1:3]) == ["network", "inspect"]:
            return json.dumps(_network_document(spec))
        if list(argv[1:3]) == ["network", "rm"]:
            removed = True
            return name
        raise AssertionError(argv)

    robolab_worker.cleanup_internal_network(spec, runner)

    assert removed is True
    assert calls[-1][1:3] == ["network", "ls"]


def test_model_source_staging_verifies_bundle_head_and_clean_checkout(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.name", "Test"], check=True)
    (repository / "tracked.txt").write_text("pinned\n")
    subprocess.run(["git", "-C", str(repository), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-q", "-m", "fixture"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    source_bundle = tmp_path / "source.bundle"
    subprocess.run(["git", "-C", str(repository), "bundle", "create", str(source_bundle), "HEAD"], check=True)

    spec = make_spec()
    spec["model_source"] = {
        "s3_uri": f"s3://{robolab_worker.BUCKET}/source/openpi-{commit}-complete.bundle",
        "version_id": "fixture-version",
        "sha256": repro_worker.sha256_file(source_bundle),
        "commit": commit,
    }
    runtime_spec = robolab_worker._runtime_spec(spec)  # noqa: SLF001
    run_root = tmp_path / "run"
    run_root.mkdir()
    subprocess_runner = repro_worker.SubprocessRunner()

    def runner(argv):
        if list(argv[:3]) == ["aws", "s3api", "get-object"]:
            pathlib.Path(argv[-1]).write_bytes(source_bundle.read_bytes())
            return json.dumps({"VersionId": "fixture-version"})
        return subprocess_runner(argv)

    checkout, evidence = robolab_worker.stage_model_source(spec, runtime_spec, run_root, runner)

    assert evidence["head_commit"] == commit
    assert evidence["source_clean"] is True
    assert evidence["source_fsck_full"] is True
    assert (checkout / "tracked.txt").read_text() == "pinned\n"
    assert oct((checkout / "tracked.txt").stat().st_mode & 0o777) == "0o444"


def test_controller_source_evidence_requires_full_object_integrity():
    spec = make_spec()
    evidence = {
        "schema_version": 1,
        "source": dict(spec["source"]),
        "bundle_sha256_actual": spec["source"]["sha256"],
        "head_commit": spec["source"]["commit"],
        "source_clean": True,
        "source_fsck_full": True,
    }

    robolab_worker.validate_controller_source_evidence(spec, evidence)
    evidence["source_fsck_full"] = False
    with pytest.raises(repro_worker.WorkerError, match="fsck"):
        robolab_worker.validate_controller_source_evidence(spec, evidence)


def launch_metadata(now: dt.datetime, command_sha256: str) -> dict:
    return {
        "project": robolab_worker.PROJECT,
        "category": "evaluation",
        "workload": "evaluation",
        "command_sha256": command_sha256,
        "deadline_utc": (now + dt.timedelta(hours=4)).isoformat(),
        "instance_type": robolab_worker.INSTANCE_TYPE,
        "instance_count": 1,
        "purchase_option": "On-Demand",
        "retain_after_command": False,
        "reservation_id": "12345678-1234-4234-9234-123456789abc",
        "projected_compute_usd": 12.76802,
        "reserved_hours": 4.25,
    }


def instance_identity() -> dict:
    return {
        "accountId": robolab_worker.ACCOUNT_ID,
        "region": robolab_worker.REGION,
        "instanceId": "i-0123456789abcdef0",
        "instanceType": robolab_worker.INSTANCE_TYPE,
        "imageId": robolab_worker.EVALUATION_AMI_ID,
    }


def test_launch_and_host_validation_binds_command_ami_driver_and_on_demand(tmp_path):
    command = tmp_path / "run-command.sh"
    command.write_text("#!/bin/bash\ntrue\n")
    command_sha = hashlib.sha256(command.read_bytes()).hexdigest()
    now = dt.datetime(2026, 8, 4, tzinfo=dt.UTC)

    _, _, instance_id, instance_type = robolab_worker.validate_launch_and_host(
        make_spec(),
        launch_metadata(now, command_sha),
        instance_identity(),
        robolab_worker.EVALUATION_DRIVER_VERSION,
        now=now,
        command_path=command,
    )

    assert instance_id == "i-0123456789abcdef0"
    assert instance_type == robolab_worker.INSTANCE_TYPE


@pytest.mark.parametrize(
    ("mutation", "driver"),
    [
        (("metadata", "purchase_option", "Spot"), robolab_worker.EVALUATION_DRIVER_VERSION),
        (("metadata", "workload", "shallow_training"), robolab_worker.EVALUATION_DRIVER_VERSION),
        (("identity", "imageId", "ami-00000000000000000"), robolab_worker.EVALUATION_DRIVER_VERSION),
        (("identity", "instanceType", "g7e.4xlarge"), robolab_worker.EVALUATION_DRIVER_VERSION),
        (None, "595.71.05"),
    ],
)
def test_launch_and_host_validation_rejects_wrong_runtime(tmp_path, mutation, driver):
    command = tmp_path / "run-command.sh"
    command.write_text("true\n")
    command_sha = hashlib.sha256(command.read_bytes()).hexdigest()
    now = dt.datetime(2026, 8, 4, tzinfo=dt.UTC)
    metadata = launch_metadata(now, command_sha)
    identity = instance_identity()
    if mutation is not None:
        location, key, value = mutation
        {"metadata": metadata, "identity": identity}[location][key] = value

    with pytest.raises(repro_worker.WorkerError):
        robolab_worker.validate_launch_and_host(
            make_spec(),
            metadata,
            identity,
            driver,
            now=now,
            command_path=command,
        )


def test_evaluator_image_identity_requires_all_three_provenance_labels():
    labels = {
        "org.opencontainers.image.revision": robolab_worker.ROBOLAB_GIT_SHA,
        "ai.openpi.client-revision": robolab_worker.ROBOLAB_CLIENT_GIT_SHA,
        "ai.openpi.isaaclab-base-digest": robolab_worker.ISAACLAB_BASE_DIGEST,
    }
    assert robolab_worker.validate_evaluator_image_identity([robolab_worker.EVALUATOR_IMAGE["uri"]], labels) == [
        robolab_worker.EVALUATOR_IMAGE["uri"]
    ]

    for key in labels:
        wrong = dict(labels)
        wrong[key] = "wrong"
        with pytest.raises(repro_worker.WorkerError, match="provenance labels"):
            robolab_worker.validate_evaluator_image_identity([robolab_worker.EVALUATOR_IMAGE["uri"]], wrong)


def test_policy_cleanup_stops_captures_and_removes(tmp_path):
    calls = []

    def runner(argv):
        calls.append(list(argv))
        return ""

    log = tmp_path / "logs" / "policy.log"
    robolab_worker.cleanup_policy_container(
        "pi05-test-policy",
        log,
        30,
        runner,
        log_capture=lambda name: f"logs for {name}\n".encode(),
    )

    assert calls == [
        ["docker", "stop", "--time", "30", "pi05-test-policy"],
        ["docker", "rm", "--force", "pi05-test-policy"],
    ]
    assert log.read_text() == "logs for pi05-test-policy\n"


def test_runtime_cleanup_removes_network_even_when_policy_cleanup_fails(tmp_path, monkeypatch):
    calls = []

    def fail_policy(*_args, **_kwargs):
        calls.append("policy")
        raise repro_worker.WorkerError("policy cleanup broke")

    def clean_network(_spec, _runner):
        calls.append("network")

    monkeypatch.setattr(robolab_worker, "cleanup_policy_container", fail_policy)
    monkeypatch.setattr(robolab_worker, "cleanup_internal_network", clean_network)
    outcome = {}

    with pytest.raises(repro_worker.WorkerError, match="runtime-resource cleanup failed"):
        robolab_worker.cleanup_runtime_resources(
            spec=make_spec(),
            policy_container_owned=True,
            network_owned=True,
            policy_log=tmp_path / "policy.log",
            stop_grace_seconds=30,
            runner=lambda _argv: "",
            outcome=outcome,
        )

    assert calls == ["policy", "network"]
    assert outcome == {"policy_container_removed": False, "internal_network_removed": True}


def test_policy_readiness_is_probed_inside_policy_container():
    calls = []

    def runner(argv):
        calls.append(list(argv))
        return ""

    robolab_worker.wait_for_policy_server(
        "pi05-policy",
        dt.datetime.now(dt.UTC) + dt.timedelta(minutes=1),
        runner,
        monotonic=iter((0.0, 0.1)).__next__,
        sleep=lambda _seconds: None,
    )

    assert calls[0][:4] == ["docker", "exec", "pi05-policy", "python"]
    assert "127.0.0.1:8000/healthz" in calls[0][-1]


def test_wait_for_policy_server_rejects_early_exit():
    with pytest.raises(repro_worker.WorkerError, match="exited before readiness"):
        robolab_worker.wait_for_policy_server(
            "pi05-policy",
            dt.datetime.now(dt.UTC) + dt.timedelta(minutes=1),
            lambda _argv: "false 9",
            monotonic=iter((0.0, 0.1)).__next__,
            sleep=lambda _seconds: None,
            port_probe=lambda: False,
        )


def _native_records() -> list[dict]:
    records = []
    for task in robolab_worker.TASKS:
        records.extend(
            [
                {
                    "env_name": task,
                    "task_name": task,
                    "policy": "pi05",
                    "instruction_type": "default",
                    "instruction": f"instruction for {task}",
                    "success": episode % 2 == 0,
                    "episode": episode,
                    "run": episode // robolab_worker.NUM_ENVS,
                    "env_id": episode % robolab_worker.NUM_ENVS,
                    "run_name": f"{task}_{episode // robolab_worker.NUM_ENVS}",
                    "dt": 1.0 / 15.0,
                    "metrics": {"ee_path_length": float(episode + 1), "ee_sparc": -float(episode + 1)},
                }
                for episode in range(robolab_worker.EPISODES_PER_TASK)
            ]
        )
    return records


class PartialS3Runner:
    def __init__(self):
        self.objects = {}
        self.calls = []

    @staticmethod
    def _option(argv, name):
        return argv[argv.index(name) + 1]

    def __call__(self, argv):
        argv = list(argv)
        self.calls.append(argv)
        operation = argv[1:3]
        if operation == ["s3api", "list-object-versions"]:
            key = self._option(argv, "--prefix")
            if key not in self.objects:
                return json.dumps({"Versions": [], "DeleteMarkers": [], "IsTruncated": False})
            return json.dumps(
                {
                    "Versions": [{"Key": key, "VersionId": self.objects[key]["version"], "IsLatest": True}],
                    "DeleteMarkers": [],
                    "IsTruncated": False,
                }
            )
        if operation == ["s3api", "put-object"]:
            key = self._option(argv, "--key")
            assert key not in self.objects
            assert self._option(argv, "--if-none-match") == "*"
            body = pathlib.Path(self._option(argv, "--body")).read_bytes()
            metadata = dict(item.split("=", 1) for item in self._option(argv, "--metadata").split(","))
            self.objects[key] = {"body": body, "metadata": metadata, "version": f"version-{len(self.objects) + 1}"}
            return json.dumps({"VersionId": self.objects[key]["version"], "ServerSideEncryption": "AES256"})
        if operation == ["s3api", "head-object"]:
            key = self._option(argv, "--key")
            item = self.objects[key]
            return json.dumps(
                {
                    "VersionId": item["version"],
                    "ServerSideEncryption": "AES256",
                    "ContentLength": len(item["body"]),
                    "Metadata": item["metadata"],
                }
            )
        if operation == ["s3api", "get-object"]:
            key = self._option(argv, "--key")
            pathlib.Path(argv[-1]).write_bytes(self.objects[key]["body"])
            return "{}"
        raise AssertionError(argv)


def _output_manager(tmp_path, spec, runner):
    root = tmp_path / "output"
    for relative in (
        ".ready",
        ".receipts",
        ".spool",
        ".active",
        "checkpoints",
        "logs",
        "manifests",
        "artifacts",
    ):
        (root / relative).mkdir(parents=True)
    return repro_worker.OutputManager(robolab_worker._runtime_spec(spec), root, runner)  # noqa: SLF001


def test_partial_episode_publication_keeps_only_complete_batches_and_is_create_once(tmp_path):
    spec = make_spec()
    runner = PartialS3Runner()
    manager = _output_manager(tmp_path, spec, runner)
    results = tmp_path / "episode_results.jsonl"
    records = _native_records()
    results.write_text("".join(json.dumps(record) + "\n" for record in records[:27]))
    publisher = robolab_worker.PartialEpisodePublisher(spec, results, manager, runner)

    publisher()
    publisher()

    assert len(publisher.receipts) == 1
    assert publisher.receipts[0]["s3_uri"].startswith(
        f"s3://{robolab_worker.BUCKET}/runs/{spec['run_id']}/artifacts/robolab-partials/snapshot-0020-"
    )
    snapshot_path = next(
        path
        for path in (manager.root / "artifacts" / "robolab-partials").glob("snapshot-*.json")
        if not path.name.endswith(".receipt.json")
    )
    snapshot = json.loads(snapshot_path.read_text())
    assert snapshot["record_count"] == 20
    assert snapshot["complete_run_groups"] == 2
    puts = [call for call in runner.calls if call[1:3] == ["s3api", "put-object"]]
    assert len(puts) == 1
    with pytest.raises(repro_worker.WorkerError, match="prior object"):
        robolab_worker.publish_partial_snapshot_object(
            spec=spec,
            output_root=manager.root,
            path=snapshot_path,
            runner=runner,
        )
    assert len([call for call in runner.calls if call[1:3] == ["s3api", "put-object"]]) == 1


@pytest.mark.parametrize(
    "history",
    [
        {"Versions": [], "DeleteMarkers": [], "IsTruncated": True},
        {"Versions": [], "DeleteMarkers": []},
        {"Versions": {}, "DeleteMarkers": [], "IsTruncated": False},
        {"Versions": [None], "DeleteMarkers": [], "IsTruncated": False},
        {
            "Versions": [{"Key": "runs/example/snapshot.json", "VersionId": "v1"}],
            "DeleteMarkers": [],
            "IsTruncated": False,
        },
        {
            "Versions": [],
            "DeleteMarkers": [{"Key": "runs/example/snapshot.json", "VersionId": "v1"}],
            "IsTruncated": False,
        },
    ],
)
def test_partial_object_history_rejects_truncation_and_malformed_entries(history):
    def runner(_argv):
        return json.dumps(history)

    with pytest.raises(repro_worker.WorkerError, match=r"history|truncated"):
        robolab_worker._exact_object_history(  # noqa: SLF001
            bucket=robolab_worker.BUCKET,
            key="runs/example/snapshot.json",
            runner=runner,
        )


def test_partial_snapshot_publisher_rejects_symlink_payload(tmp_path):
    spec = make_spec()
    output_root = tmp_path / "output"
    snapshot_root = output_root / "artifacts" / "robolab-partials"
    snapshot_root.mkdir(parents=True)
    target = snapshot_root / "target.json"
    target.write_text("{}\n")
    link = snapshot_root / "snapshot.json"
    link.symlink_to(target)

    with pytest.raises(repro_worker.WorkerError, match="non-symlink"):
        robolab_worker.publish_partial_snapshot_object(
            spec=spec,
            output_root=output_root,
            path=link,
            runner=lambda _argv: "{}",
        )


def test_version_pinned_partial_snapshot_restores_exact_prefix_without_recount(tmp_path):
    parent_spec = make_spec()
    runner = PartialS3Runner()
    manager = _output_manager(tmp_path / "parent", parent_spec, runner)
    results = tmp_path / "parent-results.jsonl"
    records = _native_records()[:20]
    results.write_text("".join(json.dumps(record) + "\n" for record in records))
    publisher = robolab_worker.PartialEpisodePublisher(parent_spec, results, manager, runner)
    publisher()
    pin = {key: publisher.receipts[0][key] for key in ("s3_uri", "version_id", "sha256")}
    child_spec = robolab_worker.make_spec(
        run_id="robolab-base-20260804t130000z-a2",
        source=controller_source(),
        model_source=model_source(),
        continuation={"parent_run_id": parent_spec["run_id"], "snapshot": pin},
    )
    child_root = tmp_path / "child"
    child_root.mkdir()
    restored_results = child_root / "output" / robolab_worker.OUTPUT_FOLDER_NAME / "episode_results.jsonl"

    evidence = robolab_worker.restore_partial_snapshot(
        child_spec,
        robolab_worker._runtime_spec(child_spec),  # noqa: SLF001
        child_root,
        restored_results,
        runner,
    )

    assert evidence["record_count"] == 20
    restored = repro_robolab_report.load_results(restored_results)
    assert (
        len(repro_robolab_report.validate_native_continuation(restored, mode="intermediate", num_envs=10, num_runs=5))
        == 20
    )
    assert len({(record["task_name"], record["episode"]) for record in restored}) == 20


def _sealed_identity(results: pathlib.Path, spec: dict | None = None) -> dict:
    spec = spec or make_spec()
    model_sha256 = robolab_worker.PYTORCH_TEACHER["payload_objects"][-1]["sha256"]
    return {
        "schema_version": 1,
        "benchmark": "robolab",
        "stage": "base",
        "stage_identity": f"base-sha256:{model_sha256}",
        "checkpoint": {
            "model_path": "/mnt/openpi/checkpoints/pi05_droid_jointpos_pytorch/model.safetensors",
            "model_sha256": model_sha256,
        },
        "results": {"path": "episode_results.jsonl", "sha256": repro_worker.sha256_file(results)},
        "runtime": {
            "image_digest": robolab_worker.EVALUATOR_IMAGE["uri"],
            "robolab_git_sha": robolab_worker.ROBOLAB_GIT_SHA,
            "openpi_client_git_sha": robolab_worker.ROBOLAB_CLIENT_GIT_SHA,
            "isaac_sim_version": "5.0.0",
            "isaac_lab_version": robolab_worker.EVALUATOR_IMAGE["isaac_lab_version"],
        },
        "policy_server": robolab_worker.policy_server_identity(spec),
        "evaluation": {
            "mode": "intermediate",
            "tasks": list(robolab_worker.TASKS),
            "episodes_per_task": robolab_worker.EPISODES_PER_TASK,
            "num_envs": robolab_worker.NUM_ENVS,
            "num_runs": robolab_worker.NUM_RUNS,
            "policy": "pi05",
            "policy_server_seed": robolab_worker.POLICY_SERVER_SEED,
            "environment_seed": robolab_worker.ENVIRONMENT_SEED,
            "instruction_type": "default",
            "open_loop_horizon": 15,
        },
    }


def test_summary_is_bound_to_sealed_identity_and_exact_counts(tmp_path):
    spec = make_spec()
    results = tmp_path / "episode_results.jsonl"
    results.write_text("".join(json.dumps(record) + "\n" for record in _native_records()))
    identity = tmp_path / "run-identity.json"
    identity.write_text(json.dumps(_sealed_identity(results, spec)))

    summary = robolab_worker.summarize_results(spec, results, identity)

    assert summary["aggregate_success_rate"] == 0.5
    assert all(value["episodes"] == 50 for value in summary["tasks"].values())
    assert summary["model_sha256"] == "3212bbd9737caf175ba238193a9e1e3b7b16a4c5d1c4b586ad3d65d58deb5117"


@pytest.mark.parametrize(
    ("section", "key", "replacement"),
    [
        ("checkpoint", "model_sha256", "f" * 64),
        ("runtime", "image_digest", "sha256:" + "e" * 64),
        ("results", "sha256", "d" * 64),
        ("evaluation", "policy_server_seed", 7),
        ("evaluation", "open_loop_horizon", 10),
        ("policy_server", "image_digest", "sha256:" + "c" * 64),
        ("policy_server", "command_sha256", "b" * 64),
    ],
)
def test_summary_rejects_any_sealed_identity_change(tmp_path, section, key, replacement):
    spec = make_spec()
    results = tmp_path / "episode_results.jsonl"
    results.write_text("".join(json.dumps(record) + "\n" for record in _native_records()))
    document = _sealed_identity(results, spec)
    document[section][key] = replacement
    identity = tmp_path / "run-identity.json"
    identity.write_text(json.dumps(document))

    with pytest.raises(repro_worker.WorkerError, match="sealed RoboLab run identity"):
        robolab_worker.summarize_results(spec, results, identity)


def test_summary_rejects_native_results_that_do_not_match_the_seal_contract(tmp_path):
    spec = make_spec()
    records = _native_records()
    records[0]["episode"] = 1
    results = tmp_path / "episode_results.jsonl"
    results.write_text("".join(json.dumps(record) + "\n" for record in records))
    identity = tmp_path / "run-identity.json"
    identity.write_text(json.dumps(_sealed_identity(results, spec)))

    with pytest.raises(repro_worker.WorkerError, match="could not be summarized"):
        robolab_worker.summarize_results(spec, results, identity)


class FakeManager:
    def __init__(self, root: pathlib.Path):
        self.root = root
        self.spec = {"run_id": "robolab-base-test"}
        self.events = []

    def commit_expected_outputs(self):
        self.events.append("commit-outputs")
        return []

    def sync_once(self):
        self.events.append("sync")
        if "run-manifest-marker" in self.events:
            return [{"marker": "run-manifest.ready.json", "artifacts": [{"version_id": "v1"}]}]
        return []

    def create_marker(self, _kind, _paths, name):
        self.events.append("run-manifest-marker" if name == "run-manifest.ready.json" else "final-marker")


def test_publication_is_payload_first_manifest_then_final_evidence(tmp_path):
    root = tmp_path / "output"
    (root / "manifests").mkdir(parents=True)
    manager = FakeManager(root)

    robolab_worker.publish_terminal_manifests(
        manager,
        {"status": "succeeded"},
        commit_expected_outputs=True,
    )

    assert manager.events == [
        "commit-outputs",
        "sync",
        "run-manifest-marker",
        "sync",
        "final-marker",
        "sync",
    ]
    assert (root / "manifests" / "run-manifest.json").is_file()
    assert (root / "manifests" / "final-sync-evidence.json").is_file()


def test_pre_scratch_failure_uses_emergency_output_manager_and_terminal_manifest(tmp_path, monkeypatch):
    captured = {}

    def fail_before_scratch(*_args, **_kwargs):
        raise repro_worker.WorkerError("scratch selection failed")

    def capture_publication(manager, manifest, *, commit_expected_outputs):
        captured["root"] = manager.root
        captured["manifest"] = manifest
        captured["commit"] = commit_expected_outputs

    monkeypatch.setattr(robolab_worker, "_execute_worker_with_scratch", fail_before_scratch)
    monkeypatch.setattr(robolab_worker, "publish_terminal_manifests", capture_publication)

    exit_code = robolab_worker.execute_worker(
        make_spec(),
        {"schema_version": 1},
        {"project": robolab_worker.PROJECT},
        runner=lambda _argv: "",
        emergency_root=tmp_path / "emergency",
    )

    assert exit_code == 3
    assert captured["commit"] is False
    assert captured["manifest"]["status"] == "failed"
    assert captured["manifest"]["failure_phase"] == "before-dedicated-output-manager"
    assert "scratch selection failed" in captured["manifest"]["failure"]
    assert captured["root"].is_dir()

    captured.clear()
    exit_code = robolab_worker.execute_worker(
        make_spec(),
        {},
        {},
        runner=lambda _argv: "",
        emergency_root=tmp_path / "startup-emergency",
        startup_failure=repro_worker.WorkerError("source evidence is missing"),
    )
    assert exit_code == 3
    assert captured["manifest"]["failure_phase"] == "startup-evidence-load"
    assert "source evidence is missing" in captured["manifest"]["failure"]


def test_execute_cli_routes_missing_startup_evidence_through_emergency_publication(tmp_path, monkeypatch):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(make_spec()))
    captured = {}

    def capture_execute(spec, source_evidence, launch_metadata, **kwargs):
        captured["spec"] = spec
        captured["source_evidence"] = source_evidence
        captured["launch_metadata"] = launch_metadata
        captured["startup_failure"] = kwargs.get("startup_failure")
        return 3

    monkeypatch.setattr(robolab_worker, "execute_worker", capture_execute)

    assert (
        robolab_worker.main(
            [
                "run",
                "--spec",
                str(spec_path),
                "--source-evidence",
                str(tmp_path / "missing-source-evidence.json"),
                "--launch-metadata",
                str(tmp_path / "missing-launch-metadata.json"),
                "--execute",
            ]
        )
        == 3
    )
    assert captured["spec"]["run_id"] == make_spec()["run_id"]
    assert captured["source_evidence"] == {}
    assert captured["launch_metadata"] == {}
    assert isinstance(captured["startup_failure"], repro_worker.WorkerError)


def test_rendered_bootstrap_is_dry_by_default_and_has_separate_execute_gate():
    kwargs = {
        "spec_s3_uri": f"s3://{robolab_worker.BUCKET}/specs/robolab-base.json",
        "spec_version_id": "spec-version",
        "spec_sha256": "a" * 64,
    }
    dry = robolab_worker.render_bootstrap_command(**kwargs, execute=False)
    live = robolab_worker.render_bootstrap_command(**kwargs, execute=True)

    assert 'repro_robolab_worker.py" run' in dry
    assert "rev-parse --is-shallow-repository" in dry
    assert "fsck --full --no-dangling" in dry
    assert '"source_fsck_full":True' in dry
    assert "--launch-metadata /opt/pi05/launch-metadata.json --execute" not in dry
    assert "--launch-metadata /opt/pi05/launch-metadata.json --execute" in live


def test_runbook_spec_publication_requires_exact_singleton_no_delete_history():
    runbook = (pathlib.Path(__file__).resolve().parents[1] / "repro" / "ROBOLAB_EVAL_RUNBOOK.md").read_text()
    block = runbook.split("export ROBOLAB_SPEC_SHA256=", 1)[1].split("There are two independent execution gates.", 1)[0]

    assert "set -euo pipefail" in runbook.split("export ROBOLAB_SPEC_SHA256=", 1)[0][-100:]
    assert block.count("list-object-versions") == 2
    assert block.count("--max-keys 10") == 2
    assert block.count(".IsTruncated == false") == 2
    assert block.count(".DeleteMarkers[]?") == 2
    assert "length == 0" in block
    assert ".[0].VersionId == $version" in block
    assert ".[0].IsLatest == true" in block
    assert "--if-none-match '*'" in block
    assert '.ServerSideEncryption == "AES256"' in block
    assert ".Metadata.sha256 == $sha256" in block
    assert '--version-id "$ROBOLAB_SPEC_VERSION_ID"' in block


def test_runbook_worker_spec_pins_complete_controller_and_model_bundles():
    runbook = (pathlib.Path(__file__).resolve().parents[1] / "repro" / "ROBOLAB_EVAL_RUNBOOK.md").read_text()

    for flag in (
        "--model-source-s3-uri",
        "--model-source-version-id",
        "--model-source-sha256",
        "--model-source-commit",
    ):
        assert runbook.count(flag) == 2
    assert "openpi-${CONTROLLER_COMMIT}-complete.bundle" in runbook
    assert "openpi-${MODEL_SOURCE_COMMIT}-complete.bundle" in runbook
    assert "CN9PJHZ3oHC3hb7lwDTH9p3JEAVQmUhh" in runbook
    assert "9be1f91dfec636d1cbb63ad87b166e301b98835b91a6212f73fa5b5350d0f7b5" in runbook


def test_runbook_manual_seal_binds_policy_server_runtime_and_source():
    runbook = (pathlib.Path(__file__).resolve().parents[1] / "repro" / "ROBOLAB_EVAL_RUNBOOK.md").read_text()
    seal_block = runbook.split("policy_command_sha256 ()", 1)[1].split("## 4. Emit promotion evidence", 1)[0]

    for flag in (
        "--policy-image-digest",
        "--policy-source-s3-uri",
        "--policy-source-version-id",
        "--policy-source-sha256",
        "--policy-source-commit",
        "--policy-config",
        "--policy-command-sha256",
    ):
        assert flag in seal_block
    assert '"python","scripts/serve_policy.py"' in seal_block
    assert '"--policy.dir",checkpoint' in seal_block


def test_dry_run_cli_performs_no_execution(tmp_path, capsys):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(make_spec()))

    assert robolab_worker.main(["run", "--spec", str(spec_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "dry-run"
    assert output["mutations_authorized"] is False
    assert output["evaluation"]["episodes_per_task"] == 50
