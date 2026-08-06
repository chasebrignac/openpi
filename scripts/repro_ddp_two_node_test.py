from __future__ import annotations

import json
import pathlib
import types

import pytest

from scripts import repro_ddp_two_node

# Tests intentionally exercise fail-closed pure helpers.
# ruff: noqa: SLF001


def _values() -> dict[str, str]:
    digest = "sha256:" + "b" * 64
    run_id = "ddp-two-node-20260805t230000z"
    return {
        "PI05_RUN_ID": run_id,
        "PI05_ORCHESTRATOR_SHA256": "a" * 64,
        "PI05_SMOKE_KEY": f"control/ddp-validation/{run_id}/repro_ddp_smoke.py",
        "PI05_SMOKE_VERSION_ID": "version.1_A-b",
        "PI05_SMOKE_SHA256": "c" * 64,
        "PI05_IMAGE_URI": f"{repro_ddp_two_node.EXPECTED_REPOSITORY}@{digest}",
        "PI05_IMAGE_DIGEST": digest,
        "PI05_IMAGE_SOURCE_SHA": "d" * 40,
    }


def test_validate_go_returns_exact_rank_master_and_peer() -> None:
    values = _values()
    identity = {
        "instance_id": "i-00000000000000001",
        "private_ip": "172.31.1.11",
    }
    launch = {"command_sha256": "e" * 64, "reservation_id": "reservation"}
    go = {
        "schema_version": 1,
        "run_id": values["PI05_RUN_ID"],
        "account_id": repro_ddp_two_node.EXPECTED_ACCOUNT,
        "region": repro_ddp_two_node.EXPECTED_REGION,
        "instance_type": repro_ddp_two_node.EXPECTED_INSTANCE_TYPE,
        "world_size": 2,
        "image_uri": values["PI05_IMAGE_URI"],
        "image_digest": values["PI05_IMAGE_DIGEST"],
        "image_source_sha": values["PI05_IMAGE_SOURCE_SHA"],
        "orchestrator_sha256": values["PI05_ORCHESTRATOR_SHA256"],
        "smoke_key": values["PI05_SMOKE_KEY"],
        "smoke_version_id": values["PI05_SMOKE_VERSION_ID"],
        "smoke_sha256": values["PI05_SMOKE_SHA256"],
        "command_sha256": launch["command_sha256"],
        "reservation_id": launch["reservation_id"],
        "nodes": [
            {"rank": 0, "instance_id": identity["instance_id"], "private_ip": identity["private_ip"]},
            {"rank": 1, "instance_id": "i-00000000000000002", "private_ip": "172.31.1.12"},
        ],
    }

    assert repro_ddp_two_node._validate_go(go, values, identity, launch) == (0, "172.31.1.11", "172.31.1.12")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda go: go.update(world_size=8),
        lambda go: go.update(image_digest="sha256:" + "f" * 64),
        lambda go: go["nodes"].append(go["nodes"][0].copy()),
        lambda go: go["nodes"][1].update(rank=0),
        lambda go: go["nodes"][1].update(private_ip=go["nodes"][0]["private_ip"]),
    ],
)
def test_validate_go_rejects_mismatched_or_duplicate_contract(mutation) -> None:
    values = _values()
    identity = {"instance_id": "i-00000000000000001", "private_ip": "172.31.1.11"}
    launch = {"command_sha256": "e" * 64, "reservation_id": "reservation"}
    go = {
        "schema_version": 1,
        "run_id": values["PI05_RUN_ID"],
        "account_id": repro_ddp_two_node.EXPECTED_ACCOUNT,
        "region": repro_ddp_two_node.EXPECTED_REGION,
        "instance_type": repro_ddp_two_node.EXPECTED_INSTANCE_TYPE,
        "world_size": 2,
        "image_uri": values["PI05_IMAGE_URI"],
        "image_digest": values["PI05_IMAGE_DIGEST"],
        "image_source_sha": values["PI05_IMAGE_SOURCE_SHA"],
        "orchestrator_sha256": values["PI05_ORCHESTRATOR_SHA256"],
        "smoke_key": values["PI05_SMOKE_KEY"],
        "smoke_version_id": values["PI05_SMOKE_VERSION_ID"],
        "smoke_sha256": values["PI05_SMOKE_SHA256"],
        "command_sha256": launch["command_sha256"],
        "reservation_id": launch["reservation_id"],
        "nodes": [
            {"rank": 0, "instance_id": identity["instance_id"], "private_ip": identity["private_ip"]},
            {"rank": 1, "instance_id": "i-00000000000000002", "private_ip": "172.31.1.12"},
        ],
    }
    mutation(go)

    with pytest.raises(repro_ddp_two_node.ValidationError):
        repro_ddp_two_node._validate_go(go, values, identity, launch)


def test_inventory_hashes_regular_files_and_rejects_symlinks(tmp_path: pathlib.Path) -> None:
    (tmp_path / "result.json").write_text("ok\n")
    inventory = repro_ddp_two_node._inventory(tmp_path)
    assert inventory == [
        {
            "path": "result.json",
            "bytes": 3,
            "sha256": "dc51b8c96c2d745df3bd5590d990230a482fd247123599548e0632fdbf97fc22",
        }
    ]
    (tmp_path / "link").symlink_to(tmp_path / "result.json")
    with pytest.raises(repro_ddp_two_node.ValidationError, match="symlink"):
        repro_ddp_two_node._inventory(tmp_path)


def test_network_interface_uses_route_device(monkeypatch) -> None:
    monkeypatch.setattr(
        repro_ddp_two_node,
        "_run",
        lambda *args, **kwargs: types.SimpleNamespace(stdout=json.dumps([{"dev": "ens5"}]).encode()),
    )
    assert repro_ddp_two_node._network_interface("172.31.1.12") == "ens5"


def test_docker_base_binds_two_node_tcp_and_disables_unneeded_transports(tmp_path: pathlib.Path) -> None:
    values = _values()
    identity = repro_ddp_two_node._prepare_nonroot_identity(tmp_path / "contract")
    command = repro_ddp_two_node._docker_base(
        values,
        name="test",
        rank=1,
        master_ip="172.31.1.11",
        interface="ens5",
        output_dir=tmp_path,
        smoke_path=tmp_path / "smoke.py",
        port=29400,
        nonroot_identity=identity,
    )
    rendered = " ".join(command)
    assert "--network host" in rendered
    assert "NCCL_SOCKET_IFNAME=ens5" in rendered
    assert "GLOO_SOCKET_IFNAME=ens5" in rendered
    assert "NCCL_IB_DISABLE=1" in rendered
    assert "NCCL_CUMEM_ENABLE=0" in rendered
    assert "--nnodes=2" in rendered
    assert "--node-rank=1" in rendered
    assert "--user 1000:1000" in rendered
    assert "dst=/etc/passwd,readonly" in rendered
    assert "dst=/etc/group,readonly" in rendered
    assert "HOME=/output/.home" in rendered
    assert "USER=pi05" in rendered
    assert "TORCHINDUCTOR_CACHE_DIR=/output/.torchinductor-cache" in rendered


def test_nonroot_identity_is_minimal_readonly_and_create_once(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "attempt"
    identity = repro_ddp_two_node._prepare_nonroot_identity(root)
    assert identity["passwd_path"].read_text() == repro_ddp_two_node.PASSWD_CONTENT
    assert identity["group_path"].read_text() == repro_ddp_two_node.GROUP_CONTENT
    assert identity["passwd_path"].stat().st_mode & 0o777 == 0o444
    assert identity["group_path"].stat().st_mode & 0o777 == 0o444
    repro_ddp_two_node._validate_nonroot_identity(identity)
    with pytest.raises(FileExistsError):
        repro_ddp_two_node._prepare_nonroot_identity(root)


def test_identity_preflight_is_networkless_and_checks_pwd_contract(tmp_path: pathlib.Path) -> None:
    values = _values()
    identity = repro_ddp_two_node._prepare_nonroot_identity(tmp_path / "contract")
    command = repro_ddp_two_node._identity_preflight_command(
        values,
        name="identity-preflight",
        output_dir=tmp_path / "output",
        nonroot_identity=identity,
    )
    rendered = " ".join(command)
    assert "--network none" in rendered
    assert "--gpus" not in command
    assert "getent passwd 1000" in rendered
    assert "pwd.getpwuid(os.getuid())" in rendered
    assert "nonroot_identity_preflight=PASS" in rendered


def test_openpi_debug_uses_compatible_full_width_vlm_and_disables_compile(tmp_path, monkeypatch) -> None:
    values = _values()
    identity = repro_ddp_two_node._prepare_nonroot_identity(tmp_path / "contract")
    captured = {}

    def fake_run(arguments, name, log_path, timeout_seconds):
        captured.update(arguments=arguments, name=name, log_path=log_path, timeout_seconds=timeout_seconds)
        return 1.25

    monkeypatch.setattr(repro_ddp_two_node, "_run_container", fake_run)
    seconds = repro_ddp_two_node._run_openpi_debug(
        values,
        rank=0,
        master_ip="172.31.1.11",
        interface="ens5",
        actual_root=tmp_path / "actual",
        smoke_path=tmp_path / "smoke.py",
        port=30475,
        nonroot_identity=identity,
        experiment="debug-compatible",
        container_label="compatible-a1",
        log_path=tmp_path / "openpi.log",
    )

    assert seconds == 1.25
    command = captured["arguments"]
    assert command[command.index("--model.paligemma-variant") + 1] == "gemma_2b"
    assert command[command.index("--model.action-expert-variant") + 1] == "dummy"
    assert command[command.index("--model.pytorch-compile-mode") + 1] == "None"


@pytest.mark.parametrize("attempt", ["x", "UPPER-a1", "bad_underscore", "-starts-hyphen", "ends-"])
def test_recovery_attempt_rejects_unsafe_slugs(attempt: str) -> None:
    with pytest.raises(repro_ddp_two_node.ValidationError):
        repro_ddp_two_node._validate_recovery_attempt(attempt)


def test_recovery_prefix_and_port_are_fresh_and_deterministic() -> None:
    attempt = "uid1000-a1"
    run_id = _values()["PI05_RUN_ID"]
    assert repro_ddp_two_node._validate_recovery_attempt(attempt) == attempt
    port = repro_ddp_two_node._recovery_port(attempt)
    assert 29500 <= port <= 30499
    assert port not in {29400, 29401}
    assert port == repro_ddp_two_node._recovery_port(attempt)
    assert repro_ddp_two_node._recovery_prefix(run_id, attempt) == (
        f"diagnostics/two-node-ddp/{run_id}/recoveries/{attempt}"
    )
