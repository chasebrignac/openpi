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
    command = repro_ddp_two_node._docker_base(
        values,
        name="test",
        rank=1,
        master_ip="172.31.1.11",
        interface="ens5",
        output_dir=tmp_path,
        smoke_path=tmp_path / "smoke.py",
        port=29400,
    )
    rendered = " ".join(command)
    assert "--network host" in rendered
    assert "NCCL_SOCKET_IFNAME=ens5" in rendered
    assert "GLOO_SOCKET_IFNAME=ens5" in rendered
    assert "NCCL_IB_DISABLE=1" in rendered
    assert "NCCL_CUMEM_ENABLE=0" in rendered
    assert "--nnodes=2" in rendered
    assert "--node-rank=1" in rendered
