import base64
import hashlib
import io
import json
import urllib.parse

import pytest

from scripts import repro_stage_checkpoints
from scripts import repro_stage_data


class Response(io.BytesIO):
    def __init__(self, payload, *, status=200):
        super().__init__(payload)
        self.status = status

    def getcode(self):
        return self.status


def object_record(spec, relative, payload, *, generation="123"):
    return {
        "name": f"{spec.object_prefix}{relative}",
        "generation": generation,
        "bytes": len(payload),
        "md5_base64": base64.b64encode(hashlib.md5(payload, usedforsecurity=False).digest()).decode(),
        "crc32c_base64": "AAAAAA==",
        "updated": "2026-01-01T00:00:00Z",
    }


def test_gcs_inventory_is_generation_pinned_and_revision_is_stable():
    spec = repro_stage_checkpoints.CHECKPOINTS["libero"]
    pages = [
        {
            "items": [
                {
                    "name": f"{spec.object_prefix}params/b",
                    "generation": "2",
                    "size": "20",
                    "crc32c": "b",
                    "updated": "2026-01-01T00:00:01Z",
                }
            ],
            "nextPageToken": "next",
        },
        {
            "items": [
                {
                    "name": f"{spec.object_prefix}params/a",
                    "generation": "1",
                    "size": "10",
                    "md5Hash": "a",
                    "crc32c": "c",
                    "updated": "2026-01-01T00:00:00Z",
                }
            ]
        },
    ]

    def opener(_url):
        return Response(json.dumps(pages.pop(0)).encode())

    inventory = repro_stage_checkpoints.list_gcs_inventory(spec, opener=opener)
    assert [item["name"] for item in inventory] == [
        f"{spec.object_prefix}params/a",
        f"{spec.object_prefix}params/b",
    ]
    revision = repro_stage_checkpoints.inventory_revision(inventory)
    assert len(revision) == 64
    assert revision == repro_stage_checkpoints.inventory_revision(list(reversed(inventory)))


def test_robolab_teacher_uses_public_joint_position_checkpoint():
    spec = repro_stage_checkpoints.CHECKPOINTS["droid_jointpos"]
    assert spec.source_uri == "gs://openpi-assets-simeval/pi05_droid_jointpos"
    assert spec.bucket == "openpi-assets-simeval"
    assert spec.object_prefix == "pi05_droid_jointpos/"
    assert spec.local_dirname == "pi05_droid_jointpos"


def test_download_uses_exact_object_generation_and_checks_md5(tmp_path):
    spec = repro_stage_checkpoints.CHECKPOINTS["droid"]
    payload = b"checkpoint bytes"
    item = object_record(spec, "params/chunk", payload, generation="987654321")
    requests = []

    def opener(request):
        requests.append(request)
        return Response(payload)

    path = repro_stage_checkpoints.download_gcs_object(spec, tmp_path, item, opener=opener)
    assert path.read_bytes() == payload
    assert "generation=987654321" in requests[0].full_url
    assert urllib.parse.quote(item["name"], safe="") in requests[0].full_url

    path.write_bytes(b"same byte length")
    assert len(path.read_bytes()) == len(payload)
    with pytest.raises(repro_stage_data.StageError, match="size/MD5 validation"):
        repro_stage_checkpoints.download_gcs_object(
            spec,
            tmp_path,
            item,
            opener=lambda _request: Response(b"wrong checkpoint"),
        )


def test_checkpoint_manifest_binds_gcs_inventory_and_local_sha256(tmp_path):
    spec = repro_stage_checkpoints.CHECKPOINTS["libero"]
    root = tmp_path / spec.local_dirname
    payloads = {"assets/stats.json": b"{}", "params/chunk": b"weights"}
    inventory = []
    for index, (relative, payload) in enumerate(payloads.items()):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        inventory.append(object_record(spec, relative, payload, generation=str(index + 1)))

    manifest = repro_stage_checkpoints.build_checkpoint_manifest(spec, root, inventory, hash_workers=2)
    assert manifest["source"]["revision"] == repro_stage_checkpoints.inventory_revision(inventory)
    assert manifest["source"]["objects"] == inventory
    hashes = {item["path"]: item["sha256"] for item in manifest["files"]}
    assert hashes["params/chunk"] == hashlib.sha256(b"weights").hexdigest()
    target = repro_stage_checkpoints.checkpoint_s3_target(
        "s3://bucket/checkpoints", spec, manifest["source"]["revision"]
    )
    assert manifest["source"]["revision"] in target.snapshot_uri


def test_checkpoint_dry_run_has_no_network_or_aws(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"aws": {"account_id": "752160877725", "region": "us-east-2"}}))

    def unexpected(*_args, **_kwargs):
        raise AssertionError("dry-run performed an external action")

    monkeypatch.setattr(repro_stage_checkpoints, "list_gcs_inventory", unexpected)
    monkeypatch.setattr(repro_stage_checkpoints, "upload_checkpoint", unexpected)
    result = repro_stage_checkpoints.main(
        [
            "stage",
            "--checkpoint",
            "libero",
            "--local-root",
            str(tmp_path / "checkpoints"),
            "--s3-root",
            "s3://bucket/checkpoints",
            "--config",
            str(config),
        ]
    )
    assert result == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "dry-run"
    assert plan["source_revision"].startswith("resolved from")
    assert plan["mutations_authorized"] is False
