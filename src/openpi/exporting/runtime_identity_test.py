from __future__ import annotations

import io
import json
import subprocess

import pytest

from openpi.exporting import runtime_identity

IMAGE = "sha256:" + "1" * 64
INSTANCE_ID = "i-0123456789abcdef0"
INSTANCE_TYPE = "g7e.4xlarge"
GPU_ROW = "GPU-11111111-1111-1111-1111-111111111111, NVIDIA L40S, 595.71.05"


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_worker_environment_binds_image_instance_and_type_without_network():
    def unexpected_opener(*args, **kwargs):
        del args, kwargs
        raise AssertionError("IMDS must not be queried inside a network-none worker")

    identity = runtime_identity.require_live_runtime_identity(
        image_digest=IMAGE,
        instance_type=INSTANCE_TYPE,
        instance_id=INSTANCE_ID,
        environ={
            "PI05_IMAGE_DIGEST": IMAGE,
            "PI05_INSTANCE_ID": INSTANCE_ID,
            "PI05_INSTANCE_TYPE": INSTANCE_TYPE,
        },
        imds_opener=unexpected_opener,
    )

    assert identity.manifest_runtime == {
        "image_digest": IMAGE,
        "instance_type": INSTANCE_TYPE,
        "instance_id": INSTANCE_ID,
    }
    assert identity.instance_identity_source == "worker-environment"


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        (
            {
                "PI05_IMAGE_DIGEST": "sha256:" + "2" * 64,
                "PI05_INSTANCE_ID": INSTANCE_ID,
                "PI05_INSTANCE_TYPE": INSTANCE_TYPE,
            },
            "image digest differs",
        ),
        (
            {
                "PI05_IMAGE_DIGEST": IMAGE,
                "PI05_INSTANCE_ID": "i-11111111111111111",
                "PI05_INSTANCE_TYPE": INSTANCE_TYPE,
            },
            "instance ID differs",
        ),
        (
            {
                "PI05_IMAGE_DIGEST": IMAGE,
                "PI05_INSTANCE_ID": INSTANCE_ID,
                "PI05_INSTANCE_TYPE": "g6e.4xlarge",
            },
            "instance type differs",
        ),
    ],
)
def test_worker_environment_rejects_forged_declared_identity(environment, message):
    with pytest.raises(ValueError, match=message):
        runtime_identity.require_live_runtime_identity(
            image_digest=IMAGE,
            instance_type=INSTANCE_TYPE,
            instance_id=INSTANCE_ID,
            environ=environment,
        )


def test_image_binding_is_required_even_when_imds_is_available():
    with pytest.raises(RuntimeError, match="PI05_IMAGE_DIGEST is required"):
        runtime_identity.require_live_runtime_identity(
            image_digest=IMAGE,
            instance_type=INSTANCE_TYPE,
            instance_id=INSTANCE_ID,
            environ={},
        )


@pytest.mark.parametrize(
    "environment",
    [
        {"PI05_IMAGE_DIGEST": IMAGE, "PI05_INSTANCE_ID": INSTANCE_ID},
        {"PI05_IMAGE_DIGEST": IMAGE, "PI05_INSTANCE_TYPE": INSTANCE_TYPE},
    ],
)
def test_worker_instance_id_and_type_must_be_injected_together(environment):
    with pytest.raises(RuntimeError, match="must be injected together"):
        runtime_identity.require_live_runtime_identity(
            image_digest=IMAGE,
            instance_type=INSTANCE_TYPE,
            instance_id=INSTANCE_ID,
            environ=environment,
        )


def test_host_execution_binds_instance_to_imdsv2():
    calls = []
    document = {
        "accountId": runtime_identity.EXPECTED_AWS_ACCOUNT,
        "region": runtime_identity.EXPECTED_AWS_REGION,
        "instanceId": INSTANCE_ID,
        "instanceType": INSTANCE_TYPE,
    }

    def opener(request, *, timeout):
        calls.append((request.full_url, request.get_method(), timeout))
        if request.full_url.endswith("/api/token"):
            return _Response(b"token")
        assert request.get_header("X-aws-ec2-metadata-token") == "token"
        return _Response(json.dumps(document).encode())

    identity = runtime_identity.require_live_runtime_identity(
        image_digest=IMAGE,
        instance_type=INSTANCE_TYPE,
        instance_id=INSTANCE_ID,
        environ={"PI05_IMAGE_DIGEST": IMAGE},
        imds_opener=opener,
    )

    assert identity.instance_identity_source == "imds-v2"
    assert [method for _, method, _ in calls] == ["PUT", "GET"]


def test_host_execution_rejects_a_different_live_instance():
    document = {
        "accountId": runtime_identity.EXPECTED_AWS_ACCOUNT,
        "region": runtime_identity.EXPECTED_AWS_REGION,
        "instanceId": "i-11111111111111111",
        "instanceType": INSTANCE_TYPE,
    }

    def opener(request, *, timeout):
        del timeout
        return _Response(b"token" if request.full_url.endswith("/api/token") else json.dumps(document).encode())

    with pytest.raises(ValueError, match="differs from live IMDS"):
        runtime_identity.require_live_runtime_identity(
            image_digest=IMAGE,
            instance_type=INSTANCE_TYPE,
            instance_id=INSTANCE_ID,
            environ={"PI05_IMAGE_DIGEST": IMAGE},
            imds_opener=opener,
        )


def test_export_chain_requires_one_exact_image_digest():
    assert runtime_identity.require_same_image_digest(IMAGE, IMAGE) == IMAGE
    with pytest.raises(ValueError, match="must use one image digest"):
        runtime_identity.require_same_image_digest(IMAGE, "sha256:" + "2" * 64)


@pytest.mark.parametrize(
    "rows",
    [
        (),
        ("GPU-not-a-uuid, NVIDIA L40S, 595.71.05",),
        ("GPU-11111111-1111-1111-1111-111111111111, NVIDIA L40S",),
        ("GPU-11111111-1111-1111-1111-111111111111, NVIDIA L40S, unknown",),
        (GPU_ROW, GPU_ROW),
    ],
)
def test_gpu_inventory_rejects_empty_malformed_or_duplicate_rows(rows):
    with pytest.raises(RuntimeError):
        runtime_identity.validate_gpu_inventory(rows)


def test_gpu_inventory_query_returns_canonical_rows_and_fails_closed():
    assert runtime_identity.query_gpu_inventory(lambda command, text: GPU_ROW + "\n") == (GPU_ROW,)

    def failed(command, *, text):
        raise subprocess.CalledProcessError(1, command)

    with pytest.raises(RuntimeError, match="GPU identity is required"):
        runtime_identity.query_gpu_inventory(failed)
