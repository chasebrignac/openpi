from __future__ import annotations

import pytest

from scripts import serve_tensorrt_policy


def test_server_rejects_cross_image_identity_before_loading_policy(monkeypatch, tmp_path):
    image = "sha256:" + "1" * 64
    instance_id = "i-0123456789abcdef0"
    monkeypatch.setenv("PI05_IMAGE_DIGEST", "sha256:" + "2" * 64)
    monkeypatch.setenv("PI05_INSTANCE_ID", instance_id)
    monkeypatch.setenv("PI05_INSTANCE_TYPE", "g7e.4xlarge")
    monkeypatch.setattr(serve_tensorrt_policy, "create_policy", lambda args: pytest.fail("policy must not load"))

    args = serve_tensorrt_policy.Args(
        artifact_dir=tmp_path / "artifacts",
        checkpoint_dir=tmp_path / "checkpoint",
        precision=serve_tensorrt_policy.Precision.FP8,
        track=serve_tensorrt_policy.Track.LIBERO,
        dataset="physical-intelligence/libero",
        dataset_revision="revision",
        image_digest=image,
        instance_type="g7e.4xlarge",
        instance_id=instance_id,
    )

    with pytest.raises(ValueError, match="image digest differs"):
        serve_tensorrt_policy.main(args)
