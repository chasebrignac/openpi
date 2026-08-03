import argparse

from scripts import repro_manifest


def test_hashes_artifacts_and_records_invocation(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"openpi")
    monkeypatch.setattr(
        repro_manifest,
        "git_output",
        lambda *args: "" if args[0] == "status" else "15a9616a00943ada6c20a0f158e3adb39df2ccac",
    )
    args = argparse.Namespace(
        run_id="test",
        image="image",
        image_digest="sha256:test",
        instance_type="g7e.4xlarge",
        instance_id="i-test",
        region="us-east-2",
        dataset="dataset",
        dataset_revision="revision",
        training_config="config",
        seed=7,
        steps=5,
        command="python train.py --steps 5",
        cost_reservation="reservation",
        actual_cost_usd=None,
        metrics_json='{"loss": 1.0}',
        artifact=[artifact],
    )
    manifest = repro_manifest.create_manifest(args)
    assert manifest["source"]["dirty"] is False
    assert manifest["experiment"]["command_argv"] == ["python", "train.py", "--steps", "5"]
    assert manifest["artifacts"][0]["sha256"] == "84ef98c9061878e43674ee319fbe773769b185118bc5b150e38c8cce69055cc0"
