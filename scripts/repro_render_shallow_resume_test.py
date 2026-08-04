import copy

import pytest

from scripts import repro_render_shallow_resume
from scripts.repro_render_snapflow_test import accepted_checkpoint
from scripts.repro_render_snapflow_test import artifact
from scripts.repro_render_snapflow_test import base_spec


def controller_source() -> dict:
    return {
        "s3_uri": "s3://pi05-repro-752160877725-us-east-2/source/controller-b19.bundle",
        "version_id": "controller-b19-v1",
        "sha256": "9" * 64,
        "commit": "8" * 40,
    }


@pytest.mark.parametrize(
    ("track", "source_step", "target_step"),
    [("libero", 5_000, 10_000), ("droid", 2_000, 5_000)],
)
def test_rendered_long_lane_resume_is_worker_valid(track, source_step, target_step):
    rendered = repro_render_shallow_resume.render_shallow_resume_spec(
        base_spec(track),
        accepted_checkpoint(track, source_step),
        controller_source(),
        track=track,
        target_step=target_step,
        run_id=f"{track}-shallow-long-{target_step}",
    )

    config = f"pi05_{track}_l09_distill"
    command = rendered["container"]["command"]
    assert command[:3] == ["python", "scripts/train_pytorch.py", config]
    assert command[command.index("--num-train-steps") + 1] == str(target_step)
    assert rendered["resume_checkpoint"] == {
        "artifact_name": "accepted_shallow",
        "target": f"{config}/{track}-shallow/{source_step}",
    }
    assert rendered["expected_outputs"][0]["publish_destination"] == (f"{config}/{track}-shallow/{target_step}")
    assert rendered["controller_source"] == controller_source()
    if track == "droid":
        assert command[-1] == "--no-wandb-enabled"
        assert "--num-workers" not in command
    else:
        assert "--no-wandb-enabled" not in command


def test_renderer_rejects_skipped_gate_and_missing_teacher():
    with pytest.raises(repro_render_shallow_resume.RenderError, match="must follow"):
        repro_render_shallow_resume.render_shallow_resume_spec(
            base_spec("droid"),
            accepted_checkpoint("droid", 2_000),
            controller_source(),
            track="droid",
            target_step=10_000,
            run_id="droid-shallow-long-10000",
        )

    missing = copy.deepcopy(base_spec("libero"))
    missing["artifacts"] = [item for item in missing["artifacts"] if item["name"] != "libero_teacher_pytorch"]
    with pytest.raises(repro_render_shallow_resume.RenderError, match="libero_teacher_pytorch"):
        repro_render_shallow_resume.render_shallow_resume_spec(
            missing,
            accepted_checkpoint("libero", 5_000),
            controller_source(),
            track="libero",
            target_step=10_000,
            run_id="libero-shallow-long-10000",
        )


def test_renderer_rejects_checkpoint_name_collision():
    checkpoint = artifact("libero", "checkpoint", "pi05_libero_l09_distill/libero-shallow/5000")
    with pytest.raises(repro_render_shallow_resume.RenderError, match="collides"):
        repro_render_shallow_resume.render_shallow_resume_spec(
            base_spec("libero"),
            checkpoint,
            controller_source(),
            track="libero",
            target_step=10_000,
            run_id="libero-shallow-long-10000",
        )
