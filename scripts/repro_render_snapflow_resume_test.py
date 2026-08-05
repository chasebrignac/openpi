import copy

import pytest

from scripts import repro_render_snapflow_resume
from scripts.repro_render_shallow_resume_test import controller_source
from scripts.repro_render_snapflow_test import artifact
from scripts.repro_render_snapflow_test import base_spec


def accepted_snapflow(track: str, step: int) -> dict:
    config = f"pi05_{track}_l09_snapflow"
    descriptor = artifact("accepted_snapflow", "checkpoint", f"{config}/{track}-snapflow/{step}")
    descriptor["revision"] = "e" * 64
    return descriptor


@pytest.mark.parametrize("track", ["libero", "droid"])
def test_rendered_snapflow_resume_is_worker_valid(track):
    rendered = repro_render_snapflow_resume.render_snapflow_resume_spec(
        base_spec(track),
        accepted_snapflow(track, 5_000),
        controller_source(),
        track=track,
        target_step=10_000,
        run_id=f"{track}-snapflow-long-10000",
    )

    config = f"pi05_{track}_l09_snapflow"
    command = rendered["container"]["command"]
    assert command[:3] == ["python", "scripts/train_pytorch.py", config]
    assert command[command.index("--num-train-steps") + 1] == "10000"
    assert command[command.index("--save-interval") + 1] == "5000"
    assert rendered["resume_checkpoint"] == {
        "artifact_name": "accepted_snapflow",
        "target": f"{config}/{track}-snapflow/5000",
    }
    assert rendered["expected_outputs"][0]["publish_destination"] == f"{config}/{track}-snapflow/10000"
    assert rendered["controller_source"] == controller_source()
    assert [item["name"] for item in rendered["artifacts"]] == [
        track,
        "libero_teacher_jax" if track == "libero" else "droid_jointpos_teacher_jax",
        "accepted_snapflow",
    ]
    if track == "droid":
        assert command[-3:] == ["--num-workers", "0", "--no-wandb-enabled"]
    else:
        assert "--num-workers" not in command


def test_renderer_rejects_skipped_gate_and_missing_jax():
    with pytest.raises(repro_render_snapflow_resume.RenderError, match="must follow"):
        repro_render_snapflow_resume.render_snapflow_resume_spec(
            base_spec("libero"),
            accepted_snapflow("libero", 5_000),
            controller_source(),
            track="libero",
            target_step=20_000,
            run_id="libero-snapflow-long-20000",
        )

    missing = copy.deepcopy(base_spec("droid"))
    missing["artifacts"] = [item for item in missing["artifacts"] if item["name"] != "droid_jointpos_teacher_jax"]
    with pytest.raises(repro_render_snapflow_resume.RenderError, match="droid_jointpos_teacher_jax"):
        repro_render_snapflow_resume.render_snapflow_resume_spec(
            missing,
            accepted_snapflow("droid", 5_000),
            controller_source(),
            track="droid",
            target_step=10_000,
            run_id="droid-snapflow-long-10000",
        )


def test_renderer_rejects_wrong_config_and_checkpoint_name_collision():
    with pytest.raises(repro_render_snapflow_resume.RenderError, match="accepted source must be"):
        repro_render_snapflow_resume.render_snapflow_resume_spec(
            base_spec("libero"),
            accepted_snapflow("droid", 5_000),
            controller_source(),
            track="libero",
            target_step=10_000,
            run_id="libero-snapflow-long-10000",
        )

    checkpoint = artifact("libero", "checkpoint", "pi05_libero_l09_snapflow/libero-snapflow/5000")
    with pytest.raises(repro_render_snapflow_resume.RenderError, match="collides"):
        repro_render_snapflow_resume.render_snapflow_resume_spec(
            base_spec("libero"),
            checkpoint,
            controller_source(),
            track="libero",
            target_step=10_000,
            run_id="libero-snapflow-long-10000",
        )
