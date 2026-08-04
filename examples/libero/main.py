from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import logging
import math
import pathlib
import re

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    save_videos: bool = True
    results_out_path: str = ""  # Defaults to data/libero/results/<suite>-<stage>-seed-<seed>.jsonl.
    overwrite_results: bool = False
    stage: str = "base"  # Included in every result record; use the same seed for paired stages.
    runtime_contract_path: str = ""  # Reproduction image contract; empty only for local development.
    expected_libero_revision: str = ""  # Must accompany runtime_contract_path.

    seed: int = 7  # Random Seed (for reproducibility)


def make_pair_id(*, suite: str, task_id: int, init_index: int, seed: int) -> str:
    """Return the stage-independent identifier used to pair two rollouts."""
    return f"libero:{suite}:task-{task_id:03d}:init-{init_index:03d}:seed-{seed}"


def make_episode_record(
    *,
    suite: str,
    task: str,
    task_id: int,
    init_index: int,
    seed: int,
    stage: str,
    success: bool,
    steps: int,
    error: str | None = None,
    libero_revision: str | None = None,
) -> dict[str, object]:
    """Build one quality-report-compatible LIBERO episode record."""
    if not stage.strip():
        raise ValueError("stage must be non-empty")
    record: dict[str, object] = {
        "pair_id": make_pair_id(suite=suite, task_id=task_id, init_index=init_index, seed=seed),
        "stage": stage,
        "benchmark": "libero",
        "suite": suite,
        "task": task,
        "task_id": task_id,
        "success": bool(success),
        "seed": seed,
        "init_index": init_index,
        "steps": steps,
    }
    if error is not None:
        record["error"] = error
    if libero_revision is not None:
        record["libero_revision"] = libero_revision
    return record


def validate_runtime_contract(path: pathlib.Path, *, expected_libero_revision: str) -> dict[str, object]:
    """Fail closed if the simulator or dependency lock differs from the requested image contract."""
    if not expected_libero_revision:
        raise ValueError("expected_libero_revision must be non-empty when validating a runtime contract")
    try:
        contract = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"LIBERO runtime contract does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"LIBERO runtime contract is invalid JSON: {path}") from exc
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        raise ValueError("LIBERO runtime contract must be a schema-version-1 object")
    simulator = contract.get("simulator")
    if not isinstance(simulator, dict) or simulator.get("revision") != expected_libero_revision:
        raise ValueError("LIBERO runtime contract simulator revision mismatch")
    requirements = contract.get("requirements")
    if not isinstance(requirements, dict):
        raise ValueError("LIBERO runtime contract has no requirements identity")
    installed_path = pathlib.Path(str(requirements.get("installed_path", "")))
    expected_sha256 = requirements.get("sha256")
    if not installed_path.is_absolute() or not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("LIBERO runtime contract requirements identity is invalid")
    try:
        actual_sha256 = hashlib.sha256(installed_path.read_bytes()).hexdigest()
    except FileNotFoundError as exc:
        raise ValueError(f"LIBERO requirements lock does not exist: {installed_path}") from exc
    if actual_sha256 != expected_sha256:
        raise ValueError("LIBERO requirements lock hash differs from the runtime contract")
    return contract


def select_init_indices(*, requested: int, available: int, suite: str, task_id: int) -> range:
    """Select the first N fixed initial states, failing instead of silently truncating."""
    if requested <= 0:
        raise ValueError("num_trials_per_task must be positive")
    if requested > available:
        raise ValueError(
            f"{suite} task {task_id} has {available} fixed initial states, but {requested} trials were requested"
        )
    return range(requested)


def resolve_results_path(args: Args) -> pathlib.Path:
    if args.results_out_path:
        return pathlib.Path(args.results_out_path)
    stage_segment = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.stage).strip("-.")
    if not stage_segment:
        raise ValueError("stage must contain at least one filename-safe character")
    filename = f"{args.task_suite_name}-{stage_segment}-seed-{args.seed}.jsonl"
    return pathlib.Path("data/libero/results") / filename


class EpisodeResultWriter:
    """Flush every completed episode to an exclusive JSONL result file."""

    def __init__(self, output_path: pathlib.Path, *, overwrite: bool):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self._stream = output_path.open("w" if overwrite else "x", encoding="utf-8")
        self.count = 0

    def write(self, record: dict[str, object]) -> None:
        self._stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self._stream.flush()
        self.count += 1

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> EpisodeResultWriter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()


def max_steps_for_suite(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220  # longest training demo has 193 steps
    if task_suite_name == "libero_object":
        return 280  # longest training demo has 254 steps
    if task_suite_name == "libero_goal":
        return 300  # longest training demo has 270 steps
    if task_suite_name == "libero_10":
        return 520  # longest training demo has 505 steps
    if task_suite_name == "libero_90":
        return 400  # longest training demo has 373 steps
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _get_task_suite(task_suite_name: str):
    # Keep LIBERO imports lazy so result-schema tests do not need the simulator installed.
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()[task_suite_name]()


def _make_client(host: str, port: int):
    return _websocket_client_policy.WebsocketClientPolicy(host, port)


def eval_libero(args: Args) -> None:
    if not args.stage.strip():
        raise ValueError("stage must be non-empty")
    if bool(args.runtime_contract_path) != bool(args.expected_libero_revision):
        raise ValueError("runtime_contract_path and expected_libero_revision must be provided together")
    if args.runtime_contract_path:
        validate_runtime_contract(
            pathlib.Path(args.runtime_contract_path), expected_libero_revision=args.expected_libero_revision
        )

    # The fixed init-state index and this environment seed define a pair. Start the
    # policy server with the same seed for both stages as documented in the runbook.
    np.random.seed(args.seed)

    task_suite = _get_task_suite(args.task_suite_name)
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")
    max_steps = max_steps_for_suite(args.task_suite_name)

    # Validate the whole run before creating its output file. This prevents a
    # requested 50-trial run from silently becoming a shorter result set.
    task_plan = []
    for task_id in range(num_tasks_in_suite):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        init_indices = select_init_indices(
            requested=args.num_trials_per_task,
            available=len(initial_states),
            suite=args.task_suite_name,
            task_id=task_id,
        )
        task_plan.append((task_id, task, initial_states, init_indices))

    expected_episodes = num_tasks_in_suite * args.num_trials_per_task
    results_path = resolve_results_path(args)
    if results_path.exists() and not args.overwrite_results:
        raise FileExistsError(f"result file already exists: {results_path}")
    logging.info("Episode results: %s", results_path)
    if args.save_videos:
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    client = _make_client(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    with EpisodeResultWriter(results_path, overwrite=args.overwrite_results) as result_writer:
        for task_id, task, initial_states, init_indices in tqdm.tqdm(task_plan):
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            task_episodes, task_successes = 0, 0
            try:
                for init_index in tqdm.tqdm(init_indices):
                    logging.info("\nTask: %s", task_description)
                    action_plan = collections.deque()
                    replay_images = []
                    success = False
                    rollout_error = None
                    t = 0

                    logging.info("Starting episode %d...", task_episodes + 1)
                    try:
                        env.reset()
                        obs = env.set_init_state(initial_states[init_index])

                        while t < max_steps + args.num_steps_wait:
                            # Do nothing first because the simulator drops objects and they need time to settle.
                            if t < args.num_steps_wait:
                                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                                t += 1
                                continue

                            # IMPORTANT: rotate 180 degrees to match train preprocessing
                            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                            img = image_tools.convert_to_uint8(
                                image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                            )
                            wrist_img = image_tools.convert_to_uint8(
                                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                            )

                            if args.save_videos:
                                replay_images.append(img)

                            if not action_plan:
                                element = {
                                    "observation/image": img,
                                    "observation/wrist_image": wrist_img,
                                    "observation/state": np.concatenate(
                                        (
                                            obs["robot0_eef_pos"],
                                            _quat2axisangle(obs["robot0_eef_quat"]),
                                            obs["robot0_gripper_qpos"],
                                        )
                                    ),
                                    "prompt": str(task_description),
                                }

                                action_chunk = client.infer(element)["actions"]
                                assert len(action_chunk) >= args.replan_steps, (
                                    f"We want to replan every {args.replan_steps} steps, but policy only predicts "
                                    f"{len(action_chunk)} steps."
                                )
                                action_plan.extend(action_chunk[: args.replan_steps])

                            action = action_plan.popleft()
                            obs, _, done, _ = env.step(action.tolist())
                            t += 1
                            if done:
                                success = True
                                break
                    except Exception:
                        # Transport, model-output, and simulator exceptions are infrastructure failures,
                        # not unsuccessful policy rollouts. Abort so a broken base/final pair cannot
                        # masquerade as matching quality.
                        logging.exception("Rollout infrastructure failed; aborting evaluation")
                        raise

                    task_episodes += 1
                    total_episodes += 1
                    if success:
                        task_successes += 1
                        total_successes += 1

                    result_writer.write(
                        make_episode_record(
                            suite=args.task_suite_name,
                            task=str(task_description),
                            task_id=task_id,
                            init_index=init_index,
                            seed=args.seed,
                            stage=args.stage,
                            success=success,
                            steps=t,
                            error=rollout_error,
                            libero_revision=args.expected_libero_revision or None,
                        )
                    )

                    if args.save_videos and replay_images:
                        suffix = "success" if success else "failure"
                        stage_segment = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.stage).strip("-.")
                        video_name = (
                            f"{args.task_suite_name}-task-{task_id:03d}-init-{init_index:03d}-"
                            f"{stage_segment}-{suffix}.mp4"
                        )
                        try:
                            imageio.mimwrite(
                                pathlib.Path(args.video_out_path) / video_name,
                                [np.asarray(x) for x in replay_images],
                                fps=10,
                            )
                        except Exception:
                            logging.exception("Could not write replay video %s", video_name)

                    logging.info("Success: %s", success)
                    logging.info("# episodes completed so far: %d", total_episodes)
                    logging.info(
                        "# successes: %d (%.1f%%)",
                        total_successes,
                        total_successes / total_episodes * 100,
                    )
            finally:
                env.close()

            logging.info("Current task success rate: %s", float(task_successes) / float(task_episodes))
            logging.info("Current total success rate: %s", float(total_successes) / float(total_episodes))

        if result_writer.count != expected_episodes:
            raise RuntimeError(f"wrote {result_writer.count} episodes; expected exactly {expected_episodes}")

    logging.info("Total success rate: %s", float(total_successes) / float(total_episodes))
    logging.info("Total episodes: %d", total_episodes)


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
