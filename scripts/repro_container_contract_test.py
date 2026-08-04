import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_training_container_pins_match_reproduction_config() -> None:
    reproduction = json.loads((ROOT / "repro/reproduction.json").read_text())
    dockerfile = (ROOT / "repro/Dockerfile").read_text()
    worker = (ROOT / "scripts/repro_worker.py").read_text()

    expected_args = {
        "BASE_IMAGE": reproduction["aws"]["base_container"]["uri"],
        "OPENPI_SHA": reproduction["source"]["openpi_commit"],
        "LEROBOT_V2_SHA": reproduction["source"]["lerobot_v2_commit"],
        "LEROBOT_V3_SHA": reproduction["source"]["lerobot_v3_commit"],
    }
    for name, value in expected_args.items():
        assert f"ARG {name}={value}" in dockerfile
    assert f'"v2": "{expected_args["LEROBOT_V2_SHA"]}"' in worker
    assert f'"v3": "{expected_args["LEROBOT_V3_SHA"]}"' in worker

    assert "LABEL ai.openpi.lerobot-runtime=${LEROBOT_RUNTIME}" in dockerfile
    assert "LABEL ai.openpi.lerobot-revision=${LEROBOT_SHA}" in dockerfile
    assert (
        f'LABEL ai.openpi.image-purpose="{reproduction["aws"]["base_container"]["worker_image_purpose"]}"' in dockerfile
    )
    assert "lerobot.git@${LEROBOT_SHA}" in dockerfile
    assert reproduction["source"]["lerobot_v2_commit"] in (ROOT / "uv.lock").read_text()

    tokenizer_pin = re.search(r"^ARG PALIGEMMA_TOKENIZER_SHA256=([0-9a-f]{64})$", dockerfile, re.MULTILINE)
    assert tokenizer_pin is not None
    assert "LABEL ai.openpi.paligemma-tokenizer-sha256=${PALIGEMMA_TOKENIZER_SHA256}" in dockerfile
    assert "sha256sum --check --strict" in dockerfile
    assert "HF_HOME=/cache/huggingface" in dockerfile
    assert "install -d -m 0777 /cache" in dockerfile
    assert "torch.version.cuda == '12.8'" in dockerfile
    assert "--excludes /tmp/dlc-provided-requirements.txt" in dockerfile
    assert "--no-emit-package lerobot" in dockerfile
    assert "--require-hashes" in dockerfile
    assert "--no-hashes" not in dockerfile
    assert "nvidia-cuda-runtime-cu12" in dockerfile
    assert "--prune gym-aloha" in dockerfile
    assert "('gym_aloha', 'mujoco')" in dockerfile
    assert "config.get_config('pi05_libero_l09_distill')" in dockerfile
    assert "config.get_config('pi05_droid_l09_distill')" in dockerfile
    assert 'LABEL ai.openpi.simulator-runtime="external"' in dockerfile
    assert reproduction["aws"]["base_container"]["video_decoder"] == "pyav"
    assert f'LABEL ai.openpi.video-decoder="{reproduction["aws"]["base_container"]["video_decoder"]}"' in dockerfile
    assert f'POLICY_VIDEO_DECODER = "{reproduction["aws"]["base_container"]["video_decoder"]}"' in worker
    ort_version = reproduction["aws"]["base_container"]["onnxruntime_gpu_version"]
    assert f"ARG ONNXRUNTIME_GPU_VERSION={ort_version}" in dockerfile
    assert "onnxruntime-gpu==${ONNXRUNTIME_GPU_VERSION}" in dockerfile
    assert "LABEL ai.openpi.onnxruntime-gpu-version=${ONNXRUNTIME_GPU_VERSION}" in dockerfile
    assert f'POLICY_ONNXRUNTIME_GPU_VERSION = "{ort_version}"' in worker
    assert "python scripts/smoke_lerobot_video.py" in dockerfile
    assert "from torchcodec.decoders import VideoDecoder" not in dockerfile
    for package_version in ("0.5.3", "1.17.0", ort_version, "2.7.1", "0.4.0", "0.22.1", "4.53.2"):
        assert package_version in dockerfile


def test_tensorrt_compiler_container_has_a_distinct_exact_toolchain_contract() -> None:
    reproduction = json.loads((ROOT / "repro/reproduction.json").read_text())
    dockerfile = (ROOT / "repro/Dockerfile.tensorrt").read_text()
    worker = (ROOT / "scripts/repro_worker.py").read_text()
    container = reproduction["aws"]["tensorrt_container"]
    toolchain = container["compiler_toolchain"]

    assert f'LABEL ai.openpi.image-purpose="{container["worker_image_purpose"]}"' in dockerfile
    expected_args = {
        "TENSORRT_VERSION": toolchain["tensorrt_version"],
        "CUDA_VERSION": toolchain["cuda_version"],
        "MODELOPT_VERSION": toolchain["modelopt_version"],
        "TORCH_VERSION": toolchain["torch_version"],
        "ONNX_VERSION": toolchain["onnx_version"],
        "ONNXRUNTIME_GPU_VERSION": toolchain["onnxruntime_gpu_version"],
    }
    for name, value in expected_args.items():
        assert f"ARG {name}={value}" in dockerfile
        assert f'"{name.lower()}": "{value}"' in worker

    labels = {
        "TENSORRT_VERSION": "ai.openpi.tensorrt-version",
        "CUDA_VERSION": "ai.openpi.cuda-version",
        "MODELOPT_VERSION": "ai.openpi.modelopt-version",
        "TORCH_VERSION": "ai.openpi.torch-version",
        "ONNX_VERSION": "ai.openpi.onnx-version",
        "ONNXRUNTIME_GPU_VERSION": "ai.openpi.onnxruntime-gpu-version",
    }
    for name, label in labels.items():
        assert f"LABEL {label}=${{{name}}}" in dockerfile
    # TensorRT 11 prints its version for ``trtexec --version`` but exits 1
    # because it then expects a model.  The image build must use a successful
    # executable/help signature smoke instead.
    assert "trtexec --version" not in dockerfile
    assert "command -v trtexec" in dockerfile
    assert "trtexec --help" in dockerfile
    assert "TensorRT.trtexec [TensorRT v110000] [b114]" in dockerfile
    assert "ai.openpi.lerobot-runtime" not in dockerfile
    assert "ai.openpi.lerobot-revision" not in dockerfile

    worker_runbook = (ROOT / "repro/WORKER_RUNBOOK.md").read_text()
    export_runbook = (ROOT / "repro/EXPORT_RUNBOOK.md").read_text()
    assert '"purpose": "tensorrt-compiler"' in worker_runbook
    assert '"purpose": "tensorrt-policy"' in export_runbook
    for runbook in (worker_runbook, export_runbook):
        for key, value in toolchain.items():
            assert f'"{key}": "{value}"' in runbook
    assert "compiler image that claims LeRobot" in worker_runbook.replace("\n", " ")
    assert "rejects a graph-only compiler image for" in export_runbook.replace("\n", " ")
    assert '"purpose": "policy"' in worker_runbook

    # Publication is allowed only after exercising both GPU execution paths,
    # not merely importing their Python modules.
    assert 'providers=["CUDAExecutionProvider"]' in export_runbook
    assert "session.disable_cpu_ep_fallback" in export_runbook
    assert "np.testing.assert_array_equal(actual, np.asarray([3], np.float32))" in export_runbook
    assert "--onnx=/tmp/pi05-cuda-smoke.onnx" in export_runbook
    assert "--saveEngine=/tmp/pi05-cuda-smoke.engine" in export_runbook
    assert "--iterations=1" in export_runbook
    assert 'grep -F "&&&& PASSED TensorRT.trtexec"' in export_runbook
    compiler_push = export_runbook.index('docker push "$ECR_REPOSITORY:$COMPILER_TAG"')
    assert export_runbook.index('providers=["CUDAExecutionProvider"]') < compiler_push
    assert export_runbook.index("--saveEngine=/tmp/pi05-cuda-smoke.engine") < compiler_push

    mirror = container["amd64_ecr_mirror"]
    assert mirror["uri"].startswith(
        f"{reproduction['aws']['account_id']}.dkr.ecr.{reproduction['aws']['region']}.amazonaws.com/"
    )
    assert "@sha256:" in mirror["uri"]
    assert mirror["source_index_digest"] == container["uri"].rsplit("@", 1)[1]
    assert mirror["platform"] == "linux/amd64"
    assert mirror["compressed_bytes"] > 0
    assert mirror["uri"].rsplit("@", 1)[1] in export_runbook


def test_official_publication_uses_committed_amd64_context_and_both_registries() -> None:
    runbook = (ROOT / "RUNBOOK.md").read_text()

    assert 'git archive --format=tar "$SOURCE_COMMIT" | docker build' in runbook
    assert "--platform linux/amd64" in runbook
    assert '--build-arg LEROBOT_SHA="$LIBERO_LEROBOT_SHA"' in runbook
    assert '--build-arg LEROBOT_SHA="$DROID_LEROBOT_SHA"' in runbook
    assert "763104351884.dkr.ecr.us-east-2.amazonaws.com" in runbook
    assert "752160877725.dkr.ecr.us-east-2.amazonaws.com" in runbook
    assert 'export LIBERO_TAG="libero-v2-$SOURCE_COMMIT"' in runbook
    assert 'export DROID_TAG="droid-v3-$SOURCE_COMMIT"' in runbook
    assert "ai.openpi.lerobot-runtime" in runbook
    assert "ai.openpi.lerobot-revision" in runbook
    assert "ai.openpi.image-purpose" in runbook
    assert "ai.openpi.paligemma-tokenizer-sha256" in runbook
    assert "ai.openpi.video-decoder" in runbook
    assert "ai.openpi.onnxruntime-gpu-version" in runbook
    assert "smoke_pyav_decoder()" in runbook
    assert "docker run --rm --gpus all --network none --user 1000:1000" in runbook
    assert "torch.cuda.is_available()" in runbook
    assert "jax.devices()" in runbook
    assert 'options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")' in runbook
    assert 'providers=["CUDAExecutionProvider"]' in runbook


def test_source_bundle_seed_and_single_corpus_contracts_are_documented() -> None:
    worker_runbook = (ROOT / "repro/WORKER_RUNBOOK.md").read_text()
    bootstrap = (ROOT / "repro/worker-bootstrap.sh").read_text()
    main_runbook = (ROOT / "RUNBOOK.md").read_text()
    training_runbook = (ROOT / "repro/TRAINING_RUNBOOK.md").read_text()
    export_runbook = (ROOT / "repro/EXPORT_RUNBOOK.md").read_text()

    assert 'test -z "$(git status --porcelain)"' in worker_runbook
    assert "status --porcelain=v1 --untracked-files=all" in bootstrap
    assert '"source_clean": True' in bootstrap
    assert 'git -C "${verify_repo}" bundle verify "${source_bundle}"' in bootstrap
    assert 'git bundle list-heads "${source_bundle}" HEAD' in bootstrap
    assert "source bundle HEAD mismatch" in bootstrap
    assert "git bundle list-heads /tmp/openpi.bundle HEAD" in worker_runbook
    assert '--version-id "$SOURCE_VERSION_ID"' in worker_runbook
    assert 'SOURCE_BUNDLE_KEY="source/openpi-$SOURCE_COMMIT.bundle"' in worker_runbook
    assert "--if-none-match '*'" in worker_runbook
    assert "SOURCE_FINAL_HISTORY_JSON" in worker_runbook
    assert "openpi-SOURCE_GIT_COMMIT.bundle" in worker_runbook
    assert "source-commit=$SOURCE_COMMIT,sha256=$SOURCE_BUNDLE_SHA256" in worker_runbook
    assert '"--seed", "42"' in worker_runbook
    assert '"seed": 42' in worker_runbook
    assert '"seed": 0' not in worker_runbook

    for canonical_name in ("libero-heldout.npz", "droid-heldout.npz"):
        assert canonical_name in main_runbook
        assert canonical_name in training_runbook
        assert canonical_name in export_runbook
    assert "pi05_libero.npz" not in main_runbook
    assert "pi05_droid_jointpos.npz" not in main_runbook
    assert "do not create a second framework-only corpus" in main_runbook
    assert "export REPRO_RUN_ID=pi05-aws-repro-001" in main_runbook
    assert 'export REPRO_RUN_ID="pi05-aws-repro-$ATTEMPT_ID"' in training_runbook
    assert main_runbook.count('--run-id "$REPRO_RUN_ID"') >= 2
    assert len(re.findall(r"^\s+--data-split-seed 42 \\$", main_runbook, re.MULTILINE)) == 2
    assert len(re.findall(r"^\s+--data-split-seed 42 \\$", training_runbook, re.MULTILINE)) == 2
    assert "positional record slice" in training_runbook
    assert main_runbook.count("--equivalence-report /mnt/openpi/evidence/") == 4

    assert '"purpose": "libero-evaluator"' in worker_runbook
    assert '"NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics"' in worker_runbook
    assert '"MUJOCO_EGL_DEVICE_ID": "0"' in worker_runbook


def test_compiled_droid_examples_use_the_digest_bound_retained_session_wrapper() -> None:
    runbook = (ROOT / "repro/EXPORT_RUNBOOK.md").read_text()

    assert 'export DROID_RUNTIME_IMAGE="$ECR_REPOSITORY@$DROID_POLICY_DIGEST"' in runbook
    assert 'export DROID_IMAGE_DIGEST="${DROID_RUNTIME_IMAGE##*@}"' in runbook
    assert "run_droid_phase()" in runbook
    wrapper = runbook[runbook.index("run_droid_phase()") : runbook.index("Every LIBERO `python ...` example")]
    for contract in (
        "status --porcelain",
        "rev-parse HEAD",
        "--gpus all --network none",
        'src="$PI05_SOURCE_CHECKOUT",dst=/workspace/openpi,readonly',
        "src=/mnt/openpi,dst=/mnt/openpi,readonly",
        "--env PI05_SOURCE_SHA",
        '--env PI05_IMAGE_DIGEST="$DROID_IMAGE_DIGEST"',
        "--env PI05_INSTANCE_ID --env PI05_INSTANCE_TYPE",
        '"$DROID_RUNTIME_IMAGE" "$@"',
    ):
        assert contract in wrapper

    required_commands = (
        "scripts/repro_make_calibration.py generate",
        "scripts/repro_make_calibration.py validate",
        "scripts/export_pi05_onnx.py",
        "scripts/repro_make_action_limits.py",
        "scripts/validate_pi05_onnx.py",
        "scripts/quantize_pi05_fp8.py",
        "scripts/build_tensorrt_engines.py",
        "scripts/benchmark_pi05_latency.py",
        "scripts/serve_tensorrt_policy.py",
    )
    for command in required_commands:
        assert f"run_droid_phase python {command}" in runbook or (
            command == "scripts/serve_tensorrt_policy.py"
            and f"run_droid_phase /opt/modelopt/bin/python {command}" in runbook
        )
    assert "compile with the final evaluator" not in runbook


def test_robolab_runbook_records_the_completed_paid_smoke() -> None:
    runbook = (ROOT / "repro/ROBOLAB_EVAL_RUNBOOK.md").read_text()

    assert "all 128 tests" in runbook
    assert "smoke_exit_code=0" in runbook
    assert "AIVks2vssJ5y8yT5.WDeKbiJA0rsvngM" in runbook
    assert "has not been launched or passed" not in runbook


def test_docker_context_excludes_credentials_and_generated_model_payloads() -> None:
    patterns = set((ROOT / ".dockerignore").read_text().splitlines())

    assert {
        ".env",
        ".env.*",
        ".git",
        ".venv",
        "artifacts",
        "checkpoints",
        "data",
        "output",
        "runs",
        "*.engine",
        "*.onnx",
        "*.safetensors",
        "*.pt",
    } <= patterns
