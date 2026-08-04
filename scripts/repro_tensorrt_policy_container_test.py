import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
POLICY_DOCKERFILE = ROOT / "repro/Dockerfile.tensorrt-policy"
COMPILER_DOCKERFILE = ROOT / "repro/Dockerfile.tensorrt"
TRAINING_DOCKERFILE = ROOT / "repro/Dockerfile"


def _arg(text: str, name: str) -> str:
    match = re.search(rf"^ARG {re.escape(name)}=([^\s]+)$", text, re.MULTILINE)
    assert match is not None, name
    return match.group(1)


def _label(text: str, name: str) -> str:
    match = re.search(rf'^LABEL {re.escape(name)}=(?:"([^"]*)"|([^\s]+))$', text, re.MULTILINE)
    assert match is not None, name
    return match.group(1) or match.group(2)


def _inputs():
    reproduction = json.loads((ROOT / "repro/reproduction.json").read_text())
    return reproduction, POLICY_DOCKERFILE.read_text(), COMPILER_DOCKERFILE.read_text()


def test_policy_derives_only_from_account_local_digest_pinned_compiler_at_same_source():
    reproduction, policy, _ = _inputs()
    account = reproduction["aws"]["account_id"]
    region = reproduction["aws"]["region"]

    assert policy.startswith(
        "# Combined TensorRT 11 engine runtime plus the minimal OpenPI transform and\n# WebSocket policy stack."
    )
    assert policy.count("ARG TENSORRT_COMPILER_IMAGE") == 2
    assert "FROM ${TENSORRT_COMPILER_IMAGE}" in policy
    assert "ARG TENSORRT_COMPILER_IMAGE=" not in policy
    assert "nvcr.io/" not in policy
    assert rf"^{account}\.dkr\.ecr\.{region}\.amazonaws\.com/" in policy
    assert "@sha256:[0-9a-f]{64}$" in policy
    assert 'test "${SOURCE_SHA}" = "${TENSORRT_COMPILER_SOURCE_SHA}"' in policy
    assert "> /tmp/parent-openpi-source.sha256" in policy
    assert "> /tmp/policy-openpi-source.sha256" in policy
    assert "cmp /tmp/parent-openpi-source.sha256 /tmp/policy-openpi-source.sha256" in policy
    assert _label(policy, "org.opencontainers.image.revision") == "${SOURCE_SHA}"
    assert _label(policy, "ai.openpi.parent-tensorrt-compiler-image") == "${TENSORRT_COMPILER_IMAGE}"
    assert _label(policy, "ai.openpi.parent-tensorrt-compiler-source-revision") == ("${TENSORRT_COMPILER_SOURCE_SHA}")
    assert _label(policy, "ai.openpi.parent-image-purpose") == "tensorrt-compiler"


def test_policy_retains_every_reproduction_compiler_toolchain_pin_and_label():
    reproduction, policy, compiler = _inputs()
    toolchain = reproduction["aws"]["tensorrt_container"]["compiler_toolchain"]
    contracts = {
        "TENSORRT_VERSION": ("tensorrt_version", "ai.openpi.tensorrt-version"),
        "CUDA_VERSION": ("cuda_version", "ai.openpi.cuda-version"),
        "MODELOPT_VERSION": ("modelopt_version", "ai.openpi.modelopt-version"),
        "TORCH_VERSION": ("torch_version", "ai.openpi.torch-version"),
        "ONNX_VERSION": ("onnx_version", "ai.openpi.onnx-version"),
        "ONNXRUNTIME_GPU_VERSION": (
            "onnxruntime_gpu_version",
            "ai.openpi.onnxruntime-gpu-version",
        ),
    }
    for argument, (config_key, label) in contracts.items():
        expected = toolchain[config_key]
        assert _arg(policy, argument) == expected
        assert _arg(compiler, argument) == expected
        assert _label(policy, label) == f"${{{argument}}}"
        assert _label(compiler, label) == f"${{{argument}}}"

    assert '/opt/modelopt/bin/python -c "import modelopt, onnx, onnxruntime, tensorrt, torch' in policy
    assert policy.count("tensorrt.__version__") >= 2
    assert policy.count("modelopt.__version__") >= 2
    assert "command -v trtexec" in policy
    assert "trtexec --version" not in policy


def test_runtime_is_frozen_trimmed_and_installed_into_inherited_modelopt_venv():
    _, policy, _ = _inputs()
    lock_sha = hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    assert _arg(policy, "UV_LOCK_SHA256") == lock_sha
    assert _label(policy, "ai.openpi.runtime-lock-sha256") == "${UV_LOCK_SHA256}"
    assert 'echo "${UV_LOCK_SHA256}  uv.lock" | sha256sum --check --strict' in policy
    assert "uv export --frozen --no-dev --no-emit-workspace" in policy
    assert "--require-hashes" in policy
    assert "--no-hashes" not in policy
    for package in ("lerobot", "gcsfs", "gym-aloha", "imageio", "opencv-python", "polars", "wandb"):
        assert f"--prune {package}" in policy

    assert "uv venv" not in policy
    assert policy.count("uv pip install --python /opt/modelopt/bin/python") == 3
    assert "--requirements /tmp/openpi-policy-locked-requirements.txt" in policy
    assert "--excludes /tmp/compiler-provided-requirements.txt" in policy
    for compiler_owned in (
        "jax-cuda12-pjrt",
        "jax-cuda12-plugin",
        "ml-dtypes",
        "modelopt",
        "nvidia-modelopt",
        "onnx",
        "onnxruntime",
        "onnxruntime-gpu",
        "torch",
        "torchcodec",
        "torchvision",
        "triton",
        "nvidia-cublas-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cudnn-cu12",
        "nvidia-nccl-cu12",
    ):
        assert compiler_owned in policy


def test_both_workspace_packages_are_source_late_no_dependency_installs():
    _, policy, _ = _inputs()
    source_copy = policy.index("COPY . /opt/openpi")
    locked_install = policy.index("--requirements /tmp/openpi-policy-locked-requirements.txt")
    tokenizer = policy.index("paligemma_tokenizer.model")
    lerobot_install = policy.index("lerobot @ git+https://github.com/huggingface/lerobot.git@${LEROBOT_SHA}")
    workspace_install = policy.index("/opt/openpi/packages/openpi-client /opt/openpi")
    assert locked_install < tokenizer < lerobot_install < source_copy < workspace_install
    workspace_run = policy[source_copy : policy.index("# These checks are imports")]
    assert "uv pip install --python /opt/modelopt/bin/python --no-cache-dir --no-deps" in workspace_run
    assert "/opt/openpi/packages/openpi-client /opt/openpi" in workspace_run
    assert "transformers_replace" in workspace_run


def test_lerobot_variants_are_exact_and_share_all_earlier_layers():
    reproduction, policy, _ = _inputs()
    training = TRAINING_DOCKERFILE.read_text()
    v2 = reproduction["source"]["lerobot_v2_commit"]
    v3 = reproduction["source"]["lerobot_v3_commit"]
    assert _arg(policy, "LEROBOT_V2_SHA") == v2
    assert _arg(policy, "LEROBOT_V3_SHA") == v3
    assert _arg(training, "LEROBOT_V2_SHA") == v2
    assert _arg(training, "LEROBOT_V3_SHA") == v3
    assert '"v2:${LEROBOT_V2_SHA}") EXPECTED_LEROBOT_VERSION=0.1.0' in policy
    assert '"v3:${LEROBOT_V3_SHA}") EXPECTED_LEROBOT_VERSION=0.4.3' in policy
    assert '"lerobot @ git+https://github.com/huggingface/lerobot.git@${LEROBOT_SHA}"' in policy
    assert "--no-cache-dir --no-deps" in policy
    assert _label(policy, "ai.openpi.lerobot-runtime") == "${LEROBOT_RUNTIME}"
    assert _label(policy, "ai.openpi.lerobot-revision") == "${LEROBOT_SHA}"
    assert policy.index("uv export --frozen") < policy.index('case "${LEROBOT_RUNTIME}:${LEROBOT_SHA}"')
    assert policy.index("PALIGEMMA_TOKENIZER_SHA256") < policy.index('case "${LEROBOT_RUNTIME}:${LEROBOT_SHA}"')


def test_tokenizer_is_the_same_cached_hash_pinned_policy_asset():
    _, policy, _ = _inputs()
    training = TRAINING_DOCKERFILE.read_text()
    assert _arg(policy, "PALIGEMMA_TOKENIZER_SHA256") == _arg(training, "PALIGEMMA_TOKENIZER_SHA256")
    assert "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model" in policy
    assert "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model" in policy
    assert "sha256sum --check --strict" in policy
    assert "chmod 0444" in policy
    assert _label(policy, "ai.openpi.paligemma-tokenizer-sha256") == "${PALIGEMMA_TOKENIZER_SHA256}"
    assert "HF_HOME=/cache/huggingface" in policy
    assert "install -d -m 0777 /cache" in policy


def test_policy_runtime_labels_entrypoint_and_import_smokes_are_distinct():
    reproduction, policy, _ = _inputs()
    assert _arg(policy, "OPENPI_SHA") == reproduction["source"]["openpi_commit"]
    assert _label(policy, "ai.openpi.upstream-revision") == "${OPENPI_SHA}"
    assert _label(policy, "ai.openpi.image-purpose") == "tensorrt-policy"
    assert _label(policy, "ai.openpi.policy-runtime") == "openpi-transform-websocket"
    assert _label(policy, "ai.openpi.policy-python") == "/opt/modelopt/bin/python"
    assert _label(policy, "ai.openpi.policy-protocol") == "openpi-policy-websocket-v1"
    assert "ENTRYPOINT []" in policy
    assert 'CMD ["bash"]' in policy
    assert "JAX_PLATFORMS=cpu" in policy
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility" in policy
    for import_contract in (
        "from openpi.exporting import tensorrt_policy",
        "from openpi.models.tokenizer import PaligemmaTokenizer",
        "from openpi.serving.websocket_policy_server import WebsocketPolicyServer",
        "from openpi.training import config",
        "from openpi_client import base_policy, msgpack_numpy",
        "pi05_libero_l09_snapflow",
        "pi05_droid_l09_snapflow",
    ):
        assert import_contract in policy


def test_build_smokes_do_not_claim_target_gpu_validation():
    _, policy, _ = _inputs()
    forbidden_build_checks = (
        "nvidia-smi",
        "torch.cuda.is_available",
        "torch.cuda.get_device",
        "deserialize_cuda_engine",
        "--gpus",
        "CUDAExecutionProvider",
    )
    for forbidden in forbidden_build_checks:
        assert forbidden not in policy
    assert "do not" in policy[policy.index("# These checks are imports") :].lower()
    assert "GPU validation" in policy
