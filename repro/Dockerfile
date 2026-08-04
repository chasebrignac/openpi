ARG BASE_IMAGE=763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training@sha256:f861e28dd9b8ecc86722bd37306b2f7b7e74336bc9c98dba37223cf031d04a7a
FROM ${BASE_IMAGE}

ARG SOURCE_SHA
ARG OPENPI_SHA=15a9616a00943ada6c20a0f158e3adb39df2ccac
ARG LEROBOT_RUNTIME=v2
ARG LEROBOT_SHA
ARG LEROBOT_V2_SHA=0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
ARG LEROBOT_V3_SHA=0b067df57d21d3a02d6c511f1609172fa39ac29b
ARG PALIGEMMA_TOKENIZER_SHA256=8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6
ARG ONNXRUNTIME_GPU_VERSION=1.26.0

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/cache/huggingface \
    OPENPI_DATA_HOME=/opt/openpi-cache \
    PYTHONPATH=/workspace/openpi/src:/workspace/openpi:/opt/openpi/src:/opt/openpi \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends curl ffmpeg git git-lfs jq rsync \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir uv==0.11.32 \
    && install -d -m 0777 /cache

WORKDIR /opt/openpi

# Copy only dependency metadata before the expensive frozen installation. This
# layer is shared by the v2/v3 builds and survives ordinary source fixes during
# the manual replay; a dependency edit still invalidates it as it should.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY packages/openpi-client/pyproject.toml packages/openpi-client/pyproject.toml

# The AWS DLC already contains the exact torch 2.7.1 / torchvision 0.22.1 /
# CUDA 12.8 stack. The PyPI lock's x86-64 torch dependency closure is CUDA 12.6,
# so explicitly exclude the DLC-provided packages rather than downgrading its
# native libraries. The lock also selects the `gym-aloha` simulator dependency
# branch, whose MuJoCo 2.3.7 has no CPython 3.12 Linux wheel. Policy training,
# conversion, and serving do not use that branch; LIBERO and RoboLab each
# provide a separate pinned simulator runtime. Pruning its root removes the
# complete unused branch rather than leaving an installed package with a
# missing dependency. Exporting from the frozen lock still pins every installed
# project dependency. Workspace packages are installed without dependency
# resolution after the source tree is copied below.
# ONNX 1.19 requires ml-dtypes APIs newer than OpenPI's pinned 0.4.1; ONNX 1.17
# supports opset 21 (required for ModelOpt FP8) without that conflict.
RUN printf '%s\n' \
        torch torchvision \
        nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvcc-cu12 \
        nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 \
        nvidia-cufft-cu12 nvidia-cufile-cu12 nvidia-curand-cu12 \
        nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 \
        nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 \
        > /tmp/dlc-provided-requirements.txt \
    && uv export --frozen --no-dev --no-emit-workspace --no-emit-package lerobot \
        --prune gym-aloha \
        --output-file /tmp/openpi-locked-requirements.txt \
    && uv pip install --system --no-cache-dir \
        --requirements /tmp/openpi-locked-requirements.txt \
        --excludes /tmp/dlc-provided-requirements.txt \
        --require-hashes \
    && uv pip install --system --no-cache-dir \
        huggingface-hub==0.33.1 \
        onnx==1.17.0 \
        onnxruntime-gpu==${ONNXRUNTIME_GPU_VERSION} \
        pytest==8.4.1 \
    && rm -f /tmp/dlc-provided-requirements.txt /tmp/openpi-locked-requirements.txt

# Golden-corpus generation invokes the PaliGemma tokenizer. Cache its exact
# bytes in the image so every `--network none` conversion/equivalence run is
# self-contained and does not resolve a mutable GCS object at runtime.
RUN mkdir -p "${OPENPI_DATA_HOME}/big_vision" \
    && curl --fail --location --silent --show-error \
      https://storage.googleapis.com/big_vision/paligemma_tokenizer.model \
      --output "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model" \
    && echo "${PALIGEMMA_TOKENIZER_SHA256}  ${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model" \
      | sha256sum --check --strict \
    && chmod 0777 "${OPENPI_DATA_HOME}" "${OPENPI_DATA_HOME}/big_vision" \
    && chmod 0444 "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model"

# MolmoAct2 uses LeRobot v3.0, while the released LIBERO dataset is v2.0.
# Validate the exact runtime/revision pair before installing it. The v3 data
# dependencies are isolated from OpenPI's NumPy<2/JAX constraint, and the
# unused rerun UI stays on the compatible lock version.
RUN case "${LEROBOT_RUNTIME}:${LEROBOT_SHA}" in \
      "v2:${LEROBOT_V2_SHA}") \
        grep -Fq "${LEROBOT_SHA}" uv.lock \
        && uv pip install --system --no-cache-dir --no-deps \
          "lerobot @ git+https://github.com/huggingface/lerobot.git@${LEROBOT_SHA}" \
        ;; \
      "v3:${LEROBOT_V3_SHA}") \
        uv pip install --system --no-cache-dir \
          accelerate==1.14.0 av==15.1.0 datasets==4.1.1 fsspec==2025.9.0 \
          huggingface-hub==0.35.3 pyarrow==25.0.0 \
        && uv pip install --system --no-cache-dir --reinstall --no-deps \
          "lerobot @ git+https://github.com/huggingface/lerobot.git@${LEROBOT_SHA}" \
        ;; \
      *) echo "LeRobot runtime/revision pair is not the pinned reproduction pair" >&2; exit 2 ;; \
    esac

# Source changes invalidate only the small editable-install and verification
# tail, not the frozen dependency layers above.
COPY . /opt/openpi
RUN printf '%s\n' "${SOURCE_SHA}" | grep -Eq '^[0-9a-f]{40}$' \
    && uv pip install --system --no-cache-dir --no-deps \
      --editable packages/openpi-client --editable .

# OpenPI pins transformers but also carries two source replacements required by
# its PyTorch pi0 implementation. Apply them inside the image exactly as the
# upstream installation instructions require.
RUN TRANSFORMERS_DIR="$(python -c 'import pathlib, transformers; print(pathlib.Path(transformers.__file__).parent)')" \
    && cp -r src/openpi/models_pytorch/transformers_replace/* "${TRANSFORMERS_DIR}/"

RUN python -c "import importlib.metadata as m, importlib.util, onnx, onnxruntime, torch; expected = {'jax': '0.5.3', 'jaxlib': '0.5.3', 'onnx': '1.17.0', 'onnxruntime-gpu': '${ONNXRUNTIME_GPU_VERSION}', 'pytest': '8.4.1', 'torch': '2.7.1', 'torchcodec': '0.4.0', 'torchvision': '0.22.1', 'transformers': '4.53.2'}; actual = {name: m.version(name).split('+', 1)[0] for name in expected}; print(actual, torch.version.cuda, onnx.__version__, onnxruntime.__version__); assert actual == expected, (actual, expected); assert all(importlib.util.find_spec(name) is None for name in ('gym_aloha', 'mujoco')); assert onnx.__version__ == expected['onnx']; assert onnxruntime.__version__ == expected['onnxruntime-gpu']; assert torch.version.cuda == '12.8', torch.version.cuda" \
    && python -c "from openpi.training import config; assert config.get_config('pi05_libero_l09_distill').name == 'pi05_libero_l09_distill'; assert config.get_config('pi05_droid_l09_distill').name == 'pi05_droid_l09_distill'" \
    && python scripts/smoke_lerobot_video.py \
    && if [ "${LEROBOT_RUNTIME}" = "v3" ]; then \
         python -c "import importlib.metadata as m; from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata; expected = {'av': '15.1.0', 'datasets': '4.1.1', 'huggingface-hub': '0.35.3', 'lerobot': '0.4.3', 'pyarrow': '25.0.0'}; actual = {name: m.version(name) for name in expected}; print(LeRobotDataset, LeRobotDatasetMetadata, actual); assert actual == expected, (actual, expected)"; \
       else \
         python -c "import importlib.metadata as m; from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata; expected = {'av': '17.0.0', 'datasets': '3.6.0', 'huggingface-hub': '0.33.1', 'lerobot': '0.1.0', 'pyarrow': '20.0.0'}; actual = {name: m.version(name) for name in expected}; print(LeRobotDataset, LeRobotDatasetMetadata, actual); assert actual == expected, (actual, expected)"; \
       fi

LABEL org.opencontainers.image.revision=${SOURCE_SHA}
LABEL ai.openpi.upstream-revision=${OPENPI_SHA}
LABEL ai.openpi.image-purpose="policy"
LABEL ai.openpi.lerobot-runtime=${LEROBOT_RUNTIME}
LABEL ai.openpi.lerobot-revision=${LEROBOT_SHA}
LABEL ai.openpi.paligemma-tokenizer-sha256=${PALIGEMMA_TOKENIZER_SHA256}
LABEL ai.openpi.simulator-runtime="external"
LABEL ai.openpi.video-decoder="pyav"
LABEL ai.openpi.onnxruntime-gpu-version=${ONNXRUNTIME_GPU_VERSION}
LABEL org.opencontainers.image.title="pi05-aws-repro"

ENTRYPOINT []
CMD ["bash"]
