ARG BASE_IMAGE=763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training@sha256:f861e28dd9b8ecc86722bd37306b2f7b7e74336bc9c98dba37223cf031d04a7a
FROM ${BASE_IMAGE}

ARG OPENPI_SHA=15a9616a00943ada6c20a0f158e3adb39df2ccac
LABEL org.opencontainers.image.revision=${OPENPI_SHA}
LABEL org.opencontainers.image.title="pi05-aws-repro"

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/mnt/openpi-cache/huggingface \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends git git-lfs jq rsync \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir uv==0.11.32

WORKDIR /opt/openpi
COPY . /opt/openpi

# The AWS DLC already contains the exact torch 2.7.1 / CUDA 12.8 base.  Installing
# into its system environment avoids a second copy of the multi-gigabyte CUDA stack.
RUN uv pip install --system --no-cache-dir -e . \
    && uv pip install --system --no-cache-dir \
        huggingface-hub==0.33.1 \
        onnx==1.19.1 \
        onnxruntime-gpu==1.28.0 \
        "nvidia-modelopt[onnx]==0.45.0" \
        pytest==8.4.1

RUN test "$(python -c 'import torch; print(torch.__version__.split("+", 1)[0])')" = "2.7.1" \
    && python -c "import onnx, torch; print(torch.__version__, onnx.__version__)"

ENTRYPOINT []
CMD ["bash"]
