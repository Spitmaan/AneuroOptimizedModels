# =============================================================================
# ANeurologic Phase 5 — Advanced Optimization
# Base: NVIDIA L4T PyTorch (JetPack 6.x, CUDA 12.6, aarch64)
# Includes: TensorRT-LLM v0.12.0-jetson, lm-eval, Go 1.22, KV-cache quant tools
# =============================================================================

FROM nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.3-py3

LABEL maintainer="ANeurologic"
LABEL version="5.0"
LABEL description="Phase 5: Advanced Optimization — TurboQuant, TensorRT-LLM, Go Serving, Distillation"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential cmake ninja-build \
    python3-pip python3-dev python3-setuptools \
    libssl-dev libffi-dev libopenblas-dev \
    ca-certificates gnupg lsb-release \
    jq htop nvtop \
    && rm -rf /var/lib/apt/lists/*

# ── Go 1.22.5 (arm64) ────────────────────────────────────────────────────────
ARG GO_VERSION=1.22.5
RUN wget -q https://go.dev/dl/go${GO_VERSION}.linux-arm64.tar.gz \
    && tar -C /usr/local -xzf go${GO_VERSION}.linux-arm64.tar.gz \
    && rm go${GO_VERSION}.linux-arm64.tar.gz

ENV PATH="/usr/local/go/bin:/go/bin:${PATH}"
ENV GOPATH="/go"
ENV GOMODCACHE="/go/pkg/mod"

# Verify Go installed
RUN go version

# ── Python — core ML stack ───────────────────────────────────────────────────
# Upgrade pip first; then install in stages to keep layer cache useful
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt /workspace/requirements.txt
RUN pip3 install --no-cache-dir -r /workspace/requirements.txt

# ── lm-eval (EleutherAI Language Model Evaluation Harness) ───────────────────
# Pinned for reproducibility; supports GSM8K and ARC out of the box
RUN pip3 install --no-cache-dir lm-eval>=0.4.3

# ── Clone TensorRT-LLM v0.12.0-jetson ────────────────────────────────────────
RUN git clone --branch v0.12.0-jetson --depth 1 \
    https://github.com/NVIDIA/TensorRT-LLM.git \
    /workspace/TensorRT-LLM \
    && echo "TensorRT-LLM v0.12.0-jetson cloned"

# Install TensorRT-LLM Python requirements (aarch64-compatible subset)
RUN pip3 install --no-cache-dir \
    -r /workspace/TensorRT-LLM/requirements-jetson.txt 2>/dev/null || \
    echo "WARNING: requirements-jetson.txt not found or partial install — OK for Stage 1"

# Try pip wheel for TRT-LLM (may only be available for x86; Jetson may need source build)
RUN pip3 install --no-cache-dir \
    --extra-index-url https://pypi.nvidia.com/ \
    tensorrt-llm==0.12.0 2>/dev/null || \
    echo "INFO: tensorrt-llm pip wheel not available for aarch64 — Stage 5 will build from source"

# ── Clone QJL (Google Research KV-cache quantization, AAAI 2025) ─────────────
RUN git clone --depth 1 \
    https://github.com/amirzandieh/QJL.git \
    /workspace/repos/QJL \
    && echo "QJL cloned"

# ── Clone KIVI (2-bit KV cache, ICML 2024 — battle-tested alternative) ───────
RUN git clone --depth 1 \
    https://github.com/jy-yuan/KIVI.git \
    /workspace/repos/KIVI \
    && echo "KIVI cloned"

# ── Clone gollama.cpp (Go bindings for llama.cpp via purego) ─────────────────
RUN mkdir -p /go/src/github.com/dianlight && \
    git clone --depth 1 \
    https://github.com/dianlight/gollama.cpp.git \
    /go/src/github.com/dianlight/gollama.cpp \
    && echo "gollama.cpp cloned"

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONPATH="/workspace:/workspace/repos/QJL:/workspace/repos/KIVI"
ENV WORKSPACE="/workspace"
ENV HF_HOME="/workspace/models/hf_cache"
ENV TRANSFORMERS_CACHE="/workspace/models/hf_cache"
# Jetson UMA memory optimizations
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"
ENV CUDA_MODULE_LOADING=LAZY
ENV OMP_NUM_THREADS=4

WORKDIR /workspace
CMD ["/bin/bash"]
