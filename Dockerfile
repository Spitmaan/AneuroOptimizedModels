# =============================================================================
# ANeurologic Phase 5 — Advanced Optimization
# Base: openpi:r36.4.tegra-aarch64-cu126-22.04-cuda_12.6
#   Already present on Jetson Orin Nano — no download needed
#   Ubuntu 22.04, Python 3.10, CUDA 12.6, JetPack 6.2 (L4T r36.4.7)
#   PyTorch 2.5.0-nv24.08 (installed via pip from Jetson AI Lab index)
#
# Includes:
#   - Go 1.22.5 (arm64)                    — Stage 4 Go inference server
#   - TensorRT-LLM v0.12.0-jetson (clone)  — Stage 5 engine build
#   - lm-eval ≥0.4.3                       — Stage 2 reasoning benchmarks
#   - QJL (AAAI 2025) + KIVI (ICML 2024)  — Stage 3 KV-cache compression
#   - gollama.cpp (purego)                 — Stage 4 Go bindings
#   - FastAPI + openai client              — Stage 6 teacher API
# =============================================================================

FROM openpi:r36.4.tegra-aarch64-cu126-22.04-cuda_12.6

LABEL maintainer="ANeurologic Edge AI Initiative"
LABEL phase="Phase 5 - Advanced Optimization"
LABEL jetpack="6.2 (L4T r36.4.7)"

WORKDIR /workspace

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl \
    build-essential cmake ninja-build \
    python3-dev python3-pip \
    libssl-dev libffi-dev libopenblas-dev \
    libopenmpi-dev pkg-config \
    ca-certificates jq htop \
    && rm -rf /var/lib/apt/lists/*

# ── libcudss stub (same workaround as Phase 3 & 4) ───────────────────────────
COPY stub_cudss.c /tmp/stub_cudss.c
RUN gcc -shared -fPIC -Wl,-soname,libcudss.so.0 \
        -o /usr/local/lib/libcudss.so.0 /tmp/stub_cudss.c \
    && ln -sf /usr/local/lib/libcudss.so.0 /usr/local/lib/libcudss.so \
    && ldconfig \
    && rm /tmp/stub_cudss.c

# ── PyTorch (Jetson CUDA 12.6 wheel, same as Phase 4) ────────────────────────
RUN pip3 install --no-cache-dir torch torchvision

# ── Go 1.22.5 (arm64) ────────────────────────────────────────────────────────
ARG GO_VERSION=1.22.5
RUN wget -q https://go.dev/dl/go${GO_VERSION}.linux-arm64.tar.gz \
    && tar -C /usr/local -xzf go${GO_VERSION}.linux-arm64.tar.gz \
    && rm go${GO_VERSION}.linux-arm64.tar.gz
ENV PATH="/usr/local/go/bin:/go/bin:${PATH}"
ENV GOPATH="/go"
ENV GOMODCACHE="/go/pkg/mod"
RUN go version

# ── Python requirements ───────────────────────────────────────────────────────
COPY requirements.txt /workspace/requirements.txt
RUN pip3 install --no-cache-dir -r /workspace/requirements.txt

# ── lm-eval (EleutherAI evaluation harness) ───────────────────────────────────
# GSM8K (math/logic) and ARC-Challenge (science reasoning) support built-in
RUN pip3 install --no-cache-dir "lm-eval>=0.4.3"

# ── Clone TensorRT-LLM v0.12.0-jetson ────────────────────────────────────────
# Note: pip wheel not available for aarch64 — Stage 5 will build from source.
# Cloning here so the source tree is baked into the image.
RUN git clone --branch v0.12.0-jetson --depth 1 \
    https://github.com/NVIDIA/TensorRT-LLM.git \
    /workspace/TensorRT-LLM \
    && echo "TRT-LLM v0.12.0-jetson cloned ($(wc -l < /workspace/TensorRT-LLM/README4Jetson.md) lines in README4Jetson.md)"

# Install TRT-LLM Jetson Python deps (subset that doesn't require build)
RUN if [ -f /workspace/TensorRT-LLM/requirements-jetson.txt ]; then \
        pip3 install --no-cache-dir \
            -r /workspace/TensorRT-LLM/requirements-jetson.txt 2>/dev/null || true; \
    fi

# ── Clone KV-cache quantization repos (Stage 3) ──────────────────────────────
# QJL: 1-bit JL-transform KV quantization (AAAI 2025, Google Research)
RUN git clone --depth 1 \
    https://github.com/amirzandieh/QJL.git \
    /workspace/repos/QJL

# KIVI: 2-bit asymmetric KV quantization (ICML 2024) — battle-tested fallback
RUN git clone --depth 1 \
    https://github.com/jy-yuan/KIVI.git \
    /workspace/repos/KIVI && \
    cd /workspace/repos/KIVI && \
    pip3 install --no-cache-dir -e . 2>/dev/null || true

# ── Clone gollama.cpp — Go bindings for llama.cpp via purego (Stage 4) ───────
RUN mkdir -p /go/src/github.com/dianlight && \
    git clone --depth 1 \
    https://github.com/dianlight/gollama.cpp.git \
    /go/src/github.com/dianlight/gollama.cpp

# ── Directory layout ─────────────────────────────────────────────────────────
RUN mkdir -p \
    /workspace/scripts \
    /workspace/models/hf_cache \
    /workspace/models/gguf \
    /workspace/models/trt_engines \
    /workspace/outputs/logs \
    /workspace/outputs/reports \
    /workspace/go_server

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONPATH="/workspace:/workspace/repos/QJL:/workspace/repos/KIVI"
ENV HF_HOME="/workspace/models/hf_cache"
ENV TRANSFORMERS_CACHE="/workspace/models/hf_cache"
# Jetson UMA memory management (proven in Phase 4)
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"
ENV CUDA_MODULE_LOADING=LAZY
ENV OMP_NUM_THREADS=4
ENV PYTHONUNBUFFERED=1

COPY scripts/ /workspace/scripts/
# go_server/ is built in Stage 4 and mounted via docker-compose volume

CMD ["/bin/bash"]
