# Stage 1 — Environment Setup & Verification

**Project:** ANeurologic Phase 5 — Advanced Optimization
**Hardware:** NVIDIA Jetson Orin Nano 8 GB
**Date:** 2026-03-28
**Results JSON:** `outputs/logs/stage1_env_results.json`

---

## Results Summary

| Check | Result | Value |
|-------|--------|-------|
| Python | ✅ | 3.10.12 |
| Platform | ✅ | aarch64 / Linux |
| PyTorch | ✅ | 2.10.0 |
| CUDA | ✅ | 12.6 (True) |
| GPU Device | ✅ | Orin — 7.99 GB VRAM |
| CUDA Max Allocatable | ✅ | ~3.0 GB |
| TensorRT | ✅ | System TensorRT present |
| TensorRT-LLM pip | ⚠️ | Not available for aarch64 — will build from source in Stage 5 |
| TRT-LLM source | ✅ | `/workspace/TensorRT-LLM` (v0.12.0-jetson cloned) |
| Go | ✅ | go1.22.5 linux/arm64 |
| gollama.cpp | ✅ | `/go/src/github.com/dianlight/gollama.cpp` cloned |
| lm-eval | ✅ | 0.4.11 |
| QJL repo | ✅ | 15 Python files |
| KIVI repo | ✅ | 21 Python files |
| transformers | ✅ | 5.4.0 |
| HuggingFace Hub | ✅ | Reachable |
| Power mode | ✅ | Jetson power mode active |
| Swap | ✅ | 23 GB swap available |
| System RAM | ✅ | 7.4 GiB total, 5.3 GiB available |

**Total: 19 checks — 18 passed, 1 warning (expected), 0 failed.**

---

## Environment Details

### Base Image

`openpi:r36.4.tegra-aarch64-cu126-22.04-cuda_12.6`

Already present on the Jetson Orin Nano from Phase 3/4. Ubuntu 22.04, Python 3.10, CUDA 12.6, JetPack 6.2 (L4T r36.4.7). No download needed — avoids pulling a 5+ GB base image.

### Critical: Host Library Mounts

The container requires host library paths to access cuDNN, TensorRT, and other Jetson system libraries:

```yaml
volumes:
  - /usr/lib/aarch64-linux-gnu:/host-libs:ro    # cuDNN 9, NCCL, etc.
  - /usr/local/cuda/lib64:/host-cuda-libs:ro    # CUDA runtime libs
environment:
  - LD_LIBRARY_PATH=/host-libs:/host-cuda-libs:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu
```

Without these mounts, `import torch` fails with:
```
ImportError: libcudnn.so.9: cannot open shared object file: No such file or directory
```

This pattern is identical to Phase 3 (SNN) and Phase 4 (JEPA) — required by all Phase containers on this Jetson.

### Volume Strategy: Specific Subdirectory Mounts

The compose file mounts **specific subdirectories** (`scripts/`, `outputs/`, `go_server/`) rather than the entire project root `.:/workspace`. This preserves baked-in repos at:
- `/workspace/TensorRT-LLM` (TensorRT-LLM v0.12.0-jetson)
- `/workspace/repos/QJL` (QJL CUDA kernel, AAAI 2025)
- `/workspace/repos/KIVI` (KIVI 2-bit KV, ICML 2024)

If `.:/workspace` were used, these would be masked by the host's (empty) repo directory.

### bitsandbytes on Jetson

The Jetson AI Lab pip index only provides pre-release dev builds of bitsandbytes:
- `0.47.0.dev0`
- `0.48.0.dev0+ff389db`

Standard `pip install bitsandbytes>=0.43.0` fails because pip skips pre-releases by default. Fix: `pip install --pre bitsandbytes`.

### TensorRT-LLM (Expected Warning)

No pip wheel for TensorRT-LLM exists for aarch64/Jetson. The v0.12.0-jetson branch must be built from source. The source is cloned at `/workspace/TensorRT-LLM`. Stage 5 will build the wheel and install it.

### Go 1.22.5

Installed from the official Go arm64 binary at `/usr/local/go`. Verified with `go version go1.22.5 linux/arm64`.

### gollama.cpp

Cloned from `github.com/dianlight/gollama.cpp` — Go bindings for llama.cpp via `purego` (no CGo required). This is Stage 4's primary Go inference approach. The Ollama Go API serves as the production-grade fallback.

---

## How to Reproduce

```bash
# 1. Clone and build
git clone git@github.com:Spitmaan/AneuroOptimizedModels.git
cd AneuroOptimizedModels
docker compose build        # ~15 min (clones TRT-LLM, QJL, KIVI, gollama.cpp)
docker compose up -d

# 2. Verify environment
docker exec aneurologic_phase5 \
    python3 /workspace/scripts/stage1_env/verify_env.py
```

---

## Issues Encountered & Fixes

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.3-py3` not found | Image doesn't exist for Jetson JetPack 6 | Use `openpi:r36.4.tegra-aarch64-cu126-22.04-cuda_12.6` (already on device) |
| `COPY go_server/` build failed | Empty directory not tracked by git | Remove COPY; add `.gitkeep`; go_server built in Stage 4 |
| `bitsandbytes>=0.43.0` install failed | Jetson index only has dev builds; pip skips pre-releases | `pip install --pre bitsandbytes` |
| `libcudnn.so.9: cannot open shared object file` | cuDNN not in container's library path | Mount `/usr/lib/aarch64-linux-gnu:/host-libs:ro`; set `LD_LIBRARY_PATH` |
| Repos show 0 py files | `.:/workspace` bind-mount masked baked-in `/workspace/repos/` | Mount only specific subdirs (`scripts/`, `outputs/`, `go_server/`) |
| CUDA allocatable: 0.0 GB | Jetson UMA throws non-RuntimeError on large alloc | Catch `Exception` (not just `RuntimeError`); use float16 |
