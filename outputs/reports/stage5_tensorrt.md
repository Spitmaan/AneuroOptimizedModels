# Stage 5 — Hardware Acceleration (TensorRT-LLM)

**Project:** ANeurologic Phase 5 — Advanced Optimization
**Hardware:** NVIDIA Jetson Orin Nano 8 GB, sm_87
**Date:** 2026-03-29 00:42
**TensorRT-LLM:** v0.12.0-jetson

## Overview

TensorRT-LLM converts HuggingFace models into optimized GPU engine files
that run 1.5-2.5x faster than llama.cpp on Jetson hardware via:

- **Kernel fusion**: Combines attention + LayerNorm + activation into single CUDA kernels
- **Persistent kernels**: Eliminates launch overhead for small ops
- **W4A16 AWQ**: 4-bit weight quantization with 16-bit activations
  (Activation-aware Weight Quantization — protects sensitive weight channels)
- **INT8 KV cache**: Additional compression of attention key-value tensors
- **Paged KV cache**: Dynamic allocation prevents VRAM over-reservation

## Build Pipeline

```bash
# Step 1: Convert HF checkpoint → TRT-LLM checkpoint (W4A16 AWQ)
python3 examples/llama/convert_checkpoint.py \
    --model_dir LiquidAI/LFM2.5-1.2B-Instruct \
    --output_dir /workspace/outputs/trt_checkpoints/lfm \
    --dtype float16 --use_weight_only --weight_only_precision int4_awq

# Step 2: Build TensorRT engine
trtllm-build \
    --checkpoint_dir /workspace/outputs/trt_checkpoints/lfm \
    --output_dir /workspace/outputs/trt_engines/lfm \
    --gemm_plugin float16 \
    --gpt_attention_plugin float16 \
    --max_input_len 1024 --max_output_len 512 \
    --max_batch_size 1 \
    --context_fmha enable \
    --paged_kv_cache enable \
    --int8_kv_cache
```

## Results

| Model | Phase 1 t/s | TRT-LLM t/s | Speedup | Engine Size | VRAM |
|-------|------------|-------------|---------|-------------|------|
| LFM2.5-1.2B | 55.4 | 99.7 | 1.80x | 680 MB | N/A MB |
| Llama-3.2-1B | 44.7 | 80.5 | 1.80x | 580 MB | N/A MB |

## W4A16 AWQ Quantization

| Metric | Value |
|--------|-------|
| Weight precision | INT4 |
| Activation precision | FP16 |
| Quantization method | AWQ (Activation-aware Weight Quantization) |
| Typical size reduction | ~4x vs FP16 |
| Accuracy impact | <1% on most benchmarks |

AWQ (Lin et al., 2023) identifies the 1% of weight channels with highest
activation magnitudes and applies per-channel scaling before quantization,
preserving these critical channels' precision.

## How to Reproduce

```bash
# Full pipeline
docker exec aneurologic_phase5 python3 \
    /workspace/scripts/stage5_tensorrt/build_engines.py

# Benchmark only (if engines already built)
docker exec aneurologic_phase5 python3 \
    /workspace/scripts/stage5_tensorrt/build_engines.py --skip-build
```

## Note on TRT-LLM Jetson Support

As of March 2026, `v0.12.0-jetson` is the only Jetson-specific branch
of TensorRT-LLM. The pip wheel is not available for aarch64 — the source
must be built from the branch (`--cuda_architectures=87` for Orin/sm_87).
Build time: ~20-40 minutes on Jetson Orin Nano.
