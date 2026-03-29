# Stage 3b — KV Compression: Speed & Accuracy Impact

**Project:** ANeurologic Phase 5 — Advanced Optimization
**Hardware:** NVIDIA Jetson Orin Nano 8 GB, sm_87
**Date:** 2026-03-29 18:55
**Model:** Qwen2.5-0.5B-Instruct (NF4 4-bit, device_map=auto)
**Speed test:** 48 tokens after 5 warmup steps
**Accuracy test:** 20 samples each, 3-shot

## Throughput (tokens/sec)

Compression applied at **every generation step**.
VRAM Δ = peak allocated during generation vs. before.

| Method | t/s | vs Baseline | VRAM Δ (MB) | KV Ratio |
|--------|----:|------------:|------------:|--------:|
| Baseline | 7.27 | 1.00x | 18.7 | 1.00x |
| PolarQuant | 5.45 | 0.75x | 18.7 | 7.53x |
| KIVI-2bit | 5.8 | 0.80x | 18.7 | 2.29x |
| KIVI-4bit | 5.74 | 0.79x | 18.7 | 1.88x |

## Accuracy Impact

**GSM8K** — math word problems, 3-shot generation, exact number match.
**ARC-Challenge** — 3-shot two-pass: context → compress KV → score choices.
Stage 2 baselines (no compression): GSM8K 7.0% | ARC-Challenge 57.0%

| Method | GSM8K | vs Baseline | ARC-Challenge | vs Baseline |
|--------|------:|------------:|--------------:|------------:|
| Baseline | 5.0% | -2.0pp | 30.0% | -27.0pp |
| PolarQuant | 0.0% | -7.0pp | 10.0% | -47.0pp |
| KIVI-2bit | 0.0% | -7.0pp | 25.0% | -32.0pp |
| KIVI-4bit | 10.0% | +3.0pp | 30.0% | -27.0pp |

## Notes

- **pp** = percentage points vs Stage 2 baseline (no compression)
- **VRAM Δ** includes overhead from compress/decompress operations
- Compression+decompression runs on CPU; GPU transfer adds latency
- In a production CUDA kernel, compress/decompress would run on-GPU
  with no transfer overhead — real-world t/s would be higher
- **QJL** not included in speed/accuracy tests: requires modifying
  the attention kernel to use sketch scores instead of exact dot products
  (Pearson-r quality: 0.62 at sketch_dim=64 from Stage 3)

## Method Summary

| Method | Compression | Principle | Best For |
|--------|-------------|-----------|----------|
| PolarQuant | 7.53x | Polar coords (mag + angle) | Large contexts, moderate accuracy loss OK |
| KIVI-2bit | 2.29x | Asymmetric min-max quant | Production (battle-tested) |
| KIVI-4bit | 1.88x | Asymmetric min-max quant | Near-lossless, conservative |
| QJL | 16–64x (K only) | 1-bit JL sketch | Research / custom attention kernels |
