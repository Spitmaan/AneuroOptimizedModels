# Stage 3 — TurboQuant KV Cache Compression

**Project:** ANeurologic Phase 5 — Advanced Optimization
**Hardware:** NVIDIA Jetson Orin Nano 8 GB
**Date:** 2026-03-29 00:35

## Background

TurboQuant (Google Research, ICLR 2026; arXiv:2504.19874) is a suite of
KV-cache compression algorithms announced March 25, 2026. The suite comprises:

| Component | Paper | Venue |
|-----------|-------|-------|
| **PolarQuant** | arXiv:2502.02617 | AISTATS 2026 |
| **QJL** | arXiv:2406.03482 | AAAI 2025 |
| **TurboQuant** (full system) | arXiv:2504.19874 | ICLR 2026 |
| **KIVI** (production baseline) | arXiv:2402.02750 | ICML 2024 |

No open-source library for TurboQuant exists yet. PolarQuant and QJL are
implemented here from the papers; QJL's CUDA kernels
([github.com/amirzandieh/QJL](https://github.com/amirzandieh/QJL)) are used
where available. KIVI is fully open and serves as the production fallback.

---

## Results Summary

| Model | Method | Tensor | Ratio | RMSE | Cos Sim |
|-------|--------|--------|-------|------|---------|
| LFM2.5-1.2B | PolarQuant (8+2 bit) | K | 7.5x | 0.41127 | 0.9157 |
| LFM2.5-1.2B | PolarQuant (8+2 bit) | V | 7.5x | 0.40880 | 0.9159 |
| LFM2.5-1.2B | KIVI (2bit) | K | 2.3x | 0.39206 | 0.9319 |
| LFM2.5-1.2B | KIVI (2bit) | V | 2.3x | 0.38980 | 0.9319 |
| LFM2.5-1.2B | KIVI (4bit) | K | 1.9x | 0.07784 | 0.9969 |
| LFM2.5-1.2B | KIVI (4bit) | V | 1.9x | 0.07755 | 0.9969 |
| LFM2.5-1.2B | QJL (sketch=16) | K | 64.0x | Pearson-r=0.3662 | — |
| LFM2.5-1.2B | QJL (sketch=32) | K | 32.0x | Pearson-r=0.4904 | — |
| LFM2.5-1.2B | QJL (sketch=64) | K | 16.0x | Pearson-r=0.6216 | — |
| Qwen2.5-0.5B | PolarQuant (8+2 bit) | K | 7.5x | 0.41025 | 0.9152 |
| Qwen2.5-0.5B | PolarQuant (8+2 bit) | V | 7.5x | 0.41332 | 0.9145 |
| Qwen2.5-0.5B | KIVI (2bit) | K | 2.3x | 0.39181 | 0.9306 |
| Qwen2.5-0.5B | KIVI (2bit) | V | 2.3x | 0.39257 | 0.9312 |
| Qwen2.5-0.5B | KIVI (4bit) | K | 1.9x | 0.07798 | 0.9969 |
| Qwen2.5-0.5B | KIVI (4bit) | V | 1.9x | 0.07818 | 0.9969 |
| Qwen2.5-0.5B | QJL (sketch=16) | K | 64.0x | Pearson-r=0.3684 | — |
| Qwen2.5-0.5B | QJL (sketch=32) | K | 32.0x | Pearson-r=0.4912 | — |
| Qwen2.5-0.5B | QJL (sketch=64) | K | 16.0x | Pearson-r=0.6210 | — |

---

## Algorithm Notes

### PolarQuant
- Transforms KV vectors to polar coordinates (magnitude + angular components)
- 8-bit magnitude + 2-bit angular → ~6x compression vs fp16
- Data-oblivious: no calibration dataset required

### QJL (1-bit JL Transform)
- Projects keys with a random JL matrix A ∈ R^{m×D}, then takes sign bits
- K-cache compressed to m bits per token (m << D); typically m=16-32
- Score estimation via asymmetric estimator: (π/2m) · Q_proj · q^T
- Zero overhead: no per-block scale/zero-point constants stored

### KIVI (2-bit production baseline)
- Per-channel asymmetric min-max quantization, group_size=32
- Residual buffer: last 64-128 tokens kept in fp16
- Drop-in replacement for KV cache in any transformer

## How to Reproduce

```bash
docker exec aneurologic_phase5 python3 \
    /workspace/scripts/stage3_turboquant/kv_compression.py --method all
```
