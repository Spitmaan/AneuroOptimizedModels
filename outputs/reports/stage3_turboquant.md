# Stage 3 — TurboQuant KV Cache Compression: Full Benchmark

**Project:** ANeurologic Phase 5 — Advanced Optimization
**Hardware:** NVIDIA Jetson Orin Nano 8 GB UMA (sm_87)
**Date:** 2026-03-29 20:37

## What is TurboQuant?

TurboQuant (Google Research, arXiv:2504.19874, ICLR 2026) is a **framework** that
unifies and extends two separately-published algorithms. There is no `pip install`.

| Component | Paper | Venue | Role |
|-----------|-------|-------|------|
| **PolarQuant** | arXiv:2502.02617 | AISTATS 2026 | K+V compression via polar transform |
| **QJL** | arXiv:2406.03482 | AAAI 2025 | K-only 1-bit JL sketch (score estimation) |
| **KIVI** | arXiv:2402.02750 | ICML 2024 | Production 2/4-bit group quant baseline |

---

## Feasibility: Can Compression Increase Speed and Accuracy?

**Speed — Yes, but only with a CUDA kernel (not CPU roundtrip).**

Current implementation compresses/decompresses on CPU at each step.
This adds ~20-25% overhead despite the smaller KV footprint in VRAM.
In a production CUDA kernel (e.g., KIVI's official implementation):

- Smaller KV → fewer memory reads during `Q @ K^T` → lower memory bandwidth usage
- On Jetson Orin's UMA, memory bandwidth is the primary bottleneck for attention
- Estimated real speedup: 1.5-3x for long sequences (1024+ tokens) where KV dominates
- For short sequences (<256 tokens), KV overhead is small → marginal gain

**Accuracy — No for short-context tasks; Yes for OOM-limited long-context tasks.**

Lossy compression fundamentally cannot add information — accuracy can only
decrease or stay the same for tasks within normal context length.
EXCEPTION: when a task would OOM without compression, compression turns a
crash into a correct answer. This is the primary motivation for edge deployment.

| Method | Accuracy impact | Speed impact (CUDA) | Best use case |
|--------|----------------|--------------------:|---------------|
| PolarQuant | Significant degradation at 0.5-1.2B scale | High (7.53x KV) | Large models (7B+) |
| KIVI-2bit | Moderate degradation | Medium (2.29x KV) | Production edge |
| KIVI-4bit | Near-zero degradation | Mild (1.88x KV) | Conservative production |
| QJL | N/A (kernel required) | Very high (16-64x K) | Research / custom kernels |

---

## Experimental Setup

| Setting | Value |
|---------|-------|
| Quantization | 4-bit NF4 (BitsAndBytesConfig, double quantized) |
| Speed test | 48 generated tokens, 5 warmup steps |
| Accuracy test | 20 samples, 3-shot prompts |
| GSM8K method | Generation loop, exact number match (regex) |
| ARC method | Two-pass: context → compress KV → score A/B/C/D |
| Compression hook | Every generation step (GPU tensor → CPU compress → GPU) |

---

## Results

### LFM2.5-1.2B

**Cache type:** `Lfm2HybridConvCache`  |  **VRAM at load:** 803.3 MB

#### Throughput & VRAM

| Method | KV Ratio | t/s | vs Baseline | Peak VRAM (eval) MB |
|--------|--------:|----:|------------:|--------------------:|
| Baseline | 1.00x | 12.12 | 1.00x | 881.7 |
| PolarQuant | 7.53x | 9.43 | 0.78x | 881.7 |
| KIVI-2bit | 2.29x | 10.18 | 0.84x | 881.7 |
| KIVI-4bit | 1.88x | 10.19 | 0.84x | 881.7 |
| QJL | 16–64x (K) | N/A* | N/A* | N/A* |

#### GSM8K Accuracy (math word problems, 3-shot generation)

| Method | Score | Correct/Total | Approx t/s | Peak VRAM MB |
|--------|------:|:-------------:|-----------:|-------------:|
| Baseline | 5.0% | 1/20 | 5.32 | 899.6 |
| PolarQuant | 0.0% | 0/20 | 4.44 | 899.6 |
| KIVI-2bit | 0.0% | 0/20 | 4.42 | 899.6 |
| KIVI-4bit | 0.0% | 0/20 | 4.42 | 899.6 |
| QJL | N/A* | N/A* | N/A* | N/A* |

#### ARC-Challenge Accuracy (science MCQ, 3-shot two-pass)

| Method | Score | Correct/Total | Peak VRAM MB |
|--------|------:|:-------------:|-------------:|
| Baseline | 20.0% | 4/20 | 941.0 |
| PolarQuant | 20.0% | 4/20 | 941.0 |
| KIVI-2bit | 25.0% | 5/20 | 941.0 |
| KIVI-4bit | 20.0% | 4/20 | 941.0 |
| QJL | N/A* | N/A* | N/A* |

*QJL requires modifying the attention kernel to use the asymmetric score
estimator `(π/2m) · Q_proj · q^T`. Speed/accuracy tests are not applicable
because simply using the 1-bit sketch as K tensors produces garbage attention.
Score estimation quality from Stage 3: Pearson-r = 0.62 at sketch_dim=64 (16x ratio).

---

### Qwen2.5-0.5B

**Cache type:** `DynamicCache`  |  **VRAM at load:** 466.0 MB

#### Throughput & VRAM

| Method | KV Ratio | t/s | vs Baseline | Peak VRAM (eval) MB |
|--------|--------:|----:|------------:|--------------------:|
| Baseline | 1.00x | 7.06 | 1.00x | 486.2 |
| PolarQuant | 7.53x | 5.28 | 0.75x | 486.2 |
| KIVI-2bit | 2.29x | 5.7 | 0.81x | 486.2 |
| KIVI-4bit | 1.88x | 5.66 | 0.80x | 486.2 |
| QJL | 16–64x (K) | N/A* | N/A* | N/A* |

#### GSM8K Accuracy (math word problems, 3-shot generation)

| Method | Score | Correct/Total | Approx t/s | Peak VRAM MB |
|--------|------:|:-------------:|-----------:|-------------:|
| Baseline | 5.0% | 1/20 | 3.2 | 558.4 |
| PolarQuant | 0.0% | 0/20 | 2.33 | 558.4 |
| KIVI-2bit | 0.0% | 0/20 | 2.55 | 558.4 |
| KIVI-4bit | 10.0% | 2/20 | 2.56 | 558.4 |
| QJL | N/A* | N/A* | N/A* | N/A* |

#### ARC-Challenge Accuracy (science MCQ, 3-shot two-pass)

| Method | Score | Correct/Total | Peak VRAM MB |
|--------|------:|:-------------:|-------------:|
| Baseline | 30.0% | 6/20 | 651.6 |
| PolarQuant | 10.0% | 2/20 | 651.6 |
| KIVI-2bit | 20.0% | 4/20 | 651.6 |
| KIVI-4bit | 30.0% | 6/20 | 651.6 |
| QJL | N/A* | N/A* | N/A* |

*QJL requires modifying the attention kernel to use the asymmetric score
estimator `(π/2m) · Q_proj · q^T`. Speed/accuracy tests are not applicable
because simply using the 1-bit sketch as K tensors produces garbage attention.
Score estimation quality from Stage 3: Pearson-r = 0.62 at sketch_dim=64 (16x ratio).

---

## Method Summary

| Method | KV Ratio | Principle | Accuracy Impact | Production Ready |
|--------|--------:|-----------|----------------|:---------------:|
| Baseline | 1x | — | — | ✅ |
| PolarQuant | 7.53x | Polar coords (mag + angle) | Significant at <2B scale | ⚠️ Large models only |
| KIVI-2bit | 2.29x | Asymmetric group quant | Moderate | ✅ |
| KIVI-4bit | 1.88x | Asymmetric group quant | Near-zero | ✅ |
| QJL | 16–64x (K) | 1-bit JL sketch | Requires kernel | 🔬 Research |

## How to Reproduce

```bash
# Full benchmark (both models, all methods, speed + accuracy)
docker exec aneurologic_phase5 python3 \
    /workspace/scripts/stage3_turboquant/stage3_comprehensive.py

# Single model
docker exec aneurologic_phase5 python3 \
    /workspace/scripts/stage3_turboquant/stage3_comprehensive.py --model qwen

# Speed only (faster)
docker exec aneurologic_phase5 python3 \
    /workspace/scripts/stage3_turboquant/stage3_comprehensive.py --speed-only
```
