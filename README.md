# ANeurologic Phase 5 — Advanced Edge Optimization

Extreme compression and acceleration of small language models on **NVIDIA Jetson Orin Nano 8 GB**.
Part of the [ANeurologic](https://github.com/Spitmaan) initiative.

Builds on Phase 1 (modelgarden) baselines with a systematic pipeline targeting production-ready deployment on constrained edge hardware. Every optimization step is measured for both **speed** (tokens/s) and **accuracy** (GSM8K, ARC-Challenge) simultaneously.

---

## Hardware Target

| Component | Value |
|-----------|-------|
| **Board** | NVIDIA Jetson Orin Nano Developer Kit 8 GB |
| **GPU** | 1024-core Ampere (sm_87) |
| **Memory** | 8 GB LPDDR5 — UMA (CPU + GPU shared) |
| **Memory bandwidth** | ~68 GB/s |
| **JetPack** | 6.2 (L4T r36.4) |
| **CUDA** | 12.6 |

---

## Models

| Model | Size | Format | Best Config |
|-------|------|--------|-------------|
| LiquidAI LFM2.5-1.2B-Instruct | 1.2B params | GGUF Q4_K_S | `-fa 1 -ngl 99` → **53.22 t/s**, 70% ARC |
| Meta Llama-3.2-1B-Instruct | 1B params | GGUF Q4_K_S | `-fa 1 -ngl 99` → **50.15 t/s**, 35% ARC |
| Alibaba Qwen2.5-0.5B-Instruct | 0.5B params | GGUF Q3_K_M | `-fa 1 -ngl 99` → **93.92 t/s**, 40% ARC |

---

## Stage Overview

| Stage | Title | Status | Report |
|-------|-------|--------|--------|
| **1** | Environment Verification | ✅ 18/19 checks passed | [stage1_environment.md](outputs/reports/stage1_environment.md) |
| **2** | Baseline Reasoning Accuracy | ✅ 2 of 3 models evaluated | [stage2_baseline.md](outputs/reports/stage2_baseline.md) |
| **3** | TurboQuant KV Cache Compression | ✅ All methods benchmarked | [stage3_turboquant.md](outputs/reports/stage3_turboquant.md) · [stage3_perf.md](outputs/reports/stage3_perf.md) |
| **4** | Go-Native Inference Server | ✅ Compiled on Jetson | *(inline below)* |
| **5** | Hardware Acceleration (TensorRT-LLM) | ✅ Estimated results | [stage5_tensorrt.md](outputs/reports/stage5_tensorrt.md) |
| **6** | Knowledge Distillation | ✅ +26.6% accuracy | [stage6_distillation.md](outputs/reports/stage6_distillation.md) |
| **7** | Final Aggregated Report | ✅ Generated | [stage7_phase5_report.md](outputs/reports/stage7_phase5_report.md) |
| **Edge Opt. Ladder** | llama.cpp host optimization (Stages I–VII) | ✅ Complete (2026-03-29) | [edge_optimization_report.md](outputs/reports/edge_optimization_report.md) |

---

## Edge Optimization Results (llama.cpp host, v8510)

Apple-to-apple: every step changes one variable and measures pp512 t/s, tg128 t/s, GSM8K%, and ARC-Challenge% simultaneously.

| Model | Config | pp512 t/s | tg128 t/s | GSM8K | ARC | vs Baseline |
|-------|--------|----------:|----------:|------:|----:|-------------|
| LFM2.5-1.2B | Baseline (Q4_K_M) | 1875 | 49.81 | 5% | 60% | — |
| LFM2.5-1.2B | +Flash Attn | 2107 | 51.64 | 5% | 60% | +3.7% tg ✅ |
| LFM2.5-1.2B | +FA +KV-q8 | 718 | 40.39 | 5% | 65% | Regression ❌ |
| LFM2.5-1.2B | +FA +KV-q4+q4 | 2105 | 50.95 | 5% | 65% | Neutral |
| **LFM2.5-1.2B** | **Q4_K_S + FA** | **2080** | **53.22** | **5%** | **70%** | **+6.8% tg, +10% ARC ✅** |
| LFM2.5-1.2B | Q3_K_M + FA | 1934 | 39.48 | 0% | 45% | Worse both ❌ |
| Llama-3.2-1B | Baseline (Q4_K_M) | 1639 | 44.64 | 0% | 35% | — |
| Llama-3.2-1B | +Flash Attn | 2190 | 48.77 | 0% | 35% | +9.2% tg ✅ |
| **Llama-3.2-1B** | **Q4_K_S + FA** | **2204** | **50.15** | **5%** | **35%** | **+12.3% tg ✅** |
| Llama-3.2-1B | Q3_K_M + FA | 2004 | 39.25 | 5% | 25% | Worse both ❌ |
| Qwen2.5-0.5B | Baseline (Q4_K_M) | 2684 | 80.24 | 0% | 40% | — |
| Qwen2.5-0.5B | +Flash Attn | 3626 | 89.43 | 0% | 40% | +11.5% tg ✅ |
| Qwen2.5-0.5B | Q4_K_S + FA | 3638 | 90.67 | 0% | 40% | Marginal |
| **Qwen2.5-0.5B** | **Q3_K_M + FA** | **3721** | **93.92** | **5%** | **40%** | **+17% tg ✅** |

**Key findings:**
- Flash Attention (`-fa 1`) is a free win on all models (+3–12% tg, +12–35% pp)
- Q4_K_S beats Q4_K_M consistently: smaller model = less memory BW per token
- Q3_K_M benefits sub-1B models only; harmful for ≥1B (dequant cost > bandwidth savings)
- KV cache quantization regresses at 512-token context (all types); crossover at 4K+ context
- CPU thread count has zero effect at `-ngl 99` — GPU does all compute

### Tier 1 — KV Quantization Type Sweep (Stages VIII–X) ✅ Complete

| Stage | Config | pp512 t/s | tg128 t/s | GSM8K | ARC | Verdict |
|-------|--------|----------:|----------:|------:|----:|---------|
| **VIII** | LFM2.5-1.2B Q4_K_S+FA +ctk-**q4_1** | 485 | 40.78 | 5% | **75%** | Speed ❌, ARC +5% (marginal) |
| **IX** | LFM2.5-1.2B Q4_K_S+FA +ctk-**iq4_nl** | 528 | 35.54 | 5% | 65% | Speed ❌, ARC −5% ❌ |
| **X** | LFM2.5-1.2B Q4_K_S+FA +ctk-**q5_0** | 537 | 40.23 | 5% | 55% | Speed ❌, ARC −15% ❌ |

All KV quantization types at 512-token context cause 4–5× pp512 regression and ~25% tg regression. KV cache (~4 MB at 512 tok) is too small for compression savings to overcome encode/decode overhead. Real benefit expected at 4K+ context (Stage XV). **Full Tier 1 report:** [outputs/reports/tier1_kv_quant_types.md](outputs/reports/tier1_kv_quant_types.md)

### Tier 2 — Chat Template + IQ Quantization (Stages XI–XIV) ✅ Complete (XIV blocked)

| Stage | Config | pp512 t/s | tg128 t/s | GSM8K | ARC | Verdict |
|-------|--------|----------:|----------:|------:|----:|---------|
| **XI** | Llama-3.2-1B Q4_K_S+FA + **chat template** | 2204 | 50.16 | **40%** | **40%** | ✅ GSM8K 5%→**40%** — chat template unlocks instruction following |
| **XII** | Qwen2.5-0.5B **IQ3_M**+FA | 3501 | 90.96 | 0% | 40% | ❌ Marginally slower than Q3_K_M; no benefit at 0.5B |
| **XIII** | Llama-3.2-1B **IQ4_XS**+FA | 2386 | **54.37** | 0% | **45%** | ✅ +8.4% tg, +10% ARC — **new Llama winner** |
| **XIV** | LFM2.5-1.2B IQ4_XS | — | — | — | — | ❌ BLOCKED — no GGUF source available |

**New optimal configs after Tier 2:**
- **Llama-3.2-1B**: IQ4_XS + FA → **54.37 t/s**, 45% ARC (use chat template for 40% GSM8K)
- Qwen2.5-0.5B and LFM2.5-1.2B: unchanged

**Full Tier 2 report:** [outputs/reports/tier2_model_quant_and_template.md](outputs/reports/tier2_model_quant_and_template.md)

### Tier 3 — Context-Length and Runtime Experiments (Stages XV–XVII) ✅ Complete (XVII blocked)

| Stage | Config | Result | Verdict |
|-------|--------|--------|---------|
| **XV** | All models × f16 + q4_0 KV at **4096-token context** | q4_0 neutral at 4K (−0.2–0.8% pp vs f16); q4_1/q5_0 still −94–95% | ✅ q4_0 production-safe at 4K |
| **XVI** | KIVI-2bit via Q2_K whitelist + rebuild | Server crashes (ggml_abort) — no CUDA quantize kernel for Q2_K in v8510 | ❌ Not viable |
| **XVII** | ExLlamaV2 v0.3.2 | pip install OK; import fails — CUDA driver 12060 too old for torch 2.11.0 | ❌ Blocked on JetPack 6.2 |

**Production recommendation from Tier 3:** For any deployment using 4K+ context, add `-ctk q4_0 -ctv q4_0`. Zero speed penalty, ~2× KV VRAM savings, enables 2× longer contexts within the 8 GB UMA budget.

**Full Tier 3 report:** [outputs/reports/tier3_long_context_kv_exllama.md](outputs/reports/tier3_long_context_kv_exllama.md)

---

### Tier 4 — IQ4_XS, EAGLE-3, TensorRT-LLM (Stages XVIII–XX) 🔄 In Progress (XX pending)

| Stage | Title | Status | Key Finding |
|-------|-------|--------|-------------|
| **XVIII** | IQ4_XS for LFM2.5 from patched F16 | ✅ Complete | **+10.8% tg128** (58.98 vs 53.22 t/s); ARC −10pp — calibration corpus too narrow. Speed win real; accuracy recoverable with better imatrix. |
| **XIX** | EAGLE-3 speculative decoding | ❌ Blocked | Requires external GPU for training. SSM+attention hybrid arch complicates standard EAGLE approach. ~2–3× speedup potential. |
| **XX** | TensorRT-LLM W4A16 | ✅ Partial (Qwen only) | Qwen2.5-0.5B: **+86% pp, +7.4% tg** (100.87 t/s). LFM2.5 blocked (SSM arch). Llama blocked (gated HF). |

**Updated deployment configs after Tier 4:**

| Model | Best Speed Config | tg t/s | Best Accuracy Config | Notes |
|-------|-----------------|-------:|---------------------|-------|
| LFM2.5-1.2B | IQ4_XS + FA (llama.cpp) | **58.98** | Q4_K_S + FA | ARC 60% vs 70% tradeoff |
| Llama-3.2-1B | IQ4_XS + FA + chat template | 54.37 | same | 40% GSM8K, 45% ARC |
| **Qwen2.5-0.5B** | **TRT-LLM W4A16** | **100.87** | Q3_K_M + FA (llama.cpp) | TRT: +7.4% tg, +86% pp |

**Full Tier 4 report:** [outputs/reports/tier4_iq4xs_eagle_trtllm.md](outputs/reports/tier4_iq4xs_eagle_trtllm.md)

Full analysis, all configs, failed methods, and the future roadmap (Stages VIII–XXII):
**[outputs/reports/edge_optimization_report.md](outputs/reports/edge_optimization_report.md)**

---

## Stage 1 — Environment Verification

**Status: ✅ 18/19 checks passed** | [Full report](outputs/reports/stage1_environment.md)

Verifies all dependencies inside the Docker container before model work begins.

| Component | Status |
|-----------|--------|
| PyTorch 2.10.0 + CUDA 12.6 | ✅ |
| GPU (Orin, 7.99 GB UMA) | ✅ |
| lm-eval 0.4.11 | ✅ |
| TensorRT-LLM pip wheel | ❌ (not available for aarch64 — source build deferred to Stage 5) |
| TRT-LLM source (`/workspace/TensorRT-LLM`) | ✅ |
| KIVI + QJL repos | ✅ |

**Key finding:** Jetson UMA means CPU and GPU share the same 8 GB. CUDA allocation limit in practice is ~3 GB due to OS and Docker overhead. This constrains model loading and gradient storage.

---

## Stage 2 — Baseline Reasoning Accuracy

**Status: ✅ 2 of 3 models evaluated** | [Full report](outputs/reports/stage2_baseline.md)

Establishes pre-optimization accuracy on GSM8K and ARC-Challenge (5-shot, 100 samples each, 4-bit NF4).

| Model | GSM8K | ARC-Challenge | Peak VRAM |
|-------|------:|-------------:|----------:|
| LFM2.5-1.2B | 9.0% | **72.0%** | 1,124 MB |
| Qwen2.5-0.5B | 7.0% | 57.0% | 1,178 MB |
| Qwen2.5-1.5B | ❌ OOM | ❌ OOM | — |

**Note:** `lm-eval` was attempted first but fails on Jetson UMA (CUDA allocator assertion). Used a custom eval loop with `BitsAndBytesConfig` + `device_map="auto"` instead.

---

## Stage 3 — TurboQuant KV Cache Compression

**Status: ✅ All methods benchmarked** | [Compression quality](outputs/reports/stage3_turboquant.md) · [Speed + accuracy](outputs/reports/stage3_perf.md)

Applies KV cache compression algorithms inside the Docker container using PyTorch attention hooks.

| Method | KV Ratio | Speed overhead | Accuracy vs baseline |
|--------|--------:|---------------|---------------------|
| **KIVI-4bit** | 1.88× | −16–20% (CPU roundtrip) | **Matches baseline on both models** |
| KIVI-2bit | 2.29× | −16–19% | Slight degradation |
| PolarQuant | 7.53× | −22–25% | Degrades at <2B scale |
| QJL | 16–64× | N/A† | Cannot be run end-to-end† |

†QJL requires a custom CUDA attention kernel — the 1-bit sketch cannot directly replace K tensors.

**KIVI-4bit (1.88×) is the standout:** near-lossless compression (`cosine sim = 0.997`) with zero accuracy drop in end-to-end eval.

---

## Stage 4 — Go-Native Inference Server

**Status: ✅ Compiled and tested on Jetson**

Production-ready concurrent inference server in pure Go (zero external dependencies) targeting multi-client edge deployment.

- OpenAI-compatible API (`/v1/completions`, `/v1/chat/completions`)
- Goroutine worker pool with counting semaphore (default: 4 concurrent)
- Prometheus `/metrics` endpoint
- Compiled: `go build -o aneurologic-server .` on Jetson (`go1.22.5 linux/arm64`)

```bash
cd go_server && go build -o aneurologic-server .
./aneurologic-server &
```

---

## Stage 5 — Hardware Acceleration (TensorRT-LLM)

**Status: ✅ Estimated results (engine build deferred)** | [Full report](outputs/reports/stage5_tensorrt.md)

TRT-LLM converts HuggingFace weights into serialized GPU engine files with fused CUDA kernels — the highest non-training speedup available on Jetson.

| Model | Best llama.cpp t/s | TRT-LLM est. t/s | Speedup |
|-------|------------------:|----------------:|--------:|
| LFM2.5-1.2B | 53.22 (Q4_K_S+FA) | **~99.7** | ~1.8× |
| Llama-3.2-1B | 50.15 (Q4_K_S+FA) | **~80.5** | ~1.8× |

**Optimization stack:** W4A16 AWQ (4-bit weights, 16-bit activations) + INT8 KV cache + kernel fusion + paged KV cache.

**Why deferred:** TRT-LLM pip wheel is x86_64 only. Jetson aarch64 requires building from the `v0.12.0-jetson` branch from source (~40 min, `--cuda_architectures=87`).

```bash
# When ready to build:
docker exec aneurologic_phase5 python3 /workspace/scripts/stage5_tensorrt/build_engines.py
# Add --build-trtllm flag to trigger the actual ~40-min source build
```

---

## Stage 6 — Knowledge Distillation

**Status: ✅ +26.6% accuracy improvement on Qwen2.5-0.5B** | [Full report](outputs/reports/stage6_distillation.md)

Transfers domain knowledge (aerospace telemetry classification) from a large teacher (Llama-3.1-70B synthetic) into the student (Qwen2.5-0.5B) using KL-divergence on soft distributions (Hinton 2015).

| Metric | Value |
|--------|-------|
| Trainable parameters | 657,667 (0.13% — LoRA rank=8 on q/v_proj + classification head) |
| Accuracy (epoch 1) | 26.7% |
| Accuracy (epoch 5) | 40.0% |
| **Accuracy (held-out eval)** | **53.3%** |

**Why Qwen, not LFM:** LFM2.5-1.2B failed during backprop through its hybrid conv layers (`NVML_SUCCESS INTERNAL ASSERT FAILED`). Conv architecture stores activations differently; gradient accumulation exceeds the 3 GB CUDA limit on Jetson UMA.

---

## Stage 7 — Final Aggregated Report

**Status: ✅ Generated** | [Full report](outputs/reports/stage7_phase5_report.md)

Aggregates all stage JSON outputs into a unified leaderboard, per-stage tables, paper attribution, and production deployment recommendations.

---

## Quick Start

### 1. Clone and build the container (on Jetson)

```bash
git clone git@github.com:Spitmaan/AneuroOptimizedModels.git
cd AneuroOptimizedModels
docker compose build
docker compose up -d
```

### 2. Run Docker pipeline stages (inside container)

```bash
# Stage 1 — Verify environment
docker exec aneurologic_phase5 python3 /workspace/scripts/stage1_env/verify_env.py

# Stage 2 — Baseline accuracy
docker exec aneurologic_phase5 python3 /workspace/scripts/stage2_baseline/baseline_reasoning.py

# Stage 3 — KV compression quality
docker exec aneurologic_phase5 python3 /workspace/scripts/stage3_turboquant/kv_compression.py
# Stage 3 comprehensive (speed + accuracy)
docker exec aneurologic_phase5 python3 /workspace/scripts/stage3_turboquant/stage3_comprehensive.py

# Stage 4 — Go server
cd go_server && go build -o aneurologic-server .

# Stage 5 — TRT-LLM (estimated by default; --build-trtllm for full ~40-min build)
docker exec aneurologic_phase5 python3 /workspace/scripts/stage5_tensorrt/build_engines.py

# Stage 6 — Distillation
docker exec aneurologic_phase5 python3 /workspace/scripts/stage6_distillation/distill.py

# Stage 7 — Final report
docker exec aneurologic_phase5 python3 /workspace/scripts/stage7_report/generate_report.py
```

### 3. Run edge optimization benchmarks (on Jetson host, not Docker)

```bash
# Run on Jetson HOST — requires llama.cpp v8510 built at /home/spitman/tools/llama.cpp/
python3 scripts/edge_optimization/bench_gguf.py \
  --model /path/to/model.gguf \
  --label "MyLabel" \
  --flags "-fa 1"

# Print comparison table from all saved results
python3 scripts/edge_optimization/bench_gguf.py --compare
```

Results accumulate in `outputs/logs/edge_opt/results.json`.
