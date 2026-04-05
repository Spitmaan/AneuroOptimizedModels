# ANeurologic — Comprehensive Project Report

**Date:** 2026-04-03
**Hardware:** NVIDIA Jetson Orin Nano 8 GB (sm_87, 68 GB/s UMA, JetPack 6.2)
**Secondary:** NVIDIA DGX Spark GB10 (sm_121 Blackwell, 119 GB UMA, used for model export/training)

---

## Executive Summary

ANeurologic investigates brain-inspired and efficient AI models on edge GPU hardware. Across four completed phases, we benchmarked conventional language models, spiking neural networks, JEPA world models, and pursued extreme optimization — systematically measuring speed, accuracy, energy, and memory on a single Jetson Orin Nano 8 GB.

**Key results:**

| Milestone | Best Result | Phase |
|-----------|------------|-------|
| Fastest >1B language model | **73.4 t/s** (Llama-3.2-1B, MLC-LLM) | 6 |
| Fastest 0.5B language model | **100.87 t/s** (Qwen2.5-0.5B, TRT-LLM W4A16) | 5 |
| Best accuracy at speed | **45% ARC, 40% GSM8K, 54 t/s** (Llama-3.2-1B IQ4_XS+chat) | 5 |
| Highest spike sparsity | **99.94%** (Spikformer-4-384, LIF neurons) | 3 |
| Lowest SNN energy | **48.45 mJ/image** (Spikformer on Jetson GPU) | 3 |
| JEPA real-time vision | **74.3 FPS, 421 MB** (V-JEPA 2.1 ViT-B/16) | 4 |
| JEPA-to-SLM pipeline | **294.5 ms** end-to-end video reasoning | 4 |
| JEPA-to-SNN bridge | **19.8 ms**, 87.5% input sparsity (latency coding) | 4 |
| Knowledge distillation | **+26.7 pp** accuracy (Qwen 0.5B aerospace task) | 5 |
| 100 t/s for >1B params | **NOT ACHIEVED** — 73.4 t/s ceiling | 6 |

---

## 1. Hardware Platform

| Component | Specification |
|-----------|--------------|
| Board | NVIDIA Jetson Orin Nano Developer Kit 8 GB |
| GPU | 1024-core Ampere (sm_87), no FP8 hardware |
| Memory | 8 GB LPDDR5 — Unified Memory Architecture (CPU + GPU shared) |
| Memory bandwidth | ~68 GB/s theoretical |
| NvMap IOVM limit | ~2.5 GB virtual address space (hard constraint) |
| JetPack | 6.2 (L4T r36.4), CUDA 12.6, TensorRT 10.3.0 |
| Power | 15 W max (MAXN mode) |

The UMA architecture means GPU memory allocations compete with system RAM. The 2.5 GB NvMap IOVM limit proved to be the single most impactful constraint across all phases — blocking speculative decoding, TRT-LLM gemm_plugin, and multi-model inference.

---

## 2. Phase 3 — Spiking Neural Networks on Edge Hardware

**Goal:** Validate energy and speed claims of SNNs vs conventional ANNs on Jetson.
**Repository:** `modelgardensnn/`

### Models Benchmarked

| Model | Type | Params | Speed | VRAM | Spike Sparsity |
|-------|------|-------:|------:|-----:|---------------:|
| SpikeGPT-OWT-216M | Language (RWKV-RNN) | 215.4M | 12.3 tok/s | 0.70 GB | ~100% |
| SpikeGPT-BC-184M | Language (char-level) | 184M | 9.0 tok/s | 0.73 GB | ~100% |
| Spikformer-4-384 | Vision (ViT+LIF) | 9.3M | 61.7 FPS | 0.85 GB | **99.94%** |
| QSD-Large | Vision (LSQ quant) | 55.6M | 3.2 FPS | 0.46 GB | 16.14% |
| SpikeYOLO-23M | Detection | 23M | 5.4 FPS | 0.68 GB | unmeasured |

### Energy Profiling (jtop, Jetson GPU rail)

| Model | GPU Power (incr.) | Energy/unit | Board Energy/unit |
|-------|------------------:|------------:|------------------:|
| Spikformer-4-384 | +2.98 W | **48.45 mJ/img** | 101.94 mJ/img |
| SpikeGPT-OWT | +1.03 W | 88.15 mJ/tok | 191.62 mJ/tok |
| SpikeYOLO-23M | +2.89 W | 554.6 mJ/img | 1,133 mJ/img |

### Key Findings

1. **Sparsity divergence:** True LIF spiking (Spikformer 99.94%) is fundamentally different from quantization-based "spiking" (QSD 6-16%). Only LIF models have neuromorphic hardware potential.
2. **CUDA penalty:** SNNs are 3-8x slower than equivalent ANNs on GPU — timestep iteration and sparse ops don't map well to CUDA's SIMT architecture.
3. **Memory efficiency:** All SNNs use < 0.85 GB VRAM (34-63% less than ANNs) — consistent advantage.
4. **Neuromorphic projection:** On Loihi 2, Spikformer's 99.94% sparsity would yield ~1,000x energy savings vs GPU execution. The value proposition is hardware-dependent, not architectural.

---

## 3. Phase 4 — JEPA World Models

**Goal:** Bridge JEPA continuous embeddings to both SLMs (for reasoning) and SNNs (for neuromorphic deployment).
**Repository:** `modelgardenJEPA/`

### Components Benchmarked

| Component | Params | Speed | Latency | VRAM |
|-----------|-------:|------:|--------:|-----:|
| V-JEPA 2.1 ViT-B/16 | 88.0M | 74.3 FPS | 13.46 ms/frame | 420.6 MB |
| LeWorldModel (proxy) | 19.2M | 6,613 FPS | 1.21 ms/seq | 89.0 MB |
| JEPA → SLM pipeline | — | 3.4 FPS | 294.51 ms e2e | 719.2 MB |
| Spiking bridge (rate) | — | — | 20.09 ms | 241 MB |
| Spiking bridge (latency) | — | — | 19.80 ms | 245 MB |

### Pipeline Architecture

```
Video frames (8×224×224)
    → V-JEPA 2.1 encoder (105.84 ms)
    → AdaptiveAvgPool1d + Linear + LayerNorm (0.84 ms)
    → Qwen2.5-0.5B 4-bit NF4 prefill (187.82 ms)
    → Text output
```

### Spiking World Model

| Coding | Encode | SNN Forward | Decode | Total | Input Sparsity | Output Sparsity |
|--------|-------:|------------:|-------:|------:|---------------:|----------------:|
| Rate | 2.03 ms | 17.72 ms | 0.34 ms | 20.09 ms | 50.2% | 99.5% |
| Latency | 2.30 ms | 16.65 ms | 0.85 ms | 19.80 ms | **87.5%** | **100.0%** |

### Key Findings

1. **V-JEPA is real-time capable:** 74.3 FPS exceeds Spikformer (61.7 FPS) with less VRAM — a viable vision backbone for edge deployment.
2. **SLM prefill dominates:** In the JEPA→SLM pipeline, 63.8% of latency is SLM prefill. Faster SLMs directly improve multimodal reasoning speed.
3. **Latency coding wins:** 87.5% input sparsity (vs 50.2% for rate coding) makes latency-coded JEPA→SNN more suitable for neuromorphic hardware.
4. **LeWorldModel gap:** Weights not publicly available (Google Drive inaccessible). Proxy architecture benchmarked for throughput only.

---

## 4. Phase 5 — Edge Optimization Ladder

**Goal:** Systematically maximize speed AND accuracy of LFM2.5-1.2B, Llama-3.2-1B, and Qwen2.5-0.5B on Jetson. One change at a time, measure both.
**Repository:** `AneuroOptimizedModels/`

### Optimization Progression (Llama-3.2-1B example)

| Step | Config | tg128 t/s | ARC | Change |
|------|--------|----------:|----:|--------|
| Baseline | Q4_K_M | 44.64 | 35% | Reference |
| +Flash Attention | Q4_K_M + FA | 48.77 | 35% | +9.2% speed |
| +Smaller quant | Q4_K_S + FA | 50.15 | 35% | +12.3% total |
| +IQ quantization | IQ4_XS + FA | 54.37 | 45% | +21.8% speed, +10pp ARC |
| +Chat template | IQ4_XS + FA + chat | 54.37 | 45% | +35pp GSM8K (5→40%) |

### Best Configs per Model (Phase 5, llama.cpp)

| Model | Quant | Flags | tg128 t/s | GSM8K | ARC |
|-------|-------|-------|----------:|------:|----:|
| LFM2.5-1.2B | IQ4_XS | FA, ngl 99 | **65.64** | — | 60% |
| Llama-3.2-1B | IQ4_XS | FA, ngl 99, chat | **54.37** | 40% | 45% |
| Qwen2.5-0.5B | Q3_K_M | FA, ngl 99 | **93.92** | 5% | 40% |

### TensorRT-LLM Results

| Model | Config | Speed | vs llama.cpp |
|-------|--------|------:|:-------------|
| Qwen2.5-0.5B | W4A16 | **100.87 t/s** | +7.4% (winner) |
| LFM2.5-1.2B | — | BLOCKED | SSM unsupported |
| Llama-3.2-1B | FP16 no-gemm | 44.08 t/s | -19% (slower) |

### Knowledge Distillation

- **Task:** Aerospace telemetry classification (10 categories)
- **Teacher:** Llama-3.1-70B (via Together.ai API)
- **Student:** Qwen2.5-0.5B with LoRA rank=8
- **Result:** 26.67% → **53.33% accuracy** (+26.66 pp) in 5 epochs
- Trainable parameters: 657,667

### Key Findings

1. **Flash Attention is free:** +9-12% speed, no accuracy cost. Always enable.
2. **IQ4_XS > Q4_K_S:** imatrix-aware quantization gives +8-11% speed AND better accuracy for Llama.
3. **Chat template matters:** Llama GSM8K jumps 5% → 40% when using `/v1/chat/completions` instead of raw completion.
4. **KV cache quant at short context:** All KV quantization types cause 4-5x pp512 regression. Only viable at 4K+ context where q4_0 is neutral.
5. **TRT-LLM limited on Jetson:** Only benefits Qwen 0.5B. Gemm_plugin OOMs (NvMap IOVM). LFM SSM unsupported.

---

## 5. Phase 6 — The 100 t/s Quest

**Goal:** Achieve ≥100 t/s token generation with >1B parameters on Jetson Orin Nano.
**Result:** NOT ACHIEVED — **73.4 t/s** ceiling (Llama-3.2-1B, MLC-LLM).

### The Bandwidth Ceiling

Token generation is memory-bandwidth bound. At 68 GB/s:

| Model Size | INT4 Weight BW/token | Theoretical Max | Realistic (72% eff.) |
|------------|---------------------|-----------------|---------------------|
| 0.5B | 247 MB | 275 t/s | ~100 t/s |
| **1B** | **500 MB** | **136 t/s** | **~98 t/s** |
| 1.2B | 600 MB | 113 t/s | ~81 t/s |
| 1.5B | 750 MB | 91 t/s | ~65 t/s |

At 72% BW efficiency (measured), 1B params is right at the edge of 100 t/s. The only path was speculative decoding for a ~1.4x effective speedup.

### All Paths Attempted

| # | Path | Result | Blocker |
|---|------|--------|---------|
| 1 | llama-speculative 3B+1B | BLOCKED | NvMap IOVM: can't load both models |
| 2 | TRT-LLM gemm_plugin | BLOCKED | NvMap IOVM: 1 GB serialization buffer |
| 3 | llama.cpp EAGLE | BLOCKED | Not implemented in llama.cpp |
| 4 | Prompt lookup decoding | FAILED | Sync overhead > speedup |
| 5 | Smaller draft model | BLOCKED | No smaller Llama-3/LFM variant exists |
| 6 | MLC-LLM EAGLE-2 (TIR) | **8.9 t/s** | Tree-attention kernels 8x slower |
| 6B | MLC-LLM EAGLE-2 (FlashInfer) | **9.3 t/s** | Same — framework-level bug |
| 7 | TRT Edge-LLM INT4+EAGLE3 | **24.1 t/s** | Vocab reduction breaks EAGLE3 |
| 8 | TRT Edge-LLM FP8+EAGLE3 | BLOCKED | sm_87 has no FP8 hardware |
| 9 | TRT Edge-LLM INT8 SQ | BLOCKED | 5.9 GB ONNX > IOVM |
| 10 | MLC-LLM EAGLE-2 (trained) | BLOCKED | Better head can't fix broken engine |

### MLC-LLM Verified Baselines (Clean 5-Run)

| Model | Runtime | t/s (avg) | t/s (peak) | BW Efficiency |
|-------|---------|----------:|-----------:|--------------:|
| Llama-3.2-1B | MLC-LLM q4f16_1 | 72.7 | **73.4** | **71.6%** |
| LFM2.5-1.2B | llama.cpp IQ4_XS+FA | 65.64 | — | 60.9% |

### Why 100 t/s Is Unreachable for >1B

1. **Speculative decoding broken:** MLC-LLM EAGLE-2 is 8x slower (framework bug); TRT Edge-LLM EAGLE3 blocked by NvMap/FP8/vocab mismatch.
2. **No draft models exist:** Llama-3.2-1B is the smallest in its family. LFM2.5 has no smaller variant.
3. **BW efficiency already 72%:** Near the practical ceiling — no room for algorithmic speedup without spec decoding.
4. **Hardware constraints:** 68 GB/s bandwidth, 2.5 GB IOVM, no FP8 (sm_87).

---

## 6. Cross-Phase Summary

### Final Leaderboard — All Models on Jetson Orin Nano 8 GB

| Rank | Model | Params | Runtime | Speed | Accuracy | VRAM |
|------|-------|-------:|---------|------:|----------|-----:|
| 1 | Qwen2.5-0.5B | 0.5B | TRT-LLM W4A16 | **100.87 t/s** | 40% ARC | 445 MB |
| 2 | Qwen2.5-0.5B | 0.5B | llama.cpp Q3_K_M+FA | 93.92 t/s | 40% ARC | — |
| 3 | **Llama-3.2-1B** | **1B** | **MLC-LLM q4f16_1** | **73.4 t/s** | **45% ARC** | **663 MB** |
| 4 | LFM2.5-1.2B | 1.2B | llama.cpp IQ4_XS+FA | 65.64 t/s | 60% ARC | 633 MB |
| 5 | Llama-3.2-1B | 1B | llama.cpp IQ4_XS+FA+chat | 54.37 t/s | 45% ARC, 40% GSM8K | 709 MB |
| 6 | V-JEPA 2.1 ViT-B | 88M | PyTorch | 74.3 FPS | — | 421 MB |
| 7 | Spikformer-4-384 | 9.3M | spikingjelly+CUDA | 61.7 FPS | 99.94% sparse | 850 MB |
| 8 | SpikeGPT-OWT | 215M | RWKV+CUDA | 12.3 tok/s | — | 700 MB |

### Runtimes Compared (for Llama-3.2-1B)

| Runtime | t/s | Notes |
|---------|----:|-------|
| MLC-LLM q4f16_1 | **73.4** | Best overall; FlashInfer attention |
| llama.cpp IQ4_XS+FA | 60.28 | Best open-source CLI tool |
| llama.cpp Q4_K_S+FA | 53.39 | Simpler quant, good accuracy |
| TRT-LLM FP16 | 44.08 | Worse — no gemm_plugin on Jetson |

### Energy Landscape

| Model Type | Representative | Energy/unit | Hardware |
|-----------|----------------|------------|----------|
| ANN (LLM) | LFM2.5-1.2B Q4_K_S | ~14.8 mJ/tok (est. at 15W, 67.5 tok/s) | Jetson GPU |
| SNN (vision) | Spikformer | 48.45 mJ/img | Jetson GPU |
| SNN (language) | SpikeGPT-216M | 88.15 mJ/tok | Jetson GPU |
| SNN (projected) | Spikformer on Loihi 2 | ~0.05 mJ/img (est.) | Neuromorphic |

On GPU, ANNs are more energy-efficient than SNNs. The SNN value proposition requires neuromorphic hardware.

---

## 7. Technical Lessons

### What Worked
- **Systematic optimization ladder:** One change at a time with measured results prevented false conclusions.
- **IQ quantization:** imatrix-aware quantization (IQ4_XS) is strictly better than static k-quant (Q4_K_S) for models with enough channel variation.
- **Flash Attention:** Free 9-12% speed win on all models. No downside.
- **Chat templates:** Unlocked 35pp GSM8K improvement on Llama — a prompt engineering win, not a model change.
- **MLC-LLM:** 73.4 t/s beats llama.cpp (60.3 t/s) by 21.7% for Llama-3.2-1B — the compiled TVM approach pays off for autoregressive decode.

### What Didn't Work
- **Speculative decoding on Jetson:** Every path blocked — NvMap IOVM prevents multi-model loading, EAGLE-2/3 engines are broken or too slow on sm_87.
- **TRT-LLM on 8 GB Jetson:** Only viable for 0.5B models. The NvMap IOVM limit and missing gemm_plugin make it slower than llama.cpp for larger models.
- **KV cache quantization at short context:** All types cause severe regression at 512 tokens. Only neutral at 4K+.
- **ExLlamaV2:** CUDA driver too old for JetPack 6.2.
- **FP8 anything:** sm_87 (Ampere) has no FP8 hardware. Requires sm_89+ (Ada) or sm_90+ (Hopper).

### Hardware Bottlenecks (Jetson Orin Nano 8 GB)

| Bottleneck | Impact | Workaround |
|-----------|--------|------------|
| 68 GB/s bandwidth | Caps single-sequence decode at ~136 t/s for 1B INT4 | None — physics |
| 2.5 GB NvMap IOVM | Blocks multi-model, large TRT engines, gemm_plugin | Smaller models, single-model inference |
| No FP8 (sm_87) | Blocks TRT Edge-LLM FP8, ExLlamaV2 FP8 | INT4/INT8 only |
| 8 GB UMA total | Limits model + KV cache + runtime to ~2 GB usable GPU | Aggressive quantization, small context |

---

## 8. Recommendations

### For Production Deployment (Today)

| Use Case | Recommended Stack | Expected Performance |
|----------|------------------|---------------------|
| Fast >1B LLM inference | MLC-LLM + Llama-3.2-1B q4f16_1 | 73.4 t/s, 45% ARC |
| Best accuracy >1B | llama.cpp + LFM2.5-1.2B IQ4_XS+FA | 65.6 t/s, 60% ARC |
| Fastest LLM (any size) | TRT-LLM + Qwen2.5-0.5B W4A16 | 100.9 t/s |
| Real-time vision | V-JEPA 2.1 ViT-B/16 | 74.3 FPS, 421 MB |
| Multimodal reasoning | V-JEPA → Qwen2.5-0.5B pipeline | 3.4 FPS, 719 MB |
| Lowest memory vision | Spiking world model (latency) | 19.8 ms, 245 MB |

### For Future Work

1. **JetPack 7.x / newer MLC-LLM:** May fix EAGLE-2 engine and enable spec decoding → 100+ t/s for >1B.
2. **Neuromorphic hardware (Loihi 2, Akida):** Deploy Spikformer (99.94% sparsity) for ~1,000x energy improvement over GPU.
3. **Fine-tuned JEPA→SNN:** Train with SNN loss + structured pruning + ALIF neurons for semantic (not random) spiking embeddings.
4. **Larger Jetson (Orin NX 16 GB):** Doubles IOVM budget → speculative decoding may fit; 102 GB/s bandwidth improves ceiling to ~100 t/s for 1B.

---

## Appendix: Repository Structure

```
Aneurologic/
├── modelgarden/          # Phase 1 — Base model inference benchmarks
├── modelgardensnn/       # Phase 3 — Spiking Neural Networks
├── modelgardenJEPA/      # Phase 4 — JEPA World Models
├── AneuroOptimizedModels/ # Phase 5+6 — Edge optimization & 100 t/s quest
├── AN_MoE/               # MoE + SNN integration experiments
├── Web/                  # Project website
└── picoclaw/             # CLI tooling
```

## Appendix: Jetson Model Files

All GGUF models at `/home/spitman/Projects/Aneurologic/modelgarden/jetson-containers/data/models/standardized/`.
MLC-LLM models at `.../data/models/mlc/`.
TRT-LLM engines at container-internal paths (aneurologic_phase5).
