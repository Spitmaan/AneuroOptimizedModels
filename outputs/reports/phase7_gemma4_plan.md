# Phase 7 — Gemma 4 on Jetson Orin Nano

**Goal:** Benchmark Google Gemma 4 E2B and E4B on Jetson Orin Nano 8 GB — speed, accuracy, and multimodal capabilities. Apply the Edge Optimization Ladder from Phase 5.
**Date:** 2026-04-11
**Hardware:** Jetson Orin Nano 8 GB (sm_87, 68 GB/s UMA, JetPack 6.2)

---

## Why Gemma 4?

Gemma 4 E2B is a fundamentally different value proposition than Llama-3.2-1B or LFM2.5-1.2B:

| Capability | Llama-3.2-1B | LFM2.5-1.2B | Gemma 4 E2B |
|-----------|:------------:|:------------:|:-----------:|
| Text generation | 73.4 t/s | 65.6 t/s | ~16-22 t/s (est.) |
| Image understanding | No | No | Yes |
| Audio understanding | No | No | Yes |
| Video understanding | No | No | Yes |
| Function calling | No | No | Yes |
| Thinking mode | No | No | Yes |
| Context window | 128K | 32K | 128K |
| Languages | ~30 | ~30 | 140+ |
| MMLU Pro | ~35% (est.) | — | 60.0% |
| License | Llama | Proprietary | Apache 2.0 |

It won't win on raw text speed, but it fills the **smart multimodal edge assistant** slot that no model in our current lineup covers.

---

## Model Specifications

### Gemma 4 E2B

| Property | Value |
|----------|-------|
| Effective params | 2.3B |
| Total params (w/ PLE) | **5.1B** |
| Architecture | Dense (NOT MoE) |
| Layers | 35 |
| Attention | Hybrid: sliding window (512) + global |
| Context | 128K tokens |
| Vocabulary | 262K tokens |
| Vision encoder | ~150M params |
| Audio encoder | ~300M params |
| Modalities | Text, Image, Audio, Video |

### Gemma 4 E4B

| Property | Value |
|----------|-------|
| Effective params | 4.5B |
| Total params (w/ PLE) | **8B** |
| Architecture | Dense (NOT MoE) |
| Layers | 42 |
| Attention | Hybrid: sliding window (512) + global |
| Context | 128K tokens |
| Vocabulary | 262K tokens |

### Critical Note: "E2B" Is NOT 2B for Inference

Per-Layer Embeddings (PLE) inflate total parameters to 5.1B for E2B and 8B for E4B. During inference, ALL parameters are read every token (dense model). Bandwidth cost is proportional to total params, not "effective" params.

### GGUF Sizes (Unsloth)

| Quant | E2B | E4B |
|-------|----:|----:|
| IQ3_XXS | 2.37 GB | 3.70 GB |
| Q3_K_M | 2.54 GB | 4.06 GB |
| IQ4_XS | 2.98 GB | 4.72 GB |
| Q4_K_M | 3.11 GB | 4.98 GB |
| Q8_0 | 5.05 GB | 8.19 GB |

### IOVM Feasibility (Jetson Orin Nano, ~2.5 GB limit)

| Config | Weight Size | Full GPU? | Expected t/s |
|--------|------------|:---------:|-------------:|
| E2B IQ3_XXS | 2.37 GB | Likely YES | ~22-28 |
| E2B Q3_K_M | 2.54 GB | BORDERLINE | ~20-25 |
| E2B IQ4_XS | 2.98 GB | Probably NO | ~16-22 (partial) |
| E2B Q4_K_M | 3.11 GB | NO | ~15-20 (partial) |
| E4B IQ3_XXS | 3.70 GB | NO | ~12-15 (partial) |
| E4B Q4_K_M | 4.98 GB | NO | ~8-10 (heavy partial) |

Bandwidth formula: `68 GB/s * 0.72 efficiency / weight_size_GB = estimated t/s`

---

## Execution Plan

### Stage 1 — E2B Quick Feasibility

**Status:** PENDING

1. Pull official NVIDIA container: `ghcr.io/nvidia-ai-iot/llama_cpp:gemma4-jetson-orin`
2. Download E2B GGUFs from `unsloth/gemma-4-E2B-it-GGUF`: Q3_K_M, IQ4_XS, Q4_K_M
3. Test full GPU offload (`-ngl 99`) with each quant
4. If IOVM OOM: find max `-ngl N` that fits
5. Benchmark: pp512, tg128, VRAM usage
6. Record which quant fits and at what offload level

### Stage 2 — E2B Edge Optimization Ladder

**Status:** PENDING

Apply Phase 5 methodology:
1. Flash Attention (`-fa 1`) — expect +9-12%
2. Quant comparison: IQ3_XXS vs Q3_K_M vs IQ4_XS vs Q4_K_M
3. Context window: test at `-c 2048` and `-c 4096` (reduce from 128K default)
4. Find optimal config for this architecture
5. Compare vs Phase 5 best configs (Llama 73.4, LFM 65.6)

### Stage 3 — E2B Accuracy Benchmarks

**Status:** PENDING

1. GSM8K (math reasoning) — same methodology as Phase 5
2. ARC-Challenge (science reasoning) — same methodology
3. Test with and without thinking mode (`enable_thinking=True`)
4. Compare vs Llama-3.2-1B (45% ARC, 40% GSM8K) and LFM2.5 (60% ARC)

### Stage 4 — E2B Multimodal

**Status:** PENDING

1. Image understanding via llama.cpp `--mmproj` or multimodal API
2. Basic VQA tasks (describe image, OCR, object detection)
3. Compare vs LFM2-VL-450M from Phase 1 (42.1 t/s gen)
4. Audio transcription if llama.cpp Gemma 4 audio is supported

### Stage 5 — E4B Assessment

**Status:** PENDING

1. Download E4B IQ3_XXS (3.70 GB) and Q3_K_M (4.06 GB)
2. Test with partial GPU offload — find max `-ngl N`
3. If runnable: benchmark speed and accuracy
4. If OOM or too slow (<5 t/s): document and skip
5. Compare E4B accuracy vs E2B to determine if the speed cost is worth it

### Stage 6 — MLC-LLM Check (Stretch Goal)

**Status:** PENDING

1. Check if `dustynv/mlc:0.20.0-r36.4.0` supports `gemma4` model type
2. If yes: compile E2B with q4f16_1 and benchmark (potential +20% vs llama.cpp)
3. If no: check for newer MLC-LLM containers with Gemma 4 support

---

## Not Attempting

| Approach | Why |
|----------|-----|
| TRT-LLM | Same IOVM/gemm_plugin issues as Phase 5-6 |
| TRT Edge-LLM | No EAGLE3 for Gemma 4 yet |
| Speculative decoding | No draft model; EAGLE broken on sm_87 |
| E4B Q4_K_M+ | 5+ GB won't fit; CPU-heavy = very slow |
| E2B Q8_0 | 5.05 GB won't fit in IOVM |

---

## Results Log

*(Updated as stages complete)*

