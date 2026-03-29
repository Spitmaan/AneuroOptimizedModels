# Stage 2 — Baseline Reasoning Benchmarks

**Project:** ANeurologic Phase 5 — Advanced Optimization
**Hardware:** NVIDIA Jetson Orin Nano 8 GB
**Date:** 2026-03-29 00:31
**Tasks:** GSM8K (math, 5-shot), ARC-Challenge (science, 5-shot)
**Limit:** 100 examples per task
**Quantization:** 4-bit NF4 (bitsandbytes, same as Phase 4 SLM fix)

> These scores are the **reasoning floor** — the baseline that
> Stages 3–6 (TurboQuant, TensorRT-LLM, Distillation) must preserve or exceed.

## Note on lm_eval

lm_eval was attempted first but fails on Jetson due to
`CUDACachingAllocator.cpp:1154` assertion in PyTorch/Jetson when
`model.to(device)` is called (even for 0.5B models). Root cause:
lm_eval's HF backend does not use `device_map='auto'` + BitsAndBytesConfig.
This custom loop replicates the same scoring methodology (5-shot,
exact-match for GSM8K, log-likelihood MC for ARC) with proper
4-bit NF4 loading to stay within Jetson's 3 GB CUDA budget.

---

## Results Summary

| Model | Phase 1 t/s | GSM8K | ARC-Challenge | VRAM (eval) |
|-------|------------|-------|---------------|-------------|
| LFM2.5-1.2B | 55.4 t/s | 9.0% | 72.0% | 1123.6 MB |
| Qwen2.5-1.5B | 21.3 t/s | Error | Error | ? MB |
| Qwen2.5-0.5B | N/A t/s | 7.0% | 57.0% | 1178.1 MB |

---

## Per-Model Detail

### LFM2.5-1.2B

- **HuggingFace ID**: `LiquidAI/LFM2.5-1.2B-Instruct`
- **Phase 1 throughput**: 55.4 t/s (Q4_K_M, llama.cpp)
- **Phase 1 VRAM**: 1.2 GB
- **Eval quantization**: 4-bit NF4
- **Peak VRAM (load)**: 839.0 MB
- **Peak VRAM (eval)**: 1123.6 MB

  **gsm8k**: 9.0%  (9/100 correct, 555s, metric: `exact_match_number`)
  **arc_challenge**: 72.0%  (72/100 correct, 179s, metric: `log_likelihood_mc`)

### Qwen2.5-1.5B

- **HuggingFace ID**: `?`
- **Phase 1 throughput**: 21.3 t/s (Q4_K_M, llama.cpp)
- **Phase 1 VRAM**: 1.5 GB
- **Eval quantization**: 4-bit NF4
- **Peak VRAM (load)**: ? MB
- **Peak VRAM (eval)**: ? MB


  > *Note: Non-gated Llama-class proxy (same param range as Llama-3.2-1B)*

### Qwen2.5-0.5B

- **HuggingFace ID**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Phase 1 throughput**: N/A t/s (Q4_K_M, llama.cpp)
- **Phase 1 VRAM**: 0.5 GB
- **Eval quantization**: 4-bit NF4
- **Peak VRAM (load)**: 1116.1 MB
- **Peak VRAM (eval)**: 1178.1 MB

  **gsm8k**: 7.0%  (7/100 correct, 877s, metric: `exact_match_number`)
  **arc_challenge**: 57.0%  (57/100 correct, 98s, metric: `log_likelihood_mc`)

---

## Methodology

### GSM8K
- 5-shot chain-of-thought prompting
- Model generates free-form text; final number extracted via regex
- Match against ground truth number from `#### N` annotation

### ARC-Challenge
- 5-shot multiple-choice prompting (A/B/C/D)
- Log-likelihood scoring: score P(label | prompt) for each choice
- Highest log-likelihood wins (standard for MC evaluation)

## How to Reproduce

```bash
# Full eval (all models, 100 examples per task)
docker exec aneurologic_phase5 python3 \
    /workspace/scripts/stage2_baseline/baseline_reasoning.py

# Quick eval (50 examples, specific models)
docker exec aneurologic_phase5 python3 \
    /workspace/scripts/stage2_baseline/baseline_reasoning.py \
    --models qwen05b --limit 50
```
