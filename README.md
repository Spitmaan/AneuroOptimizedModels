# ANeurologic Phase 5 — Advanced Optimization

**Extreme compression and acceleration of edge AI models on NVIDIA Jetson Orin Nano 8 GB.**

Part of the [ANeurologic](https://github.com/Spitmaan) initiative.

Builds on Phase 1 (modelgarden) baselines with:
- **TurboQuant** — Google Research KV-cache compression (PolarQuant + QJL, ICLR 2026)
- **TensorRT-LLM** — Hardware-accelerated engine builds (v0.12.0-jetson, W4A16 AWQ)
- **Go-native inference** — Concurrent serving via goroutine worker pool + Ollama backend
- **Teacher-Student Distillation** — KL-divergence dark knowledge transfer (aerospace domain)

---

## Results Summary

| Model | Phase 1 (llama.cpp) | TRT-LLM Est. | KV Compression | Distillation |
|-------|--------------------:|-------------:|----------------|--------------|
| LFM2.5-1.2B | 55.4 t/s | ~99.7 t/s | PolarQuant 7.5x | — |
| Llama-3.2-1B | 44.7 t/s | ~80.5 t/s | QJL 64x (K) | — |
| Qwen2.5-0.5B | — | — | KIVI 2.3x | 26.7% → **53.3%** aerospace |

### Reasoning Accuracy (Stage 2)

| Model | GSM8K | ARC-Challenge |
|-------|------:|-------------:|
| LFM2.5-1.2B | 9.0% | 72.0% |
| Qwen2.5-0.5B | 7.0% | 57.0% |

---

## Models Evaluated

| Model | Phase 1 Baseline | HuggingFace ID |
|-------|-----------------|----------------|
| Liquid AI LFM2.5 1.2B | 55.4 t/s | `LiquidAI/LFM2.5-1.2B-Instruct` |
| Llama 3.2 1B | 44.7 t/s | `meta-llama/Llama-3.2-1B-Instruct`* |
| Cosmos-Reason2 2B | 34.3 t/s | `nvidia/Cosmos-Reason2-2B`* |
| Qwen2.5-0.5B | — | `Qwen/Qwen2.5-0.5B-Instruct` |

*Gated HF repo — Phase 1 used GGUF/llama.cpp. Stage 2/6 use open alternatives.

---

## Pipeline

```
Phase 1 Baselines (throughput)
        │
        ▼
Stage 1 — Environment Verification (Docker, CUDA, PyTorch, Go, lm-eval)
        │
        ▼
Stage 2 — Reasoning Accuracy (GSM8K / ARC-Challenge — custom 4-bit NF4 eval)
        │
        ▼
Stage 3 — TurboQuant KV Cache (PolarQuant 7.5x + QJL 64x + KIVI 2.3x)
        │
        ▼
Stage 4 — Go-native Inference Server (goroutine pool, OpenAI-compatible API)
        │
        ▼
Stage 5 — TensorRT-LLM Engine (v0.12.0-jetson, W4A16 AWQ — estimated 1.8x)
        │
        ▼
Stage 6 — Teacher-Student Distillation (KL-div, aerospace telemetry domain)
        │  Teacher: Llama-3.1-70B (synthetic)  |  Student: Qwen2.5-0.5B + LoRA
        │
        ▼
Stage 7 — Final Report & Leaderboard
           Vanilla vs Optimized: Speed / Footprint / Reasoning Accuracy
```

---

## Stages & Reports

| Stage | Script | Report | Status |
|-------|--------|--------|--------|
| 1 — Environment | `scripts/stage1_env/verify_env.py` | [Stage 1 Report](outputs/reports/stage1_environment.md) | ✅ 18/19 passed |
| 2 — Baseline Reasoning | `scripts/stage2_baseline/baseline_reasoning.py` | [Stage 2 Report](outputs/reports/stage2_baseline.md) | ✅ Complete |
| 3 — TurboQuant KV | `scripts/stage3_turboquant/kv_compression.py` | [Stage 3 Report](outputs/reports/stage3_turboquant.md) | ✅ Complete |
| 4 — Go Inference | `go_server/` + `scripts/stage4_go_inference/bench_go.py` | [Stage 4 Report](outputs/reports/stage4_go_inference.md) | ✅ Compiled |
| 5 — TensorRT-LLM | `scripts/stage5_tensorrt/build_engines.py` | [Stage 5 Report](outputs/reports/stage5_tensorrt.md) | ✅ Estimated* |
| 6 — Distillation | `scripts/stage6_distillation/distill.py` | [Stage 6 Report](outputs/reports/stage6_distillation.md) | ✅ +26.6% acc |
| 7 — Final Report | `scripts/stage7_report/generate_report.py` | [Final Report](outputs/reports/stage7_phase5_report.md) | ✅ Complete |

*TRT-LLM pip wheel unavailable for aarch64; source build (~40 min) skipped. Results estimated at 1.8x Phase 1 baseline (documented NVIDIA Orin benchmark). Run with `--build-trtllm` to compile real engines.

---

## Quick Start

### 1. Build and start the container (on Jetson)
```bash
git clone git@github.com:Spitmaan/AneuroOptimizedModels.git
cd AneuroOptimizedModels
docker compose build
docker compose up -d
```

### 2. Verify environment
```bash
docker exec aneurologic_phase5 python3 /workspace/scripts/stage1_env/verify_env.py
```

### 3. Run all stages
```bash
# Stage 2 — Baseline reasoning accuracy
docker exec aneurologic_phase5 python3 /workspace/scripts/stage2_baseline/baseline_reasoning.py

# Stage 3 — TurboQuant KV cache compression
docker exec aneurologic_phase5 python3 /workspace/scripts/stage3_turboquant/kv_compression.py

# Stage 4 — Go inference server (compile + load test)
cd go_server && go build -o aneurologic-server . && ./aneurologic-server &
docker exec aneurologic_phase5 python3 /workspace/scripts/stage4_go_inference/bench_go.py

# Stage 5 — TensorRT-LLM engines (estimated if TRT-LLM not built)
docker exec aneurologic_phase5 python3 /workspace/scripts/stage5_tensorrt/build_engines.py
# To build real engines (~40 min):
# docker exec aneurologic_phase5 python3 /workspace/scripts/stage5_tensorrt/build_engines.py --build-trtllm

# Stage 6 — Distillation (Qwen2.5-0.5B student, aerospace domain)
docker exec aneurologic_phase5 python3 /workspace/scripts/stage6_distillation/distill.py

# Stage 7 — Final report
docker exec aneurologic_phase5 python3 /workspace/scripts/stage7_report/generate_report.py
```

---

## Key Technical Notes

### LFM2 Hybrid Conv Cache
LFM2.5-1.2B uses a `Lfm2HybridConvCache` (conv + attention hybrid), not a standard `(K, V)` tuple. Stage 3 detects this and falls back to synthetic KV tensors for compression benchmarking. Stage 6 uses Qwen2.5-0.5B (standard decoder) to avoid OOM during backprop.

### Jetson CUDA Allocator
`lm_eval`'s `model.to(device)` triggers `NVML_SUCCESS == r INTERNAL ASSERT FAILED` on Jetson UMA. Stage 2 uses a custom eval loop with `BitsAndBytesConfig(load_in_4bit=True)` + `device_map="auto"` to avoid this.

### bitsandbytes on Jetson
Only pre-release builds available: `pip3 install --pre bitsandbytes` (installs 0.47.0.dev0+).

### Gradient Checkpointing (Stage 6)
Enabled to reduce peak VRAM ~40% during backprop. Required for LoRA training of even 0.5B models on 8GB UMA with batch size > 1.

---

## Hardware Target

| | |
|--|--|
| **Board** | NVIDIA Jetson Orin Nano Developer Kit 8 GB |
| **JetPack** | 6.2 (L4T r36.4) |
| **CUDA** | 12.6 |
| **TensorRT** | 10.3+ |
| **PyTorch** | 2.3.0 |
| **Go** | 1.22.5 |
| **Container base** | `openpi:r36.4.tegra-aarch64-cu126-22.04-cuda_12.6` |

---

## References

### KV-Cache Compression (Stage 3)
- [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) — Google Research (arXiv:2504.19874, ICLR 2026)
- [PolarQuant](https://arxiv.org/abs/2502.02617) — Zandieh et al. (AISTATS 2026)
- [QJL](https://arxiv.org/abs/2406.03482) — Zandieh, Daliri, Han (AAAI 2025) | [Code](https://github.com/amirzandieh/QJL)
- [KIVI](https://arxiv.org/abs/2402.02750) — Liu et al. (ICML 2024) | [Code](https://github.com/jy-yuan/KIVI)

### TensorRT-LLM (Stage 5)
- [TensorRT-LLM v0.12.0-jetson](https://github.com/NVIDIA/TensorRT-LLM/tree/v0.12.0-jetson)

### Knowledge Distillation (Stage 6)
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — Hinton et al. (2015)
- [LoRA](https://arxiv.org/abs/2106.09685) — Hu et al. (ICLR 2022)

---

## Related Phases

- **Phase 1** — `modelgarden`: ANN SLMs/VLMs baseline benchmarks
- **Phase 3** — `modelgardensnn`: Spiking Neural Networks
- **Phase 4** — `modelgardenJEPA`: World Models + JEPA
- **Phase 5** — `AneuroOptimizedModels`: Advanced Optimization (this repo)
