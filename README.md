# ANeurologic Phase 5 — Advanced Optimization

**Extreme compression and acceleration of edge AI models on NVIDIA Jetson Orin Nano 8 GB.**

Part of the [ANeurologic](https://github.com/Spitmaan) initiative.

Builds on Phase 1 (modelgarden) baselines with:
- **TurboQuant** — Google Research KV-cache compression (PolarQuant + QJL, ICLR 2026)
- **TensorRT-LLM** — Hardware-accelerated engine builds (v0.12.0-jetson)
- **Go-native inference** — Concurrent serving via gollama.cpp + purego
- **Teacher-Student Distillation** — KL-divergence dark knowledge transfer

---

## Selected Models (from Phase 1 top performers)

| Model | Type | Phase 1 Baseline | HuggingFace ID |
|-------|------|-----------------|----------------|
| Liquid AI LFM2.5 1.2B | SLM | 55.4 t/s | `LiquidAI/LFM2.5-1.2B-Instruct` |
| Llama 3.2 1B | SLM | 44.7 t/s | `meta-llama/Llama-3.2-1B-Instruct` |
| Cosmos-Reason2 2B | SLM | 34.3 t/s | `nvidia/Cosmos-Reason2-2B` |
| LFM2-VL-450M | VLM | 42.1 t/s | `LiquidAI/LFM2-VL-450M` |
| LFM2-VL-1.6B | VLM | 24.8 t/s | `LiquidAI/LFM2-VL-1.6B` |

---

## Pipeline

```
Phase 1 Baselines (throughput)
        │
        ▼
Stage 2 — Reasoning Accuracy (GSM8K / ARC-Challenge via lm-eval)
        │
        ▼
Stage 3 — TurboQuant KV Cache (QJL 1-bit + PolarQuant + KIVI fallback)
        │  Target: ~6x KV-cache reduction, ~8x attention speedup
        │
        ▼
Stage 4 — Go-native Inference (gollama.cpp / purego + Ollama Go API)
        │  Multi-client concurrent serving, mem-safe, t/s benchmark
        │
        ▼
Stage 5 — TensorRT-LLM Engine (v0.12.0-jetson, W4A16 AWQ)
        │  trtllm-build → .engine files → throughput benchmark
        │
        ▼
Stage 6 — Teacher-Student Distillation (KL-divergence, specialized domain)
        │  Teacher: Llama-3.1-70B via API  |  Student: LFM2.5 / Llama 3.2 1B
        │
        ▼
Stage 7 — Final Report & Leaderboard
           Vanilla vs Optimized: Speed / Footprint / Reasoning Accuracy
```

---

## Stages & Reports

| Stage | Script | Report | Status |
|-------|--------|--------|--------|
| 1 — Environment | `scripts/stage1_env/verify_env.py` | [Stage 1 Report](outputs/reports/stage1_environment.md) | 🔄 |
| 2 — Baseline Reasoning | `scripts/stage2_baseline/baseline_reasoning.py` | [Stage 2 Report](outputs/reports/stage2_baseline.md) | ⏳ |
| 3 — TurboQuant KV | `scripts/stage3_turboquant/kv_compression.py` | [Stage 3 Report](outputs/reports/stage3_turboquant.md) | ⏳ |
| 4 — Go Inference | `go_server/` + `scripts/stage4_go_inference/bench_go.py` | [Stage 4 Report](outputs/reports/stage4_go_inference.md) | ⏳ |
| 5 — TensorRT-LLM | `scripts/stage5_tensorrt/build_engines.py` | [Stage 5 Report](outputs/reports/stage5_tensorrt.md) | ⏳ |
| 6 — Distillation | `scripts/stage6_distillation/distill.py` | [Stage 6 Report](outputs/reports/stage6_distillation.md) | ⏳ |
| 7 — Final Report | `scripts/stage7_report/generate_report.py` | [Stage 7 Report](outputs/reports/stage7_phase5_report.md) | ⏳ |

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

### 3. Run stages (sequentially, with user approval between each)
```bash
# Stage 2 — Baseline reasoning accuracy
docker exec aneurologic_phase5 python3 /workspace/scripts/stage2_baseline/baseline_reasoning.py

# Stage 3 — TurboQuant KV cache compression
docker exec aneurologic_phase5 python3 /workspace/scripts/stage3_turboquant/kv_compression.py

# Stage 4 — Go inference server
cd go_server && go run main.go &
docker exec aneurologic_phase5 python3 /workspace/scripts/stage4_go_inference/bench_go.py

# Stage 5 — TensorRT-LLM engines
docker exec aneurologic_phase5 python3 /workspace/scripts/stage5_tensorrt/build_engines.py

# Stage 6 — Distillation
docker exec aneurologic_phase5 python3 /workspace/scripts/stage6_distillation/distill.py

# Stage 7 — Final report
docker exec aneurologic_phase5 python3 /workspace/scripts/stage7_report/generate_report.py
```

---

## Hardware Target

| | |
|--|--|
| **Board** | NVIDIA Jetson Orin Nano Developer Kit 8 GB |
| **JetPack** | 6.x (r36.4) |
| **CUDA** | 12.6 |
| **TensorRT** | 10.3+ |
| **PyTorch** | 2.3.0 |
| **Go** | 1.22.5 |
| **Container base** | `nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.3-py3` |

---

## References

### KV-Cache Compression (Stage 3)
- [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) — Google Research (arXiv:2504.19874, ICLR 2026)
- [PolarQuant](https://arxiv.org/abs/2502.02617) — Zandieh et al. (AISTATS 2026)
- [QJL](https://arxiv.org/abs/2406.03482) — Zandieh, Daliri, Han (AAAI 2025) | [Code](https://github.com/amirzandieh/QJL)
- [KIVI](https://arxiv.org/abs/2402.02750) — Liu et al. (ICML 2024) | [Code](https://github.com/jy-yuan/KIVI)

### TensorRT-LLM (Stage 5)
- [TensorRT-LLM v0.12.0-jetson](https://github.com/NVIDIA/TensorRT-LLM/tree/v0.12.0-jetson)

### Go Inference (Stage 4)
- [gollama.cpp](https://github.com/dianlight/gollama.cpp) — purego Go bindings for llama.cpp
- [Ollama Go API](https://pkg.go.dev/github.com/ollama/ollama/api)

---

## Related Phases

- **Phase 1** — `modelgarden`: ANN SLMs/VLMs baseline benchmarks
- **Phase 3** — `modelgardensnn`: Spiking Neural Networks
- **Phase 4** — `modelgardenJEPA`: World Models + JEPA
- **Phase 5** — `AneuroOptimizedModels`: Advanced Optimization (this repo)
