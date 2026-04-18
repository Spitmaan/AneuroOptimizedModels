# Phase 13 — Extreme Optimization of Top LLM / VLM / VLA Models

**Project:** `AneuroOptimizedModels`
**Date:** 2026-04-17
**Goal:** Take the best-performing LLM, VLM, and VLA we've identified across Phases 1–12 and push each through a systematic optimization ladder — quantization, distillation, runtime compilation, kernel fusion — measuring speed *and* accuracy at every rung. Output: production-ready, measurably-better replacements for the current lineup.

**Depends on:**
- Phase 10 complete (new model lineup deployed)
- Phase 11 closeout (techniques from Orion-Lite KB available)
- Phase 12 complete (real VLA baselines to optimize)

---

## Target Models (starting points)

From Phase 10 benchmarks on Orin Nano 8GB:

| Category | Champion (baseline) | Baseline speed | Baseline accuracy |
|---|---|---:|---|
| **LLM** | Aneuro Cortex Ultra (Llama 3.2 1B MLC q4f16_1) | 73 t/s | ~60% ARC-C |
| **LLM (speed)** | Aneuro Nova (Gemma 3 270M Q4_K_M) | 63.7 t/s | TBM — probably 30-ish% ARC-C |
| **VLM** | Aneuro Lumen (LFM2-VL 1.6B Q4_0) | 35.7 t/s | Visual QA TBM |
| **VLA** | Aneuro Tendon (SmolVLA 450M fp16) — Phase 12 | ~500 ms/step | ~78% LIBERO (claimed) |

*Each of these gets its own full optimization ladder in Phase 13.*

---

## Methodology — the "Optimization Ladder"

Extends the Phase 5 framework (`bench_gguf.py`) to all three modalities. Every rung is measured with the same protocol:

**Protocol per rung:**
1. Load model on Jetson Orin Nano 8GB
2. Warmup (5 runs, discarded)
3. 20 timed runs, report median + p95
4. VRAM peak (`nvidia-smi` equivalent + `free -m`)
5. Accuracy on a task-appropriate held-out set (GSM8K/ARC-C for LLM; MM-Bench mini for VLM; LIBERO-Spatial subset for VLA)
6. Record in `outputs/phase13/<model>_<rung>.json`

**Acceptance criteria for promoting a rung to production:**
- Speed gain ≥ 15% vs previous rung
- Accuracy drop ≤ 2 pp
- Peak VRAM unchanged or lower

---

## Ladder A — LLM (Aneuro Cortex Ultra → "Cortex Prime")

### Rung 0 — Baseline
- Model: Llama 3.2 1B MLC q4f16_1 (Cortex Ultra, Phase 8)
- Measured: 73 t/s, peak ~1.0 GB VRAM

### Rung 1 — Chat-template + KV-cache polish
- Apply best prompt template (Phase 5 Stage XIV finding — non-default templates +3–5% accuracy for free)
- Enable Q8 KV cache via `--cache-type-k q8_0 --cache-type-v q8_0`
- Expected: +5–10% speed, no accuracy change

### Rung 2 — MLC imatrix recalibration
- Rebuild MLC compiled library with imatrix-calibrated weights (imatrix generated on ARC + GSM8K + domain conversations)
- Expected: +2–5% accuracy at same speed

### Rung 3 — TVM kernel recompile for sm_87
- Re-run MLC `compile --quantization q4f16_1 --target sm_87` with explicit kernel selection
- Investigate whether swap-to-FP8 compute (NVFP4) is usable on Orin (Ampere sm_87 doesn't have FP8 — decision: **skip, not supported**)

### Rung 4 — Distilled draft model for speculative decoding
- Train a 200M-class draft model from this Cortex Ultra teacher (feature-mimic L1, per Orion-Lite recipe from Phase 11 KB)
- Run llama-server / MLC with `--draft-model`
- Expected: 1.5–2× wall-clock speedup on structured outputs

### Rung 5 — Task-specific distillation (stretch)
- Distill a 200M "Cortex Mini" specifically for the chat domain using our existing conversation logs
- Apache-2.0 SmolLM2 or Gemma 3 270M as student architecture
- Expected: ~50–60% of Cortex Ultra's quality at 3× speed

### Output: **Aneuro Cortex Prime** — pushed through at least Rungs 1-3 (Rungs 4-5 if time allows)

---

## Ladder B — VLM (Aneuro Lumen → "Lumen Prime")

### Rung 0 — Baseline
- Model: LFM2-VL 1.6B Q4_0 GGUF
- Measured: 35.7 t/s, ~2.5 GB RAM

### Rung 1 — Better quant (IQ4_XS)
- Rebuild with imatrix-aware IQ4_XS quantization
- LFM/LiquidAI doesn't ship imatrix files — generate our own from a general VL calibration set
- Expected: same speed, +5% accuracy (per Phase 5 Llama findings IQ4_XS beat Q4_K_S)

### Rung 2 — Smaller mmproj
- Current mmproj Q8 is ~200 MB. Try F16 mmproj vs Q4 mmproj — Phase 9 Iris work showed Q8 mmproj works but we haven't swept this
- Measure: accuracy on our mini VQA set at Q4 / Q6 / Q8 / F16 mmproj

### Rung 3 — Token-compression prefix
- Apply JEPA-style adaptive-pool visual token compression before SLM prefill (we proved the pattern in Phase 4)
- 2304 → 32 visual prefix tokens
- Expected: ~50% prefill speedup, unknown accuracy cost — must ablate

### Rung 4 — Replace vision encoder (research)
- Current: LFM2's native vision tower
- Candidate: DINOv2-small (~22M params, well-studied)
- Requires retraining the projection layer — real work
- Expected: potentially huge speedup on image tokenization, unknown quality

### Rung 5 — Full MLC-LLM compile of LFM2-VL
- LFM is SSM-hybrid so MLC doesn't support it as of April 2026 — track this as a blocked rung
- Alternative: export via ONNX → TensorRT for Jetson
- Expected: 2× speedup if the graph compiles cleanly

### Output: **Aneuro Lumen Prime** — Rungs 1-3 are quick wins; 4-5 are research projects

---

## Ladder C — VLA (Aneuro Tendon → "Tendon Prime")

### Rung 0 — Baseline
- Model: SmolVLA 450M fp16 (Phase 12)
- Measured: ~500 ms/step on Nano

### Rung 1 — bf16 / fp16 cast discipline
- Audit any fp32 ops that slipped through in LeRobot's default code path
- Force end-to-end fp16 (vision encoder, flow-matching expert, action head)
- Expected: 20–30% speedup

### Rung 2 — Flow-matching step reduction
- Default: 10 denoising steps per action chunk
- Try 6 / 4 / 2 steps; measure LIBERO accuracy at each
- Expected: linear speedup with smooth accuracy knee around 4 steps

### Rung 3 — Action chunking + caching
- Plan once → execute N steps without re-running vision (async pattern from SmolVLA paper)
- Requires the serve_smolvla.py API to expose separate `plan()` and `step()` endpoints
- Expected: 3–5× effective throughput on multi-step tasks

### Rung 4 — Q8 dynamic quantization on action expert
- PyTorch dynamic int8 on all Linear layers in the action expert
- Keep vision encoder in fp16 (flow-matching diffusion is more sensitive)
- Expected: 1.3–1.5× speedup, measure accuracy on LIBERO-Spatial

### Rung 5 — NanoVLA-style routed inference (research)
- Route "easy" actions through a lightweight fast-path; send "hard" ones through full SmolVLA
- NanoVLA paper (arXiv 2510.25122) reports 52× edge speedup — ambitious target
- Classifier trained on prior action confidence scores

### Rung 6 — Distilled SmolVLA-Tiny (stretch)
- 200M-class student with our Phase 11 feature-mimic recipe, trained on our own LeRobot demos if we have any
- Expected: ~2–3× faster than SmolVLA, unknown accuracy cost

### Output: **Aneuro Tendon Prime** — Rungs 1–3 are must-do; 4 is measurement; 5–6 are research

---

## Cross-cutting: Pi-0-FAST via GGUF (bonus track)

Parallel to Ladder C, explore what Phase 12 deferred:
- Port PaliGemma weights to llama.cpp
- Quantize to Q4_K_M
- Implement FAST detokenizer as C++ post-processor
- Benchmark against SmolVLA on LIBERO
- Literature target: LiteVLA-Edge 150 ms on AGX Orin — Nano expected 2–3× slower

**If** successful, this becomes the fifth ANEUROLOGIC VLA: **Aneuro Sinew** (autoregressive, quantized) to complement Tendon (flow-matching, fp16).

---

## Action Items (summary)

| Step | Artifact |
|---|---|
| 1 | `scripts/bench_jetson_unified.py` — extend bench_gguf.py to VLM + VLA |
| 2 | Run Ladder A Rungs 0-3; commit results |
| 3 | Run Ladder B Rungs 0-3; commit |
| 4 | Run Ladder C Rungs 0-4; commit |
| 5 | Stretch: Rungs 4-5 on each ladder |
| 6 | Update `modelgarden/docs/edge_optimization_kb.md` with measured findings |
| 7 | Deploy the Prime variants to angate: new entries (Cortex Prime, Lumen Prime, Tendon Prime) next to originals |
| 8 | Closeout: `AneuroOptimizedModels/docs/phase13_closeout.md` with results table + recommendations |

## Deliverables

- `outputs/phase13/` — per-rung JSON + CSV summary
- Updated lineup in `aneurologic-server/angate/models.json` (3 new "Prime" entries)
- One master results table in closeout
- Additions to Phase 11's `edge_optimization_kb.md` with measured numbers

## Risks

1. **Orin Nano sm_87 has no FP8** — NVFP4 / FP8 rungs not applicable
2. **MLC-LLM doesn't support LFM** — Ladder B Rung 5 blocked
3. **Speculative decoding may hurt on short responses** — Ladder A Rung 4 could regress on chat
4. **LIBERO eval needs a simulator** — might have to rely on claims + sanity tests if we can't set up LIBERO on DGX Spark
5. **Distilled drafts / students need training data** — might bottleneck on dataset curation

## Time Estimate

| Phase | Work Days |
|---|---:|
| Ladder A (LLM) | 2–3 |
| Ladder B (VLM) | 2–3 |
| Ladder C (VLA) | 3–4 |
| Bonus: Pi-0-FAST GGUF | 3–5 |
| Docs + deploy + closeout | 1–2 |
| **Total** | **~11–17 days** |

## Approval Gate

- [ ] Ladder scope confirmed (A + B + C) or trimmed
- [ ] Bonus Pi-0-FAST track approved or deferred
- [ ] Prime naming OK (Cortex Prime / Lumen Prime / Tendon Prime / Sinew)
- [ ] Order of execution (A → B → C or parallel on separate weeks)
- [ ] Phase 11 + 12 completed before starting
