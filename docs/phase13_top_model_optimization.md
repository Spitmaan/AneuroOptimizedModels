# Phase 13 — Extreme Optimization of Top LLM / VLM / VLA Models

**Project:** `AneuroOptimizedModels`
**Date:** 2026-04-17 · **Revised:** 2026-04-18 (locked-in decisions)
**Goal:** Take the best-performing LLM, VLM, and VLA we've identified across Phases 1–12 and push each through a systematic optimization ladder — quantization, distillation, runtime compilation, kernel fusion — measuring speed *and* accuracy at every rung. Output: production-ready, measurably-better replacements for the current lineup.

## Locked Decisions (2026-04-18)

- **VLA ladder kept** (Tendon Prime etc.), in addition to LLM + VLM ladders.
- **Execution:** **sequential** — Ladder A (LLM) → B (VLM) → C (VLA), with detailed reporting after each rung and ladder. User decides whether to continue after each completion.
- **Heavy-duty training in scope** — distillation-aware training, custom draft models, from-scratch students all allowed. Extra sub-phases (13.1, 13.2, …) can spin out as needed.
- **Nerve Prime track** added to Ladder C, conditional on Phase 12 producing Nerve-F and/or Nerve-D.
- **Training compute:** DGX Spark (ARM64 Blackwell) for all training / distillation / export work. Checkpoints committed with tags per long-running-training policy.

**Depends on:**
- Phase 10 complete (new model lineup deployed)
- Phase 11 closeout (2026-04-20) — Orion-Lite KB entries live at `Aneurologic_Memory/knowledge_base/edge_optimization_kb.md` §B.1, §F.6; Jetson-measured student decoder latency + VRAM at `modelgarden/docs/phase11_orion_lite_closeout.md` §2.  **Note:** our own retrained student is deferred until Rung 4 of this phase fires — published ablation checkpoints were sufficient to close Phase 11.
- Phase 12 complete (real VLA baselines to optimize, including Nerve-F/Nerve-D for Prime track)

---

## Target Models (starting points)

From Phase 10 benchmarks on Orin Nano 8GB + Phase 12 VLA results:

| Category | Champion (baseline) | Baseline speed | Baseline accuracy | Target name |
|---|---|---:|---|---|
| **LLM** | Aneuro Cortex Ultra (Llama 3.2 1B MLC q4f16_1) | 73 t/s | ~60% ARC-C | **Aneuro Cortex Prime** |
| **LLM (speed)** | Aneuro Nova (Gemma 3 270M Q4_K_M) | 63.7 t/s | TBM | **Aneuro Nova Prime** (stretch) |
| **VLM** | Aneuro Lumen (LFM2-VL 1.6B Q4_0) | 35.7 t/s | Visual QA TBM | **Aneuro Lumen Prime** |
| **VLA (flow)** | Aneuro Tendon (SmolVLA 450M fp16) | ~500 ms/step | ~78% LIBERO (claimed) | **Aneuro Tendon Prime** |
| **VLA (autoregressive)** | Aneuro Sinew (π₀-FAST Q4 GGUF) — Phase 12 | 150–300 ms/step | TBM | **Aneuro Sinew Prime** |
| **VLA (our own, flow)** | Aneuro Nerve-F — Phase 12 output | TBD | TBD on LIBERO | **Aneuro Nerve-F Prime** *(cond.)* |
| **VLA (our own, discrete)** | Aneuro Nerve-D — Phase 12 output | TBD | TBD on LIBERO | **Aneuro Nerve-D Prime** *(cond.)* |

*Each of these gets its own full optimization ladder in Phase 13. Sequential execution (Ladder A → B → C).*

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
- Train a 200M-class draft model from this Cortex Ultra teacher
- **Distillation recipe: L1 feature-mimic on the final hidden state**, inherited from Phase 11 Orion-Lite closeout.  See:
  - [`Aneurologic_Memory/knowledge_base/edge_optimization_kb.md`](../../Aneurologic_Memory/knowledge_base/edge_optimization_kb.md) §B.1 — the primitive + why L1 beats L2/KL/Huber on the tail.
  - [`modelgarden/docs/orion_lite_transferable_techniques.md`](../../modelgarden/docs/orion_lite_transferable_techniques.md) §1 — concrete application pattern for chat.
  - [`modelgarden/plans/phase11_orion_lite_research.md`](../../modelgarden/plans/phase11_orion_lite_research.md) §7 — note that Phase 11's Step 4.5-full retrain is *deferred until this Rung fires* (we need to own the full training pipeline when adapting to a new teacher/student pair).
- Concrete shape for Cortex Ultra: freeze tokenizer-embedding + first-K attention blocks of the Llama 3.2 1B teacher; train a 200 M tail (SmolLM2 or Gemma 3 270M architecture, ~5-6 transformer layers) with `ℒ = ℒ_mimic + α·ℒ_lm_ce`, where `ℒ_mimic = (1/B·C)·Σ‖T_student − T_teacher‖₁` on the final hidden state and `ℒ_lm_ce` is the usual next-token CE for stability.
- Run llama-server / MLC with `--draft-model`
- Expected: 1.5–2× wall-clock speedup on structured outputs
- **Prerequisite when this Rung fires:** Phase 11 deferred workstreams (Step 4.5-full training pipeline on Spark + Step 4.6 cross-domain distillation) both get pulled in together — single setup investment amortises across both.

### Rung 5 — Task-specific distillation (stretch)
- Distill a 200M "Cortex Mini" specifically for the chat domain using our existing conversation logs
- Apache-2.0 SmolLM2 or Gemma 3 270M as student architecture
- **Same L1 feature-mimic recipe** as Rung 4 with conversation-log-filtered teacher samples.
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

## Ladder D — Compass C.2 full-pipeline optimization (new 2026-04-24)

Aneuro Compass C.2 (Phase 12 Path C.2) shipped 2026-04-24 with the full EVA-02-L + QT-Former + student-decoder pipeline live on Jetson Orin Nano.  The pipeline is **functional** and end-to-end correct (0/0 missing/unexpected keys on both EVA and QT-Former checkpoint loads, `pipeline=c2_full` flag confirmed in /score responses, trajectory directionally sensitive to image content), but the measured latency is **~167 s/frame** because EVA-02-L runs on the 6-core CPU to sidestep Jetson IOVM OOMs when the full pipeline is co-resident on GPU.  This ladder makes Compass C.2 practical for live-camera use.

**Target:** Aneuro Compass Prime — same Orion-Lite weights, same /score contract, but ≤ 2-5 s/frame on Jetson Orin Nano 8 GB.  Output format and ego_fut_preds shape unchanged (backwards compat with the /demo/kinetic UI).

**Dependencies:**
- Phase 12 Path C.2 shipped (code at `aneurologic-server/angate/compass_c2/{eva_vit.py, qt_former.py, attention.py}` + `serve_compass.py`).
- Weights at `/home/spitman/Projects/OrionLite/weights/orion_lite.fp16.pth` on Jetson (3.48 GB fused fp16 checkpoint with `img_backbone.*` + `pts_bbox_head.{input_projection, query_embedding, transformer, output_projection}.*` + `student_model.*` prefixes already verified).
- Ladder D depends on Ladder C Rungs 0-3 for shared tooling (bench harness, memory-peak audit script).

### Rung D.1 — BnB 4-bit quantisation on EVAViT weights

- **What:** post-load, walk `eva.model` and replace every `nn.Linear` + `nn.Conv2d` with its `bitsandbytes.nn.Linear4bit` / equivalent, matching the Retina / Reflex recipe from Phase 12.  Quantisation happens lazily on first `.cuda()`.
- **Why:** EVA-02-L is 303 M params @ fp16 ≈ 600 MB of weights.  NF4 drops that to ~160 MB, which (per Reflex) is the difference between NvMap-fragmentation OOM and surviving the bulk GPU move.  Compute stays in fp16 via `bnb_4bit_compute_dtype`.
- **How:**
  1. Import `bitsandbytes.nn.Linear4bit` + `Params4bit` in `compass_c2/__init__.py` via a helper `replace_linear_4bit(module, skip_names=(...))` (reusable across future C.x rungs).
  2. Call helper on `eva` right after `load_state_dict`, before `.cuda()`.
  3. Skip `patch_embed.proj` (Conv2d to embed_dim) — that path is not a Linear and is small anyway.
  4. Measure peak VRAM during a single EVA forward with the 4-bit model loaded on GPU; if it fits (expect ~800 MB vs current OOM), move EVA back to GPU.
- **Expected:** 30-50× speedup (EVA forward drops from ~150 s on CPU to ~3-5 s on GPU; total /score to ~5 s).
- **Risk:** BnB's `bnb_4bit_compute_dtype` requires an aarch64 wheel with Jetson cu126 — `pypi.jetson-ai-lab.io/jp6/cu126/` ships `bitsandbytes-0.48.0.dev0+ff389db` which is what Retina + Reflex both use.  Known to work.  Accuracy drop on EVA is unmeasured; BnB 4-bit NF4 is typically <1 % accuracy loss on ViT-scale models, but the Orion-Lite trajectory head may be sensitive to fine-grained vision details.  Monitor trajectory direction on a known synthetic scene before/after.
- **Acceptance:** speedup ≥ 20×, GPU peak ≤ 3 GB, trajectory direction unchanged on the `(120, 130, 80)` synthetic test frame within 15 % magnitude.

### Rung D.2 — 320×320 EVA with rope-buffer filter

- **What:** keep EVA on GPU (post Rung D.1 or independent of it) but at half resolution — 320×320 input → 20×20 grid → 400 vision tokens instead of 1600.
- **Why:** attention memory scales O(N²) with tokens; 320×320 is 16× less peak activation than 640×640.  Allows GPU placement even without 4-bit.
- **How:**
  1. Change `_EVA_IMG_SIZE = 320` in `serve_compass.py`.
  2. Handle the `blocks.*.attn.rope.freqs_{cos,sin}` buffer size mismatch (checkpoint [1600,64] vs new-model [400,64]) by filtering those 48 buffer keys out of `eva_sd` before `load_state_dict` — they're deterministic functions of `img_size` and get re-initialised correctly during `__init__`.
  3. Verify patch_embed and abs_pos_embed still load; ViT convention is to bilinearly interpolate the abs_pos_embed to the new grid, but the EVAViT code may need a `--interpolate-pos-embed` helper call.
  4. Attempt this rung independently (without Rung D.1) to measure the "cheap quant-free" speedup alone.
- **Expected:** EVA forward ~0.5-1 s on GPU at 320×320 (vs 150 s CPU at 640×640).  Accuracy regression because Orion-Lite was trained at 640.  Some trajectory-direction degradation on scenes with fine-grained detail.
- **Risk:** the trained/inference resolution mismatch is a real distribution shift.  On the `/demo/kinetic` UI it may produce visibly wrong trajectories on complex scenes.  Compare against Rung D.1 for the accuracy/speed trade.
- **Acceptance:** speedup ≥ 50×, trajectory direction matches D.1's output within 30 % magnitude on 5 varied test frames (straight road, left curve, right curve, stop line, turn).

### Rung D.3 — Faithful 3D camera-projected PE port

- **What:** replace the `Sinusoidal2DPE` proxy in `qt_former.py` with the actual 3D-camera-frustum PE Orion-Lite's parent detector constructs, so QT-Former's cross-attention gets the same positional information it was trained with.
- **Why:** the Phase 12 C.2 ship notes "Sinusoidal2DPE is a proxy for Orion-Lite's 3D camera-projected PE; trajectories are directionally plausible but not guaranteed faithful."  A rigorous correctness pass needs the real PE.
- **How:**
  1. Trace the `pos_embed` construction in Orion-Lite's parent detector file (`mmcv/models/detectors/orion.py` or similar — not yet in the extracted `compass_c2/` tree).  It's typically built from camera intrinsics + a 3D grid of reference points in ego frame, then projected via the LiDAR-to-camera matrix.
  2. For a single-camera demo (no multi-view, no LiDAR), simplify to a fixed intrinsics matrix (calibrate against any real Orion-Lite config value).
  3. Package the PE constructor as `compass_c2.pos_embed_3d.build_position_embed(img_size, intrinsics)` returning a buffer of shape `[H*W, 384]` to concatenate with the 2D sinusoidal.
  4. Re-verify /score output on the same deterministic test image; compare trajectories against D.1/D.2 variants — expect sharper directional confidence (baseline C.2 is 71 % on a synthetic frame; a faithful PE should push this up or improve on more-realistic scenes).
- **Expected:** no latency change; correctness improvement on realistic camera frames.
- **Risk:** the pc_range / coordinate-frame mismatch is subtle.  Worst case we regress trajectory correctness rather than improve it.  Gate behind an A/B flag (`--c2-pe=2d` vs `--c2-pe=3d`) so we can ship both and measure.
- **Acceptance:** at least one of {sharper top-action confidence, lower trajectory noise, better LIBERO subset score if we can run one} vs Rung D.2.

### Ladder D — Output: Aneuro Compass Prime

- Rungs D.1 + D.2 combined should drop /score latency from 167 s → **single-digit seconds**, making Compass live-camera usable.
- Rung D.3 is the correctness-faithfulness bump; gated A/B so we don't ship regressions.
- Order: D.1 first (cheap 4-bit quant, zero distribution shift), then D.2 (resolution cut if D.1 alone isn't fast enough), then D.3 (correctness polish).

---

## Nerve Prime Track — extends Ladder C, conditional on Phase 12

If Phase 12 successfully produces Aneuro Nerve-F and/or Nerve-D (Path F custom-trained Phi-3.5-vision VLAs), Phase 13 optimizes them further:

### Ladder C-Nerve (parallel to Tendon ladder)

**Rung 0 — Baseline:** Nerve-F (flow, INT4-quant) and Nerve-D (discrete FAST, INT4 GGUF) as shipped in Phase 12

**Rung 1 — Distillation-aware fine-tune:** Re-train the last N action-head layers with QAT (quantization-aware training) on DGX Spark, keeping Phi-3.5-vision backbone frozen. Expected: recover ~1-2 pp LIBERO lost to naive INT4.

**Rung 2 — Student backbone compression:** Distil Phi-3.5-vision into a smaller backbone (e.g., SmolLM2-1.7B + SigLIP). Full-pipeline retraining on DGX Spark, multi-day. **High effort, high reward.** This is a research sub-phase (13.2).

**Rung 3 — GGUF export (Nerve-D only):** Port the Phi-3.5-vision + FAST-head architecture to llama.cpp (once mainline llama.cpp supports Phi-3.5-vision, or via custom patch). Q4_K_M quant. Target latency <200 ms on Nano.

**Rung 4 — TRT-Edge-LLM export (Nerve-D only):** If the llama.cpp path works, try TRT-Edge-LLM export next. Target sub-100 ms.

**Output:** Aneuro Nerve-F Prime + Aneuro Nerve-D Prime (one or both, depending on what converges).

---

## Sub-phase numbering convention

Per user's "extra versions/steps" directive, heavy-duty work that expands scope gets its own sub-phase:

- **Phase 13.1** — Core optimization ladders (Rungs 1–3 of A, B, C). Most rungs land here.
- **Phase 13.2** — Distillation-aware training across A/B/C Rung 4–5 (requires days of DGX Spark time). Includes distilled-backbone Nerve Prime.
- **Phase 13.3** — Export to exotic runtimes (TRT-Edge-LLM VLA export, ONNX → TensorRT chains, etc.). Separate track because blockers are research-grade.
- **Phase 13.4** — Cross-model speculative-decoding survey (see dedicated section below). Includes custom draft-model training on DGX Spark where an off-the-shelf pair isn't available.

Each sub-phase stands on its own — user can approve advancement independently after reviewing the previous sub-phase's results.

---

## Phase 13.4 — Cross-Model Speculative Decoding Survey

**Goal:** Systematically evaluate speculative-decoding options for every chat-capable Aneuro model. Where a same-family small model already exists, pair it. Where it doesn't, fine-tune a draft on DGX Spark. Output: a table of shipped "target + draft" pairings delivering 1.5–3× wall-clock speedup with measured accuracy retention.

**Why this sub-phase exists:** Speculative decoding is the single highest-leverage inference-time optimization available to us — it's the difference between 73 t/s and 130+ t/s on Cortex Ultra, or between Sage's painful 3 t/s CPU experience and a usable ~6 t/s. Most of our top models have natural draft candidates already inside the Aneuro lineup (Axon↔Sage, Nova↔Pulse) — we just haven't wired them up.

### 13.4.1 — Method survey

| Method | Requires | Best for | Verdict for us |
|---|---|---|---|
| **Classical draft-model speculative** (vanilla) | Same tokenizer family; draft ≤ ~1/8 target params | Any autoregressive LLM pair | **Primary track.** llama.cpp `--draft-model` ships it. |
| **EAGLE-2** (feature-level AR, tree beam) | Training of a small "EAGLE head" on target's features | Max quality speedup | Stretch — requires Spark training + llama.cpp support is partial |
| **Medusa** (parallel heads on target) | Fine-tune target with extra heads | Targets where we control training | Defer — rewrites every target |
| **LayerSkip / self-spec** | Supported runtime | Targets without a natural small partner | Backup for Synapse (LFM2.5 1.2B has no draft) |
| **Prompt-lookup decoding** (PLD, n-gram from context) | Nothing | Long-context doc chat | Cheap add-on, try everywhere |

**Primary strategy:** Ship **classical draft-model** for every chat LLM where we already have a same-family small sibling in the lineup. Use **LayerSkip / PLD** as fallback. EAGLE-2 and Medusa are Phase 13.4 stretches, trained on Spark only if the primary pairings underdeliver.

### 13.4.2 — Per-model pairing table

Rows with ✅ are ready-to-ship (both models already in our lineup); ⚙️ are small training jobs on Spark; 🔶 are research stretches.

| Target (our name) | Underlying | Draft candidate (our name) | Underlying draft | Tokenizer match | Approach | Expected speedup | Status |
|---|---|---|---|---|---|---:|---|
| **Aneuro Sage** | Qwen 3 8B (CPU, ~3 t/s) | **Aneuro Axon** | Qwen 3 1.7B | ✅ exact | llama.cpp `--draft-model` | **2.0–2.5×** (→ ~6–7 t/s) | ✅ ship in 13.4.1 |
| **Aneuro Pulse** | Gemma 3 1B | **Aneuro Nova** | Gemma 3 270M | ✅ exact | llama.cpp `--draft-model` | 1.5–1.8× | ✅ ship in 13.4.1 |
| **Aneuro Vision** | Gemma 4 E2B (CPU) | **Aneuro Nova** | Gemma 3 270M | ⚠️ close (G4/G3 diverge on multimodal tokens) | text-only draft; skip on image turns | 1.3–1.5× on text | ✅ conditional |
| **Aneuro Facet** | Gemma 4 E2B (CPU) | same as Vision | — | — | same | 1.3–1.5× on text | ✅ conditional |
| **Aneuro Axon** | Qwen 3 1.7B | **Qwen 3 0.6B** (not in lineup) | — | ✅ exact | need to pull Qwen 3 0.6B GGUF; add hidden draft-only entry | 1.6–2× | ⚙️ fetch + wire |
| **Aneuro Cortex** | Llama 3.2 1B | **Llama 3.2 1B Draft** | distill from self | n/a (custom) | Train 270M draft on Spark via Orion-Lite recipe (Phase 11 KB) | 1.5–2× | ⚙️ 2–3d Spark |
| **Aneuro Cortex Ultra** | Llama 3.2 1B (MLC compiled) | Same draft as Cortex | — | same | MLC-LLM speculative-decoding support status: **to verify** before wiring; fallback to llama.cpp twin (Cortex, non-MLC) | 1.5–2× | 🔶 runtime gate |
| **Aneuro Synapse** | LFM2.5 1.2B | (no sibling) | — | n/a | **LayerSkip** or train a 270M LFM-family distillate on Spark | 1.3–1.8× (LayerSkip) / 1.6–2× (distill) | ⚙️ / 🔶 |
| **Aneuro Iris** | moondream2 2B (VLM) | (no sibling) | — | n/a | VLM speculative is immature; try text-only draft for text-heavy responses | 1.2× at most | 🔶 research |
| **Aneuro Lumen** | LFM2-VL 1.6B (VLM) | (no sibling) | — | n/a | Same as Iris; LFM tokenizer shared with Synapse, so the Synapse draft could work for the text portion | 1.2–1.4× on text | 🔶 research |
| **Aneuro Oracle** | Granite Vision 3.2 2B (VLM) | (no sibling) | — | n/a | Same as Iris | 1.2× at most | 🔶 research |
| **Aneuro Forge** | FunctionGemma 270M | (too small) | — | — | Skip — draft would need to be ≤33M, not worth it | — | skip |
| **Aneuro Nova** | Gemma 3 270M | (too small) | — | — | Skip — Nova *is* a draft candidate for others | — | skip |
| **Aneuro Spark** | Qwen 2.5 0.5B | (too small) | — | — | Skip | — | skip |

### 13.4.3 — Work breakdown

**Rung 0 — Runtime support audit (0.5 day)**
- Confirm llama.cpp `--draft-model` flag works on our Jetson build with the exact Docker images we use (`ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin`). Benchmark Sage + Axon draft on a 30-prompt eval set.
- Confirm MLC-LLM 4.x speculative-decoding status for the Cortex Ultra pairing (as of April 2026 this is partial — may need a fallback to the non-MLC Cortex twin).

**Rung 1 — Ship the four ready-to-go pairings (1.5 days)**
1. **Sage + Axon** — add a `draft_model` field to `models.json`; manager.go starts both containers or uses an existing `--draft-model` flag in start_cmd. Benchmark: baseline 3 t/s vs spec-decoded target.
2. **Pulse + Nova** — same wiring pattern.
3. **Vision + Nova (text-only turns)** — requires request-time inspection of whether the turn includes an image; skip speculative on multimodal turns.
4. **Facet + Nova (text-only turns)** — same.

Each pairing: measure speedup and accuracy (GSM8K + our conversation eval). Acceptance: ≥ 1.4× speedup, accuracy drop < 1 pp.

**Rung 2 — Fetch + wire Qwen 3 0.6B for Axon (0.5 day)**
- Download `ggml-org/Qwen3-0.6B-GGUF` (or equivalent), add a hidden `axon-draft` entry, wire to Axon's start_cmd.

**Rung 3 — Train custom drafts on DGX Spark (3–5 days per draft)**

Target 1: **Llama 3.2 1B → 270M draft** (serves Cortex + Cortex Ultra)
- Base: SmolLM2 360M-ish randomly-init → distill with Orion-Lite feature-mimic L1 from our Llama 3.2 1B GGUF (ungquantized reference).
- Dataset: 50k sampled turns from our chat logs (anonymized) + ARC-C/GSM8K + OpenAssistant.
- Tokenizer: reuse Llama 3.2's tokenizer so GGUF pairing works out of box.
- Success criterion: ≥ 30% wall-clock speedup at < 2 pp accuracy drop on Cortex.

Target 2 (stretch): **LFM2.5 1.2B → 270M draft** for Synapse.
- Only if LayerSkip rung doesn't deliver.

**Rung 4 — LayerSkip fallback (1 day)**
- llama.cpp + models with `layerskip_enabled` expose a self-speculative mode. Test on Synapse + Lumen.
- Cheap, ships without training, may give 1.2–1.4× for free.

**Rung 5 — EAGLE-2 stretch (2–3 days on Spark)**
- If any primary pairing underdelivers (< 1.3× measured), train an EAGLE head on that target.
- Only attempted after Rungs 0–4 results are in hand.

### 13.4.4 — Integration with `backend` package

Phase 10-2's `Driver` interface stays clean — speculative is a per-model config concern, not a driver-type concern. Add:

```jsonc
// models.json fragment
"sage": {
  ...
  "speculative": {
    "draft_model_id": "axon",     // references another entry in models.json
    "draft_flag_template": "--draft-model /blobs/{{.DraftBlob}} --draft-max 16"
  }
}
```

At load time, the driver resolves `draft_model_id` to its blob path and renders the flag into the `start_cmd`. If the draft isn't available (model not downloaded, speculative disabled by user), the driver falls back to plain autoregressive.

### 13.4.5 — Deliverables

1. **Pairings table, measured** — [`outputs/phase13_4/speculative_pairings.md`](../outputs/phase13_4/speculative_pairings.md) with baseline + speculative speed, accuracy delta per pairing, decision to ship or not.
2. **Trained drafts (if Rung 3 runs)** — checkpoints on DGX Spark with tags, GGUF exports on Jetson HF cache.
3. **`models.json` entries** — speculative field on every shipped target; hidden `*-draft` entries for drafts not in the user-facing lineup.
4. **Closeout memo** — [`outputs/phase13_4/closeout.md`](../outputs/phase13_4/closeout.md) feeding into `edge_optimization_kb.md`.

### 13.4.6 — Risks

| Risk | Mitigation |
|---|---|
| Tokenizer mismatch corrupts output | Restrict primary track to **exact tokenizer** pairs (Qwen 3 family, Gemma 3 family); fall back to LayerSkip for mismatches |
| MLC-LLM lacks speculative support | Ship Cortex Ultra speculative via a swap to llama.cpp runtime (measured earlier at 60 t/s for the non-MLC variant — spec decoding may pay the MLC gap back) |
| Draft accuracy too poor → verify rejection rate > 70% | Tune `--draft-min-p` / acceptance threshold; retrain with more aligned data if needed |
| Custom-trained drafts leak training-data patterns | Standard alignment/safety pass before shipping |
| VLM speculative (Iris/Lumen/Oracle) remains research-grade | Mark as 🔶 in the pairings table; revisit in a future phase if EAGLE-VLM papers mature |

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

## Approval Gate — CLEARED (2026-04-18)

| Q | Answer |
|---|---|
| Q13.1 | Keep VLA ladder (Ladder C) — yes |
| Q13.2 | Sequential execution A → B → C, detail report after each |
| Q13.3 | Heavy-duty / distillation-aware training kept in scope; sub-phases 13.1–13.3 defined |
| Q13.4 | Nerve Prime track added, conditional on Phase 12 output |
| Prime naming | Confirmed (Cortex / Lumen / Tendon / Sinew / Nerve-F / Nerve-D Prime + Nova Prime stretch) |

No blocking questions. Ready to execute after Phases 10 + 11 + 12.
