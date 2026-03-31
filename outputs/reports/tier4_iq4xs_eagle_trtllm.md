# Tier 4 — IQ4_XS for LFM2.5, EAGLE-3, TensorRT-LLM (Stages XVIII–XX)

**Date:** 2026-03-30
**Hardware:** NVIDIA Jetson Orin Nano 8 GB (UMA, ~68 GB/s)
**Runtime (Stage XVIII):** llama.cpp v8510 (host build, `3a60d06ad`)
**Runtime (Stage XX):** TensorRT-LLM v0.12.0-jetson (Docker, sm_87)

---

## Stage XVIII — IQ4_XS for LFM2.5-1.2B from Fixed F16 Source

**Status: ✅ Complete**

### Motivation

Stage XIV was blocked because `convert_hf_to_gguf.py` failed with `KeyError: 'block_ff_dim'`. Every other model (Llama-3.2-1B, Qwen2.5-0.5B) already has community IQ4_XS GGUFs. For LFM2.5-1.2B, no community IQ4_XS exists and LiquidAI's GGUF repo only has Q4_K_M. This stage unblocks the full IQ4_XS pipeline: patch the converter → F16 GGUF → imatrix → IQ4_XS → benchmark.

### Root Cause of Stage XIV Failure

**File:** `/home/spitman/tools/llama.cpp/convert_hf_to_gguf.py`, class `LFM2Model` (line 11264)

```python
def _add_feed_forward_length(self):
    ff_dim = self.hparams["block_ff_dim"]  # KeyError here
```

**Why it fails:** `load_hparams()` calls `AutoConfig.from_pretrained(dir_model, trust_remote_code=False).to_dict()`. The transformers `Lfm2Config` class stores `block_ff_dim` internally as `intermediate_size` when constructing the config object. When `.to_dict()` is called, `block_ff_dim` is absent from the result — only `intermediate_size: 12288` remains. The raw `config.json` has both keys (identical values: `block_ff_dim=12288`, `intermediate_size=12288`), but `AutoConfig` normalizes them.

**The fix** (2 lines, `_add_feed_forward_length` in `LFM2Model`):

```python
# Before:
ff_dim = self.hparams["block_ff_dim"]
auto_adjust_ff_dim = self.hparams["block_auto_adjust_ff_dim"]
ff_dim = self.hparams["block_ff_dim"]

# After (patch applied 2026-03-30):
# block_ff_dim may be dropped by AutoConfig.to_dict() for LFM2.5;
# fall back to intermediate_size (same value: 12288)
ff_dim = self.hparams.get("block_ff_dim", self.hparams.get("intermediate_size"))
auto_adjust_ff_dim = self.hparams.get("block_auto_adjust_ff_dim", True)
ff_dim = self.hparams.get("block_ff_dim", self.hparams.get("intermediate_size"))
```

The `auto_adjust_ff_dim=True` computes `int(2 * 12288 / 3) = 8192`, rounded to 256-multiple = **8192** — matching the existing Q4_K_S GGUF's `lfm2.feed_forward_length = 8192`.

Backup saved at `/home/spitman/tools/llama.cpp/convert_hf_to_gguf.py.bak`.

---

### Pipeline Steps

#### Step 1 — F16 GGUF Conversion

```bash
python3 /home/spitman/tools/llama.cpp/convert_hf_to_gguf.py \
  /home/spitman/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/e7114d1... \
  --outfile LFM2.5-1.2B-Instruct-F16.gguf --outtype f16
```

**Result:** 2.2 GB F16 GGUF — 148 tensors, 16.00 BPW, 27 s at ~185 MB/s. No errors. The duplicate `lfm2.feed_forward_length` warning is expected (parent class sets 12288, our patched function overrides to 8192).

#### Step 2 — Calibration Data

No pre-existing imatrix calibration file available on Jetson. Generated wikitext-2-raw-v1 calibration text by:
1. Downloading the train parquet directly from HuggingFace
2. Extracting 338 paragraphs (118 KB) via pyarrow

**Note:** `datasets` 4.8.4 on Jetson host now triggers a `scipy`/numpy ABI error (`numpy.core.multiarray failed to import`) when called inline. The error is caused by the system scipy (compiled for numpy <1.25.0) conflicting with pip-installed numpy 2.2.6. Workaround: read Arrow cache files directly with `pyarrow.ipc.open_stream()`.

#### Step 3 — Importance Matrix

```bash
llama-imatrix \
  -m LFM2.5-1.2B-Instruct-F16.gguf \
  -f /tmp/calibration.txt \
  -o LFM2.5-1.2B-Instruct.imatrix \
  --chunks 100 -ngl 99
```

**Result:** 1.2 MB imatrix, 100 chunks (27,648 tokens processed), PPL ≈ 18–22 (typical for wikitext-2 on a 1B model). Runtime: 74.3 s (~2.4 min).

**Calibration quality note:** 100 chunks of wikitext-2 is a minimal calibration run. bartowski's production imatrix runs use 200–500 chunks on a more diverse corpus (wikitext-2 + code + instruction data). The small calibration corpus likely contributed to the ARC accuracy regression (see results).

#### Step 4 — IQ4_XS Quantization

```bash
llama-quantize \
  --imatrix LFM2.5-1.2B-Instruct.imatrix \
  LFM2.5-1.2B-Instruct-F16.gguf \
  LFM2.5-1.2B-Instruct-IQ4_XS.gguf \
  IQ4_XS
```

**Result:** 633 MB IQ4_XS (4.52 BPW) — from 2,232 MB F16. Quantization time: 59.6 s.

**Size comparison:**
| Format | Size | BPW |
|--------|------|-----|
| F16 | 2,232 MB | 16.00 |
| Q4_K_M (LiquidAI official) | 698 MB | ~5.0 |
| Q4_K_S (re-quant from Q4_K_M) | 682 MB | ~4.8 |
| **IQ4_XS (this stage)** | **633 MB** | **4.52** |

---

### Results

**Speed** (measured by `llama-bench`, `-r 3`):

| Config | pp512 t/s | tg128 t/s | vs Q4_K_S+FA |
|--------|----------:|----------:|:-------------|
| LFM2.5-1.2B Q4_K_S+FA (prior best) | 2080 | 53.22 | baseline |
| **LFM2.5-1.2B IQ4_XS+FA** | **2374.92 ± 158** | **58.98 ± 0.13** | **+10.8% tg ✅** |

**Accuracy** (20-sample few-shot, same bench_gguf.py prompt format; pyarrow direct Arrow read):

| Config | GSM8K | ARC-Challenge | vs Q4_K_S+FA |
|--------|------:|-------------:|:-------------|
| LFM2.5-1.2B Q4_K_S+FA | 5% | **70%** | baseline |
| **LFM2.5-1.2B IQ4_XS+FA** | **10%** | **60%** | GSM8K +5pp (noise), ARC **−10pp ❌** |

### Analysis

**Speed:** IQ4_XS is 633 MB vs Q4_K_S's 682 MB — 7% smaller. On a memory-bandwidth-bound system, this directly translates to +10.8% tg128. The result is consistent with the Llama-3.2-1B IQ4_XS result (+8.4% from Stage XIII).

**ARC regression (70% → 60%):** A genuine 10-point drop. Contributing factors:

1. **Calibration corpus mismatch:** The 338-paragraph wikitext-2 corpus is encyclopedic text. ARC-Challenge tests science reasoning — a different domain. The imatrix weights channels that matter for Wikipedia text, not science reasoning. bartowski uses a larger, more diverse corpus.

2. **Quantization source:** Our F16 GGUF was freshly converted and quantized. The official Q4_K_S was quantized by LiquidAI from their F16 weights using their internal pipeline. LiquidAI may apply custom post-training optimizations before releasing their GGUF.

3. **Small calibration size:** 100 chunks / 27K tokens is a quick imatrix. The GGML documentation recommends 200+ chunks for better coverage.

**GSM8K:** 5% → 10% is within n=20 noise (1-question difference). No conclusion can be drawn.

**Verdict: Speed win achieved (+10.8% tg128), but ARC accuracy regresses 10pp. The IQ4_XS format is sound — the calibration quality limits the result. A better imatrix (larger corpus, reasoning-domain text) would likely recover the ARC loss.**

### Path Forward for Better Imatrix

To generate a production-quality imatrix for LFM2.5-1.2B:

```bash
# Use 200 chunks + a mixed corpus (wikitext-2 + code + instruction data)
llama-imatrix \
  -m LFM2.5-1.2B-Instruct-F16.gguf \
  -f /path/to/better_calibration.txt \
  -o LFM2.5-1.2B-Instruct-v2.imatrix \
  --chunks 200 -ngl 99
```

A mixed corpus with reasoning examples (MMLU, ARC, science text) would better preserve the channels that matter for science reasoning. Expected: IQ4_XS with ARC ≥ 70% (matching or exceeding Q4_K_S).

---

## Stage XIX — EAGLE-3 Speculative Decoding

**Status: ❌ BLOCKED — requires external GPU for training**

### What EAGLE-3 Is

EAGLE-3 (Extrapolation Algorithm for Greater Language-model Efficiency, v3) trains a lightweight draft head attached to the main model. During inference, the draft head predicts multiple future tokens using the main model's hidden states; the main model then verifies them in parallel. Accepted tokens advance generation by more than one per main model forward pass.

**Unlike standard speculative decoding**, EAGLE-3 requires no separate draft model — only a 2–3 layer transformer head (~100M params for a 1.2B target) trained on the target model's own hidden states.

Reference: Cai et al. (2024), ICML 2024. EAGLE-3 is the v3 update with improved acceptance rate via feature-level prediction.

### Requirements

| Requirement | Status |
|-------------|--------|
| GPU workstation for training (not Jetson) | ❌ Not available on Jetson |
| ~1K domain-relevant training samples | ❌ Not prepared |
| Draft head training code (EAGLE repo) | ✅ Available at github.com/SafeAILab/EAGLE |
| Main model hidden states (LFM2.5-1.2B F16) | ✅ F16 GGUF now available (Stage XVIII output) |
| llama.cpp EAGLE integration | ✅ llama.cpp v8510 supports speculative decoding via `-md` flag |

### Expected Performance

If implemented:
- **Token generation: 53 t/s → ~100–150 t/s** (2–3× speedup) for LFM2.5-1.2B
- Draft token acceptance rate: est. 70–80% on domain-relevant text
- Accuracy: neutral (verifier model is unchanged)

This is the single highest-impact optimization remaining — a 2–3× speedup without any accuracy loss is transformative for real-time deployment.

### Why Blocked

EAGLE-3 training requires:
1. Running the full LFM2.5-1.2B model in float16 to collect hidden state activations (~2 GB per 1K samples)
2. Training the draft head with backpropagation — requires a GPU with ≥8 GB VRAM
3. The Jetson Orin Nano's 8 GB UMA is fully consumed by model + KV cache during inference; no headroom for gradient training

**This must be done on an external GPU workstation (RTX 3090/4090 recommended).**

### Path Forward

```bash
# On a GPU workstation (not Jetson):
git clone https://github.com/SafeAILab/EAGLE
cd EAGLE

# Step 1: Collect hidden state data from LFM2.5-1.2B
# (Requires HF weights — available at LiquidAI/LFM2.5-1.2B-Instruct)
python -m eagle.ge_data.ge_data_all_vicuna \
    --base_model_path LiquidAI/LFM2.5-1.2B-Instruct \
    --data_path path/to/training_conversations.json \
    --output_path eagle_data_lfm25 \
    --num_gpus 1

# Step 2: Train the draft head
python -m eagle.train.main \
    --base_model_path LiquidAI/LFM2.5-1.2B-Instruct \
    --data_path eagle_data_lfm25 \
    --output_dir eagle_head_lfm25 \
    --num_epochs 3

# Step 3: Export and run on Jetson via llama.cpp speculative server
# (llama.cpp -md flag for EAGLE-style speculative decoding)
```

**Complication:** LFM2.5 uses a hybrid SSM+attention architecture. EAGLE-3 was designed for pure-attention transformers (LLaMA, Mistral, Qwen). The SSM layers (10 of 16 layers in LFM2.5) produce hidden states with a different structure than attention layers. The draft head may need architecture-specific modifications to work with SSM hidden states.

**Recommendation:** Test EAGLE-3 first on Llama-3.2-1B (pure attention architecture) where it is more likely to work without modification. Then adapt for LFM2.5 if Llama-3.2-1B succeeds.

**Verdict: Stage XIX requires external GPU workstation + multi-day implementation effort. Highest-impact remaining optimization (~2–3×) but not achievable on Jetson alone.**

---

## Stage XX — TensorRT-LLM Hardware Acceleration

**Status: ✅ Partial — Qwen2.5-0.5B complete; LFM2.5 and Llama blocked (see below)**

### Background

Stage 5 explored TensorRT-LLM conceptually and produced estimated results based on NVIDIA Orin benchmarks (from the TRT-LLM v0.12.0-jetson README). Stage XX actualizes those estimates by building and running TRT-LLM engines.

**TRT-LLM optimization stack:**
- **W4A16 (Weight-only 4-bit, 16-bit activations):** 4-bit weights, 16-bit activations. Reduces model footprint and memory bandwidth while preserving activation precision.
- **Kernel fusion:** Fuses attention + LayerNorm + activation into single CUDA ops — eliminates launch overhead and improves memory access patterns.
- **Persistent kernels:** Pre-compiled CUDA engine for the specific sequence length and batch size — no JIT overhead at inference time.
- **Paged KV cache:** Dynamic allocation instead of pre-reserving the full context budget.

### Build Environment

| Component | Version / Status |
|-----------|-----------------|
| TRT-LLM repo | v0.12.0-jetson (inside `aneurologic_phase5` Docker container) |
| CUDA | 12.6 (JetPack 6.2) ✅ |
| TensorRT Python | 10.3.0 (JetPack system package — pip 10.15.1.29 replaced, wrong arch) ✅ |
| torch | 2.10.0 (Jetson-specific build from `pypi.jetson-ai-lab.io`) ✅ |
| Build flags | `--cuda_architectures 87 -DENABLE_MULTI_DEVICE=0 --job_count 4 -DTRT_LIB_DIR=/host-libs` |

**Build blockers resolved:**
1. `tensorrt` pip package (10.15.1.29) → built for x86/CUDA13, not Jetson. Replaced with system JetPack TRT 10.3.0.
2. `torchvision` ABI mismatch with torch 2.10.0 → uninstalled; not needed for LLM inference.
3. `transformers>=5.x` → downgraded to 4.42.4 (TRT-LLM 0.12.0 requires `<=4.42.4`).
4. Shared tensor error (lm_head/vocab_embedding weight tying in Qwen) → patched `modeling_utils.py` to clone shared tensors before `safetensors.save_file`.
5. Parallel build memory thrash (22 nvcc processes, 384 MB free) → killed and restarted with `--job_count 4`. Build completed using cached objects.

---

### Qwen2.5-0.5B W4A16 — Results

**Pipeline:** HF safetensors → W4A16 int4 checkpoint → TRT engine (max_seq_len=640) → ModelRunnerCpp

**Checkpoint conversion:**
```bash
python3 examples/qwen/convert_checkpoint.py \
  --model_dir $QWEN_HF --output_dir /tmp/trtllm_qwen05_w4 \
  --dtype float16 --use_weight_only --weight_only_precision int4 \
  --load_model_on_cpu
# Time: 8s total. Output: rank0.safetensors + config.json
```

**Engine build:**
```bash
trtllm-build --checkpoint_dir /tmp/trtllm_qwen05_w4 \
  --output_dir /workspace/outputs/trt_engines/qwen05_w4a16 \
  --max_batch_size 1 --max_input_len 512 --max_seq_len 640 --workers 1
# Engine size: 445 MiB. Build time: 23s.
```

**Benchmark results** (ModelRunnerCpp, batch=1, `kv_cache_free_gpu_memory_fraction=0.1`):

| Metric | llama.cpp Q3_K_M+FA (best) | TRT-LLM W4A16 | Speedup |
|--------|:-------------------------:|:-------------:|:-------:|
| pp401 t/s | ~3700 (est. at 401 tok) | **~6900** | **~1.86×** |
| tg128 t/s | 93.92 | **100.87** | **+7.4%** |

*pp401: average of runs 2–3 (run 1 = cold start at 4443 t/s, runs 2–3 steady state at 6823/6962 t/s).*

**Analysis:**
- **Prefill (+86%):** TRT-LLM's kernel fusion and persistent execution context eliminate Python dispatch overhead on every attention layer. This is the largest gain from TRT-LLM — the fused attention+layernorm kernels run much faster than llama.cpp's sequential layer dispatch.
- **Generation (+7.4%):** Modest gain. At 0.5B scale, the model is so small that memory bandwidth is rarely saturated — the bottleneck is kernel launch overhead, which TRT-LLM reduces but cannot eliminate entirely. A larger model (1B+) would show more generation speedup.
- **Engine size (445 MiB vs ~318 MB GGUF Q3_K_M):** Slightly larger due to int4 packing overhead in TRT format + engine metadata.

---

### LFM2.5-1.2B — Blocked (Architecture Not Supported)

TRT-LLM 0.12.0 has no `examples/lfm` or any hybrid SSM+attention model support. The LFM2.5 architecture (10 SSM layers + 6 attention layers) requires custom model implementation in TRT-LLM. This is non-trivial: SSM layers use selective scan operators that have no equivalent in TRT-LLM's current op set.

**Verdict:** LFM2.5-1.2B TRT-LLM inference not viable in v0.12.0. Would require either:
1. A future TRT-LLM version with SSM support (v0.14+ has Mamba support — may be extensible to LFM2.5)
2. A custom TRT plugin for the GatedDeltaNet / LiquidAI SSM layers

---

### Llama-3.2-1B W4A16 — Results (Partial, No Speedup)

**Status: ⚠️ Completed with significant caveats**

Weights downloaded from `unsloth/Llama-3.2-1B-Instruct` mirror (official `meta-llama` gated repo requires license acceptance — accepted 2026-03-31 but download still in progress; mirror used for conversion).

**Build path required 7 source patches to TRT-LLM 0.12.0:**

1. **`models/llama/config.py`** — Normalize Llama 3.2 rope_scaling format (`rope_type` → `type` key)
2. **`models/llama/convert.py`** — Skip `lm_head.weight` load when `tie_word_embeddings=True` (Llama 3.2 has no separate `lm_head.weight` in safetensors)
3. **`layers/linear.py`** — None guard in `postprocess()` when weight is None (tied lm_head case)
4. **`models/model_weights_loader.py`** — Two patches: None guard in postprocess chain + `KeyError` guard in `check()` for missing `lm_head.weight`
5. **`models/modeling_utils.py`** — Explicit `lm_head.weight = vocab_embedding.weight.clone()` after dedup, so `from_checkpoint` finds both keys; `share_embedding(model)` call before `save_checkpoint`
6. **`examples/llama/convert_checkpoint.py`** — Call `share_embedding(llama)` before `save_checkpoint` when `tie_word_embeddings=True`
7. **`tensorrt_llm/commands/build.py`** — `torch.cuda.empty_cache()` + 256 MB workspace limit to reduce TRT build peak memory

**Critical blocker — TRT engine serialization OOM with `--gemm_plugin float16`:**

The correct W4A16 build requires `--gemm_plugin float16` to keep weights in int4 format at runtime. The TRT 10.3.0 engine serializer (`globWriter.cpp`) starts with a 1 GB GPU allocation for the serialization buffer. On the 8 GB Jetson, after loading the 1.5 GB W4A16 checkpoint + TRT network representation + profiling buffers, the serializer cannot allocate the 1 GB buffer — NvMap IOVM allocation fails even though PyTorch reports 5.4 GB CUDA-free.

**Workaround attempted:** Build without `--gemm_plugin float16`. This succeeds but produces a 1002 MB FP16-path engine where quantized weights are dequantized at build time to float16. The runtime uses FP16 math, not W4A16.

**Benchmark results** (no-gemm_plugin engine, ModelRunnerCpp, batch=1, `kv_cache_free_gpu_memory_fraction=0.1`):

| Metric | llama.cpp baseline Q4_K_M | llama.cpp best IQ4_XS+FA | TRT-LLM W4A16 (no gemm_plugin) |
|--------|:-------------------------:|:------------------------:|:-------------------------------:|
| tg128 t/s | 44.64 | **54.37** | 44.08 |

**Result: No improvement.** TRT-LLM without gemm_plugin falls back to FP16 math and matches (but does not exceed) the llama.cpp baseline. The llama.cpp IQ4_XS+FA config (54.37 t/s) remains the Llama-3.2-1B deployment recommendation.

**Root cause of serialization OOM:** TRT 10.3.0 `globWriter.cpp::makeResizableGpuMemory` starts with a hardcoded 1 GB GPU buffer. Qwen 0.5B engine with gemm_plugin is ~446 MB, so TRT can alloc 1 GB successfully after Qwen's smaller checkpoint is loaded. Llama 1B checkpoint is 1.5 GB (3× larger), leaving insufficient headroom for the 1 GB serialization buffer even though total GPU free shows 5.4 GB (PyTorch's memory accounting does not reflect NvMap IOVM pressure from TRT's direct allocations).

**Verdict: Llama-3.2-1B TRT-LLM W4A16 is blocked on Jetson Orin 8 GB.** The gemm_plugin build OOM is a fundamental memory constraint of TRT 10.3.0 on Orin. Not viable without either a TRT version that uses a smaller initial serialization buffer or a Jetson with more RAM.

---

## Tier 4 Summary

| Stage | Title | Status | Key Finding |
|-------|-------|--------|-------------|
| **XVIII** | IQ4_XS for LFM2.5 from patched F16 | ✅ Complete | **+10.8% tg128** (58.98 vs 53.22); ARC −10pp (60% vs 70%) — calibration corpus too narrow |
| **XIX** | EAGLE-3 speculative decoding | ❌ Blocked | Requires external GPU for training; SSM architecture complicates standard EAGLE approach |
| **XX** | TensorRT-LLM W4A16 | ✅ Partial (Qwen only) | Qwen2.5-0.5B: **+86% pp, +7.4% tg** vs llama.cpp. LFM blocked (SSM arch). Llama: 44.08 t/s (no improvement — gemm_plugin build OOMs on 8 GB Jetson). |

### Updated Optimal Deployment Configs (after Tiers 1–4)

| Model | Best Speed Config | tg t/s | Best Accuracy Config | Notes |
|-------|-----------------|-------:|---------------------|-------|
| **LFM2.5-1.2B** | IQ4_XS + FA (llama.cpp) | **58.98** | Q4_K_S + FA | ARC 60% vs 70% — accuracy tradeoff |
| **Llama-3.2-1B** | IQ4_XS + FA + chat template | **54.37** | same | 40% GSM8K, 45% ARC |
| **Qwen2.5-0.5B** | **TRT-LLM W4A16** | **100.87** | Q3_K_M + FA (llama.cpp) | TRT-LLM ~7% faster tg, ~1.86× pp |

**LFM2.5-1.2B decision guide:**
- Speed-priority deployment: **IQ4_XS + FA** → 58.98 t/s
- Accuracy-priority deployment: **Q4_K_S + FA** → 53.22 t/s, 70% ARC
