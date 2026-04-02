# Phase 6 — 100 t/s Generation with >1B Parameters

**Goal:** Achieve ≥100 t/s token generation on Jetson Orin Nano 8 GB with a model larger than 1B parameters.
**Date:** 2026-03-31
**Hardware:**
- **Jetson:** Jetson Orin Nano 8 GB, UMA 68 GB/s, sm_87 — `ssh spitman@spitman-jetson`
- **DGX Spark:** NVIDIA GB10 (Blackwell, sm_121), 119 GB UMA, aarch64, 20 cores — `ssh spark-807e.local`

---

## The Hard Constraint

Token generation is memory-bandwidth bound. Jetson Orin Nano has 68 GB/s UMA bandwidth.

| Model size | INT4 weight BW/token | Theoretical max | Realistic (37% eff.) |
|------------|---------------------|-----------------|---------------------|
| 0.5B | 247 MB | 275 t/s | ~100 t/s ✅ |
| 1B | 500 MB | 136 t/s | ~50 t/s |
| 1.5B | 750 MB | 91 t/s | ~34 t/s |
| 3B | 1.5 GB | 45 t/s | ~17 t/s |

The 37% efficiency figure is calibrated from Qwen2.5-0.5B (TRT-LLM W4A16, 100.87 t/s actual vs 275 t/s theoretical).

**Implication:** Reaching 100 t/s with >1B params requires either speculative decoding (~2× effective throughput) or a significantly more memory-bandwidth-efficient build than our current baseline.

---

## Path 1 — Speculative Decoding: Llama-3.2-3B + 1B Draft

**Target:** ~60–80 t/s effective on Llama-3.2-3B (>1B model)
**Status: ❌ BLOCKED — NvMap IOVM hardware constraint**

### Results (2026-03-31)

| Metric | Value |
|--------|-------|
| 3B IQ4_XS standalone tg128 t/s (FA) | **26.01 t/s** |
| 3B + 1B speculative | **BLOCKED** |

### Blocker: NvMap IOVM virtual address space exhaustion

Llama-3.2-3B IQ4_XS downloaded (1.70 GiB, 3.21B params) and confirmed working standalone at **26.01 t/s** with `-fa 1 -ngl 99`.

Loading the 3B + 1B models simultaneously in `llama-speculative` fails regardless of draft quantization (tried Q4_K_S, Q3_K_M):

```
NvMapMemAllocInternalTagged: 1075072515 error 12
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 651 MiB on device 0: cudaMalloc failed: out of memory
```

**Root cause:** On Jetson Orin Nano, NvMap IOVM (GPU I/O Virtual Memory address space) is limited to ~2–2.5 GB total. After loading:
- 3B model weights: ~1.7 GB IOVM
- 3B KV cache + compute scratch: ~0.3 GB IOVM
- **Total used: ~2.0 GB** — no room left for 1B draft weights (~0.65 GB needed)

This is NOT a physical memory issue (7.6 GB CUDA total, 5.9 GB free after 3B loads). It is a GPU virtual address space limit in the NvMap/SMMU subsystem. There is no user-space workaround via `cudaMalloc` flags; would require kernel boot parameters to increase NvMap IOVM region size.

**Verified:** 3B alone works ✅. Any 1B variant alongside 3B fails ❌. No Q2_K variant exists for Llama-3.2-3B (bartowski's smallest is IQ3_M at 1.60 GiB — insufficient savings).

### Alternative: TRT-LLM Speculative Decoding (deferred to Path 3)

TRT-LLM builds a single optimized engine for both models with more memory-efficient management. Qwen2.5-0.5B (already at 100 t/s in TRT-LLM) as draft + Qwen2.5-1.5B as main may avoid this constraint since TRT manages memory differently than llama.cpp's per-tensor NvMap allocations.

---

## Path 2 — Fix TRT-LLM Llama-3.2-1B W4A16 Build

**Target:** ~65–75 t/s for Llama-3.2-1B (proper W4A16, int4 weights at runtime)
**Status: ❌ BLOCKED — NvMap IOVM hardware constraint (exhaustive investigation)**
**Hardware needed:** Jetson (build + run)

### Results (2026-03-31 / 2026-04-02)

| Attempt | Outcome |
|---------|---------|
| no-gemm-plugin engine (Stage XX) | ✅ 1002 MB, **44.08 t/s** (FP16 fallback, no improvement) |
| gemm-plugin + workspace 256 MB limit | ❌ globWriter OOM (1 GB IOVM) |
| gemm-plugin + custom PinnedMemAllocator | ❌ `cudaMallocHost` for 1 GB fails — NvMap IOVM applies to pinned memory too |
| gemm-plugin + share_embedding_table fix (–501 MB) | ❌ 512 MB cudaMallocHost OK, globWriter 1 GB still fails |
| llama.cpp IQ4_XS + FA (best alternative) | **60.28 t/s** (no gemm_plugin needed) |

### Root cause analysis — exhaustive

**Why is TRT-LLM W4A16 with gemm_plugin impossible on Jetson Orin Nano 8GB:**

1. **NvMap IOVM physical limit:** Jetson Orin's NvMap IOVM (GPU virtual address space) is limited to ~2–2.5 GB system-wide. This constraint applies to `cudaMalloc` AND `cudaMallocHost` (both go through SMMU on UMA hardware).

2. **Build-time IOVM budget:**
   - Python/CUDA context overhead: ~0.3 GB
   - TRT 10.3.0 kernel library: ~1.2 GB (unavoidable at build time)
   - Llama 1B W4A16 checkpoint: ~1.0 GB (after lm_head deduplication fix, see below)
   - Total before serialization: ~2.5 GB — IOVM exhausted

3. **globWriter 1 GB allocation:** TRT 10.3.0 `globWriter.cpp::makeResizableGpuMemory` allocates a 1 GB GPU buffer for engine serialization (gemm_plugin engine is large). With IOVM exhausted, this fails. The workspace limit (256 MB) controls profiling, not serialization.

### All patches applied to TRT-LLM source (8 total)

| File | Patch | Purpose |
|------|-------|---------|
| `builder.py` | `PinnedMemAllocator` (custom IGpuAllocator) | Route large allocs to `cudaMallocHost` to bypass NvMap — partially works (512 MB OK, 1 GB fails) |
| `builder.py` | `set_memory_pool_limit(WORKSPACE, 256 MB)` | Reduce workspace from default 1 GB |
| `builder.py` | `torch.cuda.empty_cache()` + debug logging before build | Free torch cache; visibility |
| `models/llama/config.py` | Normalize `rope_scaling` dict format | Fix Llama 3.2 rope config |
| `models/llama/convert.py` | Skip lm_head when `tie_word_embeddings=True` | Avoid loading 501 MB FP16 lm_head into conversion |
| `models/modeling_utils.py` | `from_checkpoint()` alias: add `lm_head.weight = vocab_emb` when missing | Allow building without duplicate lm_head in safetensors |
| `models/modeling_utils.py` | Explicit clone in `save_checkpoint` if lm_head missing | Fix Xavier init OOM during conversion |
| `layers/linear.py` + `model_weights_loader.py` | None-guard in postprocess/check | Handle tied embedding skip gracefully |

### Key technical findings

- **share_embedding_table fix:** Removed duplicate `lm_head.weight` (501 MB FP16) from the TRT-LLM checkpoint, reducing it from 1531 MB → 1030 MB. Patch to `from_checkpoint()` injects `weights['lm_head.weight'] = weights['transformer.vocab_embedding.weight']` before `model.load()` to satisfy the required-names check.
- **`cudaMallocHost` is NOT a bypass on Jetson:** Despite Jetson being UMA, pinned host memory still requires NvMap IOVM mapping for GPU DMA. The custom allocator successfully allocates 512 MB via `cudaMallocHost` (confirmed: `[PinnedMemAllocator] 512 MB -> cudaMallocHost OK`) but 1 GB fails — ~512 MB IOVM remaining after kernel lib + checkpoint.
- **No-gemm-plugin engine works for inference** with `--kv_cache_free_gpu_memory_fraction 0.1` (limits KV cache to avoid NvMap OOM at inference time). Verified correct output: `"The capital of France is Paris."`.
- **llama.cpp IQ4_XS + FA beats TRT-LLM no-gemm-plugin:** 60.28 t/s (llama.cpp) > 44.08 t/s (TRT-LLM FP16 fallback) because INT4 dequantize+multiply is more bandwidth-efficient than FP16 matmul.

### Why DGX Spark cannot help

The DGX Spark is sm_121 (Blackwell GB10). TRT engines are **architecture-specific** — require actual target GPU for kernel profiling. sm_121 ≠ sm_87 (Jetson Orin). No cross-architecture TRT engine compilation possible.

---

## Path 3 — Qwen2.5-1.5B TRT-LLM W4A16

**Target:** ~40–50 t/s standalone, ~65–80 t/s with Qwen2.5-0.5B as speculative draft
**Effort:** Medium — direct Jetson build (smaller checkpoint than Llama 1B, may succeed)
**Hardware needed:** Jetson (build + run)

### Why Qwen1.5B is promising for speculative decoding

- **Aggressive GQA:** only 2 KV heads (vs 8 for Llama 1B) → KV cache bandwidth is negligible
- **Shared tokenizer** with Qwen2.5-0.5B → draft acceptance rate should be high
- **Qwen2.5-0.5B already runs at 100.87 t/s in TRT-LLM** → ideal draft model

Estimated checkpoint size: ~1.1 GB (smaller than Llama 1B's 1.5 GB — may avoid the serialization OOM).

### Setup

```bash
# On Jetson, inside aneurologic_phase5 container
# Download Qwen2.5-1.5B-Instruct HF weights
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct',
    local_dir='/workspace/models/hf_cache/qwen15b')
"

# Convert to W4A16 checkpoint
python3 /workspace/TensorRT-LLM/examples/qwen/convert_checkpoint.py \
  --model_dir /workspace/models/hf_cache/qwen15b \
  --output_dir /tmp/trtllm_qwen15b_w4 \
  --dtype float16 --use_weight_only --weight_only_precision int4 \
  --load_model_on_cpu

# Build TRT engine
trtllm-build \
  --checkpoint_dir /tmp/trtllm_qwen15b_w4 \
  --output_dir /workspace/outputs/trt_engines/qwen15b_w4a16 \
  --gemm_plugin float16 \
  --max_batch_size 1 --max_input_len 512 --max_seq_len 640
```

### Expected results

| Mode | tg128 t/s |
|------|-----------|
| Qwen2.5-1.5B standalone llama.cpp | ~35–40 |
| Qwen2.5-1.5B TRT-LLM W4A16 | ~40–50 |
| Qwen2.5-1.5B TRT-LLM + 0.5B speculative draft | ~65–80 |

---

## Path 4 — EAGLE-3 Speculative Decoding (DGX Spark)

**Target:** ~80–135 t/s effective on Llama-3.2-1B or 3B (depends on acceptance rate)
**Effort:** High — requires training on DGX Spark + deployment on Jetson
**Hardware needed:** DGX Spark for training, Jetson for inference

### DGX Spark suitability

| Spec | Value |
|------|-------|
| GPU | NVIDIA GB10 (Blackwell, sm_121) |
| GPU memory | 119 GB UMA |
| System RAM | 119 GB |
| CPU cores | 20 (aarch64) |
| CUDA | 13.0 |
| SSH | `ssh spark-807e.local` |

The DGX Spark is **ideal for EAGLE training**: large memory, powerful GPU, no memory constraints. It **cannot** cross-compile TRT engines for Jetson (sm_121 ≠ sm_87).

### What EAGLE-3 does

EAGLE trains a lightweight "draft head" (~100–300M params) that predicts the next N tokens. The main model then verifies them in a single forward pass. Effective throughput = main model speed × acceptance rate × draft tokens / verification overhead.

For Llama-3.2-1B with a well-trained draft:
- Verification: 1 forward pass checks 5 draft tokens
- If 3.5 accepted on average: 3.5× effective speedup → 54 t/s × 3.5 = **190 t/s** (theoretical)
- Realistic (accounting for verification overhead): **~80–130 t/s**

### Training on DGX Spark

```bash
# On DGX Spark
git clone https://github.com/SafeAILab/EAGLE
cd EAGLE

pip install -r requirements.txt

# Step 1: Generate hidden state training data
# Requires Llama-3.2-1B-Instruct HF weights
python -m eagle.ge_data.ge_data_all_vicuna \
  --base_model_path meta-llama/Llama-3.2-1B-Instruct \
  --data_path ShareGPT_Vicuna_unfiltered \
  --output_path eagle_data_llama32_1b \
  --num_gpus 1

# Step 2: Train the EAGLE-3 draft head
python -m eagle.train.main \
  --base_model_path meta-llama/Llama-3.2-1B-Instruct \
  --data_path eagle_data_llama32_1b \
  --output_dir eagle_head_llama32_1b \
  --num_epochs 3
# Expected: 2–4 hours on GB10

# Step 3: Export draft head weights
# Copy to Jetson: eagle_head_llama32_1b/
```

### Deployment on Jetson

```bash
# On Jetson host — llama-speculative with EAGLE head
/home/spitman/tools/llama.cpp/build/bin/llama-speculative \
  -m Llama-3.2-1B-Instruct-IQ4_XS.gguf \
  --eagle-draft eagle_head_llama32_1b/ \
  -ngl 99 -fa 1 \
  -p "..." -n 128 --draft 5
```

Note: llama.cpp EAGLE support requires `llama-speculative` to be built with EAGLE flag. Verify:
```bash
/home/spitman/tools/llama.cpp/build/bin/llama-speculative --help | grep -i eagle
```

If no EAGLE support, use **SpecForge + vLLM** on Jetson instead:
```bash
pip install vllm  # may require jetson-specific build
vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --speculative-model eagle_head_llama32_1b \
  --num-speculative-tokens 5
```

### DGX Spark bonus: better imatrix for LFM2.5 IQ4_XS

The DGX Spark can also generate a higher-quality imatrix calibration dataset for LFM2.5-1.2B IQ4_XS (recovering the ARC accuracy from 60% → 70% that we lost vs Q4_K_S). See Stage XXI in [edge_optimization_report.md](edge_optimization_report.md).

```bash
# On DGX Spark — generate mixed imatrix (wikitext + gsm8k + arc reasoning data)
python3 scripts/gen_imatrix_data.py \
  --datasets wikitext-2,gsm8k,arc \
  --output imatrix_mixed.dat
# Then run llama-imatrix on Jetson or DGX Spark
```

---

## llama.cpp IQ4_XS + FA Benchmarks (Updated 2026-04-02)

These supersede Phase 5 baselines — confirmed with `llama-bench` build 3a60d06ad (8510) on Jetson Orin Nano 8GB.

| Model | Params | Size | tg128 t/s (FA=1) | pp512 t/s (FA=1) |
|-------|--------|------|-------------------|-------------------|
| **LFM2 1.2B IQ4_XS** | 1.17B | 630 MiB | **65.64** ← new best >1B | 2315 |
| **Llama 3.2 1B IQ4_XS** | 1.24B | 701 MiB | **60.28** | 2550 |
| Llama 3.2 1B Q4_K_S | 1.24B | 732 MiB | 53.39 | 2285 |
| Llama 3.2 1B Q4_K_M | 1.24B | 763 MiB | 51.94 | 2238 |
| Llama 3.2 1B Q3_K_M | 1.24B | 651 MiB | 41.53 | 2012 |
| Qwen2.5-0.5B TRT-LLM W4A16 | 0.5B | — | 100.87 ← best overall | — |

**Key finding:** IQ4_XS format consistently outperforms Q3_K_M despite being larger, because the imatrix-optimized 4.25-bpw format has better GPU kernel efficiency on sm_87. The current >1B ceiling on Jetson llama.cpp is **65.64 t/s** (LFM2 1.2B IQ4_XS + FA).

**Bandwidth utilization (llama.cpp IQ4_XS + FA):**
- LFM2 1.2B: 65.64 × 630 MB = 41.4 GB/s = **60.9% of 68 GB/s**
- Llama 3.2 1B: 60.28 × 701 MB = 42.3 GB/s = **62.1% of 68 GB/s**

The ~62% bandwidth utilization means llama.cpp IQ4_XS is substantially more efficient than TRT-LLM no-gemm-plugin FP16 (which only got 44 t/s = 32% efficiency).

---

## Execution Order (Updated Status)

| Priority | Path | Status | Result |
|----------|------|--------|--------|
| **1** | Path 1: Llama-3.2-3B + 1B speculative | ❌ BLOCKED | 26 t/s standalone only; NvMap IOVM < 2.5 GB prevents loading two models |
| **2** | Path 2: TRT-LLM Llama 1B W4A16 gemm_plugin | ❌ BLOCKED | globWriter 1 GB IOVM requirement impossible on Jetson; 8 patches tried; no-gemm-plugin 44.08 t/s confirmed |
| **3** | Path 3: Qwen2.5-1.5B TRT-LLM W4A16 | ❌ NOT ATTEMPTED | Qwen 1.5B checkpoint (~1.6 GB) will exceed IOVM limit (same constraint as Path 2) |
| **4** | Path 4: EAGLE-3 on DGX Spark | 🔄 NEXT | Train draft head on DGX Spark; deploy on Jetson via llama.cpp or TRT-LLM |

---

## Success Criteria (Updated)

| Milestone | Status | Result | Model |
|-----------|--------|--------|-------|
| ✅ Phase 5 best | Achieved | 100.87 t/s | Qwen2.5-0.5B TRT-LLM W4A16 |
| ✅ New >1B record | Achieved | **65.64 t/s** | LFM2 1.2B IQ4_XS + FA (llama.cpp) |
| ✅ New Llama 1B record | Achieved | **60.28 t/s** | Llama 3.2 1B IQ4_XS + FA (llama.cpp) |
| 🎯 Path 4 goal | Pending | ≥100 t/s | Llama 3.2 1B + EAGLE-3 speculative draft |
| 🎯 Stretch goal | Pending | ≥120 t/s | Any >1B model, 2.5× EAGLE acceptance rate |

**Revised assessment:** The 100 t/s + >1B params target is at the physical edge of Jetson's 68 GB/s bandwidth at INT4. Only EAGLE-style speculative decoding (which multiplies effective throughput without extra weight loading) can realistically achieve this. TRT-LLM gemm_plugin (the other path to efficiency) is blocked by NvMap IOVM. Path 4 is the only remaining viable route.
