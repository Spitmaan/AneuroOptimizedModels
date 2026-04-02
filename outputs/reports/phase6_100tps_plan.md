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

## Path 5 — Lookup Decoding (llama-lookup, n-gram speculative) ❌ BLOCKED

**Status: BLOCKED — per-step CPU-GPU sync overhead exceeds compute gain on Jetson UMA**
**Tested: 2026-04-02**

### What was tested

`llama-lookup` n-gram based speculative decoding for Llama-3.2-1B IQ4_XS + FA.
- Cold run: dynamic cache starts empty, fills during generation
- Warm run: static cache from run 1 (same n-grams)

### Results

| Metric | Cold (dynamic) | Warm (static) |
|--------|---------------|---------------|
| Acceptance rate | 26.3% | 26.4% |
| Draft tokens per step | 5 | 5 |
| Avg accepted per step | 0.64 | 0.74 |
| GPU eval t/s (per step) | 56.0 | 54.9 |
| Wall-clock generation t/s | **~35 t/s** | **~35 t/s** |
| Baseline (no speculative) | — | **60.88 t/s** |

### Root cause

Each speculative decoding step requires CPU↔GPU synchronization:
1. CPU: query n-gram hash table → propose 5 draft tokens
2. GPU: verify all 5 drafts + generate final token in one forward pass
3. CPU: accept/reject each draft token, update KV cache state
4. Repeat

On a discrete GPU, this sync overhead is negligible (~1 ms). On Jetson UMA, per-step overhead is ~28 ms vs ~18 ms GPU compute time. Total overhead per step exceeds the compute savings from multi-token verification.

**Result:** 42% slower than baseline. `llama-lookup-create` also crashes (core dump) — no static cache from external corpus is possible in v8510.

**Conclusion:** Lookup decoding cannot overcome the Jetson UMA per-step sync cost. Not viable.

---

---

## Why Previous Assessments Were Incomplete: MLC-LLM and TRT Edge-LLM

All five blocked paths used one of two frameworks:
- **llama.cpp**: Open-source, general-purpose. No EAGLE support. Lookup decoding blocked by sync overhead.
- **TRT-LLM (standard)**: Server/datacenter-oriented. Its Python `trtllm-build` uses a C++ serialization buffer (`globWriter`) that requires 1 GB GPU allocation — hitting the NvMap IOVM constraint on 8 GB Jetson.

**Neither is the right tool for Jetson.** Two purpose-built frameworks exist for Jetson:

### Framework 1: MLC-LLM (Machine Learning Compilation)

**MLC-LLM** ([github.com/mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm)) uses Apache TVM to compile models to device-specific fused CUDA kernels. Pre-built container available immediately for JetPack 6.2:

- Pre-built `dustynv/mlc:0.20.0-r36.4.0` works on JetPack 6.2.1 (no build needed)
- TVM kernel fusion eliminates launch overhead that costs llama.cpp ~20% throughput
- Pre-quantized q4f16_1 models available on HuggingFace (no export step)
- Potential: 70–90 t/s standalone; EAGLE-2 speculative decoding available in some builds

### Framework 2: TensorRT Edge-LLM

**TensorRT Edge-LLM** ([github.com/NVIDIA/TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)) — purpose-built C++ inference runtime for embedded platforms (Jetson, DRIVE).

| | TRT-LLM (what we used) | TRT Edge-LLM (new) |
|---|---|---|
| Target | Server/cloud GPU | Embedded: Jetson, DRIVE |
| Engine builder | Python `trtllm-build` + globWriter (~1 GB GPU alloc) | C++ `llm_build` binary (edge-optimized, lower memory) |
| EAGLE-3 support | No | **Yes** |
| NVFP4 quantization | No | Yes |
| Jetson Orin support | v0.12.0-jetson branch (64GB AGX tested) | **✅ JetPack 6.x via jetson-containers v0.5.0** |
| Export + build location | On-device (blocked by IOVM for engine build) | **On Jetson (both export and build run inside container)** |
| Pre-built container | No | **No pre-built; must build locally via `./build.sh`** |

**Deployment correction:** Official TRT Edge-LLM docs target x86-64 host + Jetson Thor (JetPack 7.1). However, `dusty-nv/jetson-containers` v0.5.0 package (`requires: '>=36'`) builds both the Python export tools AND C++ runtime directly on Jetson, working around the platform requirement. DGX Spark is NOT needed.

**Key insight:** The C++ `llm_build` avoids the globWriter issue. The Python export step (AWQ quantization + ONNX) runs inside the container on Jetson — for 1B–3B models, the export memory footprint stays under the IOVM limit.

**IOVM analysis for our models:**
- Llama-3.2-1B INT4: ~700 MB IOVM
- EAGLE-3 head (draft): ~50–200 MB IOVM
- C++ runtime overhead: ~200 MB
- **Total: ~1.0–1.1 GB → well within 2.5 GB IOVM budget**

---

## Path 6 — MLC-LLM Baseline (Fastest Path to Data)

**Target:** Beat llama.cpp 60.88 t/s baseline with TVM-compiled kernels; confirm if MLC alone can approach 100 t/s  
**Status: 🔄 NEXT — container pull in progress**  
**Expected: 70–90 t/s standalone** (TVM kernel fusion typically +20–40% over llama.cpp generic CUDA)

MLC-LLM uses Apache TVM to compile models to device-specific fused CUDA kernels, eliminating the kernel launch overhead that costs llama.cpp ~15–20% throughput. Pre-built container is available immediately — no build step.

### Step 1: Pull and verify MLC-LLM container on Jetson

```bash
# On Jetson (already in progress):
docker pull dustynv/mlc:0.20.0-r36.4.0

# Verify container
docker run --rm --runtime nvidia dustynv/mlc:0.20.0-r36.4.0 \
    python3 -c "import mlc_llm; print(mlc_llm.__version__)"
```

### Step 2: Pre-flight MAXN SUPER mode

```bash
ssh spitman@spitman-jetson
sudo nvpmodel -m 0      # MAXN SUPER mode (25W max, full GPU clocks)
sudo jetson_clocks      # lock all clocks to maximum frequency
cat /sys/devices/57000000.gpu/devfreq/57000000.gpu/cur_freq  # confirm GPU at max
```

### Step 3: Download pre-quantized Llama-3.2-1B q4f16_1

MLC uses its own quantization format (q4f16_1 = INT4 grouped quantization, FP16 activations). Pre-quantized weights are available on HuggingFace:

```bash
docker run --rm --runtime nvidia \
    -v /home/spitman/models/mlc:/models \
    dustynv/mlc:0.20.0-r36.4.0 \
    bash -c "
        huggingface-cli download mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC \
            --local-dir /models/llama32-1b-q4f16_1
    "
```

If not available, quantize from HF weights:
```bash
docker run --rm --runtime nvidia \
    -v /home/spitman/models/mlc:/models \
    dustynv/mlc:0.20.0-r36.4.0 \
    bash -c "
        python3 -m mlc_llm convert_weight \
            /models/llama32-1b-hf \
            --quantization q4f16_1 \
            --output /models/llama32-1b-q4f16_1
    "
```

### Step 4: Compile TVM kernel for Jetson sm_87

```bash
docker run --rm --runtime nvidia \
    -v /home/spitman/models/mlc:/models \
    dustynv/mlc:0.20.0-r36.4.0 \
    bash -c "
        python3 -m mlc_llm gen_config \
            /models/llama32-1b-q4f16_1 \
            --quantization q4f16_1 \
            --conv-template llama-3 \
            --output /models/llama32-1b-q4f16_1

        python3 -m mlc_llm compile \
            /models/llama32-1b-q4f16_1 \
            --device cuda \
            --output /models/llama32-1b-q4f16_1/lib.so
    "
```

### Step 5: Benchmark

```bash
docker run --rm --runtime nvidia \
    -v /home/spitman/models/mlc:/models \
    dustynv/mlc:0.20.0-r36.4.0 \
    bash -c "
        python3 -m mlc_llm bench \
            --model /models/llama32-1b-q4f16_1 \
            --device cuda \
            --generate-length 128 \
            --prompt-len 512
    "
```

Compare result against **60.88 t/s llama.cpp IQ4_XS+FA baseline**.

### Expected outcome

- **Best case:** 80–95 t/s (TVM fused kernels, good BW utilization on sm_87)
- **Likely case:** 65–80 t/s — meaningful improvement; motivates EAGLE-2 path
- **Stretch:** MLC + EAGLE-2 head may hit 100 t/s if acceptance rate is good
- **IOVM risk:** LOW — MLC q4f16_1 1B model ~1 GB; stays well under limit

---

## Path 7 — TRT Edge-LLM via jetson-containers (EAGLE-3 Primary Path)

**Target:** Verify TRT Edge-LLM pipeline works end-to-end on Jetson JetPack 6.2; foundation for EAGLE-3  
**Status: 🔒 Start after Path 6 data collected**  
**Expected standalone: 55–75 t/s** (baseline before EAGLE-3 multiplier)

### Step 1: Build tensorrt_edgellm container on Jetson

No pre-built container exists for JetPack 6.2. The jetson-containers `tensorrt_edgellm:0.5.0` package (`requires: '>=36'`) builds both the Python export tools and C++ runtime directly on Jetson.

**Build time estimate: 45–90 minutes** (clones repo, pip install, cmake + make -j6)

```bash
ssh spitman@spitman-jetson
cd /home/spitman/Projects/Aneurologic/modelgarden/jetson-containers

# Pre-flight: MAXN SUPER mode (do this before any long compute task)
sudo nvpmodel -m 0 && sudo jetson_clocks

# Build the container (runs install.sh + build.sh inside Docker)
./build.sh tensorrt_edgellm:0.5.0
```

Monitor progress:
```bash
docker logs -f $(docker ps -q --filter ancestor=jetson-containers-build) 2>/dev/null
```

After build, the container should be named `tensorrt_edgellm:0.5.0`. Verify:
```bash
docker run --rm --runtime nvidia tensorrt_edgellm:0.5.0 \
    python3 /home/spitman/Projects/Aneurologic/modelgarden/jetson-containers/packages/llm/tensorrt_edgellm/test.py
# Expected: "tensorrt_edgellm version: 0.5.0", "llm_build --help: OK"
```

### Step 2: Export Llama-3.2-1B INT4 inside container on Jetson

The export runs inside the container. For a 1B model, AWQ quantization calibration uses ~1.5 GB GPU memory — safe under IOVM limit.

```bash
docker run --rm --runtime nvidia \
    -v /home/spitman/models/trt_edge:/models \
    -v /home/spitman/Projects/Aneurologic/modelgarden/jetson-containers/data/models:/hf_models \
    tensorrt_edgellm:0.5.0 \
    bash -c "
        # Download Llama-3.2-1B-Instruct if not present
        huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
            --local-dir /hf_models/Llama-3.2-1B-Instruct

        # Quantize + export to ONNX format
        tensorrt-edgellm-quantize-llm \
            --model_dir /hf_models/Llama-3.2-1B-Instruct \
            --quantization int4_awq \
            --output_dir /models/llama32_1b_onnx \
            --dtype float16

        tensorrt-edgellm-export-llm \
            --model_dir /hf_models/Llama-3.2-1B-Instruct \
            --quantized_dir /models/llama32_1b_onnx \
            --output_dir /models/llama32_1b_exported
    "
```

If `tensorrt-edgellm-quantize-llm` and `tensorrt-edgellm-export-llm` are not separate commands, use the unified script:
```bash
# Check what export scripts are available in the repo
docker run --rm tensorrt_edgellm:0.5.0 \
    bash -c "ls /opt/TensorRT-Edge-LLM/examples/ && find /opt/TensorRT-Edge-LLM -name 'export*.py' | head -10"
```

**If export OOMs (IOVM hit during calibration):**
```bash
# Increase swap before running export
sudo fallocate -l 4G /swapfile2 && sudo chmod 600 /swapfile2
sudo mkswap /swapfile2 && sudo swapon /swapfile2
# Re-run export; remove swap after: sudo swapoff /swapfile2
```

### Step 3: Build TRT engine on Jetson

```bash
docker run --rm --runtime nvidia \
    -v /home/spitman/models/trt_edge:/models \
    tensorrt_edgellm:0.5.0 \
    bash -c "
        llm_build \
            --model_dir /models/llama32_1b_exported \
            --output_file /models/llama32_1b.engine \
            --max_batch_size 1 \
            --max_input_len 2048 \
            --max_output_len 512 \
            --mmap
    "
```

Binary paths inside container (confirmed from test.py):
- `llm_build` → `/opt/TensorRT-Edge-LLM/build/examples/llm/llm_build`
- `llm_inference` → `/opt/TensorRT-Edge-LLM/build/examples/llm/llm_inference`
- Both are in `PATH` via container's `ENV PATH` setting

### Step 4: Benchmark vs baseline

```bash
docker run --rm --runtime nvidia \
    -v /home/spitman/models/trt_edge:/models \
    tensorrt_edgellm:0.5.0 \
    bash -c "
        echo 'Running 128-token generation benchmark...'
        time llm_inference \
            --engine_file /models/llama32_1b.engine \
            --max_new_tokens 128
    "
```

Compare wall-clock t/s vs **60.88 t/s llama.cpp IQ4_XS+FA baseline**.

### Expected outcome

- **Best case:** 65–80 t/s (TRT-optimized kernels, better BW utilization)
- **Likely case:** 55–65 t/s (bandwidth-bound at 1B; TRT overhead amortized)
- **Risk:** Build may fail if JetPack 6.2 libs incompatible with TRT Edge-LLM 0.5.0; fallback is to check out an older branch

---

## Path 7B — TRT Edge-LLM + EAGLE-3 Speculative Decoding (Primary 100 t/s Path)

**Target:** ≥100 t/s with Llama-3.2-1B via EAGLE-3 draft head  
**Status: 🔒 BLOCKED on Path 7 baseline success**  
**Expected: 120–180 t/s** (EAGLE-3 2–3× multiplier on ~60 t/s base)  
**IOVM budget:** ~1.1 GB (base 700 MB + EAGLE head 200 MB + runtime 200 MB) — fits

### Why EAGLE-3 works on Jetson where standard speculative decoding doesn't

Standard speculative decoding (Path 1) required loading two full models → IOVM exhaustion.
EAGLE-3 uses a **draft head**: a tiny (~50–300 MB) neural network that predicts future tokens using the hidden states from the main model's last layer. No second full model needed.

EAGLE-3 also solves the CPU-GPU sync problem that killed lookup decoding (Path 5): because the draft head runs as a fused CUDA kernel alongside the main model, there is no CPU round-trip per step.

| | Lookup decoding (Path 5, failed) | EAGLE-3 (Path 7B) |
|---|---|---|
| Draft source | CPU n-gram hash table | GPU draft head (fused) |
| CPU-GPU sync per step | Yes (~28 ms overhead) | No (kernel-level) |
| IOVM cost | None (no model) | ~50–300 MB |
| Acceptance rate | 26% | 60–80% (learned, model-specific) |

### Step 1: Locate or train an EAGLE-3 head for Llama-3.2-1B

**Option A — Check for pre-trained head on HuggingFace:**
```bash
# Search for EAGLE-3 heads for Llama-3.2-1B
huggingface-cli search --type model "eagle llama-3.2-1b" 2>/dev/null | head -10
# Direct probe:
huggingface-cli download yuhuili/EAGLE3-LLaMA3.2-Instruct-1B --local-dir /tmp/eagle3-probe 2>&1 | head -3
# If 404 → not published; proceed to Option B
```

**Option B — Train EAGLE-3 head on DGX Spark:**

DGX Spark (119 GB UMA, sm_121 Blackwell) is suitable for draft head training (~1–2 hours).

```bash
ssh spark-807e.local
python3 -m venv ~/eagle_train_env && source ~/eagle_train_env/bin/activate
pip install torch transformers accelerate datasets

git clone https://github.com/SafeAILab/EAGLE && cd EAGLE
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir ./llama32-1b

# Generate training data from base model hidden states
python3 -m eagle.ge_data.allocation \
    --model_name_or_path ./llama32-1b \
    --outdir ./eagle_data

# Train draft head (check EAGLE repo for EAGLE-3 branch/flag)
python3 -m eagle.train.main \
    --tmpdir ./eagle_data \
    --cpdir ./eagle_head_llama32_1b \
    --basepath ./llama32-1b

# Transfer head to Jetson
scp -r ./eagle_head_llama32_1b spitman@spitman-jetson:/home/spitman/models/trt_edge/
```

### Step 2: Export speculative engine (inside container on Jetson)

```bash
docker run --rm --runtime nvidia \
    -v /home/spitman/models/trt_edge:/models \
    tensorrt_edgellm:0.5.0 \
    bash -c "
        llm_build \
            --model_dir /models/llama32_1b_exported \
            --speculative_model_dir /models/eagle_head_llama32_1b \
            --output_file /models/llama32_1b_eagle3.engine \
            --max_draft_len 5 \
            --max_batch_size 1 \
            --max_input_len 2048 \
            --max_output_len 512 \
            --mmap
    "
```

### Step 3: Benchmark speculative engine

```bash
docker run --rm --runtime nvidia \
    -v /home/spitman/models/trt_edge:/models \
    tensorrt_edgellm:0.5.0 \
    bash -c "time llm_inference --engine_file /models/llama32_1b_eagle3.engine --max_new_tokens 128"
```

EAGLE-3 effective throughput = base × (1 + acceptance × draft_len)  
Conservative (50% acceptance, 3 drafts): 65 × 2.5 / 1.2 ≈ **135 t/s**  
With 70% acceptance, 5 drafts: 65 × 4.5 / 1.2 ≈ **244 t/s** (theoretical max)

---

---

## Multi-Model Sweep: Bandwidth Ceilings and EAGLE-3 Targets

Paths 8–10 extend the TRT Edge-LLM approach to Qwen2.5, LFM, and larger Llama models. The strategy is to **start large (7–8B) and work down** — largest model that can hit 100 t/s wins.

### Memory fit on 8 GB Jetson (INT4 AWQ)

| Model | INT4 size | KV cache (2K ctx) | Runtime | Total | Fits 8 GB? |
|-------|----------:|------------------:|--------:|------:|:----------:|
| Llama-3.2-1B | ~0.7 GB | ~0.1 GB | ~0.3 GB | ~1.1 GB | ✅ |
| Llama-3.2-3B | ~1.9 GB | ~0.2 GB | ~0.3 GB | ~2.4 GB | ✅ |
| Qwen2.5-1.5B | ~1.0 GB | ~0.1 GB | ~0.3 GB | ~1.4 GB | ✅ |
| Qwen2.5-3B | ~1.9 GB | ~0.2 GB | ~0.3 GB | ~2.4 GB | ✅ |
| Qwen2.5-7B | ~4.0 GB | ~0.4 GB | ~0.4 GB | ~4.8 GB | ✅ (tight) |
| Llama-3.1-8B | ~4.5 GB | ~0.4 GB | ~0.4 GB | ~5.3 GB | ✅ (tight) |
| Qwen2.5-14B | ~8.5 GB | — | — | >8 GB | ❌ |
| LFM2.5-1.2B | ~0.7 GB | ~0.1 GB | ~0.3 GB | ~1.1 GB | ✅ (arch risk) |

OS + background processes take ~1.5 GB, so practical headroom is ~6.5 GB usable.

### EAGLE-3 effective throughput projections

| Model | Standalone t/s (est.) | 2.5× EAGLE | 3.5× EAGLE | Hits 100 t/s? |
|-------|----------------------:|:----------:|:----------:|:-------------:|
| Llama-3.2-1B | ~60 | 150 | 210 | ✅ |
| Llama-3.2-3B | ~26 | 65 | 91 | ❌ (marginal) |
| Qwen2.5-1.5B | ~50 | 125 | 175 | ✅ |
| Qwen2.5-3B | ~28 | 70 | 98 | ❌ (marginal) |
| Qwen2.5-7B | ~14 | 35 | 49 | ❌ |
| Llama-3.1-8B | ~12 | 30 | 42 | ❌ |

**Key finding from this analysis:** 100 t/s is achievable with 1–1.5B models via EAGLE-3. 3B models are borderline (depends on acceptance rate). 7–8B models cannot realistically hit 100 t/s on 68 GB/s bandwidth regardless of EAGLE acceleration — but they establish absolute performance records for the largest models deployable on this hardware.

**Test order rationale:** Start 7B/8B (largest possible, benchmark record) → 3B (borderline, may hit 100) → 1.5B (Qwen2.5 is a stronger architecture than Llama at same size) → LFM2.5 (arch uncertainty).

---

## Path 8 — TRT Edge-LLM: Larger Llama Models (3B + 8B)

**Status: 🔒 After Path 7 (pipeline validated)**  
**Goal:** Largest Llama model that fits 8 GB; benchmark record for >3B on Jetson; 3B + EAGLE-3 may hit 100 t/s

Export and build run inside the `tensorrt_edgellm:0.5.0` container on Jetson (same pipeline as Path 7). 3B model export uses ~3 GB GPU memory during AWQ calibration — verify IOVM headroom before starting.

### 8A: Llama-3.2-3B (we already have IQ4_XS GGUF at 1.7 GB)

We confirmed Llama-3.2-3B-IQ4_XS at **26.01 t/s standalone** in Path 1. TRT Edge-LLM INT4 AWQ may improve this via custom kernels. EAGLE-3 + 3B needs ~75%+ acceptance to hit 100 t/s — achievable for on-domain tasks.

```bash
docker run --rm --runtime nvidia \
    -v /home/spitman/models/trt_edge:/models \
    -v /home/spitman/Projects/Aneurologic/modelgarden/jetson-containers/data/models:/hf_models \
    tensorrt_edgellm:0.5.0 \
    bash -c "
        huggingface-cli download meta-llama/Llama-3.2-3B-Instruct \
            --local-dir /hf_models/Llama-3.2-3B-Instruct
        tensorrt-edgellm-quantize-llm \
            --model_dir /hf_models/Llama-3.2-3B-Instruct \
            --quantization int4_awq \
            --output_dir /models/llama32_3b_onnx --dtype float16
        tensorrt-edgellm-export-llm \
            --model_dir /hf_models/Llama-3.2-3B-Instruct \
            --quantized_dir /models/llama32_3b_onnx \
            --output_dir /models/llama32_3b_exported
        llm_build \
            --model_dir /models/llama32_3b_exported \
            --output_file /models/llama32_3b.engine \
            --max_batch_size 1 --max_input_len 2048 --max_output_len 512 --mmap
    "
```

EAGLE-3 head for 3B:
```bash
# Check for pre-trained head
huggingface-cli download yuhuili/EAGLE3-LLaMA3.2-Instruct-3B --local-dir /tmp/eagle3-3b-probe 2>&1 | head -3
# If not found: train on DGX Spark (same pipeline as Path 7B Step 1 Option B)
```

Export size ~1.9 GB INT4, EAGLE head ~150 MB.

### 8B: Llama-3.1-8B (start here — largest viable model on 8 GB)

Llama-3.1-8B INT4 AWQ ≈ 4.5 GB. On our 8 GB Nano, engine fit is the primary risk. EAGLE-3 at 8B cannot hit 100 t/s (bandwidth-limited to ~42 t/s max) but sets the absolute largest-model benchmark.

**Note:** 8B AWQ quantization calibration requires ~20+ GB system RAM (FP16 forward passes). Jetson has only 8 GB UMA — this step must run on DGX Spark.

```bash
# On DGX Spark — 8B quantization only (too large for Jetson 8 GB RAM)
ssh spark-807e.local
python3 -m venv ~/trt_edge_export && source ~/trt_edge_export/bin/activate
pip install tensorrt_edge_llm || \
    (git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git && cd TensorRT-Edge-LLM && pip install .)

huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ~/models/llama31-8b

tensorrt-edgellm-quantize-llm \
    --model_dir ~/models/llama31-8b \
    --quantization int4_awq \
    --output_dir ~/models/llama31_8b_onnx --dtype float16

tensorrt-edgellm-export-llm \
    --model_dir ~/models/llama31-8b \
    --quantized_dir ~/models/llama31_8b_onnx \
    --output_dir ~/models/llama31_8b_exported

# Transfer exported ONNX to Jetson
scp -r ~/models/llama31_8b_exported spitman@spitman-jetson:/home/spitman/models/trt_edge/
```

Then build TRT engine on Jetson (export already done; only `llm_build` runs on device):
```bash
docker run --rm --runtime nvidia \
    -v /home/spitman/models/trt_edge:/models \
    tensorrt_edgellm:0.5.0 \
    bash -c "
        llm_build \
            --model_dir /models/llama31_8b_exported \
            --output_file /models/llama31_8b.engine \
            --max_batch_size 1 --max_input_len 1024 --max_output_len 256 --mmap
    "
```

EAGLE-3 head for 8B (most likely to exist pre-trained):
```bash
huggingface-cli download yuhuili/EAGLE3-LLaMA3.1-Instruct-8B --local-dir /tmp/eagle3-8b-probe 2>&1 | head -3
```

**Memory risk on Jetson:** 8B INT4 ≈ 4.5 GB + KV + runtime ≈ 5.3 GB total. With OS at ~1.5 GB, total ≈ 6.8 GB — within 8 GB but tight. If Jetson OOMs during build:
- Reduce `--max_input_len` to 512 (smaller KV cache)
- Use swap during build only (not inference)

### Expected 8B results

| Config | Expected t/s | Notes |
|--------|-------------:|-------|
| Llama-3.1-8B INT4 standalone | ~12–15 t/s | Bandwidth-bound; 68 GB/s / ~4.5 GB = ~15 t/s theoretical |
| Llama-3.1-8B + EAGLE-3 (50% accept) | ~30–40 t/s | Not 100 t/s, but largest model ever run on this hardware |
| Llama-3.2-3B INT4 standalone | ~25–30 t/s | |
| Llama-3.2-3B + EAGLE-3 (70% accept) | ~80–100 t/s | Borderline 100 t/s |

---

## Path 9 — TRT Edge-LLM: Qwen2.5 Series (7B, 3B, 1.5B)

**Status: 🔒 After Path 8**  
**Goal:** Qwen2.5-7B is the priority (largest model with confirmed TRT Edge-LLM support); Qwen2.5-1.5B is the best candidate for 100 t/s after Llama-1B  
**Architecture note:** Qwen2.5 is explicitly listed in TRT Edge-LLM's supported model list (Qwen 2/2.5/3). No architecture risk.

### 9A: Qwen2.5-7B — start here (largest Qwen that fits 8 GB)

```bash
# On DGX Spark
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./qwen25-7b

python3 export.py \
    --model_dir ./qwen25-7b \
    --quantization int4_awq \
    --output_dir ./qwen25_7b_exported \
    --dtype float16 \
    --max_input_len 2048 \
    --max_output_len 512
```

EAGLE-3 head for Qwen2.5-7B:
```bash
huggingface-cli download yuhuili/EAGLE3-Qwen2.5-7B-Instruct --local-dir ./eagle3-qwen7b 2>&1 | head -5
```

### 9B: Qwen2.5-3B

```bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir ./qwen25-3b

python3 export.py --model_dir ./qwen25-3b --quantization int4_awq \
    --output_dir ./qwen25_3b_exported --dtype float16 \
    --max_input_len 2048 --max_output_len 512
```

### 9C: Qwen2.5-1.5B — secondary 100 t/s candidate

Qwen2.5-1.5B is a strong architecture with known high benchmark scores at 1.5B scale. Standalone ~50 t/s expected; EAGLE-3 target ~125–175 t/s. Likely the second model (after Llama-1B) to hit 100 t/s.

```bash
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./qwen25-1p5b

python3 export.py --model_dir ./qwen25-1p5b --quantization int4_awq \
    --output_dir ./qwen25_1p5b_exported --dtype float16 \
    --max_input_len 2048 --max_output_len 512
```

EAGLE-3 head for Qwen2.5-1.5B:
```bash
huggingface-cli download yuhuili/EAGLE3-Qwen2.5-1.5B-Instruct --local-dir ./eagle3-qwen1p5b 2>&1 | head -5
```

### Expected Qwen2.5 results

| Config | Expected t/s | Notes |
|--------|-------------:|-------|
| Qwen2.5-7B INT4 standalone | ~14 t/s | ~4 GB model; bandwidth-bound same as Llama-8B |
| Qwen2.5-7B + EAGLE-3 (55% accept) | ~35–45 t/s | Best result for a 7B model on 8 GB |
| Qwen2.5-3B INT4 standalone | ~28 t/s | |
| Qwen2.5-3B + EAGLE-3 (70% accept) | ~80–100 t/s | Borderline 100 t/s |
| Qwen2.5-1.5B INT4 standalone | ~50 t/s | Qwen architecture efficient at small scale |
| Qwen2.5-1.5B + EAGLE-3 (65% accept) | ~120–150 t/s | Strong 100 t/s candidate |

Note: We already have Qwen2.5-0.5B at 100.87 t/s via TRT-LLM W4A16 (Path 5/Phase 5). The Qwen2.5-1.5B + EAGLE-3 result would be the first time a **>1B Qwen** hits 100 t/s.

---

## Path 10 — TRT Edge-LLM: LFM2.5-1.2B

**Status: 🔒 After Path 9**  
**Goal:** LFM2.5 via TRT Edge-LLM — potential improvement over our current best (65.64 t/s IQ4_XS + FA)  
**⚠️ Architecture risk: HIGH**

### Why LFM2.5 is last

LFM2.5-1.2B is a **Recurrent Hybrid Model (LSTM + Attention)** — not a standard Transformer. TRT Edge-LLM's officially supported model list includes only attention-based architectures (Llama, Qwen, DeepSeek). SSM/hybrid architectures were explicitly blocked in standard TRT-LLM (Path 2/Phase 5 Stage XX).

**Check before attempting:**
```bash
# On DGX Spark — try export and see if it fails at model load
huggingface-cli download liquid-ai/LFM2.5-1.2B-Instruct --local-dir ./lfm25-1b

python3 export.py --model_dir ./lfm25-1b --quantization int4_awq \
    --output_dir ./lfm25_exported --dtype float16 2>&1 | head -20
# If it fails with "unsupported architecture" or KeyError — blocked, same as TRT-LLM
# If it succeeds — proceed to Jetson engine build
```

EAGLE-3 for LFM2.5: pre-trained heads almost certainly don't exist (non-standard arch). Training an EAGLE-3 head for a hybrid LSTM model requires adapting the EAGLE framework to produce LSTM hidden states rather than attention hidden states — significantly more complex than the Llama/Qwen case.

### What success looks like

If TRT Edge-LLM supports LFM2.5:
- Standalone INT4: ~65–75 t/s (may match or slightly beat llama.cpp IQ4_XS)
- With EAGLE-3 (if trainable): ~150–200 t/s — the largest potential gain
- **IOVM budget:** 1.2B INT4 ≈ 700 MB + EAGLE head ≈ 200 MB + runtime ≈ 300 MB = 1.2 GB ✅

### LFM3 consideration

LiquidAI may release LFM3 variants in larger sizes (3B, 7B) after 2026. If available, the same Path 10 workflow applies — check architecture support in TRT Edge-LLM first, then export on DGX Spark.

---

## Execution Order (Full — Phase 6)

| Priority | Path | Status | Result |
|----------|------|--------|--------|
| **1** | Path 1: Llama-3.2-3B + 1B speculative | ❌ BLOCKED | 26 t/s standalone; NvMap IOVM prevents loading two models |
| **2** | Path 2: TRT-LLM Llama 1B W4A16 gemm_plugin | ❌ BLOCKED | globWriter 1 GB IOVM; 8 patches tried; TRT-LLM is wrong tool for 8 GB Jetson |
| **3** | Path 3: Qwen2.5-1.5B TRT-LLM W4A16 | ❌ SKIPPED | Same IOVM constraint as Path 2 |
| **4** | Path 4: EAGLE-3 via llama.cpp | ❌ BLOCKED | llama.cpp has no EAGLE support |
| **5** | Path 5: Lookup decoding (llama-lookup) | ❌ BLOCKED | 26% acceptance; CPU-GPU sync overhead = 42% slower than baseline |
| **6** | **Path 6: MLC-LLM Llama-3.2-1B (fast baseline)** | 🔄 **NEXT** | Pre-built container; TVM kernels; pull in progress |
| **7** | **Path 7: TRT Edge-LLM Llama-3.2-1B (jetson-containers build)** | 🔒 After Path 6 | Build container on Jetson; export + C++ engine; EAGLE-3 in 7B |
| **7B** | **Path 7B: TRT Edge-LLM Llama-3.2-1B + EAGLE-3** | 🔒 After Path 7 | Primary 100 t/s route; 1B + EAGLE-3 = 120–180 t/s |
| **8** | **Path 8: TRT Edge-LLM Llama-3.1-8B + Llama-3.2-3B** | 🔒 After Path 7B | 8B needs DGX Spark for AWQ; 3B runs on Jetson; 3B borderline 100 t/s |
| **9** | **Path 9: TRT Edge-LLM Qwen2.5-7B / 3B / 1.5B** | 🔒 After Path 8 | Qwen2.5-1.5B strong 100 t/s candidate; 7B sets large-model record |
| **10** | **Path 10: TRT Edge-LLM LFM2.5-1.2B** | 🔒 After Path 9 | Architecture support uncertain; highest upside if it works |

---

## Success Criteria (Final)

| Milestone | Status | Target | Model |
|-----------|--------|--------|-------|
| ✅ Phase 5 best | Achieved | 100.87 t/s | Qwen2.5-0.5B TRT-LLM W4A16 |
| ✅ >1B record (llama.cpp) | Achieved | **65.64 t/s** | LFM2 1.2B IQ4_XS + FA |
| ✅ Llama 1B record (llama.cpp) | Achieved | **60.28 t/s** | Llama 3.2 1B IQ4_XS + FA |
| 🔄 Path 6 gate | Pending | Pipeline works | TRT Edge-LLM on JetPack 6.2 confirmed |
| 🎯 Primary: 100 t/s with >1B | Pending | ≥100 t/s | Llama 3.2 1B + EAGLE-3 |
| 🎯 Qwen 100 t/s with >1B | Pending | ≥100 t/s | Qwen2.5-1.5B + EAGLE-3 |
| 🎯 Largest model record | Pending | Best t/s | Llama-3.1-8B or Qwen2.5-7B |
| 🎯 3B 100 t/s stretch | Pending | ≥100 t/s | Llama-3.2-3B or Qwen2.5-3B + EAGLE-3 |

**Overall strategy:** Validate the TRT Edge-LLM + EAGLE-3 pipeline on Llama-3.2-1B (Paths 6–7), then sweep larger Llama (Path 8), Qwen2.5 (Path 9), and LFM2.5 (Path 10). Each path follows the identical DGX Spark export → Jetson `llm_build` workflow. Path 6 is the gate — if JetPack 6.2 experimental support fails, all subsequent paths are blocked by the same issue.
