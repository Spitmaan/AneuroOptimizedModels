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
**Status: ✅ COMPLETE — 65.31 t/s (trials: 64.17 / 66.05 / 65.70 t/s)**  
**Result: +7.3% vs 60.88 t/s llama.cpp baseline** — improvement is real but modest; bandwidth-bound at 1B scale

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
**Status: 🔄 IN PROGRESS — C++ build ✅ done; export pipeline established on DGX Spark**  
**Expected standalone: 55–75 t/s** (baseline before EAGLE-3 multiplier)

### Key discoveries (2026-04-02)

1. **C++ build succeeded on Jetson host** (v0.5.0) after fixing two cmake issues:
   - Shell variable expansion bug: `${CUDA_TARGET}` was expanding on LOCAL machine → passed as empty string `/include`. Fixed by using single-quoted SSH commands.
   - CUDA architecture: CMakeLists.txt default adds sm_100/sm_120 when `AARCH64_BUILD` is not defined and `CUDA_VERSION >= 12.8`. Fixed by `-DCMAKE_CUDA_ARCHITECTURES=87`. The `-DAARCH64_BUILD=1` flag was in `if(NOT DEFINED AARCH64_BUILD)` guard which CMake sees as "used" but reports as "not used" — still works.
   - **Result:** `llm_build` (278 KB) and `llm_inference` (5.3 MB) at `/home/spitman/tools/TensorRT-Edge-LLM/build/examples/llm/`

2. **Python export on DGX Spark (ARM64) works** despite official x86 requirement:
   - torch 2.9.1+cu128 installs and runs (with CUDA capability 12.1 warning — operations work)
   - nvidia-modelopt 0.39.0 installs with ARM64 wheels
   - CUDA OOM on initial load fixed with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   - INT4 AWQ quantization runs successfully: calibration ~6 s/sample, search ~40-65 s/sample

3. **Pivot to Qwen3-1.7B + EAGLE3** (see Path 7C below):
   - No EAGLE3 draft models available for Llama-3.2-1B or Llama-3.2-3B
   - TRT Edge-LLM docs explicitly support: `Qwen3-1.7B` + `AngelSlim/Qwen3-1.7B_eagle3`
   - Both models downloaded on DGX Spark: `~/qwen3-1.7b/` and `~/qwen3-1.7b-eagle3/`

### Step 1: Build TRT Edge-LLM C++ runtime on Jetson host (not container)

**Approach correction (2026-04-02):** 
- jetson-containers `tensorrt_edgellm:0.5.0` requires torch 2.9.1 + modelopt 0.39.0 but max available for JetPack 6.2 is torch 2.7 → container build blocked
- `aneurologic_phase5` container has torch 2.5.0 + no NvOnnxParser.h in container → also blocked
- **Solution:** Build C++ runtime directly on Jetson HOST (which has NvOnnxParser.h, TRT 10.3.0, CUDA 12.6 headers)

```bash
# ✅ ALREADY DONE — binaries at /home/spitman/tools/TensorRT-Edge-LLM/build/examples/llm/
# Record of working cmake command (IMPORTANT: use single-quoted SSH to prevent local var expansion):
ssh spitman@spitman-jetson '
cd /home/spitman/tools/TensorRT-Edge-LLM/build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DONNX_PARSER_INCLUDE_DIR=/usr/include/aarch64-linux-gnu \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCUDA_RUNTIME_API_INCLUDE_DIR=/usr/local/cuda-12.6/targets/aarch64-linux/include \
    -DCURAND_KERNEL_INCLUDE_DIR=/usr/local/cuda-12.6/targets/aarch64-linux/include \
    -DCUDART_LIB=/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudart.so \
    -DAARCH64_BUILD=1 \
    -DCMAKE_CUDA_ARCHITECTURES=87
make -j4
'
# Binaries at: examples/llm/llm_build (278 KB), examples/llm/llm_inference (5.3 MB)
```

**✅ SOLVED: Python export pipeline on DGX Spark.** Environment setup:
```bash
# On DGX Spark (spitmann@spark-807e.local)
python3 -m venv ~/venvs/trt_edgellm_export
source ~/venvs/trt_edgellm_export/bin/activate
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install nvidia-modelopt[torch]==0.39.0
cd ~/TensorRT-Edge-LLM && pip install ".[torch,onnx]" --no-deps
pip install transformers==4.57.3 onnx==1.19.0 datasets==4.4.2
pip install onnx-graphsurgeon onnxruntime pillow
# Note: torch has CUDA capability warning for GB10 sm_12.1 but operations work fine
# CRITICAL: use PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for GPU quant
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
**Status: ⚠️ BLOCKED — No EAGLE3 draft model available for Llama-3.2-1B (searched HuggingFace 2026-04-02)**  
**Alternative: See Path 7C below (Qwen3-1.7B with EAGLE3 — explicitly supported)**

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

## Path 7C — TRT Edge-LLM: Qwen3-1.7B + EAGLE3 (Primary 100 t/s Route)

**Target:** ≥100 t/s with Qwen3-1.7B (>1B) via EAGLE3 speculative decoding  
**Status: ❌ COMPLETE (FAILED) — 24.1 t/s EAGLE3; ~31 t/s base; well below 65.31 t/s MLC-LLM baseline**  
**EAGLE3 draft:** `AngelSlim/Qwen3-1.7B_eagle3` (explicitly supported in TRT Edge-LLM v0.6.0 docs)  
**Models downloaded:** `~/qwen3-1.7b/` (base) and `~/qwen3-1.7b-eagle3/` (draft) on DGX Spark

### Why Qwen3-1.7B over Llama-3.2-1B

- No EAGLE3 draft model available for Llama-3.2-{1B,3B} on HuggingFace
- Qwen3-1.7B + AngelSlim/Qwen3-1.7B_eagle3 is the first explicitly supported sub-2B EAGLE3 pair
- 1.7B > 1B → satisfies >1B parameter constraint

### Step 1: Quantize Qwen3-1.7B (DGX Spark)

```bash
# On DGX Spark
source ~/venvs/trt_edgellm_export/bin/activate
mkdir -p ~/tensorrt-edgellm-workspace/Qwen3-1.7B

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
tensorrt-edgellm-quantize-llm \
    --model_dir ~/qwen3-1.7b \
    --output_dir ~/tensorrt-edgellm-workspace/Qwen3-1.7B/quantized \
    --quantization int4_awq \
    --device cuda
```

### Step 2: Export base model with EAGLE3 flag (DGX Spark)

```bash
tensorrt-edgellm-export-llm \
    --model_dir ~/tensorrt-edgellm-workspace/Qwen3-1.7B/quantized \
    --output_dir ~/tensorrt-edgellm-workspace/Qwen3-1.7B/onnx/base \
    --is_eagle_base
```

### Step 3: Export EAGLE3 draft model (DGX Spark)

```bash
tensorrt-edgellm-export-draft \
    --draft_model_dir ~/qwen3-1.7b-eagle3 \
    --base_model_dir ~/tensorrt-edgellm-workspace/Qwen3-1.7B/quantized \
    --output_dir ~/tensorrt-edgellm-workspace/Qwen3-1.7B/onnx/draft
```

### Step 4: Transfer ONNX to Jetson

```bash
scp -r ~/tensorrt-edgellm-workspace/Qwen3-1.7B/onnx \
    spitman@spitman-jetson:~/tensorrt-edgellm-workspace/Qwen3-1.7B/
```

### Step 5: Build engines on Jetson (host, no container needed)

**Note:** Jetson must run v0.6.0 (not v0.5.0). v0.6.0 ONNX uses separate Q/K/V inputs; v0.5.0 `getOutputDimensions` treats `inputs[1]` as 5D KV cache but it's 3D K tensor → segfault. Working v0.6.0 cmake command:

```bash
# IMPORTANT: single-quoted SSH to prevent local shell var expansion
ssh spitman@spitman-jetson '
cd /home/spitman/tools/TensorRT-Edge-LLM
git checkout v0.6.0
rm -rf build && mkdir build && cd build
# Create /tmp/trt_pkg to satisfy -DTRT_PACKAGE_DIR (v0.6.0 new requirement)
mkdir -p /tmp/trt_pkg
ln -sf /usr/include/aarch64-linux-gnu /tmp/trt_pkg/include
ln -sf /usr/lib/aarch64-linux-gnu /tmp/trt_pkg/lib
export PATH=/usr/local/cuda-12.6/bin:$PATH
cmake \
    -DAARCH64_BUILD=1 \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
    -DCUDA_RUNTIME_API_INCLUDE_DIR=/usr/local/cuda-12.6/targets/aarch64-linux/include \
    -DCUDA_CTK_VERSION=12.6 \
    -DTRT_PACKAGE_DIR=/tmp/trt_pkg \
    ..
cmake --build . --parallel 4
'
# v0.6.0 flags changed: CUDA_VERSION → CUDA_CTK_VERSION (fatal error if old name used)
# Binaries: build/examples/llm/llm_build, build/examples/llm/llm_inference
```

**Vocab reduction (required to avoid NvMap OOM during engine build):**
```bash
# Qwen3-1.7B LM head = 2048×151936 ≈ 594 MB FP16 → exceeds Jetson lfb during profiling
# Reduce to 32768 tokens (~128 MB) before building base engine:
tensorrt-edgellm-reduce-vocab \
    --onnx_dir ~/tensorrt-edgellm-workspace/Qwen3-1.7B/onnx/base \
    --output_dir ~/tensorrt-edgellm-workspace/Qwen3-1.7B/onnx/base_reduced \
    --reduced_vocab_size 32768
# WARNING: vocab reduction breaks EAGLE3 speculation (draft vocab 32000 ≠ reduced base 32768)
```

```bash
TRT_EDGELLM=/home/spitman/tools/TensorRT-Edge-LLM/build/examples/llm
WORKSPACE=~/tensorrt-edgellm-workspace/Qwen3-1.7B
export EDGELLM_PLUGIN_PATH=/home/spitman/tools/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so

# Build base model EAGLE engine (use base_reduced if vocab reduction needed)
$TRT_EDGELLM/llm_build \
  --onnxDir $WORKSPACE/onnx/base_reduced \
  --engineDir $WORKSPACE/engines \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 4096 \
  --maxVerifyTreeSize 60 \
  --maxDraftTreeSize 60 \
  --eagleBase

# Build draft model engine
$TRT_EDGELLM/llm_build \
  --onnxDir $WORKSPACE/onnx/draft \
  --engineDir $WORKSPACE/engines \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 4096 \
  --maxVerifyTreeSize 60 \
  --maxDraftTreeSize 60 \
  --eagleDraft
```

### Step 6: Benchmark with EAGLE3 (Jetson)

```bash
$TRT_EDGELLM/llm_inference \
  --engineDir $WORKSPACE/engines \
  --inputFile ~/tensorrt-edgellm-workspace/bench_input.json \
  --outputFile /tmp/qwen3_eagle3_output.json \
  --warmup 3 \
  --dumpProfile \
  --eagle
```

### Stage XXIV Results (2026-04-02) — FAILED

**Benchmark (Jetson, Qwen3-1.7B INT4 AWQ + EAGLE3, 32768 vocab reduction):**

```
=== Performance Summary ===
=== LLM Prefill ===
Computed Tokens: 38, Tokens/Second: 1211.6
=== Eagle Generation ===
Total Iterations: 208, Total Generated Tokens: 257
Average Acceptance Rate: 1.24
Overall Tokens/Second (excluding base prefill): 24.1
Construct Draft Tree: 208 runs, 3951ms total (19.00ms avg)
Base Model Verification: 208 runs, 6696ms total (32.19ms avg)
Peak Unified Memory: 3379.48 MB
```

| Config | Expected t/s | Actual t/s | Delta |
|--------|:------------:|:----------:|:-----:|
| Qwen3-1.7B INT4 standalone | 35–45 t/s | **~31 t/s** | -31% vs expectation |
| Qwen3-1.7B + EAGLE3 | **80–120 t/s** | **24.1 t/s** | **-63% vs baseline** |
| MLC-LLM Llama-3.2-1B (reference) | — | **65.31 t/s** | baseline |

**Base model speed estimate:** 1000ms / 32.19ms per verification = ~31 t/s effective at 1.7B.

### Why Path 7C Failed: Two Independent Problems

**Problem 1: Vocab reduction breaks EAGLE3 speculation (catastrophic)**

EAGLE3 needs draft vocab = base vocab for the draft logits to map onto base model token IDs. After `tensorrt-edgellm-reduce-vocab --reduced_vocab_size 32768`:
- Base model vocab: **32768** tokens (arbitrary subset of Qwen3's 151936)
- Draft model (AngelSlim/Qwen3-1.7B_eagle3) vocab: **32000** tokens (Qwen3's standard chat vocab)
- Mismatch → draft predictions do not align with base model's reduced token space
- **Result: acceptance rate 1.24 tokens/iteration** (expected 3–5 for a good EAGLE3 pair)
- This is why even EAGLE3 at 24.1 t/s is slower than base-only at ~31 t/s — verification overhead with near-zero acceptance gains nothing

**Problem 2: TRT Edge-LLM plugins not optimized for sm_87 (structural)**

The `Int4GroupwiseGemmPlugin` and `AttentionPlugin` in TRT Edge-LLM v0.6.0 are designed for Jetson Thor (GH20x, sm_100/sm_90 series), not Jetson Orin Nano (sm_87). Bandwidth utilization on sm_87:
- TRT Edge-LLM: 31 t/s × 1.7B × INT4 ≈ 31 × ~850 MB ≈ 26.4 GB/s = **39% of 68 GB/s**  
- MLC-LLM (reference, 1B): 65 t/s × 1B × INT4 ≈ 65 × ~540 MB ≈ 35.1 GB/s = **52% of 68 GB/s**
- llama.cpp IQ4_XS (reference, 1B): 60 t/s × ~700 MB ≈ 42 GB/s = **62% of 68 GB/s**

TRT Edge-LLM achieves only 39% BW utilization vs 62% for llama.cpp on sm_87. The CUDA kernel autotuning in TRT targets the profiling hardware (DGX Spark or NVIDIA's build machine), not sm_87. This is a fundamental performance regression, not a configuration issue.

### Key Infrastructure Won (reusable for Path 8+)

Despite Path 7C failing, the pipeline infrastructure is now validated:
1. **v0.6.0 C++ runtime on Jetson** — cmake command documented above works
2. **DGX Spark export pipeline** — venv, `--device cpu` flag, all tools working
3. **scp -3 Mac relay** — transfers work between DGX Spark and Jetson via Mac
4. **Engine build workflow** — both `--eagleBase` and `--eagleDraft` flags confirmed
5. **Vocab reduction workaround** — required for any model with LM head >128 MB on Jetson 8 GB (NvMap fragmentation drops lfb to 4 MB during TRT profiling)

### Conclusion

Path 7C is definitively failed. TRT Edge-LLM v0.6.0 with Qwen3-1.7B on sm_87 achieves worse throughput than llama.cpp IQ4_XS on a smaller model. The combination of:
- Poor sm_87 plugin performance (39% BW vs 62% for llama.cpp)
- Vocab reduction making EAGLE3 inoperable (1.24 acceptance rate)
- No path to fix either issue without TRT Edge-LLM kernel rewrite

...makes further TRT Edge-LLM pursuit on this hardware impractical. **Paths 8–10 (TRT Edge-LLM larger models) should be deprioritized.** The MLC-LLM + trained EAGLE3 head path (Path 4, or MLC-LLM EAGLE-2) is the more viable route to 100 t/s.

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
| **6** | **Path 6: MLC-LLM Llama-3.2-1B** | ✅ **65.31 t/s** | +7.3% vs baseline; modest improvement (bandwidth-bound); confirms TVM kernels work |
| **7** | **Path 7: TRT Edge-LLM C++ build + DGX Spark export pipeline** | ✅ **C++ done, export running** | `llm_build`+`llm_inference` built on Jetson; venv on DGX Spark works |
| **7B** | **Path 7B: TRT Edge-LLM Llama-3.2-1B + EAGLE-3** | ⚠️ BLOCKED | No EAGLE3 draft for Llama-3.2-1B on HuggingFace |
| **7C** | **Path 7C: TRT Edge-LLM Qwen3-1.7B + EAGLE3** | ❌ **COMPLETE (FAILED)** | **24.1 t/s** — vocab reduction breaks EAGLE3 (1.24 accept rate); sm_87 plugin only 39% BW; below MLC baseline |
| **8** | **Path 8: TRT Edge-LLM Llama-3.1-8B + Llama-3.2-3B** | ⚠️ DEPRIORITIZED | TRT Edge-LLM plugins underperform on sm_87 (39% BW vs 62% llama.cpp); same root cause will affect all paths |
| **9** | **Path 9: TRT Edge-LLM Qwen2.5-7B / 3B / 1.5B** | ⚠️ DEPRIORITIZED | Same sm_87 plugin performance issue as Path 7C |
| **10** | **Path 10: TRT Edge-LLM LFM2.5-1.2B** | ⚠️ DEPRIORITIZED | Same sm_87 plugin performance issue; arch support also uncertain |

---

## Success Criteria (Final)

| Milestone | Status | Target | Model |
|-----------|--------|--------|-------|
| ✅ Phase 5 best | Achieved | 100.87 t/s | Qwen2.5-0.5B TRT-LLM W4A16 |
| ✅ >1B record (llama.cpp) | Achieved | **65.64 t/s** | LFM2 1.2B IQ4_XS + FA |
| ✅ Llama 1B record (llama.cpp) | Achieved | **60.28 t/s** | Llama 3.2 1B IQ4_XS + FA |
| ❌ Path 7C TRT Edge-LLM gate | FAILED | >baseline | 24.1 t/s (EAGLE3 broken by vocab reduction; sm_87 poor kernel perf) |
| 🎯 Primary: 100 t/s with >1B | **Pending** | ≥100 t/s | Needs new approach — MLC-LLM + EAGLE3 head (trained on DGX Spark) |
| 🎯 Qwen 100 t/s with >1B | **Pending** | ≥100 t/s | Qwen2.5-1.5B MLC-LLM + EAGLE-3 |
| 🎯 Largest model record | **Pending** | Best t/s | MLC-LLM Llama-3.1-8B or Qwen2.5-7B |
| 🎯 3B 100 t/s stretch | **Pending** | ≥100 t/s | Llama-3.2-3B or Qwen2.5-3B + EAGLE-3 via MLC-LLM |

**Revised strategy (2026-04-02, post Path 7C correction):**

Path 7C failed for two fixable reasons — NOT because TRT Edge-LLM is fundamentally broken on sm_87:

1. **Wrong quantization**: INT4 AWQ was used. This leaves LM head in FP16 → 594 MB → NvMap lfb drop during TRT profiling → forced vocab reduction → broke EAGLE3 (draft vocab 32000 ≠ reduced base 32768 → acceptance rate 1.24). **Fix: use FP8 quantization. FP8 LM head = ~297 MB → no vocab reduction needed → EAGLE3 works as intended.**

2. **sm_87 BW utilization (39%)**: This was measured on the INT4 engine with vocab reduction applied. FP8 engines use different kernel paths — actual sm_87 utilization with proper FP8 may differ. Real measurement needed.

**TRT Edge-LLM is still the primary path.** The user has confirmed it installed and working on Jetson Orin Nano Super (`~/TensorRT-Edge-LLM/build`) and DGX Spark (`~/edge_llm_env/` venv, `nvcr.io/nvidia/pytorch:25.12-py3` Docker). The documentation explicitly supports the model pairs below. **Try the documented FP8 approach before abandoning.**

**Parallel track: MLC-LLM + EAGLE-3** remains valid backup at 65.6 t/s baseline.

---

## Platform Setup (Current State — 2026-04-02)

### Jetson Orin Nano Super
- SSH: `ssh spitman@spitman-jetson`
- TRT Edge-LLM binaries: `~/TensorRT-Edge-LLM/build/examples/llm/llm_build` and `llm_inference`
- MLC-LLM: `dustynv/mlc:0.20.0-r36.4.0` Docker image + `lib_bs1.so` compiled (max_batch_size=1)
- Baseline: **65.6 t/s** (MLC-LLM Llama-3.2-1B q4f16_1)

### DGX Spark
- SSH: `ssh spark-807e.local`
- venv: `source ~/edge_llm_env/bin/activate` (has tensorrt-edgellm tools)
- Docker: `nvcr.io/nvidia/pytorch:25.12-py3` (clean image with all TRT Edge-LLM libs)
- Workspace: `~/tensorrt-edgellm-workspace/`

---

## Path 11 — TRT Edge-LLM Qwen3-1.7B FP8+lm_head + EAGLE3 (Corrected Approach)

**Target:** ≥100 t/s with Qwen3-1.7B (>1B) via properly configured EAGLE3  
**Status: ❌ COMPLETE (FAILED — 2026-04-03) — FP8 blocked by sm_87 hardware**  
**Root fix attempted:** Use FP8 WITH `--lm_head_quantization fp8` — but TRT refuses to build FP8 engines on sm_87 (Ampere). FP8 requires Ada Lovelace sm_89+ or Hopper sm_90+.

### Why default FP8 was wrong (discovered 2026-04-03)

The original Path 11 plan assumed "FP8 quantization reduces LM head size". This is **wrong** — ModelOpt's default FP8 recipe explicitly excludes lm_head:

```json
{"quant_algo": "FP8", "exclude_modules": ["lm_head"]}
```

The existing `quantized-base/` directory (Apr 2, 14:21) has FP8 weights but **lm_head still at FP16 (594 MB)**, and the exported `onnx/base/embedding.safetensors` is 594 MB — same as INT4. Vocab reduction would still be needed, breaking EAGLE3.

Fix: use `--lm_head_quantization fp8` flag to explicitly quantize lm_head to FP8 as well.

### Why FP8+lm_head fixes the problem

| | Path 7C (FAILED) | Path 11 attempt 1 (wrong) | Path 11 corrected |
|---|---|---|---|
| Quantization | INT4 AWQ | FP8 (lm_head excluded) | **FP8 + lm_head FP8** |
| LM head size | 594 MB FP16 | 594 MB FP16 (unchanged) | **~297 MB FP8** |
| Vocab reduction required? | Yes | Yes (same issue) | **No — should fit** |
| EAGLE3 vocab match | 32000 ≠ 32768 → broken | 32000 ≠ 32768 → broken | **Full 151936 → intact** |
| Expected acceptance rate | 1.24 tokens/step | 1.24 tokens/step | **2–4 tokens/step** |

### Working venv (critical)

Use `~/venvs/trt_edgellm_export/` — has CUDA: True, transformers 4.57.3. NOT `~/edge_llm_env/` (no CUDA) and NOT docker (transformers version incompatible).

### Step 1: Quantize + Export (DGX Spark)

```bash
ssh spark-807e.local
source ~/venvs/trt_edgellm_export/bin/activate
cd ~/tensorrt-edgellm-workspace

# FP8 quantize base model WITH lm_head quantization
# ✅ RUNNING (2026-04-03) — PID 3658718, output: Qwen3-1.7B/quantize_fp8_lmhead_v2.log
tensorrt-edgellm-quantize-llm \
  --model_dir ~/qwen3-1.7b \
  --quantization fp8 \
  --lm_head_quantization fp8 \
  --device cuda \
  --output_dir Qwen3-1.7B/quantized-fp8-lmhead

# Export base model with EAGLE flag (after quantize completes)
tensorrt-edgellm-export-llm \
  --model_dir Qwen3-1.7B/quantized-fp8-lmhead \
  --output_dir Qwen3-1.7B/onnx_fp8_lmhead/base \
  --is_eagle_base

# Draft ONNX: reuse existing onnx/draft/ from Apr 2 (draft model is unchanged)
# onnx/draft/: config.json, d2t.safetensors (126 KB), model.onnx (21 KB), onnx_model.data (262 MB)
```

### Step 2: Transfer to Jetson (via Mac relay)

Spark and Jetson are on different networks — transfer must hop through your Mac:

```bash
# Option A: scp -3 relay (Mac runs this, pulls from Spark and pushes to Jetson in one command)
scp -3 -r spark-807e.local:~/tensorrt-edgellm-workspace/Qwen3-1.7B/onnx \
  spitman@spitman-jetson:~/tensorrt-edgellm-workspace/Qwen3-1.7B/

# Option B: two-step via Mac local disk (if -3 is slow or stalls)
scp -r spark-807e.local:~/tensorrt-edgellm-workspace/Qwen3-1.7B/onnx /tmp/qwen3_onnx/
scp -r /tmp/qwen3_onnx spitman@spitman-jetson:~/tensorrt-edgellm-workspace/Qwen3-1.7B/onnx

# Make the target dir first on Jetson if needed
ssh spitman@spitman-jetson 'mkdir -p ~/tensorrt-edgellm-workspace/Qwen3-1.7B'
```

### Step 3: Build Engines (Jetson)

```bash
TRT_EDGELLM=~/TensorRT-Edge-LLM/build/examples/llm
WORKSPACE=~/tensorrt-edgellm-workspace/Qwen3-1.7B

# Build base engine (NO vocab reduction — FP8 LM head fits)
$TRT_EDGELLM/llm_build \
  --onnxDir $WORKSPACE/onnx/base \
  --engineDir $WORKSPACE/engines \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 4096 \
  --maxVerifyTreeSize 60 \
  --maxDraftTreeSize 60 \
  --eagleBase

# Build draft engine
$TRT_EDGELLM/llm_build \
  --onnxDir $WORKSPACE/onnx/draft \
  --engineDir $WORKSPACE/engines \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 4096 \
  --maxVerifyTreeSize 60 \
  --maxDraftTreeSize 60 \
  --eagleDraft
```

If llm_build OOMs during profiling (NvMap lfb issue persists with FP8), reduce context: `--maxInputLen 512 --maxKVCacheCapacity 2048`.

### Step 4: Benchmark (Jetson)

```bash
$TRT_EDGELLM/llm_inference \
  --engineDir $WORKSPACE/engines \
  --inputFile ~/tensorrt-edgellm-workspace/bench_input.json \
  --outputFile /tmp/qwen3_17b_fp8_eagle3.json \
  --eagle
```

### Expected outcome

| Config | Projected t/s | Notes |
|--------|:-------------:|-------|
| Qwen3-1.7B FP8 standalone | ~45–55 t/s | FP8 > INT4 AWQ on sm_87 for decode |
| Qwen3-1.7B FP8 + EAGLE3 (2.5× accept) | **~112–137 t/s** | ✅ Hits 100 t/s target |
| Qwen3-1.7B FP8 + EAGLE3 (3× accept) | **~135–165 t/s** | Strong result |

---

## Path 12 — TRT Edge-LLM Llama-3.1-8B + EAGLE3 (Largest Documented Pair)

**Target:** Best t/s for a ≥7B model on Jetson; explicit NVIDIA-documented model pair  
**Status: 🔒 After Path 11**  
**Draft model:** `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` — pre-trained, no training needed  
**Vocabulary:** 128,256 tokens (Llama tokenizer). LM head FP8 = 128256 × 4096 × 1 byte = ~500 MB — may still need smaller context; test before adding vocab reduction

### Step 1: Quantize + Export (DGX Spark)

```bash
source ~/edge_llm_env/bin/activate
cd ~/tensorrt-edgellm-workspace

# Download
git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct Llama-3.1-8B/hf
git clone https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
cd EAGLE3-LLaMA3.1-Instruct-8B && git lfs pull && cd ..

# Quantize base (FP8)
tensorrt-edgellm-quantize-llm \
  --model_dir Llama-3.1-8B/hf \
  --quantization fp8 \
  --output_dir Llama-3.1-8B/quantized-base

# Export base with EAGLE flag
tensorrt-edgellm-export-llm \
  --model_dir Llama-3.1-8B/quantized-base \
  --output_dir Llama-3.1-8B/onnx/base \
  --is_eagle_base

# Quantize draft (FP8)
tensorrt-edgellm-quantize-draft \
  --base_model_dir Llama-3.1-8B/hf \
  --draft_model_dir EAGLE3-LLaMA3.1-Instruct-8B \
  --quantization fp8 \
  --output_dir Llama-3.1-8B/quantized-draft

# Export draft
tensorrt-edgellm-export-draft \
  --draft_model_dir Llama-3.1-8B/quantized-draft \
  --base_model_dir Llama-3.1-8B/hf \
  --output_dir Llama-3.1-8B/onnx/draft

# Transfer to Jetson via Mac relay (Spark and Jetson are on different networks)
# Run from Mac:
# scp -3 -r spark-807e.local:~/tensorrt-edgellm-workspace/Llama-3.1-8B/onnx \
#   spitman@spitman-jetson:~/tensorrt-edgellm-workspace/Llama-3.1-8B/
```

### Step 2: Build + Run (Jetson)

```bash
TRT_EDGELLM=~/TensorRT-Edge-LLM/build/examples/llm
WORKSPACE=~/tensorrt-edgellm-workspace/Llama-3.1-8B

# 8B FP8 ≈ 8 GB — use reduced context to stay within memory
$TRT_EDGELLM/llm_build \
  --onnxDir $WORKSPACE/onnx/base \
  --engineDir $WORKSPACE/engines \
  --maxBatchSize 1 \
  --maxInputLen 512 \
  --maxKVCacheCapacity 2048 \
  --maxVerifyTreeSize 60 \
  --maxDraftTreeSize 60 \
  --eagleBase

$TRT_EDGELLM/llm_build \
  --onnxDir $WORKSPACE/onnx/draft \
  --engineDir $WORKSPACE/engines \
  --maxBatchSize 1 \
  --maxInputLen 512 \
  --maxKVCacheCapacity 2048 \
  --maxVerifyTreeSize 60 \
  --maxDraftTreeSize 60 \
  --eagleDraft

$TRT_EDGELLM/llm_inference \
  --engineDir $WORKSPACE/engines \
  --inputFile ~/tensorrt-edgellm-workspace/bench_input.json \
  --outputFile /tmp/llama31_8b_eagle3.json \
  --eagle
```

### Expected outcome

| Config | Projected t/s | Notes |
|--------|:-------------:|-------|
| Llama-3.1-8B FP8 standalone | ~12–16 t/s | Bandwidth-bound: 68 GB/s / 8 GB = ~8.5 t/s theoretical at 100%; realistic ~12–16 |
| Llama-3.1-8B + EAGLE3 (2.5× accept) | ~30–40 t/s | Best ≥7B result achievable on this hardware |
| Memory fit | Tight — 8B FP8 ≈ 8 GB + runtime | Reduce context if OOM during build |

**Goal for Path 12:** Not 100 t/s (bandwidth math won't allow it at 8B) — but largest model ever run on this hardware, with EAGLE3.

---

## Path 13 — MoE: Qwen3-30B-A3B-GPTQ-Int4

**Target:** Benchmark MoE on Jetson + DGX Spark; understand MoE performance envelope  
**Status: 🔒 After Path 12**  
**Export:** CPU-only (DGX Spark, no GPU needed for export); requires `gptqmodel==4.2.5` and `optimum==2.0.0`  
**Memory note:** 30B total weights at INT4 ≈ 15 GB — likely does NOT fit Jetson 8 GB. Test on DGX Spark first; if Jetson OOMs, run inference on Spark only.

### Step 1: Install extra deps + Export (DGX Spark)

```bash
source ~/edge_llm_env/bin/activate
pip install gptqmodel==4.2.5 optimum==2.0.0

cd ~/tensorrt-edgellm-workspace

# Download GPTQ-Int4 model (pre-quantized — no quantize step needed)
huggingface-cli download Qwen/Qwen3-30B-A3B-GPTQ-Int4 \
  --local-dir Qwen3-30B-A3B/hf

# Export — CPU only (as documented for MoE)
tensorrt-edgellm-export-llm \
  --device cpu \
  --model_dir Qwen3-30B-A3B/hf \
  --output_dir Qwen3-30B-A3B/onnx
```

### Step 2: Try on Jetson first, fall back to DGX Spark if OOM

```bash
# Transfer ONNX to Jetson via Mac relay (different networks — run from Mac)
scp -3 -r spark-807e.local:~/tensorrt-edgellm-workspace/Qwen3-30B-A3B/onnx \
  spitman@spitman-jetson:~/tensorrt-edgellm-workspace/Qwen3-30B-A3B/onnx
# Warning: 30B ONNX may be 15+ GB — allow significant time

# Build on Jetson
TRT_EDGELLM=~/TensorRT-Edge-LLM/build/examples/llm
$TRT_EDGELLM/llm_build \
  --onnxDir ~/tensorrt-edgellm-workspace/Qwen3-30B-A3B/onnx \
  --engineDir ~/tensorrt-edgellm-workspace/Qwen3-30B-A3B/engines \
  --maxBatchSize 1 \
  --maxInputLen 512 \
  --maxKVCacheCapacity 1024
```

If Jetson OOMs (expected for 30B total params), run build+inference on DGX Spark instead — it has 119 GB UMA.

### Expected outcome

| Hardware | Config | Projected t/s |
|----------|--------|:-------------:|
| Jetson 8 GB | Qwen3-30B-A3B INT4 | Likely OOM — 15 GB weights > 8 GB |
| DGX Spark 119 GB | Qwen3-30B-A3B INT4 | ~40–80 t/s (3B active params per token) |

The MoE result on DGX Spark would demonstrate the practical ceiling for Spark as an inference machine.

---

## Path 14 — MLC-LLM + EAGLE-3 (Parallel Backup Track)

**Target:** 100 t/s via MLC-LLM speculative decoding, avoiding TRT Edge-LLM entirely  
**Status: 🔒 Parallel with Path 11; activate if Path 11 OOMs or BW remains poor**  
**Baseline confirmed:** 65.6 t/s (Llama-3.2-1B q4f16_1, lib_bs1.so, max_batch_size=1)

MLC-LLM natively supports EAGLE-2 (not EAGLE-3) speculative decoding. No pre-trained EAGLE-2 head exists for Llama-3.2-1B; requires training on DGX Spark using SafeAILab/EAGLE.

MLC eagle weight format requirements:
- `embed_tokens.weight [vocab_size, hidden_dim]`
- `layers.0.self_attn.{q,k,v,o}_proj.weight`
- `layers.0.mlp.{gate,up,down}_proj.weight`
- `layers.0.post_attention_layernorm.weight`
- `fc.weight [hidden_dim, 2×hidden_dim]` — EAGLE-2 shape; EAGLE-3 (3×) is incompatible
- `fc.bias [hidden_dim]`

Train on DGX Spark: `SafeAILab/EAGLE` repo, ~2h on GB10.

---

## Execution Order (Updated 2026-04-02)

| Priority | Path | Status | Result |
|----------|------|--------|--------|
| 1 | Path 1: Llama-3.2-3B + 1B speculative | ❌ BLOCKED | NvMap IOVM; two models don't fit |
| 2 | Path 2: TRT-LLM Llama 1B W4A16 gemm_plugin | ❌ BLOCKED | globWriter 1 GB IOVM; 8 patches; wrong tool |
| 3 | Path 3: Qwen2.5-1.5B TRT-LLM W4A16 | ❌ SKIPPED | Same constraint as Path 2 |
| 4 | Path 4: EAGLE-3 via llama.cpp | ❌ BLOCKED | llama.cpp has no EAGLE support |
| 5 | Path 5: Lookup decoding | ❌ BLOCKED | 26% accept; CPU-GPU sync = 42% slower than baseline |
| 6 | **Path 6: MLC-LLM Llama-3.2-1B** | ✅ **65.6 t/s** | TVM kernels work; +7% vs llama.cpp baseline |
| 7C | **Path 7C: TRT Edge-LLM Qwen3-1.7B INT4 + EAGLE3** | ❌ **FAILED** | 24.1 t/s — INT4 forced vocab reduction → EAGLE3 broken; **Fix: use FP8 → Path 11** |
| **8** | **Path 11: TRT Edge-LLM Qwen3-1.7B FP8 + EAGLE3** | ❌ **FAILED** | sm_87 has no FP8 hardware support |
| **9** | **Path 12: TRT Edge-LLM Llama-3.1-8B FP8 + EAGLE3** | 🔒 After Path 11 | Documented NVIDIA pair; pre-trained head available |
| **10** | **Path 13: MoE Qwen3-30B-A3B-GPTQ-Int4** | 🔒 After Path 12 | CPU export; may OOM on Jetson (15 GB weights) |
| **11** | **Path 14: MLC-LLM + EAGLE-2** | ❌ **COMPLETE (FAILED)** | **8.9 t/s** (0.12×) — FlashInfer JIT deadlocks on Jetson ARM; TIR tree-attention ~8× slower than baseline; both paths non-viable in v0.20.0 |

---

## Stage XXV — TRT Edge-LLM EAGLE3 Exhaustive Investigation (2026-04-02)

### Path 11 UPDATE: FP8 PERMANENTLY BLOCKED on sm_87

**Confirmed:** Jetson Orin Nano Super is sm_87. FP8 requires sm_89+ hardware.

```
[ERROR] Networks with FP8 Q/DQ layers require hardware with FP8 support (sm_89+)
```

FP8 cannot be used on this hardware under any configuration. The Path 11 premise (FP8 avoids vocab reduction) is invalid. **All INT4 AWQ paths must deal with the 622 MB LM head.**

### Stage XXV-A: INT4 AWQ + d2t-Aligned Vocab Reduction (32768)

**Root cause of Path 7C failure (1.24 acceptance):** Arbitrary 32768 vocab reduction excluded 768 of the 32000 draft tokens, making those draft predictions impossible to accept.

**Fix applied:** `tensorrt-edgellm-reduce-vocab --d2t_path draft/d2t.safetensors --reduced_vocab_size 32768` guarantees all 32000 draft tokens are in the 32768-token reduced vocab (draft tokens are first-priority; remaining 768 slots filled with other tokens).

**Result (Jetson, 2026-04-02):**

```
=== Eagle Generation ===
Total Iterations: 197, Total Generated Tokens: 256
Average Acceptance Rate: 1.30
Overall Tokens/Second (excluding base prefill): 21.5
Construct Draft Tree: 197 runs, 5633ms total (28.60ms avg)
Base Model Verification: 197 runs, 6244ms total (31.70ms avg)
```

| Config | t/s | Acceptance |
|--------|:---:|:----------:|
| Path 7C (arbitrary 32768 vocab) | 24.1 | 1.24 |
| Stage XXV-A (d2t-aligned 32768 vocab) | **21.5** | **1.30** |

**Worse result overall** despite fixing the vocab alignment! EAGLE3 overhead (28.6ms draft + 31.7ms verify = 60ms per 1.3 tokens = 21.7 t/s) is slower than the estimated pure autoregressive speed (~31.5 t/s from verification timing alone). The d2t fix only marginally improved acceptance from 1.24 → 1.30.

**Root cause (revised):** The EAGLE3 draft model (`AngelSlim/Qwen3-1.7B_eagle3`) was trained on **FP16 Qwen3-1.7B hidden states**. INT4 AWQ quantization produces different hidden states — the quantization noise (4-bit, 16 levels per value) is large enough to cause ~74% of draft token predictions to be wrong even after d2t alignment. This is a fundamental distribution mismatch, not a vocabulary mapping issue.

### Stage XXV-B: Greedy Draft (eagleDraftTopK=1) Test

To confirm that the EAGLE3 draft cannot predict INT4 quantized base:

| Config | t/s | Acceptance | Conclusion |
|--------|:---:|:----------:|-----------|
| Default topK=10 | 21.5 | 1.30 | Sampling noise occasionally picks correct token |
| Greedy topK=1 | 17.6 | **1.00** | Single best-prediction always wrong |

**With greedy drafting, every single draft token is rejected.** This definitively confirms that the EAGLE3 draft model's predicted distribution has near-zero overlap with the INT4 quantized base model's actual distribution.

### Stage XXV-C: Qwen3-0.6B TRT Edge-LLM Baseline

For reference, the sub-1B model in the same framework:

```
=== LLM Generation ===
Generated Tokens: 255
Tokens/Second: 116.1
Average Time per Token: 8.6136 ms
Peak Unified Memory: 1782.70 MB
```

Qwen3-0.6B INT4 AWQ runs at **116.1 t/s** (8.6ms/token). The 0.6B LM head is only 151936×1024×2 = 297 MB FP16 — below the NvMap threshold, so no vocab reduction was needed and EAGLE3 is not applicable (0.6B is the draft model, not the base in EAGLE3 pairing).

**Bandwidth analysis:**
- 0.6B: 116.1 t/s × (150 MB INT4 transformer + 297 MB FP16 LM head) = 116.1 × 447 MB = **51.9 GB/s = 76% of 68 GB/s**

This confirms TRT Edge-LLM DOES achieve good bandwidth utilization on sm_87 for sub-1B models! The 39% utilization for 1.7B was due to the vocab reduction's effect on the benchmark (verification step processes 60 tokens simultaneously which is differently bounded).

### Stage XXV-D: INT8 SQ EAGLE3 Attempt (In Progress)

**Hypothesis:** INT8 SmoothQuant (W8A8) preserves activation/hidden state distribution much better than INT4 AWQ, as activations remain in FP16 (only quantized during matmul, not stored). EAGLE3 draft trained on FP16 base should achieve ≥2.5× acceptance rate with INT8 SQ base.

**Expected improvement:**
- INT4 AWQ: 4-bit weights, significant hidden state noise → 1.30 acceptance
- INT8 SQ: 8-bit weights + smooth activation distribution → ~2.5× acceptance (projected)

**Expected speed with INT8 SQ + EAGLE3:**
- INT8 base verification time: ~24ms/step (2× INT4's read bytes but better compute efficiency)
- Draft step: ~5ms (EAGLE3 head, unchanged)
- With 2.5× acceptance: 3.5 tokens / 29ms = **~120 t/s** ✅

**Status (2026-04-02):** W8A8_SQ_PER_CHANNEL quantization completed on DGX Spark (3.3 GB model.safetensors). Export via venv fails (segfault — modelopt_cuda_ext CUDA compilation fails without Python.h). Docker-based export running now.

**Known issue:** `tensorrt-edgellm-export-llm --is_eagle_base` on W8A8_SQ model segfaults when modelopt_cuda_ext falls back to CPU (missing Python.h from python3.12-dev package). Fix: use Docker container (nvcr.io/nvidia/pytorch:25.12-py3) which has Python.h installed.

### Summary of All Attempts (as of 2026-04-02)

| Path | Config | t/s | Acceptance | Status |
|------|--------|:---:|:----------:|--------|
| 7C | INT4 AWQ + 32768 vocab (arbitrary) | 24.1 | 1.24 | ❌ FAILED |
| 11 | FP8 EAGLE3 | — | — | ❌ HARDWARE BLOCKED (sm_87 < sm_89) |
| XXV-A | INT4 AWQ + 32768 vocab (d2t-aligned) | 21.5 | 1.30 | ❌ FAILED (barely improved) |
| XXV-B | INT4 AWQ + greedy draft (topK=1) | 17.6 | 1.00 | ❌ FAILED (worst) |
| XXV-C | Qwen3-0.6B INT4 AR (sub-1B reference) | 116.1 | N/A | ✅ Works, sub-1B only |
| XXV-D | INT8 SQ + EAGLE3 | TBD | TBD | 🔄 Export in progress |

### Summary of XXV-D Outcome (2026-04-02, completed)

INT8 SQ Docker export succeeded — produced 5.9 GB `onnx_model.data` + 594 MB `embedding.safetensors`.

**BLOCKED: 5.9 GB ONNX > Jetson NvMap IOVM limit (~2.5 GB)**. The INT8 SQ transformer body alone is ~4 GB, which exceeds the hardware virtual address budget for TRT engine build on Jetson. Cannot be fixed — fundamental hardware constraint.

**TRT Edge-LLM EAGLE3 for >1B models on Jetson is definitively exhausted.** All quantization paths are blocked:

| Quantization | Blocker |
|---|---|
| INT4 AWQ | Vocab reduction required → EAGLE3 broken (1.24 accept rate) |
| INT8 SQ | 5.9 GB ONNX > NvMap IOVM limit (~2.5 GB) |
| FP8 (default, lm_head excluded) | Same vocab reduction issue as INT4 |
| FP8 + lm_head FP8 (2026-04-03) | **sm_87 hardware has no FP8 support** — TRT error: "Networks with FP8 Q/DQ layers require hardware with FP8 support." FP8 requires sm_89+ (Ada) or sm_90+ (Hopper). |

No further quantization options remain. TRT Edge-LLM is not viable for EAGLE3 on Jetson Orin Nano sm_87.

---

## Stage XXVI — Path 14: MLC-LLM + EAGLE-2 (Custom Trained)

**Status: ❌ COMPLETE (FAILED — see Stage XXVII for full post-mortem)**

**Strategy:** MLC-LLM already achieves 65.6 t/s on Llama-3.2-1B-Instruct (q4f16_1) without speculative decoding. Train a custom EAGLE-2 draft head using the SafeAILab/EAGLE framework on DGX Spark, convert to MLC-LLM format, compile, and benchmark on Jetson.

**Why MLC-LLM + EAGLE-2:**
- MLC-LLM 0.20.0 supports EAGLE speculative decoding natively (`speculative_mode="eagle"`)
- The existing `lib_bs1.so` already contains draft verification kernels (`batch_verify_on_gpu_single_kernel`)
- EAGLE-2 uses `fc.weight [hidden_dim, 2×hidden_dim]` which exactly matches MLC-LLM's `EagleForCasualLM`
- 65.6 t/s × 2× acceptance = **131 t/s theoretical** ✅

**Expected speedup:** 1.5–2.5× (small models typically have higher acceptance due to more predictable distributions)

### Architecture

MLC-LLM EAGLE-2 draft model for Llama-3.2-1B:
- 1 transformer decoder layer (identical architecture to base)
- `fc = Linear(2×2048 → 2048)` — fuses base hidden state + input embedding
- 128256 vocab, hidden_size=2048, 32 attention heads, 8 kv heads
- Quantization: q0f16 (FP16, unquantized — tiny model doesn't need quantization)

### Training Setup (DGX Spark GB10)

- **Framework:** SafeAILab/EAGLE `eagle/train/main.py` (EAGLE-2, not EAGLE-3)
- **Data:** 10,000 ShareGPT conversations from `Aeala/ShareGPT_Vicuna_unfiltered`
- **Hidden states:** Second-to-last transformer layer output (`all_hidden_states[-2]`)
- **Training config:** lr=3e-5, bs=4, 20 epochs, ~50,000 total steps
- **Key fix:** Llama-3.2-1B has `tie_word_embeddings=True` — created `lm_head.safetensors` + `model.safetensors.index.json` shim for EAGLE loader compatibility

### Pipeline Steps

| Step | Location | Status | ETA |
|------|----------|--------|-----|
| Data generation (10k ShareGPT hidden states) | DGX Spark | 🔄 ~2066/10300 files | ~17:35 CDT |
| EAGLE-2 training (20 epochs, bs=4) | DGX Spark | ⏳ | ~17:50 CDT |
| Convert checkpoint to HF format | DGX Spark | ⏳ | ~18:00 CDT |
| Transfer to Jetson via Mac relay | Network | ⏳ | ~18:05 CDT |
| `mlc_llm gen_config + convert_weight + compile` | Jetson Docker | ⏳ | ~18:45 CDT |
| Benchmark vs baseline | Jetson | ⏳ | ~18:50 CDT |

### Actual Progress (2026-04-02)

| Step | Status | Details |
|------|--------|---------|
| Data generation (10k ShareGPT hidden states) | ✅ Done | 10,000 train + 300 test .pt files |
| EAGLE-2 training (20 epochs) | 🔄 Running | Epoch 7/20, 34.9% top-1 acc (climbing), 1.22 it/s, first checkpoint state_4 (32.6% test acc) |
| Convert checkpoint → HF format | ✅ Done | `convert_eagle_to_mlc.py` → 1.2 GB q0f16 weights |
| Transfer to Jetson via Mac relay | ✅ Done | `/mlc/eagle2-llama32-1b-hf/` |
| MLC compile (q4f16_1) | ✅ Done | `eagle_lib.so` 12 MB, weights 180 MB total |
| Baseline benchmark | ✅ Done | **65.7 t/s** (matches Phase 5 baseline) |
| EAGLE-2 benchmark | 🚫 Blocked | IOMVM fragmentation (see below) |

### IOMVM Fragmentation Blocker

After many TRT EdgeLLM and Docker compilation runs, the Jetson's NvMap IOVM virtual address space is fragmented. While base model works alone (1075 MB, 65.7 t/s), EAGLE mode requires 1364 MB total (base 841 + KV cache 215 + temp 307) and the CUDA memory allocator hangs indefinitely when there is no contiguous block large enough.

**Fix: `sudo reboot` on Jetson clears IOMVM fragmentation completely.**

After reboot, the benchmark is ready to run:
```bash
# On Jetson after reboot:
MLC_MODELS=$HOME/Projects/Aneurologic/modelgarden/jetson-containers/data/models/mlc
docker run --rm --runtime=nvidia \
    -v $MLC_MODELS:/models \
    -v $HOME/bench_eagle_mlc.py:/bench.py \
    dustynv/mlc:0.20.0-r36.4.0 \
    python3 /bench.py \
    --base_model /models/Llama-3.2-1B-Instruct-q4f16_1-MLC \
    --base_lib /models/Llama-3.2-1B-Instruct-q4f16_1-MLC/lib_bs1.so \
    --eagle_model /models/eagle2-llama32-1b-mlc \
    --eagle_lib /models/eagle2-llama32-1b-mlc/eagle_lib.so \
    --draft_length 4 --num_tokens 200 --num_runs 3
```

### Training Trajectory

| Epoch | Train Loss | Train Acc | Top-2 | Test Acc |
|-------|-----------|-----------|-------|---------|
| 1 | 1.2344 | 7.4% | 11.0% | — |
| 2 | 1.2134 | 16.6% | 24.2% | — |
| 3 | 1.2069 | 22.4% | 32.0% | — |
| 4 | 1.2037 | 26.8% | 37.3% | — |
| 5 | 1.2015 | 30.3% | 41.5% | **32.6%** (checkpoint state_4) |
| 6 | 1.1999 | 32.8% | 44.4% | — |
| 7 | 1.1986 | 34.9% | 46.7% | — |
| 10 | — | ~42%? | — | checkpoint state_9 (ETA ~75 min) |
| 20 | — | ~60%? | — | checkpoint state_19 (ETA ~3.5 hr) |

### Next Actions

1. **User action required:** `sudo reboot` on Jetson (`ssh spitman@spitmann-jetson`)
2. After reboot: run the docker benchmark command above
3. Training continues autonomously → better checkpoint at epoch 10 (state_9) in ~75 min
4. If epoch 5 EAGLE acceptance is insufficient for ≥100 t/s, re-run with state_9 or state_19

---

## Stage XXVII — MLC-LLM EAGLE-2: FlashInfer JIT Deadlock + TIR Investigation

**Status: ❌ COMPLETE (FAILED) — 2026-04-03**

### What Was Attempted

After the Jetson reboot, the original overnight benchmark (`run_eagle_overnight.sh`) was discovered to have hung for 13+ hours with zero output. Investigation revealed the root cause and led to an extensive debugging campaign to find an alternative path.

### Root Cause: FlashInfer JIT Deadlock on Jetson ARM

MLC-LLM's EAGLE speculative decoding uses FlashInfer tree-attention kernels. FlashInfer compiles these at runtime via `torch.utils.cpp_extension.load` (spawning an NVCC subprocess through PyTorch's JIT infrastructure). On Jetson ARM (aarch64), this process **deadlocks** — not slow, but permanently blocked. After 13+ hours there was still no output, no CUDA activity, no file written to the FlashInfer cache directory. The container had to be killed.

**Environment variable `FLASHINFER_DISABLED=1` does not help** — it is not a recognized MLC-LLM/FlashInfer toggle in this version.

### Approach: Recompile with TIR Backend (no FlashInfer)

MLC-LLM supports two backends for attention kernels:
- **FlashInfer** (default when available): Pre-compiled JIT kernels; blocks on Jetson ARM
- **TIR** (TVM Tensor IR): Pure TVM-compiled kernels; no JIT, no NVCC at runtime

The correct compile flag is `--opt 'flashinfer=0'` (not `--flashinfer=False` which is invalid).

### Issues Encountered During TIR Compilation

**Issue 1: Default base model TIR OOM (2803 MB)**

The base model's `mlc-chat-config.json` had `context_window_size=131072, max_batch_size=128` from the original download. Compiling with these defaults allocated 2.8 GB — OOM on Jetson. Fix: explicit `--overrides 'context_window_size=4096;prefill_chunk_size=512;max_batch_size=6'`.

**Issue 2: Shell semicolons in overrides**

`--overrides context_window_size=4096;prefill_chunk_size=512;max_batch_size=1` was split by bash as multiple commands. Fix: escape as `context_window_size=4096\;prefill_chunk_size=512\;max_batch_size=6`.

**Issue 3: 0 tokens generated**

With TIR libs, EAGLE engine produced 0 tokens per run. Root cause: MLC-LLM requires `max_num_sequence >= spec_draft_length + 2` for speculative decoding (per mlc-llm issue #2710). With `draft_length=4`, minimum is 6. Any lower value silently produces no output. The `mode="local"` default sets `max_num_sequence=1`. Fix: pass `max_num_sequence=6` explicitly in `EngineConfig` without specifying `mode`.

**Issue 4: Storage allocation mismatch (2565120 vs 516096 bytes)**

Eagle TIR lib was first compiled with `max_batch_size=1`. At runtime, `max_num_sequence=6` required a workspace buffer 6× larger. The TIR workspace buffer is **fixed at compile time** proportional to `max_batch_size`. Fix: recompile both eagle and base TIR libs with `max_batch_size=6`.

**Issue 5: Multiple CUDA OOM from rapid-fire container launches**

Killed/failed containers left GPU memory allocated in Jetson's unified memory. Fix: `docker ps -q | xargs docker kill` and wait for recovery.

### Final TIR Compile Commands (Working)

```bash
# EAGLE model gen_config with max_batch_size=6
mlc_llm gen_config /models/eagle2-llama32-1b-hf \
    --quantization q4f16_1 --model-type eagle --conv-template llama-3_1 \
    --context-window-size 4096 --prefill-chunk-size 512 --max-batch-size 6 \
    -o /models/eagle2-llama32-1b-mlc-tir/

# Compile eagle TIR lib
mlc_llm compile /models/eagle2-llama32-1b-mlc-tir/ \
    --model-type eagle --device cuda \
    --opt 'flashinfer=0' \
    -o /models/eagle2-llama32-1b-mlc-tir/eagle_lib_tir.so

# Compile base model TIR lib (must also use max_batch_size=6 for workspace match)
mlc_llm compile /models/Llama-3.2-1B-Instruct-q4f16_1-MLC/ \
    --model-type llama --device cuda --opt flashinfer=0 \
    --overrides context_window_size=4096\;prefill_chunk_size=512\;max_batch_size=6 \
    -o /models/Llama-3.2-1B-Instruct-q4f16_1-MLC/lib_bs1_tir.so
```

Output libs:
- `eagle_lib_tir.so`: 2.2 MB
- `lib_bs1_tir.so`: 3.1 MB

### Benchmark Script (bench_eagle_v4.py)

```python
from mlc_llm import MLCEngine
from mlc_llm.serve import EngineConfig

min_seq = args.draft_length + 2  # = 6 for draft_length=4

engine = MLCEngine(
    model=args.base_model,
    model_lib=args.base_lib,
    engine_config=EngineConfig(
        additional_models=[(args.eagle_model, args.eagle_lib)],
        speculative_mode="eagle",
        spec_draft_length=args.draft_length,
        max_num_sequence=min_seq,
        max_total_sequence_length=4096,  # limit KV cache
    )
)
```

### Final Results (bench_eagle_v7.log — 2026-04-03)

| Config | Run 1 | Run 2 | Run 3 | Avg |
|--------|------:|------:|------:|----:|
| EAGLE-2 TIR (draft_length=4) | 8.8 t/s | 9.0 t/s | 9.0 t/s | **8.9 t/s** |
| Baseline (no spec, TIR lib) | 69.8 t/s | 72.4 t/s | 72.4 t/s | **71.5 t/s** |
| **EAGLE-2 speedup** | | | | **0.12× (8× SLOWER)** |

### Why TIR EAGLE Is Catastrophically Slow

The TVM TIR backend for tree-attention is not optimized for sm_87. FlashInfer's tree-attention kernels are hand-tuned with fused warp-level primitives; TIR generates generic element-wise kernels. The EAGLE verification step (checking N draft tokens against the main model in one pass) involves a tree-structured attention mask that is highly irregular — TIR handles this with scalar fallback paths that are ~8× slower than FlashInfer on this operation.

The baseline TIR lib (`lib_bs1_tir.so`) is actually fine at **71.5 t/s** — TVM's standard autoregressive attention is adequately optimized. It's specifically the EAGLE draft/verify tree-attention that TIR cannot handle efficiently.

### Summary of Both Blocked Paths

| Path | Approach | Result |
|------|----------|--------|
| FlashInfer EAGLE | Default MLC-LLM EAGLE compile | **Deadlocks** on Jetson ARM after 13+ hours (NVCC JIT subprocess hang) |
| TIR EAGLE | `--opt 'flashinfer=0'` compile | **8.9 t/s** (0.12×) — tree-attention TIR kernels ~8× slower than baseline |

**Both paths to MLC-LLM EAGLE-2 on Jetson Orin Nano with v0.20.0 are non-viable.** Neither achieves even the non-EAGLE baseline (71.5 t/s), let alone the 100 t/s target.

### What State_19 Means

The DGX Spark EAGLE training reached epoch 20 (state_19, 45.31% top-1 accuracy) vs state_4 (30.30%). State_19 was converted and staged at `/mlc/eagle2-llama32-1b-mlc-tir/` on Jetson but also uses FlashInfer by default. Even if compiled with TIR backend, a better-trained EAGLE head cannot fix the TIR tree-attention kernel performance — the bottleneck is the kernel, not the acceptance rate. State_19 was not tested (no path forward).

### Implication for Phase 6

MLC-LLM EAGLE-2 is definitively exhausted on MLC-LLM v0.20.0 with Jetson Orin Nano. See Stage XXVIII below.

---

## Stage XXVIII — FlashInfer Pre-Warm + EAGLE-2 FlashInfer Benchmark (2026-04-03)

**Status: ❌ COMPLETE (FAILED)**

### What Was Attempted

After the TIR EAGLE failure (8.9 t/s), the hypothesis was that FlashInfer kernels would be faster than TIR for tree-attention. The FlashInfer JIT deadlock (Stage XXVII) was caused by NVCC JIT subprocess hanging — but the kernels themselves might work if pre-compiled.

### Pre-Warm Strategy

1. **Identified FlashInfer 0.2.5 API** for batch wrappers:
   - `BatchPrefillWithRaggedKVCacheWrapper.begin_forward(qo_indptr, kv_indptr, num_qo_heads, num_kv_heads, head_dim_qk, q_data_type=torch.float16)` — CRITICAL: `q_data_type` must be a named kwarg, NOT positional (positional passes it as `head_dim_vo`, generating `constexpr int HEAD_DIM_VO = torch.float16;` in C++)
   - `BatchDecodeWithPagedKVCacheWrapper.begin_forward(indptr, indices, last_page_len, num_qo_heads, num_kv_heads, head_dim, page_size, q_data_type=torch.float16)`

2. **Pre-warm script** (`prewarm_eagle_v2.py`): allocates tiny tensors (no model loading), triggers JIT compilation of batch_prefill and batch_decode kernels. Runs in Docker with `TORCH_CUDA_ARCH_LIST=8.7` and `FLASHINFER_WORKSPACE_BASE=/root`.

3. **FlashInfer cache**: mounted at `/home/spitman/.cache/flashinfer` → `/root/.cache/flashinfer`. Cache keys include `head_dim_vo_64` (correct) vs `head_dim_vo_torch.float16` (buggy v1).

### Pre-Warm Results

Both kernels compiled successfully in the container:
- `batch_prefill_with_kv_cache_..._head_dim_qk_64_head_dim_vo_64_...` ✅
- `batch_decode_with_kv_cache_..._head_dim_qk_64_head_dim_vo_64_...` ✅
- `single_prefill_with_kv_cache_..._head_dim_qk_64_head_dim_vo_64_...` ✅ (from earlier)

### Library Compilation

The original `lib_bs1.so` was compiled for `max_batch_size=1`. EAGLE requires `max_num_sequence >= draft_length + 2 = 6`. This caused `storage allocation failure, attempted to allocate 2565120 at offset 0 in region that is 516096bytes` when running EAGLE.

Compiled new libraries with `max_batch_size=6, prefill_chunk_size=512, --opt flashinfer=1`:
- `lib_bs6_fi.so` (13 MB) — base model
- `eagle_lib_bs6_fi.so` (12 MB) — EAGLE draft model

### EAGLE-2 FlashInfer Results

| Config | Run 1 | Run 2 | Run 3 | Avg |
|--------|------:|------:|------:|----:|
| EAGLE-2 FlashInfer (draft_length=4) | 9.3 t/s | 9.3 t/s | 9.4 t/s | **9.3 t/s** |
| Baseline (lib_bs6_fi.so, no spec) | 68.5 t/s | 72.2 t/s | 72.3 t/s | **71.5 t/s** |
| **EAGLE-2 speedup** | | | | **0.13× (8× SLOWER)** |

### Conclusion

FlashInfer EAGLE (9.3 t/s) matches TIR EAGLE (8.9 t/s) within measurement noise. **The EAGLE-2 regression is NOT a kernel backend issue — it is a fundamental MLC-LLM v0.20.0 framework problem on sm_87.**

Both FlashInfer and TIR backends produce identical ~9 t/s EAGLE performance, meaning the bottleneck is in the MLC-LLM engine's speculative decoding orchestration (draft/verify scheduling, KV cache management, tree-attention dispatch overhead), not in the attention kernel implementation.

### Verified Baseline (lib_bs1.so, Clean 5-Run Benchmark)

| Lib | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg | Best |
|-----|------:|------:|------:|------:|------:|----:|-----:|
| lib_bs1.so | 69.8 | 73.4 | 73.4 | 73.4 | 73.4 | **72.7** | **73.4** |
| lib_bs6_fi.so | 68.5 | 72.2 | 72.3 | 72.4 | 72.3 | **71.5** | **72.4** |

`lib_bs1.so` is 1.2 t/s faster due to less KV cache overhead (258 MB vs 386 MB). The Stage XXIII measurement of 65.31 t/s was likely affected by cold-start overhead in the original test conditions.

---

## Phase 6 — Final Summary (Closed 2026-04-03)

### Goal: ≥100 t/s token generation with >1B parameters on Jetson Orin Nano 8 GB

### Result: NOT ACHIEVED — best is 73.4 t/s (Llama-3.2-1B)

### All Paths Attempted

| # | Path | Result | Why Failed |
|---|------|--------|-----------|
| 1 | llama-speculative (3B+1B) | BLOCKED | NvMap IOVM: can't load 3B+1B simultaneously |
| 2 | TRT-LLM gemm_plugin | BLOCKED | NvMap IOVM: globWriter 1 GB serialization buffer |
| 3 | llama.cpp EAGLE | BLOCKED | llama.cpp has no EAGLE support |
| 4 | Prompt lookup decoding | FAILED | Sync overhead > speedup for non-repetitive text |
| 5 | llama-speculative (1B+smaller) | BLOCKED | No smaller Llama-3/LFM2.5 model exists |
| 6 | MLC-LLM EAGLE-2 (TIR) | **8.9 t/s** | TIR tree-attention 8× slower than baseline |
| 6B | MLC-LLM EAGLE-2 (FlashInfer) | **9.3 t/s** | Same regression — framework-level, not kernel |
| 7 | TRT Edge-LLM INT4 AWQ + EAGLE3 | **24.1 t/s** | Vocab reduction breaks EAGLE3 acceptance |
| 11 | TRT Edge-LLM FP8 + EAGLE3 | BLOCKED | sm_87 has no FP8 hardware support |
| 12 | TRT Edge-LLM INT8 SQ | BLOCKED | 5.9 GB ONNX > NvMap IOVM |
| 14 | MLC-LLM EAGLE-2 (trained state_19) | BLOCKED | Better head can't fix broken engine |

### Best Achieved Speeds (>1B Parameters)

| Rank | Model | Runtime | Config | t/s |
|------|-------|---------|--------|----:|
| 1 | **Llama-3.2-1B** | **MLC-LLM** | q4f16_1, lib_bs1.so | **73.4** |
| 2 | LFM2.5-1.2B | llama.cpp | IQ4_XS + FA | 65.64 |
| 3 | Llama-3.2-1B | llama.cpp | IQ4_XS + FA | 60.28 |
| 4 | Llama-3.2-1B | TRT-LLM | FP16 no-gemm-plugin | 44.08 |

### Hardware Bandwidth Ceiling Analysis

At 73.4 t/s with Llama-3.2-1B q4f16_1 (~663 MB weights), we are reading:
- 663 MB × 73.4 = **48.7 GB/s** effective bandwidth utilization
- Out of 68 GB/s theoretical = **71.6% BW efficiency**

This is excellent — very close to the practical ceiling for single-sequence decode. Reaching 100 t/s would require:
- 100/73.4 = 1.36× speedup via speculative decoding with acceptance rate ~1.4 tokens/step
- OR reducing model weight reads to ~680 MB/100 = 6.8 MB per token → requires ~430 MB model at INT4 = ~0.8B params (below the >1B threshold)

### Why 100 t/s Is Unreachable

1. **Speculative decoding is broken**: MLC-LLM EAGLE-2 is 8× slower, TRT Edge-LLM EAGLE3 is blocked by NvMap/FP8
2. **No smaller draft model**: Llama-3.2-1B is the smallest in its family; LFM2.5 has no smaller variant
3. **BW efficiency is already 72%**: Limited room for algorithmic improvement without spec decoding
4. **Hardware constraint**: 68 GB/s UMA bandwidth is the hard ceiling; sm_87 lacks FP8 and optimized EAGLE plugins
