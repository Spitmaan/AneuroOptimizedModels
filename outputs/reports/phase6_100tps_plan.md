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

## Why Previous Assessments Were Incomplete: TRT Edge-LLM

All five blocked paths used one of two frameworks:
- **llama.cpp**: Open-source, general-purpose. No EAGLE support. Lookup decoding blocked by sync overhead.
- **TRT-LLM (standard)**: Server/datacenter-oriented. Its Python `trtllm-build` uses a C++ serialization buffer (`globWriter`) that requires 1 GB GPU allocation — hitting the NvMap IOVM constraint on 8 GB Jetson.

**Neither is the right tool for Jetson.** NVIDIA maintains a third, purpose-built framework:

**TensorRT Edge-LLM** ([github.com/NVIDIA/TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)) — a high-performance C++ inference runtime specifically for embedded platforms (Jetson, DRIVE). It is architecturally separate from TRT-LLM:

| | TRT-LLM (what we used) | TRT Edge-LLM (new) |
|---|---|---|
| Target | Server/cloud GPU | Embedded: Jetson, DRIVE |
| Engine builder | Python `trtllm-build` + globWriter (~1 GB GPU alloc) | C++ `llm_build` binary (edge-optimized, lower memory) |
| EAGLE-3 support | No | **Yes** |
| NVFP4 quantization | No | Yes |
| Jetson Orin support | v0.12.0-jetson branch (64GB AGX tested) | **Experimental (JetPack 6.2.x)** |
| Export location | On-device (blocked by IOVM on 8 GB) | **On host (DGX Spark exports, Jetson only builds)** |

**Key insight:** The C++ `llm_build` in TRT Edge-LLM is purpose-designed for sub-8 GB devices. The 1 GB globWriter buffer that killed Path 2 is a Python-layer artifact of standard TRT-LLM. TRT Edge-LLM's engine builder has a fundamentally different serialization architecture targeting Jetson's memory envelope.

**IOVM analysis for our models:**
- Llama-3.2-1B INT4: ~700 MB IOVM
- EAGLE-3 head (draft): ~50–200 MB IOVM
- C++ runtime overhead: ~200 MB
- **Total: ~1.0–1.1 GB → well within 2.5 GB IOVM budget**

This is why EAGLE-3 is viable via TRT Edge-LLM but was not via llama.cpp (no support) or standard TRT-LLM (full-model draft = IOVM exhaustion).

---

## Path 6 — TRT Edge-LLM Llama-3.2-1B (Baseline Run)

**Target:** Verify TRT Edge-LLM pipeline works on Jetson Orin JetPack 6.2.x; establish baseline vs llama.cpp  
**Status: 🔄 NEXT**  
**Expected: 50–75 t/s** (may match llama.cpp; confirms pipeline before EAGLE-3)

### Step 1: Setup on DGX Spark (`spark-807e.local`)

DGX Spark (CUDA 13.0, aarch64, 119 GB UMA) satisfies the export pipeline requirements (CUDA 12.x+, GPU with CC 8.0+).

**Option A — pip install (preferred):**
```bash
# On DGX Spark
sudo apt update && sudo apt install python3-pip python3-venv -y
python3 -m venv ~/edge_llm_env
source ~/edge_llm_env/bin/activate
pip install --upgrade pip setuptools wheel

# Install TRT Edge-LLM Python export tools
pip install tensorrt_edge_llm

# Clone repo for access to example export scripts
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
pip install -r requirements.txt

# Verify — should print version and recognize Blackwell GB10
python3 -c "import tensorrt_edge_llm, tensorrt as trt; print(tensorrt_edge_llm.__version__, trt.__version__)"
```

**Option B — NGC container (isolated, avoids CUDA version conflicts):**
```bash
# Pull official NVIDIA container (includes TRT, CUDA, all dependencies)
docker pull nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev
docker run --rm -it --gpus all -v $(pwd):/workspace \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev
```

**CUDA 13.0 compatibility note:** TRT Edge-LLM docs list CUDA 12.x+ as requirement. DGX Spark runs CUDA 13.0 — verify the installed wheel works before proceeding. If `import tensorrt_edge_llm` fails, use the NGC container instead (Option B), which bundles a tested CUDA version.

### Step 2: Export Llama-3.2-1B on DGX Spark

Llama-3.2-1B is the target: already have HF weights or download fresh.

```bash
# On DGX Spark
# Option A: download fresh
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir ./llama32-1b

# Export with INT4 AWQ quantization (recommended for 8 GB Jetson)
# Exact command depends on TRT Edge-LLM version — check examples/ dir
tensorrt-edgellm-quantize-llm \
    --model_dir ./llama32-1b \
    --quantization int4_awq \
    --output_dir ./llama32_1b_onnx \
    --dtype float16

tensorrt-edgellm-export-llm \
    --model_dir ./llama32-1b \
    --quantized_dir ./llama32_1b_onnx \
    --output_dir ./llama32_1b_onnx_final
```

Or use the unified export script if available in the repo:
```bash
python3 export.py \
    --model_dir ./llama32-1b \
    --quantization int4_awq \
    --output_dir ./llama32_1b_exported \
    --dtype float16 \
    --max_input_len 2048 \
    --max_output_len 512
```

**Important:** Check the actual script names and flags against the repo's `examples/` directory — the exact CLI interface varies by version. The Quick Start Guide (URL: `nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/quick-start-guide.html`) uses Qwen3-0.6B as reference; adapt for Llama-3.2-1B.

### Step 3: Transfer to Jetson and Build C++ Engine

**Pre-flight: set Jetson to MAXN SUPER mode** (required for 67 TOPS; do this first every session):
```bash
ssh spitman@spitman-jetson
sudo nvpmodel -m 0      # MAXN SUPER mode (25W max, full GPU clocks)
sudo jetson_clocks      # lock all clocks to maximum frequency
```

**Transfer ONNX export to Jetson:**
```bash
# From DGX Spark
scp -r ./llama32_1b_exported spitman@spitman-jetson:/home/spitman/models/trt_edge/
```

**Build C++ runtime on Jetson:**
```bash
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
mkdir -p build && cd build

# Option A — from official Quick Start docs:
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/aarch64_linux_toolchain.cmake

# Option B — from guide (may be required for Jetson Orin target):
# cmake .. -DCMAKE_BUILD_TYPE=Release -DEMBEDDED_TARGET=jetson-orin

# Try Option A first; if CMake errors, try Option B or both flags combined
make -j$(nproc)  # 6 CPU cores on Jetson Orin Nano
```

**Build the TRT engine** (C++ `engine_builder` / `llm_build` — same binary, different path aliases depending on version):
```bash
# Binary location varies by TRT Edge-LLM version:
# Quick Start docs: ./llm_build
# Compiled binary: ./examples/engine_builder/engine_builder
# Try both; use whichever exists after 'make'

./llm_build \
    --model_dir /home/spitman/models/trt_edge/llama32_1b_exported \
    --output_file /home/spitman/models/trt_edge/llama32_1b.engine \
    --max_batch_size 1 \
    --max_input_len 2048 \
    --max_output_len 512 \
    --mmap  # use --mmap if available: limits peak memory during serialization on 8 GB device
```

**Monitor IOVM during build** (in a separate terminal):
```bash
dmesg -w | grep -i nvmap
```

**If engine build OOMs** (Jetson crashes or NvMap error during build — not inference):
```bash
# Increase swap temporarily
sudo fallocate -l 4G /swapfile2
sudo chmod 600 /swapfile2
sudo mkswap /swapfile2
sudo swapon /swapfile2
# Re-run engine build, then remove swap when done
```

**Alternative: jetson-containers pre-built environment** (avoids C++ build entirely if it fails):
```bash
git clone https://github.com/dusty-nv/jetson-containers.git
cd jetson-containers
./run.sh $(./autotag tensorrt_edgellm)
# Provides a container with TRT Edge-LLM runtime pre-compiled for JetPack 6.2
```

**Quick inference test** using the chat binary:
```bash
# Quick Start docs: llm_inference --engine_file ... --input_file ...
# Guide binary: ./examples/chat/chat --engine_file ... --tokenizer_dir ...
# Use whichever binary is compiled; both confirm the engine loaded correctly
./examples/chat/chat \
    --engine_file /home/spitman/models/trt_edge/llama32_1b.engine \
    --tokenizer_dir /home/spitman/models/trt_edge/llama32_1b_exported/tokenizer
```

### Step 4: Benchmark vs llama.cpp baseline

TRT Edge-LLM uses a CLI binary for inference (`llm_inference`), not a Python loop. To integrate with our benchmark framework, adapt `bench_gguf.py` to call `llm_inference` with timing.

**Important note on the guide's benchmarking script** (`docs/TensorRT_Edge-LLM_Guide_for_Jetson_Orin_Nano_Super.md`):
The Python snippet using `tensorrt_edge_llm.LLMRuntime` and `runtime.generate()` does not match the actual TRT Edge-LLM API (which uses C++ CLI tools). Treat it as a conceptual template only. The actual inference command is:
```bash
# Create input JSON
echo '{"input_ids": [1, 2, 3...], "max_new_tokens": 128}' > /tmp/input.json

# Run inference with timing
time ./llm_inference \
    --engine_file /home/spitman/models/trt_edge/llama32_1b.engine \
    --input_file /tmp/input.json
```

For throughput benchmarking, measure wall-clock time over N=128 output tokens with a fixed 512-token prompt. Compare directly to `llama-bench -n 128 -p 512 -fa 1 -ngl 99` baseline of **60.28 t/s**.

### Expected outcome

- **Best case:** TRT Edge-LLM C++ runtime slightly outperforms llama.cpp IQ4_XS (~65–75 t/s) due to custom kernels
- **Likely case:** Similar performance (~55–65 t/s) — both are bandwidth-bound at 1B scale
- **Risk:** JetPack 6.2.x experimental support may require patching; C++ build may have JetPack 6 compatibility issues
- **IOVM risk:** LOW — 1B model + runtime stays well under 2 GB

---

## Path 7 — TRT Edge-LLM + EAGLE-3 Speculative Decoding (Primary 100 t/s Path)

**Target:** ≥100 t/s with Llama-3.2-1B via EAGLE-3 draft head  
**Status: 🔒 BLOCKED on Path 6 success**  
**Expected: 120–180 t/s** (EAGLE-3 2–3× multiplier on ~60 t/s base)  
**IOVM budget:** ~1.1 GB (base 700 MB + EAGLE head 200 MB + runtime 200 MB) — fits

### Why EAGLE-3 works on Jetson where standard speculative decoding doesn't

Standard speculative decoding (Path 1) required loading two full models → IOVM exhaustion.
EAGLE-3 uses a **draft head**: a tiny (~50–300 MB) neural network that predicts future tokens using the hidden states from the main model's last layer. No second full model needed.

EAGLE-3 also solves the CPU-GPU sync problem that killed lookup decoding (Path 5): because the draft head runs as a fused CUDA kernel alongside the main model, there is no CPU round-trip per step.

| | Lookup decoding (Path 5, failed) | EAGLE-3 (Path 7) |
|---|---|---|
| Draft source | CPU n-gram hash table | GPU draft head (fused) |
| CPU-GPU sync per step | Yes (~28 ms overhead) | No (kernel-level) |
| IOVM cost | None (no model) | ~50–300 MB |
| Acceptance rate | 26% | 60–80% (learned, model-specific) |

### Step 1: Locate or train an EAGLE-3 head for Llama-3.2-1B

**Option A — Check for pre-trained head on HuggingFace:**

The EAGLE team (`yuhuili` on HuggingFace) publishes pre-trained EAGLE/EAGLE-3 heads. Known published heads as of early 2026:
- `yuhuili/EAGLE-LLaMA3-Instruct-8B` (EAGLE-2 for Llama-3-8B — confirmed in guide)
- `yuhuili/EAGLE3-LLaMA3.2-Instruct-1B` (EAGLE-3 for Llama-3.2-1B — **check if published**)

```bash
# On DGX Spark — search for 1B head
huggingface-cli search --type model "eagle llama-3.2-1b"
huggingface-cli search --type model "eagle3 llama"
# Direct check:
huggingface-cli download yuhuili/EAGLE3-LLaMA3.2-Instruct-1B --local-dir ./eagle3-head-1b 2>&1 | head -5
# If 404 → not published; proceed to Option B (train on DGX Spark)
```

**Option B — Train EAGLE-3 head on DGX Spark (if no pre-trained head exists):**

DGX Spark (119 GB UMA, sm_121 Blackwell) is suitable for training a draft head. Training time for a 1B-target EAGLE head is ~1–2 hours on a comparable GPU.

```bash
# On DGX Spark — install prerequisites
source ~/edge_llm_env/bin/activate
pip install torch transformers accelerate datasets

# Clone EAGLE repo
git clone https://github.com/SafeAILab/EAGLE
cd EAGLE

# Download base model
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir ./llama32-1b

# Train EAGLE-3 draft head
# (EAGLE repo provides train.py — check for EAGLE3 vs EAGLE2 branch)
python3 -m eagle.ge_data.allocation \
    --model_name_or_path ./llama32-1b \
    --outdir ./eagle_data

python3 -m eagle.train.main \
    --tmpdir ./eagle_data \
    --cpdir ./eagle_head_llama32_1b \
    --basepath ./llama32-1b
```

Note: EAGLE-3 specifically (`eagle3` branch or flag) is what TRT Edge-LLM supports. Check the EAGLE repo for the EAGLE-3 training path.

### Step 2: Export with EAGLE-3 head on DGX Spark

```bash
# Export base model + speculative head together
python3 export.py \
    --model_dir ./llama32-1b \
    --quantization int4_awq \
    --output_dir ./llama32_1b_base \
    --dtype float16

python3 export.py \
    --model_dir ./eagle_head_llama32_1b \
    --output_dir ./llama32_1b_eagle_head \
    --speculative_mode eagle3 \
    --dtype float16
```

### Step 3: Build speculative engine on Jetson

```bash
# Transfer both directories to Jetson
scp -r ./llama32_1b_base ./llama32_1b_eagle_head spitman@spitman-jetson:/home/spitman/models/trt_edge/

# On Jetson — build combined speculative engine
# Binary alias: ./llm_build OR ./examples/engine_builder/engine_builder (same tool)
./llm_build \
    --model_dir /home/spitman/models/trt_edge/llama32_1b_base \
    --speculative_model_dir /home/spitman/models/trt_edge/llama32_1b_eagle_head \
    --output_file /home/spitman/models/trt_edge/llama32_1b_eagle3.engine \
    --max_draft_len 5 \
    --max_batch_size 1 \
    --max_input_len 2048 \
    --max_output_len 512 \
    --mmap  # add if flag exists — reduces peak memory during serialization
```

**⚠️ Critical: avoid the guide's `trtllm-build` command**

The "Speculative Engine Example" section in `docs/TensorRT_Edge-LLM_Guide_for_Jetson_Orin_Nano_Super.md` shows:
```bash
# DO NOT USE — this is standard TRT-LLM (server product), not TRT Edge-LLM
trtllm-build --checkpoint_dir ... --speculative_decoding_mode draft_tokens_external ...
```
This command (`trtllm-build`) belongs to standard TRT-LLM — the same Python-based tool that hit the globWriter 1 GB IOVM blocker in Path 2. The guide incorrectly mixes TRT-LLM API into an Edge-LLM section. The correct speculative build path for TRT Edge-LLM is `engine_builder --speculative_model_dir`, not `trtllm-build`.

### Step 4: Benchmark

Benchmark `llm_inference` with speculative engine vs baseline `llama-bench`. Target metric: wall-clock tokens/second (not the per-step eval rate — EAGLE changes the effective tokens-per-step ratio).

EAGLE-3 effective throughput = (base tokens/s) × (1 + acceptance_rate × draft_len)

With 70% acceptance and 5 drafts: 60 × (1 + 0.7 × 5) / verification_overhead ≈ 60 × 3.5 / 1.2 ≈ **175 t/s** theoretical (optimistic).  
Conservative (50% acceptance, 3 drafts): 60 × 2.5 / 1.2 ≈ **125 t/s**.

### Prerequisites check

Before running Path 7, confirm with a quick test:
```bash
/home/spitman/tools/llama.cpp/build/bin/llama-speculative --help | grep -i eagle
# Expected: nothing (we confirmed no EAGLE in llama.cpp)
# This forces TRT Edge-LLM as the path
```

---

## Execution Order (Updated — Phase 6 Reopened)

| Priority | Path | Status | Result |
|----------|------|--------|--------|
| **1** | Path 1: Llama-3.2-3B + 1B speculative | ❌ BLOCKED | 26 t/s standalone; NvMap IOVM prevents loading two models |
| **2** | Path 2: TRT-LLM Llama 1B W4A16 gemm_plugin | ❌ BLOCKED | globWriter 1 GB IOVM; 8 patches tried; TRT-LLM is wrong tool for 8 GB Jetson |
| **3** | Path 3: Qwen2.5-1.5B TRT-LLM W4A16 | ❌ SKIPPED | Same IOVM constraint as Path 2 |
| **4** | Path 4: EAGLE-3 via llama.cpp | ❌ BLOCKED | llama.cpp has no EAGLE support |
| **5** | Path 5: Lookup decoding (llama-lookup) | ❌ BLOCKED | 26% acceptance; CPU-GPU sync overhead = 42% slower than baseline |
| **6** | **Path 6: TRT Edge-LLM Llama-3.2-1B (baseline)** | 🔄 **NEXT** | Setup DGX Spark + export + C++ engine build on Jetson |
| **7** | **Path 7: TRT Edge-LLM + EAGLE-3** | 🔒 After Path 6 | Draft head + speculative engine; primary 100 t/s route |

---

## Success Criteria (Updated)

| Milestone | Status | Result | Model |
|-----------|--------|--------|-------|
| ✅ Phase 5 best | Achieved | 100.87 t/s | Qwen2.5-0.5B TRT-LLM W4A16 |
| ✅ New >1B record | Achieved | **65.64 t/s** | LFM2 1.2B IQ4_XS + FA (llama.cpp) |
| ✅ New Llama 1B record | Achieved | **60.28 t/s** | Llama 3.2 1B IQ4_XS + FA (llama.cpp) |
| 🔄 Path 6 gate | Pending | TRT Edge-LLM works | Jetson Orin JetPack 6.2 experimental support confirmed |
| 🎯 Path 7 primary goal | Pending | ≥100 t/s | Llama 3.2 1B + EAGLE-3 via TRT Edge-LLM |
| 🎯 Stretch goal | Pending | ≥150 t/s | Llama 3.2 1B + EAGLE-3 (70%+ acceptance) |

**Revised assessment:** TRT Edge-LLM (a separate NVIDIA framework from TRT-LLM) was not considered in Paths 1–5. It is purpose-built for embedded Jetson/DRIVE hardware and supports EAGLE-3 speculative decoding — the two capabilities that blocked all previous paths. DGX Spark handles the heavy export/quantization; Jetson's C++ `llm_build` avoids the globWriter IOVM bottleneck. JetPack 6.2.x is listed as experimental (vs officially supported JetPack 7.1), so compatibility testing is the first risk to resolve (Path 6).
