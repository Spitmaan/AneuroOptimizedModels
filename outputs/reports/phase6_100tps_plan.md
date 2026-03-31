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
**Effort:** Low — testable today, no new builds required
**Hardware needed:** Jetson only

### Why this is the best first step

Llama 3.2 has both a **3B** and **1B** model with **identical tokenizer and architecture**. This is the ideal speculative decoding pair — high expected acceptance rate. The `llama-speculative` binary already exists on Jetson.

### Setup

```bash
# On Jetson host — download Llama-3.2-3B IQ4_XS (bartowski)
cd /home/spitman/Projects/Aneurologic/modelgarden/jetson-containers/data/models/standardized/
wget "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-IQ4_XS.gguf"
```

### Benchmark command

```bash
# Run speculative decoding: 3B main + 1B draft
/home/spitman/tools/llama.cpp/build/bin/llama-speculative \
  -m Llama-3.2-3B-Instruct-IQ4_XS.gguf \
  -md Llama-3.2-1B-Instruct-IQ4_XS.gguf \
  -ngl 99 -ngl-draft 99 -fa 1 \
  -p "$(python3 -c \"print('word ' * 200)\")" \
  -n 128 --draft 5 -e 2>&1 | grep -E 'eval time|speed|accepted'
```

For proper bench against `bench_gguf.py` framework:

```bash
python3 scripts/edge_optimization/bench_gguf.py \
  --model Llama-3.2-3B-Instruct-IQ4_XS.gguf \
  --speculative-draft Llama-3.2-1B-Instruct-IQ4_XS.gguf \
  --draft-tokens 5 \
  --extra-args "-ngl 99 -ngl-draft 99 -fa 1"
```

### Expected results

- Llama-3.2-3B standalone (IQ4_XS+FA): ~27–32 t/s
- With 1B draft (5-token lookahead, ~70% acceptance): **~55–75 t/s effective**
- Best case (high acceptance): 80+ t/s

### Measure and record

| Metric | Value |
|--------|-------|
| 3B standalone tg128 t/s | TBD |
| Speculative tg128 t/s (draft=5) | TBD |
| Accepted tokens % | TBD |

---

## Path 2 — Fix TRT-LLM Llama-3.2-1B W4A16 Build

**Target:** ~65–75 t/s for Llama-3.2-1B (proper W4A16, int4 weights at runtime)
**Effort:** Medium — requires diagnosing and fixing the TRT build OOM
**Hardware needed:** Jetson (build + run)

### Background

Our Stage XX attempt built Llama-3.2-1B without `--gemm_plugin float16` (FP16 fallback): 44.08 t/s, no improvement. The proper W4A16 build (with gemm_plugin) fails at engine serialization:

> `[TRT] [E] [globWriter.cpp::makeResizableGpuMemory::434] Error Code 2: OutOfMemory (Requested size was 1073741824 bytes.)`

TRT 10.3.0's engine serializer starts with a hardcoded 1 GB GPU buffer. With the 1.5 GB Llama checkpoint loaded in GPU + TRT build state, NvMap IOVM allocation fails.

### Why DGX Spark cannot help here

The DGX Spark is sm_121 (Blackwell GB10). TRT engines are **architecture-specific** and require the actual target GPU for kernel profiling. sm_121 ≠ sm_87 (Jetson Orin). There is no cross-architecture TRT engine compilation.

### Fix options (in priority order)

**Option A — Reduce lm_head to INT4 (quantize the vocabulary embedding too)**
The root cause is the 501 MB float16 lm_head (128K vocab × 2048 hidden). If lm_head is quantized to INT4, the checkpoint drops to ~1.0 GB and the engine serialization buffer allocation may succeed.

```bash
# In container — convert with quantized lm_head
python3 examples/llama/convert_checkpoint.py \
  --model_dir /tmp/llama32_1b \
  --output_dir /tmp/trtllm_llama32_w4_qlmhead \
  --dtype float16 \
  --use_weight_only --weight_only_precision int4 \
  --quant_lm_head \    # flag to also quantize lm_head
  --load_model_on_cpu
```

**Option B — Use pre-built engines from NVIDIA jetson-containers**
NVIDIA's `jetson-containers` project may have pre-built TRT engines for Llama 3.2 1B. Check:
```bash
# On Jetson
docker run --rm -it --runtime nvidia nvcr.io/nvidia/l4t-tensorrt-llm:r36.4 \
  ls /opt/tensorrt_llm/examples/llama/
```

**Option C — Build at reduced max_seq_len (force smaller engine footprint during serialization)**
The engine metadata grows with max_seq_len. Try `--max_seq_len 256 --max_input_len 128`:
```bash
trtllm-build \
  --checkpoint_dir /tmp/trtllm_llama32_w4 \
  --output_dir /workspace/outputs/trt_engines/llama32_w4a16 \
  --gemm_plugin float16 \
  --max_batch_size 1 --max_input_len 128 --max_seq_len 256
```

### Expected results once unblocked

| Metric | llama.cpp IQ4_XS+FA (current best) | TRT-LLM W4A16 (expected) |
|--------|:-----------------------------------:|:------------------------:|
| tg128 t/s | 54.37 | ~65–75 |

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

## Execution Order

| Priority | Path | Expected result | Blocking? |
|----------|------|----------------|-----------|
| **1** | [Path 1] Llama-3.2-3B + 1B speculative | ~60–80 t/s on 3B | Nothing (test today) |
| **2** | [Path 2] Fix TRT-LLM Llama 1B build | ~65–75 t/s on 1B | Try Option A (quantize lm_head) |
| **3** | [Path 3] Qwen2.5-1.5B + 0.5B draft | ~65–80 t/s on 1.5B | Download Qwen 1.5B weights |
| **4** | [Path 4] EAGLE-3 on DGX Spark | ~80–130 t/s on 1B | Multi-day training effort |

---

## Success Criteria

| Milestone | Target | Model |
|-----------|--------|-------|
| ✅ Achieved (Phase 5) | 100.87 t/s | Qwen2.5-0.5B TRT-LLM W4A16 |
| 🎯 Path 1 goal | ≥60 t/s effective | Llama-3.2-3B with 1B draft |
| 🎯 Path 2/3 goal | ≥65 t/s | Any 1–1.5B model, TRT-LLM W4A16 |
| 🎯 Stretch goal | ≥100 t/s | Any >1B model, EAGLE or speculative |
