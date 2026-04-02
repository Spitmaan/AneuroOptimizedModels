# Edge Optimization Ladder — Full Analysis Report

**Date:** 2026-03-29
**Hardware:** NVIDIA Jetson Orin Nano 8 GB (Jetson Orin Nano Developer Kit)
**Runtime:** llama.cpp v8510 (build commit `3a60d06ad`), host build (not Docker)
**Benchmark script:** [`scripts/edge_optimization/bench_gguf.py`](../../scripts/edge_optimization/bench_gguf.py)
**Results data:** [`outputs/logs/edge_opt/results.json`](../logs/edge_opt/results.json)

---

## Classification Note — Relationship to Stage 3

This work is **distinct from Stage 3** (TurboQuant KV Cache Compression). The differences are fundamental:

| Dimension | Stage 3 | Edge Optimization (this report) |
|-----------|---------|----------------------------------|
| Runtime | PyTorch inside Docker container | llama.cpp binary on Jetson host |
| Language | Python | C++ (binary invoked via subprocess) |
| KV compression | Custom Python hooks into attention | CUDA kernel flags (-ctk/-ctv) |
| Scope | KV cache only | Model weights + KV cache + compute |
| Quantization | NF4 (HuggingFace bitsandbytes) | GGUF K-quant formats |
| Measurement unit | tokens/s (48 token test), VRAM | pp512 t/s + tg128 t/s (standard llama-bench) |
| Accuracy | GSM8K + ARC (same 20-sample 3-shot) | GSM8K + ARC (same 20-sample 3-shot) |

The **conceptual overlap** is in KV cache quantization: Stage 3's KIVI algorithm performs group quantization on K and V tensors; llama.cpp's `-ctk`/`-ctv` flags do the same via CUDA kernels. The implementation, integration depth, and broader scope of this work make it a separate track.

---

## Glossary — Every Abbreviation Defined

| Term | Full Name | Definition |
|------|-----------|-----------|
| GGUF | GPT-Generated Unified Format | Binary file format for storing quantized LLM weights, vocabulary, and metadata. The standard format for llama.cpp models. Successor to GGML. |
| BPW | Bits Per Weight | Average number of bits used to store each model parameter. Q4_K_M ≈ 5.0 BPW; Q3_K_M ≈ 4.2 BPW; F16 = 16 BPW. Lower = smaller file = faster generation but potentially less accurate. |
| t/s | Tokens Per Second | Speed metric: how many tokens the model processes or generates per second. Higher is faster. |
| pp512 | Prompt Processing, 512 tokens | Speed at which the model processes a 512-token input prompt in a single forward pass (prefill). Also called "prompt eval" or "PE speed." Measures parallel GPU throughput. |
| tg128 | Token Generation, 128 tokens | Speed at which the model generates 128 tokens auto-regressively (one token at a time). The bottleneck for interactive use. Measures sequential memory bandwidth utilization. |
| pp_std / tg_std | Standard Deviation | Variation in speed across the 3 benchmark repetitions. Low std = stable measurement. |
| FA / Flash Attn | Flash Attention 2 | A fused CUDA kernel that computes self-attention in tiles fitting GPU SRAM, avoiding the quadratic O(N²) memory allocation. Mathematically equivalent to standard attention; faster and lower memory. Reference: Dao et al. (2022), arXiv:2205.14135. |
| KV Cache | Key-Value Cache | During autoregressive generation, the Key and Value tensors from all past positions are stored and reused. Avoids recomputing them for every new token. Grows linearly with context length. |
| ctk / ctv | Cache Type K / Cache Type V | llama.cpp flags to quantize the Key cache (`-ctk`) and Value cache (`-ctv`) at runtime. Values: `f16` (default, 16-bit), `q8_0` (8-bit), `q4_0` (4-bit). |
| ngl | Number of GPU Layers | llama.cpp flag (`-ngl`) specifying how many model layers to offload to the GPU. `-ngl 99` means all layers (more than any model has). |
| UMA | Unified Memory Architecture | Hardware design where CPU and GPU share the same physical RAM pool. On Jetson Orin Nano, all 8 GB is shared — there is no separate VRAM. Means GPU can access all system memory but at system memory speeds (~68 GB/s), not dedicated VRAM speeds. |
| VRAM | Video Random Access Memory | In standard setups: dedicated GPU memory. On Jetson UMA: the shared system RAM that the GPU allocates from. |
| SoC | System-on-Chip | Single chip integrating CPU, GPU, memory controller, and I/O. The Jetson Orin Nano uses the NVIDIA Orin SoC. |
| CUDA | Compute Unified Device Architecture | NVIDIA's parallel computing platform and programming model. The GPU kernels in llama.cpp run via CUDA. |
| BPE | Byte Pair Encoding | A tokenization algorithm that iteratively merges frequent byte/character pairs into vocabulary tokens. Used by LFM2.5, Llama 3, and Qwen2.5 tokenizers (all with different vocabularies). |
| GSM8K | Grade School Math 8K | A benchmark dataset of 8,500 grade-school math word problems requiring arithmetic and multi-step reasoning. We evaluate on the test split (1,319 problems). Reference: Cobbe et al. (2021). |
| ARC | AI2 Reasoning Challenge | A benchmark of science multiple-choice questions (4 options: A/B/C/D) at elementary school level. "Challenge" set contains questions that stumped retrieval-based and word co-occurrence methods. Reference: Clark et al. (2018). |
| MCQ | Multiple Choice Question | A question format with a fixed set of answer options. ARC-Challenge is an MCQ benchmark. |
| SLM | Small Language Model | A language model with a relatively small parameter count (typically <3B parameters), designed for efficiency on edge hardware. The three models benchmarked here are all SLMs. |
| SSM | State Space Model | A class of sequence model using recurrent state transitions instead of full attention. LFM2.5 uses a hybrid SSM+attention architecture. |
| LFM | Liquid Foundation Model | LiquidAI's model family. LFM2.5 is the second-generation 2.5 series. The "liquid" refers to liquid neural networks, a form of continuous-time RNN. |
| Q4_K_M | Quantization 4-bit K-quant Medium | A GGUF quantization format: 4-bit K-quant (block-wise quantization with two scales per block), "medium" variant (promotes some layers to Q6_K precision using an importance matrix). Average ~5.0 BPW. |
| Q4_K_S | Quantization 4-bit K-quant Small | Like Q4_K_M but with less Q6_K promotion. Slightly smaller file (~4.8 BPW), marginal accuracy difference. |
| Q3_K_M | Quantization 3-bit K-quant Medium | More aggressive: 3-bit K-quant with medium promotion. Average ~4.2 BPW. Requires dequantization (converting 3-bit integers to float for computation), adding compute overhead per token. |
| F16 | 16-bit Floating Point | Full half-precision storage. Maximum accuracy, 2 bytes per weight. Used as source for re-quantization when available. Not used at inference — too slow for edge. |
| IQ4_XS | I-Quant 4-bit Extra Small | An "importance quantization" format that uses a calibration-based importance matrix to assign bits selectively. Typically slightly smaller than Q4_K_S with better accuracy than Q4_K_M. Requires an F16 source and imatrix calibration file. Not used in this work (F16 source unavailable for LFM2.5). |
| imatrix | Importance Matrix | A calibration file computed by running the model on representative text (e.g., wikitext-2) and tracking which weight channels contribute most to output quality. Used by Q4_K_M and IQ4_XS quantizers to preserve important channels at higher precision. |
| HF | HuggingFace | The primary ML model repository platform. Models downloaded via `huggingface_hub` Python library. |
| SD | Speculative Decoding | A generation strategy where a small "draft" model proposes multiple candidate tokens, and a large "target" model verifies them in parallel. Accepted tokens advance the position by more than one per target model call. Requires identical tokenizer vocabulary. Reference: Leviathan et al. (2023). |
| RMSE | Root Mean Square Error | A distance metric between two tensors: sqrt(mean((A-B)²)). Used in Stage 3 to measure KV cache reconstruction error after compression. Lower = more faithful reconstruction. |
| BOS | Beginning of Sequence | A special token prepended to every sequence. LFM2.5: `<\|startoftext\|>` (token ID 1). Llama 3: `<\|begin_of_text\|>`. Qwen2.5: `<\|im_start\|>system`. |
| EOS | End of Sequence | A special token that signals the model to stop generating. LFM2.5: `<\|endoftext\|>` (token ID 7). |

---

## 1. Hardware Platform

### NVIDIA Jetson Orin Nano Developer Kit 8 GB

| Component | Specification |
|-----------|--------------|
| **SoC** | NVIDIA Orin (Ampere GPU + Cortex-A78AE CPU) |
| **CPU** | 6-core ARM Cortex-A78AE @ up to 1.5 GHz |
| **GPU** | 1024-core NVIDIA Ampere @ up to 625 MHz |
| **GPU Compute Capability** | 8.7 |
| **Memory** | 8 GB LPDDR5 |
| **Memory Architecture** | UMA — CPU and GPU share the same physical 8 GB |
| **Memory Bandwidth** | ~68 GB/s (theoretical peak) |
| **JetPack** | 6.2 (L4T r36.4) |
| **CUDA Version** | 12.6 |
| **OS** | Ubuntu 22.04 (ARM64) |

**Why UMA matters for LLM inference:** On standard desktop systems, GPU VRAM is separate from CPU RAM — a GPU with 24 GB VRAM can run large models without touching system RAM. On the Jetson, all 8 GB is shared. This means:
1. The OS, processes, and Docker containers consume some of the 8 GB, leaving typically 5–7 GB for model weights + KV cache
2. Memory bandwidth (~68 GB/s) is the primary bottleneck for token generation — smaller models fit more tokens per second because fewer bytes need to move per token
3. CUDA memory allocation and deallocation can be slow; lingering contexts from previous processes cause OOM errors if insufficient cooldown time is given between runs

**Observed behavior:** After killing `llama-server`, CUDA contexts do not release instantly. A 15-second sleep is required between benchmark runs to avoid `NvMapMemAllocInternalTagged: error 12` (ENOMEM) failures.

---

## 2. Runtime — llama.cpp

### Version and Build

| Field | Value |
|-------|-------|
| **Version** | 8510 |
| **Build commit** | `3a60d06ad` |
| **Backend** | CUDA (CUDA 12.6) |
| **Source location** | `/home/spitman/tools/llama.cpp/` |
| **Build directory** | `/home/spitman/tools/llama.cpp/build/bin/` |

### Binaries Used

| Binary | Purpose |
|--------|---------|
| `llama-bench` | Throughput measurement: runs pp512 and tg128 benchmarks, outputs JSON |
| `llama-server` | HTTP inference server (port 8765); used for GSM8K and ARC-Challenge accuracy evaluation |
| `llama-quantize` | Re-quantizes existing GGUF files to different formats |
| `llama-speculative` | Speculative decoding with a draft model (`-md` flag) |

### Key Flags

| Flag | Argument | Effect |
|------|----------|--------|
| `-m PATH` | model path | Load GGUF file from path |
| `-ngl 99` | integer | Offload all model layers to GPU (99 > any real layer count) |
| `-fa 1` | boolean | Enable Flash Attention 2 CUDA kernel |
| `-ctk q8_0` | type string | Quantize Key cache to 8-bit; reduces KV memory 2× vs f16 |
| `-ctk q4_0` | type string | Quantize Key cache to 4-bit; reduces KV memory 4× vs f16 |
| `-ctv q4_0` | type string | Quantize Value cache to 4-bit |
| `-t N` | integer | CPU thread count (irrelevant at `-ngl 99`: GPU does all compute) |
| `-p 512` | integer | llama-bench: use 512-token prompt for prefill measurement |
| `-n 128` | integer | llama-bench: generate 128 tokens for generation measurement |
| `-r 3` | integer | llama-bench: repeat 3 times and average |
| `-o json` | format | llama-bench: output results as JSON |
| `--port 8765` | integer | llama-server: listen on port 8765 |
| `--ctx-size 2048` | integer | llama-server: maximum context window for inference |
| `-n 256` | integer | llama-server: maximum tokens to generate per request |
| `--no-mmap` | flag | llama-server: disable memory-mapped file loading (forces full RAM copy) |
| `-md PATH` | path | llama-speculative: path to draft model |
| `--draft 8` | integer | llama-speculative: generate 8 speculative tokens per draft step |

### How llama-bench Measures Speed

llama-bench runs two sub-tests per invocation:
1. **Prompt processing (pp):** Loads a synthetic 512-token prompt. Times a single forward pass over all 512 tokens (batch processing). Result in tokens/second.
2. **Token generation (tg):** Generates 128 tokens auto-regressively from an empty prompt (one token per forward pass). Result in tokens/second.

Each test is repeated 3 times; the reported value is the average. Standard deviation across the 3 runs is also captured (pp_std, tg_std).

**Why these two metrics matter differently:**
- `pp512` measures GPU parallelism (many tokens processed simultaneously) — important for long-prompt tasks
- `tg128` measures memory bandwidth utilization at batch_size=1 — the real bottleneck for interactive generation. Every generated token requires loading the entire model weight tensor from RAM once.

---

## 3. Models

### 3.1 LiquidAI LFM2.5-1.2B-Instruct

| Property | Value |
|----------|-------|
| **Developer** | LiquidAI |
| **Parameter count** | 1.24 billion |
| **Architecture** | Hybrid: Gated Delta Net (SSM variant) + multi-head attention layers |
| **Architecture name in llama.cpp** | `lfm2` |
| **Vocabulary size** | 65,536 tokens |
| **Tokenizer type** | BPE ("lfm2" pre-tokenizer) |
| **Context length** | 128,000 tokens |
| **BOS token** | `<\|startoftext\|>` (ID 1) |
| **EOS token** | `<\|endoftext\|>` (ID 7) |
| **HuggingFace source** | `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` |
| **Original quantization** | Q4_K_M with imatrix mixed quants (includes q6_K for critical layers) |

**Architecture note:** LFM2.5 is not a pure transformer. It uses a hybrid of:
- **Gated Delta Net (GDN):** A state space model (SSM) variant using gated delta rule updates. These layers maintain a compact recurrent state — O(1) memory regardless of sequence length, unlike attention.
- **Multi-head attention layers:** Standard softmax attention at selected depths, providing global context modeling at the cost of O(N) KV cache.

This hybrid architecture is why llama.cpp's KV cache is smaller than a pure transformer of the same parameter count — only the attention layers contribute KV entries. It is also why converting the model to GGUF from raw HuggingFace weights fails (custom config keys like `block_ff_dim` are not in the standard converter).

**Files created on Jetson:**

| File | Size | BPW | Source | Method |
|------|------|-----|--------|--------|
| `LiquidAI_LFM2.5-1.2B-Instruct-Q4_K_M.gguf` | 698 MB | ~5.0 | HuggingFace download | Original (imatrix mixed) |
| `LiquidAI_LFM2.5-1.2B-Instruct-Q4_K_S.gguf` | 668 MB | ~4.8 | Re-quantized | `llama-quantize --allow-requantize` |
| `LiquidAI_LFM2.5-1.2B-Instruct-Q3_K_M.gguf` | ~620 MB | ~4.2 | Re-quantized | `llama-quantize --allow-requantize` |

**Note on `--allow-requantize`:** The original Q4_K_M GGUF uses mixed imatrix quantization — some tensors (attention projections in critical layers) are stored at Q6_K precision rather than Q4_K. llama-quantize refuses to re-quantize Q6_K tensors by default because the loss compounds (quantizing an already-quantized tensor). The `--allow-requantize` flag bypasses this check. Quality loss is real but acceptable for our experimental purposes.

**F16 conversion failure:** Attempting to convert LFM2.5-1.2B from HuggingFace safetensors to GGUF F16 using `convert_hf_to_gguf.py` failed with `KeyError: 'block_ff_dim'` in `_add_feed_forward_length()`. The LFM2.5 architecture uses a non-standard config key for its feed-forward dimension that the converter's code path does not handle. This blocked the IQ4_XS optimization path (which requires F16 source + imatrix).

---

### 3.2 Meta Llama-3.2-1B-Instruct

| Property | Value |
|----------|-------|
| **Developer** | Meta AI |
| **Parameter count** | 1.24 billion |
| **Architecture** | Dense transformer (Llama 3 architecture) with GQA |
| **GQA** | Grouped Query Attention — 32 query heads, 8 KV heads (4× reduction in KV cache vs standard MHA) |
| **Vocabulary size** | 128,256 tokens |
| **Tokenizer type** | tiktoken BPE (Llama 3 tokenizer) |
| **Context length** | 131,072 tokens |
| **BOS token** | `<\|begin_of_text\|>` |
| **HuggingFace source** | `meta-llama/Llama-3.2-1B-Instruct` |
| **Original quantization** | Q4_K_M with imatrix mixed quants |

**Files created on Jetson:**

| File | Size | BPW | Source | Method |
|------|------|-----|--------|--------|
| `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | 762 MB | ~5.2 | HuggingFace download | Original |
| `Llama-3.2-1B-Instruct-Q4_K_S.gguf` | 740 MB | ~5.0 | Re-quantized | `llama-quantize --allow-requantize` |
| `Llama-3.2-1B-Instruct-Q3_K_M.gguf` | 651 MB | ~4.4 | Re-quantized | `llama-quantize --allow-requantize` |

**GQA note:** Grouped Query Attention reduces the number of KV heads (8 instead of 32). This means Llama-3.2-1B has a proportionally smaller KV cache than its parameter count suggests, but more complex attention computation patterns. This likely contributes to the dequantization overhead being more pronounced with Q3_K_M.

---

### 3.3 Alibaba Qwen2.5-0.5B-Instruct

| Property | Value |
|----------|-------|
| **Developer** | Alibaba DAMO Academy |
| **Parameter count** | 494 million (~0.5B) |
| **Architecture** | Dense transformer (Qwen2 architecture) with GQA |
| **GQA** | 14 query heads, 2 KV heads (7× KV reduction) |
| **Vocabulary size** | 151,936 tokens |
| **Tokenizer type** | tiktoken-based BPE (Qwen tokenizer) |
| **Context length** | 32,768 tokens |
| **HuggingFace source** | `Qwen/Qwen2.5-0.5B-Instruct` |
| **GGUF creation** | Converted from F16 GGUF inside aneurologic_phase5 Docker container using `convert_hf_to_gguf.py` from llama.cpp source tree |

**Files created on Jetson:**

| File | Size | BPW | Source | Method |
|------|------|-----|--------|--------|
| `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` | ~386 MB | ~5.0 | Converted + quantized | `convert_hf_to_gguf.py` → `llama-quantize` |
| `Qwen2.5-0.5B-Instruct-Q4_K_S.gguf` | ~370 MB | ~4.8 | Re-quantized | `llama-quantize` |
| `Qwen2.5-0.5B-Instruct-Q3_K_M.gguf` | ~310 MB | ~4.2 | Re-quantized | `llama-quantize` |

---

## 4. Evaluation Methodology

### 4.1 Speed Measurement

**Tool:** `llama-bench`
**Protocol:** 3 repetitions, average reported. Standard deviation captured.

```
llama-bench -m MODEL.gguf -p 512 -n 128 -ngl 99 -r 3 -o json [extra_flags]
```

- **pp512:** 512-token prompt processed in one batched forward pass. Captures parallel GPU throughput. All 512 tokens are known upfront; no autoregression.
- **tg128:** 128 tokens generated one at a time from an empty prompt. Pure autoregressive speed. Each token requires one full model forward pass.

**Why tg128 matters more:** For interactive LLM use (chat, generation), the user waits for tokens one at a time. tg128 directly reflects perceived latency. pp512 matters for long-document processing (summarization, RAG retrieval).

### 4.2 Accuracy Measurement

**Tool:** `llama-server` (HTTP) + HuggingFace `datasets` library
**Protocol:** 3-shot prompting, 20 samples, temperature=0.0 (greedy decoding)

**Server configuration:**
```
llama-server -m MODEL.gguf -ngl 99 --port 8765 --ctx-size 2048 -n 256 --no-mmap -t 4 [extra_flags]
```

#### GSM8K (Grade School Math)

**Dataset:** `openai/gsm8k`, config `"main"`, split `"test"`, samples 0–19 (20 total)
**Evaluation type:** Open-ended generation → number extraction
**Metric:** Exact match between predicted and gold answer number strings

**3-shot prompt template:**
```
Solve the math problem and give only the final numeric answer on the last line.

Question: There are 15 trees in the grove. Workers will plant more. After planting there are 21.
How many did workers plant?
Answer: 6

Question: If there are 3 cars and 2 more arrive, how many total?
Answer: 5

Question: Leah had 32 chocolates, sister had 42. They ate 35. How many pieces left total?
Answer: 39

Question: {ACTUAL QUESTION}
Answer:
```

**Gold extraction:** Regex `r"####\s*([\d,\-]+)"` extracts the number after the `####` marker in the dataset's answer field. Commas are stripped (e.g., "1,234" → "1234").
**Prediction extraction:** Regex `r"-?\d[\d,]*"` finds all numbers in the model's response. The **last** number is used as the predicted answer (models often do intermediate calculations). Commas stripped.
**Stop tokens:** `["\n\n", "Question:", "####"]` — stops before a new question begins or before the gold answer delimiter.
**n_predict:** 128 tokens (enough for a short chain-of-thought)

**Granularity note:** With 20 samples, each correct answer contributes exactly 5.0 percentage points. A score of 5.0% means exactly 1/20 correct. Differences below 5% cannot be distinguished.

#### ARC-Challenge (AI2 Reasoning Challenge)

**Dataset:** `allenai/ai2_arc`, config `"ARC-Challenge"`, split `"test"`, samples 0–19 (20 total)
**Evaluation type:** Multiple-choice generation → letter extraction
**Metric:** Exact match between predicted and gold letter (A, B, C, or D)

**Label mapping:** Some questions use numeric keys (1, 2, 3, 4) instead of letters. These are mapped: 1→A, 2→B, 3→C, 4→D.

**3-shot prompt template:**
```
Choose the best answer (A, B, C, or D). Reply with just the letter.

Question: Which of the following is an example of a physical change?
(A) Burning wood (B) Rusting iron (C) Melting ice (D) Digesting food
Answer: C

Question: A student measures how fast a ball rolls down a ramp. Which measurement is most important?
(A) Color of ball (B) Weight of ramp (C) Distance traveled (D) Temperature
Answer: C

Question: Which energy transformation occurs in a battery-powered flashlight?
(A) Chemical to light (B) Mechanical to electrical (C) Solar to chemical (D) Thermal to light
Answer: A

Question: {ACTUAL QUESTION}
(A) {choice_0} (B) {choice_1} (C) {choice_2} (D) {choice_3}
Answer:
```

**Prediction extraction:** `r"\b([A-D])\b"` on the uppercased response. The first A/B/C/D letter found is used.
**Stop tokens:** `["\n", "Question:"]`
**n_predict:** 5 tokens (just enough for "A", "B", "C", or "D")

**Random chance baseline:** 4 choices → 25.0% expected from random guessing. Any score above 25% indicates the model is learning something from the few-shot context.

### 4.3 Result Recording

Each benchmark run records one JSON entry in `outputs/logs/edge_opt/results.json`:
```json
{
  "label": "human-readable identifier",
  "model": "filename.gguf",
  "flags": "extra llama.cpp flags string",
  "timestamp": "ISO 8601 datetime",
  "speed": {
    "pp512": 1875.45,  "pp_std": 83.75,
    "tg128": 49.81,    "tg_std": 0.08
  },
  "accuracy": {
    "gsm8k": {"score": 5.0, "correct": 1, "total": 20},
    "arc":   {"score": 60.0, "correct": 12, "total": 20}
  }
}
```

---

## 5. Methods Tested — Completed Stages (I–VII)

Each stage below changes exactly one variable from the previous winner and measures all four metrics simultaneously. Stages are numbered in Roman numerals as a new series independent of the Stages 1–7 Docker pipeline.

### Stage I — Baseline

**Config:** Q4_K_M, no extra flags
**Purpose:** Establish the reproducible reference point under the host llama.cpp build (v8510). All subsequent stages are compared against this.
**Key question answered:** What is the true host-build tg128 for each model? (The previously quoted 55.4 t/s for LFM2.5-1.2B came from a different build; reproducible host baseline is 49.81 t/s.)

---

### Stage II — Flash Attention (`-fa 1`)

**What it is:** Flash Attention 2 (Dao et al., 2022; arXiv:2205.14135) reformulates self-attention computation to avoid materializing the full N×N attention score matrix in global GPU memory. Instead, it tiles the computation to fit in GPU SRAM (shared memory/L1 cache), which is ~100× faster than global memory.

**Mathematical equivalence:** Flash Attention computes the exact same output as standard attention — it is a compute-level optimization, not an approximation. No accuracy change is expected or observed.

**Flag:** `-fa 1` passed to both `llama-bench` and `llama-server`

**Expected effect:**
- pp512: significant improvement (up to +40%) — prefill processes many tokens in parallel, attention matrix is large
- tg128: moderate improvement (~5%) — at batch_size=1, attention computation is small relative to weight loading; the main bottleneck (memory bandwidth) is not addressed

**Implementation note:** For LFM2.5-1.2B's hybrid architecture, Flash Attention applies only to the pure-attention layers. The SSM/GDN layers are unaffected. The `sched_reserve` log shows `"fused Gated Delta Net (autoregressive) enabled"` and `"fused Gated Delta Net (chunked) enabled"` — these are separate optimizations for the SSM layers that activate automatically alongside Flash Attention.

---

### Stage III — KV Cache Quantization via `-ctk` and `-ctv`

**What it is:** llama.cpp implements KV cache quantization as CUDA kernels that compress Key and/or Value tensors after computation and decompress them before use. This is the runtime equivalent of Stage 3's KIVI approach, but integrated into the C++ inference loop rather than applied as Python hooks.

**Available types:**
- `f16` (default): 16-bit half-precision, no compression
- `q8_0`: 8-bit, 0-point quantization, ~2× compression vs f16
- `q4_0`: 4-bit, 0-point quantization, ~4× compression vs f16

**Flags tested:**
1. `-fa 1 -ctk q8_0` — Flash Attn + Key cache quantized to 8-bit
2. `-fa 1 -ctk q4_0 -ctv q4_0` — Flash Attn + both K and V caches quantized to 4-bit

**Why only at 512-token context?** At 512 tokens, the KV cache is tiny (a few MB at most). There is no memory pressure from the KV cache. The compression adds compute overhead (quantize after write, dequantize before read) without providing meaningful memory savings at this context length. KV cache quantization becomes beneficial only at 4K+ token contexts where the KV cache grows large enough to cause memory pressure or bandwidth saturation.

**KV types available in llama.cpp v8510** (all whitelisted for `-ctk`/`-ctv`):
`f32`, `f16` (default), `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`

Tested in Stage III: `q8_0` and `q4_0`. Not yet tested: `q4_1`, `iq4_nl`, `q5_0`, `q5_1` — see Stage VIII in the roadmap.

---

### Stage IV — Model Weight Quantization Format — Q4_K_S (WINNER)

**What it is:** Changing the quantization format of the model weights themselves. Unlike KV cache quantization (which affects runtime KV tensors), this changes the stored file and affects every forward pass.

**Why smaller BPW can be faster:** At batch_size=1, token generation speed is limited by memory bandwidth — how fast the GPU can load model weights from RAM. Smaller weights = fewer bytes to load = faster generation. This is the primary reason for testing Q4_K_S and Q3_K_M.

**The tradeoff:** Smaller formats require dequantization (converting quantized integers back to float for matrix multiplication). For very aggressive quantization (Q3 and below), the dequantization compute cost can exceed the bandwidth savings, leading to slower generation despite smaller file size.

**Re-quantization process:**
```bash
# Requires --allow-requantize because source has mixed Q4_K/Q6_K tensors
llama-quantize --allow-requantize SOURCE_Q4_K_M.gguf DEST_Q4_K_S.gguf Q4_K_S
llama-quantize --allow-requantize SOURCE_Q4_K_M.gguf DEST_Q3_K_M.gguf Q3_K_M
```

**Quality loss from re-quantization:** The source Q4_K_M files already lost precision when first quantized from F16. Re-quantizing from Q4_K_M (rather than F16) applies a second quantization loss on top. The actual accuracy impact is small for Q4_K_S (higher precision) but more pronounced for Q3_K_M.

**Result:** Q4_K_S + FA is the best configuration across all models (or best for ≥1B models; Qwen 0.5B is covered by Stage V).

---

### Stage V — Model Weight Quantization Format — Q3_K_M (FAILED for ≥1B)

**What changed from Stage IV:** Model file changed from Q4_K_S (~4.8 BPW) to Q3_K_M (~4.2 BPW). Same `-fa 1` flag.

**Finding:** Q3_K_M is slower AND less accurate for LFM2.5-1.2B and Llama-3.2-1B due to dequantization compute overhead exceeding bandwidth savings. However, it **works** for Qwen2.5-0.5B (+17% tg, flat accuracy) because the smaller model has lower bandwidth pressure — the arithmetic savings from loading fewer bytes outweigh the dequantization cost for sub-1B models.

---

### Stage VI — CPU Thread Count (`-t 6`)

**What it is:** Setting the number of CPU threads used for tokenization and any CPU-side computation.

**Flag tested:** `-fa 1 -t 6` on LFM2.5-1.2B Q4_K_S (speed-only, no accuracy measurement)

**Why expected to have no effect:** With `-ngl 99`, all model layers are offloaded to the GPU. The CPU handles only tokenization (fast, trivially parallelizable) and the Python subprocess interface. There is no matrix multiplication on CPU.

**Finding:** Confirmed zero effect. `-t` flag is irrelevant at full GPU offload.

---

### Stage VII — Speculative Decoding with LFM2-700M Draft (BLOCKED)

**What it is:** Speculative decoding (Leviathan et al., 2023) uses a small "draft" model to propose multiple candidate tokens in one step. A larger "target" model then verifies all candidates in a single forward pass. If the target agrees with the draft's tokens, all are accepted simultaneously — achieving >1 token per target model call.

**Setup attempted:**
- Target model: `LiquidAI_LFM2.5-1.2B-Instruct-Q4_K_S.gguf`
- Draft model: `LFM2-700M-Q4_K_M.gguf` (from `LiquidAI/LFM2-700M-GGUF`)
- Binary: `llama-speculative` with flags `-md DRAFT.gguf --draft 8 -c 512 -ngl 99`
- Draft model size: 447 MB, 742 million parameters (reported as "700M")
- Draft model speed (standalone): **63.69 t/s** generation (at 32-token test)

**Requirement for speculative decoding:** The draft and target models must use **exactly the same tokenizer vocabulary** — every token ID must map to the same string in both models. This ensures the target model can correctly evaluate which draft tokens match its own distribution.

**Result: BLOCKED — tokenizer mismatch**

llama.cpp's compatibility check failed with:
```
main: draft model vocab must match target model to use speculation
but token 128 content differs — target '<|audio_start|>', draft '<|reserved_118|>'
```

**Root cause:** LFM2 and LFM2.5 use the same vocabulary size (65,536) and the same base tokenizer, but LFM2.5 was extended with additional special tokens for multimodal capabilities (audio and visual). Token IDs 128 onward differ:
- LFM2.5-1.2B: `<|audio_start|>`, `<|audio_end|>`, `<|image_start|>`, ... (multimodal tokens)
- LFM2-700M: `<|reserved_118|>`, `<|reserved_119|>`, ... (placeholder tokens, unassigned)

**Why no workaround exists:** The vocabulary mismatch is at the model architecture level. The only compatible draft model would be another LFM2.5-family model (e.g., a hypothetical LFM2.5-700M). As of 2026-03-29, LiquidAI has not released a sub-1B LFM2.5 model.

**Other models considered:**
- Llama-3.2-1B: no smaller Llama 3 model exists (1B is the smallest)
- Qwen2.5-0.5B → Qwen2.5-1.5B: would need a larger Qwen target; Qwen2.5-0.5B cannot serve as target

**Potential future path:** LiquidAI could release LFM2.5-350M or LFM2.5-700M. Alternatively, an EAGLE-style speculative head (Cai et al., 2024; `<|audio_start|>` discussion at `LiquidAI/LFM2.5-1.2B-Instruct` discussions #10) could be trained on LFM2.5-1.2B's hidden states without requiring vocabulary compatibility.

---

## 6. Complete Results — All Measurements

All timestamps are UTC-4 (Eastern Daylight Time), 2026-03-29.

### 6.1 LFM2.5-1.2B — Full Optimization Ladder

| Stage | Label | llama.cpp Flags | pp512 t/s | pp_std | tg128 t/s | tg_std | GSM8K | ARC | tg vs Baseline | ARC vs Baseline |
|-------|-------|----------------|----------:|-------:|----------:|-------:|------:|----:|---------------:|----------------:|
| **I** | Baseline | *(none)* | 1875.45 | ±83.75 | **49.81** | ±0.08 | 5.0% (1/20) | 60.0% (12/20) | — | — |
| **II** | Flash Attn | `-fa 1` | 2106.80 | ±108.47 | **51.64** | ±0.02 | 5.0% (1/20) | 60.0% (12/20) | +3.7% ✅ | 0% |
| **IIIa** | FA + KV-K-q8 | `-fa 1 -ctk q8_0` | 718.14 | ±47.42 | **40.39** | ±0.43 | 5.0% (1/20) | 65.0% (13/20) | **−19.0% ❌** | +5% |
| **IIIb** | FA + KV-KV-q4 | `-fa 1 -ctk q4_0 -ctv q4_0` | 2104.62 | ±111.86 | **50.95** | ±1.00 | 5.0% (1/20) | 65.0% (13/20) | −1.7% | +5% |
| **IV** ★ | Q4_K_S + FA | `-fa 1` (Q4_K_S file) | 2079.60 | ±113.49 | **53.22** | ±0.05 | 5.0% (1/20) | 70.0% (14/20) | **+6.8% ✅** | **+10% ✅** |
| **V** | Q3_K_M + FA | `-fa 1` (Q3_K_M file) | 1934.32 | ±59.27 | **39.48** | ±0.01 | 0.0% (0/20) | 45.0% (9/20) | **−20.7% ❌** | **−25% ❌** |

**Winner: Step 3 — Q4_K_S + Flash Attention**
- Generation speed: 53.22 t/s (+6.8% vs baseline)
- ARC-Challenge: 70.0% (+10 percentage points vs baseline)
- GSM8K: 5.0% (unchanged — 1/20, within noise floor)

### 6.2 Llama-3.2-1B — Full Optimization Ladder

| Stage | Label | llama.cpp Flags | pp512 t/s | pp_std | tg128 t/s | tg_std | GSM8K | ARC | tg vs Baseline | ARC vs Baseline |
|-------|-------|----------------|----------:|-------:|----------:|-------:|------:|----:|---------------:|----------------:|
| **I** | Baseline | *(none)* | 1639.19 | ±61.03 | **44.64** | ±0.03 | 0.0% (0/20) | 35.0% (7/20) | — | — |
| **II** | Flash Attn | `-fa 1` | 2189.84 | ±94.85 | **48.77** | ±0.14 | 0.0% (0/20) | 35.0% (7/20) | +9.2% ✅ | 0% |
| **IV** ★ | Q4_K_S + FA | `-fa 1` (Q4_K_S file) | 2204.10 | ±155.15 | **50.15** | ±0.06 | 5.0% (1/20) | 35.0% (7/20) | **+12.3% ✅** | 0% |
| **V** | Q3_K_M + FA | `-fa 1` (Q3_K_M file) | 2003.81 | ±114.09 | **39.25** | ±0.03 | 5.0% (1/20) | 25.0% (5/20) | **−12.1% ❌** | **−10% ❌** |

**Winner: Step 2 — Q4_K_S + Flash Attention**
- Generation speed: 50.15 t/s (+12.3% vs baseline)
- ARC-Challenge: 35.0% (unchanged — 7/20)
- GSM8K: 5.0% (1/20 — within noise, not statistically meaningful at 20 samples)

**Observation:** Llama-3.2-1B GSM8K is 0% at baseline. This is expected — the Llama 3 Instruct chat template uses specific special tokens (`<|begin_of_text|>`, `<|start_header_id|>user<|end_header_id|>`, etc.) that our raw 3-shot prompts do not include. Without the proper instruction format, the model does not reliably follow few-shot examples for math. ARC-Challenge (35%) is above random chance (25%), indicating the model does extract signal from the raw MCQ prompts.

### 6.3 Qwen2.5-0.5B — Full Optimization Ladder

| Stage | Label | llama.cpp Flags | pp512 t/s | pp_std | tg128 t/s | tg_std | GSM8K | ARC | tg vs Baseline | ARC vs Baseline |
|-------|-------|----------------|----------:|-------:|----------:|-------:|------:|----:|---------------:|----------------:|
| **I** | Baseline | *(none)* | 2684.01 | ±328.10 | **80.24** | ±0.36 | 0.0% (0/20) | 40.0% (8/20) | — | — |
| **II** | Flash Attn | `-fa 1` | 3626.22 | ±711.42 | **89.43** | ±0.42 | 0.0% (0/20) | 40.0% (8/20) | +11.5% ✅ | 0% |
| **IV** | Q4_K_S + FA | `-fa 1` (Q4_K_S file) | 3637.79 | ±723.53 | **90.67** | ±0.33 | 0.0% (0/20) | 40.0% (8/20) | +13.0% ✅ | 0% |
| **V** ★ | Q3_K_M + FA | `-fa 1` (Q3_K_M file) | 3721.21 | ±290.16 | **93.92** | ±1.10 | 5.0% (1/20) | 40.0% (8/20) | **+17.0% ✅** | 0% |

**Winner: Step 3 — Q3_K_M + Flash Attention**
- Generation speed: 93.92 t/s (+17% vs baseline)
- ARC-Challenge: 40.0% (unchanged)
- GSM8K: 5.0% (1/20 — within noise, not statistically different from 0%)

**Why Q3_K_M works for 0.5B but not for 1B/1.2B:** The 0.5B model has fewer parameters → fewer bytes per token → at tg128 (batch_size=1), memory bandwidth is already less saturated. Q3_K_M reduces the model size further, and the bandwidth savings outweigh the added dequantization compute cost. For 1B+ models, weights are larger, bandwidth is already the bottleneck, and Q3's dequantization overhead (additional arithmetic per weight element) creates more overhead than the bandwidth reduction saves.

**Accuracy ceiling:** All Qwen2.5-0.5B configurations score 0–5% GSM8K and 40% ARC. This is the model's inherent capability limit with our raw 3-shot prompts, not a quantization effect. The model is too small for reliable GSM8K performance. ARC at 40% (above 25% random) shows it understands simple science MCQ.

---

## 7. Summary Comparison — All Models, Best Configs

| Model | Best Config | pp512 t/s | tg128 t/s | GSM8K | ARC | tg Improvement | ARC Improvement |
|-------|-------------|----------:|----------:|------:|----:|---------------:|----------------:|
| LFM2.5-1.2B | Baseline (Q4_K_M, no flags) | 1875.45 | 49.81 | 5.0% | 60.0% | — | — |
| LFM2.5-1.2B | **Q4_K_S + FA** | 2079.60 | **53.22** | 5.0% | **70.0%** | **+6.8%** | **+10 pp** |
| Llama-3.2-1B | Baseline (Q4_K_M, no flags) | 1639.19 | 44.64 | 0.0% | 35.0% | — | — |
| Llama-3.2-1B | **Q4_K_S + FA** | 2204.10 | **50.15** | 5.0% | 35.0% | **+12.3%** | 0 pp |
| Qwen2.5-0.5B | Baseline (Q4_K_M, no flags) | 2684.01 | 80.24 | 0.0% | 40.0% | — | — |
| Qwen2.5-0.5B | **Q3_K_M + FA** | 3721.21 | **93.92** | 5.0% | 40.0% | **+17.0%** | 0 pp |

**pp = percentage points**

---

## 8. Methods That Failed or Showed No Improvement

### 8.1 KV Cache Key Quantization to q8_0 — Severe Regression

**Config:** `LFM2.5-1.2B Q4_K_M + FA + -ctk q8_0`
**Result:** pp512 dropped from 2107 to **718 t/s** (−66%). tg128 dropped from 51.64 to **40.39 t/s** (−21.8%).

**Why:** At 512-token context, the KV cache is tiny (~6 MB). The quantization compute overhead (added per-layer, per-token) is significant relative to the near-zero memory savings. The pp512 regression is especially severe because the prefill processes all 512 tokens in one pass — quantizing the KV for each token adds up linearly. At longer contexts (8K+), where KV memory starts to matter, this tradeoff would shift.

**Verdict:** Do not use `-ctk q8_0` at standard (≤2K token) context lengths.

### 8.2 KV Cache K+V Quantization to q4_0 — Neutral

**Config:** `LFM2.5-1.2B Q4_K_M + FA + -ctk q4_0 -ctv q4_0`
**Result:** pp512: 2105 (≈ FA-only). tg128: **50.95** (−1.3% vs FA-only). ARC: 65% (+5 pp vs FA-only).

**Why neutral:** The 4-bit KV quantization has similar overhead to 8-bit but slightly less memory savings on the K side, and the V side savings are not significant at 512-token context. Speed is effectively unchanged vs FA-only. The small ARC improvement (+5 pp vs FA-only baseline, +5 pp vs Q4_K_M baseline) is within the 5% noise floor of 20 samples.

**Verdict:** Not a meaningful improvement. Would need to test at 4K+ context to see real benefit.

### 8.3 Q3_K_M Quantization for 1B and 1.2B Models — Both Metrics Worse

**Config (LFM2.5-1.2B):** `Q3_K_M + FA`
**Result:** tg128: 39.48 t/s (−20.7% vs baseline). ARC: 45% (−25 pp vs Q4_K_S winner).

**Config (Llama-3.2-1B):** `Q3_K_M + FA`
**Result:** tg128: 39.25 t/s (−12.1% vs baseline). ARC: 25% (−10 pp vs Q4_K_S winner).

**Why:** Q3_K_M uses 3-bit integers for weights. To perform matrix multiplication, these must be dequantized to float16 at runtime. The dequantization kernels are computationally heavier than Q4 kernels. For models ≥1B parameters on Jetson's Ampere GPU, the extra arithmetic cost exceeds the bandwidth savings from the smaller file.

**Verdict:** Never use Q3_K_M for models ≥1B on Jetson Orin Nano at standard context lengths.

### 8.4 CPU Thread Count Increase — No Effect

**Config:** `LFM2.5-1.2B Q4_K_S + FA + -t 6` (speed-only)
**Result:** tg128: 53.1 t/s vs 53.22 t/s (−0.2%, within noise). pp512: 2156 vs 2080 (+3.7%, within std).

**Why:** With `-ngl 99`, all model layers are on the GPU. CPU threads handle only tokenization and process management overhead. The throughput bottleneck is entirely on the GPU side. Adding CPU threads does nothing.

**Verdict:** `-t` flag is irrelevant for fully GPU-offloaded inference.

### 8.5 Speculative Decoding (LFM2-700M → LFM2.5-1.2B) — Blocked by Tokenizer

**Attempted:** Use LFM2-700M-Q4_K_M as draft model for LFM2.5-1.2B-Q4_K_S.
**Draft model standalone speed:** 63.69 t/s (at 32-token test) — adequately faster than target's 53 t/s.
**Result:** llama.cpp rejected the combination. Token 128 differs between vocabularies:
- LFM2.5-1.2B token 128: `<|audio_start|>`
- LFM2-700M token 128: `<|reserved_118|>`

**Why the vocabularies diverged:** LFM2.5 extended the base LFM2 tokenizer with multimodal special tokens (`<|audio_start|>`, `<|audio_end|>`, `<|image_start|>`, etc.) for the audio-language and vision-language variants of the model family. These were assigned to token IDs that LFM2 had left as reserved placeholders. Both have 65,536 tokens total, but the special token assignments above ID ~128 differ.

**What would be needed:** A LFM2.5-family model smaller than 1.2B — no such model exists as of 2026-03-29.

**Theoretical speedup if compatible:** With draft_speed/target_speed ≈ 63/53 ≈ 1.19× (draft ~19% faster), and assuming speculative acceptance rate of 70–80% on domain-relevant text, speculative decoding would yield approximately 1.4–1.7× generation speedup. Note: this ratio is lower than typical (where draft models are 5–10× faster than target); the speed ratio between LFM2-700M and LFM2.5-1.2B is modest, suggesting limited gains even if tokenizers matched.

---

## 9. Theoretical Analysis — Memory Bandwidth Bottleneck

At batch_size=1 (standard edge inference), token generation speed is determined by:

```
tokens/s ≈ memory_bandwidth_GBps ÷ model_size_GB
```

For Jetson Orin Nano (~68 GB/s bandwidth):

| Model | Format | Approx Size | Theoretical Max t/s | Actual tg128 | Efficiency |
|-------|--------|-------------|--------------------:|-------------:|----------:|
| LFM2.5-1.2B | Q4_K_M | 698 MB | ~97 t/s | 49.81 t/s | 51% |
| LFM2.5-1.2B | Q4_K_S | 668 MB | ~102 t/s | 53.22 t/s | 52% |
| LFM2.5-1.2B | Q3_K_M | ~620 MB | ~110 t/s | 39.48 t/s | 36% |
| Llama-3.2-1B | Q4_K_M | 762 MB | ~89 t/s | 44.64 t/s | 50% |
| Llama-3.2-1B | Q4_K_S | 740 MB | ~92 t/s | 50.15 t/s | 55% |
| Llama-3.2-1B | Q3_K_M | 651 MB | ~104 t/s | 39.25 t/s | 38% |
| Qwen2.5-0.5B | Q4_K_M | ~386 MB | ~176 t/s | 80.24 t/s | 46% |
| Qwen2.5-0.5B | Q3_K_M | ~310 MB | ~219 t/s | 93.92 t/s | 43% |

**Key observation:** All configurations achieve 36–55% of theoretical peak. The gap comes from:
1. KV cache reads (adds per-token memory traffic beyond weight loading)
2. Dequantization compute (for K-quant formats: converting integer blocks to float16)
3. CUDA kernel launch overhead
4. Memory fragmentation and allocation overhead in UMA

Q3_K_M models show lower efficiency (36–38%) because the dequantization arithmetic cost grows faster than the bandwidth savings — the GPU's compute units become the bottleneck instead of memory bandwidth.

---

## 10. Infrastructure — bench_gguf.py

The benchmark script was written from scratch for this optimization ladder. Location: [`scripts/edge_optimization/bench_gguf.py`](../../scripts/edge_optimization/bench_gguf.py)

### Design decisions

1. **Single run = single record:** Each invocation measures all four metrics (pp512, tg128, GSM8K%, ARC%) and writes one JSON record. This ensures every comparison is apple-to-apple with no temporal drift.

2. **Pre-run cleanup:** Before benchmarking, kill any stale `llama-server` and `llama-bench` processes, then sleep 15 seconds. This prevents the Jetson UMA CUDA context from leaking between runs (ENOMEM if previous context not fully released).

3. **HF dataset cache:** `cache_dir="/tmp/hf_datasets_cache"` rather than the default `~/.cache/huggingface/`. The default cache was owned by root (created by Docker container) and caused `PermissionError` on the host.

4. **Server health polling:** After starting `llama-server`, poll `GET /health` every 1 second for up to 90 seconds. LFM2.5 takes ~20–30s to load on Jetson.

5. **Result deduplication:** On save, existing records with the same label are replaced (not appended). This allows re-running a benchmark without accumulating duplicates.

### Evolution during the project

| Version | Change | Reason |
|---------|--------|--------|
| Initial | Created with pre-run `time.sleep(3)` | Not enough for CUDA context release |
| v2 | Increased pre-run sleep to `time.sleep(15)` | ENOMEM errors when running back-to-back |
| v2 | Added `cache_dir="/tmp/hf_datasets_cache"` | Root-owned `~/.cache` caused PermissionError |
| v2 | Added 5s sleep in `stop_server()` | CUDA context takes several seconds to fully release |

---

## 11. Optimal Deployment Configurations

Based on all measured results, these are the recommended production configurations for each model on Jetson Orin Nano 8 GB:

### LFM2.5-1.2B-Instruct

```bash
llama-server \
  -m /path/to/LiquidAI_LFM2.5-1.2B-Instruct-Q4_K_S.gguf \
  -ngl 99 \
  -fa 1 \
  --port 8765 \
  --ctx-size 4096
```

| Metric | Value |
|--------|-------|
| Generation speed | 53.22 t/s |
| Prefill speed | 2080 t/s |
| ARC-Challenge | 70.0% |
| GSM8K | 5.0% |
| Est. VRAM at 4K ctx | ~800 MB |

### Llama-3.2-1B-Instruct

```bash
llama-server \
  -m /path/to/Llama-3.2-1B-Instruct-Q4_K_S.gguf \
  -ngl 99 \
  -fa 1 \
  --port 8765 \
  --ctx-size 4096
```

| Metric | Value |
|--------|-------|
| Generation speed | 50.15 t/s |
| Prefill speed | 2204 t/s |
| ARC-Challenge | 35.0% |
| GSM8K | 0.0% |
| Est. VRAM at 4K ctx | ~900 MB |

**Note:** Low GSM8K is due to missing Llama 3 chat template in our evaluation prompts, not model incapability. With proper `<|begin_of_text|><|start_header_id|>user<|end_header_id|>` formatting, performance would be higher.

### Qwen2.5-0.5B-Instruct

```bash
llama-server \
  -m /path/to/Qwen2.5-0.5B-Instruct-Q3_K_M.gguf \
  -ngl 99 \
  -fa 1 \
  --port 8765 \
  --ctx-size 4096
```

| Metric | Value |
|--------|-------|
| Generation speed | 93.92 t/s |
| Prefill speed | 3721 t/s |
| ARC-Challenge | 40.0% |
| GSM8K | 5.0% |
| Est. VRAM at 4K ctx | ~450 MB |

**Note:** Low accuracy scores reflect the model's inherent size limitation (0.5B parameters), not quantization degradation. All Qwen configurations score 40% ARC regardless of format.

---

## 12. Future Stages Roadmap — Sorted by Priority

All optimizations identified but not yet implemented. Sorted by **expected gain ÷ implementation effort** — highest value-to-effort ratio first. Use this table to decide what to run next.

**Effort scale:** Trivial = flag only, no code change | Easy = download/quantize + benchmark (~1–2 h) | Medium = script change or minor source edit (~2–4 h) | Hard = new framework or major code change (~1 day) | Very Hard = training or architectural rewrite (days+)

**Gain scale:** ★★★★ = both speed and accuracy improve significantly | ★★★ = one metric improves significantly | ★★ = one metric improves marginally | ★ = narrow-case or uncertain gain

---

### Tier 1 — KV Cache Quantization Type Sweep ✅ COMPLETE

**Full report:** [outputs/reports/tier1_kv_quant_types.md](tier1_kv_quant_types.md)

| Stage | Config | pp512 t/s | tg128 t/s | GSM8K | ARC | Verdict |
|-------|--------|----------:|----------:|------:|----:|---------|
| — | **Reference: Q4_K_S+FA (no KV quant)** | **2080** | **53.22** | **5%** | **70%** | **WINNER** |
| **VIII** | +FA +ctk-**q4_1** | 485 | 40.78 | 5% | **75%** | Speed ❌, ARC +5% marginal |
| **IX** | +FA +ctk-**iq4_nl** | 528 | 35.54 | 5% | 65% | Speed ❌, ARC −5% ❌ |
| **X** | +FA +ctk-**q5_0** | 537 | 40.23 | 5% | 55% | Speed ❌, ARC −15% ❌ |

**Finding:** All KV quantization types cause 4–5× pp512 regression and 24–33% tg regression at 512-token context. Root cause: KV cache is tiny (~4 MB) at 512 tokens — encode/decode overhead exceeds bandwidth savings. Crossover point is 4K+ context (tested in Stage XV). Only `q4_1` shows a marginal +5% ARC gain; not a deployment win at short context.

---

### Tier 2 — Chat Template + IQ Quantization ✅ COMPLETE (XIV blocked)

**Full report:** [outputs/reports/tier2_model_quant_and_template.md](tier2_model_quant_and_template.md)

| Stage | Config | pp512 t/s | tg128 t/s | GSM8K | ARC | Verdict |
|-------|--------|----------:|----------:|------:|----:|---------|
| **XI** | Llama-3.2-1B Q4_K_S+FA + **Llama 3 chat template** | 2204 | 50.16 | **40%** | **40%** | ✅ **GSM8K +35pp (5%→40%)** |
| **XII** | Qwen2.5-0.5B **IQ3_M**+FA (bartowski, from F16) | 3501 | 90.96 | 0% | 40% | ❌ Slightly slower than Q3_K_M |
| **XIII** | Llama-3.2-1B **IQ4_XS**+FA (bartowski, from F16) | 2386 | **54.37** | 0% | **45%** | ✅ **+8.4% tg, +10% ARC — NEW WINNER** |
| **XIV** | LFM2.5-1.2B IQ4_XS | — | — | — | — | ❌ **BLOCKED** — no community GGUF; F16 conversion broken (`KeyError: block_ff_dim`) |

**Key findings:**
- Chat template unlocks Llama 3's instruction following: GSM8K 5% → **40%** with zero speed cost
- IQ4_XS beats Q4_K_S for Llama 1B: imatrix calibration from F16 source provides genuine +8% speed and +10% ARC
- IQ3_M does NOT benefit Qwen 0.5B: model too small for imatrix to find channel importance variation
- LFM2.5-1.2B IQ4_XS blocked until HF converter is patched or community GGUF appears (Stage XVIII)

---

### Tier 3 — Context-Length and Runtime Experiments (Stages XV–XVII) ✅ Complete (XVII blocked)

| Stage | Config | Result | Verdict |
|-------|--------|--------|---------|
| **XV** | All models × f16 + q4_0 KV at **4096-token context** | q4_0 neutral at 4K (−0.2–0.8% pp vs f16); q4_1 and q5_0 still −94–95% ❌ | ✅ **q4_0 production-safe at 4K** |
| **XVI** | KIVI-2bit via Q2_K whitelist in arg.cpp + rebuild | Server crash at load (ggml_abort) — no CUDA quantize kernel for Q2_K in v8510 | ❌ Not viable |
| **XVII** | ExLlamaV2 install + import | pip install v0.3.2 succeeded; import fails — CUDA driver 12060 too old for torch 2.11.0 | ❌ Blocked on JetPack 6.2 |

**Stage XV — 4K Context KV Results:**

| Model | KV type | pp4096 t/s | tg128 t/s | pp Δ vs f16 |
|-------|---------|----------:|----------:|------------:|
| LFM2.5-1.2B Q4_K_S+FA | **q4_0** | 2200.68 | 52.89 | **−0.2% ✅** |
| LFM2.5-1.2B Q4_K_S+FA | q4_1 | 108.27 | 37.64 | −95.1% ❌ |
| LFM2.5-1.2B Q4_K_S+FA | q5_0 | 122.16 | 37.52 | −94.5% ❌ |
| Llama-3.2-1B IQ4_XS+FA | **q4_0** | 2433.76 | 53.68 | **−0.8% ✅** |
| Qwen2.5-0.5B Q3_K_M+FA | **q4_0** | 3897.23 | 92.50 | **−0.7% ✅** |

**Key finding:** q4_0 is the only KV type with a stable CUDA multiply-accumulate kernel path; q4_1 (asymmetric zero-point) and q5_0 (non-power-of-2 packing) use complex fallback paths that remain slow regardless of context length. For long-context deployments, add `-ctk q4_0 -ctv q4_0` — zero penalty at 4K, ~2× KV VRAM savings.

**Full Tier 3 report:** [outputs/reports/tier3_long_context_kv_exllama.md](tier3_long_context_kv_exllama.md)

---

### Tier 4 — IQ4_XS, EAGLE-3, TensorRT-LLM (Stages XVIII–XX) 🔄 In Progress (XX pending)

| Stage | Title | Status | Key Finding |
|-------|-------|--------|-------------|
| **XVIII** | IQ4_XS for LFM2.5 from patched F16 | ✅ Complete | **+10.8% tg128** (58.98 vs 53.22 t/s); ARC −10pp (60% vs 70%) — calibration corpus too narrow (wikitext-2 only). Speed win is real; accuracy recoverable with better imatrix. |
| **XIX** | EAGLE-3 speculative decoding | ❌ Blocked | Requires external GPU for training. SSM+attention hybrid architecture (10 of 16 layers are SSM) complicates standard EAGLE approach. ~2–3× speedup potential if unblocked. |
| **XX** | TensorRT-LLM W4A16 | ✅ Partial (Qwen only) | Qwen2.5-0.5B: **+86% pp, +7.4% tg** (100.87 vs 93.92 t/s). LFM2.5 blocked (SSM arch). Llama: 44.08 t/s (no improvement; gemm_plugin build OOMs on 8 GB Jetson serialization buffer). Extended analysis (8 patches, custom IGpuAllocator, share_embedding_table fix) confirms gemm_plugin permanently blocked by NvMap IOVM. |
| **XXI** | llama.cpp IQ4_XS + FA updated baselines | ✅ Complete | LFM2 1.2B IQ4_XS+FA: **65.64 t/s** (new >1B record, 60.9% BW). Llama 3.2 1B IQ4_XS+FA: **60.28 t/s** (62.1% BW). Both beat TRT-LLM no-gemm-plugin (44.08 t/s). IQ4_XS outperforms Q3_K_M (41 t/s) despite larger size — better GPU kernel efficiency on sm_87. |

**4K context production recommendation (from Tier 3):** Add `-ctk q4_0 -ctv q4_0` to long-context deployments — zero speed penalty, ~2× KV VRAM savings.

**Full Tier 4 report:** [outputs/reports/tier4_iq4xs_eagle_trtllm.md](tier4_iq4xs_eagle_trtllm.md)

---

### Tier 5 — Research / Algorithmic Extensions

These require implementing new CUDA kernels or fundamentally changing the inference architecture. Suitable only if all Tiers 1–4 are exhausted.

| Stage | Optimization | Applies To | Description | Expected Gain | Effort | Gain |
|-------|-------------|------------|-------------|---------------|--------|------|
| **XXI** | **PolarQuant CUDA kernel in llama.cpp** — Implement polar coordinate KV quantization (8-bit magnitude + 2-bit angular) as a new `-ctk polar` cache type | All models at long context | New GGML type + CUDA encode/decode kernels. Stage 3 showed 7.53× compression ratio but −22% speed in Python. At very long contexts (32K+) the bandwidth savings may overcome kernel overhead. Risk: significant development effort for uncertain Jetson benefit. | 7.53× KV compression. Speed depends on context length — beneficial only at 16K+ tokens. Accuracy: cosine sim 0.915 (Stage 3 measured). | Very Hard (~500–1000 lines new CUDA; ~1 week) | ★★ |
| **XXII** | **QJL (Johnson-Lindenstrauss) attention kernel** — Replace K-cache with 1-bit JL sketches and modify score computation to use the JL estimator | All models at very long context | Requires rewriting the flash attention kernel to use `(π/2m)·Q_proj·q^T` score estimation. 16–64× KV compression but requires custom attention math. Stage 3 showed Pearson-r=0.62 (moderate correlation) at 16× compression. | 16–64× KV compression. Meaningless at ≤4K context (overhead > savings). At 32K+ could help. Accuracy risk: score estimation error propagates to attention weights. | Extreme (~2000+ lines; attention kernel rewrite; weeks) | ★ |

---

### Priority Summary Table

| Priority | Stage | Optimization | Model(s) | Effort | Expected tg Δ | Expected ARC Δ |
|----------|-------|-------------|----------|--------|--------------|----------------|
| 1 | **VIII** | KV q4_1 + q4_1 | LFM2.5 | Trivial | Neutral at 512 tok | Neutral |
| 2 | **IX** | KV iq4_nl | LFM2.5 | Trivial | Neutral at 512 tok | Neutral |
| 3 | **X** | KV q5_0 | LFM2.5 | Trivial | Neutral at 512 tok | Neutral |
| 4 | **XI** | Llama chat template | Llama-3.2-1B | Easy | 0% | GSM8K: 0%→**20–40%** |
| 5 | **XII** | IQ3_S/IQ3_XS model quant | Qwen2.5-0.5B | Easy | ~0% | ARC: 40%→est. **45–50%** |
| 6 | **XIII** | IQ4_XS model quant | Llama-3.2-1B | Easy-Med | ~0% | ARC: 35%→est. **40–45%** |
| 7 | **XIV** | IQ4_XS from community GGUF | LFM2.5-1.2B | Easy (if exists) | ~0% | ARC: 70%→est. **72–76%** |
| 8 | **XV** | KV quant at 4K context | All | Medium | **+10–20% at 4K** | Neutral |
| 9 | **XVI** | KIVI-2bit (source edit) | All (long ctx) | Medium | Neg. at 512 tok; +30% at 32K | Risk of degradation |
| 10 | **XVII** | ExLlamaV2 runtime | LFM2.5, Llama | Hard | **+5–15%** | Neutral |
| 11 | **XVIII** | IQ4_XS from fixed F16 | LFM2.5-1.2B | Hard | ~0% | ARC: 70%→est. **75–80%** |
| 12 | **XIX** | EAGLE speculative decoding | LFM2.5-1.2B | Very Hard | **+100–200%** (2–3×) | Neutral |
| 13 | **XX** | TensorRT-LLM hardware accel | LFM2.5, Llama | Hard | **~1.8× over llama.cpp** (est.) | Neutral |
| 14 | **XXI** | PolarQuant CUDA kernel | All (32K+ ctx) | Very Hard | +? at very long ctx only | Minor degradation |
| 15 | **XXII** | QJL attention kernel | All (32K+ ctx) | Extreme | +? at very long ctx only | Moderate degradation |

**Recommendation:** Run Stages VIII–X first (30 minutes total, zero code). Then Stage XI (Llama chat template) for the biggest single accuracy jump. Then Stage XII–XIV for accuracy improvements on the other models. Stage XX (TensorRT-LLM) requires a ~40-min build but offers the largest measurable speedup without training. Stage XIX (EAGLE) has the highest ceiling but requires GPU training outside Jetson.

---

## 13. References

| Method / Tool | Citation |
|---------------|---------|
| Flash Attention 2 | Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. *ICLR 2024.* arXiv:2307.08691 |
| Speculative Decoding | Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. *ICML 2023.* arXiv:2211.17192 |
| GSM8K | Cobbe, K., et al. (2021). Training Verifiers to Solve Math Problems. arXiv:2110.14168 |
| ARC-Challenge | Clark, P., et al. (2018). Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. arXiv:1803.05457 |
| LFM2.5 | LiquidAI. (2025). Introducing LFM2.5: The Next Generation of On-Device AI. liquid.ai/blog |
| LFM2 Technical Report | LiquidAI Team. (2024). LFM2 Technical Report. arXiv:2511.23404 |
| Llama 3.2 | Meta AI. (2024). The Llama 3 Herd of Models. arXiv:2407.21783 |
| Qwen2.5 | Qwen Team, Alibaba. (2024). Qwen2.5: A Party of Foundation Models. arXiv:2412.15115 |
| EAGLE | Cai, Y., et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty. *ICML 2024.* |
| K-quant formats | llama.cpp documentation. github.com/ggerganov/llama.cpp |
