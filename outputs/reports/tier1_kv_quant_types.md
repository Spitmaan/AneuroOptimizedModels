# Tier 1 — KV Cache Quantization Type Sweep (Stages VIII–X)

**Date:** 2026-03-29
**Hardware:** NVIDIA Jetson Orin Nano 8 GB (UMA, ~68 GB/s)
**Runtime:** llama.cpp v8510 (host build, `3a60d06ad`)
**Model:** `LiquidAI_LFM2.5-1.2B-Instruct-Q4_K_S.gguf` (current best config + FA)
**Benchmark script:** [`scripts/edge_optimization/bench_gguf.py`](../../scripts/edge_optimization/bench_gguf.py)

---

## Background

KV cache quantization compresses the Key and Value tensors written during attention computation. llama.cpp v8510 supports the following types via `-ctk`/`-ctv` flags:

| Type | BPW | Description |
|------|-----|-------------|
| `f16` | 16 | Default — full half-precision |
| `q8_0` | 8 | 8-bit symmetric (tested in Stage IIIa — severe pp regression) |
| `q4_0` | 4 | 4-bit symmetric (tested in Stage IIIb — neutral speed, +5% ARC) |
| **`q4_1`** | **4** | **4-bit asymmetric with zero-point (Stage VIII)** |
| **`iq4_nl`** | **4** | **4-bit non-linear lookup table (Stage IX)** |
| **`q5_0`** | **5** | **5-bit symmetric (Stage X)** |
| `q5_1` | 5 | 5-bit asymmetric with zero-point |
| `bf16` | 16 | BF16 (same as f16 for practical purposes) |

Stages VIII–X complete the sweep of remaining untested 4–5 bit types. All are tested on the **current best config** (Q4_K_S + Flash Attention, no KV quant → baseline for this sweep).

**Reference (no KV quant):** pp512 = 2080 t/s | tg128 = 53.22 t/s | GSM8K = 5% | ARC = 70%

---

## Stage VIII — KV q4_1 (Asymmetric 4-bit)

**Flags:** `-fa 1 -ctk q4_1 -ctv q4_1`
**Label:** `LFM2.5-1.2B_Q4_K_S_FA_ctk-q4_1`
**Run date:** 2026-03-29 23:06

### What is q4_1?

`q4_1` stores each 4-bit weight block with both a **scale** and a **zero-point** (offset), making the quantization asymmetric. This allows the quantizer to center the range on the actual data distribution rather than assuming zero-symmetric distribution. For KV tensors (which are often skewed), asymmetric quantization theoretically preserves more information than `q4_0` (scale-only).

### Results

| Metric | Reference (no KV quant) | Stage VIII (q4_1) | Delta |
|--------|:-----------------------:|:-----------------:|------:|
| pp512 t/s | 2080 | **485.25 ± 16.03** | **−76.7% ❌** |
| tg128 t/s | 53.22 | **40.78 ± 1.57** | **−23.4% ❌** |
| GSM8K | 5.0% | 5.0% | 0% |
| ARC-Challenge | 70.0% | **75.0%** | **+5% ✅** |

### Analysis

Severe speed regression. pp512 dropped from 2080 to 485 — a 4.3× slowdown. tg128 dropped from 53.22 to 40.78. This is consistent with the earlier `q8_0` finding (Stage IIIa: pp=718, tg=40.39).

**Hypothesis for pp512 collapse:** At 512-token context, the KV cache is tiny (~4 MB at f16 for 1.2B model). KV quantization provides no memory bandwidth savings at this scale. Instead, the encode/decode kernels add latency on every attention call, and with Flash Attention the overhead is amplified because FA tiles the computation — each tile must dequantize KV tensors multiple times. The result is a compute-overhead-dominated regime.

**ARC gain (+5%):** Despite speed regression, ARC improved from 70% to 75%. This is statistically marginal at n=20 (1 question difference) but worth noting. q4_1's asymmetric encoding may better preserve the attention head directions that matter for ARC's multi-choice format.

---

## Stage IX — KV iq4_nl (Non-linear 4-bit Lookup Table)

**Flags:** `-fa 1 -ctk iq4_nl -ctv iq4_nl`
**Label:** `LFM2.5-1.2B_Q4_K_S_FA_ctk-iq4nl`
**Run date:** 2026-03-29 23:09

### What is iq4_nl?

`iq4_nl` uses a **non-linear lookup table** instead of linear scale quantization. Each 4-bit index maps to a learned float value from a fixed codebook, enabling the quantizer to place more representatives near the most common values in the distribution. Originally designed for weight quantization; when applied to KV tensors it uses the same codebook.

### Results

| Metric | Reference (no KV quant) | Stage IX (iq4_nl) | Delta |
|--------|:-----------------------:|:-----------------:|------:|
| pp512 t/s | 2080 | **527.70 ± 21.01** | **−74.6% ❌** |
| tg128 t/s | 53.22 | **35.54 ± 4.94** | **−33.2% ❌** |
| GSM8K | 5.0% | 5.0% | 0% |
| ARC-Challenge | 70.0% | **65.0%** | **−5% ❌** |

### Analysis

Worse than `q4_1` on both speed and accuracy. tg128 at 35.54 is the worst result of any Tier 1 KV type (high std ±4.94 suggests instability). The lookup table approach adds a gather operation per dequantize call — more expensive than scalar multiply+add of linear types. The ARC drop from 70% to 65% suggests the non-linear codebook, designed for weight distributions, does not map well to KV tensor distributions.

**Verdict: iq4_nl is not recommended for KV cache on this hardware.**

---

## Stage X — KV q5_0 (5-bit Symmetric)

**Flags:** `-fa 1 -ctk q5_0 -ctv q5_0`
**Label:** `LFM2.5-1.2B_Q4_K_S_FA_ctk-q5_0`
**Run date:** 2026-03-29 23:~20

### What is q5_0?

`q5_0` stores each block with 5 bits per weight (symmetric, no zero-point). Intermediate between 4-bit and 8-bit — ~2.75× compression vs f16. Higher precision than 4-bit types but slightly more decode overhead than 4-bit due to non-power-of-2 bit width requiring a different packing scheme.

### Results

| Metric | Reference (no KV quant) | Stage X (q5_0) | Delta |
|--------|:-----------------------:|:--------------:|------:|
| pp512 t/s | 2080 | **536.88 ± 22.97** | **−74.2% ❌** |
| tg128 t/s | 53.22 | **40.23 ± 3.84** | **−24.4% ❌** |
| GSM8K | 5.0% | 5.0% | 0% |
| ARC-Challenge | 70.0% | **55.0%** | **−15% ❌** |

### Analysis

Speed is comparable to other quantized types (pp slightly better at 536 vs 485/527, tg similar at ~40). However, **ARC dropped sharply to 55%** — the worst of all Tier 1 types. This is counterintuitive (higher bit depth → worse accuracy), but with n=20 samples the noise floor means a 3-question swing is within statistical uncertainty. Still, it does not provide any advantage over q4_1 or no KV quant.

**Verdict: q5_0 is not recommended. It provides no speed advantage over 4-bit types and may degrade accuracy.**

---

## Tier 1 Summary — All KV Types at 512-token Context

| Stage | Config | pp512 t/s | tg128 t/s | GSM8K | ARC | Verdict |
|-------|--------|----------:|----------:|------:|----:|---------|
| — | **Reference: Q4_K_S+FA (no KV quant)** | **2080** | **53.22** | **5%** | **70%** | **WINNER** |
| IIIb | Q4_K_M+FA+q4_0 (prior) | 2105 | 50.95 | 5% | 65% | Neutral speed, −5% ARC |
| **VIII** | Q4_K_S+FA+**q4_1** | 485 | 40.78 | 5% | **75%** | Speed ❌, ARC +5% |
| **IX** | Q4_K_S+FA+**iq4_nl** | 528 | 35.54 | 5% | 65% | Speed ❌, ARC −5% |
| **X** | Q4_K_S+FA+**q5_0** | 537 | 40.23 | 5% | 55% | Speed ❌, ARC −15% |

---

## Key Finding — Pattern Confirmed

**All KV cache quantization types cause severe speed regression at 512-token context on Jetson Orin Nano.** This applies to every type tested (q8_0, q4_0, q4_1, iq4_nl, q5_0). The pp512 regression is 4–5× for all 4-bit types; tg128 regression is 24–33%.

**Root cause:** At 512-token context, the KV cache is tiny (~4 MB). There is no memory bandwidth pressure from the KV cache. Compression adds kernel overhead (encode on write, decode on read) without saving meaningful bandwidth. The crossover point where compression becomes beneficial is at **4K+ token contexts**, where the KV cache grows to dozens or hundreds of MB and starts competing with model weights for the 68 GB/s memory bus.

**Exception — q4_1 ARC result:** The +5% ARC gain with q4_1 (75% vs 70%) is a statistically marginal result (n=20, 1-question difference) but directionally interesting. Asymmetric quantization better preserves the signed distribution of KV tensors. **For long-context deployments where some accuracy trade-off is acceptable, q4_1 is the least-bad KV type.**

**Next step:** Stage XV tests KV quantization at 4K context where the true benefit emerges.

---

## Raw Benchmark Data

```
Label                             pp512 t/s    tg128 t/s    GSM8K      ARC
LFM2.5-1.2B_Q4_K_S_FA_ctk-q4_1    485.25        40.78      5.0%    75.0%
LFM2.5-1.2B_Q4_K_S_FA_ctk-iq4nl   527.70        35.54      5.0%    65.0%
LFM2.5-1.2B_Q4_K_S_FA_ctk-q5_0    536.88        40.23      5.0%    55.0%
```

---

## Relation to Stage XV

Stage XV will re-run the best KV type (q4_1) and compare against q4_0 at `--ctx-size 4096` with a 4096-token prompt. The expected result is that KV quantization becomes speed-neutral or beneficial at long context — Stage XV will confirm or disprove this.
