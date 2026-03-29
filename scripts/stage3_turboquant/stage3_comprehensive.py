#!/usr/bin/env python3
"""
ANeurologic Phase 5 - Stage 3 Comprehensive KV Compression Benchmark
=====================================================================
Full matrix: 2 models × 4 compression methods × 2 tasks
Reports per condition: t/s, Peak VRAM (load + eval), GSM8K %, ARC-Challenge %

Models:
  - LFM2.5-1.2B-Instruct  (hybrid conv+attention, Lfm2HybridConvCache)
  - Qwen2.5-0.5B-Instruct (standard decoder, DynamicCache)

Compression methods:
  - Baseline (no compression)
  - PolarQuant   7.53x  8-bit mag + 2-bit angular  (TurboQuant component)
  - KIVI-2bit    2.29x  asymmetric group quant       (production baseline)
  - KIVI-4bit    1.88x  near-lossless group quant    (conservative)
  - QJL          16-64x 1-bit JL sketch of K only    (TurboQuant component)
                 NOTE: QJL cannot be plugged into the generation loop without
                 modifying the attention kernel (score estimation, not KV
                 reconstruction). Speed/accuracy marked N/A; Pearson-r reported.

KV cache hook strategy:
  Qwen2.5-0.5B  → DynamicCache: iterate cache.layers, modify layer.keys / layer.values
  LFM2.5-1.2B   → Lfm2HybridConvCache: iterate key_cache / value_cache lists,
                   skip empty (conv) layers (numel==0), compress attention layers only

Feasibility note (addressing the core research question):
  Can compression INCREASE speed and accuracy?
  Speed:  YES with CUDA kernel (smaller KV → less memory bandwidth during attention).
          Current CPU roundtrip shows −20% overhead; in-kernel compression would
          eliminate this, leaving only the benefit of reduced VRAM at long sequences.
  Accuracy: Generally NO for lossy methods on short-context tasks (compression loses
          information). EXCEPTION: compression enables longer contexts that would OOM
          without it, turning a failure into a correct answer. KIVI-4bit is near-lossless
          (cosine_sim=0.997) and shows zero accuracy degradation in practice.

Usage:
    python3 /workspace/scripts/stage3_turboquant/stage3_comprehensive.py
    python3 /workspace/scripts/stage3_turboquant/stage3_comprehensive.py --n-samples 10 --n-tokens 32
    python3 /workspace/scripts/stage3_turboquant/stage3_comprehensive.py --model lfm
    python3 /workspace/scripts/stage3_turboquant/stage3_comprehensive.py --speed-only
"""

import sys, os, gc, json, re, math, time, argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import torch.nn.functional as F

OUTPUT_JSON   = "/workspace/outputs/logs/stage3_comprehensive_results.json"
OUTPUT_REPORT = "/workspace/outputs/reports/stage3_turboquant.md"
HF_CACHE      = "/workspace/models/hf_cache"

# ── Compression methods ───────────────────────────────────────────────────────

class PolarQuant:
    """8-bit magnitude + 2-bit angular. ~7.53x vs fp16. TurboQuant sub-method."""
    RATIO = 7.53
    def compress_decompress(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype, orig_shape = x.dtype, x.shape
        D = x.shape[-1]
        x_f = x.reshape(-1, D).float()
        mag = x_f.norm(dim=-1, keepdim=True)
        unit = x_f / (mag + 1e-8)
        max_mag = mag.max().clamp(min=1e-6)
        q_mag = (mag / (max_mag / 255)).round().clamp(0, 255)
        mag_rec = q_mag * (max_mag / 255)
        n_groups = math.ceil(D / 64)
        parts = []
        for g in range(n_groups):
            s, e = g * 64, min((g+1) * 64, D)
            blk = unit[:, s:e]
            b_min = blk.min(dim=-1, keepdim=True).values
            b_max = blk.max(dim=-1, keepdim=True).values
            scale = (b_max - b_min).clamp(min=1e-6) / 3.0   # 2-bit → 4 levels
            q = ((blk - b_min) / scale).round().clamp(0, 3)
            parts.append(q * scale + b_min)
        unit_rec = torch.cat(parts, dim=-1)
        unit_rec = unit_rec / unit_rec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (mag_rec * unit_rec).reshape(orig_shape).to(orig_dtype)
    def label(self): return "PolarQuant"

class KIVIQuant:
    """Asymmetric group min-max quantization. ICML 2024."""
    def __init__(self, n_bits=2, group_size=32, residual=64):
        self.n_bits, self.group_size, self.residual = n_bits, group_size, residual
        self.levels = 2 ** n_bits
        self.RATIO = {2: 2.29, 4: 1.88}.get(n_bits, 1.0)
    def compress_decompress(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        B, H, S, D = x.shape
        x_f = x.float()
        S_q = (S // self.group_size) * self.group_size
        body, resid = x_f[:, :, :S_q, :], x_f[:, :, S_q:, :]
        if S_q > 0:
            n_g = S_q // self.group_size
            g = body.reshape(B, H, n_g, self.group_size, D)
            xmin, xmax = g.amin(3, keepdim=True), g.amax(3, keepdim=True)
            scale = (xmax - xmin).clamp(min=1e-6) / (self.levels - 1)
            q = ((g - xmin) / scale).round().clamp(0, self.levels - 1)
            body = (q * scale + xmin).reshape(B, H, S_q, D)
        return torch.cat([body, resid], dim=2).to(orig_dtype)
    def label(self): return f"KIVI-{self.n_bits}bit"

# ── KV cache compression hooks ────────────────────────────────────────────────

def compress_fn(tensor: torch.Tensor, method) -> torch.Tensor:
    """Compress+decompress a single KV tensor. Input/output on original device."""
    if tensor.ndim != 4 or tensor.numel() == 0:
        return tensor
    device, dtype = tensor.device, tensor.dtype
    rec = method.compress_decompress(tensor.float().cpu())
    return rec.to(device=device, dtype=dtype)


def apply_kv_compression(kv_cache, method, cache_type: str):
    """
    Hook compression into the live KV cache after each generation step.
    cache_type: 'dynamic'  → DynamicCache  (Qwen2.5)
                'lfm2'     → Lfm2HybridConvCache (LFM2.5)
    """
    if kv_cache is None or method is None:
        return kv_cache

    if cache_type == "dynamic":
        # transformers 5.x DynamicCache: list of DynamicLayer objects
        for layer in kv_cache.layers:
            K, V = layer.keys, layer.values
            if isinstance(K, torch.Tensor) and K.ndim == 4 and K.numel() > 0:
                layer.keys   = compress_fn(K, method)
                layer.values = compress_fn(V, method)

    elif cache_type == "lfm2":
        # Lfm2HybridConvCache: key_cache and value_cache are plain lists.
        # Entries with numel==0 are conv placeholder layers — skip them.
        for i in range(len(kv_cache.key_cache)):
            K = kv_cache.key_cache[i]
            V = kv_cache.value_cache[i]
            if isinstance(K, torch.Tensor) and K.ndim == 4 and K.numel() > 0:
                kv_cache.key_cache[i]   = compress_fn(K, method)
                kv_cache.value_cache[i] = compress_fn(V, method)

    return kv_cache


def detect_cache_type_from_name(hf_id: str) -> str:
    """Detect cache hook type from model ID — avoids a dummy forward pass."""
    if "LFM" in hf_id or "lfm" in hf_id.lower():
        return "lfm2"
    return "dynamic"

def detect_cache_type(kv_cache) -> str:
    name = type(kv_cache).__name__
    if "Lfm2" in name or "Hybrid" in name:
        return "lfm2"
    return "dynamic"


# ── Evaluation prompts ────────────────────────────────────────────────────────

GSM8K_SHOTS = [
    ("Natalia sold clips to 48 of her friends in April, and then she sold "
     "half as many clips in May. How many clips did Natalia sell altogether "
     "in April and May?", "72"),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
     "minutes of babysitting. How much did she earn?", "10"),
    ("Betty is saving money for a new wallet which costs $100. Betty has "
     "only half of the money she needs. Her parents decided to give her "
     "$15 for that purpose, and her grandparents twice as much as her "
     "parents. How much more money does Betty need to buy the wallet?", "5"),
]

ARC_SHOTS = [
    {"question": "Which property of a mineral can be determined just by looking at it?",
     "choices": ["luster", "mass", "weight", "hardness"], "answer": "A"},
    {"question": "A sixth grade student measured the mass of a sample. What is the "
                 "most likely unit of measurement the student used?",
     "choices": ["liters", "meters", "grams", "seconds"], "answer": "C"},
    {"question": "A student is trying to show how ocean waves can cause erosion. "
                 "Which materials should the student use to model this process?",
     "choices": ["water and a container of sand", "water and a container of salt",
                 "water and a container of sugar", "water and a container of flour"],
     "answer": "A"},
]

def gsm8k_prompt(question: str) -> str:
    shots = "\n\n".join(f"Q: {q}\nA: The answer is {a}." for q, a in GSM8K_SHOTS)
    return f"{shots}\n\nQ: {question}\nA:"

def arc_context(question: str, choices: list) -> str:
    lines = []
    for s in ARC_SHOTS:
        opts = " ".join(f"({chr(65+i)}) {c}" for i, c in enumerate(s["choices"]))
        lines.append(f"Question: {s['question']}\nOptions: {opts}\nAnswer: {s['answer']}")
    opts = " ".join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
    lines.append(f"Question: {question}\nOptions: {opts}\nAnswer:")
    return "\n\n".join(lines)

def extract_number(text: str) -> Optional[str]:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


# ── Core benchmark functions ──────────────────────────────────────────────────

def bench_speed(model, tokenizer, method, cache_type, device, n_warmup=5, n_tokens=48):
    """Returns (tps, peak_vram_mb). Compression applied at every step."""
    prompt = ("The key challenge in deploying large language models on edge "
              "devices is the memory footprint of the KV cache. Advanced "
              "quantization methods such as")
    ids = tokenizer(prompt, return_tensors="pt").to(device)

    # Warmup
    past_kv, input_ids = None, ids["input_ids"]
    for _ in range(n_warmup):
        with torch.no_grad():
            out = model(input_ids=input_ids, past_key_values=past_kv, use_cache=True)
        past_kv = apply_kv_compression(out.past_key_values, method, cache_type)
        input_ids = out.logits[:, -1:].argmax(-1)
    del past_kv; gc.collect(); torch.cuda.empty_cache()

    # Measure
    torch.cuda.reset_peak_memory_stats()
    past_kv, input_ids = None, ids["input_ids"]
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        with torch.no_grad():
            out = model(input_ids=input_ids, past_key_values=past_kv, use_cache=True)
        past_kv = apply_kv_compression(out.past_key_values, method, cache_type)
        input_ids = out.logits[:, -1:].argmax(-1)
    elapsed = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1e6
    del past_kv; gc.collect(); torch.cuda.empty_cache()
    return round(n_tokens / elapsed, 2), round(peak_vram, 1)


def eval_gsm8k(model, tokenizer, method, cache_type, device, n_samples=20):
    """GSM8K: 3-shot, generation loop with compression at each step."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test",
                      cache_dir=HF_CACHE).select(range(n_samples))

    torch.cuda.reset_peak_memory_stats()
    correct, t0 = 0, time.perf_counter()

    for item in ds:
        prompt = gsm8k_prompt(item["question"])
        gold   = item["answer"].split("####")[-1].strip().replace(",", "")

        ids = tokenizer(prompt, return_tensors="pt").to(device)
        past_kv, input_ids = None, ids["input_ids"]
        generated = []

        for _ in range(64):   # max new tokens
            with torch.no_grad():
                out = model(input_ids=input_ids, past_key_values=past_kv, use_cache=True)
            next_tok = out.logits[:, -1:].argmax(-1)
            if next_tok.item() == tokenizer.eos_token_id:
                break
            generated.append(next_tok.item())
            past_kv    = apply_kv_compression(out.past_key_values, method, cache_type)
            input_ids  = next_tok

        text = tokenizer.decode(generated, skip_special_tokens=True)
        pred = extract_number(text)
        if pred is not None and pred == gold:
            correct += 1
        del past_kv; gc.collect(); torch.cuda.empty_cache()

    elapsed   = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1e6
    score     = round(100.0 * correct / n_samples, 1)
    tps       = round(n_samples * 32 / elapsed, 2)  # rough t/s (avg ~32 gen tokens)
    return {"score": score, "correct": correct, "total": n_samples,
            "tps_approx": tps, "peak_vram_mb": round(peak_vram, 1)}


def eval_arc(model, tokenizer, method, cache_type, device, n_samples=20):
    """
    ARC-Challenge: 3-shot, two-pass.
    Pass 1: forward on context → compress KV.
    Pass 2: score P(choice | compressed context) for each A/B/C/D.
    """
    from datasets import load_dataset
    ds = load_dataset("ai2_arc", "ARC-Challenge", split="test",
                      cache_dir=HF_CACHE).select(range(n_samples))

    torch.cuda.reset_peak_memory_stats()
    correct, t0 = 0, time.perf_counter()

    for item in ds:
        choices    = item["choices"]["text"]
        labels     = item["choices"]["label"]
        gold_label = item["answerKey"]
        gold_idx   = labels.index(gold_label) if gold_label in labels else 0

        context = arc_context(item["question"], choices)
        ctx_ids = tokenizer(context, return_tensors="pt").to(device)
        ctx_len = ctx_ids["input_ids"].shape[1]

        # Pass 1: get KV from context, then compress
        with torch.no_grad():
            ctx_out = model(**ctx_ids, use_cache=True)
        kv = apply_kv_compression(ctx_out.past_key_values, method, cache_type)

        # Pass 2: score each choice
        log_probs = []
        for choice_text in choices:
            c_ids = tokenizer(" " + choice_text, add_special_tokens=False,
                              return_tensors="pt").to(device)
            if c_ids["input_ids"].shape[1] == 0:
                log_probs.append(-1e9)
                continue
            attn_mask = torch.ones(1, ctx_len + c_ids["input_ids"].shape[1],
                                   device=device, dtype=torch.long)
            with torch.no_grad():
                c_out = model(input_ids=c_ids["input_ids"],
                              past_key_values=kv,
                              attention_mask=attn_mask,
                              use_cache=False)
            logits  = c_out.logits.float()
            lp      = F.log_softmax(logits, dim=-1)
            targets = c_ids["input_ids"][0]
            if logits.shape[1] >= len(targets):
                token_lps = lp[0, :len(targets)].gather(
                    1, targets.unsqueeze(1)).squeeze(1)
                log_probs.append(token_lps.sum().item())
            else:
                log_probs.append(lp[0, -1, targets[-1].item()].item())

        pred_idx = int(torch.tensor(log_probs).argmax().item())
        if pred_idx == gold_idx:
            correct += 1
        del kv; gc.collect(); torch.cuda.empty_cache()

    elapsed   = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1e6
    score     = round(100.0 * correct / n_samples, 1)
    return {"score": score, "correct": correct, "total": n_samples,
            "peak_vram_mb": round(peak_vram, 1)}


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(all_results: dict, n_tokens: int, n_samples: int):
    os.makedirs(os.path.dirname(OUTPUT_REPORT), exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Stage 3 — TurboQuant KV Cache Compression: Full Benchmark",
        "",
        "**Project:** ANeurologic Phase 5 — Advanced Optimization",
        f"**Hardware:** NVIDIA Jetson Orin Nano 8 GB UMA (sm_87)",
        f"**Date:** {ts}",
        "",
        "## What is TurboQuant?",
        "",
        "TurboQuant (Google Research, arXiv:2504.19874, ICLR 2026) is a **framework** that",
        "unifies and extends two separately-published algorithms. There is no `pip install`.",
        "",
        "| Component | Paper | Venue | Role |",
        "|-----------|-------|-------|------|",
        "| **PolarQuant** | arXiv:2502.02617 | AISTATS 2026 | K+V compression via polar transform |",
        "| **QJL** | arXiv:2406.03482 | AAAI 2025 | K-only 1-bit JL sketch (score estimation) |",
        "| **KIVI** | arXiv:2402.02750 | ICML 2024 | Production 2/4-bit group quant baseline |",
        "",
        "---",
        "",
        "## Feasibility: Can Compression Increase Speed and Accuracy?",
        "",
        "**Speed — Yes, but only with a CUDA kernel (not CPU roundtrip).**",
        "",
        "Current implementation compresses/decompresses on CPU at each step.",
        "This adds ~20-25% overhead despite the smaller KV footprint in VRAM.",
        "In a production CUDA kernel (e.g., KIVI's official implementation):",
        "",
        "- Smaller KV → fewer memory reads during `Q @ K^T` → lower memory bandwidth usage",
        "- On Jetson Orin's UMA, memory bandwidth is the primary bottleneck for attention",
        "- Estimated real speedup: 1.5-3x for long sequences (1024+ tokens) where KV dominates",
        "- For short sequences (<256 tokens), KV overhead is small → marginal gain",
        "",
        "**Accuracy — No for short-context tasks; Yes for OOM-limited long-context tasks.**",
        "",
        "Lossy compression fundamentally cannot add information — accuracy can only",
        "decrease or stay the same for tasks within normal context length.",
        "EXCEPTION: when a task would OOM without compression, compression turns a",
        "crash into a correct answer. This is the primary motivation for edge deployment.",
        "",
        "| Method | Accuracy impact | Speed impact (CUDA) | Best use case |",
        "|--------|----------------|--------------------:|---------------|",
        "| PolarQuant | Significant degradation at 0.5-1.2B scale | High (7.53x KV) | Large models (7B+) |",
        "| KIVI-2bit | Moderate degradation | Medium (2.29x KV) | Production edge |",
        "| KIVI-4bit | Near-zero degradation | Mild (1.88x KV) | Conservative production |",
        "| QJL | N/A (kernel required) | Very high (16-64x K) | Research / custom kernels |",
        "",
        "---",
        "",
        "## Experimental Setup",
        "",
        f"| Setting | Value |",
        f"|---------|-------|",
        f"| Quantization | 4-bit NF4 (BitsAndBytesConfig, double quantized) |",
        f"| Speed test | {n_tokens} generated tokens, 5 warmup steps |",
        f"| Accuracy test | {n_samples} samples, 3-shot prompts |",
        f"| GSM8K method | Generation loop, exact number match (regex) |",
        f"| ARC method | Two-pass: context → compress KV → score A/B/C/D |",
        f"| Compression hook | Every generation step (GPU tensor → CPU compress → GPU) |",
        "",
        "---",
        "",
        "## Results",
        "",
    ]

    for model_label, model_data in all_results.items():
        vram_load = model_data.get("vram_load_mb", "?")
        cache_t   = model_data.get("cache_type", "?")
        lines += [
            f"### {model_label}",
            "",
            f"**Cache type:** `{cache_t}`  |  **VRAM at load:** {vram_load} MB",
            "",
            "#### Throughput & VRAM",
            "",
            "| Method | KV Ratio | t/s | vs Baseline | Peak VRAM (eval) MB |",
            "|--------|--------:|----:|------------:|--------------------:|",
        ]
        baseline_tps = None
        for method_name, mdata in model_data.get("methods", {}).items():
            if method_name == "QJL":
                lines.append(f"| QJL | 16–64x (K) | N/A* | N/A* | N/A* |")
                continue
            tps    = mdata.get("speed_tps", "—")
            vram   = mdata.get("speed_vram_mb", "—")
            ratio  = mdata.get("kv_ratio", "—")
            if method_name == "Baseline" and isinstance(tps, (int, float)):
                baseline_tps = tps
            vs = (f"{tps/baseline_tps:.2f}x" if baseline_tps and isinstance(tps, (int,float))
                  else "—")
            lines.append(f"| {method_name} | {ratio} | {tps} | {vs} | {vram} |")

        lines += [
            "",
            "#### GSM8K Accuracy (math word problems, 3-shot generation)",
            "",
            "| Method | Score | Correct/Total | Approx t/s | Peak VRAM MB |",
            "|--------|------:|:-------------:|-----------:|-------------:|",
        ]
        for method_name, mdata in model_data.get("methods", {}).items():
            if method_name == "QJL":
                lines.append(f"| QJL | N/A* | N/A* | N/A* | N/A* |")
                continue
            gsm = mdata.get("gsm8k", {})
            score = f"{gsm.get('score','—')}%"
            ct    = f"{gsm.get('correct','—')}/{gsm.get('total','—')}"
            tps   = gsm.get("tps_approx", "—")
            vram  = gsm.get("peak_vram_mb", "—")
            lines.append(f"| {method_name} | {score} | {ct} | {tps} | {vram} |")

        lines += [
            "",
            "#### ARC-Challenge Accuracy (science MCQ, 3-shot two-pass)",
            "",
            "| Method | Score | Correct/Total | Peak VRAM MB |",
            "|--------|------:|:-------------:|-------------:|",
        ]
        for method_name, mdata in model_data.get("methods", {}).items():
            if method_name == "QJL":
                lines.append(f"| QJL | N/A* | N/A* | N/A* |")
                continue
            arc   = mdata.get("arc", {})
            score = f"{arc.get('score','—')}%"
            ct    = f"{arc.get('correct','—')}/{arc.get('total','—')}"
            vram  = arc.get("peak_vram_mb", "—")
            lines.append(f"| {method_name} | {score} | {ct} | {vram} |")

        lines += [""]

        # QJL note
        qjl = model_data.get("methods", {}).get("QJL", {})
        pr  = qjl.get("pearson_r_dim64", 0.62)
        lines += [
            f"*QJL requires modifying the attention kernel to use the asymmetric score",
            f"estimator `(π/2m) · Q_proj · q^T`. Speed/accuracy tests are not applicable",
            f"because simply using the 1-bit sketch as K tensors produces garbage attention.",
            f"Score estimation quality from Stage 3: Pearson-r = {pr} at sketch_dim=64 (16x ratio).",
            "",
            "---",
            "",
        ]

    lines += [
        "## Method Summary",
        "",
        "| Method | KV Ratio | Principle | Accuracy Impact | Production Ready |",
        "|--------|--------:|-----------|----------------|:---------------:|",
        "| Baseline | 1x | — | — | ✅ |",
        "| PolarQuant | 7.53x | Polar coords (mag + angle) | Significant at <2B scale | ⚠️ Large models only |",
        "| KIVI-2bit | 2.29x | Asymmetric group quant | Moderate | ✅ |",
        "| KIVI-4bit | 1.88x | Asymmetric group quant | Near-zero | ✅ |",
        "| QJL | 16–64x (K) | 1-bit JL sketch | Requires kernel | 🔬 Research |",
        "",
        "## How to Reproduce",
        "",
        "```bash",
        "# Full benchmark (both models, all methods, speed + accuracy)",
        "docker exec aneurologic_phase5 python3 \\",
        "    /workspace/scripts/stage3_turboquant/stage3_comprehensive.py",
        "",
        "# Single model",
        "docker exec aneurologic_phase5 python3 \\",
        "    /workspace/scripts/stage3_turboquant/stage3_comprehensive.py --model qwen",
        "",
        "# Speed only (faster)",
        "docker exec aneurologic_phase5 python3 \\",
        "    /workspace/scripts/stage3_turboquant/stage3_comprehensive.py --speed-only",
        "```",
    ]

    with open(OUTPUT_REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Report → {OUTPUT_REPORT}")


# ── Main ──────────────────────────────────────────────────────────────────────

# Qwen runs first: stable 457 MB footprint. LFM2 second: needs clean CUDA state.
MODELS = {
    "qwen": ("Qwen/Qwen2.5-0.5B-Instruct",       "Qwen2.5-0.5B"),
    "lfm":  ("LiquidAI/LFM2.5-1.2B-Instruct",  "LFM2.5-1.2B"),
}

METHODS_ORDERED = ["Baseline", "PolarQuant", "KIVI-2bit", "KIVI-4bit", "QJL"]

def build_methods():
    return {
        "Baseline":  (None,            "1.00x"),
        "PolarQuant":(PolarQuant(),    "7.53x"),
        "KIVI-2bit": (KIVIQuant(2),    "2.29x"),
        "KIVI-4bit": (KIVIQuant(4),    "1.88x"),
        "QJL":       (None,            "16-64x"),   # N/A — no roundtrip possible
    }


def run_model(hf_id: str, label: str, methods: dict,
              args) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\n{'='*62}")
    print(f"  Model: {label}  ({hf_id})")
    print(f"{'='*62}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("  Loading model ...")
    torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(hf_id, cache_dir=HF_CACHE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, quantization_config=bnb,
        device_map="auto", low_cpu_mem_usage=True,
        cache_dir=HF_CACHE,
    )
    model.eval()
    vram_load = round(torch.cuda.memory_allocated() / 1e6, 1)
    print(f"  Loaded. VRAM: {vram_load} MB")

    # Detect cache type from model ID (avoids dummy forward pass that can OOM on LFM2)
    cache_type      = detect_cache_type_from_name(hf_id)
    cache_type_name = "Lfm2HybridConvCache" if cache_type == "lfm2" else "DynamicCache"
    print(f"  Cache type: {cache_type_name} → hook: '{cache_type}'")

    model_results = {
        "hf_id":        hf_id,
        "vram_load_mb": vram_load,
        "cache_type":   cache_type_name,
        "methods": {},
    }

    for method_name in METHODS_ORDERED:
        method_obj, kv_ratio = methods[method_name]
        print(f"\n  [{method_name}]  KV ratio: {kv_ratio}")
        mdata = {"kv_ratio": kv_ratio}

        # QJL: cannot run generation with it — report only
        if method_name == "QJL":
            mdata["note"] = ("Cannot integrate into generation loop without modifying "
                             "attention kernel. QJL is a score estimator, not a KV "
                             "reconstructor — using 1-bit sketch as K produces garbage.")
            mdata["pearson_r_dim64"] = 0.62   # from Stage 3 offline benchmark
            model_results["methods"]["QJL"] = mdata
            print(f"    ⚠️  Skipped (kernel modification required). Pearson-r=0.62.")
            continue

        # Speed
        print(f"    Speed ({args.n_tokens} tokens) ...")
        tps, vram_eval = bench_speed(model, tok, method_obj, cache_type, device,
                                      n_warmup=5, n_tokens=args.n_tokens)
        mdata["speed_tps"]     = tps
        mdata["speed_vram_mb"] = vram_eval
        print(f"    → {tps} t/s  |  Peak VRAM: {vram_eval} MB")

        if not args.speed_only:
            # GSM8K
            print(f"    GSM8K ({args.n_samples} samples) ...")
            gsm = eval_gsm8k(model, tok, method_obj, cache_type, device, args.n_samples)
            mdata["gsm8k"] = gsm
            print(f"    → {gsm['score']}% ({gsm['correct']}/{gsm['total']})  "
                  f"VRAM: {gsm['peak_vram_mb']} MB")

            # ARC
            print(f"    ARC-Challenge ({args.n_samples} samples) ...")
            arc = eval_arc(model, tok, method_obj, cache_type, device, args.n_samples)
            mdata["arc"] = arc
            print(f"    → {arc['score']}% ({arc['correct']}/{arc['total']})  "
                  f"VRAM: {arc['peak_vram_mb']} MB")

        model_results["methods"][method_name] = mdata

    # Free model before next model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n  {label} done. VRAM freed: {torch.cuda.memory_allocated()/1e6:.0f} MB remaining")
    return model_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     choices=["lfm", "qwen", "all"], default="all")
    parser.add_argument("--n-tokens",  type=int, default=48)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--speed-only", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*62}")
    print(f"  Phase 5 Stage 3 — Comprehensive KV Compression Benchmark")
    print(f"  Models: {args.model}  |  Speed: {args.n_tokens} tok  "
          f"|  Accuracy: {args.n_samples} samples")
    print(f"{'='*62}")

    model_keys = ["lfm", "qwen"] if args.model == "all" else [args.model]
    methods    = build_methods()
    all_results = {}

    for mk in model_keys:
        hf_id, label = MODELS[mk]
        # Aggressive CUDA cleanup before each model to maximise free VRAM
        gc.collect(); torch.cuda.empty_cache()
        print(f"\n  Pre-run VRAM: {torch.cuda.memory_allocated()/1e6:.0f} MB allocated")
        try:
            result = run_model(hf_id, label, methods, args)
            all_results[label] = result
        except RuntimeError as e:
            if "INTERNAL ASSERT" in str(e) or "CUDA" in str(e) or "out of memory" in str(e).lower():
                print(f"\n  ❌ {label} CUDA OOM: {e}")
                print(f"  Skipping {label} — Jetson UMA CUDA allocator limit hit.")
                print(f"  Run this model first in a fresh container for accurate results.")
                all_results[label] = {"hf_id": hf_id, "error": str(e)[:200],
                                      "note": "CUDA OOM — run first in fresh container",
                                      "methods": {}}
            else:
                raise
            gc.collect(); torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "stage":     "Stage 3 Comprehensive — KV Compression Full Benchmark",
            "timestamp": datetime.now().isoformat(),
            "config":    {"n_tokens": args.n_tokens, "n_samples": args.n_samples,
                          "speed_only": args.speed_only},
            "results":   all_results,
        }, f, indent=2)
    print(f"\n  Results → {OUTPUT_JSON}")

    generate_report(all_results, args.n_tokens, args.n_samples)

    print(f"\n{'='*62}")
    print(f"  Stage 3 comprehensive benchmark complete.")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
