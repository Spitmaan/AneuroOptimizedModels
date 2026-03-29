#!/usr/bin/env python3
"""
ANeurologic Phase 5 - Stage 3b: KV Compression Performance & Accuracy Benchmark
================================================================================
Measures the REAL impact of each TurboQuant KV compression method on:
  1. Generation throughput (tokens/sec) — compression applied at every step
  2. Task accuracy on GSM8K and ARC-Challenge — compression active during inference

This directly answers: "does compressing the KV cache degrade quality, and
what is the speed/memory trade-off on Jetson Orin Nano?"

Methods compared:
  Baseline      no compression
  PolarQuant    8-bit magnitude + 2-bit angular  (~7.53x ratio, cosine_sim=0.916)
  KIVI-2bit     2-bit asymmetric group quant     (~2.29x ratio, cosine_sim=0.932)
  KIVI-4bit     4-bit asymmetric group quant     (~1.88x ratio, cosine_sim=0.997)

Note on QJL: QJL compresses K to a 1-bit JL sketch and estimates attention
scores directly from the sketch — it does NOT reconstruct the original K.
Plugging it into standard attention requires modifying the model's attention
kernel (not done here). QJL quality is captured by Pearson-r in Stage 3.

Stage 2 baselines (Qwen2.5-0.5B, NF4 4-bit, no compression):
  GSM8K 7.0%  |  ARC-Challenge 57.0%

Usage:
    python3 /workspace/scripts/stage3_turboquant/stage3_perf_bench.py
    python3 /workspace/scripts/stage3_turboquant/stage3_perf_bench.py --n-tokens 32 --n-samples 20
    python3 /workspace/scripts/stage3_turboquant/stage3_perf_bench.py --speed-only
"""

import sys, os, gc, json, re, math, time, argparse
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional

import torch
import torch.nn.functional as F

OUTPUT_JSON   = "/workspace/outputs/logs/stage3_perf_results.json"
HF_CACHE      = "/workspace/models/hf_cache"

# ── Stage 2 baselines for delta reporting ────────────────────────────────────
STAGE2_BASELINES = {
    "Qwen2.5-0.5B": {"gsm8k": 7.0, "arc_challenge": 57.0}
}

# ── Compression implementations (self-contained, same logic as kv_compression.py)

class PolarQuant:
    """8-bit magnitude + 2-bit angular quantization (~7.53x vs fp16)."""
    def __init__(self, n_bits_mag=8, n_bits_ang=2, group_size=64):
        self.n_bits_mag = n_bits_mag
        self.n_bits_ang = n_bits_ang
        self.group_size = group_size

    def compress_decompress(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        orig_shape = x.shape
        D = x.shape[-1]
        x_f = x.reshape(-1, D).float()
        mag = x_f.norm(dim=-1, keepdim=True)
        unit = x_f / (mag + 1e-8)
        max_mag = mag.max().clamp(min=1e-6)
        mag_scale = max_mag / (2**self.n_bits_mag - 1)
        q_mag = (mag / mag_scale).round().clamp(0, 2**self.n_bits_mag - 1)
        mag_rec = q_mag * mag_scale
        n_groups = math.ceil(D / self.group_size)
        unit_parts = []
        for g in range(n_groups):
            s, e = g * self.group_size, min((g+1) * self.group_size, D)
            blk = unit[:, s:e]
            b_min = blk.min(dim=-1, keepdim=True).values
            b_max = blk.max(dim=-1, keepdim=True).values
            scale = (b_max - b_min).clamp(min=1e-6) / (2**self.n_bits_ang - 1)
            q_blk = ((blk - b_min) / scale).round().clamp(0, 2**self.n_bits_ang - 1)
            unit_parts.append(q_blk * scale + b_min)
        unit_rec = torch.cat(unit_parts, dim=-1)
        unit_rec = unit_rec / unit_rec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (mag_rec * unit_rec).reshape(orig_shape).to(orig_dtype)

    def name(self): return "PolarQuant"


class KIVIQuant:
    """Asymmetric group quantization (2-bit or 4-bit). ICML 2024."""
    def __init__(self, n_bits=2, group_size=32, residual_length=64):
        self.n_bits = n_bits
        self.group_size = group_size
        self.residual_length = residual_length
        self.levels = 2**n_bits

    def compress_decompress(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        B, H, S, D = x.shape
        x_f = x.float()
        S_round = (S // self.group_size) * self.group_size
        x_body  = x_f[:, :, :S_round, :]
        x_resid = x_f[:, :, S_round:, :]
        if S_round > 0:
            n_g = S_round // self.group_size
            x_g  = x_body.reshape(B, H, n_g, self.group_size, D)
            xmin = x_g.amin(dim=3, keepdim=True)
            xmax = x_g.amax(dim=3, keepdim=True)
            scale = (xmax - xmin).clamp(min=1e-6) / (self.levels - 1)
            q    = ((x_g - xmin) / scale).round().clamp(0, self.levels - 1)
            rec  = q * scale + xmin
            x_body = rec.reshape(B, H, S_round, D)
        return torch.cat([x_body, x_resid], dim=2).to(orig_dtype)

    def name(self): return f"KIVI-{self.n_bits}bit"


# ── DynamicCache KV hook (transformers 5.x) ──────────────────────────────────

def apply_kv_compression(kv_cache, method):
    """
    Compress+decompress every layer's K and V tensors in-place.
    Works with transformers 5.x DynamicCache (layer.keys / layer.values).
    """
    if kv_cache is None or method is None:
        return kv_cache
    for layer in kv_cache.layers:
        K = layer.keys
        V = layer.values
        if not (isinstance(K, torch.Tensor) and K.ndim == 4 and K.numel() > 0):
            continue
        device, dtype = K.device, K.dtype
        layer.keys   = method.compress_decompress(K.float().cpu()).to(device=device, dtype=dtype)
        layer.values = method.compress_decompress(V.float().cpu()).to(device=device, dtype=dtype)
    return kv_cache


# ── Generation throughput benchmark ──────────────────────────────────────────

def bench_speed(model, tokenizer, method, device, n_warmup=5, n_tokens=48):
    """
    Measure tokens/sec for greedy generation with KV compression at each step.
    Returns (tps, vram_delta_mb).
    """
    prompt = ("The key challenge in deploying large language models on edge "
              "devices is the memory footprint of the KV cache, which grows "
              "linearly with sequence length. Compression methods such as")
    ids = tokenizer(prompt, return_tensors="pt").to(device)

    method_label = method.name() if method else "Baseline"

    # Warmup
    past_kv = None
    input_ids = ids["input_ids"]
    for _ in range(n_warmup):
        with torch.no_grad():
            out = model(input_ids=input_ids, past_key_values=past_kv, use_cache=True)
        past_kv = apply_kv_compression(out.past_key_values, method)
        input_ids = out.logits[:, -1:].argmax(-1)
    del past_kv; gc.collect(); torch.cuda.empty_cache()

    # Measurement
    vram_before = torch.cuda.memory_allocated() / 1e6
    past_kv = None
    input_ids = ids["input_ids"]
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        with torch.no_grad():
            out = model(input_ids=input_ids, past_key_values=past_kv, use_cache=True)
        past_kv = apply_kv_compression(out.past_key_values, method)
        input_ids = out.logits[:, -1:].argmax(-1)
    elapsed = time.perf_counter() - t0
    vram_peak = torch.cuda.max_memory_allocated() / 1e6
    del past_kv; gc.collect(); torch.cuda.empty_cache()

    tps = n_tokens / elapsed
    vram_delta = vram_peak - vram_before
    print(f"    {method_label:12s}  {tps:6.1f} t/s  VRAM Δ: {vram_delta:+.0f} MB")
    return round(tps, 2), round(vram_delta, 1)


# ── GSM8K accuracy with compression ──────────────────────────────────────────

GSM8K_SHOTS = [
    ("Natalia sold clips to 48 of her friends in April, and then she sold "
     "half as many clips in May. How many clips did Natalia sell altogether "
     "in April and May?",
     "72"),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
     "minutes of babysitting. How much did she earn?",
     "10"),
    ("Betty is saving money for a new wallet which costs $100. Betty has "
     "only half of the money she needs. Her parents decided to give her "
     "$15 for that purpose, and her grandparents twice as much as her "
     "parents. How much more money does Betty need to buy the wallet?",
     "5"),
]

def make_gsm8k_prompt(question: str) -> str:
    shots = "\n\n".join(
        f"Q: {q}\nA: The answer is {a}." for q, a in GSM8K_SHOTS
    )
    return f"{shots}\n\nQ: {question}\nA:"


def extract_number(text: str) -> Optional[str]:
    # Look for the last number in the text
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def generate_with_compression(model, tokenizer, prompt: str, method,
                               device, max_new=64) -> str:
    """Single generation run with KV compression at every step."""
    ids = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = ids["input_ids"]
    past_kv = None
    generated = []

    for _ in range(max_new):
        with torch.no_grad():
            out = model(input_ids=input_ids, past_key_values=past_kv, use_cache=True)
        next_tok = out.logits[:, -1:].argmax(-1)
        tok_id = next_tok.item()
        if tok_id == tokenizer.eos_token_id:
            break
        generated.append(tok_id)
        past_kv = apply_kv_compression(out.past_key_values, method)
        input_ids = next_tok

    del past_kv; gc.collect(); torch.cuda.empty_cache()
    return tokenizer.decode(generated, skip_special_tokens=True)


def eval_gsm8k(model, tokenizer, method, device, n_samples=30) -> dict:
    """Evaluate GSM8K with KV compression active during generation."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test",
                      cache_dir=HF_CACHE).select(range(n_samples))

    method_label = method.name() if method else "Baseline"
    correct = 0
    for item in ds:
        prompt = make_gsm8k_prompt(item["question"])
        answer_raw = item["answer"].split("####")[-1].strip().replace(",", "")
        generated = generate_with_compression(
            model, tokenizer, prompt, method, device, max_new=64
        )
        pred = extract_number(generated)
        if pred is not None and pred == answer_raw:
            correct += 1

    score = round(100.0 * correct / n_samples, 1)
    print(f"    {method_label:12s}  GSM8K {correct}/{n_samples} = {score}%")
    return {"correct": correct, "total": n_samples, "score": score}


# ── ARC-Challenge accuracy with compression ───────────────────────────────────

ARC_SHOTS = [
    {
        "question": "Which property of a mineral can be determined just by looking at it?",
        "choices": ["luster", "mass", "weight", "hardness"],
        "answer": "A",
    },
    {
        "question": "A sixth grade student measured the mass of a sample. What is the most "
                    "likely unit of measurement the student used?",
        "choices": ["liters", "meters", "grams", "seconds"],
        "answer": "C",
    },
    {
        "question": "A student is trying to show how ocean waves can cause erosion. Which "
                    "materials should the student use to model this process?",
        "choices": [
            "water and a container of sand",
            "water and a container of salt",
            "water and a container of sugar",
            "water and a container of flour",
        ],
        "answer": "A",
    },
]

def make_arc_context(shots, question, choices) -> str:
    """Build 3-shot ARC context without the final choices."""
    lines = []
    for s in shots:
        opts = " ".join(f"({chr(65+i)}) {c}" for i, c in enumerate(s["choices"]))
        lines.append(f"Question: {s['question']}\nOptions: {opts}\nAnswer: {s['answer']}")
    opts = " ".join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
    lines.append(f"Question: {question}\nOptions: {opts}\nAnswer:")
    return "\n\n".join(lines)


def score_arc_two_pass(model, tokenizer, context: str, choices: list,
                        method, device) -> int:
    """
    Two-pass ARC scoring:
      Pass 1: forward on context → get KV → compress
      Pass 2: for each choice, score P(choice | compressed KV)
    Returns index of highest-scoring choice.
    """
    # Pass 1: encode context and get compressed KV
    ctx_ids = tokenizer(context, return_tensors="pt").to(device)
    with torch.no_grad():
        ctx_out = model(**ctx_ids, use_cache=True)
    kv = apply_kv_compression(ctx_out.past_key_values, method)
    ctx_seq_len = ctx_ids["input_ids"].shape[1]

    log_probs = []
    for choice_text in choices:
        choice_ids = tokenizer(
            " " + choice_text, add_special_tokens=False, return_tensors="pt"
        ).to(device)
        c_ids = choice_ids["input_ids"]
        if c_ids.shape[1] == 0:
            log_probs.append(-1e9)
            continue

        # Build attention mask spanning context + choice
        total_len = ctx_seq_len + c_ids.shape[1]
        attn_mask = torch.ones(1, total_len, device=device, dtype=torch.long)

        with torch.no_grad():
            choice_out = model(
                input_ids=c_ids,
                past_key_values=kv,
                attention_mask=attn_mask,
                use_cache=False,
            )

        # Log-prob of each choice token
        logits = choice_out.logits                          # [1, L, V]
        lp = F.log_softmax(logits.float(), dim=-1)
        if c_ids.shape[1] == 1:
            lp_token = lp[0, 0, c_ids[0, 0].item()]
        else:
            # Sum log-probs: logits[i] predicts token[i+1]
            targets = c_ids[0, 1:]                          # [L-1]
            lp_token = lp[0, :-1].gather(1, targets.unsqueeze(1)).squeeze(1).sum()
        log_probs.append(lp_token.item())

    del kv; gc.collect(); torch.cuda.empty_cache()
    return int(torch.tensor(log_probs).argmax().item())


def eval_arc(model, tokenizer, method, device, n_samples=30) -> dict:
    """Evaluate ARC-Challenge with two-pass KV compression."""
    from datasets import load_dataset
    ds = load_dataset("ai2_arc", "ARC-Challenge", split="test",
                      cache_dir=HF_CACHE).select(range(n_samples))

    method_label = method.name() if method else "Baseline"
    correct = 0
    for item in ds:
        choices = item["choices"]["text"]
        labels  = item["choices"]["label"]          # ['A','B','C','D']
        gold_label = item["answerKey"]
        gold_idx   = labels.index(gold_label) if gold_label in labels else 0

        context = make_arc_context(ARC_SHOTS, item["question"], choices)
        pred_idx = score_arc_two_pass(
            model, tokenizer, context, choices, method, device
        )
        if pred_idx == gold_idx:
            correct += 1

    score = round(100.0 * correct / n_samples, 1)
    print(f"    {method_label:12s}  ARC {correct}/{n_samples} = {score}%")
    return {"correct": correct, "total": n_samples, "score": score}


# ── Report ────────────────────────────────────────────────────────────────────

def generate_report(results: dict, n_tokens: int, n_samples: int):
    """Append or write Stage 3b performance report."""
    report_path = "/workspace/outputs/reports/stage3_perf.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Stage 3b — KV Compression: Speed & Accuracy Impact",
        "",
        "**Project:** ANeurologic Phase 5 — Advanced Optimization",
        f"**Hardware:** NVIDIA Jetson Orin Nano 8 GB, sm_87",
        f"**Date:** {ts}",
        f"**Model:** Qwen2.5-0.5B-Instruct (NF4 4-bit, device_map=auto)",
        f"**Speed test:** {n_tokens} tokens after 5 warmup steps",
        f"**Accuracy test:** {n_samples} samples each, 3-shot",
        "",
        "## Throughput (tokens/sec)",
        "",
        "Compression applied at **every generation step**.",
        "VRAM Δ = peak allocated during generation vs. before.",
        "",
        "| Method | t/s | vs Baseline | VRAM Δ (MB) | KV Ratio |",
        "|--------|----:|------------:|------------:|--------:|",
    ]

    baseline_tps = results.get("Baseline", {}).get("tps", None)
    ratios = {
        "Baseline":   "1.00x",
        "PolarQuant": "7.53x",
        "KIVI-2bit":  "2.29x",
        "KIVI-4bit":  "1.88x",
    }
    for method_name, data in results.items():
        tps      = data.get("tps", "—")
        vram_d   = data.get("vram_delta_mb", "—")
        vs_base  = (f"{tps/baseline_tps:.2f}x" if isinstance(tps, (int,float))
                    and baseline_tps else "—")
        ratio    = ratios.get(method_name, "—")
        lines.append(f"| {method_name} | {tps} | {vs_base} | {vram_d} | {ratio} |")

    lines += [
        "",
        "## Accuracy Impact",
        "",
        "**GSM8K** — math word problems, 3-shot generation, exact number match.",
        "**ARC-Challenge** — 3-shot two-pass: context → compress KV → score choices.",
        "Stage 2 baselines (no compression): GSM8K 7.0% | ARC-Challenge 57.0%",
        "",
        "| Method | GSM8K | vs Baseline | ARC-Challenge | vs Baseline |",
        "|--------|------:|------------:|--------------:|------------:|",
    ]

    for method_name, data in results.items():
        gsm = data.get("gsm8k", {})
        arc = data.get("arc", {})
        gsm_s  = f"{gsm.get('score','—')}%" if gsm else "—"
        arc_s  = f"{arc.get('score','—')}%" if arc else "—"

        s2_gsm = STAGE2_BASELINES["Qwen2.5-0.5B"]["gsm8k"]
        s2_arc = STAGE2_BASELINES["Qwen2.5-0.5B"]["arc_challenge"]

        if gsm and isinstance(gsm.get("score"), (int,float)):
            gsm_d = f"{gsm['score'] - s2_gsm:+.1f}pp"
        else:
            gsm_d = "—"
        if arc and isinstance(arc.get("score"), (int,float)):
            arc_d = f"{arc['score'] - s2_arc:+.1f}pp"
        else:
            arc_d = "—"

        lines.append(f"| {method_name} | {gsm_s} | {gsm_d} | {arc_s} | {arc_d} |")

    lines += [
        "",
        "## Notes",
        "",
        "- **pp** = percentage points vs Stage 2 baseline (no compression)",
        "- **VRAM Δ** includes overhead from compress/decompress operations",
        "- Compression+decompression runs on CPU; GPU transfer adds latency",
        "- In a production CUDA kernel, compress/decompress would run on-GPU",
        "  with no transfer overhead — real-world t/s would be higher",
        "- **QJL** not included in speed/accuracy tests: requires modifying",
        "  the attention kernel to use sketch scores instead of exact dot products",
        "  (Pearson-r quality: 0.62 at sketch_dim=64 from Stage 3)",
        "",
        "## Method Summary",
        "",
        "| Method | Compression | Principle | Best For |",
        "|--------|-------------|-----------|----------|",
        "| PolarQuant | 7.53x | Polar coords (mag + angle) | Large contexts, moderate accuracy loss OK |",
        "| KIVI-2bit | 2.29x | Asymmetric min-max quant | Production (battle-tested) |",
        "| KIVI-4bit | 1.88x | Asymmetric min-max quant | Near-lossless, conservative |",
        "| QJL | 16–64x (K only) | 1-bit JL sketch | Research / custom attention kernels |",
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Report → {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tokens",  type=int, default=48,
                        help="Tokens to generate per speed test (default 48)")
    parser.add_argument("--n-samples", type=int, default=30,
                        help="Eval samples per task (default 30)")
    parser.add_argument("--speed-only", action="store_true",
                        help="Skip accuracy eval (faster)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HF model ID")
    args = parser.parse_args()

    print(f"\n{'='*62}")
    print(f"  Phase 5 Stage 3b — KV Compression: Speed & Accuracy")
    print(f"  Model: {args.model}")
    print(f"  Speed: {args.n_tokens} tokens | Accuracy: {args.n_samples} samples")
    print(f"{'='*62}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"\n[1/3] Loading model ...")
    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=HF_CACHE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb,
        device_map="auto", low_cpu_mem_usage=True,
        cache_dir=HF_CACHE,
    )
    model.eval()
    print(f"  Loaded. VRAM: {torch.cuda.memory_allocated()/1e6:.0f} MB")

    methods = [
        None,                           # Baseline
        PolarQuant(8, 2, 64),
        KIVIQuant(2, 32, 64),
        KIVIQuant(4, 32, 64),
    ]

    # ── Speed test ─────────────────────────────────────────────────────────
    print(f"\n[2/3] Speed test ({args.n_tokens} tokens each) ...")
    print(f"    {'Method':12s}  {'t/s':>6}  VRAM Δ")
    results = {}
    for method in methods:
        label = method.name() if method else "Baseline"
        torch.cuda.reset_peak_memory_stats()
        tps, vram_d = bench_speed(model, tok, method, device,
                                   n_warmup=5, n_tokens=args.n_tokens)
        results[label] = {"tps": tps, "vram_delta_mb": vram_d}

    # ── Accuracy test ───────────────────────────────────────────────────────
    if not args.speed_only:
        print(f"\n[3/3] Accuracy eval ({args.n_samples} samples each) ...")

        print(f"\n  GSM8K (3-shot generation, exact number match):")
        for method in methods:
            label = method.name() if method else "Baseline"
            gsm = eval_gsm8k(model, tok, method, device, args.n_samples)
            results[label]["gsm8k"] = gsm

        print(f"\n  ARC-Challenge (3-shot, two-pass context→compress→score):")
        for method in methods:
            label = method.name() if method else "Baseline"
            arc = eval_arc(model, tok, method, device, args.n_samples)
            results[label]["arc"] = arc
    else:
        print(f"\n[3/3] Skipped (--speed-only)")

    # ── Save ────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "stage":     "Stage 3b - KV Compression Performance",
            "timestamp": datetime.now().isoformat(),
            "model":     args.model,
            "config":    {"n_tokens": args.n_tokens, "n_samples": args.n_samples,
                          "speed_only": args.speed_only},
            "stage2_baselines": STAGE2_BASELINES,
            "results":   results,
        }, f, indent=2)
    print(f"\n  Results → {OUTPUT_JSON}")

    generate_report(results, args.n_tokens, args.n_samples)

    print(f"\n{'='*62}")
    print(f"  Stage 3b complete.")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
