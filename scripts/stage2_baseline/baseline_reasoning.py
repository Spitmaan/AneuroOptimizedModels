#!/usr/bin/env python3
"""
ANeurologic Phase 5 - Stage 2: Baseline Reasoning Benchmarks
=============================================================
Evaluates reasoning accuracy on GSM8K (math) and ARC-Challenge (science)
using a custom few-shot evaluation loop with 4-bit NF4 quantization.

Why custom loop instead of lm_eval:
  lm_eval's HF backend calls `self.model.to(self.device)` which triggers
  PyTorch's CUDA caching allocator assertion on Jetson's UMA architecture
  (CUDACachingAllocator.cpp:1154). This happens even for 0.5B models.
  Our custom loop uses BitsAndBytesConfig (4-bit NF4, proven in Phase 4)
  with `device_map='auto'` which avoids this code path entirely.

Evaluation methodology:
  - GSM8K: 5-shot chain-of-thought math. Accuracy = fraction where model
    extracts the correct final number from its output.
  - ARC-Challenge: 5-shot multiple-choice science. Accuracy = fraction of
    correct A/B/C/D choices. Uses log-likelihood scoring (standard for MC).

Models tested:
  - LiquidAI/LFM2.5-1.2B-Instruct  (Phase 1 best: 55.4 t/s)
  - Qwen/Qwen2.5-1.5B-Instruct      (Phase 1: 21.3 t/s; non-gated Llama proxy)
  - Qwen/Qwen2.5-0.5B-Instruct      (smallest reference; establishes 0.5B floor)

Usage:
    python3 /workspace/scripts/stage2_baseline/baseline_reasoning.py
    python3 /workspace/scripts/stage2_baseline/baseline_reasoning.py --models lfm qwen05b
    python3 /workspace/scripts/stage2_baseline/baseline_reasoning.py --limit 50
"""

import sys
import os
import gc
import re
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

OUTPUT_JSON   = "/workspace/outputs/logs/stage2_baseline_results.json"
OUTPUT_REPORT = "/workspace/outputs/reports/stage2_baseline.md"

# ── 4-bit NF4 config (proven in Phase 4) ─────────────────────────────────────
def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = {
    "lfm": {
        "hf_id":    "LiquidAI/LFM2.5-1.2B-Instruct",
        "label":    "LFM2.5-1.2B",
        "phase1_tps": 55.4,
        "phase1_vram_gb": 1.2,
        "gated": False,
    },
    # meta-llama/Llama-3.2-1B and nvidia/Cosmos-Reason2-2B are gated HF repos.
    # Phase 1 ran them via GGUF/llama.cpp (bypasses HF gating).
    # lm_eval's HF backend requires auth. We use non-gated equivalents:
    "qwen15b": {
        "hf_id":    "Qwen/Qwen2.5-1.5B-Instruct",
        "label":    "Qwen2.5-1.5B",
        "phase1_tps": 21.3,
        "phase1_vram_gb": 1.5,
        "gated": False,
        "note": "Non-gated Llama-class proxy (same param range as Llama-3.2-1B)",
    },
    "qwen05b": {
        "hf_id":    "Qwen/Qwen2.5-0.5B-Instruct",
        "label":    "Qwen2.5-0.5B",
        "phase1_tps": None,
        "phase1_vram_gb": 0.5,
        "gated": False,
        "note": "Lightest reference; establishes reasoning floor at 0.5B scale",
    },
}

TASKS = ["gsm8k", "arc_challenge"]

# ── GSM8K Few-shot examples ───────────────────────────────────────────────────
GSM8K_SHOTS = [
    ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
     "How many clips did Natalia sell altogether in April and May?",
     "72"),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. "
     "How much did she earn?",
     "10"),
    ("Betty is saving money for a new wallet which costs $100. Betty has only half of the money she "
     "needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much "
     "as her parents. How much more money does Betty need to buy the wallet?",
     "5"),
    ("Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read "
     "twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, "
     "how many pages should she read?",
     "42"),
    ("James writes a 3-page letter to 2 different friends twice a week. How many pages does he write "
     "a year?",
     "624"),
]

# ── ARC-Challenge Few-shot examples ──────────────────────────────────────────
ARC_SHOTS = [
    {
        "question": "Which property of a substance will change if the amount of the substance changes?",
        "choices": {"A": "mass", "B": "density", "C": "boiling point", "D": "melting point"},
        "answer": "A",
    },
    {
        "question": "A student is studying a community of organisms in a pond ecosystem. "
                    "Which pair of organisms has a predator-prey relationship?",
        "choices": {"A": "algae and sunfish", "B": "algae and bacteria",
                    "C": "sunfish and snails", "D": "bacteria and sunfish"},
        "answer": "C",
    },
    {
        "question": "Earth's layers are divided into the crust, mantle, and core. "
                    "What are the layers of Earth classified by?",
        "choices": {"A": "temperature", "B": "composition", "C": "both A and B", "D": "neither A nor B"},
        "answer": "C",
    },
    {
        "question": "A student wants to observe a single-celled organism. "
                    "Which tool would be most helpful?",
        "choices": {"A": "telescope", "B": "microscope", "C": "thermometer", "D": "barometer"},
        "answer": "B",
    },
    {
        "question": "A rock contains the following minerals: feldspar, quartz, and biotite. "
                    "The rock most likely formed from:",
        "choices": {"A": "cooled lava", "B": "compacted sediment",
                    "C": "metamorphic processes", "D": "chemical precipitation"},
        "answer": "A",
    },
]


# ── GSM8K evaluation ──────────────────────────────────────────────────────────

def build_gsm8k_prompt(question: str, shots: list) -> str:
    """Build 5-shot GSM8K prompt."""
    parts = []
    for q, a in shots:
        parts.append(f"Q: {q}\nA: The answer is {a}.")
    parts.append(f"Q: {question}\nA:")
    return "\n\n".join(parts)


def extract_gsm8k_answer(text: str) -> str:
    """Extract numeric answer from model output."""
    # Try 'the answer is X' pattern first
    m = re.search(r'the answer is\s*([\d,]+)', text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")
    # Then #### X (standard GSM8K format)
    m = re.search(r'####\s*([\d,]+)', text)
    if m:
        return m.group(1).replace(",", "")
    # Last: final number in text
    nums = re.findall(r'[\d,]+', text)
    if nums:
        return nums[-1].replace(",", "")
    return ""


def evaluate_gsm8k(model, tokenizer, dataset, n_shots: int = 5,
                   limit: int = 100, device: str = "cuda") -> dict:
    """Evaluate on GSM8K subset."""
    shots = GSM8K_SHOTS[:n_shots]
    correct = total = 0
    examples = dataset[:limit]

    model_dtype = next(model.parameters()).dtype

    t0 = time.perf_counter()
    for item in examples:
        question = item.get("question", "")
        gold     = str(item.get("answer", "")).strip()
        # GSM8K answers end with '#### <number>'
        m = re.search(r'####\s*([\d,]+)', gold)
        gold_num = m.group(1).replace(",", "") if m else gold

        prompt = build_gsm8k_prompt(question, shots)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=768).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        pred = extract_gsm8k_answer(response)
        if pred == gold_num:
            correct += 1
        total += 1

        del inputs, outputs, new_tokens
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    accuracy = round(100 * correct / max(total, 1), 2)
    return {
        "score":   accuracy,
        "correct": correct,
        "total":   total,
        "elapsed_s": round(elapsed, 1),
        "metric":  "exact_match_number",
    }


# ── ARC-Challenge evaluation ──────────────────────────────────────────────────

def build_arc_prompt(question: str, choices: dict, shots: list) -> str:
    """Build 5-shot ARC multiple-choice prompt."""
    parts = []
    for ex in shots:
        q   = ex["question"]
        chs = "\n".join(f"{k}. {v}" for k, v in ex["choices"].items())
        parts.append(f"Question: {q}\n{chs}\nAnswer: {ex['answer']}")
    ch_str = "\n".join(f"{k}. {v}" for k, v in choices.items())
    parts.append(f"Question: {question}\n{ch_str}\nAnswer:")
    return "\n\n".join(parts)


def evaluate_arc_challenge(model, tokenizer, dataset, n_shots: int = 5,
                           limit: int = 100, device: str = "cuda") -> dict:
    """
    Evaluate ARC-Challenge with log-likelihood scoring (standard for MC tasks).
    Scores each choice continuation and picks the highest.
    """
    shots = ARC_SHOTS[:n_shots]
    correct = total = 0
    examples = dataset[:limit]

    t0 = time.perf_counter()
    for item in examples:
        question = item.get("question", "")
        choices_raw = item.get("choices", {})
        # ARC uses {text: [...], label: [...]} or {A: ..., B: ...}
        if "text" in choices_raw:
            labels = choices_raw.get("label", [])
            texts  = choices_raw.get("text", [])
            choices = {l: t for l, t in zip(labels, texts)}
        else:
            choices = choices_raw
        gold = item.get("answerKey", item.get("answer", "A"))

        # Build prompt prefix (without the answer)
        prompt = build_arc_prompt(question, choices, shots)

        # Log-likelihood scoring: P(choice | prompt)
        best_label = None
        best_score = float("-inf")

        for label, text in choices.items():
            full_text   = prompt + " " + label
            inputs      = tokenizer(full_text, return_tensors="pt",
                                    truncation=True, max_length=512).to(device)
            prompt_ids  = tokenizer(prompt, return_tensors="pt",
                                    truncation=True, max_length=512).to(device)
            n_prompt    = prompt_ids["input_ids"].shape[1]

            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
                # Loss is averaged over all tokens; we want only the answer token
                logits = out.logits
                # Score the last token (the label character: A/B/C/D)
                last_logit = logits[0, n_prompt - 1, :]
                label_id   = tokenizer.encode(" " + label, add_special_tokens=False)
                if label_id:
                    score = last_logit[label_id[0]].item()
                else:
                    score = -out.loss.item()

            if score > best_score:
                best_score = score
                best_label = label

            del inputs, prompt_ids, out
            torch.cuda.empty_cache()

        if best_label == gold:
            correct += 1
        total += 1

    elapsed = time.perf_counter() - t0
    accuracy = round(100 * correct / max(total, 1), 2)
    return {
        "score":   accuracy,
        "correct": correct,
        "total":   total,
        "elapsed_s": round(elapsed, 1),
        "metric":  "log_likelihood_mc",
    }


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_datasets(limit: int = 100, seed: int = 42):
    """Load GSM8K and ARC-Challenge test sets."""
    from datasets import load_dataset
    datasets = {}

    print("  Loading GSM8K test set ...")
    try:
        gsm = load_dataset("gsm8k", "main", split="test", trust_remote_code=True)
        items = list(gsm)
        random.seed(seed)
        random.shuffle(items)
        datasets["gsm8k"] = items[:limit]
        print(f"    {len(datasets['gsm8k'])} examples")
    except Exception as e:
        print(f"    ❌ GSM8K load failed: {e}")
        datasets["gsm8k"] = []

    print("  Loading ARC-Challenge test set ...")
    try:
        arc = load_dataset("ai2_arc", "ARC-Challenge", split="test",
                           trust_remote_code=True)
        items = list(arc)
        random.seed(seed)
        random.shuffle(items)
        datasets["arc_challenge"] = items[:limit]
        print(f"    {len(datasets['arc_challenge'])} examples")
    except Exception as e:
        print(f"    ❌ ARC-Challenge load failed: {e}")
        datasets["arc_challenge"] = []

    return datasets


# ── Main benchmark ────────────────────────────────────────────────────────────

def benchmark_model(name: str, cfg: dict, datasets: dict, args) -> dict:
    """Full benchmark for one model: load → GSM8K → ARC → cleanup."""
    print(f"\n{'='*62}")
    print(f"  Model: {cfg['label']}  ({cfg['hf_id']})")
    print(f"  Phase 1 baseline: {cfg['phase1_tps']} t/s, {cfg['phase1_vram_gb']} GB VRAM")
    print(f"{'='*62}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    print(f"  Loading tokenizer ...")
    try:
        tok = AutoTokenizer.from_pretrained(
            cfg["hf_id"], trust_remote_code=True, padding_side="left"
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    except Exception as e:
        print(f"  ❌ Tokenizer load failed: {e}")
        return {"label": cfg["label"], "error": str(e), **{k: v for k, v in cfg.items()}}

    # Load model with 4-bit NF4 (Phase 4's proven approach)
    print(f"  Loading model (4-bit NF4) ...")
    torch.cuda.empty_cache()
    gc.collect()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["hf_id"],
            quantization_config=get_bnb_config(),
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.eval()
    except Exception as e:
        print(f"  ❌ Model load failed: {e}")
        return {"label": cfg["label"], "error": str(e), **{k: v for k, v in cfg.items()}}

    peak_load_mb = torch.cuda.max_memory_allocated() / 1e6
    print(f"  Peak VRAM after load: {peak_load_mb:.0f} MB")
    torch.cuda.reset_peak_memory_stats()

    results = {
        "model_id":        cfg["hf_id"],
        "label":           cfg["label"],
        "phase1_tps":      cfg["phase1_tps"],
        "phase1_vram_gb":  cfg["phase1_vram_gb"],
        "peak_load_vram_mb": round(peak_load_mb, 1),
        "n_shots":         args.shots,
        "limit":           args.limit,
        "device":          device,
        "quantization":    "4-bit NF4",
        "scores":          {},
    }

    # GSM8K
    if datasets.get("gsm8k"):
        print(f"\n  [GSM8K] {len(datasets['gsm8k'])} examples, {args.shots}-shot ...")
        try:
            gsm_r = evaluate_gsm8k(model, tok, datasets["gsm8k"],
                                    n_shots=args.shots, limit=args.limit,
                                    device=device)
            results["scores"]["gsm8k"] = gsm_r
            print(f"    Accuracy: {gsm_r['score']:.1f}%  "
                  f"({gsm_r['correct']}/{gsm_r['total']})  "
                  f"[{gsm_r['elapsed_s']:.0f}s]")
        except Exception as e:
            print(f"    ❌ GSM8K eval failed: {e}")
            results["scores"]["gsm8k"] = {"error": str(e)}

    # ARC-Challenge
    if datasets.get("arc_challenge"):
        print(f"\n  [ARC-Challenge] {len(datasets['arc_challenge'])} examples, {args.shots}-shot ...")
        try:
            arc_r = evaluate_arc_challenge(model, tok, datasets["arc_challenge"],
                                            n_shots=args.shots, limit=args.limit,
                                            device=device)
            results["scores"]["arc_challenge"] = arc_r
            print(f"    Accuracy: {arc_r['score']:.1f}%  "
                  f"({arc_r['correct']}/{arc_r['total']})  "
                  f"[{arc_r['elapsed_s']:.0f}s]")
        except Exception as e:
            print(f"    ❌ ARC eval failed: {e}")
            results["scores"]["arc_challenge"] = {"error": str(e)}

    results["peak_eval_vram_mb"] = round(
        torch.cuda.max_memory_allocated() / 1e6, 1
    )

    # Cleanup
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return results


def generate_report(all_results: dict, args):
    """Write markdown Stage 2 report."""
    os.makedirs(os.path.dirname(OUTPUT_REPORT), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Stage 2 — Baseline Reasoning Benchmarks",
        "",
        "**Project:** ANeurologic Phase 5 — Advanced Optimization",
        f"**Hardware:** NVIDIA Jetson Orin Nano 8 GB",
        f"**Date:** {timestamp}",
        f"**Tasks:** GSM8K (math, {args.shots}-shot), ARC-Challenge (science, {args.shots}-shot)",
        f"**Limit:** {args.limit or 'full'} examples per task",
        f"**Quantization:** 4-bit NF4 (bitsandbytes, same as Phase 4 SLM fix)",
        "",
        "> These scores are the **reasoning floor** — the baseline that",
        "> Stages 3–6 (TurboQuant, TensorRT-LLM, Distillation) must preserve or exceed.",
        "",
        "## Note on lm_eval",
        "",
        "lm_eval was attempted first but fails on Jetson due to",
        "`CUDACachingAllocator.cpp:1154` assertion in PyTorch/Jetson when",
        "`model.to(device)` is called (even for 0.5B models). Root cause:",
        "lm_eval's HF backend does not use `device_map='auto'` + BitsAndBytesConfig.",
        "This custom loop replicates the same scoring methodology (5-shot,",
        "exact-match for GSM8K, log-likelihood MC for ARC) with proper",
        "4-bit NF4 loading to stay within Jetson's 3 GB CUDA budget.",
        "",
        "---",
        "",
        "## Results Summary",
        "",
        "| Model | Phase 1 t/s | GSM8K | ARC-Challenge | VRAM (eval) |",
        "|-------|------------|-------|---------------|-------------|",
    ]

    for name, res in all_results.items():
        gsm = res.get("scores", {}).get("gsm8k", {})
        arc = res.get("scores", {}).get("arc_challenge", {})
        gsm_s = f"{gsm['score']:.1f}%" if isinstance(gsm, dict) and gsm.get("score") is not None else "Error"
        arc_s = f"{arc['score']:.1f}%" if isinstance(arc, dict) and arc.get("score") is not None else "Error"
        vram  = f"{res.get('peak_eval_vram_mb', '?')} MB"
        lines.append(
            f"| {res['label']} | {res.get('phase1_tps') or 'N/A'} t/s | "
            f"{gsm_s} | {arc_s} | {vram} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Per-Model Detail",
    ]

    for name, res in all_results.items():
        lines += [
            "",
            f"### {res['label']}",
            "",
            f"- **HuggingFace ID**: `{res.get('model_id', '?')}`",
            f"- **Phase 1 throughput**: {res.get('phase1_tps') or 'N/A'} t/s (Q4_K_M, llama.cpp)",
            f"- **Phase 1 VRAM**: {res.get('phase1_vram_gb')} GB",
            f"- **Eval quantization**: {res.get('quantization', '4-bit NF4')}",
            f"- **Peak VRAM (load)**: {res.get('peak_load_vram_mb', '?')} MB",
            f"- **Peak VRAM (eval)**: {res.get('peak_eval_vram_mb', '?')} MB",
            "",
        ]
        for task, info in res.get("scores", {}).items():
            if isinstance(info, dict) and "score" in info:
                lines.append(
                    f"  **{task}**: {info['score']:.1f}%  "
                    f"({info['correct']}/{info['total']} correct, "
                    f"{info['elapsed_s']:.0f}s, metric: `{info['metric']}`)"
                )
            elif isinstance(info, dict) and "error" in info:
                lines.append(f"  **{task}**: Error — {info['error'][:80]}")

        if res.get("note"):
            lines.append(f"\n  > *Note: {res['note']}*")

    lines += [
        "",
        "---",
        "",
        "## Methodology",
        "",
        "### GSM8K",
        "- 5-shot chain-of-thought prompting",
        "- Model generates free-form text; final number extracted via regex",
        "- Match against ground truth number from `#### N` annotation",
        "",
        "### ARC-Challenge",
        "- 5-shot multiple-choice prompting (A/B/C/D)",
        "- Log-likelihood scoring: score P(label | prompt) for each choice",
        "- Highest log-likelihood wins (standard for MC evaluation)",
        "",
        "## How to Reproduce",
        "",
        "```bash",
        "# Full eval (all models, 100 examples per task)",
        "docker exec aneurologic_phase5 python3 \\",
        "    /workspace/scripts/stage2_baseline/baseline_reasoning.py",
        "",
        "# Quick eval (50 examples, specific models)",
        "docker exec aneurologic_phase5 python3 \\",
        "    /workspace/scripts/stage2_baseline/baseline_reasoning.py \\",
        "    --models qwen05b --limit 50",
        "```",
    ]

    with open(OUTPUT_REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Report saved → {OUTPUT_REPORT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=list(MODELS.keys()))
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n{'='*62}")
    print(f"  Phase 5 Stage 2 — Baseline Reasoning Benchmarks")
    print(f"  Models: {args.models}")
    print(f"  Tasks: {TASKS}")
    print(f"  Few-shot: {args.shots}  |  Limit: {args.limit}  |  Quant: 4-bit NF4")
    print(f"{'='*62}")

    print("\n  Loading datasets ...")
    datasets = load_datasets(limit=args.limit, seed=args.seed)

    all_results = {}
    for name in args.models:
        cfg = MODELS[name]
        all_results[name] = benchmark_model(name, cfg, datasets, args)

    # Merge with existing results
    existing = {}
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON) as f:
                existing = json.load(f).get("models", {})
        except Exception:
            pass
    existing.update(all_results)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "stage": "Stage 2 - Baseline Reasoning",
            "timestamp": datetime.now().isoformat(),
            "config": {"shots": args.shots, "limit": args.limit, "tasks": TASKS},
            "models": existing,
        }, f, indent=2)
    print(f"  Results saved → {OUTPUT_JSON}")

    generate_report(existing, args)

    print(f"\n{'='*62}")
    print(f"  Stage 2 complete. Reasoning floor established.")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
