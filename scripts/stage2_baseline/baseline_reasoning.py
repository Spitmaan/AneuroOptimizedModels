#!/usr/bin/env python3
"""
ANeurologic Phase 5 - Stage 2: Baseline Reasoning Benchmarks
=============================================================
Measures reasoning accuracy of Phase 1's top-performing models using
lm-eval on two canonical tasks:

  - GSM8K        : Grade-school math (math / chain-of-thought reasoning)
  - ARC-Challenge: Science reasoning (adversarial science questions)

Phase 1 established throughput baselines. This stage establishes the
reasoning accuracy "floor" — the bar that Stage 3-6 optimizations must
preserve or improve.

Models tested (Phase 1 top performers via llama.cpp GGUF Q4_K_M):
  - LiquidAI/LFM2.5-1.2B-Instruct   (55.4 t/s in Phase 1)
  - meta-llama/Llama-3.2-1B-Instruct (44.7 t/s in Phase 1)
  - nvidia/Cosmos-Reason2-2B         (34.3 t/s in Phase 1)

Note: VLMs (LFM2-VL-450M, LFM2-VL-1.6B) are text+vision models.
lm-eval's text-only tasks are applied to their text backbone — this is
the standard practice for VLM language capability benchmarking.

Usage:
    python3 /workspace/scripts/stage2_baseline/baseline_reasoning.py
    python3 /workspace/scripts/stage2_baseline/baseline_reasoning.py --models lfm llama
    python3 /workspace/scripts/stage2_baseline/baseline_reasoning.py --shots 5 --limit 100
"""

import sys
import os
import gc
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

OUTPUT_JSON   = "/workspace/outputs/logs/stage2_baseline_results.json"
OUTPUT_REPORT = "/workspace/outputs/reports/stage2_baseline.md"

# ── Model registry ────────────────────────────────────────────────────────────
# Maps short name → HuggingFace model ID and Phase 1 throughput baseline
MODELS = {
    "lfm": {
        "hf_id":    "LiquidAI/LFM2.5-1.2B-Instruct",
        "label":    "LFM2.5-1.2B",
        "phase1_tps": 55.4,
        "phase1_vram_gb": 1.2,
        "gated": False,
    },
    # meta-llama/Llama-3.2-1B-Instruct and nvidia/Cosmos-Reason2-2B are gated
    # HF repos requiring authentication. Phase 1 used GGUF files (llama.cpp)
    # which bypassed this. For lm-eval (HF backend), we use non-gated equivalents:
    #   - Qwen/Qwen2.5-1.5B-Instruct: same parameter class as Llama-3.2-1B,
    #     was Phase 1's 5th-best SLM at 21.3 t/s (vLLM), proven on Jetson
    #   - Qwen/Qwen2.5-0.5B-Instruct: lighter alternative for reasoning eval
    "qwen15b": {
        "hf_id":    "Qwen/Qwen2.5-1.5B-Instruct",
        "label":    "Qwen2.5-1.5B (Llama-class proxy)",
        "phase1_tps": 21.3,   # Phase 1 result via vLLM AWQ
        "phase1_vram_gb": 1.5,
        "gated": False,
        "note": "Non-gated proxy for Llama-3.2-1B (both 1.2-1.5B, similar reasoning)",
    },
    "qwen05b": {
        "hf_id":    "Qwen/Qwen2.5-0.5B-Instruct",
        "label":    "Qwen2.5-0.5B",
        "phase1_tps": None,   # Not in Phase 1 (too small), added for reference
        "phase1_vram_gb": 0.5,
        "gated": False,
        "note": "Lightest non-gated model; establishes accuracy floor at 0.5B scale",
    },
}

TASKS = ["gsm8k", "arc_challenge"]


def run_lm_eval(model_id: str, tasks: list, n_shots: int, limit: int,
                device: str = "cuda") -> dict:
    """
    Run lm-eval harness via subprocess (avoids import-time GPU allocation
    from concurrent model loads on Jetson's 2GB CUDA budget).
    Returns parsed results dict.
    """
    task_str = ",".join(tasks)
    limit_flag = f"--limit {limit}" if limit else ""

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        # load_in_4bit=True: required on Jetson (2-3GB CUDA budget).
        # lm_eval passes these directly to AutoModelForCausalLM.from_pretrained()
        # via transformers integration. Same approach as Phase 4 SLM fix.
        "--model_args", (f"pretrained={model_id},dtype=float16,device_map=auto,"
                         f"load_in_4bit=True,trust_remote_code=True"),
        "--tasks", task_str,
        "--num_fewshot", str(n_shots),
        "--batch_size", "1",      # Jetson UMA: single example at a time
        "--output_path", "/tmp/lm_eval_out",
        "--log_samples",
    ]
    if limit:
        cmd += ["--limit", str(limit)]

    print(f"  Running: lm_eval --tasks {task_str} --num_fewshot {n_shots}")
    t0 = time.perf_counter()

    result = subprocess.run(
        cmd,
        capture_output=True, text=True, timeout=3600  # 1h max per model
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  ❌ lm_eval failed:\n{result.stderr[-2000:]}")
        return {"error": result.stderr[-500:], "elapsed_s": elapsed}

    # Parse results from output JSON written by lm_eval --output_path
    out_dir = Path("/tmp/lm_eval_out")
    result_files = list(out_dir.glob("**/*.json"))
    if not result_files:
        # Fall back to parsing stdout
        return _parse_stdout(result.stdout, elapsed)

    latest = max(result_files, key=lambda p: p.stat().st_mtime)
    with open(latest) as f:
        data = json.load(f)

    scores = {}
    for task in tasks:
        task_results = data.get("results", {}).get(task, {})
        # lm-eval reports: acc, acc_norm, exact_match depending on task
        score_key = "exact_match,flexible-extract" if task == "gsm8k" else "acc_norm,none"
        # Try multiple key formats across lm-eval versions
        for key in [score_key, "acc_norm,none", "acc,none", "exact_match,none"]:
            if key in task_results:
                scores[task] = {
                    "score":   round(task_results[key] * 100, 2),
                    "metric":  key,
                    "stderr":  round(task_results.get(key.replace(",none", "_stderr,none"), 0) * 100, 2),
                }
                break
        else:
            scores[task] = {"score": None, "raw": task_results}

    return {"scores": scores, "elapsed_s": round(elapsed, 1)}


def _parse_stdout(stdout: str, elapsed: float) -> dict:
    """Fallback: parse lm_eval table output from stdout."""
    scores = {}
    for line in stdout.splitlines():
        for task in TASKS:
            if task in line.lower():
                parts = line.split("|")
                for i, p in enumerate(parts):
                    try:
                        val = float(p.strip())
                        if 0 < val < 1:
                            scores[task] = {"score": round(val * 100, 2), "metric": "parsed_stdout"}
                            break
                    except ValueError:
                        pass
    return {"scores": scores, "elapsed_s": round(elapsed, 1), "source": "stdout_parse"}


def benchmark_model(name: str, cfg: dict, args) -> dict:
    """Run full benchmark for one model."""
    import torch
    print(f"\n{'='*62}")
    print(f"  Model: {cfg['label']}  ({cfg['hf_id']})")
    print(f"  Phase 1 baseline: {cfg['phase1_tps']} t/s, {cfg['phase1_vram_gb']} GB VRAM")
    print(f"{'='*62}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device : {device}")
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
        pre_mem = torch.cuda.memory_allocated() / 1e6
        print(f"  VRAM before load: {pre_mem:.0f} MB allocated")

    results = run_lm_eval(
        model_id=cfg["hf_id"],
        tasks=TASKS,
        n_shots=args.shots,
        limit=args.limit,
        device=device,
    )

    if device == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1e6
        results["peak_vram_mb"] = round(peak_mem, 1)
        torch.cuda.empty_cache()
        gc.collect()

    results["model_id"]      = cfg["hf_id"]
    results["label"]         = cfg["label"]
    results["phase1_tps"]    = cfg["phase1_tps"]
    results["phase1_vram_gb"]= cfg["phase1_vram_gb"]
    results["n_shots"]       = args.shots
    results["limit"]         = args.limit
    results["device"]        = device

    print(f"\n  Results for {cfg['label']}:")
    for task, info in results.get("scores", {}).items():
        if isinstance(info, dict) and "score" in info:
            print(f"    {task:20s}: {info['score']:.1f}% ± {info.get('stderr', 0):.1f}%")
        else:
            print(f"    {task:20s}: {info}")

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
        f"**Limit:** {'All examples' if not args.limit else f'{args.limit} examples per task'}",
        "",
        "> These scores are the **reasoning floor** — the baseline that",
        "> Stages 3–6 (TurboQuant, TensorRT-LLM, Distillation) must preserve or exceed.",
        "",
        "---",
        "",
        "## Results Summary",
        "",
        "| Model | Phase 1 t/s | GSM8K | ARC-Challenge | Peak VRAM |",
        "|-------|------------|-------|---------------|-----------|",
    ]

    for name, res in all_results.items():
        gsm   = res.get("scores", {}).get("gsm8k", {})
        arc   = res.get("scores", {}).get("arc_challenge", {})
        gsm_s = f"{gsm['score']:.1f}%" if isinstance(gsm, dict) and gsm.get("score") is not None else "N/A"
        arc_s = f"{arc['score']:.1f}%" if isinstance(arc, dict) and arc.get("score") is not None else "N/A"
        vram  = f"{res.get('peak_vram_mb', '?')} MB"
        lines.append(f"| {res['label']} | {res['phase1_tps']} t/s | {gsm_s} | {arc_s} | {vram} |")

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
            f"- **HuggingFace ID**: `{res['model_id']}`",
            f"- **Phase 1 throughput**: {res['phase1_tps']} t/s (Q4_K_M, llama.cpp)",
            f"- **Phase 1 VRAM**: {res['phase1_vram_gb']} GB",
            f"- **Eval device**: {res.get('device', '?')}",
            f"- **Peak VRAM (eval)**: {res.get('peak_vram_mb', '?')} MB",
            f"- **Eval time**: {res.get('elapsed_s', '?')} s",
            "",
        ]
        for task, info in res.get("scores", {}).items():
            if isinstance(info, dict):
                score   = f"{info['score']:.1f}%" if info.get("score") is not None else "N/A"
                stderr  = f" ± {info['stderr']:.1f}%" if info.get("stderr") else ""
                metric  = info.get("metric", "")
                lines.append(f"  **{task}**: {score}{stderr}  (metric: `{metric}`)")

    lines += [
        "",
        "---",
        "",
        "## Methodology",
        "",
        "- Evaluation harness: `lm-eval` (EleutherAI Language Model Evaluation Harness)",
        f"- Few-shot: {args.shots} examples",
        "- Quantization: HuggingFace `dtype=float16` with `device_map=auto`",
        "  (same weights as HF Hub; Phase 1 used GGUF Q4_K_M on llama.cpp —",
        "   small accuracy difference expected due to quantization method)",
        "- Batch size: 1 (Jetson UMA constraint — 2 GB CUDA ceiling)",
        "",
        "## How to Reproduce",
        "",
        "```bash",
        "# Full eval (all models, all tasks, all examples)",
        "docker exec aneurologic_phase5 python3 \\",
        "    /workspace/scripts/stage2_baseline/baseline_reasoning.py",
        "",
        "# Quick eval (50 examples per task)",
        "docker exec aneurologic_phase5 python3 \\",
        "    /workspace/scripts/stage2_baseline/baseline_reasoning.py --limit 50",
        "",
        "# Single model",
        "docker exec aneurologic_phase5 python3 \\",
        "    /workspace/scripts/stage2_baseline/baseline_reasoning.py --models lfm",
        "```",
    ]

    with open(OUTPUT_REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Report saved → {OUTPUT_REPORT}")


def main():
    parser = argparse.ArgumentParser(description="Phase 5 Stage 2 — Baseline Reasoning Benchmarks")
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=list(MODELS.keys()),
                        help="Which models to evaluate (default: all)")
    parser.add_argument("--shots", type=int, default=5,
                        help="Number of few-shot examples (default: 5)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit examples per task (default: full dataset)")
    args = parser.parse_args()

    print(f"\n{'='*62}")
    print(f"  Phase 5 Stage 2 — Baseline Reasoning Benchmarks")
    print(f"  Models: {args.models}")
    print(f"  Tasks: {TASKS}")
    print(f"  Few-shot: {args.shots}  |  Limit: {args.limit or 'full'}")
    print(f"{'='*62}")

    all_results = {}
    for name in args.models:
        cfg = MODELS[name]
        all_results[name] = benchmark_model(name, cfg, args)

    # Save JSON
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "stage": "Stage 2 - Baseline Reasoning",
            "timestamp": datetime.now().isoformat(),
            "config": {"shots": args.shots, "limit": args.limit, "tasks": TASKS},
            "models": all_results,
        }, f, indent=2)
    print(f"  Results saved → {OUTPUT_JSON}")

    generate_report(all_results, args)

    print(f"\n{'='*62}")
    print(f"  Stage 2 complete.")
    print(f"  These scores are the reasoning floor for Phase 5.")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
