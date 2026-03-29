#!/usr/bin/env python3
"""
GGUF Edge Optimizer Benchmark
==============================
Measures throughput (pp + tg t/s) AND accuracy (GSM8K + ARC-Challenge)
for any GGUF model with configurable llama.cpp flags.

Each step in the optimization ladder produces ONE result record so that
apple-to-apple comparisons are possible across all dimensions simultaneously.

Usage:
  python3 bench_gguf.py --model /path/to/model.gguf
  python3 bench_gguf.py --model /path/to/model.gguf --label "Flash-Attn" --flags "-fa 1"
  python3 bench_gguf.py --model /path/to/model.gguf --speed-only   # skip accuracy (faster)
  python3 bench_gguf.py --compare  # print comparison table from results JSON
"""

import argparse
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# ── Paths ────────────────────────────────────────────────────────────────────
LLAMA_BENCH  = Path("/home/spitman/tools/llama.cpp/build/bin/llama-bench")
LLAMA_SERVER = Path("/home/spitman/tools/llama.cpp/build/bin/llama-server")
SERVER_PORT  = 8765
RESULTS_JSON = Path(__file__).parent.parent.parent / "outputs" / "logs" / "edge_opt" / "results.json"

# ── Few-shot prompts ──────────────────────────────────────────────────────────
GSM8K_SHOTS = (
    "Solve the math problem and give only the final numeric answer on the last line.\n\n"
    "Question: There are 15 trees in the grove. Workers will plant more. After planting there are 21. "
    "How many did workers plant?\nAnswer: 6\n\n"
    "Question: If there are 3 cars and 2 more arrive, how many total?\nAnswer: 5\n\n"
    "Question: Leah had 32 chocolates, sister had 42. They ate 35. How many pieces left total?\nAnswer: 39\n\n"
    "Question: "
)

ARC_SHOTS = (
    "Choose the best answer (A, B, C, or D). Reply with just the letter.\n\n"
    "Question: Which of the following is an example of a physical change?\n"
    "(A) Burning wood (B) Rusting iron (C) Melting ice (D) Digesting food\nAnswer: C\n\n"
    "Question: A student measures how fast a ball rolls down a ramp. Which measurement is most important?\n"
    "(A) Color of ball (B) Weight of ramp (C) Distance traveled (D) Temperature\nAnswer: C\n\n"
    "Question: Which energy transformation occurs in a battery-powered flashlight?\n"
    "(A) Chemical to light (B) Mechanical to electrical (C) Solar to chemical (D) Thermal to light\nAnswer: A\n\n"
    "Question: "
)


# ── Speed benchmark ───────────────────────────────────────────────────────────
def run_speed_bench(model_path: str, extra_flags: list[str]) -> dict:
    """Run llama-bench and return {pp512, tg128, pp_std, tg_std}."""
    cmd = [
        str(LLAMA_BENCH),
        "-m", model_path,
        "-p", "512",
        "-n", "128",
        "-ngl", "99",
        "-r", "3",
        "-o", "json",
    ] + extra_flags
    print(f"  Speed: {' '.join(cmd[-10:])}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    stderr_clean = "\n".join(
        l for l in result.stderr.splitlines()
        if not any(x in l for x in ["ggml_cuda", "GGML_CUDA", "compute capability", "VMM"])
    )
    if stderr_clean.strip():
        print(f"  [bench stderr] {stderr_clean[:200]}")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"  ERROR: llama-bench output not valid JSON:\n{result.stdout[:500]}")
        return {"pp512": 0.0, "tg128": 0.0, "pp_std": 0.0, "tg_std": 0.0}

    out = {"pp512": 0.0, "tg128": 0.0, "pp_std": 0.0, "tg_std": 0.0}
    for entry in data:
        t = entry.get("n_gen", 0)
        p = entry.get("n_prompt", 0)
        tps = entry.get("avg_ts", 0.0)
        std = entry.get("stddev_ts", 0.0)
        if p == 512 and t == 0:
            out["pp512"] = round(tps, 2)
            out["pp_std"] = round(std, 2)
        elif t == 128 and p == 0:
            out["tg128"] = round(tps, 2)
            out["tg_std"] = round(std, 2)
    return out


# ── llama-server helpers ──────────────────────────────────────────────────────
_server_proc = None

def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def start_server(model_path: str, extra_flags: list[str]) -> None:
    global _server_proc
    if _port_in_use(SERVER_PORT):
        print(f"  WARNING: port {SERVER_PORT} already in use — assuming server is ready")
        return
    cmd = [
        str(LLAMA_SERVER),
        "-m", model_path,
        "-ngl", "99",
        "--port", str(SERVER_PORT),
        "--ctx-size", "2048",
        "-n", "256",
        "--no-mmap",
        "-t", "4",
    ] + extra_flags
    print(f"  Starting llama-server on port {SERVER_PORT} ...")
    _server_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    for _ in range(90):
        time.sleep(1)
        if _port_in_use(SERVER_PORT):
            try:
                r = requests.get(f"http://127.0.0.1:{SERVER_PORT}/health", timeout=2)
                if r.status_code == 200:
                    print("  Server ready.")
                    return
            except Exception:
                pass
    raise RuntimeError("llama-server failed to start within 90s")

def stop_server() -> None:
    global _server_proc
    if _server_proc is not None:
        try:
            os.killpg(os.getpgid(_server_proc.pid), signal.SIGTERM)
            _server_proc.wait(timeout=10)
        except Exception:
            pass
        _server_proc = None
    # Kill any stale llama-server on this port (e.g. from a previous crash)
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    # Wait for Jetson UMA CUDA context to fully release
    time.sleep(5)

def complete(prompt: str, n_predict: int = 256, temperature: float = 0.0,
             stop: list[str] | None = None) -> str:
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "stop": stop or ["\n\n\n", "Question:", "Answer:"],
    }
    r = requests.post(
        f"http://127.0.0.1:{SERVER_PORT}/completion",
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("content", "").strip()


# ── GSM8K evaluation ──────────────────────────────────────────────────────────
_HF_CACHE = "/tmp/hf_datasets_cache"

def eval_gsm8k(n_samples: int = 20) -> dict:
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=_HF_CACHE)
    except Exception as e:
        print(f"  ERROR loading GSM8K: {e}")
        return {"score": 0.0, "correct": 0, "total": 0}

    samples = list(ds.select(range(n_samples)))
    correct = 0
    for i, s in enumerate(samples):
        q = s["question"]
        gold_match = re.search(r"####\s*([\d,\-]+)", s["answer"])
        gold = gold_match.group(1).replace(",", "").strip() if gold_match else ""

        prompt = GSM8K_SHOTS + q + "\nAnswer:"
        try:
            response = complete(prompt, n_predict=128, temperature=0.0,
                                stop=["\n\n", "Question:", "####"])
        except Exception as e:
            print(f"  [gsm8k {i}] server error: {e}")
            continue

        # extract last number from response
        nums = re.findall(r"-?\d[\d,]*", response)
        pred = nums[-1].replace(",", "") if nums else ""
        hit = (pred == gold)
        if hit:
            correct += 1
        if (i + 1) % 5 == 0:
            print(f"    GSM8K {i+1}/{n_samples}: {correct}/{i+1} correct so far")

    score = round(100.0 * correct / n_samples, 1)
    print(f"  GSM8K result: {correct}/{n_samples} = {score}%")
    return {"score": score, "correct": correct, "total": n_samples}


# ── ARC-Challenge evaluation ──────────────────────────────────────────────────
def eval_arc(n_samples: int = 20) -> dict:
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", cache_dir=_HF_CACHE)
    except Exception as e:
        print(f"  ERROR loading ARC-Challenge: {e}")
        return {"score": 0.0, "correct": 0, "total": 0}

    samples = list(ds.select(range(n_samples)))
    correct = 0
    label_map = {"1": "A", "2": "B", "3": "C", "4": "D",
                 "A": "A", "B": "B", "C": "C", "D": "D"}

    for i, s in enumerate(samples):
        q = s["question"]
        choices = s["choices"]["text"]
        labels  = s["choices"]["label"]
        gold_key = label_map.get(s["answerKey"], s["answerKey"])

        # Build MCQ prompt
        choice_str = " ".join(f"({labels[j]}) {choices[j]}" for j in range(len(choices)))
        prompt = ARC_SHOTS + q + "\n" + choice_str + "\nAnswer:"

        try:
            response = complete(prompt, n_predict=5, temperature=0.0,
                                stop=["\n", "Question:"])
        except Exception as e:
            print(f"  [arc {i}] server error: {e}")
            continue

        # find first A/B/C/D in response
        match = re.search(r"\b([A-D])\b", response.upper())
        pred = match.group(1) if match else ""
        hit = (pred == gold_key)
        if hit:
            correct += 1
        if (i + 1) % 5 == 0:
            print(f"    ARC {i+1}/{n_samples}: {correct}/{i+1} correct so far")

    score = round(100.0 * correct / n_samples, 1)
    print(f"  ARC-Challenge result: {correct}/{n_samples} = {score}%")
    return {"score": score, "correct": correct, "total": n_samples}


# ── Results I/O ───────────────────────────────────────────────────────────────
def load_results() -> list:
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON) as f:
            return json.load(f)
    return []

def save_results(results: list) -> None:
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

def print_table(results: list) -> None:
    if not results:
        print("No results yet.")
        return
    header = f"{'Label':<30} {'pp512 t/s':>12} {'tg128 t/s':>12} {'GSM8K':>8} {'ARC':>8}"
    print("\n" + "=" * 75)
    print(header)
    print("-" * 75)
    for r in results:
        spd = r.get("speed", {})
        acc = r.get("accuracy", {})
        gsm = acc.get("gsm8k", {}).get("score", "—")
        arc = acc.get("arc", {}).get("score", "—")
        pp  = spd.get("pp512", "—")
        tg  = spd.get("tg128", "—")
        label = r["label"][:29]
        gsm_s = f"{gsm}%" if isinstance(gsm, float) else "—"
        arc_s = f"{arc}%" if isinstance(arc, float) else "—"
        print(f"{label:<30} {pp:>12} {tg:>12} {gsm_s:>8} {arc_s:>8}")
    print("=" * 75)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GGUF optimizer benchmark")
    parser.add_argument("--model",  default="", help="Path to GGUF file")
    parser.add_argument("--label",  default="", help="Human-readable label for this run")
    parser.add_argument("--flags",  default="", help='Extra llama.cpp flags as a string, e.g. "-fa 1 -ctk q8_0"')
    parser.add_argument("--n-samples", type=int, default=20, help="Samples per accuracy task")
    parser.add_argument("--speed-only", action="store_true", help="Skip accuracy eval")
    parser.add_argument("--compare", action="store_true", help="Print comparison table and exit")
    args = parser.parse_args()

    if args.compare:
        print_table(load_results())
        return

    if not args.model:
        parser.error("--model is required unless --compare is used")

    model_path = str(Path(args.model).resolve())
    if not Path(model_path).exists():
        print(f"ERROR: model file not found: {model_path}")
        sys.exit(1)

    extra_flags = shlex.split(args.flags) if args.flags else []
    label = args.label or (Path(model_path).stem + ("  " + args.flags if args.flags else ""))

    print(f"\n{'='*60}")
    print(f"  Model : {Path(model_path).name}")
    print(f"  Label : {label}")
    print(f"  Flags : {args.flags or '(none — baseline)'}")
    print(f"{'='*60}\n")

    record = {
        "label": label,
        "model": Path(model_path).name,
        "flags": args.flags,
        "timestamp": datetime.now().isoformat(),
        "speed": {},
        "accuracy": {},
    }

    # Ensure any stale llama processes are gone before we start
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    subprocess.run(["pkill", "-f", "llama-bench"], capture_output=True)
    time.sleep(3)

    # Step 1: Speed
    print("[1/3] Throughput benchmark (llama-bench, 3 repetitions) ...")
    record["speed"] = run_speed_bench(model_path, extra_flags)
    print(f"  → pp512: {record['speed']['pp512']} ± {record['speed']['pp_std']} t/s")
    print(f"  → tg128: {record['speed']['tg128']} ± {record['speed']['tg_std']} t/s")

    if args.speed_only:
        print("\n[speed-only mode — skipping accuracy eval]")
    else:
        # Step 2: Start server
        print("\n[2/3] Starting llama-server for accuracy eval ...")
        try:
            start_server(model_path, extra_flags)
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            save_results(load_results() + [record])
            return

        # Step 3: Accuracy
        print("\n[3/3] Accuracy evaluation ...")
        print(f"  GSM8K ({args.n_samples} samples) ...")
        record["accuracy"]["gsm8k"] = eval_gsm8k(args.n_samples)
        print(f"  ARC-Challenge ({args.n_samples} samples) ...")
        record["accuracy"]["arc"] = eval_arc(args.n_samples)
        stop_server()

    # Save
    all_results = load_results()
    # Replace existing record with same label if present
    all_results = [r for r in all_results if r["label"] != label]
    all_results.append(record)
    save_results(all_results)
    print(f"\n  Results saved → {RESULTS_JSON}")

    # Print full table
    print_table(all_results)


if __name__ == "__main__":
    main()
