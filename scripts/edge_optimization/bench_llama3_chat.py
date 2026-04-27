#!/usr/bin/env python3
"""
Stage XI — Llama-3.2-1B with proper chat template
===================================================
Tests Llama-3.2-1B-Instruct Q4_K_S+FA using the Llama 3 instruction format
via /v1/chat/completions endpoint. The model is instruction-tuned for chat;
raw prompts (used in prior benchmarks) do not activate its instruction following.

Chat format applied by llama-server:
  <|begin_of_text|><|start_header_id|>system<|end_header_id|>
  {system}<|eot_id|><|start_header_id|>user<|end_header_id|>
  {user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Speed benchmark: identical to bench_gguf.py (llama-bench, pp512+tg128).
Accuracy benchmark: uses /v1/chat/completions with multi-turn few-shot.
"""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

import os
_LLAMA_BUILD = Path(os.environ.get(
    "LLAMA_CPP_BUILD_DIR",
    str(Path.home() / "tools" / "llama.cpp" / "build"),
))
LLAMA_BENCH  = _LLAMA_BUILD / "bin" / "llama-bench"
LLAMA_SERVER = _LLAMA_BUILD / "bin" / "llama-server"
SERVER_PORT  = 8765
RESULTS_JSON = Path(__file__).parent.parent.parent / "outputs" / "logs" / "edge_opt" / "results.json"
_HF_CACHE    = "/tmp/hf_datasets_cache"

# ── GSM8K few-shot as chat turns ──────────────────────────────────────────────
GSM8K_SYSTEM = "You are a math tutor. Solve each problem step by step, then give only the final numeric answer on the last line."

GSM8K_FEW_SHOT = [
    {"role": "user",      "content": "There are 15 trees in the grove. Workers will plant more. After planting there are 21. How many trees did workers plant?"},
    {"role": "assistant", "content": "21 - 15 = 6\n6"},
    {"role": "user",      "content": "If there are 3 cars in a parking lot and 2 more arrive, how many cars are in the lot?"},
    {"role": "assistant", "content": "3 + 2 = 5\n5"},
    {"role": "user",      "content": "Leah had 32 chocolates and her sister had 42. If they ate 35 total, how many pieces are left?"},
    {"role": "assistant", "content": "32 + 42 = 74\n74 - 35 = 39\n39"},
]

# ── ARC few-shot as chat turns ─────────────────────────────────────────────────
ARC_SYSTEM = "You are a science teacher. Answer multiple choice questions with just the letter (A, B, C, or D)."

ARC_FEW_SHOT = [
    {"role": "user",      "content": "Which of the following is an example of a physical change?\n(A) Burning wood (B) Rusting iron (C) Melting ice (D) Digesting food"},
    {"role": "assistant", "content": "C"},
    {"role": "user",      "content": "A student measures how fast a ball rolls down a ramp. Which measurement is most important?\n(A) Color of ball (B) Weight of ramp (C) Distance traveled (D) Temperature"},
    {"role": "assistant", "content": "C"},
    {"role": "user",      "content": "Which energy transformation occurs in a battery-powered flashlight?\n(A) Chemical to light (B) Mechanical to electrical (C) Solar to chemical (D) Thermal to light"},
    {"role": "assistant", "content": "A"},
]

# ── Speed benchmark ───────────────────────────────────────────────────────────
def run_speed_bench(model_path: str, extra_flags: list) -> dict:
    cmd = [str(LLAMA_BENCH), "-m", model_path, "-p", "512", "-n", "128",
           "-ngl", "99", "-r", "3", "-o", "json"] + extra_flags
    print(f"  Speed: {' '.join(cmd[-8:])}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"  ERROR: llama-bench non-JSON: {result.stdout[:200]}")
        return {"pp512": 0.0, "tg128": 0.0, "pp_std": 0.0, "tg_std": 0.0}
    out = {"pp512": 0.0, "tg128": 0.0, "pp_std": 0.0, "tg_std": 0.0}
    for entry in data:
        if entry.get("n_prompt") == 512 and entry.get("n_gen") == 0:
            out["pp512"] = round(entry.get("avg_ts", 0.0), 2)
            out["pp_std"] = round(entry.get("stddev_ts", 0.0), 2)
        elif entry.get("n_gen") == 128 and entry.get("n_prompt") == 0:
            out["tg128"] = round(entry.get("avg_ts", 0.0), 2)
            out["tg_std"] = round(entry.get("stddev_ts", 0.0), 2)
    return out

# ── Server management ─────────────────────────────────────────────────────────
_server_proc = None

def _port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def start_server(model_path: str, extra_flags: list) -> None:
    global _server_proc
    if _port_in_use(SERVER_PORT):
        print(f"  WARNING: port {SERVER_PORT} already in use")
        return
    cmd = [str(LLAMA_SERVER), "-m", model_path, "-ngl", "99",
           "--port", str(SERVER_PORT), "--ctx-size", "2048", "-n", "256",
           "--no-mmap", "-t", "4"] + extra_flags
    print(f"  Starting llama-server on port {SERVER_PORT} ...")
    _server_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                    preexec_fn=os.setsid)
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
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    time.sleep(5)

def chat_complete(messages: list, n_predict: int = 64, temperature: float = 0.0) -> str:
    payload = {"messages": messages, "max_tokens": n_predict, "temperature": temperature}
    r = requests.post(f"http://127.0.0.1:{SERVER_PORT}/v1/chat/completions",
                      json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

# ── GSM8K eval with chat template ─────────────────────────────────────────────
def eval_gsm8k_chat(n_samples: int = 20) -> dict:
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=_HF_CACHE)
    except Exception as e:
        print(f"  ERROR loading GSM8K: {e}")
        return {"score": 0.0, "correct": 0, "total": 0}
    samples = list(ds.select(range(n_samples)))
    correct = 0
    for i, s in enumerate(samples):
        gold_match = re.search(r"####\s*([\d,\-]+)", s["answer"])
        gold = gold_match.group(1).replace(",", "").strip() if gold_match else ""
        messages = [{"role": "system", "content": GSM8K_SYSTEM}] + \
                   GSM8K_FEW_SHOT + \
                   [{"role": "user", "content": s["question"]}]
        try:
            response = chat_complete(messages, n_predict=128, temperature=0.0)
        except Exception as e:
            print(f"  [gsm8k {i}] error: {e}")
            continue
        nums = re.findall(r"-?\d[\d,]*", response)
        pred = nums[-1].replace(",", "") if nums else ""
        if pred == gold:
            correct += 1
        if (i + 1) % 5 == 0:
            print(f"    GSM8K {i+1}/{n_samples}: {correct}/{i+1} correct")
    score = round(100.0 * correct / n_samples, 1)
    print(f"  GSM8K result: {correct}/{n_samples} = {score}%")
    return {"score": score, "correct": correct, "total": n_samples}

# ── ARC eval with chat template ────────────────────────────────────────────────
def eval_arc_chat(n_samples: int = 20) -> dict:
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", cache_dir=_HF_CACHE)
    except Exception as e:
        print(f"  ERROR loading ARC: {e}")
        return {"score": 0.0, "correct": 0, "total": 0}
    samples = list(ds.select(range(n_samples)))
    label_map = {"1": "A", "2": "B", "3": "C", "4": "D",
                 "A": "A", "B": "B", "C": "C", "D": "D"}
    correct = 0
    for i, s in enumerate(samples):
        gold_key = label_map.get(s["answerKey"], s["answerKey"])
        choices = s["choices"]["text"]
        labels  = s["choices"]["label"]
        choice_str = " ".join(f"({labels[j]}) {choices[j]}" for j in range(len(choices)))
        user_msg = f"{s['question']}\n{choice_str}"
        messages = [{"role": "system", "content": ARC_SYSTEM}] + \
                   ARC_FEW_SHOT + \
                   [{"role": "user", "content": user_msg}]
        try:
            response = chat_complete(messages, n_predict=5, temperature=0.0)
        except Exception as e:
            print(f"  [arc {i}] error: {e}")
            continue
        match = re.search(r"\b([A-D])\b", response.upper())
        pred = match.group(1) if match else ""
        if pred == gold_key:
            correct += 1
        if (i + 1) % 5 == 0:
            print(f"    ARC {i+1}/{n_samples}: {correct}/{i+1} correct")
    score = round(100.0 * correct / n_samples, 1)
    print(f"  ARC result: {correct}/{n_samples} = {score}%")
    return {"score": score, "correct": correct, "total": n_samples}

# ── Results I/O ───────────────────────────────────────────────────────────────
def load_results():
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON) as f:
            return json.load(f)
    return []

def save_results(results):
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

def print_table(results):
    if not results:
        print("No results yet."); return
    print("\n" + "=" * 75)
    print(f"{'Label':<30} {'pp512 t/s':>12} {'tg128 t/s':>12} {'GSM8K':>8} {'ARC':>8}")
    print("-" * 75)
    for r in results:
        spd = r.get("speed", {}); acc = r.get("accuracy", {})
        gsm = acc.get("gsm8k", {}).get("score", "—")
        arc = acc.get("arc", {}).get("score", "—")
        gsm_s = f"{gsm}%" if isinstance(gsm, float) else "—"
        arc_s = f"{arc}%" if isinstance(arc, float) else "—"
        print(f"{r['label'][:29]:<30} {spd.get('pp512','—'):>12} {spd.get('tg128','—'):>12} {gsm_s:>8} {arc_s:>8}")
    print("=" * 75)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Llama-3 chat template benchmark")
    parser.add_argument("--model",  required=True)
    parser.add_argument("--label",  default="")
    parser.add_argument("--flags",  default="-fa 1")
    parser.add_argument("--n-samples", type=int, default=20)
    args = parser.parse_args()

    import shlex
    model_path  = str(Path(args.model).resolve())
    extra_flags = shlex.split(args.flags)
    label = args.label or (Path(model_path).stem + "_chat_" + args.flags)

    print(f"\n{'='*60}")
    print(f"  Model : {Path(model_path).name}")
    print(f"  Label : {label}")
    print(f"  Flags : {args.flags}")
    print(f"  Mode  : Llama-3 chat template (/v1/chat/completions)")
    print(f"{'='*60}\n")

    record = {"label": label, "model": Path(model_path).name, "flags": args.flags,
              "timestamp": datetime.now().isoformat(), "speed": {}, "accuracy": {},
              "notes": "Llama-3 chat template via /v1/chat/completions"}

    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    subprocess.run(["pkill", "-f", "llama-bench"],  capture_output=True)
    time.sleep(15)

    print("[1/3] Speed benchmark ...")
    record["speed"] = run_speed_bench(model_path, extra_flags)
    print(f"  → pp512: {record['speed']['pp512']} ± {record['speed']['pp_std']} t/s")
    print(f"  → tg128: {record['speed']['tg128']} ± {record['speed']['tg_std']} t/s")

    print("\n[2/3] Starting llama-server ...")
    try:
        start_server(model_path, extra_flags)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        save_results(load_results() + [record])
        return

    print("\n[3/3] Accuracy eval (chat template) ...")
    print(f"  GSM8K ({args.n_samples} samples) ...")
    record["accuracy"]["gsm8k"] = eval_gsm8k_chat(args.n_samples)
    print(f"  ARC-Challenge ({args.n_samples} samples) ...")
    record["accuracy"]["arc"] = eval_arc_chat(args.n_samples)
    stop_server()

    all_results = load_results()
    all_results = [r for r in all_results if r["label"] != label]
    all_results.append(record)
    save_results(all_results)
    print(f"\n  Results saved → {RESULTS_JSON}")
    print_table(all_results)

if __name__ == "__main__":
    main()
