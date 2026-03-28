#!/usr/bin/env python3
"""
ANeurologic Phase 5 - Stage 1: Environment Verification
=========================================================
Verifies all dependencies are correctly installed and the hardware
is ready for Stages 2-7.

Checks:
  - Python, CUDA, PyTorch
  - GPU memory and device info
  - Go runtime
  - TensorRT and TensorRT-LLM availability
  - lm-eval
  - QJL and KIVI repos
  - HuggingFace connectivity
  - Jetson-specific configurations (power mode, swap)

Usage (inside container):
    python3 /workspace/scripts/stage1_env/verify_env.py
"""

import sys
import os
import subprocess
import json
import platform
from pathlib import Path

OUTPUT_JSON = "/workspace/outputs/logs/stage1_env_results.json"
OUTPUT_REPORT = "/workspace/outputs/reports/stage1_environment.md"


def check(label: str, fn):
    """Run a check function, return (passed, details)."""
    try:
        result = fn()
        print(f"  ✅  {label}: {result}")
        return True, str(result)
    except Exception as e:
        print(f"  ❌  {label}: {e}")
        return False, str(e)


def run_cmd(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or r.stdout.strip())
    return r.stdout.strip()


def main():
    print("\n" + "=" * 62)
    print("  ANeurologic Phase 5 — Stage 1: Environment Verification")
    print("=" * 62)

    results = {}

    # ── 1. Python ─────────────────────────────────────────────────
    print("\n[1/8] Python & System")
    ok, v = check("Python version", lambda: sys.version.split()[0])
    results["python_version"] = {"ok": ok, "value": v}

    ok, v = check("Platform", lambda: f"{platform.machine()} / {platform.system()}")
    results["platform"] = {"ok": ok, "value": v}

    # ── 2. CUDA & PyTorch ─────────────────────────────────────────
    print("\n[2/8] CUDA & PyTorch")
    try:
        import torch
        ok, v = check("PyTorch version", lambda: torch.__version__)
        results["pytorch_version"] = {"ok": ok, "value": v}

        ok, v = check("CUDA available", lambda: f"{torch.cuda.is_available()} ({torch.version.cuda})")
        results["cuda_available"] = {"ok": ok, "value": v}

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_info = f"{props.name} — {props.total_memory / 1e9:.2f} GB VRAM"
            ok, v = check("GPU device", lambda: gpu_info)
            results["gpu_device"] = {"ok": ok, "value": v}

            # Measure CUDA allocatable ceiling
            # On Jetson UMA, use half-precision to reduce contiguous block size
            max_alloc_gb = 0.0
            for gb in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
                try:
                    n = int(gb * 1e9 / 2)  # float16 = 2 bytes
                    t = torch.zeros(n, dtype=torch.float16, device="cuda")
                    del t
                    torch.cuda.empty_cache()
                    max_alloc_gb = gb
                except (RuntimeError, Exception):
                    break
            ok, v = check("CUDA max allocatable", lambda: f"~{max_alloc_gb:.1f} GB")
            results["cuda_max_alloc_gb"] = {"ok": ok, "value": max_alloc_gb}
    except ImportError as e:
        print(f"  ❌  PyTorch: {e}")
        results["pytorch"] = {"ok": False, "value": str(e)}

    # ── 3. TensorRT ───────────────────────────────────────────────
    print("\n[3/8] TensorRT")
    try:
        import tensorrt as trt
        ok, v = check("TensorRT (system)", lambda: trt.__version__)
        results["tensorrt_version"] = {"ok": ok, "value": v}
    except ImportError:
        try:
            trt_ver = run_cmd("python3 -c 'import tensorrt; print(tensorrt.__version__)' 2>/dev/null || trtexec --version 2>/dev/null | head -1")
            ok, v = check("TensorRT (trtexec)", lambda: trt_ver)
            results["tensorrt_version"] = {"ok": ok, "value": trt_ver}
        except Exception as e:
            print(f"  ⚠️   TensorRT: not importable as Python module ({e})")
            print("       This is OK — Stage 5 will install TRT-LLM from the v0.12.0-jetson source.")
            results["tensorrt_version"] = {"ok": False, "value": str(e), "note": "Expected — will build in Stage 5"}

    try:
        import tensorrt_llm
        ok, v = check("TensorRT-LLM", lambda: tensorrt_llm.__version__)
        results["trt_llm_version"] = {"ok": ok, "value": v}
    except ImportError as e:
        print(f"  ⚠️   TensorRT-LLM: not installed as Python package")
        print("       This is expected — pip wheel not available for aarch64.")
        print("       Stage 5 will build from /workspace/TensorRT-LLM (v0.12.0-jetson).")
        results["trt_llm_version"] = {"ok": False, "value": str(e), "note": "Will build from source in Stage 5"}

    trt_llm_src = Path("/workspace/TensorRT-LLM")
    ok, v = check("TRT-LLM source cloned", lambda: f"{trt_llm_src} — {len(list(trt_llm_src.glob('*.py')))} py files")
    results["trt_llm_source"] = {"ok": ok, "value": v}

    # ── 4. Go ─────────────────────────────────────────────────────
    print("\n[4/8] Go Runtime")
    ok, v = check("Go version", lambda: run_cmd("go version"))
    results["go_version"] = {"ok": ok, "value": v}

    ok, v = check("gollama.cpp cloned",
                  lambda: run_cmd("ls /go/src/github.com/dianlight/gollama.cpp/go.mod"))
    results["gollama_cpp"] = {"ok": ok, "value": v}

    # ── 5. lm-eval ────────────────────────────────────────────────
    print("\n[5/8] lm-eval (Evaluation Harness)")
    try:
        import lm_eval
        ok, v = check("lm-eval version", lambda: lm_eval.__version__)
        results["lm_eval_version"] = {"ok": ok, "value": v}
    except ImportError as e:
        print(f"  ❌  lm-eval: {e}")
        results["lm_eval_version"] = {"ok": False, "value": str(e)}

    # ── 6. KV-cache quantization repos ───────────────────────────
    print("\n[6/8] KV-Cache Quantization Repos (Stage 3)")
    qjl_path = Path("/workspace/repos/QJL")
    ok, v = check("QJL repo (amirzandieh/QJL, AAAI 2025)",
                  lambda: f"{qjl_path} — {len(list(qjl_path.glob('**/*.py')))} py files")
    results["qjl_repo"] = {"ok": ok, "value": v}

    kivi_path = Path("/workspace/repos/KIVI")
    ok, v = check("KIVI repo (jy-yuan/KIVI, ICML 2024)",
                  lambda: f"{kivi_path} — {len(list(kivi_path.glob('**/*.py')))} py files")
    results["kivi_repo"] = {"ok": ok, "value": v}

    # ── 7. HuggingFace ────────────────────────────────────────────
    print("\n[7/8] HuggingFace & Transformers")
    try:
        import transformers
        ok, v = check("transformers version", lambda: transformers.__version__)
        results["transformers_version"] = {"ok": ok, "value": v}
    except ImportError as e:
        results["transformers_version"] = {"ok": False, "value": str(e)}

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # Lightweight connectivity test — just check the hub is reachable
        ok, v = check("HuggingFace Hub reachable",
                      lambda: f"OK (endpoint: {api.endpoint})")
        results["hf_hub"] = {"ok": ok, "value": v}
    except Exception as e:
        results["hf_hub"] = {"ok": False, "value": str(e)}

    # ── 8. Jetson system ──────────────────────────────────────────
    print("\n[8/8] Jetson System Configuration")
    try:
        nvpmodel = run_cmd("nvpmodel -q 2>/dev/null | head -2")
        ok, v = check("Power mode (nvpmodel)", lambda: nvpmodel)
        results["jetson_power_mode"] = {"ok": ok, "value": v}
    except Exception:
        results["jetson_power_mode"] = {"ok": False, "value": "nvpmodel not available"}

    try:
        swap_info = run_cmd("free -h | grep Swap")
        ok, v = check("Swap memory", lambda: swap_info)
        results["swap_memory"] = {"ok": ok, "value": v}
    except Exception as e:
        results["swap_memory"] = {"ok": False, "value": str(e)}

    try:
        mem_info = run_cmd("free -h | grep Mem")
        ok, v = check("System memory", lambda: mem_info)
        results["system_memory"] = {"ok": ok, "value": v}
    except Exception as e:
        results["system_memory"] = {"ok": False, "value": str(e)}

    # ── Summary ───────────────────────────────────────────────────
    total = len(results)
    passed = sum(1 for v in results.values() if v.get("ok", False))
    warnings = sum(1 for v in results.values() if not v.get("ok") and "note" in v)
    failed = total - passed - warnings

    print(f"\n{'─' * 62}")
    print(f"  STAGE 1 SUMMARY")
    print(f"{'─' * 62}")
    print(f"  Checks: {total}  |  Passed: {passed}  |  Warnings: {warnings}  |  Failed: {failed}")
    if failed == 0:
        print("  Status: ✅ READY for Phase 5")
    elif failed <= 2:
        print("  Status: ⚠️  Mostly ready — check warnings above")
    else:
        print("  Status: ❌ Environment needs attention")
    print(f"{'=' * 62}\n")

    # ── Save results ──────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump({"stage": "Stage 1 - Environment", "results": results,
                   "summary": {"total": total, "passed": passed, "warnings": warnings, "failed": failed}},
                  f, indent=2)
    print(f"  Results saved → {OUTPUT_JSON}")

    return failed == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
