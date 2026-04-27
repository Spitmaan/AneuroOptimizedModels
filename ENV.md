# AneuroOptimizedModels environment variables

This repo's [`env.sh`](env.sh) sources the parent monorepo's
[`Aneurologic/env.sh`](../env.sh) when present, falls back to repo-local
defaults when cloned standalone.

For shared variables see the [parent ENV.md](../ENV.md).

## Variables specific to AneuroOptimizedModels

| Variable | Default | Used by |
|---|---|---|
| `LLAMA_CPP_BUILD_DIR` | `$ANEURO_LLAMACPP_DIR/build` | `scripts/edge_optimization/bench_*.py` (`llama-bench`, `llama-server` paths) |
| `HF_CACHE_DIR` | `$ANEURO_HF_CACHE_HF` | `docker-compose.yml` HF cache mount (compose-style legacy var name) |
| `OPTIMIZED_DIR` | dir of this `env.sh` | repo root |

`LLAMA_CPP_BUILD_DIR` and `HF_CACHE_DIR` are the legacy var names used
inside the Phase 5 bench scripts. They mirror
`$ANEURO_LLAMACPP_DIR/build` and `$ANEURO_HF_CACHE_HF` from the parent
env.sh for consistency.

## Override examples

Use a custom llama.cpp build:
```bash
LLAMA_CPP_BUILD_DIR=/opt/llama.cpp/build \
  python3 scripts/edge_optimization/bench_gguf.py
```

Use a non-default HF cache for docker-compose:
```bash
echo "HF_CACHE_DIR=/big-disk/hf_cache" > .env
docker compose up
```

(Compose reads `.env` automatically — no need to source `env.sh` first
for compose itself, just set the var.)
