#!/usr/bin/env bash
# AneuroOptimizedModels/env.sh
#
# Sources Aneurologic/env.sh (parent monorepo) when present.  Falls
# back to standalone defaults when this repo is cloned on its own.

_THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PARENT_ENV="$(cd "$_THIS_DIR/.." 2>/dev/null && pwd)/env.sh"

if [ -n "$_PARENT_ENV" ] && [ -f "$_PARENT_ENV" ]; then
    source "$_PARENT_ENV"
else
    export ANEURO_HOME="${ANEURO_HOME:-$HOME}"
    export ANEURO_TOOLS="${ANEURO_TOOLS:-$ANEURO_HOME/tools}"
    export ANEURO_LLAMACPP_DIR="${ANEURO_LLAMACPP_DIR:-$ANEURO_TOOLS/llama.cpp}"
    export ANEURO_HF_CACHE_HF="${ANEURO_HF_CACHE_HF:-$ANEURO_HOME/.cache/huggingface}"
fi

# ── Repo-specific ─────────────────────────────────────────────────────
# Phase 5 edge optimization scripts read $LLAMA_CPP_BUILD_DIR (legacy
# name, kept for backwards-compat).  Same value as $ANEURO_LLAMACPP_DIR/build.
export LLAMA_CPP_BUILD_DIR="${LLAMA_CPP_BUILD_DIR:-$ANEURO_LLAMACPP_DIR/build}"

# docker-compose.yml's HF cache mount uses HF_CACHE_DIR (legacy compose-
# style env var).  Mirror to ANEURO_HF_CACHE_HF for consistency.
export HF_CACHE_DIR="${HF_CACHE_DIR:-$ANEURO_HF_CACHE_HF}"

export OPTIMIZED_DIR="${OPTIMIZED_DIR:-$_THIS_DIR}"
