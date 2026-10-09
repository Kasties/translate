#!/usr/bin/env bash
# Source this before setup/training/evaluation to keep large caches on the SSD.
TDDE19_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$TDDE19_ROOT/artifacts/cache/uv}"
export HF_HOME="${TDDE19_HF_HOME:-$TDDE19_ROOT/artifacts/cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$TDDE19_ROOT/artifacts/cache/xdg}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$TDDE19_ROOT/artifacts/cache/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$TDDE19_ROOT/artifacts/cache/triton}"
export TOKENIZERS_PARALLELISM=false
