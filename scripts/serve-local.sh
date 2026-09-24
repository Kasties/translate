#!/usr/bin/env bash
set -euo pipefail

models_dir=${S2S_MODELS_DIR:-/mnt/ssd/models}
server=${LLAMA_SERVER:-${models_dir}/llama.cpp/build/bin/llama-server}
gguf=${S2S_GGUF:-${models_dir}/gemma-4-12b/gemma-4-12b-q8_0.gguf}
alias=${S2S_MODEL:-google/gemma-4-12B}
port=${S2S_PORT:-8080}
ctx=${S2S_CONTEXT:-4096}
template=${S2S_CHAT_TEMPLATE:-$(dirname "$0")/gemma4-base.jinja}

if [[ ! -x "$server" ]]; then
    printf 'llama-server missing: %s\n' "$server" >&2
    exit 1
fi
if [[ ! -s "$gguf" ]]; then
    printf 'GGUF missing: %s (run scripts/setup-local.sh)\n' "$gguf" >&2
    exit 1
fi

args=(--model "$gguf" --alias "$alias" --host 127.0.0.1 --port "$port"
      --ctx-size "$ctx" --n-gpu-layers 99 --parallel 1 --jinja --reasoning-format none)
if [[ "$template" != auto ]]; then
    args+=(--chat-template-file "$template")
fi
exec "$server" "${args[@]}"
