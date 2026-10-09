#!/usr/bin/env bash
# Serve the pinned base and trained adapter together for matched EuroEval runs.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/experiment-env.sh"
cd "$TDDE19_ROOT"
ADAPTER_PATH="${1:-artifacts/finetuning/gemma4-base-nlu-pilot/adapter}"
ADAPTER_PATH="$(realpath "$ADAPTER_PATH")"
test -f "$ADAPTER_PATH/adapter_config.json"
exec .venv-eval/bin/vllm serve google/gemma-4-12B \
  --revision 023679ed352de9bb66cc873c9009ce3482585c08 \
  --host 127.0.0.1 --port 8000 \
  --served-model-name gemma4-base \
  --dtype bfloat16 --quantization bitsandbytes \
  --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --max-num-seqs 16 --enforce-eager \
  --enable-lora --max-lora-rank 16 \
  --lora-modules "gemma4-nlu=$ADAPTER_PATH" \
  --language-model-only
