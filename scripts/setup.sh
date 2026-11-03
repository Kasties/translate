#!/usr/bin/env bash
# Create/update the three pinned environments on a fresh machine, then check it.
#   bash scripts/setup.sh              # .venv (data generation), .venv-train, .venv-eval
#   bash scripts/setup.sh --llamacpp   # also build llama.cpp + GGUFs (generator.backend: llamacpp)
# Needs: uv, an NVIDIA driver for CUDA 13 (torch 2.13 / vLLM wheels bring their own CUDA runtime).
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/experiment-env.sh"
cd "$TDDE19_ROOT"

if ! command -v uv >/dev/null; then
    printf 'uv is missing: curl -LsSf https://astral.sh/uv/install.sh | sh\n' >&2
    exit 1
fi

printf '== .venv (data generation)\n'
uv sync --locked

for pair in "train:requirements-training.lock" "eval:requirements-evaluation.lock"; do
    name=${pair%%:*}
    lock=${pair#*:}
    printf '== .venv-%s (%s)\n' "$name" "$lock"
    if [[ ! -x ".venv-$name/bin/python" ]]; then
        uv venv --python 3.13 ".venv-$name"
    fi
    uv pip install --python ".venv-$name/bin/python" -r "$lock"
done

if [[ "${1:-}" == "--llamacpp" ]]; then
    printf '== llama.cpp\n'
    bash scripts/setup-local.sh
fi

chmod +x pipeline
if [[ ! -f "$HOME/.cache/huggingface/token" && -z "${HF_TOKEN:-}" ]]; then
    printf '\nGemma models are gated: accept the license on huggingface.co, then run\n'
    printf '  .venv/bin/hf auth login\n'
fi
printf '\n'
./pipeline doctor
printf '\nNext: ./pipeline run smoke-test   (tiny models, checks every stage)\n'
