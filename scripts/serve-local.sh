#!/usr/bin/env bash
set -euo pipefail

models_dir=${S2S_MODELS_DIR:-/mnt/ssd/models}
server=${LLAMA_SERVER:-${models_dir}/llama.cpp/build/bin/llama-server}
profile=${S2S_MODEL_PROFILE:-mtp}
case "$profile" in
    mtp)
        default_gguf=${models_dir}/gemma-4-12b-it/gemma-4-12B-it-Q8_0.gguf
        default_alias=google/gemma-4-12B-it
        default_template=auto
        default_spec=draft-mtp
        ;;
    base)
        default_gguf=${models_dir}/gemma-4-12b/gemma-4-12b-q8_0.gguf
        default_alias=google/gemma-4-12B
        default_template=$(dirname "$0")/gemma4-base.jinja
        default_spec=ngram-simple
        ;;
    *) printf 'S2S_MODEL_PROFILE must be mtp or base\n' >&2; exit 1 ;;
esac
gguf=${S2S_GGUF:-$default_gguf}
alias=${S2S_MODEL:-$default_alias}
port=${S2S_PORT:-8080}
parallel=${S2S_PARALLEL:-2}
if [[ ! "$parallel" =~ ^[1-9][0-9]*$ ]]; then
    printf 'S2S_PARALLEL must be a positive integer\n' >&2
    exit 1
fi
# llama.cpp divides this total context between slots; retain 4096 per request.
ctx=${S2S_CONTEXT:-$((4096 * parallel))}
template=${S2S_CHAT_TEMPLATE:-$default_template}
spec=${S2S_SPEC_TYPE:-$default_spec}
draft_tokens=${S2S_DRAFT_TOKENS:-3}
mtp_draft=${S2S_MTP_DRAFT:-}
if [[ "$profile" == mtp && "$spec" == draft-mtp && "$gguf" == "$default_gguf" && -z "$mtp_draft" ]]; then
    mtp_draft=${models_dir}/gemma-4-12b-it/mtp-gemma-4-12B-it-Q8_0.gguf
fi

if [[ ! -x "$server" ]]; then
    printf 'llama-server missing: %s\n' "$server" >&2
    exit 1
fi

case "$spec" in
    none|ngram-simple|draft-mtp) ;;
    *) printf 'S2S_SPEC_TYPE must be none, ngram-simple or draft-mtp\n' >&2; exit 1 ;;
esac
if [[ ! "$parallel" =~ ^[1-9][0-9]*$ || ! "$ctx" =~ ^[1-9][0-9]*$ || ! "$draft_tokens" =~ ^[1-9][0-9]*$ ]]; then
    printf 'S2S_PARALLEL, S2S_CONTEXT and S2S_DRAFT_TOKENS must be positive integers\n' >&2
    exit 1
fi
if [[ -n "$mtp_draft" && ( "$spec" != draft-mtp || ! -s "$mtp_draft" ) ]]; then
    printf 'S2S_MTP_DRAFT requires draft-mtp and a nonempty, matching MTP GGUF\n' >&2
    exit 1
fi
if [[ "$spec" == draft-mtp && -z "$mtp_draft" && "$gguf" == "${models_dir}/gemma-4-12b/gemma-4-12b-q8_0.gguf" ]]; then
    printf 'The default Gemma 4 12B GGUF has no MTP heads. Use ngram-simple, or supply a model with MTP heads/a matching S2S_MTP_DRAFT.\n' >&2
    exit 1
fi
if [[ ! -s "$gguf" ]]; then
    printf 'GGUF missing: %s (run scripts/setup-local.sh)\n' "$gguf" >&2
    exit 1
fi

args=(--model "$gguf" --alias "$alias" --host 127.0.0.1 --port "$port"
      --ctx-size "$ctx" --n-gpu-layers 99 --parallel "$parallel" --jinja --reasoning-format none
      --flash-attn on)
if [[ "$spec" != none ]]; then
    help_output=$("$server" --help 2>&1)
    if [[ "$help_output" != *"--spec-type"* || "$help_output" != *"--spec-draft-n-max"* || "$help_output" != *"$spec"* ]]; then
        printf 'This llama-server does not support %s with the required flags; rebuild using setup-local.sh or set S2S_SPEC_TYPE=none\n' "$spec" >&2
        exit 1
    fi
    args+=(--spec-type "$spec" --spec-draft-n-max "$draft_tokens")
fi
if [[ -n "$mtp_draft" ]]; then
    args+=(--model-draft "$mtp_draft")
fi
if [[ "$template" != auto ]]; then
    args+=(--chat-template-file "$template")
fi
if [[ "$profile" == mtp ]]; then
    args+=(--chat-template-kwargs '{"enable_thinking":false}')
fi
printf 'Serving %s: speculation=%s, draft_tokens=%s, slots=%s, total_context=%s\n' \
    "$alias" "$spec" "$draft_tokens" "$parallel" "$ctx" >&2
exec "$server" "${args[@]}"
