#!/usr/bin/env bash
# Build llama.cpp with CUDA and convert the exact official Google checkpoint to GGUF.
set -euo pipefail

models_dir=${S2S_MODELS_DIR:-/mnt/ssd/models}
export UV_CACHE_DIR=${UV_CACHE_DIR:-${models_dir}/uv-cache}
source_dir="${models_dir}/gemma-4-12b/hf"
llama_dir="${models_dir}/llama.cpp"
gguf="${models_dir}/gemma-4-12b/gemma-4-12b-q8_0.gguf"
revision=023679ed352de9bb66cc873c9009ce3482585c08
commit=4ceb1719101f32637b841206c172f3f058ffc182
expected_sha=fe054ae05ff7f44318fd8ae90d58992531455c7ed31356704088f0f2d8c8009a

mkdir -p "$source_dir"
if [[ ! -d "$llama_dir/.git" ]]; then
    git clone https://github.com/ggml-org/llama.cpp.git "$llama_dir"
    git -c "safe.directory=$llama_dir" -C "$llama_dir" checkout "$commit"
fi
actual_commit=$(git -c "safe.directory=$llama_dir" -C "$llama_dir" rev-parse HEAD)
if [[ "$actual_commit" != "$commit" ]]; then
    printf 'Existing llama.cpp checkout is %s; expected %s. Choose a fresh S2S_MODELS_DIR.\n' "$actual_commit" "$commit" >&2
    exit 1
fi
cmake -S "$llama_dir" -B "$llama_dir/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build "$llama_dir/build" --target llama-server -j "${S2S_BUILD_JOBS:-8}"

base="https://huggingface.co/google/gemma-4-12B/resolve/$revision"
for name in config.json generation_config.json processor_config.json tokenizer.json tokenizer_config.json README.md; do
    if [[ ! -s "$source_dir/$name" ]]; then
        curl -fL --retry 3 "$base/$name" -o "$source_dir/$name"
    fi
done
if [[ ! -s "$source_dir/model.safetensors" ]]; then
    curl -fL --retry 3 "$base/model.safetensors" -o "$source_dir/model.safetensors.part"
    mv "$source_dir/model.safetensors.part" "$source_dir/model.safetensors"
fi
printf '%s  %s\n' "$expected_sha" "$source_dir/model.safetensors" | sha256sum --check

if [[ ! -s "$gguf" ]]; then
    mkdir -p "${models_dir}/conversion-tmp"
    export TMPDIR="${models_dir}/conversion-tmp"
    uv venv "$llama_dir/.venv"
    uv pip install --python "$llama_dir/.venv/bin/python" \
        'torch==2.11.0' 'numpy~=2.2.6' 'sentencepiece>=0.1.98,<0.3.0' \
        'gguf>=0.1.0' 'safetensors>=0.4' 'transformers>=5.0' 'protobuf>=4.21,<5'
    "$llama_dir/.venv/bin/python" "$llama_dir/convert_hf_to_gguf.py" "$source_dir" \
        --outtype q8_0 --outfile "$gguf.part" --use-temp-file
    mv "$gguf.part" "$gguf"
fi
printf 'Ready: %s\n' "$gguf"
