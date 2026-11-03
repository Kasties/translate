# Experiment pipeline

One command runs the whole project: pick source data, a generator model and a
model to fine-tune, and the pipeline serves, generates, curates, trains and
benchmarks, stopping every model server afterwards.

```
sources ──► generate ──► sft ──► eval
Korp/HF/    vLLM or      LoRA    EuroEval + idioms,
local       llama.cpp    (TRL)   base vs. base+adapter
```

## New machine

```bash
git clone <repo> && cd TDDE19
bash scripts/setup.sh            # .venv, .venv-train, .venv-eval from the lock files, then ./pipeline doctor
.venv/bin/hf auth login          # Gemma is gated: accept its license on huggingface.co first
./pipeline run smoke-test        # tiny models, ~15 min: proves every stage works
```

`./pipeline doctor` checks the environments, that PyTorch has kernels for the
GPU architecture (Blackwell is `sm_120`), the Hugging Face login, free disk
and ports. The lock files pin torch 2.13 / vLLM 0.30 wheels built for CUDA 13,
which support Blackwell; only the NVIDIA driver must be recent enough.
Models and datasets are cached under `artifacts/cache/` (override with
`TDDE19_HF_HOME`). On a machine with a shared Hugging Face cache, export
`HF_HUB_CACHE` instead; the pipeline then keeps it and leaves `HF_HOME` alone.

## Run an experiment

```bash
./pipeline new                       # asks for data source + models, writes configs/pipeline/<name>.yaml
./pipeline run <name> --dry-run      # print every command without running anything
./pipeline run <name>                # run all stages
./pipeline status <name>             # what is done, and the command to continue
```

An experiment file lists only what differs from
[configs/pipeline/default.yaml](../configs/pipeline/default.yaml), which
documents every setting. `<name>` can be a path, or a file name in
`configs/pipeline/` or `configs/pipeline/examples/`. Any setting can be
overridden without editing files:

```bash
./pipeline run my-exp --set sft.model=Qwen/Qwen3-8B-Base --set sft.epochs=2
```

Typos in setting names are errors. `./pipeline show <name>` prints the merged
configuration.

## Choosing data

| `data.source` | What it reads |
| --- | --- |
| `korp` | Svenska Wikipedia through the Korp API, as before. Responses are cached in `data/raw/`. |
| `hf` | A Hugging Face dataset: `data.hf.dataset`, `subset`, `split`, `max_documents`. Parquet/JSONL shards are streamed; nothing else is downloaded. |
| `local` | `data.local.path`: a file, directory or glob of `.txt`/`.md` (one document each) or `.jsonl`/`.json`/`.parquet`/`.csv` records. |
| `experiment:<name>` | Another experiment's passages, unchanged. |

For `hf` and `local`, `data.fields` names the text, title, id, url and
(optionally) topic columns. Text is re-tokenized into Korp's spacing, split
into paragraphs of 25–450 tokens and given the topic whose `nlu.topics`
keyword it contains, exactly as the Korp query does. Unmatched paragraphs
are dropped rather than given an invented label. If a topic has no passages in
a split, the stage warns: read more documents, change the keywords with
`datagen.overrides: {nlu: {topics: ...}}`, or generate fewer tasks. A
`.jsonl` that already has the passage schema (for example
`data/nlu/passages.jsonl`) is reused as-is.

## Choosing models

| Setting | Used for | Default |
| --- | --- | --- |
| `generator.model` | writing and curating synthetic data | `google/gemma-4-12B-it` |
| `sft.model` | the model that is fine-tuned | `google/gemma-4-12B` |
| `eval.model` | the benchmarked base (defaults to `sft.model`) | |

Use any Hugging Face id. The generator runs in vLLM (`generator.backend: vllm`),
in llama.cpp with a GGUF (`llamacpp`, see the README), or on a server you
started yourself (`external` + `generator.endpoint`). Training loads the
architecture the checkpoint declares, so Gemma, Llama, Qwen, Mistral and
similar models work without code changes. Pin `revision` for
reproducible runs.

On a 96 GB card the defaults train in bf16 (`sft.quantization: none`), with
8 examples per step and 2 accumulation steps; that is the same effective
batch of 16 as the 4090 pilot. These defaults do not fit a 24 GB card: the
12B model alone is about 24 GB in bf16. `configs/pipeline/examples/rtx4090-pilot.yaml`
holds the 24 GB settings (llama.cpp Q8_0 + MTP generator, QLoRA, 4-bit
evaluation) and can be layered over any experiment as a hardware profile:

```bash
./pipeline run register-native --profile rtx4090-pilot
export S2S_PIPELINE_PROFILE=rtx4090-pilot   # or make it the default on that machine
```

The profile applies after the experiment file and before `--set`; `--profile none`
ignores the environment variable. The run header names the active profile. If a
model server dies with a CUDA out-of-memory error, the error message suggests the
profile.

## Starting from any stage

Finished stages are skipped, so rerunning a command continues after an
interruption. Generation resumes its batches. `--from`, `--to` and `--only`
select stages, and any stage input can point at existing data:

```bash
# Fine-tune a different model on the checked-in dataset (no generation):
./pipeline run sft-qwen-on-nlu1000 --from sft

# Same data as an earlier experiment, new model, as a one-liner:
./pipeline run --name qwen3-8b --from sft --set sft.model=Qwen/Qwen3-8B-Base \
    --set sft.train_file=experiment:korp-gemma4

# Train on a Hugging Face dataset, or mix sources:
./pipeline run --name mix --from sft \
    --set 'sft.train_file=[experiment:korp-gemma4, hf:org/swedish-sft:train]'

# Only benchmark (a base model, or an adapter from another experiment):
./pipeline run eval-only --only eval
./pipeline run --name reeval --only eval --set eval.adapter=experiment:korp-gemma4

# Generate more rows for a build that stopped at max_batches:
./pipeline run my-exp --from generate --set datagen.max_batches=10
```

`sft.train_file` accepts pipeline exports (checked against their
`accepted.jsonl` provenance), other `.jsonl`/`.json`/`.parquet` files and
`hf:org/name[:split]` or `hf:org/name@subset[:split]`. Rows may be
`messages`/`conversations`, `prompt`+`completion`, `instruction`/`input`/`output`
or `question`+`answer`. Only the final assistant turn is trained on.

A finished stage is never redone with different settings. For example,
changing `sft.learning_rate` after training is refused; run under a new
`--name` instead and reuse the earlier stages with `experiment:<old name>`.
Serving settings (ports, memory) and enabling another benchmark are allowed.

## Outputs

| Path | Contents |
| --- | --- |
| `artifacts/pipeline/<name>/pipeline.json` | resolved config, per-stage settings, status and timing |
| `artifacts/pipeline/<name>/logs/` | one log per stage and per model server |
| `artifacts/pipeline/<name>/data/nlu/` | source passages + manifest |
| `artifacts/source2synth/runs/<name>/` | `train.jsonl`, `accepted.jsonl`, `report.json` (child batches alongside) |
| `artifacts/finetuning/<name>/` | `adapter/`, `manifest.json`, trainer state |
| `artifacts/euroeval/<name>/`, `artifacts/idioms/<name>/` | per-model results + manifest |

These are the same locations as the manual commands, so
`scripts/export-experiments.py` and the lab book pick pipeline runs up
unchanged. The base and fine-tuned models are served as `base` and `tuned`
(`eval.names`).

## Troubleshooting

* **Port already in use**: an interrupted run's server may still be alive.
  Check with `nvidia-smi`, stop it, or change `generator.port`/`eval.port`.
* **Server exits during startup**: the error shows the end of its log. Common
  causes are GPU memory (lower `gpu_memory_utilization`/`max_model_len`) and a
  gated model without login.
* **`nvcc fatal: Unknown option` / FlashInfer "Ninja build failed"**: FlashInfer
  JIT-compiles kernels with the system CUDA toolkit, which must match the wheels'
  CUDA 13. The pipeline sets `VLLM_USE_FLASHINFER_SAMPLER=0` to avoid this; for
  manual `vllm serve`, export it yourself or install a CUDA 13 toolkit.
* **"no usable rows in this batch"**: every candidate of one task failed schema
  validation, common with small or new generator models. The seeds are recorded
  as rejected and the build continues. If a task never succeeds, the model
  cannot follow that task's JSON schema.
* **Resume refused after changing generator settings**: `nlu-build` protects
  a build from mixing configurations. Use a new `--name`, or deliberately
  record the change with `--set datagen.refresh_inputs=true`.
* **Idioms fails with "no BOS token"**: the benchmark's protocol needs one
  (e.g. Qwen tokenizers have none). Set `eval.idioms.enabled: false`.
* **Chat-tuned model as the base for EuroEval**: set
  `eval.euroeval.generative_type: instruction_tuned`.
