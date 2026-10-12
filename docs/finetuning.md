# TRL fine-tuning and EuroEval pilot

This is a one-epoch QLoRA pilot on `google/gemma-4-12B` **base**, using the
existing curated Swedish NLU export. It is a baseline experiment, not the
two-stage intermediate-model training and curation procedure from Source2Synth.

The default input currently contains 853 examples. Its `nlu-build` report is
still incomplete relative to the 1,000-row target; NER and GED are underrepresented.
The training manifest records the actual counts and SHA-256 hashes. Training
checks each exported answer against `accepted.jsonl` and rejects held-out `eval`
provenance. All 853 rows are used; there is no random row-level validation split
that could place the same source article on both sides.

## Setup

Run from the repository root, using Python 3.13 and the RTX 4090. The separate
environments leave the existing DataDesigner environment unchanged. Sourcing
the environment script puts large package/model/compiler caches under `artifacts/`
on the project disk. The initial setup downloads the original Hugging Face weights;
the existing GGUF inference files cannot be used directly for TRL training.

```bash
source scripts/experiment-env.sh
uv venv --python 3.13 .venv-train
uv pip install --python .venv-train/bin/python -r requirements-training.lock
uv venv --python 3.13 .venv-eval
uv pip install --python .venv-eval/bin/python -r requirements-evaluation.lock
```

## Train

Inspect/edit `configs/finetune.yaml`. Defaults are one epoch, NF4 double
quantization, BF16 compute, LoRA rank 16 / alpha 32, learning rate 2e-4,
one example per device and 16 gradient accumulation steps. Attention and MLP
projections are trained; the pretrained weights stay frozen.

```bash
# Check data/provenance without downloading or loading a model.
.venv/bin/python -m source2synth.finetune --dry-run

# Train the one-epoch pilot.
source scripts/experiment-env.sh
.venv-train/bin/python -m source2synth.finetune

# Optional separate two-step plumbing check, not a quality experiment.
.venv-train/bin/python -m source2synth.finetune --max-steps 2 \
  --output-dir artifacts/finetuning/gemma4-smoke
```

Output directories must be empty to prevent accidental replacement. The final
adapter and tokenizer are in `artifacts/finetuning/gemma4-base-nlu-pilot/adapter`;
the parent directory contains the manifest, metrics, trainer state and checkpoint.
The base model revision is pinned in the config and retained in the adapter.

The base checkpoint has no chat template. Training uses an explicit
`SYSTEM: / USER: / ASSISTANT:` template, saved with the tokenizer. Prompt labels
are masked with `-100`; only the answer and its ending contribute to the loss.
Overlong examples fail with their row number instead of silently truncating the
answer. Gradient checkpointing and TRL's chunked cross-entropy reduce VRAM use.
Stop other GPU model servers before training or serving evaluation.

## Compare with EuroEval

After training, serve the base and adapter in a single vLLM process:

```bash
bash scripts/serve-eval.sh
```

In another terminal:

```bash
source scripts/experiment-env.sh
.venv-eval/bin/python -m source2synth.euroeval
```

The default pilot evaluates **SweReC** (sentiment) and **ScaLA-sv** (linguistic
acceptability), using validation data and one iteration. Both model aliases use
the same base-model few-shot prompting, context limit and 4-bit inference backend.
Thus the comparison measures the adapter's effect without changing the evaluation
prompt protocol. This does not evaluate the adapter's chat instruction-following
behavior, and it does not cover all seven synthetic training tasks.

The benchmark uses the official EuroEval datasets, prompts and scoring, through
its custom inference API support. The synthetic training set is not substituted
for an official benchmark. EuroEval test splits are reserved for final evaluation.
These one-iteration development scores are **not leaderboard-comparable**, and
small differences do not establish a reliable improvement. Wikipedia-grounded
training also does not establish absence of overlap with Wikipedia-based benchmarks.

EuroEval runs through `source2synth/euroeval_cli.py`, which applies one fix to the
API backend. EuroEval's native vLLM backend strips few-shot prompts when the
tokenizer attaches the space to the label (`" ja"`), but its API backend does not.
Every Gemma prompt then ended in `": "` and most generations were not labels
(`<strong>ja</strong>`, `0`, blank). EuroEval replaced those with the "closest"
label, so scores in `artifacts/euroeval/gemma4-nlu-pilot/` are **invalid**. The
wrapper applies EuroEval's own `should_prompts_be_stripped` rule with the pinned
tokenizer and changes nothing else. To keep every prompt, raw generation and parsed
label, use `python -m source2synth.euroeval_debug`. Its `--no-strip-prompts` option
reproduces the original bug.

Each model gets its own results JSONL and log under
`artifacts/euroeval/gemma4-nlu-pilot/`, with a shared manifest linking the exact
training run. To expand the comparison or increase repetitions, select a new output:

```bash
.venv-eval/bin/python -m source2synth.euroeval \
  --datasets swerec scala-sv suc3 multi-wiki-qa-sv --iterations 10 \
  --output-dir artifacts/euroeval/gemma4-nlu-expanded
```

## Swedish idioms (minimal pairs)

[liu-nlp/swedish-idioms](https://huggingface.co/datasets/liu-nlp/swedish-idioms)
([Kunz, TACL 2026](https://arxiv.org/abs/2602.03484)) tests whether a model prefers
a conventional Swedish idiom over a plausible non-idiomatic keyword, e.g. *anar
ugglor i mossen* vs *anar råttor i mossen*. It has 478 pairs; 100 form a challenge
subset of idioms without an English or German counterpart. It is not a EuroEval
dataset and needs no prompt, so it runs against the same `serve-eval.sh` server:

```bash
source scripts/experiment-env.sh
.venv-eval/bin/python -m source2synth.idioms
```

`source2synth/idioms.py` follows the paper's reference scorer
(`jekunz/idiomatic-language`, `compute_perplexity_unbatched`): each stripped
sentence is tokenized with BOS by the pinned tokenizer, vLLM returns the logprob
of every following token, and a pair is correct only when the idiomatic sentence
has strictly lower mean-token perplexity. The dataset revision is pinned and its
SHA-256 recorded. Results go to `artifacts/idioms/gemma4-nlu-pilot/`:
per-pair perplexities in `<model>/predictions.jsonl`, and accuracy with 95%
Wilson intervals (as in the paper's code) plus an exact McNemar test on the
paired base/adapter disagreements in `summary.json`. Use `--limit 20` and a new
`--output-dir` for a smoke test. The paper scored bf16 Hugging Face models; this
uses the 4-bit vLLM backend, so compare numbers within the project only. The
dataset's generative (question → keyword) setup is not implemented.

References: [TRL SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer),
[Gemma 4 Unified](https://huggingface.co/docs/transformers/model_doc/gemma4_unified),
[EuroEval usage](https://github.com/EuroEval/EuroEval/blob/main/src/frontend/md/python-package.md),
[vLLM BitsAndBytes](https://docs.vllm.ai/en/v0.30.0/features/quantization/bnb/).
