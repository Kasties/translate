# Swedish Source2Synth: Korp → EuroEval NLU

This project uses [NeMo DataDesigner](https://docs.nvidia.com/nemo/datadesigner/) to construct and curate Swedish synthetic examples grounded in [Svenska Wikipedia in Korp](https://spraakbanken.gu.se/korp/#?corpus=wikipedia-sv). The earlier two-document QA pipeline and its method are preserved in [the MHQA guide](docs/source2synth-mhqa.md). This workflow adapts the real-source construction and independent solving/curation pattern of [Source2Synth](https://arxiv.org/abs/2409.08239v2); it does not claim to reproduce the paper's fine-tuning results.

## Quick start: the experiment pipeline

`./pipeline` runs everything with one command: source passages (Korp, a Hugging Face
dataset or local files) → synthetic data generation and curation → LoRA fine-tuning →
EuroEval + idioms benchmarks. It starts and stops the model servers itself and can start
at any stage, for example fine-tuning a new model on existing data. See
[the pipeline guide](docs/pipeline.md).

```bash
bash scripts/setup.sh                 # environments from the lock files + ./pipeline doctor
./pipeline run smoke-test             # tiny models: checks every stage on a new machine
./pipeline new                        # pick data + models interactively
./pipeline run my-exp                 # all stages; rerun to continue
./pipeline run --name qwen3 --from sft --set sft.model=Qwen/Qwen3-8B-Base \
    --set sft.train_file=artifacts/source2synth/runs/nlu-1000/train.jsonl
```

The sections below document the individual commands the pipeline runs.

## Register-first native Swedish targets

`register-native` samples documents by register and multilingual embedding cluster,
with a configurable 60% instruction backtranslation / 25% idiomatic rewriting /
10% grounded response / 5% NLU mix. Backtranslation and rewriting use the original
human passage as the assistant target; generated prompt text stays masked during
training. Riksdag debates, Wikipedia and Arbetsförmedlingen job ads are sampled from Korp; Litteraturbanken
(Korp `LB-OPEN`) is configured but off by default. Korp's forum corpora are
sentence-scrambled, so forum/opinion/advice text needs coherent local exports.

```bash
uv sync --locked --extra registers
./pipeline run register-native --only sources
./pipeline run register-native --from generate
```

Matched random and keyword baselines, source attribution, resumable generation,
and diversity per generated token are documented in the
[register data guide](docs/register-data.md). Existing NLU experiments keep their
original configuration and commands.

## Local model

The default is [Google's instruction-tuned Gemma 4 12B](https://huggingface.co/google/gemma-4-12B-it)
with its [matching MTP assistant](https://huggingface.co/google/gemma-4-12B-it-assistant).
[scripts/setup-local.sh](scripts/setup-local.sh) builds a pinned llama.cpp revision and
downloads Q8_0 GGUFs published by [the llama.cpp maintainers](https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF),
pinned to revision `e3e681731089efaa3f0917336944ac64752db8ba`. Both files are verified
against SHA-256 before use. The instruction-tuned checkpoint supplies its own chat
template; thinking is disabled for these short structured tasks.

The setup machine has an RTX 4090 with 24 GB VRAM. The target and assistant downloads
total about 13.1 GB. Requirements: `git`, `cmake`, CUDA toolkit, `uv` and network access.
The earlier base checkpoint and existing run artifacts are preserved.

```bash
uv sync --locked
bash scripts/setup-local.sh
bash scripts/serve-local.sh           # leave this terminal running; IT + MTP by default
```

In a second terminal:

```bash
uv run python -m source2synth model-check
```

The server listens on `127.0.0.1:8080`; DataDesigner connects to its OpenAI-compatible `/v1` API. To switch GGUF models later, use matching server alias and client ID:

```bash
S2S_GGUF=/absolute/path/other.gguf S2S_MODEL=my-local-model S2S_CHAT_TEMPLATE=auto S2S_SPEC_TYPE=none bash scripts/serve-local.sh
S2S_MODEL=my-local-model uv run python -m source2synth model-check
S2S_MODEL=my-local-model uv run python -m source2synth nlu-generate --records 2 --run-name alternative-model
```

`S2S_MODELS_DIR`, `LLAMA_SERVER`, `S2S_PORT`, `S2S_CONTEXT`, `S2S_PARALLEL`, `S2S_GGUF` and `S2S_MODEL` configure the launcher. `S2S_CHAT_TEMPLATE=auto` uses a replacement GGUF's embedded chat template; otherwise set a Jinja file. `model.id` and `model.endpoint` in [configs/source2synth.yaml](configs/source2synth.yaml) configure DataDesigner. `S2S_MODEL` and `S2S_ENDPOINT` override YAML, or pass `--endpoint` before a CLI subcommand and `--model` to `nlu-generate`/`nlu-curate`. `model-check` catches a mismatched model ID before generation. The local loopback server needs no API key.

### MTP and inference throughput

The launcher uses `--spec-type draft-mtp` with the matching assistant GGUF by default,
three draft tokens, and two server slots with 4096 context tokens each. DataDesigner's
`model.max_parallel_requests` is also 2. The target verifies the assistant's proposals.
See [Google's MTP overview](https://ai.google.dev/gemma/docs/mtp/overview) and
[llama.cpp's speculative decoding documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md).

`S2S_SPEC_TYPE=none` disables speculation for a baseline; `ngram-simple` needs no
additional weights. `S2S_DRAFT_TOKENS` controls the draft length. For another compatible
model, set `S2S_GGUF`, `S2S_MODEL`, and optionally `S2S_MTP_DRAFT`. Models containing
native MTP heads need no assistant file. The launcher checks server flag support;
llama.cpp validates architecture/weights on loading. An arbitrary draft model is not
a matching MTP assistant.

`S2S_CONTEXT` is the **total** context across slots. Increase it when increasing
`S2S_PARALLEL`, and keep the client's concurrency within the server's slot count.
Measure your workload before increasing either setting further.

The original `google/gemma-4-12B` base GGUF has no MTP heads. To reproduce the original
model setup, retaining its explicit role template and using n-gram speculation:

```bash
S2S_MODEL_PROFILE=base bash scripts/setup-local.sh  # only if the old files are missing
S2S_MODEL_PROFILE=base bash scripts/serve-local.sh
S2S_MODEL=google/gemma-4-12B uv run python -m source2synth model-check
```

To compare throughput, restart the server with each setting and run the same benchmark
while no generation job is running:

```bash
uv run python scripts/benchmark-local.py --parallel 2 --output artifacts/benchmark.json
```

The benchmark reports end-to-end output tokens/second, timings, JSON completion
validity and a prompt hash. It warms up the model and disables prompt-cache reuse.
For an MTP comparison, keep the IT model and parallelism unchanged, changing only
`S2S_SPEC_TYPE=none` versus `draft-mtp`. To compare the old base setup, launch with
`S2S_MODEL_PROFILE=base S2S_SPEC_TYPE=none S2S_PARALLEL=1` and benchmark with
`--model google/gemma-4-12B --parallel 1`. This is a small throughput check, not a
quality evaluation.

## Build and curate NLU data

To request an **exact number of accepted, deduplicated examples**, use `nlu-build`:

```bash
uv run python -m source2synth nlu-build --target 1000 --batch-size 25 --max-batches 10 --balance-tasks --run-name nlu-1000
```

`--target` is the final row count; `--batch-size` is the number of candidate rows **per task** in each batch. The command generates and curates repeatedly, then writes `artifacts/source2synth/runs/nlu-1000/train.jsonl` and `report.json`. If it reaches the batch limit first, the report has `complete: false` and the accepted rows are preserved. Continue with the same target, batch size, task selection, split, offset and model settings:

```bash
uv run python -m source2synth nlu-build --target 1000 --batch-size 25 --max-batches 10 --balance-tasks --run-name nlu-1000 --resume
```

`--max-batches` limits new batches *per invocation* and may change on resume. Child runs retain the raw DataDesigner responses and rejection reasons. A resumed build reuses completed child runs after an interruption. The final export is capped at exactly `--target`; extra accepted rows from its last batch go to `surplus.jsonl`. With `--balance-tasks`, each task gets an accepted-row quota differing by at most one;
only deficient tasks receive further generation batches. Excess rows are retained in
`surplus.jsonl`. This balances task families, not the labels within each family:
label counts and solver disagreements remain visible in the report. Without this flag,
the build retains the original total-count behavior. `--tasks`, `--split`, `--offset`, `--model` and `--endpoint` work as with the individual commands. For `--split eval`, the accepted rows are saved without a training export. The expanded Korp pool has 1,267 passages, including 568 in slice1. The defaults fetch
100 hits per query and keep up to 100 articles per topic/split. Reports and saved
source metadata retain article provenance; task scheduling also prefers source excerpts
with relevant entity or answer-type clues.

After changing source data, configuration, implementation, or enabling task balancing on an existing
build, use an explicit input refresh. Normal `--resume` still rejects changed inputs.
For the handed-over `nlu-1000` build (offset remains **0** in the command; its saved
state calculates the next offset automatically):

```bash
uv run python -m source2synth nlu-build --target 1000 --batch-size 25 --max-batches 20 \
  --run-name nlu-1000 --resume --refresh-inputs --balance-tasks
```

Input refresh records old/new source, configuration and implementation fingerprints in `state.json` and
`report.json`, retains completed child runs, and uses a new child-run namespace so an
interrupted batch from older inputs cannot be reused accidentally. Existing accepted
rows remain historical curated examples; refreshing inputs does not re-review them.
Reaching the batch limit still means `complete: false`, especially if a weak model
cannot fill the NER quota. Do not infer dataset readiness from total row count alone.

The lower-level commands remain available when you want to inspect each generation and curation step:

```bash
uv run python -m source2synth prepare-nlu
uv run python -m source2synth validate-nlu
uv run python -m source2synth nlu-generate --split slice1 --tasks all --records 12 --run-name nlu-candidates-v1
uv run python -m source2synth nlu-curate --source-run artifacts/source2synth/runs/nlu-candidates-v1 --run-name nlu-curated-v1
```

`--records` is a candidate count **per task**, so the example asks for 84 candidates. Start with `--records 1` or `2` for a smoke test. Each run preserves DataDesigner outputs, raw responses, accepted/rejected rows, reasons, model configuration and a report. Curation exports `train.jsonl` with only the requested short label/answer. It may accept zero rows. For Source2Synth's intermediate training slice, generate with `--split slice0`; `uncurated_train.jsonl` is explicitly labeled. Generate slice1 separately, then curate for final training. `--split eval` remains held out and never exports a training file. These scripts prepare data; they do not fine-tune the model.

Use `--offset 12 --records 12` for another batch of seeds. Pass multiple generation run paths after `nlu-curate --source-run` to combine batches from the same split; duplicate inputs and seed IDs are rejected. Every curated row retains its generation run provenance.

If separate curation passes accept complementary rows, merge their accepted outputs without another model call:

```bash
uv run python -m source2synth nlu-merge --source-run artifacts/source2synth/runs/nlu-curated-v1 artifacts/source2synth/runs/nlu-curated-v2 --run-name nlu-merged
```

The merge requires one split and removes duplicate seed IDs or task inputs. Its report records each curation run and model ID.

```bash
uv run python -m source2synth prepare-nlu --offline  # reuse cached Korp responses
```

`prepare-nlu` samples six topic strata from the same Korp corpus, then keeps at most one passage per article and topic. Article IDs determine the 45/45/10 source split before construction. This is a keyword-conditioned sample, not a representative one, and a matched keyword alone is not a topic label. Candidate schedules cycle labels, topics, styles and difficulty. Reports give accepted counts and label distribution after filtering.

| [EuroEval task](https://euroeval.com/tasks/) | Synthetic example and output |
| --- | --- |
| [Classification](https://euroeval.com/tasks/classification) | Source-grounded topic rewrite → one of six declared topic labels |
| [Sentiment](https://euroeval.com/tasks/sentiment-classification) | Synthetic anonymous opinion about source facts → `positive`, `neutral`, `negative` |
| [NER](https://euroeval.com/tasks/named-entity-recognition) | Contiguous source excerpt → per-token BIO labels for person, organisation, place, other |
| [Linguistic acceptability](https://euroeval.com/tasks/linguistic-acceptability) | Source-grounded sentence and controlled grammar edit → `correct` or `incorrect` |
| [Reading comprehension](https://euroeval.com/tasks/reading-comprehension) | Unchanged source passage and exact short answer span → answer text |
| [NLI](https://euroeval.com/tasks/natural-language-inference) | Source-supported premise and controlled hypothesis → `entailment`, `neutral`, `contradiction` |
| [Grammatical error detection](https://euroeval.com/tasks/grammatical-error-detection) | Verb moved within the same clause → per-token `o`/`b-err` tags |

Labels and token formats follow [EuroEval's task definitions](https://github.com/EuroEval/EuroEval/blob/main/src/euroeval/tasks.py). These are **EuroEval-style synthetic examples**, not an official EuroEval dataset or benchmark score. Generic classification uses dataset-specific labels, so the six topics here are a declared local set. GED targets verb placement because that is EuroEval's GED phenomenon. No EuroEval test samples are used as source data.

Each accepted row includes source URL, corpus ID, DOI, displayed license, query URL, retrieval timestamp and response hash. NER annotates a fixed excerpt from Korp; reading-comprehension evidence comes from the exact source answer span. Deterministic checks cover source quotes, answer spans, BIO lengths and labels, controlled edits, duplicates and article splits. DataDesigner independently solves and reviews candidates. NER generation and independent solving return entity spans; Python converts both to
BIO tags. Agreement still requires every entity and type to match, and review must
pass. Valid BIO positions alone never exempt NER from independent validation. GED
solving returns error-token indices instead of a long tag array. For controlled
acceptability edits and verb-placement tags, solver disagreement remains an explicit
audit field and the semantic review must pass; an edit alone does not prove linguistic
incorrectness. The unsafe insertion of negation before an arbitrary finite verb has
been replaced with verb duplication.

For reading comprehension, answers are mapped to whole-token source spans to restore
Korp punctuation spacing and case. Questions may use other Swedish constructions;
review must verify their requested answer type. Cause answers may use up to 16 words;
other answers remain capped at 8. Answer matching permits narrow place/time/quantity
prefix variants, never arbitrary substring overlap, changed numbers, or paraphrases.
Only solver mismatches with a passing review are retried, up to
`nlu.curation_attempts` (default 3). Retry prompts never include the proposed answer
or earlier predictions. All predictions are saved in `prediction_attempts`.

Re-curate selected tasks from existing generation runs without paying for generation
again, for example `nlu-curate --tasks named-entity-recognition reading-comprehension
--source-run ... --run-name rechecked`. All old candidates are rechecked against
current deterministic rules before any model call. Model review is fallible, and accepted rows still need human spot checks before serious training or evaluation.

| File | Purpose |
| --- | --- |
| [configs/source2synth.yaml](configs/source2synth.yaml) | Source strata and model settings |
| [source2synth/nlu_sources.py](source2synth/nlu_sources.py) | Korp sampling and candidate schedule |
| [source2synth/nlu_tasks.py](source2synth/nlu_tasks.py) | Task schemas, validation and short-answer exports |
| [source2synth/nlu_pipeline.py](source2synth/nlu_pipeline.py) | DataDesigner generation, solving and review |
| [scripts/setup-local.sh](scripts/setup-local.sh) | llama.cpp build and official model conversion |
| [scripts/serve-local.sh](scripts/serve-local.sh) | Loopback llama.cpp launcher |

The Korp corpus is listed by Språkbanken under CC-BY-4.0 with [DOI 10.23695/7dz6-ek80](https://doi.org/10.23695/7dz6-ek80). Keep `accepted.jsonl` and the source manifest with shared training data so attribution is retained. The repository license is separate from corpus and model terms.

## Fine-tune with TRL and benchmark with EuroEval

The [fine-tuning guide](docs/finetuning.md) provides a one-epoch QLoRA pilot on
**Gemma 4 12B base**, using the current curated export, followed by a matched
before/after EuroEval validation comparison on Swedish sentiment and linguistic
acceptability. Start with `python -m source2synth.finetune --dry-run` to validate
the data. The guide includes separate training/evaluation environments, GPU
commands, output locations and the limits of the pilot scores. The same served
models can also be scored on the [Swedish idioms minimal-pair benchmark](docs/finetuning.md#swedish-idioms-minimal-pairs)
with `.venv-eval/bin/python -m source2synth.idioms`.

## Moving to the paper's training stages

The current NLU workflow is an adaptation of Source2Synth, not the paper's MHQA/TQA
experiment. For the next stage, generate **slice0** and train an intermediate model
on its `uncurated_train.jsonl`; then serve that checkpoint and independently curate
**slice1** using `--model`/`--endpoint`, before final training and held-out evaluation.
The source split is assigned by article before generation, and `eval` never exports
training data. See [Source2Synth §3.2–3.3](https://arxiv.org/pdf/2409.08239) and the
[existing MHQA guide](docs/source2synth-mhqa.md). NLU training, imputation, and EuroEval
execution are separate from these data-preparation commands; the pilot above
implements direct curated-data training and benchmarking. Imputation and the
paper's complete two-stage training procedure remain unimplemented. A slice1 run
curated by the original base model is a baseline, not the paper's intermediate-model
curation stage.
