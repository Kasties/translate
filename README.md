# Swedish Source2Synth: Korp → EuroEval NLU

This project uses [NeMo DataDesigner](https://docs.nvidia.com/nemo/datadesigner/) to construct and curate Swedish synthetic examples grounded in [Svenska Wikipedia in Korp](https://spraakbanken.gu.se/korp/#?corpus=wikipedia-sv). The earlier two-document QA pipeline and its method are preserved in [the MHQA guide](docs/source2synth-mhqa.md). This workflow adapts the real-source construction and independent solving/curation pattern of [Source2Synth](https://arxiv.org/abs/2409.08239v2); it does not claim to reproduce the paper's fine-tuning results.

## Local model

The default is the exact [official `google/gemma-4-12B` checkpoint](https://huggingface.co/google/gemma-4-12B), pinned to revision `023679ed352de9bb66cc873c9009ce3482585c08`. [scripts/setup-local.sh](scripts/setup-local.sh) verifies the downloaded safetensors SHA-256, builds [llama.cpp](https://github.com/ggml-org/llama.cpp) with CUDA and converts these weights to Q8_0 GGUF. The checkpoint has no chat template, so the server uses an explicit role template in [scripts/gemma4-base.jinja](scripts/gemma4-base.jinja). The checkpoint is not instruction tuned; inspect acceptance rates and manually review labels.

The setup machine has an RTX 4090 with 24 GB VRAM. Allow at least 24 GB for the original download and additional disk space for GGUF. Requirements: `git`, `cmake`, CUDA toolkit, `uv` and network access.

```bash
uv sync --locked
bash scripts/setup-local.sh
bash scripts/serve-local.sh           # leave this terminal running
```

In a second terminal:

```bash
uv run python -m source2synth model-check
```

The server listens on `127.0.0.1:8080`; DataDesigner connects to its OpenAI-compatible `/v1` API. To switch GGUF models later, use matching server alias and client ID:

```bash
S2S_GGUF=/absolute/path/other.gguf S2S_MODEL=my-local-model S2S_CHAT_TEMPLATE=auto bash scripts/serve-local.sh
S2S_MODEL=my-local-model uv run python -m source2synth model-check
S2S_MODEL=my-local-model uv run python -m source2synth nlu-generate --records 2 --run-name alternative-model
```

`S2S_MODELS_DIR`, `LLAMA_SERVER`, `S2S_PORT`, `S2S_CONTEXT`, `S2S_GGUF` and `S2S_MODEL` configure the launcher. `S2S_CHAT_TEMPLATE=auto` uses a replacement GGUF's embedded chat template; otherwise set a Jinja file. `model.id` and `model.endpoint` in [configs/source2synth.yaml](configs/source2synth.yaml) configure DataDesigner. `S2S_MODEL` and `S2S_ENDPOINT` override YAML, or pass `--endpoint` before a CLI subcommand and `--model` to `nlu-generate`/`nlu-curate`. `model-check` catches a mismatched model ID before generation. The local loopback server needs no API key.

## Build and curate NLU data

To request an **exact number of accepted, deduplicated examples**, use `nlu-build`:

```bash
uv run python -m source2synth nlu-build --target 1000 --batch-size 25 --max-batches 10 --run-name nlu-1000
```

`--target` is the final row count; `--batch-size` is the number of candidate rows **per task** in each batch. The command generates and curates repeatedly, then writes `artifacts/source2synth/runs/nlu-1000/train.jsonl` and `report.json`. If it reaches the batch limit first, the report has `complete: false` and the accepted rows are preserved. Continue with the same target, batch size, task selection, split, offset and model settings:

```bash
uv run python -m source2synth nlu-build --target 1000 --batch-size 25 --max-batches 10 --run-name nlu-1000 --resume
```

`--max-batches` limits new batches *per invocation* and may change on resume. Child runs retain the raw DataDesigner responses and rejection reasons. A resumed build reuses completed child runs after an interruption. The final export is capped at exactly `--target`; extra accepted rows from its last batch go to `surplus.jsonl`. The report shows task and label counts, which may still be uneven. `--tasks`, `--split`, `--offset`, `--model` and `--endpoint` work as with the individual commands. For `--split eval`, the accepted rows are saved without a training export. The initial Korp pool has 319 passages; expand the sampling settings in `configs/source2synth.yaml` and rerun `prepare-nlu` before a large build if more distinct source articles are needed.

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

Each accepted row includes source URL, corpus ID, DOI, displayed license, query URL, retrieval timestamp and response hash. NER annotates a fixed excerpt from Korp; reading-comprehension evidence comes from the exact source answer span. Deterministic checks cover source quotes, answer spans, BIO lengths and labels, controlled edits, duplicates and article splits. DataDesigner independently solves and reviews candidates. For controlled acceptability edits and verb-placement tags, the saved deterministic transformation is the label authority: a solver disagreement is retained and counted in the curation report, while the separate source/task review must still pass. Other tasks require solver agreement and review. Model review is fallible, and accepted rows still need human spot checks before serious training or evaluation.

| File | Purpose |
| --- | --- |
| [configs/source2synth.yaml](configs/source2synth.yaml) | Source strata and model settings |
| [source2synth/nlu_sources.py](source2synth/nlu_sources.py) | Korp sampling and candidate schedule |
| [source2synth/nlu_tasks.py](source2synth/nlu_tasks.py) | Task schemas, validation and short-answer exports |
| [source2synth/nlu_pipeline.py](source2synth/nlu_pipeline.py) | DataDesigner generation, solving and review |
| [scripts/setup-local.sh](scripts/setup-local.sh) | llama.cpp build and official model conversion |
| [scripts/serve-local.sh](scripts/serve-local.sh) | Loopback llama.cpp launcher |

The Korp corpus is listed by Språkbanken under CC-BY-4.0 with [DOI 10.23695/7dz6-ek80](https://doi.org/10.23695/7dz6-ek80). Keep `accepted.jsonl` and the source manifest with shared training data so attribution is retained. The repository license is separate from corpus and model terms.
