# Swedish Source2Synth with Korp and DataDesigner

This project turns **Svenska Wikipedia (`WIKIPEDIA-SV`) in Korp** into source-grounded Swedish two-hop question answering data, using the installed **NVIDIA NeMo DataDesigner 0.9.2**. Generation, imputation, independent solving, and quality review use DataDesigner LLM columns followed by schema-validated custom columns.

The selected corpus is [documented by Språkbanken](https://spraakbanken.gu.se/resurser/wikipedia-sv), [searchable in Korp](https://spraakbanken.gu.se/korp/#?corpus=wikipedia-sv), and identified by [DOI 10.23695/7dz6-ek80](https://doi.org/10.23695/7dz6-ek80). Wikipedia is among the corpora whose [sentence order is preserved](https://spraakbanken.gu.se/faq/hur-kan-jag-fa-tillgang-till-icke-meningsomkastade-versioner-av-sprakbankens-korpusar), which makes it suitable for passage-based QA.

## Quick start

```bash
uv sync --locked
uv run python -m source2synth prepare
uv run python -m source2synth validate
uv run python -m unittest discover -s tests -v
```

`prepare` makes at most 13 public Korp requests before HTTP retries: metadata plus 40 hits for each of 12 editable entities. The default sample is bounded; it does not download the multi-gigabyte corpus. Responses are cached. To rebuild exactly from the saved responses:

```bash
uv run python -m source2synth prepare --offline
```

The initial extraction produced **327 passages, 40 slice0 pairs, 40 slice1 pairs, and 12 evaluation pairs**. Pair counts can change after a corpus update if the raw cache is replaced. The exact corpus metadata, acquisition time, query URLs, response hashes, revision permalinks where available, and seed hashes are retained locally.

The live NVIDIA smoke test produced **one curated training example** in `artifacts/source2synth/runs/nvidia-curated-final/train.jsonl`, with full source provenance and model outputs in the adjacent `accepted.jsonl`. Its lineage is `nvidia-verified-generation` → `nvidia-verified-curation` → `nvidia-curated-final`. Five source pairs yielded two generation candidates; one survived final curation. Model formatting failures and rejected candidates are recorded separately. This tiny baseline pilot verifies execution, not dataset quality at training scale. No model was fine-tuned.

## NVIDIA generation

Set your key in your shell; it is neither embedded in configuration nor loaded from `.env` automatically. Bash can read it without displaying it or adding it to command history:

```bash
read -rs -p 'NVIDIA API key: ' NVIDIA_API_KEY
export NVIDIA_API_KEY
uv run python -m source2synth generate --split slice0 --records 5 --run-name slice0-v1
uv run python -m source2synth generate --split slice1 --records 5 --run-name slice1-v1
```

The default is `nvidia/nemotron-3-super-120b-a12b`, with two concurrent requests, 4,096 output tokens, and thinking disabled for structured output. The installed DataDesigner default, `nvidia/nemotron-3-nano-30b-a3b`, returned HTTP 410 on 2026-09-18: NVIDIA retired that endpoint on September 1. [Super model configuration](https://docs.nvidia.com/nim/large-language-models/2.0.4/turbo/get-started-nemotron-3-super-120b-a12b.html).

Generation makes approximately three model calls per row, plus provider checks and possible retries. Rows may be rejected or dropped; inspect `report.json`, `rejected.jsonl`, and the DataDesigner artifacts. `--records` is a **candidate budget**, not a promised accepted count. Runs use a new directory and refuse to overwrite one. Use `--offset 5 --records 5` to process the next five seeds. Each run exports its own batch; latest-run pointers select only that run, not a concatenation of previous batches.

Run a baseline curation smoke test, using a count no larger than the accepted slice1 count:

```bash
uv run python -m source2synth curate --mode baseline --records 2 --run-name baseline-v1
```

This uses the pretrained instruction model as curator. It is explicitly labeled **baseline**, because it has not undergone the paper's intermediate fine-tuning. Curation makes six model calls per candidate: imputation, recomposition, three independent answers, and quality review, plus checks/retries. A valid run can accept zero rows.

To reapply deterministic filtering to saved curation outputs without additional model calls:

```bash
uv run python -m source2synth refilter \
  --source-run artifacts/source2synth/runs/baseline-v1 --run-name refilter-v1
```

## Method and training boundary

The implementation follows the MHQA branch of [Source2Synth, arXiv:2409.08239v2](https://arxiv.org/abs/2409.08239v2), sections 3.1–3.3:

| Stage | Implementation |
| --- | --- |
| Source and seed | Two different Wikipedia articles with a shared named entity E |
| Construction | Q1 from D1 answers E; Q2 asks about E in D2; combine into Q |
| Slice training | Export uncurated slice0 as `train.jsonl` for intermediate supervised fine-tuning |
| Imputation | Hide Q1; reconstruct it from Q, Q2, E and D1; recompose Q |
| Answerability | Solve the recomposed Q from D1 and D2 in three independent calls; retain only if at least one answer matches the original |
| Final training | Export accepted curated slice1 as conversational supervised training data |

**Fine-tuning is an external stage, not performed by DataDesigner or this setup.** The repository prepares training data and accepts an already trained and served curator. It does not install a GPU trainer, download model weights, or claim to reproduce the paper's experimental results. To execute the complete training experiment:

1. Generate a sufficiently large slice0 and slice1; inspect the accepted and rejected examples.
2. Train your chosen base model on `artifacts/source2synth/runs/<slice0-run>/train.jsonl` using a trainer that supports the `messages` format. Each response includes two short subanswers, supporting quotes, and the final answer.
3. Serve the resulting intermediate checkpoint through an OpenAI-compatible endpoint.
4. Copy `configs/training-manifest.example.json` to a separate file, fill in the actual model/checkpoint/command, and set `slice0_train_sha256` to the **file SHA-256** reported by `sha256sum`.
5. Run paper-mode curation. The manifest must match both the requested served model ID and the latest slice0 training file. The provenance is a user-supplied record, not an inspection of remote model weights.

```bash
sha256sum artifacts/source2synth/runs/slice0-v1/train.jsonl
export S2S_ENDPOINT=http://localhost:8000/v1
export S2S_API_KEY=local
uv run python -m source2synth curate --mode paper \
  --curator-model source2synth-sv-intermediate \
  --training-manifest /path/to/actual-training-manifest.json \
  --records 2 --run-name paper-v1
```

6. Fine-tune a fresh copy of the selected base model on the resulting curated `train.jsonl`. Compare it with the base model and the slice0-trained intermediate model on the same held-out evaluation data. Preserve training commands, versions, hyperparameters, and checkpoints for both training runs.

For a basic held-out synthetic QA evaluation:

```bash
# Return to NVIDIA for reference generation.
unset S2S_ENDPOINT S2S_API_KEY
uv run python -m source2synth generate --split eval --records 5 --run-name eval-v1
uv run python -m source2synth evaluate \
  --model nvidia/nemotron-3-super-120b-a12b \
  --records 2 --run-name eval-base-v1
# Set S2S_ENDPOINT/S2S_API_KEY and repeat evaluate with your served model ID.
```

Evaluation prompts contain only Q and the documents. Missing predictions count as incorrect. The score is normalized exact match against synthetic reference answers; manually review those references before drawing model-quality conclusions. This is not HotpotQA or an independent human-annotated benchmark.

## Deliberate adaptations and validation

- Swedish passages and prompts replace the paper's English Wikipedia material. Entity-conditioned Korp searches replace hyperlink-based article discovery. The bootstrap vocabulary covers people, institutions, and events; this is not a representative Wikipedia sample.
- Articles are assigned to 45% slice0 / 45% slice1 / 10% evaluation using a stable hash **before pairing**. The two training seed slices are then capped to equal sizes. Acceptance rates may differ. No source article appears across splits, though shared facts and entities may still occur.
- Evidence quotes must occur as contiguous tokens in the cited passage; punctuation, case and token order are preserved while typographic spacing is ignored. The bridge must occur in both documents. Same-article pairs, exposed bridge/answer strings, abstentions, invalid schemas, and duplicate questions within a run are rejected.
- Curation uses normalized exact equality (case/whitespace/punctuation normalization, retaining Swedish accents, and ignoring an optional `den` before a Swedish date) rather than substring matching. No reference answer or original Q1 is included in the independent solver prompts. The model can naturally see the answer in source text.
- A separate quality review checks grounding, Swedish, consistency of the subquestions, and whether both documents are necessary. This semantic review remains fallible and should be spot-checked. Its access to the reference answer does not affect the independent answerability calls.
- Model/API errors are recorded in DataDesigner artifacts and incomplete-record counts. Invalid rows are never silently converted into valid QA. Configurations can be compiled without credentials; the behavioral suite mocks the HTTP boundary and exercises the real DataDesigner engine.
- The custom parser accepts plain JSON or a single Markdown JSON block. This handles valid NVIDIA responses rejected by DataDesigner 0.9.2's fence-only structured parser. It never repairs malformed JSON or fills missing required fields. Raw outputs remain alongside the validated objects for auditing. Exported YAML files are configuration snapshots; use the Python builders/CLI to recreate the custom parser functions.

## Files and outputs

| Path | Purpose |
| --- | --- |
| `configs/source2synth.yaml` | Corpus sampling, model, and filtering settings |
| `source2synth/korp.py` | Cached API ingestion, provenance, article partitioning, pair building |
| `source2synth/pipeline.py` | DataDesigner schemas, Swedish prompts, column dependencies |
| `source2synth/curation.py` | Deterministic checks, rejection reasons, SFT export |
| `source2synth/__main__.py` | Preparation, validation, generation, curation, evaluation CLI |
| `data/raw/*.json` | Cached request/response envelopes |
| `data/prepared/*.parquet` | Source-pair seed datasets |
| `data/prepared/passages.jsonl` | Extracted real passages and provenance |
| `data/prepared/manifest.json` | Sampling configuration, source version and hashes |
| `artifacts/source2synth/configs/` | Compiled DataDesigner YAML configurations; curation/evaluation schema fixtures are not datasets |
| `artifacts/source2synth/runs/<run>/` | Accepted/rejected JSONL, training JSONL, report, configuration, DataDesigner outputs |

Use `--config`, `--data-dir`, and `--output-dir` **before** the subcommand for separate experiments. `latest_slice0.json`, `latest_slice1.json`, and `latest_eval.json` select the generation runs used by subsequent stages. Use a separate output directory when comparing independent experiments.

Generated data and credentials are ignored by Git. Språkbanken lists this corpus under **CC-BY-4.0**; that label, its DOI and source links are propagated. The article URLs and revision permalinks preserve attribution to the underlying Wikipedia material. Keep `accepted.jsonl` and the corpus manifest alongside `train.jsonl` when distributing derivatives: the training-only file intentionally has no extra metadata fields. No blanket license is asserted for newly generated content.

API references: [Korp v8](https://ws.spraakbanken.gu.se/docs/korp), [DataDesigner seed datasets](https://docs.nvidia.com/nemo/datadesigner/concepts/seed-datasets), [model providers](https://docs.nvidia.com/nemo/datadesigner/concepts/models/model-providers).
