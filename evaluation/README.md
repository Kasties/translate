# Evaluation

Evaluates checkpoints on two suites and collects the results in one table:

- **Minimal-pair suite**: does the model prefer the better of two Swedish sentences? Four datasets: Idioms, DaLAJ, ScaLA and Translationese. Runs on a CUDA GPU or the CPU.
- **EuroEval suite**: Swedish downstream tasks from [EuroEval](https://euroeval.com/datasets/swedish), run through the pinned `euroeval==18.1.0`. Six datasets: `scala-sv`, `swedn`, `multi-ifeval-sv`, `skolprov`, `multi-wiki-qa-sv` and `swerec`. Needs a CUDA GPU.

This folder is a uv project of its own. Run every command from inside it.

## Install

You need [uv](https://docs.astral.sh/uv/) and, for the EuroEval suite, a Linux machine with an NVIDIA GPU. uv installs Python 3.12 and every dependency at the versions in `uv.lock`.

```bash
cd evaluation
uv sync
uv run hf auth login        # a Hugging Face token; EuroEval's datasets refuse anonymous downloads
```

## Reference run

One small instruction-tuned model, SmolLM2-135M-Instruct (`reference_run.yaml`), through both suites: every minimal-pair dataset, and every EuroEval dataset except `multi-ifeval-sv`, the slowest. Use it to check that the evaluation works on a new machine.

```bash
uv run python -m evaluation run --model-list reference_run.yaml --suite minimal-pair euroeval \
    --euroeval-dataset scala-sv --euroeval-dataset swedn --euroeval-dataset skolprov \
    --euroeval-dataset multi-wiki-qa-sv --euroeval-dataset swerec --no-upload
uv run python -m evaluation table --include-quick-runs
```

It is a quick run: its numbers check the pipeline and are never reported. On a Kaggle T4 it took 13 minutes: about 4 for the minimal-pair suite and 9 for the EuroEval suite, most of them spent installing EuroEval and vLLM, which happens once per machine. Its output should list the minimal-pair results and EuroEval's `Completed 5 benchmarks`, and the results table should have one row. On the T4 it gave, for example:

| Column | Value |
|---|---|
| Idioms (all) preference accuracy (mean) | 45.82 |
| ScaLA (word deleted) preference accuracy (mean) | 64.93 |
| Translationese (human vs OPUS) preference accuracy (mean) | 49.67 |
| `scala-sv` mcc | 0.00 |
| `swedn` chr_f3pp | 26.93 |

Another GPU or the CPU moves the minimal-pair numbers by a few pairs. A model this small is at chance on most EuroEval tasks, so an MCC near 0 is expected.

Without a GPU, run the minimal-pair suite only: leave out `--suite minimal-pair euroeval` and the `--euroeval-dataset` flags. On a laptop CPU it takes about 30 minutes, 25 of them for DaLAJ; add `--minimal-pair-dataset Idioms` for a one-minute check.

## Evaluating checkpoints

```bash
uv run python -m evaluation run --model-list models.yaml --suite minimal-pair euroeval   # every checkpoint in a model list
uv run python -m evaluation run --model <repo id> --revision <commit hash>               # one checkpoint, pinned
uv run python -m evaluation run --model path/to/checkpoint                               # a local checkpoint folder
```

**Checkpoints.** A checkpoint is a merged Hugging Face model: full weights, the tokenizer, and the chat template it was trained with. EuroEval needs the chat template. An adapter must be merged into its base model first. On a GPU without native bf16, such as a T4, store the checkpoint in bf16, not float32: EuroEval then switches it to float16, while a float32 checkpoint makes vLLM fail.

**The model list** is a YAML file with one entry per checkpoint. Each checkpoint is evaluated in turn, as a run of its own, and its values tag its result records. Only `reference` is required:

```yaml
- reference: <org>/<checkpoint repo>   # Hugging Face repo id or local path
  revision: "<commit hash>"            # quoted, so YAML reads it as text; Hugging Face repos only
  model_family: <base model repo id>
  condition: seed-grounded             # seed-grounded, translated, MAGPIE or control; empty for a base model
  training_method: full                # full or lora; empty for a base model
  random_seed: 42
  suites: [minimal-pair]               # optional: run only these of the suites chosen with --suite
```

A base model gets `suites: [minimal-pair]`: EuroEval prompts base models differently and skips some tasks for them. With `--model` instead of a model list, the values come from `checkpoint_metadata.json` next to the checkpoint, if there is one.

**Suites and datasets.** `--suite` picks `minimal-pair` (the default), `euroeval`, or both. `--minimal-pair-dataset` and `--euroeval-dataset` pick datasets, repeated for more; by default a suite runs all of its datasets.

| Suite | Dataset | Pairs or examples | Reported per |
|---|---|---|---|
| minimal-pair | Idioms | 478 pairs | all, challenge (100 pairs) |
| minimal-pair | DaLAJ | 20,948 pairs | error type: lexical, morphology, orthography, syntax, punctuation |
| minimal-pair | ScaLA | 1,663 pairs | neighbouring words swapped, word deleted |
| minimal-pair | Translationese | 1,200 pairs | human vs OPUS, human vs GPT |
| EuroEval | `scala-sv` | linguistic acceptability | MCC, macro-F1 |
| EuroEval | `swedn` | summarisation | chrF3++, chrF4++ |
| EuroEval | `multi-ifeval-sv` | instruction following | instruction accuracy |
| EuroEval | `skolprov` | knowledge (school exam questions) | MCC, accuracy |
| EuroEval | `multi-wiki-qa-sv` | reading comprehension | F1, exact match |
| EuroEval | `swerec` | sentiment | MCC, macro-F1 |

**Quick runs and full runs.** Every run is a quick run unless you add `--full-run`. Quick runs check the pipeline and are left out of the results table by default; only full runs are reported. For the EuroEval suite, a quick run uses the validation split once, and a full run the test split ten times, which is about 80 times the work. For the minimal-pair suite both score every pair. Make all reported runs of one experiment on the same GPU type: bf16 scores shift by a few pairs between GPU types. Each result record names its GPU.

## Results

Each run writes a folder of its own, `results/runs/<run>/`, named after the UTC time it started and the machine:

- `result_records.jsonl`: one result record per dataset, subset and metric, with the checkpoint, its model list values, the value as a fraction, the 95% interval (`lower`, `upper`), the number of items, quick run or full run, the GPU and a timestamp.
- `pair_scores.jsonl`: one row per minimal pair, with both sentences, the subsets it counts toward, and each sentence's summed log-probability and token count.
- `euroeval_benchmark_results.jsonl`: EuroEval's own output, for a run of the EuroEval suite.

```bash
uv run python -m evaluation table                          # full runs only; add --include-quick-runs
uv run python -m streamlit run evaluation/dashboard.py     # charts, and the minimal pairs two checkpoints disagree on
```

The table has one row per checkpoint and one column per dataset, subset and metric. It is printed and written to `results/results_table.md` (percent with the 95% interval, ready for a report) and `results/results_table.csv` (fractions). A checkpoint run more than once shows its latest run.

## Sharing results

Without `--no-upload`, a run uploads its folder to a private Hugging Face dataset repo when it ends, so runs from Colab, Kaggle and lab machines end up in one place. Name the repo in the `RESULTS_REPO` environment variable; your token needs write access to it. The repo is created on the first upload.

```bash
export RESULTS_REPO=<org or user>/<results repo>
uv run python -m evaluation run --model-list reference_run.yaml     # uploads when it ends
uv run python -m evaluation download                                # fetches every run, on any machine
```

Without `RESULTS_REPO`, a run that would upload stops before it starts and says so; add `--no-upload` to keep its files on this machine only. A failed upload leaves the run's files in `results/runs/<run>/`. Upload them again with:

```bash
uv run hf upload $RESULTS_REPO results/runs/<run> runs/<run> --repo-type dataset
```

## Scoring

**Minimal pairs** follow Kunz (2026), "Preferences for idiomatic language are acquired slowly — and forgotten quickly: A case study on Swedish":

- Each sentence is scored alone as raw text, with no prompt and no chat template, for base and instruction-tuned models alike. The tokenizer adds its default special tokens.
- Preference accuracy (mean) counts a pair as right when the better sentence has the strictly higher mean log-probability per token. Ties count as wrong. It is the number to report.
- Preference accuracy (sum) compares summed log-probabilities instead. It favours the shorter sentence: on ScaLA's word-deleted pairs, where the worse sentence is shorter, it falls to about 7%.
- Intervals are 95% Wilson intervals.
- DaLAJ keeps only rows with a single error label, as the paper does, and has no pooled number. ScaLA uses all 1,663 pairs in the paper's CSVs, where the paper reports 552 and 601. Translationese strips trailing punctuation before scoring, as its paper's code does.

**EuroEval** results are converted into the same result records: values as fractions, the metric name without EuroEval's `test_` prefix, no subset, and no item count. A quick run gives no interval, because EuroEval computes it over its ten iterations.
