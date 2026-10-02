# Evaluation

This folder scores checkpoints on two suites and collects the results in one table.

- The minimal-pair suite checks whether a model gives the better of two Swedish sentences the higher probability. It has four datasets: Idioms, DaLAJ, ScaLA and Translationese. It runs on a CUDA GPU or the CPU.
- The EuroEval suite runs Swedish tasks from [EuroEval](https://euroeval.com/datasets/swedish) 18.1.0. By default it runs five: `scala-sv`, `swedn`, `skolprov`, `multi-wiki-qa-sv` and `swerec`. It needs a Linux machine with an NVIDIA GPU.

The folder is a uv project of its own. Run every command from inside it.

## Install

Install [uv](https://docs.astral.sh/uv/). It installs Python 3.12 and the exact dependency versions in `uv.lock`.

```bash
cd evaluation
uv sync
uv run hf auth login
```

The login needs a Hugging Face token. EuroEval downloads its datasets with it.

## Reference run

The reference run checks that the evaluation works on a new machine. It runs one small instruction-tuned model, SmolLM2-135M-Instruct, through both suites with the default datasets.

```bash
uv run python -m evaluation run --config reference_run.yaml
uv run python -m evaluation table --include-quick-runs
```

On a Kaggle T4 the run took 13 minutes. The minimal-pair suite took 4 of them. The EuroEval suite took 9, mostly to install EuroEval and vLLM, which happens once per machine.

The run should end without an error, and the results table should have one row. The T4 gave these values:

| Column | Value |
|---|---|
| Idioms (all) preference accuracy (mean) | 45.82 |
| ScaLA (word deleted) preference accuracy (mean) | 64.93 |
| Translationese (human vs OPUS) preference accuracy (mean) | 49.67 |
| `scala-sv` mcc | 0.00 |
| `swedn` chr_f3pp | 26.93 |

Another GPU, or the CPU, changes the minimal-pair values by a few pairs. A model this small answers most EuroEval tasks at chance level, so an MCC near 0 is normal.

On a machine without a GPU, add `suites: [minimal-pair]` to a copy of `reference_run.yaml`. A laptop CPU then needs about 30 minutes, 25 of them for DaLAJ.

## Evaluating your checkpoints

### Checkpoints

A checkpoint is a merged Hugging Face model with full weights, the tokenizer and the chat template it was trained with. EuroEval needs the chat template. Merge an adapter into its base model before you evaluate it.

On a GPU without native bf16, such as a T4, store the checkpoint in bf16. EuroEval switches a bf16 checkpoint to float16 on such a GPU. A float32 checkpoint makes vLLM fail there.

### Run config

A run config is a YAML file that describes a whole run. Only `models` is required. Every other field has a default, so a short file runs both suites with the default datasets. `reference_run.yaml` is an example.

```yaml
upload: true
models:
  - reference: Qwen/Qwen3.5-2B-Base           # a base model
    model_family: Qwen/Qwen3.5-2B-Base
    suites: [minimal-pair]
  - reference: <org>/<checkpoint repo>        # a fine-tuned checkpoint
    revision: "<commit hash>"
    model_family: Qwen/Qwen3.5-2B-Base
    condition: seed-grounded
    training_method: full
    random_seed: 42
```

```bash
uv run python -m evaluation run --config <your run config>.yaml
```

The fields of the run config apply to every checkpoint in it:

| Field | What it sets | Default |
|---|---|---|
| `models` | the checkpoints to evaluate, one entry each | required |
| `suites` | the suites to run | `[minimal-pair, euroeval]` |
| `minimal_pair_datasets` | the datasets of the minimal-pair suite | all four |
| `euroeval_datasets` | the datasets of the EuroEval suite | the five in the table below |
| `full_run` | `true` marks the results as reportable | `false` |
| `upload` | `true` uploads the run to the repo in `RESULTS_REPO` | `false` |

The fields of an entry under `models` describe one checkpoint:

| Field | What it is |
|---|---|
| `reference` | the Hugging Face repo id or a local folder. Required |
| `revision` | the commit hash, in quotes. Hugging Face repos only |
| `model_family`, `condition`, `training_method`, `random_seed` | labels for the results table: the base model, seed-grounded, translated, MAGPIE or control, full or lora, and the seed |
| `suites` | a limit for this checkpoint only. A base model gets `[minimal-pair]`, because EuroEval prompts base models differently and skips some tasks for them |

The command evaluates the checkpoints one after another, each as a run of its own.

### One checkpoint

For a quick check of one checkpoint, give it on the command line instead of in a run config:

```bash
uv run python -m evaluation run --model <repo id> --revision <commit hash> --no-upload
```

It runs the same defaults as a run config. `--suite`, `--minimal-pair-dataset` and `--euroeval-dataset` change them, and the dataset flags can be repeated. `--full-run` marks the results as reportable. The command reads the checkpoint's values from `checkpoint_metadata.json` next to the checkpoint, if that file exists.

### Suites and datasets

To change the datasets, write the whole list you want in the run config. Leave a dataset out to remove it, and add a name to add one. Any Swedish dataset on the [EuroEval list](https://euroeval.com/datasets/swedish) works, such as `swedish-facts` or `belebele-sv`.

```yaml
suites: [minimal-pair]                                   # no EuroEval suite
minimal_pair_datasets: [Idioms, ScaLA]                   # two of the four
euroeval_datasets: [scala-sv, swerec, multi-ifeval-sv]   # two of the default five, plus multi-ifeval-sv
```

The default datasets are these:

| Suite | Dataset | Size or task | Reported |
|---|---|---|---|
| minimal-pair | Idioms | 478 pairs | all, and the 100-pair challenge subset |
| minimal-pair | DaLAJ | 20,948 pairs | per error type: lexical, morphology, orthography, syntax, punctuation |
| minimal-pair | ScaLA | 1,663 pairs | per corruption: neighbouring words swapped, word deleted |
| minimal-pair | Translationese | 1,200 pairs | human vs OPUS, human vs GPT |
| EuroEval | `scala-sv` | linguistic acceptability | MCC, macro-F1 |
| EuroEval | `swedn` | summarisation | chrF3++, chrF4++ |
| EuroEval | `skolprov` | school exam questions | MCC, accuracy |
| EuroEval | `multi-wiki-qa-sv` | reading comprehension | F1, exact match |
| EuroEval | `swerec` | sentiment | MCC, macro-F1 |

`multi-ifeval-sv`, instruction following, is not a default because it is the slowest. Add it when you need it.

### Quick runs and full runs

Every run is a quick run unless `full_run` is true. The results table leaves quick runs out, and you never report their numbers. A EuroEval quick run scores the validation split once. A EuroEval full run scores the test split ten times, which is about 80 times the work. The minimal-pair suite scores every pair in both.

Use the same GPU type for every reported run of one experiment. Scores in bf16 change by a few pairs between GPU types. Each result record names its GPU.

## Results

Each run writes a folder of its own, `results/runs/<run>/`, named after the UTC time it started and the machine. The folder holds three files:

- `result_records.jsonl` has one record per dataset, subset and metric. A record holds the checkpoint and its run config values, the value as a fraction, the 95% interval in `lower` and `upper`, the number of items, the run type, the GPU and a timestamp.
- `pair_scores.jsonl` has one row per minimal pair. A row holds both sentences, the subsets the pair counts toward, and the summed log-probability and token count of each sentence.
- `euroeval_benchmark_results.jsonl` holds EuroEval's own output for a run of the EuroEval suite.

```bash
uv run python -m evaluation table                          # full runs only, add --include-quick-runs for all
uv run python -m streamlit run evaluation/dashboard.py     # charts per dataset, and the pairs two checkpoints disagree on
```

The table has one row per checkpoint and one column per dataset, subset and metric. The command prints it and writes `results/results_table.md`, with percentages and 95% intervals, and `results/results_table.csv`, with fractions. A checkpoint that ran more than once shows its latest run.

## Sharing results

With `upload: true`, a run uploads its folder to a private Hugging Face dataset repo when it ends. Runs from Colab, Kaggle and lab machines then end up in one place. The `RESULTS_REPO` environment variable names the repo, and your token needs write access to it. The first upload creates the repo.

```bash
export RESULTS_REPO=<org or user>/<results repo>
uv run python -m evaluation download          # fetches every run, on any machine with read access
```

If `RESULTS_REPO` is empty, a run that would upload stops before it starts. If an upload fails, the run's files stay in `results/runs/<run>/`. Upload them again with:

```bash
uv run hf upload $RESULTS_REPO results/runs/<run> runs/<run> --repo-type dataset
```

## How scoring works

The minimal-pair suite follows the protocol of Kunz (2026, Transactions of the Association for Computational Linguistics, volume 14).

- The scorer scores each sentence alone as raw text, with no prompt and no chat template, for base models and instruction-tuned models alike. The tokenizer adds its default special tokens.
- Preference accuracy (mean) counts a pair as right when the better sentence has the strictly higher mean log-probability per token. A tie counts as wrong. Report this number.
- Preference accuracy (sum) compares summed log-probabilities. It favours the shorter sentence. On ScaLA's word-deleted pairs, where the worse sentence is shorter, it drops to about 7%.
- Each value has a 95% Wilson interval.
- DaLAJ keeps only rows with a single error label, as the paper does, and has no pooled value. ScaLA uses all 1,663 pairs in the paper's CSV files, where the paper reports 552 and 601. Translationese removes trailing punctuation before scoring, as the paper's code does.

The command converts EuroEval's output into the same kind of result records. It stores values as fractions, drops EuroEval's `test_` prefix from metric names, and leaves the subset and the item count empty. A quick run has no interval, because EuroEval computes intervals over its ten iterations.
