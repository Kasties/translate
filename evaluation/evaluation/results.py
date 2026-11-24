"""Result records: one metric value for one checkpoint on one dataset, with its checkpoint metadata.

Each evaluation run writes its result records and pair scores to JSON Lines files in a folder of
its own, results/runs/<run>, and uploads that folder to the results repo. No run changes another
run's files, so runs from different machines never conflict. The results table pivots the result
records of every run into checkpoints x metrics.
"""

import csv
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import create_repo, hf_hub_download, snapshot_download, upload_folder
from huggingface_hub.errors import EntryNotFoundError

CHECKPOINT_METADATA = "checkpoint_metadata.json"
RESULT_RECORDS = "result_records.jsonl"
PAIR_SCORES = "pair_scores.jsonl"
# The keys of match_keys in append_results: a pair score belongs to the result records that share them.
MATCH_KEYS = ["checkpoint", "revision", "suite", "dataset", "timestamp"]
# The fields of a model list entry. They tag the checkpoint's result records.
# training_examples is the number of rows the checkpoint was trained on (the manifest's data.rows).
MODEL_LIST_FIELDS = ["reference", "revision", "model_family", "training_data", "training_method", "training_examples"]
SUITES = ["minimal-pair", "euroeval"]


def new_run_dir(results_dir):
    """Return the folder for a new run's files, named after the time it started and the machine."""
    return results_dir / "runs" / f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S-%f}-{socket.gethostname()}"


def results_repo():
    """Return the private Hugging Face dataset repo that collects every run's files, from the
    RESULTS_REPO environment variable."""
    repo = os.environ.get("RESULTS_REPO")
    if not repo:
        raise RuntimeError(
            "Set the RESULTS_REPO environment variable to a Hugging Face dataset repo your token can write to, "
            "such as <org>/<results repo>, or run with --no-upload. See README.md, \"Sharing results\"."
        )
    return repo


def upload_run(run_dir):
    """Upload one run's folder to the results repo, creating the repo on the first upload."""
    repo = results_repo()
    create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=run_dir,
        path_in_repo=f"runs/{run_dir.name}",
        commit_message=f"Add run {run_dir.name}",
    )


def download_runs(results_dir):
    """Download every run's folder from the results repo into results_dir/runs."""
    snapshot_download(results_repo(), repo_type="dataset", local_dir=results_dir, allow_patterns="runs/*")


def load_model_list(entries):
    """Return the checkpoints in a model list, one dict per entry with every field in MODEL_LIST_FIELDS.
    A field the entry leaves out is None."""
    for entry in entries:
        if "reference" not in entry or set(entry) - set(MODEL_LIST_FIELDS):
            raise ValueError(f"Model list entry {entry} needs a reference and only these fields: {MODEL_LIST_FIELDS}")
    return [{field: entry.get(field) for field in MODEL_LIST_FIELDS} for entry in entries]


def checkpoint_from_metadata(reference, revision=None):
    """Return one checkpoint as a model list entry, with the checkpoint metadata saved next to it. The
    model family is its base model. A checkpoint without that file, such as a base model, gets
    empty values."""
    if Path(reference).is_dir():
        path = Path(reference) / CHECKPOINT_METADATA
        metadata = json.loads(path.read_text()) if path.exists() else {}
    else:
        try:
            metadata = json.loads(Path(hf_hub_download(reference, CHECKPOINT_METADATA, revision=revision)).read_text())
        except EntryNotFoundError:
            metadata = {}
    return {
        "reference": reference,
        "revision": revision,
        "model_family": metadata.get("base_model"),
        "training_data": metadata.get("training_data"),
        "training_method": metadata.get("training_method"),
        "training_examples": metadata.get("training_examples"),
    }


def append_results(run_dir, checkpoint, suite, dataset, results, pair_scores, full_run):
    """Append one result record per result, tagged with the checkpoint's model list values, and the
    pair scores behind them, to the files in run_dir."""
    # A pair score belongs to the result records that share these keys.
    match_keys = {
        "checkpoint": checkpoint["reference"],
        "revision": checkpoint["revision"],
        "suite": suite,
        "dataset": dataset,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    records = [
        {
            **match_keys,
            "model_family": checkpoint["model_family"],
            "training_data": checkpoint["training_data"],
            "training_method": checkpoint["training_method"],
            "training_examples": checkpoint.get("training_examples"),
            **result,
            "run": "full" if full_run else "quick",
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        }
        for result in results
    ]
    append_jsonl(run_dir / RESULT_RECORDS, records)
    append_jsonl(run_dir / PAIR_SCORES, [{**match_keys, **scores} for scores in pair_scores])


def append_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


# The metric the results table shows for a minimal-pair dataset. A EuroEval dataset shows the first metric EuroEval
# lists, its main one. results_table.csv keeps every metric.
REPORTED_MINIMAL_PAIR_METRIC = "preference accuracy (mean)"
METRIC_NAMES = {"mcc": "MCC", "macro_f1": "macro-F1", "chr_f3pp": "chrF3++", "chr_f4pp": "chrF4++", "f1": "F1", "em": "exact match"}
# The columns of results_table.csv, one row per result record.
CSV_FIELDS = [
    "checkpoint", "revision", "model_family", "training_data", "training_method", "training_examples", "run", "suite", "dataset", "subset",
    "metric", "value", "lower", "upper", "items", "gpu", "timestamp",
]  # fmt: skip


def read_runs(results_dir, name):
    """Return the rows of the JSON Lines file called name, such as RESULT_RECORDS, from every run in
    results_dir/runs."""
    return [
        json.loads(line)
        for path in (results_dir / "runs").glob(f"*/{name}")
        for line in path.read_text().splitlines()
    ]


def write_results_table(results_dir, include_quick_runs=False):
    """Build the results table from the result records of every run in results_dir/runs. It has one row per dataset
    and subset, with the metric a report uses, and one column per checkpoint. Write it to results_table.md, in percent
    with the 95% interval, and write every latest result record to results_table.csv. Return the table as plain text
    for the terminal. A checkpoint run more than once shows its latest value. Quick runs are left out unless
    include_quick_runs is set; then a checkpoint's quick runs get a column of their own."""
    records = read_runs(results_dir, RESULT_RECORDS)
    if not records:
        raise FileNotFoundError(f"No result records in {results_dir / 'runs'}; run the download command first")
    latest = {}
    # Oldest first, so a later run replaces an earlier one. Timestamps are UTC, so they sort as text.
    for record in sorted(records, key=lambda record: record["timestamp"]):
        if record["run"] == "full" or include_quick_runs:
            key = (record["checkpoint"], record["revision"], record["run"], record["dataset"], record["subset"])
            latest[(*key, record["metric"])] = record
    if not latest:
        return f"No full runs yet. Add --include-quick-runs to see the {len(records)} quick-run result records.\n"

    with (results_dir / "results_table.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_FIELDS)
        # Runs from before 2026-10-02 have no training_data field.
        writer.writerows([record.get(field) for field in CSV_FIELDS] for record in latest.values())

    rows, euroeval_metrics = {}, {}
    for record in latest.values():
        if record["suite"] == "minimal-pair":
            if record["metric"] != REPORTED_MINIMAL_PAIR_METRIC:
                continue
            row = f"{record['dataset']} ({record['subset']})"
        else:
            metric = euroeval_metrics.setdefault(record["dataset"], record["metric"])
            if record["metric"] != metric:
                continue
            row = f"{record['dataset']} {METRIC_NAMES.get(metric, metric)}"
        rows.setdefault(row, {})[(record["checkpoint"], record["revision"], record["run"])] = record

    examples = {column: record for cells in rows.values() for column, record in cells.items()}
    # Group the columns by model family, then training data; empty labels first.
    columns = sorted(
        examples,
        key=lambda column: (
            examples[column]["model_family"] or "", examples[column].get("training_data") or "", column[0], column[2]
        ),
    )
    # A column is named after its repo or folder, plus the revision when two checkpoints share that name.
    checkpoints = {column[:2] for column in columns}
    short = {checkpoint: checkpoint[0].rstrip("/").split("/")[-1] for checkpoint in checkpoints}
    headers = [
        f"{short[column[:2]]}@{column[1][:7]}"
        if column[1] and list(short.values()).count(short[column[:2]]) > 1
        else short[column[:2]]
        for column in columns
    ]
    label_rows = [
        ["Model family", *((examples[column]["model_family"] or "").split("/")[-1] for column in columns)],
        ["Training data", *((examples[column].get("training_data") or "") for column in columns)],
        # Runs from before 2026-11-24 have no training_examples field.
        ["Training examples", *(str(examples[column].get("training_examples") or "") for column in columns)],
    ]
    if include_quick_runs:
        label_rows.append(["Run", *(column[2] for column in columns)])

    def cell(record, interval):
        if record is None:
            return ""
        if not interval or record["lower"] is None:
            return f"{100 * record['value']:.1f}"
        return f"{100 * record['value']:.1f} [{100 * record['lower']:.1f}, {100 * record['upper']:.1f}]"

    def table(interval):
        body = [[row, *(cell(cells.get(column), interval) for column in columns)] for row, cells in rows.items()]
        return [["Dataset", *headers], *label_rows, *body]

    markdown = table(interval=True)
    lines = [f"| {' | '.join(markdown[0])} |", f"| {' | '.join(['---'] * len(markdown[0]))} |"]
    lines += [f"| {' | '.join(line)} |" for line in markdown[1:]]
    (results_dir / "results_table.md").write_text("\n".join(lines) + "\n")

    text = table(interval=False)
    widths = [max(len(line[index]) for line in text) for index in range(len(text[0]))]
    return "".join(
        "  ".join([line[0].ljust(widths[0]), *(value.rjust(width) for value, width in zip(line[1:], widths[1:]))])
        .rstrip() + "\n"
        for line in text
    ) + f"\nWrote {results_dir / 'results_table.md'} (with 95% intervals) and {results_dir / 'results_table.csv'} (every metric).\n"
