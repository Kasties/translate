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
MODEL_LIST_FIELDS = ["reference", "revision", "model_family", "training_data", "training_method"]
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


# The columns that label a checkpoint's row in the results table: (result record field, Markdown header).
ROW_LABELS = [
    ("model_family", "Model family"),
    ("training_data", "Training data"),
    ("checkpoint", "Checkpoint"),
]


def read_runs(results_dir, name):
    """Return the rows of the JSON Lines file called name, such as RESULT_RECORDS, from every run in
    results_dir/runs."""
    return [
        json.loads(line)
        for path in (results_dir / "runs").glob(f"*/{name}")
        for line in path.read_text().splitlines()
    ]


def write_results_table(results_dir, include_quick_runs=False):
    """Pivot the result records of every run in results_dir/runs into one row per checkpoint and one
    column per dataset, subset and metric, write it as results_table.md (percent, with the confidence interval) and
    results_table.csv (the values as stored), and return the Markdown. A checkpoint run more than
    once shows its latest value. Quick runs are left out unless include_quick_runs is set; then
    each run gets its own row.
    """
    labels = ROW_LABELS + [("run", "Run")] if include_quick_runs else ROW_LABELS
    records = read_runs(results_dir, RESULT_RECORDS)
    if not records:
        raise FileNotFoundError(f"No result records in {results_dir / 'runs'}; run the download command first")
    records_by_row, columns = {}, {}
    # Oldest first, so a later run replaces an earlier one. Timestamps are UTC, so they sort as text.
    for record in sorted(records, key=lambda record: record["timestamp"]):
        if record["run"] == "quick" and not include_quick_runs:
            continue
        checkpoint = f"{record['checkpoint']}@{record['revision']}" if record["revision"] else record["checkpoint"]
        # Runs from before 2026-10-02 have no training_data field.
        row = tuple(checkpoint if field == "checkpoint" else record.get(field) for field, _ in labels)
        # A EuroEval dataset has no subsets.
        subset = f" ({record['subset']})" if record["subset"] else ""
        column = f"{record['dataset']}{subset} {record['metric']}"
        columns[column] = None
        records_by_row.setdefault(row, {})[column] = record
    # Group the rows by model family, then training data; empty labels first.
    rows = sorted(records_by_row, key=lambda row: [(value is not None, value) for value in row])
    cells = [[records_by_row[row].get(column) for column in columns] for row in rows]

    def percent(record):
        if record is None:
            return ""
        if record["lower"] is None:
            return f"{100 * record['value']:.2f}"
        return f"{100 * record['value']:.2f} [{100 * record['lower']:.2f}, {100 * record['upper']:.2f}]"

    table_rows = [
        [header for _, header in labels] + list(columns),
        ["---"] * (len(labels) + len(columns)),
        *(
            ["" if value is None else str(value) for value in row] + [percent(record) for record in records]
            for row, records in zip(rows, cells)
        ),
    ]
    markdown = "".join(f"| {' | '.join(table_row)} |\n" for table_row in table_rows)
    (results_dir / "results_table.md").write_text(markdown)

    with (results_dir / "results_table.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([field for field, _ in labels] + list(columns))
        for row, records in zip(rows, cells):
            writer.writerow([*row, *(record and record["value"] for record in records)])
    return markdown
