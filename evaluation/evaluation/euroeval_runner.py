"""EuroEval suite: run the pinned EuroEval CLI and convert its output into result records.

EuroEval runs through uvx in its own environment, because its vLLM dependency pins torch and
transformers versions that conflict with ours. It appends one line per model and dataset to
euroeval_benchmark_results.jsonl, in the Every Eval Ever schema v0.2.1.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import torch

from evaluation.results import append_jsonl, append_results

EUROEVAL_REQUIREMENT = "euroeval[generative]==18.1.0"
# The datasets a run uses when it names none. multi-ifeval-sv, the slowest, is left out by default.
EUROEVAL_DATASETS = ["scala-sv", "swedn", "skolprov", "multi-wiki-qa-sv", "swerec"]
# Every Swedish dataset of EuroEval 18.1.0, official and unofficial. A run can name any of them.
EUROEVAL_SWEDISH_DATASETS = [
    "scala-sv", "swedn", "multi-ifeval-sv", "skolprov", "multi-wiki-qa-sv", "swerec", "swedish-facts",
    "suc3", "winogrande-sv", "valeu-sv", "zebra-puzzles-easy-sv", "ragtruth-sv", "scandiqa-sv", "belebele-sv",
    "mmlu-sv", "arc-sv", "multiloko-sv", "hellaswag-sv", "goldenswag-sv", "schibsted-sv", "svd-seo-title",
    "ifeval-sv", "gerlangmod-sv", "zebra-puzzles-hard-sv",
]  # fmt: skip
EUROEVAL_OUTPUT = "euroeval_benchmark_results.jsonl"


def run_euroeval(run_dir, checkpoint, datasets, full_run):
    """Run EuroEval on one checkpoint, keep its output lines in run_dir and append the result records.

    A quick run uses the validation split with one iteration, a full run the test split with 10.
    """
    reference, revision = checkpoint["reference"], checkpoint["revision"]
    # vLLM needs most of the GPU memory free when it starts; give back what the minimal-pair suite
    # left cached in this process.
    torch.cuda.empty_cache()
    # EuroEval runs in a fresh folder, so it never skips a dataset or reuses model outputs cached
    # by an earlier run of a checkpoint with the same reference.
    # ponytail: downloads the model and datasets again on every run; share a cache folder if that gets slow.
    with tempfile.TemporaryDirectory() as folder:
        # EuroEval runs in that folder, so a local checkpoint needs its absolute path.
        model = str(Path(reference).resolve()) if Path(reference).is_dir() else reference
        run_arguments = ["--evaluate-test-split", "--num-iterations", "10"] if full_run else ["--num-iterations", "1"]
        try:
            subprocess.run(
                ["uvx", "--python", "3.12", "--from", EUROEVAL_REQUIREMENT, "euroeval"]
                + ["--model", f"{model}@{revision}" if revision else model]
                + [argument for dataset in datasets for argument in ("--dataset", dataset)]
                + ["--generative-type", "instruction_tuned", "--raise-errors"]
                + run_arguments,
                cwd=folder,
                check=True,
            )
        finally:
            # EuroEval writes a line when each dataset finishes, so a failed run keeps the datasets it finished.
            output = Path(folder) / EUROEVAL_OUTPUT
            lines = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
            append_jsonl(run_dir / EUROEVAL_OUTPUT, lines)
            for line in lines:
                append_euroeval_results(run_dir, checkpoint, line, full_run)


def append_euroeval_results(run_dir, checkpoint, output, full_run):
    """Append one result record per metric in one line of EuroEval's output, parsed from JSON."""
    results = []
    for result in output["evaluation_results"]:
        details = result["score_details"]
        # A run with one iteration has no confidence interval.
        interval = details.get("uncertainty", {}).get("confidence_interval", {})
        results.append(
            {
                # A EuroEval dataset has no subsets.
                "subset": None,
                # EuroEval prefixes every metric with "test_", on the validation split too.
                "metric": result["evaluation_name"].removeprefix("test_"),
                # EuroEval reports percent; our values are fractions like preference accuracy.
                "value": details["score"] / 100,
                "lower": interval["lower"] / 100 if interval else None,
                "upper": interval["upper"] / 100 if interval else None,
                # Not reported by EuroEval.
                "items": None,
            }
        )
    dataset = output["eval_library"]["additional_details"]["dataset"]
    append_results(run_dir, checkpoint, "euroeval", dataset, results, [], full_run)
