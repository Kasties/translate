"""Evaluate checkpoints and build the results table.

uv run python -m evaluation run --config example_run.yaml [--results-dir DIR]
uv run python -m evaluation run --model HuggingFaceTB/SmolLM2-135M [--revision REVISION]
    [--suite minimal-pair euroeval] [--minimal-pair-dataset Idioms ...] [--euroeval-dataset scala-sv ...]
    [--full-run] [--no-upload] [--results-dir DIR]
uv run python -m evaluation download [--results-dir DIR]
uv run python -m evaluation table [--include-quick-runs] [--results-dir DIR]
"""

import argparse
from pathlib import Path

import yaml

from evaluation.euroeval_runner import EUROEVAL_DATASETS, EUROEVAL_SWEDISH_DATASETS, run_euroeval
from evaluation.minimal_pairs import MINIMAL_PAIR_DATASETS, evaluate, load_model
from evaluation.results import (
    SUITES,
    append_results,
    download_runs,
    checkpoint_from_metadata,
    load_model_list,
    new_run_dir,
    results_repo,
    upload_run,
    write_results_table,
)


# The fields of a run config. models holds model list entries; the others choose what every checkpoint runs.
RUN_CONFIG_FIELDS = ["suites", "minimal_pair_datasets", "euroeval_datasets", "full_run", "upload", "models"]


def load_run_config(path):
    """Return the run config at path as the settings of a run. A field left out takes the command's default,
    except upload, which is off unless the config turns it on."""
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or "models" not in config or set(config) - set(RUN_CONFIG_FIELDS):
        raise ValueError(f"The run config {path} needs models and only these fields: {RUN_CONFIG_FIELDS}")
    for field, known in [
        ("suites", SUITES),
        ("minimal_pair_datasets", list(MINIMAL_PAIR_DATASETS)),
        ("euroeval_datasets", EUROEVAL_SWEDISH_DATASETS),
    ]:
        unknown = set(config.get(field) or []) - set(known)
        if unknown:
            raise ValueError(f"The run config {path} has unknown {field} {sorted(unknown)}; choose from {known}")
    return {
        "checkpoints": load_model_list(config["models"]),
        "suites": config.get("suites") or SUITES,
        "minimal_pair_datasets": config.get("minimal_pair_datasets") or list(MINIMAL_PAIR_DATASETS),
        "euroeval_datasets": config.get("euroeval_datasets") or EUROEVAL_DATASETS,
        "full_run": bool(config.get("full_run")),
        "upload": bool(config.get("upload")),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser(
        "run", help="evaluate checkpoints on the chosen suites, write the result records and upload them"
    )
    config_or_model = run.add_mutually_exclusive_group(required=True)
    config_or_model.add_argument(
        "--config",
        type=Path,
        help="a run config, such as example_run.yaml: the checkpoints, suites, datasets, quick or full run, and upload",
    )
    config_or_model.add_argument("--model", help="Hugging Face repo id or local path of one checkpoint")
    run.add_argument("--revision", help="with --model: branch, tag or commit hash of the Hugging Face repo")
    run.add_argument(
        "--suite",
        nargs="+",
        choices=SUITES,
        help="the suites to run; default is both. The EuroEval suite needs a CUDA GPU",
    )
    run.add_argument(
        "--minimal-pair-dataset",
        action="append",
        choices=MINIMAL_PAIR_DATASETS,
        help="a minimal-pair dataset to run; repeat for more. Default is all of them",
    )
    run.add_argument(
        "--euroeval-dataset",
        action="append",
        choices=EUROEVAL_SWEDISH_DATASETS,
        metavar="EUROEVAL_DATASET",
        help=f"a Swedish EuroEval dataset to run; repeat for more. Default is {', '.join(EUROEVAL_DATASETS)}",
    )
    run.add_argument("--full-run", action="store_true", help="mark the results as reportable; default is a quick run")
    run.add_argument("--no-upload", action="store_true", help="keep the run's files on this machine only")

    download = commands.add_parser("download", help="download every run's files from the results repo")

    table = commands.add_parser("table", help="write the result records as a results table in Markdown and CSV")
    table.add_argument(
        "--include-quick-runs", action="store_true", help="add quick-run rows; default is full runs only"
    )

    for command in (run, download, table):
        command.add_argument("--results-dir", type=Path, default=Path("results"), help="folder of the results files")
    args = parser.parse_args(argv)

    if args.command == "run":
        if args.config:
            if any([args.revision, args.suite, args.minimal_pair_dataset, args.euroeval_dataset, args.full_run,
                    args.no_upload]):  # fmt: skip
                parser.error("with --config, set the suites, datasets, full_run, upload and revisions in the run config")
            settings = load_run_config(args.config)
        else:
            settings = {
                "checkpoints": [checkpoint_from_metadata(args.model, args.revision)],
                "suites": args.suite or SUITES,
                "minimal_pair_datasets": args.minimal_pair_dataset or list(MINIMAL_PAIR_DATASETS),
                "euroeval_datasets": args.euroeval_dataset or EUROEVAL_DATASETS,
                "full_run": args.full_run,
                "upload": not args.no_upload,
            }
        checkpoints, suites = settings["checkpoints"], settings["suites"]
        for checkpoint in checkpoints:
            # EuroEval cannot load a local checkpoint directory given with a revision.
            if checkpoint["revision"] and Path(checkpoint["reference"]).is_dir():
                parser.error(
                    f"{checkpoint['reference']} is a local checkpoint directory, which has no revisions; "
                    "give a revision only with a Hugging Face repo id"
                )
        # Stop before any run, not after the scoring, when the run's files could not be uploaded.
        if settings["upload"]:
            try:
                results_repo()
            except RuntimeError as error:
                parser.error(str(error))
        for checkpoint in checkpoints:
            # Every checkpoint gets the same evaluation: the run's suites and datasets.
            print(f"Evaluating {checkpoint['reference']}")
            run_dir = new_run_dir(args.results_dir)
            try:
                if "minimal-pair" in suites:
                    run_minimal_pair_suite(run_dir, checkpoint, settings["minimal_pair_datasets"], settings["full_run"])
                if "euroeval" in suites:
                    run_euroeval(run_dir, checkpoint, settings["euroeval_datasets"], settings["full_run"])
            finally:
                # A failed run still uploads the datasets it finished.
                if run_dir.exists() and settings["upload"]:
                    upload_run(run_dir)
    elif args.command == "download":
        try:
            download_runs(args.results_dir)
        except RuntimeError as error:
            parser.error(str(error))
    else:
        print(write_results_table(args.results_dir, args.include_quick_runs), end="")


def run_minimal_pair_suite(run_dir, checkpoint, datasets, full_run):
    """Evaluate one checkpoint on the minimal-pair datasets, write the result records to run_dir and print them."""
    model, tokenizer = load_model(checkpoint["reference"], checkpoint["revision"])
    for dataset in datasets:
        results, pair_scores = evaluate(model, tokenizer, MINIMAL_PAIR_DATASETS[dataset]())
        append_results(run_dir, checkpoint, "minimal-pair", dataset, results, pair_scores, full_run)

        for result in results:
            print(
                f"{dataset} ({result['subset']}), {result['items']} pairs: {result['metric']} "
                f"{100 * result['value']:.2f} [{100 * result['lower']:.2f}, {100 * result['upper']:.2f}]"
            )


if __name__ == "__main__":
    main()
