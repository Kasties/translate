"""Evaluate checkpoints and build the results table.

uv run python -m evaluation run (--model-list models.yaml | --model HuggingFaceTB/SmolLM2-135M [--revision REVISION])
    [--suite minimal-pair euroeval] [--minimal-pair-dataset Idioms ...] [--euroeval-dataset scala-sv ...]
    [--full-run] [--no-upload] [--results-dir DIR]
uv run python -m evaluation download [--results-dir DIR]
uv run python -m evaluation table [--include-quick-runs] [--results-dir DIR]
"""

import argparse
from pathlib import Path

from evaluation.euroeval_runner import EUROEVAL_DATASETS, run_euroeval
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser(
        "run", help="evaluate checkpoints on the chosen suites, write the result records and upload them"
    )
    model_or_model_list = run.add_mutually_exclusive_group(required=True)
    model_or_model_list.add_argument(
        "--model-list", type=Path, help="the model list, such as models.yaml; each checkpoint in it is its own run"
    )
    model_or_model_list.add_argument("--model", help="Hugging Face repo id or local path of one checkpoint")
    run.add_argument("--revision", help="with --model: branch, tag or commit hash of the Hugging Face repo")
    run.add_argument(
        "--suite",
        nargs="+",
        choices=SUITES,
        default=["minimal-pair"],
        help="the suites to run; default is the minimal-pair suite. The EuroEval suite needs a CUDA GPU",
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
        choices=EUROEVAL_DATASETS,
        help="a EuroEval dataset to run; repeat for more. Default is all six",
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
        if args.model_list and args.revision:
            parser.error("--revision goes with --model; give a model list entry its revision in the model list")
        if args.model_list:
            checkpoints = load_model_list(args.model_list)
        else:
            checkpoints = [checkpoint_from_metadata(args.model, args.revision)]
        for checkpoint in checkpoints:
            # EuroEval cannot load a local checkpoint directory given with a revision.
            if checkpoint["revision"] and Path(checkpoint["reference"]).is_dir():
                parser.error(
                    f"{checkpoint['reference']} is a local checkpoint directory, which has no revisions; "
                    "give a revision only with a Hugging Face repo id"
                )
        # Stop before any run, not after the scoring, when the run's files could not be uploaded.
        if not args.no_upload:
            try:
                results_repo()
            except RuntimeError as error:
                parser.error(str(error))
        for checkpoint in checkpoints:
            # A model list entry with suites runs only those of the chosen suites, so a base model can
            # skip the EuroEval suite. A checkpoint that runs no suite writes no run folder.
            suites = [suite for suite in args.suite if suite in (checkpoint.get("suites") or args.suite)]
            skipped = [suite for suite in args.suite if suite not in suites]
            print(f"Evaluating {checkpoint['reference']}" + (f", skipping {', '.join(skipped)}" if skipped else ""))
            run_dir = new_run_dir(args.results_dir)
            try:
                if "minimal-pair" in suites:
                    run_minimal_pair_suite(
                        run_dir, checkpoint, args.minimal_pair_dataset or MINIMAL_PAIR_DATASETS, args.full_run
                    )
                if "euroeval" in suites:
                    run_euroeval(run_dir, checkpoint, args.euroeval_dataset or EUROEVAL_DATASETS, args.full_run)
            finally:
                # A failed run still uploads the datasets it finished.
                if run_dir.exists() and not args.no_upload:
                    upload_run(run_dir)
    elif args.command == "download":
        download_runs(args.results_dir)
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
