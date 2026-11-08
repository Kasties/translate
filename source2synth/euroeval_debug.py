"""Re-run the EuroEval pilot while preserving every prompt, raw generation and parsed label.

EuroEval silently maps unparseable generations to the "closest" candidate label and
deletes its model cache afterwards, so the stock run cannot tell formatting failures
from wrong answers. This wraps EuroEval's own label extraction (prompts, few-shot
sampling and scoring are untouched) and writes one JSONL row per sample.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from .common import ROOT, write_json
from .euroeval_cli import install_prompt_stripping, load_tokenizer

RECORDS: list[dict] = []


def install_recorder() -> None:
    from euroeval.task_group_utils import sequence_classification as sc

    original = sc.extract_labels_from_generation

    def recording_extract(input_batch, model_output, dataset_config, model_config, **kw):
        failed_before = len(model_output.failed_instances)
        labels = original(
            input_batch=input_batch,
            model_output=model_output,
            dataset_config=dataset_config,
            model_config=model_config,
            **kw,
        )
        unmatched = {
            f["sample_index"] for f in model_output.failed_instances[failed_before:]
        }
        gold_map = dataset_config.prompt_label_mapping
        for idx, raw in enumerate(model_output.sequences):
            gold = input_batch["label"][idx]
            first_step = None
            if model_output.scores is not None and model_output.scores[idx]:
                first_step = model_output.scores[idx][0][:5]
            RECORDS.append(
                {
                    "model": model_config.model_id,
                    "dataset": dataset_config.name,
                    "prompt": input_batch["text"][idx],
                    "raw_output": raw,
                    "parsed_label": labels[idx],
                    "gold_label": gold_map.get(gold, gold),
                    "label_found": idx not in unmatched,
                    "used_logprobs": model_output.scores is not None,
                    "first_step_top_logprobs": first_step,
                }
            )
        return labels

    sc.extract_labels_from_generation = recording_extract


def summarise(records: list[dict]) -> dict:
    found = [r for r in records if r["label_found"]]
    return {
        "samples": len(records),
        "label_found": len(found),
        "fallback": len(records) - len(found),
        "accuracy_all": sum(r["parsed_label"] == r["gold_label"] for r in records)
        / max(len(records), 1),
        "accuracy_when_found": sum(r["parsed_label"] == r["gold_label"] for r in found)
        / max(len(found), 1),
        "gold_distribution": Counter(r["gold_label"] for r in records),
        "parsed_distribution": Counter(r["parsed_label"] for r in records),
        "fallback_parsed_distribution": Counter(
            r["parsed_label"] for r in records if not r["label_found"]
        ),
        "top_raw_outputs": Counter(r["raw_output"] for r in records).most_common(15),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--datasets", nargs="+", default=["swerec", "scala-sv"])
    parser.add_argument("--models", nargs="+", default=["gemma4-base", "gemma4-nlu"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/euroeval/gemma4-nlu-pilot-debug",
    )
    parser.add_argument(
        "--training-run",
        type=Path,
        default=ROOT / "artifacts/finetuning/gemma4-base-nlu-pilot",
    )
    parser.add_argument(
        "--no-strip-prompts",
        action="store_true",
        help="Reproduce stock EuroEval API behaviour (prompts end in ': ')",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("Output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    from euroeval import Benchmarker
    from euroeval.enums import GenerativeType

    install_recorder()
    if not args.no_strip_prompts:
        config = json.loads((args.training_run / "manifest.json").read_text())["config"]
        install_prompt_stripping(load_tokenizer(config["model"], config["revision"]))
    # Same protocol as source2synth.euroeval so numbers are directly comparable.
    benchmarker = Benchmarker(
        dataset=args.datasets,
        api_base=args.api_base,
        api_key="local",
        generative_type=GenerativeType.BASE,
        few_shot=True,
        evaluate_test_split=False,
        num_iterations=1,
        cache_dir=str(output_dir / "cache"),
        raise_errors=True,
        max_context_length=4096,
        progress_bar=False,
    )
    summary: dict = {
        "euroeval_version": version("euroeval"),
        "prompts_stripped": not args.no_strip_prompts,
        "runs": {},
    }
    for model in args.models:
        run_dir = output_dir / model
        run_dir.mkdir()
        start = len(RECORDS)
        cwd = os.getcwd()
        os.chdir(run_dir)
        try:
            benchmarker.benchmark(model=model)
        finally:
            os.chdir(cwd)
        # Benchmarker writes results relative to the cwd it was created in.
        results = Path(cwd) / "euroeval_benchmark_results.jsonl"
        if results.is_file():
            results.rename(run_dir / results.name)
        records = RECORDS[start:]
        with (run_dir / "predictions.jsonl").open("w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        for dataset in args.datasets:
            summary["runs"][f"{model}/{dataset}"] = summarise(
                [r for r in records if r["dataset"] == dataset]
            )
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
