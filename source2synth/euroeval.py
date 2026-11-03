"""Run matched base/adapter EuroEval validation benchmarks against serve-eval.sh."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from importlib.metadata import version
from pathlib import Path

from .common import ROOT, write_json


def benchmark_commands(
    python: Path,
    api_base: str,
    datasets: list[str],
    iterations: int,
    cache_dir: Path,
    tokenizer: str,
    tokenizer_revision: str,
    models: list[str] = ("gemma4-base", "gemma4-nlu"),
    generative_type: str = "base",
    max_context_length: int = 4096,
) -> list[tuple[str, list[str]]]:
    # Explicit base + few-shot keeps the input protocol identical before/after
    # SFT. Automatic type detection would otherwise change prompting as well.
    # euroeval_cli strips base prompts like EuroEval's own vLLM backend does.
    common = [
        str(python),
        "-m",
        "source2synth.euroeval_cli",
        "--tokenizer",
        tokenizer,
        "--tokenizer-revision",
        tokenizer_revision,
        "--",
        "--api-base",
        api_base,
        "--api-key",
        "local",
        "--generative-type",
        generative_type,
        "--few-shot",
        "--evaluate-val-split",
        "--num-iterations",
        str(iterations),
        "--cache-dir",
        str(cache_dir),
        "--raise-errors",
        "--max-context-length",
        str(max_context_length),
    ]
    for dataset in datasets:
        common.extend(["--dataset", dataset])
    return [(name, common + ["--model", name]) for name in models]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--datasets", nargs="+", default=["swerec", "scala-sv"])
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/euroeval/gemma4-nlu-pilot"
    )
    parser.add_argument(
        "--training-run",
        default=str(ROOT / "artifacts/finetuning/gemma4-base-nlu-pilot"),
        help="Training output linked in the manifest; 'none' to evaluate without one",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gemma4-base", "gemma4-nlu"],
        help="Served model names; the first is the baseline",
    )
    parser.add_argument("--tokenizer", help="Default: the training run's base model")
    parser.add_argument("--tokenizer-revision", help="Default: the training run's revision")
    parser.add_argument("--generative-type", choices=["base", "instruction_tuned"], default="base")
    parser.add_argument("--max-context-length", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    manifest = None
    if args.training_run != "none":
        manifest = json.loads((Path(args.training_run) / "manifest.json").read_text())
    if not args.tokenizer and not manifest:
        parser.error("--tokenizer is required with --training-run none")
    tokenizer = args.tokenizer or manifest["config"]["model"]
    revision = args.tokenizer_revision or (
        manifest["config"].get("revision") if manifest and not args.tokenizer else None
    )
    commands = benchmark_commands(
        Path(sys.executable),
        args.api_base,
        args.datasets,
        args.iterations,
        args.output_dir.resolve() / "cache",
        tokenizer,
        revision or "main",
        args.models,
        args.generative_type,
        args.max_context_length,
    )
    if args.dry_run:
        print(json.dumps(dict(commands), indent=2))
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(
            "Output directory must be empty; choose a new --output-dir to avoid reusing stale results"
        )
    if manifest and manifest["status"] != "complete":
        parser.error("Training run is not complete")
    with urllib.request.urlopen(
        args.api_base.rstrip("/") + "/models", timeout=10
    ) as response:
        served = {m["id"] for m in json.load(response)["data"]}
    if not {name for name, _ in commands}.issubset(served):
        parser.error(
            f"Not served: {sorted({n for n, _ in commands} - served)}; start scripts/serve-eval.sh"
        )
    experiment = {
        "status": "running",
        "euroeval_version": version("euroeval"),
        "training_manifest": manifest,
        "models": args.models,
        "tokenizer": {"model": tokenizer, "revision": revision},
        "datasets": args.datasets,
        "iterations": args.iterations,
        "split": "validation",
        "few_shot": True,
        "generative_type": args.generative_type,
        "api_base": args.api_base,
        "commands": dict(commands),
        "prompt_stripping": "EuroEval vLLM-backend rule applied to LiteLLM base prompts",
        "note": ("Pilot validation scores; one iteration is not a leaderboard-comparable evaluation."
                 if args.iterations == 1 else
                 f"Validation split, {args.iterations} iterations through the local API server; compare "
                 "against this project's own base runs rather than leaderboard rows."),
    }
    write_json(args.output_dir / "manifest.json", experiment)
    for name, command in commands:
        run_dir = args.output_dir.resolve() / name
        run_dir.mkdir()
        with (run_dir / "run.log").open("w") as log:
            print(f"Evaluating {name}; log: {run_dir / 'run.log'}", flush=True)
            subprocess.run(
                command,
                cwd=run_dir,
                env={**os.environ, "PYTHONPATH": str(ROOT)},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        results_file = run_dir / "euroeval_benchmark_results.jsonl"
        if not results_file.is_file() or not results_file.read_text().strip():
            raise RuntimeError(
                f"EuroEval produced no results for {name}; inspect {run_dir / 'run.log'}"
            )
    experiment["status"] = "complete"
    write_json(args.output_dir / "manifest.json", experiment)
    print(f"Results saved under {args.output_dir}")


if __name__ == "__main__":
    main()
