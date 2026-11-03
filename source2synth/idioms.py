"""Swedish idiom minimal-pair benchmark (Kunz, TACL 2026) against serve-eval.sh.

Dataset: https://huggingface.co/datasets/liu-nlp/swedish-idioms
Paper:   https://arxiv.org/abs/2602.03484

Each of the 478 items pairs a sentence containing a conventional Swedish idiom
with an identical sentence whose idiomatic keyword is replaced by a plausible
distractor ("anar ugglor i mossen" vs "anar råttor i mossen"). As in the paper's
reference code (jekunz/idiomatic-language, eval/eval_utils.py,
compute_perplexity_unbatched), each stripped sentence is tokenized with BOS,
every token after BOS is scored, and the pair counts as correct only when the
idiomatic sentence has strictly lower mean-token perplexity. Accuracy is
reported on all items and on the 100-item challenge subset (idioms without an
English/German counterpart). This is not a EuroEval dataset; it has no
train/validation/test split and no prompt, so there is no few-shot variant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

from .common import ROOT, write_json, write_jsonl

DATASET = "liu-nlp/swedish-idioms"
DATASET_REVISION = "3d33f55db6c45a82a6a237f3cbee6490af08327d"
DATASET_FILE = "data/train-00000-of-00001.parquet"
POSITIVE = "Positive (Minimal Pairs Setup)"
NEGATIVE = "Negative (Minimal Pairs Setup)"
CHALLENGE = "Challenge Set"


def load_items() -> tuple[list[dict], str]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        DATASET, DATASET_FILE, repo_type="dataset", revision=DATASET_REVISION
    )
    sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    items = [
        {
            "index": index,
            "question": row["Question (Generative Setup)"],
            "answer": row["Answer (Generative Setup)"],
            # Same whitespace handling as the reference implementation.
            "positive": str(row[POSITIVE]).strip(),
            "negative": str(row[NEGATIVE]).strip(),
            "challenge": row[CHALLENGE] == "x",
        }
        for index, row in pd.read_parquet(path).iterrows()
    ]
    return items, sha256


def mean_nll(token_ids: list[int], prompt_logprobs: list) -> float:
    """Mean negative log-likelihood of every token after BOS, as HF labels=input_ids."""
    if len(prompt_logprobs) != len(token_ids) or prompt_logprobs[0] is not None:
        raise ValueError("prompt_logprobs do not align with the submitted tokens")
    nll = []
    for token, entry in zip(token_ids[1:], prompt_logprobs[1:]):
        if str(token) not in entry:
            raise ValueError(f"No logprob returned for submitted token {token}")
        nll.append(-entry[str(token)]["logprob"])
    return sum(nll) / len(nll)


def score_sentence(api_base: str, model: str, token_ids: list[int]) -> float:
    request = urllib.request.Request(
        api_base.rstrip("/") + "/completions",
        data=json.dumps(
            {
                "model": model,
                "prompt": token_ids,
                "max_tokens": 1,
                "temperature": 0,
                "prompt_logprobs": 0,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        choice = json.load(response)["choices"][0]
    return mean_nll(token_ids, choice["prompt_logprobs"])


def wilson(correct: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson interval, matching the paper's compute_confidence_interval."""
    if total == 0:
        return 0.0, 1.0
    p = correct / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def accuracy(rows: list[dict]) -> dict:
    correct = sum(r["correct"] for r in rows)
    low, high = wilson(correct, len(rows))
    return {
        "accuracy": correct / len(rows) if rows else None,
        "correct": correct,
        "total": len(rows),
        "ci95": [low, high],
    }


def subsets(rows: list[dict]) -> dict[str, list[dict]]:
    return {"all": rows, "challenge": [r for r in rows if r["challenge"]]}


def summarise(predictions: dict[str, list[dict]]) -> dict:
    summary: dict = {
        "models": {
            model: {name: accuracy(part) for name, part in subsets(rows).items()}
            for model, rows in predictions.items()
        }
    }
    models = list(predictions)
    if len(models) == 2:
        a, b = models
        summary["paired"] = {}
        for name, rows_a in subsets(predictions[a]).items():
            rows_b = subsets(predictions[b])[name]
            only_a = sum(x["correct"] and not y["correct"] for x, y in zip(rows_a, rows_b))
            only_b = sum(y["correct"] and not x["correct"] for x, y in zip(rows_a, rows_b))
            summary["paired"][name] = {
                f"only_{a}": only_a,
                f"only_{b}": only_b,
                "mcnemar_exact_p": mcnemar_exact(only_a, only_b),
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--models", nargs="+", default=["gemma4-base", "gemma4-nlu"])
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/idioms/gemma4-nlu-pilot"
    )
    parser.add_argument(
        "--training-run",
        default=str(ROOT / "artifacts/finetuning/gemma4-base-nlu-pilot"),
        help="Training output linked in the manifest; 'none' to evaluate without one",
    )
    parser.add_argument("--tokenizer", help="Default: the training run's base model")
    parser.add_argument("--tokenizer-revision", help="Default: the training run's revision")
    parser.add_argument("--limit", type=int, help="Score only the first N items (smoke test)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("Output directory must be empty")
    manifest = None
    if args.training_run != "none":
        manifest = json.loads((Path(args.training_run) / "manifest.json").read_text())
    if not args.tokenizer and not manifest:
        parser.error("--tokenizer is required with --training-run none")
    with urllib.request.urlopen(args.api_base.rstrip("/") + "/models", timeout=10) as response:
        served = {m["id"] for m in json.load(response)["data"]}
    if not set(args.models) <= served:
        parser.error(f"Not served: {sorted(set(args.models) - served)}; start scripts/serve-eval.sh")

    from transformers import AutoTokenizer

    tokenizer_name = args.tokenizer or manifest["config"]["model"]
    revision = args.tokenizer_revision or (
        manifest["config"].get("revision") if manifest and not args.tokenizer else None
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=revision)
    if tokenizer.bos_token_id is None:
        raise SystemExit(
            f"{tokenizer_name} has no BOS token; the paper's perplexity protocol scores "
            "every token after BOS. Skip this benchmark for this model."
        )
    items, sha256 = load_items()
    items = items[: args.limit] if args.limit else items
    for item in items:
        for side in ("positive", "negative"):
            ids = tokenizer(item[side])["input_ids"]
            if ids[0] != tokenizer.bos_token_id or len(ids) < 2:
                raise RuntimeError(f"Unexpected tokenization for item {item['index']}")
            item[f"{side}_ids"] = ids

    experiment = {
        "status": "running",
        "benchmark": "swedish-idioms",
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "dataset_sha256": sha256,
        "items": len(items),
        "models": args.models,
        "api_base": args.api_base,
        "tokenizer": {"model": tokenizer_name, "revision": revision},
        "training_manifest": manifest,
        "metric": "accuracy: mean-token perplexity(idiomatic) < perplexity(distractor), ties wrong",
        "transformers_version": version("transformers"),
        "note": "Paper protocol on the vLLM server used for EuroEval; the paper scored "
        "bf16 HF models, so absolute numbers are comparable within this project only.",
    }
    write_json(output_dir / "manifest.json", experiment)
    predictions = {}
    with ThreadPoolExecutor(args.workers) as pool:
        for model in args.models:
            print(f"Scoring {len(items)} pairs with {model}", flush=True)
            jobs = [
                (pool.submit(score_sentence, args.api_base, model, i["positive_ids"]),
                 pool.submit(score_sentence, args.api_base, model, i["negative_ids"]))
                for i in items
            ]
            rows = []
            for item, (positive, negative) in zip(items, jobs):
                nll_pos, nll_neg = positive.result(), negative.result()
                rows.append(
                    {
                        "index": item["index"],
                        "challenge": item["challenge"],
                        "answer": item["answer"],
                        "positive": item["positive"],
                        "negative": item["negative"],
                        "positive_tokens": len(item["positive_ids"]) - 1,
                        "negative_tokens": len(item["negative_ids"]) - 1,
                        "positive_ppl": math.exp(nll_pos),
                        "negative_ppl": math.exp(nll_neg),
                        "correct": nll_pos < nll_neg,
                    }
                )
            write_jsonl(output_dir / model / "predictions.jsonl", rows)
            predictions[model] = rows
    summary = summarise(predictions)
    write_json(output_dir / "summary.json", summary)
    experiment["status"] = "complete"
    write_json(output_dir / "manifest.json", experiment)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
