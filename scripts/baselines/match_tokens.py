"""Draw MURI and Magpie subsets with the same token budget as one of our training sets.

    .venv-train/bin/python scripts/baselines/match_tokens.py \
        --reference artifacts/source2synth/runs/register-native-gemma4-31b-gen/train.jsonl \
        --baseline muri=artifacts/baselines/baselines/muri/data/processed/train.jsonl \
        --baseline magpie=artifacts/baselines/baselines/magpie/data/processed/train.jsonl \
        --output artifacts/baselines/matched

Follows baselines/tokenize_and_match.py on feature/baseline-datasets: whole examples are counted
as the model sees them after the chat template, and a seeded random order is taken until the
budget would be exceeded. Two changes: the budget is the reference set's total instead of the
smallest baseline's, and examples longer than --max-length are left out of the pool, because
finetune.py refuses to truncate them (MURI has a few raw web texts of up to 37,500 tokens).
Counting uses the template our adapters were trained with (USER:/ASSISTANT: turns), because the
base model has none.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--baseline", action="append", required=True, help="name=path to a messages JSONL")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--tokenizer", default="google/gemma-4-12B")
parser.add_argument("--template", type=Path,
                    default=Path("artifacts/finetuning/register-native-gemma4-31b-gen/adapter/chat_template.jinja"))
parser.add_argument("--budget-tokens", type=int,
                    help="fixed budget instead of the reference's total; the same seed draws in the same order, "
                         "so a larger budget keeps every example of a smaller one")
parser.add_argument("--max-length", type=int, default=2048)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
tokenizer.chat_template = args.template.read_text(encoding="utf-8")


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def lengths(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Tokens per whole example (template adds BOS and turn markers) and per assistant answer."""
    texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False) for r in rows]
    answers = [" ".join(m["content"] for m in r["messages"] if m["role"] == "assistant") for r in rows]
    whole = [len(x) for x in tokenizer(texts, add_special_tokens=False)["input_ids"]]
    answer = [len(x) for x in tokenizer(answers, add_special_tokens=False)["input_ids"]]
    return np.array(whole), np.array(answer)


reference = read(args.reference)
ref_whole, ref_answer = lengths(reference)
budget = args.budget_tokens or int(ref_whole.sum())
report = {"reference": {"path": str(args.reference), "examples": len(reference), "tokens": int(ref_whole.sum()),
                        "answer_tokens": int(ref_answer.sum()), "median_tokens": float(np.median(ref_whole))},
          "budget_tokens": budget, "tokenizer": args.tokenizer, "max_length": args.max_length, "seed": args.seed}
print(f"reference: {len(reference)} examples, {ref_whole.sum():,} tokens (answers {ref_answer.sum():,}); "
      f"budget {budget:,} tokens")
for spec in args.baseline:
    name, path = spec.split("=", 1)
    rows = read(Path(path))
    whole, answer = lengths(rows)
    fits = whole <= args.max_length
    order = np.random.default_rng(args.seed).permutation(len(rows))
    order = order[fits[order]]
    keep = np.sort(order[np.cumsum(whole[order]) <= budget])
    out = args.output / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(rows[i], ensure_ascii=False) + "\n" for i in keep), encoding="utf-8")
    report[name] = {"source": path, "available": len(rows), "examples": len(keep), "tokens": int(whole[keep].sum()),
                    "answer_tokens": int(answer[keep].sum()), "median_tokens": float(np.median(whole[keep])),
                    "too_long_in_pool": int((~fits).sum()), "output": str(out)}
    print(f"{name}: {(~fits).sum()} of {len(rows):,} examples over {args.max_length} tokens left out; "
          f"kept {len(keep):,}, {whole[keep].sum():,} tokens "
          f"(answers {answer[keep].sum():,}, median {np.median(whole[keep]):.0f}) -> {out}")
(args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
