"""Count tokens in the processed MURI and Magpie baselines and trim both to a shared token budget.

Run: uv run python -m baselines.tokenize_and_match [--tokenizer NAME] [--target-tokens N]

Reads baselines/<name>/data/processed/<SPLIT>.jsonl and writes <SPLIT>_matched.jsonl next to it.
Examples are drawn in a seeded random order until the budget would be exceeded, so each trimmed
set is a random subsample rather than the head of the file.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from baselines.common import ROOT, read_jsonl, write_jsonl

# Placeholder until the fine-tuning model is confirmed. google/gemma-* is gated on Hugging Face:
# accept the license on the model page and set HF_TOKEN, or use an ungated copy of the same
# tokenizer (e.g. unsloth/gemma-3-4b-it).
TOKENIZER = "google/gemma-3-4b-it"
# Shared per-dataset token budget, pending confirmation with the supervisor.
# None = use the smallest dataset's total, so every baseline is trimmed to the same size.
TARGET_TOKENS: int | None = None
SPLIT = "train"
SEED = 0
BATCH = 1000

DATASETS = {
    "muri": ROOT / "baselines/muri/data/processed",
    "magpie": ROOT / "baselines/magpie/data/processed",
}


def load_tokenizer(name: str):
    try:
        return AutoTokenizer.from_pretrained(name)
    except OSError as error:
        raise SystemExit(f"Could not load tokenizer {name!r}. If the model is gated, accept its license on "
                         f"Hugging Face and set HF_TOKEN, or pass --tokenizer with an ungated copy.\n{error}")


def count_tokens(tokenizer, rows: list[dict]) -> np.ndarray:
    """Token count per example, as the model sees it after applying the chat template."""
    if tokenizer.chat_template:
        texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False) for r in rows]
        special = False  # The template already inserts BOS and turn markers.
    else:
        texts = ["\n\n".join(m["content"] for m in r["messages"]) for r in rows]
        special = True
    lengths = []
    for start in range(0, len(texts), BATCH):
        ids = tokenizer(texts[start:start + BATCH], add_special_tokens=special)["input_ids"]
        lengths.extend(len(x) for x in ids)
    return np.array(lengths, dtype=np.int64)


def trim(lengths: np.ndarray, budget: int, seed: int) -> np.ndarray:
    """Indices (in original order) of a random subset whose total stays within the budget."""
    order = np.random.default_rng(seed).permutation(len(lengths))
    keep = order[np.cumsum(lengths[order]) <= budget]
    return np.sort(keep)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", default=TOKENIZER)
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    print(f"Tokenizer: {args.tokenizer}")

    data = {}
    for name, directory in DATASETS.items():
        path = directory / f"{SPLIT}.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} not found; run baselines.{name}.format_for_finetuning first.")
        rows = read_jsonl(path)
        lengths = count_tokens(tokenizer, rows)
        data[name] = (rows, lengths)
        print(f"{name}: {len(rows):,} examples, {lengths.sum():,} tokens "
              f"(mean {lengths.mean():.0f}, median {np.median(lengths):.0f}, max {lengths.max():,})")

    budget = args.target_tokens or min(int(lengths.sum()) for _, lengths in data.values())
    print(f"\nTarget budget per dataset: {budget:,} tokens"
          + ("" if args.target_tokens else " (smallest dataset's total)"))

    for name, (rows, lengths) in data.items():
        if lengths.sum() < budget:
            print(f"WARNING: {name} has only {lengths.sum():,} tokens, below the budget; keeping all of it.")
        keep = trim(lengths, budget, SEED)
        out = DATASETS[name] / f"{SPLIT}_matched.jsonl"
        write_jsonl(out, [rows[i] for i in keep])
        print(f"{name}: kept {len(keep):,}/{len(rows):,} examples, {lengths[keep].sum():,} tokens -> {out}")


if __name__ == "__main__":
    main()
