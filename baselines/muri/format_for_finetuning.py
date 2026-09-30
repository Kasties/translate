"""Convert Swedish MURI-IT to chat-format JSONL (data/processed/<split>.jsonl).

Run: uv run python -m baselines.muri.format_for_finetuning [--instruction COL --response COL]
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from baselines.common import column_args, detect_columns, to_chat, write_jsonl

PROCESSED = Path(__file__).resolve().parent / "data/processed"


def main() -> None:
    args = column_args(__doc__)
    source = PROCESSED / "swedish.parquet"
    if not source.exists():
        raise SystemExit(f"{source} not found; run baselines.muri.filter_swedish first.")
    df = pd.read_parquet(source)
    instruction, response = detect_columns(list(df.columns), args.instruction, args.response)
    for split, part in df.groupby("split", sort=False):
        print(f"[{split}]")
        rows = to_chat(part.reset_index(drop=True), f"muri-{split}", instruction, response)
        write_jsonl(PROCESSED / f"{split}.jsonl", rows)
        print(f"  -> {PROCESSED / f'{split}.jsonl'}")


if __name__ == "__main__":
    main()
