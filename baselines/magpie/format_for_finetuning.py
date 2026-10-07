"""Convert Magpie to chat-format JSONL (data/processed/<split>.jsonl).

Run: uv run python -m baselines.magpie.format_for_finetuning [--instruction COL --response COL]
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from baselines.common import column_args, to_chat, write_jsonl

HERE = Path(__file__).resolve().parent
RAW = HERE / "data/raw"
PROCESSED = HERE / "data/processed"


def main() -> None:
    args = column_args(__doc__)
    files = sorted(RAW.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No raw data in {RAW}; run baselines.magpie.download first.")
    for path in files:
        print(f"[{path.stem}]")
        rows = to_chat(pd.read_parquet(path), f"magpie-{path.stem}", args.instruction, args.response)
        write_jsonl(PROCESSED / f"{path.stem}.jsonl", rows)
        print(f"  -> {PROCESSED / f'{path.stem}.jsonl'}")


if __name__ == "__main__":
    main()
