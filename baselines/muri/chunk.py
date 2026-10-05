"""Split chat-format MURI into fixed-size chunks for staged fine-tuning.

Run: uv run python -m baselines.muri.chunk [--split train] [--size 2000]

Writes data/processed/chunks/<split>_chunk_01_of_NN.jsonl, ... in the original row order
(the source mix is already interleaved, so each chunk is representative).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from baselines.common import read_jsonl, write_jsonl

PROCESSED = Path(__file__).resolve().parent / "data/processed"
CHUNKS = PROCESSED / "chunks"
CHUNK_SIZE = 2000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="train")
    parser.add_argument("--size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    source = PROCESSED / f"{args.split}.jsonl"
    if not source.exists():
        raise SystemExit(f"{source} not found; run baselines.muri.format_for_finetuning first.")
    rows = read_jsonl(source)
    total = math.ceil(len(rows) / args.size)

    CHUNKS.mkdir(parents=True, exist_ok=True)
    for old in CHUNKS.glob(f"{args.split}_chunk_*.jsonl"):
        old.unlink()
    for n in range(total):
        part = rows[n * args.size:(n + 1) * args.size]
        path = CHUNKS / f"{args.split}_chunk_{n + 1:02d}_of_{total:02d}.jsonl"
        write_jsonl(path, part)
        print(f"{path.name}: {len(part):,} rows ({part[0]['id']} .. {part[-1]['id']})")
    print(f"{len(rows):,} rows -> {total} chunks in {CHUNKS}")


if __name__ == "__main__":
    main()
