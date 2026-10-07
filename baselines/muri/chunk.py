"""Split chat-format MURI into fixed-size chunks for staged fine-tuning.

Run: uv run python -m baselines.muri.chunk [--split train] [--size 2000]

Writes data/processed/chunks/<split>_chunk_01_of_NN.jsonl, ... in the original row order
(the source mix is already interleaved, so each chunk is representative).
"""
from __future__ import annotations

from pathlib import Path

from baselines.common import chunk_main

PROCESSED = Path(__file__).resolve().parent / "data/processed"

if __name__ == "__main__":
    chunk_main("muri", PROCESSED, __doc__)
