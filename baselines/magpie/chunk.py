"""Split chat-format Swedish Magpie into fixed-size chunks for staged fine-tuning.

Run: uv run python -m baselines.magpie.chunk [--size 2000]

Writes data/processed/chunks/train_chunk_01_of_NN.jsonl, ... in the original row order
(task types are already interleaved, so each chunk is representative).
"""
from __future__ import annotations

from pathlib import Path

from baselines.common import chunk_main

PROCESSED = Path(__file__).resolve().parent / "data/processed"

if __name__ == "__main__":
    chunk_main("magpie", PROCESSED, __doc__)
