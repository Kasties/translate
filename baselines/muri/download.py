"""Download MURI-IT (Köksal et al., 2024) from Hugging Face into data/raw/.

Run: uv run python -m baselines.muri.download
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import DatasetDict, load_dataset
from datasets.exceptions import DatasetNotFoundError

RAW = Path(__file__).resolve().parent / "data/raw"

# The language-split repo has one config per ISO 639-3 code, so only Swedish is fetched.
PRIMARY = ("akoksal/muri-it-language-split", "swe")
# The full multilingual dump (~200 languages); filter_swedish.py narrows it down.
FALLBACK = ("akoksal/muri-it", None)


def download() -> tuple[str, str | None, DatasetDict]:
    try:
        return (*PRIMARY, load_dataset(*PRIMARY))
    except (DatasetNotFoundError, ValueError, FileNotFoundError) as error:
        print(f"{PRIMARY[0]} ({PRIMARY[1]}) unavailable ({error}); falling back to {FALLBACK[0]}")
        return (*FALLBACK, load_dataset(FALLBACK[0]))


def main() -> None:
    repo, config, dataset = download()
    RAW.mkdir(parents=True, exist_ok=True)
    for old in RAW.glob("*.parquet"):
        old.unlink()
    for split, data in dataset.items():
        data.to_parquet(RAW / f"{split}.parquet")
    (RAW / "source.json").write_text(json.dumps({"repo": repo, "config": config}, indent=2) + "\n", encoding="utf-8")

    print(f"Source: {repo}" + (f" (config {config})" if config else ""))
    print(f"Columns: {next(iter(dataset.values())).column_names}")
    for split, data in dataset.items():
        print(f"  {split}: {data.num_rows:,} rows")
    print(f"Total: {sum(d.num_rows for d in dataset.values()):,} rows -> {RAW}")


if __name__ == "__main__":
    main()
