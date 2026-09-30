"""Filter raw MURI-IT to Swedish and save to data/processed/swedish.parquet.

Run: uv run python -m baselines.muri.filter_swedish
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "data/raw"
OUT = HERE / "data/processed/swedish.parquet"

# MURI-IT labels languages with ISO 639-3 codes ("swe"); "sv" is accepted too in case that changes.
SWEDISH_CODES = {"swe", "sv"}


def main() -> None:
    files = sorted(RAW.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No raw data in {RAW}; run baselines.muri.download first.")
    # Keep the original split, including for the fallback dump where it is only in the filename.
    df = pd.concat([pd.read_parquet(f).assign(split=f.stem) for f in files], ignore_index=True)

    swedish = df["language"].str.lower().isin(SWEDISH_CODES)
    if "language_name" in df:
        swedish |= df["language_name"].str.lower().eq("swedish")
    if not swedish.any():
        raise SystemExit(f"No Swedish rows found; languages present include {sorted(df['language'].unique())[:20]}")
    df = df[swedish].reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"Swedish rows: {len(df):,} -> {OUT}")
    print(df["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
