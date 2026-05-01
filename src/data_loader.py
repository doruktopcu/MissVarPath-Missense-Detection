"""Load the curated OpenCRAVAT/ClinVar CSV.

The raw CSV has 777 columns and ~22k rows; ~98 are identifier-like text columns
(transcripts, rsIDs, HGVS strings, JSON-encoded mappings) that contain commas
inside quoted fields. We always read with the standard pandas CSV parser
(c-engine, quoting respected) and let dtypes be inferred.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import LABEL_COL, RAW_CSV


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Expected label column '{LABEL_COL}' not found in {path}")
    return df


if __name__ == "__main__":
    df = load_raw()
    print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]:,} cols from {RAW_CSV}")
    print(df[LABEL_COL].value_counts(dropna=False))
