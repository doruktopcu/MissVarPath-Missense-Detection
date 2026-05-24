"""Build 3-class and 5-class training parquets that include VUS as a labelled class.

Pulls VUS rows from `data/missense_VUS_pro_set_annotated.csv`, filters to strict
`clinvar__sig == "Uncertain significance"` (drops NaN and any drifted labels),
runs the same column-pruning + numeric-coercion + tidy-rename steps used by
`src/preprocessing.run()`, aligns to the 208-feature training schema with
training-set medians for any missing column, and concatenates with the existing
`outputs/preprocessing/missense_processed.parquet`.

Outputs:
  - outputs/preprocessing/missense_3class.parquet   (target_3: 0=Benign, 1=Pathogenic, 2=VUS)
  - outputs/preprocessing/missense_5class.parquet   (target_5: 0..3 as in target_4, 4=VUS)

Run: python -m scripts.build_vus_train_parquets
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PREPROC_DIR, PROCESSED_PARQUET
from src.preprocessing import (
    coerce_numeric_objects,
    drop_identifier_columns,
    drop_label_leakage,
    drop_remaining_text_columns,
    tidy_column,
)

VUS_CSV = Path("data/missense_VUS_pro_set_annotated.csv")
OUT_3CLASS = PREPROC_DIR / "missense_3class.parquet"
OUT_5CLASS = PREPROC_DIR / "missense_5class.parquet"

META_COLS = {"clinvar_sig", "target_4", "target_2", "target_3", "target_5", "gene_symbol"}


def _tidy_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {c: tidy_column(c) for c in df.columns}
    if "clinvar__sig" in df.columns:
        rename_map["clinvar__sig"] = "clinvar_sig"
    if "base__hugo" in df.columns:
        rename_map["base__hugo"] = "gene_symbol"
    return df.rename(columns=rename_map)


def _prepare_vus(features: list[str], train_medians: pd.Series) -> pd.DataFrame:
    raw = pd.read_csv(VUS_CSV, low_memory=False)
    print(f"[build] VUS raw: {len(raw)} rows × {raw.shape[1]} cols")

    # Strict-VUS filter requested by the user: keep only rows whose clinvar__sig
    # is exactly "Uncertain significance"; drop NaN and any other label.
    mask = raw["clinvar__sig"] == "Uncertain significance"
    raw = raw[mask].copy()
    print(f"[build] After strict-VUS filter: {len(raw)} rows")

    df, _ = drop_label_leakage(raw)
    df, _ = drop_identifier_columns(df)
    df, _ = coerce_numeric_objects(df)
    df, _ = drop_remaining_text_columns(df)
    df = _tidy_columns(df)

    # Align to training feature manifest; impute missing columns with training medians.
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"[build] {len(missing)} features absent from VUS; imputing with training medians.")
        for f in missing:
            df[f] = np.nan

    feat = df[features].copy().fillna(train_medians).fillna(0.0)
    gene = df["gene_symbol"] if "gene_symbol" in df.columns else pd.Series(
        ["UNKNOWN"] * len(df), index=df.index)
    out = feat.assign(
        clinvar_sig="Uncertain significance",
        gene_symbol=gene.astype(str).values,
    )
    return out.reset_index(drop=True)


def main() -> None:
    train = pd.read_parquet(PROCESSED_PARQUET)
    features = [c for c in train.columns if c not in META_COLS]
    print(f"[build] Training corpus: {len(train)} rows × {len(features)} features")
    train_medians = train[features].median(numeric_only=True)

    vus = _prepare_vus(features, train_medians)
    print(f"[build] VUS prepared: {len(vus)} rows × {len(features)} features")

    # --- 3-class parquet: collapse training Likely-* into definitive labels ---
    # 0 = Benign-side (target_4 in {0,1}), 1 = Pathogenic-side (target_4 in {2,3})
    # 2 = VUS
    train_3 = train.copy()
    train_3["target_3"] = (train_3["target_4"] >= 2).astype("int64")
    vus_3 = vus.copy()
    vus_3["target_3"] = 2
    cols_3 = features + ["clinvar_sig", "gene_symbol", "target_3"]
    df_3 = pd.concat(
        [train_3[cols_3], vus_3[cols_3]], ignore_index=True,
    )
    df_3.to_parquet(OUT_3CLASS, index=False)
    print(f"[build] Wrote {OUT_3CLASS}  shape={df_3.shape}")
    print(df_3["target_3"].value_counts().sort_index().to_string())

    # --- 5-class parquet: keep target_4 (0..3), VUS = 4 ---
    train_5 = train.copy()
    train_5["target_5"] = train_5["target_4"].astype("int64")
    vus_5 = vus.copy()
    vus_5["target_5"] = 4
    cols_5 = features + ["clinvar_sig", "gene_symbol", "target_5"]
    df_5 = pd.concat(
        [train_5[cols_5], vus_5[cols_5]], ignore_index=True,
    )
    df_5.to_parquet(OUT_5CLASS, index=False)
    print(f"[build] Wrote {OUT_5CLASS}  shape={df_5.shape}")
    print(df_5["target_5"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
