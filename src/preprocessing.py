"""Data preprocessing.

Steps:
  1. Load raw CSV.
  2. Tidy column names: lowercase, replace double underscores with single,
     strip non-alphanumeric.
  3. Encode label (`clinvar__sig`) into 4-class and 2-class integer targets.
  4. Drop columns that leak the target (everything in the `clinvar` annotator
     group except the label itself, plus any column whose name embeds
     'pathogenic' / 'benign' / 'sig' / 'rev_stat'/ etc.).
  5. Drop identifier-like columns (transcripts, rsIDs, HGVS, JSON mappings,
     PubMed IDs, disease names, free text, URLs).
  6. Coerce object columns that are >=95% numerically convertible.
  7. After coercion, drop columns that are still object-typed (free text /
     JSON-like) -- we'll do feature engineering later, per the proposal.
  8. Drop columns with >95% missingness.
  9. Fill missing values: numeric -> median; remaining categoricals -> mode.
     Boolean columns are cast to int with NA filled to 0.
 10. Drop near-zero-variance columns (single unique non-NaN value).
 11. Save processed parquet + manifests for kept/dropped columns.

Run as a module: `python -m src.preprocessing`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import (
    CLASS_2,
    CLASS_4,
    DROPPED_COLS_TXT,
    FEATURE_LIST_TXT,
    LABEL_COL,
    PREPROC_DIR,
    PROCESSED_PARQUET,
)
from .data_loader import load_raw
from .utils import get_logger

LOG = get_logger("preprocess")

# --- Patterns for columns to drop -------------------------------------------------

# Identifier / free-text / non-feature columns. These should NOT be used as
# model features; we will engineer sequence features later (k-mers, BLAST).
ID_PATTERNS = [
    r"_id$",
    r"__id$",
    r"\bid$",
    r"rsid",
    r"dbsnp",
    r"hgvs",
    r"transcript",
    r"all_mappings",
    r"note_variant",
    r"preferred_name",
    r"disease_name",
    r"disease_ref",
    r"pubmed",
    r"gene_info",
    r"uniprot_id",
    r"allele_id",
    r"somatic_disease",
    r"onc_disease",
    r"sig_conf",
    r"rev_stat",
    r"_url$",
    r"description$",
    r"diseases$",
    r"epi_id",
    r"cosmic_id",
    r"civic__",  # whole annotator is text/curation
    r"litvar",
    r"denovo__pubmed",
    r"clinvar__hgvs",
    r"clinvar__id",
    r"^clinvar__dbvar_id$",
    r"^base__cchange$",
    r"^base__achange$",
    r"^base__so$",
    r"^base__exonno$",
    r"^base__chrom$",
    r"^base__pos$",
    r"^base__ref_base$",
    r"^base__alt_base$",
    r"^base__coding$",
    r"^base__uid$",
    r"^base__gposend$",
]

# Anything from the clinvar annotator besides the label leaks the target.
LABEL_LEAK_PREFIXES = ("clinvar__", "clinvar_acmg__")
LABEL_LEAK_KEEP = {LABEL_COL}

# Object/text columns we preserve all the way through because they are
# metadata used as grouping keys (not features). Excluded from feature_cols.
PRESERVE_TEXT = {"base__hugo"}


@dataclass
class PreprocessResult:
    df: pd.DataFrame
    feature_cols: list[str]
    dropped: dict[str, list[str]] = field(default_factory=dict)


def tidy_column(c: str) -> str:
    out = c.strip().lower()
    out = out.replace("__", "_")
    out = re.sub(r"[^0-9a-z_]+", "_", out)
    out = re.sub(r"_+", "_", out)
    return out.strip("_")


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(re.search(p, name, re.IGNORECASE) for p in patterns)


def filter_label(df: pd.DataFrame) -> pd.DataFrame:
    valid = set(CLASS_4.keys())
    before = len(df)
    df = df[df[LABEL_COL].isin(valid)].copy()
    LOG.info("Filtered to valid 4-class labels: %d -> %d", before, len(df))
    return df


def drop_label_leakage(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    leak = [c for c in df.columns
            if c.startswith(LABEL_LEAK_PREFIXES) and c not in LABEL_LEAK_KEEP]
    LOG.info("Dropping %d label-leakage columns (clinvar/clinvar_acmg).", len(leak))
    return df.drop(columns=leak), leak


def drop_identifier_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    drop = [c for c in df.columns
            if c != LABEL_COL and _matches_any(c, ID_PATTERNS)]
    LOG.info("Dropping %d identifier / free-text columns.", len(drop))
    return df.drop(columns=drop), drop


def coerce_numeric_objects(df: pd.DataFrame, threshold: float = 0.95) -> tuple[pd.DataFrame, list[str]]:
    coerced = []
    for c in df.columns:
        if c == LABEL_COL or df[c].dtype != object:
            continue
        new = pd.to_numeric(df[c], errors="coerce")
        if new.notna().mean() >= threshold:
            df[c] = new
            coerced.append(c)
    LOG.info("Coerced %d object columns to numeric (>=%.0f%% parseable).",
             len(coerced), threshold * 100)
    return df, coerced


def drop_remaining_text_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    text_cols = [c for c in df.columns
                 if c != LABEL_COL and c not in PRESERVE_TEXT and df[c].dtype == object]
    LOG.info("Dropping %d remaining object-typed columns (free text / JSON-ish).",
             len(text_cols))
    return df.drop(columns=text_cols), text_cols


def drop_high_missing(df: pd.DataFrame, max_missing: float = 0.50) -> tuple[pd.DataFrame, list[str]]:
    miss = df.drop(columns=[LABEL_COL]).isna().mean()
    drop = miss[miss > max_missing].index.tolist()
    LOG.info("Dropping %d columns with >%.0f%% missing.", len(drop), max_missing * 100)
    return df.drop(columns=drop), drop


def cast_booleans(df: pd.DataFrame) -> pd.DataFrame:
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    for c in bool_cols:
        df[c] = df[c].astype("Int64")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric -> median. Any remaining non-numeric -> mode."""
    feature_df = df.drop(columns=[LABEL_COL])
    num_cols = feature_df.select_dtypes(include="number").columns.tolist()
    other_cols = [c for c in feature_df.columns if c not in num_cols]

    if num_cols:
        medians = feature_df[num_cols].median(numeric_only=True)
        feature_df[num_cols] = feature_df[num_cols].fillna(medians)
    if other_cols:
        for c in other_cols:
            mode = feature_df[c].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "missing"
            feature_df[c] = feature_df[c].fillna(fill)

    df_out = pd.concat([feature_df, df[[LABEL_COL]]], axis=1)
    LOG.info("Imputed: %d numeric columns (median), %d non-numeric (mode).",
             len(num_cols), len(other_cols))
    return df_out


def drop_zero_variance(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feats = df.drop(columns=[LABEL_COL])
    nunique = feats.nunique(dropna=False)
    drop = nunique[nunique <= 1].index.tolist()
    LOG.info("Dropping %d zero/near-zero-variance columns.", len(drop))
    return df.drop(columns=drop), drop


def add_label_codes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target_4"] = df[LABEL_COL].map(CLASS_4).astype("int64")
    df["target_2"] = df[LABEL_COL].map(CLASS_2).astype("int64")
    return df


def run() -> PreprocessResult:
    df = load_raw()
    LOG.info("Loaded raw: %s × %s", *df.shape)

    df = filter_label(df)

    df, leak = drop_label_leakage(df)
    df, ids = drop_identifier_columns(df)
    df, coerced = coerce_numeric_objects(df)
    df, text = drop_remaining_text_columns(df)
    df = cast_booleans(df)
    df, hi_miss = drop_high_missing(df)
    df = impute_missing(df)
    df, zero_var = drop_zero_variance(df)
    df = add_label_codes(df)

    # tidy column names (after all dropping is done so manifests stay readable)
    rename_map = {c: tidy_column(c) for c in df.columns}
    # avoid clobbering label/target cols if they tidy-collide
    rename_map[LABEL_COL] = "clinvar_sig"
    rename_map["target_4"] = "target_4"
    rename_map["target_2"] = "target_2"
    if "base__hugo" in df.columns:
        rename_map["base__hugo"] = "gene_symbol"
    df = df.rename(columns=rename_map)

    META_COLS = {"clinvar_sig", "target_4", "target_2", "gene_symbol"}
    feature_cols = [c for c in df.columns if c not in META_COLS]

    LOG.info("Final shape: %s × %s   (features: %d)",
             df.shape[0], df.shape[1], len(feature_cols))

    # Save artifacts
    df.to_parquet(PROCESSED_PARQUET, index=False)
    FEATURE_LIST_TXT.write_text("\n".join(feature_cols))
    DROPPED_COLS_TXT.write_text(
        "## Label-leakage (clinvar*)\n" + "\n".join(leak) +
        "\n\n## Identifier / text\n" + "\n".join(ids) +
        "\n\n## Coerced object->numeric\n" + "\n".join(coerced) +
        "\n\n## Remaining text dropped\n" + "\n".join(text) +
        "\n\n## High-missingness (>95%)\n" + "\n".join(hi_miss) +
        "\n\n## Zero-variance\n" + "\n".join(zero_var) + "\n"
    )

    summary = {
        "rows": int(df.shape[0]),
        "cols_after": int(df.shape[1]),
        "features": int(len(feature_cols)),
        "dropped_label_leakage": len(leak),
        "dropped_identifiers": len(ids),
        "coerced_object_to_numeric": len(coerced),
        "dropped_remaining_text": len(text),
        "dropped_high_missing": len(hi_miss),
        "dropped_zero_variance": len(zero_var),
    }
    pd.Series(summary).to_json(PREPROC_DIR / "summary.json", indent=2)
    LOG.info("Saved processed parquet to %s", PROCESSED_PARQUET)

    return PreprocessResult(df=df, feature_cols=feature_cols, dropped={
        "label_leakage": leak,
        "identifiers": ids,
        "coerced_numeric": coerced,
        "text_dropped": text,
        "high_missing": hi_miss,
        "zero_variance": zero_var,
    })


if __name__ == "__main__":
    run()
