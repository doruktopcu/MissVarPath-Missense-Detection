"""Join the existing tabular preprocessed parquet with k-mer + BLAST features
keyed by (chrom, pos, ref, alt). Produces ``outputs/preprocessing/missense_augmented.parquet``.

Variants with failed flank fetches are dropped (the BLAST/k-mer columns would
be all-NaN for them; we'd rather train on a slightly smaller, fully-featured
dataset than carry sentinel rows).

Run as:
    python -m src.build_augmented_dataset
"""
from __future__ import annotations

import pandas as pd

from .config import (
    AUGMENTED_PARQUET,
    BLAST_FEATURES_PARQUET,
    KMER_PARQUET,
    PROCESSED_PARQUET,
    RAW_CSV,
)
from .utils import get_logger

LOG = get_logger("augment")


def _load_raw_keys() -> pd.DataFrame:
    raw = pd.read_csv(RAW_CSV,
                      usecols=["base__chrom", "base__pos", "base__ref_base",
                               "base__alt_base"],
                      low_memory=False)
    raw.columns = ["chrom", "pos", "ref", "alt"]
    raw["chrom"] = raw["chrom"].str.replace("^chr", "", regex=True).replace({"M": "MT"})
    raw["ref"] = raw["ref"].str.upper()
    raw["alt"] = raw["alt"].str.upper()
    raw["variant_id"] = (raw["chrom"].astype(str) + "_" +
                         raw["pos"].astype(str) + "_" +
                         raw["ref"] + "_" + raw["alt"])
    return raw


def run() -> pd.DataFrame:
    base = pd.read_parquet(PROCESSED_PARQUET)
    LOG.info("Base processed parquet: %s", base.shape)

    raw_keys = _load_raw_keys()
    if len(raw_keys) != len(base):
        raise RuntimeError(
            f"Row-count mismatch: raw={len(raw_keys)} processed={len(base)}; "
            "preprocessing must drop labels in the same order as the raw CSV.")
    base = pd.concat([raw_keys.reset_index(drop=True),
                      base.reset_index(drop=True)], axis=1)

    kmer = pd.read_parquet(KMER_PARQUET)
    blast = pd.read_parquet(BLAST_FEATURES_PARQUET)
    kmer = kmer.drop_duplicates(subset=["chrom", "pos", "ref", "alt"], keep="first")
    blast = blast.drop_duplicates(subset="variant_id", keep="first")
    LOG.info("k-mer features: %s   BLAST features: %s", kmer.shape, blast.shape)

    # k-mer is keyed by (chrom, pos, ref, alt); BLAST by variant_id.
    out = base.merge(kmer, on=["chrom", "pos", "ref", "alt"], how="left")
    out = out.merge(blast, on="variant_id", how="left")

    before = len(out)
    out = out.dropna(subset=["entropy_ref", "blast_n_hits"]).reset_index(drop=True)
    LOG.info("Dropped %d rows missing flank/BLAST features. Final: %s",
             before - len(out), out.shape)

    # Drop the join keys before saving — they're identifier-like and shouldn't
    # be features. Keep target columns and label.
    drop_keys = ["chrom", "pos", "ref", "alt", "variant_id"]
    out = out.drop(columns=drop_keys)

    out.to_parquet(AUGMENTED_PARQUET, index=False)
    LOG.info("Saved augmented parquet: %s  shape=%s", AUGMENTED_PARQUET, out.shape)
    return out


if __name__ == "__main__":
    run()
