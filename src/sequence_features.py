"""K-mer feature engineering on the ±25 bp flanks fetched in src.sequence_fetch.

Produced features (per variant, 2 sets — ref and alt):
    kmer_<XYZ>           — count of k-mer XYZ in the flank window (k=3 by default)
    Plus aggregate ref-vs-alt deltas:
    kmer_diff_<XYZ>      — alt count − ref count
    gc_ref / gc_alt      — GC fraction
    entropy_ref / entropy_alt
                         — Shannon entropy over single-base distribution

The proposal explicitly calls for k-mer features with frame shift; here we use
overlapping (frame-shift-1) k-mers, which is the standard formulation.

For 51 bp windows and k=3, each variant yields:
    49 ref k-mer counts + 49 alt k-mer counts + 64 diff k-mers (max k-mer space)
    + 4 aggregates (gc_ref, gc_alt, entropy_ref, entropy_alt)

But the alphabet has only 4 letters {A,C,G,T}, so the full k-mer space is 4^k.
We materialise the full 4^k columns so that train/test splits stay dense.
"""
from __future__ import annotations

from collections import Counter
from itertools import product
from math import log2

import numpy as np
import pandas as pd

from .config import FLANKS_PARQUET, KMER_PARQUET
from .utils import get_logger

LOG = get_logger("seqfeat")

ALPHABET = ("A", "C", "G", "T")


def all_kmers(k: int) -> list[str]:
    return ["".join(p) for p in product(ALPHABET, repeat=k)]


def kmer_count(seq: str, k: int) -> Counter:
    if not seq or len(seq) < k:
        return Counter()
    # Skip k-mers that contain characters outside the alphabet (e.g. 'N').
    valid = set(ALPHABET)
    return Counter(
        seq[i:i + k]
        for i in range(len(seq) - k + 1)
        if set(seq[i:i + k]).issubset(valid)
    )


def gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    s = seq.upper()
    n = sum(1 for c in s if c in ("A", "C", "G", "T"))
    if n == 0:
        return 0.0
    return (s.count("G") + s.count("C")) / n


def shannon_entropy(seq: str) -> float:
    if not seq:
        return 0.0
    counts = Counter(c for c in seq.upper() if c in ALPHABET)
    n = sum(counts.values())
    if n == 0:
        return 0.0
    return -sum((c / n) * log2(c / n) for c in counts.values())


def featurise(flanks: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Produce a wide DataFrame of k-mer + aggregate features keyed by
    (chrom, pos, ref, alt)."""
    kmers = all_kmers(k)
    LOG.info("Computing %d-mer features over %d flanks (k-mer space size=%d).",
             k, len(flanks), len(kmers))

    rows = []
    flanks_ok = flanks[flanks["ok"]].reset_index(drop=True)
    for _, r in flanks_ok.iterrows():
        ref_seq = r["ref_flank"]
        alt_seq = r["alt_flank"]
        ref_c = kmer_count(ref_seq, k)
        alt_c = kmer_count(alt_seq, k)
        feat = {"chrom": r["chrom"], "pos": int(r["pos"]),
                "ref": r["ref"], "alt": r["alt"]}
        for km in kmers:
            feat[f"kmer_ref_{km}"] = ref_c.get(km, 0)
            feat[f"kmer_alt_{km}"] = alt_c.get(km, 0)
            feat[f"kmer_diff_{km}"] = alt_c.get(km, 0) - ref_c.get(km, 0)
        feat["gc_ref"] = gc_content(ref_seq)
        feat["gc_alt"] = gc_content(alt_seq)
        feat["entropy_ref"] = shannon_entropy(ref_seq)
        feat["entropy_alt"] = shannon_entropy(alt_seq)
        rows.append(feat)
    out = pd.DataFrame(rows)
    LOG.info("k-mer features shape: %s", out.shape)
    return out


def run(k: int = 3) -> pd.DataFrame:
    flanks = pd.read_parquet(FLANKS_PARQUET)
    feats = featurise(flanks, k=k)
    feats.to_parquet(KMER_PARQUET, index=False)
    LOG.info("Saved %s", KMER_PARQUET)
    return feats


if __name__ == "__main__":
    run()
