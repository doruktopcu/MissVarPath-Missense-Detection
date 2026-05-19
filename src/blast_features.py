"""BLAST-derived "same-locus neighbour-label" features.

Honest framing — please read before interpreting these features:

This script runs ``blastn`` with ``word_size=7``, ``evalue=10`` on the 51 bp
flanks fetched in ``src.sequence_fetch`` (±25 bp around each variant). The
database is built from the *training* set's ref-flanks; each variant's
alt-flank is queried against it, the top-K hits are pulled, and their
training-set labels are aggregated into features (counts of pathogenic-tier
vs. benign-tier hits, mean bit score, nearest-hit label, etc.).

This is *not* a homology-based feature engineering step in the biologically
meaningful sense of the word, for two structural reasons:

  1. **The query substrate is too short and too narrow.** 51 bp DNA windows
     with ``word_size=7`` and ``evalue=10`` are well below the regime where
     BLAST produces statistically meaningful homology calls. The search
     space is tiny and the parameters are permissive; a "hit" largely
     reflects local 7-mer matching, not orthology or paralogy.

  2. **The hits are dominated by same-locus / same-gene proximity, not
     biology.** ClinVar has many variants per gene. Two variants 10 bp apart
     share ~80% of their flank sequence by construction, so they will trivially
     be each other's top BLAST hits regardless of any biological similarity.
     The ``blast_n_pathogenic`` / ``blast_n_benign`` features therefore behave
     mostly as a *neighbour-label proxy for the surrounding locus* — useful
     signal, but not "homology" in the usual sense. Under gene-stratified CV
     (``--cv-mode gene``) this signal should largely collapse, which is the
     empirical test for what the features actually encode.

A *properly* homology-driven version of this idea would need substantially
more infrastructure than fits this course project:

  - **A large protein reference database** — e.g., UniRef50/90 or a curated
    pathogenic/benign variant corpus — millions of sequences rather than the
    ~17.5k train-set flanks we use here.
  - **Protein-level BLAST (``blastp``)** on a window of amino acids around the
    variant (with the substitution applied), so that cross-gene paralog hits
    can contribute, instead of DNA flanks where the only neighbours that match
    are positionally adjacent ClinVar entries.
  - **Tighter cutoffs** (e.g., ``evalue ≤ 1e-3``, ``pident ≥ 90``) and length
    filters so only meaningful hits feed the feature aggregator.

We chose the DNA-flank / train-DB formulation deliberately because the
project does not assume access to a UniRef-scale reference DB, and because
even this proxy is informative as long as it is reported honestly. The
report frames these as **label-aware locus-neighbour features**, not as
homology features, and the no-VEP / gene-stratified ablations are designed
to quantify how much of the lift survives once same-gene leakage is removed.

Implementation notes (kept from earlier revisions):

  - The BLAST DB is built only from training-set variants (stratified 80/20
    split with ``RANDOM_STATE``=42, matched to ``src.train``). Test-set
    variants are queried against it but never indexed.
  - Self-hits are excluded via the ``qseqid != sseqid`` filter in
    ``_aggregate_hits``, but a train variant queried during k-fold CV would
    still see *other in-fold* train variants in the DB. BLAST features are
    therefore clean for the held-out test rows but mildly optimistic on
    training rows; the augmented leaderboard reads them under the same
    80/20 holdout used for training. OOF BLAST features for the 5-fold CV
    would require rebuilding the DB inside each fold and are not generated
    by this script.

Feature schema (per variant, top-K=10 hits):
    blast_n_hits             - number of BLAST hits (>=1 e-value threshold)
    blast_top_bit            - bit score of the best hit
    blast_top_pident         - % identity of the best hit
    blast_n_pathogenic       - count of top-K hits whose label is pathogenic-tier (target_4 in {2,3})
    blast_n_benign           - count of top-K hits whose label is benign-tier (target_4 in {0,1})
    blast_p_pathogenic_top1  - 1 if top hit is pathogenic-tier, 0 otherwise
    blast_mean_bit_path      - mean bit score over pathogenic-tier hits among top K
    blast_mean_bit_benign    - mean bit score over benign-tier hits among top K
    blast_target_4_top1      - target_4 label of the nearest hit (-1 if no hit)
    blast_target_2_top1      - target_2 label of the nearest hit (-1 if no hit)

Run as:
    python -m src.blast_features
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    BLAST_FEATURES_PARQUET,
    FLANKS_PARQUET,
    PROCESSED_PARQUET,
    RANDOM_STATE,
    RAW_CSV,
    SEQUENCES_DIR,
    TEST_SIZE,
)
from .utils import get_logger

LOG = get_logger("blastfeat")

TOP_K = 10
EVALUE = 10.0  # permissive; flanks are short
WORD_SIZE = 7  # short-read friendly default for short queries


def _check_blast_tools() -> None:
    for tool in ("makeblastdb", "blastn"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"`{tool}` not found on PATH; install BLAST+")


def _write_fasta(df: pd.DataFrame, seq_col: str, out_path: Path) -> None:
    with out_path.open("w") as f:
        for _, r in df.iterrows():
            f.write(f">{r['variant_id']}\n{r[seq_col]}\n")


def _make_db(fasta: Path, db_prefix: Path) -> None:
    cmd = ["makeblastdb", "-in", str(fasta), "-dbtype", "nucl",
           "-out", str(db_prefix), "-parse_seqids"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"makeblastdb failed (rc={res.returncode})\n"
            f"stdout: {res.stdout}\nstderr: {res.stderr}")


def _blast(query_fasta: Path, db_prefix: Path, out_tsv: Path,
           top_k: int = TOP_K) -> None:
    cmd = [
        "blastn",
        "-query", str(query_fasta),
        "-db", str(db_prefix),
        "-out", str(out_tsv),
        "-outfmt", "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore",
        "-evalue", str(EVALUE),
        "-word_size", str(WORD_SIZE),
        "-max_target_seqs", str(top_k + 1),  # +1 because the variant may self-hit
        "-num_threads", "4",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"blastn failed (rc={res.returncode})\n"
            f"stdout: {res.stdout}\nstderr: {res.stderr}")


def _aggregate_hits(hits: pd.DataFrame, train_labels: pd.DataFrame,
                    top_k: int) -> pd.DataFrame:
    """For each query, pick top_k hits ranked by bitscore desc and aggregate."""
    hits = hits.merge(
        train_labels[["variant_id", "target_4", "target_2"]].rename(
            columns={"variant_id": "sseqid"}),
        on="sseqid", how="left",
    )
    # Drop self-hits (qseqid == sseqid) -- they only matter for train rows
    # querying themselves. For test rows the variant id won't be in the DB.
    hits = hits[hits["qseqid"] != hits["sseqid"]].copy()

    hits.sort_values(["qseqid", "bitscore"], ascending=[True, False], inplace=True)
    hits["rank"] = hits.groupby("qseqid").cumcount() + 1
    top = hits[hits["rank"] <= top_k]

    rows = []
    for qid, grp in top.groupby("qseqid"):
        path_mask = grp["target_4"].isin([2, 3])
        ben_mask = grp["target_4"].isin([0, 1])
        rec = {
            "variant_id": qid,
            "blast_n_hits": int(len(grp)),
            "blast_top_bit": float(grp.iloc[0]["bitscore"]),
            "blast_top_pident": float(grp.iloc[0]["pident"]),
            "blast_n_pathogenic": int(path_mask.sum()),
            "blast_n_benign": int(ben_mask.sum()),
            "blast_p_pathogenic_top1": int(path_mask.iloc[0]) if len(grp) else 0,
            "blast_mean_bit_path": float(grp.loc[path_mask, "bitscore"].mean())
                                    if path_mask.any() else 0.0,
            "blast_mean_bit_benign": float(grp.loc[ben_mask, "bitscore"].mean())
                                      if ben_mask.any() else 0.0,
            "blast_target_4_top1": int(grp.iloc[0]["target_4"])
                                    if pd.notna(grp.iloc[0]["target_4"]) else -1,
            "blast_target_2_top1": int(grp.iloc[0]["target_2"])
                                    if pd.notna(grp.iloc[0]["target_2"]) else -1,
        }
        rows.append(rec)
    return pd.DataFrame(rows)


def run(top_k: int = TOP_K) -> pd.DataFrame:
    _check_blast_tools()
    flanks = pd.read_parquet(FLANKS_PARQUET)
    flanks = flanks[flanks["ok"]].copy()
    flanks["variant_id"] = (flanks["chrom"].astype(str) + "_" +
                            flanks["pos"].astype(str) + "_" +
                            flanks["ref"] + "_" + flanks["alt"])
    n_before = len(flanks)
    flanks = flanks.drop_duplicates(subset="variant_id", keep="first").reset_index(drop=True)
    if n_before != len(flanks):
        LOG.info("Deduplicated flanks: %d -> %d", n_before, len(flanks))

    # Pull labels from the processed parquet. Need to recreate variant_id from
    # the *raw* CSV chrom/pos/ref/alt -- those columns were dropped during
    # preprocessing, so reload them here.
    raw = pd.read_csv(
        RAW_CSV,
        usecols=["base__chrom", "base__pos", "base__ref_base", "base__alt_base",
                 "clinvar__sig"],
        low_memory=False,
    )
    raw.columns = ["chrom", "pos", "ref", "alt", "label"]
    raw["chrom"] = raw["chrom"].str.replace("^chr", "", regex=True)
    raw["chrom"] = raw["chrom"].replace({"M": "MT"})
    raw["ref"] = raw["ref"].str.upper()
    raw["alt"] = raw["alt"].str.upper()
    raw["variant_id"] = (raw["chrom"].astype(str) + "_" +
                         raw["pos"].astype(str) + "_" +
                         raw["ref"] + "_" + raw["alt"])
    label_to_4 = {"Benign": 0, "Likely benign": 1,
                  "Likely pathogenic": 2, "Pathogenic": 3}
    label_to_2 = {"Benign": 0, "Likely benign": 0,
                  "Likely pathogenic": 1, "Pathogenic": 1}
    raw["target_4"] = raw["label"].map(label_to_4)
    raw["target_2"] = raw["label"].map(label_to_2)
    raw = raw.drop_duplicates(subset="variant_id", keep="first").reset_index(drop=True)

    # Stratified 80/20 split — same RANDOM_STATE as src.train so the BLAST
    # train DB stays consistent with downstream model training.
    idx_tr, idx_te = train_test_split(
        np.arange(len(raw)),
        test_size=TEST_SIZE,
        stratify=raw["target_4"],
        random_state=RANDOM_STATE,
    )
    raw_tr = raw.iloc[idx_tr][["variant_id", "target_4", "target_2"]].copy()
    LOG.info("Train DB candidates: %d  Query (all): %d", len(raw_tr), len(raw))

    flanks_with_label = flanks.merge(
        raw[["variant_id", "target_4", "target_2"]], on="variant_id", how="inner")
    train_flanks = flanks_with_label[flanks_with_label["variant_id"].isin(raw_tr["variant_id"])]
    LOG.info("Train flanks indexed in BLAST DB: %d", len(train_flanks))

    with tempfile.TemporaryDirectory(dir=str(SEQUENCES_DIR)) as tmpdir:
        tmp = Path(tmpdir)
        train_fa = tmp / "train.fa"
        query_fa = tmp / "query.fa"
        db_prefix = tmp / "train_db"
        out_tsv = tmp / "blast.tsv"

        _write_fasta(train_flanks, "ref_flank", train_fa)
        _write_fasta(flanks_with_label, "alt_flank", query_fa)

        LOG.info("makeblastdb...")
        _make_db(train_fa, db_prefix)
        LOG.info("blastn...")
        _blast(query_fa, db_prefix, out_tsv)

        LOG.info("Reading hits...")
        cols = ["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
                "qstart", "qend", "sstart", "send", "evalue", "bitscore"]
        hits = pd.read_csv(out_tsv, sep="\t", header=None, names=cols)
        LOG.info("Total hits: %d  Unique queries with hits: %d",
                 len(hits), hits["qseqid"].nunique())

    feat = _aggregate_hits(hits, raw_tr, top_k=top_k)

    # Fill variants with no hits with neutral defaults so they merge cleanly
    all_query_ids = flanks_with_label["variant_id"].drop_duplicates()
    feat = (pd.DataFrame({"variant_id": all_query_ids})
            .merge(feat, on="variant_id", how="left"))
    fill_zero = ["blast_n_hits", "blast_top_bit", "blast_top_pident",
                 "blast_n_pathogenic", "blast_n_benign",
                 "blast_p_pathogenic_top1",
                 "blast_mean_bit_path", "blast_mean_bit_benign"]
    feat[fill_zero] = feat[fill_zero].fillna(0)
    feat["blast_target_4_top1"] = feat["blast_target_4_top1"].fillna(-1).astype(int)
    feat["blast_target_2_top1"] = feat["blast_target_2_top1"].fillna(-1).astype(int)

    feat.to_parquet(BLAST_FEATURES_PARQUET, index=False)
    LOG.info("Saved %s  shape=%s", BLAST_FEATURES_PARQUET, feat.shape)
    return feat


if __name__ == "__main__":
    run()
