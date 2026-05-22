"""Curate a balanced VUS missense set from raw ClinVar TSV exports.

Inputs (project root):
    VUS_missense_expert.txt   -- 3-star ("reviewed by expert panel") missense VUS
    missense_VUS.txt          -- 2-star + 3-star missense VUS (larger pool)

Output:
    data/missense_VUS_pro_set.txt    -- OpenCRAVAT TSV-ready, target size matches
                                        the other classes (5,468 by default).
    data/missense_VUS_pro_set_2x.txt -- Optional twice-target version
                                        (10,936) if --also-2x is passed.

Filters applied:
    * Variant type = "single nucleotide variant"   (missense ⇒ SNV)
    * Has GRCh38Chromosome + GRCh38Location populated
    * Canonical SPDI parseable into 4 colon-separated fields
    * Germline review status in {3-star expert, 2-star multi-submitter no-conflict}

Composition rule:
    * Include ALL 3-star (expert) rows that survive the SNV / coordinate filters.
    * Fill remaining slots up to --target-size with a random sample of 2-star
      rows (random_state=42, matching project RANDOM_STATE).

Output schema (tab-separated, no header — matches the OpenCRAVAT TSV input):
    chrom    pos    strand    ref    alt    sample
    e.g.     chr1   925946   +     C    G    s0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42

EXPERT_STATUS = "reviewed by expert panel"          # 3-star
MULTI_NO_CONFLICT = "criteria provided, multiple submitters, no conflicts"  # 2-star


def _parse_spdi(spdi: str) -> tuple[str, str] | None:
    """Pull (ref, alt) from a Canonical SPDI like 'NC_000001.11:925945:C:G'.
    Empty string in either slot is converted to '-' (OpenCRAVAT convention)."""
    if not isinstance(spdi, str) or spdi.count(":") < 3:
        return None
    parts = spdi.split(":")
    ref = parts[2] if parts[2] != "" else "-"
    alt = parts[3] if parts[3] != "" else "-"
    return ref, alt


def _load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    keep_cols = [
        "GRCh38Chromosome", "GRCh38Location", "Canonical SPDI",
        "Variant type", "Germline review status", "VariationID",
    ]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing expected columns: {missing}")
    df = df[keep_cols].copy()

    # SNV filter (missense ⇒ SNV).
    df = df[df["Variant type"] == "single nucleotide variant"]

    # Coordinate filter.
    df = df.dropna(subset=["GRCh38Chromosome", "GRCh38Location"])
    df = df[df["GRCh38Location"].str.match(r"^\d+$")]

    # Parse SPDI for ref/alt.
    parsed = df["Canonical SPDI"].apply(_parse_spdi)
    df = df[parsed.notna()].copy()
    df["ref"] = parsed[parsed.notna()].apply(lambda x: x[0])
    df["alt"] = parsed[parsed.notna()].apply(lambda x: x[1])

    df["chrom"] = "chr" + df["GRCh38Chromosome"].astype(str)
    df["pos"] = df["GRCh38Location"].astype(int)
    df["strand"] = "+"
    df["sample"] = "s0"
    df["spdi_key"] = df["Canonical SPDI"]      # used for dedup

    return df.reset_index(drop=True)


def _to_opencravat_tsv(df: pd.DataFrame, out_path: Path) -> None:
    cols = ["chrom", "pos", "strand", "ref", "alt", "sample"]
    df[cols].to_csv(out_path, sep="\t", index=False, header=False)
    print(f"[build_vus_pro_set] Wrote {len(df):,} variants to {out_path}")


def build(target_size: int, expert_path: Path, main_path: Path,
          out_path: Path) -> pd.DataFrame:
    expert = _load_and_clean(expert_path)
    main = _load_and_clean(main_path)

    n_expert_raw = len(expert)
    n_main_raw = len(main)

    # 3-star set: everything in expert that's flagged as expert review.
    three_star = expert[expert["Germline review status"] == EXPERT_STATUS].copy()
    if len(three_star) == 0:
        # Fall back to whatever is in the expert file
        three_star = expert.copy()

    # 2-star set: from main, exclude any SPDI keys already in three_star.
    two_star_pool = main[main["Germline review status"] == MULTI_NO_CONFLICT].copy()
    two_star_pool = two_star_pool[~two_star_pool["spdi_key"].isin(three_star["spdi_key"])]

    n_3 = len(three_star)
    n_2_pool = len(two_star_pool)
    print(f"[build_vus_pro_set] SNV-filtered counts:")
    print(f"  expert file:       {n_expert_raw:,} rows  ->  3-star kept: {n_3:,}")
    print(f"  main file:         {n_main_raw:,} rows  ->  2-star pool: {n_2_pool:,}")
    print(f"  target output:     {target_size:,}")

    need_two_star = max(0, target_size - n_3)
    if need_two_star > n_2_pool:
        print(f"  [warn] 2-star pool ({n_2_pool:,}) smaller than slots to fill ({need_two_star:,}); using whole pool.")
        sampled_two_star = two_star_pool
    elif need_two_star == 0:
        print(f"  [warn] 3-star alone exceeds target — truncating expert set to {target_size:,}.")
        three_star = three_star.sample(n=target_size, random_state=RANDOM_STATE)
        sampled_two_star = two_star_pool.iloc[0:0]
    else:
        sampled_two_star = two_star_pool.sample(n=need_two_star, random_state=RANDOM_STATE)

    out = pd.concat([three_star, sampled_two_star], ignore_index=True)
    out = out.drop_duplicates(subset="spdi_key", keep="first").reset_index(drop=True)
    print(f"  composed set:      {len(out):,}  ({len(three_star):,} expert + {len(sampled_two_star):,} sampled 2-star)")

    _to_opencravat_tsv(out, out_path)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-size", type=int, default=5468,
                   help="Desired total VUS count. Default 5468 (matches other classes).")
    p.add_argument("--expert", type=Path,
                   default=PROJECT_ROOT / "VUS_missense_expert.txt")
    p.add_argument("--main", type=Path,
                   default=PROJECT_ROOT / "missense_VUS.txt")
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "data" / "missense_VUS_pro_set.txt")
    p.add_argument("--also-2x", action="store_true",
                   help="Also write a 2× target-size companion file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build(args.target_size, args.expert, args.main, args.output)
    if args.also_2x:
        out2x = args.output.with_name(args.output.stem + "_2x" + args.output.suffix)
        print()
        build(args.target_size * 2, args.expert, args.main, out2x)


if __name__ == "__main__":
    main()
