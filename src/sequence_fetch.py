"""Fetch ±FLANK_SIZE bp around each variant from Ensembl REST (GRCh38).

Output schema (parquet):
    chrom    str    e.g. '1', '17', 'X'
    pos      int64  1-based variant position
    ref      str
    alt      str
    ref_flank str   2*FLANK_SIZE+1 nt centred on the variant; centre is `ref`
    alt_flank str   same window with the centre base substituted to `alt`
    ok        bool  False if the fetched window doesn't match the reported ref base

Usage:
    python -m src.sequence_fetch                    # full dataset, with cache
    python -m src.sequence_fetch --limit 200        # demo subset
    python -m src.sequence_fetch --refresh          # re-fetch ignoring cache
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from .config import (
    ENSEMBL_ASSEMBLY,
    ENSEMBL_REST,
    FLANK_SIZE,
    FLANKS_PARQUET,
    RAW_CSV,
)
from .utils import get_logger

LOG = get_logger("seqfetch")

# Ensembl rate limit: 15 req/sec. Batch endpoint accepts up to 50 regions/POST.
BATCH_SIZE = 50
INTER_BATCH_SLEEP = 0.07  # ~14 req/sec, well under the limit


def _normalize_chrom(chrom: str) -> str:
    """Ensembl wants '1', 'X', 'MT' -- not 'chr1'."""
    c = str(chrom).strip()
    if c.lower().startswith("chr"):
        c = c[3:]
    if c == "M":
        c = "MT"
    return c


def _build_regions(df: pd.DataFrame, flank: int) -> list[str]:
    out = []
    for _, row in df.iterrows():
        chrom = _normalize_chrom(row["chrom"])
        start = int(row["pos"]) - flank
        end = int(row["pos"]) + flank
        out.append(f"{chrom}:{start}-{end}:1")  # :1 = + strand
    return out


def _post_batch(session: requests.Session, regions: list[str]) -> dict[str, str]:
    url = f"{ENSEMBL_REST}/sequence/region/human"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    payload = {"regions": regions, "coord_system_version": ENSEMBL_ASSEMBLY}
    for attempt in range(5):
        try:
            r = session.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        except requests.RequestException as e:
            LOG.warning("Network error on batch (attempt %d): %s", attempt + 1, e)
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", "1"))
            LOG.warning("429; sleeping %.1fs", wait)
            time.sleep(wait + 0.5)
            continue
        if r.status_code != 200:
            LOG.warning("HTTP %s on batch (attempt %d); body=%s",
                        r.status_code, attempt + 1, r.text[:200])
            time.sleep(2 ** attempt)
            continue
        # Ensembl returns a 'query' field that matches the requested region;
        # the 'id' is a different (full-coord) format. Always key by 'query'.
        out = {item.get("query", item["id"]): item["seq"].upper() for item in r.json()}
        return out
    raise RuntimeError("Ensembl batch failed after 5 attempts")


def _apply_alt(ref_flank: str, alt_base: str, flank: int) -> str:
    if len(ref_flank) != 2 * flank + 1:
        return ref_flank  # malformed; caller will mark ok=False
    return ref_flank[:flank] + alt_base.upper() + ref_flank[flank + 1:]


def fetch_flanks(df: pd.DataFrame, flank: int = FLANK_SIZE,
                 batch_size: int = BATCH_SIZE) -> pd.DataFrame:
    """Fetch ref/alt flanks for the given variants. Pure -- no caching here."""
    df = df.reset_index(drop=True)
    regions = _build_regions(df, flank)

    seq_map: dict[str, str] = {}
    session = requests.Session()
    n_batches = (len(regions) + batch_size - 1) // batch_size
    for i in range(0, len(regions), batch_size):
        batch = regions[i:i + batch_size]
        seq_map.update(_post_batch(session, batch))
        if (i // batch_size) % 20 == 0:
            LOG.info("  fetched %d/%d batches  (%d sequences cached)",
                     i // batch_size, n_batches, len(seq_map))
        time.sleep(INTER_BATCH_SLEEP)

    rows = []
    for region, (_, row) in zip(regions, df.iterrows()):
        ref_flank = seq_map.get(region, "")
        ref_base = str(row["ref"]).upper()
        alt_base = str(row["alt"]).upper()
        ok = (len(ref_flank) == 2 * flank + 1 and
              ref_flank[flank:flank + 1] == ref_base)
        alt_flank = _apply_alt(ref_flank, alt_base, flank) if ok else ""
        rows.append({
            "chrom": _normalize_chrom(row["chrom"]),
            "pos": int(row["pos"]),
            "ref": ref_base,
            "alt": alt_base,
            "ref_flank": ref_flank,
            "alt_flank": alt_flank,
            "ok": bool(ok),
        })
    return pd.DataFrame(rows)


def _read_variants(path: Path = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path,
                     usecols=["base__chrom", "base__pos", "base__ref_base", "base__alt_base"],
                     low_memory=False)
    df.columns = ["chrom", "pos", "ref", "alt"]
    return df


def run(limit: int | None = None, refresh: bool = False) -> pd.DataFrame:
    df = _read_variants()
    if limit:
        df = df.head(limit).copy()
    LOG.info("Variants to fetch: %d  (flank=%d, ±%d bp)", len(df), FLANK_SIZE, FLANK_SIZE)

    if FLANKS_PARQUET.exists() and not refresh:
        cached = pd.read_parquet(FLANKS_PARQUET)
        LOG.info("Cache: %d rows in %s", len(cached), FLANKS_PARQUET)
        # Determine missing rows by (chrom, pos, ref, alt)
        key_cols = ["chrom", "pos", "ref", "alt"]
        df_norm = df.copy()
        df_norm["chrom"] = df_norm["chrom"].map(_normalize_chrom)
        df_norm["ref"] = df_norm["ref"].str.upper()
        df_norm["alt"] = df_norm["alt"].str.upper()
        merged = df_norm.merge(cached, on=key_cols, how="left", indicator=True)
        missing = df_norm[merged["_merge"] == "left_only"]
        LOG.info("Missing: %d rows; will fetch.", len(missing))
        if len(missing) == 0:
            return cached
        new = fetch_flanks(missing)
        out = pd.concat([cached, new], ignore_index=True).drop_duplicates(key_cols)
    else:
        out = fetch_flanks(df)

    out.to_parquet(FLANKS_PARQUET, index=False)
    LOG.info("Saved %d flanks to %s  (ok=%d, fail=%d)",
             len(out), FLANKS_PARQUET, int(out["ok"].sum()), int((~out["ok"]).sum()))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="Only fetch first N variants (for demo/testing).")
    p.add_argument("--refresh", action="store_true",
                   help="Ignore cache and re-fetch.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(limit=args.limit, refresh=args.refresh)
