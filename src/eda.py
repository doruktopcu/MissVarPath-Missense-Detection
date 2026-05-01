"""Exploratory data analysis on the raw curated dataset.

Produces:
  - shape / dtype overview
  - label balance (4-class and 2-class collapse)
  - missingness profile (per-column and per-annotator-group)
  - numeric vs categorical column counts
  - per-feature label-conditional summary for the highest-signal annotators
    (alphamissense, cadd, revel, sift, polyphen, phylop, gnomad freq)
  - a small correlation heatmap among VEP score features

All artifacts go to outputs/eda/.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import (
    CLASS_2,
    CLASS_4_NAMES,
    EDA_DIR,
    LABEL_COL,
)
from .data_loader import load_raw
from .utils import get_logger

LOG = get_logger("eda")

# Annotators whose scores we already know are informative VEP outputs.
KEY_SCORE_PATTERNS = [
    r"^alphamissense__am_pathogenicity$",
    r"^cadd__phred$",
    r"^cadd_exome__phred$",
    r"^revel__score$",
    r"^revel__rankscore$",
    r"^sift__score$",
    r"^polyphen2__hdiv$",
    r"^polyphen2__hvar$",
    r"^phylop__phylop100_vert$",
    r"^phylop__phylop17_primate$",
    r"^phastcons__phastcons100_vert$",
    r"^gnomad4__af$",
    r"^gnomad4__af_popmax$",
    r"^metarnn__score$",
    r"^bayesdel__score$",
    r"^fathmm__score$",
    r"^mutationtaster__score$",
    r"^provean__score$",
    r"^esm1b__score$",
    r"^eve__score$",
    r"^primateai__score$",
    r"^mvp__score$",
]


def _annotator_group(col: str) -> str:
    return col.split("__")[0]


def overview(df: pd.DataFrame) -> dict:
    summary = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "n_numeric": int(df.select_dtypes(include="number").shape[1]),
        "n_object": int(df.select_dtypes(include="object").shape[1]),
        "n_bool": int(df.select_dtypes(include="bool").shape[1]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }
    return summary


def label_distribution(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    counts = df[LABEL_COL].value_counts(dropna=False).rename_axis(LABEL_COL).reset_index(name="count")
    counts.to_csv(out_dir / "label_distribution_4class.csv", index=False)

    # 2-class collapse
    df2 = df[LABEL_COL].map(lambda v: "Pathogenic_or_Likely" if CLASS_2.get(v) == 1
                            else ("Benign_or_Likely" if CLASS_2.get(v) == 0 else "Other"))
    counts2 = df2.value_counts(dropna=False).rename_axis("label_2class").reset_index(name="count")
    counts2.to_csv(out_dir / "label_distribution_2class.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.barplot(data=counts, x=LABEL_COL, y="count", ax=axes[0],
                order=CLASS_4_NAMES, color="#4C72B0")
    axes[0].set_title("4-class label balance")
    axes[0].set_xlabel("")
    plt.setp(axes[0].get_xticklabels(), rotation=20, ha="right")
    sns.barplot(data=counts2, x="label_2class", y="count", ax=axes[1], color="#55A868")
    axes[1].set_title("2-class collapse")
    axes[1].set_xlabel("")
    plt.setp(axes[1].get_xticklabels(), rotation=10, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "label_distribution.png", dpi=150)
    plt.close(fig)
    return counts


def missingness_profile(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    miss = df.isna().mean().sort_values(ascending=False).rename("missing_frac")
    out = miss.to_frame()
    out["dtype"] = df.dtypes.astype(str).reindex(out.index)
    out["annotator"] = [_annotator_group(c) for c in out.index]
    out.to_csv(out_dir / "missingness_per_column.csv", index_label="column")

    # Per-annotator average missingness
    grp = out.groupby("annotator")["missing_frac"].agg(["mean", "min", "max", "count"])
    grp = grp.sort_values("mean", ascending=False)
    grp.to_csv(out_dir / "missingness_per_annotator.csv")

    # Histogram of missingness
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(miss.values, bins=40, ax=ax, color="#C44E52")
    ax.set_xlabel("Per-column missing fraction")
    ax.set_ylabel("Number of columns")
    ax.set_title("Missingness distribution across 777 columns")
    fig.tight_layout()
    fig.savefig(out_dir / "missingness_histogram.png", dpi=150)
    plt.close(fig)

    # Top-30 most missing
    top = miss.head(30).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(top.index, top.values, color="#C44E52")
    ax.set_xlabel("Missing fraction")
    ax.set_title("Top 30 columns by missingness")
    fig.tight_layout()
    fig.savefig(out_dir / "missingness_top30.png", dpi=150)
    plt.close(fig)

    return out


def annotator_group_counts(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    groups = Counter(_annotator_group(c) for c in df.columns)
    g = (pd.Series(groups).rename("n_columns").sort_values(ascending=False)
         .rename_axis("annotator").reset_index())
    g.to_csv(out_dir / "annotator_group_counts.csv", index=False)
    return g


def key_score_summary(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    cols = [c for c in df.columns if any(re.match(p, c) for p in KEY_SCORE_PATTERNS)]
    if not cols:
        LOG.warning("No key VEP score columns matched the patterns.")
        return pd.DataFrame()
    sub = df[cols + [LABEL_COL]].copy()
    rows = []
    for c in cols:
        s = pd.to_numeric(sub[c], errors="coerce")
        rows.append({
            "feature": c,
            "missing_frac": float(s.isna().mean()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "median": float(s.median()),
            "min": float(s.min()),
            "max": float(s.max()),
        })
    summary = pd.DataFrame(rows).sort_values("feature")
    summary.to_csv(out_dir / "key_vep_score_summary.csv", index=False)

    # Box plots per class for each VEP score (one figure with subplots)
    n = len(cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
    axes = np.atleast_1d(axes).flatten()
    plot_df = sub.copy()
    plot_df[LABEL_COL] = pd.Categorical(plot_df[LABEL_COL], categories=CLASS_4_NAMES, ordered=True)
    for i, c in enumerate(cols):
        ax = axes[i]
        plot_df["_v"] = pd.to_numeric(plot_df[c], errors="coerce")
        ok = plot_df.dropna(subset=["_v"])
        if ok.empty:
            ax.axis("off")
            ax.set_title(f"{c}\n(all NaN)", fontsize=7)
            continue
        sns.boxplot(data=ok, x=LABEL_COL, y="_v", hue=LABEL_COL,
                    ax=ax, order=CLASS_4_NAMES, hue_order=CLASS_4_NAMES,
                    showfliers=False, palette="Set2", legend=False)
        ax.set_title(c, fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Key VEP scores by ClinVar class", y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "key_vep_scores_by_class.png", dpi=140)
    plt.close(fig)

    # Correlation heatmap among key scores (numeric only, using pairwise complete)
    num = sub[cols].apply(pd.to_numeric, errors="coerce")
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(min(14, 0.5 * len(cols) + 4),
                                    min(14, 0.5 * len(cols) + 4)))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax,
                xticklabels=True, yticklabels=True, cbar_kws={"shrink": 0.6})
    ax.set_title("Pearson correlation among key VEP scores")
    plt.setp(ax.get_xticklabels(), rotation=80, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "key_vep_score_correlation.png", dpi=140)
    plt.close(fig)
    return summary


def numeric_vs_text_split(df: pd.DataFrame, out_dir: Path) -> dict:
    # Many "object" columns are numeric strings or contain mixed types;
    # try coercing each object col and report what's salvageable.
    coercion = []
    for c in df.columns:
        if df[c].dtype == object:
            coerced = pd.to_numeric(df[c], errors="coerce")
            ok_frac = coerced.notna().mean()
            coercion.append((c, float(ok_frac)))
    coerc = pd.DataFrame(coercion, columns=["column", "numeric_coercion_frac"])
    coerc = coerc.sort_values("numeric_coercion_frac", ascending=False)
    coerc.to_csv(out_dir / "object_columns_numeric_coercion.csv", index=False)

    summary = {
        "n_object_cols": int((df.dtypes == object).sum()),
        "n_object_recoverable_as_numeric_>=95pct": int((coerc["numeric_coercion_frac"] >= 0.95).sum()),
        "n_object_recoverable_as_numeric_>=50pct": int((coerc["numeric_coercion_frac"] >= 0.50).sum()),
        "n_object_pure_text_<5pct": int((coerc["numeric_coercion_frac"] < 0.05).sum()),
    }
    return summary


def main() -> None:
    LOG.info("Loading data...")
    df = load_raw()

    LOG.info("Overview...")
    ov = overview(df)
    LOG.info(f"Shape: {ov['n_rows']:,} × {ov['n_cols']:,}  |  ~{ov['memory_mb']} MB")

    LOG.info("Label distribution...")
    counts = label_distribution(df, EDA_DIR)
    LOG.info("\n%s", counts.to_string(index=False))

    LOG.info("Annotator group counts...")
    annotator_group_counts(df, EDA_DIR)

    LOG.info("Missingness profile...")
    missingness_profile(df, EDA_DIR)

    LOG.info("Object-column numeric coercion analysis...")
    coerc_summary = numeric_vs_text_split(df, EDA_DIR)
    LOG.info("Object-col coercion summary: %s", coerc_summary)

    LOG.info("Key VEP score summary + class-conditioned plots...")
    key_score_summary(df, EDA_DIR)

    overview_summary = {**ov, "object_coercion": coerc_summary}
    pd.Series(overview_summary).to_json(EDA_DIR / "overview.json", indent=2)

    LOG.info("EDA artifacts written to %s", EDA_DIR)


if __name__ == "__main__":
    main()
