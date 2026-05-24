"""Materialise CSV versions of the LaTeX tables in final_report.tex.

Each output lives under outputs/reports/tables_csv/ with a name that mirrors
the LaTeX label (e.g. tab:gene-strat → gene_strat.csv). The intent is that
anyone can open the spreadsheet of every aggregated table in the report
without having to re-extract them from the .tex source.

Run: python -m scripts.build_report_table_csvs
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path("outputs/reports/tables_csv")
OUT.mkdir(parents=True, exist_ok=True)
REPORTS = Path("outputs/reports")


def _read(setting: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS / setting / "leaderboard.csv")


# ----- tab:gene-strat: 4-class & 2-class canonical vs. gene-CV -----
def gene_strat() -> None:
    rows = []
    for task, base, gene in [("4-class", "4class_final_no_adaboost", "4class_gene"),
                             ("2-class", "2class_final_no_adaboost", "2class_gene")]:
        b = _read(base)[["model", "holdout_macro_f1"]].rename(
            columns={"holdout_macro_f1": f"{task}_canonical"})
        g = _read(gene)[["model", "holdout_macro_f1"]].rename(
            columns={"holdout_macro_f1": f"{task}_gene_cv"})
        m = b.merge(g, on="model")
        m[f"{task}_delta"] = m[f"{task}_gene_cv"] - m[f"{task}_canonical"]
        rows.append(m)
    out = rows[0].merge(rows[1], on="model")
    out.to_csv(OUT / "gene_strat.csv", index=False)


# ----- tab:no-vep: 4-class no-VEP kfold + gene-CV -----
def no_vep() -> None:
    kf = _read("4class_augmented_no_vep")[["model", "holdout_macro_f1"]].rename(
        columns={"holdout_macro_f1": "no_vep_kfold"})
    gn = _read("4class_augmented_gene_no_vep")[["model", "holdout_macro_f1"]].rename(
        columns={"holdout_macro_f1": "no_vep_gene_cv"})
    out = kf.merge(gn, on="model", how="outer")
    out.to_csv(OUT / "no_vep.csv", index=False)


# ----- tab:raw-only: augmented vs raw-only on 4-class & 2-class -----
def raw_only() -> None:
    rows = []
    for task, aug, raw in [
        ("4-class", "4class_augmented", "4class_augmented_raw_only"),
        ("2-class", "2class_augmented", "2class_augmented_raw_only"),
    ]:
        a = _read(aug)[["model", "holdout_macro_f1"]].rename(
            columns={"holdout_macro_f1": f"{task}_augmented"})
        r = _read(raw)[["model", "holdout_macro_f1"]].rename(
            columns={"holdout_macro_f1": f"{task}_raw_only"})
        m = a.merge(r, on="model", how="outer")
        m[f"{task}_delta"] = m[f"{task}_raw_only"] - m[f"{task}_augmented"]
        rows.append(m)
    out = rows[0].merge(rows[1], on="model", how="outer")
    out.to_csv(OUT / "raw_only.csv", index=False)


# ----- tab:vus-class: VUS-as-class headlines -----
def vus_class() -> None:
    metrics = ["holdout_acc", "holdout_macro_f1", "holdout_mcc"]
    out = []
    for label, dirname in [("2-class no-VUS", "2class_final_no_adaboost"),
                           ("3-class with VUS", "3class_vus"),
                           ("4-class no-VUS", "4class_final_no_adaboost"),
                           ("5-class with VUS", "5class_vus")]:
        df = _read(dirname)
        hgb = df[df["model"] == "HistGradientBoosting"][metrics].iloc[0]
        out.append({"study": label, **hgb.to_dict()})
    pd.DataFrame(out).to_csv(OUT / "vus_class.csv", index=False)


# ----- tab:ablation-headline: HistGB group-LOO deltas -----
def ablation_headline() -> None:
    src = REPORTS / "feature_ablation" / "headline_histgb.csv"
    if src.exists():
        # Copy with explicit task ordering
        df = pd.read_csv(src, index_col=0)
        cols = ["4class", "2class", "5class", "3class", "max_abs"]
        df = df[[c for c in cols if c in df.columns]]
        df.to_csv(OUT / "ablation_headline.csv")


# ----- tab:ablation-perfeat: per-feature permutation importance -----
def ablation_per_feature() -> None:
    src = REPORTS / "feature_ablation" / "per_feature_waste.csv"
    if src.exists():
        df = pd.read_csv(src)
        # Pivot to wide: rows=feature (with group prefix), cols=task, values=importance
        wide = df.pivot_table(index=["group", "feature"], columns="task",
                              values="importance_mean")
        wide["max_abs"] = wide.abs().max(axis=1)
        wide = wide.sort_values("max_abs", ascending=False)
        # Keep only those that cross the 0.001 floor on at least one task
        wide = wide[wide["max_abs"] > 0.001]
        wide.to_csv(OUT / "ablation_per_feature.csv")


# ----- tab:lean-headline: Lean A / Lean B HistGB headlines -----
def lean_headline() -> None:
    out = []
    for task, base in [("4-class", "4class_final_no_adaboost"),
                       ("2-class", "2class_final_no_adaboost"),
                       ("3-class", "3class_vus"),
                       ("5-class", "5class_vus")]:
        b = _read(base)
        la = _read(f"{task.replace('-','')}_lean")
        lb = _read(f"{task.replace('-','')}_leanB")
        h_b = b[b["model"] == "HistGradientBoosting"]["holdout_macro_f1"].iloc[0]
        h_la = la[la["model"] == "HistGradientBoosting"]["holdout_macro_f1"].iloc[0]
        h_lb = lb[lb["model"] == "HistGradientBoosting"]["holdout_macro_f1"].iloc[0]
        out.append({"task": task, "baseline": h_b, "lean_A": h_la, "lean_B": h_lb,
                    "lean_A_delta": h_la - h_b, "lean_B_delta": h_lb - h_b})
    pd.DataFrame(out).to_csv(OUT / "lean_headline.csv", index=False)


# ----- tab:lean-full: Lean B full leaderboard per task -----
def lean_full() -> None:
    rows = []
    for task in ["4class", "2class", "3class", "5class"]:
        lb = _read(f"{task}_leanB")
        lb["task"] = task
        rows.append(lb)
    df = pd.concat(rows)
    out = df.pivot(index="model", columns="task", values="holdout_macro_f1")
    out = out[["2class", "3class", "4class", "5class"]]
    out.to_csv(OUT / "lean_full.csv")


# ----- tab:anchors: anchor numbers across the report -----
def anchors() -> None:
    rows = [
        {"regime": "Canonical kfold (untuned)",          "top_model": "HistGB/AdaBoost",
         "4class_f1": 0.7928, "2class_f1": 0.9870},
        {"regime": "Final tuned no-AdaBoost (canonical)", "top_model": "HistGB",
         "4class_f1": 0.7950, "2class_f1": 0.9872},
        {"regime": "Augmented kfold (k-mer + BLAST)",     "top_model": "HistGB",
         "4class_f1": 0.7948, "2class_f1": 0.9904},
        {"regime": "Gene-stratified CV",                  "top_model": "HistGB",
         "4class_f1": 0.7658, "2class_f1": 0.9874},
        {"regime": "No-VEP kfold (augmented)",            "top_model": "HistGB",
         "4class_f1": 0.7727, "2class_f1": 0.9764},
        {"regime": "No-VEP gene-CV (augmented)",          "top_model": "HistGB",
         "4class_f1": 0.7611, "2class_f1": float("nan")},
        {"regime": "Raw-only (augmented)",                "top_model": "HistGB",
         "4class_f1": 0.7212, "2class_f1": 0.9339},
        {"regime": "DITTO removed (canonical kfold)",     "top_model": "HistGB",
         "4class_f1": 0.7911, "2class_f1": float("nan")},
        {"regime": "Lean B (113 features)",               "top_model": "HistGB",
         "4class_f1": 0.7900, "2class_f1": 0.9863},
    ]
    rows.append({"regime": "3-class with VUS",   "top_model": "HistGB",
                 "4class_f1": 0.9459, "2class_f1": float("nan")})
    rows.append({"regime": "5-class with VUS",   "top_model": "HistGB",
                 "4class_f1": 0.7984, "2class_f1": float("nan")})
    pd.DataFrame(rows).to_csv(OUT / "anchors.csv", index=False)


# ----- tab:canonical leaderboards (4c, 2c) - already exist as leaderboard.csv,
#       but copy with explicit names too -----
def canonical_leaderboards() -> None:
    for task, dirname in [("4-class", "4class_final_no_adaboost"),
                          ("2-class", "2class_final_no_adaboost"),
                          ("3-class_vus", "3class_vus"),
                          ("5-class_vus", "5class_vus")]:
        df = _read(dirname)
        df.to_csv(OUT / f"leaderboard_{task.replace('-','')}.csv", index=False)


# ----- tab:tuning - hand-built table; no source CSV exists - skip -----


def main() -> None:
    gene_strat()
    no_vep()
    raw_only()
    vus_class()
    ablation_headline()
    ablation_per_feature()
    lean_headline()
    lean_full()
    anchors()
    canonical_leaderboards()
    for p in sorted(OUT.iterdir()):
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
