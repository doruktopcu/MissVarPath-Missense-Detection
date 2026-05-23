"""Compile the Phase A feature-ablation results into reportable tables.

For each (task, model, group) tuple, computes:
  delta = baseline_holdout_macro_f1 - ablated_holdout_macro_f1

Positive delta = removing the group hurt the model (group was useful).
Negative delta = removing the group helped (group was redundant/noisy).
|delta| < 0.005 ≈ noise floor.

Outputs (under outputs/reports/feature_ablation/):
  - master.csv              long-format: task, model, group, baseline_f1,
                            ablated_f1, delta, n_features_dropped
  - {task}_pivot.csv        per-task wide: rows=group, cols=model, values=delta
  - headline_histgb.csv     compact: rows=group, cols=task, values=HistGB delta
                            (the headline table for the report)
  - waste_groups.txt        groups whose HistGB delta < 0.005 on every task
                            (candidates for Phase B per-feature drill-in)

Run: python -m scripts.summarize_feature_ablation
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Mirror the groups defined in run_feature_ablation.py so we don't duplicate.
from scripts.run_feature_ablation import GROUPS, TASKS

# Baselines on disk for each task (full 10-model + AdaBoost).
BASELINE_DIRS = {
    "4class": "4class_final_no_adaboost",
    "2class": "2class_final_no_adaboost",
    "3class": "3class_vus",
    "5class": "5class_vus",
}
OUT_DIR = Path("outputs/reports/feature_ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_lb(report_dir: str) -> pd.DataFrame:
    return pd.read_csv(f"outputs/reports/{report_dir}/leaderboard.csv")


def _count_dropped(task: str, group: str) -> int:
    """How many features got dropped — read from the ablated dir's features.txt."""
    bp = Path(f"outputs/reports/{BASELINE_DIRS[task]}/features.txt")
    ap = Path(f"outputs/reports/{task}_ablate_{group}/features.txt")
    if not (bp.exists() and ap.exists()):
        return -1
    return len(bp.read_text().splitlines()) - len(ap.read_text().splitlines())


def main() -> None:
    rows = []
    missing = []
    for task in TASKS:
        base = _load_lb(BASELINE_DIRS[task])[["model", "holdout_macro_f1"]].rename(
            columns={"holdout_macro_f1": "baseline_f1"})
        for group, _prefixes in GROUPS:
            ab_dir = f"{task}_ablate_{group}"
            ab_path = Path(f"outputs/reports/{ab_dir}/leaderboard.csv")
            if not ab_path.exists():
                missing.append(f"{task}/{group}")
                continue
            ab = pd.read_csv(ab_path)[["model", "holdout_macro_f1"]].rename(
                columns={"holdout_macro_f1": "ablated_f1"})
            merged = base.merge(ab, on="model", how="inner")
            merged["task"] = task
            merged["group"] = group
            merged["n_dropped"] = _count_dropped(task, group)
            merged["delta"] = merged["baseline_f1"] - merged["ablated_f1"]
            rows.append(merged[["task", "model", "group", "n_dropped",
                                "baseline_f1", "ablated_f1", "delta"]])

    if missing:
        print(f"[summarize] missing leaderboards (skipped): {len(missing)}")
        for m in missing[:10]:
            print(f"  - {m}")

    master = pd.concat(rows, ignore_index=True)
    master_path = OUT_DIR / "master.csv"
    master.to_csv(master_path, index=False)
    print(f"[summarize] wrote {master_path}  ({len(master)} rows)")

    # Per-task wide pivot: rows=group, cols=model, values=delta
    for task in TASKS:
        sub = master[master["task"] == task]
        pivot = sub.pivot(index="group", columns="model", values="delta")
        # Add HistGB-only column ordering for readability + row mean
        model_order = [
            "HistGradientBoosting", "AdaBoost", "LinearSVC", "DecisionTree",
            "SGDClassifier", "LDA", "QDA", "RidgeClassifier", "KNN",
            "NearestCentroid", "CosineSimilarity",
        ]
        cols = [m for m in model_order if m in pivot.columns]
        pivot = pivot[cols]
        pivot["mean_delta"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("mean_delta", ascending=False)
        path = OUT_DIR / f"{task}_pivot.csv"
        pivot.to_csv(path)
        print(f"[summarize] wrote {path}")

    # Headline table: HistGB delta per group × task
    head = master[master["model"] == "HistGradientBoosting"].pivot(
        index="group", columns="task", values="delta")
    head = head[[t for t in TASKS if t in head.columns]]
    head["max_abs"] = head.abs().max(axis=1)
    head = head.sort_values("max_abs", ascending=False)
    head_path = OUT_DIR / "headline_histgb.csv"
    head.to_csv(head_path)
    print(f"[summarize] wrote {head_path}")
    print("\n=== Headline (HistGB delta = baseline - ablated; positive = group was useful) ===")
    print(head.round(4).to_string())

    # Waste groups: HistGB |delta| < 0.005 on every task (i.e., removal never
    # moves macro-F1 above noise floor). Candidates for Phase B drill-in.
    waste = head[head["max_abs"] < 0.005].index.tolist()
    (OUT_DIR / "waste_groups.txt").write_text("\n".join(waste) + ("\n" if waste else ""))
    print(f"\n=== Waste groups (HistGB delta < 0.005 on every task) ===")
    for g in waste:
        print(f"  - {g}")


if __name__ == "__main__":
    main()
