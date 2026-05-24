"""Phase B: per-feature permutation importance inside the waste groups.

After Phase A flags some functional groups as 'waste' (HistGB macro-F1 delta
below the 0.005 noise floor on every task), this script does a fine-grained
permutation-importance check inside each waste group, per task, to confirm
that no individual feature inside the group is pulling weight that the
group-level LOO smeared out.

Methodology (per task × per waste group):
  1. Load the trained tuned HistGB joblib + feature manifest.
  2. Recreate the held-out 20% split with the same RANDOM_STATE.
  3. Compute baseline macro-F1 once.
  4. For each feature in the waste group, shuffle its column on the held-out
     X (n_repeats=10), measure macro-F1 drop. Report mean ± std.
  5. Anything whose mean drop exceeds 0.001 macro-F1 is *not* waste — surface it.

Output: outputs/reports/feature_ablation/per_feature_waste.csv

Run: python -m scripts.waste_group_per_feature
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from scripts.run_feature_ablation import GROUPS, TASKS
from src.config import (
    CLASS_2_NAMES, CLASS_3_NAMES, CLASS_4_NAMES, CLASS_5_NAMES,
    PROCESSED_PARQUET, RANDOM_STATE, TEST_SIZE,
    VUS_3CLASS_PARQUET, VUS_5CLASS_PARQUET,
)
from src.train import META_COLS

TASK_PARQUET = {
    "4class": PROCESSED_PARQUET,
    "2class": PROCESSED_PARQUET,
    "3class": VUS_3CLASS_PARQUET,
    "5class": VUS_5CLASS_PARQUET,
}
TASK_TARGET = {"4class": "target_4", "2class": "target_2",
               "3class": "target_3", "5class": "target_5"}
TASK_BASELINE_DIR = {
    "4class": "4class_final_no_adaboost",
    "2class": "2class_final_no_adaboost",
    "3class": "3class_vus",
    "5class": "5class_vus",
}

OUT_DIR = Path("outputs/reports/feature_ablation")
WASTE_FILE = OUT_DIR / "waste_groups.txt"
OUT_CSV = OUT_DIR / "per_feature_waste.csv"

GROUP_PREFIXES = dict(GROUPS)
N_REPEATS = 10


def _features_for_group(all_feats: list[str], group: str) -> list[str]:
    prefixes = GROUP_PREFIXES[group]
    return [f for f in all_feats if any(f.startswith(p) for p in prefixes)]


def _eval_task(task: str, waste_groups: list[str]) -> list[dict]:
    parq = pd.read_parquet(TASK_PARQUET[task])
    feats = [c for c in parq.columns if c not in META_COLS]
    target = TASK_TARGET[task]
    X = parq[feats].to_numpy(dtype=np.float32)
    y = parq[target].to_numpy(dtype=np.int64)
    _, X_te, _, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    model = joblib.load(f"outputs/models/{TASK_BASELINE_DIR[task]}/histgradientboosting.joblib")

    base_f1 = f1_score(y_te, model.predict(X_te), average="macro")
    print(f"  [{task}] baseline macro-F1 = {base_f1:.4f}")

    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for g in waste_groups:
        cols_in_group = _features_for_group(feats, g)
        for f_name in cols_in_group:
            f_idx = feats.index(f_name)
            drops = []
            for _ in range(N_REPEATS):
                X_perm = X_te.copy()
                rng.shuffle(X_perm[:, f_idx])
                p_f1 = f1_score(y_te, model.predict(X_perm), average="macro")
                drops.append(base_f1 - p_f1)
            arr = np.asarray(drops)
            rows.append({
                "task": task,
                "group": g,
                "feature": f_name,
                "baseline_f1": float(base_f1),
                "importance_mean": float(arr.mean()),
                "importance_std": float(arr.std()),
            })
    return rows


def main() -> None:
    if not WASTE_FILE.exists():
        # Soft-skip: print clearly and return rather than SystemExit so the
        # subprocess.run() in the Colab notebook surfaces the message via stdout.
        print(f"[per-feature] {WASTE_FILE} not found — run "
              f"`scripts.summarize_feature_ablation` first to produce it. "
              f"Skipping Phase B.")
        return
    waste = [g for g in WASTE_FILE.read_text().splitlines() if g.strip()]
    if not waste:
        print("[per-feature] No waste groups flagged by Phase A — nothing to drill in.")
        return
    print(f"[per-feature] {len(waste)} waste groups to drill into: {waste}")

    all_rows: list[dict] = []
    for task in TASKS:
        print(f"\n[per-feature] task = {task}")
        all_rows.extend(_eval_task(task, waste))

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["task", "group", "importance_mean"], ascending=[True, True, False])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[per-feature] wrote {OUT_CSV}  ({len(df)} rows)")

    # Surface anything that beats the 0.001 noise floor — these are *not* waste.
    keepers = df[df["importance_mean"] > 0.001].sort_values(
        "importance_mean", ascending=False)
    print(f"\n=== Features inside 'waste' groups with importance > 0.001 ===")
    print(keepers[["task", "group", "feature", "importance_mean",
                   "importance_std"]].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
