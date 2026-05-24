"""Compile a grand model x setting summary CSV.

Walks every setting (canonical / tuned / regime / ablation / VUS / lean), pulls
out the per-model held-out (accuracy, macro_f1, macro_precision, macro_recall)
quadruple, and produces a wide table with rows=model, columns=setting.metric.

Where the model joblib + features manifest are available on disk, we recompute
macro_precision and macro_recall from scratch (sklearn.metrics.precision_score
/ recall_score with average='macro') by re-applying the same train/test split
and any --drop-prefixes/--keep-prefixes the setting used. Where joblibs are
not present (some legacy runs), we fall back to parsing the
classification_report.md (sklearn classification_report textual output).

Output:
  - outputs/reports/grand_summary.csv     wide: rows=model, cols=setting.metric
  - outputs/reports/grand_summary_long.csv long-format master

Run: python -m scripts.build_grand_summary
"""
from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.config import (AUGMENTED_PARQUET, PROCESSED_PARQUET, RANDOM_STATE,
                        TEST_SIZE, VUS_3CLASS_PARQUET, VUS_5CLASS_PARQUET)
from src.train import META_COLS

REPORTS = Path("outputs/reports")
MODELS = Path("outputs/models")


def _parquet_for(setting_dir: str) -> Path:
    """Pick parquet by setting name.

    `4class_no_ditto` deliberately does NOT match the augmented condition even
    though both contain `_no_`; the no-DITTO ablation was run on the base
    parquet (only `--drop-prefixes ditto_`, no `--variant augmented`).
    """
    if setting_dir.startswith("3class"):
        return VUS_3CLASS_PARQUET
    if setting_dir.startswith("5class"):
        return VUS_5CLASS_PARQUET
    if "augmented" in setting_dir:
        return AUGMENTED_PARQUET
    return PROCESSED_PARQUET


def _target_col(setting_dir: str) -> str:
    if setting_dir.startswith("4class"):
        return "target_4"
    if setting_dir.startswith("2class"):
        return "target_2"
    if setting_dir.startswith("3class"):
        return "target_3"
    if setting_dir.startswith("5class"):
        return "target_5"
    raise ValueError(setting_dir)


def _is_gene_cv(setting_dir: str) -> bool:
    return "_gene" in setting_dir


def _load_features_manifest(setting_dir: Path) -> list[str] | None:
    fpath = setting_dir / "features.txt"
    if not fpath.exists():
        return None
    feats = [l.strip() for l in fpath.read_text().splitlines() if l.strip()]
    return feats


def _split_for(setting_dir: str, parq: pd.DataFrame, features: list[str]
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = _target_col(setting_dir)
    X = parq[features].to_numpy(dtype=np.float32)
    y = parq[target].to_numpy(dtype=np.int64)
    if _is_gene_cv(setting_dir):
        groups = parq["gene_symbol"].astype(str).to_numpy()
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                random_state=RANDOM_STATE)
        tr, te = next(gss.split(X, y, groups))
    else:
        idx = np.arange(len(X))
        tr, te = train_test_split(idx, test_size=TEST_SIZE, stratify=y,
                                  random_state=RANDOM_STATE)
    return X[tr], X[te], y[tr], y[te]


_SLUG_RE = re.compile(r"[^a-z0-9]+")
def _slugify(s: str) -> str:
    return _SLUG_RE.sub("", s.lower())


def _metrics_from_joblib(setting_dir_name: str, model_name: str
                         ) -> dict | None:
    """Recompute holdout metrics from the trained joblib + features manifest."""
    setting_path = REPORTS / setting_dir_name
    feats = _load_features_manifest(setting_path)
    if feats is None:
        return None
    slug = _slugify(model_name)
    jpath = MODELS / setting_dir_name / f"{slug}.joblib"
    if not jpath.exists():
        return None
    parq = pd.read_parquet(_parquet_for(setting_dir_name))
    # Some settings dropped a target column; just use the features manifest.
    try:
        _, X_te, _, y_te = _split_for(setting_dir_name, parq, feats)
    except KeyError:
        return None
    model = joblib.load(jpath)
    try:
        y_pred = model.predict(X_te)
    except Exception:
        return None
    return {
        "accuracy": accuracy_score(y_te, y_pred),
        "macro_f1": f1_score(y_te, y_pred, average="macro"),
        "macro_precision": precision_score(y_te, y_pred, average="macro",
                                           zero_division=0),
        "macro_recall": recall_score(y_te, y_pred, average="macro",
                                     zero_division=0),
    }


def _metrics_from_classification_report(setting_dir_name: str,
                                        model_name: str) -> dict | None:
    """Fallback: read macro avg from sklearn classification_report textual file."""
    slug = _slugify(model_name)
    fpath = REPORTS / setting_dir_name / f"{slug}_classification_report.md"
    if not fpath.exists():
        return None
    text = fpath.read_text()
    # Lines look like:
    #    macro avg     0.7987    0.7997    0.7984      5460
    m_macro = re.search(
        r"macro avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+\d+", text)
    if not m_macro:
        return None
    macro_prec = float(m_macro.group(1))
    macro_rec = float(m_macro.group(2))
    macro_f1 = float(m_macro.group(3))
    m_acc = re.search(r"\baccuracy\b\s+([0-9.]+)\s+\d+", text)
    acc = float(m_acc.group(1)) if m_acc else float("nan")
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
    }


def _metrics_for(setting_dir_name: str, model_name: str) -> dict | None:
    out = _metrics_from_joblib(setting_dir_name, model_name)
    if out is not None:
        return out
    return _metrics_from_classification_report(setting_dir_name, model_name)


# --- Settings to include in the grand table -----------------------------------
# (label, on-disk dir under outputs/reports/)
SETTINGS: list[tuple[str, str]] = [
    # Canonical (untuned + tuned)
    ("4class_canonical_kfold",        "4class"),
    ("4class_tuned",                  "4class_final_no_adaboost"),
    ("2class_canonical_kfold",        "2class"),
    ("2class_tuned",                  "2class_final_no_adaboost"),
    # Gene-stratified
    ("4class_gene_cv",                "4class_gene"),
    ("2class_gene_cv",                "2class_gene"),
    # Augmented (k-mer + BLAST)
    ("4class_augmented",              "4class_augmented"),
    ("2class_augmented",              "2class_augmented"),
    # No-VEP / VUS scenario
    ("4class_no_vep_kfold",           "4class_augmented_no_vep"),
    ("2class_no_vep_kfold",           "2class_augmented_no_vep"),
    ("4class_no_vep_gene",            "4class_augmented_gene_no_vep"),
    # Raw-only
    ("4class_raw_only",               "4class_augmented_raw_only"),
    ("2class_raw_only",               "2class_augmented_raw_only"),
    # DITTO ablation
    ("4class_no_ditto",               "4class_no_ditto"),
    # VUS-as-class
    ("3class_with_vus",               "3class_vus"),
    ("5class_with_vus",               "5class_vus"),
    # Ablation-optimal lean models
    ("4class_lean_A",                 "4class_lean"),
    ("4class_lean_B",                 "4class_leanB"),
    ("2class_lean_A",                 "2class_lean"),
    ("2class_lean_B",                 "2class_leanB"),
    ("3class_lean_A",                 "3class_lean"),
    ("3class_lean_B",                 "3class_leanB"),
    ("5class_lean_A",                 "5class_lean"),
    ("5class_lean_B",                 "5class_leanB"),
]

MODELS_ORDER = [
    "HistGradientBoosting", "AdaBoost", "LinearSVC", "DecisionTree",
    "SGDClassifier", "LDA", "QDA", "RidgeClassifier", "KNN",
    "NearestCentroid", "CosineSimilarity",
]


def main() -> None:
    long_rows = []
    for label, dirname in SETTINGS:
        lb_path = REPORTS / dirname / "leaderboard.csv"
        if not lb_path.exists():
            print(f"[grand] MISSING leaderboard: {dirname}")
            continue
        lb = pd.read_csv(lb_path)
        for _, row in lb.iterrows():
            m = row["model"]
            metrics = _metrics_for(dirname, m)
            if metrics is None:
                # Fall back to leaderboard's coarse metrics (no prec/recall).
                metrics = {
                    "accuracy": row["holdout_acc"],
                    "macro_f1": row["holdout_macro_f1"],
                    "macro_precision": float("nan"),
                    "macro_recall": float("nan"),
                }
            long_rows.append({
                "model": m,
                "setting": label,
                **metrics,
            })
        print(f"[grand] {label:30s}  done  ({len(lb)} models)")

    long = pd.DataFrame(long_rows)
    long_path = REPORTS / "grand_summary_long.csv"
    long.to_csv(long_path, index=False)
    print(f"\n[grand] wrote {long_path}  ({len(long)} rows)")

    # Wide pivot: rows=model, cols=setting.metric
    metric_order = ["macro_f1", "accuracy", "macro_precision", "macro_recall"]
    wide_blocks = []
    for label, _dirname in SETTINGS:
        sub = long[long["setting"] == label].set_index("model")
        block = sub[metric_order].rename(
            columns={mc: f"{label}.{mc}" for mc in metric_order})
        wide_blocks.append(block)
    wide = pd.concat(wide_blocks, axis=1)
    # Order rows by canonical model order, drop models that never appear.
    available = [m for m in MODELS_ORDER if m in wide.index]
    wide = wide.loc[available]
    wide_path = REPORTS / "grand_summary.csv"
    wide.to_csv(wide_path)
    print(f"[grand] wrote {wide_path}  ({wide.shape[0]} rows x {wide.shape[1]} cols)")

    # Compact human-glance table: HistGB row, F1 only.
    hgb_f1 = (wide.loc["HistGradientBoosting"]
              .filter(regex=r"\.macro_f1$")
              .rename(lambda c: c.replace(".macro_f1", "")))
    print("\n=== Headline (HistGradientBoosting macro-F1 across settings) ===")
    print(hgb_f1.round(4).to_string())


if __name__ == "__main__":
    main()
