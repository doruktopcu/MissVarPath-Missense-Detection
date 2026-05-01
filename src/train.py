"""Training & evaluation runner.

Procedure (per the project plan):
  1. Load processed parquet.
  2. Stratified 80/20 train/test split.
  3. For each model:
        a. 5-fold stratified CV on the *training* portion -> per-fold metrics.
        b. Refit on the full training portion.
        c. Evaluate on the held-out 20% test set.
        d. Save a classification report (md), confusion matrix (png),
           and per-fold metrics (json).
  4. Aggregate model leaderboard and write to outputs/reports/leaderboard.csv.

Example
-------
    python -m src.train --task 4class
    python -m src.train --task 2class --models RandomForest LightGBM
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import (
    CLASS_2_NAMES,
    CLASS_4_NAMES,
    N_SPLITS,
    PROCESSED_PARQUET,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
)
from .models import MODEL_SPECS, ModelSpec, get_model
from .utils import (
    get_logger,
    metrics_summary,
    plot_confusion_matrix,
    save_json,
    slugify,
    write_classification_report,
)

LOG = get_logger("train")


def load_processed(task: str) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    df = pd.read_parquet(PROCESSED_PARQUET)
    target_col = "target_4" if task == "4class" else "target_2"
    class_names = CLASS_4_NAMES if task == "4class" else CLASS_2_NAMES
    feature_cols = [c for c in df.columns
                    if c not in {"clinvar_sig", "target_4", "target_2"}]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.int64)
    return X, y, feature_cols, class_names


def cv_evaluate(spec: ModelSpec, X: np.ndarray, y: np.ndarray,
                n_classes: int) -> list[dict]:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        t0 = time.time()
        model = spec.builder(n_classes)
        model.fit(X[tr_idx], y[tr_idx])
        y_pred = model.predict(X[va_idx])
        try:
            y_proba = model.predict_proba(X[va_idx])
        except (AttributeError, NotImplementedError):
            y_proba = None
        m = {
            "fold": fold,
            "accuracy": float(accuracy_score(y[va_idx], y_pred)),
            "macro_f1": float(f1_score(y[va_idx], y_pred, average="macro")),
            "weighted_f1": float(f1_score(y[va_idx], y_pred, average="weighted")),
            "fit_seconds": round(time.time() - t0, 2),
        }
        fold_metrics.append(m)
        LOG.info("  fold %d/%d  acc=%.4f  macroF1=%.4f  (%.1fs)",
                 fold, N_SPLITS, m["accuracy"], m["macro_f1"], m["fit_seconds"])
    return fold_metrics


def evaluate_holdout(spec: ModelSpec, X_tr, y_tr, X_te, y_te,
                     class_names: list[str], task_dir: Path,
                     n_classes: int) -> dict:
    model = spec.builder(n_classes)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    fit_s = round(time.time() - t0, 2)
    y_pred = model.predict(X_te)
    try:
        y_proba = model.predict_proba(X_te)
    except (AttributeError, NotImplementedError):
        y_proba = None

    summary = metrics_summary(y_te, y_pred, y_proba, class_names)
    summary["fit_seconds"] = fit_s

    slug = slugify(spec.name)
    report_path = task_dir / f"{slug}_classification_report.md"
    write_classification_report(y_te, y_pred, class_names,
                                title=f"{spec.name} — held-out test set",
                                save_path=report_path)
    plot_confusion_matrix(y_te, y_pred, class_names,
                          title=f"{spec.name} — held-out (counts)",
                          save_path=task_dir / f"{slug}_confusion_matrix.png")
    plot_confusion_matrix(y_te, y_pred, class_names,
                          title=f"{spec.name} — held-out (row-normalized)",
                          save_path=task_dir / f"{slug}_confusion_matrix_normalized.png",
                          normalize=True)
    return summary


def run(task: str, model_names: list[str] | None = None) -> pd.DataFrame:
    X, y, feature_cols, class_names = load_processed(task)
    n_classes = len(class_names)
    LOG.info("Task=%s  X=%s  y unique=%s  features=%d",
             task, X.shape, np.unique(y, return_counts=True), len(feature_cols))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    LOG.info("Train: %s  Test: %s", X_tr.shape, X_te.shape)

    task_dir = REPORTS_DIR / task
    task_dir.mkdir(parents=True, exist_ok=True)

    specs = [get_model(n) for n in model_names] if model_names else MODEL_SPECS
    leaderboard_rows = []
    for spec in specs:
        LOG.info("=" * 70)
        LOG.info("Model: %s  (family=%s)", spec.name, spec.family)
        try:
            cv = cv_evaluate(spec, X_tr, y_tr, n_classes)
        except Exception as e:
            LOG.exception("CV failed for %s: %s", spec.name, e)
            continue
        cv_df = pd.DataFrame(cv)
        cv_summary = {
            "cv_acc_mean": float(cv_df["accuracy"].mean()),
            "cv_acc_std": float(cv_df["accuracy"].std()),
            "cv_macro_f1_mean": float(cv_df["macro_f1"].mean()),
            "cv_macro_f1_std": float(cv_df["macro_f1"].std()),
        }
        LOG.info("  CV: acc=%.4f±%.4f  macroF1=%.4f±%.4f",
                 cv_summary["cv_acc_mean"], cv_summary["cv_acc_std"],
                 cv_summary["cv_macro_f1_mean"], cv_summary["cv_macro_f1_std"])
        try:
            holdout = evaluate_holdout(spec, X_tr, y_tr, X_te, y_te,
                                       class_names, task_dir, n_classes)
        except Exception as e:
            LOG.exception("Holdout eval failed for %s: %s", spec.name, e)
            continue

        save_json({
            "model": spec.name,
            "task": task,
            "cv": cv,
            "cv_summary": cv_summary,
            "holdout": holdout,
        }, task_dir / f"{slugify(spec.name)}_metrics.json")

        leaderboard_rows.append({
            "model": spec.name,
            "family": spec.family,
            **cv_summary,
            "holdout_acc": holdout["accuracy"],
            "holdout_macro_f1": holdout["macro_f1"],
            "holdout_weighted_f1": holdout["weighted_f1"],
            "holdout_mcc": holdout["mcc"],
            "holdout_roc_auc_macro": holdout.get("roc_auc_ovr_macro", float("nan")),
            "fit_seconds": holdout["fit_seconds"],
        })

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        by="holdout_macro_f1", ascending=False)
    leaderboard.to_csv(task_dir / "leaderboard.csv", index=False)
    LOG.info("Leaderboard:\n%s", leaderboard.to_string(index=False))
    return leaderboard


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["4class", "2class"], default="4class")
    p.add_argument("--models", nargs="*", default=None,
                   help="Optional subset of model names to run.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.task, args.models)
