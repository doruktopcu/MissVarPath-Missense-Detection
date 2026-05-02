"""Training & evaluation runner.

Procedure:
  1. Load processed parquet (base or augmented).
  2. Optionally drop columns by prefix list (ablation).
  3. Train/test split:
       --cv-mode kfold (default): stratified 80/20 + StratifiedKFold(5)
       --cv-mode gene: GroupShuffleSplit by gene_symbol + StratifiedGroupKFold(5)
  4. For each model: 5-fold CV on training portion, holdout on test portion.
  5. Save classification report (md), confusion matrix (PNG), metrics JSON.
  6. Aggregate leaderboard at outputs/reports/<task>[_<variant>][_<tag>]/leaderboard.csv.

Examples
--------
    # base baseline (#6)
    python -m src.train --task 4class

    # augmented baseline (#8)
    python -m src.train --task 4class --variant augmented

    # gene-stratified CV on base
    python -m src.train --task 4class --cv-mode gene --tag gene

    # VEP-score ablation on augmented (VUS scenario)
    python -m src.train --task 4class --variant augmented \\
        --drop-prefixes alphamissense_ cadd_ cadd_exome_ revel_ sift_ \\
                        metarnn_ bayesdel_ fathmm_ mutationtaster_ provean_ \\
                        esm1b_ eve_ primateai_ mvp_ ditto_ mutation_assessor_ \\
                        vest_ chasmplus mutpred1_ aloft_ gmvp_ \\
        --tag no_vep
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

from .config import (
    AUGMENTED_PARQUET,
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

META_COLS = {"clinvar_sig", "target_4", "target_2", "gene_symbol"}


def load_processed(task: str, variant: str = "base",
                   drop_prefixes: list[str] | None = None,
                   keep_prefixes: list[str] | None = None,
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Returns (X, y, groups, feature_cols, class_names).

    `groups` is the gene-symbol array (length = X.shape[0]); used by gene-CV.
    `drop_prefixes` and `keep_prefixes` are mutually exclusive — use one or the other.
    """
    parquet = AUGMENTED_PARQUET if variant == "augmented" else PROCESSED_PARQUET
    df = pd.read_parquet(parquet)
    target_col = "target_4" if task == "4class" else "target_2"
    class_names = CLASS_4_NAMES if task == "4class" else CLASS_2_NAMES

    feature_cols = [c for c in df.columns if c not in META_COLS]
    if drop_prefixes and keep_prefixes:
        raise ValueError("Use either --drop-prefixes or --keep-prefixes, not both.")
    if drop_prefixes:
        before = len(feature_cols)
        feature_cols = [c for c in feature_cols
                        if not any(c.startswith(p) for p in drop_prefixes)]
        LOG.info("Ablation drop: %d → %d features (dropped %d by prefix)",
                 before, len(feature_cols), before - len(feature_cols))
    if keep_prefixes:
        before = len(feature_cols)
        feature_cols = [c for c in feature_cols
                        if any(c.startswith(p) for p in keep_prefixes)]
        LOG.info("Ablation keep: %d → %d features (kept %d by prefix)",
                 before, len(feature_cols), len(feature_cols))

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.int64)
    groups = (df["gene_symbol"].astype(str).to_numpy()
              if "gene_symbol" in df.columns else np.zeros(len(df), dtype=object))
    return X, y, groups, feature_cols, class_names


def split_train_test(X, y, groups, cv_mode: str
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                np.ndarray, np.ndarray, np.ndarray]:
    if cv_mode == "gene":
        # Hold out 20% of *genes* (and their variants).
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                random_state=RANDOM_STATE)
        tr, te = next(gss.split(X, y, groups))
        n_train_genes = len(np.unique(groups[tr]))
        n_test_genes = len(np.unique(groups[te]))
        overlap = set(groups[tr]) & set(groups[te])
        LOG.info("Gene-grouped split: %d train genes, %d test genes, %d overlap",
                 n_train_genes, n_test_genes, len(overlap))
    else:
        idx = np.arange(len(X))
        tr, te = train_test_split(idx, test_size=TEST_SIZE, stratify=y,
                                  random_state=RANDOM_STATE)
    return X[tr], X[te], y[tr], y[te], groups[tr], groups[te]


def _make_cv(cv_mode: str):
    if cv_mode == "gene":
        return StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                    random_state=RANDOM_STATE)
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                           random_state=RANDOM_STATE)


def cv_evaluate(spec: ModelSpec, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                n_classes: int, cv_mode: str) -> list[dict]:
    cv = _make_cv(cv_mode)
    split_args = (X, y, groups) if cv_mode == "gene" else (X, y)
    fold_metrics = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(*split_args), start=1):
        t0 = time.time()
        model = spec.builder(n_classes)
        model.fit(X[tr_idx], y[tr_idx])
        y_pred = model.predict(X[va_idx])
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
    write_classification_report(y_te, y_pred, class_names,
                                title=f"{spec.name} — held-out test set",
                                save_path=task_dir / f"{slug}_classification_report.md")
    plot_confusion_matrix(y_te, y_pred, class_names,
                          title=f"{spec.name} — held-out (counts)",
                          save_path=task_dir / f"{slug}_confusion_matrix.png")
    plot_confusion_matrix(y_te, y_pred, class_names,
                          title=f"{spec.name} — held-out (row-normalized)",
                          save_path=task_dir / f"{slug}_confusion_matrix_normalized.png",
                          normalize=True)
    return summary


def _task_dir_name(task: str, variant: str, cv_mode: str, tag: str | None) -> str:
    parts = [task]
    if variant != "base":
        parts.append(variant)
    if cv_mode != "kfold":
        parts.append(cv_mode)
    if tag and tag not in parts:
        parts.append(tag)
    return "_".join(parts)


def run(task: str, model_names: list[str] | None = None,
        variant: str = "base", cv_mode: str = "kfold",
        drop_prefixes: list[str] | None = None,
        keep_prefixes: list[str] | None = None,
        tag: str | None = None) -> pd.DataFrame:
    X, y, groups, feature_cols, class_names = load_processed(
        task, variant=variant, drop_prefixes=drop_prefixes,
        keep_prefixes=keep_prefixes)
    n_classes = len(class_names)
    LOG.info("Task=%s  variant=%s  cv_mode=%s  tag=%s  X=%s  features=%d  unique_genes=%d",
             task, variant, cv_mode, tag, X.shape, len(feature_cols),
             len(np.unique(groups)))

    X_tr, X_te, y_tr, y_te, g_tr, g_te = split_train_test(X, y, groups, cv_mode)
    LOG.info("Train: %s  Test: %s", X_tr.shape, X_te.shape)

    task_dir = REPORTS_DIR / _task_dir_name(task, variant, cv_mode, tag)
    task_dir.mkdir(parents=True, exist_ok=True)

    # Persist the feature manifest so it's traceable later.
    (task_dir / "features.txt").write_text("\n".join(feature_cols))

    specs = [get_model(n) for n in model_names] if model_names else MODEL_SPECS
    leaderboard_rows = []
    for spec in specs:
        LOG.info("=" * 70)
        LOG.info("Model: %s  (family=%s)", spec.name, spec.family)
        try:
            cv = cv_evaluate(spec, X_tr, y_tr, g_tr, n_classes, cv_mode)
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
            "variant": variant,
            "cv_mode": cv_mode,
            "tag": tag,
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
    p.add_argument("--variant", choices=["base", "augmented"], default="base",
                   help="'base' = tabular only; 'augmented' = + k-mer + BLAST features.")
    p.add_argument("--cv-mode", choices=["kfold", "gene"], default="kfold",
                   help="'kfold' = StratifiedKFold(5); 'gene' = StratifiedGroupKFold by gene_symbol with GroupShuffleSplit holdout.")
    p.add_argument("--drop-prefixes", nargs="*", default=None,
                   help="Column-name prefixes to drop (ablation).")
    p.add_argument("--keep-prefixes", nargs="*", default=None,
                   help="Column-name prefixes to keep — drops everything else.")
    p.add_argument("--tag", default=None, help="Tag appended to the report dir.")
    p.add_argument("--models", nargs="*", default=None,
                   help="Optional subset of model names to run.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.task, args.models, variant=args.variant, cv_mode=args.cv_mode,
        drop_prefixes=args.drop_prefixes, keep_prefixes=args.keep_prefixes,
        tag=args.tag)
