"""Tight grid-search hyperparameter tuning (no Optuna).

Generic runner that sweeps a per-model grid by 5-fold StratifiedKFold (or
StratifiedGroupKFold under --cv-mode gene) macro-F1 on the 80% train portion,
then refits the winning combination on the full train portion and evaluates
on the held-out 20%.

Outputs (under outputs/tuning/<task>[_<cv_mode>]/<model>/):
    grid_results.csv      every (combo, fold) — cv_acc, cv_macro_f1, fit_seconds
    grid_summary.csv      one row per combo — cv_macro_f1 mean/std + ranks
    best_config.json      winning hyperparameters + holdout metrics

Grids are defined in the ``GRIDS`` dict at the bottom of this module (already
populated for all 10 tunable models). Bad parameter combinations are caught
per-combo: the offending combo is logged and skipped, and the run continues
with the remaining combos rather than aborting.

    python -m src.tune --task 4class --models <name1> <name2> ...
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import product
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

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .config import (
    CLASS_2_NAMES,
    CLASS_4_NAMES,
    N_SPLITS,
    OUTPUTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from .train import load_processed
from .utils import get_logger, metrics_summary, save_json, slugify

LOG = get_logger("tune")

TUNE_DIR = OUTPUTS_DIR / "tuning"


# -------------------- Grids --------------------------------------------------
# Each entry: name -> callable returning (grid, param_names, builder).
#   grid:          list of parameter tuples
#   param_names:   tuple of names, same length as each tuple in `grid`
#   builder:       fn(n_classes, params_tuple) -> sklearn-compatible estimator


def _decision_tree_grid():
    grid = list(product(
        [None, 10, 20, 30],          # max_depth
        [1, 5, 10, 20],               # min_samples_leaf
        [0.0, 0.001, 0.005],          # ccp_alpha
    ))
    def builder(_n_classes, params):
        max_depth, min_samples_leaf, ccp_alpha = params
        return DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            class_weight="balanced", random_state=RANDOM_STATE,
        )
    return grid, ("max_depth", "min_samples_leaf", "ccp_alpha"), builder


def _knn_grid():
    grid = list(product(
        [5, 15, 30, 50, 100],         # n_neighbors
        ["uniform", "distance"],       # weights
        ["minkowski", "cosine"],       # metric
    ))
    def builder(_n_classes, params):
        n_neighbors, weights, metric = params
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=n_neighbors, weights=weights, metric=metric,
                n_jobs=-1,
            )),
        ])
    return grid, ("n_neighbors", "weights", "metric"), builder


def _qda_grid():
    grid = [(rp,) for rp in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]]
    def builder(_n_classes, params):
        reg_param, = params
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", QuadraticDiscriminantAnalysis(reg_param=reg_param)),
        ])
    return grid, ("reg_param",), builder


def _sgd_grid():
    # SGDClassifier with log_loss; alpha is the main regularization knob.
    # elasticnet rows include l1_ratio implicitly = 0.15.
    grid = [
        (1e-5, "l2",         2000),
        (1e-5, "l2",         5000),
        (1e-4, "l2",         2000),
        (1e-4, "l2",         5000),
        (1e-3, "l2",         2000),
        (1e-3, "l2",         5000),
        (1e-2, "l2",         5000),
        (1e-5, "elasticnet", 5000),
        (1e-4, "elasticnet", 5000),
        (1e-3, "elasticnet", 5000),
    ]
    def builder(_n_classes, params):
        alpha, penalty, max_iter = params
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SGDClassifier(
                loss="log_loss", alpha=alpha, penalty=penalty,
                l1_ratio=0.15, max_iter=max_iter,
                tol=1e-3, early_stopping=True,
                class_weight="balanced",
                n_jobs=-1, random_state=RANDOM_STATE,
            )),
        ])
    return grid, ("alpha", "penalty", "max_iter"), builder


def _lda_grid():
    # solver=svd doesn't support shrinkage; skip it.
    grid = []
    for solver in ["lsqr", "eigen"]:
        for shrinkage in [None, "auto", 0.1, 0.3, 0.5, 0.7, 0.9]:
            grid.append((solver, shrinkage))
    def builder(_n_classes, params):
        solver, shrinkage = params
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(
                solver=solver, shrinkage=shrinkage,
            )),
        ])
    return grid, ("solver", "shrinkage"), builder


def _ridge_grid():
    grid = [(a,) for a in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]]
    def builder(_n_classes, params):
        alpha, = params
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RidgeClassifier(
                alpha=alpha, class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ])
    return grid, ("alpha",), builder


def _nearest_centroid_grid():
    grid = []
    for metric in ["euclidean", "manhattan"]:
        for shrink in [None, 0.1, 0.5, 1.0, 2.0, 4.0]:
            grid.append((metric, shrink))
    def builder(_n_classes, params):
        metric, shrink_threshold = params
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", NearestCentroid(
                metric=metric, shrink_threshold=shrink_threshold,
            )),
        ])
    return grid, ("metric", "shrink_threshold"), builder


def _linear_svc_grid():
    grid = list(product(
        [0.01, 0.1, 1.0, 10.0, 100.0],   # C
        [5000, 10000],                    # max_iter
    ))
    def builder(_n_classes, params):
        C, max_iter = params
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(
                C=C, max_iter=max_iter, dual="auto",
                class_weight="balanced", random_state=RANDOM_STATE,
            )),
        ])
    return grid, ("C", "max_iter"), builder


def _hist_gbm_grid():
    # learning_rate × max_iter paired (LR-iter trade), crossed with capacity knobs
    grid = []
    for lr, max_iter in [(0.1, 300), (0.05, 600), (0.03, 1200), (0.02, 2000)]:
        for max_leaf_nodes in [31, 63, 127]:
            for min_samples_leaf in [20, 50]:
                for l2_reg in [0.0, 1.0]:
                    grid.append((lr, max_iter, max_leaf_nodes, min_samples_leaf, l2_reg))
    def builder(_n_classes, params):
        lr, max_iter, max_leaf_nodes, min_samples_leaf, l2_reg = params
        return HistGradientBoostingClassifier(
            learning_rate=lr, max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_reg,
            max_depth=None,
            random_state=RANDOM_STATE,
        )
    return grid, ("learning_rate", "max_iter", "max_leaf_nodes",
                  "min_samples_leaf", "l2_regularization"), builder


def _adaboost_grid():
    grid = list(product(
        [100, 300, 500],   # n_estimators
        [0.1, 0.5, 1.0],   # learning_rate
        [1, 3, 5],          # base estimator max_depth
    ))
    def builder(_n_classes, params):
        n_estimators, lr, base_depth = params
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=base_depth,
                                              random_state=RANDOM_STATE),
            n_estimators=n_estimators, learning_rate=lr,
            random_state=RANDOM_STATE,
        )
    return grid, ("n_estimators", "learning_rate", "base_depth"), builder


GRIDS: dict[str, callable] = {
    # --- Phase 1: fast batch ---
    "DecisionTree":         _decision_tree_grid,
    "KNN":                  _knn_grid,
    "QDA":                  _qda_grid,
    "SGDClassifier":        _sgd_grid,
    "LDA":                  _lda_grid,
    "RidgeClassifier":      _ridge_grid,
    "NearestCentroid":      _nearest_centroid_grid,
    "LinearSVC":            _linear_svc_grid,
    # --- Phase 2: heavy ---
    "HistGradientBoosting": _hist_gbm_grid,
    "AdaBoost":             _adaboost_grid,
}


# -------------------- CV runner ---------------------------------------------

def _split(X, y, groups, cv_mode):
    if cv_mode == "gene":
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                random_state=RANDOM_STATE)
        return next(gss.split(X, y, groups))
    idx = np.arange(len(X))
    return train_test_split(idx, test_size=TEST_SIZE, stratify=y,
                            random_state=RANDOM_STATE)


def _cv_iter(X_tr, y_tr, groups_tr, cv_mode):
    if cv_mode == "gene":
        skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                   random_state=RANDOM_STATE)
        return skf.split(X_tr, y_tr, groups_tr)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=RANDOM_STATE)
    return skf.split(X_tr, y_tr)


def tune_model(model_name, task, variant, cv_mode):
    if model_name not in GRIDS:
        raise KeyError(
            f"No tuning grid defined for {model_name!r}. "
            f"Add an entry to GRIDS in src/tune.py."
        )

    X, y, groups, feature_cols, class_names = load_processed(
        task, variant=variant, drop_prefixes=None)
    n_classes = len(class_names)
    LOG.info("Tuning %s  task=%s  variant=%s  cv=%s  features=%d  classes=%d",
             model_name, task, variant, cv_mode, len(feature_cols), n_classes)

    grid, param_names, builder = GRIDS[model_name]()
    LOG.info("Grid size: %d combinations", len(grid))

    tr, te = _split(X, y, groups, cv_mode)
    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[te], y[te]
    groups_tr = groups[tr]
    LOG.info("Train: %s  Holdout: %s", X_tr.shape, X_te.shape)

    grid_rows = []
    skipped_combos: list[tuple[int, tuple, str]] = []
    for combo_i, params in enumerate(grid):
        cv_accs, cv_f1s, fit_secs = [], [], []
        cv = list(_cv_iter(X_tr, y_tr, groups_tr, cv_mode))
        combo_failed = False
        for fold_i, (tri, vai) in enumerate(cv, start=1):
            t0 = time.time()
            try:
                model = builder(n_classes, params)
                model.fit(X_tr[tri], y_tr[tri])
                yp = model.predict(X_tr[vai])
            except Exception as e:
                combo_failed = True
                skipped_combos.append((combo_i, params, f"{type(e).__name__}: {e}"))
                LOG.warning(
                    "  combo %2d/%d %s — skipped on fold %d (%s: %s)",
                    combo_i + 1, len(grid),
                    dict(zip(param_names, params)),
                    fold_i, type(e).__name__, str(e).split('\n')[0][:120],
                )
                break
            fit_secs.append(time.time() - t0)
            cv_accs.append(accuracy_score(y_tr[vai], yp))
            cv_f1s.append(f1_score(y_tr[vai], yp, average="macro"))
            grid_rows.append({
                "combo": combo_i, "fold": fold_i,
                **dict(zip(param_names, [str(p) for p in params])),
                "cv_acc": cv_accs[-1], "cv_macro_f1": cv_f1s[-1],
                "fit_seconds": fit_secs[-1],
            })
        if combo_failed:
            continue
        LOG.info("  combo %2d/%d  %s  cv_macro_f1=%.4f±%.4f  (%.1fs/fold)",
                 combo_i + 1, len(grid),
                 dict(zip(param_names, params)),
                 float(np.mean(cv_f1s)), float(np.std(cv_f1s)),
                 float(np.mean(fit_secs)))
    if skipped_combos:
        LOG.warning("Skipped %d/%d combos for %s due to fit errors.",
                    len(skipped_combos), len(grid), model_name)
    if not grid_rows:
        LOG.error("Every combo failed for %s — nothing to summarise. Skipping.", model_name)
        return

    grid_df = pd.DataFrame(grid_rows)
    summary = (grid_df
               .groupby(["combo", *param_names])
               .agg(cv_macro_f1_mean=("cv_macro_f1", "mean"),
                    cv_macro_f1_std=("cv_macro_f1", "std"),
                    cv_acc_mean=("cv_acc", "mean"),
                    cv_acc_std=("cv_acc", "std"),
                    fit_seconds_mean=("fit_seconds", "mean"))
               .reset_index()
               .sort_values("cv_macro_f1_mean", ascending=False))
    summary["rank"] = np.arange(1, len(summary) + 1)

    best = summary.iloc[0].to_dict()
    LOG.info("Best combo: %s (cv_macro_f1=%.4f±%.4f)",
             {n: best[n] for n in param_names},
             best["cv_macro_f1_mean"], best["cv_macro_f1_std"])

    # Refit best on full train portion; evaluate on holdout.
    best_params = tuple(grid[int(best["combo"])])
    best_model = builder(n_classes, best_params)
    t0 = time.time()
    best_model.fit(X_tr, y_tr)
    fit_seconds_full = time.time() - t0
    y_pred = best_model.predict(X_te)
    try:
        y_proba = best_model.predict_proba(X_te)
    except Exception:
        y_proba = None
    holdout = metrics_summary(y_te, y_pred, y_proba, class_names)
    holdout["fit_seconds"] = fit_seconds_full
    LOG.info("Holdout (best): acc=%.4f  macroF1=%.4f  MCC=%.4f",
             holdout["accuracy"], holdout["macro_f1"], holdout["mcc"])

    out_dir = TUNE_DIR / (f"{task}_{cv_mode}" if cv_mode != "kfold" else task) / slugify(model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(out_dir / "grid_results.csv", index=False)
    summary.to_csv(out_dir / "grid_summary.csv", index=False)
    save_json({
        "model": model_name, "task": task, "variant": variant, "cv_mode": cv_mode,
        "param_names": list(param_names),
        "best_params": dict(zip(param_names, [str(p) for p in best_params])),
        "best_cv_macro_f1_mean": float(best["cv_macro_f1_mean"]),
        "best_cv_macro_f1_std": float(best["cv_macro_f1_std"]),
        "holdout": holdout,
    }, out_dir / "best_config.json")
    LOG.info("Saved tuning artifacts to %s", out_dir)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["4class", "2class"], default="4class")
    p.add_argument("--variant", choices=["base", "augmented"], default="base")
    p.add_argument("--cv-mode", choices=["kfold", "gene"], default="kfold")
    p.add_argument("--models", nargs="+", required=True,
                   help="Model names to tune (must each have an entry in GRIDS).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for name in args.models:
        tune_model(name, args.task, args.variant, args.cv_mode)
