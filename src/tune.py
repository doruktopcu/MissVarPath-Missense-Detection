"""Tight grid-search hyperparameter tuning (no Optuna).

Sweeps small grids for the four model families that respond meaningfully to
their main capacity knobs:

    XGBoost        n_estimators x max_depth        (LR fixed at 0.05)
    LightGBM       n_estimators x num_leaves       (LR fixed at 0.05)
    CatBoost       iterations   x depth            (LR fixed at 0.05)
    ShallowNN_MLP  batch_size   x hidden_layer_sizes

Each combination is scored by 5-fold StratifiedKFold (or StratifiedGroupKFold
under --cv-mode gene) macro-F1 mean on the 80%% train portion, mirroring the
baseline setup in src/train.py. The winning config per model is refit on the
full train portion and evaluated on the held-out 20%%.

Outputs (under outputs/tuning/<task>[_<cv_mode>]/<model>/):
    grid_results.csv      every (combo, fold) — cv_acc, cv_macro_f1, fit_seconds
    grid_summary.csv      one row per combo — cv_macro_f1 mean/std + ranks
    best_config.json      winning hyperparameters + holdout metrics

Usage:
    python -m src.tune --task 4class --models XGBoost LightGBM CatBoost ShallowNN_MLP
    python -m src.tune --task 4class --cv-mode gene --models LightGBM XGBoost
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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

def _xgb_grid():
    from xgboost import XGBClassifier
    grid = list(product([100, 300, 500], [4, 6, 8]))
    def builder(n_classes, params):
        n_estimators, max_depth = params
        return XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            num_class=n_classes if n_classes > 2 else None,
            tree_method="hist",
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            n_jobs=-1, random_state=RANDOM_STATE,
        )
    return grid, ("n_estimators", "max_depth"), builder


def _lgb_grid():
    from lightgbm import LGBMClassifier
    grid = list(product([100, 300, 500], [31, 63, 127]))
    def builder(n_classes, params):
        n_estimators, num_leaves = params
        return LGBMClassifier(
            n_estimators=n_estimators, num_leaves=num_leaves,
            learning_rate=0.05, max_depth=-1,
            subsample=0.9, colsample_bytree=0.9,
            objective="multiclass" if n_classes > 2 else "binary",
            num_class=n_classes if n_classes > 2 else 1,
            n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
        )
    return grid, ("n_estimators", "num_leaves"), builder


def _cat_grid():
    from catboost import CatBoostClassifier
    grid = list(product([100, 300, 500], [4, 6, 8]))
    def builder(n_classes, params):
        iterations, depth = params
        return CatBoostClassifier(
            iterations=iterations, depth=depth, learning_rate=0.05,
            loss_function="MultiClass" if n_classes > 2 else "Logloss",
            random_seed=RANDOM_STATE, verbose=False, thread_count=-1,
        )
    return grid, ("iterations", "depth"), builder


def _mlp_grid():
    grid = list(product([64, 128, 256], [(64,), (128,), (256,)]))
    def builder(_n_classes, params):
        batch_size, hidden = params
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=hidden, activation="relu", solver="adam",
                alpha=1e-4, batch_size=batch_size, learning_rate_init=1e-3,
                max_iter=400, n_iter_no_change=15, tol=1e-5,
                early_stopping=True, random_state=RANDOM_STATE,
            )),
        ])
    return grid, ("batch_size", "hidden_layer_sizes"), builder


GRIDS = {
    "XGBoost": _xgb_grid,
    "LightGBM": _lgb_grid,
    "CatBoost": _cat_grid,
    "ShallowNN_MLP": _mlp_grid,
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
    for combo_i, params in enumerate(grid):
        cv_accs, cv_f1s, fit_secs = [], [], []
        cv = list(_cv_iter(X_tr, y_tr, groups_tr, cv_mode))
        for fold_i, (tri, vai) in enumerate(cv, start=1):
            t0 = time.time()
            model = builder(n_classes, params)
            model.fit(X_tr[tri], y_tr[tri])
            yp = model.predict(X_tr[vai])
            fit_secs.append(time.time() - t0)
            cv_accs.append(accuracy_score(y_tr[vai], yp))
            cv_f1s.append(f1_score(y_tr[vai], yp, average="macro"))
            grid_rows.append({
                "combo": combo_i, "fold": fold_i,
                **dict(zip(param_names, [str(p) for p in params])),
                "cv_acc": cv_accs[-1], "cv_macro_f1": cv_f1s[-1],
                "fit_seconds": fit_secs[-1],
            })
        LOG.info("  combo %2d/%d  %s  cv_macro_f1=%.4f±%.4f  (%.1fs/fold)",
                 combo_i + 1, len(grid),
                 dict(zip(param_names, params)),
                 float(np.mean(cv_f1s)), float(np.std(cv_f1s)),
                 float(np.mean(fit_secs)))

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
    p.add_argument("--models", nargs="+", default=list(GRIDS.keys()),
                   choices=list(GRIDS.keys()))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for name in args.models:
        tune_model(name, args.task, args.variant, args.cv_mode)
