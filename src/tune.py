"""Tight grid-search hyperparameter tuning (no Optuna).

Generic runner that sweeps a per-model grid by 5-fold StratifiedKFold (or
StratifiedGroupKFold under --cv-mode gene) macro-F1 on the 80% train portion,
then refits the winning combination on the full train portion and evaluates
on the held-out 20%.

Outputs (under outputs/tuning/<task>[_<cv_mode>]/<model>/):
    grid_results.csv      every (combo, fold) — cv_acc, cv_macro_f1, fit_seconds
    grid_summary.csv      one row per combo — cv_macro_f1 mean/std + ranks
    best_config.json      winning hyperparameters + holdout metrics

Grids are defined in the ``GRIDS`` dict at the bottom of this module — refill
for whichever models you want to tune, then call:

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
#
# Refill this dict per the active model suite before invoking the runner.

GRIDS: dict[str, callable] = {}


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
    p.add_argument("--models", nargs="+", required=True,
                   help="Model names to tune (must each have an entry in GRIDS).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for name in args.models:
        tune_model(name, args.task, args.variant, args.cv_mode)
