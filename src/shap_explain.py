"""SHAP attributions for the canonical and VUS models (proposal §4 deliverable).

For tree boosters (XGBoost / LightGBM / CatBoost) we use ``shap.TreeExplainer``
which is exact and fast. The output for a 4-class problem is a list of 4
(N, F) arrays; we aggregate to global importance with mean(|SHAP|) per feature.

Usage:
    python -m src.shap_explain --task 4class --model LightGBM --variant base \\
        --tag canonical
    python -m src.shap_explain --task 4class --model CatBoost --variant augmented \\
        --cv-mode gene --drop-prefixes <VEP list> --tag vus

Outputs (under outputs/shap/<run_tag>/):
    summary_bar.png          - top-30 features by mean(|SHAP|) (bar chart)
    summary_beeswarm.png     - top-30 features (per-instance dots; class 0 head)
    feature_importance.csv   - all features ranked by mean(|SHAP|), per class
    shap_values.npz          - cached SHAP arrays + feature names
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import (
    GroupShuffleSplit,
    train_test_split,
)

from .config import (
    CLASS_2_NAMES,
    CLASS_4_NAMES,
    RANDOM_STATE,
    SHAP_DIR,
    TEST_SIZE,
)
from .models import get_model
from .train import META_COLS, load_processed
from .utils import get_logger

LOG = get_logger("shap")


def _split(X, y, groups, cv_mode: str):
    if cv_mode == "gene":
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                random_state=RANDOM_STATE)
        tr, te = next(gss.split(X, y, groups))
    else:
        idx = np.arange(len(X))
        tr, te = train_test_split(idx, test_size=TEST_SIZE, stratify=y,
                                  random_state=RANDOM_STATE)
    return tr, te


def _to_per_class_array(shap_values, n_classes: int, n_samples: int,
                        n_features: int) -> np.ndarray:
    """Normalise the various SHAP output shapes to (n_classes, N, F)."""
    if isinstance(shap_values, list):
        arr = np.stack(shap_values, axis=0)  # (C, N, F)
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            # XGBoost / LightGBM may return (N, F, C) or (C, N, F)
            if arr.shape == (n_samples, n_features, n_classes):
                arr = arr.transpose(2, 0, 1)
            elif arr.shape == (n_classes, n_samples, n_features):
                pass
            else:
                raise ValueError(f"Unexpected SHAP shape: {arr.shape}")
        elif arr.ndim == 2 and n_classes == 2:
            # binary classifier: shape (N, F) for the positive class
            arr = np.stack([-arr, arr], axis=0)
        else:
            raise ValueError(f"Unexpected SHAP shape: {arr.shape}")
    return arr  # (C, N, F)


def _as_underlying_estimator(model):
    """Strip a sklearn Pipeline if present, returning the raw classifier."""
    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline):
        return model.named_steps.get("clf", model.steps[-1][1])
    return model


def _pipeline_preprocessor(model):
    """If `model` is a Pipeline, return the prefix that transforms inputs (everything
    except the final step). Returns None for bare estimators."""
    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline) and len(model.steps) > 1:
        prefix = Pipeline(model.steps[:-1])
        return prefix
    return None


def run(task: str, model_name: str, variant: str = "base",
        cv_mode: str = "kfold", drop_prefixes: list[str] | None = None,
        tag: str = "shap", n_explain: int = 1000) -> None:
    X, y, groups, feature_cols, class_names = load_processed(
        task, variant=variant, drop_prefixes=drop_prefixes)
    n_classes = len(class_names)
    LOG.info("Task=%s model=%s variant=%s cv_mode=%s features=%d classes=%d",
             task, model_name, variant, cv_mode, len(feature_cols), n_classes)

    tr, te = _split(X, y, groups, cv_mode)
    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[te], y[te]
    LOG.info("Train: %s  Test: %s", X_tr.shape, X_te.shape)

    spec = get_model(model_name)
    model = spec.builder(n_classes)
    LOG.info("Fitting %s on full train portion...", model_name)
    model.fit(X_tr, y_tr)

    explainer_input = _as_underlying_estimator(model)
    preprocessor = _pipeline_preprocessor(model)

    # Sample test rows for explanation (1000 keeps SHAP fast and the plots clear).
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(X_te), size=min(n_explain, len(X_te)), replace=False)
    X_explain = X_te[sample_idx]
    y_explain = y_te[sample_idx]

    # Linear models (LogReg, etc.): use LinearExplainer with a scaled background.
    is_linear = hasattr(explainer_input, "coef_")
    if is_linear:
        bg_idx = rng.choice(len(X_tr), size=min(200, len(X_tr)), replace=False)
        X_bg = X_tr[bg_idx]
        if preprocessor is not None:
            X_bg = preprocessor.transform(X_bg)
            X_explain_for_shap = preprocessor.transform(X_explain)
        else:
            X_explain_for_shap = X_explain
        LOG.info("Explaining %d rows with LinearExplainer (background=%d)...",
                 len(X_explain), len(X_bg))
        explainer = shap.LinearExplainer(explainer_input, X_bg)
        raw = explainer.shap_values(X_explain_for_shap)
    else:
        LOG.info("Explaining %d held-out rows with TreeExplainer...", len(X_explain))
        explainer = shap.TreeExplainer(explainer_input)
        raw = explainer.shap_values(X_explain)
    shap_arr = _to_per_class_array(raw, n_classes, len(X_explain), len(feature_cols))
    LOG.info("SHAP values shape: %s  (classes, samples, features)", shap_arr.shape)

    out_dir = SHAP_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-class mean(|SHAP|) feature importance
    per_class = np.abs(shap_arr).mean(axis=1)  # (C, F)
    overall = per_class.mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap_overall": overall,
        **{f"mean_abs_shap_{class_names[i]}": per_class[i]
           for i in range(n_classes)},
    }).sort_values("mean_abs_shap_overall", ascending=False)
    importance_df.to_csv(out_dir / "feature_importance.csv", index=False)
    LOG.info("Top-15 features by mean(|SHAP|):\n%s",
             importance_df.head(15).to_string(index=False))

    # Save raw SHAP for downstream (per-instance) analysis.
    np.savez_compressed(
        out_dir / "shap_values.npz",
        shap_values=shap_arr,
        X_explain=X_explain,
        y_explain=y_explain,
        feature_names=np.array(feature_cols, dtype=object),
        class_names=np.array(class_names, dtype=object),
    )

    # Bar plot of top-30 overall.
    top = importance_df.head(30).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(top["feature"], top["mean_abs_shap_overall"], color="#4C72B0")
    ax.set_xlabel("mean(|SHAP|) across classes")
    ax.set_title(f"Top-30 features — {model_name} ({tag})")
    fig.tight_layout()
    fig.savefig(out_dir / "summary_bar.png", dpi=140)
    plt.close(fig)

    # Beeswarm for class 0 (Benign) — gives the per-instance shape.
    try:
        class_label = class_names[0]
        plt.figure(figsize=(8, 9))
        shap.summary_plot(
            shap_arr[0], X_explain, feature_names=feature_cols,
            max_display=30, show=False, plot_size=None,
        )
        plt.title(f"SHAP beeswarm — class={class_label} — {model_name} ({tag})")
        plt.tight_layout()
        plt.savefig(out_dir / "summary_beeswarm.png", dpi=140)
        plt.close()
    except Exception as e:
        LOG.warning("Beeswarm plot failed (%s); bar chart still saved.", e)

    LOG.info("SHAP artifacts saved under %s", out_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["4class", "2class"], default="4class")
    p.add_argument("--model", required=True,
                   help="Model name (must be in MODEL_SPECS).")
    p.add_argument("--variant", choices=["base", "augmented"], default="base")
    p.add_argument("--cv-mode", choices=["kfold", "gene"], default="kfold")
    p.add_argument("--drop-prefixes", nargs="*", default=None)
    p.add_argument("--tag", default="shap")
    p.add_argument("--n-explain", type=int, default=1000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.task, args.model, variant=args.variant, cv_mode=args.cv_mode,
        drop_prefixes=args.drop_prefixes, tag=args.tag, n_explain=args.n_explain)
