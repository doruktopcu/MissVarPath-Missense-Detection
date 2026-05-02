"""Per-class SHAP analysis on top of cached `shap_values.npz` artifacts.

For each pair of classes, computes the *contrastive* SHAP — the per-feature
mean(|SHAP_class_a − SHAP_class_b|) — i.e. which features drive the model's
choice between class A and class B. This is the signal you actually want when
the headline error is "Likely benign vs Benign", not the global mean(|SHAP|)
that summary_bar.png shows.

Usage:
    python -m src.shap_per_class --tag canonical_lightgbm
    python -m src.shap_per_class --tag vus_catboost
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import SHAP_DIR
from .utils import get_logger

LOG = get_logger("shap_perclass")


def _load(tag: str):
    npz = np.load(SHAP_DIR / tag / "shap_values.npz", allow_pickle=True)
    shap_arr = npz["shap_values"]              # (C, N, F)
    feature_names = list(npz["feature_names"])
    class_names = list(npz["class_names"])
    return shap_arr, feature_names, class_names


def _contrastive(shap_arr: np.ndarray, ci: int, cj: int) -> np.ndarray:
    """mean(|SHAP_ci - SHAP_cj|) per feature — magnitude of disagreement."""
    return np.abs(shap_arr[ci] - shap_arr[cj]).mean(axis=0)


def _plot_top(values: np.ndarray, names, title: str, out_path: Path,
              top_n: int = 25):
    order = np.argsort(values)[::-1][:top_n]
    df = pd.DataFrame({"feature": [names[i] for i in order],
                       "score": values[order]}).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.28)))
    ax.barh(df["feature"], df["score"], color="#937860")
    ax.set_xlabel("mean(|SHAP_a − SHAP_b|) over 1,000 held-out rows")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run(tag: str) -> None:
    shap_arr, names, class_names = _load(tag)
    n_classes = len(class_names)
    LOG.info("Tag=%s  shape=%s  classes=%s", tag, shap_arr.shape, class_names)

    out_dir = SHAP_DIR / tag / "per_class"
    out_dir.mkdir(parents=True, exist_ok=True)

    # The two diagnostic boundaries from #6:
    #   Benign(0) vs Likely benign(1)        — hardest "low-pathogenicity" call
    #   Likely pathogenic(2) vs Pathogenic(3) — hardest "high-pathogenicity" call
    # Plus the easy benign-vs-pathogenic sanity check.
    pairs = [(0, 1), (2, 3), (0, 3)]
    if n_classes != 4:
        # Fall back to all unordered pairs if the cached run was 2-class.
        pairs = [(i, j) for i in range(n_classes) for j in range(i + 1, n_classes)]

    rows = []
    for (i, j) in pairs:
        scores = _contrastive(shap_arr, i, j)
        order = np.argsort(scores)[::-1]
        for rank, idx in enumerate(order, start=1):
            rows.append({
                "class_a": class_names[i], "class_b": class_names[j],
                "rank": rank, "feature": names[idx],
                "mean_abs_shap_diff": float(scores[idx]),
            })
        slug = f"{class_names[i].replace(' ', '_')}_vs_{class_names[j].replace(' ', '_')}"
        _plot_top(scores, names,
                  title=f"Top features distinguishing {class_names[i]} vs {class_names[j]}",
                  out_path=out_dir / f"{slug}.png")
        LOG.info("Pair %s vs %s — top 5: %s",
                 class_names[i], class_names[j],
                 [(names[k], round(scores[k], 4)) for k in order[:5]])

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "contrastive_importance.csv", index=False)
    LOG.info("Saved %d ranked rows -> %s", len(df), out_dir / "contrastive_importance.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True,
                   help="Tag of an existing SHAP run under outputs/shap/<tag>/.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.tag)
