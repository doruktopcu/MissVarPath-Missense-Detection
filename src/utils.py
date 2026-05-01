"""Shared helpers: logging, plotting, metric reporting."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def get_logger(name: str = "mvp") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                                     datefmt="%H:%M:%S"))
    logger.addHandler(h)
    return logger


def slugify(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_").lower()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    title: str,
    save_path: Path,
    normalize: bool = False,
) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_to_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_to_plot = np.nan_to_num(cm_to_plot)
        fmt = ".2f"
    else:
        cm_to_plot = cm
        fmt = "d"
    fig, ax = plt.subplots(figsize=(max(4, len(class_names) * 1.2),
                                    max(3.5, len(class_names) * 1.0)))
    sns.heatmap(cm_to_plot, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return cm


def metrics_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    class_names: Sequence[str],
) -> dict:
    report = classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                   target_names=class_names, output_dict=True,
                                   zero_division=0)
    out = {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "f1_per_class": {c: report[c]["f1-score"] for c in class_names},
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_proba is not None:
        try:
            out["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            )
        except ValueError:
            out["roc_auc_ovr_macro"] = float("nan")
    return out


def write_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    title: str,
    save_path: Path,
) -> str:
    text = classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                 target_names=class_names, digits=4, zero_division=0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(f"# {title}\n\n```\n{text}\n```\n")
    return text


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
