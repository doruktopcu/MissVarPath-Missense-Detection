"""Model zoo.

A unified ``ModelSpec`` interface lets ``train.py`` iterate over heterogeneous
sklearn estimators without special casing inside the training loop.

Active suite (11 models — classical baselines plus two fast ensembles,
all M1 Pro-friendly):
    Classical / linear:  KNN, NearestCentroid, CosineSimilarity, DecisionTree,
                         LDA, QDA, LinearSVC, RidgeClassifier, SGDClassifier
    Ensembles:           AdaBoost, HistGradientBoosting
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
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

from .config import RANDOM_STATE


# ---------------- Custom estimators --------------------------------------------

class CosineSimilarityClassifier(BaseEstimator, ClassifierMixin):
    """Predicts the class whose L2-normalised mean vector is most cosine-similar
    to the input. Equivalent to NearestCentroid with a cosine distance metric.
    """

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        centroids = np.stack([X[y == c].mean(axis=0) for c in self.classes_])
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.centroids_ = centroids / norms
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_norm = X / norms
        sim = X_norm @ self.centroids_.T  # (N, C)
        return self.classes_[np.argmax(sim, axis=1)]


@dataclass
class ModelSpec:
    name: str
    builder: Callable[[int], object]
    needs_scaling: bool = False
    family: str = "sklearn"


# ---------------- Classical / linear builders ----------------------------------

def _knn(_n_classes: int):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance",
                                      n_jobs=-1)),
    ])


def _nearest_centroid(_n_classes: int):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", NearestCentroid()),
    ])


def _cosine_similarity(_n_classes: int):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CosineSimilarityClassifier()),
    ])


def _decision_tree(_n_classes: int):
    return DecisionTreeClassifier(
        max_depth=None, class_weight="balanced", random_state=RANDOM_STATE,
    )


def _lda(_n_classes: int):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])


def _qda(_n_classes: int):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", QuadraticDiscriminantAnalysis(reg_param=0.1)),
    ])


def _linear_svc(_n_classes: int):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(C=1.0, max_iter=5000, dual="auto",
                          class_weight="balanced",
                          random_state=RANDOM_STATE)),
    ])


def _ridge(_n_classes: int):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RidgeClassifier(alpha=1.0, class_weight="balanced",
                                 random_state=RANDOM_STATE)),
    ])


def _sgd(_n_classes: int):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SGDClassifier(loss="log_loss", alpha=1e-4,
                              max_iter=1000, tol=1e-3, early_stopping=True,
                              class_weight="balanced",
                              n_jobs=-1, random_state=RANDOM_STATE)),
    ])


# ---------------- Ensemble builders --------------------------------------------

def _adaboost(_n_classes: int):
    return AdaBoostClassifier(
        n_estimators=200, learning_rate=0.5, random_state=RANDOM_STATE,
    )


def _hist_gbm(_n_classes: int):
    return HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.05, max_depth=None, max_leaf_nodes=63,
        l2_regularization=0.0, random_state=RANDOM_STATE,
    )


# ---------------- Public registry ----------------------------------------------

MODEL_SPECS: list[ModelSpec] = [
    # --- Classical / linear ---
    ModelSpec("KNN", _knn, needs_scaling=True),
    ModelSpec("NearestCentroid", _nearest_centroid, needs_scaling=True),
    ModelSpec("CosineSimilarity", _cosine_similarity, needs_scaling=True),
    ModelSpec("DecisionTree", _decision_tree),
    ModelSpec("LDA", _lda, needs_scaling=True),
    ModelSpec("QDA", _qda, needs_scaling=True),
    ModelSpec("LinearSVC", _linear_svc, needs_scaling=True),
    ModelSpec("RidgeClassifier", _ridge, needs_scaling=True),
    ModelSpec("SGDClassifier", _sgd, needs_scaling=True),
    # --- Ensembles ---
    ModelSpec("AdaBoost", _adaboost),
    ModelSpec("HistGradientBoosting", _hist_gbm),
]


def get_model(name: str) -> ModelSpec:
    for spec in MODEL_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown model: {name}")
