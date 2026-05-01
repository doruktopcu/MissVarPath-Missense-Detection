"""Model zoo.

A unified ``ModelSpec`` interface lets ``train.py`` iterate over heterogeneous
estimators (sklearn / XGBoost / LightGBM / CatBoost / PyTorch) without special
casing inside the training loop.

Models included (10 total per the project plan):
    1.  Logistic Regression
    2.  Random Forest
    3.  Extra Trees
    4.  XGBoost
    5.  LightGBM
    6.  CatBoost
    7.  Shallow Neural Network (1 hidden layer MLP)
    8.  CNN 1D            (PyTorch)
    9.  LSTM              (PyTorch)
    10. RNN               (PyTorch)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE


@dataclass
class ModelSpec:
    name: str
    builder: Callable[[int], object]
    needs_scaling: bool = False
    family: str = "sklearn"  # 'sklearn' | 'torch'


# ---------------- Sklearn-style builders ---------------------------------------

def _logreg(_n_classes: int):
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # robust if any sparse-like inputs
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs",
                                    multi_class="auto", n_jobs=-1,
                                    random_state=RANDOM_STATE)),
    ])


def _random_forest(_n_classes: int):
    return RandomForestClassifier(
        n_estimators=500, max_depth=None, n_jobs=-1,
        class_weight="balanced", random_state=RANDOM_STATE,
    )


def _extra_trees(_n_classes: int):
    return ExtraTreesClassifier(
        n_estimators=500, n_jobs=-1, class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def _xgboost(n_classes: int):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob" if n_classes > 2 else "binary:logistic",
        num_class=n_classes if n_classes > 2 else None,
        tree_method="hist",
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def _lightgbm(n_classes: int):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multiclass" if n_classes > 2 else "binary",
        num_class=n_classes if n_classes > 2 else 1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=-1,
    )


def _catboost(n_classes: int):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        loss_function="MultiClass" if n_classes > 2 else "Logloss",
        random_seed=RANDOM_STATE,
        verbose=False,
        thread_count=-1,
    )


def _shallow_nn(_n_classes: int):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=400,
            n_iter_no_change=15,
            tol=1e-5,
            early_stopping=True,
            random_state=RANDOM_STATE,
        )),
    ])


# ---------------- PyTorch builders ---------------------------------------------

def _torch_cnn1d(_n_classes: int):
    from .torch_models import TorchClassifier, CNN1D
    return TorchClassifier(model_factory=CNN1D, epochs=25, batch_size=256, lr=1e-3)


def _torch_lstm(_n_classes: int):
    from .torch_models import TorchClassifier, LSTMNet
    return TorchClassifier(model_factory=LSTMNet, epochs=25, batch_size=256, lr=1e-3)


def _torch_rnn(_n_classes: int):
    from .torch_models import TorchClassifier, RNNNet
    return TorchClassifier(model_factory=RNNNet, epochs=25, batch_size=256, lr=1e-3)


# ---------------- Public registry ----------------------------------------------

MODEL_SPECS: list[ModelSpec] = [
    ModelSpec("LogisticRegression", _logreg, needs_scaling=True),
    ModelSpec("RandomForest", _random_forest),
    ModelSpec("ExtraTrees", _extra_trees),
    ModelSpec("XGBoost", _xgboost),
    ModelSpec("LightGBM", _lightgbm),
    ModelSpec("CatBoost", _catboost),
    ModelSpec("ShallowNN_MLP", _shallow_nn, needs_scaling=True),
    ModelSpec("CNN1D", _torch_cnn1d, needs_scaling=True, family="torch"),
    ModelSpec("LSTM", _torch_lstm, needs_scaling=True, family="torch"),
    ModelSpec("RNN", _torch_rnn, needs_scaling=True, family="torch"),
]


def get_model(name: str) -> ModelSpec:
    for spec in MODEL_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown model: {name}")
