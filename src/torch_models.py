"""PyTorch implementations of CNN1D / LSTM / RNN with a sklearn-compatible
``TorchClassifier`` wrapper so they can be dropped into the same evaluation
loop as the gradient-boosted trees.

Note: the inputs are tabular features, not literal sequences. For CNN/LSTM/RNN
we treat the flat feature vector as a 1-D "sequence" so the architectures
have something to consume. This is a known caveat documented in the
progress report; the proposal mentions sequence/k-mer features for a later
iteration.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _device(n_features: int = 0) -> torch.device:
    """CUDA when available; MPS only when feature count is divisible by 8 to
    dodge the AdaptiveAvgPool input-divisibility crash on Apple Silicon. Other-
    wise fall back to CPU. Set MVP_FORCE_CPU=1 to force CPU regardless."""
    import os
    if os.environ.get("MVP_FORCE_CPU") == "1":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and (n_features == 0 or n_features % 8 == 0):
        return torch.device("mps")
    return torch.device("cpu")


class CNN1D(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        # Use a fixed pool target of 8 only when input divides cleanly; else
        # fall back to global pooling. The MPS device guard already steers
        # off-divisible inputs to CPU, but this keeps the architecture sane
        # in either case.
        pool_size = 8 if n_features % 8 == 0 else 1
        flat_dim = 64 * pool_size
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(pool_size),
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.unsqueeze(1))  # (B, 1, F)


class LSTMNet(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F) -> (B, F, 1)
        out, _ = self.lstm(x.unsqueeze(-1))
        h = out.mean(dim=1)
        return self.head(h)


class RNNNet(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: int = 64):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden, num_layers=1,
                          batch_first=True, nonlinearity="tanh")
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x.unsqueeze(-1))
        h = out.mean(dim=1)
        return self.head(h)


class TorchClassifier(BaseEstimator, ClassifierMixin):
    """Thin sklearn-compatible wrapper.

    Standard-scales features (assumes float32 already, but scales to be safe),
    trains with Adam + cross-entropy, exposes ``predict`` and ``predict_proba``.
    """

    def __init__(
        self,
        model_factory: Callable[[int, int], nn.Module],
        epochs: int = 25,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        verbose: bool = False,
    ):
        self.model_factory = model_factory
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose

    def _to_tensor(self, X) -> torch.Tensor:
        return torch.as_tensor(np.asarray(X, dtype=np.float32))

    def fit(self, X, y):
        X = self._to_tensor(X)
        y = torch.as_tensor(np.asarray(y, dtype=np.int64))
        self.classes_ = np.unique(y.numpy())
        n_classes = int(self.classes_.size)
        n_features = X.shape[1]

        self.scaler_ = StandardScaler()
        X_np = self.scaler_.fit_transform(X.numpy())
        X = torch.as_tensor(X_np, dtype=torch.float32)

        self.device_ = _device(n_features=n_features)
        self.model_ = self.model_factory(n_features, n_classes).to(self.device_)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                            drop_last=False)
        self.model_.train()
        for ep in range(self.epochs):
            running = 0.0
            n = 0
            for xb, yb in loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                opt.zero_grad()
                logits = self.model_(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                running += loss.item() * xb.size(0)
                n += xb.size(0)
            if self.verbose:
                print(f"  epoch {ep+1}/{self.epochs}  loss={running/n:.4f}")
        return self

    def _forward(self, X) -> np.ndarray:
        X = self._to_tensor(X).numpy()
        X = self.scaler_.transform(X)
        X = torch.as_tensor(X, dtype=torch.float32)
        self.model_.eval()
        out_chunks: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, X.shape[0], self.batch_size):
                xb = X[i:i + self.batch_size].to(self.device_)
                logits = self.model_(xb)
                out_chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(out_chunks, axis=0)

    def predict_proba(self, X) -> np.ndarray:
        return self._forward(X)

    def predict(self, X) -> np.ndarray:
        proba = self._forward(X)
        return self.classes_[np.argmax(proba, axis=1)]
