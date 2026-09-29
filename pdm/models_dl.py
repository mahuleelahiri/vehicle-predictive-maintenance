"""Deep learning sequence models (PyTorch): LSTM, GRU and Transformer encoder."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import SEED


class RNNClassifier(nn.Module):
    def __init__(self, n_features: int, cell: str = "lstm", hidden: int = 64, layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(n_features, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, heads: int = 4, layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=128, dropout=dropout,
                                         batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        h = self.encoder(self.pos(self.proj(x)))
        return self.head(h.mean(dim=1)).squeeze(-1)


def build_dl_model(name: str, n_features: int) -> nn.Module:
    if name in ("lstm", "gru"):
        return RNNClassifier(n_features, cell=name)
    if name == "transformer":
        return TransformerClassifier(n_features)
    raise ValueError(f"Unknown DL model: {name}")


def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train_dl_model(model: nn.Module, X_tr, y_tr, X_val, y_val, epochs: int = 12,
                   batch_size: int = 256, lr: float = 1e-3, patience: int = 3, verbose=True):
    """Train with weighted BCE; keep the epoch with the best validation PR-AUC."""
    torch.manual_seed(SEED)
    device = get_device()
    model.to(device)
    pos_weight = torch.tensor(float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)),
                              dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                        batch_size=batch_size, shuffle=True)
    best_ap, best_state, bad = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(xb)
        ap = average_precision_score(y_val, predict_proba_dl(model, X_val))
        if verbose:
            print(f"    epoch {epoch:2d}  loss={total / len(X_tr):.4f}  val PR-AUC={ap:.4f}")
        if ap > best_ap:
            best_ap, best_state, bad = ap, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.cpu()


@torch.no_grad()
def predict_proba_dl(model: nn.Module, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    probs = [torch.sigmoid(model(torch.from_numpy(X[i:i + batch_size]).to(device))).cpu().numpy()
             for i in range(0, len(X), batch_size)]
    return np.concatenate(probs)
