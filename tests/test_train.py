"""Tests for training loop and utilities (Phase 3 ML-NDT)."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure local worktree src/ takes precedence over any installed package
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

from utriage.ml.train import (  # noqa: E402
    TrainConfig,
    TrainResult,
    loss_fn,
    pick_device,
    predict,
    set_seed,
    train_model,
)


class TinyConvNet(nn.Module):
    """Tiny two-head model for unit testing."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.clf = nn.Linear(8, 1)
        self.reg = nn.Sequential(nn.Linear(8, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = torch.relu(self.conv(x))
        pooled = self.pool(feat).flatten(1)
        logit = self.clf(pooled).squeeze(-1)
        size = self.reg(pooled).squeeze(-1)
        return logit, size


class FlatValNet(nn.Module):
    """Zero-learning model with constant output to test flat validation loss."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        zeros = (self.bias * 0.0).expand(b)
        return zeros, zeros


class ToyUTDataset(Dataset):
    """Toy dataset with 64 1x16x16 patches where flaw = mean of x > 0."""

    def __init__(self, n_items: int = 64, seed: int = 42) -> None:
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        half = n_items // 2
        pos = torch.randn(half, 1, 16, 16, generator=rng) + 0.8
        neg = torch.randn(n_items - half, 1, 16, 16, generator=rng) - 0.8
        self.x = torch.cat([pos, neg], dim=0)
        means = self.x.view(n_items, -1).mean(dim=1)
        self.flaws = (means > 0).float()
        clamped = torch.clamp(means, 0.1, 1.0)
        self.sizes = torch.where(self.flaws > 0.5, clamped, torch.zeros_like(means))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.flaws[idx], self.sizes[idx]


def test_pick_device() -> None:
    assert pick_device("cpu") == torch.device("cpu")
    auto_dev = pick_device("auto")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert auto_dev.type == expected


@pytest.mark.parametrize(
    ("size_pred_val", "flaw_val"),
    [(0.0, 0.0), (10.0, 0.0), (100.0, 0.0)],
)
def test_loss_fn_ignores_size_on_unflawed_items(size_pred_val: float, flaw_val: float) -> None:
    logit = torch.tensor([0.2, -0.4])
    flaw = torch.tensor([flaw_val, 0.0])
    size = torch.tensor([0.0, 0.0])
    pred1 = torch.tensor([0.0, 0.0])
    pred2 = torch.tensor([size_pred_val, size_pred_val])
    loss1 = loss_fn(logit, pred1, flaw, size, w=1.0)
    loss2 = loss_fn(logit, pred2, flaw, size, w=1.0)
    assert torch.allclose(loss1, loss2)


def test_train_model_decreases_loss_and_high_accuracy() -> None:
    ds = ToyUTDataset(64, seed=42)
    model = TinyConvNet()
    cfg = TrainConfig(epochs=15, batch_size=16, lr=0.01, device="cpu", seed=42)
    res = train_model(model, ds, ds, cfg)
    assert isinstance(res, TrainResult)
    assert res.history[-1]["train_loss"] < res.history[0]["train_loss"]
    assert res.history[-1]["val_acc"] > 0.8
    assert res.best_val_loss <= res.history[0]["val_loss"]


def test_early_stopping_stops_when_val_loss_flat() -> None:
    ds = ToyUTDataset(32, seed=42)
    model = FlatValNet()
    cfg = TrainConfig(epochs=6, batch_size=16, lr=0.001, patience=1, device="cpu", seed=42)
    res = train_model(model, ds, ds, cfg)
    assert len(res.history) < 6
    assert len(res.history) == 2


def test_predict_returns_arrays_in_range() -> None:
    ds = ToyUTDataset(64, seed=42)
    model = TinyConvNet()
    cfg = TrainConfig(batch_size=16, device="cpu")
    probs, sizes = predict(model, ds, cfg)
    assert isinstance(probs, np.ndarray)
    assert isinstance(sizes, np.ndarray)
    assert len(probs) == len(ds)
    assert len(sizes) == len(ds)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.all((sizes >= 0.0) & (sizes <= 1.0))


def test_set_seed_makes_runs_identical() -> None:
    ds = ToyUTDataset(32, seed=7)
    cfg = TrainConfig(epochs=3, batch_size=16, lr=0.01, device="cpu", seed=99)

    set_seed(99)
    m1 = TinyConvNet()
    r1 = train_model(m1, ds, ds, cfg)

    set_seed(99)
    m2 = TinyConvNet()
    r2 = train_model(m2, ds, ds, cfg)

    losses1 = [row["train_loss"] for row in r1.history]
    losses2 = [row["train_loss"] for row in r2.history]
    assert losses1 == pytest.approx(losses2)
    for p1, p2 in zip(m1.parameters(), m2.parameters(), strict=True):
        assert torch.allclose(p1, p2)
