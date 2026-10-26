"""Tests for WindowNet architecture, TrainerAdapter, and WindowDataset."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from utriage.ml.bar_models import (  # noqa: E402
    TrainerAdapter,
    WindowDataset,
    WindowNet,
    count_params,
    flaw_logit,
)
from utriage.ml.train import TrainConfig, TrainResult, train_model  # noqa: E402


@pytest.mark.parametrize("in_channels", [1, 9])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_window_net_shapes(in_channels: int, batch_size: int) -> None:
    """WindowNet outputs (B, n_classes) logits and (B,) sizes for C=1 and C=9."""
    n_classes = 3
    net = WindowNet(in_channels=in_channels, width=16, n_classes=n_classes)
    net.eval()
    x = torch.randn(batch_size, in_channels, 128)
    logits, sizes = net(x)
    assert logits.shape == (batch_size, n_classes)
    assert sizes.shape == (batch_size,)


def test_count_params_under_1m() -> None:
    """WindowNet with width 32 has parameter count strictly under 1M."""
    net = WindowNet(width=32)
    total = count_params(net)
    assert 50_000 < total < 1_000_000


def test_eval_mode_deterministic() -> None:
    """WindowNet outputs are deterministic in eval mode."""
    net = WindowNet(width=16, dropout=0.5)
    net.eval()
    x = torch.randn(2, 1, 128)
    with torch.no_grad():
        l1, s1 = net(x)
        l2, s2 = net(x)
    assert torch.allclose(l1, l2)
    assert torch.allclose(s1, s2)


def test_train_mode_dropout_different() -> None:
    """WindowNet with dropout produces stochastic outputs in train mode."""
    torch.manual_seed(42)
    net = WindowNet(width=16, dropout=0.5)
    net.train()
    x = torch.randn(2, 1, 128)
    l1, _ = net(x)
    l2, _ = net(x)
    assert not torch.allclose(l1, l2)


@pytest.mark.parametrize(
    ("z0", "z1"),
    [(1.0, 2.0), (-0.5, 1.5), (3.0, -1.0)],
)
def test_flaw_logit_equals_binary_logit(z0: float, z1: float) -> None:
    """For n_classes=2, flaw_logit matches manual z1 - z0 log-odds."""
    logits = torch.tensor([[z0, z1]])
    result = flaw_logit(logits)
    expected = torch.tensor([z1 - z0])
    assert torch.allclose(result, expected)


def test_flaw_logit_three_classes() -> None:
    """For n_classes=3, flaw_logit matches z1 - logsumexp(z0, z2)."""
    logits = torch.tensor([[1.0, 2.5, 0.5]])
    computed = flaw_logit(logits)
    expected = 2.5 - torch.log(torch.exp(torch.tensor(1.0)) + torch.exp(torch.tensor(0.5)))
    assert torch.allclose(computed, expected)
    prob_flaw = torch.sigmoid(computed)
    softmax_flaw = torch.softmax(logits, dim=-1)[0, 1]
    assert torch.allclose(prob_flaw, softmax_flaw)


@pytest.mark.parametrize("x_dim", [2, 3])
def test_window_dataset_items(x_dim: int) -> None:
    """WindowDataset returns properly typed tensors and maps flaw to y==1."""
    n = 8
    x_raw = np.random.randn(n, 128).astype(np.float32)
    if x_dim == 3:
        x_raw = x_raw[:, np.newaxis, :]
    y = np.array([0, 1, 2, 1, 0, 0, 1, 2])
    size_mm = np.linspace(0.5, 2.5, n).astype(np.float32)
    ds = WindowDataset(x=x_raw, y=y, size_mm=size_mm)

    assert len(ds) == n
    item_x, item_flaw, item_size = ds[1]
    assert item_x.shape == (1, 128)
    assert item_flaw.item() == 1.0
    assert np.isclose(item_size.item(), size_mm[1])

    _, neg_flaw, _ = ds[0]
    assert neg_flaw.item() == 0.0


def test_trainer_adapter_train_model() -> None:
    """TrainerAdapter executes cleanly in train_model on toy WindowDataset."""
    rng = np.random.default_rng(42)
    n = 64
    x = rng.standard_normal((n, 1, 128)).astype(np.float32)
    y = rng.choice([0, 1, 2], size=n)
    size_mm = rng.uniform(0.5, 2.5, size=n).astype(np.float32)

    ds_train = WindowDataset(x=x, y=y, size_mm=size_mm)
    ds_val = WindowDataset(x=x[:16], y=y[:16], size_mm=size_mm[:16])

    net = WindowNet(in_channels=1, width=16, n_classes=3)
    adapter = TrainerAdapter(net)
    cfg = TrainConfig(epochs=2, batch_size=16, device="cpu", amp=False)

    result = train_model(adapter, ds_train, ds_val, cfg)
    assert isinstance(result, TrainResult)
    assert result.best_epoch >= 1
    assert len(result.history) == 2
