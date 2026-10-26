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


def _make_window_net(
    in_channels: int = 1,
    width: int = 16,
    n_classes: int = 3,
    dropout: float = 0.0,
) -> WindowNet:
    """Construct a WindowNet instance with default width=16 and n_classes=3."""
    return WindowNet(in_channels=in_channels, width=width, n_classes=n_classes, dropout=dropout)


def _assert_shapes(
    logits: torch.Tensor,
    sizes: torch.Tensor,
    batch_size: int,
    n_classes: int = 3,
) -> None:
    """Assert WindowNet output logits and sizes tensor shapes."""
    assert logits.shape == (batch_size, n_classes)
    assert sizes.shape == (batch_size,)


def _assert_close(
    l1: torch.Tensor,
    s1: torch.Tensor,
    l2: torch.Tensor,
    s2: torch.Tensor,
) -> None:
    """Assert output logits and sizes are elementwise close."""
    assert torch.allclose(l1, l2)
    assert torch.allclose(s1, s2)


def make_toy_dataset(
    n: int = 64,
    in_channels: int = 1,
    seed: int = 42,
) -> tuple[WindowDataset, WindowDataset]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, in_channels, 128)).astype(np.float32)
    y = rng.choice([0, 1, 2], size=n)
    size_mm = rng.uniform(0.5, 2.5, size=n).astype(np.float32)
    ds_train = WindowDataset(x=x, y=y, size_mm=size_mm)
    ds_val = WindowDataset(x=x[: n // 4], y=y[: n // 4], size_mm=size_mm[: n // 4])
    return ds_train, ds_val


@pytest.fixture
def dropout_net() -> WindowNet:
    return _make_window_net(width=16, dropout=0.5)


@pytest.mark.parametrize("in_channels", [1, 9])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_window_net_shapes(in_channels: int, batch_size: int) -> None:
    """WindowNet outputs (B, n_classes) logits and (B,) sizes for C=1 and C=9."""
    n_classes = 3
    net = _make_window_net(in_channels=in_channels, width=16, n_classes=n_classes)
    net.eval()
    x = torch.randn(batch_size, in_channels, 128)
    logits, sizes = net(x)
    _assert_shapes(logits, sizes, batch_size, n_classes)


def test_count_params_under_1m() -> None:
    """WindowNet with width 32 has parameter count strictly under 1M."""
    net = _make_window_net(width=32)
    total = count_params(net)
    assert 50_000 < total < 1_000_000


@pytest.mark.parametrize(
    ("is_train", "expected_deterministic"),
    [(False, True), (True, False)],
    ids=["eval_deterministic", "train_stochastic"],
)
def test_dropout_net_mode(
    dropout_net: WindowNet,
    is_train: bool,
    expected_deterministic: bool,
) -> None:
    """WindowNet with dropout is deterministic in eval mode and stochastic in train mode."""
    x = torch.randn(2, 1, 128)
    if is_train:
        torch.manual_seed(42)
        dropout_net.train()
        l1, _ = dropout_net(x)
        l2, _ = dropout_net(x)
        assert not torch.allclose(l1, l2)
    else:
        dropout_net.eval()
        with torch.no_grad():
            l1, s1 = dropout_net(x)
            l2, s2 = dropout_net(x)
        _assert_shapes(l1, s1, 2)
        _assert_shapes(l2, s2, 2)
        _assert_close(l1, s1, l2, s2)


@pytest.mark.parametrize(
    ("logits_vals", "expected_val"),
    [
        ([1.0, 2.0], 2.0 - 1.0),
        ([-0.5, 1.5], 1.5 - (-0.5)),
        ([3.0, -1.0], -1.0 - 3.0),
        (
            [1.0, 2.5, 0.5],
            2.5 - float(np.log(np.exp(1.0) + np.exp(0.5))),
        ),
    ],
    ids=["binary_pos", "binary_neg", "binary_large_neg", "three_classes"],
)
def test_flaw_logit_formula(logits_vals: list[float], expected_val: float) -> None:
    """flaw_logit matches manual logit contrast and sigmoid matches softmax probability."""
    logits = torch.tensor([logits_vals])
    computed = flaw_logit(logits)
    expected = torch.tensor([expected_val])
    assert torch.allclose(computed, expected)
    prob_flaw = torch.sigmoid(computed)
    softmax_flaw = torch.softmax(logits, dim=-1)[0, 1]
    assert torch.allclose(prob_flaw, softmax_flaw)


@pytest.mark.parametrize("x_dim", [2, 3])
@pytest.mark.parametrize(("idx", "expected_flaw"), [(1, 1.0), (0, 0.0)])
def test_window_dataset_items(x_dim: int, idx: int, expected_flaw: float) -> None:
    """WindowDataset returns properly typed tensors and maps flaw to y==1."""
    n = 8
    x_raw = np.random.randn(n, 128).astype(np.float32)
    if x_dim == 3:
        x_raw = x_raw[:, np.newaxis, :]
    y = np.array([0, 1, 2, 1, 0, 0, 1, 2])
    size_mm = np.linspace(0.5, 2.5, n).astype(np.float32)
    ds = WindowDataset(x=x_raw, y=y, size_mm=size_mm)

    assert len(ds) == n
    item_x, item_flaw, item_size = ds[idx]
    assert item_x.shape == (1, 128)
    assert item_flaw.item() == expected_flaw
    assert np.isclose(item_size.item(), size_mm[idx])


@pytest.mark.parametrize("in_channels", [1, 9])
def test_trainer_adapter_train_model(in_channels: int) -> None:
    """TrainerAdapter executes cleanly in train_model on toy WindowDataset."""
    ds_train, ds_val = make_toy_dataset(n=64, in_channels=in_channels, seed=42)
    net = _make_window_net(in_channels=in_channels, width=16, n_classes=3)
    adapter = TrainerAdapter(net)
    cfg = TrainConfig(epochs=2, batch_size=16, device="cpu", amp=False)

    result = train_model(adapter, ds_train, ds_val, cfg)
    assert isinstance(result, TrainResult)
    assert result.best_epoch >= 1
    assert len(result.history) == 2
