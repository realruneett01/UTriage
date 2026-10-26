"""Tests for MlNdtNet neural network architecture and parameter counting."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure local workspace src is prioritized over external paths
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        sys.modules["utriage"].__path__.insert(0, _pkg_dir)
if "utriage.ml" in sys.modules:
    _ml_dir = str(Path(_SRC) / "utriage" / "ml")
    if _ml_dir not in sys.modules["utriage.ml"].__path__:
        sys.modules["utriage.ml"].__path__.insert(0, _ml_dir)

import pytest  # noqa: E402
import torch  # noqa: E402

from utriage.ml.models import BasicBlock, MlNdtNet, count_params  # noqa: E402


@pytest.mark.parametrize("spatial_size", [256, 128])
def test_output_shapes(spatial_size: int) -> None:
    """Output shapes are (B,) for logit and size across input resolutions."""
    model = MlNdtNet(width=32)
    model.eval()
    batch_size = 2
    x = torch.randn(batch_size, 1, spatial_size, spatial_size)
    logit, size = model(x)
    assert logit.shape == (batch_size,)
    assert size.shape == (batch_size,)


def test_count_params_range() -> None:
    """Parameter count for width 32 is between 0.5M and 6M."""
    model = MlNdtNet(width=32)
    total = count_params(model)
    assert 500_000 <= total <= 6_000_000


def test_gradient_flows_to_stem() -> None:
    """Gradients propagate back to the stem conv weights."""
    model = MlNdtNet(width=32)
    model.train()
    x = torch.randn(2, 1, 128, 128)
    logit, size = model(x)
    loss = logit.sum() + size.sum()
    loss.backward()
    stem_conv = model.stem[0]
    assert stem_conv.weight.grad is not None
    assert torch.any(stem_conv.weight.grad != 0.0)


def test_eval_mode_deterministic() -> None:
    """Eval mode produces identical outputs for identical inputs."""
    model = MlNdtNet(width=32, dropout=0.5)
    model.eval()
    x = torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        logit1, size1 = model(x)
        logit2, size2 = model(x)
    assert torch.allclose(logit1, logit2)
    assert torch.allclose(size1, size2)


def test_train_mode_dropout_different() -> None:
    """Train mode with dropout produces stochastic outputs."""
    torch.manual_seed(42)
    model = MlNdtNet(width=32, dropout=0.5)
    model.train()
    x = torch.randn(2, 1, 128, 128)
    logit1, _ = model(x)
    logit2, _ = model(x)
    assert not torch.allclose(logit1, logit2)


@pytest.mark.parametrize("stride", [1, 2])
def test_basic_block_shapes(stride: int) -> None:
    """BasicBlock outputs expected tensor shapes with stride 1 and 2."""
    out_ch = 64 if stride == 2 else 32
    block = BasicBlock(in_channels=32, out_channels=out_ch, stride=stride)
    x = torch.randn(2, 32, 16, 16)
    out = block(x)
    expected_hw = 8 if stride == 2 else 16
    assert out.shape == (2, out_ch, expected_hw, expected_hw)
