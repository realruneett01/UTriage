"""Tests for self-supervised masked-patch denoising pretraining."""

from __future__ import annotations

import copy
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
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from utriage.ml.anomaly import ConvAE2D  # noqa: E402
from utriage.ml.pretrain import (  # noqa: E402
    PretrainConfig,
    encoder_state,
    mask_patches,
    masked_loss,
    pretrain,
    pretrain_epoch,
)


def _make_loader(
    x: torch.Tensor,
    with_labels: bool = False,
    batch_size: int = 4,
) -> DataLoader:
    """Create a DataLoader yielding tensors or (x, y) tuples."""
    if with_labels:
        y = torch.zeros(len(x), dtype=torch.long)
        ds = TensorDataset(x, y)
    else:
        ds = TensorDataset(x)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def test_pretrain_config_defaults_and_immutability() -> None:
    """PretrainConfig dataclass has expected defaults and is frozen."""
    cfg = PretrainConfig()
    assert cfg.mask_ratio == 0.5
    assert cfg.patch == 16
    assert cfg.epochs == 1
    assert cfg.lr == 1e-3
    assert cfg.seed == 0

    with pytest.raises(AttributeError):
        cfg.epochs = 5  # type: ignore[misc]


def test_mask_patches_ratio_and_zeroing() -> None:
    """mask_patches zeroes exactly the masked patches and matches ratio."""
    gen = torch.Generator().manual_seed(123)
    x = torch.ones(2, 1, 64, 64)
    cfg = PretrainConfig(mask_ratio=0.5, patch=16)

    masked, mask = mask_patches(x, cfg, generator=gen)

    assert masked.shape == (2, 1, 64, 64)
    assert mask.shape == (2, 1, 64, 64)
    assert abs(mask.float().mean().item() - 0.5) < 1e-4

    is_masked = mask > 0.5
    assert torch.all(masked[is_masked] == 0.0)
    assert torch.all(masked[~is_masked] == 1.0)


def test_mask_patches_invalid_dimensions() -> None:
    """mask_patches rejects spatial dimensions not divisible by patch."""
    x = torch.zeros(1, 1, 50, 50)
    cfg = PretrainConfig(patch=16)
    with pytest.raises(ValueError, match="divisible by patch size"):
        mask_patches(x, cfg)


def test_masked_loss_ignores_unmasked_pixels() -> None:
    """masked_loss computes MSE only over masked pixels."""
    target = torch.randn(2, 1, 32, 32)
    mask = torch.zeros(2, 1, 32, 32)
    mask[:, :, :16, :16] = 1.0  # mask top-left patch

    recon_a = target.clone()
    assert masked_loss(recon_a, target, mask).item() == 0.0

    # Perturb unmasked pixels only: loss should remain 0.0
    recon_b = target.clone()
    recon_b[:, :, 16:, 16:] += 50.0
    loss_b = masked_loss(recon_b, target, mask)
    assert loss_b.item() == 0.0

    # Perturb masked pixels: loss must increase
    recon_c = target.clone()
    recon_c[:, :, :16, :16] += 2.0
    loss_c = masked_loss(recon_c, target, mask)
    assert abs(loss_c.item() - 4.0) < 1e-4


@pytest.mark.parametrize("with_labels", [False, True])
def test_pretrain_epoch_handles_loader_formats(with_labels: bool) -> None:
    """pretrain_epoch succeeds for both tensor-only and (x, y) loaders."""
    x = torch.randn(4, 1, 32, 32)
    loader = _make_loader(x, with_labels=with_labels, batch_size=2)
    model = ConvAE2D()
    cfg = PretrainConfig(patch=16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loss = pretrain_epoch(model, loader, opt, cfg, device="cpu")
    assert isinstance(loss, float)
    assert loss > 0.0


def test_pretrain_three_epochs_reduce_loss() -> None:
    """Three epochs on a fixed batch reduce reconstruction loss."""
    x = torch.ones(4, 1, 32, 32)
    loader = _make_loader(x, with_labels=False, batch_size=4)
    model = ConvAE2D()
    cfg = PretrainConfig(mask_ratio=0.5, patch=16, epochs=3, lr=1e-3, seed=42)

    losses = pretrain(model, loader, cfg, device="cpu")

    assert len(losses) == 3
    assert losses[-1] < losses[0]


def test_pretrain_deterministic_for_seed() -> None:
    """Same seed yields identical epoch losses for identical initial models."""
    torch.manual_seed(77)
    x = torch.randn(4, 1, 32, 32)
    loader = _make_loader(x, with_labels=False, batch_size=4)

    torch.manual_seed(10)
    model_a = ConvAE2D()
    model_b = ConvAE2D()
    model_b.load_state_dict(copy.deepcopy(model_a.state_dict()))

    cfg = PretrainConfig(epochs=2, seed=123)
    losses_a = pretrain(model_a, loader, cfg, device="cpu")
    losses_b = pretrain(model_b, loader, cfg, device="cpu")

    assert losses_a == losses_b


def test_encoder_state_extraction_and_loading() -> None:
    """encoder_state returns valid state_dict that can initialize another encoder."""
    model_src = ConvAE2D()
    state = encoder_state(model_src)

    assert isinstance(state, dict)
    assert "0.weight" in state

    model_dst = ConvAE2D()
    model_dst.encoder.load_state_dict(state)

    for p_src, p_dst in zip(
        model_src.encoder.parameters(),
        model_dst.encoder.parameters(),
        strict=True,
    ):
        assert torch.allclose(p_src, p_dst)
