"""Masked-patch denoising self-supervised pretraining on unlabelled ultrasonic images."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from utriage.ml.train import pick_device, set_seed

if TYPE_CHECKING:
    import torch.nn as nn
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PretrainConfig:
    """Hyperparameters for self-supervised masked autoencoder pretraining."""

    mask_ratio: float = 0.5
    patch: int = 16
    epochs: int = 1
    lr: float = 1e-3
    seed: int = 0


def mask_patches(
    x: torch.Tensor,
    cfg: PretrainConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask square patches of input images at specified ratio.

    Parameters
    ----------
    x : torch.Tensor
        Input image tensor of shape (B, C, H, W).
    cfg : PretrainConfig
        Configuration containing patch size and mask ratio.
    generator : torch.Generator | None, optional
        RNG generator for reproducible random masking.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of (masked_x, mask) where mask is 1.0 at masked pixels and 0.0 elsewhere.
    """
    b, _, h, w = x.shape
    p = cfg.patch
    if h % p != 0 or w % p != 0:
        msg = f"Spatial dimensions ({h}, {w}) must be divisible by patch size {p}"
        raise ValueError(msg)
    num_h, num_w = h // p, w // p
    num_patches = num_h * num_w
    num_masked = int(round(num_patches * cfg.mask_ratio))
    rand_dev = generator.device if generator is not None else x.device
    rand = torch.rand((b, num_patches), generator=generator, device=rand_dev).to(x.device)
    order = torch.argsort(rand, dim=-1)
    patch_mask = (order < num_masked).to(dtype=x.dtype).view(b, 1, num_h, num_w)
    mask = patch_mask.repeat_interleave(p, dim=-2).repeat_interleave(p, dim=-1)
    masked = x * (1.0 - mask)
    return masked, mask


def masked_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Calculate mean squared error over masked pixels only.

    Parameters
    ----------
    recon : torch.Tensor
        Reconstructed image from autoencoder.
    target : torch.Tensor
        Original unmasked image.
    mask : torch.Tensor
        Binary mask indicating masked pixels with 1.0 (or True).

    Returns
    -------
    torch.Tensor
        MSE loss over masked pixels only.
    """
    mask_bool = mask > 0.5 if mask.dtype.is_floating_point else mask.bool()
    if not mask_bool.any():
        return torch.zeros((), device=recon.device, dtype=recon.dtype)
    return F.mse_loss(recon[mask_bool], target[mask_bool])


def _resolve_device(device: torch.device | str | None) -> torch.device:
    """Resolve compute device, defaulting to pick_device('auto')."""
    if device is not None:
        return torch.device(device)
    dev = pick_device("auto")
    logger.info("Pretrain compute device: %s", dev)
    return dev


def _extract_x(batch: Any) -> torch.Tensor:
    """Extract input tensor from tensor or (x, y) tuple, ignoring labels."""
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def pretrain_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    opt: torch.optim.Optimizer,
    cfg: PretrainConfig,
    device: torch.device | str | None = None,
) -> float:
    """Train model for one epoch on unlabelled masked images and return mean loss."""
    dev = _resolve_device(device)
    model.train()
    model.to(dev)
    use_amp = dev.type == "cuda"
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp
        else contextlib.nullcontext()
    )
    total_loss, total_count = 0.0, 0
    for batch in loader:
        x = _extract_x(batch).to(dev)
        masked, mask = mask_patches(x, cfg)
        opt.zero_grad()
        with amp_ctx:
            recon = model(masked)
            loss = masked_loss(recon, x, mask)
        loss.backward()
        opt.step()
        total_loss += float(loss.item()) * len(x)
        total_count += len(x)
    return total_loss / max(1, total_count)


def pretrain(
    model: nn.Module,
    loader: DataLoader[Any],
    cfg: PretrainConfig | None = None,
    device: torch.device | str | None = None,
) -> list[float]:
    """Execute masked autoencoder pretraining for configured epochs.

    Parameters
    ----------
    model : nn.Module
        Autoencoder model to pretrain.
    loader : DataLoader
        DataLoader yielding unlabelled images or (x, y) tuples.
    cfg : PretrainConfig | None, optional
        Pretraining hyperparameters (defaults to PretrainConfig()).
    device : torch.device | str | None, optional
        Target device for training.

    Returns
    -------
    list[float]
        List of average training losses per epoch.
    """
    config = cfg if cfg is not None else PretrainConfig()
    set_seed(config.seed)
    dev = _resolve_device(device)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    losses: list[float] = []
    for _ in range(config.epochs):
        losses.append(pretrain_epoch(model, loader, opt, config, dev))
    return losses


def encoder_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract encoder state_dict from autoencoder for downstream fine-tuning.

    Parameters
    ----------
    model : nn.Module
        Autoencoder containing an encoder submodule.

    Returns
    -------
    dict[str, torch.Tensor]
        State dict of the encoder component.
    """
    if hasattr(model, "encoder"):
        return model.encoder.state_dict()
    return model.state_dict()
