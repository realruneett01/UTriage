"""Unsupervised anomaly detector trained exclusively on flaw-free ultrasonic data.

Includes ConvAE1D for A-scan windows, ConvAE2D for B-scan images, automatic
wall echo gating, moving-average anomaly scoring, empirical quantile thresholding,
and autoencoder training routines.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def count_params(model: nn.Module) -> int:
    """Return total number of parameters in a neural network module."""
    return sum(p.numel() for p in model.parameters())


class ConvAE1D(nn.Module):
    """Small 1D convolutional autoencoder for ultrasonic A-scan windows.

    Uses 3 conv downsampling layers and 3 transposed conv upsampling layers,
    maintaining fewer than 50,000 parameters for efficient CPU inference.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c2, c3, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(c3, c2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(c2, c1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(c1, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct 1D A-scan windows with shape (B, 1, L)."""
        return self.decoder(self.encoder(x))


class ConvAE2D(nn.Module):
    """Small 2D convolutional autoencoder for ultrasonic B-scan images.

    Uses stride-2 convolutions and transposed convolutions, maintaining
    fewer than 200,000 parameters for low-latency inference.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c1, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct 2D B-scan images with shape (B, 1, H, W)."""
        return self.decoder(self.encoder(x))


def gate_walls(
    ascan: np.ndarray,
    front_idx: int,
    back_idx: int,
    guard: int = 0,
) -> np.ndarray:
    """Return a mask that zeroes samples within guard distance of wall echoes.

    Parameters
    ----------
    ascan : np.ndarray
        A-scan signal array of shape (..., L).
    front_idx : int
        Index of the detected front-wall echo.
    back_idx : int
        Index of the detected back-wall echo.
    guard : int, optional
        Sample half-width around echoes to zero out (default 0).

    Returns
    -------
    np.ndarray
        Float32 mask with wall regions zeroed out and bulk material set to 1.0.
    """
    mask = np.ones_like(ascan, dtype=np.float32)
    length = ascan.shape[-1]
    if front_idx >= 0:
        f_start = max(0, front_idx - guard)
        f_end = min(length, front_idx + guard + 1)
        mask[..., f_start:f_end] = 0.0
    if back_idx >= 0:
        b_start = max(0, back_idx - guard)
        b_end = min(length, back_idx + guard + 1)
        mask[..., b_start:b_end] = 0.0
    return mask


def _smooth_error(err: torch.Tensor, smooth_window: int = 5) -> torch.Tensor:
    """Apply moving-average filter of specified window size along signal dimensions."""
    pad = smooth_window // 2
    if err.ndim == 3:
        w_1d = 1.0 / float(smooth_window)
        k_1d = torch.full((1, 1, smooth_window), w_1d, device=err.device, dtype=err.dtype)
        return F.conv1d(err, k_1d, padding=pad)
    if err.ndim == 4:
        w_2d = 1.0 / float(smooth_window * smooth_window)
        k_2d = torch.full(
            (1, 1, smooth_window, smooth_window), w_2d, device=err.device, dtype=err.dtype
        )
        return F.conv2d(err, k_2d, padding=pad)
    msg = f"Unsupported input tensor rank: {err.ndim}"
    raise ValueError(msg)


def anomaly_score(
    model: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor | np.ndarray | None = None,
    smooth_window: int = 5,
) -> torch.Tensor:
    """Compute per-sample max moving-average squared reconstruction error.

    Parameters
    ----------
    model : nn.Module
        Trained autoencoder model.
    x : torch.Tensor
        Input tensor of shape (B, 1, L) or (B, 1, H, W).
    mask : torch.Tensor | np.ndarray | None, optional
        Mask zeroing wall echoes (1 for bulk material, 0 for wall guards).
    smooth_window : int, optional
        Moving-average filter window size (default 5).

    Returns
    -------
    torch.Tensor
        Per-sample maximum smoothed error of shape (B,).
    """
    model.eval()
    with torch.no_grad():
        recon = model(x)
        sq_err = (x - recon) ** 2
        mask_t = None
        if mask is not None:
            mask_t = torch.as_tensor(mask, device=x.device, dtype=sq_err.dtype)
            sq_err = sq_err * mask_t
        smoothed = _smooth_error(sq_err, smooth_window=smooth_window)
        if mask_t is not None:
            smoothed = smoothed * mask_t
        return smoothed.view(x.shape[0], -1).amax(dim=-1)


def fit_threshold(clean_scores: np.ndarray, false_call_rate: float) -> float:
    """Calculate the (1 - rate) empirical quantile threshold on clean scores.

    Parameters
    ----------
    clean_scores : np.ndarray
        Array of scores computed on flaw-free validation data.
    false_call_rate : float
        Target false call rate in [0.0, 1.0].

    Returns
    -------
    float
        Threshold value above which an anomaly is flagged.
    """
    if false_call_rate < 0.0 or false_call_rate > 1.0:
        msg = f"false_call_rate must be in [0.0, 1.0], got {false_call_rate}"
        raise ValueError(msg)
    arr = np.asarray(clean_scores, dtype=float).ravel()
    if arr.size == 0:
        msg = "clean_scores cannot be empty"
        raise ValueError(msg)
    q = 1.0 - false_call_rate
    return float(np.quantile(arr, q))


def _resolve_device(device: torch.device | str | None) -> torch.device:
    """Resolve compute device, defaulting to CUDA if available, else CPU."""
    if device is not None:
        return torch.device(device)
    dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(dev_str)
    logger.info("Autoencoder training device selected: %s", dev)
    return dev


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    dev: torch.device,
) -> float:
    """Execute a single autoencoder training epoch and return average MSE loss."""
    model.train()
    total_loss = 0.0
    batch_count = 0
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(dev)
        optimizer.zero_grad()
        recon = model(x)
        loss = F.mse_loss(recon, x)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        batch_count += 1
    return total_loss / max(1, batch_count)


def train_autoencoder(
    model: nn.Module,
    clean_loader: DataLoader[Any],
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device | str | None = None,
) -> list[float]:
    """Train autoencoder on flaw-free data with MSE loss and AdamW optimizer.

    Parameters
    ----------
    model : nn.Module
        Autoencoder to train.
    clean_loader : DataLoader
        DataLoader providing flaw-free training batches.
    epochs : int, optional
        Number of training epochs (default 10).
    lr : float, optional
        Learning rate (default 1e-3).
    weight_decay : float, optional
        Weight decay coefficient (default 1e-4).
    device : torch.device | str | None, optional
        Target device for training.

    Returns
    -------
    list[float]
        Average MSE training loss per epoch.
    """
    dev = _resolve_device(device)
    model.to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses: list[float] = []
    for _ in range(epochs):
        epoch_loss = _train_one_epoch(model, clean_loader, optimizer, dev)
        losses.append(epoch_loss)
    return losses
