"""Lightweight hybrid flaw detector building blocks (project plan Phase 8 / ADR 0013).

Provides LiteNet (depthwise-separable CNN with <= 100k parameters), physics-informed
data augmentations, unsupervised Ledoit-Wolf Mahalanobis scoring on clean physics features,
and hybrid flag fusion combining supervised and unsupervised detection signals.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.covariance import LedoitWolf

from utriage.ml.physics_features import FEATURE_NAMES, extract_features

# TIME_AXIS identifies the ultrasonic fast-time / depth axis.
# Found in src/utriage/ml/physics_features.py (lines 5-8):
# "Axis 0 (rows): Time / depth axis. In ML-NDT weld PAUT data, rows correspond to
#  A-scan time samples. Max-pooling along axis 0 acts as envelope detection over
#  ultrasonic wave cycles."
# In 2D images [H, W] and PyTorch image tensors [..., H, W], rows correspond to dim -2.
TIME_AXIS: int = -2

# SCAN_AXIS identifies the lateral transducer scanning axis (columns, dim -1).
# Found in src/utriage/ml/physics_features.py (lines 9-10):
# "Axis 1 (columns): Lateral / transducer scan axis."
SCAN_AXIS: int = -1


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise-separable convolutional block with batch normalization and ReLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2) -> None:
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_ch,
                bias=False,
            ),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class LiteNet(nn.Module):
    """Lightweight CNN architecture for ultrasonic flaw classification."""

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv2d(1, w, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
        )
        self.block1 = DepthwiseSeparableBlock(w, w, stride=2)
        self.block2 = DepthwiseSeparableBlock(w, 2 * w, stride=2)
        self.block3 = DepthwiseSeparableBlock(2 * w, 4 * w, stride=2)
        self.block4 = DepthwiseSeparableBlock(4 * w, 4 * w, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(4 * w, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        feat = self.block1(feat)
        feat = self.block2(feat)
        feat = self.block3(feat)
        feat = self.block4(feat)
        pooled = self.pool(feat).flatten(start_dim=1)
        return self.head(pooled).squeeze(-1)


def count_params(model: nn.Module) -> int:
    """Return total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())


def _apply_gain(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Scale per-sample amplitudes by random gain in U(0.5, 2.0)."""
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    gain = torch.empty(shape, device=gen.device, dtype=x.dtype).uniform_(0.5, 2.0, generator=gen)
    return x * gain.to(x.device)


def _apply_offset(x: torch.Tensor, gen: torch.Generator, std: torch.Tensor) -> torch.Tensor:
    """Add per-sample DC baseline offset in U(-0.2, 0.2) * std."""
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    scale = torch.empty(shape, device=gen.device, dtype=x.dtype).uniform_(-0.2, 0.2, generator=gen)
    offset = scale.to(x.device) * std
    return x + offset


def _apply_noise(x: torch.Tensor, gen: torch.Generator, std: torch.Tensor) -> torch.Tensor:
    """Add Gaussian noise with standard deviation in U(0.0, 0.1) * std."""
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    scale = torch.empty(shape, device=gen.device, dtype=x.dtype).uniform_(0.0, 0.1, generator=gen)
    noise_std = scale.to(x.device) * std
    noise = torch.randn(x.shape, device=gen.device, dtype=x.dtype, generator=gen).to(x.device)
    return x + noise * noise_std


def _circular_shift(x: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """Perform circular roll along TIME_AXIS given per-sample shifts."""
    if x.shape[0] == 0:
        return x.clone()
    h = x.shape[TIME_AXIS]
    grid_shape = [1] * x.ndim
    grid_shape[TIME_AXIS] = h
    shift_shape = [1] * x.ndim
    shift_shape[0] = x.shape[0]
    grid = torch.arange(h, device=x.device, dtype=torch.int64).view(grid_shape)
    shifts_tensor = shifts.to(device=x.device).reshape(shift_shape)
    idx = ((grid - shifts_tensor) % h).long().expand_as(x)
    return torch.gather(x, dim=TIME_AXIS, index=idx)


def augment_batch(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Apply physical augmentations to batch with deterministic fixed generator.

    Performs per-sample gain U(0.5, 2.0), DC offset U(-0.2, 0.2) * std,
    additive Gaussian noise with std U(0, 0.1) * std, random circular roll
    along TIME_AXIS up to +-16 samples, and random flip along SCAN_AXIS (p=0.5).
    """
    if x.ndim < 3:
        raise ValueError(f"Expected batch tensor with ndim >= 3, got shape {tuple(x.shape)}.")
    if x.shape[0] == 0:
        return x.clone()

    std = x.float().std(dim=(-2, -1), keepdim=True).to(x.dtype)
    x_aug = _apply_gain(x, gen)
    x_aug = _apply_offset(x_aug, gen, std)
    x_aug = _apply_noise(x_aug, gen, std)
    shifts = torch.randint(-16, 17, (x.shape[0],), device=gen.device, generator=gen)
    x_aug = _circular_shift(x_aug, shifts)
    mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = (torch.rand(mask_shape, device=gen.device, generator=gen) < 0.5).to(x.device)
    return torch.where(mask, x_aug.flip(SCAN_AXIS), x_aug)


class CleanFeatureModel:
    """Unsupervised anomaly scoring model fitted on clean-only feature rows."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None

    def fit(self, feats: np.ndarray) -> CleanFeatureModel:
        """Fit LedoitWolf covariance on clean features and store mean and precision."""
        arr = np.asarray(feats, dtype=np.float64)
        if any([arr.size == 0, arr.shape[0] == 0]):
            raise ValueError("Cannot fit CleanFeatureModel on empty input.")
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {arr.shape}.")
        estimator = LedoitWolf().fit(arr)
        self.mean_ = np.asarray(estimator.location_, dtype=np.float64)
        self.precision_ = np.asarray(estimator.precision_, dtype=np.float64)
        return self

    def score(self, feats: np.ndarray) -> np.ndarray:
        """Compute non-negative Mahalanobis distance scores for input features."""
        if any([self.mean_ is None, self.precision_ is None]):
            raise ValueError("CleanFeatureModel must be fitted before scoring.")
        arr = np.asarray(feats, dtype=np.float64)
        if any([arr.size == 0, arr.shape[0] == 0]):
            raise ValueError("Cannot score empty input.")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {arr.shape}.")
        delta = arr - self.mean_
        sq_dist = np.sum((delta @ self.precision_) * delta, axis=1)
        return np.sqrt(np.maximum(0.0, sq_dist))


def image_features(images: np.ndarray) -> np.ndarray:
    """Extract physics features for batch of 2D images [N, H, W] -> [N, F]."""
    arr = np.asarray(images)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array of shape [N, H, W], got shape {arr.shape}.")
    if arr.shape[0] == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    rows = [extract_features(arr[i]) for i in range(arr.shape[0])]
    return np.stack(rows, axis=0).astype(np.float64)


def clean_threshold(clean_scores: np.ndarray, target_fcr: float) -> float:
    """Compute quantile anomaly threshold on clean scores for target false-call rate."""
    if not (0.0 < target_fcr < 1.0):
        raise ValueError(f"target_fcr must be strictly between 0 and 1, got {target_fcr}.")
    arr = np.asarray(clean_scores, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("clean_scores cannot be empty.")
    return float(np.quantile(arr, 1.0 - target_fcr, method="higher"))


def hybrid_flags(
    p_cnn: np.ndarray,
    anom: np.ndarray,
    t_cnn: float,
    t_anom: float,
) -> np.ndarray:
    """Return boolean OR flags: (p_cnn >= t_cnn) | (anom >= t_anom)."""
    p_arr = np.asarray(p_cnn)
    a_arr = np.asarray(anom)
    if p_arr.shape != a_arr.shape:
        raise ValueError(f"Shape mismatch: {p_arr.shape} != {a_arr.shape}.")
    return (p_arr >= t_cnn) | (a_arr >= t_anom)
