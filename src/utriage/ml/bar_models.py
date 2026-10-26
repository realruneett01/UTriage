"""1D residual CNN and dataset utilities for bar ultrasonic windows (project plan 4.1).

Architectures operate on DAC-normalised A-scan windows of shape (B, C, 128).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from utriage.ml.models import count_params


def conv1d_k3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """Return a 3x1 convolution with padding 1 and no bias."""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def make_shortcut1d(in_ch: int, out_ch: int, stride: int) -> nn.Module:
    """Projection shortcut if shape changes, otherwise identity."""
    if stride != 1 or in_ch != out_ch:
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch),
        )
    return nn.Identity()


class BasicBlock1D(nn.Module):
    """Residual basic block for 1D sequences with two 3x1 convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = conv1d_k3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv1d_k3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = make_shortcut1d(in_channels, out_channels, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out + identity)


class WindowNet(nn.Module):
    """1D residual CNN on (B, C, 128) windows with classification and size heads."""

    def __init__(
        self,
        in_channels: int = 1,
        width: int = 32,
        n_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),
        )
        self.stage1 = BasicBlock1D(width, width, stride=1)
        self.stage2 = BasicBlock1D(width, 2 * width, stride=2)
        self.stage3 = BasicBlock1D(2 * width, 4 * width, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.clf_head = nn.Linear(8 * width, n_classes)
        self.size_head = nn.Linear(8 * width, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        avg_feat = self.avg_pool(feat).flatten(start_dim=1)
        max_feat = self.max_pool(feat).flatten(start_dim=1)
        pooled = torch.cat([avg_feat, max_feat], dim=1)
        dropped = self.dropout(pooled)
        class_logits = self.clf_head(dropped)
        size = self.size_head(dropped)[:, 0]
        return class_logits, size


def flaw_logit(class_logits: torch.Tensor, flaw_idx: int = 1) -> torch.Tensor:
    """Calculate log-odds of flaw vs not-flaw (shape (B,)).

    Computes logsumexp over flaw class minus logsumexp over the others.
    """
    if class_logits.ndim == 1:
        class_logits = class_logits.unsqueeze(0)
    if class_logits.shape[1] == 1:
        return class_logits[:, 0]
    flaw_col = class_logits[:, flaw_idx]
    other_cols = torch.cat([class_logits[:, :flaw_idx], class_logits[:, flaw_idx + 1 :]], dim=1)
    return flaw_col - torch.logsumexp(other_cols, dim=1)


class TrainerAdapter(nn.Module):
    """Wraps a WindowNet so forward returns (flaw_logit, size) for train_model."""

    def __init__(self, net: WindowNet, flaw_idx: int = 1) -> None:
        super().__init__()
        self.net = net
        self.flaw_idx = flaw_idx

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        class_logits, size = self.net(x)
        return flaw_logit(class_logits, flaw_idx=self.flaw_idx), size


class WindowDataset(Dataset):
    """Dataset of DAC-normalised windows with flaw labels and size measurements."""

    def __init__(
        self,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        size_mm: np.ndarray | torch.Tensor,
    ) -> None:
        super().__init__()
        if isinstance(x, np.ndarray):
            x_arr = x[:, np.newaxis, :] if x.ndim == 2 else x
            self.x = torch.as_tensor(x_arr, dtype=torch.float32)
        else:
            self.x = x.unsqueeze(1).float() if x.ndim == 2 else x.float()

        y_arr = y.cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        sz_arr = size_mm.cpu().numpy() if isinstance(size_mm, torch.Tensor) else np.asarray(size_mm)

        self.y = y_arr
        self.size_mm = sz_arr
        self.flaws = torch.as_tensor(self.y == 1, dtype=torch.float32)
        self.sizes = torch.as_tensor(self.size_mm, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.flaws[idx], self.sizes[idx]


__all__ = [
    "BasicBlock1D",
    "TrainerAdapter",
    "WindowDataset",
    "WindowNet",
    "count_params",
    "flaw_logit",
]
