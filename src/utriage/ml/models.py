"""Neural network architectures for ML-NDT crack detection and sizing (project plan 3.1).

A small residual CNN with dual heads: classification logit and size regression.
"""

from __future__ import annotations

import torch
from torch import nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """Return a 3x3 convolution with padding 1 and no bias."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def make_shortcut(in_ch: int, out_ch: int, stride: int) -> nn.Module:
    """Projection shortcut if shape changes, otherwise identity."""
    if stride != 1 or in_ch != out_ch:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        )
    return nn.Identity()


class BasicBlock(nn.Module):
    """Residual basic block with two 3x3 convolutions and optional projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = make_shortcut(in_channels, out_channels, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out + identity)
        return out


class MlNdtNet(nn.Module):
    """Residual CNN with concatenated global pooling and dual detection/size heads."""

    def __init__(
        self,
        width: int = 32,
        first_pool: tuple[int, int] = (2, 2),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            conv3x3(1, width),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=first_pool),
        )
        self.stage1 = BasicBlock(width, width, stride=1)
        self.stage2 = BasicBlock(width, 2 * width, stride=2)
        self.stage3 = BasicBlock(2 * width, 4 * width, stride=2)
        self.stage4 = BasicBlock(4 * width, 8 * width, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(16 * width, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.stage4(feat)
        avg_feat = self.avg_pool(feat).flatten(start_dim=1)
        max_feat = self.max_pool(feat).flatten(start_dim=1)
        pooled = torch.cat([avg_feat, max_feat], dim=1)
        dropped = self.dropout(pooled)
        out = self.head(dropped)
        logit = out[:, 0]
        size = out[:, 1]
        return logit, size


def count_params(model: nn.Module) -> int:
    """Return total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())
