"""Invariant crack representation modules and contrastive losses.

Provides physics-informed spatial stretch transforms, LiteNet-based invariant feature
encoder, projection head, downstream classifier, and contrastive learning losses (SimCLR
NT-Xent and Supervised Contrastive SupCon).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utriage.ml.lite_hybrid import SCAN_AXIS, TIME_AXIS, LiteNet, augment_batch


class LiteEncoder(nn.Module):
    """Backbone feature encoder reusing LiteNet convolutional layers."""

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        net = LiteNet(width=width)
        self.stem = net.stem
        self.block1 = net.block1
        self.block2 = net.block2
        self.block3 = net.block3
        self.block4 = net.block4
        self.pool = net.pool
        self._out_dim = 4 * width

    @property
    def out_dim(self) -> int:
        """Return output feature dimension."""
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward ultrasonic image [B, 1, H, W] to pooled representations [B, 4*width]."""
        feat = self.stem(x)
        feat = self.block1(feat)
        feat = self.block2(feat)
        feat = self.block3(feat)
        feat = self.block4(feat)
        return self.pool(feat).flatten(start_dim=1)


class ProjectionHead(nn.Module):
    """MLP projection head mapping encoder features to an L2-normalized hypersphere."""

    def __init__(self, in_dim: int, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project representation vectors to L2-normalized embeddings [B, out_dim]."""
        proj = self.net(x)
        return F.normalize(proj, p=2.0, dim=-1)


class InvariantClassifier(nn.Module):
    """Downstream flaw classifier using a frozen or fine-tuned LiteEncoder."""

    def __init__(self, encoder: LiteEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.out_dim, 1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Extract encoder feature representation [B, D]."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict flaw classification logits [B]."""
        feats = self.embed(x)
        return self.head(feats).squeeze(-1)


def _crop_or_pad(
    res: torch.Tensor,
    is_time: bool,
    h_orig: int,
    w_orig: int,
) -> torch.Tensor:
    """Center-crop or zero-pad resampled tensor back to target spatial dimensions."""
    if is_time:
        h_new = res.shape[-2]
        if h_new > h_orig:
            start = (h_new - h_orig) // 2
            return res[..., start : start + h_orig, :]
        pad_top = (h_orig - h_new) // 2
        return F.pad(res, (0, 0, pad_top, h_orig - h_new - pad_top))
    w_new = res.shape[-1]
    if w_new > w_orig:
        start = (w_new - w_orig) // 2
        return res[..., start : start + w_orig]
    pad_left = (w_orig - w_new) // 2
    return F.pad(res, (pad_left, w_orig - w_new - pad_left, 0, 0))


def _resample_sample(
    sample: torch.Tensor,
    factor: float,
    is_time: bool,
) -> torch.Tensor:
    """Resample single 4D tensor along selected spatial axis."""
    h_orig, w_orig = sample.shape[-2], sample.shape[-1]
    if abs(factor - 1.0) < 1e-6:
        return sample
    h_new = max(1, int(round(h_orig * factor))) if is_time else h_orig
    w_new = w_orig if is_time else max(1, int(round(w_orig * factor)))
    if all([h_new == h_orig, w_new == w_orig]):
        return sample
    res = F.interpolate(sample, size=(h_new, w_new), mode="bilinear", align_corners=False)
    return _crop_or_pad(res, is_time, h_orig, w_orig)


def _is_time_axis(ndim: int, axis: int) -> bool:
    """Return whether target axis matches the fast-time / depth dimension."""
    norm_axis = axis if axis >= 0 else ndim + axis
    if norm_axis == (ndim - 2):
        return True
    if norm_axis == (ndim - 1):
        return False
    raise ValueError(f"Unsupported axis {axis}. Must be -2 (time) or -1 (scan).")


def _prepare_factors(factor: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Flatten factor tensor and validate batch size alignment."""
    f_arr = factor.view(-1)
    if any([len(f_arr) == 1, batch_size == 1]):
        return f_arr.expand(batch_size)
    if len(f_arr) != batch_size:
        raise ValueError(f"factor length {len(f_arr)} does not match batch {batch_size}.")
    return f_arr


def stretch_axis(x: torch.Tensor, factor: torch.Tensor, axis: int) -> torch.Tensor:
    """Resample per-sample along an axis and crop or pad back to input size.

    Physics approximation: stretching along TIME_AXIS simulates acoustic velocity
    or depth variation, while stretching along SCAN_AXIS approximates crack lateral
    extent variations. Note that this geometric scaling is an idealized acoustic model.
    """
    if x.ndim < 3:
        raise ValueError(f"Expected tensor with ndim >= 3, got shape {tuple(x.shape)}.")
    if x.shape[0] == 0:
        return x.clone()
    is_time = _is_time_axis(x.ndim, axis)
    f_arr = _prepare_factors(factor, x.shape[0])
    is_3d = x.ndim == 3
    x_4d = x.unsqueeze(1) if is_3d else x
    res_list = [
        _resample_sample(x_4d[i : i + 1], float(f_arr[i].item()), is_time)
        for i in range(x.shape[0])
    ]
    out = torch.cat(res_list, dim=0)
    return out.squeeze(1) if is_3d else out


def physics_view(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Generate physically-augmented view under sound-velocity and flaw-geometry stretch."""
    aug = augment_batch(x, gen)
    b = x.shape[0]
    if b == 0:
        return aug
    t_factors = (
        torch.empty((b,), device=gen.device, dtype=torch.float32)
        .uniform_(0.85, 1.15, generator=gen)
        .to(x.device)
    )
    s_factors = (
        torch.empty((b,), device=gen.device, dtype=torch.float32)
        .uniform_(0.80, 1.25, generator=gen)
        .to(x.device)
    )
    x_time = stretch_axis(aug, t_factors, TIME_AXIS)
    return stretch_axis(x_time, s_factors, SCAN_AXIS)


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """SimCLR NT-Xent loss on normalized view pairs with 2B - 2 negative samples."""
    if z1.shape != z2.shape:
        raise ValueError(f"Mismatch in view shapes: {z1.shape} vs {z2.shape}.")
    if z1.ndim != 2:
        raise ValueError(f"Expected 2D embedding tensors [B, D], got ndim {z1.ndim}.")
    b = z1.shape[0]
    if b < 1:
        raise ValueError("Batch size must be at least 1.")
    if temperature <= 0.0:
        raise ValueError(f"Temperature must be positive, got {temperature}.")
    # float32 so the -1e9 mask cannot overflow under fp16 autocast
    norm_z1 = F.normalize(z1.float(), p=2.0, dim=-1)
    norm_z2 = F.normalize(z2.float(), p=2.0, dim=-1)
    z = torch.cat([norm_z1, norm_z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    diag_mask = torch.eye(2 * b, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(diag_mask, -1e9)
    targets = torch.cat(
        [
            torch.arange(b, 2 * b, device=z.device),
            torch.arange(0, b, device=z.device),
        ]
    )
    return F.cross_entropy(sim, targets)


def _validate_supcon_inputs(z: torch.Tensor, labels: torch.Tensor, temp: float) -> None:
    """Validate dimensionality and shape constraints for SupCon loss."""
    if z.ndim != 2:
        raise ValueError(f"Expected 2D tensor [N, D], got ndim {z.ndim}.")
    if z.shape[0] < 2:
        raise ValueError(f"N must be at least 2, got {z.shape[0]}.")
    if any([labels.ndim != 1, labels.shape[0] != z.shape[0]]):
        raise ValueError(f"labels must be 1D tensor [N], got {labels.shape}.")
    if temp <= 0.0:
        raise ValueError(f"Temperature must be positive, got {temp}.")


def supcon_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020) with positive-less anchor skipping."""
    _validate_supcon_inputs(z, labels, temperature)
    norm_z = F.normalize(z.float(), p=2.0, dim=-1)  # float32: fp16-safe mask
    sim = torch.matmul(norm_z, norm_z.T) / temperature
    eye = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    label_col = labels.contiguous().view(-1, 1)
    pos_mask = torch.eq(label_col, label_col.T) & (~eye)
    num_pos = pos_mask.sum(dim=1)
    valid_anchors = num_pos > 0
    if not valid_anchors.any():
        return torch.tensor(0.0, device=z.device)

    sim_den = sim.masked_fill(eye, -1e9)
    log_probs = (sim - torch.logsumexp(sim_den, dim=1, keepdim=True)) * pos_mask.float()
    anchor_loss = -(log_probs.sum(dim=1) / num_pos.clamp(min=1))
    return anchor_loss[valid_anchors].mean()
