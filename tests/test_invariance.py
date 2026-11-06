"""Unit tests for invariant crack representation modules and losses."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pytest  # noqa: E402
import torch  # noqa: E402

from utriage.ml.invariance import (  # noqa: E402
    InvariantClassifier,
    LiteEncoder,
    ProjectionHead,
    nt_xent_loss,
    physics_view,
    stretch_axis,
    supcon_loss,
)
from utriage.ml.lite_hybrid import SCAN_AXIS, TIME_AXIS  # noqa: E402


def test_lite_encoder_output_shapes() -> None:
    """Verify LiteEncoder output dimensions on 256x256 and fast 64x64 tensors."""
    enc16 = LiteEncoder(width=16)
    x_full = torch.randn(2, 1, 256, 256)
    out_full = enc16(x_full)
    assert out_full.shape == torch.Size([2, 4 * 16])
    assert enc16.out_dim == 64

    enc4 = LiteEncoder(width=4)
    x_fast = torch.randn(2, 1, 64, 64)
    out_fast = enc4(x_fast)
    assert out_fast.shape == torch.Size([2, 4 * 4])
    assert enc4.out_dim == 16


@pytest.mark.parametrize("out_dim", [32, 64])
def test_projection_head_unit_norm(out_dim: int) -> None:
    """Verify ProjectionHead outputs L2-normalized unit vectors."""
    head = ProjectionHead(in_dim=16, out_dim=out_dim)
    feats = torch.randn(5, 16)
    proj = head(feats)
    assert proj.shape == torch.Size([5, out_dim])
    norms = torch.norm(proj, p=2.0, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_invariant_classifier_shapes(batch_size: int) -> None:
    """Verify InvariantClassifier returns logits [B] and embeddings [B, D]."""
    encoder = LiteEncoder(width=4)
    model = InvariantClassifier(encoder)
    x = torch.randn(batch_size, 1, 64, 64)
    logits = model(x)
    embeds = model.embed(x)
    assert logits.shape == torch.Size([batch_size])
    assert embeds.shape == torch.Size([batch_size, encoder.out_dim])


@pytest.mark.parametrize("axis", [TIME_AXIS, SCAN_AXIS])
def test_stretch_axis_factor_one_identity(axis: int) -> None:
    """Verify factor 1.0 returns the input tensor exactly."""
    x = torch.randn(3, 1, 64, 64)
    factors = torch.ones(3)
    out = stretch_axis(x, factors, axis)
    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-5)


@pytest.mark.parametrize(
    ("axis", "orig_idx", "expected_idx"),
    [
        (TIME_AXIS, (40, 32), 48),
        (SCAN_AXIS, (32, 40), 48),
    ],
)
def test_stretch_axis_peak_shift(
    axis: int,
    orig_idx: tuple[int, int],
    expected_idx: int,
) -> None:
    """Verify factor 2.0 moves an off-center peak by the expected amount."""
    pattern = torch.zeros(1, 1, 64, 64)
    pattern[0, 0, orig_idx[0], orig_idx[1]] = 10.0
    factor = torch.tensor([2.0])
    out = stretch_axis(pattern, factor, axis)
    assert out.shape == pattern.shape
    res_slice = out[0, 0, :, 32] if axis == TIME_AXIS else out[0, 0, 32, :]
    assert abs(res_slice.argmax().item() - expected_idx) <= 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stretch_axis_dtype_and_shape(dtype: torch.dtype) -> None:
    """Verify stretch_axis preserves shape and dtype across scales."""
    x = torch.randn(2, 1, 32, 32, dtype=dtype)
    factors = torch.tensor([0.85, 1.15], dtype=torch.float32)
    out = stretch_axis(x, factors, SCAN_AXIS)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


@pytest.mark.parametrize("axis", [TIME_AXIS, SCAN_AXIS])
def test_stretch_axis_batched_matches_individual(axis: int) -> None:
    """Verify batched stretch_axis matches per-sample individual calls."""
    torch.manual_seed(42)
    x = torch.randn(4, 1, 32, 32)
    factors = torch.tensor([0.8, 1.0, 1.2, 1.5])
    batched = stretch_axis(x, factors, axis)
    singles = torch.cat(
        [stretch_axis(x[i : i + 1], factors[i : i + 1], axis) for i in range(4)],
        dim=0,
    )
    assert batched.shape == x.shape
    assert torch.allclose(batched, singles, atol=1e-5)


@pytest.mark.parametrize("axis", [TIME_AXIS, SCAN_AXIS])
def test_stretch_axis_deterministic_and_no_nan(axis: int) -> None:
    """Verify stretch_axis is deterministic and contains no NaN values."""
    torch.manual_seed(42)
    x = torch.randn(4, 1, 32, 32)
    factors = torch.tensor([0.75, 1.0, 1.33, 2.0])
    out1 = stretch_axis(x, factors, axis)
    out2 = stretch_axis(x, factors, axis)
    assert torch.equal(out1, out2)
    assert not torch.isnan(out1).any()


def test_physics_view_deterministic_and_preserves_meta() -> None:
    """Verify physics_view reproducibility and tensor metadata preservation."""
    x = torch.randn(2, 1, 64, 64, dtype=torch.float32)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    v1 = physics_view(x, g1)
    v2 = physics_view(x, g2)
    assert torch.allclose(v1, v2)
    assert v1.shape == x.shape
    assert v1.dtype == x.dtype
    assert v1.device == x.device


def test_nt_xent_loss_identical_vs_random() -> None:
    """Verify NT-Xent loss is lower for identical views than random pairs."""
    z1 = torch.randn(8, 16)
    loss_identical = nt_xent_loss(z1, z1.clone())
    loss_random = nt_xent_loss(z1, torch.randn(8, 16))
    assert loss_identical < loss_random
    assert torch.isfinite(loss_identical)


def test_supcon_loss_clusters_and_edge_cases() -> None:
    """Verify SupCon loss clustering sensitivity, single-member labels, and errors."""
    z_good = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1, 1])
    loss_good = supcon_loss(z_good, labels)
    loss_shuffled = supcon_loss(z_good, torch.tensor([0, 1, 0, 1]))
    assert loss_good < loss_shuffled

    single_labels = torch.tensor([0, 0, 1])
    z_single = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    loss_single = supcon_loss(z_single, single_labels)
    assert torch.isfinite(loss_single)
    assert loss_single.ndim == 0

    with pytest.raises(ValueError, match="at least 2"):
        supcon_loss(torch.randn(1, 4), torch.tensor([0]))


def test_losses_accept_fp16_embeddings() -> None:
    """Contrastive losses stay finite for fp16 inputs (AMP autocast)."""
    z1 = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1).half()
    z2 = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1).half()
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    assert torch.isfinite(nt_xent_loss(z1, z2))
    assert torch.isfinite(supcon_loss(torch.cat([z1, z2]), labels))
