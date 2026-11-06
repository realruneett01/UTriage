"""Unit and property tests for lightweight hybrid flaw detector building blocks."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from utriage.ml.lite_hybrid import (  # noqa: E402
    SCAN_AXIS,
    TIME_AXIS,
    CleanFeatureModel,
    LiteNet,
    _circular_shift,
    augment_batch,
    clean_threshold,
    count_params,
    hybrid_flags,
    image_features,
)


def test_litenet_output_shape_and_params() -> None:
    """Verify LiteNet output shape [B] for [2,1,256,256] and parameter bound <= 100k."""
    model = LiteNet(width=16)
    x = torch.randn(2, 1, 256, 256, dtype=torch.float32)
    out = model(x)
    assert out.shape == torch.Size([2])
    assert count_params(model) <= 100_000


@pytest.mark.parametrize("batch_size", [1, 3])
def test_litenet_variable_batch_sizes(batch_size: int) -> None:
    """Verify LiteNet preserves batch dimension across variable batch sizes."""
    model = LiteNet(width=16)
    x = torch.randn(batch_size, 1, 64, 64, dtype=torch.float32)
    assert model(x).shape == torch.Size([batch_size])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_augment_batch_shape_and_dtype(dtype: torch.dtype) -> None:
    """Verify augment_batch keeps input shape, dtype, and device."""
    gen = torch.Generator().manual_seed(101)
    x = torch.randn(4, 1, 32, 32, dtype=dtype)
    out = augment_batch(x, gen)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_augment_batch_deterministic_per_seed() -> None:
    """Verify augment_batch is identical when evaluated with fixed generator seed."""
    x = torch.randn(4, 1, 32, 32, dtype=torch.float32)
    gen1 = torch.Generator().manual_seed(202)
    out1 = augment_batch(x, gen1)
    gen2 = torch.Generator().manual_seed(202)
    out2 = augment_batch(x, gen2)
    assert torch.equal(out1, out2)


def test_augment_batch_different_seeds_differ() -> None:
    """Verify augment_batch outputs differ when run with different seeds."""
    x = torch.randn(4, 1, 32, 32, dtype=torch.float32)
    out1 = augment_batch(x, torch.Generator().manual_seed(202))
    out2 = augment_batch(x, torch.Generator().manual_seed(303))
    assert not torch.equal(out1, out2)


@pytest.mark.parametrize("shifts_list", [[0, 5, -7, 16], [-16, 1, -1, 8]])
def test_circular_shift_matches_torch_roll(shifts_list: list[int]) -> None:
    """Verify _circular_shift matches per-sample torch.roll along TIME_AXIS."""
    x = torch.randn(len(shifts_list), 1, 32, 20, dtype=torch.float32)
    shifts = torch.tensor(shifts_list, dtype=torch.int64)
    out = _circular_shift(x, shifts)
    expected = torch.stack(
        [torch.roll(x[i], shifts=s, dims=TIME_AXIS) for i, s in enumerate(shifts_list)],
        dim=0,
    )
    assert torch.equal(out, expected)


def test_circular_shift_empty() -> None:
    """Verify _circular_shift handles empty batch tensor gracefully."""
    x = torch.empty(0, 1, 16, 16, dtype=torch.float32)
    shifts = torch.empty(0, dtype=torch.int64)
    out = _circular_shift(x, shifts)
    assert out.shape == x.shape


def test_augment_batch_gain_range_respected() -> None:
    """Verify per-sample gain stays within U(0.5, 2.0) on zero-std inputs."""
    gen = torch.Generator().manual_seed(303)
    x = torch.ones(50, 1, 16, 16, dtype=torch.float32)
    out = augment_batch(x, gen)
    gains = out[:, 0, 0, 0]
    assert torch.all(gains >= 0.5)
    assert torch.all(gains <= 2.0)
    assert float(gains.min()) < 0.85
    assert float(gains.max()) > 1.65


def test_augment_batch_roll_range_respected() -> None:
    """Verify circular time roll stays within +-16 samples along TIME_AXIS."""
    gen = torch.Generator().manual_seed(404)
    x = torch.zeros(60, 1, 64, 16, dtype=torch.float32)
    x[:, 0, 32, :] = 10.0
    out = augment_batch(x, gen)
    active_rows = torch.argmax(out.abs()[:, 0, :, 0], dim=-1)
    shifts = active_rows - 32
    assert torch.all(shifts >= -16)
    assert torch.all(shifts <= 16)
    assert float(shifts.min()) < 0
    assert float(shifts.max()) > 0


def test_time_and_scan_axis_constants() -> None:
    """Verify TIME_AXIS and SCAN_AXIS constants match ML-NDT orientation."""
    assert TIME_AXIS in (-2, 0, 2)
    assert SCAN_AXIS in (-1, 1, 3)


def test_clean_feature_model_separation_and_non_negative() -> None:
    """Verify CleanFeatureModel scores clean-like rows lower than shifted rows."""
    rng = np.random.default_rng(505)
    clean = rng.normal(loc=0.0, scale=1.0, size=(120, 8))
    shifted = rng.normal(loc=6.0, scale=1.0, size=(25, 8))

    model = CleanFeatureModel().fit(clean)
    scores_clean = model.score(clean[:25])
    scores_shifted = model.score(shifted)

    assert np.all(scores_clean >= 0.0)
    assert np.all(scores_shifted >= 0.0)
    assert float(np.mean(scores_clean)) < float(np.mean(scores_shifted))


@pytest.mark.parametrize(
    "empty_feats",
    [np.empty((0, 8)), np.array([])],
)
def test_clean_feature_model_empty_raises(empty_feats: np.ndarray) -> None:
    """Verify CleanFeatureModel raises ValueError on empty input."""
    model = CleanFeatureModel()
    with pytest.raises(ValueError, match="empty"):
        model.fit(empty_feats)
    model.fit(np.ones((10, 8)))
    with pytest.raises(ValueError, match="empty"):
        model.score(empty_feats)


def test_clean_feature_model_unfitted_raises() -> None:
    """Verify CleanFeatureModel raises ValueError when scoring before fit."""
    model = CleanFeatureModel()
    with pytest.raises(ValueError, match="fitted"):
        model.score(np.ones((5, 8)))


def test_image_features_shape_and_extraction() -> None:
    """Verify image_features transforms [N, H, W] images into [N, F] feature matrix."""
    imgs = np.zeros((3, 32, 32), dtype=np.float64)
    feats = image_features(imgs)
    assert feats.shape == (3, 8)
    assert feats.dtype == np.float64


def test_image_features_empty_returns_empty() -> None:
    """Verify image_features handles empty image array gracefully."""
    empty_imgs = np.empty((0, 32, 32), dtype=np.float64)
    feats = image_features(empty_imgs)
    assert feats.shape == (0, 8)


def test_clean_threshold_quantiles() -> None:
    """Verify clean_threshold computes higher quantile for false call rate."""
    scores = np.arange(1, 101, dtype=np.float64)
    thresh = clean_threshold(scores, target_fcr=0.05)
    expected = float(np.quantile(scores, 0.95, method="higher"))
    assert thresh == expected


@pytest.mark.parametrize("bad_fcr", [-0.1, 0.0, 1.0, 1.5])
def test_clean_threshold_invalid_fcr_raises(bad_fcr: float) -> None:
    """Verify clean_threshold rejects target_fcr outside open interval (0, 1)."""
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        clean_threshold(np.ones(10), target_fcr=bad_fcr)


def test_clean_threshold_empty_raises() -> None:
    """Verify clean_threshold raises ValueError on empty input array."""
    with pytest.raises(ValueError, match="empty"):
        clean_threshold(np.array([]), target_fcr=0.05)


def test_hybrid_flags_or_semantics() -> None:
    """Verify hybrid_flags executes boolean OR without dropping detections."""
    p_cnn = np.array([0.1, 0.9, 0.2, 0.8])
    anom = np.array([0.3, 0.1, 0.7, 0.9])
    flags = hybrid_flags(p_cnn, anom, t_cnn=0.5, t_anom=0.5)
    expected = np.array([False, True, True, True])
    assert np.array_equal(flags, expected)


def test_hybrid_flags_shape_mismatch_raises() -> None:
    """Verify hybrid_flags raises ValueError when input array shapes mismatch."""
    with pytest.raises(ValueError, match="mismatch"):
        hybrid_flags(np.ones(4), np.ones(5), 0.5, 0.5)


@settings(max_examples=100, deadline=None)
@given(
    pairs=st.lists(
        st.tuples(
            st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
            st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
        ),
        min_size=1,
        max_size=20,
    ),
    t_cnn=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
    t_anom=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
)
def test_hybrid_flags_hypothesis(
    pairs: list[tuple[float, float]],
    t_cnn: float,
    t_anom: float,
) -> None:
    """Property test: hybrid_flags equals boolean OR and never drops detections."""
    p_cnn = np.array([p for p, _ in pairs], dtype=float)
    anom = np.array([a for _, a in pairs], dtype=float)
    flags = hybrid_flags(p_cnn, anom, t_cnn, t_anom)

    cnn_mask = p_cnn >= t_cnn
    anom_mask = anom >= t_anom
    expected = cnn_mask | anom_mask

    assert np.array_equal(flags, expected)
    assert np.all(flags[cnn_mask])
    assert np.all(flags[anom_mask])
