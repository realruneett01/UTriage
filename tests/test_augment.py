"""Tests for physics-aware ultrasonic data augmentation (Phase 3-4 ML-NDT)."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from utriage.ml.augment import (  # noqa: E402
    AugmentConfig,
    add_noise,
    augment,
    gain_change,
    implant_virtual_flaw,
    pulse_shape_change,
    time_shift,
)


@pytest.fixture
def rng() -> np.random.Generator:
    """Fixed RNG fixture for reproducible testing."""
    return np.random.default_rng(42)


@pytest.fixture
def test_signal_1d() -> np.ndarray:
    """Synthetic 1D A-scan pulse for testing."""
    t = np.linspace(-3, 3, 64, dtype=np.float32)
    return np.exp(-(t**2)) * np.cos(2 * np.pi * t)


@pytest.fixture
def test_image_2d() -> np.ndarray:
    """Synthetic 2D B-scan slice for testing."""
    x = np.linspace(-2, 2, 32, dtype=np.float32)
    y = np.linspace(-2, 2, 48, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    return np.exp(-(xx**2 + yy**2)).astype(np.float32)


@pytest.mark.parametrize("shape", [(64,), (2, 64), (16, 32)])
def test_shapes_preserved(shape: tuple[int, ...], rng: np.random.Generator) -> None:
    """Ensure all augmentation functions preserve array shapes and types."""
    x = np.ones(shape, dtype=np.float32)
    cfg = AugmentConfig(p=1.0)

    out_gain = gain_change(x, rng, gain_db=3.0)
    out_shift = time_shift(x, rng, max_shift=4)
    out_noise = add_noise(x, rng, noise_std=0.02)
    out_pulse = pulse_shape_change(x, rng, pulse_stretch=0.1)
    out_all = augment(x, cfg, rng)

    for out in (out_gain, out_shift, out_noise, out_pulse, out_all):
        assert out.shape == x.shape
        assert out.dtype == x.dtype


@pytest.mark.parametrize("dim", ["1d", "2d"])
def test_input_never_mutated(
    dim: str,
    test_signal_1d: np.ndarray,
    test_image_2d: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Ensure original arrays are never modified in-place."""
    base = test_signal_1d if dim == "1d" else test_image_2d
    orig = base.copy()
    cfg = AugmentConfig(p=1.0)

    _ = gain_change(base, rng)
    _ = time_shift(base, rng)
    _ = add_noise(base, rng)
    _ = pulse_shape_change(base, rng)
    _ = augment(base, cfg, rng)

    patch = np.ones((4,) if dim == "1d" else (4, 4), dtype=np.float32)
    pos = 4 if dim == "1d" else (4, 4)
    _ = implant_virtual_flaw(base, patch, pos)

    np.testing.assert_array_equal(base, orig)


def test_same_seed_same_output(test_signal_1d: np.ndarray) -> None:
    """Verify augment produces deterministic output with identical seeds."""
    cfg = AugmentConfig(p=0.5)
    rng1 = np.random.default_rng(12345)
    rng2 = np.random.default_rng(12345)

    out1 = augment(test_signal_1d, cfg, rng1)
    out2 = augment(test_signal_1d, cfg, rng2)

    np.testing.assert_array_almost_equal(out1, out2)


@pytest.mark.parametrize("shift_val", [3, -3, 0, 100])
def test_time_shift_zero_fills(shift_val: int) -> None:
    """Verify time shift shifts along the last axis with zero filling, not wrapping."""
    dummy_rng = np.random.default_rng(0)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    out = time_shift(x, dummy_rng, shift=shift_val)

    if shift_val == 3:
        np.testing.assert_array_equal(out, [0.0, 0.0, 0.0, 1.0, 2.0])
    elif shift_val == -3:
        np.testing.assert_array_equal(out, [4.0, 5.0, 0.0, 0.0, 0.0])
    elif shift_val == 0:
        np.testing.assert_array_equal(out, x)
    else:
        np.testing.assert_array_equal(out, np.zeros_like(x))


def test_gain_change_fixed_draw() -> None:
    """Verify gain change scales amplitudes by 10**(g/20) at a fixed draw."""
    x = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    dummy_rng = np.random.default_rng(0)

    out_identity = gain_change(x, dummy_rng, gain_db=0.0)
    np.testing.assert_array_almost_equal(out_identity, x)

    expected_scale = 10.0 ** (6.0 / 20.0)
    out_scaled = gain_change(x, dummy_rng, gain_db=6.0, gain_val=6.0)
    np.testing.assert_array_almost_equal(out_scaled, x * expected_scale)


def test_add_noise_relative_to_rms() -> None:
    """Verify added noise variance scales with signal RMS."""
    rng1 = np.random.default_rng(999)
    rng2 = np.random.default_rng(999)
    x = np.ones(1000, dtype=np.float32) * 4.0
    out = add_noise(x, rng1, noise_std=0.05)
    noise = out - x

    assert abs(float(np.std(noise)) - 0.20) < 0.02

    zero_sig = np.zeros(10, dtype=np.float32)
    np.testing.assert_array_equal(add_noise(zero_sig, rng2, noise_std=0.1), zero_sig)
    np.testing.assert_array_equal(add_noise(x, rng2, noise_std=0.0), x)


@pytest.mark.parametrize("factor", [1.25, 0.75, 1.0])
def test_pulse_shape_change_stretch_compress(factor: float) -> None:
    """Verify pulse shape change resamples along last axis with crop/pad."""
    dummy_rng = np.random.default_rng(0)
    x = np.sin(np.linspace(0, np.pi, 30, dtype=np.float32))
    out = pulse_shape_change(x, dummy_rng, factor=factor)

    assert out.shape == x.shape
    if factor < 1.0:
        expected_len = int(round(30 * factor))
        np.testing.assert_array_equal(out[expected_len:], 0.0)
    elif factor == 1.0:
        np.testing.assert_array_equal(out, x)


def test_pulse_shape_edge_cases() -> None:
    """Verify pulse shape change with small length or zero stretch returns copy."""
    dummy_rng = np.random.default_rng(0)
    short_x = np.array([3.14], dtype=np.float32)
    np.testing.assert_array_equal(pulse_shape_change(short_x, dummy_rng), short_x)

    x = np.ones(10, dtype=np.float32)
    np.testing.assert_array_equal(pulse_shape_change(x, dummy_rng, pulse_stretch=0.0), x)


def test_implant_adds_exact_patch_1d() -> None:
    """Verify 1D virtual-flaw implantation adds patch at exact position."""
    clean = np.zeros(16, dtype=np.float32)
    patch = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = implant_virtual_flaw(clean, patch, position=5, scale=2.0)

    expected = np.zeros(16, dtype=np.float32)
    expected[5:8] = patch * 2.0
    np.testing.assert_array_equal(out, expected)


def test_implant_adds_exact_patch_2d() -> None:
    """Verify 2D virtual-flaw implantation adds patch at exact 2D position."""
    clean = np.zeros((8, 8), dtype=np.float32)
    patch = np.ones((2, 3), dtype=np.float32) * 5.0
    out = implant_virtual_flaw(clean, patch, position=(2, 3), scale=1.0)

    expected = np.zeros((8, 8), dtype=np.float32)
    expected[2:4, 3:6] = patch
    np.testing.assert_array_equal(out, expected)


def test_implant_broadcast_1d_patch_to_2d_clean() -> None:
    """Verify implanting 1D patch into 2D clean with scalar position."""
    clean = np.zeros((2, 10), dtype=np.float32)
    patch_1d = np.array([1.0, 2.0], dtype=np.float32)
    out = implant_virtual_flaw(clean, patch_1d, position=3, scale=1.0)

    expected = np.zeros((2, 10), dtype=np.float32)
    expected[0, 3:5] = patch_1d
    np.testing.assert_array_equal(out, expected)


def test_augment_probability_zero_and_one(test_signal_1d: np.ndarray) -> None:
    """Verify augment respects probability p=0 and p=1."""
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(1)

    out_p0 = augment(test_signal_1d, AugmentConfig(p=0.0), rng1)
    np.testing.assert_array_equal(out_p0, test_signal_1d)

    out_p1 = augment(test_signal_1d, AugmentConfig(p=1.0), rng2)
    assert not np.array_equal(out_p1, test_signal_1d)
