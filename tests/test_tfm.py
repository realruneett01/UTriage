"""Tests for Total Focusing Method (TFM) and Full Matrix Capture (FMC)."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from utriage.imaging.tfm import (  # noqa: E402
    ArrayGeometry,
    simulate_point_fmc,
    tfm_image,
)


@pytest.fixture
def array_geom() -> ArrayGeometry:
    """16 elements, pitch 0.6 mm, fs 50 MHz, c 5900 m/s."""
    pitch_m = 0.6e-3
    elem_x = (np.arange(16) - 7.5) * pitch_m
    return ArrayGeometry(elem_x_m=elem_x, fs_hz=50e6, c_m_s=5900.0, t0_s=0.0)


def test_zero_fmc_returns_zero_image(array_geom: ArrayGeometry) -> None:
    """Zero FMC matrix must produce an all-zero image."""
    x_m = np.arange(-2e-3, 5e-3, 0.1e-3)
    z_m = np.arange(10e-3, 20e-3, 0.1e-3)
    fmc = np.zeros((16, 16, 500), dtype=float)
    img = tfm_image(fmc, array_geom, x_m, z_m)

    assert img.shape == (len(z_m), len(x_m))
    assert np.all(img == 0.0)


def test_simulate_point_fmc_properties(array_geom: ArrayGeometry) -> None:
    """Simulated point FMC has correct shape, non-trivial values, and finite numbers."""
    fmc = simulate_point_fmc(array_geom, (2e-3, 15e-3), n_t=600, f0_hz=5e6)

    assert fmc.shape == (16, 16, 600)
    assert np.isfinite(fmc).all()
    assert np.max(np.abs(fmc)) > 0.0


def test_tfm_output_shape_and_non_negativity(array_geom: ArrayGeometry) -> None:
    """TFM image output shape matches (len(z), len(x)) and is non-negative."""
    fmc = simulate_point_fmc(array_geom, (2e-3, 15e-3), n_t=600, f0_hz=5e6)
    x_m = np.linspace(0.0, 4.0e-3, 41)
    z_m = np.linspace(12.0e-3, 18.0e-3, 61)
    img = tfm_image(fmc, array_geom, x_m, z_m)

    assert img.shape == (len(z_m), len(x_m))
    assert np.all(img >= 0.0)


@pytest.mark.parametrize(
    ("scatterer_x_mm", "scatterer_z_mm"),
    [
        (2.0, 15.0),
        (-1.0, 12.0),
    ],
)
def test_point_scatterer_within_one_pixel(
    array_geom: ArrayGeometry,
    scatterer_x_mm: float,
    scatterer_z_mm: float,
) -> None:
    """Image argmax must be within 1 pixel (0.1 mm) of the true scatterer location."""
    grid_step = 0.1e-3
    xs = scatterer_x_mm * 1e-3
    zs = scatterer_z_mm * 1e-3

    x_m = np.arange(-3.0e-3, 5.01e-3, grid_step)
    z_m = np.arange(8.0e-3, 20.01e-3, grid_step)

    fmc = simulate_point_fmc(array_geom, (xs, zs), n_t=600, f0_hz=5e6, cycles=3.0)
    img = tfm_image(fmc, array_geom, x_m, z_m)

    max_z_idx, max_x_idx = np.unravel_index(np.argmax(img), img.shape)
    detected_x = x_m[max_x_idx]
    detected_z = z_m[max_z_idx]

    assert abs(detected_x - xs) <= grid_step + 1e-9
    assert abs(detected_z - zs) <= grid_step + 1e-9
