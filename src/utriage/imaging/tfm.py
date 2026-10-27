"""Total Focusing Method (TFM) and Full Matrix Capture (FMC) for linear contact arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert


@dataclass(frozen=True)
class ArrayGeometry:
    """Acoustic and spatial configuration for a 1D linear contact array."""

    elem_x_m: np.ndarray
    fs_hz: float
    c_m_s: float
    t0_s: float = 0.0


def simulate_point_fmc(
    geom: ArrayGeometry,
    scatterer_xz: tuple[float, float],
    n_t: int,
    f0_hz: float,
    cycles: float = 3.0,
) -> np.ndarray:
    """Simulate Full Matrix Capture (FMC) data for a single point scatterer.

    Output is Synthetic data.
    """
    xs, zs = scatterer_xz
    elem_x = np.asarray(geom.elem_x_m, dtype=float)
    d_elem = np.sqrt((elem_x - xs) ** 2 + zs**2)
    tof = (d_elem[:, None] + d_elem[None, :]) / geom.c_m_s

    time_s = geom.t0_s + np.arange(n_t) / geom.fs_hz
    dt = time_s[None, None, :] - tof[:, :, None]
    sigma_s = cycles / (2.0 * np.pi * f0_hz)

    envelope = np.exp(-0.5 * (dt / sigma_s) ** 2)
    carrier = np.cos(2.0 * np.pi * f0_hz * dt)
    return envelope * carrier


def _interpolate_tx(
    sig_tx: np.ndarray,
    sample_indices: np.ndarray,
    n_t: int,
) -> np.ndarray:
    """Linearly interpolate complex signals for one transmitter across receivers and pixels."""
    k0 = np.floor(sample_indices).astype(int)
    k1 = k0 + 1
    weight = sample_indices - k0

    valid = (sample_indices >= 0) & (sample_indices <= n_t - 1)
    k0_c = np.clip(k0, 0, n_t - 1)
    k1_c = np.clip(k1, 0, n_t - 1)

    rx_indices = np.arange(sig_tx.shape[0])[:, None, None]
    interp_vals = (1.0 - weight) * sig_tx[rx_indices, k0_c] + weight * sig_tx[rx_indices, k1_c]
    interp_vals[~valid] = 0.0
    return interp_vals.sum(axis=0)


def tfm_image(
    fmc: np.ndarray,
    geom: ArrayGeometry,
    x_m: np.ndarray,
    z_m: np.ndarray,
) -> np.ndarray:
    """Compute Total Focusing Method (TFM) image for a linear contact array.

    Delays and sums FMC analytic signals at t = (d_tx + d_rx)/c - t0 over a 2D pixel grid.
    Returns non-negative envelope image with shape (len(z_m), len(x_m)).
    """
    analytic = hilbert(fmc, axis=-1)
    n_tx, n_rx, n_t = fmc.shape
    x_grid, z_grid = np.meshgrid(x_m, z_m)

    elem_x = np.asarray(geom.elem_x_m, dtype=float)
    dist = np.sqrt((x_grid[None, :, :] - elem_x[:, None, None]) ** 2 + z_grid[None, :, :] ** 2)
    dist_rx = dist[:n_rx]

    image_complex = np.zeros((len(z_m), len(x_m)), dtype=complex)
    for tx in range(n_tx):
        sample_indices = ((dist[tx : tx + 1] + dist_rx) / geom.c_m_s - geom.t0_s) * geom.fs_hz
        image_complex += _interpolate_tx(analytic[tx], sample_indices, n_t)

    return np.abs(image_complex)
