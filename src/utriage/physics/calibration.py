"""ASTM E127-style flat calibration block simulation and DAC measurement.

Output is Synthetic data.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from utriage.physics.simulator import (
    BarSpec,
    Material,
    Probe,
    attenuation_factor,
    beam_factor,
    depth_of_time,
    envelope,
    fbh_amplitude,
    grain_noise,
    pulse,
    reflection_coefficient,
    round_trip_transmission,
)


@dataclass(frozen=True)
class DacPoint:
    """One point on a Distance-Amplitude Correction (DAC) curve."""

    depth_mm: float
    amplitude: float


@dataclass(frozen=True)
class CalibrationBlock:
    """ASTM E127-style flat calibration block set specification."""

    material: Material
    probe: Probe
    water: Material
    fbh_diameter_mm: float = 0.8
    metal_paths_mm: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0)
    thickness_mm: float = 40.0
    seed: int = 0


def _calibration_time_axis(
    t_if_us: float, thickness_mm: float, c_m: float, sampling_mhz: float
) -> np.ndarray:
    """Construct discrete time axis from t_if - 2 us to t_bw + 3 us."""
    start = t_if_us - 2.0
    stop = t_if_us + 2.0 * thickness_mm / c_m + 3.0
    n_t = int(np.ceil((stop - start) * sampling_mhz))
    return start + np.arange(n_t) / sampling_mhz


def _add_block_echoes(
    rf: np.ndarray,
    time_us: np.ndarray,
    t_if_us: float,
    block: CalibrationBlock,
) -> None:
    """Add interface, back-wall, and per-hole FBH echoes into rf."""
    mat, probe, water = block.material, block.probe, block.water
    c_m = mat.velocity_m_s * 1e-3
    r_wm = reflection_coefficient(water.impedance, mat.impedance)
    t_rt = round_trip_transmission(water.impedance, mat.impedance)
    d = block.thickness_mm
    bw = float(-r_wm * t_rt * attenuation_factor(d, mat, probe) * beam_factor(d, probe))
    t_bw = t_if_us + 2.0 * d / c_m

    rf += r_wm * pulse(time_us - t_if_us, probe)
    rf += bw * pulse(time_us - t_bw, probe)
    for i, z in enumerate(block.metal_paths_mm):
        t_z = t_if_us + 2.0 * z / c_m
        amp = fbh_amplitude(block.fbh_diameter_mm, z, mat, probe, water)
        rf[i] += (-amp) * pulse(time_us - t_z, probe)


def simulate_calibration_block(
    block: CalibrationBlock,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Simulate ASTM E127-style flat calibration block A-scans (Synthetic data)."""
    probe, mat, water = block.probe, block.material, block.water
    c_w = water.velocity_m_s * 1e-3
    c_m = mat.velocity_m_s * 1e-3
    t_if_us = 2.0 * probe.water_path_mm / c_w
    n_paths = len(block.metal_paths_mm)
    time_us = _calibration_time_axis(t_if_us, block.thickness_mm, c_m, probe.sampling_mhz)

    rng = np.random.default_rng(block.seed)
    spec = BarSpec(
        material=mat,
        probe=probe,
        water=water,
        diameter_mm=block.thickness_mm,
        seed=block.seed,
    )
    rf = grain_noise((n_paths,), time_us, np.full(n_paths, t_if_us), spec, rng)
    _add_block_echoes(rf, time_us, t_if_us, block)

    rf *= 10.0 ** (probe.gain_db / 20.0)
    rf += probe.electronic_noise * rng.standard_normal(rf.shape)
    ascans = np.clip(rf, -1.0, 1.0).astype(np.float32)
    return time_us, ascans, t_if_us


def measure_dac(
    time_us: np.ndarray,
    ascans: np.ndarray,
    t_if_us: float,
    block: CalibrationBlock,
) -> list[DacPoint]:
    """Measure peak envelope within +-1.5 mm depth around each metal path."""
    env = envelope(ascans)
    depth = depth_of_time(time_us, np.asarray(t_if_us), block.material)
    points: list[DacPoint] = []
    for i, z in enumerate(block.metal_paths_mm):
        gate = (depth >= z - 1.5) & (depth <= z + 1.5)
        peak = float(np.max(env[i, gate]))
        points.append(DacPoint(depth_mm=float(z), amplitude=peak))
    return points


def dac_curve(points: list[DacPoint]) -> Callable[[np.ndarray], np.ndarray]:
    """Return linear interpolation of amplitude vs depth (clamped at ends)."""
    if not points:
        raise ValueError("points list must not be empty")
    sorted_points = sorted(points, key=lambda p: p.depth_mm)
    xp = np.array([p.depth_mm for p in sorted_points], dtype=np.float64)
    fp = np.array([p.amplitude for p in sorted_points], dtype=np.float64)

    def curve(depth_mm: np.ndarray) -> np.ndarray:
        return np.interp(depth_mm, xp, fp)

    return curve
