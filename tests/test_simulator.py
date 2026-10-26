"""Phase 1 unit tests for the immersion UT bar simulator."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from utriage.physics.simulator import (
    Artefact,
    BarScan,
    BarSpec,
    Reflector,
    depth_of_time,
    envelope,
    fbh_amplitude,
    load_material,
    load_probe,
    simulate_bar,
)


@pytest.fixture
def base_quiet_spec() -> BarSpec:
    """Quiet spec with zero grain and electronic noise, zero attenuation."""
    mat = dataclasses.replace(
        load_material("ti64"),
        grain_noise=0.0,
        attenuation_db_mm_mhz=0.0,
    )
    probe = dataclasses.replace(load_probe("p5"), electronic_noise=0.0)
    return BarSpec(
        material=mat,
        probe=probe,
        water=load_material("water"),
        diameter_mm=30.0,
        length_mm=4.0,
        axial_step_mm=1.0,
        n_angle=8,
        seed=42,
    )


@pytest.fixture
def quiet_scan_fbh(base_quiet_spec: BarSpec) -> BarScan:
    """Quiet scan with one FBH at depth 10 mm on grid point (axial=2, angle=0)."""
    ref = Reflector(axial_mm=2.0, angle_deg=0.0, depth_mm=10.0, diameter_mm=1.0)
    spec = dataclasses.replace(base_quiet_spec, reflectors=(ref,))
    return simulate_bar(spec)


def _window_peak(
    time_us: np.ndarray,
    env: np.ndarray,
    t_center: float,
    half_window: float = 0.5,
) -> tuple[float, float]:
    """Return peak time and amplitude in a window around t_center."""
    mask = np.abs(time_us - t_center) <= half_window
    idx = int(np.argmax(np.where(mask, env, -1.0)))
    return float(time_us[idx]), float(env[idx])


def _region_mask(
    scan: BarScan,
    axial_bounds: tuple[float, float],
    angle_bounds: tuple[float, float],
) -> np.ndarray:
    """Return 2D boolean mask of (axial, angle) locations within bounds."""
    in_ax = (scan.axial_mm >= axial_bounds[0]) & (scan.axial_mm <= axial_bounds[1])
    in_an = (scan.angle_deg >= angle_bounds[0]) & (scan.angle_deg <= angle_bounds[1])
    return in_ax[:, None] & in_an[None, :]


@pytest.mark.parametrize("target", ["interface", "fbh", "backwall"])
def test_arrival_times(quiet_scan_fbh: BarScan, target: str) -> None:
    scan = quiet_scan_fbh
    c_w = scan.spec.water.velocity_m_s * 1e-3
    c_m = scan.spec.material.velocity_m_s * 1e-3
    t_if = 2.0 * scan.spec.probe.water_path_mm / c_w
    expected_times = {
        "interface": t_if,
        "fbh": t_if + 2.0 * 10.0 / c_m,
        "backwall": t_if + 2.0 * scan.spec.diameter_mm / c_m,
    }
    t_exp = expected_times[target]
    env = envelope(scan.ascans[2, 0])
    t_peak, _ = _window_peak(scan.time_us, env, t_exp, half_window=0.5)
    assert abs(t_peak - t_exp) <= 2.0 / scan.fs_mhz


@pytest.mark.parametrize("depth_mm", [8.0, 15.0, 20.0])
def test_amplitude_exactness(base_quiet_spec: BarSpec, depth_mm: float) -> None:
    ref = Reflector(axial_mm=2.0, angle_deg=0.0, depth_mm=depth_mm, diameter_mm=1.0)
    spec = dataclasses.replace(base_quiet_spec, reflectors=(ref,))
    scan = simulate_bar(spec)
    env = envelope(scan.ascans[2, 0])
    c_w = spec.water.velocity_m_s * 1e-3
    c_m = spec.material.velocity_m_s * 1e-3
    t_fbh = (2.0 * spec.probe.water_path_mm / c_w) + (2.0 * depth_mm / c_m)
    _, peak_val = _window_peak(scan.time_us, env, t_fbh, half_window=0.5)
    expected = abs(fbh_amplitude(1.0, depth_mm, spec.material, spec.probe, spec.water))
    assert abs(peak_val - expected) / expected < 0.03


def test_larger_fbh_larger_echo(base_quiet_spec: BarSpec) -> None:
    peaks: list[float] = []
    for dia in (0.8, 1.6):
        ref = Reflector(axial_mm=2.0, angle_deg=0.0, depth_mm=10.0, diameter_mm=dia)
        spec = dataclasses.replace(base_quiet_spec, reflectors=(ref,))
        scan = simulate_bar(spec)
        c_w = spec.water.velocity_m_s * 1e-3
        c_m = spec.material.velocity_m_s * 1e-3
        t_fbh = (2.0 * spec.probe.water_path_mm / c_w) + (2.0 * 10.0 / c_m)
        _, peak = _window_peak(scan.time_us, envelope(scan.ascans[2, 0]), t_fbh)
        peaks.append(peak)
    ratio = peaks[1] / peaks[0]
    assert abs(ratio - 4.0) / 4.0 < 0.05


def test_grain_noise_scaling(base_quiet_spec: BarSpec) -> None:
    stds: list[float] = []
    for gn in (0.02, 0.04):
        mat = dataclasses.replace(base_quiet_spec.material, grain_noise=gn)
        spec = dataclasses.replace(base_quiet_spec, material=mat, reflectors=(), seed=42)
        scan = simulate_bar(spec)
        depth = depth_of_time(scan.time_us[None, None, :], scan.t_if_us[..., None], mat)
        mask = (depth >= 5.0) & (depth <= 20.0)
        stds.append(float(np.std(scan.ascans[mask])))
    ratio = stds[1] / stds[0]
    assert abs(ratio - 2.0) <= 0.1


def test_determinism(base_quiet_spec: BarSpec) -> None:
    mat = load_material("ti64")
    spec1 = dataclasses.replace(base_quiet_spec, material=mat, seed=42)
    spec2 = dataclasses.replace(base_quiet_spec, material=mat, seed=42)
    spec3 = dataclasses.replace(base_quiet_spec, material=mat, seed=99)
    scan1 = simulate_bar(spec1)
    scan2 = simulate_bar(spec2)
    scan3 = simulate_bar(spec3)
    assert np.array_equal(scan1.ascans, scan2.ascans)
    assert not np.array_equal(scan1.ascans, scan3.ascans)


def _verify_missing_coverage(scan: BarScan, in_region: np.ndarray) -> None:
    assert np.all(scan.ascans[in_region] == 0.0)


def _verify_lost_backwall(scan: BarScan, in_region: np.ndarray) -> None:
    c_m = scan.spec.material.velocity_m_s * 1e-3
    t_bw = scan.t_if_us + 2.0 * scan.spec.diameter_mm / c_m
    env = envelope(scan.ascans)
    bw_peaks = np.array(
        [
            _window_peak(scan.time_us, env[i, j], t_bw[i, j])[1]
            for i in range(scan.axial_mm.size)
            for j in range(scan.angle_deg.size)
        ]
    ).reshape(scan.ascans.shape[:2])
    assert bw_peaks[in_region].max() < 0.2 * bw_peaks[~in_region].min()


def _verify_waterpath_drift(scan: BarScan, in_region: np.ndarray) -> None:
    assert scan.t_if_us[in_region].min() > scan.t_if_us[~in_region].max()


def _verify_bubble(scan: BarScan, in_region: np.ndarray) -> None:
    env = envelope(scan.ascans)
    if_peaks = np.array(
        [
            _window_peak(scan.time_us, env[i, j], scan.t_if_us[i, j])[1]
            for i in range(scan.axial_mm.size)
            for j in range(scan.angle_deg.size)
        ]
    ).reshape(scan.ascans.shape[:2])
    assert if_peaks[in_region].max() < if_peaks[~in_region].min()


_ARTEFACT_VERIFIERS = {
    "missing_coverage": _verify_missing_coverage,
    "lost_backwall": _verify_lost_backwall,
    "waterpath_drift": _verify_waterpath_drift,
    "bubble": _verify_bubble,
}


@pytest.mark.parametrize("kind", list(_ARTEFACT_VERIFIERS.keys()))
def test_artefacts(base_quiet_spec: BarSpec, kind: str) -> None:
    art = Artefact(kind=kind, axial_mm=(1.0, 2.0), angle_deg=(0.0, 90.0), severity=1.0)
    spec = dataclasses.replace(base_quiet_spec, artefacts=(art,))
    scan = simulate_bar(spec)
    in_region = _region_mask(scan, (1.0, 2.0), (0.0, 90.0))
    _ARTEFACT_VERIFIERS[kind](scan, in_region)


def test_output_contract(base_quiet_spec: BarSpec) -> None:
    scan = simulate_bar(base_quiet_spec)
    expected_shape = (len(scan.axial_mm), scan.spec.n_angle, len(scan.time_us))
    assert scan.ascans.dtype == np.float32
    assert scan.ascans.shape == expected_shape
    assert (scan.ascans >= -1.0).all()
    assert (scan.ascans <= 1.0).all()
