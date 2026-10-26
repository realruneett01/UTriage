"""Tests for conventional rules detection engine (project plan 2.1)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure local worktree src/ takes precedence over any installed package
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from utriage.physics.simulator import (  # noqa: E402
    BarScan,
    BarSpec,
    Material,
    Probe,
    Reflector,
    envelope,
    fbh_amplitude,
    load_material,
    load_probe,
    simulate_bar,
)
from utriage.rules.config import AcceptanceConfig, load_acceptance  # noqa: E402
from utriage.rules.engine import (  # noqa: E402
    Dac,
    detect_indications,
    interface_times,
    label_clusters,
    local_snr,
)


@dataclass(frozen=True)
class SimContext:
    material: Material
    probe: Probe
    water: Material
    cfg: AcceptanceConfig
    dac: Dac


@pytest.fixture
def ctx() -> SimContext:
    mat = load_material("ti64")
    probe = load_probe("p5")
    water = load_material("water")
    cfg = load_acceptance()
    dac = np.vectorize(lambda z: abs(fbh_amplitude(0.8, float(z), mat, probe, water)))
    return SimContext(mat, probe, water, cfg, dac)


def _sim_bar(
    c: SimContext,
    reflectors: tuple[Reflector, ...] = (),
    grain_scale: float = 0.2,
    seed: int = 42,
) -> BarScan:
    low_noise = Material(
        name=c.material.name,
        velocity_m_s=c.material.velocity_m_s,
        density_kg_m3=c.material.density_kg_m3,
        attenuation_db_mm_mhz=c.material.attenuation_db_mm_mhz,
        grain_noise=c.material.grain_noise * grain_scale,
    )
    spec = BarSpec(
        material=low_noise,
        probe=c.probe,
        water=c.water,
        diameter_mm=30.0,
        length_mm=12.0,
        axial_step_mm=1.0,
        n_angle=24,
        reflectors=reflectors,
        seed=seed,
    )
    return simulate_bar(spec)


def test_interface_times_exact_step() -> None:
    time_us = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    env = np.array([0.1, 0.2, 0.4, 0.8, 0.3])
    t_if = interface_times(env, time_us, threshold=0.3)
    assert t_if == pytest.approx(3.0)
    no_cross = interface_times(env, time_us, threshold=0.9)
    assert np.isnan(no_cross)


def test_interface_times_2d() -> None:
    time_us = np.array([1.0, 2.0, 3.0, 4.0])
    env = np.array([[0.0, 0.5, 0.8, 0.2], [0.0, 0.1, 0.2, 0.0]])
    t_if = interface_times(env, time_us, threshold=0.4)
    assert t_if[0] == pytest.approx(2.0)
    assert np.isnan(t_if[1])


def test_label_clusters_wraps_angle() -> None:
    mask = np.zeros((10, 24), dtype=bool)
    mask[3, 0] = True
    mask[3, 23] = True
    mask[4, 0] = True
    mask[8, 12] = True
    labeled, n_clusters = label_clusters(mask)
    assert n_clusters == 2
    assert labeled[3, 0] == labeled[3, 23] == labeled[4, 0]
    assert labeled[8, 12] != labeled[3, 0]


@pytest.mark.parametrize("depth_mm", [8.0, 15.0])
def test_planted_fbh_detected(ctx: SimContext, depth_mm: float) -> None:
    ref = Reflector(axial_mm=6.0, angle_deg=45.0, depth_mm=depth_mm, diameter_mm=0.8)
    scan = _sim_bar(ctx, reflectors=(ref,), grain_scale=0.2, seed=42)
    inds = detect_indications(scan, ctx.dac, ctx.cfg)
    assert len(inds) >= 1
    top = inds[0]
    assert 80.0 <= top.amplitude_pct <= 125.0
    assert abs(top.depth_mm - depth_mm) <= 1.0


def test_pure_noise_gives_no_indication(ctx: SimContext) -> None:
    scan = _sim_bar(ctx, reflectors=(), grain_scale=0.5, seed=123)
    inds = detect_indications(scan, ctx.dac, ctx.cfg)
    assert len(inds) == 0


def test_fbh_diameter_scaling(ctx: SimContext) -> None:
    ref_small = Reflector(axial_mm=6.0, angle_deg=45.0, depth_mm=10.0, diameter_mm=0.8)
    ref_large = Reflector(axial_mm=6.0, angle_deg=45.0, depth_mm=10.0, diameter_mm=1.6)
    scan_s = _sim_bar(ctx, reflectors=(ref_small,), grain_scale=0.1, seed=7)
    scan_l = _sim_bar(ctx, reflectors=(ref_large,), grain_scale=0.1, seed=7)
    ind_s = detect_indications(scan_s, ctx.dac, ctx.cfg)[0]
    ind_l = detect_indications(scan_l, ctx.dac, ctx.cfg)[0]
    ratio = ind_l.amplitude_pct / ind_s.amplitude_pct
    assert 3.0 <= ratio <= 5.0


def test_snr_planted_fbh(ctx: SimContext) -> None:
    ref = Reflector(axial_mm=6.0, angle_deg=45.0, depth_mm=10.0, diameter_mm=0.8)
    scan = _sim_bar(ctx, reflectors=(ref,), grain_scale=0.2, seed=99)
    inds = detect_indications(scan, ctx.dac, ctx.cfg)
    assert len(inds) >= 1
    top = inds[0]
    assert top.snr_db > 0.0
    env = envelope(scan.ascans)
    t_if = interface_times(env, scan.time_us, ctx.cfg.gates.interface_threshold)
    noise_snr = local_snr(env, scan.time_us, t_if, scan, 1, 18, 10.0, ctx.cfg)
    assert top.snr_db > noise_snr
