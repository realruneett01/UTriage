"""Tests for conventional rules detection engine (project plan 2.1)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

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
    EnvelopeView,
    Peak,
    amplitude_map,
    detect_indications,
    interface_times,
    label_clusters,
    local_snr,
)
from utriage.rules.types import Indication  # noqa: E402


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
    setup = SimpleNamespace(material=mat, probe=probe, water=water)
    dac = np.vectorize(lambda z: abs(fbh_amplitude(0.8, float(z), setup)))
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


def _fbh(diameter_mm: float = 0.8, depth_mm: float = 10.0) -> Reflector:
    return Reflector(axial_mm=6.0, angle_deg=45.0, depth_mm=depth_mm, diameter_mm=diameter_mm)


def _primary_indication(scan: BarScan, ctx: SimContext) -> Indication:
    inds = detect_indications(scan, ctx.dac, ctx.cfg)
    assert len(inds) >= 1
    return inds[0]


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [
        (0.3, 3.0),
        (0.9, np.nan),
    ],
)
def test_interface_times_exact_step(threshold: float, expected: float) -> None:
    time_us = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    env = np.array([0.1, 0.2, 0.4, 0.8, 0.3])
    t_if = interface_times(env, time_us, threshold=threshold)
    if np.isnan(expected):
        assert np.isnan(t_if)
    else:
        assert t_if == pytest.approx(expected)


@pytest.mark.parametrize(
    ("idx", "expected"),
    [
        (0, 2.0),
        (1, np.nan),
    ],
)
def test_interface_times_2d(idx: int, expected: float) -> None:
    time_us = np.array([1.0, 2.0, 3.0, 4.0])
    env = np.array([[0.0, 0.5, 0.8, 0.2], [0.0, 0.1, 0.2, 0.0]])
    t_if = interface_times(env, time_us, threshold=0.4)
    if np.isnan(expected):
        assert np.isnan(t_if[idx])
    else:
        assert t_if[idx] == pytest.approx(expected)


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
    scan = _sim_bar(ctx, reflectors=(_fbh(depth_mm=depth_mm),), seed=42)
    top = _primary_indication(scan, ctx)
    assert 80.0 <= top.amplitude_pct <= 125.0
    assert abs(top.depth_mm - depth_mm) <= 1.0


def test_pure_noise_gives_no_indication(ctx: SimContext) -> None:
    scan = _sim_bar(ctx, reflectors=(), grain_scale=0.5, seed=123)
    inds = detect_indications(scan, ctx.dac, ctx.cfg)
    assert len(inds) == 0


def test_fbh_diameter_scaling(ctx: SimContext) -> None:
    ind_s = _primary_indication(_sim_bar(ctx, (_fbh(0.8),), grain_scale=0.1, seed=7), ctx)
    ind_l = _primary_indication(_sim_bar(ctx, (_fbh(1.6),), grain_scale=0.1, seed=7), ctx)
    ratio = ind_l.amplitude_pct / ind_s.amplitude_pct
    assert 3.0 <= ratio <= 5.0


def test_amplitude_map(ctx: SimContext) -> None:
    scan = _sim_bar(ctx, (_fbh(),), seed=42)
    env = envelope(scan.ascans)
    t_if = interface_times(env, scan.time_us, ctx.cfg.gates.interface_threshold)
    view = EnvelopeView(env=env, time_us=scan.time_us, t_if_us=t_if, scan=scan)
    pct_map, depth_map = amplitude_map(view, ctx.dac, ctx.cfg)
    assert pct_map.shape == scan.ascans.shape[:2]
    assert depth_map.shape == scan.ascans.shape[:2]
    assert np.nanmax(pct_map) > 80.0


def test_snr_planted_fbh(ctx: SimContext) -> None:
    scan = _sim_bar(ctx, (_fbh(),), seed=99)
    top = _primary_indication(scan, ctx)
    assert top.snr_db > 0.0
    env = envelope(scan.ascans)
    t_if = interface_times(env, scan.time_us, ctx.cfg.gates.interface_threshold)
    view = EnvelopeView(env=env, time_us=scan.time_us, t_if_us=t_if, scan=scan)
    noise_snr = local_snr(view, Peak(1, 18, 10.0), ctx.cfg)
    assert top.snr_db > noise_snr
