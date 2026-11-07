"""Tests for titanium macrozone grain noise simulator. Synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from utriage.physics.simulator import (  # noqa: E402
    BarSpec,
    depth_of_time,
    load_material,
    load_probe,
    simulate_bar,
)
from utriage.physics.ti_noise import (  # noqa: E402
    TiNoiseConfig,
    load_ti_noise,
    simulate_ti_bar,
    strength_field,
    zone_index,
    zone_level,
)


def test_strength_field_mean_and_autocorrelation() -> None:
    cfg = TiNoiseConfig(
        corr_axial_mm=6.0,
        corr_angle_deg=15.0,
        field_sigma=0.4,
        zone_edges_mm=(2.0, 12.0, 22.0, 36.0),
        zone_levels=(1.0, 1.6, 1.3),
    )
    rng = np.random.default_rng(0)
    field = strength_field(64, 120, (1.0, 3.0), cfg, rng)
    assert abs(float(np.mean(field)) - 1.0) < 0.1
    assert float(np.std(field)) > 0.0

    corr_ax = float(np.corrcoef(field[:-1, :].ravel(), field[1:, :].ravel())[0, 1])
    corr_an = float(np.corrcoef(field[:, :-1].ravel(), field[:, 1:].ravel())[0, 1])
    corr_wrap = float(np.corrcoef(field[:, 0], field[:, -1])[0, 1])

    rng_white = np.random.default_rng(0)
    white = rng_white.standard_normal((64, 120))
    corr_white_ax = float(np.corrcoef(white[:-1, :].ravel(), white[1:, :].ravel())[0, 1])

    assert corr_ax > 0.0
    assert corr_an > 0.0
    assert corr_ax > corr_white_ax
    assert corr_wrap > 0.3


def test_strength_field_zero_sigma() -> None:
    cfg = TiNoiseConfig(
        corr_axial_mm=6.0,
        corr_angle_deg=15.0,
        field_sigma=0.0,
        zone_edges_mm=(2.0, 12.0, 22.0, 36.0),
        zone_levels=(1.0, 1.6, 1.3),
    )
    field = strength_field(64, 120, (1.0, 3.0), cfg)
    assert np.allclose(field, 1.0)


@pytest.mark.parametrize(
    ("depth", "expected_level", "expected_idx"),
    [
        (0.0, 1.0, None),
        (1.0, 1.0, None),
        (2.0, 1.0, 0),
        (5.0, 1.0, 0),
        (12.0, 1.6, 1),
        (18.0, 1.6, 1),
        (22.0, 1.3, 2),
        (30.0, 1.3, 2),
        (36.0, 1.0, None),
        (45.0, 1.0, None),
    ],
)
def test_zone_level_and_index(
    depth: float,
    expected_level: float,
    expected_idx: int | None,
) -> None:
    cfg = TiNoiseConfig(
        corr_axial_mm=6.0,
        corr_angle_deg=15.0,
        field_sigma=0.4,
        zone_edges_mm=(2.0, 12.0, 22.0, 36.0),
        zone_levels=(1.0, 1.6, 1.3),
    )
    assert zone_index(depth, cfg) == expected_idx
    lvl = zone_level(np.array([depth]), cfg)
    assert np.isclose(lvl[0], expected_level)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "corr_axial_mm": 6.0,
                "corr_angle_deg": 15.0,
                "field_sigma": 0.4,
                "zone_edges_mm": (2.0, 12.0),
                "zone_levels": (1.0, 1.6),
            },
            "zone_levels length",
        ),
        (
            {
                "corr_axial_mm": 6.0,
                "corr_angle_deg": 15.0,
                "field_sigma": 0.4,
                "zone_edges_mm": (2.0,),
                "zone_levels": (),
            },
            "at least 2 edges",
        ),
        (
            {
                "corr_axial_mm": 6.0,
                "corr_angle_deg": 15.0,
                "field_sigma": 0.4,
                "zone_edges_mm": (12.0, 2.0),
                "zone_levels": (1.0,),
            },
            "strictly increasing",
        ),
        (
            {
                "corr_axial_mm": 6.0,
                "corr_angle_deg": 15.0,
                "field_sigma": 0.4,
                "zone_edges_mm": (2.0, 2.0),
                "zone_levels": (1.0,),
            },
            "strictly increasing",
        ),
        (
            {
                "corr_axial_mm": 6.0,
                "corr_angle_deg": 15.0,
                "field_sigma": -0.1,
                "zone_edges_mm": (2.0, 12.0),
                "zone_levels": (1.0,),
            },
            "field_sigma",
        ),
        (
            {
                "corr_axial_mm": 0.0,
                "corr_angle_deg": 15.0,
                "field_sigma": 0.4,
                "zone_edges_mm": (2.0, 12.0),
                "zone_levels": (1.0,),
            },
            "correlation lengths",
        ),
        (
            {
                "corr_axial_mm": 6.0,
                "corr_angle_deg": -1.0,
                "field_sigma": 0.4,
                "zone_edges_mm": (2.0, 12.0),
                "zone_levels": (1.0,),
            },
            "correlation lengths",
        ),
    ],
)
def test_config_validation_errors(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TiNoiseConfig(**kwargs)


def test_load_ti_noise_real_yaml() -> None:
    cfg = load_ti_noise()
    assert cfg.corr_axial_mm == 6.0
    assert cfg.corr_angle_deg == 15.0
    assert cfg.field_sigma == 0.4
    assert cfg.zone_edges_mm == (2.0, 12.0, 22.0, 36.0)
    assert cfg.zone_levels == (1.0, 1.6, 1.3)


def test_simulate_ti_bar_properties() -> None:
    mat = load_material("ti64")
    probe = load_probe()
    water = load_material("water")
    spec = BarSpec(
        material=mat,
        probe=probe,
        water=water,
        length_mm=16.0,
        n_angle=24,
        seed=42,
    )
    cfg = load_ti_noise()

    scan1 = simulate_ti_bar(spec, cfg)
    scan2 = simulate_ti_bar(spec, cfg)
    assert np.array_equal(scan1.ascans, scan2.ascans)

    scan_base = simulate_bar(spec)
    assert scan1.ascans.shape == scan_base.ascans.shape

    depth = depth_of_time(scan1.time_us[None, None, :], scan1.t_if_us[..., None], spec.material)
    gate = (depth >= 14.0) & (depth <= 20.0)
    rms_ti = float(np.sqrt(np.mean(scan1.ascans[gate] ** 2)))
    rms_base = float(np.sqrt(np.mean(scan_base.ascans[gate] ** 2)))
    assert rms_ti != rms_base
    assert abs(rms_ti - rms_base) > 1e-4
