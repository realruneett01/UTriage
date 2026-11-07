"""Tests for titanium zone calibration. Synthetic data."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from utriage.physics.simulator import (  # noqa: E402
    BarSpec,
    load_material,
    load_probe,
)
from utriage.physics.ti_calibration import (  # noqa: E402
    CalibrationRecord,
    TiCalConfig,
    ZoneCalibration,
    calibrate_zones,
    hole_index,
    load_ti_calibration,
    noise_stats,
    record_to_dict,
    reference_block_spec,
    signal_a,
    zone_alarm,
)
from utriage.physics.ti_noise import (  # noqa: E402
    load_ti_noise,
)


@pytest.fixture
def default_cal() -> TiCalConfig:
    """Default calibration configuration."""
    return TiCalConfig(
        ref_fbh_mm=0.8,
        signal_window_px=3,
        noise_exclude_px=6,
        noise_margin=0.10,
        cap_fraction=0.70,
        min_snr=2.0,
        block_seed=1000,
    )


def test_load_ti_calibration_yaml(default_cal: TiCalConfig) -> None:
    """Load calibration parameters from configs/ti_mode.yaml."""
    cfg = load_ti_calibration()
    assert cfg == default_cal


@pytest.mark.parametrize(
    (
        "a",
        "b",
        "c",
        "expected_snr",
        "expected_alarm",
        "expected_cap_bound",
        "expected_inspectable",
        "expected_reason",
    ),
    [
        (1.0, 0.1, 0.3, 4.5, 0.33, False, True, ""),
        (0.4, 0.1, 0.35, 1.2, 0.28, True, False, "alarm cap is not above peak noise"),
        (0.45, 0.1, 0.3, 1.75, 0.315, True, False, "SNR below placeholder minimum"),
        (1.0, 0.3, 0.2, None, 0.22, False, False, "degenerate noise region"),
        (1.0, 0.3, 0.3, None, 0.33, False, False, "degenerate noise region"),
    ],
)
def test_zone_alarm_branches(
    a: float,
    b: float,
    c: float,
    expected_snr: float | None,
    expected_alarm: float,
    expected_cap_bound: bool,
    expected_inspectable: bool,
    expected_reason: str,
    default_cal: TiCalConfig,
) -> None:
    """Test zone_alarm logic across all operating branches."""
    snr, alarm, cap_bound, inspectable, reason = zone_alarm(a, b, c, default_cal)
    if expected_snr is None:
        assert math.isnan(snr)
    else:
        assert np.isclose(snr, expected_snr)
    assert np.isclose(alarm, expected_alarm)
    assert cap_bound == expected_cap_bound
    assert inspectable == expected_inspectable
    assert reason == expected_reason


def test_signal_a_angle_wrap(default_cal: TiCalConfig) -> None:
    """Test signal_a 3x3 averaging with periodic angle wrap at column 0."""
    cscan = np.zeros((8, 12), dtype=float)
    # 3x3 patch around (row=4, col=0) wrapping to col=11 and col=1
    patch = np.array([[1.0, 2.0, 3.0], [4.0, 10.0, 5.0], [6.0, 7.0, 8.0]])
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            cscan[4 + dr, (0 + dc) % 12] = patch[dr + 1, dc + 1]

    # Search from (4, 1); peak is at (4, 0)
    result = signal_a(cscan, (4, 1), default_cal)
    assert np.isclose(result, float(np.mean(patch)))


def test_noise_stats_exclusion(default_cal: TiCalConfig) -> None:
    """Test noise_stats excludes designated hole regions and handles empty masks."""
    cscan = np.ones((15, 15), dtype=float)
    cscan[7, 7] = 99.0
    b, c = noise_stats(cscan, [(7, 7)], default_cal)
    assert np.isclose(b, 1.0)
    assert np.isclose(c, 1.0)

    # Empty mask error
    tiny_cscan = np.ones((5, 5), dtype=float)
    with pytest.raises(ValueError, match="No noise pixels remain"):
        noise_stats(tiny_cscan, [(2, 2)], default_cal)


def test_reference_block_spec_and_indices(default_cal: TiCalConfig) -> None:
    """Test reference block generation and reflector index mapping."""
    mat = load_material("ti64")
    probe = load_probe("p5")
    water = load_material("water")
    part = BarSpec(material=mat, probe=probe, water=water, length_mm=64.0, n_angle=120)
    noise = load_ti_noise()

    ref_spec = reference_block_spec(part, noise, default_cal)
    assert len(ref_spec.reflectors) == 3
    assert ref_spec.seed == default_cal.block_seed
    assert ref_spec.artefacts == ()

    expected_depths = (7.0, 17.0, 29.0)
    for refl, exp_d in zip(ref_spec.reflectors, expected_depths, strict=True):
        assert refl.kind == "fbh"
        assert refl.diameter_mm == default_cal.ref_fbh_mm
        assert np.isclose(refl.depth_mm, exp_d)
        idx = hole_index(ref_spec, refl)
        assert idx[0] == round(refl.axial_mm / ref_spec.axial_step_mm)


def test_calibrate_zones_small_bar() -> None:
    """Test calibrate_zones determinism and basic properties on a small synthetic bar."""
    mat = load_material("ti64")
    probe = load_probe("p5")
    water = load_material("water")
    part = BarSpec(
        material=mat,
        probe=probe,
        water=water,
        length_mm=24.0,
        diameter_mm=50.0,
        axial_step_mm=1.0,
        n_angle=36,
    )
    noise = load_ti_noise()
    cal = load_ti_calibration()

    rec1 = calibrate_zones(part, noise, cal)
    rec2 = calibrate_zones(part, noise, cal)

    assert rec1.calib_id == rec2.calib_id
    assert len(rec1.calib_id) == 16
    assert len(rec1.zones) == len(noise.zone_levels)
    assert rec1.ref_fbh_mm == cal.ref_fbh_mm
    assert rec1.label == "Synthetic data: simulated reference block"

    for z in rec1.zones:
        assert z.signal_a > z.noise_mean_b


def test_record_to_dict_serialization() -> None:
    """Test record_to_dict JSON serialization and NaN handling."""
    zone = ZoneCalibration(
        zone=0,
        depth_mm=(2.0, 12.0),
        ref_depth_mm=7.0,
        signal_a=0.8,
        noise_mean_b=0.2,
        noise_peak_c=0.3,
        snr=float("nan"),
        alarm_level=0.33,
        cap_bound=False,
        inspectable=False,
        reason="degenerate noise region",
    )
    rec = CalibrationRecord(
        calib_id="0123456789abcdef",
        zones=(zone,),
        ref_fbh_mm=0.8,
        block_seed=1000,
        config_sha256="deadbeef",
    )
    d = record_to_dict(rec)
    raw = json.dumps(d)
    parsed = json.loads(raw)
    assert parsed["calib_id"] == "0123456789abcdef"
    assert parsed["zones"][0]["snr"] is None
