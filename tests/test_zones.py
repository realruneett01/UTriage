"""Tests for depth zone alarms and indication conversion."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from utriage.physics.simulator import BarScan, BarSpec, Material, Probe  # noqa: E402
from utriage.rules.zones import ZoneAlarm, ZoneGate, to_indications, zone_alarms  # noqa: E402


def _make_test_scan(
    shape: tuple[int, int, int] = (6, 12, 64),
    step_mm: float = 1.0,
) -> BarScan:
    """Build a deterministic BarScan for unit tests."""
    n_ax, n_an, n_t = shape
    mat = Material("ti64", 6100.0, 4420.0, 0.02, 0.0)
    probe = Probe(5.0, 0.5, 25.0, 10.0, 15.0, 2.0, 50.0, 20.0, 0.0, 1.0)
    water = Material("water", 1480.0, 1000.0, 0.002, 0.0)
    spec = BarSpec(
        material=mat,
        probe=probe,
        water=water,
        diameter_mm=40.0,
        length_mm=n_ax * step_mm,
        axial_step_mm=step_mm,
        n_angle=n_an,
    )
    ascans = np.zeros(shape, dtype=np.float32)
    t0_us = 10.0
    fs_mhz = 50.0
    axial_mm = np.arange(n_ax, dtype=float) * step_mm
    angle_deg = np.arange(n_an, dtype=float) * (360.0 / n_an)
    t_if_us = np.full((n_ax, n_an), t0_us)
    missing = np.zeros((n_ax, n_an), dtype=bool)
    return BarScan(ascans, t0_us, fs_mhz, axial_mm, angle_deg, t_if_us, spec, missing)


def _sample_ok_alarm() -> ZoneAlarm:
    """Create sample ok alarm."""
    return ZoneAlarm(
        kind="alarm",
        zone=1,
        calib_id="block_a",
        axial_idx=2,
        angle_idx=4,
        axial_mm=2.0,
        angle_deg=120.0,
        depth_mm=5.5,
        peak=0.6,
        pct_of_alarm=120.0,
        n_ascans=3,
        axial_extent_mm=2.0,
    )


def _sample_skip_alarm() -> ZoneAlarm:
    """Create sample uninspectable alarm."""
    return ZoneAlarm(
        kind="zone_not_inspectable",
        zone=2,
        calib_id="block_b",
        axial_idx=-1,
        angle_idx=-1,
        axial_mm=float("nan"),
        angle_deg=float("nan"),
        depth_mm=10.0,
        peak=float("nan"),
        pct_of_alarm=float("nan"),
        n_ascans=72,
        axial_extent_mm=float("nan"),
    )


def test_planted_reflector_alarm() -> None:
    """Check detection of planted reflector above threshold."""
    scan = _make_test_scan()
    scan.ascans[2, 3, 20] = 0.8
    gate = ZoneGate(
        zone=1, depth_mm=(0.5, 3.0), alarm_level=0.5, inspectable=True, calib_id="cal_1"
    )
    alarms = zone_alarms(scan, [gate])
    assert len(alarms) == 1
    alarm = alarms[0]
    expected_attrs = ("alarm", 1, "cal_1", 2, 3, 1)
    actual_attrs = (
        alarm.kind,
        alarm.zone,
        alarm.calib_id,
        alarm.axial_idx,
        alarm.angle_idx,
        alarm.n_ascans,
    )
    assert actual_attrs == expected_attrs
    assert (alarm.peak, alarm.pct_of_alarm) == (
        pytest.approx(0.8, rel=1e-3),
        pytest.approx(160.0, rel=1e-3),
    )
    assert (alarm.axial_extent_mm, 0.5 <= alarm.depth_mm <= 3.0) == (pytest.approx(1.0), True)


@pytest.mark.parametrize("alarm_level,expected_count", [(0.5, 1), (999.0, 0)])
def test_alarm_threshold_suppression(alarm_level: float, expected_count: int) -> None:
    """Check that high threshold yields no alarms."""
    scan = _make_test_scan()
    scan.ascans[1, 1, 20] = 0.7
    gate = ZoneGate(
        zone=1, depth_mm=(0.5, 3.0), alarm_level=alarm_level, inspectable=True, calib_id="c"
    )
    alarms = zone_alarms(scan, [gate])
    assert len(alarms) == expected_count


def test_gate_not_inspectable() -> None:
    """Check behavior when zone is marked uninspectable."""
    scan = _make_test_scan()
    gate = ZoneGate(
        zone=2, depth_mm=(2.0, 6.0), alarm_level=0.4, inspectable=False, calib_id="cal_bad"
    )
    alarms = zone_alarms(scan, [gate])
    assert len(alarms) == 1
    alarm = alarms[0]
    expected = ("zone_not_inspectable", 2, "cal_bad", -1, -1, 6 * 12)
    actual = (
        alarm.kind,
        alarm.zone,
        alarm.calib_id,
        alarm.axial_idx,
        alarm.angle_idx,
        alarm.n_ascans,
    )
    assert actual == expected
    nans_ok = all(
        np.isnan(
            [alarm.axial_mm, alarm.angle_deg, alarm.peak, alarm.pct_of_alarm, alarm.axial_extent_mm]
        )
    )
    assert (alarm.depth_mm, nans_ok) == (pytest.approx(4.0), True)


def test_cluster_wrapping_angle() -> None:
    """Check that clusters spanning angle boundary merge."""
    scan = _make_test_scan()
    scan.ascans[2, 0, 20] = 0.8
    scan.ascans[2, 11, 20] = 0.75
    gate = ZoneGate(
        zone=1, depth_mm=(0.5, 3.0), alarm_level=0.5, inspectable=True, calib_id="cal_wrap"
    )
    alarms = zone_alarms(scan, [gate])
    assert len(alarms) == 1
    alarm = alarms[0]
    assert (alarm.n_ascans, alarm.axial_idx, alarm.angle_idx) == (2, 2, 0)
    assert alarm.peak == pytest.approx(0.8, rel=1e-3)


def test_missing_ascans_never_alarm() -> None:
    """Check that missing A-scans do not trigger alarms."""
    scan = _make_test_scan()
    scan.ascans[2, 3, 20] = 0.9
    scan.missing[2, 3] = True
    gate = ZoneGate(
        zone=1, depth_mm=(0.5, 3.0), alarm_level=0.1, inspectable=True, calib_id="cal_m"
    )
    alarms = zone_alarms(scan, [gate])
    uninspect_gate = ZoneGate(
        zone=1, depth_mm=(1.0, 5.0), alarm_level=0.1, inspectable=False, calib_id="cal_m"
    )
    uninspect_alarms = zone_alarms(scan, [uninspect_gate])
    assert (len(alarms), len(uninspect_alarms)) == (0, 1)
    assert uninspect_alarms[0].n_ascans == (6 * 12) - 1


def test_to_indications_skips_not_inspectable() -> None:
    """Check conversion to indications and reasons tuple."""
    alarms = [_sample_ok_alarm(), _sample_skip_alarm()]
    inds = to_indications(alarms, start_id=42)
    assert len(inds) == 1
    ind = inds[0]
    expected_tuple = (42, 2.0, 120.0, 5.5, 120.0, 2.0, 3, 2, 4, "reportable")
    actual_tuple = (
        ind.id,
        ind.axial_mm,
        ind.angle_deg,
        ind.depth_mm,
        ind.amplitude_pct,
        ind.axial_length_mm,
        ind.n_ascans,
        ind.axial_idx,
        ind.angle_idx,
        ind.category,
    )
    assert actual_tuple == expected_tuple
    assert all(np.isnan([ind.snr_db, ind.circ_length_mm]))
    assert ind.reasons == ("zone 1 alarm", "calibration block_a")


def test_alarms_sorting_order() -> None:
    """Check alarm ordering by zone then descending peak."""
    scan = _make_test_scan()
    scan.ascans[1, 1, 20] = 0.6
    scan.ascans[3, 3, 20] = 0.9
    gate1 = ZoneGate(zone=1, depth_mm=(0.5, 3.0), alarm_level=0.5, inspectable=True, calib_id="c1")
    gate2 = ZoneGate(zone=2, depth_mm=(0.5, 3.0), alarm_level=0.5, inspectable=False, calib_id="c2")
    alarms = zone_alarms(scan, [gate2, gate1])
    peaks = [a.peak for a in alarms[:2]]
    assert ([a.zone for a in alarms], peaks) == (
        [1, 1, 2],
        [pytest.approx(0.9, rel=1e-3), pytest.approx(0.6, rel=1e-3)],
    )
