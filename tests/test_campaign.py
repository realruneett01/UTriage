"""Tests for triage campaign evaluation and review effort metrics."""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Ensure local src/ is imported in worktrees
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from utriage.physics.simulator import (
    BarScan,
    BarSpec,
    Reflector,
    load_material,
    load_probe,
)
from utriage.rules.scoring import MatchTol
from utriage.rules.types import Indication
from utriage.triage.campaign import item_is_real, random_order_effort, review_effort
from utriage.triage.fusion import MlFinding, TriageItem


def _make_dummy_spec() -> BarSpec:
    """Create minimal BarSpec with one planted reflector."""
    mat = load_material("ti64")
    probe = load_probe("p5")
    water = load_material("water")
    ref = Reflector(axial_mm=10.0, angle_deg=30.0, depth_mm=5.0, diameter_mm=0.8)
    return BarSpec(
        material=mat,
        probe=probe,
        water=water,
        diameter_mm=40.0,
        length_mm=32.0,
        axial_step_mm=1.0,
        n_angle=60,
        reflectors=(ref,),
    )


def _make_dummy_scan(spec: BarSpec) -> BarScan:
    """Create minimal BarScan with coordinate axes matching spec."""
    n_ax = int(spec.length_mm / spec.axial_step_mm)
    axial_mm = np.linspace(0.0, spec.length_mm, n_ax, endpoint=False)
    angle_deg = np.linspace(0.0, 360.0, spec.n_angle, endpoint=False)
    ascans = np.zeros((n_ax, spec.n_angle, 64), dtype=np.float32)
    t_if = np.zeros((n_ax, spec.n_angle), dtype=np.float32)
    return BarScan(
        ascans=ascans,
        t0_us=0.0,
        fs_mhz=50.0,
        axial_mm=axial_mm,
        angle_deg=angle_deg,
        t_if_us=t_if,
        spec=spec,
    )


def _make_rule_indication(axial_mm: float, angle_deg: float, depth_mm: float) -> Indication:
    """Create dummy Indication for testing."""
    return Indication(
        id=1,
        axial_mm=axial_mm,
        angle_deg=angle_deg,
        depth_mm=depth_mm,
        amplitude_pct=75.0,
        snr_db=12.0,
        axial_length_mm=1.0,
        circ_length_mm=1.0,
        n_ascans=1,
        axial_idx=int(axial_mm),
        angle_idx=int(angle_deg / 6.0),
        category="rejectable",
        reasons=("test",),
    )


@pytest.mark.parametrize(
    ("ordered_real", "n_real_total", "expected"),
    [
        ([True, True, True, False, False], 3, 3),
        ([False, False, True, True, True], 3, 5),
        ([True, False, True, False, True], 3, 5),
        ([True, True, False, False], 3, None),
        ([], 3, None),
        ([], 0, 0),
        ([False, False], 0, 0),
    ],
)
def test_review_effort_scenarios(
    ordered_real: list[bool],
    n_real_total: int,
    expected: int | None,
) -> None:
    """Verify review effort handles all-real-first, real-last, and missing flaws."""
    assert review_effort(ordered_real, n_real_total) == expected


def test_random_order_effort_bounds() -> None:
    """Verify random permutation effort lies between best and worst cases."""
    flags = [True, True, True, False, False]
    n_real = 3
    effort = random_order_effort(flags, n_real, n_perm=100, seed=42)
    assert 3.0 <= effort <= 5.0


def test_random_order_effort_missing() -> None:
    """Verify random effort is NaN when flaws are missing from the list."""
    flags = [True, False, False]
    assert math.isnan(random_order_effort(flags, 3, n_perm=50, seed=0))


def test_item_is_real_rules_matched_and_unmatched() -> None:
    """Verify rules item matching with spec reflector within tolerance."""
    spec = _make_dummy_spec()
    scan = _make_dummy_scan(spec)

    ind_match = _make_rule_indication(10.0, 30.0, 5.0)
    item_match = TriageItem(
        source="rules",
        category="rejectable",
        indication=ind_match,
        ml=None,
    )
    assert item_is_real(item_match, spec, scan) is True

    ind_noise = _make_rule_indication(25.0, 180.0, 20.0)
    item_noise = TriageItem(
        source="rules",
        category="rejectable",
        indication=ind_noise,
        ml=None,
    )
    assert item_is_real(item_noise, spec, scan) is False


def test_item_is_real_ml_matched_and_unmatched() -> None:
    """Verify ML item coordinate conversion via scan axes matches reflector."""
    spec = _make_dummy_spec()
    scan = _make_dummy_scan(spec)

    finding_match = MlFinding(
        axial_idx=10,
        angle_idx=5,
        depth_mm=5.0,
        score=0.92,
        uncertainty=0.05,
    )
    item_match = TriageItem(
        source="ml",
        category="ml_flag",
        indication=None,
        ml=finding_match,
    )
    assert item_is_real(item_match, spec, scan) is True

    finding_miss = MlFinding(
        axial_idx=28,
        angle_idx=40,
        depth_mm=18.0,
        score=0.88,
        uncertainty=0.08,
    )
    item_miss = TriageItem(
        source="ml",
        category="ml_flag",
        indication=None,
        ml=finding_miss,
    )
    assert item_is_real(item_miss, spec, scan) is False


def test_item_is_real_empty_item_and_custom_tol() -> None:
    """Verify empty item returns False and strict tolerance behaves correctly."""
    spec = _make_dummy_spec()
    scan = _make_dummy_scan(spec)
    empty_item = TriageItem(source="none", category="none", indication=None, ml=None)
    assert item_is_real(empty_item, spec, scan) is False

    ind = _make_rule_indication(11.5, 30.0, 5.0)
    item = TriageItem(source="rules", category="rejectable", indication=ind, ml=None)
    strict_tol = MatchTol(axial_mm=0.5, arc_mm=0.5, depth_mm=0.5)
    assert item_is_real(item, spec, scan, tol=strict_tol) is False
