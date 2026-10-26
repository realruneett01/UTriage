"""Tests for scoring rules engine detections and watchdog faults (project plan 2.4)."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        sys.modules["utriage"].__path__.insert(0, _pkg_dir)

import pytest  # noqa: E402

from utriage.physics.simulator import (  # noqa: E402
    Artefact,
    BarSpec,
    Material,
    Probe,
    Reflector,
)
from utriage.rules.scoring import (  # noqa: E402
    MatchTol,
    fault_overlaps,
    matches,
    score_bar,
)
from utriage.rules.types import Indication, QualityFault  # noqa: E402


def _dummy_spec(
    reflectors: tuple[Reflector, ...] = (),
    diameter_mm: float = 40.0,
) -> BarSpec:
    mat = Material(
        name="dummy",
        velocity_m_s=6000.0,
        density_kg_m3=4500.0,
        attenuation_db_mm_mhz=0.01,
        grain_noise=0.02,
    )
    probe = Probe(
        frequency_mhz=5.0,
        fractional_bandwidth=0.5,
        water_path_mm=25.0,
        focal_depth_mm=15.0,
        focal_zone_mm=10.0,
        beam_sigma_mm=1.5,
        sampling_mhz=50.0,
        gain_db=20.0,
        electronic_noise=0.01,
        k_fbh=1.0,
    )
    return BarSpec(
        material=mat,
        probe=probe,
        water=mat,
        diameter_mm=diameter_mm,
        length_mm=48.0,
        axial_step_mm=1.0,
        n_angle=72,
        reflectors=reflectors,
    )


def _make_ind(
    pos: tuple[float, float, float],
    category: str = "reportable",
    ind_id: int = 1,
) -> Indication:
    ax, ang, depth = pos
    return Indication(
        id=ind_id,
        axial_mm=ax,
        angle_deg=ang,
        depth_mm=depth,
        amplitude_pct=80.0,
        snr_db=12.0,
        axial_length_mm=1.0,
        circ_length_mm=1.0,
        n_ascans=1,
        axial_idx=int(ax),
        angle_idx=int(ang),
        category=category,
    )


@pytest.mark.parametrize(
    ("ind_pos", "ref_pos", "expected"),
    [
        ((10.0, 45.0, 10.0), (10.0, 45.0, 10.0), True),
        ((12.0, 45.0, 10.0), (10.0, 45.0, 10.0), True),
        ((12.5, 45.0, 10.0), (10.0, 45.0, 10.0), False),
        ((10.0, 45.0, 12.0), (10.0, 45.0, 10.0), True),
        ((10.0, 45.0, 12.5), (10.0, 45.0, 10.0), False),
        ((10.0, 359.0, 10.0), (10.0, 1.0, 10.0), True),
        ((10.0, 1.0, 10.0), (10.0, 359.0, 10.0), True),
        ((10.0, 0.0, 10.0), (10.0, 180.0, 10.0), False),
    ],
)
def test_matches_tolerance_and_angle_wrap(
    ind_pos: tuple[float, float, float],
    ref_pos: tuple[float, float, float],
    expected: bool,
) -> None:
    spec = _dummy_spec(diameter_mm=40.0)
    ind = _make_ind(ind_pos)
    ref = Reflector(
        axial_mm=ref_pos[0],
        angle_deg=ref_pos[1],
        depth_mm=ref_pos[2],
        diameter_mm=0.8,
    )
    assert matches(ind, ref, spec, MatchTol()) == expected


@pytest.mark.parametrize(
    ("cat", "is_match_expected", "is_fc_expected"),
    [
        ("reportable", True, False),
        ("rejectable", True, False),
        ("insignificant", False, False),
        ("data_quality_fault", False, False),
    ],
)
def test_score_bar_categories(
    cat: str,
    is_match_expected: bool,
    is_fc_expected: bool,
) -> None:
    ref = Reflector(axial_mm=10.0, angle_deg=20.0, depth_mm=8.0, diameter_mm=0.8)
    spec = _dummy_spec(reflectors=(ref,))
    matching_ind = _make_ind((10.0, 20.0, 8.0), category=cat, ind_id=1)
    score = score_bar([matching_ind], spec)
    assert score.hits == [is_match_expected]
    assert score.false_calls == (1 if is_fc_expected else 0)


def test_score_bar_multiple_indications_and_false_calls() -> None:
    ref1 = Reflector(axial_mm=8.0, angle_deg=0.0, depth_mm=10.0, diameter_mm=0.8)
    ref2 = Reflector(axial_mm=24.0, angle_deg=120.0, depth_mm=15.0, diameter_mm=1.0)
    spec = _dummy_spec(reflectors=(ref1, ref2), diameter_mm=40.0)
    ind1 = _make_ind((8.5, 1.0, 10.2), category="reportable", ind_id=1)
    ind2 = _make_ind((8.2, 359.5, 9.8), category="rejectable", ind_id=2)
    ind_fc = _make_ind((40.0, 240.0, 12.0), category="reportable", ind_id=3)

    score = score_bar([ind1, ind2, ind_fc], spec)
    assert score.hits == [True, False]
    assert score.false_calls == 1
    assert score.n_ascans == 48 * 72


@pytest.mark.parametrize(
    ("fault_bounds", "art_bounds", "expected"),
    [
        (((10.0, 20.0), (50.0, 100.0)), ((15.0, 25.0), (80.0, 120.0)), True),
        (((10.0, 20.0), (50.0, 100.0)), ((10.0, 20.0), (50.0, 100.0)), True),
        (((20.0, 25.0), (50.0, 100.0)), ((10.0, 20.0), (50.0, 100.0)), True),
        (((25.0, 30.0), (50.0, 100.0)), ((10.0, 20.0), (50.0, 100.0)), False),
        (((10.0, 20.0), (110.0, 150.0)), ((10.0, 20.0), (50.0, 100.0)), False),
    ],
)
def test_fault_overlaps(
    fault_bounds: tuple[tuple[float, float], tuple[float, float]],
    art_bounds: tuple[tuple[float, float], tuple[float, float]],
    expected: bool,
) -> None:
    fault = QualityFault(
        kind="interface_drop",
        axial_mm=fault_bounds[0],
        angle_deg=fault_bounds[1],
        n_ascans=10,
    )
    art = Artefact(
        kind="bubble",
        axial_mm=art_bounds[0],
        angle_deg=art_bounds[1],
        severity=1.0,
    )
    assert fault_overlaps(fault, art) == expected
