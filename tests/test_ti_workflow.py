"""Titanium mode inspection workflow tests. Synthetic data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure local src/ is imported in worktrees
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from utriage.physics.simulator import BarScan, BarSpec, Reflector, load_material, load_probe
from utriage.physics.ti_calibration import calibrate_zones
from utriage.physics.ti_noise import simulate_ti_bar
from utriage.ti_workflow import (
    TiConfigs,
    TiInspection,
    gates_from_record,
    inspect_ti,
    inspection_to_dict,
    load_ti_configs,
)
from utriage.triage.fusion import MlFinding, rules_flags_preserved

ADVISORY_TEXT = "UTriage never accepts or passes a part; the certified inspector decides."

finding_strategy = st.builds(
    MlFinding,
    axial_idx=st.integers(min_value=0, max_value=23),
    angle_idx=st.integers(min_value=0, max_value=35),
    depth_mm=st.floats(min_value=0.0, max_value=36.0, allow_nan=False),
    score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    uncertainty=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    grain_noise_prob=st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    ),
    evidence=st.lists(
        st.sampled_from(["noise_peak", "echo_peak"]),
        max_size=2,
    ).map(tuple),
)


@pytest.fixture(scope="module")
def ti_setup() -> tuple[BarScan, TiConfigs]:
    """Module-level small titanium scan fixture. Synthetic data."""
    mat = load_material("ti64")
    prb = load_probe("p5")
    wtr = load_material("water")
    refl = Reflector(
        axial_mm=12.0,
        angle_deg=0.0,
        depth_mm=17.0,
        diameter_mm=0.8,
        kind="fbh",
    )
    spec = BarSpec(
        material=mat,
        probe=prb,
        water=wtr,
        length_mm=24.0,
        n_angle=36,
        reflectors=(refl,),
        seed=42,
    )
    cfgs = load_ti_configs()
    scan = simulate_ti_bar(spec, cfgs.noise)
    return scan, cfgs


def test_gates_from_record(ti_setup: tuple[BarScan, TiConfigs]) -> None:
    """Build gates from calibration record and check mapping."""
    scan, cfgs = ti_setup
    rec = calibrate_zones(scan.spec, cfgs.noise, cfgs.cal)
    gates = gates_from_record(rec)
    assert len(gates) == len(rec.zones)
    for g, z in zip(gates, rec.zones, strict=True):
        assert g.zone == z.zone
        assert g.depth_mm == z.depth_mm
        assert g.alarm_level == z.alarm_level
        assert g.inspectable == z.inspectable
        assert g.calib_id == rec.calib_id


def test_inspect_ti_calib_id(ti_setup: tuple[BarScan, TiConfigs]) -> None:
    """Check inspect_ti returns alarms linked with calibration id."""
    scan, cfgs = ti_setup
    insp = inspect_ti(scan, "bar_01", cfgs)
    assert insp.record.calib_id != ""
    assert len(insp.alarms) > 0
    assert all(a.calib_id == insp.record.calib_id for a in insp.alarms)
    assert insp.synthetic is True
    assert insp.bar_id == "bar_01"


@settings(max_examples=25, deadline=None, derandomize=True)
@given(findings=st.lists(finding_strategy, max_size=10))
def test_inspect_ti_property_safety(
    ti_setup: tuple[BarScan, TiConfigs],
    findings: list[MlFinding],
) -> None:
    """Property test: rules indications remain unchanged and count holds."""
    scan, cfgs = ti_setup
    insp = inspect_ti(scan, "bar_prop", cfgs, findings)
    assert len(insp.items) >= len(insp.rule_indications)
    assert rules_flags_preserved(list(insp.rule_indications), list(insp.items))
    for ind in insp.rule_indications:
        matching = [it for it in insp.items if it.indication == ind]
        assert len(matching) == 1
        assert matching[0].category == ind.category


def test_inspect_ti_monkeypatch_fuse_raises(
    ti_setup: tuple[BarScan, TiConfigs],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RuntimeError is raised when fuse drops a rules item."""
    scan, cfgs = ti_setup

    def bad_fuse(rule_inds: list, findings: list, cfg: object = None) -> list:
        return []

    monkeypatch.setattr("utriage.ti_workflow.fuse", bad_fuse)
    with pytest.raises(RuntimeError, match="ML removed or changed a rules flag"):
        inspect_ti(scan, "bar_bad", cfgs)


def test_inspection_to_dict_json_serializable(ti_setup: tuple[BarScan, TiConfigs]) -> None:
    """Dictionary is JSON serializable and holds required keys."""
    scan, cfgs = ti_setup
    insp = inspect_ti(scan, "bar_01", cfgs)
    d = inspection_to_dict(insp)
    serialized = json.dumps(d)
    assert isinstance(serialized, str)
    assert d["advisory"] == ADVISORY_TEXT
    assert d["bar_id"] == "bar_01"
    assert d["synthetic"] is True
    assert d["label"] == "Synthetic data"
    assert "calibration" in d
    assert "alarms" in d
    assert "items" in d
    assert d["placeholder_limits"] == insp.evaluation.placeholder_limits
    assert d["n_rules_indications"] == len(insp.rule_indications)
    assert d["n_zone_alarms"] + d["n_zones_not_inspectable"] == len(insp.alarms)


def test_inspection_to_dict_real(ti_setup: tuple[BarScan, TiConfigs]) -> None:
    """When synthetic is False, label is omitted and synthetic is False."""
    scan, cfgs = ti_setup
    insp = inspect_ti(scan, "bar_real", cfgs, synthetic=False)
    d = inspection_to_dict(insp)
    assert d["synthetic"] is False
    assert "label" not in d
    assert json.dumps(d)


@pytest.fixture(scope="module")
def clean_ti_setup() -> tuple[BarScan, TiConfigs, BarSpec]:
    """Module-level clean titanium scan fixture without reflectors. Synthetic data."""
    mat = load_material("ti64")
    prb = load_probe("p5")
    wtr = load_material("water")
    spec = BarSpec(
        material=mat,
        probe=prb,
        water=wtr,
        length_mm=24.0,
        n_angle=36,
        reflectors=(),
        seed=42,
    )
    cfgs = load_ti_configs()
    scan = simulate_ti_bar(spec, cfgs.noise)
    return scan, cfgs, spec


@pytest.fixture(scope="module")
def clean_ti_inspection(
    clean_ti_setup: tuple[BarScan, TiConfigs, BarSpec],
) -> tuple[TiInspection, TiConfigs, BarSpec]:
    """Module-level clean titanium inspection fixture. Synthetic data."""
    scan, cfgs, spec = clean_ti_setup
    insp = inspect_ti(scan, "bar_clean", cfgs)
    return insp, cfgs, spec


def test_inspect_ti_calib_id_matches_original_spec(
    clean_ti_inspection: tuple[TiInspection, TiConfigs, BarSpec],
) -> None:
    """Inspection calibration id matches calibration from original spec."""
    insp, cfgs, spec = clean_ti_inspection
    expected = calibrate_zones(spec, cfgs.noise, cfgs.cal)
    assert insp.record.calib_id == expected.calib_id


def test_calibration_zones_noise_levels(
    clean_ti_inspection: tuple[TiInspection, TiConfigs, BarSpec],
) -> None:
    """Every zone in record has peak noise strictly above mean noise."""
    insp, _, _ = clean_ti_inspection
    assert len(insp.record.zones) > 0
    assert all(z.noise_peak_c > z.noise_mean_b > 0.0 for z in insp.record.zones)


def test_clean_bar_alarm_count_low(
    clean_ti_inspection: tuple[TiInspection, TiConfigs, BarSpec],
) -> None:
    """Clean bar without reflectors produces very few zone alarms."""
    insp, _, _ = clean_ti_inspection
    alarm_count = sum(1 for a in insp.alarms if a.kind == "alarm")
    assert alarm_count <= 3
