"""Tests for titanium zone evaluation experiment (Synthetic data)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from experiments.run_ti_zones import (
    DEFINITIONS,
    FlawTarget,
    RunConfig,
    alarm_matches_hole,
    is_alarm_hit,
    parse_args,
    run_ti_zones,
)

from utriage.physics.simulator import BarSpec, load_material, load_probe
from utriage.rules.zones import ZoneAlarm


@pytest.fixture
def tiny_part() -> BarSpec:
    """Tiny bar spec for fast test execution."""
    ti64 = load_material("ti64")
    water = load_material("water")
    probe = load_probe("p5")
    return BarSpec(
        material=ti64,
        probe=probe,
        water=water,
        length_mm=16.0,
        n_angle=24,
    )


@pytest.mark.parametrize(
    ("ax_idx", "an_idx", "expected"),
    [
        (0, 0, True),
        (2, 0, True),
        (3, 0, False),
        (0, 23, True),
        (0, 22, True),
        (0, 21, False),
        (0, 1, True),
        (2, 22, True),
        (3, 22, False),
    ],
)
def test_angle_wrap_matching(ax_idx: int, an_idx: int, expected: bool) -> None:
    """Test axial and angle wrap distance matching for flaw detection."""
    alarm = ZoneAlarm(
        kind="alarm",
        zone=0,
        calib_id="c0",
        axial_idx=ax_idx,
        angle_idx=an_idx,
        axial_mm=float(ax_idx),
        angle_deg=float(an_idx * 15),
        depth_mm=5.0,
        peak=0.5,
        pct_of_alarm=100.0,
        n_ascans=1,
        axial_extent_mm=1.0,
    )
    res = alarm_matches_hole(alarm, hole_ax=0, hole_an=0, n_angle=24)
    assert res == expected


@pytest.mark.parametrize(
    ("arm_name", "alarm_zone", "target_zone", "expected"),
    [
        ("zone", 0, 0, True),
        ("zone", 1, 0, False),
        ("global", 0, 0, True),
        ("global", 1, 0, True),
    ],
)
def test_arm_zone_matching(
    arm_name: str,
    alarm_zone: int,
    target_zone: int,
    expected: bool,
) -> None:
    """Test zone arm checks zone identity while global arm accepts any zone."""
    alarm = ZoneAlarm(
        kind="alarm",
        zone=alarm_zone,
        calib_id="c0",
        axial_idx=0,
        angle_idx=0,
        axial_mm=0.0,
        angle_deg=0.0,
        depth_mm=5.0,
        peak=0.5,
        pct_of_alarm=100.0,
        n_ascans=1,
        axial_extent_mm=1.0,
    )
    target = FlawTarget(axial_idx=0, angle_idx=0, depth_mm=5.0, zone=target_zone)
    assert is_alarm_hit(alarm, target, n_angle=24, arm_name=arm_name) == expected


@pytest.mark.parametrize(
    ("args", "expect_error"),
    [
        (["--quick"], True),
        (["--quick", "--seed", "1"], True),
        (["--n-clean", "5"], False),
    ],
)
def test_quick_without_out_errors(args: list[str], expect_error: bool) -> None:
    """Check that --quick without explicit --out raises ValueError."""
    if expect_error:
        with pytest.raises(ValueError, match="--quick requires explicit --out"):
            parse_args(args)
    else:
        cfg = parse_args(args)
        assert cfg.n_clean == 5


def test_definitions_present() -> None:
    """Check required conceptual definitions are present in DEFINITIONS."""
    required = {
        "global_arm",
        "zone_arm",
        "structured_noise",
        "flat_noise",
        "detection",
        "clean_evaluation",
    }
    assert required.issubset(DEFINITIONS.keys())
    assert "min zone A" in DEFINITIONS["global_arm"]
    assert "max zone C" in DEFINITIONS["global_arm"]


def test_payload_keys_and_structure(tmp_path: Path, tiny_part: BarSpec) -> None:
    """Check payload contains all expected keys and metadata."""
    out_file = tmp_path / "ti_zone_metrics.json"
    cfg = RunConfig(n_clean=2, n_flaw=2, seed=42, out_path=out_file, quick=True)
    payload = run_ti_zones(cfg, part=tiny_part)
    expected_keys = {
        "label",
        "config_sha256",
        "definitions",
        "counts",
        "seeds",
        "runtime_s",
        "noise_models",
        "calibration",
    }
    assert expected_keys.issubset(payload.keys())
    assert payload["label"] == (
        "Synthetic data (titanium-mode simulator; configs/ti_mode.yaml placeholders, UNVERIFIED)"
    )
    assert len(payload["config_sha256"]) == 64
    assert out_file.exists()


def _check_ci_bracket(record: dict[str, Any]) -> None:
    """Helper to check rate is bracketed by confidence interval."""
    rate = record["rate"]
    low = record["ci_low"]
    high = record["ci_high"]
    assert low <= rate + 1e-9
    assert high >= rate - 1e-9


def _check_arm_ci_brackets(arm_data: dict[str, Any]) -> None:
    """Check confidence interval bounds for clean and flawed records in an arm."""
    _check_ci_bracket(arm_data["clean"]["bars_with_alarm"])
    for zone_flaw in arm_data["flawed"]["per_zone"]:
        _check_ci_bracket(zone_flaw)


def test_clopper_pearson_intervals_bracket_rate(
    tmp_path: Path,
    tiny_part: BarSpec,
) -> None:
    """Check Clopper-Pearson confidence intervals bracket empirical rates."""
    out_file = tmp_path / "metrics.json"
    cfg = RunConfig(n_clean=2, n_flaw=2, seed=123, out_path=out_file, quick=True)
    payload = run_ti_zones(cfg, part=tiny_part)
    all_arms = [arm for m in payload["noise_models"].values() for arm in m["arms"].values()]
    for arm in all_arms:
        _check_arm_ci_brackets(arm)


def test_main_cli_with_monkeypatch(
    tmp_path: Path,
    tiny_part: BarSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check main CLI execution with monkeypatched tiny part spec."""
    import experiments.run_ti_zones as rtz

    monkeypatch.setattr(rtz, "default_part_spec", lambda: tiny_part)
    out_file = tmp_path / "cli_metrics.json"
    payload = rtz.main(["--quick", "--out", str(out_file), "--seed", "7"])
    assert payload["counts"]["n_clean"] == 2
    assert payload["counts"]["n_flaw"] == 2
    assert out_file.exists()
