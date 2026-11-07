"""Tests for titanium inspection report generation. Synthetic data."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from utriage.physics.simulator import (  # noqa: E402
    BarScan,
    BarSpec,
    load_material,
    load_probe,
    simulate_bar,
)
from utriage.physics.ti_calibration import (  # noqa: E402
    CalibrationRecord,
    ZoneCalibration,
)
from utriage.report.ti_report import (  # noqa: E402
    TiReportInput,
    crop_png_b64,
    render_ti_report,
    write_ti_report,
)
from utriage.rules.types import Indication  # noqa: E402
from utriage.rules.zones import ZoneAlarm  # noqa: E402
from utriage.triage.fusion import MlFinding, TriageItem  # noqa: E402


def _build_tiny_scan() -> BarScan:
    """Build a fast simulated scan with length_mm=16 and n_angle=24."""
    mat = load_material("ti64")
    probe = load_probe("p5")
    water = load_material("water")
    spec = BarSpec(
        material=mat,
        probe=probe,
        water=water,
        length_mm=16.0,
        diameter_mm=40.0,
        axial_step_mm=2.0,
        n_angle=24,
        seed=10,
    )
    return simulate_bar(spec)


def _make_test_zone(
    zone: int,
    depth: tuple[float, float],
    alarm: float,
    reason: str = "",
) -> ZoneCalibration:
    """Create a calibrated test zone instance."""
    insp = reason == ""
    return ZoneCalibration(
        zone=zone,
        depth_mm=depth,
        ref_depth_mm=(depth[0] + depth[1]) / 2.0,
        signal_a=0.85 if insp else 0.40,
        noise_mean_b=0.15 if insp else 0.10,
        noise_peak_c=0.35 if insp else 0.32,
        snr=3.5 if insp else 1.36,
        alarm_level=alarm,
        cap_bound=not insp,
        inspectable=insp,
        reason=reason,
    )


def _build_test_record() -> CalibrationRecord:
    """Build a calibration record with one inspectable and one uninspectable zone."""
    z0 = _make_test_zone(0, (2.0, 12.0), 0.385, reason="")
    z1 = _make_test_zone(1, (12.0, 22.0), 0.28, reason="alarm cap is not above peak noise")
    return CalibrationRecord(
        calib_id="calib_test_9876",
        zones=(z0, z1),
        ref_fbh_mm=0.8,
        block_seed=1000,
        config_sha256="0123456789abcdef0123456789abcdef",
        label="Synthetic data: test calibration block",
    )


def _build_test_alarms(calib_id: str) -> tuple[ZoneAlarm, ...]:
    """Build test zone alarms for a calibrated record."""
    a0 = ZoneAlarm("alarm", 0, calib_id, 2, 4, 4.0, 60.0, 6.5, 0.45, 116.9, 3, 2.0)
    nan = float("nan")
    a1 = ZoneAlarm("zone_not_inspectable", 1, calib_id, -1, -1, nan, nan, 17.0, nan, nan, 192, nan)
    return (a0, a1)


def _build_test_items(calib_id: str) -> tuple[TriageItem, ...]:
    """Build test triage items for rules and ML sources."""
    ind0 = Indication(
        id=1,
        axial_mm=4.0,
        angle_deg=60.0,
        depth_mm=6.5,
        amplitude_pct=116.9,
        snr_db=float("nan"),
        axial_length_mm=2.0,
        circ_length_mm=float("nan"),
        n_ascans=3,
        axial_idx=2,
        angle_idx=4,
        category="reportable",
        reasons=("zone 0 alarm", f"calibration {calib_id}"),
    )
    item0 = TriageItem("rules", "reportable", ind0, None, ("zone_alarm",))
    ml1 = MlFinding(3, 8, 8.0, 0.75, 0.12, evidence=("candidate peak",))
    item1 = TriageItem("ml", "ml_flag", None, ml1, ("advisory_only",))
    return (item0, item1)


def _build_test_data() -> tuple[
    BarScan, CalibrationRecord, tuple[ZoneAlarm, ...], tuple[TriageItem, ...]
]:
    """Build test fixtures for scan, calibration, alarms, and triage items."""
    scan = _build_tiny_scan()
    record = _build_test_record()
    alarms = _build_test_alarms(record.calib_id)
    items = _build_test_items(record.calib_id)
    return scan, record, alarms, items


@pytest.fixture
def base_input() -> TiReportInput:
    """Fixture providing a standard test input."""
    scan, record, alarms, items = _build_test_data()
    return TiReportInput(
        bar_id="BAR-TEST-001",
        scan=scan,
        record=record,
        alarms=alarms,
        items=items,
        synthetic=True,
        placeholder_limits=True,
        generated_at="2026-11-07T12:00:00Z",
    )


def test_crop_png_b64_output() -> None:
    """Check that crop_png_b64 returns valid base64 PNG data starting with PNG magic bytes."""
    cscan = np.linspace(0.0, 1.0, 200).reshape((10, 20))
    b64_data = crop_png_b64(cscan, idx=(4, 0), half=3, alarm_level=0.5)
    raw_bytes = base64.b64decode(b64_data)
    assert raw_bytes.startswith(b"\x89PNG")


@pytest.mark.parametrize(
    "expected_heading",
    [
        "Metadata",
        "Zone Calibration",
        "Inspectability Warnings",
        "Flag List",
        "Inspector Sign-Off",
    ],
)
def test_section_headings_present(base_input: TiReportInput, expected_heading: str) -> None:
    """Check that each required section heading is present in the rendered report."""
    output = render_ti_report(base_input)
    assert expected_heading in output


@pytest.mark.parametrize(
    "expected_text",
    [
        "Advisory tool - decisions by certified inspector",
        "Synthetic data",
        "Calibration and limits are PLACEHOLDERS (UNVERIFIED - Ruri to check)",
        "Titanium-mode values in configs/ti_mode.yaml are placeholders",
        "calib_test_9876",
        "0123456789abcdef",
        "data:image/png;base64,",
        (
            "Zone 1 (12.0-22.0 mm) cannot be inspected at this sensitivity: "
            "alarm cap is not above peak noise. Inspector action required."
        ),
        "UTriage never accepts or passes a part.",
        "Inspector decision",
    ],
)
def test_report_content_elements(base_input: TiReportInput, expected_text: str) -> None:
    """Check required text elements and snippets in rendered report."""
    output = render_ti_report(base_input)
    assert expected_text in output


def test_no_http_or_script_in_output(base_input: TiReportInput) -> None:
    """Check that the output has no script tags and no http strings for security."""
    output = render_ti_report(base_input)
    assert "http" not in output.lower()
    assert "<script" not in output.lower()


def test_bar_id_html_escaping(base_input: TiReportInput) -> None:
    """Check HTML escaping on bar_id containing markup."""
    inp = TiReportInput(
        bar_id="BAR-<b>-001",
        scan=base_input.scan,
        record=base_input.record,
        alarms=base_input.alarms,
        items=base_input.items,
        synthetic=base_input.synthetic,
        placeholder_limits=base_input.placeholder_limits,
        generated_at=base_input.generated_at,
    )
    output = render_ti_report(inp)
    assert "BAR-&lt;b&gt;-001" in output
    assert "BAR-<b>-001" not in output


def test_no_prefilled_inspector_decisions(base_input: TiReportInput) -> None:
    """Check that inspector decision cells remain blank."""
    output = render_ti_report(base_input)
    assert "<td>Accept</td>" not in output
    assert "<td>Reject</td>" not in output
    assert "<td></td>" in output


def test_write_ti_report(base_input: TiReportInput, tmp_path: Path) -> None:
    """Check writing report to a file and assessing file content."""
    target = tmp_path / "subdir" / "ti_report.html"
    res = write_ti_report(base_input, target)
    assert res == target
    assert target.is_file()
    content = target.read_text(encoding="utf-8")
    assert base_input.record.calib_id in content


def test_crop_cap_at_thirty(base_input: TiReportInput) -> None:
    """Check that C-scan crops are capped at 30 flags with a notice."""
    ind = base_input.items[0].indication
    assert ind is not None
    many_items = tuple(
        TriageItem(
            source="rules",
            category="reportable",
            indication=ind,
            ml=None,
            review=(),
        )
        for _ in range(35)
    )
    inp = TiReportInput(
        bar_id="BAR-MANY-FLAGS",
        scan=base_input.scan,
        record=base_input.record,
        alarms=base_input.alarms,
        items=many_items,
        synthetic=True,
        placeholder_limits=False,
        generated_at=base_input.generated_at,
    )
    output = render_ti_report(inp)
    assert output.count("data:image/png;base64,") == 30
    assert "5 flags not imaged" in output
