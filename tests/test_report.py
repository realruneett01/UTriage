"""Tests for offline HTML inspection report generation (project plan 6.2 item 6)."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from utriage.report import Decision, ReportInput, render_report, write_report  # noqa: E402
from utriage.rules.pipeline import Evaluation  # noqa: E402
from utriage.rules.types import Indication, QualityFault, QualityReport  # noqa: E402
from utriage.triage.fusion import MlFinding, TriageItem  # noqa: E402


def _make_sample_indication(
    ind_id: int = 1,
    category: str = "rejectable",
    amplitude_pct: float = 85.0,
) -> Indication:
    return Indication(
        id=ind_id,
        axial_mm=15.0,
        angle_deg=45.0,
        depth_mm=10.0,
        amplitude_pct=amplitude_pct,
        snr_db=12.5,
        axial_length_mm=2.5,
        circ_length_mm=3.0,
        n_ascans=4,
        axial_idx=3,
        angle_idx=9,
        category=category,
        reasons=("amplitude above threshold",),
    )


def _make_report_input(
    *,
    synthetic: bool = False,
    placeholder_limits: bool = False,
    faults: list[QualityFault] | None = None,
    items: list[TriageItem] | None = None,
    decisions: list[Decision] | None = None,
) -> ReportInput:
    actual_faults = faults or []
    quality = QualityReport(
        faults=actual_faults,
        fault_mask=np.zeros((10, 10), dtype=bool),
    )
    evaluation = Evaluation(
        indications=[_make_sample_indication()],
        quality=quality,
        summary={"rejectable": 1},
        placeholder_limits=placeholder_limits,
        label="synthetic_bar_01",
    )
    actual_items = (
        items
        if items is not None
        else [
            TriageItem(
                source="rules",
                category="rejectable",
                indication=_make_sample_indication(),
                ml=None,
                review=("borderline_low_snr",),
            )
        ]
    )
    return ReportInput(
        bar_id="BAR-TEST-001",
        evaluation=evaluation,
        items=actual_items,
        decisions=decisions or [],
        synthetic=synthetic,
        generated_at="2026-10-27T00:00:00Z",
    )


def test_banner_always_present() -> None:
    inp = _make_report_input()
    html = render_report(inp)
    assert "Advisory tool - decisions by certified inspector" in html


@pytest.mark.parametrize("synthetic", [True, False])
def test_synthetic_data_label_present_iff_synthetic(synthetic: bool) -> None:
    inp = _make_report_input(synthetic=synthetic)
    html = render_report(inp)
    if synthetic:
        assert "Synthetic data" in html
    else:
        assert "Synthetic data" not in html


@pytest.mark.parametrize("placeholder_limits", [True, False])
def test_placeholder_warning_present_iff_placeholder_limits(
    placeholder_limits: bool,
) -> None:
    inp = _make_report_input(placeholder_limits=placeholder_limits)
    html = render_report(inp)
    if placeholder_limits:
        assert "PLACEHOLDER" in html
    else:
        assert "PLACEHOLDER" not in html


def test_html_escaping_of_comment_containing_script() -> None:
    attack_payload = "<script>alert('xss')</script> & \"injection\""
    decisions = [
        Decision(
            item_index=0,
            decision="rejected",
            comment=attack_payload,
            inspector="Inspector <1>",
        )
    ]
    inp = _make_report_input(decisions=decisions)
    html = render_report(inp)
    assert "<script>" not in html
    assert "</script>" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;/script&gt;" in html
    assert "Inspector &lt;1&gt;" in html


def test_no_http_or_https_in_output() -> None:
    inp = _make_report_input(
        faults=[
            QualityFault(
                kind="interface_drop",
                axial_mm=(10.0, 20.0),
                angle_deg=(0.0, 30.0),
                n_ascans=5,
            )
        ]
    )
    html = render_report(inp)
    assert "http://" not in html
    assert "https://" not in html


def test_decisions_appear_in_right_rows() -> None:
    ind1 = _make_sample_indication(ind_id=1, amplitude_pct=90.0)
    ind2 = _make_sample_indication(ind_id=2, amplitude_pct=70.0)
    finding = MlFinding(
        axial_idx=5,
        angle_idx=8,
        depth_mm=12.0,
        score=0.85,
        uncertainty=0.15,
        grain_noise_prob=0.1,
    )
    item0 = TriageItem(source="rules", category="rejectable", indication=ind1, ml=None)
    item1 = TriageItem(source="rules", category="reportable", indication=ind2, ml=None)
    item2 = TriageItem(source="ml", category="ml_flag", indication=None, ml=finding)
    decisions = [
        Decision(0, "rejected", "Gross flaw", "Inspector A"),
        Decision(2, "rescan", "Uncertain ML finding", "Inspector B"),
    ]
    inp = _make_report_input(items=[item0, item1, item2], decisions=decisions)
    html = render_report(inp)

    assert "Gross flaw" in html
    assert "Uncertain ML finding" in html
    tbody = html.split("<tbody>")[1].split("</tbody>")[0]
    rows = [r for r in tbody.split("<tr>") if r.strip()]
    assert len(rows) == 3
    assert "Gross flaw" in rows[0]
    assert "Gross flaw" not in rows[1]
    assert "Uncertain ML finding" not in rows[1]
    assert "Uncertain ML finding" in rows[2]


def test_deterministic_output_for_same_input() -> None:
    inp = _make_report_input(synthetic=True, placeholder_limits=True)
    html1 = render_report(inp)
    html2 = render_report(inp)
    assert html1 == html2


def test_data_quality_section_faults_and_rescan() -> None:
    fault = QualityFault(
        kind="interface_missing",
        axial_mm=(5.0, 15.0),
        angle_deg=(10.0, 40.0),
        n_ascans=12,
    )
    inp_with_fault = _make_report_input(faults=[fault])
    html_fault = render_report(inp_with_fault)
    assert "interface_missing" in html_fault
    assert "Re-scan recommended" in html_fault

    inp_no_fault = _make_report_input(faults=[])
    html_clean = render_report(inp_no_fault)
    assert "Re-scan recommended" not in html_clean


def test_signature_block_present() -> None:
    inp = _make_report_input()
    html = render_report(inp)
    assert "Inspector name" in html
    assert "Level" in html
    assert "Signature" in html
    assert "Date" in html


def test_write_report(tmp_path: Path) -> None:
    inp = _make_report_input()
    out_file = tmp_path / "sub" / "report.html"
    returned_path = write_report(inp, out_file)
    assert returned_path == out_file
    assert out_file.is_file()
    assert out_file.read_text(encoding="utf-8") == render_report(inp)
