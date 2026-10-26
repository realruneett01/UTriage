"""Offline HTML inspection report generation (project plan 6.2 item 6).

Advisory only: UTriage never accepts or passes a part. Every decision
belongs to the certified inspector.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

from utriage.rules.pipeline import Evaluation
from utriage.rules.types import QualityFault, QualityReport
from utriage.triage.fusion import TriageItem


@dataclass(frozen=True)
class Decision:
    """Inspector decision and note for a single triage item."""

    item_index: int
    decision: str = ""
    comment: str = ""
    inspector: str = ""


@dataclass(frozen=True)
class ReportInput:
    """Input payload for deterministic offline report rendering."""

    bar_id: str
    evaluation: Evaluation
    items: list[TriageItem]
    decisions: list[Decision]
    synthetic: bool
    generated_at: str


@dataclass(frozen=True)
class _RowMetrics:
    axial: str
    angle: str
    depth: str
    amp: str
    snr: str
    reasons: str


def _render_styles() -> str:
    """Return inline CSS styles including print layout without external links."""
    return """<style>
body {
  font-family: sans-serif;
  margin: 24px;
  color: #111;
  line-height: 1.4;
}
.banner.advisory {
  background: #e0f2fe;
  border: 1px solid #0284c7;
  color: #0369a1;
  padding: 10px;
  font-weight: bold;
  text-align: center;
  margin-bottom: 16px;
  border-radius: 4px;
}
.badge.synthetic {
  display: inline-block;
  background: #fef3c7;
  border: 1px solid #d97706;
  color: #92400e;
  padding: 4px 8px;
  font-weight: bold;
  margin-bottom: 12px;
  border-radius: 4px;
}
.warning.placeholder {
  background: #fee2e2;
  border: 1px solid #dc2626;
  color: #991b1b;
  padding: 10px;
  font-weight: bold;
  margin-bottom: 16px;
  border-radius: 4px;
}
.rescan-warning {
  color: #dc2626;
  font-weight: bold;
}
.metadata {
  margin-bottom: 16px;
  font-size: 14px;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 24px;
  font-size: 13px;
}
th, td {
  border: 1px solid #cbd5e1;
  padding: 6px 8px;
  text-align: left;
}
th {
  background: #f1f5f9;
  font-weight: 600;
}
tr:nth-child(even) {
  background: #f8fafc;
}
.signature-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-top: 16px;
}
@media print {
  body {
    margin: 0;
    font-size: 10pt;
  }
  .banner.advisory {
    border: 1px solid #000;
    background: none;
    color: #000;
  }
  table {
    page-break-inside: auto;
  }
  tr {
    page-break-inside: avoid;
  }
  @page {
    margin: 1.5cm;
  }
}
</style>"""


def _render_header(inp: ReportInput) -> str:
    """Render top banners, metadata, and placeholder limits warning."""
    lines = ['<div class="banner advisory">Advisory tool - decisions by certified inspector</div>']
    if inp.synthetic:
        lines.append('<div class="badge synthetic">Synthetic data</div>')
    if inp.evaluation.placeholder_limits:
        lines.append(
            '<div class="warning placeholder">'
            "WARNING: Acceptance limits are PLACEHOLDERS (UNVERIFIED)"
            "</div>"
        )
    lines.append('<div class="metadata">')
    lines.append(f"<p><strong>Bar ID:</strong> {html.escape(inp.bar_id)}</p>")
    lines.append(f"<p><strong>Generated at:</strong> {html.escape(inp.generated_at)}</p>")
    lines.append("</div>")
    return "\n".join(lines)


def _render_fault_row(fault: QualityFault) -> str:
    """Render a single data quality fault table row."""
    kind = html.escape(fault.kind)
    axial = f"{fault.axial_mm[0]:.1f} - {fault.axial_mm[1]:.1f} mm"
    angle = f"{fault.angle_deg[0]:.1f} - {fault.angle_deg[1]:.1f} deg"
    return f"<tr><td>{kind}</td><td>{axial}</td><td>{angle}</td><td>{fault.n_ascans}</td></tr>"


def _render_quality_section(quality: QualityReport) -> str:
    """Render the data quality watchdog section."""
    lines = ["<section><h2>Data Quality</h2>"]
    if quality.rescan_recommended:
        lines.append('<p class="rescan-warning"><strong>Re-scan recommended</strong></p>')
    if not quality.faults:
        lines.append("<p>No data-quality faults detected.</p></section>")
        return "\n".join(lines)
    lines.append("<table><thead><tr><th>Fault Kind</th><th>Axial Range</th>")
    lines.append("<th>Angle Range</th><th>A-scans</th></tr></thead><tbody>")
    lines.extend(_render_fault_row(f) for f in quality.faults)
    lines.append("</tbody></table></section>")
    return "\n".join(lines)


def _render_decision_cell(dec: Decision | None) -> str:
    """Render inspector decision content for a table cell."""
    if dec is None:
        return "<td></td>"
    parts: list[str] = []
    if dec.decision:
        parts.append(f"<strong>{html.escape(dec.decision)}</strong>")
    if dec.comment:
        parts.append(f"<span>{html.escape(dec.comment)}</span>")
    if dec.inspector:
        parts.append(f"<small>{html.escape(dec.inspector)}</small>")
    content = "<br>".join(parts)
    return f"<td>{content}</td>"


def _extract_metrics(item: TriageItem) -> _RowMetrics:
    """Extract spatial and amplitude metrics from an indication or ML finding."""
    if item.indication is not None:
        ind = item.indication
        reasons = ", ".join(ind.reasons) if ind.reasons else "-"
        return _RowMetrics(
            axial=f"{ind.axial_mm:.1f}",
            angle=f"{ind.angle_deg:.1f}",
            depth=f"{ind.depth_mm:.1f}",
            amp=f"{ind.amplitude_pct:.1f}%",
            snr=f"{ind.snr_db:.1f}",
            reasons=reasons,
        )
    if item.ml is not None:
        ml = item.ml
        reasons = ", ".join(ml.evidence) if ml.evidence else "-"
        return _RowMetrics(
            axial=f"{ml.axial_idx} (idx)",
            angle=f"{ml.angle_idx} (idx)",
            depth=f"{ml.depth_mm:.1f}",
            amp="-",
            snr="-",
            reasons=reasons,
        )
    return _RowMetrics("-", "-", "-", "-", "-", "-")


def _format_ml_metrics(item: TriageItem) -> tuple[str, str]:
    """Format ML score and uncertainty strings."""
    if item.ml is None:
        return "-", "-"
    return f"{item.ml.score:.2f}", f"{item.ml.uncertainty:.2f}"


def _render_triage_row(item: TriageItem, dec: Decision | None) -> str:
    """Render a single triage table row."""
    m = _extract_metrics(item)
    ml_score, uncertainty = _format_ml_metrics(item)
    review = ", ".join(item.review) if item.review else "-"
    decision_cell = _render_decision_cell(dec)
    return (
        f"<tr>"
        f"<td>{html.escape(item.source)}</td>"
        f"<td>{html.escape(item.category)}</td>"
        f"<td>{html.escape(m.axial)}</td>"
        f"<td>{html.escape(m.angle)}</td>"
        f"<td>{html.escape(m.depth)}</td>"
        f"<td>{html.escape(m.amp)}</td>"
        f"<td>{html.escape(m.snr)}</td>"
        f"<td>{html.escape(ml_score)}</td>"
        f"<td>{html.escape(uncertainty)}</td>"
        f"<td>{html.escape(review)}</td>"
        f"<td>{html.escape(m.reasons)}</td>"
        f"{decision_cell}"
        f"</tr>"
    )


def _render_triage_table(items: list[TriageItem], decisions: list[Decision]) -> str:
    """Render the triage indications table in given rank order."""
    dec_map = {d.item_index: d for d in decisions}
    lines = [
        "<section><h2>Triage Indications</h2>",
        "<table><thead><tr>",
        "<th>Source</th><th>Category</th><th>Axial</th><th>Angle</th>",
        "<th>Depth</th><th>Amp (%)</th><th>SNR (dB)</th><th>ML Score</th>",
        "<th>Uncertainty</th><th>Review Flags</th><th>Reasons</th>",
        "<th>Inspector Decision</th>",
        "</tr></thead><tbody>",
    ]
    for idx, item in enumerate(items):
        lines.append(_render_triage_row(item, dec_map.get(idx)))
    lines.append("</tbody></table></section>")
    return "\n".join(lines)


def _render_signature_block() -> str:
    """Render blank signature block for certified inspector."""
    return (
        "<section><h2>Inspector Sign-Off</h2>"
        '<div class="signature-grid">'
        "<div><strong>Inspector name:</strong> ______________________</div>"
        "<div><strong>Level:</strong> ______________________</div>"
        "<div><strong>Signature:</strong> ______________________</div>"
        "<div><strong>Date:</strong> ______________________</div>"
        "</div></section>"
    )


def render_report(inp: ReportInput) -> str:
    """Render full offline HTML inspection report."""
    styles = _render_styles()
    header = _render_header(inp)
    quality = _render_quality_section(inp.evaluation.quality)
    triage = _render_triage_table(inp.items, inp.decisions)
    sign = _render_signature_block()
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>UTriage Inspection Report - {html.escape(inp.bar_id)}</title>\n"
        f"{styles}\n"
        "</head>\n"
        "<body>\n"
        f"{header}\n"
        f"{quality}\n"
        f"{triage}\n"
        f"{sign}\n"
        "</body>\n"
        "</html>"
    )


def write_report(inp: ReportInput, path: Path) -> Path:
    """Write rendered report to path as UTF-8 and return path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_report(inp), encoding="utf-8")
    return target
