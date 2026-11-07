"""Titanium mode inspection report generation. Synthetic data."""

from __future__ import annotations

import base64
import dataclasses
import html
import io
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utriage.physics.simulator import BarScan, cscan_peak
from utriage.physics.ti_calibration import CalibrationRecord, ZoneCalibration
from utriage.report.html import _render_signature_block, _render_styles
from utriage.rules.zones import ZoneAlarm
from utriage.triage.fusion import TriageItem


@dataclass(frozen=True)
class TiReportInput:
    """Input payload for titanium mode inspection report."""

    bar_id: str
    scan: BarScan
    record: CalibrationRecord
    alarms: tuple[ZoneAlarm, ...]
    items: tuple[TriageItem, ...]
    synthetic: bool
    placeholder_limits: bool
    generated_at: str


@dataclass(frozen=True)
class _Coords:
    axial: str
    angle: str
    depth: str
    idx: tuple[int, int]


@dataclass(frozen=True)
class _FlagRow:
    index: int
    source: str
    category: str
    zone_str: str
    coords: _Coords
    amp_str: str
    snr_str: str
    reasons_str: str
    review_str: str
    crop_cell: str


@dataclass
class _ScanCropContext:
    scan: BarScan
    cache: dict[int, np.ndarray]
    count: int = 0


def crop_png_b64(
    cscan: np.ndarray,
    idx: tuple[int, int],
    half: int,
    alarm_level: float,
) -> str:
    """Render C-scan patch with angle wrap and contour at alarm level as base64 PNG."""
    n_ax, n_an = cscan.shape
    r_idx = np.clip(idx[0] + np.arange(-half, half + 1), 0, n_ax - 1)
    c_idx = (idx[1] + np.arange(-half, half + 1)) % n_an
    crop = cscan[np.ix_(r_idx, c_idx)]

    fig, ax = plt.subplots(figsize=(2.4, 2.4), dpi=100)
    try:
        ax.imshow(crop, cmap="viridis", origin="lower", aspect="auto")
        if np.nanmin(crop) < alarm_level < np.nanmax(crop):
            ax.contour(crop, levels=[alarm_level], colors="red", linewidths=1.5)
        ax.axis("off")
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    finally:
        plt.close(fig)


def _render_banners(inp: TiReportInput) -> str:
    """Render top advisory, placeholder, and synthetic banners."""
    lines = ['<div class="banner advisory">Advisory tool - decisions by certified inspector</div>']
    if inp.synthetic:
        lines.append('<div class="badge synthetic">Synthetic data</div>')
    if inp.placeholder_limits:
        lines.append(
            '<div class="warning placeholder">'
            "Calibration and limits are PLACEHOLDERS (UNVERIFIED - Ruri to check)"
            "</div>"
        )
    lines.append(
        '<p class="note">Titanium-mode values in configs/ti_mode.yaml are placeholders</p>'
    )
    return "\n".join(lines)


def _render_metadata(inp: TiReportInput) -> str:
    """Render inspection metadata section."""
    rec = inp.record
    lines = [
        "<section><h2>Metadata</h2>",
        '<div class="metadata">',
        f"<p><strong>Bar ID:</strong> {html.escape(inp.bar_id)}</p>",
        f"<p><strong>Generated at:</strong> {html.escape(inp.generated_at)}</p>",
        f"<p><strong>Calibration ID:</strong> {html.escape(rec.calib_id)}</p>",
        f"<p><strong>Config SHA-256:</strong> {html.escape(rec.config_sha256[:16])}</p>",
        f"<p><strong>Reference FBH diameter:</strong> {rec.ref_fbh_mm:.2f} mm</p>",
        f"<p><strong>Block seed:</strong> {rec.block_seed}</p>",
        f"<p><strong>Record label:</strong> {html.escape(rec.label)}</p>",
        "</div></section>",
    ]
    return "\n".join(lines)


def _cal_row(z: ZoneCalibration) -> str:
    """Format one row in zone calibration table."""
    snr_str = f"{z.snr:.2f}" if not math.isnan(z.snr) else "-"
    cap_str = "Yes" if z.cap_bound else "No"
    insp_str = "Yes" if z.inspectable else "No"
    reason = html.escape(z.reason) if z.reason else "-"
    depth = f"{z.depth_mm[0]:.1f} - {z.depth_mm[1]:.1f}"
    return (
        f"<tr>"
        f"<td>{z.zone}</td><td>{depth}</td>"
        f"<td>{z.signal_a:.3f}</td><td>{z.noise_mean_b:.3f}</td><td>{z.noise_peak_c:.3f}</td>"
        f"<td>{snr_str}</td><td>{z.alarm_level:.3f}</td>"
        f"<td>{cap_str}</td><td>{insp_str}</td><td>{reason}</td>"
        f"</tr>"
    )


def _render_calibration_section(rec: CalibrationRecord) -> str:
    """Render depth zone calibration table."""
    lines = [
        "<section><h2>Zone Calibration</h2>",
        "<table><thead><tr>",
        "<th>Zone</th><th>Depth Range (mm)</th><th>A</th><th>B</th><th>C</th>",
        "<th>SNR=(A-B)/(C-B)</th><th>Alarm Level</th><th>Cap Bound</th>",
        "<th>Inspectable</th><th>Reason</th>",
        "</tr></thead><tbody>",
    ]
    lines.extend(_cal_row(z) for z in rec.zones)
    lines.append("</tbody></table></section>")
    return "\n".join(lines)


def _render_inspectability_warnings(rec: CalibrationRecord) -> str:
    """Render warning boxes for depth zones that cannot be inspected."""
    uninspectable = [z for z in rec.zones if not z.inspectable]
    if not uninspectable:
        return ""
    lines = ["<section><h2>Inspectability Warnings</h2>"]
    for z in uninspectable:
        reason = html.escape(z.reason)
        box = (
            f'<div class="warning not-inspectable">'
            f"Zone {z.zone} ({z.depth_mm[0]:.1f}-{z.depth_mm[1]:.1f} mm) cannot be "
            f"inspected at this sensitivity: {reason}. Inspector action required."
            f"</div>"
        )
        lines.append(box)
    lines.append("</section>")
    return "\n".join(lines)


def _extract_zone(reasons: tuple[str, ...]) -> int | None:
    """Extract zone number from indication reasons matching 'zone k alarm'."""
    for r in reasons:
        match = re.match(r"^zone (\d+) alarm$", r)
        if match:
            return int(match.group(1))
    return None


def _extract_coords(item: TriageItem, scan: BarScan) -> _Coords:
    """Extract spatial coordinates and discrete grid index."""
    if item.indication is not None:
        ind = item.indication
        return _Coords(
            axial=f"{ind.axial_mm:.1f}",
            angle=f"{ind.angle_deg:.1f}",
            depth=f"{ind.depth_mm:.1f}",
            idx=(ind.axial_idx, ind.angle_idx),
        )
    if item.ml is not None:
        ml = item.ml
        ax = f"{float(scan.axial_mm[ml.axial_idx]):.1f}"
        ang = f"{float(scan.angle_deg[ml.angle_idx]):.1f}"
        return _Coords(
            axial=ax,
            angle=ang,
            depth=f"{ml.depth_mm:.1f}",
            idx=(ml.axial_idx, ml.angle_idx),
        )
    return _Coords(axial="-", angle="-", depth="-", idx=(0, 0))


def _extract_flag_peak(
    item: TriageItem,
    z_cal: ZoneCalibration,
    alarms: tuple[ZoneAlarm, ...],
) -> float | None:
    """Extract peak amplitude for an item in a calibrated zone."""
    if item.indication is not None:
        return float(item.indication.amplitude_pct * z_cal.alarm_level / 100.0)
    for a in alarms:
        if a.kind == "alarm" and a.zone == z_cal.zone:
            return float(a.peak)
    return None


def _calculate_flag_snr(
    item: TriageItem,
    z_cal: ZoneCalibration | None,
    alarms: tuple[ZoneAlarm, ...],
) -> str:
    """Calculate indication SNR = (peak - B) / (C - B) for zone alarms."""
    if z_cal is None:
        return "-"
    denom = z_cal.noise_peak_c - z_cal.noise_mean_b
    if denom <= 0.0:
        return "-"
    peak = _extract_flag_peak(item, z_cal, alarms)
    if peak is None:
        return "-"
    return f"{(peak - z_cal.noise_mean_b) / denom:.2f}"


def _build_crop_cell(
    cscan: np.ndarray,
    idx: tuple[int, int],
    alarm_level: float,
    count: int,
) -> str:
    """Render image cell or cap message for zone alarms."""
    if count > 30:
        return "<td>Not imaged (cap of 30)</td>"
    b64 = crop_png_b64(cscan, idx, half=8, alarm_level=alarm_level)
    return (
        f'<td><img src="data:image/png;base64,{b64}" '
        f'alt="C-scan crop" style="max-width:120px;height:auto;" /></td>'
    )


def _resolve_crop_cell(
    ctx: _ScanCropContext,
    z_cal: ZoneCalibration | None,
    idx: tuple[int, int],
) -> str:
    """Build crop table cell and update crop count."""
    if z_cal is None:
        return "<td>-</td>"
    ctx.count += 1
    if z_cal.zone not in ctx.cache:
        ctx.cache[z_cal.zone] = cscan_peak(ctx.scan, z_cal.depth_mm[0], z_cal.depth_mm[1])
    return _build_crop_cell(ctx.cache[z_cal.zone], idx, z_cal.alarm_level, ctx.count)


def _format_flag_html_row(row: _FlagRow) -> str:
    """Render a single HTML row for triage flags."""
    return (
        f"<tr>"
        f"<td>{row.index}</td>"
        f"<td>{html.escape(row.source)}</td>"
        f"<td>{html.escape(row.category)}</td>"
        f"<td>{html.escape(row.zone_str)}</td>"
        f"<td>{html.escape(row.coords.axial)}</td>"
        f"<td>{html.escape(row.coords.angle)}</td>"
        f"<td>{html.escape(row.coords.depth)}</td>"
        f"<td>{html.escape(row.amp_str)}</td>"
        f"<td>{html.escape(row.snr_str)}</td>"
        f"<td>{html.escape(row.reasons_str)}</td>"
        f"<td>{html.escape(row.review_str)}</td>"
        f"{row.crop_cell}"
        f"<td></td>"
        f"</tr>"
    )


def _build_row_data(
    item: TriageItem,
    ctx: _ScanCropContext,
    record: CalibrationRecord,
    alarms: tuple[ZoneAlarm, ...],
) -> _FlagRow:
    """Construct one row data structure for a triage item."""
    reasons = item.indication.reasons if item.indication else (item.ml.evidence if item.ml else ())
    zk = _extract_zone(reasons)
    coords = _extract_coords(item, ctx.scan)
    z_cal = next((z for z in record.zones if z.zone == zk), None)
    snr_str = _calculate_flag_snr(item, z_cal, alarms)
    crop_cell = _resolve_crop_cell(ctx, z_cal, coords.idx)
    amp_str = f"{item.indication.amplitude_pct:.1f}%" if item.indication else "-"
    return _FlagRow(
        index=0,
        source=item.source,
        category=item.category,
        zone_str=str(zk) if zk is not None else "-",
        coords=coords,
        amp_str=amp_str,
        snr_str=snr_str,
        reasons_str=", ".join(reasons) if reasons else "-",
        review_str=", ".join(item.review) if item.review else "-",
        crop_cell=crop_cell,
    )


def _render_flag_rows(inp: TiReportInput) -> tuple[list[str], int]:
    """Render all flag table rows and count total zone flags."""
    ctx = _ScanCropContext(scan=inp.scan, cache={})
    rows = []
    for idx, item in enumerate(inp.items, start=1):
        row = _build_row_data(item, ctx, inp.record, inp.alarms)
        rows.append(_format_flag_html_row(dataclasses.replace(row, index=idx)))
    return rows, ctx.count


def _render_flags_section(inp: TiReportInput) -> str:
    """Render indications and triage flags table with C-scan crops."""
    rows, total_zone_crops = _render_flag_rows(inp)
    lines = [
        "<section><h2>Flag List</h2>",
        "<table><thead><tr>",
        "<th>#</th><th>Source</th><th>Category</th><th>Zone</th>",
        "<th>Axial (mm)</th><th>Angle (deg)</th><th>Depth (mm)</th>",
        "<th>% of Alarm or Reference</th><th>Indication SNR</th>",
        "<th>Reasons</th><th>Review Tags</th><th>C-Scan Crop</th>",
        "<th>Inspector decision</th>",
        "</tr></thead><tbody>",
    ]
    lines.extend(rows)
    lines.append("</tbody></table>")
    if total_zone_crops > 30:
        lines.append(
            f'<p class="note">Showing C-scan crops for 30 zone flags '
            f"({total_zone_crops - 30} flags not imaged due to 30-crop limit).</p>"
        )
    else:
        lines.append(f'<p class="note">All {total_zone_crops} zone flags imaged.</p>')
    lines.append("</section>")
    return "\n".join(lines)


def _extra_styles() -> str:
    """Return supplemental styles for warnings and footer."""
    return (
        "<style>\n"
        ".warning.not-inspectable {\n"
        "  background: #fee2e2;\n"
        "  border: 1px solid #dc2626;\n"
        "  color: #991b1b;\n"
        "  padding: 10px;\n"
        "  font-weight: bold;\n"
        "  margin-bottom: 12px;\n"
        "  border-radius: 4px;\n"
        "}\n"
        "footer {\n"
        "  margin-top: 32px;\n"
        "  border-top: 1px solid #cbd5e1;\n"
        "  padding-top: 12px;\n"
        "  font-size: 13px;\n"
        "}\n"
        "</style>"
    )


def _render_page_body(inp: TiReportInput) -> str:
    """Render main body components of titanium inspection report."""
    return "\n".join(
        [
            _render_banners(inp),
            _render_metadata(inp),
            _render_calibration_section(inp.record),
            _render_inspectability_warnings(inp.record),
            _render_flags_section(inp),
            _render_signature_block(),
            "<footer><p>UTriage never accepts or passes a part.</p></footer>",
        ]
    )


def render_ti_report(inp: TiReportInput) -> str:
    """Render complete offline HTML inspection report for titanium mode."""
    css = f"{_render_styles()}\n{_extra_styles()}"
    title = f"UTriage Titanium Inspection Report - {html.escape(inp.bar_id)}"
    body = _render_page_body(inp)
    return (
        f"<!DOCTYPE html>\n"
        f'<html lang="en">\n'
        f"<head>\n"
        f'<meta charset="utf-8">\n'
        f"<title>{title}</title>\n"
        f"{css}\n"
        f"</head>\n"
        f"<body>\n"
        f"{body}\n"
        f"</body>\n"
        f"</html>"
    )


def write_ti_report(inp: TiReportInput, path: Path) -> Path:
    """Write rendered titanium report to file as UTF-8."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_ti_report(inp), encoding="utf-8")
    return target
