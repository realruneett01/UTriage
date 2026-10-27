"""False call classification, agreement analysis, and trade-off curves (ADR 0010).

Provides pure helpers for categorizing false calls across detectors, computing
detection-vs-false-call trade-off curves, and synthesising reports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HIGH_SNR_THRESHOLD: float = 5.0
LOW_MARGIN_RATIO: float = 0.10
NEAR_EDGE_MARGIN_PIXELS: float = 25.0


@dataclass(frozen=True)
class FalseCallEvidence:
    """Evidence metrics used to classify a false call indication."""

    near_edge: bool = False
    snr: float = 0.0
    agreement_count: int = 1
    score: float = 0.0
    threshold: float = 1.0


@dataclass(frozen=True)
class FalseCallRecord:
    """Comprehensive details of an unflawed indication flagged by a detector."""

    batch_stem: str
    image_idx: int
    flagged_detectors: list[str]
    agreement_count: int
    scores: dict[str, float]
    margins: dict[str, float]
    thresholds: dict[str, float]
    image_snr: float
    near_edge: bool
    peak_row: int
    peak_col: int
    anomaly_peak_row: int
    anomaly_peak_col: int
    is_near_miss_supervised: bool
    cause: str = "unclassified"
    png_path: str = ""


def is_low_margin(score: float, threshold: float, ratio: float = LOW_MARGIN_RATIO) -> bool:
    """Check if score is within ratio of the threshold."""
    if threshold <= 0.0:
        return False
    return abs(score - threshold) <= (threshold * ratio)


def is_near_miss_supervised(
    score: float,
    threshold: float = 0.5,
    margin_ratio: float = LOW_MARGIN_RATIO,
) -> bool:
    """Check whether supervised CNN score represents a borderline clean near-miss."""
    if threshold <= 0.0:
        return False
    lower_bound = threshold * (1.0 - margin_ratio)
    return lower_bound <= score < threshold


def _extract_evidence(data: FalseCallEvidence | dict[str, Any]) -> FalseCallEvidence:
    """Normalize input evidence representation into a FalseCallEvidence object."""
    if isinstance(data, FalseCallEvidence):
        return data
    score = float(data.get("score", 0.0))
    threshold = float(data.get("threshold", 1.0))
    if "scores" in data and "thresholds" in data:
        detectors = data.get("flagged_detectors", list(data["scores"].keys()))
        for d in detectors:
            s_d = float(data["scores"].get(d, 0.0))
            t_d = float(data["thresholds"].get(d, 1.0))
            if is_low_margin(s_d, t_d):
                score, threshold = s_d, t_d
                break
    return FalseCallEvidence(
        near_edge=bool(data.get("near_edge", False)),
        snr=float(data.get("image_snr", data.get("snr", 0.0))),
        agreement_count=int(data.get("agreement_count", len(data.get("flagged_detectors", [1])))),
        score=score,
        threshold=threshold,
    )


def classify_false_call_cause(
    evidence: FalseCallEvidence | dict[str, Any],
    snr_thresh: float = HIGH_SNR_THRESHOLD,
    margin_ratio: float = LOW_MARGIN_RATIO,
) -> str:
    """Classify false call cause using documented analysis heuristics."""
    ev = _extract_evidence(evidence)
    if ev.near_edge:
        return "near_edge"
    if ev.snr >= snr_thresh:
        return "high_snr_structure"
    if is_low_margin(ev.score, ev.threshold, margin_ratio):
        return "low_margin"
    if ev.agreement_count > 1:
        return "multi_detector_agreement"
    return "unclassified"


def build_agreement_table(
    records: list[dict[str, Any]] | list[FalseCallRecord],
    detectors: list[str] | None = None,
) -> dict[str, Any]:
    """Compute agreement count distribution and detector overlap."""
    det_list = detectors or ["supervised", "anomaly", "h2", "h3"]
    counts_by_agreement: dict[int, int] = {i: 0 for i in range(1, len(det_list) + 1)}
    by_detector: dict[str, int] = {d: 0 for d in det_list}
    pairwise: dict[str, dict[str, int]] = {d: {o: 0 for o in det_list} for d in det_list}

    for item in records:
        rec = item if isinstance(item, dict) else item.__dict__
        flagged = rec.get("flagged_detectors", [])
        agr = int(rec.get("agreement_count", len(flagged)))
        counts_by_agreement[agr] = counts_by_agreement.get(agr, 0) + 1
        for d in flagged:
            if d in by_detector:
                by_detector[d] += 1
            for o in flagged:
                if d in pairwise and o in pairwise[d]:
                    pairwise[d][o] += 1

    return {
        "counts_by_agreement": counts_by_agreement,
        "by_detector": by_detector,
        "pairwise_overlap": pairwise,
        "total_flagged": len(records),
    }


def _evaluate_threshold_step(
    scores: np.ndarray,
    y_true: np.ndarray,
    thresh: float,
) -> tuple[float, float]:
    """Compute detection rate and false call rate at a single threshold."""
    flawed_mask = y_true == 1
    clean_mask = y_true == 0
    n_flawed = int(np.sum(flawed_mask))
    n_clean = int(np.sum(clean_mask))

    flaw_hits = int(np.sum((scores >= thresh) & flawed_mask))
    clean_hits = int(np.sum((scores >= thresh) & clean_mask))

    det_rate = float(flaw_hits / n_flawed) if n_flawed > 0 else 0.0
    fc_rate = float(clean_hits / n_clean) if n_clean > 0 else 0.0
    return det_rate, fc_rate


def compute_tradeoff_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray | None = None,
    operating_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute detection rate vs false-call rate across ascending thresholds."""
    if thresholds is None:
        min_v = float(np.min(scores)) if len(scores) > 0 else 0.0
        max_v = float(np.max(scores)) if len(scores) > 0 else 1.0
        thresh_arr = np.linspace(min_v, max_v, 50)
    else:
        thresh_arr = np.sort(thresholds)

    steps = [_evaluate_threshold_step(scores, y_true, float(t)) for t in thresh_arr]
    op_dr, op_fcr = _evaluate_threshold_step(scores, y_true, float(operating_threshold))
    return {
        "thresholds": [float(t) for t in thresh_arr],
        "detection_rates": [s[0] for s in steps],
        "false_call_rates": [s[1] for s in steps],
        "operating_point": {
            "threshold": float(operating_threshold),
            "detection_rate": op_dr,
            "false_call_rate": op_fcr,
        },
    }


def plot_tradeoff_curve(
    curve_data: dict[str, Any],
    detector_name: str,
    out_png: Path,
) -> None:
    """Plot detection rate vs false-call rate curve and mark operating point."""
    fc_rates, det_rates = curve_data["false_call_rates"], curve_data["detection_rates"]
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(fc_rates, det_rates, color="navy", lw=1.5, label=f"{detector_name} curve")

    op = curve_data.get("operating_point", {})
    if op:
        t_val = op.get("threshold", 0.0)
        fcr_val, dr_val = op.get("false_call_rate", 0.0), op.get("detection_rate", 0.0)
        label = f"Pre-registered T={t_val:.2f} (FCR={fcr_val:.3f}, DR={dr_val:.3f})"
        ax.scatter([fcr_val], [dr_val], color="crimson", marker="*", s=90, zorder=4, label=label)

    ax.set_title(f"Trade-off: {detector_name}\nReal data (ML-NDT) - Exploratory", fontsize=9)
    ax.set_xlabel("False Call Rate (unflawed test images)", fontsize=8)
    ax.set_ylabel("Detection Rate (held-out cracks)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=72)
    plt.close(fig)


def plot_flagged_image(
    img: np.ndarray,
    err_map: np.ndarray,
    png_path: Path,
    title: str = "",
) -> None:
    """Save 6x3 PNG showing B-scan image and anomaly reconstruction error map."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.0, 3.0))
    ax1.imshow(img, cmap="viridis", aspect="auto")
    ax1.set_title(title or "B-scan image", fontsize=8)
    ax1.set_xlabel("Scan col", fontsize=7)
    ax1.set_ylabel("Depth sample", fontsize=7)
    ax1.tick_params(labelsize=6)

    ax2.imshow(err_map, cmap="hot", aspect="auto")
    ax2.set_title("Anomaly error map", fontsize=8)
    ax2.set_xlabel("Scan col", fontsize=7)
    ax2.set_ylabel("Depth sample", fontsize=7)
    ax2.tick_params(labelsize=6)

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=72, pil_kwargs={"optimize": True})
    plt.close(fig)


def _format_cause_table(cause_counts: dict[str, int], total: int) -> str:
    """Format markdown table rows for classified failure causes."""
    rows = ["| Cause | Count | Percentage |", "|---|---|---|"]
    for cause, cnt in cause_counts.items():
        pct = (cnt / total * 100.0) if total > 0 else 0.0
        rows.append(f"| `{cause}` | {cnt} | {pct:.1f}% |")
    return "\n".join(rows)


def _format_agreement_table(table: dict[str, Any]) -> str:
    """Format agreement distribution as markdown table."""
    rows = ["| Agreement Count | Scans Flagged |", "|---|---|"]
    for agr, cnt in table.get("counts_by_agreement", {}).items():
        rows.append(f"| {agr} | {cnt} |")
    return "\n".join(rows)


def build_false_call_report_md(summary_data: dict[str, Any]) -> str:
    """Synthesise false call markdown report generated from counts only."""
    total = int(summary_data.get("n_flagged", 0))
    n_clean = int(summary_data.get("n_clean", 778))
    causes_md = _format_cause_table(summary_data.get("cause_counts", {}), total)
    agr_md = _format_agreement_table(summary_data.get("agreement_table", {}))

    return (
        "# False Call Analysis and Operating Point Trade-Offs\n\n"
        f"**Dataset**: {summary_data.get('label', 'Real data (ML-NDT)')}\n\n"
        "> [!NOTE]\n"
        "> The pre-registered thresholds are unchanged. Trade-off curves are presented "
        "for choosing an operating point; any change to operating thresholds is exploratory.\n\n"
        f"## Summary Statistics\n\n"
        f"- Clean test images evaluated: {n_clean}\n"
        f"- Total clean images flagged by any detector: {total}\n\n"
        f"## Agreement Across Detectors\n\n{agr_md}\n\n"
        f"## Cause Classification\n\n{causes_md}\n"
    )


def save_false_call_analysis(
    payload: dict[str, Any],
    json_path: Path,
    md_path: Path,
) -> None:
    """Write false call analysis JSON and markdown reports."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_content = build_false_call_report_md(payload)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_content, encoding="utf-8")
