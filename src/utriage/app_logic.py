"""Application logic and session orchestration for UTriage (pure, no Streamlit)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from utriage.audit import append_entry  # noqa: E402
from utriage.io.adapters import read_scan  # noqa: E402
from utriage.ml.bar_infer import load_window_model, scan_bar  # noqa: E402
from utriage.physics.simulator import BarScan, cscan_peak, envelope  # noqa: E402
from utriage.report.html import Decision  # noqa: E402
from utriage.rules.config import load_acceptance  # noqa: E402
from utriage.rules.pipeline import Evaluation, dac_for, evaluate_scan  # noqa: E402
from utriage.triage.fusion import MlFinding, TriageItem, fuse, rank  # noqa: E402

VALID_DECISIONS = frozenset({"accepted", "rejected", "rescan"})


@dataclass
class Session:
    """Inspector session containing scan, evaluations, triage items, and decisions."""

    scan: BarScan
    evaluation: Evaluation
    items: list[TriageItem]
    decisions: list[Decision]
    synthetic: bool
    model_loaded: bool
    bar_id: str = ""


def _is_synthetic(path: Path, scan: BarScan) -> bool:
    """Check whether a scan is synthetic from path or sibling json metadata."""
    norm_path = path.as_posix().lower()
    if "data/samples" in norm_path or "data/synthetic" in norm_path:
        return True
    json_path = path.with_suffix(".json")
    if json_path.is_file():
        try:
            with json_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict) and "synthetic" in str(data.get("label", "")).lower():
                    return True
        except (OSError, json.JSONDecodeError):
            pass
    return "synthetic" in getattr(scan, "label", "").lower()


def _run_ml_if_available(
    model_path: Path | None,
    scan: BarScan,
) -> tuple[list[MlFinding], bool]:
    """Run WindowNet bar inference if model checkpoint exists on disk."""
    if model_path is None:
        return [], False
    p = Path(model_path)
    if not p.is_file():
        return [], False
    model, _ = load_window_model(p)
    return scan_bar(model, scan), True


def load_session(path: Path, model_path: Path | None = None) -> Session:
    """Load scan, evaluate rules, run ML if available, fuse, and rank triage items."""
    path = Path(path)
    scan = read_scan(path)
    cfg = load_acceptance()
    eval_res = evaluate_scan(scan, cfg)
    findings, model_loaded = _run_ml_if_available(model_path, scan)
    fused_items = fuse(eval_res.indications, findings)
    ranked_items = rank(fused_items)
    is_synth = _is_synthetic(path, scan)
    return Session(
        scan=scan,
        evaluation=eval_res,
        items=ranked_items,
        decisions=[],
        synthetic=is_synth,
        model_loaded=model_loaded,
        bar_id=path.stem,
    )


def _append_audit_row(log_path: Path, row: list[str]) -> None:
    """Append decision audit log row, creating parent directories and header if needed."""
    target = Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_header = not target.exists() or target.stat().st_size == 0
    with target.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(
                [
                    "timestamp",
                    "bar_id",
                    "item_index",
                    "source",
                    "category",
                    "decision",
                    "comment",
                    "inspector",
                ]
            )
        writer.writerow(row)


def _normalize_decision(decision: str) -> str:
    """Validate and normalize inspector decision string."""
    norm = decision.strip().lower()
    norm = "rescan" if norm == "re-scan" else norm
    if norm not in VALID_DECISIONS:
        raise ValueError(f"decision must be accepted, rejected or rescan, got: {decision}")
    return norm


def record_decision(
    session: Session,
    item_index: int,
    decision: str,
    comment: str,
    inspector: str,
    log_path: Path,
) -> None:
    """Record an inspector disposition in session and append to the audit log."""
    norm_dec = _normalize_decision(decision)
    item = session.items[item_index] if 0 <= item_index < len(session.items) else None
    src, cat = (item.source, item.category) if item else ("", "")
    now_iso = datetime.now(UTC).isoformat()
    bar_id = session.bar_id or "bar"

    session.decisions.append(
        Decision(
            item_index=item_index,
            decision=norm_dec,
            comment=comment,
            inspector=inspector,
        )
    )
    _append_audit_row(
        log_path,
        [now_iso, bar_id, str(item_index), src, cat, norm_dec, comment, inspector],
    )
    record = {
        "bar_id": bar_id,
        "item_index": item_index,
        "source": src,
        "category": cat,
        "decision": norm_dec,
        "comment": comment,
        "inspector": inspector,
        "ts": now_iso,
    }
    append_entry(Path(log_path).with_suffix(".jsonl"), record)


def _plot_ascan_gates(
    ax: plt.Axes,
    t_if: float,
    v_mm_us: float,
    diameter_mm: float,
) -> None:
    """Plot gate markers on an A-scan."""
    cfg = load_acceptance()
    t_front = t_if + 2.0 * cfg.gates.front_dead_zone_mm / v_mm_us
    t_flaw_end = t_if + 2.0 * (diameter_mm - cfg.gates.back_margin_mm) / v_mm_us
    t_bw = t_if + 2.0 * diameter_mm / v_mm_us

    ax.axvline(t_front, color="purple", linestyle="--", label="Front dead zone")
    ax.axvline(t_flaw_end, color="darkorange", linestyle="--", label="Flaw gate end")
    ax.axvline(t_bw, color="green", linestyle=":", label="Back-wall echo")


def ascan_figure(session: Session, ia: int, ja: int) -> Figure:
    """Build matplotlib figure with RF, envelope, gates, and DAC curve."""
    rf = session.scan.ascans[ia, ja, :]
    env = envelope(rf)
    t = session.scan.time_us
    t_if = float(session.scan.t_if_us[ia, ja])
    v_mm_us = session.scan.spec.material.velocity_m_s * 1e-3
    d_mm = float(session.scan.spec.diameter_mm)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, rf, label="RF", color="steelblue", alpha=0.6)
    ax.plot(t, env, label="Envelope", color="crimson", linewidth=1.5)

    _plot_ascan_gates(ax, t_if, v_mm_us, d_mm)

    dac = dac_for(session.scan)
    depths = (t - t_if) * v_mm_us / 2.0
    in_gate = (t >= t_if) & (depths <= d_mm)
    ax.plot(t[in_gate], dac(depths[in_gate]), color="magenta", label="DAC (100% ref)")

    title = f"A-scan (ia={ia}, ja={ja})"
    if session.synthetic:
        title += " (Synthetic data)"
    ax.set_title(title)
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def cscan_figure(session: Session) -> Figure:
    """Build matplotlib C-scan figure with rules (o) and ML (x) indication overlays."""
    gate_end = max(2.0, session.scan.spec.diameter_mm - 2.0)
    peak = cscan_peak(session.scan, 2.0, gate_end)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    extent = [0.0, 360.0, float(session.scan.spec.length_mm), 0.0]
    im = ax.imshow(peak, aspect="auto", extent=extent, cmap="viridis")
    fig.colorbar(im, ax=ax, label="Peak Amplitude")

    rule_items = [it for it in session.items if it.indication is not None]
    if rule_items:
        ax.plot(
            [it.indication.angle_deg for it in rule_items],
            [it.indication.axial_mm for it in rule_items],
            "o",
            markeredgecolor="cyan",
            markerfacecolor="none",
            markersize=8,
            markeredgewidth=1.8,
            label="Rules indication (o)",
        )

    ml_items = [it for it in session.items if it.source == "ml" and it.ml is not None]
    if ml_items:
        ax.plot(
            [session.scan.angle_deg[it.ml.angle_idx] for it in ml_items],
            [session.scan.axial_mm[it.ml.axial_idx] for it in ml_items],
            "rx",
            markersize=8,
            markeredgewidth=1.8,
            label="ML flag (x)",
        )

    title = f"C-scan — {session.scan.spec.material.name}"
    if session.synthetic:
        title += " (Synthetic data)"
    ax.set_title(title)
    ax.set_xlabel("Angle (°)")
    ax.set_ylabel("Axial (mm)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def _parse_used_line(path: Path) -> str:
    """Extract first line starting with 'Used' from data inventory markdown."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped.startswith("Used"):
                    return stripped
    except OSError:
        # Narrow exception for unreadable inventory file
        pass
    return ""


def _load_claims_json(path: Path) -> list[dict[str, Any]]:
    """Load claims list with claim text and scope limit from JSON."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError):
        # Narrow exception for missing or malformed claims JSON
        return []
    if not isinstance(raw, list):
        return []
    result: list[dict[str, Any]] = []
    for c in raw:
        if isinstance(c, dict):
            entry = dict(c)
            entry["text"] = str(c.get("text", ""))
            entry["scope_limit"] = str(c.get("scope_limit", ""))
            result.append(entry)
    return result


def _find_matrix_png(results_dir: Path) -> Path | None:
    """Find generalization matrix PNG in figures subfolder or directly in dir."""
    nested = results_dir / "figures" / "generalization_matrix.png"
    if nested.is_file():
        return nested
    flat = results_dir / "generalization_matrix.png"
    if flat.is_file():
        return flat
    return None


def load_evidence(results_dir: Path) -> dict[str, Any]:
    """Load evidence limits, claims, and matrix figure from results directory."""
    r_dir = Path(results_dir)
    missing: list[str] = []

    inv_path = r_dir / "data_inventory.md"
    used_line = ""
    if inv_path.is_file():
        used_line = _parse_used_line(inv_path)
    else:
        missing.append("data_inventory.md")

    claims_path = r_dir / "claims.json"
    claims: list[dict[str, Any]] = []
    if claims_path.is_file():
        claims = _load_claims_json(claims_path)
    else:
        missing.append("claims.json")

    matrix_png = _find_matrix_png(r_dir)
    if matrix_png is None:
        missing.append("figures/generalization_matrix.png")

    return {
        "used_line": used_line,
        "claims": claims,
        "matrix_png": matrix_png,
        "missing": missing,
    }
