"""One-command shop-floor triage workflow and batch runner (project plan 6.2)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from utriage.app_logic import Session, load_session
from utriage.report.html import ReportInput, write_report
from utriage.rules.config import load_acceptance
from utriage.rules.pipeline import _indication_dict, evaluate_scan, evaluation_to_dict
from utriage.triage.fusion import MlFinding, TriageItem, fuse, rank

ADVISORY_BANNER = "Advisory tool - decisions by certified inspector"


@dataclass(frozen=True)
class TriageJob:
    """Specification for an automated offline triage run on a single bar scan."""

    scan: Path
    out_dir: Path
    model: Path | None = None
    acceptance: Path | None = None


def _ml_finding_dict(ml: MlFinding) -> dict:
    """Format ML finding as a JSON-safe dictionary."""
    return {
        "axial_idx": int(ml.axial_idx),
        "angle_idx": int(ml.angle_idx),
        "depth_mm": float(ml.depth_mm),
        "score": float(ml.score),
        "uncertainty": float(ml.uncertainty),
        "grain_noise_prob": (
            float(ml.grain_noise_prob) if ml.grain_noise_prob is not None else None
        ),
        "evidence": list(ml.evidence),
    }


def _triage_item_dict(item: TriageItem) -> dict:
    """Format fused triage item as a JSON-safe dictionary."""
    ind_dict = _indication_dict(item.indication) if item.indication is not None else None
    ml_dict = _ml_finding_dict(item.ml) if item.ml is not None else None
    return {
        "source": str(item.source),
        "category": str(item.category),
        "indication": ind_dict,
        "ml": ml_dict,
        "review": list(item.review),
    }


def _prepare_session(job: TriageJob) -> Session:
    """Load session, optionally applying custom acceptance rules."""
    session = load_session(job.scan, model_path=job.model)
    if job.acceptance is not None:
        cfg = load_acceptance(job.acceptance)
        session.evaluation = evaluate_scan(session.scan, cfg)
        findings = [it.ml for it in session.items if it.ml is not None]
        session.items = rank(fuse(session.evaluation.indications, findings))
    return session


def _write_triage_html(session: Session, path: Path) -> Path:
    """Render and write offline HTML report with no decisions."""
    now_iso = datetime.now(UTC).isoformat()
    report_in = ReportInput(
        bar_id=session.bar_id,
        evaluation=session.evaluation,
        items=session.items,
        decisions=[],
        synthetic=session.synthetic,
        generated_at=now_iso,
    )
    return write_report(report_in, path)


def _write_triage_json(session: Session, items: list[TriageItem], path: Path) -> Path:
    """Write serialized evaluation, ranked items, and metadata to JSON."""
    data = evaluation_to_dict(session.evaluation)
    item_dicts = [_triage_item_dict(it) for it in items]
    data["items"] = item_dicts
    data["ranked_items"] = item_dicts
    data["model_loaded"] = bool(session.model_loaded)
    data["synthetic"] = bool(session.synthetic)
    data["advisory_banner"] = ADVISORY_BANNER
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return path


def _base_summary(bar_id: str, scan: str) -> dict:
    """Initialize common fields for triage summary dictionary."""
    return {
        "bar_id": bar_id,
        "scan": scan,
        "counts_per_category": {},
        "category_counts": {},
        "counts": {},
        "n_ml_flag": 0,
        "rescan_recommended": False,
        "report_path": None,
        "json_path": None,
        "error": None,
    }


def _build_summary(
    session: Session,
    report_path: Path,
    json_path: Path,
) -> dict:
    """Assemble workflow summary statistics dictionary."""
    counts: dict[str, int] = {}
    for item in session.items:
        counts[item.category] = counts.get(item.category, 0) + 1
    n_ml_flag = sum(1 for it in session.items if it.category == "ml_flag")
    rescan = bool(session.evaluation.quality.rescan_recommended)
    d = _base_summary(session.bar_id, str(session.bar_id))
    d.update(
        {
            "counts_per_category": counts,
            "category_counts": counts,
            "counts": counts,
            "n_ml_flag": n_ml_flag,
            "rescan_recommended": rescan,
            "report_path": str(report_path),
            "json_path": str(json_path),
        }
    )
    return d


def run_triage(job: TriageJob) -> dict:
    """Execute end-to-end triage on a single bar scan and write artefacts."""
    session = _prepare_session(job)
    bar_id = session.bar_id or Path(job.scan).stem
    html_path = job.out_dir / f"{bar_id}_report.html"
    json_path = job.out_dir / f"{bar_id}_triage.json"
    _write_triage_html(session, html_path)
    _write_triage_json(session, session.items, json_path)
    return _build_summary(session, html_path, json_path)


def _failing_summary(scan: Path, error_msg: str) -> dict:
    """Build summary dictionary for a failed scan."""
    d = _base_summary(scan.stem, str(scan))
    d["error"] = error_msg
    return d


def run_batch(
    scans: list[Path],
    out_dir: Path,
    model: Path | None = None,
    acceptance: Path | None = None,
) -> list[dict]:
    """Execute triage across a collection of scans, recording failures."""
    results: list[dict] = []
    for scan in scans:
        job = TriageJob(scan=scan, out_dir=out_dir, model=model, acceptance=acceptance)
        try:
            results.append(run_triage(job))
        except (OSError, ValueError, RuntimeError, KeyError) as exc:  # scan load or eval failure
            results.append(_failing_summary(scan, str(exc)))
    return results
