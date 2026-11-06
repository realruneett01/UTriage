"""Quantitative claim generation for Phase 8 ML experiments.

Extracts verified quantitative claims from separability, robust LOCO, and lightweight
hybrid experiment results without inventing results or thresholds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from utriage.report.claims import Claim

SCOPE_LIMIT: str = (
    "ML-NDT weld, 3 physical cracks, leave-one-crack-out, "
    "clean test scans from held-out batches of the same weld"
)


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load JSON from path, returning None on failure."""
    if not path.exists():
        print(f"Skipping claims requiring {path.name}: file not found", file=sys.stderr)
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as err:  # narrow reason: unreadable or corrupt JSON
        print(f"Skipping claims requiring {path.name}: {err}", file=sys.stderr)
        return None


def _lookup_verdict_key(data: dict[str, Any], key: str) -> Any:
    """Lookup key directly or within nested verdicts dictionary."""
    if key in data:
        return data[key]
    nested = data.get("verdicts")
    return nested.get(key) if isinstance(nested, dict) else None


def _find_verdict_entry(data: dict[str, Any], key: str) -> Any:
    """Lookup raw verdict entry by key, prefix, or global verdict field."""
    val = _lookup_verdict_key(data, key)
    if val is not None:
        return val
    prefix = key.split("_")[0]
    return data.get(prefix, data.get("verdict"))


def _format_verdict_value(val: Any) -> str | None:
    """Format verdict value as clean string."""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, dict):
        if "verdict" in val:
            return str(val["verdict"]).strip()
        items = [f"{k}: {v}" for k, v in val.items() if isinstance(v, (str, int, float))]
        return ", ".join(items)
    return None


def _extract_verdict(verdicts_data: dict[str, Any] | None, claim_key: str) -> str | None:
    """Extract verdict string verbatim from preregistration verdicts dictionary."""
    if not isinstance(verdicts_data, dict):
        return None
    val = _find_verdict_entry(verdicts_data, claim_key)
    return _format_verdict_value(val)


def _append_verdict(text: str, verdict: str | None) -> str:
    """Append verbatim verdict to claim text if provided."""
    if not verdict:
        return text
    clean_v = verdict.strip()
    term = "" if clean_v.endswith(".") else "."
    return f"{text} Verdict: {clean_v}{term}"


def _format_separability_fold(fold: dict[str, Any]) -> tuple[str, dict[str, float]]:
    """Format single fold metrics for separability evaluation."""
    held_out = str(fold.get("held_out", ""))
    auroc = float(fold.get("auroc", 0.0))
    ap = float(fold.get("average_precision", 0.0))
    op = fold.get("operating_points", {}).get("0.05", {})
    det = op.get("detection", {})
    fc = op.get("false_calls", {})
    k_det, n_det = int(det.get("k", 0)), int(det.get("n", 0))
    k_fc, n_fc = int(fc.get("k", 0)), int(fc.get("n", 0))
    part = (
        f"{held_out}: AUROC {auroc:.3f}, AP {ap:.3f}, "
        f"at FCR 0.05 detection {k_det}/{n_det}, false calls {k_fc}/{n_fc} on clean_eval"
    )
    nums = {
        f"{held_out}_auroc": float(f"{auroc:.3f}"),
        f"{held_out}_ap": float(f"{ap:.3f}"),
        f"{held_out}_det_k": float(k_det),
        f"{held_out}_det_n": float(n_det),
        f"{held_out}_fc_k": float(k_fc),
        f"{held_out}_fc_n": float(n_fc),
    }
    return part, nums


def _resolve_protocol_desc(proto: str) -> str:
    """Resolve human-readable protocol description for claim text."""
    if "crack" in proto:
        return "crack-held-out early stopping"
    if "batch" in proto:
        return "batch-held-out early stopping"
    return f"{proto} early stopping"


def _build_single_separability_claim(
    proto: str,
    p_data: dict[str, Any],
    verdict: str | None,
) -> Claim | None:
    """Build a Claim for a single separability protocol."""
    from utriage.report.claims import Claim

    folds = p_data.get("folds", [])
    if not isinstance(folds, list) or not folds:
        return None
    parts: list[str] = []
    numbers: dict[str, float] = {"target_fcr": 0.05}
    for f in folds:
        part, nums = _format_separability_fold(f)
        parts.append(part)
        numbers.update(nums)
    proto_desc = _resolve_protocol_desc(proto)
    text = (
        f"Separability ({proto_desc}): {'; '.join(parts)}. Thresholds were set on clean scans only."
    )
    text = _append_verdict(text, verdict)
    return Claim(
        id=f"separability_{proto}",
        text=text,
        evidence_file="results/separability_metrics.json",
        evidence_path=f"protocols.{proto}.folds",
        scope_limit=SCOPE_LIMIT,
        numbers=numbers,
    )


def _build_separability_claims(
    r_dir: Path,
    verdicts: dict[str, Any] | None = None,
) -> list[Claim]:
    """Build claims for all separability protocols present in results."""
    data = _load_json(r_dir / "separability_metrics.json")
    if data is None:
        return []
    protos = data.get("protocols")
    if not isinstance(protos, dict):
        return []
    claims: list[Claim] = []
    for proto, p_data in protos.items():
        if isinstance(p_data, dict):
            v = _extract_verdict(verdicts, f"separability_{proto}")
            claim = _build_single_separability_claim(str(proto), p_data, v)
            if claim is not None:
                claims.append(claim)
    return claims


def _format_loco_robust_fold(fold: dict[str, Any]) -> tuple[str, dict[str, float]]:
    """Format single fold metrics for robust LOCO seed ensemble."""
    held_out = str(fold.get("held_out", ""))
    ens = fold.get("ensemble_metrics", {})
    auroc = float(ens.get("auroc", 0.0))
    ap = float(ens.get("average_precision", 0.0))
    fcr_05 = ens.get("fcr_0.05", {})
    det = fcr_05.get("detections", {})
    fc = fcr_05.get("false_calls", {})
    k_det = int(det.get("n_hit", det.get("k", 0)))
    n_det = int(det.get("n_flawed", det.get("n", 0)))
    k_fc = int(fc.get("n_false", fc.get("k", 0)))
    n_fc = int(fc.get("n_clean", fc.get("n", 0)))
    part = (
        f"{held_out}: AUROC {auroc:.3f}, AP {ap:.3f}, "
        f"at FCR 0.05 detection {k_det}/{n_det}, false calls {k_fc}/{n_fc} on clean_eval"
    )
    nums = {
        f"{held_out}_auroc": float(f"{auroc:.3f}"),
        f"{held_out}_ap": float(f"{ap:.3f}"),
        f"{held_out}_det_k": float(k_det),
        f"{held_out}_det_n": float(n_det),
        f"{held_out}_fc_k": float(k_fc),
        f"{held_out}_fc_n": float(n_fc),
    }
    return part, nums


def _resolve_seed_count(cfg: dict[str, Any], folds: list[dict[str, Any]]) -> int:
    """Resolve seed count from config or fallback to first fold seeds."""
    cfg_seeds = cfg.get("seeds")
    if isinstance(cfg_seeds, list) and cfg_seeds:
        return len(cfg_seeds)
    first_seeds = folds[0].get("seeds", [])
    return len(first_seeds) if isinstance(first_seeds, list) else 1


def _build_loco_robust_claim(
    r_dir: Path,
    verdicts: dict[str, Any] | None = None,
) -> Claim | None:
    """Build claim for robust LOCO seed ensemble."""
    from utriage.report.claims import Claim

    data = _load_json(r_dir / "loco_robust_metrics.json")
    if data is None:
        return None
    folds = data.get("folds", [])
    if not isinstance(folds, list) or not folds:
        return None
    n_seeds = _resolve_seed_count(data.get("config", {}), folds)
    parts: list[str] = []
    numbers: dict[str, float] = {"n_seeds": float(n_seeds), "target_fcr": 0.05}
    for f in folds:
        part, nums = _format_loco_robust_fold(f)
        parts.append(part)
        numbers.update(nums)
    text = (
        f"Robust LOCO seed ensemble ({n_seeds} seeds): {'; '.join(parts)}. "
        "Thresholds were set on clean scans only."
    )
    verdict = _extract_verdict(verdicts, "loco_robust")
    text = _append_verdict(text, verdict)
    return Claim(
        id="loco_robust",
        text=text,
        evidence_file="results/loco_robust_metrics.json",
        evidence_path="folds",
        scope_limit=SCOPE_LIMIT,
        numbers=numbers,
    )


def _format_lite_hybrid_fold(held_out: str, fold: dict[str, Any]) -> tuple[str, dict[str, float]]:
    """Format single fold metrics for lightweight hybrid evaluation."""
    fused_auc = float(fold.get("threshold_free", {}).get("fused", {}).get("auroc", 0.0))
    op = fold.get("operating_points", {}).get("fcr_0.05", {})
    hyb = op.get("hybrid", {})
    det = hyb.get("detection", {})
    fc = hyb.get("false_calls", {})
    k_det, n_det = int(det.get("k", 0)), int(det.get("n", 0))
    k_fc, n_fc = int(fc.get("k", 0)), int(fc.get("n", 0))
    part = (
        f"{held_out}: fused AUROC {fused_auc:.3f}, "
        f"at FCR 0.05 detection {k_det}/{n_det}, false calls {k_fc}/{n_fc}"
    )
    nums = {
        f"{held_out}_fused_auroc": float(f"{fused_auc:.3f}"),
        f"{held_out}_det_k": float(k_det),
        f"{held_out}_det_n": float(n_det),
        f"{held_out}_fc_k": float(k_fc),
        f"{held_out}_fc_n": float(n_fc),
    }
    return part, nums


def _extract_lite_latency(lat: dict[str, Any]) -> tuple[str, dict[str, float]]:
    """Extract LiteNet params, CPU median latency, and payload note."""
    litenet = lat.get("litenet", {})
    params = int(litenet.get("params", 0))
    med_ms = float(litenet.get("median_ms", 0.0))
    threads = int(lat.get("threads", 4))
    default_note = (
        f"measured on the development machine CPU limited to {threads} threads, "
        "not on the target i5 laptop"
    )
    note = str(lat.get("note", default_note))
    text = f"LiteNet {params} params, CPU median latency {med_ms:.2f} ms ({note})"
    nums = {
        "litenet_params": float(params),
        "median_ms": float(f"{med_ms:.2f}"),
        "threads": float(threads),
    }
    return text, nums


def _iter_lite_folds(folds_data: Any) -> list[tuple[str, dict[str, Any]]]:
    """Normalize folds data into a list of (held_out, fold_dict) pairs."""
    if isinstance(folds_data, dict):
        return [(str(k), v) for k, v in folds_data.items() if isinstance(v, dict)]
    if isinstance(folds_data, list):
        return [(str(f.get("held_out", "")), f) for f in folds_data if isinstance(f, dict)]
    return []


def _collect_lite_folds(folds_data: Any) -> tuple[list[str], dict[str, float]]:
    """Collect fold descriptions and numbers across dictionary or list folds."""
    parts: list[str] = []
    numbers: dict[str, float] = {}
    for held_out, fold in _iter_lite_folds(folds_data):
        p, nums = _format_lite_hybrid_fold(held_out, fold)
        parts.append(p)
        numbers.update(nums)
    return parts, numbers


def _build_lite_hybrid_claim(
    r_dir: Path,
    verdicts: dict[str, Any] | None = None,
) -> Claim | None:
    """Build claim for lightweight hybrid model and latency."""
    from utriage.report.claims import Claim

    data = _load_json(r_dir / "lite_hybrid_metrics.json")
    if data is None:
        return None
    parts, numbers = _collect_lite_folds(data.get("folds"))
    if not parts:
        return None
    numbers["target_fcr"] = 0.05
    lat_text, lat_nums = _extract_lite_latency(data.get("latency", {}))
    numbers.update(lat_nums)
    text = f"LiteNet hybrid: {'; '.join(parts)}. {lat_text}."
    verdict = _extract_verdict(verdicts, "lite_hybrid")
    text = _append_verdict(text, verdict)
    return Claim(
        id="lite_hybrid",
        text=text,
        evidence_file="results/lite_hybrid_metrics.json",
        evidence_path="folds",
        scope_limit=SCOPE_LIMIT,
        numbers=numbers,
    )


def collect_ml_claims(r_dir: Path | str) -> list[Claim]:
    """Collect quantitative claims from new ML experiment results."""
    root = Path(r_dir)
    verdicts = _load_json(root / "preregistration_verdicts.json")
    claims: list[Claim] = []

    claims.extend(_build_separability_claims(root, verdicts))

    c_robust = _build_loco_robust_claim(root, verdicts)
    if c_robust is not None:
        claims.append(c_robust)

    c_lite = _build_lite_hybrid_claim(root, verdicts)
    if c_lite is not None:
        claims.append(c_lite)

    return claims


build_separability_claims = _build_separability_claims
build_loco_robust_claim = _build_loco_robust_claim
build_lite_hybrid_claim = _build_lite_hybrid_claim
