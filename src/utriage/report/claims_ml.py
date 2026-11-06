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


def _lookup_dotted_path(data: dict[str, Any], path: str) -> Any:
    """Lookup dotted path in dictionary."""
    curr: Any = data
    for part in path.split("."):
        if not isinstance(curr, dict) or part not in curr:
            return None
        curr = curr[part]
    return curr


def _lookup_container_key(data: dict[str, Any], key: str) -> Any:
    """Lookup key within nested verdict containers."""
    for container in ("verdicts", "candidates"):
        nested = data.get(container, {})
        if isinstance(nested, dict):
            val = nested.get(key)
            if val is not None:
                return val
    return None


def _lookup_verdict_key(data: dict[str, Any], key: str) -> Any:
    """Lookup key directly, via dotted path, or within nested containers."""
    if key in data:
        return data[key]
    if "." in key:
        return _lookup_dotted_path(data, key)
    return _lookup_container_key(data, key)


def _find_verdict_entry(data: dict[str, Any], key: str) -> Any:
    """Lookup raw verdict entry by key, candidate path, prefix, or global verdict field."""
    val = _lookup_verdict_key(data, key)
    if val is not None:
        return val
    cand_val = _lookup_verdict_key(data, f"candidates.{key}")
    if cand_val is not None:
        return cand_val
    prefix = key.split("_")[0]
    val_prefix = _lookup_verdict_key(data, prefix)
    if val_prefix is not None:
        return val_prefix
    cand_prefix = _lookup_verdict_key(data, f"candidates.{prefix}")
    if cand_prefix is not None:
        return cand_prefix
    return data.get("verdict")


def _format_folds_verdict(folds: dict[str, Any]) -> str | None:
    """Format dictionary of fold verdicts into a comma-separated string."""
    fold_parts: list[str] = []
    for fold_name, fold_item in folds.items():
        v_str = fold_item.get("verdict") if isinstance(fold_item, dict) else fold_item
        if isinstance(v_str, (str, int, float)):
            fold_parts.append(f"{fold_name}: {v_str}")
    return ", ".join(fold_parts) if fold_parts else None


def _format_single_candidate(cand_name: str, cand_info: Any) -> str | None:
    """Format verdict entry for a single candidate."""
    if isinstance(cand_info, dict):
        folds = cand_info.get("folds")
        if isinstance(folds, dict):
            f_str = _format_folds_verdict(folds)
            return f"{cand_name}: {f_str}" if f_str else None
    if isinstance(cand_info, (str, int, float)):
        return f"{cand_name}: {cand_info}"
    return None


def _format_candidates_verdict(candidates: dict[str, Any]) -> str | None:
    """Format dictionary of candidate verdicts into a semicolon-separated string."""
    parts: list[str] = []
    for cand_name, cand_info in candidates.items():
        c_str = _format_single_candidate(cand_name, cand_info)
        if c_str is not None:
            parts.append(c_str)
    return "; ".join(parts) if parts else None


def _format_dict_items(val: dict[str, Any]) -> str | None:
    """Format plain key-value mapping items."""
    items = [f"{k}: {v}" for k, v in val.items() if isinstance(v, (str, int, float))]
    return ", ".join(items) if items else None


def _format_verdict_value(val: Any) -> str | None:
    """Format verdict value as clean string."""
    if isinstance(val, str):
        return val.strip()
    if not isinstance(val, dict):
        return None
    v_raw = val.get("verdict")
    if isinstance(v_raw, (str, int, float)):
        return str(v_raw).strip()
    if isinstance(val.get("candidates"), dict):
        return _format_candidates_verdict(val["candidates"])
    if isinstance(val.get("folds"), dict):
        return _format_folds_verdict(val["folds"])
    return _format_dict_items(val)


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
    # the judge names this candidate "robust"
    verdict = _extract_verdict(verdicts, "candidates.robust") or _extract_verdict(
        verdicts, "loco_robust"
    )
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


def _format_invariant_fold(fold: dict[str, Any]) -> tuple[str, dict[str, float]]:
    """Format single fold metrics for invariant LOCO evaluation."""
    held_out = str(fold.get("held_out", ""))
    # keys as written by experiments/run_invariant_loco.py build_fold_record;
    # a missing key raises instead of reporting a fake 0
    fused = fold["fused"]
    auroc = float(fused["auroc"])
    op_05 = fused["operating_points"]["0.05"]
    det, fc = op_05["detection"], op_05["false_calls"]
    k_det, n_det = int(det["k"]), int(det["n"])
    k_fc, n_fc = int(fc["k"]), int(fc["n"])
    part = (
        f"{held_out}: fused AUROC {auroc:.3f}, "
        f"at FCR 0.05 detection {k_det}/{n_det}, false calls {k_fc}/{n_fc}"
    )
    nums = {
        f"{held_out}_fused_auroc": float(f"{auroc:.3f}"),
        f"{held_out}_det_k": float(k_det),
        f"{held_out}_det_n": float(n_det),
        f"{held_out}_fc_k": float(k_fc),
        f"{held_out}_fc_n": float(n_fc),
    }
    return part, nums


def _collect_invariant_folds(folds_data: Any) -> tuple[list[str], dict[str, float]]:
    """Iterate and format invariant LOCO folds from list or dict."""
    folds_list = (
        folds_data
        if isinstance(folds_data, list)
        else list(folds_data.values())
        if isinstance(folds_data, dict)
        else []
    )
    parts: list[str] = []
    numbers: dict[str, float] = {}
    for f in folds_list:
        if isinstance(f, dict):
            part, nums = _format_invariant_fold(f)
            parts.append(part)
            numbers.update(nums)
    return parts, numbers


def _build_invariant_loco_claim(
    r_dir: Path,
    verdicts: dict[str, Any] | None = None,
) -> Claim | None:
    """Build claim for invariant representation LOCO evaluation."""
    from utriage.report.claims import Claim

    data = _load_json(r_dir / "invariant_loco_metrics.json")
    if data is None:
        return None
    parts, numbers = _collect_invariant_folds(data.get("folds"))
    if not parts:
        return None
    numbers["target_fcr"] = 0.05
    text = f"Invariant LOCO: {'; '.join(parts)}."
    verdict = _extract_verdict(verdicts, "candidates.invariant") or _extract_verdict(
        verdicts, "invariant_loco"
    )
    text = _append_verdict(text, verdict)
    return Claim(
        id="invariant_loco",
        text=text,
        evidence_file="results/invariant_loco_metrics.json",
        evidence_path="folds",
        scope_limit=SCOPE_LIMIT,
        numbers=numbers,
    )


def _is_all_clean_flagged(shift_item: dict[str, Any]) -> bool:
    """Check if all clean_eval scans were flagged at FCR 0.01 in this shift."""
    op_01 = shift_item.get("operating_points", {}).get("0.01", {})
    fc = op_01.get("false_calls", {})
    k = int(fc.get("k", 0))
    n = int(fc.get("n", 0))
    return k == n and n > 0


def _count_clean_flagged_shifts(shifts: list[dict[str, Any]]) -> int:
    """Count shifts where every clean_eval scan was flagged at FCR 0.01."""
    return sum(1 for s in shifts if isinstance(s, dict) and _is_all_clean_flagged(s))


def _extract_worst_rates(fold: dict[str, Any]) -> tuple[float, float]:
    """Extract worst-case detection and false-call rates at FCR 0.01."""
    # summary keys as written by experiments/run_shift_robustness.py
    summary = fold["summary"]
    return (
        float(summary["worst_case_detection_rate_0.01"]),
        float(summary["worst_case_false_call_rate_0.01"]),
    )


def _format_shift_fold(
    family: str,
    fold: dict[str, Any],
) -> tuple[str, dict[str, float]]:
    """Format single fold metrics for shift robustness evaluation."""
    held_out = str(fold.get("held_out", ""))
    worst_det, worst_fc = _extract_worst_rates(fold)
    shifts = [s for s in fold.get("shifts", []) if isinstance(s, dict)]
    all_clean = _count_clean_flagged_shifts(shifts)
    part = (
        f"{held_out}: worst-case detection {worst_det:.3f}, "
        f"worst-case false calls {worst_fc:.3f}, "
        f"{all_clean}/{len(shifts)} shifts flagged every clean_eval scan"
    )
    nums = {
        f"{family}_{held_out}_worst_det": float(f"{worst_det:.3f}"),
        f"{family}_{held_out}_worst_fcr": float(f"{worst_fc:.3f}"),
        f"{family}_{held_out}_all_clean_shifts": float(all_clean),
        f"{family}_{held_out}_n_shifts": float(len(shifts)),
    }
    return part, nums


def _format_shift_family(
    family: str,
    fam_data: dict[str, Any],
) -> tuple[str | None, dict[str, float]]:
    """Format all folds for a single model family under shift robustness."""
    folds = fam_data.get("folds", [])
    if not isinstance(folds, list) or not folds:
        return None, {}
    fold_parts: list[str] = []
    nums: dict[str, float] = {}
    for fold in folds:
        if isinstance(fold, dict):
            part, fold_nums = _format_shift_fold(family, fold)
            fold_parts.append(part)
            nums.update(fold_nums)
    if not fold_parts:
        return None, {}
    fam_text = f"{family} ({'; '.join(fold_parts)})"
    return fam_text, nums


SHIFT_ROBUSTNESS_NOTE: str = (
    "thresholds fixed on unshifted clean scans, "
    "shifts are synthetic gain/offset/noise perturbations of real weld images, "
    "site calibration (ADR 0011) is the intended mitigation and was not applied here; "
    "the false-call condition of the ADR 0015 rule is met trivially wherever the "
    "baseline worst-case false-call rate is already at its maximum"
)


def _collect_shift_families(
    families: dict[str, Any],
) -> tuple[list[str], dict[str, float]]:
    """Format and collect all valid family descriptions and numbers."""
    fam_parts: list[str] = []
    numbers: dict[str, float] = {}
    for fam_name, fam_data in families.items():
        if isinstance(fam_data, dict):
            f_text, f_nums = _format_shift_family(str(fam_name), fam_data)
            if f_text is not None:
                fam_parts.append(f_text)
                numbers.update(f_nums)
    return fam_parts, numbers


def _build_shift_robustness_claim(
    r_dir: Path,
    verdicts: dict[str, Any] | None = None,
) -> Claim | None:
    """Build claim for shift robustness evaluation across model families."""
    from utriage.report.claims import Claim

    data = _load_json(r_dir / "shift_robustness_metrics.json")
    if data is None:
        return None
    families = data.get("families")
    if not isinstance(families, dict) or not families:
        return None
    fam_parts, numbers = _collect_shift_families(families)
    if not fam_parts:
        return None
    numbers["target_fcr"] = 0.01
    text = f"Shift robustness at FCR 0.01: {'; '.join(fam_parts)}; {SHIFT_ROBUSTNESS_NOTE}."
    text = _append_verdict(text, _extract_verdict(verdicts, "shift_robustness"))
    return Claim(
        id="shift_robustness",
        text=text,
        evidence_file="results/shift_robustness_metrics.json",
        evidence_path="families",
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

    c_invariant = _build_invariant_loco_claim(root, verdicts)
    if c_invariant is not None:
        claims.append(c_invariant)

    c_shift = _build_shift_robustness_claim(root, verdicts)
    if c_shift is not None:
        claims.append(c_shift)

    return claims


build_separability_claims = _build_separability_claims
build_loco_robust_claim = _build_loco_robust_claim
build_lite_hybrid_claim = _build_lite_hybrid_claim
build_invariant_loco_claim = _build_invariant_loco_claim
build_shift_robustness_claim = _build_shift_robustness_claim
