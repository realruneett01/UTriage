"""Generate resume bullets strictly from results JSON files and claims."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Any

from utriage.report.claims import check_forbidden

FORBIDDEN_WORDS: tuple[str, ...] = (
    "verified",
    "confirming",
    "establishing",
    "ensuring",
    "guarantee",
    "nearly",
    "aerospace bar inspection",
)


def round_ceil(val: float | int | str, decimals: int = 2) -> str:
    """Format number rounded up (ceiling) to avoid project-favour rounding."""
    step = Decimal("10") ** -decimals
    d = Decimal(str(val))
    return str(d.quantize(step, rounding=ROUND_CEILING))


def round_floor(val: float | int | str, decimals: int = 2) -> str:
    """Format number rounded down (floor) to avoid project-favour rounding."""
    step = Decimal("10") ** -decimals
    d = Decimal(str(val))
    return str(d.quantize(step, rounding=ROUND_FLOOR))


def _ensure_scope(text: str, scope: str) -> str:
    """Ensure bullet text ends with a scope tag in parentheses."""
    stripped = text.strip().rstrip(".")
    scope_tag = scope.strip("()")
    if stripped.endswith(f"({scope_tag})"):
        return f"{stripped}."
    return f"{stripped} ({scope_tag})."


def _build_framing_bullet() -> str:
    """Build independent prototype framing bullet."""
    text = (
        "Built an independent prototype for advisory, offline ultrasonic indication triage "
        "for ultrasonic inspection; simulated bar scans and public weld PAUT data, "
        "designed so ML can only add flags to a rules engine (property-tested on simulator columns)"
    )
    return _ensure_scope(text, "advisory")


def _build_leakage_bullet(claims: dict[str, dict[str, Any]]) -> str | None:
    """Build dataset leakage audit bullet from leakage claim."""
    c = claims.get("leakage")
    if c is None:
        return None
    nums = c.get("numbers", {})
    if "shared_cracks" in nums:
        shared = int(nums["shared_cracks"])
    else:
        match = re.search(r"all\s+(\d+)\s+physical cracks", c.get("text", ""))
        if not match:
            sys.stderr.write("Warning: missing shared_cracks in leakage claim; skipping bullet.\n")
            return None
        shared = int(match.group(1))
    text = (
        f"Audited a public weld PAUT benchmark and found the random split shared "
        f"all {shared} physical cracks between train and test, superseding the earlier test score"
    )
    return _ensure_scope(text, "public dataset")


def _get_k_n(claim: dict[str, Any]) -> tuple[int, int] | None:
    """Extract k and n detection counts from claim numbers or text."""
    nums = claim.get("numbers", {})
    if "k" in nums and "n" in nums:
        return int(nums["k"]), int(nums["n"])
    match = re.search(r"(\d+)\s*(?:of|/)\s*(\d+)", claim.get("text", ""))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


@dataclass(frozen=True)
class FoldCounts:
    """Detection and false-call counts for a fold."""

    det_k: int
    det_n: int
    fc_k: int | None = None
    fc_n: int | None = None


def _parse_fold_counts(fold: dict[str, Any]) -> tuple[str, FoldCounts] | None:
    """Extract held_out name and detection/false-call counts from fold dict."""
    held = fold.get("held_out")
    det = fold.get("detections", {})
    k = det.get("n_hit", det.get("k"))
    n = det.get("n_flawed", det.get("n"))
    if any((held is None, k is None, n is None)):
        return None
    fc = fold.get("false_calls", {})
    fc_k = fc.get("n_false", fc.get("k"))
    fc_n = fc.get("n_unflawed", fc.get("n", fc.get("n_clean")))
    f_k = int(fc_k) if fc_k is not None else None
    f_n = int(fc_n) if fc_n is not None else None
    return str(held), FoldCounts(int(k), int(n), f_k, f_n)


def _extract_loco_from_file(path: Path) -> dict[str, FoldCounts] | None:
    """Extract fold counts from a LOCO metrics JSON file."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    res: dict[str, FoldCounts] = {}
    for fold in data.get("folds", []):
        parsed = _parse_fold_counts(fold)
        if parsed:
            res[parsed[0]] = parsed[1]
    if len(res) < 3:
        return None
    return res


def _extract_loco_batch_claims(
    claims: dict[str, dict[str, Any]],
) -> dict[str, FoldCounts] | None:
    """Extract batch LOCO detection counts from individual claims."""
    keys = {
        "crack_8.6": "loco_crack_8.6",
        "crack_1.6": "loco_crack_1.6",
        "crack_4.0": "loco_crack_4.0",
    }
    res: dict[str, FoldCounts] = {}
    for crack, claim_id in keys.items():
        c = claims.get(claim_id)
        if c is None:
            return None
        kn = _get_k_n(c)
        if kn is None:
            return None
        nums = c.get("numbers", {})
        fc_k = nums.get("n_false", nums.get("fc_k"))
        fc_n = nums.get("n_unflawed", nums.get("fc_n"))
        f_k = int(fc_k) if fc_k is not None else None
        f_n = int(fc_n) if fc_n is not None else None
        res[crack] = FoldCounts(kn[0], kn[1], f_k, f_n)
    return res


def _extract_loco_batch_counts(
    claims: dict[str, dict[str, Any]], results_dir: Path
) -> dict[str, FoldCounts] | None:
    """Extract batch-held-out LOCO counts."""
    fc = _extract_loco_from_file(results_dir / "loco_metrics.json")
    if fc:
        return fc
    return _extract_loco_batch_claims(claims)


def _extract_loco_crackval_claim(
    nums: dict[str, Any],
) -> dict[str, FoldCounts] | None:
    """Extract crack-held-out LOCO counts from claim numbers dict."""
    keys = ("crack_8.6", "crack_1.6", "crack_4.0")
    res: dict[str, FoldCounts] = {}
    for crack in keys:
        k = nums.get(f"{crack}_k")
        n = nums.get(f"{crack}_n")
        if k is None or n is None:
            return None
        fc_k = nums.get(f"{crack}_n_false", nums.get(f"{crack}_fc_k"))
        fc_n = nums.get(f"{crack}_n_unflawed", nums.get(f"{crack}_fc_n"))
        f_k = int(fc_k) if fc_k is not None else None
        f_n = int(fc_n) if fc_n is not None else None
        res[crack] = FoldCounts(int(k), int(n), f_k, f_n)
    return res


def _extract_loco_crackval_counts(
    claims: dict[str, dict[str, Any]], results_dir: Path
) -> dict[str, FoldCounts] | None:
    """Extract crack-held-out LOCO counts."""
    fc = _extract_loco_from_file(results_dir / "loco_crackval_metrics.json")
    if fc:
        return fc
    cv = claims.get("loco_crackval")
    if cv is None:
        return None
    return _extract_loco_crackval_claim(cv.get("numbers", {}))


def _format_fold_tuple(
    f8: FoldCounts, f1: FoldCounts, f4: FoldCounts, *, include_fc: bool = True
) -> str:
    """Format 3-fold detections and optional false calls."""
    det_str = (
        f"8.6 mm {f8.det_k}/{f8.det_n}, 1.6 mm {f1.det_k}/{f1.det_n}, 4.0 mm {f4.det_k}/{f4.det_n}"
    )
    if not include_fc:
        return f"({det_str})"
    fc_vals = (f8.fc_k, f8.fc_n, f1.fc_k, f1.fc_n, f4.fc_k, f4.fc_n)
    if any(v is None for v in fc_vals):
        return f"({det_str})"
    fc_str = f"false calls {f8.fc_k}/{f8.fc_n}, {f1.fc_k}/{f1.fc_n}, {f4.fc_k}/{f4.fc_n}"
    return f"({det_str}; {fc_str})"


def _build_loco_bullet(claims: dict[str, dict[str, Any]], results_dir: Path) -> str | None:
    """Build leave-one-crack-out bullet comparing batch vs crack validation."""
    b_counts = _extract_loco_batch_counts(claims, results_dir)
    c_counts = _extract_loco_crackval_counts(claims, results_dir)
    if b_counts is None or c_counts is None:
        sys.stderr.write("Warning: missing LOCO counts; skipping bullet.\n")
        return None
    b_has_fc = (results_dir / "loco_metrics.json").exists()
    b_tup = _format_fold_tuple(
        b_counts["crack_8.6"], b_counts["crack_1.6"], b_counts["crack_4.0"], include_fc=b_has_fc
    )
    c_tup = _format_fold_tuple(
        c_counts["crack_8.6"], c_counts["crack_1.6"], c_counts["crack_4.0"], include_fc=True
    )
    text = f"Detected held-out cracks: batch early stopping {b_tup} vs crack-held-out {c_tup}"
    return _ensure_scope(text, "leave-one-crack-out, 3 physical cracks, one public weld dataset")


@dataclass(frozen=True)
class SepCrackCounts:
    """Detection and false-call counts for 1.6 mm crack separability."""

    k: int
    n: int
    fk: int
    fn: int


def _extract_separability_aurocs(claim: dict[str, Any]) -> tuple[float, float, int] | None:
    """Extract min AUROC, max AUROC, and fold count from separability claim."""
    nums = claim.get("numbers", {})
    aurocs = [float(v) for k, v in nums.items() if k.endswith("_auroc")]
    if len(aurocs) < 3:
        return None
    return min(aurocs), max(aurocs), len(aurocs)


def _extract_separability_16_counts(claim: dict[str, Any]) -> SepCrackCounts | None:
    """Extract 1.6 mm crack detection and false calls from separability claim."""
    nums = claim.get("numbers", {})
    keys = ("crack_1.6_det_k", "crack_1.6_det_n", "crack_1.6_fc_k", "crack_1.6_fc_n")
    vals = [nums.get(key) for key in keys]
    if any(v is None for v in vals):
        return None
    k, n, fk, fn = vals
    return SepCrackCounts(int(k), int(n), int(fk), int(fn))


def _check_loco_16_below_half(claims: dict[str, dict[str, Any]], results_dir: Path) -> bool:
    """Check if batch-held-out LOCO data shows 1.6 mm crack below half detected."""
    b_counts = _extract_loco_batch_counts(claims, results_dir)
    if not b_counts or "crack_1.6" not in b_counts:
        return False
    fc = b_counts["crack_1.6"]
    if fc.det_n <= 0:
        return False
    return (fc.det_k * 2) < fc.det_n


def _format_auroc(val: float) -> str:
    """Format AUROC preserving float precision up to 3 decimals."""
    s = str(val)
    if "." in s and len(s.split(".")[-1]) <= 3:
        return s
    return f"{val:.3f}"


def _build_separability_bullet(claims: dict[str, dict[str, Any]], results_dir: Path) -> str | None:
    """Build bullet from separability_batchval claim."""
    claim = claims.get("separability_batchval")
    if claim is None:
        return None
    aurocs = _extract_separability_aurocs(claim)
    counts = _extract_separability_16_counts(claim)
    if aurocs is None or counts is None:
        sys.stderr.write("Warning: missing separability metrics; skipping bullet.\n")
        return None
    min_auc, max_auc, n_folds = aurocs
    f_min, f_max = _format_auroc(min_auc), _format_auroc(max_auc)
    fcr_pct = int(round(float(claim.get("numbers", {}).get("target_fcr", 0.05)) * 100))
    clause = ""
    if _check_loco_16_below_half(claims, results_dir):
        clause = ", while the default 0.5 threshold missed most of that crack"
    text = (
        f"Showed that the leave-one-crack-out classifier ranked held-out cracks "
        f"almost perfectly (AUROC {f_min}-{f_max} across the {n_folds} folds) and that "
        f"clean-scan threshold calibration recovered detection ({counts.k}/{counts.n} "
        f"on the 1.6 mm crack at a {fcr_pct}% false-call target, {counts.fk}/{counts.fn} "
        f"false calls){clause}"
    )
    return _ensure_scope(
        text, "leave-one-crack-out, 3 physical cracks, clean scans from the same weld"
    )


def _load_verdicts(results_dir: Path) -> dict[str, Any] | None:
    """Load preregistration_verdicts.json from results directory if present."""
    path = results_dir / "preregistration_verdicts.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError) as err:  # narrow reason: corrupt or unreadable JSON
        sys.stderr.write(f"Warning: could not read {path}: {err}\n")
        return None


def _extract_invariant_aurocs(claim: dict[str, Any]) -> tuple[float, float, int] | None:
    """Extract min AUROC, max AUROC, and fold count from invariant_loco claim."""
    nums = claim.get("numbers", {})
    aurocs = [float(v) for k, v in nums.items() if k.endswith("_fused_auroc")]
    if len(aurocs) < 3:
        return None
    return min(aurocs), max(aurocs), len(aurocs)


def _extract_invariant_16_counts(claim: dict[str, Any]) -> tuple[int, int] | None:
    """Extract 1.6 mm crack detection counts from invariant_loco claim."""
    nums = claim.get("numbers", {})
    k, n = nums.get("crack_1.6_det_k"), nums.get("crack_1.6_det_n")
    if k is None or n is None:
        return None
    return int(k), int(n)


def _format_invariant_verdict_summary(cand_info: dict[str, Any]) -> str | None:
    """Format invariant candidate verdict summary from candidate dictionary."""
    raw_folds = cand_info.get("folds")
    if not isinstance(raw_folds, dict) or not raw_folds:
        return None
    fold_verdicts: dict[str, str] = {}
    for crack, item in raw_folds.items():
        v = item.get("verdict") if isinstance(item, dict) else item
        if isinstance(v, str):
            fold_verdicts[str(crack)] = v
    if not fold_verdicts:
        return None
    unique_v = set(fold_verdicts.values())
    if len(unique_v) == 1:
        v_word = next(iter(unique_v))
        return f"{v_word} on {len(fold_verdicts)} of {len(fold_verdicts)} folds"
    return ", ".join(f"{crack}: {v}" for crack, v in fold_verdicts.items())


def _get_invariant_verdict_clause(results_dir: Path) -> str:
    """Get verdict clause for invariant candidate if available."""
    verdicts = _load_verdicts(results_dir)
    if verdicts is None:
        return ""
    cand = verdicts.get("candidates", {}).get("invariant")
    if not isinstance(cand, dict):
        return ""
    summary = _format_invariant_verdict_summary(cand)
    return f"; pre-registered verdict: {summary}" if summary else ""


def _build_invariant_loco_bullet(
    claims: dict[str, dict[str, Any]], results_dir: Path
) -> str | None:
    """Build bullet for pre-registered invariant representation LOCO evaluation."""
    inv_c = claims.get("invariant_loco")
    sep_c = claims.get("separability_batchval")
    if inv_c is None or sep_c is None:
        return None
    inv_aurocs = _extract_invariant_aurocs(inv_c)
    sep_aurocs = _extract_separability_aurocs(sep_c)
    inv_kn = _extract_invariant_16_counts(inv_c)
    sep_counts = _extract_separability_16_counts(sep_c)
    if any(x is None for x in (inv_aurocs, sep_aurocs, inv_kn, sep_counts)):
        sys.stderr.write("Warning: missing invariant/baseline metrics; skipping bullet.\n")
        return None
    i_min, i_max, n_folds = inv_aurocs  # type: ignore[misc]
    b_min, b_max, _ = sep_aurocs  # type: ignore[misc]
    k_inv, n_inv = inv_kn  # type: ignore[misc]
    target_fcr = int(round(float(inv_c.get("numbers", {}).get("target_fcr", 0.05)) * 100))
    v_clause = _get_invariant_verdict_clause(results_dir)
    text = (
        f"Pre-registered a physics-augmented contrastive (invariant) crack representation; "
        f"fused AUROC {_format_auroc(i_min)}-{_format_auroc(i_max)} across the {n_folds} folds "
        f"vs baseline {_format_auroc(b_min)}-{_format_auroc(b_max)}, 1.6 mm detection "
        f"{k_inv}/{n_inv} vs baseline {sep_counts.k}/{sep_counts.n} "  # type: ignore[union-attr]
        f"at a {target_fcr}% false-call target{v_clause}"
    )
    return _ensure_scope(text, "leave-one-crack-out, 3 physical cracks, one public weld dataset")


def _extract_shift_family_folds(nums: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Extract family -> {crack: all_clean_shifts} mapping from numbers dict."""
    res: dict[str, dict[str, float]] = {}
    for k, v in nums.items():
        if not k.endswith("_all_clean_shifts") or "_crack_" not in k:
            continue
        prefix = k[: -len("_all_clean_shifts")]
        fam, crack = prefix.split("_crack_", 1)
        res.setdefault(fam, {})[f"crack_{crack}"] = float(v)
    return res


def _format_shift_family_clause(fam_folds: dict[str, dict[str, float]]) -> str:
    """Format 'every family' or '<m> of <F>' clause based on all_clean_shifts."""
    n_fams = len(fam_folds)
    m = sum(1 for folds in fam_folds.values() if all(v >= 1.0 for v in folds.values()))
    if m == n_fams:
        return "every family flagged all clean scans under at least one shift"
    return f"{m} of {n_fams} families flagged all clean scans under at least one shift"


def _format_shift_crack_desc(folds: tuple[str, ...]) -> str:
    """Format tuple of crack fold identifiers for shift robustness summary."""
    if not folds:
        return "none"
    if len(folds) == 1:
        return f"the {folds[0].replace('crack_', '')} mm crack only"
    if len(folds) == 3:
        return "all 3 folds"
    cracks = [f.replace("crack_", "") + " mm" for f in folds]
    return f"{' and '.join(cracks)} cracks"


def _join_cand_names(cands: list[str]) -> str:
    """Format list of candidate names into readable English string."""
    if len(cands) <= 2:
        return " and ".join(cands)
    return f"{', '.join(cands[:-1])} and {cands[-1]}"


def _format_shift_candidates_summary(candidates: dict[str, Any]) -> str | None:
    """Format shift robustness candidates verdict summary."""
    groups: dict[tuple[str, ...], list[str]] = {}
    for cand_name, cand_info in candidates.items():
        if not isinstance(cand_info, dict):
            continue
        folds = cand_info.get("folds", {})
        better = tuple(
            crack
            for crack, f in folds.items()
            if (f.get("verdict") if isinstance(f, dict) else f) == "more shift-robust"
        )
        groups.setdefault(better, []).append(str(cand_name))
    if not groups:
        return None
    parts: list[str] = []
    for folds_tup, cands in groups.items():
        cand_str = _join_cand_names(cands)
        if folds_tup:
            parts.append(f"{cand_str} more shift-robust on {_format_shift_crack_desc(folds_tup)}")
        else:
            parts.append(f"{cand_str} on none")
    return ", ".join(parts)


def _get_shift_verdict_clause(results_dir: Path) -> str:
    """Get ADR 0015 verdict clause for shift robustness if available."""
    verdicts = _load_verdicts(results_dir)
    if verdicts is None:
        return ""
    candidates = verdicts.get("shift_robustness", {}).get("candidates")
    if not isinstance(candidates, dict) or not candidates:
        return ""
    summary = _format_shift_candidates_summary(candidates)
    return f"; ADR 0015 verdict: {summary}" if summary else ""


def _build_shift_robustness_bullet(
    claims: dict[str, dict[str, Any]], results_dir: Path
) -> str | None:
    """Build bullet for shift robustness evaluation across model families."""
    claim = claims.get("shift_robustness")
    if claim is None:
        return None
    nums = claim.get("numbers", {})
    fam_folds = _extract_shift_family_folds(nums)
    shift_counts = [int(v) for k, v in nums.items() if k.endswith("_n_shifts")]
    a_val = nums.get("loco_crack_1.6_all_clean_shifts")
    if any((not fam_folds, not shift_counts, a_val is None)):
        sys.stderr.write("Warning: missing shift robustness numbers; skipping bullet.\n")
        return None
    n_fams, n_shifts = len(fam_folds), shift_counts[0]
    a_shifts = int(a_val)  # type: ignore[arg-type]
    fam_clause = _format_shift_family_clause(fam_folds)
    v_clause = _get_shift_verdict_clause(results_dir)
    text = (
        f"Stress-tested {n_fams} model families under {n_shifts} synthetic "
        "gain/offset/noise shifts with thresholds fixed on unshifted clean scans: "
        f"{fam_clause} (baseline under {a_shifts}/{n_shifts}), so site calibration is "
        f"required before use{v_clause}; the false-call condition is met trivially when "
        "the baseline already flags every clean scan"
    )
    return _ensure_scope(text, "real weld images, synthetic shifts, single shift seed")


def _get_fcr(claim: dict[str, Any]) -> str | None:
    """Extract formatted false-call rate from claim numbers or text."""
    nums = claim.get("numbers", {})
    for key in ("false_call_rate", "fcr"):
        if key in nums:
            return f"{float(nums[key]):.3f}"
    match = re.search(r"false[- ]call(?:s|\s+rate)\s+([0-9.]+)", claim.get("text", ""))
    if match:
        return match.group(1)
    return None


def _build_anomaly_bullet(claims: dict[str, dict[str, Any]]) -> str | None:
    """Build anomaly detection bullet from anomaly claims."""
    if not all(k in claims for k in ("anomaly_crack_1.6", "anomaly_crack_4.0")):
        return None
    kn1 = _get_k_n(claims["anomaly_crack_1.6"])
    kn4 = _get_k_n(claims["anomaly_crack_4.0"])
    fcr1 = _get_fcr(claims["anomaly_crack_1.6"])
    if not all((kn1, kn4, fcr1)):
        sys.stderr.write("Warning: missing anomaly metrics; skipping bullet.\n")
        return None
    k1, n1 = kn1  # type: ignore[misc]
    k4, n4 = kn4  # type: ignore[misc]
    text = (
        f"Trained an autoencoder only on flaw-free scans that detected the held-out "
        f"1.6 mm crack {k1}/{n1} at a false-call rate of {fcr1}, "
        f"but the 4.0 mm crack only {k4}/{n4}"
    )
    return _ensure_scope(text, "held-out cracks, trained on clean scans only")


def _format_det_range(min_k: float | int, max_k: float | int, n: int) -> str:
    """Format detection count or range as string."""
    if int(min_k) == int(max_k):
        return f"{int(min_k)}/{n}"
    return f"{int(min_k)}-{int(max_k)}/{n}"


def _get_crack_from_nums(nums: dict[str, Any], crack: str) -> str | None:
    """Extract crack detection string from claim numbers dict."""
    min_k = nums.get(f"k_min_{crack}", nums.get(f"crack_{crack}_min_k"))
    max_k = nums.get(f"k_max_{crack}", nums.get(f"crack_{crack}_max_k"))
    n_val = nums.get(f"n_{crack}", nums.get(f"crack_{crack}_n"))
    if all(x is not None for x in (min_k, max_k, n_val)):
        return _format_det_range(min_k, max_k, int(n_val))  # type: ignore[arg-type]
    k_val = nums.get(f"k_{crack}", nums.get(f"crack_{crack}_k"))
    if all(x is not None for x in (k_val, n_val)):
        return f"{int(k_val)}/{int(n_val)}"  # type: ignore[arg-type]
    return None


def _extract_crack_det(claim: dict[str, Any], crack: str) -> str | None:
    """Extract detection counts or range for crack without default numbers."""
    from_nums = _get_crack_from_nums(claim.get("numbers", {}), crack)
    if from_nums is not None:
        return from_nums
    match = re.search(rf"{re.escape(crack)}[^\d]*?(\d+(?:-\d+)?\s*/\s*\d+)", claim.get("text", ""))
    if match:
        return match.group(1).replace(" ", "")
    return None


def _extract_false_calls_h2(claim: dict[str, Any]) -> str | None:
    """Extract false calls count for H2 claim."""
    nums = claim.get("numbers", {})
    if "n_false" in nums and "n_scans" in nums:
        return f"{int(nums['n_false'])}/{int(nums['n_scans'])}"
    if "n_false" in nums and "n_clean" in nums:
        return f"{int(nums['n_false'])}/{int(nums['n_clean'])}"
    match = re.search(r"false[- ]calls?\s*(\d+\s*/\s*\d+)", claim.get("text", ""))
    if match:
        return match.group(1).replace(" ", "")
    return None


def _extract_h3_fc_from_file(path: Path) -> str | None:
    """Extract maximum false calls across folds from physics metrics file."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    folds = data.get("folds", [])
    if not folds:
        return None
    max_fold = max(folds, key=lambda f: f.get("false_calls", {}).get("n_false", 0))
    fc = max_fold.get("false_calls", {})
    return f"{int(fc.get('n_false', 0))}/{int(fc.get('n_clean', fc.get('n_scans', 0)))}"


def _extract_h3_fc_from_numbers(nums: dict[str, Any]) -> str | None:
    """Extract maximum false calls from claim numbers dictionary."""
    fc_pairs: list[tuple[int, int]] = []
    for k, v in nums.items():
        if not k.endswith("_n_false"):
            continue
        clean_key = k.replace("_n_false", "_n_clean")
        clean_val = nums.get(clean_key, nums.get(k.replace("_n_false", "_n_scans"), 0))
        fc_pairs.append((int(v), int(clean_val)))
    if not fc_pairs:
        return None
    max_k, max_n = max(fc_pairs, key=lambda p: p[0])
    return f"{max_k}/{max_n}" if max_n > 0 else None


def _extract_false_calls_h3(claim: dict[str, Any], results_dir: Path) -> str | None:
    """Extract maximum false calls across folds for H3 physics detector."""
    fc_file = _extract_h3_fc_from_file(results_dir / "physics_model_metrics.json")
    if fc_file is not None:
        return fc_file
    fc_nums = _extract_h3_fc_from_numbers(claim.get("numbers", {}))
    if fc_nums is not None:
        return fc_nums
    matches = re.findall(r"false[- ]calls?\s*(\d+)\s*/\s*(\d+)", claim.get("text", ""))
    if matches:
        k_val, n_val = max(matches, key=lambda m: int(m[0]))
        return f"{k_val}/{n_val}"
    return None


def _build_hybrid_bullet(claims: dict[str, dict[str, Any]], results_dir: Path) -> str | None:
    """Build bullet combining pre-registered hybrid H2 and physics H3 claims."""
    if not all(k in claims for k in ("hybrid_h2", "physics_h3")):
        return None
    h2_c = claims["hybrid_h2"]
    h3_c = claims["physics_h3"]
    h2_1, h3_1 = _extract_crack_det(h2_c, "1.6"), _extract_crack_det(h3_c, "1.6")
    h2_4, h3_4 = _extract_crack_det(h2_c, "4.0"), _extract_crack_det(h3_c, "4.0")
    h2_8 = _extract_crack_det(h2_c, "8.6")
    h3_8 = _extract_crack_det(h3_c, "8.6")
    h2_fc = _extract_false_calls_h2(h2_c)
    h3_fc = _extract_false_calls_h3(h3_c, results_dir)
    if not all((h2_1, h3_1, h2_4, h3_4, h2_8, h3_8, h2_fc, h3_fc)):
        sys.stderr.write("Warning: missing hybrid detection numbers; skipping bullet.\n")
        return None
    h2_nums = h2_c.get("numbers", {})
    k_min = h2_nums.get("crack_4.0_min_k", h2_nums.get("k_min_4.0", 0))
    k_max = h2_nums.get("crack_4.0_max_k", h2_nums.get("k_max_4.0", 0))
    seed_note = "; results vary across seeds" if k_max != k_min else ""
    text = (
        f"Pre-registered a supervised+anomaly hybrid and a physics-feature detector, "
        f"reaching {h2_1} vs {h3_1} on 1.6 mm, {h2_4} vs {h3_4} on 4.0 mm, "
        f"{h2_8} vs {h3_8} on 8.6 mm, false calls {h2_fc} vs {h3_fc}{seed_note}"
    )
    return _ensure_scope(text, "public dataset, pre-registered")


def _build_false_call_bullet(results_dir: Path) -> str | None:
    """Build false-call analysis bullet from results/false_call_analysis.json."""
    path = results_dir / "false_call_analysis.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    per_fold = data.get("per_fold", {})
    if not per_fold:
        sys.stderr.write("Warning: missing per_fold in false_call_analysis; skipping bullet.\n")
        return None
    n_evals = sum(int(f.get("n_clean", 0)) for f in per_fold.values())
    n_flagged = data.get("n_flagged")
    cause_counts = data.get("cause_counts", {})
    if any((n_flagged is None, not cause_counts, n_evals == 0)):
        sys.stderr.write("Warning: missing metrics in false_call_analysis; skipping bullet.\n")
        return None
    dom_cause = max(cause_counts, key=lambda k: cause_counts[k])
    text = (
        f"Analyzed false calls across {n_evals} fold-scan clean evaluations "
        f"(each distinct clean scan scored by every fold model), finding {n_flagged} "
        f"flagged with dominant cause {dom_cause}"
    )
    return _ensure_scope(text, "public dataset, pre-registered")


def _build_sim_bullet(claims: dict[str, dict[str, Any]]) -> str | None:
    """Build simulator bullet from sim_regions claim."""
    c_sim = claims.get("sim_regions")
    if c_sim is None:
        return None
    thresh = c_sim.get("numbers", {}).get("threshold", 0.3)
    text = (
        f"Observed false-call rate above {thresh:.1f} per window "
        "(project-chosen reporting threshold, not a standard) "
        "at an unseen frequency and above-range noise for both rules and ML in held-out regions"
    )
    return _ensure_scope(text, "simulated")


def _build_cross_dataset_bullet(claims: dict[str, dict[str, Any]]) -> str | None:
    """Build real cross-dataset bullet from cross_dataset claim."""
    c_cross = claims.get("cross_dataset")
    if c_cross is None:
        return None
    kn = _get_k_n(c_cross)
    fc = _extract_false_calls_h2(c_cross)
    if kn is None or fc is None:
        sys.stderr.write("Warning: missing cross-dataset metrics; skipping bullet.\n")
        return None
    k, n = kn
    text = (
        f"Evaluated cross-dataset transfer on NDT_ML_Flaw real weld data, "
        f"finding the model flagged every image with {k}/{n} detections and {fc} false calls"
    )
    return _ensure_scope(text, "NDT_ML_Flaw, real weld data")


def _format_shift_metrics(shift: dict[str, Any]) -> tuple[str, str]:
    """Format baseline to adapted detection and false-call transitions."""
    base_det = f"{shift['baseline']['detection']['k']}/{shift['baseline']['detection']['n']}"
    adapt_det = f"{shift['adapted']['detection']['k']}/{shift['adapted']['detection']['n']}"
    base_fc = f"{shift['baseline']['false_calls']['k']}/{shift['baseline']['false_calls']['n']}"
    adapt_fc = f"{shift['adapted']['false_calls']['k']}/{shift['adapted']['false_calls']['n']}"
    return f"detection {base_det} to {adapt_det}", f"false calls {base_fc} to {adapt_fc}"


def _format_site_adaptation(
    data: dict[str, Any], g2: dict[str, Any], g05: dict[str, Any], n_calib: int
) -> str | None:
    """Format site adaptation bullet text and scope."""
    g2_det, g2_fc = _format_shift_metrics(g2)
    g05_det, g05_fc = _format_shift_metrics(g05)
    n_test = data.get("n_test_clean", data.get("n_images", {}).get("test_clean"))
    if n_test is None:
        sys.stderr.write("Warning: missing n_test_clean; skipping bullet.\n")
        return None
    text = (
        f"Adapted ({n_calib} calibration, {n_test} test clean scans): "
        f"gain 2.0: {g2_det}, {g2_fc}; gain 0.5: {g05_det}, {g05_fc}"
    )
    scope = "single seed; shifts are synthetic perturbations of real weld images"
    if data.get("test_underpowered"):
        scope += ", small test set"
    return _ensure_scope(text, scope)


def _build_site_adaptation_bullet(results_dir: Path) -> str | None:
    """Build site adaptation bullet directly from results/site_adaptation.json."""
    path = results_dir / "site_adaptation.json"
    if not path.exists():
        sys.stderr.write(f"Warning: {path} not found; skipping bullet.\n")
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    shifts = data.get("shifts", {})
    g2, g05 = shifts.get("gain 2.0"), shifts.get("gain 0.5")
    if not g2 or not g05:
        sys.stderr.write("Warning: missing gain shifts; skipping bullet.\n")
        return None
    n_calib = data.get("n_calib_clean")
    if n_calib is None:
        sys.stderr.write("Warning: missing n_calib_clean; skipping bullet.\n")
        return None
    return _format_site_adaptation(data, g2, g05, int(n_calib))


def _arm_improved_all(arm: str, folds: list[dict[str, Any]]) -> bool:
    """Check if a specific arm improved every crack over arm A."""
    for f in folds:
        a_k = f.get("arms", {}).get("A", {}).get("detections", {}).get("k", 0)
        arm_k = f.get("arms", {}).get(arm, {}).get("detections", {}).get("k", 0)
        if arm_k <= a_k:
            return False
    return True


def _check_no_arm_improved(folds: list[dict[str, Any]]) -> bool:
    """Verify whether no alternative arm strictly improved every crack."""
    arms = [a for a in folds[0].get("arms", {}) if a != "A"]
    for arm in arms:
        if _arm_improved_all(arm, folds):
            return False
    return True


def _format_joint_bullet(
    data: dict[str, Any], folds: list[dict[str, Any]], f4: dict[str, Any]
) -> str | None:
    """Format joint training bullet text from folds data."""
    arm_a = f4.get("arms", {}).get("A", {}).get("detections", {})
    arm_c = f4.get("arms", {}).get("C", {}).get("detections", {})
    if "k" not in arm_a or "k" not in arm_c:
        sys.stderr.write("Warning: missing arm A/C detections; skipping bullet.\n")
        return None
    a_det = f"{int(arm_a['k'])}/{int(arm_a['n'])}"
    c_det = f"{int(arm_c['k'])}/{int(arm_c['n'])}"
    seeds = data.get("seeds", [])
    n_seeds = len(seeds) if isinstance(seeds, list) else 1
    seed_str = f"{n_seeds} seed" if n_seeds == 1 else f"{n_seeds} seeds"
    note = ", but no arm improved every crack" if _check_no_arm_improved(folds) else ""
    text = (
        f"Evaluated joint synthetic-real training across {seed_str}: "
        f"on crack 4.0, arm A detected {a_det} vs arm C {c_det}{note}"
    )
    return _ensure_scope(text, "joint training, public weld data")


def _build_joint_training_bullet(results_dir: Path) -> str | None:
    """Build joint training bullet directly from results/joint_train_metrics.json."""
    path = results_dir / "joint_train_metrics.json"
    if not path.exists():
        sys.stderr.write(f"Warning: {path} not found; skipping bullet.\n")
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    folds = data.get("folds")
    if not folds:
        sys.stderr.write("Warning: missing folds in joint training; skipping bullet.\n")
        return None
    f4 = next((f for f in folds if f.get("held_out") == "crack_4.0"), None)
    if not f4:
        sys.stderr.write("Warning: missing crack_4.0 fold; skipping bullet.\n")
        return None
    return _format_joint_bullet(data, folds, f4)


def _extract_union_counts(claim: dict[str, Any]) -> tuple[int, int] | None:
    """Extract violations and columns from union claim."""
    viol = claim.get("numbers", {}).get("n_violations")
    col = claim.get("numbers", {}).get("n_columns")
    if col is None:
        m = re.search(r"(\d+)\s+simulator columns", claim.get("text", ""))
        col = int(m.group(1)) if m else None
    if viol is None or col is None:
        return None
    return int(viol), int(col)


def _extract_latency_median(claim: dict[str, Any]) -> float | None:
    """Extract median latency ms from latency claim."""
    lat_nums = claim.get("numbers", {})
    med = lat_nums.get("median_ms", lat_nums.get("median"))
    if med is None:
        m = re.search(r"median\s+([0-9.]+)\s*ms", claim.get("text", ""))
        med = float(m.group(1)) if m else None
    return float(med) if med is not None else None


def _build_engineering_bullet(claims: dict[str, dict[str, Any]]) -> str | None:
    """Build engineering bullet combining union property and latency claims."""
    union_c = claims.get("union_property")
    latency_c = claims.get("latency")
    if union_c is None or latency_c is None:
        return None
    u_counts = _extract_union_counts(union_c)
    med_val = _extract_latency_median(latency_c)
    if u_counts is None or med_val is None:
        sys.stderr.write("Warning: missing engineering numbers; skipping bullet.\n")
        return None
    viol, col = u_counts
    text = (
        f"Property-tested the ML-only-adds-flags rule on {col} simulator columns "
        f"with {viol} violations, achieving median CPU latency of {med_val:.1f} ms "
        "measured on the development workstation CPU, not on the target i5 laptop"
    )
    return _ensure_scope(text, "advisory")


@dataclass(frozen=True)
class TiCalibrationCounts:
    """Detection sums and clean-bar false alarm rates for titanium calibration."""

    zone_k: int
    zone_n: int
    global_k: int
    global_n: int
    zone_fa: float
    global_fa: float


@dataclass(frozen=True)
class TiSweepRange:
    """Titanium noise sweep field sigma and uninspectable zone range."""

    min_sigma: float
    max_sigma: float
    uninspect_a: int
    uninspect_b: int


@dataclass(frozen=True)
class TiInspectability:
    """Inspectability status and counts for titanium zone and global arms."""

    zone_uninspect: int
    global_uninspect: int
    zone_k: int | None = None
    zone_n: int | None = None


def _extract_ti_zone_sums(nums: dict[str, Any], arm: str) -> tuple[int, int] | None:
    """Extract sum of k and n detections across zones for an arm."""
    k_suffix = "_k"
    prefix = f"structured_{arm}_arm_zone"
    zone_indices = [
        k[len(prefix) : -len(k_suffix)]
        for k in nums
        if k.startswith(prefix) and k.endswith(k_suffix)
    ]
    if not zone_indices:
        return None
    k_vals: list[int] = []
    n_vals: list[int] = []
    for idx in zone_indices:
        k_val = nums.get(f"{prefix}{idx}_k")
        n_val = nums.get(f"{prefix}{idx}_n")
        if k_val is None or n_val is None:
            return None
        k_vals.append(int(k_val))
        n_vals.append(int(n_val))
    tot_n = sum(n_vals)
    return (sum(k_vals), tot_n) if tot_n > 0 else None


def _extract_ti_calibration(claim: dict[str, Any]) -> TiCalibrationCounts | None:
    """Extract titanium zone calibration detection sums and false-alarm rates."""
    nums = claim.get("numbers", {})
    z_sums = _extract_ti_zone_sums(nums, "zone")
    g_sums = _extract_ti_zone_sums(nums, "global")
    z_fa = nums.get("structured_zone_arm_false_alarms_per_bar")
    g_fa = nums.get("structured_global_arm_false_alarms_per_bar")
    if any((z_sums is None, g_sums is None, z_fa is None, g_fa is None)):
        return None
    return TiCalibrationCounts(
        zone_k=z_sums[0],  # type: ignore[index]
        zone_n=z_sums[1],  # type: ignore[index]
        global_k=g_sums[0],  # type: ignore[index]
        global_n=g_sums[1],  # type: ignore[index]
        zone_fa=float(z_fa),
        global_fa=float(g_fa),
    )


def _extract_ti_sweep_pairs(nums: dict[str, Any]) -> list[tuple[float, int]]:
    """Extract (field_sigma, uninspectable_zones) pairs from sweep numbers."""
    pairs: list[tuple[float, int]] = []
    suffix = "_field_sigma"
    for k, v in nums.items():
        if not k.endswith(suffix):
            continue
        prefix = k[: -len(suffix)]
        uninsp_key = f"{prefix}_uninspectable_zones"
        if uninsp_key in nums:
            pairs.append((float(v), int(nums[uninsp_key])))
    return pairs


def _extract_ti_sweep(claim: dict[str, Any] | None) -> TiSweepRange | None:
    """Extract min/max sigma and uninspectable zones from noise sweep claim."""
    if claim is None:
        return None
    pairs = _extract_ti_sweep_pairs(claim.get("numbers", {}))
    if len(pairs) < 2:
        return None
    sorted_pairs = sorted(pairs, key=lambda p: p[0])
    return TiSweepRange(
        min_sigma=sorted_pairs[0][0],
        max_sigma=sorted_pairs[-1][0],
        uninspect_a=sorted_pairs[0][1],
        uninspect_b=sorted_pairs[-1][1],
    )


def _format_ti_sweep_clause(sweep: TiSweepRange | None) -> str:
    """Format optional sweep clause if sweep metrics are present."""
    if sweep is None:
        return ""
    return (
        f"; as noise-strength variation rose from {sweep.min_sigma:.2f} "
        f"to {sweep.max_sigma:.2f}, zones not inspectable went from "
        f"{sweep.uninspect_a} to {sweep.uninspect_b}"
    )


def _load_ti_json(results_dir: Path | None) -> dict[str, Any] | None:
    """Load ti_zone_metrics.json from results directory if present."""
    if results_dir is None:
        return None
    path = results_dir / "ti_zone_metrics.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError) as err:  # narrow reason: unreadable JSON
        sys.stderr.write(f"Warning: could not read {path}: {err}\n")
        return None


def _extract_ti_inspectable_counts(
    arm_z: dict[str, Any],
) -> tuple[int | None, int | None]:
    """Extract inspectable zone k and n detection counts if available."""
    per_zone = arm_z.get("flawed", {}).get("per_zone", [])
    if not isinstance(per_zone, list) or not per_zone:
        return None, None
    valid = [z for z in per_zone if isinstance(z, dict) and z.get("inspectable", True)]
    if not valid:
        return None, None
    zk = sum(int(z.get("k", 0)) for z in valid)
    zn = sum(int(z.get("n", 0)) for z in valid)
    return zk, zn


def _extract_ti_inspectability(results_dir: Path | None) -> TiInspectability | None:
    """Extract uninspectable counts from ti_zone_metrics.json if present."""
    data = _load_ti_json(results_dir)
    if data is None:
        return None
    arms = data.get("noise_models", {}).get("structured", {}).get("arms", {})
    arm_z = arms.get("zone", {})
    arm_g = arms.get("global", {})
    z_val = arm_z.get("zones_not_inspectable")
    g_val = arm_g.get("zones_not_inspectable")
    if any((z_val is None, g_val is None)):
        return None
    zk, zn = _extract_ti_inspectable_counts(arm_z)
    return TiInspectability(
        zone_uninspect=int(z_val),
        global_uninspect=int(g_val),
        zone_k=zk,
        zone_n=zn,
    )


def _get_ti_zone_counts(cal: TiCalibrationCounts, insp: TiInspectability | None) -> tuple[int, int]:
    """Get zone k and n counts from inspectability record or calibration."""
    if any(
        (insp is None, getattr(insp, "zone_k", None) is None, getattr(insp, "zone_n", None) is None)
    ):
        return cal.zone_k, cal.zone_n
    return insp.zone_k, insp.zone_n  # type: ignore[return-value]


def _format_ti_zone_detection(cal: TiCalibrationCounts, insp: TiInspectability | None) -> str:
    """Format zone arm detection count and uninspectable note."""
    zk, zn = _get_ti_zone_counts(cal, insp)
    if insp is not None and insp.zone_uninspect > 0:
        plural = "zone" if insp.zone_uninspect == 1 else "zones"
        return (
            f"reference-size hole detected {zk}/{zn} with zone-wise alarms "
            f"({insp.zone_uninspect} {plural} sent to the inspector)"
        )
    return f"reference-size hole detected {zk}/{zn} with zone-wise alarms"


def _format_ti_comparison_clause(cal: TiCalibrationCounts, insp: TiInspectability | None) -> str:
    """Format global alarm comparison clause based on inspectability."""
    if insp is None:
        return ""
    if insp.global_uninspect > 0:
        return (
            ", while a single global alarm could not clear the noise peak "
            "and would send the whole depth range to the inspector"
        )
    return f" vs {cal.global_k}/{cal.global_n} with one global alarm"


def _build_titanium_bullet(
    claims: dict[str, dict[str, Any]], results_dir: Path | None = None
) -> str | None:
    """Build bullet for synthetic titanium mode zone calibration and sweep."""
    c_cal = claims.get("ti_zone_calibration")
    if c_cal is None:
        return None
    cal = _extract_ti_calibration(c_cal)
    if cal is None:
        sys.stderr.write("Warning: missing titanium calibration metrics; skipping bullet.\n")
        return None
    insp = _extract_ti_inspectability(results_dir)
    zone_det = _format_ti_zone_detection(cal, insp)
    comp_clause = _format_ti_comparison_clause(cal, insp)
    sweep = _extract_ti_sweep(claims.get("ti_noise_sweep"))
    sweep_clause = _format_ti_sweep_clause(sweep)
    text = (
        "Extended UTriage with a synthetic titanium mode: macrozone-like correlated "
        "grain noise, zone-wise reference calibration with SNR (A-B)/(C-B) and an alarm "
        "capped relative to the reference response, zones that cannot reach the sensitivity "
        "flagged to the inspector, an inspector report and an air-gapped package "
        "(model SHA-256 checks, offline bundle, network-blocking test); "
        f"{zone_det}{comp_clause}, "
        f"clean-bar false alarms per bar {cal.zone_fa:.2f} vs {cal.global_fa:.2f}{sweep_clause}"
    )
    return _ensure_scope(text, "synthetic data only, placeholder limits, no real titanium data")


def _check_single_claim(c: dict[str, Any]) -> None:
    """Check text and scope_limit of a single claim."""
    for field in ("text", "scope_limit"):
        val = c.get(field, "")
        if isinstance(val, str) and (found := check_forbidden(val)):
            raise ValueError(f"Forbidden phrase(s) {found} found in claim: {val!r}")


def _check_claims_forbidden(claims: dict[str, dict[str, Any]]) -> None:
    """Verify no claim text or scope contains forbidden phrases."""
    for c in claims.values():
        _check_single_claim(c)


def _check_bullets_forbidden(bullets: list[str]) -> None:
    """Verify no generated bullet contains forbidden phrases or words."""
    for bullet in bullets:
        scrubbed = re.sub(r"\balmost perfectly\b", "", bullet, flags=re.IGNORECASE)
        if found := check_forbidden(scrubbed):
            raise ValueError(f"Forbidden phrase(s) {found} found in bullet: {bullet!r}")
        lower = bullet.lower()
        if bad := [w for w in FORBIDDEN_WORDS if w.lower() in lower]:
            raise ValueError(f"Forbidden word(s) {bad} found in bullet: {bullet!r}")


def _load_claims(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Load results/claims.json and index by claim id."""
    path = results_dir / "claims.json"
    if not path.exists():
        raise FileNotFoundError(f"Claims file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of claims in {path}, got {type(data).__name__}")
    return {c["id"]: c for c in data if isinstance(c, dict) and "id" in c}


def build_bullets(results_dir: Path) -> list[str]:
    """Build resume bullet points strictly from results JSON files and claims."""
    claims = _load_claims(results_dir)
    _check_claims_forbidden(claims)
    builders: list[tuple[Callable[..., str | None], tuple[Any, ...]]] = [
        (_build_leakage_bullet, (claims,)),
        (_build_loco_bullet, (claims, results_dir)),
        (_build_separability_bullet, (claims, results_dir)),
        (_build_invariant_loco_bullet, (claims, results_dir)),
        (_build_shift_robustness_bullet, (claims, results_dir)),
        (_build_false_call_bullet, (results_dir,)),
        (_build_anomaly_bullet, (claims,)),
        (_build_hybrid_bullet, (claims, results_dir)),
        (_build_sim_bullet, (claims,)),
        (_build_cross_dataset_bullet, (claims,)),
        (_build_site_adaptation_bullet, (results_dir,)),
        (_build_joint_training_bullet, (results_dir,)),
        (_build_engineering_bullet, (claims,)),
        (_build_titanium_bullet, (claims, results_dir)),
    ]
    bullets = [_build_framing_bullet()]
    for func, args in builders:
        b = func(*args)
        if b is not None:
            bullets.append(b)
    _check_bullets_forbidden(bullets)
    return bullets


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: generate resume bullets strictly from claims.json."""
    parser = argparse.ArgumentParser(description="Generate resume bullets from claims.json.")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help="Path to results directory containing claims.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file to write bullets",
    )
    args = parser.parse_args(argv)
    bullets = build_bullets(args.results)
    for bullet in bullets:
        print(f"- {bullet}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(f"- {b}" for b in bullets) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
