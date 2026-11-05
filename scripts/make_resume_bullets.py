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
        if found := check_forbidden(bullet):
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
        (_build_false_call_bullet, (results_dir,)),
        (_build_anomaly_bullet, (claims,)),
        (_build_hybrid_bullet, (claims, results_dir)),
        (_build_sim_bullet, (claims,)),
        (_build_cross_dataset_bullet, (claims,)),
        (_build_site_adaptation_bullet, (results_dir,)),
        (_build_joint_training_bullet, (results_dir,)),
        (_build_engineering_bullet, (claims,)),
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
