"""Pre-registration success rule evaluation.

Applies the pre-registered ADR decision rules to experiment metrics
to determine candidate improvements over the baseline.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RULE_TEXT: str = (
    "verdict 'better' on a fold iff cand_auroc >= base_auroc + 0.05 "
    "AND cand_det_rate >= base_det_rate; otherwise 'no improvement'. "
    "NaN or missing values -> 'not evaluable'."
)
LITE_RULE_TEXT: str = (
    f"{RULE_TEXT} Lite additionally 'lighter' iff "
    "latency.litenet total median < latency.mlndtnet median."
)
EVIDENCE_NOTE: str = "evidence limit: 3 physical cracks in one public weld dataset"
ADR_ROBUST: str = "docs/decisions/0012-robust-loco-preregistration.md"
ADR_LITE: str = "docs/decisions/0013-lite-hybrid-preregistration.md"
ADR_INVARIANT: str = "docs/decisions/0014-invariant-representation-preregistration.md"
ADR_SHIFT: str = "docs/decisions/0015-shift-robustness-preregistration.md"
SHIFT_NOTE: str = (
    "shifts are synthetic gain/offset/noise perturbations of real ML-NDT images; "
    "thresholds fixed on unshifted clean scans"
)
SHIFT_RULE_TEXT: str = (
    "candidate is 'more shift-robust' on a fold iff cand_worst_det >= "
    "base_worst_det + 0.10 AND cand_worst_fcr <= base_worst_fcr + 0.01; "
    "otherwise 'no improvement'; NaN/missing -> 'not evaluable'."
)


@dataclass(frozen=True)
class JudgeConfig:
    """Configuration paths for pre-registration evaluation."""

    results_dir: Path = Path("results")
    out_path: Path = Path("results/preregistration_verdicts.json")


@dataclass(frozen=True)
class FoldMetrics:
    """Metrics for a single fold evaluation."""

    auroc: float
    det_k: int
    det_n: int
    det_rate: float


@dataclass(frozen=True)
class FoldVerdict:
    """Comparison verdict metrics between candidate and baseline fold."""

    base_auroc: float | None
    cand_auroc: float | None
    delta_auroc: float | None
    base_det_k: int | None
    base_det_n: int | None
    cand_det_k: int | None
    cand_det_n: int | None
    verdict: str
    lighter: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert fold verdict into serialization dictionary."""
        record: dict[str, Any] = {
            "base_auroc": self.base_auroc,
            "cand_auroc": self.cand_auroc,
            "delta_auroc": self.delta_auroc,
            "base_det_k": self.base_det_k,
            "base_det_n": self.base_det_n,
            "cand_det_k": self.cand_det_k,
            "cand_det_n": self.cand_det_n,
            "verdict": self.verdict,
        }
        if self.lighter is not None:
            record["lighter"] = self.lighter
        return record


@dataclass(frozen=True)
class ShiftFoldMetrics:
    """Worst-case metrics for a single fold under shifts."""

    worst_det: float
    worst_fcr: float


@dataclass(frozen=True)
class ShiftFoldVerdict:
    """Comparison verdict for shift robustness between candidate and baseline fold."""

    base_worst_det: float | None
    cand_worst_det: float | None
    base_worst_fcr: float | None
    cand_worst_fcr: float | None
    delta_det: float | None
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        """Convert shift fold verdict into serialization dictionary."""
        return {
            "base_worst_det": self.base_worst_det,
            "cand_worst_det": self.cand_worst_det,
            "base_worst_fcr": self.base_worst_fcr,
            "cand_worst_fcr": self.cand_worst_fcr,
            "delta_det": self.delta_det,
            "verdict": self.verdict,
        }


def _is_invalid_metrics(cand: FoldMetrics, base: FoldMetrics) -> bool:
    """Check if any metric is NaN or non-positive sample count."""
    checks = [
        math.isnan(cand.auroc),
        math.isnan(base.auroc),
        math.isnan(cand.det_rate),
        math.isnan(base.det_rate),
        cand.det_k < 0,
        cand.det_n <= 0,
        base.det_k < 0,
        base.det_n <= 0,
    ]
    return any(checks)


def _judge_fold_metrics(cand: FoldMetrics, base: FoldMetrics) -> FoldVerdict:
    """Compute delta and judge improvement when metrics are present."""
    if _is_invalid_metrics(cand, base):
        return FoldVerdict(
            base_auroc=base.auroc,
            cand_auroc=cand.auroc,
            delta_auroc=None,
            base_det_k=base.det_k,
            base_det_n=base.det_n,
            cand_det_k=cand.det_k,
            cand_det_n=cand.det_n,
            verdict="not evaluable",
        )
    delta = round(cand.auroc - base.auroc, 6)
    is_auc_better = delta >= 0.05
    is_det_better = round(cand.det_rate - base.det_rate, 6) >= 0.0
    verdict = "better" if (is_auc_better and is_det_better) else "no improvement"
    return FoldVerdict(
        base_auroc=base.auroc,
        cand_auroc=cand.auroc,
        delta_auroc=delta,
        base_det_k=base.det_k,
        base_det_n=base.det_n,
        cand_det_k=cand.det_k,
        cand_det_n=cand.det_n,
        verdict=verdict,
    )


def _extract_base_metrics(base: FoldMetrics | None) -> tuple[float | None, int | None, int | None]:
    """Extract auroc, det_k, and det_n from fold metrics if present."""
    if base is None:
        return None, None, None
    return base.auroc, base.det_k, base.det_n


def _make_missing_verdict(
    cand: FoldMetrics | None,
    base: FoldMetrics | None,
) -> FoldVerdict:
    """Construct a not evaluable verdict when baseline or candidate is missing."""
    b_auc, b_k, b_n = _extract_base_metrics(base)
    c_auc, c_k, c_n = _extract_base_metrics(cand)
    return FoldVerdict(b_auc, c_auc, None, b_k, b_n, c_k, c_n, "not evaluable")


def compute_verdict(
    cand: FoldMetrics | None,
    base: FoldMetrics | None,
) -> FoldVerdict:
    """Apply ADR pre-registered success rule comparing candidate against baseline."""
    if cand is not None and base is not None:
        return _judge_fold_metrics(cand, base)
    return _make_missing_verdict(cand, base)


def _is_invalid_shift_metrics(cand: ShiftFoldMetrics, base: ShiftFoldMetrics) -> bool:
    """Check if any shift metric is NaN."""
    checks = [
        math.isnan(cand.worst_det),
        math.isnan(base.worst_det),
        math.isnan(cand.worst_fcr),
        math.isnan(base.worst_fcr),
    ]
    return any(checks)


def _judge_shift_fold_metrics(
    cand: ShiftFoldMetrics,
    base: ShiftFoldMetrics,
) -> ShiftFoldVerdict:
    """Judge shift improvement when both candidate and baseline metrics exist."""
    if _is_invalid_shift_metrics(cand, base):
        return ShiftFoldVerdict(
            base_worst_det=base.worst_det,
            cand_worst_det=cand.worst_det,
            base_worst_fcr=base.worst_fcr,
            cand_worst_fcr=cand.worst_fcr,
            delta_det=None,
            verdict="not evaluable",
        )
    eps = 1e-12
    delta_det = round(cand.worst_det - base.worst_det, 6)
    is_det_better = cand.worst_det >= base.worst_det + 0.10 - eps
    is_fcr_better = cand.worst_fcr <= base.worst_fcr + 0.01 + eps
    verdict = "more shift-robust" if (is_det_better and is_fcr_better) else "no improvement"
    return ShiftFoldVerdict(
        base_worst_det=base.worst_det,
        cand_worst_det=cand.worst_det,
        base_worst_fcr=base.worst_fcr,
        cand_worst_fcr=cand.worst_fcr,
        delta_det=delta_det,
        verdict=verdict,
    )


def _make_missing_shift_verdict(
    cand: ShiftFoldMetrics | None,
    base: ShiftFoldMetrics | None,
) -> ShiftFoldVerdict:
    """Construct a not evaluable shift verdict when a metric is missing."""
    b_det = base.worst_det if base else None
    c_det = cand.worst_det if cand else None
    b_fcr = base.worst_fcr if base else None
    c_fcr = cand.worst_fcr if cand else None
    return ShiftFoldVerdict(b_det, c_det, b_fcr, c_fcr, None, "not evaluable")


def compute_shift_verdict(
    cand: ShiftFoldMetrics | None,
    base: ShiftFoldMetrics | None,
) -> ShiftFoldVerdict:
    """Apply ADR 0015 pre-registered shift-robustness success rule."""
    if cand is not None and base is not None:
        return _judge_shift_fold_metrics(cand, base)
    return _make_missing_shift_verdict(cand, base)


def load_baseline_metrics(path: Path) -> dict[str, FoldMetrics]:
    """Load baseline metrics from separability JSON under protocols.batchval.folds."""
    if not path.exists():
        raise FileNotFoundError(f"Baseline file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    folds_raw = data.get("protocols", {}).get("batchval", {}).get("folds", [])
    result: dict[str, FoldMetrics] = {}
    for item in folds_raw:
        held = str(item.get("held_out", ""))
        auc = float(item.get("auroc", float("nan")))
        det = item.get("operating_points", {}).get("0.05", {}).get("detection", {})
        k = int(det.get("k", -1))
        n = int(det.get("n", -1))
        rate = float(k / n) if n > 0 else float("nan")
        result[held] = FoldMetrics(auroc=auc, det_k=k, det_n=n, det_rate=rate)
    return result


def parse_robust_folds(data: dict[str, Any]) -> dict[str, FoldMetrics]:
    """Extract per-fold metrics from robust LOCO payload."""
    folds_raw = data.get("folds", [])
    result: dict[str, FoldMetrics] = {}
    for item in folds_raw:
        held = str(item.get("held_out", ""))
        ens = item.get("ensemble_metrics", {})
        auc = float(ens.get("auroc", float("nan")))
        det = ens.get("fcr_0.05", {}).get("detections", {})
        k = int(det.get("n_hit", det.get("k", -1)))
        n = int(det.get("n_flawed", det.get("n", -1)))
        rate = float(k / n) if n > 0 else float("nan")
        result[held] = FoldMetrics(auroc=auc, det_k=k, det_n=n, det_rate=rate)
    return result


def _normalize_folds_dict(raw: Any) -> list[dict[str, Any]]:
    """Convert folds data structure into a flat list of fold dicts."""
    if isinstance(raw, dict):
        return list(raw.values())
    if isinstance(raw, list):
        return raw
    return []


def parse_lite_folds(data: dict[str, Any]) -> dict[str, FoldMetrics]:
    """Extract per-fold metrics from lite hybrid LOCO payload."""
    folds_list = _normalize_folds_dict(data.get("folds"))
    result: dict[str, FoldMetrics] = {}
    for item in folds_list:
        held = str(item.get("held_out", ""))
        auc = float(item.get("threshold_free", {}).get("fused", {}).get("auroc", float("nan")))
        ops = item.get("operating_points", {}).get("fcr_0.05", {})
        det = ops.get("hybrid", {}).get("detection", {})
        k = int(det.get("k", -1))
        n = int(det.get("n", -1))
        rate = float(k / n) if n > 0 else float("nan")
        result[held] = FoldMetrics(auroc=auc, det_k=k, det_n=n, det_rate=rate)
    return result


def _extract_median_val(val: Any) -> float | None:
    """Extract numeric median from dict or numeric value."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict):
        # narrow reason: check median_ms before fallback median
        med = val.get("median_ms", val.get("median"))
        if isinstance(med, (int, float)):
            return float(med)
    return None


def parse_lite_latency(data: dict[str, Any]) -> bool | None:
    """Determine if lite candidate is lighter than baseline MlNdtNet."""
    lat = data.get("latency")
    if not isinstance(lat, dict):
        return None
    cand_obj = lat.get("total", lat.get("litenet", {}).get("total", lat.get("litenet")))
    cand_med = _extract_median_val(cand_obj)
    mlndt_med = _extract_median_val(lat.get("mlndtnet"))
    if cand_med is None or mlndt_med is None:
        return None
    return bool(cand_med < mlndt_med)


def parse_invariant_folds(
    data: dict[str, Any],
) -> tuple[str, dict[str, FoldMetrics]]:
    """Defensively parse invariant folds or return schema unknown."""
    folds_list = _normalize_folds_dict(data.get("folds"))
    if not folds_list:
        return "schema unknown", {}
    result: dict[str, FoldMetrics] = {}
    for item in folds_list:
        held = str(item.get("held_out", ""))
        # keys as written by experiments/run_invariant_loco.py build_fold_record
        fused = item.get("fused", {})
        auc_val = fused.get("auroc")
        det = fused.get("operating_points", {}).get("0.05", {}).get("detection")
        if auc_val is None or det is None:
            return "schema unknown", {}
        auc = float(auc_val)
        k = int(det.get("k", -1))
        n = int(det.get("n", -1))
        rate = float(k / n) if n > 0 else float("nan")
        result[held] = FoldMetrics(auroc=auc, det_k=k, det_n=n, det_rate=rate)
    return "evaluated", result


def judge_candidate_folds(
    cand_folds: dict[str, FoldMetrics],
    base_folds: dict[str, FoldMetrics],
    lighter: bool | None = None,
) -> dict[str, dict[str, Any]]:
    """Compare per-fold candidate metrics against baseline."""
    verdicts: dict[str, dict[str, Any]] = {}
    for held, base_m in base_folds.items():
        cand_m = cand_folds.get(held)
        v = compute_verdict(cand_m, base_m)
        if lighter is not None:
            v = FoldVerdict(
                base_auroc=v.base_auroc,
                cand_auroc=v.cand_auroc,
                delta_auroc=v.delta_auroc,
                base_det_k=v.base_det_k,
                base_det_n=v.base_det_n,
                cand_det_k=v.cand_det_k,
                cand_det_n=v.cand_det_n,
                verdict=v.verdict,
                lighter=lighter,
            )
        verdicts[held] = v.to_dict()
    return verdicts


def _build_candidate_report(
    status: str,
    adr: str,
    rule: str,
    folds: dict[str, dict[str, Any]],
    lighter: bool | None = None,
) -> dict[str, Any]:
    """Format single candidate report dictionary."""
    rep: dict[str, Any] = {
        "status": status,
        "adr": adr,
        "rule": rule,
        "note": EVIDENCE_NOTE,
        "folds": folds,
    }
    if lighter is not None:
        rep["lighter"] = lighter
    return rep


def judge_robust_candidate(
    results_dir: Path,
    base_folds: dict[str, FoldMetrics],
) -> dict[str, Any]:
    """Process robust LOCO experiment results."""
    file_path = results_dir / "loco_robust_metrics.json"
    if not file_path.exists():
        return _build_candidate_report("not run", ADR_ROBUST, RULE_TEXT, {})
    with file_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    folds_m = parse_robust_folds(data)
    verdicts = judge_candidate_folds(folds_m, base_folds)
    return _build_candidate_report("evaluated", ADR_ROBUST, RULE_TEXT, verdicts)


def judge_lite_candidate(
    results_dir: Path,
    base_folds: dict[str, FoldMetrics],
) -> dict[str, Any]:
    """Process lite hybrid experiment results."""
    file_path = results_dir / "lite_hybrid_metrics.json"
    if not file_path.exists():
        return _build_candidate_report("not run", ADR_LITE, LITE_RULE_TEXT, {})
    with file_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    folds_m = parse_lite_folds(data)
    lighter = parse_lite_latency(data)
    verdicts = judge_candidate_folds(folds_m, base_folds, lighter=lighter)
    return _build_candidate_report("evaluated", ADR_LITE, LITE_RULE_TEXT, verdicts, lighter=lighter)


def judge_invariant_candidate(
    results_dir: Path,
    base_folds: dict[str, FoldMetrics],
) -> dict[str, Any]:
    """Process invariant representation experiment results."""
    file_path = results_dir / "invariant_loco_metrics.json"
    if not file_path.exists():
        return _build_candidate_report("not run", ADR_INVARIANT, RULE_TEXT, {})
    with file_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    status, folds_m = parse_invariant_folds(data)
    if status == "schema unknown":
        return _build_candidate_report("schema unknown", ADR_INVARIANT, RULE_TEXT, {})
    verdicts = judge_candidate_folds(folds_m, base_folds)
    return _build_candidate_report("evaluated", ADR_INVARIANT, RULE_TEXT, verdicts)


def _extract_shift_metrics(item: Any) -> tuple[str, ShiftFoldMetrics] | None:
    """Extract held_out identifier and metrics from fold item if evaluated."""
    if not isinstance(item, dict) or item.get("status") != "evaluated":
        return None
    summary = item.get("summary")
    if not isinstance(summary, dict):
        return None
    det = summary.get("worst_case_detection_rate_0.01", summary.get("worst_case_detection_rate"))
    fcr = summary.get("worst_case_false_call_rate_0.01", summary.get("worst_case_false_call_rate"))
    if det is None or fcr is None:
        return None
    held = str(item.get("held_out", ""))
    return held, ShiftFoldMetrics(worst_det=float(det), worst_fcr=float(fcr))


def parse_shift_family_folds(family_data: dict[str, Any]) -> dict[str, ShiftFoldMetrics]:
    """Extract evaluated per-fold metrics from family payload."""
    folds_raw = _normalize_folds_dict(family_data.get("folds"))
    result: dict[str, ShiftFoldMetrics] = {}
    for item in folds_raw:
        parsed = _extract_shift_metrics(item)
        if parsed is not None:
            result[parsed[0]] = parsed[1]
    return result


def judge_shift_candidate_folds(
    cand_folds: dict[str, ShiftFoldMetrics],
    base_folds: dict[str, ShiftFoldMetrics],
    cand_held_outs: Sequence[str] = (),
) -> dict[str, dict[str, Any]]:
    """Compare candidate shift metrics against baseline across all folds."""
    verdicts: dict[str, dict[str, Any]] = {}
    all_folds = list(
        dict.fromkeys(list(base_folds.keys()) + list(cand_held_outs) + list(cand_folds.keys()))
    )
    for held in all_folds:
        cand_m = cand_folds.get(held)
        base_m = base_folds.get(held)
        v = compute_shift_verdict(cand_m, base_m)
        verdicts[held] = v.to_dict()
    return verdicts


def _judge_single_shift_candidate(
    family_data: dict[str, Any] | None,
    base_folds: dict[str, ShiftFoldMetrics],
) -> dict[str, Any]:
    """Judge a single candidate family against baseline shift results."""
    if not isinstance(family_data, dict) or family_data.get("status") != "evaluated":
        return {"status": "not run", "folds": {}}
    cand_folds = parse_shift_family_folds(family_data)
    raw_folds = _normalize_folds_dict(family_data.get("folds"))
    cand_held = [
        str(f.get("held_out")) for f in raw_folds if isinstance(f, dict) and "held_out" in f
    ]
    folds_dict = judge_shift_candidate_folds(cand_folds, base_folds, cand_held)
    return {"status": "evaluated", "folds": folds_dict}


def _build_shift_report(candidates: dict[str, Any]) -> dict[str, Any]:
    """Format the shift robustness section dictionary."""
    return {
        "adr": ADR_SHIFT,
        "rule": SHIFT_RULE_TEXT,
        "baseline_family": "loco",
        "note": SHIFT_NOTE,
        "candidates": candidates,
    }


def _empty_shift_candidates() -> dict[str, Any]:
    """Create empty not run candidate dictionary."""
    return {name: {"status": "not run", "folds": {}} for name in ("robust", "lite", "invariant")}


def _load_shift_base_folds(families: dict[str, Any]) -> dict[str, ShiftFoldMetrics]:
    """Extract baseline fold metrics if loco family is evaluated."""
    loco_data = families.get("loco")
    if isinstance(loco_data, dict) and loco_data.get("status") == "evaluated":
        return parse_shift_family_folds(loco_data)
    return {}


def evaluate_shift_robustness(results_dir: Path) -> dict[str, Any]:
    """Evaluate shift robustness across candidate families against loco baseline."""
    file_path = results_dir / "shift_robustness_metrics.json"
    if not file_path.exists():
        return _build_shift_report(_empty_shift_candidates())

    with file_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    families = data.get("families", {})
    base_folds = _load_shift_base_folds(families)

    candidates_out: dict[str, Any] = {}
    for name in ("robust", "lite", "invariant"):
        cand_fam = families.get(name)
        candidates_out[name] = _judge_single_shift_candidate(cand_fam, base_folds)
    return _build_shift_report(candidates_out)


def evaluate_preregistration(cfg: JudgeConfig) -> dict[str, Any]:
    """Execute evaluation of all candidate results against baseline metrics."""
    base_path = cfg.results_dir / "separability_metrics.json"
    base_folds = load_baseline_metrics(base_path)
    candidates: dict[str, Any] = {
        "robust": judge_robust_candidate(cfg.results_dir, base_folds),
        "lite": judge_lite_candidate(cfg.results_dir, base_folds),
        "invariant": judge_invariant_candidate(cfg.results_dir, base_folds),
    }
    shift_payload = evaluate_shift_robustness(cfg.results_dir)
    return {
        "rule": RULE_TEXT,
        "note": EVIDENCE_NOTE,
        "baseline_source": str(base_path),
        "candidates": candidates,
        "shift_robustness": shift_payload,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for judging pre-registered experiments."""
    parser = argparse.ArgumentParser(description="Judge pre-registered experiment outcomes.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing experiment result files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/preregistration_verdicts.json"),
        help="Path for output verdict JSON",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entrypoint evaluating candidates and writing verdicts to disk."""
    args = parse_args(argv)
    cfg = JudgeConfig(results_dir=Path(args.results_dir), out_path=Path(args.out))
    payload = evaluate_preregistration(cfg)
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload


if __name__ == "__main__":
    main()
