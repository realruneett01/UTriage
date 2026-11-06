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


def compute_verdict(
    cand: FoldMetrics | None,
    base: FoldMetrics | None,
) -> FoldVerdict:
    """Apply ADR pre-registered success rule comparing candidate against baseline."""
    if cand is None or base is None:
        return FoldVerdict(
            base_auroc=base.auroc if base else None,
            cand_auroc=cand.auroc if cand else None,
            delta_auroc=None,
            base_det_k=base.det_k if base else None,
            base_det_n=base.det_n if base else None,
            cand_det_k=cand.det_k if cand else None,
            cand_det_n=cand.det_n if cand else None,
            verdict="not evaluable",
        )
    return _judge_fold_metrics(cand, base)


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


def evaluate_preregistration(cfg: JudgeConfig) -> dict[str, Any]:
    """Execute evaluation of all candidate results against baseline metrics."""
    base_path = cfg.results_dir / "separability_metrics.json"
    base_folds = load_baseline_metrics(base_path)
    candidates: dict[str, Any] = {
        "robust": judge_robust_candidate(cfg.results_dir, base_folds),
        "lite": judge_lite_candidate(cfg.results_dir, base_folds),
        "invariant": judge_invariant_candidate(cfg.results_dir, base_folds),
    }
    return {
        "rule": RULE_TEXT,
        "note": EVIDENCE_NOTE,
        "baseline_source": str(base_path),
        "candidates": candidates,
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
