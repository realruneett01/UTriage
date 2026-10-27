"""Generalization matrix evaluation metrics (project plan Phase 8).

Includes Clopper-Pearson exact confidence bounds, detection summary, false call rates,
expected calibration error (ECE), reliability diagrams, and robust POD fitting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.stats

from utriage.pod.hitmiss import binomial_pod_bound, fit_hitmiss, fit_to_dict


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Calculate exact two-sided Clopper-Pearson confidence interval via Beta distribution."""
    if n == 0:
        return float("nan"), float("nan")
    if any([n < 0, k < 0, k > n]):
        raise ValueError(f"Invalid counts for binomial interval: k={k}, n={n}.")
    if not (0.0 < conf < 1.0):
        raise ValueError(f"Confidence level must be strictly between 0 and 1, got {conf}.")

    alpha = 1.0 - conf
    low = 0.0 if k == 0 else float(scipy.stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    high = 1.0 if k == n else float(scipy.stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return low, high


def detection_summary(hits: np.ndarray, conf: float = 0.95) -> dict[str, Any]:
    """Summarize hit detection counts, empirical rate, and two-sided Clopper-Pearson CI."""
    arr = np.asarray(hits)
    n = int(len(arr))
    if n == 0:
        low, high = clopper_pearson(0, 0, conf=conf)
        return {
            "n": 0,
            "k": 0,
            "rate": float("nan"),
            "ci_low": low,
            "ci_high": high,
            "method": "Clopper-Pearson two-sided 95",
        }

    k = int(np.sum(arr == 1))
    rate = float(k / n)
    ci_low, ci_high = clopper_pearson(k, n, conf=conf)
    return {
        "n": n,
        "k": k,
        "rate": rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "method": "Clopper-Pearson two-sided 95",
    }


def false_calls_per_scan(n_false: int, n_scans: int) -> dict[str, Any]:
    """Calculate false calls per scan rate."""
    if any([n_false < 0, n_scans < 0]):
        raise ValueError("Counts of false calls and scans must be non-negative.")
    rate = float("nan") if n_scans == 0 else float(n_false / n_scans)
    return {
        "rate": rate,
        "n_false": int(n_false),
        "n_scans": int(n_scans),
    }


def _bin_stats(p_bin: np.ndarray, y_bin: np.ndarray) -> tuple[float | None, float | None, int]:
    """Compute mean confidence, empirical accuracy, and count for one bin."""
    count = len(p_bin)
    if count == 0:
        return None, None, 0
    return float(np.mean(p_bin)), float(np.mean(y_bin)), count


def reliability_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    """Compute reliability curve with equal-width bins for JSON serialization."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if len(p) != len(y):
        raise ValueError("probs and labels must have equal length.")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_conf: list[float | None] = []
    frac_pos: list[float | None] = []
    counts: list[int] = []

    for b in range(n_bins):
        is_last = b == n_bins - 1
        mask = (p >= edges[b]) & (p <= edges[b + 1] if is_last else p < edges[b + 1])
        conf, frac, cnt = _bin_stats(p[mask], y[mask])
        mean_conf.append(conf)
        frac_pos.append(frac)
        counts.append(cnt)

    return {
        "bin_edges": [float(e) for e in edges],
        "mean_conf": mean_conf,
        "frac_pos": frac_pos,
        "counts": counts,
    }


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE) across equal-width bins."""
    curve = reliability_curve(probs, labels, n_bins=n_bins)
    n_total = sum(curve["counts"])
    if n_total == 0:
        return 0.0

    ece = 0.0
    for conf, frac, cnt in zip(curve["mean_conf"], curve["frac_pos"], curve["counts"], strict=True):
        if cnt > 0 and conf is not None and frac is not None:
            ece += (cnt / n_total) * abs(frac - conf)
    return float(ece)


def pod_or_bound(sizes: np.ndarray, hits: np.ndarray) -> dict[str, Any]:
    """Fit logistic POD if fittable; otherwise fall back to binomial bound."""
    s = np.asarray(sizes, dtype=float)
    h = np.asarray(hits, dtype=float)
    try:
        fit = fit_hitmiss(s, h)
        res = fit_to_dict(fit)
        res["fittable"] = True
        res["reason"] = None
        return res
    except (ValueError, RuntimeError) as err:
        # Fallback to binomial bound when logistic regression cannot converge or data is separable
        res = binomial_pod_bound(s, h)
        res["fittable"] = False
        res["reason"] = str(err)
        res["a90_95"] = None
        return res


def cell(
    hits: np.ndarray,
    sizes: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    n_false: int,
    n_scans: int,
    n_physical_flaws: int,
) -> dict[str, Any]:
    """Combine generalization matrix metrics and flaw/image counts for one cell."""
    det = detection_summary(hits)
    n_img = int(len(probs))
    n_flaws = int(n_physical_flaws)
    return {
        "detection": det,
        "false_calls": false_calls_per_scan(n_false, n_scans),
        "ece": expected_calibration_error(probs, labels),
        "reliability": reliability_curve(probs, labels),
        "pod": pod_or_bound(sizes, hits),
        "n_images": n_img,
        "n_physical_flaws": n_flaws,
        "counts": {
            "images": n_img,
            "distinct_physical_flaws": n_flaws,
            "hits": det["k"],
            "total_flaws_tested": det["n"],
        },
    }
