"""Hybrid detection methods combining supervised and unsupervised models (ADR 0010)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from utriage.ml.anomaly import fit_threshold
from utriage.pod.metrics import detection_summary, false_calls_per_scan


def or_union(p_sup: np.ndarray, a_score: np.ndarray, t_anom: float) -> np.ndarray:
    """Compute H1 logical OR union flag = (p_sup >= 0.5) | (a_score >= t_anom)."""
    p_arr = np.asarray(p_sup, dtype=float)
    a_arr = np.asarray(a_score, dtype=float)
    if p_arr.shape != a_arr.shape:
        raise ValueError(f"Shape mismatch: {p_arr.shape} != {a_arr.shape}")
    return (p_arr >= 0.5) | (a_arr >= float(t_anom))


def ecdf(calib_scores: np.ndarray) -> Callable[[np.ndarray | float], np.ndarray]:
    """Return empirical cumulative distribution function over calibration scores."""
    sorted_cal = np.sort(np.asarray(calib_scores, dtype=float).ravel())
    n = float(sorted_cal.size)

    def _eval(scores: np.ndarray | float) -> np.ndarray:
        arr = np.asarray(scores, dtype=float)
        if n == 0.0:
            return np.zeros_like(arr, dtype=float)
        ranks = np.searchsorted(sorted_cal, arr, side="right")
        return np.clip(ranks / n, 0.0, 1.0)

    return _eval


def fused_score(p_sup: np.ndarray, p_anom: np.ndarray) -> np.ndarray:
    """Compute average fused score s = (p_sup + p_anom) / 2."""
    p_arr = np.asarray(p_sup, dtype=float)
    a_arr = np.asarray(p_anom, dtype=float)
    if p_arr.shape != a_arr.shape:
        raise ValueError(f"Shape mismatch: {p_arr.shape} != {a_arr.shape}")
    return (p_arr + a_arr) / 2.0


def quantile_threshold(calib_scores: np.ndarray, fcr: float) -> float:
    """Compute (1 - fcr) empirical quantile threshold on calibration scores."""
    return fit_threshold(calib_scores, false_call_rate=fcr)


def evaluate_detector(flags: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Evaluate binary detector flags against ground truth labels."""
    flags_arr = np.asarray(flags, dtype=bool).ravel()
    labels_arr = np.asarray(labels, dtype=int).ravel()
    flawed_mask = labels_arr == 1
    unflawed_mask = labels_arr == 0
    hits = flags_arr[flawed_mask].astype(int)
    det = detection_summary(hits)
    n_false = int(np.sum(flags_arr[unflawed_mask]))
    n_scans = int(np.sum(unflawed_mask))
    fc = false_calls_per_scan(n_false, n_scans)
    return {
        "detection": det,
        "false_calls": fc,
    }
