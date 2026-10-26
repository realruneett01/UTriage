"""Comparison utilities between rules engine and ML models on bar windows."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.signal

from utriage.pod.hitmiss import binomial_pod_bound, fit_hitmiss, fit_to_dict

_EPS: float = 1e-6


def rules_window_score(x: np.ndarray) -> np.ndarray:
    """Peak envelope of centre channel x 100 (% of reference per window)."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        centre = arr[np.newaxis, :]
    elif arr.ndim == 2:
        centre = arr
    else:
        c_idx = arr.shape[1] // 2
        centre = arr[:, c_idx, :]
    env = np.abs(scipy.signal.hilbert(centre, axis=-1))
    scores = np.max(env, axis=-1) * 100.0
    return np.asarray(scores, dtype=np.float32)


def matched_threshold(ml_noise_scores: np.ndarray, target_fcr: float) -> float:
    """Find smallest threshold t such that mean(ml_noise_scores >= t) <= target_fcr."""
    scores = np.asarray(ml_noise_scores, dtype=np.float64)
    if len(scores) == 0:
        return 1.0
    max_val = float(np.max(scores))
    if target_fcr <= 0.0:
        return max_val + _EPS
    unique_scores = np.sort(np.unique(scores))
    counts_ge = len(scores) - np.searchsorted(np.sort(scores), unique_scores, side="left")
    rates = counts_ge / len(scores)
    valid = np.where(rates <= target_fcr)[0]
    if len(valid) == 0:
        return max_val + _EPS
    return float(unique_scores[valid[0]])


def _make_table_row(
    group: Any,
    size: float,
    hits_sub: np.ndarray,
) -> dict[str, Any]:
    n = int(len(hits_sub))
    n_hit = int(np.sum(hits_sub > 0.5))
    rate = float(n_hit / n) if n > 0 else 0.0
    return {
        "group": group,
        "size": float(size),
        "size_mm": float(size),
        "n": n,
        "n_hit": n_hit,
        "rate": rate,
        "detection_rate": rate,
    }


def detection_table(
    hits: np.ndarray,
    sizes: np.ndarray,
    groups: np.ndarray,
) -> list[dict[str, Any]]:
    """Compute detection rate and count per (group, size) pair."""
    h = np.asarray(hits, dtype=np.float64)
    s = np.asarray(sizes, dtype=np.float64)
    g = np.asarray(groups)
    if any([len(h) != len(s), len(h) != len(g)]):
        raise ValueError("hits, sizes, and groups must have equal length.")
    if len(h) == 0:
        return []
    rows: list[dict[str, Any]] = []
    for grp in np.unique(g):
        g_mask = g == grp
        for sz in np.sort(np.unique(s[g_mask])):
            mask = g_mask & (s == sz)
            grp_val = grp.item() if hasattr(grp, "item") else grp
            rows.append(_make_table_row(grp_val, float(sz), h[mask]))
    return rows


def pod_summary(sizes: np.ndarray, hits: np.ndarray) -> dict[str, Any]:
    """Calculate POD summary with logistic fit and binomial lower bound."""
    s = np.asarray(sizes, dtype=np.float64)
    h = np.asarray(hits, dtype=np.float64)
    binomial = binomial_pod_bound(s, h)
    try:
        fit = fit_hitmiss(s, h)
        return {
            "pod": fit_to_dict(fit),
            "reason": None,
            "binomial": binomial,
        }
    except ValueError as exc:
        return {
            "pod": None,
            "reason": str(exc),
            "binomial": binomial,
        }
