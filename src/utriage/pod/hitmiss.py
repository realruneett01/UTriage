"""Hit/miss probability of detection (POD) analysis (project plan Phase 3).

Follows MIL-HDBK-1823A logistic regression with delta-method confidence bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.optimize
import scipy.stats
from scipy.special import expit

_Z_95: float = 1.645
_LN_9: float = float(np.log(9.0))


@dataclass(frozen=True)
class PodFit:
    """Fitted logistic POD curve parameters and detection thresholds."""

    beta0: float
    beta1: float
    cov: tuple[tuple[float, float], tuple[float, float]]
    n: int
    a50: float
    a90: float
    a90_95: float
    method: str = "delta"


def _check_shapes_and_values(s: np.ndarray, h: np.ndarray) -> None:
    """Validate array shapes, positivity, and binary labels."""
    if any([s.ndim != 1, h.ndim != 1, len(s) != len(h)]):
        raise ValueError("Sizes and hits must be 1D arrays of equal length.")
    if len(s) == 0:
        raise ValueError("Input arrays must not be empty.")
    if np.any(s <= 0.0):
        raise ValueError("Flaw sizes must be strictly positive.")
    if not np.all((h == 0.0) | (h == 1.0)):
        raise ValueError("Hits must be binary (0 or 1).")


def _check_separation(s: np.ndarray, h: np.ndarray) -> None:
    """Raise ValueError if hit/miss outcomes are separable by flaw size."""
    n_hits = int(np.sum(h == 1.0))
    if n_hits == 0 or n_hits == len(h):
        raise ValueError("Data is separable: only hits or only misses present.")
    if np.min(s[h == 1.0]) >= np.max(s[h == 0.0]):
        raise ValueError("Data is separable: all hits are above all misses.")
    if np.min(s[h == 0.0]) >= np.max(s[h == 1.0]):
        raise ValueError("Data is separable: all misses are above all hits.")


def _validate_inputs(sizes: np.ndarray, hits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate input dimensions, positivity, and check for separable data."""
    s = np.asarray(sizes, dtype=float)
    h = np.asarray(hits, dtype=float)
    _check_shapes_and_values(s, h)
    _check_separation(s, h)
    return s, h


def _fit_mle(x: np.ndarray, hits: np.ndarray) -> tuple[float, float]:
    """Fit logistic regression parameters (beta0, beta1) via BFGS on NLL."""
    design_x = np.column_stack([np.ones_like(x), x])

    def nll(beta: np.ndarray) -> float:
        eta = design_x @ beta
        return float(np.sum(np.logaddexp(0.0, eta) - hits * eta))

    def grad(beta: np.ndarray) -> np.ndarray:
        eta = design_x @ beta
        return design_x.T @ (expit(eta) - hits)

    res = scipy.optimize.minimize(nll, x0=np.array([0.0, 1.0]), jac=grad, method="BFGS")
    if not res.success:
        raise ValueError(f"Optimization failed to converge: {res.message}")
    beta0, beta1 = float(res.x[0]), float(res.x[1])
    if beta1 <= 0.0:
        raise ValueError(f"Fitted slope beta1 must be strictly positive, got {beta1:.4f}.")
    return beta0, beta1


def _calc_cov(x: np.ndarray, beta: tuple[float, float]) -> np.ndarray:
    """Calculate covariance matrix as inverse of observed Fisher information."""
    design_x = np.column_stack([np.ones_like(x), x])
    b = np.array([beta[0], beta[1]])
    prob = expit(design_x @ b)
    weights = prob * (1.0 - prob)
    fisher = design_x.T @ (weights[:, None] * design_x)
    try:
        return np.linalg.inv(fisher)
    except np.linalg.LinAlgError as err:
        # Fisher information matrix is singular when data is poorly conditioned
        raise ValueError("Observed Fisher information matrix is singular.") from err


def _calc_thresholds(beta0: float, beta1: float, cov: np.ndarray) -> tuple[float, float, float]:
    """Calculate a50, a90, and delta-method one-sided 95% upper bound a90_95."""
    a50 = float(np.exp(-beta0 / beta1))
    ln_a90 = float((_LN_9 - beta0) / beta1)
    a90 = float(np.exp(ln_a90))
    grad_g = np.array([-1.0 / beta1, -(_LN_9 - beta0) / (beta1**2)])
    var_ln_a90 = float(grad_g @ cov @ grad_g)
    se_ln_a90 = np.sqrt(max(0.0, var_ln_a90))
    a90_95 = float(np.exp(ln_a90 + _Z_95 * se_ln_a90))
    return a50, a90, a90_95


def fit_hitmiss(sizes: np.ndarray, hits: np.ndarray) -> PodFit:
    """Fit logistic hit/miss POD model on ln(flaw sizes)."""
    s, h = _validate_inputs(sizes, hits)
    x = np.log(s)
    b0, b1 = _fit_mle(x, h)
    cov = _calc_cov(x, (b0, b1))
    a50, a90, a90_95 = _calc_thresholds(b0, b1, cov)
    cov_tuple = (
        (float(cov[0, 0]), float(cov[0, 1])),
        (float(cov[1, 0]), float(cov[1, 1])),
    )
    return PodFit(
        beta0=b0,
        beta1=b1,
        cov=cov_tuple,
        n=len(s),
        a50=a50,
        a90=a90,
        a90_95=a90_95,
        method="delta",
    )


def pod_curve(fit: PodFit, sizes: np.ndarray) -> np.ndarray:
    """Evaluate fitted probability of detection curve at given flaw sizes."""
    s = np.asarray(sizes, dtype=float)
    if np.any(s <= 0.0):
        raise ValueError("Flaw sizes must be strictly positive.")
    eta = fit.beta0 + fit.beta1 * np.log(s)
    return expit(eta)


def pod_lower_bound(fit: PodFit, sizes: np.ndarray) -> np.ndarray:
    """Evaluate one-sided 95% lower confidence bound of POD via delta method."""
    s = np.asarray(sizes, dtype=float)
    if np.any(s <= 0.0):
        raise ValueError("Flaw sizes must be strictly positive.")
    x = np.log(s)
    cov = np.asarray(fit.cov, dtype=float)
    var_eta = cov[0, 0] + 2.0 * x * cov[0, 1] + (x**2) * cov[1, 1]
    se_eta = np.sqrt(np.maximum(0.0, var_eta))
    eta = fit.beta0 + fit.beta1 * x
    eta_lower = eta - _Z_95 * se_eta
    return expit(eta_lower)


def false_call_rate(flags: np.ndarray) -> dict[str, float | int]:
    """Calculate false call rate with exact Clopper-Pearson one-sided 95% upper bound."""
    arr = np.asarray(flags)
    n = len(arr)
    if n == 0:
        raise ValueError("Flags array cannot be empty.")
    n_false = int(np.sum(arr != 0))
    rate = float(n_false / n)
    if n_false == n:
        upper95 = 1.0
    else:
        upper95 = float(scipy.stats.beta.ppf(0.95, n_false + 1, n - n_false))
    return {
        "rate": rate,
        "n": n,
        "n_false": n_false,
        "upper95": upper95,
    }


def fit_to_dict(fit: PodFit) -> dict[str, Any]:
    """Serialize PodFit dataclass to a JSON-safe dictionary."""
    return {
        "beta0": fit.beta0,
        "beta1": fit.beta1,
        "cov": [list(fit.cov[0]), list(fit.cov[1])],
        "n": fit.n,
        "a50": fit.a50,
        "a90": fit.a90,
        "a90_95": fit.a90_95,
        "method": fit.method,
    }


def binomial_pod_bound(sizes: np.ndarray, hits: np.ndarray) -> dict[str, Any]:
    """Calculate binomial Clopper-Pearson 95% lower bound for POD over tested size range."""
    s = np.asarray(sizes, dtype=float)
    h = np.asarray(hits, dtype=float)
    if any([s.ndim != 1, h.ndim != 1, len(s) != len(h)]):
        raise ValueError("Sizes and hits must be 1D arrays of equal length.")
    n_flawed = int(len(s))
    if n_flawed == 0:
        return {
            "n_flawed": 0,
            "n_hit": 0,
            "min_size": 0.0,
            "max_size": 0.0,
            "size_p10": 0.0,
            "size_p50": 0.0,
            "pod_lower95": 0.0,
            "method": "binomial Clopper-Pearson one-sided 95",
            "note": "POD lower bound over the whole tested size range, not a POD curve.",
        }
    n_hit = int(np.sum(h == 1.0))
    pod_lower95 = (
        0.0 if n_hit == 0 else float(scipy.stats.beta.ppf(0.05, n_hit, n_flawed - n_hit + 1))
    )
    return {
        "n_flawed": n_flawed,
        "n_hit": n_hit,
        "min_size": float(np.min(s)),
        "max_size": float(np.max(s)),
        "size_p10": float(np.percentile(s, 10)),
        "size_p50": float(np.percentile(s, 50)),
        "pod_lower95": pod_lower95,
        "method": "binomial Clopper-Pearson one-sided 95",
        "note": "POD lower bound over the whole tested size range, not a POD curve.",
    }
