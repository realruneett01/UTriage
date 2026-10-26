"""Continuous signal-response (ahat vs a) POD analysis (project plan Phase 4).

Follows MIL-HDBK-1823A Tobit maximum-likelihood estimation for censored responses
and delta-method confidence bounds for a90/95.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.optimize
from scipy.special import log_ndtr, ndtr
from scipy.stats import norm

_Z_90: float = 1.2816
_Z_95: float = 1.645
_EPS: float = 1e-4


@dataclass(frozen=True)
class Censoring:
    """Censoring thresholds for signal responses (noise floor or saturation)."""

    low: float | None = None
    high: float | None = None


_DEFAULT_CENSOR: Censoring = Censoring()


@dataclass(frozen=True)
class AhatFit:
    """Fitted continuous Tobit model parameters and POD thresholds."""

    beta0: float
    beta1: float
    sigma: float
    cov: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    n: int
    n_left: int
    n_right: int
    a_dec: float
    a50: float
    a90: float
    a90_95: float
    method: str = "tobit-delta"


def _check_arrays(sizes: np.ndarray, responses: np.ndarray) -> None:
    """Validate array dimensions and positivity."""
    if any([sizes.ndim != 1, responses.ndim != 1, len(sizes) != len(responses)]):
        raise ValueError("Sizes and responses must be 1D arrays of equal length.")
    if len(sizes) == 0:
        raise ValueError("Input arrays must not be empty.")
    if np.any(sizes <= 0.0) or np.any(responses <= 0.0):
        raise ValueError("Flaw sizes and responses must be strictly positive.")


def _check_censor(decision: float, censor: Censoring) -> None:
    """Validate decision threshold and censoring bounds."""
    if decision <= 0.0:
        raise ValueError("Decision threshold must be strictly positive.")
    if censor.low is not None and censor.low <= 0.0:
        raise ValueError("Censor low limit must be strictly positive.")
    if censor.high is not None and censor.high <= 0.0:
        raise ValueError("Censor high limit must be strictly positive.")
    if censor.low is not None and censor.high is not None:
        if censor.low >= censor.high:
            raise ValueError("Censor low limit must be strictly less than high limit.")


def _identify_censoring(
    responses: np.ndarray, censor: Censoring
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify boolean masks for uncensored, left-censored, and right-censored points."""
    is_left = (
        (responses <= censor.low)
        if censor.low is not None
        else np.zeros(len(responses), dtype=bool)
    )
    is_right = (
        (responses >= censor.high)
        if censor.high is not None
        else np.zeros(len(responses), dtype=bool)
    )
    is_uncensored = (~is_left) & (~is_right)
    return is_uncensored, is_left, is_right


def _init_ols(x_u: np.ndarray, y_u: np.ndarray) -> tuple[float, float, float]:
    """Initialize beta0, beta1, and ln(sigma) via OLS on uncensored observations."""
    x_mean = float(np.mean(x_u))
    y_mean = float(np.mean(y_u))
    s_xx = float(np.sum((x_u - x_mean) ** 2))
    if s_xx <= 0.0:
        raise ValueError("Uncensored flaw sizes have zero variance.")
    s_xy = float(np.sum((x_u - x_mean) * (y_u - y_mean)))
    b1 = s_xy / s_xx
    b0 = y_mean - b1 * x_mean
    residuals = y_u - (b0 + b1 * x_u)
    raw_sigma = float(np.std(residuals))
    sigma_init = max(raw_sigma, 1e-3)
    return b0, b1, float(np.log(sigma_init))


def _eval_tobit_nll(
    theta: np.ndarray,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    censor_logs: tuple[float | None, float | None],
) -> float:
    """Evaluate negative log-likelihood of Tobit model."""
    b0, b1, tau = float(theta[0]), float(theta[1]), float(theta[2])
    sigma = float(np.exp(tau))
    x, y, is_u, is_l, is_r = data
    y_low, y_high = censor_logs

    r_u = (y[is_u] - b0 - b1 * x[is_u]) / sigma
    ll = -len(r_u) * (tau + 0.5 * np.log(2.0 * np.pi)) - 0.5 * float(np.sum(r_u**2))
    if y_low is not None and np.any(is_l):
        r_l = (y_low - b0 - b1 * x[is_l]) / sigma
        ll += float(np.sum(log_ndtr(r_l)))
    if y_high is not None and np.any(is_r):
        r_r = (y_high - b0 - b1 * x[is_r]) / sigma
        ll += float(np.sum(log_ndtr(-r_r)))
    return -ll


def _eval_tobit_grad(
    theta: np.ndarray,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    censor_logs: tuple[float | None, float | None],
) -> np.ndarray:
    """Evaluate analytical gradient of Tobit negative log-likelihood."""
    b0, b1, tau = float(theta[0]), float(theta[1]), float(theta[2])
    sigma = float(np.exp(tau))
    x, y, is_u, is_l, is_r = data
    y_low, y_high = censor_logs

    r_u = (y[is_u] - b0 - b1 * x[is_u]) / sigma
    d_b0, d_b1 = -float(np.sum(r_u)) / sigma, -float(np.sum(r_u * x[is_u])) / sigma
    d_tau = float(len(r_u) - np.sum(r_u**2))
    if y_low is not None and np.any(is_l):
        r_l = (y_low - b0 - b1 * x[is_l]) / sigma
        lam_l = np.exp(norm.logpdf(r_l) - log_ndtr(r_l))
        d_b0 += float(np.sum(lam_l)) / sigma
        d_b1 += float(np.sum(lam_l * x[is_l])) / sigma
        d_tau += float(np.sum(lam_l * r_l))
    if y_high is not None and np.any(is_r):
        r_r = (y_high - b0 - b1 * x[is_r]) / sigma
        lam_r = np.exp(norm.logpdf(r_r) - log_ndtr(-r_r))
        d_b0 -= float(np.sum(lam_r)) / sigma
        d_b1 -= float(np.sum(lam_r * x[is_r])) / sigma
        d_tau -= float(np.sum(lam_r * r_r))
    return np.array([d_b0, d_b1, d_tau])


def _compute_hessian(func: Callable[[np.ndarray], float], theta: np.ndarray) -> np.ndarray:
    """Compute 3x3 numerical Hessian of scalar loss via central finite differences."""
    hess = np.zeros((3, 3), dtype=float)
    f0 = func(theta)
    for i in range(3):
        e_i = np.zeros(3)
        e_i[i] = _EPS
        hess[i, i] = (func(theta + e_i) - 2.0 * f0 + func(theta - e_i)) / (_EPS**2)
        for j in range(i + 1, 3):
            e_j = np.zeros(3)
            e_j[j] = _EPS
            f_pp = func(theta + e_i + e_j)
            f_pm = func(theta + e_i - e_j)
            f_mp = func(theta - e_i + e_j)
            f_mm = func(theta - e_i - e_j)
            diff = (f_pp - f_pm - f_mp + f_mm) / (4.0 * _EPS**2)
            hess[i, j] = diff
            hess[j, i] = diff
    return hess


def _invert_hessian(hess: np.ndarray) -> np.ndarray:
    """Invert numerical Hessian to obtain parameter covariance matrix."""
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError as err:
        # Hessian matrix is singular when data cannot identify model parameters
        raise ValueError("Observed Fisher information (Hessian) is singular.") from err
    if np.any(np.diag(cov) <= 0.0):
        raise ValueError("Covariance matrix has non-positive diagonal entries.")
    return cov


def _calc_thresholds(
    theta: np.ndarray, cov: np.ndarray, a_dec: float
) -> tuple[float, float, float]:
    """Calculate a50, a90, and delta-method one-sided 95% upper bound a90_95."""
    b0, b1, tau = float(theta[0]), float(theta[1]), float(theta[2])
    sigma = float(np.exp(tau))
    ln_adec = float(np.log(a_dec))
    mu = (ln_adec - b0) / b1
    a50 = float(np.exp(mu))
    g = mu + _Z_90 * sigma / b1
    a90 = float(np.exp(g))
    # Analytical gradient of g with respect to (beta0, beta1, ln_sigma)
    grad_g = np.array([-1.0 / b1, -g / b1, _Z_90 * sigma / b1])
    var_g = float(grad_g @ cov @ grad_g)
    se_g = float(np.sqrt(max(0.0, var_g)))
    a90_95 = float(np.exp(g + _Z_95 * se_g))
    return a50, a90, a90_95


def _run_mle(
    init_theta: np.ndarray,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    c_logs: tuple[float | None, float | None],
) -> np.ndarray:
    """Run BFGS optimization on Tobit NLL."""
    res = scipy.optimize.minimize(
        _eval_tobit_nll,
        x0=init_theta,
        args=(data, c_logs),
        jac=_eval_tobit_grad,
        method="BFGS",
    )
    if not res.success or res.x[1] <= 0.0:
        raise ValueError(f"Optimization failed or beta1 <= 0: b1={res.x[1]:.4f}, {res.message}")
    return res.x


def _make_fit(
    theta: np.ndarray,
    cov: np.ndarray,
    counts: tuple[int, int, int],
    a_dec: float,
) -> AhatFit:
    """Construct AhatFit dataclass from fitted parameters and thresholds."""
    a50, a90, a90_95 = _calc_thresholds(theta, cov, a_dec)
    cov_tuple = tuple(tuple(float(v) for v in row) for row in cov)
    return AhatFit(
        beta0=float(theta[0]),
        beta1=float(theta[1]),
        sigma=float(np.exp(theta[2])),
        cov=cov_tuple,  # type: ignore[arg-type]
        n=counts[0],
        n_left=counts[1],
        n_right=counts[2],
        a_dec=a_dec,
        a50=a50,
        a90=a90,
        a90_95=a90_95,
    )


def fit_ahat(
    sizes: np.ndarray,
    responses: np.ndarray,
    decision: float,
    censor: Censoring = _DEFAULT_CENSOR,
) -> AhatFit:
    """Fit Tobit signal-response model via MLE and estimate POD thresholds."""
    if censor is None:
        censor = _DEFAULT_CENSOR
    _check_arrays(sizes, responses)
    _check_censor(decision, censor)
    s, r = np.asarray(sizes, dtype=float), np.asarray(responses, dtype=float)
    x, y = np.log(s), np.log(r)
    is_u, is_l, is_r = _identify_censoring(r, censor)
    if np.sum(is_u) < 3:
        raise ValueError(f"Fewer than 3 uncensored points ({int(np.sum(is_u))} provided).")
    c_logs = (
        np.log(censor.low) if censor.low is not None else None,
        np.log(censor.high) if censor.high is not None else None,
    )
    data = (x, y, is_u, is_l, is_r)
    theta = _run_mle(np.array(_init_ols(x[is_u], y[is_u])), data, c_logs)
    cov = _invert_hessian(_compute_hessian(lambda th: _eval_tobit_nll(th, data, c_logs), theta))
    counts = (len(s), int(np.sum(is_l)), int(np.sum(is_r)))
    return _make_fit(theta, cov, counts, decision)


def ahat_pod(fit: AhatFit, sizes: np.ndarray) -> np.ndarray:
    """Evaluate fitted continuous probability of detection curve."""
    s = np.asarray(sizes, dtype=float)
    if np.any(s <= 0.0):
        raise ValueError("Flaw sizes must be strictly positive.")
    z = (fit.beta0 + fit.beta1 * np.log(s) - np.log(fit.a_dec)) / fit.sigma
    return ndtr(z)


def ahat_to_dict(fit: AhatFit) -> dict[str, Any]:
    """Serialize AhatFit dataclass to a JSON-safe dictionary."""
    return {
        "beta0": fit.beta0,
        "beta1": fit.beta1,
        "sigma": fit.sigma,
        "cov": [list(row) for row in fit.cov],
        "n": fit.n,
        "n_left": fit.n_left,
        "n_right": fit.n_right,
        "a_dec": fit.a_dec,
        "a50": fit.a50,
        "a90": fit.a90,
        "a90_95": fit.a90_95,
        "method": fit.method,
    }
