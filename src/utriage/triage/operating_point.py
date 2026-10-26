"""Operating point selection for ML ultrasonic flaw triage (project plan 5.3).

Procedure:
1. Scan candidate thresholds across unique flaw scores in descending order
   (highest threshold first, minimizing false calls).
2. For each candidate threshold t, compute detection hits = (flaw_scores >= t).
3. Evaluate the one-sided 95% lower confidence bound of POD at the reference size:
   - Primary: Fit logistic POD curve via fit_hitmiss and evaluate pod_lower_bound
     at ref_size using the delta method.
   - Fallback: If logistic fit fails (due to outcome separation or non-convergence),
     calculate the exact binomial Clopper-Pearson 95% lower bound via
     binomial_pod_bound on flaws with size >= ref_size.
4. Select the first (highest) threshold whose POD lower bound >= target.
   The returned OperatingPoint records the threshold, bound, method ('logistic' or
   'binomial'), false call rate on noise indications, and sample counts.
5. If no candidate threshold qualifies, return the lowest threshold with its
   computed bound and method suffixed with '(target not met)'.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utriage.pod.hitmiss import (
    binomial_pod_bound,
    false_call_rate,
    fit_hitmiss,
    pod_lower_bound,
)


@dataclass(frozen=True)
class OperatingPoint:
    """Operating point parameters and validation performance metrics."""

    threshold: float
    ref_size: float
    target: float
    pod_lower95_at_ref: float
    method: str
    false_call_rate: float
    n_flawed: int
    n_noise: int


def _validate_inputs(
    flaw_scores: np.ndarray,
    flaw_sizes: np.ndarray,
    ref_size: float,
    target: float,
) -> None:
    """Validate arrays and numeric parameters."""
    if any([flaw_scores.ndim != 1, flaw_sizes.ndim != 1]):
        raise ValueError("Flaw scores and flaw sizes must be 1D arrays.")
    if len(flaw_scores) == 0:
        raise ValueError("Flaw scores must not be empty.")
    if len(flaw_scores) != len(flaw_sizes):
        raise ValueError("Flaw scores and sizes must have the same length.")
    if any([ref_size <= 0.0, target <= 0.0, target > 1.0]):
        raise ValueError("ref_size must be positive and target must be in (0, 1].")
    if np.any(flaw_sizes <= 0.0):
        raise ValueError("Flaw sizes must be strictly positive.")


def _evaluate_candidate(
    flaw_scores: np.ndarray,
    flaw_sizes: np.ndarray,
    threshold: float,
    ref_size: float,
) -> tuple[float, str]:
    """Calculate POD lower bound at ref_size using logistic fit or binomial fallback."""
    hits = (flaw_scores >= threshold).astype(float)
    try:
        fit = fit_hitmiss(flaw_sizes, hits)
        bound_arr = pod_lower_bound(fit, np.array([ref_size], dtype=float))
        bound = float(bound_arr[0])
        if np.isnan(bound):
            raise ValueError("POD lower bound produced NaN.")
        return bound, "logistic"
    except ValueError:
        # fit_hitmiss raises ValueError on separable data, singular Fisher info, or non-convergence
        mask = flaw_sizes >= ref_size
        res = binomial_pod_bound(flaw_sizes[mask], hits[mask])
        return float(res["pod_lower95"]), "binomial"


def _calc_fcr(noise_scores: np.ndarray, threshold: float) -> float:
    """Calculate false call rate on noise indications at given threshold."""
    if len(noise_scores) == 0:
        return 0.0
    res = false_call_rate(noise_scores >= threshold)
    return float(res["rate"])


def choose_operating_point(
    flaw_scores: np.ndarray,
    flaw_sizes: np.ndarray,
    noise_scores: np.ndarray,
    ref_size: float,
    target: float = 0.90,
) -> OperatingPoint:
    """Select the highest score threshold maintaining POD lower 95% bound >= target.

    Scans candidate thresholds (unique flaw scores descending) and returns the first
    threshold where the one-sided 95% lower bound of POD at ref_size meets the target.
    If none qualifies, returns the lowest threshold with method suffixed '(target not met)'.
    """
    scores = np.asarray(flaw_scores, dtype=float)
    sizes = np.asarray(flaw_sizes, dtype=float)
    noise = np.asarray(noise_scores, dtype=float)
    _validate_inputs(scores, sizes, ref_size, target)

    candidates = np.sort(np.unique(scores))[::-1]
    last_bound = 0.0
    last_method = "binomial"

    for t in candidates:
        thresh = float(t)
        bound, method = _evaluate_candidate(scores, sizes, thresh, ref_size)
        last_bound, last_method = bound, method
        if bound >= target:
            return OperatingPoint(
                threshold=thresh,
                ref_size=float(ref_size),
                target=float(target),
                pod_lower95_at_ref=bound,
                method=method,
                false_call_rate=_calc_fcr(noise, thresh),
                n_flawed=len(scores),
                n_noise=len(noise),
            )

    lowest_t = float(candidates[-1])
    return OperatingPoint(
        threshold=lowest_t,
        ref_size=float(ref_size),
        target=float(target),
        pod_lower95_at_ref=last_bound,
        method=f"{last_method} (target not met)",
        false_call_rate=_calc_fcr(noise, lowest_t),
        n_flawed=len(scores),
        n_noise=len(noise),
    )
