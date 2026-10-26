"""Uncertainty quantification: deep ensembles and split conformal prediction.

Follows Angelopoulos and Bates (2021), 'A Gentle Introduction to Conformal Prediction'.
Provides calibrated prediction sets and ensemble uncertainty for ultrasonic indications.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def ensemble_stats(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and standard deviation across ensemble models.

    Args:
        probs: Array of shape (M models, N samples) containing probabilities.

    Returns:
        tuple (mean, std) of shape (N,) over models.
    """
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2:
        raise ValueError("probs must be a 2D array of shape (M models, N samples).")
    if p.shape[0] == 0:
        raise ValueError("probs must have at least one model row.")
    return np.mean(p, axis=0), np.std(p, axis=0)


@dataclass(frozen=True)
class Conformal:
    """Split conformal prediction calibration parameters."""

    qhat: float
    alpha: float
    n_cal: int


def fit_conformal(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    alpha: float = 0.1,
) -> Conformal:
    """Fit split conformal calibration quantile for a binary classifier.

    Computes nonconformity scores s = 1 - p(true class), where true class probability
    is cal_probs for label 1 and 1 - cal_probs for label 0.
    The quantile qhat is the ceil((n+1)(1-alpha))/n empirical quantile (method=higher)
    following Angelopoulos & Bates (2021).
    """
    p = np.asarray(cal_probs, dtype=float)
    y = np.asarray(cal_labels, dtype=int)
    if any([p.ndim != 1, y.ndim != 1, len(p) != len(y)]):
        raise ValueError("cal_probs and cal_labels must be 1D arrays of equal length.")
    if len(p) == 0:
        raise ValueError("Calibration set must not be empty.")
    if any([alpha <= 0.0, alpha >= 1.0]):
        raise ValueError("alpha must be in (0, 1).")

    scores = np.where(y == 1, 1.0 - p, p)
    n = len(scores)
    q_level = min(1.0, float(np.ceil((n + 1) * (1.0 - alpha)) / n))
    qhat = float(np.quantile(scores, q_level, method="higher"))
    return Conformal(qhat=qhat, alpha=float(alpha), n_cal=n)


def prediction_sets(conf: Conformal, probs: np.ndarray) -> np.ndarray:
    """Generate conformal prediction sets for binary classification.

    Args:
        conf: Fitted Conformal calibration object.
        probs: Array of flaw probabilities p_1 of shape (N,).

    Returns:
        Boolean array of shape (N, 2) where column 0 indicates inclusion of
        not-flaw (label 0) and column 1 indicates inclusion of flaw (label 1).
    """
    p1 = np.asarray(probs, dtype=float)
    if p1.ndim != 1:
        raise ValueError("probs must be a 1D array.")
    p0 = 1.0 - p1
    in_0 = (1.0 - p0) <= conf.qhat
    in_1 = (1.0 - p1) <= conf.qhat
    return np.column_stack([in_0, in_1])


def is_uncertain(sets: np.ndarray) -> np.ndarray:
    """Identify samples where prediction set is ambiguous (both labels or empty).

    Args:
        sets: Boolean array of shape (N, 2) from prediction_sets.

    Returns:
        Boolean array of shape (N,) True when the prediction set contains both
        labels or is empty (not a singleton set).
    """
    arr = np.asarray(sets, dtype=bool)
    if any([arr.ndim != 2, arr.shape[1] != 2]):
        raise ValueError("sets must be a 2D array of shape (N, 2).")
    return arr[:, 0] == arr[:, 1]


def coverage(sets: np.ndarray, labels: np.ndarray) -> float:
    """Calculate empirical coverage of prediction sets against true labels.

    Args:
        sets: Boolean array of shape (N, 2).
        labels: Ground truth binary labels of shape (N,).

    Returns:
        Fraction of samples where the true label is contained in the prediction set.
    """
    s = np.asarray(sets, dtype=bool)
    y = np.asarray(labels, dtype=int)
    if any([s.ndim != 2, s.shape[1] != 2, len(s) != len(y)]):
        raise ValueError("sets must have shape (N, 2) matching labels length.")
    if len(y) == 0:
        return 0.0
    covered = np.where(y == 1, s[:, 1], s[:, 0])
    return float(np.mean(covered))
