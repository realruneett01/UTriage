"""Tests for signal-response (ahat vs a) POD analysis (project plan Phase 4)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from utriage.pod.ahat import (  # noqa: E402
    AhatFit,
    Censoring,
    ahat_pod,
    ahat_to_dict,
    fit_ahat,
)


def _simulate_data(n: int = 1000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """Simulate log-uniform flaw sizes and ahat responses per project plan."""
    rng = np.random.default_rng(seed)
    ln_a = rng.uniform(np.log(0.2), np.log(5.0), size=n)
    sizes = np.exp(ln_a)
    ln_ahat = 0.5 + 1.2 * ln_a + rng.normal(0.0, 0.3, size=n)
    responses = np.exp(ln_ahat)
    q20 = float(np.percentile(responses, 20))
    return sizes, responses, q20


def test_simulation_uncensored_recovery() -> None:
    """Recover beta0, beta1, sigma within 10% on uncensored data."""
    sizes, responses, _ = _simulate_data(n=1000, seed=0)
    fit = fit_ahat(sizes, responses, decision=1.0)

    assert isinstance(fit, AhatFit)
    assert fit.method == "tobit-delta"
    assert fit.n == 1000
    assert fit.n_left == 0
    assert fit.n_right == 0
    assert abs(fit.beta0 - 0.5) / 0.5 < 0.10
    assert abs(fit.beta1 - 1.2) / 1.2 < 0.10
    assert abs(fit.sigma - 0.3) / 0.3 < 0.10


def test_simulation_left_censored_recovery() -> None:
    """Tobit recovers parameters within 10% under 20% left censoring."""
    sizes, responses, q20 = _simulate_data(n=1000, seed=0)
    censored_responses = np.maximum(responses, q20)
    ols_slope, ols_intercept = np.polyfit(np.log(sizes), np.log(censored_responses), 1)
    assert abs(ols_slope - 1.2) / 1.2 > 0.10

    censor = Censoring(low=q20)
    fit = fit_ahat(sizes, censored_responses, decision=1.0, censor=censor)
    assert fit.n_left == int(np.sum(censored_responses <= q20))
    assert abs(fit.beta0 - 0.5) / 0.5 < 0.10
    assert abs(fit.beta1 - 1.2) / 1.2 < 0.10
    assert abs(fit.sigma - 0.3) / 0.3 < 0.10


def test_analytic_a90_and_confidence_bound() -> None:
    """a90 matches analytic value within 10% and a90_95 > a90."""
    sizes, responses, q20 = _simulate_data(n=1000, seed=0)
    censor = Censoring(low=q20)
    fit = fit_ahat(sizes, responses, decision=1.0, censor=censor)

    true_mu = (np.log(1.0) - 0.5) / 1.2
    true_a90 = float(np.exp(true_mu + 1.2816 * 0.3 / 1.2))
    assert abs(fit.a90 - true_a90) / true_a90 < 0.10
    assert fit.a90_95 > fit.a90


def test_ahat_pod_monotonic_and_bounded() -> None:
    """ahat_pod output is strictly in [0, 1] and monotone increasing."""
    sizes, responses, _ = _simulate_data(n=1000, seed=0)
    fit = fit_ahat(sizes, responses, decision=1.0)

    grid = np.linspace(0.1, 6.0, 100)
    pod = ahat_pod(fit, grid)
    assert np.all(pod >= 0.0)
    assert np.all(pod <= 1.0)
    assert np.all(np.diff(pod) >= 0.0)


def test_ahat_to_dict_roundtrip() -> None:
    """ahat_to_dict serializes cleanly to JSON."""
    sizes, responses, q20 = _simulate_data(n=500, seed=1)
    fit = fit_ahat(sizes, responses, decision=1.0, censor=Censoring(low=q20))
    data = ahat_to_dict(fit)

    assert data["n"] == 500
    assert data["method"] == "tobit-delta"
    assert len(data["cov"]) == 3
    roundtrip = json.loads(json.dumps(data))
    assert roundtrip["beta0"] == fit.beta0


def test_right_censoring_recovery() -> None:
    """Right censoring at 80th percentile is correctly handled."""
    sizes, responses, _ = _simulate_data(n=1000, seed=0)
    q80 = float(np.percentile(responses, 80))
    censor = Censoring(high=q80)
    fit = fit_ahat(sizes, responses, decision=1.0, censor=censor)

    assert fit.n_right == int(np.sum(responses >= q80))
    assert abs(fit.beta0 - 0.5) / 0.5 < 0.15
    assert abs(fit.beta1 - 1.2) / 1.2 < 0.15


def test_sample_size_shrinks_confidence_bound() -> None:
    """Delta a90_95 - a90 narrows as sample size increases."""
    s_small, r_small, _ = _simulate_data(n=200, seed=42)
    s_large, r_large, _ = _simulate_data(n=2000, seed=42)
    fit_small = fit_ahat(s_small, r_small, decision=1.0)
    fit_large = fit_ahat(s_large, r_large, decision=1.0)

    delta_small = fit_small.a90_95 - fit_small.a90
    delta_large = fit_large.a90_95 - fit_large.a90
    assert delta_large < delta_small


def test_negative_slope_raises() -> None:
    """Negative relation between size and response raises ValueError."""
    sizes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    responses = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="beta1"):
        fit_ahat(sizes, responses, decision=2.0)


def test_fewer_than_three_uncensored_raises() -> None:
    """Fewer than 3 uncensored observations raises ValueError."""
    sizes = np.array([1.0, 2.0, 3.0, 4.0])
    responses = np.array([0.5, 0.6, 0.7, 0.8])
    censor = Censoring(low=0.75)
    with pytest.raises(ValueError, match="uncensored"):
        fit_ahat(sizes, responses, decision=1.0, censor=censor)


@pytest.mark.parametrize(
    ("sizes", "responses", "decision", "censor", "match_str"),
    [
        (np.array([1.0, 2.0]), np.array([1.0]), 1.0, Censoring(), "equal"),
        (np.empty(0), np.empty(0), 1.0, Censoring(), "empty"),
        (np.array([[1.0]]), np.array([[1.0]]), 1.0, Censoring(), "1D"),
        (np.array([-1.0, 2.0]), np.array([1.0, 2.0]), 1.0, Censoring(), "positive"),
        (np.array([1.0, 2.0]), np.array([0.0, 2.0]), 1.0, Censoring(), "positive"),
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), -1.0, Censoring(), "Decision"),
        (
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            1.0,
            Censoring(low=2.0, high=1.0),
            "less than",
        ),
    ],
)
def test_invalid_inputs_raise(
    sizes: np.ndarray,
    responses: np.ndarray,
    decision: float,
    censor: Censoring,
    match_str: str,
) -> None:
    """Invalid array shapes, non-positive values, or bad limits raise ValueError."""
    with pytest.raises(ValueError, match=match_str):
        fit_ahat(sizes, responses, decision=decision, censor=censor)


def test_ahat_pod_invalid_sizes_raise() -> None:
    """Non-positive sizes in ahat_pod raise ValueError."""
    sizes, responses, _ = _simulate_data(n=200, seed=0)
    fit = fit_ahat(sizes, responses, decision=1.0)
    with pytest.raises(ValueError, match="positive"):
        ahat_pod(fit, np.array([1.0, -0.5]))
