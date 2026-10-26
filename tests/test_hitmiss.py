"""Tests for hit/miss probability of detection (project plan Phase 3)."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

# Ensure local worktree src/ takes precedence over any installed package
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.special import expit  # noqa: E402

from utriage.pod.hitmiss import (  # noqa: E402
    PodFit,
    binomial_pod_bound,
    false_call_rate,
    fit_hitmiss,
    fit_to_dict,
    pod_curve,
    pod_lower_bound,
)


def _generate_synthetic_data(
    n: int, seed: int = 0, beta0: float = -2.0, beta1: float = 4.0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate log-uniform flaw sizes and hit/miss observations."""
    rng = np.random.default_rng(seed)
    sizes = np.exp(rng.uniform(np.log(0.2), np.log(5.0), size=n))
    prob = expit(beta0 + beta1 * np.log(sizes))
    hits = (rng.uniform(size=n) < prob).astype(int)
    return sizes, hits


def _fit_synthetic(n: int, seed: int = 0, beta0: float = -2.0, beta1: float = 4.0) -> PodFit:
    """Generate synthetic data and fit hit/miss model."""
    sizes, hits = _generate_synthetic_data(n, seed=seed, beta0=beta0, beta1=beta1)
    return fit_hitmiss(sizes, hits)


def _ci_width(fit: PodFit) -> float:
    """Width of 95% confidence interval on a90."""
    return float(fit.a90_95 - fit.a90)


@pytest.fixture
def sample_pod_fit() -> PodFit:
    """Standard PodFit fixture for testing evaluation and serialization."""
    return PodFit(
        beta0=-2.0,
        beta1=4.0,
        cov=((0.01, -0.005), (-0.005, 0.02)),
        n=2000,
        a50=1.65,
        a90=2.85,
        a90_95=3.10,
    )


def test_recover_known_parameters() -> None:
    """Recover known parameters within specified tolerance (project plan 03-pod)."""
    true_b0, true_b1 = -2.0, 4.0
    fit = _fit_synthetic(2000, seed=0, beta0=true_b0, beta1=true_b1)

    assert isinstance(fit, PodFit)
    assert fit.method == "delta"
    assert fit.n == 2000
    assert abs(fit.beta0 - true_b0) / abs(true_b0) < 0.10
    assert abs(fit.beta1 - true_b1) / abs(true_b1) < 0.10

    true_a90 = float(np.exp((np.log(9.0) - true_b0) / true_b1))
    assert abs(fit.a90 - true_a90) / true_a90 < 0.05
    assert fit.a90_95 > fit.a90


@pytest.mark.parametrize("sample_size", [200, 2000])
def test_fit_hitmiss_sample_sizes(sample_size: int) -> None:
    """Hit/miss fit produces valid PodFit across sample sizes 200 and 2000."""
    fit = _fit_synthetic(sample_size, seed=0)
    assert isinstance(fit, PodFit)
    assert fit.method == "delta"
    assert fit.n == sample_size
    assert fit.a90_95 > fit.a90
    assert fit.a50 < fit.a90


def test_a90_95_shrinks_with_sample_size() -> None:
    """Confidence interval on a90 narrows as sample size increases."""
    fit_200 = _fit_synthetic(200, seed=0)
    fit_2000 = _fit_synthetic(2000, seed=0)
    assert _ci_width(fit_2000) < _ci_width(fit_200)


@pytest.mark.parametrize(
    ("sizes", "hits"),
    [
        (np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 0, 1, 1])),
        (np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1])),
        (np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0])),
    ],
)
def test_separable_data_raises(sizes: np.ndarray, hits: np.ndarray) -> None:
    """Perfect or quasi-separable data raises ValueError."""
    with pytest.raises(ValueError, match="separable"):
        fit_hitmiss(sizes, hits)


def test_negative_slope_raises() -> None:
    """Data where larger sizes have fewer hits raises ValueError."""
    sizes = np.array([1.0, 1.2, 2.0, 2.5, 3.0, 3.5])
    hits = np.array([1, 1, 1, 0, 0, 0])
    with pytest.raises(ValueError, match="separable|beta1"):
        fit_hitmiss(sizes, hits)


def test_invalid_sizes_raise() -> None:
    """Zero or negative flaw sizes raise ValueError."""
    with pytest.raises(ValueError, match="positive"):
        fit_hitmiss(np.array([0.0, 1.0, 2.0]), np.array([0, 1, 1]))


def test_pod_curve_and_lower_bound() -> None:
    """Lower bound of POD curve must be <= POD curve everywhere."""
    fit = _fit_synthetic(500, seed=42)

    eval_sizes = np.linspace(0.2, 5.0, 100)
    pod = pod_curve(fit, eval_sizes)
    lower = pod_lower_bound(fit, eval_sizes)

    assert pod.shape == eval_sizes.shape
    assert lower.shape == eval_sizes.shape
    assert np.all(lower <= pod)
    assert np.all(lower >= 0.0)
    assert np.all(pod <= 1.0)


@pytest.mark.parametrize(
    ("eval_fn", "invalid_sizes"),
    [
        (pod_curve, np.array([0.0, 1.0])),
        (pod_lower_bound, np.array([-1.0, 1.0])),
    ],
    ids=["pod_curve_zero", "pod_lower_bound_negative"],
)
def test_pod_curve_invalid_sizes(
    sample_pod_fit: PodFit,
    eval_fn: Callable[[PodFit, np.ndarray], np.ndarray],
    invalid_sizes: np.ndarray,
) -> None:
    """Evaluating POD curve or lower bound on non-positive sizes raises ValueError."""
    with pytest.raises(ValueError, match="positive"):
        eval_fn(sample_pod_fit, invalid_sizes)


def test_false_call_rate_zero_of_hundred() -> None:
    """0 false calls in 100 gives upper95 bound approx 0.0295 (rule of three)."""
    flags = np.zeros(100, dtype=int)
    fcr = false_call_rate(flags)

    assert fcr["rate"] == 0.0
    assert fcr["n"] == 100
    assert fcr["n_false"] == 0
    assert pytest.approx(fcr["upper95"], rel=1e-3) == 0.02951


@pytest.mark.parametrize(
    ("flags", "expected_rate", "expected_upper95", "exact_upper"),
    [
        (np.ones(50, dtype=int), 1.0, 1.0, True),
        (np.array([0] * 90 + [1] * 10), 0.10, 0.10, False),
    ],
    ids=["all_flags", "mixed_flags"],
)
def test_false_call_rate_cases(
    flags: np.ndarray,
    expected_rate: float,
    expected_upper95: float,
    exact_upper: bool,
) -> None:
    """Test boundary and non-zero cases for false call rate."""
    fcr = false_call_rate(flags)
    assert fcr["rate"] == expected_rate
    if exact_upper:
        assert fcr["upper95"] == expected_upper95
    else:
        assert fcr["upper95"] > expected_upper95


def test_false_call_rate_empty_raises() -> None:
    """Empty flags array raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        false_call_rate(np.array([]))


def test_fit_to_dict_json_serializable(sample_pod_fit: PodFit) -> None:
    """fit_to_dict must produce valid JSON-safe dictionary."""
    d = fit_to_dict(sample_pod_fit)
    assert isinstance(d, dict)
    assert d["beta0"] == -2.0
    assert d["beta1"] == 4.0
    assert d["n"] == 2000
    assert d["method"] == "delta"

    # Must serialize without errors
    serialized = json.dumps(d)
    deserialized = json.loads(serialized)
    assert deserialized["a90"] == 2.85


@pytest.mark.parametrize(
    ("hits", "expected_hit_count", "expected_lower95", "min_threshold"),
    [
        (np.ones(29, dtype=int), 29, 0.902, 0.90),
        (np.zeros(29, dtype=int), 0, 0.0, 0.0),
    ],
    ids=["twenty_nine_of_twenty_nine", "zero_hits"],
)
def test_binomial_pod_bound_extremes(
    hits: np.ndarray,
    expected_hit_count: int,
    expected_lower95: float,
    min_threshold: float,
) -> None:
    """29/29 and 0/29 hits yield expected lower bounds (project plan Phase 3)."""
    sizes = np.linspace(1.0, 5.0, 29)
    res = binomial_pod_bound(sizes, hits)

    assert res["n_flawed"] == 29
    assert res["n_hit"] == expected_hit_count
    assert res["pod_lower95"] >= min_threshold
    assert pytest.approx(res["pod_lower95"], rel=1e-3) == expected_lower95


def test_binomial_pod_bound_keys_and_json_serializable() -> None:
    """binomial_pod_bound produces dictionary with required keys and json serialization."""
    sizes = np.array([1.2, 2.5, 3.8, 4.1])
    hits = np.array([1, 1, 0, 1])
    res = binomial_pod_bound(sizes, hits)

    expected_keys = {
        "n_flawed",
        "n_hit",
        "min_size",
        "max_size",
        "size_p10",
        "size_p50",
        "pod_lower95",
        "method",
        "note",
    }
    assert set(res.keys()) == expected_keys
    assert isinstance(res["n_flawed"], int)
    assert isinstance(res["n_hit"], int)
    assert isinstance(res["min_size"], float)
    assert isinstance(res["max_size"], float)
    assert isinstance(res["size_p10"], float)
    assert isinstance(res["size_p50"], float)
    assert isinstance(res["pod_lower95"], float)
    assert isinstance(res["method"], str)
    assert isinstance(res["note"], str)

    serialized = json.dumps(res)
    deserialized = json.loads(serialized)
    assert deserialized["n_flawed"] == 4
    assert deserialized["n_hit"] == 3
