"""Tests for hit/miss probability of detection (project plan Phase 3)."""

from __future__ import annotations

import json
import sys
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


def test_recover_known_parameters() -> None:
    """Recover known parameters within specified tolerance (project plan 03-pod)."""
    true_b0, true_b1 = -2.0, 4.0
    sizes, hits = _generate_synthetic_data(2000, seed=0, beta0=true_b0, beta1=true_b1)
    fit = fit_hitmiss(sizes, hits)

    assert isinstance(fit, PodFit)
    assert fit.method == "delta"
    assert fit.n == 2000
    assert abs(fit.beta0 - true_b0) / abs(true_b0) < 0.10
    assert abs(fit.beta1 - true_b1) / abs(true_b1) < 0.10

    true_a90 = float(np.exp((np.log(9.0) - true_b0) / true_b1))
    assert abs(fit.a90 - true_a90) / true_a90 < 0.05
    assert fit.a90_95 > fit.a90


def test_a90_95_shrinks_with_sample_size() -> None:
    """Confidence interval on a90 narrows as sample size increases."""
    sizes_200, hits_200 = _generate_synthetic_data(200, seed=0)
    sizes_2000, hits_2000 = _generate_synthetic_data(2000, seed=0)
    fit_200 = fit_hitmiss(sizes_200, hits_200)
    fit_2000 = fit_hitmiss(sizes_2000, hits_2000)

    delta_200 = fit_200.a90_95 - fit_200.a90
    delta_2000 = fit_2000.a90_95 - fit_2000.a90
    assert delta_2000 < delta_200


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
    sizes, hits = _generate_synthetic_data(500, seed=42)
    fit = fit_hitmiss(sizes, hits)

    eval_sizes = np.linspace(0.2, 5.0, 100)
    pod = pod_curve(fit, eval_sizes)
    lower = pod_lower_bound(fit, eval_sizes)

    assert pod.shape == eval_sizes.shape
    assert lower.shape == eval_sizes.shape
    assert np.all(lower <= pod)
    assert np.all(lower >= 0.0)
    assert np.all(pod <= 1.0)


def test_pod_curve_invalid_sizes() -> None:
    """Evaluating POD curve on non-positive sizes raises ValueError."""
    fit = PodFit(
        beta0=-2.0,
        beta1=4.0,
        cov=((0.01, 0.0), (0.0, 0.01)),
        n=100,
        a50=1.65,
        a90=2.85,
        a90_95=3.10,
    )
    with pytest.raises(ValueError, match="positive"):
        pod_curve(fit, np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="positive"):
        pod_lower_bound(fit, np.array([-1.0, 1.0]))


def test_false_call_rate_zero_of_hundred() -> None:
    """0 false calls in 100 gives upper95 bound approx 0.0295 (rule of three)."""
    flags = np.zeros(100, dtype=int)
    fcr = false_call_rate(flags)

    assert fcr["rate"] == 0.0
    assert fcr["n"] == 100
    assert fcr["n_false"] == 0
    assert pytest.approx(fcr["upper95"], rel=1e-3) == 0.02951


def test_false_call_rate_cases() -> None:
    """Test boundary and non-zero cases for false call rate."""
    all_flags = np.ones(50, dtype=int)
    fcr_all = false_call_rate(all_flags)
    assert fcr_all["rate"] == 1.0
    assert fcr_all["upper95"] == 1.0

    mixed_flags = np.array([0] * 90 + [1] * 10)
    fcr_mixed = false_call_rate(mixed_flags)
    assert fcr_mixed["rate"] == 0.10
    assert fcr_mixed["upper95"] > 0.10

    with pytest.raises(ValueError, match="empty"):
        false_call_rate(np.array([]))


def test_fit_to_dict_json_serializable() -> None:
    """fit_to_dict must produce valid JSON-safe dictionary."""
    fit = PodFit(
        beta0=-2.0,
        beta1=4.0,
        cov=((0.01, -0.005), (-0.005, 0.02)),
        n=2000,
        a50=1.65,
        a90=2.85,
        a90_95=3.10,
    )
    d = fit_to_dict(fit)
    assert isinstance(d, dict)
    assert d["beta0"] == -2.0
    assert d["beta1"] == 4.0
    assert d["n"] == 2000
    assert d["method"] == "delta"

    # Must serialize without errors
    serialized = json.dumps(d)
    deserialized = json.loads(serialized)
    assert deserialized["a90"] == 2.85
