"""Tests for generalization matrix metrics (project plan Phase 8)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# Ensure local worktree src/ takes precedence over any installed package
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.special import expit  # noqa: E402

from utriage.pod.metrics import (  # noqa: E402
    cell,
    clopper_pearson,
    detection_summary,
    expected_calibration_error,
    false_calls_per_scan,
    pod_or_bound,
    reliability_curve,
)


def _synthetic_hitmiss_data(n: int = 500, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate log-uniform flaw sizes and hit/miss observations."""
    rng = np.random.default_rng(seed)
    sizes = np.exp(rng.uniform(np.log(0.5), np.log(5.0), size=n))
    probs = expit(-2.0 + 3.0 * np.log(sizes))
    hits = (rng.uniform(size=n) < probs).astype(int)
    return sizes, hits


def test_clopper_pearson_known_values() -> None:
    """Clopper-Pearson interval matches known reference for k=5, n=10."""
    low, high = clopper_pearson(5, 10, conf=0.95)
    assert pytest.approx(low, abs=1e-3) == 0.187
    assert pytest.approx(high, abs=1e-3) == 0.813


@pytest.mark.parametrize(
    ("k", "n", "expected_low", "expected_high"),
    [
        (0, 10, 0.0, None),
        (10, 10, None, 1.0),
    ],
    ids=["k_zero", "k_equals_n"],
)
def test_clopper_pearson_boundary_rates(
    k: int, n: int, expected_low: float | None, expected_high: float | None
) -> None:
    """Boundary conditions k=0 and k=n produce exact 0.0 and 1.0 ends."""
    low, high = clopper_pearson(k, n, conf=0.95)
    if expected_low is not None:
        assert low == expected_low
        assert 0.0 < high < 1.0
    if expected_high is not None:
        assert high == expected_high
        assert 0.0 < low < 1.0


def test_clopper_pearson_zero_sample() -> None:
    """Sample size n=0 returns (nan, nan)."""
    low, high = clopper_pearson(0, 0, conf=0.95)
    assert math.isnan(low)
    assert math.isnan(high)


@pytest.mark.parametrize(
    ("k", "n", "conf"),
    [
        (-1, 10, 0.95),
        (11, 10, 0.95),
        (5, -1, 0.95),
        (5, 10, 0.0),
        (5, 10, 1.0),
    ],
    ids=["neg_k", "k_gt_n", "neg_n", "conf_zero", "conf_one"],
)
def test_clopper_pearson_invalid_parameters(k: int, n: int, conf: float) -> None:
    """Invalid parameter values raise ValueError."""
    with pytest.raises(ValueError):
        clopper_pearson(k, n, conf=conf)


@pytest.mark.parametrize(
    ("hits", "expected_n", "expected_k", "expected_rate"),
    [
        (np.array([1, 0, 1, 1, 0]), 5, 3, 0.6),
        (np.array([1, 1, 1]), 3, 3, 1.0),
        (np.array([0, 0]), 2, 0, 0.0),
    ],
    ids=["mixed", "all_hits", "all_misses"],
)
def test_detection_summary_standard_cases(
    hits: np.ndarray, expected_n: int, expected_k: int, expected_rate: float
) -> None:
    """Detection summary returns expected dictionary fields and rates."""
    res = detection_summary(hits)
    assert res["n"] == expected_n
    assert res["k"] == expected_k
    assert pytest.approx(res["rate"]) == expected_rate
    assert res["method"] == "Clopper-Pearson two-sided 95"
    assert res["ci_low"] <= res["rate"] <= res["ci_high"]


def test_detection_summary_empty() -> None:
    """Empty hits array yields n=0, k=0, and NaN rates."""
    res = detection_summary(np.array([]))
    assert res["n"] == 0
    assert res["k"] == 0
    assert math.isnan(res["rate"])
    assert math.isnan(res["ci_low"])
    assert math.isnan(res["ci_high"])


@pytest.mark.parametrize(
    ("n_false", "n_scans", "expected_rate"),
    [
        (2, 10, 0.2),
        (0, 5, 0.0),
        (5, 2, 2.5),
    ],
    ids=["fractional", "zero_false", "multiple_per_scan"],
)
def test_false_calls_per_scan_rates(n_false: int, n_scans: int, expected_rate: float) -> None:
    """False call rates are correctly computed per scan."""
    res = false_calls_per_scan(n_false, n_scans)
    assert pytest.approx(res["rate"]) == expected_rate
    assert res["n_false"] == n_false
    assert res["n_scans"] == n_scans


def test_false_calls_per_scan_zero_scans() -> None:
    """Zero scans returns NaN rate."""
    res = false_calls_per_scan(2, 0)
    assert math.isnan(res["rate"])
    assert res["n_false"] == 2
    assert res["n_scans"] == 0


def test_false_calls_per_scan_negative_raises() -> None:
    """Negative counts raise ValueError."""
    with pytest.raises(ValueError):
        false_calls_per_scan(-1, 5)


def test_expected_calibration_error_perfect_calibration() -> None:
    """ECE is zero when empirical accuracy matches predicted probability in every bin."""
    # Bin [0.5, 0.6) has mean conf 0.55 and exactly 55% positive (11 of 20)
    # Bin [0.2, 0.3) has mean conf 0.25 and exactly 25% positive (5 of 20)
    probs = np.array([0.55] * 20 + [0.25] * 20)
    labels = np.array([1] * 11 + [0] * 9 + [1] * 5 + [0] * 15)
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert pytest.approx(ece, abs=1e-6) == 0.0


def test_expected_calibration_error_hand_example() -> None:
    """ECE matches exact hand calculation for a 4-sample example."""
    probs = np.array([0.15, 0.15, 0.85, 0.85])
    labels = np.array([1, 0, 1, 1])
    # Bin 1 [0.1, 0.2): 2 samples, mean conf = 0.15, acc = 0.50, diff = 0.35, wt = 0.5
    # Bin 8 [0.8, 0.9): 2 samples, mean conf = 0.85, acc = 1.00, diff = 0.15, wt = 0.5
    # Weighted ECE = 0.5 * 0.35 + 0.5 * 0.15 = 0.25
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert pytest.approx(ece, abs=1e-6) == 0.25


def test_expected_calibration_error_empty() -> None:
    """Empty inputs yield 0.0 ECE."""
    ece = expected_calibration_error(np.array([]), np.array([]))
    assert ece == 0.0


def test_reliability_curve_structure_and_counts() -> None:
    """Reliability curve bin counts sum to n and empty bins are null/None."""
    probs = np.array([0.05, 0.12, 0.18, 0.55, 0.92, 1.0])
    labels = np.array([0, 0, 1, 1, 1, 1])
    curve = reliability_curve(probs, labels, n_bins=10)

    assert len(curve["bin_edges"]) == 11
    assert len(curve["mean_conf"]) == 10
    assert len(curve["frac_pos"]) == 10
    assert len(curve["counts"]) == 10
    assert sum(curve["counts"]) == len(probs)

    # Empty bins must have None for mean_conf and frac_pos
    for conf, frac, count in zip(
        curve["mean_conf"], curve["frac_pos"], curve["counts"], strict=True
    ):
        if count == 0:
            assert conf is None
            assert frac is None
        else:
            assert conf is not None
            assert frac is not None

    # Must be JSON-serializable and empty bins must serialize to null
    dumped = json.dumps(curve)
    loaded = json.loads(dumped)
    assert loaded["counts"] == curve["counts"]
    assert any(c is None for c in loaded["mean_conf"])


def test_pod_or_bound_fittable() -> None:
    """Fittable dataset returns PodFit parameters with a90_95 and fittable=True."""
    sizes, hits = _synthetic_hitmiss_data(500, seed=42)
    res = pod_or_bound(sizes, hits)
    assert res["fittable"] is True
    assert res["reason"] is None
    assert isinstance(res["a90_95"], float)
    assert res["a90_95"] > 0.0


def test_pod_or_bound_unfittable_fallback() -> None:
    """Separable or invalid data falls back to binomial POD bound with reason."""
    # All hits (separable) cannot fit logistic regression
    sizes = np.linspace(1.0, 5.0, 30)
    hits = np.ones(30, dtype=int)
    res = pod_or_bound(sizes, hits)
    assert res["fittable"] is False
    assert res["a90_95"] is None
    assert isinstance(res["reason"], str)
    assert len(res["reason"]) > 0
    assert res["pod_lower95"] >= 0.90


def test_cell_combines_metrics_and_json_serializable() -> None:
    """cell() integrates all metrics, includes counts, and serializes to JSON."""
    sizes, hits = _synthetic_hitmiss_data(100, seed=123)
    probs = np.array([0.1, 0.4, 0.7, 0.9] * 25)
    labels = np.array([0, 0, 1, 1] * 25)
    n_false = 4
    n_scans = 10
    n_physical_flaws = 3

    res = cell(
        hits=hits,
        sizes=sizes,
        probs=probs,
        labels=labels,
        n_false=n_false,
        n_scans=n_scans,
        n_physical_flaws=n_physical_flaws,
    )

    assert "detection" in res
    assert "false_calls" in res
    assert "ece" in res
    assert "reliability" in res
    assert "pod" in res
    assert res["n_images"] == 100
    assert res["n_physical_flaws"] == 3
    assert res["counts"]["images"] == 100
    assert res["counts"]["distinct_physical_flaws"] == 3

    # Verify JSON serialization without errors
    dumped = json.dumps(res)
    loaded = json.loads(dumped)
    assert loaded["n_images"] == 100
    assert loaded["detection"]["k"] == res["detection"]["k"]
