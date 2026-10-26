"""Tests for uncertainty quantification and operating point selection (project plan 5.2 and 5.3)."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure local src directory is prioritized regardless of environment
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
from scipy.special import expit

from utriage.triage.operating_point import OperatingPoint, choose_operating_point
from utriage.triage.uncertainty import (
    Conformal,
    coverage,
    ensemble_stats,
    fit_conformal,
    is_uncertain,
    prediction_sets,
)


def test_ensemble_stats_shapes_and_values() -> None:
    """Ensemble stats returns mean and std over models (axis 0)."""
    probs = np.array(
        [
            [0.2, 0.8, 0.5],
            [0.4, 0.8, 0.7],
            [0.6, 0.8, 0.9],
        ]
    )
    mean, std = ensemble_stats(probs)
    assert mean.shape == (3,)
    assert std.shape == (3,)
    np.testing.assert_allclose(mean, [0.4, 0.8, 0.7])
    expected_std = [np.std([0.2, 0.4, 0.6]), 0.0, np.std([0.5, 0.7, 0.9])]
    np.testing.assert_allclose(std, expected_std, atol=1e-7)


@pytest.mark.parametrize(
    "bad_input",
    [
        np.array([0.1, 0.2]),
        np.empty((0, 3)),
    ],
)
def test_ensemble_stats_invalid_input(bad_input: np.ndarray) -> None:
    """Ensemble stats rejects 1D or empty model inputs."""
    with pytest.raises(ValueError):
        ensemble_stats(bad_input)


def test_conformal_coverage_synthetic() -> None:
    """Split conformal coverage on calibrated synthetic data matches 1 - alpha."""
    rng = np.random.default_rng(42)
    n_cal = 2000
    cal_p = rng.uniform(0.0, 1.0, size=n_cal)
    cal_y = rng.binomial(1, cal_p)

    conf = fit_conformal(cal_p, cal_y, alpha=0.1)
    assert isinstance(conf, Conformal)
    assert conf.n_cal == n_cal
    assert conf.alpha == 0.1

    n_test = 5000
    test_p = rng.uniform(0.0, 1.0, size=n_test)
    test_y = rng.binomial(1, test_p)

    sets = prediction_sets(conf, test_p)
    cov = coverage(sets, test_y)
    assert 0.88 <= cov <= 0.94


def test_perfect_classifier_gives_singleton_sets() -> None:
    """A perfect classifier yields singletons and zero uncertainty."""
    cal_p = np.array([0.0, 1.0, 0.0, 1.0] * 50)
    cal_y = np.array([0, 1, 0, 1] * 50)
    conf = fit_conformal(cal_p, cal_y, alpha=0.1)

    test_p = np.array([0.0, 1.0])
    sets = prediction_sets(conf, test_p)
    expected = np.array([[True, False], [False, True]])
    np.testing.assert_array_equal(sets, expected)

    unc = is_uncertain(sets)
    assert not np.any(unc)


def test_is_uncertain_mixed_sets() -> None:
    """is_uncertain flags empty sets and both-label sets as uncertain."""
    mixed_sets = np.array(
        [
            [True, True],  # both labels -> uncertain
            [False, False],  # empty set -> uncertain
            [True, False],  # singleton not-flaw -> certain
            [False, True],  # singleton flaw -> certain
        ]
    )
    unc = is_uncertain(mixed_sets)
    np.testing.assert_array_equal(unc, [True, True, False, False])


@pytest.mark.parametrize(
    ("probs", "labels", "alpha"),
    [
        (np.array([]), np.array([]), 0.1),
        (np.array([0.5]), np.array([1]), 0.0),
        (np.array([0.5]), np.array([1]), 1.0),
        (np.array([0.5, 0.6]), np.array([1]), 0.1),
    ],
)
def test_conformal_validation_errors(
    probs: np.ndarray,
    labels: np.ndarray,
    alpha: float,
) -> None:
    """fit_conformal raises ValueError on invalid inputs."""
    with pytest.raises(ValueError):
        fit_conformal(probs, labels, alpha=alpha)


def test_choose_operating_point_separable_binomial() -> None:
    """Separable synthetic data follows the binomial path."""
    # 40 flaws with sizes >= 0.8 and high scores
    flaw_sizes = np.full(40, 1.0)
    flaw_scores = np.linspace(0.8, 0.99, 40)
    noise_scores = np.array([0.05, 0.10, 0.15])

    op = choose_operating_point(
        flaw_scores=flaw_scores,
        flaw_sizes=flaw_sizes,
        noise_scores=noise_scores,
        ref_size=0.8,
        target=0.90,
    )
    assert isinstance(op, OperatingPoint)
    assert op.method == "binomial"
    assert op.pod_lower95_at_ref >= 0.90
    assert op.false_call_rate == 0.0
    assert op.n_flawed == 40
    assert op.n_noise == 3


def test_choose_operating_point_overlapping_logistic() -> None:
    """Overlapping synthetic data follows the logistic regression path."""
    rng = np.random.default_rng(123)
    n = 120
    sizes = rng.uniform(0.5, 3.0, size=n)
    # Log-linear probability with noise to produce overlapping hits/misses
    logits = -0.5 + 2.0 * np.log(sizes)
    scores = expit(logits + rng.normal(0, 0.4, size=n))
    noise_scores = rng.uniform(0.0, 0.3, size=50)

    op = choose_operating_point(
        flaw_scores=scores,
        flaw_sizes=sizes,
        noise_scores=noise_scores,
        ref_size=1.2,
        target=0.80,
    )
    assert isinstance(op, OperatingPoint)
    assert op.method == "logistic"
    assert op.pod_lower95_at_ref >= 0.80
    assert op.threshold > 0.0


def test_higher_target_never_yields_higher_threshold() -> None:
    """Higher target never yields a higher threshold (monotonicity)."""
    rng = np.random.default_rng(42)
    n = 100
    sizes = rng.uniform(0.5, 3.0, size=n)
    logits = -0.2 + 1.8 * np.log(sizes)
    scores = expit(logits + rng.normal(0, 0.3, size=n))
    noise = rng.uniform(0.0, 0.4, size=30)

    op_low = choose_operating_point(scores, sizes, noise, ref_size=1.0, target=0.70)
    op_mid = choose_operating_point(scores, sizes, noise, ref_size=1.0, target=0.85)
    op_high = choose_operating_point(scores, sizes, noise, ref_size=1.0, target=0.95)

    assert op_mid.threshold <= op_low.threshold
    assert op_high.threshold <= op_mid.threshold


def test_choose_operating_point_target_not_met() -> None:
    """When target cannot be reached, lowest threshold is returned with suffixed method."""
    flaw_sizes = np.array([0.5, 0.6, 0.7])
    flaw_scores = np.array([0.5, 0.6, 0.7])
    noise_scores = np.array([0.1])

    # ref_size 1.0 has 0 flaws >= ref_size, impossible to reach 0.95
    op = choose_operating_point(
        flaw_scores=flaw_scores,
        flaw_sizes=flaw_sizes,
        noise_scores=noise_scores,
        ref_size=1.0,
        target=0.95,
    )
    assert "(target not met)" in op.method
    assert op.threshold == 0.5
