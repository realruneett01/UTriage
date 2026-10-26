"""Tests for bar-domain comparison utilities (project plan Phase 4)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Ensure local workspace src is prioritized over external paths
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        sys.modules["utriage"].__path__.insert(0, _pkg_dir)
if "utriage.ml" in sys.modules:
    _ml_dir = str(Path(_SRC) / "utriage" / "ml")
    if _ml_dir not in sys.modules["utriage.ml"].__path__:
        sys.modules["utriage.ml"].__path__.insert(0, _ml_dir)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from utriage.ml.compare import (  # noqa: E402
    detection_table,
    matched_threshold,
    pod_summary,
    rules_window_score,
)
from utriage.physics.simulator import load_probe, pulse  # noqa: E402


def test_rules_window_score_pure_pulse() -> None:
    """Test rules_window_score on a pure pulse of amplitude 1.0 gives about 100."""
    probe = load_probe("p5")
    dt = (np.arange(128) - 64) / probe.sampling_mhz
    p = pulse(dt, probe)

    score_1d = rules_window_score(p)
    assert np.isclose(score_1d[0], 100.0, atol=2.0)

    w_3d_1ch = p.reshape(1, 1, 128)
    score_3d_1ch = rules_window_score(w_3d_1ch)
    assert np.isclose(score_3d_1ch[0], 100.0, atol=2.0)

    w_3d_9ch = np.zeros((1, 9, 128), dtype=np.float32)
    w_3d_9ch[0, 4] = p
    score_3d_9ch = rules_window_score(w_3d_9ch)
    assert np.isclose(score_3d_9ch[0], 100.0, atol=2.0)


@pytest.mark.parametrize("target_fcr", [0.01, 0.05, 0.10, 0.25])
def test_matched_threshold_random_scores(target_fcr: float) -> None:
    """Test matched_threshold gives FCR <= target on random scores."""
    rng = np.random.default_rng(0)
    scores = rng.random(1000)
    thresh = matched_threshold(scores, target_fcr)
    actual_fcr = float(np.mean(scores >= thresh))
    assert actual_fcr <= target_fcr


def test_matched_threshold_zero_target() -> None:
    """Test matched_threshold gives 0 FCR when target is 0."""
    rng = np.random.default_rng(42)
    scores = rng.random(200)
    thresh = matched_threshold(scores, 0.0)
    assert thresh > float(np.max(scores))
    assert float(np.mean(scores >= thresh)) == 0.0


def test_detection_table_counts() -> None:
    """Test detection_table computes correct n and detection rate per group and size."""
    hits = np.array([1, 1, 0, 1, 0, 0], dtype=float)
    sizes = np.array([0.4, 0.4, 0.4, 0.8, 0.8, 0.8], dtype=float)
    groups = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)

    rows = detection_table(hits, sizes, groups)
    assert len(rows) == 2

    assert rows[0]["size"] == 0.4
    assert rows[0]["n"] == 3
    assert rows[0]["n_hit"] == 2
    assert np.isclose(rows[0]["rate"], 2.0 / 3.0)

    assert rows[1]["size"] == 0.8
    assert rows[1]["n"] == 3
    assert rows[1]["n_hit"] == 1
    assert np.isclose(rows[1]["rate"], 1.0 / 3.0)


def test_pod_summary_separable_data() -> None:
    """Test pod_summary returns pod None with reason on separable data and has binomial."""
    sizes = np.array([0.3, 0.4, 0.8, 1.0], dtype=float)
    hits = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)

    res = pod_summary(sizes, hits)
    assert res["pod"] is None
    assert res["reason"] is not None
    assert isinstance(res["reason"], str)
    assert "separable" in res["reason"].lower()
    assert "binomial" in res
    assert res["binomial"]["n_flawed"] == 4
    assert res["binomial"]["n_hit"] == 2
