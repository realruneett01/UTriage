"""Tests for model fit diagnosis helpers (Phase 8)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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

import pytest  # noqa: E402

from utriage.ml.diagnostics import (  # noqa: E402
    FitRule,
    data_size_curve,
    diagnose,
    learning_curve,
    plot_learning_curve,
)


@pytest.fixture
def hand_history() -> list[dict[str, float]]:
    """Sample training history mimicking TrainResult.history."""
    return [
        {"epoch": 1, "train_loss": 0.80, "val_loss": 0.90, "val_acc": 0.60},
        {"epoch": 2, "train_loss": 0.50, "val_loss": 0.60, "val_acc": 0.75},
        {"epoch": 3, "train_loss": 0.30, "val_loss": 0.40, "val_acc": 0.88},
        {"epoch": 4, "train_loss": 0.20, "val_loss": 0.55, "val_acc": 0.82},
    ]


def test_learning_curve_hand_history(hand_history: list[dict[str, float]]) -> None:
    """Verify trajectory extraction, best epoch selection, and final gap calculation."""
    res = learning_curve(hand_history)
    assert res["epoch"] == [1, 2, 3, 4]
    assert res["train_loss"] == [0.80, 0.50, 0.30, 0.20]
    assert res["val_loss"] == [0.90, 0.60, 0.40, 0.55]
    assert res["best_epoch"] == 3
    assert res["final_gap"] == pytest.approx(0.35)
    serialized = json.dumps(res)
    assert "best_epoch" in serialized


def test_learning_curve_empty() -> None:
    """Verify empty history edge case handling."""
    res = learning_curve([])
    assert res["epoch"] == []
    assert res["best_epoch"] == 0
    assert res["final_gap"] == 0.0


def test_data_size_curve_calls_run_in_order() -> None:
    """Verify that run() is invoked once per fraction in specified order."""
    called_fractions: list[float] = []

    def mock_run(fraction: float) -> dict[str, float]:
        called_fractions.append(fraction)
        return {"val_acc": fraction * 0.9, "val_loss": 1.0 - fraction * 0.5}

    fractions = (0.1, 0.25, 0.5, 1.0)
    curve = data_size_curve(fractions, mock_run)

    assert called_fractions == [0.1, 0.25, 0.5, 1.0]
    assert len(curve) == 4
    for idx, frac in enumerate(fractions):
        assert curve[idx]["fraction"] == frac
        assert curve[idx]["val_acc"] == pytest.approx(frac * 0.9)
    assert json.dumps(curve)


@pytest.mark.parametrize(
    ("train_metric", "heldout_metrics", "expected_verdict", "expected_sets"),
    [
        (
            0.70,
            {"val": 0.65},
            "underfitting",
            {"val": {"gap": 0.05, "verdict": "underfitting", "verdicts": ["underfitting"]}},
        ),
        (
            0.92,
            {"val": 0.75},
            "overfitting",
            {"val": {"gap": 0.17, "verdict": "overfitting", "verdicts": ["overfitting"]}},
        ),
        (
            0.92,
            {"val": 0.88},
            "ok",
            {"val": {"gap": 0.04, "verdict": "ok", "verdicts": ["ok"]}},
        ),
        (
            0.92,
            {"val": 0.88, "loco": 0.70},
            "overfitting",
            {
                "val": {"gap": 0.04, "verdict": "ok", "verdicts": ["ok"]},
                "loco": {"gap": 0.22, "verdict": "overfitting", "verdicts": ["overfitting"]},
            },
        ),
        (
            0.70,
            {},
            "underfitting",
            {},
        ),
        (
            0.90,
            {},
            "ok",
            {},
        ),
        (
            0.70,
            {"val": 0.55},
            "underfitting+overfitting",
            {
                "val": {
                    "gap": 0.15,
                    "verdict": "underfitting+overfitting",
                    "verdicts": ["underfitting", "overfitting"],
                }
            },
        ),
    ],
)
def test_diagnose_cases(
    train_metric: float,
    heldout_metrics: dict[str, float],
    expected_verdict: str,
    expected_sets: dict[str, dict[str, Any]],
) -> None:
    """Verify underfitting, overfitting, ok, and worst-case multi-set diagnostics."""
    rule = FitRule(target_train_metric=0.85, max_gap=0.10)
    diag = diagnose(train_metric, heldout_metrics, rule)

    assert diag["verdict"] == expected_verdict
    assert diag["verdicts"] == expected_verdict.split("+")
    assert "evaluations" not in diag
    assert "overall_verdict" not in diag
    assert "target" in diag["rule"].lower() or "underfitting" in diag["rule"].lower()

    for name, exp in expected_sets.items():
        assert diag["sets"][name]["gap"] == pytest.approx(exp["gap"])
        assert diag["sets"][name]["verdict"] == exp["verdict"]
        assert diag["sets"][name]["verdicts"] == exp["verdicts"]

    serialized = json.dumps(diag)
    assert expected_verdict in serialized


def test_fit_rule_docstring() -> None:
    """Verify FitRule docstring states diagnostic rule."""
    doc = FitRule.__doc__ or ""
    assert "underfitting if train metric < target" in doc
    assert "overfitting if (train metric - group-held-out metric) > max_gap" in doc
    assert "else ok" in doc


def test_plot_learning_curve(tmp_path: Path) -> None:
    """Verify plot_learning_curve writes a valid non-empty PNG file."""
    curve = {
        "epoch": [1, 2, 3],
        "train_loss": [0.9, 0.5, 0.2],
        "val_loss": [1.0, 0.6, 0.3],
        "best_epoch": 3,
        "final_gap": 0.1,
    }
    out_file = tmp_path / "subdir" / "curve.png"
    result_path = plot_learning_curve(curve, out_file, title="Test Learning Curve")

    assert result_path == out_file
    assert out_file.exists()
    assert out_file.stat().st_size > 0
    header = out_file.read_bytes()[:8]
    assert header == b"\x89PNG\r\n\x1a\n"
