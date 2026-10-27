"""Tests for kept model decision logic in Phase 4."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="module")
def phase4_module() -> Any:
    """Dynamically load experiments/run_phase4.py."""
    script_path = Path(__file__).resolve().parents[1] / "experiments" / "run_phase4.py"
    spec = importlib.util.spec_from_file_location("run_phase4", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_phase4"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def decide_fn(phase4_module: Any) -> Any:
    """Extract _decide_kept_model function."""
    return phase4_module._decide_kept_model


def test_real_failure_case_selects_1d(decide_fn: Any) -> None:
    """Exact real failure case: 1D without fitted POD vs 2D with fitted POD."""
    res_1d = {2.0: {"pod_summary": {"pod": None, "binomial": {"pod_lower95": 0.4353}}}}
    res_2d = {
        2.0: {
            "pod_summary": {
                "pod": {"a90_95": 0.7183},
                "binomial": {"pod_lower95": 0.5065},
            }
        }
    }
    decision = decide_fn(res_1d, res_2d, 2.0)
    assert decision["choice"] == "ml_1d"


@pytest.mark.parametrize(
    ("val_1d", "val_2d", "expected_choice"),
    [
        (0.40, 0.50, "ml_2d"),
        (0.40, 0.55, "ml_2d"),
        (0.40, 0.49, "ml_1d"),
    ],
)
def test_margin_threshold_choice(
    decide_fn: Any,
    val_1d: float,
    val_2d: float,
    expected_choice: str,
) -> None:
    """Keep simpler 1D model unless 2D bound is at least 0.1 higher."""
    res_1d = {1.5: {"pod_summary": {"pod": None, "binomial": {"pod_lower95": val_1d}}}}
    res_2d = {1.5: {"pod_summary": {"pod": None, "binomial": {"pod_lower95": val_2d}}}}
    decision = decide_fn(res_1d, res_2d, 1.5)
    assert decision["choice"] == expected_choice


def test_both_have_fitted_pod_compares_binomial_bound(decide_fn: Any) -> None:
    """When both have fitted POD, selection still relies on binomial bound."""
    res_1d = {
        2.0: {
            "pod_summary": {
                "pod": {"a90_95": 0.90},
                "binomial": {"pod_lower95": 0.70},
            }
        }
    }
    res_2d = {
        2.0: {
            "pod_summary": {
                "pod": {"a90_95": 0.40},
                "binomial": {"pod_lower95": 0.75},
            }
        }
    }
    decision = decide_fn(res_1d, res_2d, 2.0)
    assert decision["choice"] == "ml_1d"


def test_decision_reports_metric_and_values(decide_fn: Any) -> None:
    """Verify that metric name and values dict are properly populated."""
    v1 = 0.4353
    v2 = 0.5065
    res_1d = {2.0: {"pod_summary": {"pod": None, "binomial": {"pod_lower95": v1}}}}
    res_2d = {2.0: {"pod_summary": {"pod": {"a90_95": 0.7183}, "binomial": {"pod_lower95": v2}}}}
    decision = decide_fn(res_1d, res_2d, 2.0)
    assert decision["metric"] == "binomial pod_lower95"
    assert decision["values"] == {"ml_1d": v1, "ml_2d": v2}
    assert "choice" in decision
    assert "rule" in decision
    assert "reason" in decision


def test_keep_2d_margin_constant(phase4_module: Any) -> None:
    """Module constant KEEP_2D_MARGIN is defined as 0.1."""
    assert getattr(phase4_module, "KEEP_2D_MARGIN", None) == 0.1
