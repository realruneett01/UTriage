"""Tests for Phase 3 report generation, superseded notice, and report-only flag."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="module")
def phase3_module() -> Any:
    """Dynamically load experiments/run_phase3.py."""
    script_path = Path(__file__).resolve().parents[1] / "experiments" / "run_phase3.py"
    spec = importlib.util.spec_from_file_location("run_phase3", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_phase3"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def sample_metrics() -> dict[str, Any]:
    """Load existing Phase 3 metrics as test input."""
    metrics_path = Path(__file__).resolve().parents[1] / "results" / "mlndt_metrics.json"
    with metrics_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_notice_present_when_leakage_true(
    phase3_module: Any,
    tmp_path: Path,
    sample_metrics: dict[str, Any],
) -> None:
    """Verify superseded notice is written when leakage_by_physical_crack is True."""
    with (tmp_path / "leakage_audit.json").open("w", encoding="utf-8") as fh:
        json.dump({"leakage_by_physical_crack": True}, fh)

    out_path = phase3_module.write_report(sample_metrics, tmp_path)
    content = out_path.read_text(encoding="utf-8")
    expected_notice = (
        "Superseded: the random batch split shares all physical cracks between "
        "train and test (results/leakage_audit.json); see results/loco_metrics.json "
        "for leave-one-crack-out results."
    )
    assert expected_notice in content


@pytest.mark.parametrize("leakage_state", [False, "missing"])
def test_notice_absent_when_leakage_false_or_missing(
    phase3_module: Any,
    tmp_path: Path,
    sample_metrics: dict[str, Any],
    leakage_state: Any,
) -> None:
    """Verify superseded notice is omitted when leakage is False or missing."""
    if leakage_state != "missing":
        with (tmp_path / "leakage_audit.json").open("w", encoding="utf-8") as fh:
            json.dump({"leakage_by_physical_crack": leakage_state}, fh)

    out_path = phase3_module.write_report(sample_metrics, tmp_path)
    content = out_path.read_text(encoding="utf-8")
    assert "Superseded:" not in content


def test_report_only_does_not_train(
    phase3_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_metrics: dict[str, Any],
) -> None:
    """Verify --report-only rebuilds report without running model training."""
    res_dir = tmp_path / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    with (res_dir / "mlndt_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(sample_metrics, fh)

    def _fail_training(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Training was invoked during --report-only execution")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(phase3_module, "_train_restarts", _fail_training)
    monkeypatch.setattr(phase3_module, "train_model", _fail_training)
    monkeypatch.setattr(sys, "argv", ["run_phase3.py", "--report-only"])

    phase3_module.main()

    report_path = res_dir / "mlndt_report.md"
    assert report_path.exists()
