"""Tests for fit diagnostics and learning curve analysis (project plan Phase 8)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from experiments.run_fit_diagnostics import (
    FitDiagnosticsConfig,
    FitPayloadComponents,
    _assemble_payload,
    diagnose_loco_folds,
    diagnose_mlndt,
    extract_anomaly_learning_curves,
    extract_loco_learning_curves,
    extract_mlndt_learning_curve,
    plot_data_size_curve,
    run_fit_diagnostics,
)

from utriage.ml.diagnostics import FitRule, diagnose, plot_learning_curve


@pytest.fixture
def fake_mlndt_json(tmp_path: Path) -> Path:
    """Create tiny fake mlndt_metrics.json with restart history."""
    p = tmp_path / "mlndt_metrics.json"
    data = {
        "label": "Real data (ML-NDT)",
        "kept_seed": 1,
        "restarts": [
            {
                "seed": 0,
                "best_epoch": 2,
                "best_val_loss": 0.05,
                "history": [
                    {"epoch": 1, "train_loss": 0.20, "val_loss": 0.15},
                    {"epoch": 2, "train_loss": 0.10, "val_loss": 0.05},
                ],
            },
            {
                "seed": 1,
                "best_epoch": 2,
                "best_val_loss": 0.02,
                "history": [
                    {"epoch": 1, "train_loss": 0.18, "val_loss": 0.08},
                    {"epoch": 2, "train_loss": 0.05, "val_loss": 0.02},
                ],
            },
        ],
    }
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def mock_synthetic_curve_data() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate fake synthetic data size records and diagnosis."""
    sample_record = {
        "fraction": 1.0,
        "n_train_samples": 50,
        "train": {"detection_rate": 0.95, "false_call_rate": 0.01},
        "test_sets": {
            "in_distribution": {"detection_rate": 0.92, "false_call_rate": 0.01},
            "freq_10mhz": {"detection_rate": 0.88, "false_call_rate": 0.02},
            "material_third": {"detection_rate": 0.86, "false_call_rate": 0.02},
            "noise_above_range": {"detection_rate": 0.85, "false_call_rate": 0.03},
        },
    }
    sample_diag = {
        "available": True,
        "train_metric": 0.95,
        "verdict": "ok",
        "rule": (
            "underfitting if train metric < 0.9; "
            "overfitting if (train metric - group-held-out metric) > 0.1; else ok."
        ),
        "sets": {"in_distribution": {"heldout_metric": 0.92, "gap": 0.03, "verdict": "ok"}},
    }
    return [sample_record], sample_diag


def test_missing_inputs_produce_available_false(tmp_path: Path) -> None:
    """Verify missing metrics files report available=False with a clear reason."""
    missing = tmp_path / "non_existent.json"
    mlndt_res = extract_mlndt_learning_curve(missing)
    loco_res = extract_loco_learning_curves(missing)
    anom_res = extract_anomaly_learning_curves(missing)

    assert mlndt_res["available"] is False
    assert "not found" in mlndt_res["reason"].lower()
    assert loco_res["available"] is False
    assert "not found" in loco_res["reason"].lower()
    assert anom_res["available"] is False
    assert "not found" in anom_res["reason"].lower()


def test_extract_mlndt_learning_curve(fake_mlndt_json: Path) -> None:
    """Verify ML-NDT learning curve is extracted from kept restart and marked leaky."""
    res = extract_mlndt_learning_curve(fake_mlndt_json)
    assert res["available"] is True
    assert res["kept_seed"] == 1
    assert "leaky" in res["note"].lower()
    curve = res["curve"]
    assert curve["epoch"] == [1, 2]
    assert curve["train_loss"] == [0.18, 0.05]
    assert curve["best_epoch"] == 2


def _make_loco_fold_dict(has_train_det: bool, has_history: bool) -> dict[str, Any]:
    """Construct mock fold dictionary with optional history and train detection."""
    fold_entry: dict[str, Any] = {
        "held_out": "crack_B",
        "best_val_loss": 0.03,
        "detections": {"rate": 0.88, "n_hit": 44, "n_flawed": 50},
    }
    if has_history:
        fold_entry["history"] = [
            {"epoch": 1, "train_loss": 0.1, "val_loss": 0.08},
            {"epoch": 2, "train_loss": 0.05, "val_loss": 0.03},
        ]
    if has_train_det:
        fold_entry["train_detection"] = {"rate": 0.94, "n_hit": 47, "n_flawed": 50}
    return fold_entry


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ((False, False), (False, False)),
        ((True, True), (True, True)),
    ],
)
def test_loco_metrics_handling(
    tmp_path: Path,
    flags: tuple[bool, bool],
    expected: tuple[bool, bool],
) -> None:
    """Verify LOCO fold history and detection gap handling."""
    has_train_det, has_history = flags
    expected_curve_avail, expected_gap_avail = expected
    rule = FitRule(target_train_metric=0.9, max_gap=0.1)
    fold_entry = _make_loco_fold_dict(has_train_det, has_history)
    loco_file = tmp_path / "loco_test.json"
    loco_file.write_text(json.dumps({"folds": [fold_entry]}), encoding="utf-8")

    curve_res = extract_loco_learning_curves(loco_file)
    assert curve_res["available"] is expected_curve_avail

    loaded = json.loads(loco_file.read_text(encoding="utf-8"))
    gap_res = diagnose_loco_folds(loaded, rule)
    assert gap_res["available"] is expected_gap_avail


@pytest.mark.parametrize(
    ("train_metric", "heldout_metric", "expected_verdict", "expected_gap"),
    [
        (0.85, 0.80, "underfitting", 0.05),
        (0.95, 0.80, "overfitting", 0.15),
        (0.95, 0.90, "ok", 0.05),
    ],
)
def test_verdict_computed_via_diagnose(
    train_metric: float,
    heldout_metric: float,
    expected_verdict: str,
    expected_gap: float,
) -> None:
    """Verify FitRule evaluation produces correct verdicts and gaps."""
    rule = FitRule(target_train_metric=0.9, max_gap=0.1)
    res = diagnose(train_metric, {"test_set": heldout_metric}, rule)
    assert res["verdict"] == expected_verdict
    set_eval = res["sets"]["test_set"]
    assert pytest.approx(set_eval["gap"], abs=1e-5) == expected_gap
    assert set_eval["verdict"] == expected_verdict
    assert rule.rule_text in res["rule"]


def test_diagnose_mlndt_handling(fake_mlndt_json: Path) -> None:
    """Verify diagnose_mlndt handles missing train metric with available=False."""
    rule = FitRule(target_train_metric=0.9, max_gap=0.1)
    data = json.loads(fake_mlndt_json.read_text(encoding="utf-8"))
    res = diagnose_mlndt(data, rule)
    assert res["available"] is False
    assert "not recorded" in res["reason"].lower()


def test_learning_curve_plot_written(tmp_path: Path) -> None:
    """Verify learning curve PNG is written to disk with content."""
    curve = {
        "epoch": [1, 2, 3],
        "train_loss": [0.3, 0.2, 0.1],
        "val_loss": [0.35, 0.22, 0.12],
        "best_epoch": 3,
        "final_gap": 0.02,
    }
    lc_path = tmp_path / "fit_lc_test.png"
    out_lc = plot_learning_curve(curve, lc_path, "Test Curve")
    assert out_lc.exists()
    assert out_lc.stat().st_size > 0


def test_data_size_plot_written(tmp_path: Path) -> None:
    """Verify data size curve PNG is written to disk with content."""
    data_size_records = [
        {
            "fraction": 0.5,
            "test_sets": {
                "in_distribution": {"detection_rate": 0.85, "false_call_rate": 0.02},
                "freq_10mhz": {"detection_rate": 0.75, "false_call_rate": 0.03},
            },
        },
        {
            "fraction": 1.0,
            "test_sets": {
                "in_distribution": {"detection_rate": 0.92, "false_call_rate": 0.01},
                "freq_10mhz": {"detection_rate": 0.82, "false_call_rate": 0.02},
            },
        },
    ]
    ds_path = tmp_path / "fit_data_size_test.png"
    out_ds = plot_data_size_curve(data_size_records, ds_path)
    assert out_ds.exists()
    assert out_ds.stat().st_size > 0


def test_full_pipeline_mocked(
    tmp_path: Path,
    mock_synthetic_curve_data: tuple[list[dict[str, Any]], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run run_fit_diagnostics pipeline with mocked training in tmp_path."""
    mock_run = MagicMock(return_value=mock_synthetic_curve_data)
    monkeypatch.setattr(
        "experiments.run_fit_diagnostics._run_synthetic_data_size_curve",
        mock_run,
    )
    out_json = tmp_path / "fit_diagnostics.json"
    cfg = FitDiagnosticsConfig(
        mlndt_metrics_path=tmp_path / "missing_mlndt.json",
        loco_metrics_path=tmp_path / "missing_loco.json",
        anomaly_metrics_path=tmp_path / "missing_anomaly.json",
        out_json=out_json,
        figures_dir=tmp_path / "figures",
        quick=True,
    )
    result = run_fit_diagnostics(cfg)
    assert out_json.exists()
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert "fit_rule" in loaded
    assert "learning_curves" in loaded
    assert "data_size_curve" in loaded
    assert "diagnostics" in loaded
    assert loaded["diagnostics"]["synthetic_supervised"]["available"] is True
    assert loaded["diagnostics"]["synthetic_supervised"]["verdict"] == "ok"
    assert result == loaded


@pytest.mark.parametrize(
    ("payload", "expected_available", "expect_reason_substring"),
    [
        (
            {"real": {"learning_curve": {"val_loss": [0.1, 0.05]}}, "synthetic": {}},
            True,
            None,
        ),
        (
            {
                "real": {"history": [{"epoch": 1, "train_loss": 0.2, "val_loss": 0.1}]},
                "synthetic": {},
            },
            True,
            None,
        ),
        (
            {"real": {"learning_curve": None}, "synthetic": {"learning_curve": None}},
            False,
            "learning_curve is null",
        ),
    ],
)
def test_extract_anomaly_learning_curves_formats(
    tmp_path: Path,
    payload: dict[str, Any],
    expected_available: bool,
    expect_reason_substring: str | None,
) -> None:
    """Verify anomaly learning curve parsing across dict, list, and null cases."""
    p = tmp_path / "anomaly_test.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    res = extract_anomaly_learning_curves(p)
    assert res["available"] is expected_available
    if expect_reason_substring:
        assert expect_reason_substring in res["reason"]
    else:
        assert res["reason"] is None


def test_assemble_payload() -> None:
    """Verify _assemble_payload builds structured payload from FitPayloadComponents."""
    rule = FitRule(target_train_metric=0.9, max_gap=0.1)
    comp = FitPayloadComponents(
        lc_data={"mlndt": {"available": False}},
        ds_records=[{"fraction": 1.0}],
        synth_diag={"verdict": "ok"},
        rule=rule,
        fig_paths=["fig.png"],
        loco_diag={"available": False},
        mlndt_diag={"available": False},
    )
    payload = _assemble_payload(comp)
    assert payload["rule"] == rule.rule_text
    assert payload["data_size_curve"] == [{"fraction": 1.0}]
    assert payload["diagnostics"]["synthetic_supervised"]["verdict"] == "ok"
    assert payload["figures"] == ["fig.png"]
