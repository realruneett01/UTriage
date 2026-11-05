"""Unit tests for LOCO separability evaluation experiment (Phase 8).

Verifies threshold-free AUROC/AP metrics, deterministic clean calibration split,
clean-only threshold derivation, graceful missing model skipping, and payload schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from experiments.run_separability import (
    CleanSplit,
    _build_payload,
    _evaluate_protocol,
    compute_auroc,
    compute_average_precision,
    compute_operating_point,
    evaluate_fold_separability,
    main,
    split_clean_scans,
)

from utriage.ml.loco import LocoFold


@pytest.fixture
def synthetic_scores() -> tuple[np.ndarray, np.ndarray]:
    """Return synthetic binary labels and separated prediction scores."""
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=int)
    y_score = np.array([0.9, 0.85, 0.8, 0.75, 0.2, 0.15, 0.1, 0.05], dtype=float)
    return y_true, y_score


def test_auroc_separated_fake_scores(synthetic_scores: tuple[np.ndarray, np.ndarray]) -> None:
    """Verify AUROC is 1.0 for perfectly separated predictions."""
    y_true, y_score = synthetic_scores
    auroc = compute_auroc(y_true, y_score)
    avg_prec = compute_average_precision(y_true, y_score)
    assert auroc == pytest.approx(1.0)
    assert avg_prec == pytest.approx(1.0)


def test_auroc_identical_distributions() -> None:
    """Verify AUROC is ~0.5 when flaw and clean scores share identical distribution."""
    rng = np.random.default_rng(seed=123)
    n_samples = 400
    y_true = np.array([1] * n_samples + [0] * n_samples, dtype=int)
    y_score = rng.normal(loc=0.0, scale=1.0, size=2 * n_samples)
    auroc = compute_auroc(y_true, y_score)
    assert auroc == pytest.approx(0.5, abs=0.08)


def test_auroc_single_class_returns_nan() -> None:
    """Verify AUROC and AP return NaN when only one class is present."""
    y_true = np.ones(10, dtype=int)
    y_score = np.linspace(0.1, 0.9, 10)
    assert np.isnan(compute_auroc(y_true, y_score))
    assert np.isnan(compute_average_precision(y_true, y_score))


def test_clean_split_disjoint_and_deterministic() -> None:
    """Verify clean calib/eval split is strictly disjoint and deterministic by seed."""
    clean_indices = np.arange(100, dtype=int)
    split_a: CleanSplit = split_clean_scans(clean_indices, seed=42)
    split_b: CleanSplit = split_clean_scans(clean_indices, seed=42)
    split_diff: CleanSplit = split_clean_scans(clean_indices, seed=99)

    assert set(split_a.calib_indices).isdisjoint(set(split_a.eval_indices))
    assert len(split_a.calib_indices) + len(split_a.eval_indices) == len(clean_indices)
    assert np.array_equal(split_a.calib_indices, split_b.calib_indices)
    assert np.array_equal(split_a.eval_indices, split_b.eval_indices)
    assert not np.array_equal(split_a.calib_indices, split_diff.calib_indices)


@pytest.mark.parametrize(
    "flawed_shift",
    [0.0, -0.8, 5.0],
    ids=["baseline", "lowered_flawed", "elevated_flawed"],
)
def test_threshold_uses_only_clean_scores(flawed_shift: float) -> None:
    """Verify operating threshold is invariant to flawed image score values."""
    calib_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)
    eval_scores = np.array([0.15, 0.25, 0.35], dtype=float)
    flawed_scores = np.array([0.8, 0.9], dtype=float) + flawed_shift

    op = compute_operating_point(calib_scores, eval_scores, flawed_scores, target_fcr=0.05)
    expected_th = float(np.quantile(calib_scores, 0.95, method="higher"))
    assert op["threshold"] == pytest.approx(expected_th)
    assert op["false_calls"]["n"] == len(eval_scores)


def test_evaluate_fold_separability_metrics(
    synthetic_scores: tuple[np.ndarray, np.ndarray],
) -> None:
    """Verify evaluate_fold_separability returns required metric keys and valid CI bounds."""
    y_true, y_score = synthetic_scores
    res = evaluate_fold_separability(y_true, y_score, seed=0)

    assert res["n_flawed"] == 4
    assert res["n_clean"] == 4
    assert res["auroc"] == pytest.approx(1.0)
    assert res["average_precision"] == pytest.approx(1.0)
    assert "0.01" in res["operating_points"]
    assert "0.05" in res["operating_points"]

    op_05 = res["operating_points"]["0.05"]
    assert 0.0 <= op_05["detection"]["ci_low"] <= op_05["detection"]["ci_high"] <= 1.0
    assert 0.0 <= op_05["false_calls"]["ci_low"] <= op_05["false_calls"]["ci_high"] <= 1.0


def test_missing_model_files_skipped(tmp_path: Path) -> None:
    """Verify missing model files result in skipped protocol status with informative note."""
    empty_models = tmp_path / "models"
    empty_models.mkdir()
    fold = LocoFold(held_out="crack_1.6", train=[], val=[], test=[])

    from experiments.run_separability import EvalContext

    ctx = EvalContext(models_dir=empty_models, out_path=tmp_path / "out.json")
    res = _evaluate_protocol("batchval", [fold], ctx)

    assert res["status"] == "skipped"
    assert "not found" in res["note"]
    assert res["folds"] == []


def test_payload_structure_and_keys() -> None:
    """Verify complete payload conforms to expected schema and exact label/reading strings."""
    fake_protocols = {
        "batchval": {
            "description": "batch desc",
            "status": "evaluated",
            "folds": [{"held_out": "crack_1.6", "model_sha256": ["sha_1a", "sha_1b"]}],
        },
        "crackval": {
            "description": "crack desc",
            "status": "skipped",
            "note": "missing",
            "folds": [],
        },
    }
    payload = _build_payload(fake_protocols, seed=7)

    assert payload["label"] == "Real data (ML-NDT weld, 3 physical cracks)"
    assert payload["reading"] == (
        "AUROC 0.5 = no separation, 1.0 = perfect ranking; thresholds set on clean scans only"
    )
    assert payload["seed"] == 7
    assert "batchval" in payload["protocol_descriptions"]
    assert "crackval" in payload["protocol_descriptions"]
    assert payload["protocols"] == fake_protocols
    assert payload["model_sha256"] == ["sha_1a", "sha_1b"]


def test_main_cli_missing_models_graceful(tmp_path: Path) -> None:
    """Verify main CLI executes cleanly with empty models directory, outputting valid JSON."""
    empty_models = tmp_path / "models"
    empty_models.mkdir()
    out_json = tmp_path / "results" / "separability_metrics.json"

    argv = [
        "--models-dir",
        str(empty_models),
        "--out",
        str(out_json),
        "--device",
        "cpu",
        "--protocols",
        "batchval",
        "crackval",
        "--seed",
        "0",
        "--quick",
    ]
    payload = main(argv)

    assert out_json.exists()
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded["label"] == payload["label"]
    assert loaded["protocols"]["batchval"]["status"] == "skipped"
    assert loaded["protocols"]["crackval"]["status"] == "skipped"
