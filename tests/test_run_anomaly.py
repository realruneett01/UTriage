"""Tests for unsupervised anomaly detection pipeline (project plan Phase 8)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from experiments.run_anomaly import (
    AnomalyConfig,
    map_scores_to_cdf,
    prepare_real_splits,
    prepare_synthetic_data,
    run_anomaly,
)


def _create_fake_batch(
    batch_path: Path,
    flaws: list[dict[str, Any]],
    labels: list[tuple[float, float]],
    seed: int,
) -> Path:
    """Write fake .bins, .jsons, and .labels files for a single batch."""
    rng = np.random.default_rng(seed)
    n = len(flaws)
    raw = rng.integers(100, 3000, size=(n, 256, 256), dtype=np.uint16)
    raw.tofile(batch_path)

    jsons_text = "".join(json.dumps({"flaws": [f]}) for f in flaws)
    batch_path.with_suffix(".jsons").write_text(jsons_text, encoding="utf-8")

    labels_arr = np.array(labels, dtype=np.float32)
    np.savetxt(batch_path.with_suffix(".labels"), labels_arr, delimiter="\t")
    return batch_path


@pytest.fixture
def fake_mlndt_root(tmp_path: Path) -> Path:
    """Populate minimal fake ML-NDT root with training and validation batches."""
    root = tmp_path / "fake_mlndt"
    tr_dir = root / "training"
    val_dir = root / "validation"
    tr_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    flaws = [{"size": "1.6"}, {"size": "4.0"}, {"size": "8.6"}, {"size": "0.0"}]
    labels = [(1.0, 1.6), (1.0, 4.0), (1.0, 8.6), (0.0, 0.0)]

    for i in range(5):
        _create_fake_batch(tr_dir / f"batch_tr_{i:02d}.bins", flaws, labels, seed=i)
    for j in range(2):
        _create_fake_batch(val_dir / f"batch_val_{j:02d}.bins", flaws, labels, seed=100 + j)
    return root


def test_real_splits_training_never_sees_flaw_and_disjoint(fake_mlndt_root: Path) -> None:
    """Training item list has no flawed images, and calibration is disjoint from test."""
    cfg = AnomalyConfig(root=fake_mlndt_root, quick=False)
    splits = prepare_real_splits(cfg)

    # 1. Training never sees a flawed image
    assert splits.train_flaw_count == 0
    assert len(splits.train_items) > 0

    # 2. Calibration set is disjoint from test set
    cal_set = set(splits.cal_items)
    test_set = set(splits.test_unflawed_items + splits.all_flawed_items)
    assert cal_set.isdisjoint(test_set)
    assert set(splits.train_items).isdisjoint(test_set)


@pytest.mark.parametrize(
    ("scores", "expected_bounds"),
    [
        (np.array([0.0, 0.5, 1.0, 2.0]), (0.0, 1.0)),
        (np.array([-1.0, 10.0]), (0.0, 1.0)),
    ],
)
def test_map_scores_to_cdf_bounds(scores: np.ndarray, expected_bounds: tuple[float, float]) -> None:
    """Scores mapped through empirical CDF lie within [0.0, 1.0]."""
    cal_scores = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    probs = map_scores_to_cdf(scores, cal_scores)
    assert float(np.min(probs)) >= expected_bounds[0]
    assert float(np.max(probs)) <= expected_bounds[1]


def test_synthetic_data_clean_disjointness() -> None:
    """Synthetic training clean windows are disjoint from calibration and test sets."""
    tr_clean, cal_clean, test_flaw, test_clean = prepare_synthetic_data(quick=True, seed=42)
    assert len(tr_clean) > 0
    assert len(cal_clean) > 0
    assert len(test_flaw) > 0
    assert len(test_clean) > 0
    assert tr_clean.shape[1:] == (1, 128)


def _assert_cell_keys(cell_data: dict[str, Any]) -> None:
    """Verify standard pod.metrics cell keys are present and typed."""
    for key in ("detection", "false_calls", "ece", "reliability", "pod", "counts"):
        assert key in cell_data


def test_quick_pipeline_execution_and_artifacts(
    fake_mlndt_root: Path,
    tmp_path: Path,
) -> None:
    """Run quick anomaly pipeline on CPU in tmp_path under 60 s without touching data/."""
    out_json = tmp_path / "results" / "anomaly_metrics.json"
    m2d = tmp_path / "models" / "anomaly_2d.pt"
    m1d = tmp_path / "models" / "anomaly_1d.pt"

    cfg = AnomalyConfig(
        root=fake_mlndt_root,
        out_json=out_json,
        model_2d_path=m2d,
        model_1d_path=m1d,
        epochs=1,
        quick=True,
        device="cpu",
    )

    t0 = time.perf_counter()
    metrics = run_anomaly(cfg)
    elapsed = time.perf_counter() - t0

    assert elapsed < 60.0
    assert out_json.is_file()
    assert m2d.is_file()
    assert m1d.is_file()

    # Verify JSON serializability
    text = out_json.read_text(encoding="utf-8")
    loaded = json.loads(text)
    assert loaded["real"]["label"] == "Real data (ML-NDT)"
    assert loaded["synthetic"]["label"] == "Synthetic data"
    assert "probability_mapping" in loaded["real"]

    _assert_cell_keys(metrics["real"]["overall"])
    for crack in ("crack_1.6", "crack_4.0", "crack_8.6"):
        _assert_cell_keys(metrics["real"]["per_crack"][crack])
    _assert_cell_keys(metrics["synthetic"]["cell"])
