"""Tests for unsupervised anomaly detection pipeline (project plan Phase 8)."""

from __future__ import annotations

import inspect
import json
import sys
import time
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

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from experiments.run_anomaly import (  # noqa: E402
    AnomalyConfig,
    _parse_args,
    map_scores_to_cdf,
    prepare_real_splits,
    prepare_synthetic_data,
    run_anomaly,
)

from utriage.detector_config import DEFAULT_PATH, load_detector_config  # noqa: E402
from utriage.ml.anomaly import anomaly_score, gate_walls  # noqa: E402


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


def test_anomaly_functions_default_parameters() -> None:
    """Verify anomaly.py functions have defaults matching shipped config."""
    sig_gate = inspect.signature(gate_walls)
    assert sig_gate.parameters["guard"].default == 0

    sig_score = inspect.signature(anomaly_score)
    assert sig_score.parameters["smooth_window"].default == 5


@pytest.mark.parametrize(
    ("cli_args", "expected_config", "expected_overrides"),
    [
        ([], DEFAULT_PATH, []),
        (["--config", "my_cfg.yaml"], Path("my_cfg.yaml"), []),
        (["--set", "anomaly.target_fcr=0.05"], DEFAULT_PATH, ["anomaly.target_fcr=0.05"]),
    ],
)
def test_parse_args_config_and_overrides(
    cli_args: list[str],
    expected_config: Path,
    expected_overrides: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI parser correctly parses --config and repeatable --set overrides."""
    monkeypatch.setattr("sys.argv", ["run_anomaly.py", *cli_args])
    args = _parse_args()
    assert args.config == expected_config
    assert args.overrides == expected_overrides


def _assert_pipeline_metrics(metrics: dict[str, Any], loaded: dict[str, Any]) -> None:
    """Verify serialized anomaly metrics payload structure."""
    assert loaded["real"]["label"] == "Real data (ML-NDT)"
    assert loaded["synthetic"]["label"] == "Synthetic data"
    assert "probability_mapping" in loaded["real"]
    assert loaded["run_label"] == "pre-registered"
    assert loaded["detector_config"]["is_default"] is True
    assert len(loaded["detector_config"]["sha256"]) == 64
    _assert_cell_keys(metrics["real"]["overall"])
    for crack in ("crack_1.6", "crack_4.0", "crack_8.6"):
        _assert_cell_keys(metrics["real"]["per_crack"][crack])
    _assert_cell_keys(metrics["synthetic"]["cell"])


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
    assert (time.perf_counter() - t0) < 60.0
    assert out_json.is_file() and m2d.is_file() and m1d.is_file()

    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    _assert_pipeline_metrics(metrics, loaded)


def test_exploratory_config_pipeline_artifacts(
    fake_mlndt_root: Path,
    tmp_path: Path,
) -> None:
    """Verify exploratory config writes run_label and updated target_fcr."""
    out_json = tmp_path / "results" / "anomaly_exploratory.json"
    m2d = tmp_path / "models" / "anomaly_2d.pt"
    m1d = tmp_path / "models" / "anomaly_1d.pt"
    det_cfg = load_detector_config(overrides=["anomaly.target_fcr=0.05", "anomaly.smooth_window=3"])

    cfg = AnomalyConfig(
        root=fake_mlndt_root,
        out_json=out_json,
        model_2d_path=m2d,
        model_1d_path=m1d,
        epochs=1,
        quick=True,
        device="cpu",
        detector_config=det_cfg,
    )

    metrics = run_anomaly(cfg)
    assert metrics["run_label"] == "exploratory"
    assert metrics["detector_config"]["is_default"] is False

    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded["run_label"] == "exploratory"
    assert loaded["detector_config"]["is_default"] is False
    assert loaded["real"]["target_fcr"] == 0.05
    assert loaded["synthetic"]["target_fcr"] == 0.05
