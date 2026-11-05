"""Tests for lightweight hybrid leave-one-crack-out training and evaluation.

Follows TDD, code health (max 30 lines per function, CCN <= 6, max 4 parameters),
and runs on CPU in seconds using synthetic fixtures in tmp_path.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn


class _MockLiteNet(nn.Module):
    """Mock lightweight CNN."""

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.width = width
        self.conv = nn.Conv2d(1, width, kernel_size=3, padding=1)
        self.fc = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).mean(dim=(-2, -1))
        return self.fc(h).squeeze(-1)


class _MockCleanFeatureModel:
    """Mock covariance anomaly model."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None

    def fit(self, feats: np.ndarray) -> _MockCleanFeatureModel:
        arr = np.asarray(feats)
        if arr.size == 0 or arr.shape[0] == 0:
            raise ValueError("Cannot fit on empty features.")
        self.mean_ = np.mean(arr, axis=0)
        return self

    def score(self, feats: np.ndarray) -> np.ndarray:
        arr = np.asarray(feats)
        if arr.size == 0 or arr.shape[0] == 0:
            raise ValueError("Cannot score empty features.")
        if self.mean_ is None:
            raise ValueError("Model not fitted.")
        diff = arr - self.mean_
        return np.sqrt(np.sum(diff**2, axis=-1))


def _mock_count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _mock_augment_batch(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    return x


def _mock_image_features(images: np.ndarray) -> np.ndarray:
    arr = np.asarray(images)
    n = arr.shape[0] if arr.ndim >= 3 else 1
    return np.ones((n, 8), dtype=float)


def _mock_clean_threshold(clean_scores: np.ndarray, target_fcr: float) -> float:
    arr = np.asarray(clean_scores, dtype=float)
    if arr.size == 0 or not (0.0 < target_fcr < 1.0):
        raise ValueError("Invalid clean scores or target_fcr.")
    return float(np.quantile(arr, 1.0 - target_fcr, method="higher"))


def _mock_hybrid_flags(
    p_cnn: np.ndarray,
    anom: np.ndarray,
    t_cnn: float,
    t_anom: float,
) -> np.ndarray:
    return (np.asarray(p_cnn) >= t_cnn) | (np.asarray(anom) >= t_anom)


def _install_mock_lite_hybrid_if_missing() -> None:
    """Install mock utriage.ml.lite_hybrid module if not yet created on disk."""
    try:
        import utriage.ml.lite_hybrid  # noqa: F401
    except ImportError:
        mod = types.ModuleType("utriage.ml.lite_hybrid")
        mod.LiteNet = _MockLiteNet  # type: ignore[attr-defined]
        mod.count_params = _mock_count_params  # type: ignore[attr-defined]
        mod.augment_batch = _mock_augment_batch  # type: ignore[attr-defined]
        mod.CleanFeatureModel = _MockCleanFeatureModel  # type: ignore[attr-defined]
        mod.image_features = _mock_image_features  # type: ignore[attr-defined]
        mod.clean_threshold = _mock_clean_threshold  # type: ignore[attr-defined]
        mod.hybrid_flags = _mock_hybrid_flags  # type: ignore[attr-defined]
        mod.TIME_AXIS = 0  # type: ignore[attr-defined]
        sys.modules["utriage.ml.lite_hybrid"] = mod


_install_mock_lite_hybrid_if_missing()

from experiments.run_lite_hybrid import (  # noqa: E402
    clean_threshold,
    extract_clean_train_features,
    hybrid_flags,
    main,
    split_clean_test_scans,
)


def _write_single_batch(
    path: Path,
    flaws: list[dict[str, Any]],
    labels: list[tuple[float, float]],
) -> Path:
    """Write fake ML-NDT batch binary, metadata jsons, and label text files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(flaws)
    raw = np.zeros((n, 256, 256), dtype=np.uint16)
    raw.tofile(path)
    jsons = "".join(json.dumps({"flaws": [f]}) for f in flaws)
    path.with_suffix(".jsons").write_text(jsons, encoding="utf-8")
    np.savetxt(path.with_suffix(".labels"), np.array(labels, dtype=np.float32), delimiter="\t")
    return path


@pytest.fixture
def fake_batches(tmp_path: Path) -> tuple[list[Path], list[Path]]:
    """Create minimal valid ML-NDT training and validation batches."""
    tr_dir = tmp_path / "data" / "training"
    val_dir = tmp_path / "data" / "validation"
    flaws = [
        {"size": "1.6"},
        {"size": "4.0"},
        {"size": "8.6"},
        {"size": "0.0"},
        {"size": "0.0"},
        {"size": "0.0"},
    ]
    labels = [
        (1.0, 1.6),
        (1.0, 4.0),
        (1.0, 8.6),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ]
    tr_b = [_write_single_batch(tr_dir / f"b{i}.bins", flaws, labels) for i in range(10)]
    val_b = [_write_single_batch(val_dir / f"v{j}.bins", flaws, labels) for j in range(2)]
    return tr_b, val_b


def _check_payload_contents(payload: dict[str, Any]) -> None:
    """Validate top-level keys and identity fields of output payload."""
    expected = [
        "label",
        "preregistration",
        "config",
        "gpu_name",
        "wall_time_s",
        "training_summary",
        "folds",
        "latency",
    ]
    assert all(k in payload for k in expected)
    assert payload["label"] == "Real data (ML-NDT weld, 3 physical cracks)"
    assert payload["preregistration"] == "docs/decisions/0013-lite-hybrid-preregistration.md"


def test_end_to_end_quick_run_payload_keys(
    tmp_path: Path,
    fake_batches: tuple[list[Path], list[Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that end-to-end --quick run writes JSON payload with all expected keys."""
    tr_b, val_b = fake_batches
    monkeypatch.setattr(
        "experiments.run_lite_hybrid.list_batches",
        lambda split: tr_b if split == "training" else val_b,
    )
    out_file = tmp_path / "results" / "metrics.json"
    models_dir = tmp_path / "models"
    argv = [
        "--quick",
        "--device",
        "cpu",
        "--threads",
        "1",
        "--out",
        str(out_file),
        "--models-dir",
        str(models_dir),
    ]
    main(argv)

    assert out_file.exists()
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    _check_payload_contents(payload)


def test_latency_block_fields(
    tmp_path: Path,
    fake_batches: tuple[list[Path], list[Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check that latency block contains cpu_name, threads, note, and timing stats."""
    tr_b, val_b = fake_batches
    monkeypatch.setattr(
        "experiments.run_lite_hybrid.list_batches",
        lambda split: tr_b if split == "training" else val_b,
    )
    out_file = tmp_path / "metrics_lat.json"
    argv = ["--quick", "--device", "cpu", "--threads", "2", "--out", str(out_file)]
    main(argv)

    payload = json.loads(out_file.read_text(encoding="utf-8"))
    lat = payload["latency"]
    assert "cpu_name" in lat
    assert lat["threads"] == 2
    assert "measured on the development machine CPU limited to 2 threads" in lat["note"]
    for model_key in ["litenet", "anomaly", "total", "mlndtnet"]:
        assert model_key in lat
        assert lat[model_key]["median_ms"] >= 0.0
        assert lat[model_key]["p95_ms"] >= 0.0


def test_held_out_crack_never_in_train_or_features(
    fake_batches: tuple[list[Path], list[Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert held-out crack is excluded from training, validation, and feature fitting."""
    tr_b, val_b = fake_batches
    monkeypatch.setattr(
        "experiments.run_lite_hybrid.list_batches",
        lambda split: tr_b if split == "training" else val_b,
    )
    from experiments.run_lite_hybrid import resolve_folds

    folds = resolve_folds(quick=True)
    for fold in folds:
        ds_tr = [(b, idx) for b, idx in fold.train]
        feats = extract_clean_train_features(ds_tr)
        assert feats.shape[0] > 0
        assert fold.held_out not in [b.stem for b, _ in fold.train]
        assert fold.held_out not in [b.stem for b, _ in fold.val]


def test_split_clean_test_scans_disjoint_and_calib_only() -> None:
    """Verify split_clean_test_scans splits 50/50 disjointly for calibration and eval."""
    flaws = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    flawed_idx, calib, ev = split_clean_test_scans(flaws, seed=0)
    assert len(flawed_idx) == 2
    assert len(calib) == 2
    assert len(ev) == 2
    assert set(calib).isdisjoint(set(ev))
    assert set(flawed_idx).isdisjoint(set(calib))


@pytest.mark.parametrize(
    ("p_cnn", "anom", "t_cnn", "t_anom"),
    [
        (np.array([0.1, 0.8, 0.3]), np.array([1.0, 2.0, 5.0]), 0.5, 4.0),
        (np.array([0.2, 0.3]), np.array([0.5, 0.6]), 0.9, 0.9),
        (np.array([0.9, 0.95]), np.array([10.0, 12.0]), 0.1, 1.0),
    ],
)
def test_hybrid_flags_union_bound_sanity(
    p_cnn: np.ndarray,
    anom: np.ndarray,
    t_cnn: float,
    t_anom: float,
) -> None:
    """Verify hybrid flags satisfy boolean OR and union bound k_hyb <= k_cnn + k_anom."""
    flags_hyb = hybrid_flags(p_cnn, anom, t_cnn, t_anom)
    flags_cnn = p_cnn >= t_cnn
    flags_anom = anom >= t_anom
    assert np.array_equal(flags_hyb, flags_cnn | flags_anom)
    assert np.sum(flags_hyb) <= (np.sum(flags_cnn) + np.sum(flags_anom))


def test_clean_threshold_validation() -> None:
    """Ensure clean_threshold raises ValueError on empty input or invalid target FCR."""
    with pytest.raises(ValueError):
        clean_threshold(np.array([]), 0.01)
    with pytest.raises(ValueError):
        clean_threshold(np.array([0.1, 0.5]), 1.5)
