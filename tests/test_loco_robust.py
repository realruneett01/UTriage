"""Unit tests for robust leave-one-crack-out training and evaluation.

Verifies data augmentations, holdout integrity, clean-calibrated thresholding,
and end-to-end execution of the robust LOCO pipeline on CPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from experiments.run_loco_robust import (  # noqa: E402
    apply_augmentation,
    apply_gain,
    apply_noise,
    apply_offset,
    apply_scan_flip,
    apply_time_shift,
    calibrate_threshold,
    evaluate_detection_and_fcr,
    main,
    split_clean_scores,
)

from utriage.ml.datasets import list_batches  # noqa: E402
from utriage.ml.loco import PHYSICAL_CRACKS, make_folds  # noqa: E402
from utriage.ml.mlndt_meta import image_index  # noqa: E402


def _create_fake_batch(
    batch_path: Path,
    flaws: list[dict[str, Any]],
    labels: list[tuple[float, float]],
    seed: int,
) -> Path:
    """Write fake .bins, .jsons, and .labels files for a single batch."""
    rng = np.random.default_rng(seed)
    raw = rng.integers(100, 3000, size=(len(flaws), 256, 256), dtype=np.uint16)
    raw.tofile(batch_path)

    jsons_text = "".join(json.dumps({"flaws": [f]}) for f in flaws)
    batch_path.with_suffix(".jsons").write_text(jsons_text, encoding="utf-8")

    labels_arr = np.array(labels, dtype=np.float32)
    np.savetxt(batch_path.with_suffix(".labels"), labels_arr, delimiter="\t")
    return batch_path


@pytest.fixture
def fake_mlndt_root(tmp_path: Path) -> Path:
    """Populate fake ML-NDT root with training and validation batches."""
    root = tmp_path / "fake_mlndt"
    tr_dir, val_dir = root / "training", root / "validation"
    tr_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    flaws = [{"size": "1.6"}, {"size": "4.0"}, {"size": "8.6"}, {"size": "0.0"}]
    labels = [(1.0, 1.6), (1.0, 4.0), (1.0, 8.6), (0.0, 0.0)]

    for i in range(10):
        _create_fake_batch(tr_dir / f"batch_tr_{i:02d}.bins", flaws, labels, seed=i)
    for j in range(2):
        _create_fake_batch(val_dir / f"batch_val_{j:02d}.bins", flaws, labels, seed=100 + j)
    return root


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((2, 1, 64, 64), torch.float32),
        ((1, 1, 32, 32), torch.float32),
        ((3, 1, 64, 64), torch.float16),
    ],
)
def test_augmentation_keeps_shape_and_dtype(
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    """Verify data augmentation preserves tensor shape and dtype."""
    gen = torch.Generator().manual_seed(42)
    x = torch.randn(shape, dtype=dtype)
    out = apply_augmentation(x, gen)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_augmentation_determinism() -> None:
    """Verify same seed produces identical augmented outputs."""
    x = torch.randn(4, 1, 64, 64, dtype=torch.float32)
    gen1 = torch.Generator().manual_seed(123)
    gen2 = torch.Generator().manual_seed(123)
    gen3 = torch.Generator().manual_seed(456)

    out1 = apply_augmentation(x, gen1)
    out2 = apply_augmentation(x, gen2)
    out3 = apply_augmentation(x, gen3)

    assert torch.equal(out1, out2)
    assert not torch.equal(out1, out3)


def test_gain_range_respected() -> None:
    """Verify gain scaling is strictly within [0.5, 2.0]."""
    gen = torch.Generator().manual_seed(99)
    x = torch.ones(200, 1, 16, 16, dtype=torch.float32)
    out = apply_gain(x, gen, gain_min=0.5, gain_max=2.0)
    gains = out[:, 0, 0, 0]

    assert torch.all(gains >= 0.5)
    assert torch.all(gains <= 2.0)
    assert float(gains.min()) < 0.6
    assert float(gains.max()) > 1.9


def test_offset_range_respected() -> None:
    """Verify DC offset is within [-0.2 * std, 0.2 * std]."""
    gen = torch.Generator().manual_seed(101)
    x = torch.randn(200, 1, 32, 32, dtype=torch.float32) * 5.0 + 10.0
    stds = x.std(dim=(-2, -1), keepdim=True)
    out = apply_offset(x, gen, scale=0.2)
    diff = out - x

    max_allowed = 0.2 * stds + 1e-5
    assert torch.all(diff.abs() <= max_allowed)
    assert float(diff.abs().max()) > 0.1 * 5.0


def test_noise_range_respected() -> None:
    """Verify additive Gaussian noise std does not exceed 0.1 * std."""
    gen = torch.Generator().manual_seed(202)
    x = torch.randn(50, 1, 64, 64, dtype=torch.float32) * 3.0
    stds = x.std(dim=(-2, -1), keepdim=True)
    out = apply_noise(x, gen, scale=0.1)
    noise = out - x

    noise_stds = noise.std(dim=(-2, -1), keepdim=True)
    assert torch.all(noise_stds <= 0.11 * stds)


def test_time_shift_along_time_axis() -> None:
    """Verify time shift rolls along rows (dim -2) up to +-16 samples."""
    gen = torch.Generator().manual_seed(303)
    x = torch.zeros(1, 1, 64, 64, dtype=torch.float32)
    x[0, 0, 32, :] = 1.0  # single row active
    out = apply_time_shift(x, gen, max_shift=16)

    active_rows = torch.where(out[0, 0, :, 0] == 1.0)[0]
    assert len(active_rows) == 1
    shift = int(active_rows[0].item()) - 32
    assert -16 <= shift <= 16


def test_scan_flip_along_scan_axis() -> None:
    """Verify scan flip inverts columns (dim -1) without altering rows."""
    gen = torch.Generator().manual_seed(404)
    x = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 8).repeat(1, 1, 8, 1)
    out = apply_scan_flip(x, gen)

    forward = x
    reversed_x = torch.flip(x, dims=[-1])
    is_fwd = torch.equal(out, forward)
    is_rev = torch.equal(out, reversed_x)
    assert is_fwd or is_rev


@pytest.mark.parametrize("crack", PHYSICAL_CRACKS)
def test_held_out_crack_never_in_train_or_val(
    fake_mlndt_root: Path,
    crack: str,
) -> None:
    """Verify held-out crack scans never appear in fold train or val splits."""
    tr = list_batches("training", root=fake_mlndt_root)
    val = list_batches("validation", root=fake_mlndt_root)
    folds = make_folds(tr, val, seed=0, val_mode="batch")
    crack_map = {(b, idx): c for b, idx, c in image_index(tr + val)}

    fold = next(f for f in folds if f.held_out == crack)
    assert all(crack_map[it] != crack for it in fold.train)
    assert all(crack_map[it] != crack for it in fold.val)


def test_threshold_from_clean_only() -> None:
    """Verify threshold calculation depends only on clean calibration scans."""
    clean_scores = np.linspace(0.01, 0.80, 100, dtype=np.float32)
    calib, ev = split_clean_scores(clean_scores)
    th_01 = calibrate_threshold(calib, 0.01)
    th_05 = calibrate_threshold(calib, 0.05)

    assert th_01 >= th_05
    assert th_01 == float(np.quantile(calib, 0.99, method="higher"))

    # Changing flawed scores must not change thresholds
    flawed_a = np.array([0.85, 0.95], dtype=np.float32)
    flawed_b = np.array([0.1, 0.2], dtype=np.float32)
    res_a = evaluate_detection_and_fcr(flawed_a, ev, th_01, 0.01)
    res_b = evaluate_detection_and_fcr(flawed_b, ev, th_01, 0.01)

    assert res_a["threshold"] == res_b["threshold"]
    assert res_a["detections"]["rate"] == 1.0
    assert res_b["detections"]["rate"] == 0.0


def _assert_metrics_payload(data: dict[str, Any]) -> None:
    """Validate JSON payload keys and structure."""
    assert data["label"] == "Real data (ML-NDT weld, 3 physical cracks)"
    assert data["preregistration"] == "docs/decisions/0012-robust-loco-preregistration.md"
    assert "wall_time_seconds" in data
    assert len(data["folds"]) == 3
    for f in data["folds"]:
        assert "ensemble_metrics" in f
        em = f["ensemble_metrics"]
        assert all(k in em for k in ("auroc", "average_precision", "fcr_0.01", "fcr_0.05"))
        assert len(f["models"]) == 1


def _make_quick_argv(out_json: Path, models_dir: Path) -> list[str]:
    """Construct CLI argument list for quick test."""
    return [
        "--quick",
        "--device",
        "cpu",
        "--epochs",
        "1",
        "--seeds",
        "0",
        "--batch-size",
        "16",
        "--out",
        str(out_json),
        "--models-dir",
        str(models_dir),
    ]


def test_quick_path_end_to_end(
    fake_mlndt_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify --quick path runs end-to-end on synthetic data within seconds."""
    monkeypatch.setattr(
        "experiments.run_loco_robust.list_batches",
        lambda split, root=None: list_batches(split, root=fake_mlndt_root),
    )
    out_json = tmp_path / "loco_robust_metrics.json"
    models_dir = tmp_path / "models"
    main(_make_quick_argv(out_json, models_dir))

    assert out_json.exists()
    _assert_metrics_payload(json.loads(out_json.read_text(encoding="utf-8")))
    assert all((models_dir / f"robust_{c}_s0.pt").exists() for c in PHYSICAL_CRACKS)
