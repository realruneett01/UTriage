"""Tests for unsupervised site adaptation and calibration (project plan Phase 0 / ADR 0010).

Synthetic data. Fixed random seeds for deterministic verification.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        del sys.modules["utriage"]

from utriage.ml.site_adapt import (  # noqa: E402
    SiteCalibration,
    SiteStats,
    apply_site_stats,
    calibrate_site_threshold,
    fit_site_stats,
    load_site_profile,
    min_clean_scans,
    recalibrate_batchnorm,
    save_site_profile,
)


class _SmallConvBNNet(nn.Module):
    """Minimal CNN with Conv2d and BatchNorm2d for testing AdaBN."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


@pytest.mark.parametrize(
    ("fcr", "conf", "expected"),
    [
        (0.01, 0.95, 299),
        (0.05, 0.95, 59),
        (0.10, 0.95, 29),
    ],
)
def test_min_clean_scans_exact_values(fcr: float, conf: float, expected: int) -> None:
    """min_clean_scans yields exact theoretical sample sizes."""
    assert min_clean_scans(fcr, confidence=conf) == expected


@pytest.mark.parametrize("bad_fcr", [-0.01, 0.0, 1.0, 1.2])
def test_min_clean_scans_invalid_rate(bad_fcr: float) -> None:
    """min_clean_scans raises ValueError for rates outside (0, 1)."""
    with pytest.raises(ValueError, match="false_call_rate"):
        min_clean_scans(bad_fcr)


def test_calibrate_site_threshold_warns_when_too_few_scans() -> None:
    """calibrate_site_threshold raises UserWarning when n_clean < min_clean_scans."""
    rng = np.random.default_rng(123)
    scores = rng.uniform(0.0, 1.0, size=50)
    with pytest.warns(UserWarning, match="below recommended minimum"):
        calib = calibrate_site_threshold(scores, false_call_rate=0.01)
    assert calib.n_clean == 50


def test_calibrate_site_threshold_no_warning_when_sufficient() -> None:
    """calibrate_site_threshold runs silently without warning when scans are sufficient."""
    rng = np.random.default_rng(123)
    scores = rng.uniform(0.0, 1.0, size=350)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        calib = calibrate_site_threshold(scores, false_call_rate=0.01)
    assert calib.n_clean == 350
    assert calib.upper95_fcr <= 0.03


def test_calibrate_site_threshold_fcr_property_large_sample() -> None:
    """Threshold isolates target empirical FCR on 10,000 clean scores."""
    rng = np.random.default_rng(42)
    scores = rng.normal(loc=5.0, scale=1.5, size=10000)
    target_fcr = 0.02
    calib = calibrate_site_threshold(scores, false_call_rate=target_fcr)

    emp_fcr = float(np.mean(scores > calib.threshold))
    assert pytest.approx(target_fcr, abs=0.005) == emp_fcr
    assert calib.upper95_fcr >= emp_fcr


def test_adabn_shifts_running_mean_and_preserves_weights() -> None:
    """AdaBN updates running statistics toward shifted clean scans leaving weights unchanged."""
    torch.manual_seed(42)
    model = _SmallConvBNNet()
    orig_conv_weight = model.conv.weight.detach().clone()
    orig_mean = model.bn.running_mean.detach().clone()

    shift_val = 8.0
    loader = [torch.randn(8, 1, 16, 16) + shift_val for _ in range(4)]
    recalibrate_batchnorm(model, loader, device="cpu")

    assert not model.training
    assert torch.equal(model.conv.weight, orig_conv_weight)
    assert not torch.equal(model.bn.running_mean, orig_mean)
    assert float(model.bn.running_mean.mean()) > 0.0


def test_recalibrate_batchnorm_without_bn_leaves_model_unchanged() -> None:
    """A model lacking BatchNorm modules is returned completely unchanged."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU())
    orig_weight = model[0].weight.detach().clone()
    loader = [torch.randn(4, 8) + 5.0 for _ in range(2)]

    out_model = recalibrate_batchnorm(model, loader, device="cpu")
    assert out_model is model
    assert torch.equal(model[0].weight, orig_weight)


def test_fit_and_apply_site_stats() -> None:
    """fit_site_stats captures mean and std; apply_site_stats standardises images."""
    rng = np.random.default_rng(99)
    raw_images = [
        rng.normal(loc=12.0, scale=3.0, size=(16, 16)).astype(np.float32) for _ in range(6)
    ]

    stats = fit_site_stats(raw_images, label="Synthetic")
    assert stats.n_scans == 6
    assert pytest.approx(12.0, abs=0.5) == stats.mean
    assert pytest.approx(3.0, abs=0.5) == stats.std

    z_np = apply_site_stats(raw_images[0], stats)
    assert pytest.approx(0.0, abs=0.6) == float(np.mean(z_np))
    assert pytest.approx(1.0, abs=0.3) == float(np.std(z_np))

    tensor_input = torch.from_numpy(raw_images[0])
    z_torch = apply_site_stats(tensor_input, stats)
    assert isinstance(z_torch, torch.Tensor)
    assert pytest.approx(float(np.mean(z_np)), abs=1e-5) == float(z_torch.mean())


def test_profile_round_trip(tmp_path: Path) -> None:
    """Site adaptation profile saves to JSON and reloads intact."""
    stats = SiteStats(mean=14.5, std=2.8, n_scans=120, label="Synthetic")
    calib = SiteCalibration(threshold=0.88, false_call_rate=0.01, n_clean=120, upper95_fcr=0.025)
    prof_path = tmp_path / "site_profile.json"

    save_site_profile(prof_path, stats=stats, calib=calib, label="Synthetic")
    assert prof_path.exists()

    loaded = load_site_profile(prof_path)
    assert loaded.stats == stats
    assert loaded.calib == calib
    assert loaded.label == "Synthetic"
    assert loaded["stats"] == stats
    assert loaded["calib"] == calib
    assert "created" in loaded


@pytest.mark.parametrize("invalid_label", ["Plant_B", "Midhani", "Unknown", ""])
def test_invalid_label_raises_value_error(invalid_label: str, tmp_path: Path) -> None:
    """save_site_profile, load_site_profile, and SiteStats reject non-approved labels."""
    stats = SiteStats(mean=10.0, std=1.0, n_scans=10, label="Synthetic")
    calib = SiteCalibration(threshold=0.5, false_call_rate=0.05, n_clean=10, upper95_fcr=0.1)
    p = tmp_path / "test.json"

    with pytest.raises(ValueError, match="label"):
        save_site_profile(p, stats=stats, calib=calib, label=invalid_label)

    with pytest.raises(ValueError, match="label"):
        SiteStats(mean=10.0, std=1.0, n_scans=10, label=invalid_label)
