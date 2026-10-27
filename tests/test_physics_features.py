"""Tests for interpretable physics features and logistic detector (ADR 0010 H3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from experiments.run_physics_model import (
    PhysicsModelConfig,
    _parse_args,
    fit_fold_classifier,
    run_physics_experiment,
)
from sklearn.preprocessing import StandardScaler

from utriage.detector_config import DEFAULT_PATH, PhysicsConfig, load_detector_config
from utriage.ml.physics_features import (
    FEATURE_NAMES,
    compute_above_noise_area,
    compute_energy_ratio,
    compute_robust_noise,
    extract_features,
)


def _make_gaussian_echo(
    center: tuple[int, int] = (100, 150),
    sigmas: tuple[float, float] = (4.0, 6.0),
    amp: float = 2000.0,
) -> np.ndarray:
    """Generate 256x256 synthetic image with centered Gaussian echo."""
    r0, c0 = center
    sigma_r, sigma_c = sigmas
    r = np.arange(256, dtype=np.float64)[:, None]
    c = np.arange(256, dtype=np.float64)[None, :]
    exponent = -(((r - r0) ** 2) / (2.0 * sigma_r**2) + ((c - c0) ** 2) / (2.0 * sigma_c**2))
    return (amp * np.exp(exponent)).astype(np.float64)


def _create_fake_batch(
    batch_path: Path,
    flaws: list[dict[str, Any]],
    labels: list[tuple[float, float]],
    seed: int,
) -> Path:
    """Write fake .bins, .jsons, and .labels files for a single test batch."""
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
    """Populate minimal fake ML-NDT dataset root in temporary directory."""
    root = tmp_path / "fake_mlndt"
    tr_dir, val_dir = root / "training", root / "validation"
    tr_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    flaws = [{"size": "1.6"}, {"size": "4.0"}, {"size": "8.6"}, {"size": "0.0"}]
    labels = [(1.0, 1.6), (1.0, 4.0), (1.0, 8.6), (0.0, 0.0)]

    for i in range(6):
        _create_fake_batch(tr_dir / f"batch_tr_{i:02d}.bins", flaws, labels, seed=i)
    for j in range(2):
        _create_fake_batch(val_dir / f"batch_val_{j:02d}.bins", flaws, labels, seed=100 + j)
    return root


def test_feature_names_tuple() -> None:
    """Verify FEATURE_NAMES is a tuple containing 8 physics feature names."""
    assert isinstance(FEATURE_NAMES, tuple)
    assert len(FEATURE_NAMES) == 8
    expected = (
        "peak_amplitude",
        "robust_noise",
        "snr",
        "peak_depth",
        "echo_duration",
        "lateral_extent",
        "above_noise_area",
        "energy_ratio",
    )
    assert FEATURE_NAMES == expected


@pytest.mark.parametrize(
    ("feat_name", "expected_val", "abs_tol"),
    [
        ("peak_amplitude", 2000.0, 1.0),
        ("peak_depth", 100.0, 1.0),
        ("echo_duration", 9.419, 1.0),
        ("lateral_extent", 14.129, 1.0),
    ],
    ids=["peak_amp", "peak_depth", "duration_fwhm", "lateral_fwhm"],
)
def test_gaussian_echo_spatial_features(
    feat_name: str, expected_val: float, abs_tol: float
) -> None:
    """Check spatial echo features on synthetic Gaussian within 1 sample tolerance."""
    img = _make_gaussian_echo(center=(100, 150), sigmas=(4.0, 6.0), amp=2000.0)
    feats = extract_features(img)
    feat_idx = FEATURE_NAMES.index(feat_name)
    val = feats[feat_idx]
    assert pytest.approx(val, abs=abs_tol) == expected_val


def test_energy_ratio_feature() -> None:
    """Check energy concentration ratio contains majority of Gaussian energy."""
    img = _make_gaussian_echo(center=(100, 150), sigmas=(4.0, 6.0), amp=2000.0)
    ratio = compute_energy_ratio(img, r_max=100)
    assert 0.95 <= ratio <= 1.0


def test_snr_higher_with_echo_than_without() -> None:
    """Verify SNR feature is strictly higher for image containing flaw echo."""
    rng = np.random.default_rng(42)
    noise = rng.normal(loc=500.0, scale=30.0, size=(256, 256))
    noise = np.clip(noise, 0.0, 65535.0)

    echo = _make_gaussian_echo(center=(120, 120), sigmas=(5.0, 5.0), amp=2500.0)
    img_with = np.clip(noise + echo, 0.0, 65535.0)
    img_without = noise

    feats_with = extract_features(img_with)
    feats_without = extract_features(img_without)

    snr_idx = FEATURE_NAMES.index("snr")
    assert feats_with[snr_idx] > feats_without[snr_idx] * 2.0


def test_robust_noise_and_above_noise_area() -> None:
    """Check robust noise level and 6x noise area thresholding."""
    rng = np.random.default_rng(123)
    noise = rng.normal(loc=1000.0, scale=50.0, size=(256, 256))
    noise = np.clip(noise, 0.0, 65535.0)

    mad = compute_robust_noise(noise)
    assert pytest.approx(mad, abs=5.0) == 33.7  # ~50 * 0.6745

    echo = _make_gaussian_echo(center=(80, 80), sigmas=(3.0, 3.0), amp=15000.0)
    img_spiky = np.clip(noise + echo, 0.0, 65535.0)
    area = compute_above_noise_area(img_spiky, noise_level=mad)
    assert area > 10.0


def test_scaler_fitted_on_train_only() -> None:
    """Verify standard scaler is fitted strictly on training data."""
    rng = np.random.default_rng(0)
    x_tr = rng.normal(loc=10.0, scale=2.0, size=(50, 8))
    x_te = rng.normal(loc=100.0, scale=5.0, size=(20, 8))
    y_tr = np.array([0, 1] * 25)

    scaler = StandardScaler()
    res = fit_fold_classifier(x_tr, y_tr, scaler)

    np.testing.assert_allclose(scaler.mean_, np.mean(x_tr, axis=0), rtol=1e-5)
    assert not np.allclose(scaler.mean_, np.mean(x_te, axis=0), rtol=1e-2)
    assert res.classifier is not None


def test_runner_on_fake_mlndt_root(fake_mlndt_root: Path, tmp_path: Path) -> None:
    """Execute physics model experiment runner on isolated fake ML-NDT root."""
    out_json = tmp_path / "test_physics_metrics.json"
    cfg = PhysicsModelConfig(root=fake_mlndt_root, out_path=out_json, quick=True)
    results = run_physics_experiment(cfg)

    assert out_json.exists()
    assert results["label"] == "Real data (ML-NDT)"
    assert results["run_label"] == "pre-registered"
    assert results["detector_config"]["is_default"] is True
    assert isinstance(results["detector_config"]["sha256"], str)
    assert "ADR 0010 H3" in results["protocol"]
    assert results["feature_names"] == list(FEATURE_NAMES)
    assert len(results["folds"]) == 3

    for fold in results["folds"]:
        assert fold["held_out"].startswith("crack_")
        assert "detections" in fold
        assert "false_calls" in fold
        assert "coefficients" in fold
        assert "cell" in fold
        assert "k" in fold["detections"]
        assert "n_false" in fold["false_calls"]


def test_extract_features_with_custom_parameters() -> None:
    """Verify extract_features respects custom noise_multiplier and energy_window."""
    img = _make_gaussian_echo(center=(100, 150), sigmas=(4.0, 6.0), amp=2000.0)
    feats_default = extract_features(img)
    feats_custom = extract_features(img, noise_multiplier=1.0, energy_window=2)

    area_idx = FEATURE_NAMES.index("above_noise_area")
    energy_idx = FEATURE_NAMES.index("energy_ratio")

    assert feats_custom[area_idx] >= feats_default[area_idx]
    assert feats_custom[energy_idx] < feats_default[energy_idx]


def test_compute_above_noise_area_custom_multiplier() -> None:
    """Verify compute_above_noise_area scales threshold with multiplier."""
    arr = np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float64)
    noise_level = 2.0
    area_high = compute_above_noise_area(arr, noise_level=noise_level, noise_multiplier=10.0)
    area_low = compute_above_noise_area(arr, noise_level=noise_level, noise_multiplier=2.0)
    assert area_low > area_high


def test_fit_fold_classifier_custom_config() -> None:
    """Verify fit_fold_classifier applies hyperparameters from PhysicsConfig."""
    x_tr = np.array([[1.0] * 8, [2.0] * 8, [3.0] * 8, [4.0] * 8], dtype=np.float64)
    y_tr = np.array([0, 1, 0, 1])
    cfg = PhysicsConfig(C=2.5, max_iter=250, threshold=0.7)
    scaler = StandardScaler()
    res = fit_fold_classifier(x_tr, y_tr, scaler, cfg=cfg)

    assert res.classifier.C == 2.5
    assert res.classifier.max_iter == 250


@pytest.mark.parametrize(
    ("cli_args", "expected_config", "expected_overrides"),
    [
        ([], DEFAULT_PATH, []),
        (["--config", "custom.yaml"], Path("custom.yaml"), []),
        (
            ["--set", "physics.threshold=0.7", "--set", "physics.C=2.0"],
            DEFAULT_PATH,
            ["physics.threshold=0.7", "physics.C=2.0"],
        ),
    ],
    ids=["defaults", "custom_config", "repeated_overrides"],
)
def test_parse_args_options(
    cli_args: list[str],
    expected_config: Path,
    expected_overrides: list[str],
) -> None:
    """Verify _parse_args parses default and explicit config/override flags."""
    parsed = _parse_args(cli_args)
    assert parsed.config == expected_config
    assert parsed.overrides == expected_overrides


def test_runner_exploratory_config_on_fake_data(
    fake_mlndt_root: Path,
    tmp_path: Path,
) -> None:
    """Verify exploratory configuration updates run_label and detector_config."""
    cfg_path = tmp_path / "custom_detectors.yaml"
    cfg_path.write_text("physics:\n  threshold: 0.8\n", encoding="utf-8")
    det_cfg = load_detector_config(path=cfg_path)

    out_json = tmp_path / "exploratory_physics_metrics.json"
    cfg = PhysicsModelConfig(
        root=fake_mlndt_root,
        out_path=out_json,
        quick=True,
        detector_cfg=det_cfg,
    )
    results = run_physics_experiment(cfg)

    assert out_json.exists()
    assert results["run_label"] == "exploratory"
    assert results["detector_config"]["is_default"] is False
    assert len(results["detector_config"]["sha256"]) == 64
