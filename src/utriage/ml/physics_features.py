"""Hand-crafted physics feature extraction for ultrasonic inspection images.

Pre-registered design: candidate H3 of ADR 0010.

Axis convention (determined from ML-NDT dataset and utriage.ml.datasets):
- Axis 0 (rows): Time / depth axis. In ML-NDT weld PAUT data, rows correspond to
  A-scan time samples. Max-pooling along axis 0 acts as envelope detection over
  ultrasonic wave cycles.
- Axis 1 (columns): Lateral / transducer scan axis. Columns correspond to spatial
  scan positions (adjacent A-scans across the inspection aperture).

Features extracted:
1. peak_amplitude: Peak reflection amplitude (max |x|).
2. robust_noise: Robust background noise level (median absolute deviation of image).
3. snr: Signal-to-noise ratio (peak_amplitude / robust_noise).
4. peak_depth: Row index of maximum amplitude reflection along time/depth axis.
5. echo_duration: Number of time samples around peak above half-peak in peak's column (-6 dB FWHM).
6. lateral_extent: Number of columns around peak above half-peak in peak's row (-6 dB beam width).
7. above_noise_area: Count of pixels exceeding noise_multiplier x robust noise level.
8. energy_ratio: Sum of squares within +-energy_window samples of peak row divided by total energy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utriage.detector_config import PhysicsConfig
from utriage.ml.datasets import read_image, read_labels
from utriage.ml.loco import LocoFold

FEATURE_NAMES: tuple[str, ...] = (
    "peak_amplitude",
    "robust_noise",
    "snr",
    "peak_depth",
    "echo_duration",
    "lateral_extent",
    "above_noise_area",
    "energy_ratio",
)


@dataclass(frozen=True)
class FitResult:
    """Trained classifier, fitted scaler, and extracted feature coefficients."""

    classifier: LogisticRegression
    scaler: StandardScaler
    coefficients: dict[str, float]


def compute_peak_amplitude(arr: np.ndarray) -> float:
    """Compute maximum absolute acoustic reflection amplitude."""
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr)))


def compute_robust_noise(arr: np.ndarray) -> float:
    """Compute robust noise level via median absolute deviation (MAD)."""
    if arr.size == 0:
        return 0.0
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def compute_snr(peak: float, noise: float) -> float:
    """Compute signal-to-noise ratio with zero-noise safeguard."""
    if noise <= 0.0:
        return 0.0 if peak <= 0.0 else float(peak / 1e-6)
    return float(peak / noise)


def compute_peak_depth(r_max: int) -> float:
    """Return time/depth row index of peak reflection."""
    return float(r_max)


def _contiguous_count(slice_1d: np.ndarray, center_idx: int, threshold: float) -> float:
    """Count contiguous elements around center_idx meeting or exceeding threshold."""
    n = len(slice_1d)
    if any([center_idx < 0, center_idx >= n, threshold <= 0.0]):
        return 0.0
    if slice_1d[center_idx] < threshold:
        return 0.0

    left = center_idx
    while left > 0 and slice_1d[left - 1] >= threshold:
        left -= 1

    right = center_idx
    while right < n - 1 and slice_1d[right + 1] >= threshold:
        right += 1

    return float(right - left + 1)


def compute_echo_duration(col: np.ndarray, r_max: int, half_peak: float) -> float:
    """Compute temporal echo duration (-6 dB FWHM) in peak's column."""
    return _contiguous_count(col, r_max, half_peak)


def compute_lateral_extent(row: np.ndarray, c_max: int, half_peak: float) -> float:
    """Compute lateral spatial defect extent (-6 dB width) in peak's row."""
    return _contiguous_count(row, c_max, half_peak)


def compute_above_noise_area(
    arr: np.ndarray,
    noise_level: float,
    noise_multiplier: float = 6.0,
) -> float:
    """Count image pixels whose amplitude strictly exceeds noise_multiplier times noise level."""
    threshold = noise_multiplier * max(0.0, noise_level)
    return float(np.sum(np.abs(arr) > threshold))


def compute_energy_ratio(arr: np.ndarray, r_max: int, window: int = 8) -> float:
    """Compute energy fraction in +-window rows around peak to total energy."""
    if any([arr.size == 0, r_max < 0, r_max >= arr.shape[0]]):
        return 0.0
    r_start = max(0, r_max - window)
    r_end = min(arr.shape[0], r_max + window + 1)
    band = arr[r_start:r_end, :]
    band_energy = float(np.sum(band.astype(np.float64) ** 2))
    total_energy = float(np.sum(arr.astype(np.float64) ** 2))
    if total_energy <= 0.0:
        return 0.0
    return float(band_energy / total_energy)


def extract_features(
    img: np.ndarray,
    noise_multiplier: float = 6.0,
    energy_window: int = 8,
) -> np.ndarray:
    """Extract 8 hand-crafted physics features from raw 2D ultrasonic image."""
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim > 2:
        arr = arr.squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image array, got shape {arr.shape}.")

    peak = compute_peak_amplitude(arr)
    noise = compute_robust_noise(arr)
    snr = compute_snr(peak, noise)

    r_max, c_max = np.unravel_index(np.argmax(np.abs(arr)), arr.shape)
    r_idx = int(r_max)
    c_idx = int(c_max)
    half_peak = 0.5 * peak

    duration = compute_echo_duration(np.abs(arr[:, c_idx]), r_idx, half_peak)
    lateral = compute_lateral_extent(np.abs(arr[r_idx, :]), c_idx, half_peak)
    area = compute_above_noise_area(arr, noise, noise_multiplier=noise_multiplier)
    e_ratio = compute_energy_ratio(arr, r_idx, window=energy_window)

    return np.array(
        [peak, noise, snr, float(r_idx), duration, lateral, area, e_ratio],
        dtype=np.float64,
    )


def build_feature_dataset(
    items: list[tuple[Path, int]],
    noise_multiplier: float = 6.0,
    energy_window: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features, labels, and flaw sizes streaming images from disk."""
    feats: list[np.ndarray] = []
    flaws: list[int] = []
    sizes: list[float] = []
    labels_cache: dict[Path, np.ndarray] = {}

    for bins_path, idx in items:
        if bins_path not in labels_cache:
            labels_cache[bins_path] = read_labels(bins_path)
        lbl = labels_cache[bins_path]
        flaws.append(int(lbl[idx, 0]))
        sizes.append(float(lbl[idx, 1]))
        img = read_image(bins_path, idx)
        feats.append(
            extract_features(
                img,
                noise_multiplier=noise_multiplier,
                energy_window=energy_window,
            )
        )

    x_arr = np.array(feats, dtype=np.float64) if feats else np.empty((0, 8), dtype=np.float64)
    y_arr = np.array(flaws, dtype=np.int64) if flaws else np.empty(0, dtype=np.int64)
    s_arr = np.array(sizes, dtype=np.float32) if sizes else np.empty(0, dtype=np.float32)
    return x_arr, y_arr, s_arr


def fit_fold_classifier(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    scaler: StandardScaler,
    cfg: PhysicsConfig | None = None,
) -> FitResult:
    """Fit StandardScaler and LogisticRegression classifier on train fold split."""
    p_cfg = cfg if cfg is not None else PhysicsConfig()
    x_tr_scaled = scaler.fit_transform(x_tr)
    clf = LogisticRegression(
        class_weight=p_cfg.class_weight,
        C=p_cfg.C,
        max_iter=p_cfg.max_iter,
        random_state=0,
    )
    clf.fit(x_tr_scaled, y_tr)
    coefs = clf.coef_[0]
    intercept = float(clf.intercept_[0])
    coeff_dict = {name: float(coefs[i]) for i, name in enumerate(FEATURE_NAMES)}
    coeff_dict["intercept"] = intercept
    return FitResult(classifier=clf, scaler=scaler, coefficients=coeff_dict)


def score_physics_classifier(
    fold: LocoFold,
    items: list[tuple[Path, int]],
    cfg: PhysicsConfig | None = None,
) -> np.ndarray:
    """Train logistic regression on fold training features and score items."""
    p_cfg = cfg if cfg is not None else PhysicsConfig()
    x_tr, y_tr, _ = build_feature_dataset(
        fold.train,
        noise_multiplier=p_cfg.noise_multiplier,
        energy_window=p_cfg.energy_window,
    )
    scaler = StandardScaler()
    fit_res = fit_fold_classifier(x_tr, y_tr, scaler, cfg=p_cfg)
    x_te, _, _ = build_feature_dataset(
        items,
        noise_multiplier=p_cfg.noise_multiplier,
        energy_window=p_cfg.energy_window,
    )
    if len(x_te) == 0:
        return np.empty(0, dtype=np.float64)
    x_te_scaled = fit_res.scaler.transform(x_te)
    return fit_res.classifier.predict_proba(x_te_scaled)[:, 1]


score_physics_fold = score_physics_classifier
