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
7. above_noise_area: Count of pixels exceeding 6 x robust noise level.
8. energy_ratio: Sum of squares within +-8 samples of peak row divided by total energy.
"""

from __future__ import annotations

import numpy as np

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


def compute_above_noise_area(arr: np.ndarray, noise_level: float) -> float:
    """Count image pixels whose amplitude strictly exceeds 6 times noise level."""
    threshold = 6.0 * max(0.0, noise_level)
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


def extract_features(img: np.ndarray) -> np.ndarray:
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
    area = compute_above_noise_area(arr, noise)
    e_ratio = compute_energy_ratio(arr, r_idx)

    return np.array(
        [peak, noise, snr, float(r_idx), duration, lateral, area, e_ratio],
        dtype=np.float64,
    )
