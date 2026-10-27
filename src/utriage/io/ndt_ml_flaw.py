"""I/O loader and preprocessing for NDT_ML_Flaw ultrasonic dataset.

Dataset specification:
- Real flaw batches: batch_013 through batch_019 (.xz archives + .txt metadata).
- 2xx batches represent simulated flaws and are excluded.
- Format per image: 480 (spatial scan axis) x 7168 (ultrasonic RF time axis).
- Data type: little-endian int16 (2 bytes per sample -> 6,881,280 bytes/image).
- Documented flaw depth range: samples 1100-3100 along the time axis.
- Domain mapping to ML-NDT space: Hilbert envelope along time axis, time-axis crop,
  bilinear spatial-temporal zoom to (256, 256), and per-image standardisation.
"""

from __future__ import annotations

import lzma
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage
import scipy.signal

from utriage.ml.datasets import standardise

N_SCAN: int = 480
N_TIME: int = 7168
BYTES_PER_SAMPLE: int = 2
IMAGE_BYTES: int = N_SCAN * N_TIME * BYTES_PER_SAMPLE  # 6,881,280 bytes
FLAW_AREA_START: int = 1100
FLAW_AREA_END: int = 3100
TARGET_SIZE: int = 256


def is_real_batch(name: str) -> bool:
    """Check whether a batch filename represents a real flaw dataset.

    Batches 013-019 contain real flaw data. Batches starting with 2xx contain
    simulated flaws and are excluded.
    """
    match = re.search(r"batch_(\d+)", name)
    if match is None:
        return False
    digits = match.group(1)
    if digits.startswith("2"):
        return False
    return 13 <= int(digits) <= 19


def list_real_batches(root: Path | str) -> list[tuple[Path, Path]]:
    """Discover pairs of (batch_NNN.xz, batch_NNN.txt) for real flaw batches.

    Excludes 2xx batches (simulated flaws) and retains batches 013-019.
    """
    root_p = Path(root)
    if not root_p.exists():
        return []
    pairs: list[tuple[Path, Path]] = []
    for xz in sorted(root_p.glob("batch_*.xz")):
        if not is_real_batch(xz.stem):
            continue
        txt = xz.with_suffix(".txt")
        if txt.exists():
            pairs.append((xz, txt))
    return pairs


def _parse_meta_line(line: str) -> dict[str, Any] | None:
    """Parse a single whitespace-separated metadata row."""
    parts = line.strip().split()
    if len(parts) < 7:
        return None
    return {
        "flaw": int(parts[0]),
        "augmentation": float(parts[1]),
        "flaw_depth": float(parts[2]),
        "flaw_location": float(parts[3]),
        "flaw_size": float(parts[4]),
        "index_line": int(parts[5]),
        "flaw_type": parts[6],
    }


def read_meta(txt_path: Path | str) -> list[dict[str, Any]]:
    """Parse NDT_ML_Flaw metadata text file into list of per-image dictionaries."""
    path = Path(txt_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, Any]] = []
    for line in lines:
        rec = _parse_meta_line(line)
        if rec is not None:
            records.append(rec)
    return records


def _stream_chunks(
    fh: Any,
    start: int,
    stop: int | None,
) -> Iterator[np.ndarray]:
    """Iterate over image chunks from an open binary file handle."""
    if start > 0:
        fh.seek(start * IMAGE_BYTES)
    idx = start
    while stop is None or idx < stop:
        chunk = fh.read(IMAGE_BYTES)
        if len(chunk) < IMAGE_BYTES:
            break
        yield np.frombuffer(chunk, dtype="<i2").reshape((N_SCAN, N_TIME))
        idx += 1


def iter_images(
    xz_path: Path | str,
    start: int = 0,
    stop: int | None = None,
) -> Iterator[np.ndarray]:
    """Stream (480, 7168) int16 RF images from an .xz archive without loading batch."""
    path = Path(xz_path)
    with lzma.open(path, "rb") as fh:
        yield from _stream_chunks(fh, start, stop)


def _crop_time_window(img: np.ndarray, margin: int = 50) -> np.ndarray:
    """Crop the RF time axis around the documented flaw area (1100-3100 plus margin).

    Domain rationale:
    Axis 0 (length 480) is the spatial scan axis.
    Axis 1 (length 7168) is the ultrasonic A-scan time axis.
    The NDT_ML_Flaw README documents flaw depth samples in range 1100-3100, which
    only fits within the 7168-sample axis.
    """
    n_cols = img.shape[1]
    if n_cols <= TARGET_SIZE:
        return img
    start = max(0, FLAW_AREA_START - margin)
    end = min(n_cols, FLAW_AREA_END + margin)
    if start >= end:
        return img
    return img[:, start:end]


def to_mlndt_like(img: np.ndarray, margin: int = 50) -> np.ndarray:
    """Domain mapping from raw RF (480, 7168) to ML-NDT standard space (256, 256).

    1. Extract analytic signal envelope along time axis (axis 1) via Hilbert transform.
    2. Crop the time axis to the flaw depth region [1100-margin, 3100+margin].
    3. Resample both spatial (scan) and time axes to 256x256 via bilinear zoom.
    4. Apply per-image standardisation ((x - mean) / (std + 1e-5)).

    Returns float32 ndarray of shape (256, 256).
    """
    rf = img.astype(np.float32)
    env = np.abs(scipy.signal.hilbert(rf, axis=1))
    cropped = _crop_time_window(env, margin=margin)
    zy = float(TARGET_SIZE / cropped.shape[0])
    zx = float(TARGET_SIZE / cropped.shape[1])
    resampled = scipy.ndimage.zoom(cropped, (zy, zx), order=1).astype(np.float32)
    return standardise(resampled)


def _count_stream_bytes(fh: Any, buffer_size: int = 1024 * 1024) -> int:
    """Accumulate total decompressed bytes from binary stream."""
    total = 0
    for chunk in iter(lambda: fh.read(buffer_size), b""):
        total += len(chunk)
    return total


def verify_decompressed_bytes(
    xz_path: Path | str,
    n_expected: int = 1000,
) -> int:
    """Stream decompression of archive to verify bytes-per-image without RAM loading."""
    if n_expected <= 0:
        raise ValueError("n_expected must be positive.")
    path = Path(xz_path)
    with lzma.open(path, "rb") as fh:
        total_bytes = _count_stream_bytes(fh)
    return total_bytes // n_expected
