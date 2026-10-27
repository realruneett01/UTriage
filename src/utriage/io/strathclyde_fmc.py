"""Loader for Strathclyde Full Matrix Capture (FMC) ultrasonic dataset.

Loads .mat files containing FMC ultrasonic array data, element coordinates,
sampling frequency, and material sound speed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.io


class FMCRecord(dict):
    """Dictionary holding FMC dataset arrays and parameters with attribute access."""

    @property
    def fmc(self) -> np.ndarray | None:
        return self.get("fmc")

    @property
    def element_positions(self) -> np.ndarray | None:
        return self.get("element_positions")

    @property
    def sampling_frequency(self) -> float | None:
        return self.get("sampling_frequency")

    @property
    def wave_speed(self) -> float | None:
        return self.get("wave_speed")


_FMC_KEY_CANDIDATES = (
    "fmc",
    "fmc_data",
    "exp_data",
    "data",
    "rf_data",
    "signals",
    "h",
)
_ELEM_KEY_CANDIDATES = (
    "element_positions",
    "el_pos",
    "positions",
    "probe_coords",
    "rx",
    "tx",
)
_FS_KEY_CANDIDATES = (
    "sampling_frequency",
    "fs",
    "frequency",
    "sample_freq",
    "fs_mhz",
)
_C_KEY_CANDIDATES = (
    "wave_speed",
    "c",
    "velocity",
    "sound_speed",
    "v",
)


def _find_matching_key(keys: list[str], candidates: tuple[str, ...]) -> str | None:
    """Find first key in mat keys matching candidate names (case-insensitive)."""
    lower_map = {k.lower(): k for k in keys}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _extract_array(mat_dict: dict[str, Any], key: str | None) -> np.ndarray | None:
    """Extract and convert array from mat dictionary, returning None if missing."""
    if key is None or key not in mat_dict:
        return None
    val = mat_dict[key]
    if isinstance(val, np.ndarray):
        return val
    return np.asarray(val)


def _extract_float(mat_dict: dict[str, Any], key: str | None) -> float | None:
    """Extract scalar float from mat dictionary, returning None if missing."""
    if key is None or key not in mat_dict:
        return None
    val = mat_dict[key]
    try:
        if isinstance(val, np.ndarray):
            return float(val.squeeze())
        return float(val)
    except (TypeError, ValueError):  # reason: key holds non-scalar value
        return None


def load_fmc(path: str | Path) -> FMCRecord:
    """Load Full Matrix Capture dataset from a MATLAB .mat file.

    Parameters
    ----------
    path:
        File path to the .mat file.

    Returns
    -------
    FMCRecord:
        Dictionary with keys 'fmc', 'element_positions', 'sampling_frequency',
        and 'wave_speed'. Missing parameters are set to None.
    """
    file_path = Path(path)
    mat_dict = scipy.io.loadmat(file_path)
    clean_keys = [k for k in mat_dict if not k.startswith("__")]
    print(f"Keys found in {file_path.name}: {clean_keys}")

    fmc_k = _find_matching_key(clean_keys, _FMC_KEY_CANDIDATES)
    elem_k = _find_matching_key(clean_keys, _ELEM_KEY_CANDIDATES)
    fs_k = _find_matching_key(clean_keys, _FS_KEY_CANDIDATES)
    c_k = _find_matching_key(clean_keys, _C_KEY_CANDIDATES)

    return FMCRecord(
        fmc=_extract_array(mat_dict, fmc_k),
        element_positions=_extract_array(mat_dict, elem_k),
        sampling_frequency=_extract_float(mat_dict, fs_k),
        wave_speed=_extract_float(mat_dict, c_k),
    )
