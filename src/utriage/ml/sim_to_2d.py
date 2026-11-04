"""2D B-scan generation from 1D immersion simulator (Synthetic data).

Converts 1D A-scans across scan positions into 256x256 B-scan images
for joint synthetic-real ML training and evaluation (project plan Step 3.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.signal import hilbert
from torch.utils.data import Dataset

from utriage.ml.datasets import standardise
from utriage.physics.simulator import (
    BarSpec,
    Reflector,
    load_material,
    load_probe,
    simulate_bar,
)


@dataclass(frozen=True)
class SimBscanConfig:
    """Acquisition and scan settings for 2D B-scan simulation (Synthetic data)."""

    material_name: str = "ti64"
    probe_name: str = "p5"
    diameter_mm: float = 40.0
    length_mm: float = 16.0
    axial_step_mm: float = 0.25
    reflector_depth_mm: float = 10.0
    reflector_diameter_mm: float = 1.0


def _resample_2d(arr: np.ndarray, target_size: int = 256) -> np.ndarray:
    """Resample both axes of a 2D array to (target_size, target_size)."""
    n_pos, n_samp = arr.shape
    x_old = np.linspace(0.0, 1.0, n_samp)
    x_new = np.linspace(0.0, 1.0, target_size)
    interp_time = np.empty((n_pos, target_size), dtype=np.float32)
    for i in range(n_pos):
        interp_time[i] = np.interp(x_new, x_old, arr[i])

    y_old = np.linspace(0.0, 1.0, n_pos)
    y_new = np.linspace(0.0, 1.0, target_size)
    out = np.empty((target_size, target_size), dtype=np.float32)
    for j in range(target_size):
        out[:, j] = np.interp(y_new, y_old, interp_time[:, j])
    return out


def build_bscan(ascans: np.ndarray) -> np.ndarray:
    """Convert RF A-scans (n_positions, n_samples) into a 256x256 B-scan (Synthetic data)."""
    arr = np.asarray(ascans, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array (n_positions, n_samples), got shape {arr.shape}")
    env = np.abs(hilbert(arr, axis=-1)).astype(np.float32)
    resampled = _resample_2d(env, target_size=256)
    return standardise(resampled)


def flaw_region_slice(cfg: SimBscanConfig | None = None, margin: int = 10) -> tuple[slice, slice]:
    """Return (row_slice, col_slice) around the flaw center in 256x256 image (Synthetic data)."""
    c = cfg if cfg is not None else SimBscanConfig()
    mat = load_material(c.material_name)
    probe = load_probe(c.probe_name)
    water = load_material("water")

    row_center = int(round(0.5 * 255))
    t_if = 2.0 * probe.water_path_mm / (water.velocity_m_s * 1e-3)
    t_metal = 2.0 * c.diameter_mm / (mat.velocity_m_s * 1e-3)
    t_flaw = t_if + 2.0 * c.reflector_depth_mm / (mat.velocity_m_s * 1e-3)
    t_start, t_stop = t_if - 2.0, t_if + t_metal + 3.0
    col_center = int(round(((t_flaw - t_start) / (t_stop - t_start)) * 255))

    r_slice = slice(max(0, row_center - margin), min(256, row_center + margin + 1))
    c_slice = slice(max(0, col_center - margin), min(256, col_center + margin + 1))
    return r_slice, c_slice


def _build_bar_spec(seed: int, with_flaw: bool, cfg: SimBscanConfig) -> BarSpec:
    """Construct BarSpec for helical simulator with 1 angle slice (Synthetic data)."""
    reflectors = ()
    if with_flaw:
        ref = Reflector(
            axial_mm=cfg.length_mm / 2.0,
            angle_deg=0.0,
            depth_mm=cfg.reflector_depth_mm,
            diameter_mm=cfg.reflector_diameter_mm,
        )
        reflectors = (ref,)

    return BarSpec(
        material=load_material(cfg.material_name),
        probe=load_probe(cfg.probe_name),
        water=load_material("water"),
        diameter_mm=cfg.diameter_mm,
        length_mm=cfg.length_mm,
        axial_step_mm=cfg.axial_step_mm,
        n_angle=1,
        reflectors=reflectors,
        seed=seed,
    )


def simulate_bscan(
    seed: int, with_flaw: bool = True, cfg: SimBscanConfig | None = None
) -> tuple[np.ndarray, int]:
    """Simulate a 256x256 B-scan with or without flaw indication (Synthetic data)."""
    c = cfg if cfg is not None else SimBscanConfig()
    spec = _build_bar_spec(seed=seed, with_flaw=with_flaw, cfg=c)
    scan = simulate_bar(spec)
    bscan = build_bscan(scan.ascans[:, 0, :])
    label = 1 if with_flaw else 0
    return bscan, label


class SimBscanDataset(Dataset):
    """PyTorch Dataset yielding 256x256 B-scans generated on the fly (Synthetic data)."""

    def __init__(
        self,
        length: int,
        base_seed: int = 0,
        cfg: SimBscanConfig | None = None,
        with_flaw: bool | None = None,
    ) -> None:
        self.length = length
        self.base_seed = base_seed
        self.cfg = cfg if cfg is not None else SimBscanConfig()
        self.with_flaw = with_flaw

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0:
            idx += self.length
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} out of range for length {self.length}")

        has_flaw = self.with_flaw if self.with_flaw is not None else bool(idx % 2 == 1)
        image, label = simulate_bscan(seed=self.base_seed + idx, with_flaw=has_flaw, cfg=self.cfg)
        tensor = torch.from_numpy(image).unsqueeze(0)
        label_tensor = torch.tensor(float(label), dtype=torch.float32)
        return tensor, label_tensor


__all__ = [
    "SimBscanConfig",
    "SimBscanDataset",
    "build_bscan",
    "flaw_region_slice",
    "simulate_bscan",
]
