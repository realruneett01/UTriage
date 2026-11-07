"""Titanium structured grain noise simulation. Synthetic data."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from utriage.physics.simulator import (
    BarScan,
    BarSpec,
    _load_yaml,
    depth_of_time,
    grain_noise,
    simulate_bar,
)

TI_MODE_PATH = Path(__file__).resolve().parents[3] / "configs" / "ti_mode.yaml"


@dataclass(frozen=True)
class TiNoiseConfig:
    """Configuration for titanium macrozone grain noise. Synthetic data."""

    corr_axial_mm: float
    corr_angle_deg: float
    field_sigma: float
    zone_edges_mm: tuple[float, ...]
    zone_levels: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.zone_levels) != len(self.zone_edges_mm) - 1:
            raise ValueError("zone_levels length must be len(zone_edges_mm) - 1")
        if len(self.zone_edges_mm) < 2:
            raise ValueError("zone_edges_mm must have at least 2 edges")
        if any(x >= y for x, y in zip(self.zone_edges_mm, self.zone_edges_mm[1:], strict=False)):
            raise ValueError("zone_edges_mm must be strictly increasing")
        if self.field_sigma < 0.0:
            raise ValueError("field_sigma must be non-negative")
        if any(c <= 0.0 for c in (self.corr_axial_mm, self.corr_angle_deg)):
            raise ValueError("correlation lengths must be positive")


def load_ti_noise(path: Path | None = None) -> TiNoiseConfig:
    """Load titanium noise configuration. Values are placeholder, UNVERIFIED - Ruri to check."""
    data = _load_yaml(path or TI_MODE_PATH)["noise"]
    return TiNoiseConfig(
        corr_axial_mm=float(data["corr_axial_mm"]),
        corr_angle_deg=float(data["corr_angle_deg"]),
        field_sigma=float(data["field_sigma"]),
        zone_edges_mm=tuple(float(x) for x in data["zone_edges_mm"]),
        zone_levels=tuple(float(x) for x in data["zone_levels"]),
    )


def strength_field(
    n_axial: int,
    n_angle: int,
    steps: tuple[float, float],
    cfg: TiNoiseConfig,
    rng: Any = None,
) -> np.ndarray:
    """Spatially correlated lognormal multiplier field. Synthetic data."""
    if cfg.field_sigma == 0.0:
        return np.ones((n_axial, n_angle), dtype=np.float64)
    gen = np.random.default_rng(rng)
    white = gen.standard_normal((n_axial, n_angle))
    sigmas = (cfg.corr_axial_mm / steps[0], cfg.corr_angle_deg / steps[1])
    filtered = gaussian_filter(white, sigma=sigmas, mode=("nearest", "wrap"))
    std = float(np.std(filtered))
    g = filtered / std if std > 0.0 else filtered
    s = cfg.field_sigma
    return np.exp(s * g - 0.5 * s * s)


def zone_index(depth_mm: float, cfg: TiNoiseConfig) -> int | None:
    """Return zone index for a depth, or None if outside all zones."""
    edges = cfg.zone_edges_mm
    for k in range(len(edges) - 1):
        if edges[k] <= depth_mm < edges[k + 1]:
            return k
    return None


def zone_level(depth_mm: np.ndarray, cfg: TiNoiseConfig) -> np.ndarray:
    """Return depth zone noise multipliers (1.0 outside defined zones)."""
    depth = np.asarray(depth_mm, dtype=float)
    out = np.ones(depth.shape, dtype=float)
    edges = cfg.zone_edges_mm
    for k, lvl in enumerate(cfg.zone_levels):
        in_zone = (depth >= edges[k]) & (depth < edges[k + 1])
        out = np.where(in_zone, lvl, out)
    return out


def simulate_ti_bar(spec: BarSpec, cfg: TiNoiseConfig) -> BarScan:
    """Synthetic data.

    Structured noise is a simulator model choice inspired by the titanium
    grain-noise literature, not a fitted model of any real alloy.
    """
    no_grain_mat = dataclasses.replace(spec.material, grain_noise=0.0)
    no_grain_spec = dataclasses.replace(spec, material=no_grain_mat)
    base = simulate_bar(no_grain_spec)
    rng = np.random.default_rng(spec.seed + 7919)
    g = grain_noise(base.time_us, base.t_if_us, spec, rng)
    depth = depth_of_time(base.time_us[None, None, :], base.t_if_us[..., None], spec.material)
    steps = (spec.axial_step_mm, 360.0 / spec.n_angle)
    field = strength_field(base.ascans.shape[0], base.ascans.shape[1], steps, cfg, rng)
    g *= field[..., None] * zone_level(depth, cfg)
    gain = 10.0 ** (spec.probe.gain_db / 20.0)
    ascans = np.clip(base.ascans + g * gain, -1.0, 1.0).astype(np.float32)
    if base.missing.shape == ascans.shape[:2]:
        ascans[base.missing] = 0.0
    return dataclasses.replace(base, ascans=ascans)
