"""Domain randomization over the immersion UT simulator. Synthetic data.

Provides on-the-fly randomized parameter sampling, held-out region rejection,
and streaming PyTorch IterableDataset generation without disk writes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from torch.utils.data import IterableDataset, get_worker_info

from utriage.physics.dataset import DatasetConfig
from utriage.physics.simulator import (
    CONFIG_DIR,
    Artefact,
    BarScan,
    BarSpec,
    Material,
    Probe,
    Reflector,
    load_material,
    load_probe,
    simulate_bar,
)


@dataclass(frozen=True)
class Region:
    """A parameter region (e.g. held-out domain). Synthetic data."""

    name: str
    param: str
    low: float | str
    high: float | str

    def contains_value(self, val: float | str) -> bool:
        """Check if a single parameter value falls inside this region."""
        if isinstance(self.low, str):
            return str(val) == self.low or str(val) == self.high
        try:
            val_f = float(val)
            return float(self.low) <= val_f <= float(self.high)
        except (ValueError, TypeError):
            # Non-numeric comparison fallback
            return False

    def matches(self, params: dict[str, Any]) -> bool:
        """Check if a parameter dict has a value inside this region."""
        if self.param not in params:
            return False
        return self.contains_value(params[self.param])


DEFAULT_HOLDOUT_REGIONS: tuple[Region, ...] = (
    Region("freq_10mhz", "frequency_mhz", 9.0, 15.0),
    Region("material_third", "material", "maraging250", "maraging250"),
    Region("noise_above_range", "grain_noise_scale", 2.0, 3.5),
)
HOLDOUT_REGIONS: list[Region] = list(DEFAULT_HOLDOUT_REGIONS)


@dataclass(frozen=True)
class RandomizationConfig:
    """Parameter ranges for domain randomization. Synthetic data."""

    velocity_scale: tuple[float, float] = (0.95, 1.05)
    attenuation_scale: tuple[float, float] = (0.5, 2.0)
    grain_noise_scale: tuple[float, float] = (0.5, 2.0)
    frequency_mhz: tuple[float, float] = (4.0, 11.0)
    fractional_bandwidth: tuple[float, float] = (0.4, 0.8)
    focal_depth_mm: tuple[float, float] = (10.0, 25.0)
    water_path_mm: tuple[float, float] = (35.0, 65.0)
    diameter_mm: tuple[float, float] = (25.0, 60.0)
    gain_db: tuple[float, float] = (-6.0, 6.0)
    reflector_size_mm: tuple[float, float] = (0.4, 2.0)
    reflector_depth_mm: tuple[float, float] = (3.0, 35.0)
    artefact_prob_bubble: tuple[float, float] = (0.0, 0.2)
    artefact_prob_lost_backwall: tuple[float, float] = (0.0, 0.2)
    artefact_prob_roughness: tuple[float, float] = (0.0, 0.2)
    materials: tuple[str, ...] = ("ti64", "in718", "maraging250")
    not_supported: dict[str, str] = field(default_factory=dict)


def _parse_tuple(val: Any) -> tuple[float, float]:
    """Parse range tuple from loaded dict or sequence."""
    if isinstance(val, dict):
        return (float(val["min"]), float(val["max"]))
    return (float(val[0]), float(val[1]))


def load_randomization(path: Path | str | None = None) -> RandomizationConfig:
    """Load RandomizationConfig from YAML file. Synthetic data."""
    target_path = Path(path) if path is not None else CONFIG_DIR / "randomization.yaml"
    with target_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    ranges = raw.get("ranges", raw)
    not_supported = raw.get("not_supported", {})
    materials = tuple(raw.get("materials", {}).get("choices", ("ti64", "in718", "maraging250")))
    fields: dict[str, Any] = {}
    for name in RandomizationConfig.__dataclass_fields__:
        if name in ranges:
            fields[name] = _parse_tuple(ranges[name])
    return RandomizationConfig(materials=materials, not_supported=not_supported, **fields)


def _draw_raw_params(cfg: RandomizationConfig, rng: np.random.Generator) -> dict[str, Any]:
    """Draw one candidate parameter dict from config ranges. Synthetic data."""
    mat_choices = cfg.materials or ("ti64",)
    return {
        "material": str(rng.choice(mat_choices)),
        "velocity_scale": float(rng.uniform(*cfg.velocity_scale)),
        "attenuation_scale": float(rng.uniform(*cfg.attenuation_scale)),
        "grain_noise_scale": float(rng.uniform(*cfg.grain_noise_scale)),
        "frequency_mhz": float(rng.uniform(*cfg.frequency_mhz)),
        "fractional_bandwidth": float(rng.uniform(*cfg.fractional_bandwidth)),
        "focal_depth_mm": float(rng.uniform(*cfg.focal_depth_mm)),
        "water_path_mm": float(rng.uniform(*cfg.water_path_mm)),
        "diameter_mm": float(rng.uniform(*cfg.diameter_mm)),
        "gain_db": float(rng.uniform(*cfg.gain_db)),
        "reflector_size_mm": float(rng.uniform(*cfg.reflector_size_mm)),
        "reflector_depth_mm": float(rng.uniform(*cfg.reflector_depth_mm)),
        "artefact_prob_bubble": float(rng.uniform(*cfg.artefact_prob_bubble)),
        "artefact_prob_lost_backwall": float(rng.uniform(*cfg.artefact_prob_lost_backwall)),
        "artefact_prob_roughness": float(rng.uniform(*cfg.artefact_prob_roughness)),
    }


def sample_params(
    cfg: RandomizationConfig,
    rng: np.random.Generator,
    exclude: tuple[Region, ...] = (),
) -> dict[str, Any]:
    """Sample parameters using rejection sampling to avoid excluded regions. Synthetic data."""
    for _ in range(5000):
        params = _draw_raw_params(cfg, rng)
        if not any(reg.matches(params) for reg in exclude):
            return params
    raise RuntimeError("Failed to sample parameters outside excluded regions after 5000 attempts.")


def sample_in_region(
    cfg: RandomizationConfig,
    rng: np.random.Generator,
    region: Region,
) -> dict[str, Any]:
    """Sample parameter dict guaranteed to fall inside the specified region. Synthetic data."""
    params = _draw_raw_params(cfg, rng)
    if isinstance(region.low, str):
        params[region.param] = str(region.low)
    else:
        low_f = float(region.low)
        high_f = float(region.high)
        if math.isinf(high_f):
            high_f = low_f + 3.0
        params[region.param] = float(rng.uniform(low_f, high_f))
    return params


def _make_material(params: dict[str, Any], default_name: str) -> Material:
    """Build a randomized Material instance. Synthetic data."""
    mat_name = str(params.get("material", default_name))
    base_mat = load_material(mat_name)
    v_scale = float(params.get("velocity_scale", 1.0))
    a_scale = float(params.get("attenuation_scale", 1.0))
    n_scale = float(params.get("grain_noise_scale", 1.0))
    return replace(
        base_mat,
        velocity_m_s=base_mat.velocity_m_s * v_scale,
        attenuation_db_mm_mhz=base_mat.attenuation_db_mm_mhz * a_scale,
        grain_noise=base_mat.grain_noise * n_scale,
    )


def _make_probe(params: dict[str, Any], base_probe_key: str) -> Probe:
    """Build a randomized Probe instance. Synthetic data."""
    base_probe = load_probe(base_probe_key)
    return replace(
        base_probe,
        frequency_mhz=float(params.get("frequency_mhz", base_probe.frequency_mhz)),
        fractional_bandwidth=float(
            params.get("fractional_bandwidth", base_probe.fractional_bandwidth)
        ),
        focal_depth_mm=float(params.get("focal_depth_mm", base_probe.focal_depth_mm)),
        water_path_mm=float(params.get("water_path_mm", base_probe.water_path_mm)),
        gain_db=float(params.get("gain_db", base_probe.gain_db)),
    )


def _make_reflectors(
    params: dict[str, Any], base: DatasetConfig, rng: np.random.Generator
) -> tuple[Reflector, ...]:
    """Generate randomized reflectors for BarSpec. Synthetic data."""
    n_refl = int(rng.poisson(base.mean_reflectors))
    if n_refl <= 0:
        return ()
    reflectors = []
    diam_bar = float(params.get("diameter_mm", base.diameter_mm))
    max_d = max(3.0, diam_bar - 5.0)
    for _ in range(n_refl):
        is_fbh = bool(rng.random() < 0.7)
        kind = "fbh" if is_fbh else "inclusion"
        amp = 1.0 if is_fbh else float(rng.uniform(0.3, 1.0))
        d_size = float(params.get("reflector_size_mm", rng.uniform(*base.fbh_range_mm)))
        d_depth = float(params.get("reflector_depth_mm", rng.uniform(3.0, max_d)))
        d_depth = min(max_d, max(3.0, d_depth))
        ax = float(rng.uniform(0.0, max(0.0, base.length_mm - 4.0)))
        an = float(rng.uniform(0.0, 360.0))
        reflectors.append(
            Reflector(
                axial_mm=ax,
                angle_deg=an,
                depth_mm=d_depth,
                diameter_mm=d_size,
                kind=kind,
                amplitude_factor=amp,
            )
        )
    return tuple(reflectors)


def _make_one_artefact(kind: str, length_mm: float, rng: np.random.Generator) -> Artefact:
    """Generate a single artefact region. Synthetic data."""
    span_ax = min(float(rng.uniform(6.0, 20.0)), length_mm)
    ax0 = float(rng.uniform(0.0, max(0.0, length_mm - span_ax)))
    span_an = float(rng.uniform(20.0, 90.0))
    an0 = float(rng.uniform(0.0, 360.0 - span_an))
    sev = float(rng.uniform(0.5, 1.0))
    return Artefact(
        kind=kind,
        axial_mm=(ax0, ax0 + span_ax),
        angle_deg=(an0, an0 + span_an),
        severity=sev,
    )


def _make_artefacts(
    params: dict[str, Any], base: DatasetConfig, rng: np.random.Generator
) -> tuple[Artefact, ...]:
    """Generate randomized artefacts based on probabilities in params. Synthetic data."""
    kinds = (
        ("bubble", "artefact_prob_bubble"),
        ("lost_backwall", "artefact_prob_lost_backwall"),
        ("roughness", "artefact_prob_roughness"),
    )
    artefacts = []
    for kind, key in kinds:
        prob = float(params.get(key, 0.0))
        if rng.random() < prob:
            artefacts.append(_make_one_artefact(kind, base.length_mm, rng))
    return tuple(artefacts)


def make_bar_spec(params: dict[str, Any], base: DatasetConfig, rng: np.random.Generator) -> BarSpec:
    """Construct a complete BarSpec from randomized params and a base config. Synthetic data."""
    mat = _make_material(params, base.material)
    probe = _make_probe(params, base.probe)
    water = load_material("water")
    diam = float(params.get("diameter_mm", base.diameter_mm))
    refls = _make_reflectors(params, base, rng)
    arts = _make_artefacts(params, base, rng)
    seed = int(params.get("seed", rng.integers(0, 2**31 - 1)))
    return BarSpec(
        material=mat,
        probe=probe,
        water=water,
        diameter_mm=diam,
        length_mm=base.length_mm,
        axial_step_mm=1.0,
        n_angle=base.n_angle,
        reflectors=refls,
        artefacts=arts,
        seed=seed,
    )


def _get_worker_seed(base_seed: int, worker_id: int | None = None) -> int:
    """Derive deterministic seed for current worker or environment. Synthetic data."""
    if worker_id is not None:
        return base_seed + worker_id
    info = get_worker_info()
    if info is not None:
        return base_seed + info.id
    return base_seed


class RandomBarStream(IterableDataset):
    """Infinite or finite stream of simulated immersion UT bar scans. Synthetic data.

    Yields (BarScan, params) or transform(BarScan, params) generated on-the-fly.
    Nothing is written to disk.
    """

    def __init__(
        self,
        cfg: RandomizationConfig | None = None,
        base_cfg: DatasetConfig | None = None,
        base_seed: int = 42,
        transform: Callable[[BarScan, dict[str, Any]], Any] | None = None,
        *,
        exclude: tuple[Region, ...] = (),
        max_items: int | None = None,
        worker_id: int | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or load_randomization()
        self.base_cfg = base_cfg or DatasetConfig()
        self.base_seed = base_seed
        self.transform = transform
        self.exclude = exclude
        self.max_items = max_items
        self._worker_id = worker_id

    def __iter__(self) -> Iterator[Any]:
        seed = _get_worker_seed(self.base_seed, self._worker_id)
        rng = np.random.default_rng(seed)
        count = 0
        while self.max_items is None or count < self.max_items:
            params = sample_params(self.cfg, rng, exclude=self.exclude)
            spec = make_bar_spec(params, self.base_cfg, rng)
            scan = simulate_bar(spec)
            count += 1
            if self.transform is not None:
                yield self.transform(scan, params)
            else:
                yield scan, params
