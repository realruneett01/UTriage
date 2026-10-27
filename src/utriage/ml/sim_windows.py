"""Simulator held-out parameter-region generalization window generation.

Synthetic data.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from utriage.ml.bar_data import (
    WindowConfig,
    WindowSet,
    _bar_artefact_windows,
    _bar_flaw_windows,
    _bar_noise_windows,
    _BarContext,
    _concat_sets,
    _partition_grid,
)
from utriage.physics.dataset import DatasetConfig
from utriage.physics.randomize import (
    DEFAULT_HOLDOUT_REGIONS,
    Region,
    _make_one_artefact,
    load_randomization,
    make_bar_spec,
    sample_in_region,
    sample_params,
)
from utriage.physics.simulator import ARTEFACT_KINDS, simulate_bar
from utriage.rules.pipeline import dac_for


@dataclass(frozen=True)
class SimGenSettings:
    """Settings for simulation generalization study (Synthetic data)."""

    n_train_bars: int = 40
    n_test_bars: int = 10
    seed: int = 42
    epochs: int = 15


def sample_train_params(
    settings: SimGenSettings,
    regions: Sequence[Region] = DEFAULT_HOLDOUT_REGIONS,
    rng: np.random.Generator | None = None,
) -> list[dict[str, Any]]:
    """Sample in-distribution parameters excluding held-out regions."""
    gen = rng if rng is not None else np.random.default_rng(settings.seed)
    cfg = load_randomization()
    exclude = tuple(regions)
    return [sample_params(cfg, gen, exclude=exclude) for _ in range(settings.n_train_bars)]


def sample_region_params(
    settings: SimGenSettings,
    region: Region,
    rng: np.random.Generator | None = None,
) -> list[dict[str, Any]]:
    """Sample test parameters strictly inside a held-out region."""
    gen = rng if rng is not None else np.random.default_rng(settings.seed)
    cfg = load_randomization()
    return [sample_in_region(cfg, gen, region) for _ in range(settings.n_test_bars)]


def sample_artefact_params(
    settings: SimGenSettings,
    kind: str,
    regions: Sequence[Region] = DEFAULT_HOLDOUT_REGIONS,
    rng: np.random.Generator | None = None,
) -> list[dict[str, Any]]:
    """Sample in-distribution parameters and attach a forced artefact kind."""
    gen = rng if rng is not None else np.random.default_rng(settings.seed)
    cfg = load_randomization()
    exclude = tuple(regions)
    out: list[dict[str, Any]] = []
    for _ in range(settings.n_test_bars):
        p = sample_params(cfg, gen, exclude=exclude)
        p["forced_artefact"] = kind
        out.append(p)
    return out


def _cut_bar_windows(
    ctx: _BarContext,
    outside: np.ndarray,
    inside: np.ndarray,
    rng: np.random.Generator,
) -> WindowSet:
    """Cut DAC-normalised noise, flaw, and artefact windows from one scan."""
    count = 40
    sub: list[WindowSet] = [
        _bar_noise_windows(ctx, outside, rng, count),
        _bar_flaw_windows(ctx, outside, rng, count),
    ]
    if len(inside) > 0:
        art_set = _bar_artefact_windows(ctx, inside, rng, max(1, count // 4))
        if art_set is not None:
            sub.append(art_set)
    return _concat_sets(sub, ctx.cfg.patch)


def _bar_windows_from_one_param(
    params: dict[str, Any],
    wcfg: WindowConfig,
    base_ds: DatasetConfig,
    rng: np.random.Generator,
) -> WindowSet:
    """Simulate a bar from one randomized parameter dict and extract windows."""
    spec = make_bar_spec(params, base_ds, rng)
    if "forced_artefact" in params:
        art = _make_one_artefact(str(params["forced_artefact"]), spec.length_mm, rng)
        spec = replace(spec, artefacts=(art,))
    scan = simulate_bar(spec)
    dac = dac_for(scan)
    max_d = max(4.0, spec.diameter_mm - 5.0)
    depth_range = (wcfg.depth_range_mm[0], min(wcfg.depth_range_mm[1], max_d))
    bar_wcfg = replace(wcfg, depth_range_mm=depth_range)
    scale = float(params.get("grain_noise_scale", 1.0))
    ctx = _BarContext(scan, dac, bar_wcfg, scale, spec.artefacts)
    outside, inside = _partition_grid(scan, spec.artefacts)
    return _cut_bar_windows(ctx, outside, inside, rng)


def windows_from_params(
    param_list: Sequence[dict[str, Any]],
    wcfg: WindowConfig,
    rng: np.random.Generator,
) -> WindowSet:
    """Build WindowSet from simulated bars using randomized parameter dicts."""
    if not param_list:
        return _concat_sets([], wcfg.patch)
    all_sets: list[WindowSet] = []
    base_ds = DatasetConfig(length_mm=16.0, n_angle=36, mean_reflectors=0.0)
    for p in param_list:
        bar_set = _bar_windows_from_one_param(p, wcfg, base_ds, rng)
        all_sets.append(bar_set)
    return _concat_sets(all_sets, wcfg.patch)


def build_train(
    settings: SimGenSettings,
    regions: Sequence[Region] = DEFAULT_HOLDOUT_REGIONS,
    wcfg: WindowConfig | None = None,
) -> WindowSet:
    """Sample in-distribution training parameters and generate WindowSet."""
    rng = np.random.default_rng(settings.seed)
    params = sample_train_params(settings, regions, rng)
    cfg = wcfg if wcfg is not None else WindowConfig(patch=False, seed=settings.seed)
    return windows_from_params(params, cfg, rng)


def region_test_seed(seed: int | SimGenSettings, region: Region | str) -> int:
    """Derive deterministic test seed for a held-out region."""
    base_seed = seed.seed if isinstance(seed, SimGenSettings) else seed
    name = region.name if isinstance(region, Region) else str(region)
    offset = 1 + (zlib.crc32(name.encode()) % 9999)
    return base_seed + offset


def build_region_test(
    settings: SimGenSettings,
    region: Region,
    wcfg: WindowConfig | None = None,
) -> WindowSet:
    """Sample parameters in specified held-out region and generate WindowSet."""
    rng = np.random.default_rng(region_test_seed(settings.seed, region))
    params = sample_region_params(settings, region, rng)
    cfg = wcfg if wcfg is not None else WindowConfig(patch=False, seed=settings.seed)
    return windows_from_params(params, cfg, rng)


def build_artefact_stress(
    settings: SimGenSettings,
    kinds: Sequence[str] = ARTEFACT_KINDS,
    wcfg: WindowConfig | None = None,
) -> dict[str, WindowSet]:
    """Sample in-distribution parameters with forced artefacts (one test set per kind)."""
    cfg = wcfg if wcfg is not None else WindowConfig(patch=False, seed=settings.seed)
    out: dict[str, WindowSet] = {}
    for idx, kind in enumerate(kinds):
        rng = np.random.default_rng(settings.seed + 100 + idx)
        params = sample_artefact_params(settings, kind, rng=rng)
        out[kind] = windows_from_params(params, cfg, rng)
    return out


def build_in_distribution_test(
    settings: SimGenSettings,
    regions: Sequence[Region] = DEFAULT_HOLDOUT_REGIONS,
    wcfg: WindowConfig | None = None,
) -> WindowSet:
    """Sample in-distribution test parameters and generate WindowSet."""
    rng = np.random.default_rng(settings.seed + 9999)
    cfg_rand = load_randomization()
    exclude = tuple(regions)
    params = [sample_params(cfg_rand, rng, exclude=exclude) for _ in range(settings.n_test_bars)]
    cfg = wcfg if wcfg is not None else WindowConfig(patch=False, seed=settings.seed)
    return windows_from_params(params, cfg, rng)
