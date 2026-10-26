"""Bar-domain ML window extraction and virtual flaw implantation (Synthetic data)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from utriage.physics.simulator import (
    _INCLUSION_FACETS,
    ARTEFACT_KINDS,
    Artefact,
    BarScan,
    BarSpec,
    Material,
    Probe,
    Reflector,
    fbh_amplitude,
    load_material,
    load_probe,
    pulse,
    simulate_bar,
)
from utriage.rules.pipeline import dac_for

WINDOW = 128
CLASSES = ("noise", "flaw", "artefact")

_PATCH_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass(frozen=True)
class WindowConfig:
    """Configuration for cutting and implanting bar scan windows."""

    material: str = "ti64"
    probe: str = "p5"
    noise_scale: tuple[float, float] = (0.5, 2.0)
    fbh_range_mm: tuple[float, float] = (0.3, 1.6)
    depth_range_mm: tuple[float, float] = (4.0, 34.0)
    diameter_mm: float = 40.0
    patch: bool = False
    seed: int = 0


@dataclass
class WindowSet:
    """Collection of cut/implanted A-scan windows (Synthetic data)."""

    x: np.ndarray
    y: np.ndarray
    size_mm: np.ndarray
    depth_mm: np.ndarray
    noise_scale: np.ndarray

    def save(self, path: Path | str) -> None:
        """Save dataset to a compressed npz file."""
        np.savez_compressed(
            Path(path),
            x=self.x,
            y=self.y,
            size_mm=self.size_mm,
            depth_mm=self.depth_mm,
            noise_scale=self.noise_scale,
        )

    @classmethod
    def load(cls, path: Path | str) -> WindowSet:
        """Load dataset from a compressed npz file."""
        with np.load(Path(path)) as data:
            return cls(
                x=data["x"].astype(np.float32),
                y=data["y"].astype(np.int64),
                size_mm=data["size_mm"].astype(np.float32),
                depth_mm=data["depth_mm"].astype(np.float32),
                noise_scale=data["noise_scale"].astype(np.float32),
            )


@dataclass(frozen=True)
class _FlawContext:
    scan: BarScan
    dac: Any
    ref: Reflector
    patch: bool


@dataclass(frozen=True)
class _BarContext:
    scan: BarScan
    dac: Any
    cfg: WindowConfig
    scale: float
    artefacts: tuple[Artefact, ...]


@dataclass(frozen=True)
class _TestContext:
    scan: BarScan
    dac: Any
    cfg: WindowConfig
    scale: float


def _slice_window(ascan: np.ndarray, center_idx: int) -> np.ndarray:
    w = np.zeros(WINDOW, dtype=np.float32)
    s_from = center_idx - WINDOW // 2
    s_to = s_from + WINDOW
    clip_from = max(0, s_from)
    clip_to = min(ascan.shape[0], s_to)
    if clip_to > clip_from:
        w[clip_from - s_from : clip_to - s_from] = ascan[clip_from:clip_to]
    return w


def _sample_index(scan: BarScan, ax_c: int, an_c: int, depth_mm: float) -> int:
    c_m = scan.spec.material.velocity_m_s * 1e-3
    t_if = float(scan.t_if_us[ax_c, an_c]) if scan.t_if_us.ndim == 2 else float(scan.t_if_us)
    t_sample = t_if + 2.0 * depth_mm / c_m
    return int(np.round((t_sample - scan.t0_us) * scan.fs_mhz))


def _cut_site(scan: BarScan, dac: Any, site: np.ndarray, patch: bool) -> np.ndarray:
    ax_i, an_i, depth = int(site[0]), int(site[1]), float(site[2])
    n_ax, n_an = scan.ascans.shape[0], scan.ascans.shape[1]
    offsets = _PATCH_OFFSETS if patch else ((0, 0),)
    ref = max(float(np.asarray(dac(depth))), 1e-12)
    channels = np.empty((len(offsets), WINDOW), dtype=np.float32)
    for c_idx, (d_ax, d_an) in enumerate(offsets):
        ax_c = int(np.clip(ax_i + d_ax, 0, n_ax - 1))
        an_c = (an_i + d_an) % n_an
        s_idx = _sample_index(scan, ax_c, an_c, depth)
        channels[c_idx] = _slice_window(scan.ascans[ax_c, an_c], s_idx) / ref
    return channels


def cut_windows(
    scan: BarScan,
    dac: Any,
    sites: np.ndarray,
    patch: bool,
) -> np.ndarray:
    """Cut DAC-normalised windows at specified sites."""
    n_sites = len(sites)
    n_ch = 9 if patch else 1
    if n_sites == 0:
        return np.zeros((0, n_ch, WINDOW), dtype=np.float32)
    out = np.empty((n_sites, n_ch, WINDOW), dtype=np.float32)
    for i in range(n_sites):
        out[i] = _cut_site(scan, dac, sites[i], patch)
    return out


def _single_pulse_sum(probe: Probe, kind: str) -> np.ndarray:
    period = 1.0 / probe.frequency_mhz
    dt = (np.arange(WINDOW) - WINDOW // 2) / probe.sampling_mhz
    facets = _INCLUSION_FACETS if kind == "inclusion" else ((0.0, 1.0),)
    out = np.zeros(WINDOW, dtype=np.float32)
    for delay, w in facets:
        out += w * pulse(dt - delay * period, probe)
    return out


def _patch_weights(spec: BarSpec, depth_mm: float) -> np.ndarray:
    weights = np.empty(len(_PATCH_OFFSETS), dtype=np.float32)
    radius = spec.diameter_mm / 2.0 - depth_mm
    sigma2 = 2.0 * spec.probe.beam_sigma_mm**2
    ang_step = 360.0 / spec.n_angle
    for idx, (d_ax, d_an) in enumerate(_PATCH_OFFSETS):
        d_axial = d_ax * spec.axial_step_mm
        arc = abs(radius) * np.deg2rad(d_an * ang_step)
        r2 = d_axial**2 + arc**2
        w = np.exp(-r2 / sigma2)
        weights[idx] = w if w > 1e-3 else 0.0
    return weights


def _site_flaw_signal(ctx: _FlawContext, depth_mm: float) -> np.ndarray:
    scan, ref = ctx.scan, ctx.ref
    base = fbh_amplitude(ref.diameter_mm, depth_mm, scan.spec)
    gain_fac = 10.0 ** (scan.spec.probe.gain_db / 20.0)
    raw_amp = -base * ref.amplitude_factor * gain_fac
    ref_val = max(float(np.asarray(ctx.dac(depth_mm))), 1e-12)
    center = (raw_amp / ref_val) * _single_pulse_sum(scan.spec.probe, ref.kind)
    if not ctx.patch:
        return center[None, :]
    weights = _patch_weights(scan.spec, depth_mm)
    return weights[:, None] * center[None, :]


def flaw_signal(
    scan: BarScan,
    dac: Any,
    ref: Reflector,
    sites: np.ndarray,
    patch: bool,
) -> np.ndarray:
    """Noise-free echo of reflector at each site."""
    n_sites = len(sites)
    n_ch = 9 if patch else 1
    if n_sites == 0:
        return np.zeros((0, n_ch, WINDOW), dtype=np.float32)
    ctx = _FlawContext(scan=scan, dac=dac, ref=ref, patch=patch)
    out = np.empty((n_sites, n_ch, WINDOW), dtype=np.float32)
    for i in range(n_sites):
        out[i] = _site_flaw_signal(ctx, float(sites[i, 2]))
    return out


def implant(noise: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Implant flaw signal into noise window."""
    return (noise + signal).astype(np.float32)


def _concat_sets(sets: list[WindowSet], patch: bool) -> WindowSet:
    if not sets:
        n_ch = 9 if patch else 1
        return WindowSet(
            x=np.zeros((0, n_ch, WINDOW), dtype=np.float32),
            y=np.zeros(0, dtype=np.int64),
            size_mm=np.zeros(0, dtype=np.float32),
            depth_mm=np.zeros(0, dtype=np.float32),
            noise_scale=np.zeros(0, dtype=np.float32),
        )
    return WindowSet(
        x=np.concatenate([s.x for s in sets], axis=0),
        y=np.concatenate([s.y for s in sets], axis=0),
        size_mm=np.concatenate([s.size_mm for s in sets], axis=0),
        depth_mm=np.concatenate([s.depth_mm for s in sets], axis=0),
        noise_scale=np.concatenate([s.noise_scale for s in sets], axis=0),
    )


def _random_artefact(rng: np.random.Generator) -> tuple[Artefact, ...]:
    kind = str(rng.choice(ARTEFACT_KINDS))
    ax0 = float(rng.uniform(2.0, 6.0))
    ax1 = float(rng.uniform(ax0 + 2.0, 14.0))
    ang0 = float(rng.uniform(0.0, 180.0))
    ang1 = float(rng.uniform(ang0 + 30.0, min(360.0, ang0 + 120.0)))
    sev = float(rng.uniform(0.5, 1.0))
    return (Artefact(kind, (ax0, ax1), (ang0, ang1), sev),)


def _simulate_training_bar(
    cfg: WindowConfig,
    rng: np.random.Generator,
) -> tuple[BarScan, Any, float, tuple[Artefact, ...]]:
    base_mat = load_material(cfg.material)
    probe = load_probe(cfg.probe)
    water = load_material("water")
    scale = float(rng.uniform(cfg.noise_scale[0], cfg.noise_scale[1]))
    mat = Material(
        base_mat.name,
        base_mat.velocity_m_s,
        base_mat.density_kg_m3,
        base_mat.attenuation_db_mm_mhz,
        base_mat.grain_noise * scale,
    )
    artefacts = _random_artefact(rng) if rng.random() < 0.30 else ()
    spec = BarSpec(
        material=mat,
        probe=probe,
        water=water,
        diameter_mm=cfg.diameter_mm,
        length_mm=16.0,
        axial_step_mm=1.0,
        n_angle=36,
        artefacts=artefacts,
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    scan = simulate_bar(spec)
    return scan, dac_for(scan), scale, artefacts


def _partition_grid(
    scan: BarScan,
    artefacts: tuple[Artefact, ...],
) -> tuple[np.ndarray, np.ndarray]:
    n_ax, n_an = len(scan.axial_mm), len(scan.angle_deg)
    if not artefacts:
        all_pairs = np.indices((n_ax, n_an)).reshape(2, -1).T
        return all_pairs, np.zeros((0, 2), dtype=int)
    art = artefacts[0]
    ax_mask = (scan.axial_mm >= art.axial_mm[0]) & (scan.axial_mm <= art.axial_mm[1])
    an_mask = (scan.angle_deg >= art.angle_deg[0]) & (scan.angle_deg <= art.angle_deg[1])
    in_mask = ax_mask[:, None] & an_mask[None, :]
    return np.argwhere(~in_mask), np.argwhere(in_mask)


def _bar_noise_windows(
    ctx: _BarContext,
    outside: np.ndarray,
    rng: np.random.Generator,
    count: int,
) -> WindowSet:
    idx = rng.choice(len(outside), size=count, replace=True)
    depths = rng.uniform(ctx.cfg.depth_range_mm[0], ctx.cfg.depth_range_mm[1], size=count)
    sites = np.column_stack([outside[idx], depths])
    raw = cut_windows(ctx.scan, ctx.dac, sites, ctx.cfg.patch)
    x = implant(raw, np.zeros_like(raw))
    return WindowSet(
        x=x,
        y=np.full(count, CLASSES.index("noise"), dtype=np.int64),
        size_mm=np.zeros(count, dtype=np.float32),
        depth_mm=depths.astype(np.float32),
        noise_scale=np.full(count, ctx.scale, dtype=np.float32),
    )


def _bar_artefact_windows(
    ctx: _BarContext,
    inside: np.ndarray,
    rng: np.random.Generator,
    count: int,
) -> WindowSet | None:
    if count == 0 or len(inside) == 0:
        return None
    idx = rng.choice(len(inside), size=count, replace=True)
    depths = rng.uniform(ctx.cfg.depth_range_mm[0], ctx.cfg.depth_range_mm[1], size=count)
    sites = np.column_stack([inside[idx], depths])
    raw = cut_windows(ctx.scan, ctx.dac, sites, ctx.cfg.patch)
    x = implant(raw, np.zeros_like(raw))
    return WindowSet(
        x=x,
        y=np.full(count, CLASSES.index("artefact"), dtype=np.int64),
        size_mm=np.zeros(count, dtype=np.float32),
        depth_mm=depths.astype(np.float32),
        noise_scale=np.full(count, ctx.scale, dtype=np.float32),
    )


def _make_flaw_reflector(
    ctx: _BarContext,
    pair: np.ndarray,
    depth: float,
    rng: np.random.Generator,
) -> tuple[Reflector, float]:
    size = float(rng.uniform(ctx.cfg.fbh_range_mm[0], ctx.cfg.fbh_range_mm[1]))
    is_incl = rng.random() < 0.30
    kind = "inclusion" if is_incl else "fbh"
    amp_fac = float(rng.uniform(0.3, 1.0)) if is_incl else 1.0
    ax_i, an_i = int(pair[0]), int(pair[1])
    ref = Reflector(
        axial_mm=ctx.scan.axial_mm[ax_i],
        angle_deg=ctx.scan.angle_deg[an_i],
        depth_mm=depth,
        diameter_mm=size,
        kind=kind,
        amplitude_factor=amp_fac,
    )
    return ref, size


def _bar_flaw_windows(
    ctx: _BarContext,
    outside: np.ndarray,
    rng: np.random.Generator,
    count: int,
) -> WindowSet:
    idx = rng.choice(len(outside), size=count, replace=True)
    depths = rng.uniform(ctx.cfg.depth_range_mm[0], ctx.cfg.depth_range_mm[1], size=count)
    sites = np.column_stack([outside[idx], depths])
    fresh = cut_windows(ctx.scan, ctx.dac, sites, ctx.cfg.patch)
    n_ch = 9 if ctx.cfg.patch else 1
    x = np.empty((count, n_ch, WINDOW), dtype=np.float32)
    sizes = np.empty(count, dtype=np.float32)
    for k in range(count):
        ref, size = _make_flaw_reflector(ctx, outside[idx[k]], depths[k], rng)
        sig = flaw_signal(ctx.scan, ctx.dac, ref, sites[k : k + 1], ctx.cfg.patch)
        x[k] = implant(fresh[k : k + 1], sig)[0]
        sizes[k] = size
    return WindowSet(
        x=x,
        y=np.full(count, CLASSES.index("flaw"), dtype=np.int64),
        size_mm=sizes,
        depth_mm=depths.astype(np.float32),
        noise_scale=np.full(count, ctx.scale, dtype=np.float32),
    )


def make_training_set(
    cfg: WindowConfig,
    n_bars: int,
    per_bar: int,
) -> WindowSet:
    """Generate deterministic training set via virtual flaw implantation."""
    rng = np.random.default_rng(cfg.seed)
    all_sets: list[WindowSet] = []
    for _ in range(n_bars):
        scan, dac, scale, arts = _simulate_training_bar(cfg, rng)
        ctx = _BarContext(scan, dac, cfg, scale, arts)
        outside, inside = _partition_grid(scan, arts)
        all_sets.append(_bar_noise_windows(ctx, outside, rng, per_bar))
        art_set = _bar_artefact_windows(ctx, inside, rng, per_bar // 4)
        if art_set is not None:
            all_sets.append(art_set)
        all_sets.append(_bar_flaw_windows(ctx, outside, rng, per_bar))
    return _concat_sets(all_sets, cfg.patch)


def _plant_test_reflectors(
    cfg: WindowConfig,
    sizes_mm: tuple[float, ...],
    rng: np.random.Generator,
) -> tuple[Reflector, ...]:
    n_sizes = len(sizes_mm)
    ax_spacing = 32.0 / max(n_sizes, 1)
    ang_spacing = 360.0 / max(n_sizes, 1)
    refs: list[Reflector] = []
    for k, size in enumerate(sizes_mm):
        depth = float(rng.uniform(cfg.depth_range_mm[0], cfg.depth_range_mm[1]))
        ax = 8.0 + k * ax_spacing
        ang = (k * ang_spacing) % 360.0
        refs.append(Reflector(ax, ang, depth, size, "fbh"))
    return tuple(refs)


def _simulate_test_bar(
    cfg: WindowConfig,
    scale: float,
    reflectors: tuple[Reflector, ...],
    rng: np.random.Generator,
) -> tuple[BarScan, Any]:
    base_mat = load_material(cfg.material)
    probe = load_probe(cfg.probe)
    water = load_material("water")
    mat = Material(
        base_mat.name,
        base_mat.velocity_m_s,
        base_mat.density_kg_m3,
        base_mat.attenuation_db_mm_mhz,
        base_mat.grain_noise * scale,
    )
    spec = BarSpec(
        material=mat,
        probe=probe,
        water=water,
        diameter_mm=cfg.diameter_mm,
        length_mm=48.0,
        axial_step_mm=1.0,
        n_angle=72,
        reflectors=reflectors,
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    scan = simulate_bar(spec)
    return scan, dac_for(scan)


def _test_flaw_windows(
    ctx: _TestContext,
    reflectors: tuple[Reflector, ...],
) -> WindowSet:
    scan = ctx.scan
    sites = np.empty((len(reflectors), 3), dtype=np.float64)
    sizes = np.empty(len(reflectors), dtype=np.float32)
    depths = np.empty(len(reflectors), dtype=np.float32)
    for i, ref in enumerate(reflectors):
        ax_i = int(np.round(ref.axial_mm / scan.spec.axial_step_mm))
        an_i = int(np.round(ref.angle_deg / (360.0 / scan.spec.n_angle))) % scan.spec.n_angle
        sites[i] = [ax_i, an_i, ref.depth_mm]
        sizes[i] = ref.diameter_mm
        depths[i] = ref.depth_mm
    x = cut_windows(scan, ctx.dac, sites, ctx.cfg.patch)
    return WindowSet(
        x=x,
        y=np.full(len(reflectors), CLASSES.index("flaw"), dtype=np.int64),
        size_mm=sizes,
        depth_mm=depths,
        noise_scale=np.full(len(reflectors), ctx.scale, dtype=np.float32),
    )


def _is_noise_site_valid(
    scan: BarScan,
    refs: tuple[Reflector, ...],
    ax_i: int,
    an_i: int,
) -> bool:
    ax_pos = scan.axial_mm[ax_i]
    an_pos = scan.angle_deg[an_i]
    radius = scan.spec.diameter_mm / 2.0
    for r in refs:
        d_ax = ax_pos - r.axial_mm
        d_ang = (an_pos - r.angle_deg + 180.0) % 360.0 - 180.0
        arc = radius * np.deg2rad(d_ang)
        if d_ax**2 + arc**2 < 16.0:
            return False
    return True


def _test_noise_windows(
    ctx: _TestContext,
    refs: tuple[Reflector, ...],
    rng: np.random.Generator,
) -> WindowSet:
    count = len(refs)
    sites = np.empty((count, 3), dtype=np.float64)
    depths = np.empty(count, dtype=np.float32)
    n_ax, n_an = len(ctx.scan.axial_mm), len(ctx.scan.angle_deg)
    found = 0
    while found < count:
        ax_i = int(rng.integers(0, n_ax))
        an_i = int(rng.integers(0, n_an))
        if _is_noise_site_valid(ctx.scan, refs, ax_i, an_i):
            d = float(rng.uniform(ctx.cfg.depth_range_mm[0], ctx.cfg.depth_range_mm[1]))
            sites[found] = [ax_i, an_i, d]
            depths[found] = d
            found += 1
    x = cut_windows(ctx.scan, ctx.dac, sites, ctx.cfg.patch)
    return WindowSet(
        x=x,
        y=np.full(count, CLASSES.index("noise"), dtype=np.int64),
        size_mm=np.zeros(count, dtype=np.float32),
        depth_mm=depths,
        noise_scale=np.full(count, ctx.scale, dtype=np.float32),
    )


def _test_bar_windows(
    cfg: WindowConfig,
    sizes_mm: tuple[float, ...],
    scale: float,
    rng: np.random.Generator,
) -> WindowSet:
    refs = _plant_test_reflectors(cfg, sizes_mm, rng)
    scan, dac = _simulate_test_bar(cfg, scale, refs, rng)
    ctx = _TestContext(scan, dac, cfg, scale)
    flaw_set = _test_flaw_windows(ctx, refs)
    noise_set = _test_noise_windows(ctx, refs, rng)
    return _concat_sets([flaw_set, noise_set], cfg.patch)


def make_test_set(
    cfg: WindowConfig,
    n_bars: int,
    sizes_mm: tuple[float, ...],
    noise_scales: tuple[float, ...],
) -> WindowSet:
    """Generate test set using the full simulator path (no implant)."""
    rng = np.random.default_rng(cfg.seed)
    all_sets: list[WindowSet] = []
    for scale in noise_scales:
        for _ in range(n_bars):
            all_sets.append(_test_bar_windows(cfg, sizes_mm, scale, rng))
    return _concat_sets(all_sets, cfg.patch)
