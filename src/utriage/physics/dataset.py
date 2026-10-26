"""Dataset generator, save, and load for immersion UT scans. Output is Synthetic data."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from utriage.physics.simulator import (
    ARTEFACT_KINDS,
    SIM_VERSION,
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
class DatasetConfig:
    """Configuration for synthetic bar dataset generation."""

    n_bars: int = 5
    material: str = "ti64"
    probe: str = "p5"
    seed: int = 0
    diameter_mm: float = 40.0
    length_mm: float = 64.0
    n_angle: int = 120
    mean_reflectors: float = 2.0
    artefact_rate: float = 0.3
    noise_scale: tuple[float, float] = (0.5, 2.0)
    fbh_range_mm: tuple[float, float] = (0.4, 2.0)


def _random_reflector(cfg: DatasetConfig, rng: np.random.Generator) -> Reflector:
    is_fbh = bool(rng.random() < 0.7)
    kind = "fbh" if is_fbh else "inclusion"
    amp = 1.0 if is_fbh else float(rng.uniform(0.3, 1.0))
    diam = float(rng.uniform(cfg.fbh_range_mm[0], cfg.fbh_range_mm[1]))
    depth = float(rng.uniform(3.0, max(3.0, cfg.diameter_mm - 5.0)))
    ax_min = min(4.0, max(0.0, cfg.length_mm - 4.0))
    ax_max = max(4.0, cfg.length_mm - 4.0)
    axial = float(rng.uniform(ax_min, ax_max))
    angle = float(rng.uniform(0.0, 360.0))
    return Reflector(
        axial_mm=axial,
        angle_deg=angle,
        depth_mm=depth,
        diameter_mm=diam,
        kind=kind,
        amplitude_factor=amp,
    )


def _random_artefact(cfg: DatasetConfig, rng: np.random.Generator) -> Artefact:
    kind = str(rng.choice(ARTEFACT_KINDS))
    span_ax = min(float(rng.uniform(6.0, 20.0)), cfg.length_mm)
    ax0 = float(rng.uniform(0.0, max(0.0, cfg.length_mm - span_ax)))
    ax1 = min(cfg.length_mm, ax0 + span_ax)
    span_an = float(rng.uniform(20.0, 90.0))
    an0 = float(rng.uniform(0.0, 360.0 - span_an))
    an1 = an0 + span_an
    sev = float(rng.uniform(0.5, 1.0))
    return Artefact(
        kind=kind,
        axial_mm=(ax0, ax1),
        angle_deg=(an0, an1),
        severity=sev,
    )


def random_bar_spec(cfg: DatasetConfig, bar_seed: int) -> BarSpec:
    """Generate a randomized BarSpec for synthetic data generation."""
    rng = np.random.default_rng(bar_seed)
    n_refl = int(rng.poisson(cfg.mean_reflectors))
    reflectors = tuple(_random_reflector(cfg, rng) for _ in range(n_refl))

    noise_mult = float(rng.uniform(cfg.noise_scale[0], cfg.noise_scale[1]))
    base_mat = load_material(cfg.material)
    mat = replace(base_mat, grain_noise=base_mat.grain_noise * noise_mult)

    artefacts: tuple[Artefact, ...] = ()
    if rng.random() < cfg.artefact_rate:
        artefacts = (_random_artefact(cfg, rng),)

    return BarSpec(
        material=mat,
        probe=load_probe(cfg.probe),
        water=load_material("water"),
        diameter_mm=cfg.diameter_mm,
        length_mm=cfg.length_mm,
        axial_step_mm=1.0,
        n_angle=cfg.n_angle,
        reflectors=reflectors,
        artefacts=artefacts,
        seed=bar_seed,
    )


def _ground_truth_dict(scan: BarScan) -> dict:
    spec = scan.spec
    return {
        "label": "Synthetic data",
        "sim_version": SIM_VERSION,
        "seed": spec.seed,
        "material": asdict(spec.material),
        "probe": asdict(spec.probe),
        "water": asdict(spec.water),
        "diameter_mm": spec.diameter_mm,
        "length_mm": spec.length_mm,
        "axial_step_mm": spec.axial_step_mm,
        "n_angle": spec.n_angle,
        "reflectors": [asdict(r) for r in spec.reflectors],
        "artefacts": [asdict(a) for a in spec.artefacts],
    }


def save_scan(scan: BarScan, stem: Path) -> tuple[Path, Path]:
    """Save BarScan arrays as compressed NPZ and ground truth as JSON."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    npz_path = stem.with_suffix(".npz")
    json_path = stem.with_suffix(".json")

    np.savez_compressed(
        npz_path,
        ascans=scan.ascans.astype(np.float16),
        t0_us=scan.t0_us,
        fs_mhz=scan.fs_mhz,
        axial_mm=scan.axial_mm,
        angle_deg=scan.angle_deg,
        t_if_us=scan.t_if_us,
        missing=scan.missing,
    )
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(_ground_truth_dict(scan), fh, indent=2)

    return npz_path, json_path


def _rebuild_bar_spec(truth: dict) -> BarSpec:
    reflectors = tuple(Reflector(**r) for r in truth.get("reflectors", []))
    artefacts = tuple(
        Artefact(
            kind=a["kind"],
            axial_mm=(float(a["axial_mm"][0]), float(a["axial_mm"][1])),
            angle_deg=(float(a["angle_deg"][0]), float(a["angle_deg"][1])),
            severity=float(a["severity"]),
        )
        for a in truth.get("artefacts", [])
    )
    return BarSpec(
        material=Material(**truth["material"]),
        probe=Probe(**truth["probe"]),
        water=Material(**truth["water"]),
        diameter_mm=float(truth["diameter_mm"]),
        length_mm=float(truth["length_mm"]),
        axial_step_mm=float(truth["axial_step_mm"]),
        n_angle=int(truth["n_angle"]),
        reflectors=reflectors,
        artefacts=artefacts,
        seed=int(truth["seed"]),
    )


def load_scan(npz_path: Path) -> BarScan:
    """Read npz and sibling json, rebuild BarSpec and BarScan with ascans as float32."""
    npz_path = Path(npz_path)
    json_path = npz_path.with_suffix(".json")
    with json_path.open("r", encoding="utf-8") as fh:
        truth = json.load(fh)

    with np.load(npz_path) as data:
        ascans = data["ascans"].astype(np.float32)
        t0_us = float(data["t0_us"])
        fs_mhz = float(data["fs_mhz"])
        axial_mm = np.asarray(data["axial_mm"])
        angle_deg = np.asarray(data["angle_deg"])
        t_if_us = np.asarray(data["t_if_us"])
        missing = np.asarray(data["missing"])

    spec = _rebuild_bar_spec(truth)
    return BarScan(ascans, t0_us, fs_mhz, axial_mm, angle_deg, t_if_us, spec, missing)


def generate_dataset(cfg: DatasetConfig, out_dir: Path) -> list[Path]:
    """Generate synthetic dataset of scans and write manifest.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_paths: list[Path] = []
    for i in range(cfg.n_bars):
        bar_seed = cfg.seed * 1000 + i
        spec = random_bar_spec(cfg, bar_seed)
        scan = simulate_bar(spec)
        stem = out_dir / f"bar_{i:04d}"
        npz_path, _ = save_scan(scan, stem)
        npz_paths.append(npz_path)

    manifest = {
        "label": "Synthetic data",
        "sim_version": SIM_VERSION,
        "config": asdict(cfg),
        "files": [p.name for p in npz_paths],
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return npz_paths
