"""Physics-based immersion UT simulator for rotating round bars. Output is Synthetic data.

Every equation is explained in ``docs/study/01-simulator-physics.md``; every simplification is
listed in ``docs/decisions/0003-simulator-assumptions.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml
from scipy.signal import hilbert

SIM_VERSION = "1.0.0"
CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
ARTEFACT_KINDS = ("bubble", "lost_backwall", "waterpath_drift", "roughness", "missing_coverage")
REFLECTOR_KINDS = ("fbh", "inclusion")
_PRE_US = 2.0  # time recorded before the earliest interface echo
_POST_US = 3.0  # time recorded after the latest back-wall echo
_INCLUSION_FACETS = ((0.0, 1.0), (0.35, -0.6), (0.7, 0.4))  # (delay in periods, weight)


@dataclass(frozen=True)
class Material:
    """Acoustic properties of one material (see configs/materials.yaml)."""

    name: str
    velocity_m_s: float
    density_kg_m3: float
    attenuation_db_mm_mhz: float
    grain_noise: float

    @property
    def impedance(self) -> float:
        """Acoustic impedance Z = rho * c in kg/(m^2 s)."""
        return self.density_kg_m3 * self.velocity_m_s


@dataclass(frozen=True)
class Probe:
    """Probe and acquisition settings (see configs/probe.yaml)."""

    frequency_mhz: float
    fractional_bandwidth: float
    water_path_mm: float
    focal_depth_mm: float
    focal_zone_mm: float
    beam_sigma_mm: float
    sampling_mhz: float
    gain_db: float
    electronic_noise: float
    k_fbh: float


@dataclass(frozen=True)
class Reflector:
    """A planted reflector. ``amplitude_factor`` < 1 makes an inclusion weaker than its FBH size."""

    axial_mm: float
    angle_deg: float
    depth_mm: float
    diameter_mm: float
    kind: str = "fbh"
    amplitude_factor: float = 1.0


@dataclass(frozen=True)
class Artefact:
    """A planted data-quality artefact over an axial x angle region."""

    kind: str
    axial_mm: tuple[float, float]
    angle_deg: tuple[float, float]
    severity: float = 1.0


@dataclass(frozen=True)
class BarSpec:
    """Everything needed to simulate one bar scan deterministically."""

    material: Material
    probe: Probe
    water: Material
    diameter_mm: float = 40.0
    length_mm: float = 64.0
    axial_step_mm: float = 1.0
    n_angle: int = 120
    reflectors: tuple[Reflector, ...] = ()
    artefacts: tuple[Artefact, ...] = ()
    seed: int = 0


@dataclass
class BarScan:
    """Simulated helical scan: RF A-scans in a cube of (axial, angle, time). Synthetic data."""

    ascans: np.ndarray
    t0_us: float
    fs_mhz: float
    axial_mm: np.ndarray
    angle_deg: np.ndarray
    t_if_us: np.ndarray
    spec: BarSpec
    missing: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    @property
    def time_us(self) -> np.ndarray:
        """Time axis in microseconds."""
        return self.t0_us + np.arange(self.ascans.shape[-1]) / self.fs_mhz


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_material(key: str, path: Path | None = None) -> Material:
    """Load one material entry from ``configs/materials.yaml``."""
    entry = _load_yaml(path or CONFIG_DIR / "materials.yaml")[key]
    return Material(
        name=entry.get("name", key),
        velocity_m_s=float(entry["velocity_m_s"]),
        density_kg_m3=float(entry["density_kg_m3"]),
        attenuation_db_mm_mhz=float(entry["attenuation_db_mm_mhz"]),
        grain_noise=float(entry["grain_noise"]),
    )


def load_probe(key: str = "p5", path: Path | None = None) -> Probe:
    """Load one probe entry from ``configs/probe.yaml``."""
    entry = _load_yaml(path or CONFIG_DIR / "probe.yaml")[key]
    names = Probe.__dataclass_fields__
    return Probe(**{k: float(entry[k]) for k in names})


def pulse(t_us: np.ndarray, probe: Probe) -> np.ndarray:
    """Gaussian-modulated cosine p(t) = exp(-t^2 / 2s^2) cos(2 pi f t), t in microseconds."""
    sigma_f = probe.fractional_bandwidth * probe.frequency_mhz / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    s = 1.0 / (2.0 * np.pi * sigma_f)
    return np.exp(-(t_us**2) / (2.0 * s**2)) * np.cos(2.0 * np.pi * probe.frequency_mhz * t_us)


def reflection_coefficient(z1: float, z2: float) -> float:
    """Pressure reflection coefficient R = (Z2 - Z1) / (Z2 + Z1) going from medium 1 into 2."""
    return (z2 - z1) / (z2 + z1)


def round_trip_transmission(z1: float, z2: float) -> float:
    """Round-trip pressure transmission T12 * T21 = 1 - R^2."""
    return 1.0 - reflection_coefficient(z1, z2) ** 2


def attenuation_factor(
    depth_mm: np.ndarray | float, material: Material, probe: Probe
) -> np.ndarray:
    """Round-trip amplitude loss 10^(-2 a f z / 20) for metal path ``depth_mm``."""
    alpha_db_mm = material.attenuation_db_mm_mhz * probe.frequency_mhz
    return 10.0 ** (-2.0 * alpha_db_mm * np.asarray(depth_mm) / 20.0)


def beam_factor(depth_mm: np.ndarray | float, probe: Probe) -> np.ndarray:
    """On-axis focal factor G(z) = 1 / sqrt(1 + ((z - z_f) / z_R)^2)."""
    u = (np.asarray(depth_mm) - probe.focal_depth_mm) / probe.focal_zone_mm
    return 1.0 / np.sqrt(1.0 + u**2)


class Setup(Protocol):
    """Anything that names a material, a probe and the couplant (BarSpec, CalibrationBlock)."""

    material: Material
    probe: Probe
    water: Material


def fbh_amplitude(diameter_mm: float, depth_mm: float, setup: Setup) -> float:
    """Peak echo amplitude of a flat-bottom hole: k (d/1mm)^2 T_rt A(z) G(z), before gain."""
    material, probe, water = setup.material, setup.probe, setup.water
    t_rt = round_trip_transmission(water.impedance, material.impedance)
    losses = attenuation_factor(depth_mm, material, probe) * beam_factor(depth_mm, probe)
    return float(probe.k_fbh * diameter_mm**2 * t_rt * losses)


def envelope(rf: np.ndarray) -> np.ndarray:
    """Envelope |analytic signal| along the last axis (Hilbert transform)."""
    return np.abs(hilbert(rf, axis=-1))


def depth_of_time(t_us: np.ndarray, t_if_us: np.ndarray, material: Material) -> np.ndarray:
    """Metal depth in mm for time ``t_us`` after the interface echo at ``t_if_us``."""
    return (t_us - t_if_us) * material.velocity_m_s * 1e-3 / 2.0


def grain_noise(
    t_us: np.ndarray, t_if_us: np.ndarray, spec: BarSpec, rng: np.random.Generator
) -> np.ndarray:
    """Grain backscatter: white noise convolved with the pulse, scaled by depth losses.

    One noise trace is drawn per entry of ``t_if_us`` (its shape is the leading shape).
    """
    mat, probe = spec.material, spec.probe
    n_t = t_us.size
    kernel = pulse((np.arange(n_t) - n_t // 2) / probe.sampling_mhz, probe)
    kernel /= np.sqrt(np.sum(kernel**2))
    white = rng.standard_normal((*t_if_us.shape, n_t))
    nfft = 2 * n_t
    conv = np.fft.irfft(np.fft.rfft(white, nfft) * np.fft.rfft(kernel, nfft), nfft)
    colored = conv[..., n_t // 2 : n_t // 2 + n_t]
    depth = depth_of_time(t_us, t_if_us[..., None], mat)
    inside = (depth > 0) & (depth < spec.diameter_mm)
    t_rt = round_trip_transmission(spec.water.impedance, mat.impedance)
    level = mat.grain_noise * (probe.frequency_mhz / 5.0) ** 2 * t_rt
    shape = level * attenuation_factor(depth, mat, probe) * beam_factor(depth, probe)
    return colored * np.where(inside, shape, 0.0)


def _region(art: Artefact, axial: np.ndarray, angle: np.ndarray) -> np.ndarray:
    in_ax = (axial >= art.axial_mm[0]) & (axial <= art.axial_mm[1])
    in_an = (angle >= art.angle_deg[0]) & (angle <= art.angle_deg[1])
    return in_ax[:, None] & in_an[None, :]


@dataclass
class _Fields:
    """Per-A-scan multipliers and offsets that artefacts modify."""

    water_path_mm: np.ndarray
    interface: np.ndarray
    metal: np.ndarray
    backwall: np.ndarray
    bubble: np.ndarray
    surface_noise: np.ndarray
    missing: np.ndarray


def _apply_bubble(f: _Fields, m: np.ndarray, sev: float) -> None:
    f.bubble[m] = 0.5 * sev
    f.interface[m] *= 0.3
    f.metal[m] *= 0.3


def _apply_lost_backwall(f: _Fields, m: np.ndarray, sev: float) -> None:
    f.backwall[m] *= max(0.0, 1.0 - 0.95 * sev)


def _apply_drift(f: _Fields, m: np.ndarray, sev: float) -> None:
    ramp = np.cumsum(m, axis=0) / max(1, int(m.sum(axis=0).max()))
    f.water_path_mm += np.where(m, 3.0 * sev * ramp, 0.0)


def _apply_roughness(f: _Fields, m: np.ndarray, sev: float) -> None:
    f.interface[m] *= 0.6
    f.metal[m] *= 0.6
    f.surface_noise[m] = 0.05 * sev


def _apply_missing(f: _Fields, m: np.ndarray, sev: float) -> None:
    f.missing |= m


_ARTEFACT_APPLY = {
    "bubble": _apply_bubble,
    "lost_backwall": _apply_lost_backwall,
    "waterpath_drift": _apply_drift,
    "roughness": _apply_roughness,
    "missing_coverage": _apply_missing,
}


def _artefact_fields(spec: BarSpec, axial: np.ndarray, angle: np.ndarray) -> _Fields:
    shape = (axial.size, angle.size)
    f = _Fields(
        water_path_mm=np.full(shape, spec.probe.water_path_mm),
        interface=np.ones(shape),
        metal=np.ones(shape),
        backwall=np.ones(shape),
        bubble=np.zeros(shape),
        surface_noise=np.zeros(shape),
        missing=np.zeros(shape, dtype=bool),
    )
    for art in spec.artefacts:
        _ARTEFACT_APPLY[art.kind](f, _region(art, axial, angle), art.severity)
    return f


def lateral_weight(
    spec: BarSpec, ref: Reflector, axial: np.ndarray, angle: np.ndarray
) -> np.ndarray:
    """Beam weight exp(-r^2 / 2 w^2) of a reflector seen from every scan position."""
    radius = spec.diameter_mm / 2.0 - ref.depth_mm
    d_ang = (angle - ref.angle_deg + 180.0) % 360.0 - 180.0
    arc = np.abs(radius) * np.deg2rad(d_ang)
    r2 = (axial - ref.axial_mm)[:, None] ** 2 + arc[None, :] ** 2
    return np.exp(-r2 / (2.0 * spec.probe.beam_sigma_mm**2))


@dataclass
class _Canvas:
    """The A-scan cube being drawn, with its time axis, interface times and scan grid."""

    cube: np.ndarray
    t_us: np.ndarray
    t_if: np.ndarray
    axial: np.ndarray
    angle: np.ndarray
    probe: Probe


def _echo(c: _Canvas, t_echo: np.ndarray, amp: np.ndarray) -> None:
    """Add amp * p(t - t_echo) to every A-scan (only where amp is non-zero)."""
    idx = np.nonzero(amp)
    if idx[0].size:
        shifted = c.t_us[None, :] - t_echo[idx][:, None]
        c.cube[idx] += amp[idx][:, None] * pulse(shifted, c.probe)


def _reflector_echoes(c: _Canvas, spec: BarSpec, f: _Fields) -> None:
    period = 1.0 / spec.probe.frequency_mhz
    for ref in spec.reflectors:
        base = fbh_amplitude(ref.diameter_mm, ref.depth_mm, spec)
        weight = lateral_weight(spec, ref, c.axial, c.angle) * f.metal * ref.amplitude_factor
        amp = -base * np.where(weight > 1e-3, weight, 0.0)  # metal -> void: phase inverts
        t_ref = c.t_if + 2.0 * ref.depth_mm / (spec.material.velocity_m_s * 1e-3)
        facets = _INCLUSION_FACETS if ref.kind == "inclusion" else ((0.0, 1.0),)
        for delay, w in facets:
            _echo(c, t_ref + delay * period, w * amp)


def _time_axis(spec: BarSpec, t_if: np.ndarray) -> np.ndarray:
    t_metal = 2.0 * spec.diameter_mm / (spec.material.velocity_m_s * 1e-3)
    start = float(t_if.min()) - _PRE_US
    stop = float(t_if.max()) + t_metal + _POST_US
    n_t = int(np.ceil((stop - start) * spec.probe.sampling_mhz))
    return start + np.arange(n_t) / spec.probe.sampling_mhz


def _wall_echoes(c: _Canvas, spec: BarSpec, f: _Fields) -> None:
    w, m, probe = spec.water, spec.material, spec.probe
    r_wm = reflection_coefficient(w.impedance, m.impedance)
    t_rt = round_trip_transmission(w.impedance, m.impedance)
    d = spec.diameter_mm
    bw = -r_wm * t_rt * attenuation_factor(d, m, probe) * beam_factor(d, probe)
    t_bw = c.t_if + 2.0 * d / (m.velocity_m_s * 1e-3)
    _echo(c, c.t_if, r_wm * f.interface)
    _echo(c, t_bw, bw * f.backwall * f.metal)
    _echo(c, c.t_if - 0.8, f.bubble)


def simulate_bar(spec: BarSpec) -> BarScan:
    """Simulate a full helical scan of one bar. Returns RF A-scans (Synthetic data)."""
    rng = np.random.default_rng(spec.seed)
    probe = spec.probe
    axial = np.arange(0.0, spec.length_mm, spec.axial_step_mm)
    angle = np.arange(spec.n_angle) * 360.0 / spec.n_angle
    f = _artefact_fields(spec, axial, angle)
    t_if = 2.0 * f.water_path_mm / (spec.water.velocity_m_s * 1e-3)
    t_us = _time_axis(spec, t_if)
    cube = grain_noise(t_us, t_if, spec, rng) * f.metal[..., None]
    canvas = _Canvas(cube, t_us, t_if, axial, angle, probe)
    _wall_echoes(canvas, spec, f)
    _reflector_echoes(canvas, spec, f)
    near = (t_us[None, None, :] > t_if[..., None]) & (t_us[None, None, :] < t_if[..., None] + 1.0)
    cube += near * f.surface_noise[..., None] * rng.standard_normal(cube.shape)
    cube *= 10.0 ** (probe.gain_db / 20.0)
    cube += probe.electronic_noise * rng.standard_normal(cube.shape)
    cube[f.missing] = 0.0
    cube = np.clip(cube, -1.0, 1.0).astype(np.float32)
    return BarScan(cube, float(t_us[0]), probe.sampling_mhz, axial, angle, t_if, spec, f.missing)


def cscan_peak(scan: BarScan, gate_start_mm: float, gate_end_mm: float) -> np.ndarray:
    """Peak envelope inside a depth gate [start, end] mm (relative to the interface echo)."""
    depth = depth_of_time(scan.time_us[None, None, :], scan.t_if_us[..., None], scan.spec.material)
    gate = (depth >= gate_start_mm) & (depth <= gate_end_mm)
    return np.max(np.where(gate, envelope(scan.ascans), 0.0), axis=-1)
