"""Titanium zone calibration on a simulated reference block. Synthetic data."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from utriage.physics.simulator import (
    BarScan,
    BarSpec,
    Reflector,
    _load_yaml,
    cscan_peak,
)
from utriage.physics.ti_noise import (
    TI_MODE_PATH,
    TiNoiseConfig,
    simulate_ti_bar,
)


@dataclass(frozen=True)
class TiCalConfig:
    """Settings for reference block calibration. Synthetic data."""

    ref_fbh_mm: float
    signal_window_px: int
    noise_exclude_px: int
    noise_margin: float
    cap_fraction: float
    min_snr: float
    block_seed: int


def load_ti_calibration(path: Path | None = None) -> TiCalConfig:
    """Load calibration config. Values are placeholder, UNVERIFIED - Ruri to check."""
    data = _load_yaml(path or TI_MODE_PATH)["calibration"]
    return TiCalConfig(
        ref_fbh_mm=float(data["ref_fbh_mm"]),
        signal_window_px=int(data["signal_window_px"]),
        noise_exclude_px=int(data["noise_exclude_px"]),
        noise_margin=float(data["noise_margin"]),
        cap_fraction=float(data["cap_fraction"]),
        min_snr=float(data["min_snr"]),
        block_seed=int(data["block_seed"]),
    )


@dataclass(frozen=True)
class ZoneCalibration:
    """Calibration results for a single depth zone. Synthetic data."""

    zone: int
    depth_mm: tuple[float, float]
    ref_depth_mm: float
    signal_a: float
    noise_mean_b: float
    noise_peak_c: float
    snr: float
    alarm_level: float
    cap_bound: bool
    inspectable: bool
    reason: str


@dataclass(frozen=True)
class CalibrationRecord:
    """Full calibration record for all depth zones. Synthetic data."""

    calib_id: str
    zones: tuple[ZoneCalibration, ...]
    ref_fbh_mm: float
    block_seed: int
    config_sha256: str
    label: str = "Synthetic data: simulated reference block"


def reference_block_spec(
    part: BarSpec,
    noise: TiNoiseConfig,
    cal: TiCalConfig,
) -> BarSpec:
    """Simulated reference block with one FBH per depth zone. Synthetic data."""
    edges = noise.zone_edges_mm
    nz = len(edges) - 1
    refls = []
    for k in range(nz):
        mid = 0.5 * (edges[k] + edges[k + 1])
        ax = part.length_mm * (k + 1) / (nz + 1)
        ang = 360.0 * (k + 0.5) / nz
        refls.append(
            Reflector(
                axial_mm=ax,
                angle_deg=ang,
                depth_mm=mid,
                diameter_mm=cal.ref_fbh_mm,
                kind="fbh",
            )
        )
    return dataclasses.replace(
        part,
        reflectors=tuple(refls),
        artefacts=(),
        seed=cal.block_seed,
    )


def hole_index(spec: BarSpec, refl: Reflector) -> tuple[int, int]:
    """Map reflector coordinates to discrete C-scan grid indices (axial, angle)."""
    ax_idx = int(round(refl.axial_mm / spec.axial_step_mm))
    an_step = 360.0 / spec.n_angle
    an_idx = int(round(refl.angle_deg / an_step)) % spec.n_angle
    return (ax_idx, an_idx)


def signal_a(
    cscan: np.ndarray,
    idx: tuple[int, int],
    cal: TiCalConfig,
) -> float:
    """Mean of signal_window_px around peak near idx.

    Averaging form as in US 6,401,537. Synthetic data.
    """
    n_ax, n_an = cscan.shape
    rad = cal.noise_exclude_px
    r_idx = np.arange(max(0, idx[0] - rad), min(n_ax, idx[0] + rad + 1))
    c_idx = (idx[1] + np.arange(-rad, rad + 1)) % n_an
    sub = cscan[np.ix_(r_idx, c_idx)]
    lr, lc = np.unravel_index(np.argmax(sub), sub.shape)
    best_r, best_c = int(r_idx[lr]), int(c_idx[lc])

    half = cal.signal_window_px // 2
    win_r = np.clip(best_r + np.arange(-half, half + 1), 0, n_ax - 1)
    win_c = (best_c + np.arange(-half, half + 1)) % n_an
    return float(np.mean(cscan[np.ix_(win_r, win_c)]))


def noise_stats(
    cscan: np.ndarray,
    holes: list[tuple[int, int]],
    cal: TiCalConfig,
) -> tuple[float, float]:
    """Mean and peak noise envelope excluding reference hole regions. Synthetic data."""
    n_ax, n_an = cscan.shape
    mask = np.ones((n_ax, n_an), dtype=bool)
    rad = cal.noise_exclude_px
    for h_ax, h_an in holes:
        r_idx = np.arange(max(0, h_ax - rad), min(n_ax, h_ax + rad + 1))
        c_idx = (h_an + np.arange(-rad, rad + 1)) % n_an
        mask[np.ix_(r_idx, c_idx)] = False

    noise_pixels = cscan[mask]
    if noise_pixels.size == 0:
        raise ValueError("No noise pixels remain after hole exclusion")
    return float(np.mean(noise_pixels)), float(np.max(noise_pixels))


def zone_alarm(
    a: float,
    b: float,
    c: float,
    cal: TiCalConfig,
) -> tuple[float, float, bool, bool, str]:
    """Calculate zone SNR and alarm threshold.

    SNR form (A-B)/(C-B) as published in US 6,370,956;
    cap_fraction and min_snr are placeholders, UNVERIFIED - Ruri to check.
    """
    candidate = c * (1.0 + cal.noise_margin)
    cap = cal.cap_fraction * a
    alarm = min(candidate, cap)
    cap_bound = candidate > cap

    if c - b <= 0.0:
        return float("nan"), alarm, cap_bound, False, "degenerate noise region"

    snr = (a - b) / (c - b)
    if alarm <= c:
        return snr, alarm, cap_bound, False, "alarm cap is not above peak noise"
    if snr < cal.min_snr:
        return snr, alarm, cap_bound, False, "SNR below placeholder minimum"

    return snr, alarm, cap_bound, True, ""


def _zone_cals(
    scan: BarScan,
    noise: TiNoiseConfig,
    holes: list[tuple[int, int]],
    cal: TiCalConfig,
) -> tuple[ZoneCalibration, ...]:
    edges = noise.zone_edges_mm
    zones = []
    for k in range(len(noise.zone_levels)):
        cscan = cscan_peak(scan, edges[k], edges[k + 1])
        a = signal_a(cscan, holes[k], cal)
        b, c = noise_stats(cscan, holes, cal)
        snr, alarm, cap_bound, inspectable, reason = zone_alarm(a, b, c, cal)
        zones.append(
            ZoneCalibration(
                zone=k,
                depth_mm=(edges[k], edges[k + 1]),
                ref_depth_mm=scan.spec.reflectors[k].depth_mm,
                signal_a=a,
                noise_mean_b=b,
                noise_peak_c=c,
                snr=snr,
                alarm_level=alarm,
                cap_bound=cap_bound,
                inspectable=inspectable,
                reason=reason,
            )
        )
    return tuple(zones)


def _zone_to_dict(z: ZoneCalibration) -> dict[str, Any]:
    return {
        "zone": int(z.zone),
        "depth_mm": [float(z.depth_mm[0]), float(z.depth_mm[1])],
        "ref_depth_mm": float(z.ref_depth_mm),
        "signal_a": float(z.signal_a),
        "noise_mean_b": float(z.noise_mean_b),
        "noise_peak_c": float(z.noise_peak_c),
        "snr": None if math.isnan(z.snr) else float(z.snr),
        "alarm_level": float(z.alarm_level),
        "cap_bound": bool(z.cap_bound),
        "inspectable": bool(z.inspectable),
        "reason": z.reason,
    }


def record_to_dict(rec: CalibrationRecord) -> dict[str, Any]:
    """Convert calibration record to JSON-safe dictionary. Synthetic data."""
    return {
        "calib_id": rec.calib_id,
        "zones": [_zone_to_dict(z) for z in rec.zones],
        "ref_fbh_mm": float(rec.ref_fbh_mm),
        "block_seed": int(rec.block_seed),
        "config_sha256": rec.config_sha256,
        "label": rec.label,
    }


def calibrate_zones(
    part: BarSpec,
    noise: TiNoiseConfig,
    cal: TiCalConfig,
) -> CalibrationRecord:
    """Zone-wise reference calibration on simulated reference block. Synthetic data."""
    ref_block = reference_block_spec(part, noise, cal)
    scan = simulate_ti_bar(ref_block, noise)
    holes = [hole_index(ref_block, r) for r in ref_block.reflectors]
    zones = _zone_cals(scan, noise, holes, cal)
    cfg_raw = json.dumps({"noise": asdict(noise), "cal": asdict(cal)}, sort_keys=True)
    cfg_sha = hashlib.sha256(cfg_raw.encode("utf-8")).hexdigest()
    rec0 = CalibrationRecord(
        calib_id="",
        zones=zones,
        ref_fbh_mm=cal.ref_fbh_mm,
        block_seed=cal.block_seed,
        config_sha256=cfg_sha,
    )
    body = {k: v for k, v in record_to_dict(rec0).items() if k != "calib_id"}
    cid = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return dataclasses.replace(rec0, calib_id=cid)
