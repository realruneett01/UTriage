"""Data-quality watchdog for immersion ultrasonic scans (project plan Task 2.2).

Detects unphysical, corrupted, or noisy A-scans and maps them into connected fault
regions so re-scans can target the affected areas.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label

from utriage.physics.simulator import BarScan, depth_of_time, envelope
from utriage.rules.config import AcceptanceConfig
from utriage.rules.types import FAULT_KINDS, QualityFault, QualityReport


@dataclass(frozen=True)
class _QualityContext:
    scan: BarScan
    cfg: AcceptanceConfig
    env: np.ndarray
    t_if: np.ndarray
    valid: np.ndarray
    cov: np.ndarray
    if_miss: np.ndarray


def _detect_first_crossing(env: np.ndarray, time_us: np.ndarray, threshold: float) -> np.ndarray:
    cross = env >= threshold
    idx = np.argmax(cross, axis=-1)
    return time_us[idx]


def _gate_rms(env: np.ndarray, mask: np.ndarray) -> np.ndarray:
    counts = np.sum(mask, axis=-1)
    sq_sum = np.sum(np.where(mask, env**2, 0.0), axis=-1)
    mean_sq = np.divide(sq_sum, counts, out=np.zeros_like(sq_sum), where=counts > 0)
    return np.sqrt(mean_sq)


def _mask_to_faults(kind: str, mask: np.ndarray, scan: BarScan) -> list[QualityFault]:
    if not np.any(mask):
        return []
    labeled, num = label(mask, structure=np.ones((3, 3), dtype=int))
    faults: list[QualityFault] = []
    for idx in range(1, num + 1):
        coords = np.argwhere(labeled == idx)
        ax = scan.axial_mm[coords[:, 0]]
        ang = scan.angle_deg[coords[:, 1]]
        faults.append(
            QualityFault(
                kind=kind,
                axial_mm=(float(np.min(ax)), float(np.max(ax))),
                angle_deg=(float(np.min(ang)), float(np.max(ang))),
                n_ascans=int(len(coords)),
            )
        )
    return faults


def _check_coverage_gap(ctx: _QualityContext) -> np.ndarray:
    return ctx.cov


def _check_interface_missing(ctx: _QualityContext) -> np.ndarray:
    return ctx.if_miss


def _check_saturation(ctx: _QualityContext) -> np.ndarray:
    sat = np.any(np.abs(ctx.scan.ascans) >= ctx.cfg.quality.saturation_level, axis=-1)
    return sat & ctx.valid


def _check_interface_drop(ctx: _QualityContext) -> np.ndarray:
    if not np.any(ctx.valid):
        return np.zeros(ctx.valid.shape, dtype=bool)
    time_3d = ctx.scan.time_us[None, None, :]
    in_win = (time_3d >= ctx.t_if[..., None]) & (time_3d <= ctx.t_if[..., None] + 0.3)
    if_peak = np.max(np.where(in_win, ctx.env, 0.0), axis=-1)
    med = float(np.median(if_peak[ctx.valid]))
    return (if_peak < 0.5 * med) & ctx.valid


def _check_waterpath_drift(ctx: _QualityContext) -> np.ndarray:
    if not np.any(ctx.valid):
        return np.zeros(ctx.valid.shape, dtype=bool)
    c_water = ctx.scan.spec.water.velocity_m_s * 1e-3
    med_t_if = float(np.median(ctx.t_if[ctx.valid]))
    drift_mm = np.abs(ctx.t_if - med_t_if) * c_water / 2.0
    return (drift_mm > ctx.cfg.quality.waterpath_tolerance_mm) & ctx.valid


def _check_early_echo(ctx: _QualityContext) -> np.ndarray:
    if not np.any(ctx.valid):
        return np.zeros(ctx.valid.shape, dtype=bool)
    med_t_if = float(np.median(ctx.t_if[ctx.valid]))
    time_3d = ctx.scan.time_us[None, None, :]
    in_win = (time_3d >= med_t_if - 2.0) & (time_3d <= ctx.t_if[..., None] - 0.4)
    thresh = ctx.cfg.quality.early_echo_threshold
    has_echo = np.any(in_win & (ctx.env >= thresh), axis=-1)
    return has_echo & ctx.valid


def _check_backwall_loss(ctx: _QualityContext) -> np.ndarray:
    if not np.any(ctx.valid):
        return np.zeros(ctx.valid.shape, dtype=bool)
    depth = depth_of_time(
        ctx.scan.time_us[None, None, :], ctx.t_if[..., None], ctx.scan.spec.material
    )
    d, margin = ctx.scan.spec.diameter_mm, ctx.cfg.gates.back_margin_mm
    in_bw = (depth >= d - margin) & (depth <= d + margin)
    bw_peak = np.max(np.where(in_bw, ctx.env, 0.0), axis=-1)
    med = float(np.median(bw_peak[ctx.valid]))
    return (bw_peak < ctx.cfg.quality.backwall_drop_ratio * med) & ctx.valid


def _check_noise_above_limit(ctx: _QualityContext) -> np.ndarray:
    if not np.any(ctx.valid):
        return np.zeros(ctx.valid.shape, dtype=bool)
    depth = depth_of_time(
        ctx.scan.time_us[None, None, :], ctx.t_if[..., None], ctx.scan.spec.material
    )
    z_start = ctx.cfg.gates.front_dead_zone_mm
    z_end = ctx.scan.spec.diameter_mm - ctx.cfg.gates.back_margin_mm
    in_flaw = (depth >= z_start) & (depth <= z_end)
    rms = _gate_rms(ctx.env, in_flaw)
    med = float(np.median(rms[ctx.valid]))
    return (rms > ctx.cfg.quality.noise_limit_ratio * med) & ctx.valid


_CHECKERS: dict[str, Callable[[_QualityContext], np.ndarray]] = {
    "coverage_gap": _check_coverage_gap,
    "interface_missing": _check_interface_missing,
    "saturation": _check_saturation,
    "interface_drop": _check_interface_drop,
    "waterpath_drift": _check_waterpath_drift,
    "early_echo": _check_early_echo,
    "backwall_loss": _check_backwall_loss,
    "noise_above_limit": _check_noise_above_limit,
}


def check_quality(scan: BarScan, cfg: AcceptanceConfig) -> QualityReport:
    """Evaluate ultrasonic scan against data-quality acceptance criteria."""
    env = envelope(scan.ascans)
    cov = np.max(np.abs(scan.ascans), axis=-1) < cfg.quality.coverage_floor
    if_miss = (np.max(env, axis=-1) < cfg.gates.interface_threshold) & (~cov)
    valid = ~(cov | if_miss)
    t_if = _detect_first_crossing(env, scan.time_us, cfg.gates.interface_threshold)

    ctx = _QualityContext(scan, cfg, env, t_if, valid, cov, if_miss)
    masks = {kind: _CHECKERS[kind](ctx) for kind in FAULT_KINDS}
    faults = [f for kind, mask in masks.items() for f in _mask_to_faults(kind, mask, scan)]
    fault_mask = np.bitwise_or.reduce(list(masks.values()))
    return QualityReport(faults=faults, fault_mask=fault_mask, masks=masks)
