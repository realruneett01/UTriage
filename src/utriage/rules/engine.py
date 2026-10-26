"""Conventional rules detection engine (project plan 2.1).

Gate synchronisation, DAC-normalised amplitude maps, cluster labelling with angle
wrapping, local SNR, and indication extraction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.ndimage

from utriage.physics.simulator import BarScan, depth_of_time, envelope
from utriage.rules.config import AcceptanceConfig
from utriage.rules.types import Indication

Dac = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class EnvelopeView:
    """Precomputed envelope and interface-time view of a bar scan."""

    env: np.ndarray
    time_us: np.ndarray
    t_if_us: np.ndarray
    scan: BarScan


@dataclass(frozen=True)
class Peak:
    """Peak indication location and depth."""

    ia: int
    ja: int
    depth_mm: float


def interface_times(env: np.ndarray, time_us: np.ndarray, threshold: float) -> np.ndarray:
    """First time where env >= threshold along the last axis; NaN where never."""
    crossed = env >= threshold
    has_crossing = np.any(crossed, axis=-1)
    idx = np.argmax(crossed, axis=-1)
    if time_us.ndim == 1:
        t = time_us[idx]
    else:
        t = np.take_along_axis(time_us, idx[..., None], axis=-1).squeeze(-1)
    return np.where(has_crossing, t, np.nan)


def amplitude_map(
    view: EnvelopeView,
    dac: Dac,
    cfg: AcceptanceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Peak amplitude % and depth in flaw gate per A-scan."""
    env = view.env
    t_grid = view.time_us[None, None, :] if view.time_us.ndim == 1 else view.time_us
    tif_grid = view.t_if_us[..., None] if view.t_if_us.ndim == 2 else view.t_if_us
    depth = depth_of_time(t_grid, tif_grid, view.scan.spec.material)
    z_min = cfg.gates.front_dead_zone_mm
    z_max = view.scan.spec.diameter_mm - cfg.gates.back_margin_mm
    in_range = (depth >= z_min) & (depth <= z_max)
    in_gate = in_range & np.isfinite(depth)
    pct = np.zeros_like(env)
    if np.any(in_gate):
        ref = np.asarray(dac(depth[in_gate]), dtype=env.dtype)
        pct[in_gate] = 100.0 * env[in_gate] / np.maximum(ref, 1e-12)
    has_gate = np.any(in_gate, axis=-1)
    masked = np.where(in_gate, pct, -1.0)
    peak_idx = np.argmax(masked, axis=-1)
    pct_map = np.where(has_gate, np.max(masked, axis=-1), 0.0)
    d_at_peak = np.take_along_axis(depth, peak_idx[..., None], axis=-1).squeeze(-1)
    depth_map = np.where(has_gate, d_at_peak, np.nan)
    return pct_map, depth_map


def _find(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _merge_roots(parent: list[int], col0: np.ndarray, col_last: np.ndarray) -> None:
    u = np.concatenate([col0, col0[1:], col0[:-1]])
    v = np.concatenate([col_last, col_last[:-1], col_last[1:]])
    valid = (u > 0) & (v > 0)
    for a, b in zip(u[valid], v[valid], strict=False):
        ru, rv = _find(parent, int(a)), _find(parent, int(b))
        if ru != rv:
            parent[ru] = rv


def _remap_labels(labeled: np.ndarray, parent: list[int], n_feat: int) -> tuple[np.ndarray, int]:
    root_to_new: dict[int, int] = {}
    mapping = np.zeros(n_feat + 1, dtype=int)
    for lbl in range(1, n_feat + 1):
        r = _find(parent, lbl)
        if r not in root_to_new:
            root_to_new[r] = len(root_to_new) + 1
        mapping[lbl] = root_to_new[r]
    return mapping[labeled], len(root_to_new)


def label_clusters(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected labels on (axial, angle) grid with angle wrapping."""
    labeled, n_features = scipy.ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n_features == 0:
        return labeled, 0
    parent = list(range(n_features + 1))
    _merge_roots(parent, labeled[:, 0], labeled[:, -1])
    return _remap_labels(labeled, parent, n_features)


def local_snr(view: EnvelopeView, peak: Peak, cfg: AcceptanceConfig) -> float:
    """Local SNR in dB around an indication peak."""
    env = view.env
    z_peak = depth_of_time(view.time_us, view.t_if_us[peak.ia, peak.ja], view.scan.spec.material)
    m_peak = (np.abs(z_peak - peak.depth_mm) <= cfg.snr.exclude_mm) & np.isfinite(z_peak)
    has_peak = np.any(m_peak)
    peak_val = (
        float(np.max(env[peak.ia, peak.ja, m_peak]))
        if has_peak
        else float(np.max(env[peak.ia, peak.ja]))
    )
    ax = np.arange(max(0, peak.ia - 1), min(env.shape[0], peak.ia + 2))
    ang = np.unique([(peak.ja + dj) % env.shape[1] for dj in (-1, 0, 1)])
    sub_env = env[ax[:, None], ang[None, :], :]
    sub_tif = view.t_if_us[ax[:, None], ang[None, :]]
    sub_z = depth_of_time(view.time_us[None, None, :], sub_tif[..., None], view.scan.spec.material)
    dist = np.abs(sub_z - peak.depth_mm)
    in_win = (dist <= cfg.snr.noise_window_mm) & (dist > cfg.snr.exclude_mm)
    noise = sub_env[in_win & np.isfinite(dist)]
    rms = float(np.sqrt(np.mean(noise**2))) if noise.size > 0 else 0.0
    return float(20.0 * np.log10(max(peak_val, 1e-12) / max(rms, 1e-12)))


def _cluster_stats(
    pts: np.ndarray,
    pct_map: np.ndarray,
    depth_map: np.ndarray,
    scan: BarScan,
) -> tuple[int, int, float, float, float, float]:
    pcts = pct_map[pts[:, 0], pts[:, 1]]
    top = int(np.argmax(pcts))
    ia, ja = int(pts[top, 0]), int(pts[top, 1])
    d_mm = float(depth_map[ia, ja])
    ax_len = float((pts[:, 0].max() - pts[:, 0].min() + 1) * scan.spec.axial_step_mm)
    rad = max(0.0, scan.spec.diameter_mm / 2.0 - d_mm)
    d_th = 2.0 * np.pi / scan.spec.n_angle
    circ_len = float(len(np.unique(pts[:, 1])) * rad * d_th)
    return ia, ja, float(pcts[top]), d_mm, ax_len, circ_len


def _to_indication(
    i: int,
    item: tuple[float, int, int, float, float, float, int, float],
    scan: BarScan,
) -> Indication:
    amp, ia, ja, d_mm, ax_len, circ_len, n_asc, snr = item
    return Indication(
        id=i,
        axial_mm=float(scan.axial_mm[ia]),
        angle_deg=float(scan.angle_deg[ja]),
        depth_mm=d_mm,
        amplitude_pct=amp,
        snr_db=snr,
        axial_length_mm=ax_len,
        circ_length_mm=circ_len,
        n_ascans=n_asc,
        axial_idx=ia,
        angle_idx=ja,
    )


def detect_indications(scan: BarScan, dac: Dac, cfg: AcceptanceConfig) -> list[Indication]:
    """Detect and cluster ultrasonic indications above recording threshold."""
    env = envelope(scan.ascans)
    t_if = interface_times(env, scan.time_us, cfg.gates.interface_threshold)
    view = EnvelopeView(env=env, time_us=scan.time_us, t_if_us=t_if, scan=scan)
    pct_map, depth_map = amplitude_map(view, dac, cfg)
    labels, n_clusters = label_clusters(pct_map >= cfg.amplitude.record_pct)
    raw = []
    for k in range(1, n_clusters + 1):
        pts = np.argwhere(labels == k)
        ia, ja, amp, d_mm, ax_len, circ_len = _cluster_stats(pts, pct_map, depth_map, scan)
        snr = local_snr(view, Peak(ia, ja, d_mm), cfg)
        raw.append((amp, ia, ja, d_mm, ax_len, circ_len, len(pts), snr))
    raw.sort(key=lambda item: item[0], reverse=True)
    return [_to_indication(i, item, scan) for i, item in enumerate(raw)]
