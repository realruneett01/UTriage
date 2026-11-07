"""Rules engine depth zone alarms and conversion to indications."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from utriage.physics.simulator import BarScan, cscan_peak, depth_of_time, envelope
from utriage.rules.engine import label_clusters
from utriage.rules.types import Indication


@dataclass(frozen=True)
class ZoneGate:
    """Depth gate definition with alarm threshold and inspection status."""

    zone: int
    depth_mm: tuple[float, float]
    alarm_level: float
    inspectable: bool
    calib_id: str


@dataclass(frozen=True)
class ZoneAlarm:
    """Alarm flag for a cluster exceeding threshold or an uninspectable zone."""

    kind: str
    zone: int
    calib_id: str
    axial_idx: int
    angle_idx: int
    axial_mm: float
    angle_deg: float
    depth_mm: float
    peak: float
    pct_of_alarm: float
    n_ascans: int
    axial_extent_mm: float


def _depth_at_peak(
    scan: BarScan,
    ia: int,
    ja: int,
    depth_range: tuple[float, float],
) -> float:
    """Calculate depth of max envelope sample within gate at (ia, ja)."""
    env_trace = envelope(scan.ascans[ia, ja])
    depth_trace = depth_of_time(scan.time_us, scan.t_if_us[ia, ja], scan.spec.material)
    lo, hi = depth_range
    in_gate = (depth_trace >= lo) & (depth_trace <= hi)
    if not np.any(in_gate):
        return float((lo + hi) / 2.0)
    idx = int(np.argmax(np.where(in_gate, env_trace, -1.0)))
    return float(depth_trace[idx])


def _non_missing_count(scan: BarScan) -> int:
    """Count non-missing A-scans across the scan grid."""
    total = scan.axial_mm.size * scan.angle_deg.size
    is_matching = scan.missing.shape == (scan.axial_mm.size, scan.angle_deg.size)
    if scan.missing.size > 0 and is_matching:
        return int(np.sum(~scan.missing))
    return int(total)


def _build_cluster_alarm(
    scan: BarScan,
    gate: ZoneGate,
    pts: np.ndarray,
    cscan: np.ndarray,
) -> ZoneAlarm:
    """Build one alarm record for a cluster peak."""
    vals = cscan[pts[:, 0], pts[:, 1]]
    top = int(np.argmax(vals))
    ia, ja = int(pts[top, 0]), int(pts[top, 1])
    peak = float(vals[top])
    pct = float(100.0 * peak / gate.alarm_level)
    depth_mm = _depth_at_peak(scan, ia, ja, gate.depth_mm)
    span = int(pts[:, 0].max() - pts[:, 0].min() + 1)
    axial_extent = float(span * scan.spec.axial_step_mm)
    return ZoneAlarm(
        kind="alarm",
        zone=gate.zone,
        calib_id=gate.calib_id,
        axial_idx=ia,
        angle_idx=ja,
        axial_mm=float(scan.axial_mm[ia]),
        angle_deg=float(scan.angle_deg[ja]),
        depth_mm=depth_mm,
        peak=peak,
        pct_of_alarm=pct,
        n_ascans=int(pts.shape[0]),
        axial_extent_mm=axial_extent,
    )


def _not_inspectable_alarm(scan: BarScan, gate: ZoneGate) -> ZoneAlarm:
    """Build a zone_not_inspectable alarm covering the zone."""
    mid_depth = float((gate.depth_mm[0] + gate.depth_mm[1]) / 2.0)
    return ZoneAlarm(
        kind="zone_not_inspectable",
        zone=gate.zone,
        calib_id=gate.calib_id,
        axial_idx=-1,
        angle_idx=-1,
        axial_mm=float("nan"),
        angle_deg=float("nan"),
        depth_mm=mid_depth,
        peak=float("nan"),
        pct_of_alarm=float("nan"),
        n_ascans=_non_missing_count(scan),
        axial_extent_mm=float("nan"),
    )


def _process_inspectable_gate(scan: BarScan, gate: ZoneGate) -> list[ZoneAlarm]:
    """Find cluster alarms in an inspectable zone gate."""
    lo, hi = gate.depth_mm
    cscan = np.array(cscan_peak(scan, lo, hi), copy=True)
    has_missing = scan.missing.size > 0 and scan.missing.shape == cscan.shape
    if has_missing:
        cscan[scan.missing] = 0.0
    mask = cscan >= gate.alarm_level
    if has_missing:
        mask[scan.missing] = False
    labels, n_clusters = label_clusters(mask)
    alarms: list[ZoneAlarm] = []
    for k in range(1, n_clusters + 1):
        pts = np.argwhere(labels == k)
        alarms.append(_build_cluster_alarm(scan, gate, pts, cscan))
    return alarms


def zone_alarms(scan: BarScan, gates: Sequence[ZoneGate]) -> list[ZoneAlarm]:
    """Calculate zone alarms for each gate and sort by (zone, -peak)."""
    alarms: list[ZoneAlarm] = []
    for gate in gates:
        if not gate.inspectable:
            alarms.append(_not_inspectable_alarm(scan, gate))
        else:
            alarms.extend(_process_inspectable_gate(scan, gate))
    alarms.sort(key=lambda a: (a.zone, float("inf") if np.isnan(a.peak) else -a.peak))
    return alarms


def to_indications(alarms: Sequence[ZoneAlarm], start_id: int = 0) -> list[Indication]:
    """Transform zone alarms of kind 'alarm' into indications."""
    indications: list[Indication] = []
    for i, alarm in enumerate(a for a in alarms if a.kind == "alarm"):
        reasons = (f"zone {alarm.zone} alarm", f"calibration {alarm.calib_id}")
        ind = Indication(
            id=start_id + i,
            axial_mm=alarm.axial_mm,
            angle_deg=alarm.angle_deg,
            depth_mm=alarm.depth_mm,
            amplitude_pct=alarm.pct_of_alarm,
            snr_db=float("nan"),
            axial_length_mm=alarm.axial_extent_mm,
            circ_length_mm=float("nan"),
            n_ascans=alarm.n_ascans,
            axial_idx=alarm.axial_idx,
            angle_idx=alarm.angle_idx,
            category="reportable",
            reasons=reasons,
        )
        indications.append(ind)
    return indications
