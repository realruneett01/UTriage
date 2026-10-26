"""Scoring and evaluation metrics for rules engine detections and watchdog faults.

All outputs are Synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utriage.physics.simulator import Artefact, BarSpec, Reflector
from utriage.rules.types import Indication, QualityFault


@dataclass(frozen=True)
class MatchTol:
    """Tolerances for matching an indication to a known reflector."""

    axial_mm: float = 2.0
    arc_mm: float = 2.5
    depth_mm: float = 2.0


DEFAULT_TOL = MatchTol()


def matches(
    ind: Indication,
    ref: Reflector,
    spec: BarSpec,
    tol: MatchTol = DEFAULT_TOL,
) -> bool:
    """True if indication matches reflector within tolerances."""
    axial_diff = abs(ind.axial_mm - ref.axial_mm)
    depth_diff = abs(ind.depth_mm - ref.depth_mm)
    radius = spec.diameter_mm / 2.0 - ref.depth_mm
    d_ang = (ind.angle_deg - ref.angle_deg + 180.0) % 360.0 - 180.0
    arc_diff = abs(radius) * float(np.deg2rad(abs(d_ang)))
    checks = (
        axial_diff <= tol.axial_mm,
        arc_diff <= tol.arc_mm,
        depth_diff <= tol.depth_mm,
    )
    return all(checks)


@dataclass
class BarScore:
    """Scoring results for a single bar scan."""

    hits: list[bool]
    false_calls: int
    n_ascans: int


def score_bar(
    indications: list[Indication],
    spec: BarSpec,
    tol: MatchTol = DEFAULT_TOL,
) -> BarScore:
    """Score detections against reflectors in spec."""
    relevant = [ind for ind in indications if ind.category in ("reportable", "rejectable")]
    hits = [any(matches(ind, ref, spec, tol) for ind in relevant) for ref in spec.reflectors]
    false_calls = sum(
        1 for ind in relevant if not any(matches(ind, ref, spec, tol) for ref in spec.reflectors)
    )
    n_axial = len(np.arange(0.0, spec.length_mm, spec.axial_step_mm))
    n_ascans = n_axial * spec.n_angle
    return BarScore(hits=hits, false_calls=false_calls, n_ascans=n_ascans)


def _interval_overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def fault_overlaps(fault: QualityFault, art: Artefact) -> bool:
    """True if fault and artefact axial and angle intervals overlap."""
    return _interval_overlaps(fault.axial_mm, art.axial_mm) and _interval_overlaps(
        fault.angle_deg, art.angle_deg
    )
