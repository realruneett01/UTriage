"""Campaign evaluation metrics for triage ranking and review effort.

All outputs labelled Synthetic data.
"""

from __future__ import annotations

import numpy as np

from utriage.physics.simulator import BarScan, BarSpec, Reflector
from utriage.rules.scoring import MatchTol, matches
from utriage.triage.fusion import TriageItem

_DEFAULT_TOL = MatchTol()


def _coords_match(
    coords: tuple[float, float, float],
    ref: Reflector,
    spec: BarSpec,
    tol: MatchTol,
) -> bool:
    """True if coordinates (axial_mm, angle_deg, depth_mm) match reflector within tol."""
    ax_mm, ang_deg, depth_mm = coords
    axial_diff = abs(ax_mm - ref.axial_mm)
    depth_diff = abs(depth_mm - ref.depth_mm)
    radius = spec.diameter_mm / 2.0 - ref.depth_mm
    d_ang = (ang_deg - ref.angle_deg + 180.0) % 360.0 - 180.0
    arc_diff = abs(radius) * float(np.deg2rad(abs(d_ang)))
    return all(
        (
            axial_diff <= tol.axial_mm,
            arc_diff <= tol.arc_mm,
            depth_diff <= tol.depth_mm,
        )
    )


def _extract_coords(item: TriageItem, scan: BarScan) -> tuple[float, float, float] | None:
    """Extract spatial coordinates (axial_mm, angle_deg, depth_mm) from item."""
    if item.indication is not None:
        return (item.indication.axial_mm, item.indication.angle_deg, item.indication.depth_mm)
    if item.ml is not None:
        return (
            float(scan.axial_mm[item.ml.axial_idx]),
            float(scan.angle_deg[item.ml.angle_idx]),
            float(item.ml.depth_mm),
        )
    return None


def item_is_real(
    item: TriageItem,
    spec: BarSpec,
    scan: BarScan,
    tol: MatchTol = _DEFAULT_TOL,
) -> bool:
    """True if the item matches any planted reflector."""
    if item.indication is not None:
        return any(matches(item.indication, ref, spec, tol) for ref in spec.reflectors)
    coords = _extract_coords(item, scan)
    if coords is None:
        return False
    return any(_coords_match(coords, ref, spec, tol) for ref in spec.reflectors)


def review_effort(ordered_real: list[bool], n_real_total: int) -> int | None:
    """Number of items to inspect to find every real flaw; None if some missing."""
    if n_real_total == 0:
        return 0
    seen = 0
    for idx, is_real in enumerate(ordered_real, start=1):
        if is_real:
            seen += 1
            if seen == n_real_total:
                return idx
    return None


def random_order_effort(
    real_flags: list[bool],
    n_real_total: int,
    n_perm: int = 200,
    seed: int = 0,
) -> float:
    """Mean review effort over random permutations."""
    if review_effort(real_flags, n_real_total) is None:
        return float("nan")
    rng = np.random.default_rng(seed)
    flags = np.asarray(real_flags, dtype=bool)
    efforts: list[int] = []
    for _ in range(n_perm):
        perm = rng.permutation(flags).tolist()
        eff = review_effort(perm, n_real_total)
        if eff is not None:
            efforts.append(eff)
    return float(np.mean(efforts)) if efforts else float("nan")


__all__ = [
    "item_is_real",
    "random_order_effort",
    "review_effort",
]
