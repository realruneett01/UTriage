"""Shared result types of the rules engine and the data-quality watchdog."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CATEGORIES = ("rejectable", "reportable", "insignificant", "data_quality_fault")
FAULT_KINDS = (
    "interface_missing",
    "interface_drop",
    "waterpath_drift",
    "backwall_loss",
    "noise_above_limit",
    "coverage_gap",
    "saturation",
    "early_echo",
)


@dataclass(frozen=True)
class Indication:
    """One clustered indication found by the rules engine.

    ``amplitude_pct`` is the peak amplitude in % of the reference (DAC) at the same depth.
    ``category`` is empty until :func:`utriage.rules.acceptance.categorise` sets it.
    """

    id: int
    axial_mm: float
    angle_deg: float
    depth_mm: float
    amplitude_pct: float
    snr_db: float
    axial_length_mm: float
    circ_length_mm: float
    n_ascans: int
    axial_idx: int
    angle_idx: int
    category: str = ""
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class QualityFault:
    """One data-quality fault region (a connected group of affected A-scans)."""

    kind: str
    axial_mm: tuple[float, float]
    angle_deg: tuple[float, float]
    n_ascans: int


@dataclass
class QualityReport:
    """Per-bar watchdog output. ``fault_mask`` is True where any fault affects the A-scan."""

    faults: list[QualityFault]
    fault_mask: np.ndarray
    masks: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def rescan_recommended(self) -> bool:
        """True if any fault was found."""
        return bool(self.faults)
