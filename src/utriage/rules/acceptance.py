"""Acceptance rules engine: categorise indications and summarise inspection results."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from utriage.rules.config import AcceptanceConfig
from utriage.rules.types import CATEGORIES, Indication

_FAULT_REASON = "data quality fault at this location (re-scan)"


def _is_fault(fault_mask: np.ndarray | None, ind: Indication) -> bool:
    """Return True if the indication falls on a flagged data quality fault."""
    if fault_mask is None:
        return False
    return bool(fault_mask[ind.axial_idx, ind.angle_idx])


def _find_multiple_indices(
    indications: list[Indication],
    cfg: AcceptanceConfig,
) -> set[int]:
    """Find indices of indications triggering the multiple indications rule."""
    recordable = [
        (idx, ind)
        for idx, ind in enumerate(indications)
        if ind.amplitude_pct >= cfg.amplitude.record_pct
    ]
    triggered: set[int] = set()
    for _, ind in recordable:
        in_window = [
            idx
            for idx, other in recordable
            if abs(other.axial_mm - ind.axial_mm) <= cfg.multiple.window_mm
        ]
        if len(in_window) > cfg.multiple.max_count:
            triggered.update(in_window)
    return triggered


def _check_reject(
    ind: Indication,
    cfg: AcceptanceConfig,
    is_multiple: bool,
) -> str | None:
    """Return rejection reason if rejectable under rules 2, 3, or 4."""
    if ind.amplitude_pct >= cfg.amplitude.reject_pct:
        return f"amplitude {ind.amplitude_pct}% >= reject {cfg.amplitude.reject_pct}%"
    if max(ind.axial_length_mm, ind.circ_length_mm) >= cfg.linear.max_length_mm:
        return "linear indication"
    if is_multiple:
        return "multiple indications"
    return None


def _classify_indication(
    ind: Indication,
    cfg: AcceptanceConfig,
    is_fault: bool,
    is_multiple: bool,
) -> tuple[str, str | None]:
    """Determine the primary category and primary reason.

    Safety: a data-quality fault never downgrades a rejectable or reportable indication; it
    only adds a re-scan reason. Only indications below the recording level become faults.
    """
    category, reason = _amplitude_category(ind, cfg, is_multiple)
    if not is_fault:
        return category, reason
    if category == "insignificant":
        return "data_quality_fault", _FAULT_REASON
    return category, f"{reason}; {_FAULT_REASON}" if reason else _FAULT_REASON


def _amplitude_category(
    ind: Indication, cfg: AcceptanceConfig, is_multiple: bool
) -> tuple[str, str | None]:
    reject_reason = _check_reject(ind, cfg, is_multiple)
    if reject_reason is not None:
        return "rejectable", reject_reason
    if ind.amplitude_pct >= cfg.amplitude.record_pct:
        return "reportable", None
    return "insignificant", None


def _build_reasons(
    primary_reason: str | None,
    ind: Indication,
    cfg: AcceptanceConfig,
) -> tuple[str, ...]:
    """Assemble reasons tuple for an indication."""
    reasons: list[str] = []
    if primary_reason is not None:
        reasons.append(primary_reason)
    if ind.snr_db < cfg.snr.min_snr_db:
        reasons.append("low SNR (borderline)")
    if cfg.is_placeholder:
        reasons.append("limits are PLACEHOLDERS")
    return tuple(reasons)


def categorise(
    indications: list[Indication],
    cfg: AcceptanceConfig,
    fault_mask: np.ndarray | None = None,
) -> list[Indication]:
    """Categorise indications and attach reasons according to acceptance rules."""
    multiple_indices = _find_multiple_indices(indications, cfg)
    result: list[Indication] = []
    for idx, ind in enumerate(indications):
        is_fault = _is_fault(fault_mask, ind)
        is_multiple = idx in multiple_indices
        category, primary_reason = _classify_indication(ind, cfg, is_fault, is_multiple)
        reasons = _build_reasons(primary_reason, ind, cfg)
        result.append(replace(ind, category=category, reasons=reasons))
    return result


def summarise(indications: list[Indication]) -> dict[str, int]:
    """Count indications per category for all categories in CATEGORIES."""
    counts = {category: 0 for category in CATEGORIES}
    for ind in indications:
        if ind.category in counts:
            counts[ind.category] += 1
    return counts
