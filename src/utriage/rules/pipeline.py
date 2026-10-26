"""Conventional rules engine pipeline (project plan 2.5).

Coordinates data-quality checks, indication detection, acceptance categorisation,
and DAC calibration curves.
"""

from __future__ import annotations

from dataclasses import dataclass

from utriage.physics.calibration import (
    CalibrationBlock,
    dac_curve,
    measure_dac,
    simulate_calibration_block,
)
from utriage.physics.simulator import BarScan
from utriage.rules.acceptance import categorise, summarise
from utriage.rules.config import AcceptanceConfig
from utriage.rules.engine import Dac, detect_indications
from utriage.rules.quality import check_quality
from utriage.rules.types import Indication, QualityFault, QualityReport


def _metal_paths_for(diameter_mm: float) -> tuple[float, ...]:
    """Generate metal paths from 5 mm to (diameter - 3) mm in steps of 5 mm."""
    max_path = diameter_mm - 3.0
    paths = [float(p) for p in range(5, int(max_path) + 1, 5)]
    return tuple(paths) if paths else (5.0,)


def dac_for(scan: BarScan, fbh_diameter_mm: float = 0.8) -> Dac:
    """Build a calibration block, simulate A-scans, and return the DAC curve."""
    paths = _metal_paths_for(scan.spec.diameter_mm)
    block = CalibrationBlock(
        material=scan.spec.material,
        probe=scan.spec.probe,
        water=scan.spec.water,
        fbh_diameter_mm=fbh_diameter_mm,
        metal_paths_mm=paths,
        thickness_mm=scan.spec.diameter_mm,
        seed=0,
    )
    time_us, ascans, t_if_us = simulate_calibration_block(block)
    points = measure_dac(time_us, ascans, t_if_us, block)
    return dac_curve(points)


@dataclass
class Evaluation:
    """Inspection evaluation results for a single bar scan."""

    indications: list[Indication]
    quality: QualityReport
    summary: dict[str, int]
    placeholder_limits: bool
    label: str = ""


def evaluate_scan(
    scan: BarScan,
    cfg: AcceptanceConfig,
    dac: Dac | None = None,
) -> Evaluation:
    """Run data quality watchdog, detect indications, categorise, and summarise."""
    quality = check_quality(scan, cfg)
    active_dac = dac if dac is not None else dac_for(scan)
    raw_indications = detect_indications(scan, active_dac, cfg)
    categorised = categorise(raw_indications, cfg, fault_mask=quality.fault_mask)
    label = getattr(scan, "label", "")
    return Evaluation(
        indications=categorised,
        quality=quality,
        summary=summarise(categorised),
        placeholder_limits=cfg.is_placeholder,
        label=label,
    )


def _indication_dict(ind: Indication) -> dict:
    return {
        "id": int(ind.id),
        "axial_mm": float(ind.axial_mm),
        "angle_deg": float(ind.angle_deg),
        "depth_mm": float(ind.depth_mm),
        "amplitude_pct": float(ind.amplitude_pct),
        "snr_db": float(ind.snr_db),
        "axial_length_mm": float(ind.axial_length_mm),
        "circ_length_mm": float(ind.circ_length_mm),
        "n_ascans": int(ind.n_ascans),
        "axial_idx": int(ind.axial_idx),
        "angle_idx": int(ind.angle_idx),
        "category": str(ind.category),
        "reasons": list(ind.reasons),
    }


def _fault_dict(fault: QualityFault) -> dict:
    return {
        "kind": str(fault.kind),
        "axial_mm": [float(fault.axial_mm[0]), float(fault.axial_mm[1])],
        "angle_deg": [float(fault.angle_deg[0]), float(fault.angle_deg[1])],
        "n_ascans": int(fault.n_ascans),
    }


def evaluation_to_dict(ev: Evaluation) -> dict:
    """Convert an Evaluation to a JSON-safe dictionary without numpy types."""
    fault_dicts = [_fault_dict(f) for f in ev.quality.faults]
    return {
        "label": str(ev.label),
        "placeholder_limits": bool(ev.placeholder_limits),
        "summary": {str(k): int(v) for k, v in ev.summary.items()},
        "indications": [_indication_dict(ind) for ind in ev.indications],
        "quality": {
            "rescan_recommended": bool(ev.quality.rescan_recommended),
            "faults": fault_dicts,
        },
        "faults": fault_dicts,
    }
