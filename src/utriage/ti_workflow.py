"""Titanium mode inspection workflow. Synthetic data."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from utriage.physics.simulator import BarScan
from utriage.physics.ti_calibration import (
    CalibrationRecord,
    TiCalConfig,
    calibrate_zones,
    load_ti_calibration,
    record_to_dict,
)
from utriage.physics.ti_noise import (
    TiNoiseConfig,
    load_ti_noise,
)
from utriage.rules.config import AcceptanceConfig, load_acceptance
from utriage.rules.pipeline import Evaluation, evaluate_scan
from utriage.rules.types import Indication
from utriage.rules.zones import (
    ZoneAlarm,
    ZoneGate,
    to_indications,
    zone_alarms,
)
from utriage.triage.fusion import (
    MlFinding,
    TriageItem,
    fuse,
    rank,
    rules_flags_preserved,
)


@dataclass(frozen=True)
class TiConfigs:
    """Titanium configurations bundle. Synthetic data."""

    noise: TiNoiseConfig
    cal: TiCalConfig
    acceptance: AcceptanceConfig


@dataclass(frozen=True)
class TiInspection:
    """Titanium inspection result. Synthetic data."""

    bar_id: str
    scan: BarScan
    record: CalibrationRecord
    alarms: tuple[ZoneAlarm, ...]
    evaluation: Evaluation
    rule_indications: tuple[Indication, ...]
    items: tuple[TriageItem, ...]
    synthetic: bool


def gates_from_record(rec: CalibrationRecord) -> list[ZoneGate]:
    """Build zone gates from calibration record. Synthetic data."""
    return [
        ZoneGate(
            zone=z.zone,
            depth_mm=z.depth_mm,
            alarm_level=z.alarm_level,
            inspectable=z.inspectable,
            calib_id=rec.calib_id,
        )
        for z in rec.zones
    ]


def load_ti_configs() -> TiConfigs:
    """Load titanium configurations. Values are placeholder, UNVERIFIED - Ruri to check."""
    return TiConfigs(
        noise=load_ti_noise(),
        cal=load_ti_calibration(),
        acceptance=load_acceptance(),
    )


def inspect_ti(
    scan: BarScan,
    bar_id: str,
    cfgs: TiConfigs,
    findings: Sequence[MlFinding] = (),
    *,
    synthetic: bool = True,
) -> TiInspection:
    """Run titanium mode inspection. Synthetic data."""
    rec = calibrate_zones(scan.spec, cfgs.noise, cfgs.cal)
    alarms = zone_alarms(scan, gates_from_record(rec))
    ev = evaluate_scan(scan, cfgs.acceptance)
    next_id = max([i.id for i in ev.indications], default=0) + 1
    zone_inds = to_indications(alarms, start_id=next_id)
    rule_inds = list(ev.indications) + zone_inds
    items = rank(fuse(rule_inds, list(findings)))
    if not rules_flags_preserved(rule_inds, items):
        raise RuntimeError("ML removed or changed a rules flag")
    return TiInspection(
        bar_id=bar_id,
        scan=scan,
        record=rec,
        alarms=tuple(alarms),
        evaluation=ev,
        rule_indications=tuple(rule_inds),
        items=tuple(items),
        synthetic=synthetic,
    )


def _float_or_none(val: float) -> float | None:
    return None if math.isnan(val) else float(val)


def _alarm_to_dict(alarm: ZoneAlarm) -> dict[str, Any]:
    return {
        "kind": str(alarm.kind),
        "zone": int(alarm.zone),
        "calib_id": str(alarm.calib_id),
        "axial_idx": int(alarm.axial_idx),
        "angle_idx": int(alarm.angle_idx),
        "axial_mm": _float_or_none(alarm.axial_mm),
        "angle_deg": _float_or_none(alarm.angle_deg),
        "depth_mm": _float_or_none(alarm.depth_mm),
        "peak": _float_or_none(alarm.peak),
        "pct_of_alarm": _float_or_none(alarm.pct_of_alarm),
        "n_ascans": int(alarm.n_ascans),
        "axial_extent_mm": _float_or_none(alarm.axial_extent_mm),
    }


def _item_reasons(item: TriageItem) -> list[str]:
    if item.indication is not None:
        return list(item.indication.reasons)
    if item.ml is not None:
        return list(item.ml.evidence)
    return []


def _item_to_dict(item: TriageItem) -> dict[str, Any]:
    return {
        "source": str(item.source),
        "category": str(item.category),
        "review": list(item.review),
        "reasons": _item_reasons(item),
    }


def inspection_to_dict(insp: TiInspection) -> dict[str, Any]:
    """Convert inspection result to JSON-safe dictionary. Synthetic data."""
    alarms_json = [_alarm_to_dict(a) for a in insp.alarms]
    n_zone = sum(1 for a in insp.alarms if a.kind == "alarm")
    n_uninsp = sum(1 for a in insp.alarms if a.kind == "zone_not_inspectable")
    out: dict[str, Any] = {
        "bar_id": str(insp.bar_id),
        "synthetic": bool(insp.synthetic),
        "calibration": record_to_dict(insp.record),
        "alarms": alarms_json,
        "n_rules_indications": len(insp.rule_indications),
        "n_zone_alarms": n_zone,
        "n_zones_not_inspectable": n_uninsp,
        "items": [_item_to_dict(it) for it in insp.items],
        "placeholder_limits": bool(insp.evaluation.placeholder_limits),
        "advisory": "UTriage never accepts or passes a part; the certified inspector decides.",
    }
    if insp.synthetic:
        out["label"] = "Synthetic data"
    return out
