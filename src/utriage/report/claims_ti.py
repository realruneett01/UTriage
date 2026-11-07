"""Quantitative claim generation for synthetic titanium zone study.

Extracts claims from titanium zone calibration metrics and sweep results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from utriage.report.claims import Claim

TI_SCOPE_LIMIT: str = (
    "Synthetic titanium-mode simulator only; placeholder limits; "
    "no real titanium data; no conformance to any specification"
)


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load JSON from path, returning None on failure."""
    if not path.exists():
        print(f"Skipping claims requiring {path.name}: file not found", file=sys.stderr)
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as err:  # narrow reason: unreadable or corrupt JSON
        print(f"Skipping claims requiring {path.name}: {err}", file=sys.stderr)
        return None


def _format_calibration_zone(
    z_cal: dict[str, Any],
    z_flaw: dict[str, Any],
    g_flaw: dict[str, Any],
) -> tuple[str, dict[str, float]] | None:
    """Format single zone calibration and detection numbers."""
    zone_idx = z_cal.get("zone")
    snr = z_cal.get("snr")
    inspectable = z_cal.get("inspectable")
    if any(v is None for v in (zone_idx, snr, inspectable)):
        return None
    zk, zn = z_flaw.get("k"), z_flaw.get("n")
    gk, gn = g_flaw.get("k"), g_flaw.get("n")
    if any(val is None for val in (zk, zn, gk, gn)):
        return None
    insp_str = "yes" if inspectable else "no"
    text = (
        f"zone {zone_idx}: SNR {float(snr):.2f}, inspectable {insp_str}, "
        f"zone arm {int(zk)}/{int(zn)}, global arm {int(gk)}/{int(gn)}"
    )
    nums = {
        f"structured_zone{zone_idx}_snr": float(f"{float(snr):.2f}"),
        f"structured_zone_arm_zone{zone_idx}_k": float(zk),
        f"structured_zone_arm_zone{zone_idx}_n": float(zn),
        f"structured_global_arm_zone{zone_idx}_k": float(gk),
        f"structured_global_arm_zone{zone_idx}_n": float(gn),
    }
    return text, nums


def _extract_clean_fa(
    cz_clean: dict[str, Any],
    cg_clean: dict[str, Any],
) -> tuple[float, float, int] | None:
    """Extract clean-bar false alarm rates and clean bar count."""
    cz_fa = cz_clean.get("false_alarms_per_bar")
    cg_fa = cg_clean.get("false_alarms_per_bar")
    if cz_fa is None or cg_fa is None:
        return None
    b_alarm = cz_clean.get("bars_with_alarm", {})
    n_clean = b_alarm.get("n")
    if n_clean is None:
        return None
    return float(cz_fa), float(cg_fa), int(n_clean)


def _collect_calibration_zones(
    z_cals: list[dict[str, Any]],
    fz_zones: list[dict[str, Any]],
    fg_zones: list[dict[str, Any]],
) -> tuple[list[str], dict[str, float]] | None:
    """Format all zones and aggregate their numeric values."""
    if len(z_cals) != len(fz_zones) or len(z_cals) != len(fg_zones):
        return None
    zone_texts: list[str] = []
    nums: dict[str, float] = {}
    for zc, zf, gf in zip(z_cals, fz_zones, fg_zones, strict=True):
        res = _format_calibration_zone(zc, zf, gf)
        if res is None:
            return None
        zt, zn = res
        zone_texts.append(zt)
        nums.update(zn)
    return zone_texts, nums


def _format_calibration_text(
    fa_info: tuple[float, float, int],
    n_flaw: int,
    ref_fbh: float,
    zone_texts: list[str],
) -> str:
    """Format claim text for titanium calibration."""
    cz_fa, cg_fa, n_clean = fa_info
    return (
        f"Synthetic data (titanium-mode simulator). "
        f"Limits and calibration values are placeholders (UNVERIFIED). "
        f"Evaluated on {n_clean} clean bars and {n_flaw} flawed bars per zone. "
        f"Clean-bar false alarms per bar: zone arm {cz_fa:.2f} vs global arm {cg_fa:.2f}. "
        f"Detection k/n of the {ref_fbh:.1f} mm reference-size hole per zone: "
        f"{'; '.join(zone_texts)}."
    )


def _assemble_calibration_claim(
    fa_info: tuple[float, float, int],
    n_flaw: int,
    ref_fbh: float,
    zone_data: tuple[list[str], dict[str, float]],
) -> Claim:
    """Assemble final calibration Claim instance."""
    from utriage.report.claims import Claim

    cz_fa, cg_fa, n_clean = fa_info
    zone_texts, zone_nums = zone_data
    numbers = {
        "ref_hole_mm": float(f"{ref_fbh:.1f}"),
        "n_clean_bars": float(n_clean),
        "n_flawed_bars": float(n_flaw),
        "structured_zone_arm_false_alarms_per_bar": float(f"{cz_fa:.2f}"),
        "structured_global_arm_false_alarms_per_bar": float(f"{cg_fa:.2f}"),
    }
    numbers.update(zone_nums)
    text = _format_calibration_text(fa_info, n_flaw, ref_fbh, zone_texts)
    return Claim(
        id="ti_zone_calibration",
        text=text,
        evidence_file="results/ti_zone_metrics.json",
        evidence_path="noise_models.structured",
        scope_limit=TI_SCOPE_LIMIT,
        numbers=numbers,
    )


def _build_ti_calibration_claim(data: dict[str, Any]) -> Claim | None:
    """Build claim for structured model titanium zone calibration."""
    structured = data.get("noise_models", {}).get("structured")
    if not isinstance(structured, dict):
        return None
    cal = structured.get("calibration", {})
    arms = structured.get("arms", {})
    z_cals = cal.get("zones", [])
    arm_z, arm_g = arms.get("zone", {}), arms.get("global", {})
    fa_info = _extract_clean_fa(arm_z.get("clean", {}), arm_g.get("clean", {}))
    fz_zones = arm_z.get("flawed", {}).get("per_zone", [])
    fg_zones = arm_g.get("flawed", {}).get("per_zone", [])
    if fa_info is None or not z_cals or not fz_zones:
        return None
    n_flaw = fz_zones[0].get("n")
    if n_flaw is None:
        return None
    zone_res = _collect_calibration_zones(z_cals, fz_zones, fg_zones)
    if zone_res is None:
        return None
    ref_fbh = float(cal.get("ref_fbh_mm", 0.8))
    return _assemble_calibration_claim(fa_info, int(n_flaw), ref_fbh, zone_res)


def _format_sweep_model(
    m_sweep: dict[str, Any],
    m_data: dict[str, Any],
) -> tuple[str, dict[str, float], float, int] | None:
    """Format single noise model sweep entry."""
    name = m_sweep.get("name")
    sigma = m_sweep.get("field_sigma")
    if name is None or sigma is None:
        return None
    arm_z = m_data.get("arms", {}).get("zone", {})
    uninspect = arm_z.get("zones_not_inspectable")
    fz_zones = arm_z.get("flawed", {}).get("per_zone", [])
    if uninspect is None or not fz_zones:
        return None
    tot_k = sum(z.get("k", 0) for z in fz_zones)
    tot_n = sum(z.get("n", 0) for z in fz_zones)
    f_sigma = float(sigma)
    text = (
        f"{name} (field_sigma {f_sigma:.2f}): {int(uninspect)} zones not inspectable, "
        f"zone arm detection {tot_k}/{tot_n}"
    )
    nums = {
        f"{name}_field_sigma": float(f"{f_sigma:.2f}"),
        f"{name}_uninspectable_zones": float(uninspect),
        f"{name}_zone_arm_total_k": float(tot_k),
        f"{name}_zone_arm_total_n": float(tot_n),
    }
    return text, nums, f_sigma, int(uninspect)


def _is_monotone_uninspectable(pairs: list[tuple[float, int]]) -> bool:
    """Check if uninspectable count is non-decreasing as field_sigma increases."""
    if len(pairs) < 2:
        return False
    sorted_pairs = sorted(pairs, key=lambda p: p[0])
    return all(sorted_pairs[i][1] <= sorted_pairs[i + 1][1] for i in range(len(sorted_pairs) - 1))


def _assemble_sweep_claim(
    parts: list[str],
    numbers: dict[str, float],
    pairs: list[tuple[float, int]],
) -> Claim:
    """Assemble final sweep Claim instance."""
    from utriage.report.claims import Claim

    monotone_clause = ""
    if _is_monotone_uninspectable(pairs):
        monotone_clause = (
            "; as the synthetic noise-strength variation grows, zones lose inspectability"
        )
    note_msg = (
        "Default field_sigma was lowered after the first calibration run (from payload notes)."
    )
    text = f"Noise sweep across models: {'; '.join(parts)}{monotone_clause}. {note_msg}"
    return Claim(
        id="ti_noise_sweep",
        text=text,
        evidence_file="results/ti_zone_metrics.json",
        evidence_path="sweep",
        scope_limit=TI_SCOPE_LIMIT,
        numbers=numbers,
    )


def _build_ti_sweep_claim(data: dict[str, Any]) -> Claim | None:
    """Build claim for titanium noise sweep across varying field_sigma."""
    sweep = data.get("sweep")
    noise_models = data.get("noise_models")
    if not isinstance(sweep, list) or not isinstance(noise_models, dict) or not sweep:
        return None
    parts: list[str] = []
    numbers: dict[str, float] = {}
    pairs: list[tuple[float, int]] = []
    for item in sweep:
        if not isinstance(item, dict):
            return None
        name = item.get("name")
        m_data = noise_models.get(str(name))
        if not isinstance(m_data, dict):
            return None
        res = _format_sweep_model(item, m_data)
        if res is None:
            return None
        p_text, p_nums, sigma, uninsp = res
        parts.append(p_text)
        numbers.update(p_nums)
        pairs.append((sigma, uninsp))
    return _assemble_sweep_claim(parts, numbers, pairs)


def collect_ti_claims(results_dir: Path | str) -> list[Claim]:
    """Collect quantitative claims from titanium zone evaluation results."""
    root = Path(results_dir)
    data = _load_json(root / "ti_zone_metrics.json")
    if data is None:
        return []
    claims: list[Claim] = []
    c_cal = _build_ti_calibration_claim(data)
    if c_cal is not None:
        claims.append(c_cal)
    c_sweep = _build_ti_sweep_claim(data)
    if c_sweep is not None:
        claims.append(c_sweep)
    return claims


build_ti_calibration_claim = _build_ti_calibration_claim
build_ti_sweep_claim = _build_ti_sweep_claim
