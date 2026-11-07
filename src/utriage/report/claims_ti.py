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
) -> tuple[str, dict[str, float]]:
    """Format single zone calibration and detection numbers."""
    zone_idx = int(z_cal["zone"])
    snr = z_cal["snr"]
    inspectable = z_cal["inspectable"]
    if inspectable is None:
        raise TypeError("inspectable cannot be None")
    zk, zn = int(z_flaw["k"]), int(z_flaw["n"])
    gk, gn = int(g_flaw["k"]), int(g_flaw["n"])
    insp_str = "yes" if inspectable else "no"
    snr_str = f"SNR {float(snr):.2f}" if snr is not None else "SNR n/a"
    text = (
        f"zone {zone_idx}: {snr_str}, inspectable {insp_str}, "
        f"zone arm {zk}/{zn}, global arm {gk}/{gn}"
    )
    nums = {
        f"structured_zone_arm_zone{zone_idx}_k": float(zk),
        f"structured_zone_arm_zone{zone_idx}_n": float(zn),
        f"structured_global_arm_zone{zone_idx}_k": float(gk),
        f"structured_global_arm_zone{zone_idx}_n": float(gn),
    }
    if snr is not None:
        nums[f"structured_zone{zone_idx}_snr"] = float(f"{float(snr):.2f}")
    return text, nums


def _extract_clean_fa(
    cz_clean: dict[str, Any],
    cg_clean: dict[str, Any],
) -> tuple[float, float, int]:
    """Extract clean-bar false alarm rates and clean bar count."""
    cz_fa = float(cz_clean["false_alarms_per_bar"])
    cg_fa = float(cg_clean["false_alarms_per_bar"])
    b_alarm = cz_clean["bars_with_alarm"]
    n_clean = int(b_alarm["n"])
    return cz_fa, cg_fa, n_clean


def _collect_calibration_zones(
    z_cals: list[dict[str, Any]],
    fz_zones: list[dict[str, Any]],
    fg_zones: list[dict[str, Any]],
) -> tuple[list[str], dict[str, float]]:
    """Format all zones and aggregate their numeric values."""
    if len(z_cals) != len(fz_zones) or len(z_cals) != len(fg_zones):
        raise ValueError("mismatched zone counts")
    if not z_cals:
        raise ValueError("empty calibration zones")
    zone_texts: list[str] = []
    nums: dict[str, float] = {}
    for zc, zf, gf in zip(z_cals, fz_zones, fg_zones, strict=True):
        zt, zn = _format_calibration_zone(zc, zf, gf)
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
    try:
        structured = data["noise_models"]["structured"]
        cal = structured["calibration"]
        arms = structured["arms"]
        z_cals = cal["zones"]
        arm_z, arm_g = arms["zone"], arms["global"]
        fa_info = _extract_clean_fa(arm_z["clean"], arm_g["clean"])
        fz_zones = arm_z["flawed"]["per_zone"]
        fg_zones = arm_g["flawed"]["per_zone"]
        if not z_cals or not fz_zones:
            raise ValueError("empty calibration zones or flawed data")
        n_flaw = int(fz_zones[0]["n"])
        zone_res = _collect_calibration_zones(z_cals, fz_zones, fg_zones)
        ref_fbh = float(cal["ref_fbh_mm"])
        return _assemble_calibration_claim(fa_info, n_flaw, ref_fbh, zone_res)
    except (KeyError, TypeError, ValueError) as err:  # narrow reason: missing or malformed fields
        print(f"Skipping ti_zone_calibration: {err}", file=sys.stderr)
        return None


def _format_sweep_model(
    m_sweep: dict[str, Any],
    m_data: dict[str, Any],
) -> tuple[str, dict[str, float], float, int]:
    """Format single noise model sweep entry."""
    name = str(m_sweep["name"])
    f_sigma = float(m_sweep["field_sigma"])
    arm_z = m_data["arms"]["zone"]
    uninspect = int(arm_z["zones_not_inspectable"])
    fz_zones = arm_z["flawed"]["per_zone"]
    if not fz_zones:
        raise ValueError("empty per_zone in sweep model")
    tot_k = sum(int(z["k"]) for z in fz_zones)
    tot_n = sum(int(z["n"]) for z in fz_zones)
    text = (
        f"{name} (field_sigma {f_sigma:.2f}): {uninspect} zones not inspectable, "
        f"zone arm detection {tot_k}/{tot_n}"
    )
    nums = {
        f"{name}_field_sigma": float(f"{f_sigma:.2f}"),
        f"{name}_uninspectable_zones": float(uninspect),
        f"{name}_zone_arm_total_k": float(tot_k),
        f"{name}_zone_arm_total_n": float(tot_n),
    }
    return text, nums, f_sigma, uninspect


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
    try:
        sweep = data["sweep"]
        noise_models = data["noise_models"]
        if not sweep:
            raise ValueError("empty sweep")
        parts: list[str] = []
        numbers: dict[str, float] = {}
        pairs: list[tuple[float, int]] = []
        for item in sweep:
            name = str(item["name"])
            m_data = noise_models[name]
            p_text, p_nums, sigma, uninsp = _format_sweep_model(item, m_data)
            parts.append(p_text)
            numbers.update(p_nums)
            pairs.append((sigma, uninsp))
        return _assemble_sweep_claim(parts, numbers, pairs)
    except (KeyError, TypeError, ValueError) as err:  # narrow reason: missing or malformed fields
        print(f"Skipping ti_noise_sweep: {err}", file=sys.stderr)
        return None


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
