"""Synthetic study of zone-wise calibration under titanium grain noise."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Configure local workspace paths
_EXP_DIR = Path(__file__).resolve().parent
_SRC_DIR = _EXP_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from utriage.physics.simulator import (  # noqa: E402
    BarSpec,
    Reflector,
    load_material,
    load_probe,
)
from utriage.physics.ti_calibration import (  # noqa: E402
    CalibrationRecord,
    TiCalConfig,
    ZoneCalibration,
    calibrate_zones,
    hole_index,
    load_ti_calibration,
    record_to_dict,
    zone_alarm,
)
from utriage.physics.ti_noise import (  # noqa: E402
    TI_MODE_PATH,
    TiNoiseConfig,
    load_ti_noise,
    simulate_ti_bar,
)
from utriage.pod.metrics import clopper_pearson  # noqa: E402
from utriage.rules.zones import ZoneAlarm, ZoneGate, zone_alarms  # noqa: E402

LABEL_SYNTHETIC: str = (
    "Synthetic data (titanium-mode simulator; configs/ti_mode.yaml placeholders, UNVERIFIED)"
)

DEFINITIONS: dict[str, str] = {
    "global_arm": (
        "One ZoneGate spanning edges[0]..edges[-1] with alarm from "
        "zone_alarm(A=min zone A, B=mean zone B, C=max zone C, cal)."
    ),
    "zone_arm": (
        "One ZoneGate per depth zone from calibrate_zones(part, noise, cal), "
        "with alarm_level from each ZoneCalibration record."
    ),
    "structured_noise": (
        "Structured grain noise loaded from configs/ti_mode.yaml with "
        "spatially correlated lognormal multiplier and depth-dependent zone levels."
    ),
    "flat_noise": (
        "Same zone edges as structured noise but with field_sigma=0.0 "
        "and all zone_levels set to 1.0."
    ),
    "detection": (
        "Flawed bar is detected if at least one alarm of kind 'alarm' is "
        "within 2 px axial and 2 px angle with circular wrap of the hole pixel, "
        "and for zone arm whose zone contains the hole depth (for global arm any alarm)."
    ),
    "clean_evaluation": (
        "Clean bars have no planted flaws; false alarms per bar is the mean count "
        "of kind 'alarm' clusters, and bars with >=1 alarm reports k/n with "
        "exact two-sided Clopper-Pearson 95% bounds."
    ),
}


@dataclass(frozen=True)
class RunConfig:
    """Execution settings for titanium zone evaluation study."""

    n_clean: int = 20
    n_flaw: int = 20
    seed: int = 0
    out_path: Path = Path("results/ti_zone_metrics.json")
    quick: bool = False


@dataclass(frozen=True)
class FlawTarget:
    """Target reflector coordinates and depth zone."""

    axial_idx: int
    angle_idx: int
    depth_mm: float
    zone: int


@dataclass(frozen=True)
class ArmGates:
    """Container for threshold arm gates."""

    name: str
    gates: tuple[ZoneGate, ...]
    uninspectable_count: int
    global_cal: tuple[float, float, bool, bool, str] | None = None


@dataclass(frozen=True)
class EvalArms:
    """Pair of zone and global arms for evaluation."""

    zone: ArmGates
    global_arm: ArmGates


@dataclass(frozen=True)
class SimContext:
    """Context bundle for titanium simulation study."""

    part: BarSpec
    noise: TiNoiseConfig
    cal: TiCalConfig
    arms: EvalArms


def default_part_spec() -> BarSpec:
    """Default titanium bar specification with Ti-6Al-4V, p5 probe, and water."""
    ti64 = load_material("ti64")
    water = load_material("water")
    probe = load_probe("p5")
    return BarSpec(material=ti64, probe=probe, water=water)


def compute_config_sha256(path: Path = TI_MODE_PATH) -> str:
    """Compute git-free SHA256 of titanium mode config file."""
    with path.open("rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def make_noise_models(cfg_noise: TiNoiseConfig) -> dict[str, TiNoiseConfig]:
    """Build structured and flat noise configurations."""
    flat = dataclasses.replace(
        cfg_noise,
        field_sigma=0.0,
        zone_levels=tuple(1.0 for _ in cfg_noise.zone_levels),
    )
    return {"structured": cfg_noise, "flat": flat}


def build_zone_arm(rec: CalibrationRecord) -> ArmGates:
    """Build threshold gates from zone-wise calibration record."""
    gates = tuple(
        ZoneGate(
            zone=z.zone,
            depth_mm=z.depth_mm,
            alarm_level=z.alarm_level,
            inspectable=z.inspectable,
            calib_id=rec.calib_id,
        )
        for z in rec.zones
    )
    uninspect = sum(1 for z in rec.zones if not z.inspectable)
    return ArmGates(name="zone", gates=gates, uninspectable_count=uninspect)


def build_global_arm(
    rec: CalibrationRecord,
    noise: TiNoiseConfig,
    cal: TiCalConfig,
) -> ArmGates:
    """Build single global threshold gate spanning all depth zones."""
    a_val = min(z.signal_a for z in rec.zones)
    b_val = float(np.mean([z.noise_mean_b for z in rec.zones]))
    c_val = max(z.noise_peak_c for z in rec.zones)
    cal_res = zone_alarm(a_val, b_val, c_val, cal)
    snr, alarm, cap_bound, inspectable, reason = cal_res
    gate = ZoneGate(
        zone=0,
        depth_mm=(noise.zone_edges_mm[0], noise.zone_edges_mm[-1]),
        alarm_level=alarm,
        inspectable=inspectable,
        calib_id=rec.calib_id,
    )
    uninspect = 0 if inspectable else 1
    return ArmGates(
        name="global",
        gates=(gate,),
        uninspectable_count=uninspect,
        global_cal=cal_res,
    )


def alarm_matches_hole(
    alarm: ZoneAlarm,
    hole_ax: int,
    hole_an: int,
    n_angle: int,
) -> bool:
    """Check if alarm is within 2 px axial and 2 px angle with circular wrap."""
    if alarm.kind != "alarm":
        return False
    d_ax = abs(alarm.axial_idx - hole_ax)
    if d_ax > 2:
        return False
    d_an = abs(alarm.angle_idx - hole_an)
    d_wrap = min(d_an, n_angle - d_an)
    return d_wrap <= 2


def is_alarm_hit(
    alarm: ZoneAlarm,
    target: FlawTarget,
    n_angle: int,
    arm_name: str,
) -> bool:
    """Return whether alarm hits the planted flaw target."""
    if arm_name == "zone" and alarm.zone != target.zone:
        return False
    return alarm_matches_hole(alarm, target.axial_idx, target.angle_idx, n_angle)


def check_flaw_detected(
    alarms: Sequence[ZoneAlarm],
    target: FlawTarget,
    n_angle: int,
    arm_name: str,
) -> bool:
    """Return True if any alarm matches the planted flaw target."""
    return any(is_alarm_hit(a, target, n_angle, arm_name) for a in alarms)


def score_clean_bar(
    part: BarSpec,
    noise: TiNoiseConfig,
    seed: int,
    arms: EvalArms,
) -> tuple[int, int]:
    """Simulate one clean bar and return alarm counts for (zone, global) arms."""
    clean_spec = dataclasses.replace(part, reflectors=(), artefacts=(), seed=seed)
    scan = simulate_ti_bar(clean_spec, noise)
    cnt_zone = sum(1 for a in zone_alarms(scan, arms.zone.gates) if a.kind == "alarm")
    cnt_global = sum(1 for a in zone_alarms(scan, arms.global_arm.gates) if a.kind == "alarm")
    del scan
    return cnt_zone, cnt_global


def _summarize_clean(counts: Sequence[int]) -> dict[str, Any]:
    """Compute false alarm rate and Clopper-Pearson bounds."""
    n = len(counts)
    mean_alarms = float(np.mean(counts)) if n > 0 else 0.0
    k = sum(1 for c in counts if c >= 1)
    rate = float(k / n) if n > 0 else 0.0
    low, high = clopper_pearson(k, n)
    return {
        "false_alarms_per_bar": mean_alarms,
        "false_alarms_per_bar_mean": mean_alarms,
        "bars_with_alarm": {
            "k": k,
            "n": n,
            "rate": rate,
            "ci_low": low,
            "ci_high": high,
        },
        "bars_with_ge_1_alarm": {
            "k": k,
            "n": n,
            "rate": rate,
            "ci_low": low,
            "ci_high": high,
        },
    }


def evaluate_clean_bars(
    part: BarSpec,
    noise: TiNoiseConfig,
    seeds: Sequence[int],
    arms: EvalArms,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate clean bars streaming one bar at a time."""
    counts_zone: list[int] = []
    counts_global: list[int] = []
    for s in seeds:
        cz, cg = score_clean_bar(part, noise, s, arms)
        counts_zone.append(cz)
        counts_global.append(cg)
    return _summarize_clean(counts_zone), _summarize_clean(counts_global)


def make_flaw_reflector(
    part: BarSpec,
    bounds: tuple[float, float],
    cal: TiCalConfig,
    seed: int,
) -> Reflector:
    """Generate one planted FBH reflector within zone depth bounds."""
    rng = np.random.default_rng(seed)
    depth = float(rng.uniform(bounds[0] + 1.0, bounds[1] - 1.0))
    ax_max = max(4.0, part.length_mm - 4.0)
    axial = float(rng.uniform(4.0, ax_max))
    angle = float(rng.uniform(0.0, 360.0))
    return Reflector(
        axial_mm=axial,
        angle_deg=angle,
        depth_mm=depth,
        diameter_mm=cal.ref_fbh_mm,
        kind="fbh",
    )


def score_flawed_bar(
    spec: BarSpec,
    target: FlawTarget,
    noise: TiNoiseConfig,
    arms: EvalArms,
) -> tuple[int, int]:
    """Simulate flawed bar and return detection hits (zone_hit, global_hit)."""
    scan = simulate_ti_bar(spec, noise)
    az = zone_alarms(scan, arms.zone.gates)
    hit_z = 1 if check_flaw_detected(az, target, spec.n_angle, "zone") else 0
    ag = zone_alarms(scan, arms.global_arm.gates)
    hit_g = 1 if check_flaw_detected(ag, target, spec.n_angle, "global") else 0
    del scan
    return hit_z, hit_g


def evaluate_flawed_zone(
    ctx: SimContext,
    z_idx: int,
    seeds: Sequence[int],
) -> tuple[int, int]:
    """Evaluate flawed bars for one depth zone, returning hit sums."""
    lo = ctx.noise.zone_edges_mm[z_idx]
    hi = ctx.noise.zone_edges_mm[z_idx + 1]
    hits_z = 0
    hits_g = 0
    for s in seeds:
        refl = make_flaw_reflector(ctx.part, (lo, hi), ctx.cal, s)
        spec = dataclasses.replace(ctx.part, reflectors=(refl,), seed=s)
        ax_idx, an_idx = hole_index(spec, refl)
        target = FlawTarget(ax_idx, an_idx, refl.depth_mm, z_idx)
        hz, hg = score_flawed_bar(spec, target, ctx.noise, ctx.arms)
        hits_z += hz
        hits_g += hg
    return hits_z, hits_g


def summarize_zone_arm_flaw(
    k: int,
    n: int,
    z_cal: ZoneCalibration,
) -> dict[str, Any]:
    """Summarize detection and calibration metrics for a zone in zone arm."""
    rate = float(k / n) if n > 0 else 0.0
    low, high = clopper_pearson(k, n)
    snr_val = None if math.isnan(z_cal.snr) else float(z_cal.snr)
    return {
        "zone": z_cal.zone,
        "depth_mm": [float(z_cal.depth_mm[0]), float(z_cal.depth_mm[1])],
        "k": k,
        "n": n,
        "rate": rate,
        "ci_low": low,
        "ci_high": high,
        "snr": snr_val,
        "alarm_level": float(z_cal.alarm_level),
        "cap_bound": bool(z_cal.cap_bound),
        "inspectable": bool(z_cal.inspectable),
    }


def summarize_global_arm_flaw(
    k: int,
    n: int,
    z_cal: ZoneCalibration,
    global_arm: ArmGates,
) -> dict[str, Any]:
    """Summarize detection and global threshold metrics for a zone in global arm."""
    rate = float(k / n) if n > 0 else 0.0
    low, high = clopper_pearson(k, n)
    gate = global_arm.gates[0]
    g_cal = global_arm.global_cal
    snr_val = float(g_cal[0]) if (g_cal and not math.isnan(g_cal[0])) else None
    cap_bound = bool(g_cal[2]) if g_cal else False
    return {
        "zone": z_cal.zone,
        "depth_mm": [float(z_cal.depth_mm[0]), float(z_cal.depth_mm[1])],
        "k": k,
        "n": n,
        "rate": rate,
        "ci_low": low,
        "ci_high": high,
        "snr": snr_val,
        "alarm_level": float(gate.alarm_level),
        "cap_bound": cap_bound,
        "inspectable": bool(gate.inspectable),
    }


def _eval_all_flaw_zones(
    ctx: SimContext,
    cfg: RunConfig,
    rec: CalibrationRecord,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run flawed simulations across all zones and assemble summaries."""
    n_zones = len(rec.zones)
    fz_zones: list[dict[str, Any]] = []
    fg_zones: list[dict[str, Any]] = []
    for z in range(n_zones):
        seeds = [cfg.seed + cfg.n_clean + z * cfg.n_flaw + j for j in range(cfg.n_flaw)]
        hz, hg = evaluate_flawed_zone(ctx, z, seeds)
        fz_zones.append(summarize_zone_arm_flaw(hz, cfg.n_flaw, rec.zones[z]))
        fg_zones.append(
            summarize_global_arm_flaw(hg, cfg.n_flaw, rec.zones[z], ctx.arms.global_arm)
        )
    return fz_zones, fg_zones


def _build_arms(
    rec: CalibrationRecord,
    noise: TiNoiseConfig,
    cal: TiCalConfig,
) -> EvalArms:
    """Build zone and global evaluation arm gate containers."""
    zone_arm = build_zone_arm(rec)
    global_arm = build_global_arm(rec, noise, cal)
    return EvalArms(zone=zone_arm, global_arm=global_arm)


def _pack_model_metrics(
    rec: CalibrationRecord,
    clean_pair: tuple[dict[str, Any], dict[str, Any]],
    flaw_pair: tuple[list[dict[str, Any]], list[dict[str, Any]]],
    arms: EvalArms,
) -> dict[str, Any]:
    """Package calibration and arm metrics for one noise model."""
    cz_metrics, cg_metrics = clean_pair
    fz_zones, fg_zones = flaw_pair
    return {
        "calibration": record_to_dict(rec),
        "arms": {
            "zone": {
                "clean": cz_metrics,
                "flawed": {
                    "per_zone": fz_zones,
                    "zones_not_inspectable": arms.zone.uninspectable_count,
                },
                "zones_not_inspectable": arms.zone.uninspectable_count,
            },
            "global": {
                "clean": cg_metrics,
                "flawed": {
                    "per_zone": fg_zones,
                    "zones_not_inspectable": arms.global_arm.uninspectable_count,
                },
                "zones_not_inspectable": arms.global_arm.uninspectable_count,
            },
        },
    }


def evaluate_noise_model(
    part: BarSpec,
    noise: TiNoiseConfig,
    cal: TiCalConfig,
    cfg: RunConfig,
) -> dict[str, Any]:
    """Evaluate one noise model across zone and global arms."""
    rec = calibrate_zones(part, noise, cal)
    arms = _build_arms(rec, noise, cal)
    ctx = SimContext(part=part, noise=noise, cal=cal, arms=arms)
    clean_seeds = [cfg.seed + i for i in range(cfg.n_clean)]
    clean_pair = evaluate_clean_bars(part, noise, clean_seeds, arms)
    flaw_pair = _eval_all_flaw_zones(ctx, cfg, rec)
    return _pack_model_metrics(rec, clean_pair, flaw_pair, arms)


def make_seeds_dict(cfg: RunConfig, n_zones: int) -> dict[str, Any]:
    """Build deterministic seed dictionary for clean and flawed runs."""
    clean_seeds = [cfg.seed + i for i in range(cfg.n_clean)]
    flaw_seeds = [
        [cfg.seed + cfg.n_clean + z * cfg.n_flaw + j for j in range(cfg.n_flaw)]
        for z in range(n_zones)
    ]
    return {
        "base_seed": cfg.seed,
        "clean_seeds": clean_seeds,
        "flaw_seeds": flaw_seeds,
    }


def assemble_payload(
    models_data: dict[str, Any],
    cfg: RunConfig,
    sha: str,
    runtime_s: float,
) -> dict[str, Any]:
    """Assemble final JSON payload for titanium zone evaluation."""
    n_zones = len(models_data["structured"]["calibration"]["zones"])
    seeds_dict = make_seeds_dict(cfg, n_zones)
    cal_dict = {k: v["calibration"] for k, v in models_data.items()}
    return {
        "label": LABEL_SYNTHETIC,
        "config_sha256": sha,
        "definitions": DEFINITIONS,
        "counts": {"n_clean": cfg.n_clean, "n_flaw": cfg.n_flaw, "n_zones": n_zones},
        "seeds": seeds_dict,
        "runtime_s": round(runtime_s, 3),
        "noise_models": models_data,
        "models": models_data,
        "calibration": cal_dict,
    }


def run_ti_zones(
    cfg: RunConfig,
    part: BarSpec | None = None,
) -> dict[str, Any]:
    """Run titanium zone calibration and evaluation experiment."""
    t0 = time.perf_counter()
    part_spec = part if part is not None else default_part_spec()
    noise_cfg = load_ti_noise()
    cal_cfg = load_ti_calibration()
    sha = compute_config_sha256()
    models = make_noise_models(noise_cfg)
    models_data: dict[str, Any] = {}
    for name, noise in models.items():
        models_data[name] = evaluate_noise_model(part_spec, noise, cal_cfg, cfg)
    runtime = time.perf_counter() - t0
    payload = assemble_payload(models_data, cfg, sha, runtime)
    if cfg.out_path:
        cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> RunConfig:
    """Parse CLI options for titanium zone evaluation study."""
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Synthetic study of zone calibration under titanium-like noise."
    )
    parser.add_argument("--n-clean", type=int, default=20, help="Clean bar count")
    parser.add_argument("--n-flaw", type=int, default=20, help="Flawed bars per zone")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/ti_zone_metrics.json"),
        help="Output metrics path",
    )
    parser.add_argument("--quick", action="store_true", help="Quick run")
    args = parser.parse_args(raw_args)
    out_given = any(a == "--out" or a.startswith("--out=") for a in raw_args)
    if args.quick and not out_given:
        raise ValueError("--quick requires explicit --out path to prevent overwriting results/")
    n_clean = 2 if args.quick else args.n_clean
    n_flaw = 2 if args.quick else args.n_flaw
    return RunConfig(
        n_clean=n_clean,
        n_flaw=n_flaw,
        seed=args.seed,
        out_path=args.out,
        quick=args.quick,
    )


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entrypoint for titanium zone experiment."""
    cfg = parse_args(argv)
    return run_ti_zones(cfg)


if __name__ == "__main__":
    main()
