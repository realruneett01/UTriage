# ruff: noqa: E402
"""Phase 5 experiment: Triage fusion, uncertainty & ranking (Synthetic data)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Ensure local workspace src is prioritized
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        sys.modules["utriage"].__path__.insert(0, _pkg_dir)
if "utriage.triage" in sys.modules:
    _triage_dir = str(Path(_SRC) / "utriage" / "triage")
    if _triage_dir not in sys.modules["utriage.triage"].__path__:
        sys.modules["utriage.triage"].__path__.insert(0, _triage_dir)

import numpy as np
import torch

from utriage.ml.bar_data import WindowConfig, make_test_set
from utriage.ml.bar_infer import ScanSettings, load_window_model, scan_bar
from utriage.ml.bar_models import WindowNet, flaw_logit
from utriage.physics.simulator import (
    BarScan,
    BarSpec,
    Material,
    Reflector,
    load_material,
    load_probe,
    simulate_bar,
)
from utriage.rules.config import load_acceptance
from utriage.rules.pipeline import evaluate_scan
from utriage.rules.scoring import MatchTol, matches
from utriage.rules.types import Indication
from utriage.triage.campaign import item_is_real, random_order_effort, review_effort
from utriage.triage.fusion import FusionConfig, TriageItem, fuse, rank, rules_flags_preserved
from utriage.triage.operating_point import OperatingPoint, choose_operating_point
from utriage.triage.uncertainty import (
    Conformal,
    coverage,
    fit_conformal,
    is_uncertain,
    prediction_sets,
)


@dataclass(frozen=True)
class _CalData:
    probs: np.ndarray
    labels: np.ndarray
    sizes: np.ndarray


@dataclass(frozen=True)
class _RankingContext:
    model: WindowNet
    op: OperatingPoint
    rng: np.random.Generator
    bar_idx: int
    device: str


@dataclass(frozen=True)
class _RunConfig:
    quick: bool
    seed: int
    device: str


def _select_device(device_arg: str) -> str:
    """Select compute device and log choice."""
    if device_arg == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = device_arg
    print(f"Device: {dev}")
    return dev


def _find_model_path() -> Path:
    """Find WindowNet checkpoint from local directory or previous phases."""
    model_name = "bar_2d.pt"
    metrics_path = Path("results/bar_domain_metrics.json")
    if metrics_path.exists():
        try:
            with open(metrics_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            choice = meta.get("kept_model", {}).get("choice", "ml_2d")
            model_name = "bar_2d.pt" if "2d" in choice else "bar_1d.pt"
        except (json.JSONDecodeError, KeyError, OSError):
            # Fall back to default model_name if metrics cannot be parsed
            model_name = "bar_2d.pt"
    candidates = (
        Path("models") / model_name,
        Path("../p4_exp/models") / model_name,
        Path("../h2_a/models") / model_name,
    )
    for p in candidates:
        if p.exists():
            return p
    return Path("models") / model_name


@torch.no_grad()
def _predict_flaw_probs(
    model: WindowNet,
    windows: np.ndarray,
    dev: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    """Predict flaw probabilities for input windows in batches."""
    model.eval()
    model.to(dev)
    n = len(windows)
    probs = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = torch.from_numpy(windows[start:end]).to(dev)
        logits, _ = model(batch)
        probs[start:end] = torch.sigmoid(flaw_logit(logits)).cpu().numpy()
    return probs


def _run_conformal(
    all_probs: np.ndarray,
    all_labels: np.ndarray,
    split_idx: int,
) -> tuple[Conformal, dict[str, Any]]:
    """Calibrate conformal prediction on first half and evaluate on second."""
    cal_probs = all_probs[:split_idx]
    cal_labels = all_labels[:split_idx]
    eval_probs = all_probs[split_idx:]
    eval_labels = all_labels[split_idx:]

    conf = fit_conformal(cal_probs, cal_labels, alpha=0.1)
    sets = prediction_sets(conf, eval_probs)
    cov = coverage(sets, eval_labels)
    unc = is_uncertain(sets)
    unc_fraction = float(np.mean(unc))

    summary = {
        "alpha": conf.alpha,
        "coverage": float(cov),
        "uncertain_fraction": float(unc_fraction),
        "n": len(eval_labels),
    }
    return conf, summary


def _run_operating_point(
    cal: _CalData,
    eval_noise: np.ndarray,
) -> tuple[OperatingPoint, float]:
    """Select operating point on calibration set and evaluate FCR on eval."""
    cal_flaws = cal.labels == 1
    cal_noise = cal.labels == 0
    flaw_scores = cal.probs[cal_flaws]
    flaw_sizes = cal.sizes[cal_flaws]
    noise_scores = cal.probs[cal_noise]

    op = choose_operating_point(
        flaw_scores=flaw_scores,
        flaw_sizes=flaw_sizes,
        noise_scores=noise_scores,
        ref_size=0.8,
        target=0.90,
    )
    eval_fcr = float(np.mean(eval_noise >= op.threshold)) if len(eval_noise) > 0 else 0.0
    return op, eval_fcr


def _generate_planted_reflectors(rng: np.random.Generator) -> tuple[Reflector, ...]:
    """Generate 3 planted reflectors between 0.4 and 1.2 mm."""
    refs: list[Reflector] = []
    for _ in range(3):
        ax = float(rng.uniform(6.0, 26.0))
        ang = float(rng.uniform(0.0, 360.0))
        dep = float(rng.uniform(6.0, 30.0))
        sz = float(rng.uniform(0.4, 1.2))
        refs.append(Reflector(axial_mm=ax, angle_deg=ang, depth_mm=dep, diameter_mm=sz))
    return tuple(refs)


def _simulate_ranking_bar(rng: np.random.Generator, refs: tuple[Reflector, ...]) -> BarScan:
    """Simulate a single Ti64 bar with noise U(1.0, 2.0)."""
    scale = float(rng.uniform(1.0, 2.0))
    base_mat = load_material("ti64")
    mat = Material(
        name=base_mat.name,
        velocity_m_s=base_mat.velocity_m_s,
        density_kg_m3=base_mat.density_kg_m3,
        attenuation_db_mm_mhz=base_mat.attenuation_db_mm_mhz,
        grain_noise=base_mat.grain_noise * scale,
    )
    probe = load_probe("p5")
    water = load_material("water")
    spec = BarSpec(
        material=mat,
        probe=probe,
        water=water,
        diameter_mm=40.0,
        length_mm=32.0,
        axial_step_mm=1.0,
        n_angle=60,
        reflectors=refs,
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    return simulate_bar(spec)


def _coords_match(
    coords: tuple[float, float, float],
    ref: Reflector,
    spec: BarSpec,
    tol: MatchTol,
) -> bool:
    """True if coordinates match reflector within tolerance."""
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


def _item_matches_reflector(
    item: TriageItem,
    ref: Reflector,
    spec: BarSpec,
    scan: BarScan,
) -> bool:
    """Check if triage item matches a specific reflector."""
    if item.indication is not None:
        return matches(item.indication, ref, spec)
    if item.ml is not None:
        coords = (
            float(scan.axial_mm[item.ml.axial_idx]),
            float(scan.angle_deg[item.ml.angle_idx]),
            float(item.ml.depth_mm),
        )
        return _coords_match(coords, ref, spec, MatchTol())
    return False


def _evaluate_bar_flaws(
    ranked_items: list[TriageItem],
    rule_inds: list[Indication],
    spec: BarSpec,
    scan: BarScan,
) -> tuple[int, int]:
    """Count planted flaws found by rules only vs union."""
    rules_found = sum(any(matches(ind, ref, spec) for ind in rule_inds) for ref in spec.reflectors)
    union_found = sum(
        any(_item_matches_reflector(it, ref, spec, scan) for it in ranked_items)
        for ref in spec.reflectors
    )
    return int(rules_found), int(union_found)


def _run_single_bar_ranking(ctx: _RankingContext) -> dict[str, Any]:
    """Run simulation, rules detection, ML scan, fusion, and ranking for one bar."""
    refs = _generate_planted_reflectors(ctx.rng)
    scan = _simulate_ranking_bar(ctx.rng, refs)

    acceptance_cfg = load_acceptance()
    eval_res = evaluate_scan(scan, acceptance_cfg)
    rule_inds = eval_res.indications

    settings = ScanSettings(threshold=ctx.op.threshold)
    findings = scan_bar(ctx.model, scan, settings=settings, device=ctx.device)

    fused_items = fuse(rule_inds, findings, FusionConfig(ml_threshold=ctx.op.threshold))
    ranked_items = rank(fused_items)

    preserved = rules_flags_preserved(rule_inds, ranked_items)
    assert preserved, f"Bar {ctx.bar_idx}: rules indications were not preserved!"  # noqa: S101

    ranked_real = [item_is_real(it, scan.spec, scan) for it in ranked_items]
    n_real = len(scan.spec.reflectors)
    effort_r = review_effort(ranked_real, n_real)
    bar_seed = int(ctx.rng.integers(0, 2**31 - 1))
    effort_rand = random_order_effort(ranked_real, n_real, n_perm=200, seed=bar_seed)

    rules_found, union_found = _evaluate_bar_flaws(ranked_items, rule_inds, scan.spec, scan)

    return {
        "bar_idx": ctx.bar_idx,
        "n_planted": n_real,
        "rules_found": rules_found,
        "union_found": union_found,
        "effort_ranked": int(effort_r) if effort_r is not None else None,
        "effort_random": float(effort_rand) if not np.isnan(effort_rand) else None,
        "n_items": len(ranked_items),
        "n_rules": len(rule_inds),
        "n_ml_only": len(ranked_items) - len(rule_inds),
        "preserved": preserved,
    }


def _run_ranking_campaign(
    model: WindowNet,
    op: OperatingPoint,
    cfg: _RunConfig,
) -> dict[str, Any]:
    """Execute ranking evaluation across synthetic bars."""
    n_bars = 3 if cfg.quick else 20
    rng = np.random.default_rng(cfg.seed)
    per_bar: list[dict[str, Any]] = []
    union_violations = 0

    for idx in range(n_bars):
        ctx = _RankingContext(model=model, op=op, rng=rng, bar_idx=idx, device=cfg.device)
        res = _run_single_bar_ranking(ctx)
        if not res["preserved"]:
            union_violations += 1
        per_bar.append(res)
        print(
            f"Bar {idx + 1}/{n_bars}: ranked effort={res['effort_ranked']}, "
            f"rand effort={res['effort_random']}, flaws={res['union_found']}/{res['n_planted']}"
        )

    valid_ranked = [r["effort_ranked"] for r in per_bar if r["effort_ranked"] is not None]
    valid_rand = [r["effort_random"] for r in per_bar if r["effort_random"] is not None]
    total_planted = sum(r["n_planted"] for r in per_bar)
    total_rules = sum(r["rules_found"] for r in per_bar)
    total_union = sum(r["union_found"] for r in per_bar)

    return {
        "mean_effort_ranked": float(np.mean(valid_ranked)) if valid_ranked else None,
        "mean_effort_random": float(np.mean(valid_rand)) if valid_rand else None,
        "flaws_found_rules_only": float(total_rules / total_planted) if total_planted > 0 else 0.0,
        "flaws_found_union": float(total_union / total_planted) if total_planted > 0 else 0.0,
        "n_bars": len(per_bar),
        "per_bar": per_bar,
        "union_violations": union_violations,
    }


def _execute_phase5(cfg: _RunConfig) -> None:
    """Execute complete Phase 5 pipeline and write results/triage_metrics.json."""
    device = _select_device(cfg.device)
    model_path = _find_model_path()
    print(f"Loading WindowNet from {model_path}...")
    model, _ = load_window_model(model_path, device=device)
    is_patch = getattr(model.stem[0], "in_channels", 1) == 9

    n_test_bars = 2 if cfg.quick else 12
    wcfg = WindowConfig(patch=is_patch, seed=cfg.seed)
    sizes_mm = (0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2)
    noise_scales = (0.5, 1.0, 1.5, 2.0)
    print(f"Generating test windows (n_bars={n_test_bars}, patch={is_patch})...")
    wset = make_test_set(wcfg, n_bars=n_test_bars, sizes_mm=sizes_mm, noise_scales=noise_scales)

    total_bars = len(noise_scales) * n_test_bars
    windows_per_bar = 2 * len(sizes_mm)
    split_idx = (total_bars // 2) * windows_per_bar

    dev = torch.device(device)
    all_probs = _predict_flaw_probs(model, wset.x, dev)
    all_labels = (wset.y == 1).astype(int)

    conf, conformal_res = _run_conformal(all_probs, all_labels, split_idx)
    print(
        f"Conformal: coverage={conformal_res['coverage']:.3f}, "
        f"uncertain={conformal_res['uncertain_fraction']:.3f}"
    )

    cal_data = _CalData(
        probs=all_probs[:split_idx],
        labels=all_labels[:split_idx],
        sizes=wset.size_mm[:split_idx],
    )
    eval_noise = all_probs[split_idx:][all_labels[split_idx:] == 0]
    op, eval_fcr = _run_operating_point(cal_data, eval_noise)
    print(
        f"Operating point: threshold={op.threshold:.4f}, method={op.method}, "
        f"eval FCR={eval_fcr:.4f}"
    )

    ranking_summary = _run_ranking_campaign(model, op, cfg)
    print(
        f"Ranking: mean ranked={ranking_summary['mean_effort_ranked']}, "
        f"mean random={ranking_summary['mean_effort_random']}"
    )

    output = {
        "label": "Synthetic data",
        "conformal": conformal_res,
        "operating_point": asdict(op),
        "ranking": {
            "mean_effort_ranked": ranking_summary["mean_effort_ranked"],
            "mean_effort_random": ranking_summary["mean_effort_random"],
            "flaws_found_rules_only": ranking_summary["flaws_found_rules_only"],
            "flaws_found_union": ranking_summary["flaws_found_union"],
            "n_bars": ranking_summary["n_bars"],
            "per_bar": ranking_summary["per_bar"],
        },
        "union_violations": ranking_summary["union_violations"],
        "property_test": "tests/test_fusion.py 10,000 examples",
    }

    out_path = Path("results/triage_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"Results written to {out_path}")


def main() -> None:
    """CLI entry point for Phase 5 experiment."""
    parser = argparse.ArgumentParser(
        description="Phase 5: Triage fusion, uncertainty & ranking (Synthetic data)"
    )
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer bars")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('auto', 'cuda', 'cpu')",
    )
    args = parser.parse_args()
    cfg = _RunConfig(quick=args.quick, seed=args.seed, device=args.device)
    _execute_phase5(cfg)


if __name__ == "__main__":
    main()
