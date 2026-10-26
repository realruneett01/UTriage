"""Phase 2 baseline experiment: detection rate, false calls, and watchdog check.

All outputs are Synthetic data (project plan 2.4 and Phase 2 gate).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        sys.modules["utriage"].__path__.insert(0, _pkg_dir)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from utriage.physics.simulator import (  # noqa: E402
    ARTEFACT_KINDS,
    Artefact,
    BarSpec,
    Reflector,
    load_material,
    load_probe,
    simulate_bar,
)
from utriage.rules.config import AcceptanceConfig, load_acceptance  # noqa: E402
from utriage.rules.pipeline import dac_for, evaluate_scan  # noqa: E402
from utriage.rules.scoring import fault_overlaps, score_bar  # noqa: E402


@dataclasses.dataclass(frozen=True)
class SimSettings:
    """Fixed parameters for synthetic bar evaluation."""

    diameter_mm: float = 40.0
    length_mm: float = 48.0
    n_angle: int = 72
    axial_step_mm: float = 1.0


@dataclasses.dataclass(frozen=True)
class CellParams:
    """Parameters for one grid cell evaluation."""

    noise_scale: float
    fbh_mm: float
    n_bars: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 rules engine baseline")
    parser.add_argument("--quick", action="store_true", help="2 bars per cell")
    return parser.parse_args()


def _make_reflector_trio(fbh_mm: float, rng: np.random.Generator) -> tuple[Reflector, ...]:
    depths = rng.uniform(5.0, 30.0, size=3)
    axials = (8.0, 24.0, 40.0)
    angles = (0.0, 120.0, 240.0)
    return tuple(
        Reflector(axial_mm=ax, angle_deg=ang, depth_mm=float(d), diameter_mm=fbh_mm)
        for ax, ang, d in zip(axials, angles, depths, strict=True)
    )


def _eval_cell_bar(
    spec: BarSpec,
    cfg: AcceptanceConfig,
    dac: object,
) -> tuple[list[bool], int]:
    scan = simulate_bar(spec)
    evaluation = evaluate_scan(scan, cfg, dac=dac)
    score = score_bar(evaluation.indications, spec)
    return score.hits, score.false_calls


def _run_cell(
    settings: SimSettings,
    params: CellParams,
    rng: np.random.Generator,
    context: tuple[AcceptanceConfig, object, int],
) -> dict:
    cfg, dac, base_seed = context
    base_mat = load_material("ti64")
    mat = dataclasses.replace(base_mat, grain_noise=base_mat.grain_noise * params.noise_scale)
    probe, water = load_probe("p5"), load_material("water")
    all_hits: list[bool] = []
    total_fc = 0
    for i in range(params.n_bars):
        refs = _make_reflector_trio(params.fbh_mm, rng)
        spec = BarSpec(
            material=mat,
            probe=probe,
            water=water,
            diameter_mm=settings.diameter_mm,
            length_mm=settings.length_mm,
            axial_step_mm=settings.axial_step_mm,
            n_angle=settings.n_angle,
            reflectors=refs,
            seed=base_seed + i,
        )
        hits, fc = _eval_cell_bar(spec, cfg, dac)
        all_hits.extend(hits)
        total_fc += fc
    det_rate = float(np.mean(all_hits)) if all_hits else 0.0
    return {
        "noise_scale": params.noise_scale,
        "fbh_mm": params.fbh_mm,
        "n_reflectors": len(all_hits),
        "detection_rate": det_rate,
        "false_calls_per_bar": total_fc / params.n_bars,
    }


def _run_all_cells(
    settings: SimSettings,
    n_bars: int,
    rng: np.random.Generator,
    context: tuple[AcceptanceConfig, object],
) -> list[dict]:
    cfg, dac = context
    noise_grid = (0.5, 1.0, 1.5, 2.0)
    fbh_grid = (0.4, 0.6, 0.8, 1.0, 1.2, 1.6)
    cells: list[dict] = []
    seed_idx = 0
    for ns in noise_grid:
        for fbh in fbh_grid:
            params = CellParams(noise_scale=ns, fbh_mm=fbh, n_bars=n_bars)
            cell = _run_cell(settings, params, rng, (cfg, dac, seed_idx))
            cells.append(cell)
            seed_idx += n_bars
    return cells


def _eval_watchdog_bar(art: Artefact, spec: BarSpec, cfg: AcceptanceConfig) -> bool:
    scan = simulate_bar(spec)
    evaluation = evaluate_scan(scan, cfg)
    return any(fault_overlaps(fault, art) for fault in evaluation.quality.faults)


def _run_watchdog(
    settings: SimSettings,
    cfg: AcceptanceConfig,
    n_bars: int = 3,
) -> list[dict]:
    mat, probe, water = load_material("ti64"), load_probe("p5"), load_material("water")
    rows: list[dict] = []
    for kind_idx, kind in enumerate(ARTEFACT_KINDS):
        art = Artefact(kind=kind, axial_mm=(10.0, 30.0), angle_deg=(60.0, 160.0), severity=1.0)
        caught_count = 0
        for i in range(n_bars):
            spec = BarSpec(
                material=mat,
                probe=probe,
                water=water,
                diameter_mm=settings.diameter_mm,
                length_mm=settings.length_mm,
                axial_step_mm=settings.axial_step_mm,
                n_angle=settings.n_angle,
                artefacts=(art,),
                seed=1000 + kind_idx * 10 + i,
            )
            if _eval_watchdog_bar(art, spec, cfg):
                caught_count += 1
        rows.append({"kind": kind, "n": n_bars, "caught": caught_count})
    return rows


def _plot_detection(cells: list[dict], out_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    noise_levels = sorted({c["noise_scale"] for c in cells})
    for ns in noise_levels:
        pts = sorted([c for c in cells if c["noise_scale"] == ns], key=lambda x: x["fbh_mm"])
        x = [p["fbh_mm"] for p in pts]
        y = [p["detection_rate"] for p in pts]
        plt.plot(x, y, marker="o", label=f"Noise {ns}x")
    plt.xlabel("FBH diameter (mm)")
    plt.ylabel("Detection rate")
    plt.title("Rules Engine Detection Rate vs FBH Size (Synthetic data)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_false_calls(cells: list[dict], out_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    noise_levels = sorted({c["noise_scale"] for c in cells})
    avg_fc = [
        float(np.mean([c["false_calls_per_bar"] for c in cells if c["noise_scale"] == ns]))
        for ns in noise_levels
    ]
    plt.plot(noise_levels, avg_fc, marker="s", color="darkred", linewidth=2)
    plt.xlabel("Noise scale")
    plt.ylabel("False calls per bar")
    plt.title("Rules Engine False Calls vs Noise Scale (Synthetic data)")
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _check_gate(cells: list[dict], watchdog: list[dict]) -> dict:
    low_noise_ref = [c for c in cells if c["noise_scale"] == 0.5 and c["fbh_mm"] >= 0.8]
    det_pass = all(c["detection_rate"] >= 1.0 for c in low_noise_ref)
    wd_pass = all(w["caught"] == w["n"] for w in watchdog)
    return {
        "detection_rate_low_noise_at_or_above_reference": bool(det_pass),
        "all_artefacts_caught": bool(wd_pass),
        "acceptance_limits": "PLACEHOLDER",
        "passed": bool(det_pass and wd_pass),
    }


def _save_results(output: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "rules_baseline.json").open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)


def main() -> None:
    t0 = time.time()
    args = _parse_args()
    n_bars = 2 if args.quick else 4
    rng = np.random.default_rng(0)
    settings = SimSettings()
    cfg = load_acceptance()
    base_mat = load_material("ti64")
    base_spec = BarSpec(
        material=base_mat,
        probe=load_probe("p5"),
        water=load_material("water"),
        diameter_mm=settings.diameter_mm,
        length_mm=settings.length_mm,
        seed=0,
    )
    base_scan = simulate_bar(base_spec)
    dac = dac_for(base_scan)

    cells = _run_all_cells(settings, n_bars, rng, (cfg, dac))
    watchdog = _run_watchdog(settings, cfg, n_bars=3)
    gate = _check_gate(cells, watchdog)
    output = {
        "label": "Synthetic data",
        "sim_settings": dataclasses.asdict(settings),
        "grid": cells,
        "watchdog": watchdog,
        "gate": gate,
    }
    _save_results(output, Path("results"))
    fig_dir = Path("results/figures")
    _plot_detection(cells, fig_dir / "rules_detection_vs_size.png")
    _plot_false_calls(cells, fig_dir / "rules_false_calls_vs_noise.png")

    elapsed = time.time() - t0
    print(f"Phase 2 experiment completed in {elapsed:.2f} s")
    passed = gate["passed"]
    det_ok = gate["detection_rate_low_noise_at_or_above_reference"]
    wd_ok = gate["all_artefacts_caught"]
    print(f"Gate passed: {passed} (det: {det_ok}, wd: {wd_ok})")


if __name__ == "__main__":
    main()
