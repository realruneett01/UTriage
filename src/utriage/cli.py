"""Command-line interface for UTriage. Subcommands are added phase by phase."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from utriage import __version__
from utriage.physics.dataset import DatasetConfig, generate_dataset, load_scan
from utriage.physics.figures import plot_cscan
from utriage.physics.simulator import BarScan
from utriage.rules.config import DEFAULT_ACCEPTANCE, AcceptanceConfig, load_acceptance
from utriage.rules.pipeline import Evaluation, evaluate_scan, evaluation_to_dict

app = typer.Typer(
    help="UTriage: advisory, offline UT indication triage. Decisions belong to the inspector.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """UTriage: advisory and offline. It never accepts or passes a part."""


@app.command()
def version() -> None:
    """Print the UTriage version."""
    typer.echo(f"utriage {__version__}")


def _process_bar(npz_path: Path, fig_dir: Path | None, idx: int) -> None:
    """Load a simulated bar, optionally plot C-scan, and print summary."""
    scan = load_scan(npz_path)
    if fig_dir is not None:
        plot_cscan(scan, fig_dir / f"cscan_{idx:04d}.png")
    n_refl = len(scan.spec.reflectors)
    artefacts = ", ".join(a.kind for a in scan.spec.artefacts) or "none"
    typer.echo(f"{npz_path.name}: {n_refl} reflectors, artefacts: {artefacts}")


@app.command()
def simulate(
    material: Annotated[str, typer.Option("--material", help="Material name")] = "ti64",
    probe: Annotated[str, typer.Option("--probe", help="Probe config")] = "p5",
    bars: Annotated[int, typer.Option("--bars", help="Number of bars to simulate")] = 5,
    seed: Annotated[int, typer.Option("--seed", help="Random seed")] = 0,
    out: Annotated[Path | None, typer.Option("--out", help="Output directory")] = None,
    length_mm: Annotated[float, typer.Option("--length-mm", help="Bar length in mm")] = 64.0,
    n_angle: Annotated[int, typer.Option("--n-angle", help="Number of angular steps")] = 120,
    figures: Annotated[bool, typer.Option("--figures/--no-figures", help="Plot C-scans")] = True,
) -> None:
    """Generate synthetic bar UT scans with ground truth (Synthetic data)."""
    out_dir = out if out is not None else Path(f"data/synthetic/{material}")
    cfg = DatasetConfig(
        n_bars=bars,
        material=material,
        probe=probe,
        seed=seed,
        length_mm=length_mm,
        n_angle=n_angle,
    )
    npz_paths = generate_dataset(cfg, out_dir)
    fig_dir = out_dir / "figures" if figures else None

    for i, npz_path in enumerate(npz_paths):
        _process_bar(npz_path, fig_dir, i)

    typer.echo(f"Synthetic data written to {out_dir}")


def _attach_scan_label(scan: BarScan, scan_path: Path) -> None:
    json_path = scan_path.with_suffix(".json")
    if not json_path.exists():
        return
    try:
        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        scan.label = data.get("label", "")
    except (OSError, json.JSONDecodeError):  # ignore unreadable or invalid json
        scan.label = ""


def _print_evaluation(ev: Evaluation) -> None:
    typer.echo("Advisory tool - decisions by certified inspector")
    if ev.label:
        typer.echo(f"Data source: {ev.label}")
    if ev.placeholder_limits:
        typer.echo("WARNING: Acceptance limits are PLACEHOLDERS (UNVERIFIED)")
    for ind in ev.indications:
        typer.echo(
            f"Indication #{ind.id}: {ind.category} | axial={ind.axial_mm:.1f}mm "
            f"angle={ind.angle_deg:.1f}deg depth={ind.depth_mm:.1f}mm "
            f"pct={ind.amplitude_pct:.1f}% snr={ind.snr_db:.1f}dB"
        )
    for fault in ev.quality.faults:
        typer.echo(
            f"Fault: {fault.kind} | axial=({fault.axial_mm[0]:.1f}, {fault.axial_mm[1]:.1f})mm "
            f"angle=({fault.angle_deg[0]:.1f}, {fault.angle_deg[1]:.1f})deg ascans={fault.n_ascans}"
        )
    typer.echo(f"Summary: {ev.summary}")


def _write_json_out(ev: Evaluation, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(evaluation_to_dict(ev), fh, indent=2)


def _load_cfg(acceptance: Path) -> AcceptanceConfig:
    cfg_path = acceptance if acceptance.exists() else DEFAULT_ACCEPTANCE
    return load_acceptance(cfg_path)


@app.command()
def evaluate(
    scan: Annotated[Path, typer.Argument(help="Path to scan .npz file")],
    acceptance: Annotated[
        Path,
        typer.Option("--acceptance", help="Path to acceptance configuration"),
    ] = Path("configs/acceptance.yaml"),
    json_out: Annotated[
        Path | None,
        typer.Option("--json-out", help="Optional path to output JSON file"),
    ] = None,
) -> None:
    """Evaluate a bar scan against acceptance rules."""
    bar_scan = load_scan(scan)
    _attach_scan_label(bar_scan, scan)
    cfg = _load_cfg(acceptance)
    ev = evaluate_scan(bar_scan, cfg)
    _print_evaluation(ev)
    if json_out is not None:
        _write_json_out(ev, json_out)
