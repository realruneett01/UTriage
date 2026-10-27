"""Command-line interface for UTriage. Subcommands are added phase by phase."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from utriage import __version__
from utriage.integrity import MANIFEST, IntegrityError, require_locked
from utriage.physics.dataset import DatasetConfig, generate_dataset, load_scan
from utriage.physics.figures import plot_cscan
from utriage.physics.simulator import BarScan
from utriage.rules.config import DEFAULT_ACCEPTANCE, AcceptanceConfig, load_acceptance
from utriage.rules.pipeline import Evaluation, evaluate_scan, evaluation_to_dict
from utriage.workflow import ADVISORY_BANNER, run_batch

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


MANIFEST_PATH: Path | None = None
REPO_ROOT: Path | None = None


def _enforce_locked() -> None:
    """Enforce configuration and model integrity under UTRIAGE_LOCKED=1."""
    manifest_p = MANIFEST_PATH if MANIFEST_PATH is not None else MANIFEST
    root_p = REPO_ROOT if REPO_ROOT is not None else manifest_p.parents[1]
    try:
        require_locked(manifest_p, root_p)
    except IntegrityError as exc:
        typer.secho(f"ERROR: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


def _process_bar(npz_path: Path, fig_dir: Path | None, idx: int) -> None:
    """Load a simulated bar, optionally plot C-scan, and print summary."""
    scan = load_scan(npz_path)
    if fig_dir is not None:
        plot_cscan(scan, fig_dir / f"cscan_{idx:04d}.png")
    n_refl = len(scan.spec.reflectors)
    artefacts = ", ".join(a.kind for a in scan.spec.artefacts) or "none"
    typer.echo(f"{npz_path.name}: {n_refl} reflectors, artefacts: {artefacts}")


@dataclass(frozen=True)
class SimulateOptions:
    """Options for simulating synthetic bar scans."""

    material: str = "ti64"
    probe: str = "p5"
    bars: int = 5
    seed: int = 0
    out: Path | None = None
    length_mm: float = 64.0
    n_angle: int = 120
    figures: bool = True


def _run_simulate(opts: SimulateOptions) -> None:
    """Execute synthetic dataset generation with the given options."""
    out_dir = opts.out if opts.out is not None else Path(f"data/synthetic/{opts.material}")
    cfg = DatasetConfig(
        n_bars=opts.bars,
        material=opts.material,
        probe=opts.probe,
        seed=opts.seed,
        length_mm=opts.length_mm,
        n_angle=opts.n_angle,
    )
    npz_paths = generate_dataset(cfg, out_dir)
    fig_dir = out_dir / "figures" if opts.figures else None

    for i, npz_path in enumerate(npz_paths):
        _process_bar(npz_path, fig_dir, i)

    typer.echo(f"Synthetic data written to {out_dir}")


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
    _enforce_locked()
    opts = SimulateOptions(
        material=material,
        probe=probe,
        bars=bars,
        seed=seed,
        out=out,
        length_mm=length_mm,
        n_angle=n_angle,
        figures=figures,
    )
    _run_simulate(opts)


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
    _enforce_locked()
    bar_scan = load_scan(scan)
    _attach_scan_label(bar_scan, scan)
    cfg = _load_cfg(acceptance)
    ev = evaluate_scan(bar_scan, cfg)
    _print_evaluation(ev)
    if json_out is not None:
        _write_json_out(ev, json_out)


def _resolve_default_model(model: Path | None) -> Path | None:
    """Resolve model path, falling back to default checkpoint if present."""
    if model is not None:
        return model
    default_p = Path("models/bar_2d.pt")
    return default_p if default_p.is_file() else None


def _format_summary_line(summary: dict) -> str:
    """Format single-line CLI summary for a processed scan."""
    bar = summary.get("bar_id", "bar")
    counts = summary.get("counts_per_category", {})
    counts_str = ", ".join(f"{k}={v}" for k, v in counts.items()) or "none"
    rescan = "yes" if summary.get("rescan_recommended") else "no"
    report_p = summary.get("report_path") or ""
    report_name = Path(report_p).name if report_p else "-"
    return f"{bar}: indications: [{counts_str}], rescan={rescan}, report={report_name}"


def _print_scan_result(summary: dict) -> bool:
    """Print either summary line or error in red. Returns True if failed."""
    err = summary.get("error")
    if err:
        bar = summary.get("bar_id", "bar")
        typer.secho(f"{bar}: ERROR: {err}", fg=typer.colors.RED)
        return True
    typer.echo(_format_summary_line(summary))
    return False


@app.command()
def triage(
    scans: Annotated[list[Path], typer.Argument(help="Paths to scan .npz files")],
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory for reports and summaries"),
    ] = Path("out"),
    model: Annotated[
        Path | None,
        typer.Option("--model", help="Optional ML model checkpoint path"),
    ] = None,
    acceptance: Annotated[
        Path | None,
        typer.Option("--acceptance", help="Optional path to acceptance configuration"),
    ] = None,
) -> None:
    """Batch shop-floor triage: advisory evaluation, report and JSON generation."""
    _enforce_locked()
    typer.echo(ADVISORY_BANNER)
    resolved_model = _resolve_default_model(model)
    summaries = run_batch(scans, out, model=resolved_model, acceptance=acceptance)
    has_failure = False
    for s in summaries:
        if _print_scan_result(s):
            has_failure = True
    if has_failure:
        raise typer.Exit(code=1)
