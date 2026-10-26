"""Command-line interface for UTriage. Subcommands are added phase by phase."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from utriage import __version__
from utriage.physics.dataset import DatasetConfig, generate_dataset, load_scan
from utriage.physics.figures import plot_cscan

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
