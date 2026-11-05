"""Command-line interface for UTriage. Subcommands are added phase by phase."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import torch.nn as nn
import typer

from utriage import __version__
from utriage.detector_config import DEFAULT_PATH, load_detector_config
from utriage.integrity import MANIFEST, IntegrityError, require_locked
from utriage.ml.anomaly import ConvAE2D, anomaly_score
from utriage.ml.site_adapt import (
    SiteCalibration,
    apply_site_stats,
    calibrate_site_threshold,
    fit_site_stats,
    min_clean_scans,
    recalibrate_batchnorm,
    save_site_profile,
)
from utriage.ml.train import pick_device
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


@dataclass(frozen=True)
class SiteCalibrateOptions:
    """Options for site adaptation threshold calibration."""

    clean_dir: Path
    out: Path
    label: str
    model: Path = Path("models/anomaly_2d.pt")
    fcr: float | None = None


def _resolve_site_fcr(fcr: float | None) -> float:
    """Resolve false call rate from option or detector config."""
    if fcr is not None:
        return fcr
    if DEFAULT_PATH.is_file():
        cfg = load_detector_config(DEFAULT_PATH)
        return float(cfg.anomaly.target_fcr)
    typer.secho(
        "Error: --fcr is required when detector config does not exist.",
        fg=typer.colors.RED,
    )
    raise typer.Exit(code=2)


def _load_clean_images(clean_dir: Path) -> list[np.ndarray]:
    """Load all .npy scan arrays from clean directory in sorted order."""
    if not clean_dir.is_dir():
        typer.secho(f"Error: Directory not found: {clean_dir}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    paths = sorted(clean_dir.glob("*.npy"))
    if not paths:
        typer.secho(f"Error: No .npy files found in {clean_dir}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    return [np.load(p, allow_pickle=False) for p in paths]


def _load_site_model(model_path: Path, dev: torch.device) -> ConvAE2D:
    """Load trained 2D convolutional autoencoder from checkpoint."""
    if not model_path.is_file():
        typer.secho(f"Error: Model not found at {model_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    ckpt = torch.load(model_path, map_location=dev, weights_only=True)
    if isinstance(ckpt, nn.Module):
        model = ckpt
    else:
        model = ConvAE2D()
        state = ckpt
        if isinstance(state, dict):
            state = state.get("state_dict", state.get("model_state_dict", state))
        model.load_state_dict(state)
    model.to(dev)
    model.eval()
    return model


def _prepare_scan_tensors(norm_imgs: list[np.ndarray]) -> list[torch.Tensor]:
    """Convert normalized arrays into 4D PyTorch tensors (B, 1, H, W)."""
    same_shape = all(img.shape == norm_imgs[0].shape for img in norm_imgs)
    if same_shape:
        stacked = np.stack(norm_imgs, axis=0)
        t = torch.as_tensor(stacked, dtype=torch.float32)
        if t.ndim == 3:
            t = t.unsqueeze(1)
        return [t]
    tensors: list[torch.Tensor] = []
    for img in norm_imgs:
        arr = torch.as_tensor(img, dtype=torch.float32)
        while arr.ndim < 4:
            arr = arr.unsqueeze(0)
        tensors.append(arr)
    return tensors


def _score_clean_scans(
    model: nn.Module,
    tensors: list[torch.Tensor],
    dev: torch.device,
) -> list[float]:
    """Score clean scan tensors using anomaly_score reconstruction error."""
    scores: list[float] = []
    for t in tensors:
        batch_scores = anomaly_score(model, t.to(dev)).detach().cpu().numpy()
        scores.extend(float(s) for s in np.atleast_1d(batch_scores))
    return scores


def _print_site_results(calib: SiteCalibration, fcr: float) -> None:
    """Print calibration summary, sample-count warning if low, and advisory notice."""
    typer.echo(f"n_clean: {calib.n_clean}")
    typer.echo(f"threshold: {calib.threshold:.6f}")
    typer.echo(f"upper95_fcr: {calib.upper95_fcr:.4f}")
    min_scans = min_clean_scans(fcr)
    if calib.n_clean < min_scans:
        typer.echo(
            f"WARNING: clean scans ({calib.n_clean}) below recommended minimum ({min_scans}) "
            f"for false call rate {fcr} at 95% confidence."
        )
    typer.echo(
        "Advisory only - this profile changes ML thresholds, never rules limits; "
        "accept/reject stays with the certified inspector."
    )


def _run_site_calibrate(opts: SiteCalibrateOptions) -> None:
    """Execute site adaptation calibration pipeline."""
    if opts.label not in {"Real", "Synthetic"}:
        typer.secho(
            f"Error: Invalid label {opts.label!r}. Must be 'Real' or 'Synthetic'.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)
    fcr = _resolve_site_fcr(opts.fcr)
    images = _load_clean_images(opts.clean_dir)
    stats = fit_site_stats(images, label=opts.label)
    norm_imgs = [apply_site_stats(img, stats) for img in images]
    dev = pick_device("auto")
    model = _load_site_model(opts.model, dev)
    tensors = _prepare_scan_tensors(norm_imgs)
    recalibrate_batchnorm(model, tensors, device=dev)
    scores = _score_clean_scans(model, tensors, dev)
    calib = calibrate_site_threshold(scores, false_call_rate=fcr)
    save_site_profile(opts.out, stats, calib, label=opts.label)
    _print_site_results(calib, fcr)


@app.command("site-calibrate")
def site_calibrate(
    clean_dir: Annotated[Path, typer.Option("--clean-dir", help="Folder of .npy clean scans")],
    out: Annotated[Path, typer.Option("--out", help="Profile JSON output path")],
    label: Annotated[str, typer.Option("--label", help="Data provenance ('Real' or 'Synthetic')")],
    model: Annotated[
        Path,
        typer.Option("--model", help="Path to 2D autoencoder checkpoint"),
    ] = Path("models/anomaly_2d.pt"),
    fcr: Annotated[
        float | None,
        typer.Option("--fcr", help="Target false call rate (default from detector config)"),
    ] = None,
) -> None:
    """Calibrate ML anomaly detection threshold on clean reference scans."""
    _enforce_locked()
    opts = SiteCalibrateOptions(
        clean_dir=clean_dir,
        out=out,
        label=label,
        model=model,
        fcr=fcr,
    )
    _run_site_calibrate(opts)
