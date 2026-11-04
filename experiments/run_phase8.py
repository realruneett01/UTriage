"""Phase 8 reproduction and pipeline orchestration (project plan Phase 8).

Regenerates all Phase 8 experiments from scratch in sequence:
audit -> LOCO -> anomaly -> sim generalization -> cross-dataset -> hybrid ->
physics model -> false call analysis -> site adaptation -> joint train ->
diagnostics -> failures -> hardware -> data inventory -> generalization matrix ->
claims/resume bullets.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure src/ is importable
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from utriage.report.matrix import build_matrix, plot_matrix  # noqa: E402

EXPERIMENT_SCRIPTS: tuple[str, ...] = (
    "experiments/run_leakage_audit.py",
    "experiments/run_loco.py",
    "experiments/run_anomaly.py",
    "experiments/run_sim_generalization.py",
    "experiments/run_cross_dataset.py",
    "experiments/run_hybrid.py",
    "experiments/run_physics_model.py",
    "experiments/run_false_call_analysis.py",
    "experiments/run_site_adaptation.py",
    "experiments/run_joint_train.py",
    "experiments/run_fit_diagnostics.py",
    "experiments/run_failures.py",
    "experiments/run_hardware.py",
    "scripts/make_data_inventory.py",
)

POST_SCRIPTS: tuple[str, ...] = (
    "scripts/make_claims.py",
    "scripts/make_resume_bullets.py",
)

QUICK_CAPABLE: frozenset[str] = frozenset(
    {
        "run_loco.py",
        "run_anomaly.py",
        "run_sim_generalization.py",
        "run_cross_dataset.py",
        "run_hybrid.py",
        "run_physics_model.py",
        "run_false_call_analysis.py",
        "run_site_adaptation.py",
        "run_joint_train.py",
        "run_fit_diagnostics.py",
        "run_failures.py",
    }
)

CONFIG_CAPABLE: frozenset[str] = frozenset(
    {
        "run_loco.py",
        "run_anomaly.py",
        "run_hybrid.py",
        "run_physics_model.py",
        "run_false_call_analysis.py",
        "run_failures.py",
    }
)


@dataclass(frozen=True)
class Phase8Config:
    """Configuration options for Phase 8 execution."""

    assemble_only: bool = False
    quick: bool = False
    skip: tuple[str, ...] = ()
    results_dir: Path = Path("results")
    timeout_seconds: int = 1800
    config: Path | None = None
    set: tuple[str, ...] = ()


def _is_skipped(script: str, skip_list: tuple[str, ...]) -> bool:
    """Check if script path, filename, or stem matches skip entries."""
    path = Path(script)
    return any(s in (script, path.name, path.stem) for s in skip_list)


def _append_config_args(cmd: list[str], cfg: Phase8Config) -> None:
    """Append config and set override flags to command."""
    if cfg.config is not None:
        cmd.extend(["--config", str(cfg.config)])
    for override in cfg.set or ():
        cmd.extend(["--set", override])


def _build_cmd(script: str, cfg: Phase8Config) -> list[str]:
    """Form executable command line with optional arguments."""
    cmd = [sys.executable, script]
    name = Path(script).name
    if cfg.quick and name in QUICK_CAPABLE:
        cmd.append("--quick")
    if name in CONFIG_CAPABLE:
        _append_config_args(cmd, cfg)
    return cmd


def _run_single_script(script: str, cfg: Phase8Config) -> None:
    """Execute a single pipeline python script with subprocess."""
    if _is_skipped(script, cfg.skip):
        print(f"Skipping {script}", flush=True)
        return
    if not Path(script).is_file():
        print(f"Skipping {script} (file does not exist yet)", flush=True)
        return
    cmd = _build_cmd(script, cfg)
    print(f"Running {' '.join(cmd)}...", flush=True)
    subprocess.run(cmd, check=True, timeout=cfg.timeout_seconds)  # noqa: S603


def _run_experiment_stage(cfg: Phase8Config) -> None:
    """Run all primary experiment generation scripts in order."""
    if cfg.assemble_only:
        print("Assemble-only mode: skipping experiment re-runs.", flush=True)
        return
    for script in EXPERIMENT_SCRIPTS:
        _run_single_script(script, cfg)


def _run_post_stage(cfg: Phase8Config) -> None:
    """Execute available post-matrix reporting and resume bullet scripts."""
    for script in POST_SCRIPTS:
        p = Path(script)
        if not p.is_file():
            continue
        if _is_skipped(script, cfg.skip):
            print(f"Skipping {script}", flush=True)
            continue
        print(f"Running {script}...", flush=True)
        subprocess.run(  # noqa: S603
            [sys.executable, script], check=True, timeout=cfg.timeout_seconds
        )


def run_phase8(cfg: Phase8Config) -> dict[str, Path]:
    """Execute Phase 8 workflow and generate generalization matrix artifacts."""
    _run_experiment_stage(cfg)

    print("Building generalization matrix...", flush=True)
    matrix = build_matrix(cfg.results_dir)

    out_json = cfg.results_dir / "generalization_matrix.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(matrix, fh, indent=2)
    print(f"Matrix written to {out_json}", flush=True)

    out_png = cfg.results_dir / "figures" / "generalization_matrix.png"
    plot_matrix(matrix, out_png)
    print(f"Matrix heatmap written to {out_png}", flush=True)

    _run_post_stage(cfg)
    return {"matrix_json": out_json, "matrix_png": out_png}


def _add_cli_flags(parser: argparse.ArgumentParser) -> None:
    """Register boolean flags and repeatable skip options on parser."""
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="Skip experiment scripts and build matrix only",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Pass --quick to scripts that accept it",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Skip specific script by name or stem (repeatable)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to detector config YAML",
    )
    parser.add_argument(
        "--set",
        dest="set",
        action="append",
        default=[],
        help="Config override KEY=VALUE (repeatable)",
    )


def _build_cli_parser() -> argparse.ArgumentParser:
    """Configure command line argument parser for Phase 8 runner."""
    parser = argparse.ArgumentParser(description="Run Phase 8 end-to-end pipeline")
    _add_cli_flags(parser)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results directory path (default: results)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout in seconds per experiment script",
    )
    return parser


def parse_args(args: list[str] | None = None) -> Phase8Config:
    """Parse command line arguments and return Phase8Config."""
    parser = _build_cli_parser()
    parsed = parser.parse_args(args)
    return Phase8Config(
        assemble_only=parsed.assemble_only,
        quick=parsed.quick,
        skip=tuple(parsed.skip),
        results_dir=parsed.results_dir,
        timeout_seconds=parsed.timeout,
        config=parsed.config,
        set=tuple(parsed.set) if parsed.set else (),
    )


if __name__ == "__main__":
    config = parse_args()
    run_phase8(config)
