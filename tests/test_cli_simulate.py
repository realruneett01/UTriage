"""Tests for the simulate CLI subcommand and figure generation (Synthetic data)."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure local worktree src is prioritized over any site-packages editable install
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        sys.modules["utriage"].__path__.insert(0, _pkg_dir)

import pytest  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from utriage.cli import app  # noqa: E402
from utriage.physics.dataset import DatasetConfig, generate_dataset, load_scan  # noqa: E402
from utriage.physics.figures import plot_ascan, plot_cscan  # noqa: E402


@pytest.fixture
def tiny_scan(tmp_path: Path):
    """Create a single simulated bar for testing figure generation."""
    data_dir = tmp_path / "seed_data"
    cfg = DatasetConfig(n_bars=1, length_mm=6.0, n_angle=8, diameter_mm=20.0, seed=42)
    npz_paths = generate_dataset(cfg, data_dir)
    return load_scan(npz_paths[0])


def test_simulate_cli_creates_expected_files(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["simulate", "--bars", "2", "--length-mm", "6", "--n-angle", "8", "--out", str(tmp_path)],
    )
    assert result.exit_code == 0
    assert (tmp_path / "bar_0000.npz").exists()
    assert (tmp_path / "bar_0001.npz").exists()
    assert (tmp_path / "bar_0000.json").exists()
    assert (tmp_path / "bar_0001.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "figures" / "cscan_0000.png").exists()
    assert (tmp_path / "figures" / "cscan_0001.png").exists()
    assert len(list(tmp_path.rglob("*.png"))) == 2
    assert "bar_0000.npz" in result.output
    assert "reflectors" in result.output
    assert f"Synthetic data written to {tmp_path}" in result.output


def test_simulate_cli_no_figures_flag(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "simulate",
            "--bars",
            "2",
            "--length-mm",
            "6",
            "--n-angle",
            "8",
            "--out",
            str(tmp_path),
            "--no-figures",
        ],
    )
    assert result.exit_code == 0
    assert (tmp_path / "bar_0000.npz").exists()
    assert (tmp_path / "bar_0001.npz").exists()
    assert len(list(tmp_path.rglob("*.png"))) == 0


def test_plot_ascan_writes_file(tmp_path: Path, tiny_scan) -> None:
    out_path = tmp_path / "ascan.png"
    result_path = plot_ascan(tiny_scan, 0, 0, out_path)
    assert result_path == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_cscan_writes_file(tmp_path: Path, tiny_scan) -> None:
    out_path = tmp_path / "cscan.png"
    result_path = plot_cscan(tiny_scan, out_path)
    assert result_path == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


EXPECTED_SIMULATE_HELP = (
    "                                                                               \n"
    " Usage: root simulate [OPTIONS]                                                \n"
    "                                                                               \n"
    " Generate synthetic bar UT scans with ground truth (Synthetic data).           \n"
    "                                                                               \n"
    "┌─ Options ───────────────────────────────────────────────────────────────────┐\n"
    "│ --material                     <str>    Material name [default: ti64]       │\n"
    "│ --probe                        <str>    Probe config [default: p5]          │\n"
    "│ --bars                         <int>    Number of bars to simulate          │\n"
    "│                                         [default: 5]                        │\n"
    "│ --seed                         <int>    Random seed [default: 0]            │\n"
    "│ --out                          <path>   Output directory                    │\n"
    "│ --length-mm                    <float>  Bar length in mm [default: 64.0]    │\n"
    "│ --n-angle                      <int>    Number of angular steps             │\n"
    "│                                         [default: 120]                      │\n"
    "│ --figures      --no-figures             Plot C-scans [default: figures]     │\n"
    "│ --help                                  Show this message and exit.         │\n"
    "└─────────────────────────────────────────────────────────────────────────────┘\n\n"
)

_EXPECTED_HELP_OPTIONS: tuple[tuple[str, str | None], ...] = (
    ("--material", "ti64"),
    ("--probe", "p5"),
    ("--bars", "5"),
    ("--seed", "0"),
    ("--out", None),
    ("--length-mm", "64.0"),
    ("--n-angle", "120"),
    ("--figures", "figures"),
    ("--no-figures", None),
    ("--help", None),
)


def test_simulate_help_output_matches_captured() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["simulate", "--help"])
    assert result.exit_code == 0
    assert result.output == EXPECTED_SIMULATE_HELP


@pytest.mark.parametrize(("opt_name", "opt_default"), _EXPECTED_HELP_OPTIONS)
def test_simulate_help_option_names_and_defaults(opt_name: str, opt_default: str | None) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["simulate", "--help"])
    assert result.exit_code == 0
    assert opt_name in result.output
    if opt_default is not None:
        assert f"default: {opt_default}" in result.output
