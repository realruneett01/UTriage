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
