"""Tests for site-calibrate CLI command (project plan Phase 0 / ADR 0010)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        sys.modules["utriage"].__path__.insert(0, _pkg_dir)
    sys.modules.pop("utriage.cli", None)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from typer.testing import CliRunner, Result  # noqa: E402

from utriage.cli import app  # noqa: E402
from utriage.ml.anomaly import ConvAE2D  # noqa: E402


@pytest.fixture
def site_env(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create dummy model checkpoint, clean scan directory, and output path."""
    model_path = tmp_path / "model.pt"
    torch.manual_seed(42)
    model = ConvAE2D()
    torch.save(model.state_dict(), model_path)

    clean_dir = tmp_path / "clean_scans"
    clean_dir.mkdir()
    rng = np.random.default_rng(42)
    for i in range(3):
        scan_data = rng.normal(0, 1, (32, 32)).astype(np.float32)
        np.save(clean_dir / f"scan_{i:02d}.npy", scan_data)

    out_json = tmp_path / "profile.json"
    return clean_dir, model_path, out_json


def _run_cli(args: list[str]) -> Result:
    """Invoke Typer CLI runner with the provided argument list."""
    runner = CliRunner()
    return runner.invoke(app, args)


def test_site_calibrate_success(site_env: tuple[Path, Path, Path]) -> None:
    """site-calibrate generates profile with threshold and warning for few scans."""
    clean_dir, model_path, out_json = site_env
    res = _run_cli(
        [
            "site-calibrate",
            "--clean-dir",
            str(clean_dir),
            "--model",
            str(model_path),
            "--label",
            "Synthetic",
            "--out",
            str(out_json),
        ]
    )
    assert res.exit_code == 0
    assert out_json.is_file()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "threshold" in data["calib"]
    assert data["label"] == "Synthetic"
    assert "WARNING:" in res.output
    assert "n_clean: 3" in res.output
    assert "threshold:" in res.output
    assert "upper95_fcr:" in res.output
    assert "Advisory only" in res.output


@pytest.mark.parametrize("bad_label", ["Invalid", "unknown", "real", "SYNTHETIC"])
def test_site_calibrate_bad_label(site_env: tuple[Path, Path, Path], bad_label: str) -> None:
    """Invalid data label causes command to exit with non-zero status."""
    clean_dir, model_path, out_json = site_env
    res = _run_cli(
        [
            "site-calibrate",
            "--clean-dir",
            str(clean_dir),
            "--model",
            str(model_path),
            "--label",
            bad_label,
            "--out",
            str(out_json),
        ]
    )
    assert res.exit_code != 0


def test_site_calibrate_custom_fcr(site_env: tuple[Path, Path, Path]) -> None:
    """site-calibrate respects explicit --fcr and Real label."""
    clean_dir, model_path, out_json = site_env
    res = _run_cli(
        [
            "site-calibrate",
            "--clean-dir",
            str(clean_dir),
            "--model",
            str(model_path),
            "--label",
            "Real",
            "--fcr",
            "0.05",
            "--out",
            str(out_json),
        ]
    )
    assert res.exit_code == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["label"] == "Real"
    assert data["calib"]["false_call_rate"] == 0.05


def test_site_calibrate_missing_dir(site_env: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """Non-existent clean-dir causes command to exit non-zero."""
    _, model_path, out_json = site_env
    res = _run_cli(
        [
            "site-calibrate",
            "--clean-dir",
            str(tmp_path / "missing_dir"),
            "--model",
            str(model_path),
            "--label",
            "Synthetic",
            "--out",
            str(out_json),
        ]
    )
    assert res.exit_code != 0
