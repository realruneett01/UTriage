"""Tests for titanium mode inspection CLI subcommand (Synthetic data)."""

from __future__ import annotations

import dataclasses
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

import pytest  # noqa: E402
from typer.testing import CliRunner, Result  # noqa: E402

from utriage import cli  # noqa: E402
from utriage.cli import app  # noqa: E402
from utriage.physics.dataset import DatasetConfig, generate_dataset  # noqa: E402
from utriage.physics.simulator import BarSpec  # noqa: E402


@pytest.fixture(autouse=True)
def fast_demo_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reduce demo bar dimensions so simulation executes in sub-second time."""
    if hasattr(cli, "_demo_spec"):
        orig_func = cli._demo_spec

        def _fast(seed: int) -> BarSpec:
            base = orig_func(seed)
            return dataclasses.replace(base, length_mm=16.0, n_angle=24)

        monkeypatch.setattr(cli, "_demo_spec", _fast)


@pytest.fixture
def small_scan(tmp_path: Path) -> Path:
    """Create a small simulated bar scan for test execution."""
    data_dir = tmp_path / "raw_data"
    cfg = DatasetConfig(n_bars=1, length_mm=16.0, n_angle=24, diameter_mm=20.0, seed=42)
    paths = generate_dataset(cfg, data_dir)
    return paths[0]


def _run_cmd(args: list[str]) -> Result:
    """Execute CLI runner with specified argument list."""
    runner = CliRunner()
    return runner.invoke(app, args)


def _check_output_clean(text: str) -> None:
    """Check that output avoids evaluation decision wording."""
    lowered = text.lower()
    assert "accept" not in lowered
    assert "pass" not in lowered
    assert "reject" not in lowered


@pytest.mark.parametrize(
    "options",
    [
        [],
        ["--scan", "data/test.npz", "--demo-seed", "42"],
    ],
)
def test_ti_inspect_mutual_exclusion(options: list[str]) -> None:
    """Exit code 2 when neither or both input options are provided."""
    res = _run_cmd(["ti-inspect", *options])
    assert res.exit_code == 2
    assert "exactly one" in res.output.lower()


def test_ti_inspect_demo_outputs(tmp_path: Path) -> None:
    """Demo run creates JSON and HTML report files with metadata."""
    out_dir = tmp_path / "ti_demo"
    res = _run_cmd(["ti-inspect", "--demo-seed", "0", "--out", str(out_dir)])
    assert res.exit_code == 0
    json_path = out_dir / "demo_0_ti.json"
    html_path = out_dir / "demo_0_ti_report.html"
    assert json_path.is_file()
    assert html_path.is_file()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    calib_id = data["calibration"]["calib_id"]
    assert calib_id
    assert data.get("label") == "Synthetic data"

    html_content = html_path.read_text(encoding="utf-8")
    assert calib_id in html_content
    assert calib_id in res.output
    assert "Zone alarms:" in res.output
    assert "Zones not inspectable:" in res.output


def test_ti_inspect_scan_success(tmp_path: Path, small_scan: Path) -> None:
    """Running ti-inspect on saved scan file produces expected artifacts."""
    out_dir = tmp_path / "ti_scan"
    res = _run_cmd(["ti-inspect", "--scan", str(small_scan), "--out", str(out_dir)])
    assert res.exit_code == 0
    bar_stem = small_scan.stem
    json_path = out_dir / f"{bar_stem}_ti.json"
    html_path = out_dir / f"{bar_stem}_ti_report.html"
    assert json_path.is_file()
    assert html_path.is_file()
    assert "Report:" in res.output


@pytest.mark.parametrize(
    "mode",
    ["demo", "scan"],
)
def test_ti_inspect_no_forbidden_words(
    tmp_path: Path,
    small_scan: Path,
    mode: str,
) -> None:
    """CLI output contains no accept or pass wording."""
    out_dir = tmp_path / f"ti_{mode}"
    args = (
        ["ti-inspect", "--demo-seed", "1", "--out", str(out_dir)]
        if mode == "demo"
        else ["ti-inspect", "--scan", str(small_scan), "--out", str(out_dir)]
    )
    res = _run_cmd(args)
    assert res.exit_code == 0
    _check_output_clean(res.output)


@pytest.mark.parametrize(
    "bad_id",
    ["../evil", "a/b", "a\\b", "", "C:x"],
)
def test_ti_inspect_rejects_unsafe_bar_id(tmp_path: Path, bad_id: str) -> None:
    """Reject traversal and malformed bar identifiers with exit code 2."""
    out_dir = tmp_path / "ti_out"
    res = _run_cmd(["ti-inspect", "--demo-seed", "0", "--bar-id", bad_id, "--out", str(out_dir)])
    assert res.exit_code == 2
    assert "Error: invalid --bar-id" in res.output
    outside = [p for p in tmp_path.rglob("*") if p.is_file() and not p.is_relative_to(out_dir)]
    assert not outside


def test_ti_inspect_safe_bar_id_success(tmp_path: Path) -> None:
    """Safe bar identifier writes expected output files."""
    out_dir = tmp_path / "ti_safe"
    res = _run_cmd(
        ["ti-inspect", "--demo-seed", "0", "--bar-id", "bar-01.A_2", "--out", str(out_dir)]
    )
    assert res.exit_code == 0
    assert (out_dir / "bar-01.A_2_ti.json").is_file()
    assert (out_dir / "bar-01.A_2_ti_report.html").is_file()
