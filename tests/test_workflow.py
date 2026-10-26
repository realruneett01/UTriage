"""Tests for shop-floor triage workflow and CLI integration (project plan 6.2)."""

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

import pytest  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from utriage.cli import app  # noqa: E402
from utriage.workflow import TriageJob, run_batch, run_triage  # noqa: E402

_SAMPLE_SCAN = Path(__file__).resolve().parents[1] / "data" / "samples" / "ti64" / "bar_0000.npz"


@pytest.fixture
def sample_scan() -> Path:
    """Return path to committed sample bar scan."""
    assert _SAMPLE_SCAN.is_file(), f"Sample scan not found: {_SAMPLE_SCAN}"
    return _SAMPLE_SCAN


@pytest.fixture
def corrupt_scan(tmp_path: Path) -> Path:
    """Create a corrupt .npz file for failure testing."""
    bad_file = tmp_path / "corrupt_bar.npz"
    bad_file.write_bytes(b"not a valid npz header")
    return bad_file


def test_run_triage_outputs(sample_scan: Path, tmp_path: Path) -> None:
    """Verify run_triage creates html report and json triage with summary keys."""
    out_dir = tmp_path / "out"
    job = TriageJob(scan=sample_scan, out_dir=out_dir)
    summary = run_triage(job)

    expected_keys = {
        "counts_per_category",
        "n_ml_flag",
        "rescan_recommended",
        "report_path",
        "json_path",
    }
    assert expected_keys.issubset(summary.keys())
    assert summary["error"] is None

    html_file = Path(summary["report_path"])
    json_file = Path(summary["json_path"])
    assert html_file.is_file()
    assert json_file.is_file()

    with json_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    assert data["model_loaded"] is False
    assert data["synthetic"] is True
    assert "Advisory tool" in data["advisory_banner"]
    assert isinstance(data["items"], list)
    assert len(data["items"]) > 0


def test_run_batch_error_resilience(
    sample_scan: Path,
    corrupt_scan: Path,
    tmp_path: Path,
) -> None:
    """Verify run_batch continues after error and records error in summary."""
    out_dir = tmp_path / "batch_out"
    scans = [sample_scan, corrupt_scan]
    results = run_batch(scans=scans, out_dir=out_dir, model=None)

    assert len(results) == 2
    good_res, bad_res = results[0], results[1]

    assert good_res["error"] is None
    assert Path(good_res["report_path"]).is_file()

    assert bad_res["error"] is not None
    assert bad_res["report_path"] is None
    assert bad_res["bar_id"] == "corrupt_bar"


@pytest.mark.parametrize(
    ("args_builder", "expected_code", "expect_error"),
    [
        (lambda good, bad, out: ["triage", str(good), "--out", str(out)], 0, False),
        (
            lambda good, bad, out: ["triage", str(good), str(bad), "--out", str(out)],
            1,
            True,
        ),
    ],
)
def test_cli_triage_exit_codes(
    sample_scan: Path,
    corrupt_scan: Path,
    tmp_path: Path,
    args_builder: callable,
    expected_code: int,
    expect_error: bool,
) -> None:
    """Verify CLI triage command exit codes 0 and 1, banners and summaries."""
    runner = CliRunner()
    cli_out = tmp_path / f"cli_out_{expected_code}"
    args = args_builder(sample_scan, corrupt_scan, cli_out)
    result = runner.invoke(app, args)

    assert result.exit_code == expected_code
    assert "Advisory tool - decisions by certified inspector" in result.output
    assert "bar_0000" in result.output
    if expect_error:
        assert "ERROR" in result.output
