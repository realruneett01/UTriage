"""Tests checking that the inspection path makes no network calls."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from utriage.physics.dataset import save_scan  # noqa: E402
from utriage.physics.simulator import (  # noqa: E402
    BarSpec,
    Reflector,
    load_material,
    load_probe,
    simulate_bar,
)
from utriage.physics.ti_noise import (  # noqa: E402
    load_ti_noise,
    simulate_ti_bar,
)
from utriage.workflow import TriageJob, run_batch, run_triage  # noqa: E402


@pytest.fixture(autouse=True)
def guard_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block socket network calls across all tests in this module."""

    def _block(*args: object, **kwargs: object) -> None:
        raise RuntimeError("network call attempted")

    monkeypatch.setattr(socket.socket, "connect", _block)
    monkeypatch.setattr(socket.socket, "connect_ex", _block)
    monkeypatch.setattr(socket, "create_connection", _block)
    monkeypatch.setattr(socket, "getaddrinfo", _block)


def _make_small_spec() -> BarSpec:
    """Build a compact bar spec for testing."""
    material = load_material("ti64")
    probe = load_probe("p5")
    water = load_material("water")
    reflector = Reflector(
        axial_mm=8.0,
        angle_deg=0.0,
        depth_mm=10.0,
        diameter_mm=0.8,
    )
    return BarSpec(
        material=material,
        probe=probe,
        water=water,
        diameter_mm=40.0,
        length_mm=16.0,
        axial_step_mm=1.0,
        n_angle=24,
        reflectors=(reflector,),
        seed=42,
    )


def test_network_guard_sanity() -> None:
    """Check that socket network calls raise RuntimeError under the guard."""
    with pytest.raises(RuntimeError, match="network call attempted"):
        socket.create_connection(("example.com", 80))


def test_simulate_and_triage_offline(tmp_path: Path) -> None:
    """Check that bar simulation and triage run without network calls."""
    spec = _make_small_spec()
    scan = simulate_bar(spec)
    npz_path, _ = save_scan(scan, tmp_path / "small_bar")

    out_dir = tmp_path / "out"
    job = TriageJob(scan=npz_path, out_dir=out_dir, model=None)
    summary = run_triage(job)

    html_path = Path(summary["report_path"])
    assert html_path.is_file()
    html_content = html_path.read_text(encoding="utf-8")
    assert "http://" not in html_content
    assert "https://" not in html_content

    batch_out = tmp_path / "batch_out"
    batch_results = run_batch([npz_path], out_dir=batch_out, model=None)
    assert len(batch_results) == 1
    assert batch_results[0]["error"] is None


def test_simulate_ti_bar_offline() -> None:
    """Check that simulate_ti_bar runs without network calls."""
    spec = _make_small_spec()
    cfg = load_ti_noise()
    ti_scan = simulate_ti_bar(spec, cfg)
    assert ti_scan.ascans.shape[:2] == (16, 24)
    assert not np.isnan(ti_scan.ascans).any()


def test_import_modules_offline() -> None:
    """Check that importing cli and workflow makes no network calls."""
    import utriage.cli
    import utriage.workflow

    assert hasattr(utriage.cli, "app")
    assert hasattr(utriage.workflow, "run_triage")
