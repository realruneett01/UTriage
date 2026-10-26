"""Tests for the rules engine evaluation pipeline and CLI (project plan 2.5)."""

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

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from utriage.cli import app  # noqa: E402
from utriage.physics.dataset import save_scan  # noqa: E402
from utriage.physics.simulator import (  # noqa: E402
    Artefact,
    BarScan,
    BarSpec,
    Reflector,
    load_material,
    load_probe,
    simulate_bar,
)
from utriage.rules.config import load_acceptance  # noqa: E402
from utriage.rules.pipeline import (  # noqa: E402
    dac_for,
    evaluate_scan,
    evaluation_to_dict,
)


def _make_spec(
    reflectors: tuple[Reflector, ...] = (),
    artefacts: tuple[Artefact, ...] = (),
) -> BarSpec:
    return BarSpec(
        material=load_material("ti64"),
        probe=load_probe("p5"),
        water=load_material("water"),
        diameter_mm=30.0,
        length_mm=12.0,
        axial_step_mm=1.0,
        n_angle=24,
        reflectors=reflectors,
        artefacts=artefacts,
        seed=42,
    )


@pytest.fixture
def base_scan() -> BarScan:
    return simulate_bar(_make_spec())


def test_dac_for_produces_callable_dac(base_scan: BarScan) -> None:
    dac = dac_for(base_scan)
    depths = np.array([5.0, 10.0, 15.0])
    amps = dac(depths)
    assert len(amps) == 3
    assert np.all(amps > 0.0)


def test_fbh_indication_is_rejectable_or_reportable() -> None:
    ref = Reflector(
        axial_mm=6.0,
        angle_deg=60.0,
        depth_mm=26.0,
        diameter_mm=1.2,
        kind="fbh",
        amplitude_factor=1.0,
    )
    scan = simulate_bar(_make_spec(reflectors=(ref,)))
    cfg = load_acceptance()
    ev = evaluate_scan(scan, cfg)
    assert len(ev.indications) >= 1
    flagged = any(ind.category in ("rejectable", "reportable") for ind in ev.indications)
    assert flagged
    assert ev.summary["rejectable"] + ev.summary["reportable"] >= 1


def test_lost_backwall_artefact_recommends_rescan() -> None:
    art = Artefact(
        kind="lost_backwall",
        axial_mm=(3.0, 9.0),
        angle_deg=(30.0, 150.0),
        severity=1.0,
    )
    scan = simulate_bar(_make_spec(artefacts=(art,)))
    cfg = load_acceptance()
    ev = evaluate_scan(scan, cfg)
    assert ev.quality.rescan_recommended is True
    assert any(f.kind == "backwall_loss" for f in ev.quality.faults)


def test_evaluation_to_dict_is_json_dumpable(base_scan: BarScan) -> None:
    cfg = load_acceptance()
    ev = evaluate_scan(base_scan, cfg)
    ev_dict = evaluation_to_dict(ev)
    dumped = json.dumps(ev_dict)
    loaded = json.loads(dumped)
    assert "indications" in loaded
    assert "quality" in loaded
    assert "summary" in loaded
    assert "placeholder_limits" in loaded


def test_cli_evaluate_on_saved_scan(tmp_path: Path, base_scan: BarScan) -> None:
    npz_path, _ = save_scan(base_scan, tmp_path / "bar_sample")
    json_out = tmp_path / "out.json"
    runner = CliRunner()
    result = runner.invoke(app, ["evaluate", str(npz_path), "--json-out", str(json_out)])
    assert result.exit_code == 0
    assert "Advisory tool - decisions by certified inspector" in result.output
    assert json_out.exists()
    with json_out.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert "indications" in data
    assert "quality" in data
