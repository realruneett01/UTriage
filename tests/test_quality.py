"""Tests for the data-quality watchdog (project plan Task 2.2)."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src of the current worktree is prioritized over any editable install
_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from utriage.physics.simulator import (  # noqa: E402
    ARTEFACT_KINDS,
    Artefact,
    BarSpec,
    Probe,
    load_material,
    load_probe,
    simulate_bar,
)
from utriage.rules.config import AcceptanceConfig, load_acceptance  # noqa: E402
from utriage.rules.quality import check_quality  # noqa: E402


def _make_spec(
    artefacts: tuple[Artefact, ...] = (),
    probe_override: Probe | None = None,
) -> BarSpec:
    return BarSpec(
        material=load_material("ti64"),
        probe=probe_override or load_probe("p5"),
        water=load_material("water"),
        diameter_mm=30.0,
        length_mm=24.0,
        axial_step_mm=1.0,
        n_angle=36,
        artefacts=artefacts,
        seed=42,
    )


@pytest.fixture
def test_cfg() -> AcceptanceConfig:
    base = load_acceptance()
    quality = base.quality.model_copy(update={"noise_limit_ratio": 1.5})
    return base.model_copy(update={"quality": quality})


def _overlaps(r1: tuple[float, float], r2: tuple[float, float]) -> bool:
    return max(r1[0], r2[0]) <= min(r1[1], r2[1])


@pytest.mark.parametrize("use_default", [True, False])
def test_clean_bar_has_no_faults(use_default: bool, test_cfg: AcceptanceConfig) -> None:
    cfg = load_acceptance() if use_default else test_cfg
    scan = simulate_bar(_make_spec())
    report = check_quality(scan, cfg)
    assert not report.faults
    assert not np.any(report.fault_mask)
    assert not report.rescan_recommended


@pytest.mark.parametrize("kind", ARTEFACT_KINDS)
def test_planted_artefact_detected(kind: str, test_cfg: AcceptanceConfig) -> None:
    ax_region, ang_region = (8.0, 16.0), (60.0, 160.0)
    art = Artefact(kind=kind, axial_mm=ax_region, angle_deg=ang_region, severity=1.0)
    scan = simulate_bar(_make_spec(artefacts=(art,)))
    report = check_quality(scan, test_cfg)

    has_overlap = any(
        _overlaps(f.axial_mm, ax_region) and _overlaps(f.angle_deg, ang_region)
        for f in report.faults
    )
    assert has_overlap

    planted = (
        (scan.axial_mm[:, None] >= ax_region[0])
        & (scan.axial_mm[:, None] <= ax_region[1])
        & (scan.angle_deg[None, :] >= ang_region[0])
        & (scan.angle_deg[None, :] <= ang_region[1])
    )
    ratio = float(np.sum(report.fault_mask & planted) / np.sum(planted))
    assert ratio >= 0.5


def test_high_gain_reports_saturation(test_cfg: AcceptanceConfig) -> None:
    base_probe = load_probe("p5")
    high_gain = Probe(**{**base_probe.__dict__, "gain_db": 40.0})
    scan = simulate_bar(_make_spec(probe_override=high_gain))
    report = check_quality(scan, test_cfg)
    assert any(f.kind == "saturation" for f in report.faults)
    assert np.any(report.masks["saturation"])
    assert report.rescan_recommended
