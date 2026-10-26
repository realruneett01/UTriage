"""Tests for bar-domain ML inference, MC-dropout uncertainty, and NMS."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        sys.modules["utriage"].__path__.insert(0, _pkg_dir)
if "utriage.ml" in sys.modules:
    _ml_dir = str(Path(_SRC) / "utriage" / "ml")
    if _ml_dir not in sys.modules["utriage.ml"].__path__:
        sys.modules["utriage.ml"].__path__.insert(0, _ml_dir)

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from utriage.ml.bar_infer import (  # noqa: E402
    ScanSettings,
    findings_to_dicts,
    load_window_model,
    save_window_model,
    scan_bar,
)
from utriage.ml.bar_models import WindowNet  # noqa: E402
from utriage.physics.simulator import (  # noqa: E402
    BarScan,
    BarSpec,
    load_material,
    load_probe,
    simulate_bar,
)
from utriage.triage.fusion import MlFinding  # noqa: E402


@pytest.fixture
def tiny_scan() -> BarScan:
    """Fixture providing a tiny simulated bar scan (diameter 20, length 4, 8 angles)."""
    mat = load_material("ti64")
    probe = load_probe("p5")
    water = load_material("water")
    spec = BarSpec(
        material=mat,
        probe=probe,
        water=water,
        diameter_mm=20.0,
        length_mm=4.0,
        axial_step_mm=1.0,
        n_angle=8,
        seed=42,
    )
    return simulate_bar(spec)


@pytest.fixture
def tiny_model() -> WindowNet:
    """Fixture providing an untrained WindowNet of width 4."""
    torch.manual_seed(42)
    return WindowNet(in_channels=1, width=4, n_classes=3, dropout=0.2)


@pytest.fixture
def patch_model() -> WindowNet:
    """Fixture providing an untrained WindowNet of width 4 with 9 input channels."""
    return WindowNet(in_channels=9, width=4)


def _assert_uncertainties_bounded(findings: list[MlFinding]) -> None:
    """Assert non-empty findings and all uncertainties fall in [0.0, 1.0]."""
    assert len(findings) > 0
    for f in findings:
        assert 0.0 <= f.uncertainty <= 1.0


def _assert_outputs_close(
    out1: torch.Tensor,
    sz1: torch.Tensor,
    out2: torch.Tensor,
    sz2: torch.Tensor,
) -> None:
    """Assert two sets of model outputs are elementwise close."""
    assert torch.allclose(out1, out2)
    assert torch.allclose(sz1, sz2)


def test_scan_bar_findings_grid_and_gate(tiny_model: WindowNet, tiny_scan: BarScan) -> None:
    settings = ScanSettings(threshold=0.0, depth_step_mm=1.0, mc_passes=4)
    findings = scan_bar(tiny_model, tiny_scan, settings, device="cpu")
    _assert_uncertainties_bounded(findings)
    max_d = tiny_scan.spec.diameter_mm - settings.back_margin_mm
    for f in findings:
        assert 0 <= f.axial_idx < tiny_scan.ascans.shape[0]
        assert 0 <= f.angle_idx < tiny_scan.ascans.shape[1]
        assert settings.front_mm <= f.depth_mm <= max_d + 1e-6
        assert 0.0 <= f.score <= 1.0
        assert f.grain_noise_prob == pytest.approx(1.0 - f.score, abs=1e-5)
        assert len(f.evidence) == 2
        assert f.evidence[0].startswith("ML window score")
        assert f.evidence[1].startswith("MC-dropout std")


@pytest.mark.parametrize(
    "settings",
    [
        ScanSettings(threshold=1.1),
        ScanSettings(front_mm=25.0, back_margin_mm=2.0),
    ],
    ids=["threshold_above_1", "gate_outside_bar"],
)
def test_scan_bar_empty(tiny_model: WindowNet, tiny_scan: BarScan, settings: ScanSettings) -> None:
    findings = scan_bar(tiny_model, tiny_scan, settings, device="cpu")
    assert findings == []


def test_scan_bar_mc_uncertainty_range(tiny_model: WindowNet, tiny_scan: BarScan) -> None:
    settings = ScanSettings(threshold=0.0, mc_passes=6)
    findings = scan_bar(tiny_model, tiny_scan, settings, device="cpu")
    _assert_uncertainties_bounded(findings)


def test_scan_bar_batchnorm_stats_unchanged(tiny_model: WindowNet, tiny_scan: BarScan) -> None:
    bn_modules = [m for m in tiny_model.modules() if isinstance(m, nn.BatchNorm1d)]
    means_before = [m.running_mean.clone() for m in bn_modules]
    vars_before = [m.running_var.clone() for m in bn_modules]
    scan_bar(tiny_model, tiny_scan, ScanSettings(threshold=0.0), device="cpu")
    for m, m_bef, v_bef in zip(bn_modules, means_before, vars_before, strict=True):
        assert torch.equal(m.running_mean, m_bef)
        assert torch.equal(m.running_var, v_bef)


def test_load_window_model_roundtrip(tmp_path: Path, tiny_model: WindowNet) -> None:
    path = tmp_path / "model.pt"
    meta = {"in_channels": 1, "width": 4, "dropout": 0.2, "n_classes": 3}
    save_window_model(tiny_model, path, meta)
    loaded_model, loaded_meta = load_window_model(path, device="cpu")
    assert loaded_meta["width"] == 4
    assert loaded_meta["in_channels"] == 1
    assert isinstance(loaded_model, WindowNet)
    x = torch.randn(2, 1, 128)
    tiny_model.eval()
    loaded_model.eval()
    with torch.no_grad():
        out1, sz1 = tiny_model(x)
        out2, sz2 = loaded_model(x)
    _assert_outputs_close(out1, sz1, out2, sz2)


def test_scan_bar_patch_channels(patch_model: WindowNet, tiny_scan: BarScan) -> None:
    findings = scan_bar(patch_model, tiny_scan, ScanSettings(threshold=0.0), device="cpu")
    assert isinstance(findings, list)


def test_findings_to_dicts() -> None:
    findings = [
        MlFinding(
            axial_idx=2,
            angle_idx=3,
            depth_mm=5.5,
            score=0.85,
            uncertainty=0.12,
            grain_noise_prob=0.15,
            evidence=("ML window score 0.85", "MC-dropout std 0.06"),
        )
    ]
    dicts = findings_to_dicts(findings)
    assert len(dicts) == 1
    d = dicts[0]
    assert d["axial_idx"] == 2
    assert d["angle_idx"] == 3
    assert d["depth_mm"] == pytest.approx(5.5)
    assert d["score"] == pytest.approx(0.85)
    assert d["uncertainty"] == pytest.approx(0.12)
    assert d["grain_noise_prob"] == pytest.approx(0.15)
    assert d["evidence"] == ["ML window score 0.85", "MC-dropout std 0.06"]
