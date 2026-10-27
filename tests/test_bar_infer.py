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

from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from utriage.ml.bar_infer import (  # noqa: E402
    ScanSettings,
    _Candidate,
    _InferStats,
    _suppress_candidates,
    calibrate_bar_threshold,
    candidate_scores,
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


@pytest.mark.parametrize(
    ("max_flags", "expected_threshold"),
    [
        (0.5, 0.9),
        (1.0, 0.8),
        (1.5, 0.7),
        (2.0, 0.6),
    ],
)
def test_calibrate_bar_threshold_quantiles(
    monkeypatch: pytest.MonkeyPatch,
    tiny_scan: BarScan,
    max_flags: float,
    expected_threshold: float,
) -> None:
    scores_map = {
        0: np.array([0.9, 0.7, 0.5]),
        1: np.array([0.8, 0.6, 0.4]),
    }
    call_count = 0

    def mock_candidate_scores(
        model: Any,
        scan: Any,
        settings: Any = None,
        device: str = "auto",
    ) -> np.ndarray:
        nonlocal call_count
        res = scores_map.get(call_count, np.zeros(0))
        call_count += 1
        return res

    monkeypatch.setattr("utriage.ml.bar_infer.candidate_scores", mock_candidate_scores)
    model = WindowNet(in_channels=1, width=4)
    thresh = calibrate_bar_threshold(
        model,
        [tiny_scan, tiny_scan],
        max_flags_per_bar=max_flags,
    )
    assert thresh == pytest.approx(expected_threshold)


@pytest.mark.parametrize("threshold", [0.0, 0.2, 0.5, 0.8])
def test_candidate_scores_count_gte_scan_bar(
    tiny_model: WindowNet,
    tiny_scan: BarScan,
    threshold: float,
) -> None:
    settings = ScanSettings(threshold=threshold, depth_step_mm=1.0, mc_passes=1)
    torch.manual_seed(42)
    scores = candidate_scores(tiny_model, tiny_scan, settings, device="cpu")
    torch.manual_seed(42)
    findings = scan_bar(tiny_model, tiny_scan, settings, device="cpu")
    assert len(scores) >= len(findings)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_blob_3x3_yields_single_finding() -> None:
    """A 3x3 blob of connected neighbouring candidates collapses to exactly one finding."""
    cands = [
        _Candidate(
            axial_idx=ax,
            angle_idx=an,
            depth_mm=5.0,
            score=0.9 if (ax == 1 and an == 1) else 0.8,
            logit=2.0 if (ax == 1 and an == 1) else 1.0,
            uncertainty=0.1,
            std=0.05,
        )
        for ax in range(3)
        for an in range(3)
    ]
    kept = _suppress_candidates(cands, n_angle=8)
    assert len(kept) == 1
    assert kept[0].axial_idx == 1
    assert kept[0].angle_idx == 1


def test_two_separated_blobs_yield_two_findings() -> None:
    """Two spatially separated blobs yield exactly two findings."""
    blob1 = [
        _Candidate(
            axial_idx=ax,
            angle_idx=an,
            depth_mm=5.0,
            score=0.9 if (ax == 0 and an == 0) else 0.8,
            logit=2.0 if (ax == 0 and an == 0) else 1.0,
            uncertainty=0.1,
            std=0.05,
        )
        for ax in range(2)
        for an in range(2)
    ]
    blob2 = [
        _Candidate(
            axial_idx=ax,
            angle_idx=an,
            depth_mm=5.0,
            score=0.85 if (ax == 10 and an == 10) else 0.7,
            logit=1.8 if (ax == 10 and an == 10) else 0.9,
            uncertainty=0.1,
            std=0.05,
        )
        for ax in range(10, 12)
        for an in range(10, 12)
    ]
    kept = _suppress_candidates(blob1 + blob2, n_angle=20)
    assert len(kept) == 2
    assert {c.axial_idx for c in kept} == {0, 10}


def test_candidate_count_invariant_to_batch(tiny_scan: BarScan) -> None:
    """Candidate count and findings are invariant to settings.batch."""
    model = WindowNet(in_channels=1, width=4, dropout=0.0)
    settings_small = ScanSettings(threshold=0.0, batch=16, mc_passes=1)
    settings_large = ScanSettings(threshold=0.0, batch=256, mc_passes=1)
    scores_small = candidate_scores(model, tiny_scan, settings_small, device="cpu")
    scores_large = candidate_scores(model, tiny_scan, settings_large, device="cpu")
    assert len(scores_small) == len(scores_large)
    assert np.allclose(scores_small, scores_large)
    findings_small = scan_bar(model, tiny_scan, settings_small, device="cpu")
    findings_large = scan_bar(model, tiny_scan, settings_large, device="cpu")
    assert len(findings_small) == len(findings_large)


def test_calibrate_bar_threshold_high_logits_budget(
    monkeypatch: pytest.MonkeyPatch,
    tiny_scan: BarScan,
) -> None:
    """calibrate_bar_threshold gives mean flags <= k when raw scores are 1.0 in float32."""
    call_idx = 0
    logits_per_scan = {
        0: 18.0,
        1: 21.0,
    }

    def mock_eval_batches(
        model: Any,
        windows: np.ndarray,
        settings: ScanSettings,
        dev: torch.device,
    ) -> _InferStats:
        nonlocal call_idx
        n = len(windows)
        lg = np.full(n, 0.0, dtype=np.float64)
        lg[0] = logits_per_scan[call_idx % 2]
        call_idx += 1
        probs = 1.0 / (1.0 + np.exp(-lg))
        return _InferStats(
            score=probs,
            logit=lg,
            uncertainty=np.zeros(n, dtype=np.float64),
            std=np.zeros(n, dtype=np.float64),
        )

    monkeypatch.setattr("utriage.ml.bar_infer._eval_batches", mock_eval_batches)
    model = WindowNet(in_channels=1, width=4)
    clean_scans = [tiny_scan, tiny_scan]
    budget_k = 0.5
    thresh = calibrate_bar_threshold(model, clean_scans, max_flags_per_bar=budget_k)
    assert 0.99999 <= thresh < 1.0

    call_idx = 0
    f0 = scan_bar(model, clean_scans[0], ScanSettings(threshold=thresh), device="cpu")
    call_idx = 1
    f1 = scan_bar(model, clean_scans[1], ScanSettings(threshold=thresh), device="cpu")
    mean_flags = (len(f0) + len(f1)) / 2.0
    assert mean_flags <= budget_k
