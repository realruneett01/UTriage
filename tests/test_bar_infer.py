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

from utriage.ml.bar_data import WINDOW  # noqa: E402
from utriage.ml.bar_infer import (  # noqa: E402
    ScanSettings,
    _Candidate,
    _InferStats,
    _suppress_candidates,
    calibrate_bar_threshold,
    count_findings,
    finding_to_dict,
    findings_to_dicts,
    load_window_model,
    save_window_model,
    scan_bar,
    site_depths,
    site_logits,
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


def _make_candidate(
    axial_idx: int,
    angle_idx: int,
    score: float,
    logit: float,
) -> _Candidate:
    """Create a test candidate with standard depth and uncertainty."""
    return _Candidate(
        axial_idx=axial_idx,
        angle_idx=angle_idx,
        depth_mm=5.0,
        score=score,
        logit=logit,
        uncertainty=0.1,
        std=0.05,
    )


def _make_depth_test_scan(material_name: str) -> tuple[BarScan, ScanSettings]:
    """Create a simulated bar scan and scan settings for gate depth testing."""
    mat = load_material(material_name)
    probe = load_probe("p5")
    water = load_material("water")
    spec = BarSpec(
        material=mat,
        probe=probe,
        water=water,
        diameter_mm=40.0,
        length_mm=4.0,
    )
    scan = simulate_bar(spec)
    settings = ScanSettings(front_mm=2.0, back_margin_mm=2.0, depth_step_mm=0.5)
    return scan, settings


def _make_synthetic_calibration_grids() -> list[np.ndarray]:
    """Create two synthetic grids with known peaks for threshold calibration."""
    g0 = np.full((8, 12, 8), -10.0, dtype=np.float64)
    g0[1, 2, 1] = 5.0
    g0[4, 6, 4] = 3.0
    g0[6, 9, 6] = 1.0

    g1 = np.full((8, 12, 8), -10.0, dtype=np.float64)
    g1[1, 3, 1] = 4.5
    g1[4, 7, 4] = 2.5
    g1[6, 10, 6] = 0.5
    return [g0, g1]


def _assert_calibrated_scans(
    model: WindowNet,
    scans: list[BarScan],
    thresh: float,
    budget: float,
) -> None:
    """Assert scans evaluated at threshold satisfy budget and finding counts."""
    f0 = scan_bar(model, scans[0], ScanSettings(threshold=thresh), device="cpu")
    f1 = scan_bar(model, scans[1], ScanSettings(threshold=thresh), device="cpu")
    assert (len(f0) + len(f1)) / 2.0 <= budget
    assert len(f0) == 1
    assert len(f1) == 1


def _mock_high_logits_eval(logits_per_scan: dict[int, float]):
    """Return a mock evaluator producing saturated float32 logits."""
    call_idx = 0

    def mock_eval_batches(
        model: Any,
        windows: np.ndarray,
        settings: ScanSettings,
        dev: torch.device,
    ) -> _InferStats:
        nonlocal call_idx
        n = len(windows)
        lg = np.zeros(n, dtype=np.float64)
        lg[0] = logits_per_scan[call_idx % 2]
        call_idx += 1
        probs = 1.0 / (1.0 + np.exp(-lg))
        return _InferStats(
            score=probs,
            logit=lg,
            uncertainty=np.zeros(n, dtype=np.float64),
            std=np.zeros(n, dtype=np.float64),
        )

    return mock_eval_batches


def test_scan_bar_findings_grid_and_gate(tiny_model: WindowNet, tiny_scan: BarScan) -> None:
    settings = ScanSettings(threshold=-1.0, depth_step_mm=1.0, mc_passes=4)
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
        ScanSettings(threshold=100.0),
        ScanSettings(front_mm=25.0, back_margin_mm=2.0),
    ],
    ids=["threshold_above_all", "gate_outside_bar"],
)
def test_scan_bar_empty(tiny_model: WindowNet, tiny_scan: BarScan, settings: ScanSettings) -> None:
    findings = scan_bar(tiny_model, tiny_scan, settings, device="cpu")
    assert findings == []


def test_scan_bar_mc_uncertainty_range(tiny_model: WindowNet, tiny_scan: BarScan) -> None:
    settings = ScanSettings(threshold=-1.0, mc_passes=6)
    findings = scan_bar(tiny_model, tiny_scan, settings, device="cpu")
    _assert_uncertainties_bounded(findings)


def test_scan_bar_batchnorm_stats_unchanged(tiny_model: WindowNet, tiny_scan: BarScan) -> None:
    bn_modules = [m for m in tiny_model.modules() if isinstance(m, nn.BatchNorm1d)]
    means_before = [m.running_mean.clone() for m in bn_modules]
    vars_before = [m.running_var.clone() for m in bn_modules]
    scan_bar(tiny_model, tiny_scan, ScanSettings(threshold=-1.0), device="cpu")
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
    findings = scan_bar(patch_model, tiny_scan, ScanSettings(threshold=-1.0), device="cpu")
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
    expected_values = {
        "axial_idx": 2,
        "angle_idx": 3,
        "depth_mm": pytest.approx(5.5),
        "score": pytest.approx(0.85),
        "uncertainty": pytest.approx(0.12),
        "grain_noise_prob": pytest.approx(0.15),
        "evidence": ["ML window score 0.85", "MC-dropout std 0.06"],
    }
    for key, expected in expected_values.items():
        assert d[key] == expected


def test_finding_to_dict() -> None:
    finding = MlFinding(
        axial_idx=1,
        angle_idx=2,
        depth_mm=4.0,
        score=0.9,
        uncertainty=0.1,
    )
    d = finding_to_dict(finding)
    assert d["axial_idx"] == 1
    assert d["angle_idx"] == 2
    assert d["depth_mm"] == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("blob_case", "expected_count"),
    [
        ("separated_blobs", 2),
        ("angle_wrap", 1),
        ("threshold_above_all", 0),
    ],
)
def test_count_findings_handmade_grid(blob_case: str, expected_count: int) -> None:
    grid = np.full((6, 12, 6), -10.0, dtype=np.float64)
    thresh = 0.0
    if blob_case == "separated_blobs":
        grid[1:3, 2:4, 1:3] = 5.0
        grid[4:6, 8:10, 4:6] = 4.0
    elif blob_case == "angle_wrap":
        grid[2, 0, 2] = 5.0
        grid[2, 11, 2] = 5.0
    elif blob_case == "threshold_above_all":
        grid[1:3, 2:4, 1:3] = 5.0
        thresh = 10.0
    assert count_findings(grid, logit_threshold=thresh) == expected_count


def test_calibrate_bar_threshold_synthetic_grids(
    monkeypatch: pytest.MonkeyPatch,
    tiny_scan: BarScan,
) -> None:
    synthetic_grids = _make_synthetic_calibration_grids()
    depths = np.linspace(2.0, 18.0, 8)
    call_idx = 0

    def mock_site_logits(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        nonlocal call_idx
        grid = synthetic_grids[call_idx % len(synthetic_grids)]
        call_idx += 1
        return grid, depths

    monkeypatch.setattr("utriage.ml.bar_infer.site_logits", mock_site_logits)
    model = WindowNet(in_channels=1, width=4)
    clean_scans = [tiny_scan, tiny_scan]
    budget = 1.0

    thresh = calibrate_bar_threshold(model, clean_scans, max_flags_per_bar=budget)
    assert 3.0 <= thresh <= 5.0
    call_idx = 0
    _assert_calibrated_scans(model, clean_scans, thresh, budget)


def test_site_logits_shape_and_depths(tiny_model: WindowNet, tiny_scan: BarScan) -> None:
    settings = ScanSettings(threshold=-1.0, depth_step_mm=1.0, mc_passes=2)
    logits_grid, depths = site_logits(tiny_model, tiny_scan, settings, device="cpu")
    assert logits_grid.ndim == 3
    assert logits_grid.shape[0] == tiny_scan.ascans.shape[0]
    assert logits_grid.shape[1] == tiny_scan.ascans.shape[1]
    assert logits_grid.shape[2] == len(depths)
    assert logits_grid.dtype == np.float64


def _make_candidate_blob(
    peak_coord: tuple[int, int],
    bg_coords: tuple[tuple[int, int], ...],
    peak_scores: tuple[float, float],
    base_scores: tuple[float, float],
) -> list[_Candidate]:
    """Create a candidate cluster with one peak and surrounding background sites."""
    px, pa = peak_coord
    cands = [_make_candidate(px, pa, score=peak_scores[0], logit=peak_scores[1])]
    for ax, an in bg_coords:
        cands.append(_make_candidate(ax, an, score=base_scores[0], logit=base_scores[1]))
    return cands


def test_blob_3x3_yields_single_finding() -> None:
    """A 3x3 blob of connected neighbouring candidates collapses to exactly one finding."""
    bg = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2))
    cands = _make_candidate_blob((1, 1), bg, (0.9, 2.0), (0.8, 1.0))
    kept = _suppress_candidates(cands, n_angle=8)
    assert len(kept) == 1
    assert kept[0].axial_idx == 1
    assert kept[0].angle_idx == 1


def test_two_separated_blobs_yield_two_findings() -> None:
    """Two spatially separated blobs yield exactly two findings."""
    blob1_bg = ((0, 1), (1, 0), (1, 1))
    blob1 = _make_candidate_blob((0, 0), blob1_bg, (0.9, 2.0), (0.8, 1.0))
    blob2_bg = ((10, 11), (11, 10), (11, 11))
    blob2 = _make_candidate_blob((10, 10), blob2_bg, (0.85, 1.8), (0.7, 0.9))
    kept = _suppress_candidates(blob1 + blob2, n_angle=20)
    assert len(kept) == 2
    assert {c.axial_idx for c in kept} == {0, 10}


def test_site_logits_and_scan_bar_invariant_to_batch(tiny_scan: BarScan) -> None:
    """Site logits and scan_bar findings are invariant to settings.batch."""
    model = WindowNet(in_channels=1, width=4, dropout=0.0)
    settings_small = ScanSettings(threshold=-1.0, batch=16, mc_passes=1)
    settings_large = ScanSettings(threshold=-1.0, batch=256, mc_passes=1)
    grid_small, _ = site_logits(model, tiny_scan, settings_small, device="cpu")
    grid_large, _ = site_logits(model, tiny_scan, settings_large, device="cpu")
    assert grid_small.shape == grid_large.shape
    assert np.allclose(grid_small, grid_large)
    findings_small = scan_bar(model, tiny_scan, settings_small, device="cpu")
    findings_large = scan_bar(model, tiny_scan, settings_large, device="cpu")
    assert len(findings_small) == len(findings_large)


def test_calibrate_bar_threshold_high_logits_budget(
    monkeypatch: pytest.MonkeyPatch,
    tiny_scan: BarScan,
) -> None:
    """calibrate_bar_threshold gives mean flags <= k when raw scores are 1.0 in float32."""
    mock_eval = _mock_high_logits_eval({0: 18.0, 1: 21.0})
    monkeypatch.setattr("utriage.ml.bar_infer._eval_batches", mock_eval)
    model = WindowNet(in_channels=1, width=4)
    clean_scans = [tiny_scan, tiny_scan]
    budget_k = 0.5
    thresh = calibrate_bar_threshold(model, clean_scans, max_flags_per_bar=budget_k)
    assert 18.0 <= thresh <= 21.1

    f0 = scan_bar(model, clean_scans[0], ScanSettings(threshold=thresh), device="cpu")
    f1 = scan_bar(model, clean_scans[1], ScanSettings(threshold=thresh), device="cpu")
    assert (len(f0) + len(f1)) / 2.0 <= budget_k


@pytest.mark.parametrize("material_name", ["ti64", "in718"])
def test_site_depths_stays_inside_flaw_gate(material_name: str) -> None:
    """site_depths stays inside [front + half_window, D - back - half_window] for Ti64 and IN718."""
    scan, settings = _make_depth_test_scan(material_name)
    c_metal = scan.spec.material.velocity_m_s * 1e-3
    half_window = (WINDOW / 2) / scan.spec.probe.sampling_mhz * c_metal / 2.0
    d_min = settings.front_mm + half_window
    d_max = scan.spec.diameter_mm - settings.back_margin_mm - half_window

    depths = site_depths(scan, settings)
    assert len(depths) > 0
    assert np.all(depths >= d_min - 1e-9)
    assert np.all(depths <= d_max + 1e-9)


@pytest.mark.parametrize("material_name", ["ti64", "in718"])
def test_cut_windows_exclude_back_wall_echo(material_name: str) -> None:
    """No window cut at returned sites includes the back-wall echo time."""
    scan, settings = _make_depth_test_scan(material_name)
    depths = site_depths(scan, settings)
    assert len(depths) > 0

    c_m = scan.spec.material.velocity_m_s * 1e-3
    t_if = float(scan.t_if_us[0, 0]) if scan.t_if_us.ndim == 2 else float(scan.t_if_us)
    t_bw = t_if + 2.0 * scan.spec.diameter_mm / c_m
    half_window_time = (WINDOW / 2) / scan.spec.probe.sampling_mhz

    for d in depths:
        t_center = t_if + 2.0 * d / c_m
        assert t_center + half_window_time < t_bw
        assert t_center - half_window_time > t_if


def test_raw_logit_thresholds_distinguish_float32_saturated_sites(
    monkeypatch: pytest.MonkeyPatch,
    tiny_scan: BarScan,
) -> None:
    """Raw-logit thresholds distinguish sites whose probabilities are both 1.0 in float32."""
    g = np.full((1, 1, 3), -10.0, dtype=np.float64)
    g[0, 0, 0] = 20.0
    g[0, 0, 2] = 25.0

    assert np.float32(1.0 / (1.0 + np.exp(-20.0))) == 1.0
    assert np.float32(1.0 / (1.0 + np.exp(-25.0))) == 1.0

    for thresh, expected in ((22.0, 1), (15.0, 2), (30.0, 0)):
        assert count_findings(g, logit_threshold=thresh) == expected

    depths = np.array([5.0, 8.0, 11.0], dtype=np.float64)
    monkeypatch.setattr(
        "utriage.ml.bar_infer.site_logits",
        lambda *args, **kwargs: (g, depths),
    )
    model = WindowNet(in_channels=1, width=4)
    findings = scan_bar(model, tiny_scan, ScanSettings(threshold=22.0), device="cpu")
    assert len(findings) == 1
    assert findings[0].depth_mm == 11.0
    assert findings[0].score == pytest.approx(1.0)
