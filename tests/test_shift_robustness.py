"""Unit tests for site shift robustness evaluation experiment (ADR 0015).

Tests CPU-safe evaluation, unshifted threshold derivation, identical shifted image
delivery across model families, worst-case summary selection, margin sign, and payload schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from experiments.run_shift_robustness import (
    EXPERIMENT_LABEL,
    EXPERIMENT_NOTE,
    PREREGISTRATION_PATH,
    ExperimentConfig,
    build_payload,
    compute_shift_margin,
    compute_unshifted_thresholds,
    compute_worst_case_summary,
    evaluate_family_fold,
    main,
)
from run_site_adaptation import FIXED_SHIFTS

from utriage.ml.loco import LocoFold


@pytest.mark.parametrize(
    ("flawed_median", "th_01", "expected_sign"),
    [
        (0.8, 0.08, 1.0),
        (0.008, 0.08, -1.0),
        (0.08, 0.08, 0.0),
    ],
    ids=["positive_margin", "negative_margin", "zero_margin"],
)
def test_margin_calculation_sign(
    flawed_median: float,
    th_01: float,
    expected_sign: float,
) -> None:
    """Check that shift margin sign correctly reflects median score vs threshold."""
    scores = np.array([flawed_median], dtype=np.float32)
    margin = compute_shift_margin(scores, th_01)
    if expected_sign > 0:
        assert margin > 0.0
    elif expected_sign < 0:
        assert margin < 0.0
    else:
        assert margin == pytest.approx(0.0, abs=1e-6)


def test_margin_empty_scores() -> None:
    """Check that shift margin handles empty scores array gracefully."""
    empty = np.empty(0, dtype=np.float32)
    margin = compute_shift_margin(empty, 0.05)
    assert np.isnan(margin)


def _make_dummy_shift_result(d_rate: float, f_rate: float) -> dict[str, Any]:
    """Build mock single shift dictionary with given rates."""
    op = {
        "detection": {"rate": d_rate},
        "false_calls": {"rate": f_rate},
    }
    return {"operating_points": {"0.01": op, "0.05": op}}


@pytest.mark.parametrize(
    ("det_rates", "fc_rates", "expected_det", "expected_fc"),
    [
        ([0.95, 0.70, 0.85], [0.01, 0.04, 0.02], 0.70, 0.04),
        ([0.50, 0.60, 0.55], [0.10, 0.05, 0.08], 0.50, 0.10),
    ],
    ids=["case_a", "case_b"],
)
def test_worst_case_summary_selection(
    det_rates: list[float],
    fc_rates: list[float],
    expected_det: float,
    expected_fc: float,
) -> None:
    """Check that worst-case summary picks minimum detection and maximum false call."""
    shift_results = [
        _make_dummy_shift_result(d, f) for d, f in zip(det_rates, fc_rates, strict=True)
    ]
    summary = compute_worst_case_summary(shift_results)
    assert summary["worst_case_detection_rate"] == pytest.approx(expected_det)
    assert summary["worst_case_false_call_rate"] == pytest.approx(expected_fc)


@pytest.mark.parametrize(
    "flawed_shift",
    [0.0, -0.5, 2.0],
    ids=["baseline", "lowered_flawed", "elevated_flawed"],
)
def test_threshold_uses_only_calib(flawed_shift: float) -> None:
    """Check that operating threshold depends exclusively on unshifted clean calib scores."""
    calib = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float32)
    _ = np.array([0.7, 0.8, 0.9], dtype=np.float32) + flawed_shift
    th_01, th_05 = compute_unshifted_thresholds(calib)
    exp_01 = float(np.quantile(calib, 1.0 - 0.01, method="higher"))
    exp_05 = float(np.quantile(calib, 1.0 - 0.05, method="higher"))
    assert th_01 == pytest.approx(exp_01)
    assert th_05 == pytest.approx(exp_05)


def test_missing_family_recorded_as_not_run(tmp_path: Path) -> None:
    """Check that missing model checkpoints record status as not run without error."""
    empty_models = tmp_path / "models"
    empty_models.mkdir()
    fold = LocoFold(held_out="crack_1.6", train=[], val=[], test=[])
    cfg = ExperimentConfig(models_dir=empty_models, out=tmp_path / "out.json")

    res = evaluate_family_fold("loco", fold, cfg, torch.device("cpu"))
    assert res["status"] == "not run"
    assert res["shifts"] == []
    assert res["summary"] is None


class DummyTrackingNet(nn.Module):
    """Toy network recording input batches during evaluation."""

    def __init__(self, fixed_score: float = 0.6) -> None:
        super().__init__()
        self.fixed_score = fixed_score
        self.seen_inputs: list[np.ndarray] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen_inputs.append(x.detach().cpu().numpy())
        n = x.size(0)
        logit = float(np.log(self.fixed_score / (1.0 - self.fixed_score)))
        return torch.full((n, 1), logit, dtype=torch.float32, device=x.device)


def _build_synthetic_fold(tmp_path: Path) -> tuple[LocoFold, np.ndarray]:
    """Create tiny test fold with synthetic scan items."""
    img_dir = tmp_path / "fake_data"
    img_dir.mkdir(parents=True, exist_ok=True)
    items: list[tuple[Path, int]] = []
    for i in range(12):
        fake_bin = img_dir / f"batch_{i}.bin"
        fake_bin.touch()
        items.append((fake_bin, i))
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    fold = LocoFold(held_out="crack_1.6", train=[], val=[], test=items)
    return fold, labels


def _patch_evaluation_context(
    monkeypatch: pytest.MonkeyPatch,
    model_a: DummyTrackingNet,
    model_b: DummyTrackingNet,
    labels: np.ndarray,
) -> None:
    """Configure monkeypatched mocks for image loading and model evaluation."""
    monkeypatch.setattr(
        "experiments.run_shift_robustness.load_family_models",
        lambda fam, _f, _c, _d: [model_a] if fam == "loco" else [model_b],
    )
    monkeypatch.setattr(
        "experiments.run_shift_robustness.get_fold_labels",
        lambda _f: labels,
    )
    monkeypatch.setattr(
        "experiments.run_shift_robustness.read_item_image",
        lambda _it: np.full((16, 16), 0.5, dtype=np.float32),
    )


def test_all_families_see_identical_shifted_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check that all model families evaluate on exact identical shifted image sequences."""
    fold, labels = _build_synthetic_fold(tmp_path)
    model_a = DummyTrackingNet(0.7)
    model_b = DummyTrackingNet(0.3)
    _patch_evaluation_context(monkeypatch, model_a, model_b, labels)

    cfg = ExperimentConfig(
        families=("loco", "lite"),
        models_dir=tmp_path,
        seed=42,
        chunk_size=4,
    )
    dev = torch.device("cpu")
    _ = evaluate_family_fold("loco", fold, cfg, dev)
    _ = evaluate_family_fold("lite", fold, cfg, dev)

    assert len(model_a.seen_inputs) > 0
    assert len(model_a.seen_inputs) == len(model_b.seen_inputs)
    for arr_a, arr_b in zip(model_a.seen_inputs, model_b.seen_inputs, strict=True):
        assert np.array_equal(arr_a, arr_b)


def test_payload_structure_and_constants() -> None:
    """Check top-level payload structure, shifts, and constant label strings."""
    fake_families = {
        "loco": {"status": "not run", "folds": []},
        "robust": {"status": "not run", "folds": []},
    }
    sha_map = {"model.pt": "abc123sha"}
    payload = build_payload(fake_families, sha_map, seed=7)

    assert payload["label"] == EXPERIMENT_LABEL
    assert payload["preregistration"] == PREREGISTRATION_PATH
    assert payload["note"] == EXPERIMENT_NOTE
    assert payload["seed"] == 7
    assert len(payload["shifts"]) == len(FIXED_SHIFTS)
    assert payload["families"] == fake_families
    assert payload["model_sha256"] == sha_map


def test_main_cli_execution(tmp_path: Path) -> None:
    """Check main CLI parses arguments, runs without error, and outputs valid JSON."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    out_json = tmp_path / "results" / "shift_robustness_metrics.json"

    argv = [
        "--models-dir",
        str(models_dir),
        "--out",
        str(out_json),
        "--device",
        "cpu",
        "--families",
        "loco",
        "robust",
        "--seed",
        "0",
        "--quick",
    ]
    payload = main(argv)

    assert out_json.exists()
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded["label"] == payload["label"]
    assert "loco" in loaded["families"]
    assert loaded["families"]["loco"]["status"] == "not run"
