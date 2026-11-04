"""Tests for site adaptation evaluation experiment runner (project plan Phase 0 / ADR 0010).

Verifies shift application, per-shift evaluation on tiny inputs, and quick pipeline
execution using synthetic fake data in temporary directories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from experiments.run_site_adaptation import (
    DATASET_LABEL,
    FIXED_SHIFTS,
    LIMITS_TEXT,
    ShiftEvalSettings,
    ShiftEvaluationData,
    SiteAdaptationConfig,
    SiteShift,
    _parse_args,
    apply_shift,
    evaluate_shift,
    run_site_adaptation,
)

from utriage.integrity import sha256_file
from utriage.ml.anomaly import ConvAE2D


def _write_fake_batch(
    batch_path: Path,
    labels: list[tuple[float, float]],
    seed: int,
) -> Path:
    """Write fake .bins, .labels, and .jsons files for a batch of 256x256 images."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    raw = rng.integers(100, 3000, size=(n, 256, 256), dtype=np.uint16)
    raw.tofile(batch_path)

    labels_arr = np.array(labels, dtype=np.float32)
    np.savetxt(batch_path.with_suffix(".labels"), labels_arr, delimiter="\t")

    jsons_text = "".join(
        json.dumps({"flaws": [{"size": str(sz)}] if fl > 0 else []}) for fl, sz in labels
    )
    batch_path.with_suffix(".jsons").write_text(jsons_text, encoding="utf-8")
    return batch_path


@pytest.fixture
def fake_data_root(tmp_path: Path) -> Path:
    """Create fake ML-NDT validation directory with two batches in tmp_path."""
    root = tmp_path / "fake_mlndt"
    val_dir = root / "validation"
    val_dir.mkdir(parents=True)

    labels = [(0.0, 0.0)] * 8 + [(1.0, 1.6)] * 8
    for j in range(2):
        _write_fake_batch(val_dir / f"batch_val_{j:02d}.bins", labels, seed=100 + j)
    return root


def test_apply_shift_identity_unchanged() -> None:
    """Identity shift preserves input image values exactly."""
    rng = np.random.default_rng(42)
    img = rng.normal(0.0, 1.0, size=(32, 32)).astype(np.float32)
    shift = SiteShift(gain=1.0, offset=0.0, noise_std=0.0, name="identity")

    out = apply_shift(img, shift, rng)
    assert np.allclose(out, img, atol=1e-6)


def test_apply_shift_gain_scales() -> None:
    """Gain parameter scales pixel intensities linearly."""
    rng = np.random.default_rng(42)
    img = rng.normal(1.0, 0.5, size=(32, 32)).astype(np.float32)
    shift = SiteShift(gain=2.0, offset=0.0, noise_std=0.0, name="gain 2.0")

    out = apply_shift(img, shift, rng)
    assert np.allclose(out, img * 2.0, atol=1e-5)


@pytest.mark.parametrize(
    ("gain", "offset", "noise_std", "name"),
    [
        (1.0, 0.5, 0.0, "offset +0.5"),
        (0.5, 0.0, 0.0, "gain 0.5"),
        (1.0, 0.0, 0.2, "noise 0.2"),
    ],
)
def test_apply_shift_components(
    gain: float,
    offset: float,
    noise_std: float,
    name: str,
) -> None:
    """Individual shift components alter image statistics as expected."""
    rng = np.random.default_rng(99)
    img = np.zeros((64, 64), dtype=np.float32)
    shift = SiteShift(gain=gain, offset=offset, noise_std=noise_std, name=name)

    out = apply_shift(img, shift, rng)
    if noise_std == 0.0:
        assert np.allclose(out, offset, atol=1e-5)
    else:
        assert pytest.approx(float(np.std(out)), abs=0.05) == noise_std


def _assert_summary_block(summary: dict[str, Any]) -> None:
    """Verify keys and bounds for detection summary dictionary."""
    for key in ("k", "n", "rate", "ci_low", "ci_high", "interval_95"):
        assert key in summary
    assert summary["n"] >= summary["k"] >= 0
    assert 0.0 <= summary["ci_low"] <= summary["ci_high"] <= 1.0


def test_evaluate_shift_tiny_images() -> None:
    """evaluate_shift produces valid baseline and adapted results on tiny arrays."""
    rng = np.random.default_rng(77)
    cal_clean = [rng.normal(0.0, 1.0, (32, 32)).astype(np.float32) for _ in range(6)]
    test_clean = [rng.normal(0.0, 1.0, (32, 32)).astype(np.float32) for _ in range(6)]
    test_flawed = [rng.normal(2.0, 1.0, (32, 32)).astype(np.float32) for _ in range(4)]

    model = ConvAE2D()
    data = ShiftEvaluationData(cal_clean, test_clean, test_flawed, t_orig=1.5)
    settings = ShiftEvalSettings(target_fcr=0.1, device=torch.device("cpu"), seed=12)
    shift = SiteShift(gain=1.2, offset=0.5, noise_std=0.1, name="combined")

    result = evaluate_shift(model, shift, data, settings)
    assert result["shift"]["name"] == "combined"
    _assert_summary_block(result["baseline"]["false_calls"])
    _assert_summary_block(result["baseline"]["detection"])
    _assert_summary_block(result["adapted"]["false_calls"])
    _assert_summary_block(result["adapted"]["detection"])


def test_fixed_shifts_completeness() -> None:
    """Fixed shifts list contains all six required shift conditions."""
    names = [s.name for s in FIXED_SHIFTS]
    expected = ["identity", "gain 0.5", "gain 2.0", "offset +0.5", "noise 0.2", "combined"]
    assert names == expected


@pytest.mark.parametrize(
    ("cli_args", "expected_quick", "expected_fcr"),
    [
        ([], False, 0.01),
        (["--quick", "--fcr", "0.05"], True, 0.05),
    ],
)
def test_parse_args_options(
    cli_args: list[str],
    expected_quick: bool,
    expected_fcr: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Argument parser correctly extracts CLI settings."""
    monkeypatch.setattr("sys.argv", ["run_site_adaptation.py", *cli_args])
    cfg = _parse_args()
    assert cfg.quick is expected_quick
    assert pytest.approx(cfg.fcr) == expected_fcr


def _assert_payload_structure(payload: dict[str, Any], exp_sha: str) -> None:
    """Validate structure and provenance metadata of output JSON payload."""
    assert payload["label"] == DATASET_LABEL
    assert payload["dataset_label"] == DATASET_LABEL
    assert payload["limits"] == LIMITS_TEXT
    assert payload["model_sha256"] == exp_sha
    assert payload["seed"] == 0
    assert "n_images" in payload
    for shift in FIXED_SHIFTS:
        assert shift.name in payload["shifts"]
        shift_res = payload["shifts"][shift.name]
        _assert_summary_block(shift_res["baseline"]["false_calls"])
        _assert_summary_block(shift_res["adapted"]["false_calls"])


def test_quick_pipeline_execution(fake_data_root: Path, tmp_path: Path) -> None:
    """Quick pipeline execution writes valid site_adaptation.json into tmp_path."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)
    model_path = models_dir / "anomaly_2d.pt"
    torch.save(ConvAE2D().state_dict(), model_path)
    exp_sha = sha256_file(model_path)

    out_dir = tmp_path / "results"
    cfg = SiteAdaptationConfig(
        root=fake_data_root,
        out_dir=out_dir,
        models_dir=models_dir,
        model_path=model_path,
        fcr=0.05,
        device="cpu",
        seed=0,
        quick=True,
    )

    metrics = run_site_adaptation(cfg)
    out_file = out_dir / "site_adaptation.json"
    assert out_file.is_file()

    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    _assert_payload_structure(metrics, exp_sha)
    _assert_payload_structure(loaded, exp_sha)
