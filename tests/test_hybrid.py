"""Tests for hybrid detection module and experiment runner (ADR 0010)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        del sys.modules["utriage"]

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from utriage.ml.anomaly import ConvAE2D  # noqa: E402
from utriage.ml.infer import save_model  # noqa: E402
from utriage.ml.models import MlNdtNet  # noqa: E402
from utriage.triage.hybrid import (  # noqa: E402
    ecdf,
    evaluate_detector,
    fused_score,
    or_union,
    quantile_threshold,
)


def _make_fake_batch(
    batch_path: Path,
    flaws: list[dict[str, Any]],
    labels: list[tuple[float, float]],
    seed: int,
) -> Path:
    """Create a mock ML-NDT batch file tuple (.bins, .jsons, .labels)."""
    rng = np.random.default_rng(seed)
    n = len(flaws)
    raw = rng.integers(100, 3000, size=(n, 256, 256), dtype=np.uint16)
    raw.tofile(batch_path)

    jsons_text = "".join(json.dumps({"flaws": [f]}) for f in flaws)
    batch_path.with_suffix(".jsons").write_text(jsons_text, encoding="utf-8")

    labels_arr = np.array(labels, dtype=np.float32)
    np.savetxt(batch_path.with_suffix(".labels"), labels_arr, delimiter="\t")
    return batch_path


def _populate_fake_mlndt(root: Path) -> Path:
    """Generate minimal fake training and validation batches."""
    tr_dir = root / "training"
    val_dir = root / "validation"
    tr_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    flaws = [{"size": "1.6"}, {"size": "4.0"}, {"size": "8.6"}, {"size": "0.0"}]
    labels = [(1.0, 1.6), (1.0, 4.0), (1.0, 8.6), (0.0, 0.0)]

    for i in range(5):
        _make_fake_batch(tr_dir / f"batch_tr_{i:02d}.bins", flaws, labels, seed=i)
    for j in range(2):
        _make_fake_batch(val_dir / f"batch_val_{j:02d}.bins", flaws, labels, seed=100 + j)
    return root


@pytest.mark.parametrize(
    ("p_sup", "a_score", "t_anom", "expected"),
    [
        (
            [0.1, 0.49, 0.5, 0.9, 0.2],
            [0.1, 0.2, 0.05, 0.3, 0.8],
            0.5,
            [False, False, True, True, True],
        ),
        ([0.0, 1.0], [0.0, 0.0], 0.5, [False, True]),
        ([0.0, 0.0], [0.1, 0.9], 0.5, [False, True]),
    ],
)
def test_or_union_hand_arrays(
    p_sup: list[float],
    a_score: list[float],
    t_anom: float,
    expected: list[bool],
) -> None:
    """Verify or_union produces exact logical OR against thresholded inputs."""
    res = or_union(np.array(p_sup), np.array(a_score), t_anom)
    assert np.array_equal(res, np.array(expected))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_or_union_safety_property(seed: int) -> None:
    """Ensure OR-union never produces fewer positive flags than either detector."""
    rng = np.random.default_rng(seed)
    p_sup = rng.uniform(0.0, 1.0, size=50)
    a_score = rng.uniform(0.0, 2.0, size=50)
    t_anom = 1.0

    flags = or_union(p_sup, a_score, t_anom)
    flags_sup = p_sup >= 0.5
    flags_anom = a_score >= t_anom

    assert np.all(flags >= flags_sup)
    assert np.all(flags >= flags_anom)
    assert np.sum(flags) >= np.sum(flags_sup)
    assert np.sum(flags) >= np.sum(flags_anom)


def test_or_union_shape_mismatch() -> None:
    """Ensure or_union raises ValueError on differing array shapes."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        or_union(np.zeros(3), np.zeros(4), 0.5)


def test_fused_score_hand_arrays() -> None:
    """Verify fused_score averages probabilities element-wise."""
    p_sup = np.array([0.2, 0.6, 0.8])
    p_anom = np.array([0.4, 0.4, 0.2])
    fused = fused_score(p_sup, p_anom)
    expected = np.array([0.3, 0.5, 0.5])
    assert np.allclose(fused, expected)


def test_fused_score_shape_mismatch() -> None:
    """Ensure fused_score raises ValueError on shape mismatch."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        fused_score(np.zeros(2), np.zeros(3))


@pytest.mark.parametrize("fcr", [0.01, 0.05, 0.10])
def test_quantile_threshold_calibration_fcr(fcr: float) -> None:
    """Verify empirical false call rate on calibration set does not exceed target fcr."""
    calib = np.linspace(0.0, 10.0, 100)
    threshold = quantile_threshold(calib, fcr)
    empirical_fcr = float(np.mean(calib >= threshold))
    assert empirical_fcr <= fcr


def test_quantile_threshold_invalid_fcr() -> None:
    """Verify quantile_threshold validates target fcr range."""
    calib = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="false_call_rate must be in"):
        quantile_threshold(calib, 1.5)


def test_ecdf_monotone_in_unit_interval() -> None:
    """Verify empirical CDF is monotonic non-decreasing and bounded in [0, 1]."""
    calib = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    cdf_fn = ecdf(calib)

    queries = np.linspace(0.0, 6.0, 61)
    probs = cdf_fn(queries)

    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.all(np.diff(probs) >= 0.0)
    assert cdf_fn(0.0) == 0.0
    assert cdf_fn(6.0) == 1.0


def test_ecdf_empty_calibration() -> None:
    """Verify ecdf handles empty calibration arrays gracefully."""
    cdf_fn = ecdf(np.empty(0))
    probs = cdf_fn(np.array([1.0, 2.0]))
    assert np.array_equal(probs, np.zeros(2))


def test_evaluate_detector_counts() -> None:
    """Verify evaluate_detector accurately calculates detection and false call metrics."""
    flags = np.array([1, 0, 1, 1, 0, 0, 1])
    labels = np.array([1, 1, 1, 0, 0, 0, 0])
    summary = evaluate_detector(flags, labels)

    det = summary["detection"]
    fc = summary["false_calls"]

    assert det["n"] == 3
    assert det["k"] == 2
    assert np.isclose(det["rate"], 2.0 / 3.0)
    assert "ci_low" in det and "ci_high" in det

    assert fc["n_scans"] == 4
    assert fc["n_false"] == 2
    assert np.isclose(fc["rate"], 0.5)


def _setup_stubs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Create directory structure, fake ML-NDT data, and stub models."""
    fake_root = _populate_fake_mlndt(tmp_path / "fake_data")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    ae_model = ConvAE2D()
    torch.save(ae_model.state_dict(), models_dir / "anomaly_2d.pt")

    for crack in ("crack_1.6", "crack_4.0", "crack_8.6"):
        net = MlNdtNet(width=4)
        save_model(net, models_dir / f"loco_{crack}.pt", {"width": 4})
        for s in range(3):
            save_model(net, models_dir / f"loco_{crack}_seed{s}.pt", {"width": 4})

    anom_json = results_dir / "anomaly_metrics.json"
    anom_json.write_text(json.dumps({"real": {"threshold": 0.5}}), encoding="utf-8")
    out_json = results_dir / "hybrid_metrics.json"

    return fake_root, models_dir, anom_json, out_json


def test_run_hybrid_on_stubs(tmp_path: Path) -> None:
    """Verify run_hybrid runner runs on stub models and produces compliant JSON output."""
    from experiments.run_hybrid import HybridConfig, run_hybrid

    fake_root, models_dir, anom_json, out_json = _setup_stubs(tmp_path)
    cfg = HybridConfig(
        root=fake_root,
        models_dir=models_dir,
        anomaly_metrics_path=anom_json,
        out_json=out_json,
        device="cpu",
        quick=True,
    )
    result = run_hybrid(cfg)

    assert result["label"] == "Real data (ML-NDT)"
    assert "ADR 0010" in result["protocol"] or "0010" in result["protocol"]
    assert out_json.exists()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["label"] == "Real data (ML-NDT)"

    for crack in ("crack_1.6", "crack_4.0", "crack_8.6"):
        fold_data = payload["folds"][crack]
        assert "models" in fold_data
        assert "summary_3seeds" in fold_data

        weights_file = f"loco_{crack}.pt"
        w_entry = fold_data["models"][weights_file]
        assert "model_sha256" in w_entry
        assert len(w_entry["model_sha256"]) == 64

        for method in ("supervised", "anomaly", "h1", "h2"):
            assert method in w_entry["methods"]
            cell = w_entry["methods"][method]
            assert "detection" in cell
            assert "false_calls" in cell
            assert "model_sha256" in cell

            m_sum = fold_data["summary_3seeds"][method]
            assert "mean_detection_rate" in m_sum
            assert "min_detection_rate" in m_sum
            assert "max_detection_rate" in m_sum
            assert "mean_false_call_rate" in m_sum
