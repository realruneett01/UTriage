"""Tests for false call analysis and operating point trade-off curves."""

from __future__ import annotations

import json
import subprocess
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
from utriage.report.false_calls import (  # noqa: E402
    FalseCallEvidence,
    build_agreement_table,
    build_false_call_report_md,
    classify_false_call_cause,
    compute_tradeoff_curve,
    is_low_margin,
    is_near_miss_supervised,
    plot_flagged_image,
    plot_tradeoff_curve,
)


@pytest.mark.parametrize(
    ("score", "threshold", "ratio", "expected"),
    [
        (0.52, 0.50, 0.10, True),
        (0.56, 0.50, 0.10, False),
        (1.05, 1.00, 0.10, True),
        (1.20, 1.00, 0.10, False),
        (0.50, 0.00, 0.10, False),
    ],
)
def test_is_low_margin_branches(
    score: float,
    threshold: float,
    ratio: float,
    expected: bool,
) -> None:
    """Verify is_low_margin detects scores within ratio above threshold."""
    assert is_low_margin(score, threshold, ratio) is expected


@pytest.mark.parametrize(
    ("score", "threshold", "expected"),
    [
        (0.46, 0.50, True),
        (0.35, 0.50, False),
        (0.52, 0.50, False),
        (0.40, 0.00, False),
    ],
)
def test_is_near_miss_supervised(score: float, threshold: float, expected: bool) -> None:
    """Verify supervised CNN near-miss identification on borderline probabilities."""
    assert is_near_miss_supervised(score, threshold) is expected


@pytest.mark.parametrize(
    ("evidence", "expected_cause"),
    [
        (
            FalseCallEvidence(near_edge=True, snr=6.0, agreement_count=2, score=0.6, threshold=0.5),
            "near_edge",
        ),
        (
            FalseCallEvidence(
                near_edge=False, snr=6.0, agreement_count=2, score=0.8, threshold=0.5
            ),
            "high_snr_structure",
        ),
        (
            FalseCallEvidence(
                near_edge=False, snr=2.0, agreement_count=1, score=0.52, threshold=0.5
            ),
            "low_margin",
        ),
        (
            FalseCallEvidence(
                near_edge=False, snr=2.0, agreement_count=2, score=0.8, threshold=0.5
            ),
            "multi_detector_agreement",
        ),
        (
            FalseCallEvidence(
                near_edge=False, snr=2.0, agreement_count=1, score=0.8, threshold=0.5
            ),
            "unclassified",
        ),
    ],
)
def test_classify_false_call_cause_branches(
    evidence: FalseCallEvidence,
    expected_cause: str,
) -> None:
    """Verify each documented false call cause branch in priority order."""
    assert classify_false_call_cause(evidence) == expected_cause


def test_classify_false_call_cause_dict_input() -> None:
    """Verify classify_false_call_cause accepts dictionary representation."""
    meta = {"near_edge": True, "image_snr": 1.5, "agreement_count": 1}
    assert classify_false_call_cause(meta) == "near_edge"


def test_build_agreement_table_hand_inputs() -> None:
    """Verify agreement table computation on synthetic flagged records."""
    records = [
        {"flagged_detectors": ["anomaly"], "agreement_count": 1},
        {"flagged_detectors": ["anomaly", "h2"], "agreement_count": 2},
        {"flagged_detectors": ["anomaly", "h2", "h3"], "agreement_count": 3},
    ]
    table = build_agreement_table(records)
    assert table["counts_by_agreement"][1] == 1
    assert table["counts_by_agreement"][2] == 1
    assert table["counts_by_agreement"][3] == 1
    assert table["by_detector"]["anomaly"] == 3
    assert table["by_detector"]["h2"] == 2
    assert table["by_detector"]["h3"] == 1


@pytest.mark.parametrize("n_clean", [50, 100])
def test_tradeoff_curve_monotonicity(n_clean: int) -> None:
    """Verify false-call and detection rates are non-increasing with threshold."""
    rng = np.random.default_rng(42)
    scores_clean = rng.uniform(0.0, 0.4, size=n_clean)
    scores_flawed = rng.uniform(0.2, 0.9, size=40)
    y_true = np.concatenate([np.ones(40, dtype=int), np.zeros(n_clean, dtype=int)])
    scores = np.concatenate([scores_flawed, scores_clean])

    curve = compute_tradeoff_curve(y_true, scores, operating_threshold=0.3)
    fc_rates = curve["false_call_rates"]
    det_rates = curve["detection_rates"]

    assert all(fc_rates[i] >= fc_rates[i + 1] for i in range(len(fc_rates) - 1))
    assert all(det_rates[i] >= det_rates[i + 1] for i in range(len(det_rates) - 1))
    assert 0.0 <= curve["operating_point"]["false_call_rate"] <= 1.0


def test_plot_tradeoff_curve_saves_png(tmp_path: Path) -> None:
    """Verify trade-off curve rendering generates valid output PNG file."""
    curve = {
        "false_call_rates": [0.2, 0.1, 0.0],
        "detection_rates": [0.95, 0.90, 0.80],
        "operating_point": {"threshold": 0.5, "detection_rate": 0.90, "false_call_rate": 0.1},
    }
    out_png = tmp_path / "tradeoff_test.png"
    plot_tradeoff_curve(curve, "supervised", out_png)
    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_plot_flagged_image_saves_png(tmp_path: Path) -> None:
    """Verify flagged image and error map rendering generates small 6x3 PNG."""
    img = np.ones((64, 64), dtype=np.float32)
    err_map = np.zeros((64, 64), dtype=np.float32)
    out_png = tmp_path / "flagged_case.png"
    plot_flagged_image(img, err_map, out_png, title="test_case")
    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_build_false_call_report_md_content() -> None:
    """Verify false call markdown synthesis includes required labels and notice."""
    summary_data = {
        "label": "Real data (ML-NDT)",
        "n_clean": 778,
        "n_flagged": 14,
        "cause_counts": {"near_edge": 5, "high_snr_structure": 3, "unclassified": 6},
        "agreement_table": {"counts_by_agreement": {1: 10, 2: 4}, "by_detector": {"anomaly": 14}},
        "operating_points": {
            "supervised": {"threshold": 0.5, "detection_rate": 0.0, "false_call_rate": 0.0}
        },
    }
    md = build_false_call_report_md(summary_data)
    assert "Real data (ML-NDT)" in md
    assert "pre-registered thresholds are unchanged" in md
    assert "exploratory" in md
    assert "Agreement Count" in md


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


def _setup_stubs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Generate fake ML-NDT dataset and stub model checkpoints."""
    root = tmp_path / "data"
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

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)
    torch.save(ConvAE2D().state_dict(), models_dir / "anomaly_2d.pt")
    for crack in ("crack_1.6", "crack_4.0", "crack_8.6"):
        save_model(MlNdtNet(width=4), models_dir / f"loco_{crack}.pt", {"width": 4})

    out_json = tmp_path / "false_call_analysis.json"
    out_md = tmp_path / "false_call_analysis.md"
    return root, models_dir, out_json, out_md


def test_run_false_call_analysis_on_stubs(tmp_path: Path) -> None:
    """Verify runner runs on stub models and produces compliant artifacts in tmp_path."""
    from experiments.run_false_call_analysis import (
        FalseCallConfig,
        run_false_call_analysis,
    )

    root, models_dir, out_json, out_md = _setup_stubs(tmp_path)
    cfg = FalseCallConfig(
        root=root,
        models_dir=models_dir,
        out_json=out_json,
        out_md=out_md,
        figures_dir=tmp_path / "figures",
        cases_dir=tmp_path / "cases",
        quick=True,
        device="cpu",
    )
    result = run_false_call_analysis(cfg)

    assert result["label"] == "Real data (ML-NDT)"
    assert result["run_label"] == "pre-registered"
    assert result["detector_config"]["is_default"] is True
    assert len(result["detector_config"]["sha256"]) == 64
    assert out_json.exists()
    assert out_md.exists()
    assert "tradeoff_curves" in result
    for det in ("supervised", "anomaly", "h2", "h3"):
        assert (tmp_path / "figures" / f"tradeoff_{det}.png").exists()


def test_run_false_call_analysis_exploratory_config(tmp_path: Path) -> None:
    """Verify exploratory overrides update run_label and detector thresholds."""
    from experiments.run_false_call_analysis import (
        FalseCallConfig,
        run_false_call_analysis,
    )

    root, models_dir, out_json, out_md = _setup_stubs(tmp_path)
    cfg = FalseCallConfig(
        root=root,
        models_dir=models_dir,
        out_json=out_json,
        out_md=out_md,
        figures_dir=tmp_path / "figures",
        cases_dir=tmp_path / "cases",
        quick=True,
        device="cpu",
        overrides=["supervised.threshold=0.6"],
    )
    result = run_false_call_analysis(cfg)

    assert result["run_label"] == "exploratory"
    assert result["detector_config"]["is_default"] is False
    assert result["thresholds"]["supervised"] == 0.6


def test_cli_help_from_temp_dir(tmp_path: Path) -> None:
    """Verify running false call analysis script with --help from isolated directory succeeds."""
    script = str(Path(__file__).resolve().parents[1] / "experiments" / "run_false_call_analysis.py")
    res = subprocess.run(  # noqa: S603
        [sys.executable, script, "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert res.returncode == 0


def _create_test_items(tmp_path: Path) -> list[tuple[Path, int]]:
    """Generate minimal fake batch files and return item coordinates."""
    batch_dir = tmp_path / "batch_data"
    batch_dir.mkdir(parents=True)
    batch_path = batch_dir / "test_batch.bins"
    flaws = [{"size": "1.6"}, {"size": "4.0"}, {"size": "0.0"}, {"size": "0.0"}]
    labels = [(1.0, 1.6), (1.0, 4.0), (0.0, 0.0), (0.0, 0.0)]
    _make_fake_batch(batch_path, flaws, labels, seed=42)
    return [(batch_path, i) for i in range(len(flaws))]


def test_build_feature_dataset_shapes(tmp_path: Path) -> None:
    """Verify build_feature_dataset extracts correct feature, label, and size arrays."""
    from utriage.ml.physics_features import build_feature_dataset

    items = _create_test_items(tmp_path)
    x_arr, y_arr, s_arr = build_feature_dataset(items)
    assert x_arr.shape == (4, 8)
    assert y_arr.shape == (4,)
    assert s_arr.shape == (4,)

    x_empty, y_empty, s_empty = build_feature_dataset([])
    assert x_empty.shape == (0, 8)
    assert y_empty.shape == (0,)
    assert s_empty.shape == (0,)


def test_fit_fold_classifier_output(tmp_path: Path) -> None:
    """Verify fit_fold_classifier fits StandardScaler and LogisticRegression."""
    from sklearn.preprocessing import StandardScaler

    from utriage.ml.physics_features import FitResult, build_feature_dataset, fit_fold_classifier

    items = _create_test_items(tmp_path)
    x_tr, y_tr, _ = build_feature_dataset(items)
    scaler = StandardScaler()
    fit_res = fit_fold_classifier(x_tr, y_tr, scaler)

    assert isinstance(fit_res, FitResult)
    assert len(fit_res.coefficients) == 9
    assert "intercept" in fit_res.coefficients


@pytest.mark.parametrize("empty_test", [False, True], ids=["with_items", "empty_items"])
def test_score_physics_classifier_predictions(tmp_path: Path, empty_test: bool) -> None:
    """Verify score_physics_classifier outputs valid probabilities for fold test items."""
    from utriage.ml.loco import LocoFold
    from utriage.ml.physics_features import score_physics_classifier, score_physics_fold

    items = _create_test_items(tmp_path)
    fold = LocoFold(held_out="crack_1.6", train=items, val=[], test=items)
    test_items = [] if empty_test else items
    scores = score_physics_classifier(fold, test_items)

    expected_len = 0 if empty_test else len(items)
    assert len(scores) == expected_len
    if not empty_test:
        assert all(0.0 <= float(s) <= 1.0 for s in scores)
    assert score_physics_fold is score_physics_classifier
