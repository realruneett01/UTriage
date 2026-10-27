"""Tests for generalization matrix construction and reporting (project plan Phase 8).

Verifies matrix assembly, missing result handling, union property verification,
matrix plotting, and run_phase8 orchestration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pytest  # noqa: E402
from experiments.run_phase8 import Phase8Config, parse_args, run_phase8  # noqa: E402

from utriage.report.matrix import build_matrix, plot_matrix  # noqa: E402


def _make_dummy_cell(k: int = 10, n: int = 10, n_false: int = 0) -> dict[str, Any]:
    """Create a minimal valid cell dict for testing."""
    return {
        "detection": {
            "n": n,
            "k": k,
            "rate": float(k / n) if n > 0 else 0.0,
            "ci_low": 0.8,
            "ci_high": 1.0,
            "method": "Clopper-Pearson two-sided 95",
        },
        "false_calls": {"rate": float(n_false), "n_false": n_false, "n_scans": 10},
        "ece": 0.05,
        "reliability": {"counts": [n]},
        "pod": {"fittable": False, "a90_95": None},
        "n_images": n + 10,
        "n_physical_flaws": 1,
        "counts": {
            "images": n + 10,
            "distinct_physical_flaws": 1,
            "hits": k,
            "total_flaws_tested": n,
        },
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Helper to write indented JSON to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _create_fake_loco(res_dir: Path) -> None:
    """Write fake loco_metrics.json for testing."""
    loco = {
        "label": "Real data (ML-NDT)",
        "folds": [
            {
                "held_out": f"crack_{size}",
                "n_test_flawed": 20,
                "n_test_unflawed": 10,
                "detections": {"n_hit": 18, "n_flawed": 20, "rate": 0.9},
                "false_calls": {"n_false": 1, "n_unflawed": 10, "rate": 0.1, "upper95": 0.2},
                "pod": {"fittable": False, "a90_95": None},
            }
            for size in ("1.6", "4.0", "8.6")
        ],
        "official_validation": {
            "n_flawed": 25,
            "n_unflawed": 15,
            "detections": {"n_hit": 25, "n_flawed": 25, "rate": 1.0},
            "false_calls": {"n_false": 0, "n_unflawed": 15, "rate": 0.0, "upper95": 0.05},
            "pod": {"fittable": False, "a90_95": None},
        },
    }
    _write_json(res_dir / "loco_metrics.json", loco)


def _create_fake_anomaly(res_dir: Path) -> None:
    """Write fake anomaly_metrics.json for testing."""
    anomaly = {
        "real": {
            "per_crack": {
                f"crack_{s}": _make_dummy_cell(k=15, n=20, n_false=1) for s in ("1.6", "4.0", "8.6")
            }
        },
        "synthetic": {"cell": _make_dummy_cell(k=90, n=100, n_false=2)},
    }
    _write_json(res_dir / "anomaly_metrics.json", anomaly)


def _create_fake_sim(res_dir: Path) -> None:
    """Write fake sim_generalization.json for testing."""
    sim = {
        "test_sets": {
            "in_distribution": {
                "rules": _make_dummy_cell(k=80, n=100, n_false=0),
                "ml_1d": _make_dummy_cell(k=85, n=100, n_false=1),
                "union": _make_dummy_cell(k=90, n=100, n_false=1),
            },
            "freq_10mhz": {
                "rules": _make_dummy_cell(k=70, n=100, n_false=0),
                "ml_1d": _make_dummy_cell(k=75, n=100, n_false=2),
                "union": _make_dummy_cell(k=80, n=100, n_false=2),
            },
            "artefact_bubble": {
                "rules": _make_dummy_cell(k=65, n=100, n_false=1),
                "ml_1d": _make_dummy_cell(k=70, n=100, n_false=3),
                "union": _make_dummy_cell(k=75, n=100, n_false=3),
            },
        }
    }
    _write_json(res_dir / "sim_generalization.json", sim)


@pytest.fixture
def fake_results_dir(tmp_path: Path) -> Path:
    """Populate minimal fake results files in tmp_path."""
    res_dir = tmp_path / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    _create_fake_loco(res_dir)
    _create_fake_anomaly(res_dir)
    _create_fake_sim(res_dir)
    cross = {
        "models": {
            "mlndt_best.pt": {"overall": _make_dummy_cell(k=40, n=50, n_false=2)},
            "anomaly_2d.pt": {"overall": _make_dummy_cell(k=35, n=50, n_false=3)},
        }
    }
    _write_json(res_dir / "cross_dataset_metrics.json", cross)
    return res_dir


def test_build_matrix_structure_and_keys(fake_results_dir: Path) -> None:
    """Verify matrix dictionary contains all mandatory keys and row IDs."""
    mat = build_matrix(fake_results_dir)
    for expected_key in ("label_note", "columns", "rows", "sources", "union_property"):
        assert expected_key in mat
    for model_id in ("rules", "supervised", "anomaly", "fused"):
        assert model_id in mat["rows"]


def test_build_matrix_columns_and_groups(fake_results_dir: Path) -> None:
    """Check required column IDs and valid group classifications."""
    mat = build_matrix(fake_results_dir)
    col_map = {c["id"]: c for c in mat["columns"]}
    assert "mlndt_loco_crack_1.6" in col_map
    assert "mlndt_validation" in col_map
    assert "cross_ndt_ml_flaw" in col_map
    assert "sim_in_distribution" in col_map
    assert "sim_artefact_bubble" in col_map

    assert col_map["mlndt_validation"]["leaky"] is True
    assert col_map["mlndt_loco_crack_1.6"]["leaky"] is False
    assert col_map["sim_in_distribution"]["group"] == "sim_region"
    assert col_map["sim_artefact_bubble"]["group"] == "sim_artefact"


def test_build_matrix_rules_and_fused_real_unavailable(fake_results_dir: Path) -> None:
    """Real weld PAUT columns are unavailable for rules and fused models."""
    mat = build_matrix(fake_results_dir)
    rules_row = mat["rows"]["rules"]
    fused_row = mat["rows"]["fused"]
    for col_id in ("mlndt_loco_crack_1.6", "mlndt_validation", "cross_ndt_ml_flaw"):
        assert rules_row[col_id]["available"] is False
        assert "bar immersion A-scans" in rules_row[col_id]["reason"]
        assert fused_row[col_id]["available"] is False
        assert "fused scores not computed" in fused_row[col_id]["reason"]


def test_build_matrix_sim_cells_available(fake_results_dir: Path) -> None:
    """Sim columns populate rules, supervised, anomaly (in-dist), and fused rows."""
    mat = build_matrix(fake_results_dir)
    rows = mat["rows"]
    assert rows["rules"]["sim_in_distribution"]["detection"]["k"] == 80
    assert rows["supervised"]["sim_in_distribution"]["detection"]["k"] == 85
    assert rows["fused"]["sim_in_distribution"]["detection"]["k"] == 90
    assert rows["anomaly"]["sim_in_distribution"]["detection"]["k"] == 90
    assert rows["anomaly"]["sim_freq_10mhz"]["available"] is False


def test_union_property_satisfied(fake_results_dir: Path) -> None:
    """Union property holds when detections and false calls satisfy monotonic bounds."""
    mat = build_matrix(fake_results_dir)
    u_prop = mat["union_property"]
    assert u_prop["sim_in_distribution"]["holds"] is True
    assert u_prop["sim_in_distribution"]["n_violations"] == 0
    assert u_prop["mlndt_loco_crack_1.6"]["available"] is False


def test_union_property_flags_violation(fake_results_dir: Path) -> None:
    """Union property correctly flags violation when union detection rate drops."""
    sim_path = fake_results_dir / "sim_generalization.json"
    sim_data = json.loads(sim_path.read_text(encoding="utf-8"))
    # Introduce violation: union detects fewer than ml_1d
    sim_data["test_sets"]["in_distribution"]["union"]["detection"]["k"] = 50
    _write_json(sim_path, sim_data)

    mat = build_matrix(fake_results_dir)
    u_prop = mat["union_property"]["sim_in_distribution"]
    assert u_prop["holds"] is False
    assert u_prop["n_violations"] >= 1


def test_build_matrix_missing_files_graceful(tmp_path: Path) -> None:
    """Missing input result files yield available: false without raising errors."""
    empty_results = tmp_path / "empty_results"
    empty_results.mkdir()
    mat = build_matrix(empty_results)
    assert mat["rows"]["supervised"]["mlndt_loco_crack_1.6"]["available"] is False
    assert mat["rows"]["supervised"]["cross_ndt_ml_flaw"]["available"] is False
    assert mat["rows"]["anomaly"]["cross_ndt_ml_flaw"]["available"] is False


def test_plot_matrix_creates_png(fake_results_dir: Path, tmp_path: Path) -> None:
    """plot_matrix renders heatmap image and returns valid path."""
    mat = build_matrix(fake_results_dir)
    out_png = tmp_path / "test_heatmap.png"
    result_path = plot_matrix(mat, out_png)
    assert result_path.exists()
    assert result_path.stat().st_size > 0


def test_run_phase8_assemble_only(monkeypatch: pytest.MonkeyPatch, fake_results_dir: Path) -> None:
    """run_phase8 in --assemble-only mode skips experiments and calls matrix build."""
    mock_run = MagicMock()
    monkeypatch.setattr("subprocess.run", mock_run)

    cfg = Phase8Config(assemble_only=True, results_dir=fake_results_dir)
    run_phase8(cfg)

    # Subprocess should not be called for experiments during assemble_only
    exp_calls = [c for c in mock_run.call_args_list if "experiments" in str(c)]
    assert len(exp_calls) == 0

    assert (fake_results_dir / "generalization_matrix.json").exists()
    assert (fake_results_dir / "figures" / "generalization_matrix.png").exists()


def test_parse_args_defaults_and_flags() -> None:
    """CLI argument parser handles flags and skip list correctly."""
    cfg = parse_args(["--assemble-only", "--quick", "--skip", "run_loco.py"])
    assert cfg.assemble_only is True
    assert cfg.quick is True
    assert "run_loco.py" in cfg.skip
