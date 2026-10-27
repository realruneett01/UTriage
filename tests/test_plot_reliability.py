"""Tests for reliability diagram plotting (scripts/plot_reliability.py).

Verifies matrix parsing, skipping of unavailable and curve-less cells,
null bin omission, file generation per row and column group, and CLI execution.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from plot_reliability import (  # noqa: E402
    classify_column_group,
    extract_curve,
    load_matrix,
    main,
    plot_all_reliability,
)


def _make_cell(
    confs: list[float | None] | None = None,
    fracs: list[float | None] | None = None,
    ece: float | None = 0.05,
    available: bool = True,
) -> dict[str, Any]:
    """Helper to build a standardized cell dict for testing."""
    if not available:
        return {"available": False, "reason": "evaluation unavailable"}
    if confs is None or fracs is None:
        return {"available": True, "reliability": {"counts": [100]}, "ece": ece}
    return {
        "available": True,
        "ece": ece,
        "reliability": {
            "bin_edges": [0.0, 0.5, 1.0],
            "mean_conf": confs,
            "frac_pos": fracs,
            "counts": [50, 50],
        },
    }


def _write_json(path: Path, data: dict[str, Any]) -> Path:
    """Helper to write test JSON dictionary to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return path


def _assert_file_list_matches(paths: list[Path], expected_names: set[str]) -> None:
    """Helper to assert created paths match expected filenames and are non-empty."""
    actual_names = {p.name for p in paths}
    assert actual_names == expected_names
    for p in paths:
        assert p.is_file()
        assert p.stat().st_size > 0


@pytest.mark.parametrize(
    ("col_meta", "expected_group"),
    [
        ({"id": "col1", "data_label": "Real data (ML-NDT)"}, "real"),
        ({"id": "col2", "data_label": "Real data (NDT_ML_Flaw)"}, "real"),
        ({"id": "col3", "data_label": "Synthetic data"}, "synthetic"),
        ({"id": "sim_freq_10mhz"}, "synthetic"),
        ({"id": "mlndt_loco_1.6"}, "real"),
        ({"id": "custom", "group": "sim_artefact"}, "synthetic"),
    ],
)
def test_classify_column_group(col_meta: dict[str, Any], expected_group: str) -> None:
    """Verify column metadata correctly resolves to real or synthetic."""
    assert classify_column_group(col_meta) == expected_group


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        (([0.1, 0.9], [0.2, 0.8], 0.04), (2, 0.04)),
        (([0.2, None, 0.8], [0.1, None, 0.9], 0.02), (2, 0.02)),
        (([0.5], [0.6], None), (1, None)),
    ],
)
def test_extract_curve_valid(
    inputs: tuple[list[float | None], list[float | None], float | None],
    expected: tuple[int, float | None],
) -> None:
    """Valid cells produce CurveData with null bins skipped and matching ECE."""
    confs, fracs, ece = inputs
    expected_len, expected_ece = expected
    cell = _make_cell(confs=confs, fracs=fracs, ece=ece)
    curve = extract_curve("col_test", cell)
    assert curve is not None
    assert len(curve.mean_conf) == expected_len
    assert len(curve.frac_pos) == expected_len
    assert curve.ece == expected_ece


@pytest.mark.parametrize(
    "invalid_cell",
    [
        _make_cell(available=False),
        _make_cell(confs=None, fracs=None),
        _make_cell(confs=[None, None], fracs=[None, None]),
        {"available": True, "reliability": "not_a_dict"},
        "string_instead_of_dict",
        None,
    ],
)
def test_extract_curve_skipped(invalid_cell: Any) -> None:
    """Unavailable, curve-less, or all-null cells return None."""
    assert extract_curve("col_invalid", invalid_cell) is None


def test_nothing_written_when_no_curves_exist(tmp_path: Path) -> None:
    """When a matrix has no valid curves, no diagram files are written."""
    matrix = {
        "columns": [
            {"id": "c_unavail", "data_label": "Real data"},
            {"id": "c_curveless", "data_label": "Synthetic data"},
        ],
        "rows": {
            "model_empty": {
                "c_unavail": _make_cell(available=False),
                "c_curveless": _make_cell(confs=None, fracs=None),
            }
        },
    }
    written = plot_all_reliability(matrix, tmp_path)
    assert len(written) == 0
    assert list(tmp_path.glob("*.png")) == []


@pytest.mark.parametrize(
    ("row_name", "cells", "expected_files"),
    [
        (
            "model_dual",
            {
                "col_real": _make_cell(confs=[0.2, 0.8], fracs=[0.3, 0.7]),
                "col_syn": _make_cell(confs=[0.1, 0.9], fracs=[0.2, 0.8]),
            },
            {"reliability_model_dual_real.png", "reliability_model_dual_synthetic.png"},
        ),
        (
            "model_real_only",
            {
                "col_real": _make_cell(confs=[0.4], fracs=[0.5]),
                "col_syn": _make_cell(available=False),
            },
            {"reliability_model_real_only.png"},
        ),
        (
            "model_syn_only",
            {
                "col_real": _make_cell(confs=None, fracs=None),
                "col_syn": _make_cell(confs=[0.3], fracs=[0.4]),
            },
            {"reliability_model_syn_only.png"},
        ),
    ],
)
def test_files_written_per_row_and_group(
    tmp_path: Path,
    row_name: str,
    cells: dict[str, Any],
    expected_files: set[str],
) -> None:
    """Files are named by row, splitting into _real and _synthetic only when both exist."""
    matrix = {
        "columns": [
            {"id": "col_real", "data_label": "Real data"},
            {"id": "col_syn", "data_label": "Synthetic data"},
        ],
        "rows": {row_name: cells},
    }
    written = plot_all_reliability(matrix, tmp_path)
    _assert_file_list_matches(written, expected_files)


def test_main_cli_execution(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI main parses matrix JSON, writes diagrams, and prints written file paths."""
    matrix_data = {
        "columns": [
            {"id": "col_real", "data_label": "Real data (ML-NDT)"},
            {"id": "col_syn", "data_label": "Synthetic data"},
        ],
        "rows": {
            "model_a": {
                "col_real": _make_cell(confs=[0.2, 0.8], fracs=[0.2, 0.8], ece=0.01),
                "col_syn": _make_cell(confs=[0.3, 0.7], fracs=[0.4, 0.6], ece=0.03),
            }
        },
    }
    matrix_file = _write_json(tmp_path / "generalization_matrix.json", matrix_data)
    out_dir = tmp_path / "figures"

    written = main([str(matrix_file), str(out_dir)])
    captured = capsys.readouterr()

    assert len(written) == 2
    for p in written:
        assert str(p) in captured.out
        assert p.is_file()


def test_load_matrix_file_and_missing(tmp_path: Path) -> None:
    """load_matrix loads valid JSON and raises FileNotFoundError on missing files."""
    test_file = _write_json(tmp_path / "matrix.json", {"columns": [], "rows": {}})
    loaded = load_matrix(test_file)
    assert loaded["rows"] == {}

    with pytest.raises(FileNotFoundError):
        load_matrix(tmp_path / "nonexistent" / "matrix.json")
