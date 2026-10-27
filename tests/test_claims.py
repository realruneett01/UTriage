"""Tests for quantitative claims and evidence limits (project plan Phase 8)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pytest  # noqa: E402

from utriage.report.claims import (  # noqa: E402
    FORBIDDEN,
    Claim,
    check_forbidden,
    get_path,
    render_markdown,
)


def _make_cell(k: int, n: int, fc_rate: float) -> dict[str, Any]:
    """Helper to create a standard pod.metrics cell dictionary."""
    rate = k / n if n > 0 else 0.0
    return {
        "detection": {
            "n": n,
            "k": k,
            "rate": rate,
            "ci_low": max(0.0, rate - 0.05),
            "ci_high": min(1.0, rate + 0.05),
            "method": "Clopper-Pearson two-sided 95",
        },
        "false_calls": {"rate": fc_rate, "n_false": int(fc_rate * 10), "n_scans": 10},
        "ece": 0.05,
        "counts": {"images": n, "distinct_physical_flaws": 1, "hits": k, "total_flaws_tested": n},
    }


def _make_supervised_cells() -> dict[str, Any]:
    """Helper for supervised row cells."""
    return {
        "mlndt_loco_crack_1.6": _make_cell(400, 423, 0.0),
        "mlndt_loco_crack_4.0": _make_cell(390, 421, 0.0),
        "mlndt_loco_crack_8.6": _make_cell(370, 378, 0.0),
        "cross_ndt_ml_flaw": _make_cell(150, 200, 0.10),
        "sim_freq_10mhz": _make_cell(340, 400, 0.05),
        "sim_material_third": _make_cell(330, 400, 0.08),
        "sim_noise_above_range": _make_cell(280, 400, 0.35),
    }


@pytest.fixture
def fake_matrix_data() -> dict[str, Any]:
    """Return dictionary structure matching generalization_matrix.json contract."""
    cols = [
        {"id": "mlndt_loco_crack_1.6", "group": "loco"},
        {"id": "mlndt_loco_crack_4.0", "group": "loco"},
        {"id": "mlndt_loco_crack_8.6", "group": "loco"},
        {"id": "cross_ndt_ml_flaw", "group": "cross"},
        {"id": "sim_freq_10mhz", "group": "sim"},
        {"id": "sim_material_third", "group": "sim"},
        {"id": "sim_noise_above_range", "group": "sim"},
    ]
    anom = {
        "mlndt_loco_crack_1.6": _make_cell(350, 423, 0.02),
        "mlndt_loco_crack_4.0": _make_cell(360, 421, 0.02),
        "mlndt_loco_crack_8.6": _make_cell(340, 378, 0.02),
    }
    rules = {
        "sim_freq_10mhz": _make_cell(300, 400, 0.03),
        "sim_material_third": _make_cell(280, 400, 0.04),
        "sim_noise_above_range": _make_cell(240, 400, 0.31),
    }
    union = {c["id"]: {"holds": True, "n_violations": 0} for c in cols}
    return {
        "columns": cols,
        "rows": {"supervised": _make_supervised_cells(), "anomaly": anom, "rules": rules},
        "union_property": union,
    }


def _write_support_files(tmp_path: Path) -> None:
    """Write hardware, data inventory and triage test metrics."""
    hw_data = {
        "inference_machine": {"processor": "Intel Core i5-1135G7"},
        "cpu_latency_ms": {
            "mlndt": {"median_ms": 14.203, "p95_ms": 20.954, "median": 14.203, "p95": 20.954}
        },
    }
    (tmp_path / "hardware.json").write_text(json.dumps(hw_data), encoding="utf-8")

    inv_text = (
        "# Real-data evidence limits\n\n"
        "| Dataset | Licence | Material | Technique | Distinct Physical Flaws | Download Status |\n"
        "|---|---|---|---|---|---|\n"
        "| ML-NDT | LGPL-3.0 | 316L steel | PAUT | 3 | downloaded |\n"
        "| Strathclyde-FMC | CC BY 4.0 | 316L steel | FMC | 1 | not_downloaded |\n"
        "| NDT_ML_Flaw | LGPL-3.0 | dissimilar weld | PAUT | 6 | downloaded |\n\n"
        "Across the 6 accepted datasets, 3 datasets have verified counts "
        "totaling 10 distinct physical flaws.\n"
    )
    (tmp_path / "data_inventory.md").write_text(inv_text, encoding="utf-8")
    triage_data = {"union_violations": 0, "property_test": "10,000 examples"}
    (tmp_path / "triage_metrics.json").write_text(json.dumps(triage_data), encoding="utf-8")


@pytest.fixture
def fake_results_dir(tmp_path: Path, fake_matrix_data: dict[str, Any]) -> Path:
    """Write tiny fake results files into a temporary directory."""
    mat_path = tmp_path / "generalization_matrix.json"
    mat_path.write_text(json.dumps(fake_matrix_data), encoding="utf-8")
    leak_data = {
        "shared_cracks_train_test": ["crack_1.6", "crack_4.0", "crack_8.6"],
        "leakage_by_physical_crack": True,
    }
    (tmp_path / "leakage_audit.json").write_text(json.dumps(leak_data), encoding="utf-8")
    _write_support_files(tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    ("data", "path", "expected"),
    [
        ({"a": {"b": {"c": 42}}}, "a.b.c", 42),
        ({"columns": [{"id": "c0"}, {"id": "c1"}]}, "columns.1.id", "c1"),
        (
            {"rows": {"mlndt_loco_crack_1.6": {"rate": 0.95}}},
            "rows.mlndt_loco_crack_1.6.rate",
            0.95,
        ),
        ({"a": {"b": 1}, "a.b": 2}, "a.b", 2),
        ({"items": [10, 20, 30]}, "items.-1", 30),
        ({"val": 99}, "", {"val": 99}),
    ],
)
def test_get_path_valid(data: Any, path: str, expected: Any) -> None:
    """Verify get_path retrieves values for dotted keys and list indices."""
    assert get_path(data, path) == expected


@pytest.mark.parametrize(
    ("data", "path", "err_type"),
    [
        ({"a": {"b": 1}}, "a.c", KeyError),
        ({"items": [1, 2]}, "items.5", IndexError),
        ({"items": [1, 2]}, "items.not_a_number", IndexError),
        (42, "foo.bar", KeyError),
    ],
)
def test_get_path_errors(data: Any, path: str, err_type: type[Exception]) -> None:
    """Verify get_path raises expected errors on missing keys or bad indices."""
    with pytest.raises(err_type):
        get_path(data, path)


@pytest.mark.parametrize(
    "phrase",
    [
        "perfect",
        "100%",
        "for MIDHANI",
        "validated on plant data",
        "qualified",
        "deployed",
        "generalizes to all",
    ],
)
def test_check_forbidden_finds_phrases(phrase: str) -> None:
    """Verify check_forbidden detects each forbidden phrase case-insensitively."""
    assert phrase in FORBIDDEN
    variants = [phrase.lower(), phrase.upper(), phrase.title()]
    for variant in variants:
        sample = f"System claim: {variant} achieved in testing."
        found = check_forbidden(sample)
        assert len(found) >= 1
        assert any(f.lower() == phrase.lower() for f in found)


def test_check_forbidden_clean_text() -> None:
    """Verify check_forbidden returns empty list on compliant text."""
    clean = "Independent prototype evaluated on public dataset and simulated bars."
    assert check_forbidden(clean) == []


@pytest.mark.parametrize(
    ("text", "scope"),
    [
        ("Model achieves 100% detection.", "public dataset"),
        ("Independent prototype.", "validated on plant data"),
        ("Deployed on machine.", "advisory"),
    ],
)
def test_claim_forbidden_phrase_raises(text: str, scope: str) -> None:
    """Verify Claim raises ValueError when text or scope contains forbidden words."""
    with pytest.raises(ValueError, match="Forbidden phrase"):
        Claim(
            id="test_claim",
            text=text,
            evidence_file="results/test.json",
            evidence_path="path",
            scope_limit=scope,
            numbers={"val": 1.0},
        )


def test_build_claims_all_inputs(fake_results_dir: Path) -> None:
    """Verify build_claims constructs all 12 claims matching exact contract."""
    from utriage.report.claims import build_claims

    expected_ids = [
        "leakage",
        "loco_crack_1.6",
        "loco_crack_4.0",
        "loco_crack_8.6",
        "anomaly_crack_1.6",
        "anomaly_crack_4.0",
        "anomaly_crack_8.6",
        "cross_dataset",
        "sim_regions",
        "union_property",
        "latency",
        "evidence_limit",
    ]
    claims = build_claims(fake_results_dir)
    claim_ids = [c.id for c in claims]
    assert claim_ids == expected_ids
    assert len(claims) == 12


def test_build_claims_cross_dataset_missing_skipped(
    fake_results_dir: Path, fake_matrix_data: dict[str, Any]
) -> None:
    """Verify cross_dataset claim is skipped with a reason when unavailable."""
    from utriage.report.claims import build_claims

    fake_matrix_data["rows"]["supervised"]["cross_ndt_ml_flaw"] = {
        "available": False,
        "reason": "data not downloaded",
    }
    mat_path = fake_results_dir / "generalization_matrix.json"
    mat_path.write_text(json.dumps(fake_matrix_data), encoding="utf-8")

    claims = build_claims(fake_results_dir)
    claim_ids = [c.id for c in claims]
    assert "cross_dataset" not in claim_ids
    assert len(claims) == 11


def test_numbers_formatted_in_claim_text_match_evidence(fake_results_dir: Path) -> None:
    """Verify every formatted number in claim text matches raw value at evidence_path."""
    from utriage.report.claims import build_claims

    claims = build_claims(fake_results_dir)
    for c in claims:
        assert len(c.numbers) > 0
        for key, val in c.numbers.items():
            candidates = [f"{val:.2f}", f"{val:.1f}", str(int(val))]
            assert any(cand in c.text for cand in candidates), (
                f"Value {val} ({key}) not found in claim text: '{c.text}'"
            )


def test_render_markdown_has_one_row_per_claim(fake_results_dir: Path) -> None:
    """Verify markdown output has table header and exactly one row per claim."""
    from utriage.report.claims import build_claims

    claims = build_claims(fake_results_dir)
    md = render_markdown(claims)
    lines = [ln.strip() for ln in md.strip().splitlines() if ln.startswith("|")]
    assert lines[0] == "| Claim | Evidence | Scope limit |"
    assert lines[1] == "|---|---|---|"
    data_rows = lines[2:]
    assert len(data_rows) == len(claims)


def test_make_claims_script(fake_results_dir: Path) -> None:
    """Verify make_claims script writes claims.json and claims.md."""
    scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from make_claims import generate_claims

    data = generate_claims(fake_results_dir)
    assert len(data) == 12
    json_path = fake_results_dir / "claims.json"
    md_path = fake_results_dir / "claims.md"
    assert json_path.exists()
    assert md_path.exists()
