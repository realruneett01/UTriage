"""Tests for quantitative claims and evidence limits (project plan Phase 8)."""

from __future__ import annotations

import dataclasses
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
        "sim_in_distribution": _make_cell(355, 400, 0.02),
        "sim_freq_10mhz": _make_cell(340, 400, 0.05),
        "sim_material_third": _make_cell(330, 400, 0.08),
        "sim_noise_above_range": _make_cell(280, 400, 0.35),
    }


def _make_rules_cells() -> dict[str, Any]:
    """Helper for rules row cells."""
    return {
        "sim_in_distribution": _make_cell(309, 400, 0.03),
        "sim_freq_10mhz": _make_cell(300, 400, 0.03),
        "sim_material_third": _make_cell(280, 400, 0.04),
        "sim_noise_above_range": _make_cell(240, 400, 0.31),
    }


def _make_matrix_columns() -> list[dict[str, Any]]:
    """Helper returning standard mock column descriptors."""
    col_ids = [
        "mlndt_loco_crack_1.6",
        "mlndt_loco_crack_4.0",
        "mlndt_loco_crack_8.6",
        "cross_ndt_ml_flaw",
        "sim_in_distribution",
        "sim_freq_10mhz",
        "sim_material_third",
        "sim_noise_above_range",
    ]
    return [{"id": cid, "group": "sim" if cid.startswith("sim_") else "real"} for cid in col_ids]


@pytest.fixture
def fake_matrix_data() -> dict[str, Any]:
    """Return dictionary structure matching generalization_matrix.json contract."""
    cols = _make_matrix_columns()
    anom = {
        "mlndt_loco_crack_1.6": _make_cell(350, 423, 0.02),
        "mlndt_loco_crack_4.0": _make_cell(360, 421, 0.02),
        "mlndt_loco_crack_8.6": _make_cell(340, 378, 0.02),
    }
    union = {
        c["id"]: (
            {"holds": True, "n_violations": 0}
            if c["id"].startswith("sim_")
            else {"available": False, "reason": "rules unavailable"}
        )
        for c in cols
    }
    return {
        "columns": cols,
        "rows": {
            "supervised": _make_supervised_cells(),
            "anomaly": anom,
            "rules": _make_rules_cells(),
        },
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
    """Verify build_claims constructs all 13 claims matching exact contract."""
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
        "sim_in_distribution",
        "sim_regions",
        "union_property",
        "latency",
        "evidence_limit",
    ]
    claims = build_claims(fake_results_dir)
    claim_ids = [c.id for c in claims]
    assert claim_ids == expected_ids
    assert len(claims) == 13


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
    assert len(claims) == 12


def test_numbers_formatted_in_claim_text_match_evidence(fake_results_dir: Path) -> None:
    """Verify every formatted number in claim text matches raw value at evidence_path."""
    from utriage.report.claims import build_claims

    claims = build_claims(fake_results_dir)
    for c in claims:
        assert len(c.numbers) > 0
        for key, val in c.numbers.items():
            candidates = [f"{val:.3f}", f"{val:.2f}", f"{val:.1f}", str(int(val))]
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
    assert len(data) == 13
    json_path = fake_results_dir / "claims.json"
    md_path = fake_results_dir / "claims.md"
    assert json_path.exists()
    assert md_path.exists()


def test_make_claims_main(fake_results_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify make_claims main writes claims matching build_claims output."""
    scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from make_claims import main

    from utriage.report.claims import build_claims

    main([str(fake_results_dir)])
    captured = capsys.readouterr().out
    assert "Generated 13 claims" in captured
    claims = build_claims(fake_results_dir)
    expected_json = [dataclasses.asdict(c) for c in claims]
    expected_md = render_markdown(claims)
    written_json = json.loads((fake_results_dir / "claims.json").read_text(encoding="utf-8"))
    written_md = (fake_results_dir / "claims.md").read_text(encoding="utf-8")
    assert written_json == expected_json
    assert written_md == expected_md


def test_cross_dataset_high_false_calls(
    fake_results_dir: Path, fake_matrix_data: dict[str, Any]
) -> None:
    """Verify high false-call rate cross-dataset claim framing."""
    from utriage.report.claims import build_claims

    cell = {
        "detection": {"n": 3442, "k": 3442, "rate": 1.0, "ci_low": 0.999, "ci_high": 1.0},
        "false_calls": {"rate": 1.0, "n_false": 3558, "n_scans": 3558},
    }
    fake_matrix_data["rows"]["supervised"]["cross_ndt_ml_flaw"] = cell
    mat_path = fake_results_dir / "generalization_matrix.json"
    mat_path.write_text(json.dumps(fake_matrix_data), encoding="utf-8")

    claims = build_claims(fake_results_dir)
    claim = next(c for c in claims if c.id == "cross_dataset")
    assert "flagged (nearly) every image" in claim.text
    assert "3442/3442" in claim.text
    assert "3558/3558" in claim.text
    assert claim.text.endswith("no cross-dataset transfer under the documented domain mapping.")
    assert "detection success" not in claim.text.lower()
    assert not claim.text.startswith("Cross-dataset NDT_ML_Flaw: detection")


def test_rate_formatting_small_values(
    fake_results_dir: Path, fake_matrix_data: dict[str, Any]
) -> None:
    """Verify rates below 0.05 use 3 decimals and 19/4014 is not 0.00."""
    from utriage.report.claims import build_claims

    k, n = 19, 4014
    cell = {
        "detection": {"n": n, "k": k, "rate": k / n, "ci_low": 0.003, "ci_high": 0.007},
        "false_calls": {"rate": 0.002, "n_false": 2, "n_scans": 1000},
    }
    fake_matrix_data["rows"]["supervised"]["mlndt_loco_crack_1.6"] = cell
    mat_path = fake_results_dir / "generalization_matrix.json"
    mat_path.write_text(json.dumps(fake_matrix_data), encoding="utf-8")

    claims = build_claims(fake_results_dir)
    claim = next(c for c in claims if c.id == "loco_crack_1.6")
    assert "0.005" in claim.text
    assert "rate 0.00," not in claim.text


@pytest.mark.parametrize(
    ("cpu_name", "processor", "expected_in_text"),
    [
        ("Intel Core i7-12700H", "Intel64 Family 6 Model 198", "Intel Core i7-12700H"),
        (None, "Intel Core i5-1135G7", "Intel Core i5-1135G7"),
    ],
)
def test_latency_cpu_name(
    fake_results_dir: Path,
    cpu_name: str | None,
    processor: str,
    expected_in_text: str,
) -> None:
    """Verify latency claim prefers cpu_name over processor."""
    from utriage.report.claims import build_claims

    mach: dict[str, Any] = {"processor": processor}
    if cpu_name is not None:
        mach["cpu_name"] = cpu_name
    hw = {
        "inference_machine": mach,
        "cpu_latency_ms": {"mlndt": {"median_ms": 12.0, "p95_ms": 18.0}},
    }
    (fake_results_dir / "hardware.json").write_text(json.dumps(hw), encoding="utf-8")
    claims = build_claims(fake_results_dir)
    claim = next(c for c in claims if c.id == "latency")
    assert expected_in_text in claim.text
    assert claim.scope_limit == "this development machine, not the target i5 laptop"


@pytest.mark.parametrize(
    ("used_line", "expected_fragment"),
    [
        ("Used in evaluation: 3 distinct physical flaws\n", "10 listed vs 3 used in evaluation."),
        ("", "10 listed."),
    ],
)
def test_evidence_limit_used_vs_listed(
    fake_results_dir: Path, used_line: str, expected_fragment: str
) -> None:
    """Verify evidence limit claim handles both listed and used flaw counts."""
    from utriage.report.claims import build_claims

    inv_text = (
        "Across the 6 accepted datasets, 3 datasets have verified counts "
        f"totaling 10 distinct physical flaws.\n{used_line}"
    )
    (fake_results_dir / "data_inventory.md").write_text(inv_text, encoding="utf-8")
    claims = build_claims(fake_results_dir)
    claim = next(c for c in claims if c.id == "evidence_limit")
    assert expected_fragment in claim.text


def test_union_property_scope_and_columns(fake_results_dir: Path) -> None:
    """Verify union property claim scope and column count."""
    from utriage.report.claims import build_claims

    claims = build_claims(fake_results_dir)
    claim = next(c for c in claims if c.id == "union_property")
    assert claim.scope_limit == (
        "synthetic simulator columns only (the real-image columns have no rules baseline)"
    )
    assert "in all 4 synthetic simulator columns" in claim.text


def test_sim_in_distribution_claim(fake_results_dir: Path) -> None:
    """Verify sim_in_distribution claim has ML vs rules detection and false calls."""
    from utriage.report.claims import build_claims

    claims = build_claims(fake_results_dir)
    claim = next(c for c in claims if c.id == "sim_in_distribution")
    assert "Simulator in-distribution" in claim.text
    assert "ML det 0.89 vs rules 0.77" in claim.text
    assert "false calls 0.020 vs 0.030" in claim.text
    assert claim.scope_limit == "Synthetic data, UNVERIFIED constants"


def _make_crackval_fold(
    held_out: str,
    val_crack: str,
    det: tuple[int, int],
    fc: tuple[int, int],
) -> dict[str, Any]:
    """Helper creating a single fold dictionary for loco_crackval_metrics.json."""
    k, n = det
    n_false, n_unflawed = fc
    rate = k / n if n > 0 else 0.0
    fc_rate = n_false / n_unflawed if n_unflawed > 0 else 0.0
    return {
        "held_out": held_out,
        "val_crack": val_crack,
        "detections": {"n_hit": k, "n_flawed": n, "rate": rate},
        "false_calls": {"n_false": n_false, "n_unflawed": n_unflawed, "rate": fc_rate},
        "model_sha256": "fake_sha256_hash",
    }


def _make_crackval_data(*, high_fc: bool = True) -> dict[str, Any]:
    """Helper creating fake loco_crackval_metrics dictionary."""
    fc_1 = 753 if high_fc else 50
    return {
        "label": "Real data (ML-NDT)",
        "protocol": "Leave-one-crack-out with crack-held-out validation",
        "folds": [
            _make_crackval_fold("crack_1.6", "crack_4.0", (4012, 4014), (fc_1, 778)),
            _make_crackval_fold("crack_4.0", "crack_8.6", (2449, 4078), (0, 778)),
            _make_crackval_fold("crack_8.6", "crack_1.6", (3925, 3925), (11, 778)),
        ],
    }


def test_loco_crackval_claim_present_with_numbers(tmp_path: Path) -> None:
    """Verify loco_crackval claim is built from metrics with exact numbers."""
    from utriage.report.claims import build_claims

    data = _make_crackval_data(high_fc=True)
    (tmp_path / "loco_crackval_metrics.json").write_text(json.dumps(data), encoding="utf-8")
    claims = build_claims(tmp_path)
    claim = next((c for c in claims if c.id == "loco_crackval"), None)
    assert claim is not None
    assert claim.evidence_file == "results/loco_crackval_metrics.json"
    assert claim.evidence_path == "folds"
    assert claim.scope_limit == (
        "one weld PAUT public dataset, 3 physical cracks, each fold trains on a single crack"
    )
    for expected_str in (
        "Leave-one-crack-out with crack-held-out early stopping",
        "held-out crack_1.6, validation crack_4.0: detections 4012/4014, false calls 753/778",
        "held-out crack_4.0, validation crack_8.6: detections 2449/4078, false calls 0/778",
        "held-out crack_8.6, validation crack_1.6: detections 3925/3925, false calls 11/778",
    ):
        assert expected_str in claim.text
    assert claim.numbers["crack_1.6_k"] == 4012.0
    assert claim.numbers["crack_1.6_n"] == 4014.0
    assert claim.numbers["crack_1.6_n_false"] == 753.0
    assert claim.numbers["crack_1.6_n_unflawed"] == 778.0
    for key, val in claim.numbers.items():
        candidates = [f"{val:.3f}", f"{val:.2f}", f"{val:.1f}", str(int(val))]
        assert any(cand in claim.text for cand in candidates), f"{key} ({val}) not in text"


def test_loco_crackval_missing_file_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify loco_crackval claim is skipped with printed reason when file missing."""
    from utriage.report.claims import build_claims

    claims = build_claims(tmp_path)
    claim_ids = [c.id for c in claims]
    assert "loco_crackval" not in claim_ids
    captured = capsys.readouterr().err
    assert "Skipping claims requiring loco_crackval_metrics.json: file not found" in captured


@pytest.mark.parametrize(
    ("high_fc", "fold_name", "expect_named"),
    [
        (True, "crack_1.6", True),
        (False, "crack_1.6", False),
    ],
)
def test_loco_crackval_false_calls_swing_sentence(
    tmp_path: Path,
    high_fc: bool,
    fold_name: str,
    expect_named: bool,
) -> None:
    """Verify swing sentence names fold where false calls exceed half clean images."""
    from utriage.report.claims import build_claims

    data = _make_crackval_data(high_fc=high_fc)
    (tmp_path / "loco_crackval_metrics.json").write_text(json.dumps(data), encoding="utf-8")
    claims = build_claims(tmp_path)
    claim = next(c for c in claims if c.id == "loco_crackval")
    assert "Results swing strongly with the choice of early-stopping crack" in claim.text
    if expect_named:
        assert f"false calls exceed half the clean images in fold {fold_name}" in claim.text
    else:
        assert "exceed half" not in claim.text
