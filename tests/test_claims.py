"""Tests for quantitative claims and evidence limits (project plan Phase 8)."""

from __future__ import annotations

import dataclasses
import json
import re
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
    assert "flagged 7000/7000 images" in claim.text
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
        "ML-NDT (316L weld, 3 physical cracks), each fold trains on a single crack"
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


def _make_fake_hybrid_fold(
    crack: str, k_range: tuple[int, int], n: int, n_false: int, n_scans: int
) -> dict[str, Any]:
    """Helper creating a single fold dictionary for fake hybrid_metrics.json."""
    min_k, max_k = k_range
    return {
        "held_out": crack,
        "models": {
            f"loco_{crack}_seed0.pt": {
                "methods": {
                    "h2": {
                        "detection": {"k": min_k, "n": n, "rate": min_k / n},
                        "false_calls": {
                            "n_false": n_false,
                            "n_scans": n_scans,
                            "rate": n_false / n_scans,
                        },
                    }
                }
            },
            f"loco_{crack}_seed1.pt": {
                "methods": {
                    "h2": {
                        "detection": {"k": max_k, "n": n, "rate": max_k / n},
                        "false_calls": {
                            "n_false": n_false,
                            "n_scans": n_scans,
                            "rate": n_false / n_scans,
                        },
                    }
                }
            },
        },
    }


def _make_fake_hybrid_data(run_label: str = "pre-registered") -> dict[str, Any]:
    """Helper creating fake hybrid_metrics dictionary."""
    return {
        "run_label": run_label,
        "folds": {
            "crack_1.6": _make_fake_hybrid_fold("crack_1.6", (3687, 3766), 4014, 7, 778),
            "crack_4.0": _make_fake_hybrid_fold("crack_4.0", (899, 2272), 4078, 7, 778),
            "crack_8.6": _make_fake_hybrid_fold("crack_8.6", (3925, 3925), 3925, 7, 778),
        },
    }


def _make_fake_physics_data(run_label: str = "pre-registered") -> dict[str, Any]:
    """Helper creating fake physics_model_metrics dictionary."""
    return {
        "run_label": run_label,
        "folds": [
            {
                "held_out": "crack_1.6",
                "detections": {"k": 2661, "n": 4014, "rate": 2661 / 4014},
                "false_calls": {"n_false": 0, "n_clean": 778, "rate": 0.0},
            },
            {
                "held_out": "crack_4.0",
                "detections": {"k": 2578, "n": 4078, "rate": 2578 / 4078},
                "false_calls": {"n_false": 0, "n_clean": 778, "rate": 0.0},
            },
            {
                "held_out": "crack_8.6",
                "detections": {"k": 3925, "n": 3925, "rate": 1.0},
                "false_calls": {"n_false": 1, "n_clean": 778, "rate": 1 / 778},
            },
        ],
    }


def test_hybrid_h2_claim_present(tmp_path: Path) -> None:
    """Verify hybrid_h2 claim is constructed with expected numbers and scope."""
    from utriage.report.claims import build_claims

    data = _make_fake_hybrid_data()
    (tmp_path / "hybrid_metrics.json").write_text(json.dumps(data), encoding="utf-8")
    claims = build_claims(tmp_path)
    claim = next((c for c in claims if c.id == "hybrid_h2"), None)
    assert claim is not None
    assert claim.evidence_file == "results/hybrid_metrics.json"
    assert claim.evidence_path == "folds"
    assert claim.scope_limit == (
        "pre-registered ADR 0010, one weld PAUT public dataset, 3 physical cracks, seed-sensitive"
    )
    for expected in (
        "crack_1.6: detections range across seeds 3687-3766/4014, false calls 7/778",
        "crack_4.0: detections range across seeds 899-2272/4078, false calls 7/778",
        "crack_8.6: detections range across seeds 3925-3925/3925, false calls 7/778",
    ):
        assert expected in claim.text
    for val in claim.numbers.values():
        assert str(int(val)) in claim.text


def test_physics_h3_claim_present(tmp_path: Path) -> None:
    """Verify physics_h3 claim is constructed with expected numbers and scope."""
    from utriage.report.claims import build_claims

    data = _make_fake_physics_data()
    (tmp_path / "physics_model_metrics.json").write_text(json.dumps(data), encoding="utf-8")
    claims = build_claims(tmp_path)
    claim = next((c for c in claims if c.id == "physics_h3"), None)
    assert claim is not None
    assert claim.evidence_file == "results/physics_model_metrics.json"
    assert claim.evidence_path == "folds"
    assert (
        claim.scope_limit == "pre-registered ADR 0010 addendum, coefficients unstable across folds"
    )
    for expected in (
        "crack_1.6: detections 2661/4014, false calls 0/778",
        "crack_4.0: detections 2578/4078, false calls 0/778",
        "crack_8.6: detections 3925/3925, false calls 1/778",
    ):
        assert expected in claim.text
    for val in claim.numbers.values():
        assert str(int(val)) in claim.text


def test_hybrid_comparison_claim(tmp_path: Path) -> None:
    """Verify hybrid_comparison derives strongest methods dynamically from metrics."""
    from utriage.report.claims import build_claims

    h_data = _make_fake_hybrid_data()
    p_data = _make_fake_physics_data()
    (tmp_path / "hybrid_metrics.json").write_text(json.dumps(h_data), encoding="utf-8")
    (tmp_path / "physics_model_metrics.json").write_text(json.dumps(p_data), encoding="utf-8")
    claims = build_claims(tmp_path)
    claim = next((c for c in claims if c.id == "hybrid_comparison"), None)
    assert claim is not None
    assert claim.evidence_file == "results/hybrid_metrics.json"
    assert claim.evidence_path == "folds"
    assert claim.text == "H2 strongest on the 1.6 mm crack, H3 strongest on the 4.0 mm crack."
    assert claim.numbers["crack_1.6"] == 1.6
    assert claim.numbers["crack_4.0"] == 4.0


@pytest.mark.parametrize("claim_id", ["hybrid_h2", "physics_h3", "hybrid_comparison"])
def test_exploratory_prefix_on_claims(tmp_path: Path, claim_id: str) -> None:
    """Verify exploratory run_label prefixes claim text with Exploratory:."""
    from utriage.report.claims import build_claims

    h_data = _make_fake_hybrid_data(run_label="exploratory")
    p_data = _make_fake_physics_data(run_label="exploratory")
    (tmp_path / "hybrid_metrics.json").write_text(json.dumps(h_data), encoding="utf-8")
    (tmp_path / "physics_model_metrics.json").write_text(json.dumps(p_data), encoding="utf-8")
    claims = build_claims(tmp_path)
    claim = next(c for c in claims if c.id == claim_id)
    assert claim.text.startswith("Exploratory: ")


def test_missing_hybrid_files_skipped(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify hybrid claims are skipped with printed reason when files are missing."""
    from utriage.report.claims import build_claims

    claims = build_claims(tmp_path)
    ids = {c.id for c in claims}
    assert "hybrid_h2" not in ids
    assert "physics_h3" not in ids
    assert "hybrid_comparison" not in ids
    captured = capsys.readouterr().err
    assert "Skipping claims requiring hybrid_metrics.json: file not found" in captured
    assert "Skipping claims requiring physics_model_metrics.json: file not found" in captured


def _make_fake_site_adaptation_shift(
    name: str,
    b_fc: tuple[int, int, float, float, float],
    a_fc: tuple[int, int, float, float, float],
    det: tuple[tuple[int, int], tuple[int, int]] = ((40, 50), (45, 50)),
) -> dict[str, Any]:
    """Helper creating a single shift entry for fake site_adaptation.json."""
    bk, bn, br, blow, bhigh = b_fc
    ak, an, ar, alow, ahigh = a_fc
    (bk_det, bn_det), (ak_det, an_det) = det
    b_calls = {
        "k": bk,
        "n": bn,
        "rate": br,
        "ci_low": blow,
        "ci_high": bhigh,
        "interval_95": [blow, bhigh],
    }
    a_calls = {
        "k": ak,
        "n": an,
        "rate": ar,
        "ci_low": alow,
        "ci_high": ahigh,
        "interval_95": [alow, ahigh],
    }
    return {
        "shift": {"name": name, "gain": 1.0, "offset": 0.0, "noise_std": 0.0},
        "baseline": {
            "false_calls": b_calls,
            "detection": {"k": bk_det, "n": bn_det, "rate": bk_det / bn_det},
        },
        "adapted": {
            "false_calls": a_calls,
            "detection": {"k": ak_det, "n": an_det, "rate": ak_det / an_det},
        },
        "test_underpowered": True,
        "upper95_fcr_scope": 0.05,
    }


def _make_fake_site_adaptation_data() -> dict[str, Any]:
    """Helper creating fake site_adaptation metrics dictionary."""
    return {
        "label": "Real (ML-NDT weld, 3 physical cracks; shifts are synthetic perturbations)",
        "n_images": {"test_clean": 100},
        "shifts": {
            "identity": _make_fake_site_adaptation_shift(
                "identity",
                (1, 100, 0.01, 0.0002, 0.054),
                (1, 100, 0.01, 0.0002, 0.054),
            ),
            "gain 0.5": _make_fake_site_adaptation_shift(
                "gain 0.5",
                (15, 100, 0.15, 0.086, 0.235),
                (2, 100, 0.02, 0.002, 0.070),
            ),
        },
    }


def _make_fake_joint_arm(
    arm: str,
    held_out: str,
    det: tuple[int, int],
    fc: tuple[int, int],
) -> dict[str, Any]:
    """Helper creating a single arm entry for joint training tests."""
    return {
        "arm": arm,
        "held_out": held_out,
        "detections": {"k": det[0], "n": det[1], "rate": det[0] / det[1]},
        "false_calls": {"k": fc[0], "n": fc[1], "rate": fc[0] / fc[1]},
    }


def _make_fake_joint_train_data() -> dict[str, Any]:
    """Helper creating fake joint_train_metrics dictionary."""
    return {
        "label": "real metrics on ML-NDT (weld, 3 physical cracks); sim images are Synthetic data",
        "protocol": "Leave-one-crack-out joint synthetic-real training",
        "folds": [
            {
                "held_out": "crack_1.6",
                "arms": {
                    "A": _make_fake_joint_arm("A", "crack_1.6", (45, 50), (1, 50)),
                    "B": _make_fake_joint_arm("B", "crack_1.6", (48, 50), (0, 50)),
                },
            },
            {
                "held_out": "crack_4.0",
                "arms": {"A": _make_fake_joint_arm("A", "crack_4.0", (38, 40), (2, 50))},
            },
        ],
    }


def test_site_adaptation_claims_present_with_numbers(tmp_path: Path) -> None:
    """Verify site adaptation claims are built with exact numbers and scope."""
    from utriage.report.claims import build_claims

    data = _make_fake_site_adaptation_data()
    (tmp_path / "site_adaptation.json").write_text(json.dumps(data), encoding="utf-8")
    claims = build_claims(tmp_path)
    claim_map = {c.id: c for c in claims if c.id.startswith("site_adaptation_")}
    assert "site_adaptation_identity" in claim_map
    assert "site_adaptation_gain_0.5" in claim_map

    c_gain = claim_map["site_adaptation_gain_0.5"]
    assert c_gain.evidence_file == "results/site_adaptation.json"
    assert c_gain.evidence_path == "shifts.gain 0.5"
    assert c_gain.scope_limit == "Real ML-NDT images with synthetic perturbations; single seed"
    assert "baseline 15/100" in c_gain.text
    assert "adapted 2/100" in c_gain.text
    assert "detection baseline 40/50 vs adapted 45/50" in c_gain.text
    assert "n_test_clean 100" in c_gain.text
    assert "test_underpowered" in c_gain.text
    assert "upper95_fcr_scope" in c_gain.text
    for val in c_gain.numbers.values():
        cands = [f"{val:.3f}", f"{val:.2f}", f"{val:.1f}", str(int(val))]
        assert any(cand in c_gain.text for cand in cands)


def test_site_adaptation_missing_file_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify site adaptation claims are skipped with printed notice when missing."""
    from utriage.report.claims import build_claims

    claims = build_claims(tmp_path)
    ids = {c.id for c in claims}
    assert not any(i.startswith("site_adaptation_") for i in ids)
    captured = capsys.readouterr().err
    assert "Skipping claims requiring site_adaptation.json: file not found" in captured


def test_joint_train_claims_present_with_numbers(tmp_path: Path) -> None:
    """Verify joint training claims are built with detection and false call counts."""
    from utriage.report.claims import build_claims

    data = _make_fake_joint_train_data()
    (tmp_path / "joint_train_metrics.json").write_text(json.dumps(data), encoding="utf-8")
    claims = build_claims(tmp_path)
    claim_map = {c.id: c for c in claims if c.id.startswith("joint_train_")}
    assert "joint_train_crack_1.6_a" in claim_map
    assert "joint_train_crack_1.6_b" in claim_map
    assert "joint_train_crack_4.0_a" in claim_map

    c_a = claim_map["joint_train_crack_1.6_a"]
    assert c_a.evidence_file == "results/joint_train_metrics.json"
    assert c_a.evidence_path == "folds"
    assert c_a.scope_limit == (
        "real metrics on ML-NDT weld with 3 physical cracks; sim images Synthetic data"
    )
    assert "detection 45/50" in c_a.text
    assert "false calls 1/50" in c_a.text
    assert c_a.numbers["detection_k"] == 45.0
    assert c_a.numbers["detection_n"] == 50.0
    assert c_a.numbers["false_calls_k"] == 1.0
    assert c_a.numbers["false_calls_n"] == 50.0


def test_joint_train_missing_file_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify joint training claims are skipped with printed notice when missing."""
    from utriage.report.claims import build_claims

    claims = build_claims(tmp_path)
    ids = {c.id for c in claims}
    assert not any(i.startswith("joint_train_") for i in ids)
    captured = capsys.readouterr().err
    assert "Skipping claims requiring joint_train_metrics.json: file not found" in captured


def _make_fake_false_call_analysis_data() -> dict[str, Any]:
    """Helper creating fake false_call_analysis dictionary."""
    return {
        "n_clean": 778,
        "n_flagged": 14,
        "agreement_table": {
            "counts_by_agreement": {"1": 7, "2": 7},
            "by_detector": {"anomaly": 14, "h2": 7},
        },
        "cause_counts": {"high_snr_structure": 14},
        "protocol_note": "pre-registered thresholds",
    }


def test_false_call_analysis_claim_present(tmp_path: Path) -> None:
    """Verify false call analysis claim is built with clean, flagged, agreement, causes."""
    from utriage.report.claims import build_claims

    data = _make_fake_false_call_analysis_data()
    (tmp_path / "false_call_analysis.json").write_text(json.dumps(data), encoding="utf-8")
    claims = build_claims(tmp_path)
    claim = next((c for c in claims if c.id == "false_call_analysis"), None)
    assert claim is not None
    assert claim.evidence_file == "results/false_call_analysis.json"
    assert claim.evidence_path == "agreement_table"
    assert claim.scope_limit == "ML-NDT (316L weld, 3 physical cracks), flaw-free scans"
    for expected in (
        "778 clean images evaluated",
        "14 flagged by any detector",
        "high_snr_structure: 14",
    ):
        assert expected in claim.text
    assert claim.numbers["clean_images_evaluated"] == 778.0
    assert claim.numbers["flagged_by_any_detector"] == 14.0
    assert claim.numbers["cause_high_snr_structure"] == 14.0


def test_false_call_analysis_missing_file_skipped(tmp_path: Path) -> None:
    """Verify false call analysis claim is skipped when file is missing."""
    from utriage.report.claims import build_claims

    claims = build_claims(tmp_path)
    assert not any(c.id == "false_call_analysis" for c in claims)


def test_joint_train_collector_extra_fields(tmp_path: Path) -> None:
    """Verify joint training includes actual_sim_fraction, seeds, and interval_type."""
    from utriage.report.claims import build_claims

    data = _make_fake_joint_train_data()
    data["actual_sim_fraction"] = 0.42
    data["seeds"] = [0]
    data["interval_type"] = "Clopper-Pearson"
    (tmp_path / "joint_train_metrics.json").write_text(json.dumps(data), encoding="utf-8")
    claims = build_claims(tmp_path)
    claim = next(c for c in claims if c.id == "joint_train_crack_1.6_a")
    assert "actual_sim_fraction 0.42" in claim.text
    assert "seeds [0]" in claim.text
    assert "interval_type Clopper-Pearson" in claim.text
    assert claim.numbers["actual_sim_fraction"] == 0.42


@pytest.mark.parametrize(
    ("b_hits", "expect_verdict"),
    [
        ((40, 30), True),
        ((48, 45), False),
    ],
)
def test_joint_train_verdict_only_if_numbers_show_it(
    tmp_path: Path, b_hits: tuple[int, int], expect_verdict: bool
) -> None:
    """Verify 'no arm improves consistently' stated only if numbers show it."""
    from utriage.report.claims import build_claims

    data = _make_fake_joint_train_data()
    # In fold 0, arm A has (45, 50). In fold 1, arm A has (38, 40).
    data["folds"][0]["arms"]["B"] = _make_fake_joint_arm("B", "crack_1.6", (b_hits[0], 50), (0, 50))
    data["folds"][1]["arms"]["B"] = _make_fake_joint_arm("B", "crack_4.0", (b_hits[1], 40), (0, 50))
    (tmp_path / "joint_train_metrics.json").write_text(json.dumps(data), encoding="utf-8")
    claims = build_claims(tmp_path)
    summary_claim = next((c for c in claims if c.id == "joint_train_summary"), None)
    if summary_claim is not None:
        if expect_verdict:
            assert "no arm improves consistently" in summary_claim.text
        else:
            assert "no arm improves consistently" not in summary_claim.text


def test_sim_regions_module_constant_threshold(fake_results_dir: Path) -> None:
    """Verify sim_regions mentions reporting threshold chosen by project, not a standard."""
    from utriage.report.claims import HIGH_FCR_THRESHOLD, build_claims

    assert HIGH_FCR_THRESHOLD == 0.3
    claims = build_claims(fake_results_dir)
    claim = next(c for c in claims if c.id == "sim_regions")
    expected = (
        "reporting threshold 0.3 false calls per window chosen by the project, not a standard"
    )
    assert expected in claim.text


def test_loco_crack_protocol_text(fake_results_dir: Path) -> None:
    """Verify loco_crack_* claims include batch-held-out protocol and ML-NDT scope."""
    from utriage.report.claims import build_claims

    claims = build_claims(fake_results_dir)
    for crack in ("1.6", "4.0", "8.6"):
        claim = next(c for c in claims if c.id == f"loco_crack_{crack}")
        expected_proto = (
            "early stopping on validation batches that contain the training cracks (batch-held-out)"
        )
        assert expected_proto in claim.text
        assert claim.scope_limit == "ML-NDT (316L weld, 3 physical cracks), not validated on bars"


@pytest.mark.parametrize(
    "forbidden_word",
    ["verified", "confirming", "establishing", "ensuring", "guarantee", "nearly"],
)
def test_forbidden_words_not_in_generated_claims(
    fake_results_dir: Path, forbidden_word: str
) -> None:
    """Verify forbidden words do not appear in any generated claim text."""
    from utriage.report.claims import build_claims

    (fake_results_dir / "loco_crackval_metrics.json").write_text(
        json.dumps(_make_crackval_data()), encoding="utf-8"
    )
    (fake_results_dir / "hybrid_metrics.json").write_text(
        json.dumps(_make_fake_hybrid_data()), encoding="utf-8"
    )
    (fake_results_dir / "physics_model_metrics.json").write_text(
        json.dumps(_make_fake_physics_data()), encoding="utf-8"
    )
    (fake_results_dir / "site_adaptation.json").write_text(
        json.dumps(_make_fake_site_adaptation_data()), encoding="utf-8"
    )
    (fake_results_dir / "joint_train_metrics.json").write_text(
        json.dumps(_make_fake_joint_train_data()), encoding="utf-8"
    )
    (fake_results_dir / "false_call_analysis.json").write_text(
        json.dumps(_make_fake_false_call_analysis_data()), encoding="utf-8"
    )
    claims = build_claims(fake_results_dir)
    pattern = rf"\b{re.escape(forbidden_word)}\b"
    for c in claims:
        assert not re.search(pattern, c.text, re.IGNORECASE), (
            f"Forbidden word '{forbidden_word}' in claim {c.id}: '{c.text}'"
        )
