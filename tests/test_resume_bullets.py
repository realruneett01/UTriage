"""Tests for resume bullets generator (scripts/make_resume_bullets.py)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from utriage.report.claims import FORBIDDEN

_SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from make_resume_bullets import (  # noqa: E402
    FORBIDDEN_WORDS,
    _check_bullets_forbidden,
    build_bullets,
    main,
    round_ceil,
    round_floor,
)


@pytest.mark.parametrize(
    ("val", "decimals", "expected"),
    [
        (0.7231, 2, "0.73"),
        (0.7200, 2, "0.72"),
        (0.8301, 2, "0.84"),
        (0.01785, 3, "0.018"),
        (0.0, 3, "0.000"),
    ],
)
def test_rounding_ceil(val: float, decimals: int, expected: str) -> None:
    """Verify rounding ceiling never rounds down in project's favour."""
    assert round_ceil(val, decimals) == expected


@pytest.mark.parametrize(
    ("val", "decimals", "expected"),
    [
        (0.9999, 2, "0.99"),
        (0.9900, 2, "0.99"),
        (0.9975, 2, "0.99"),
        (0.8750, 2, "0.87"),
        (0.0, 2, "0.00"),
    ],
)
def test_rounding_floor(val: float, decimals: int, expected: str) -> None:
    """Verify detection rounding floor never rounds up in project's favour."""
    assert round_floor(val, decimals) == expected


def _write_json(path: Path, data: Any) -> None:
    """Helper to write arbitrary data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh)


def _write_claims(path: Path, data: list[dict[str, Any]]) -> None:
    """Helper to write claims list to JSON file."""
    _write_json(path, data)


def _make_h2_claim() -> dict[str, Any]:
    """Build representative fake claim for hybrid H2 detector."""
    return {
        "id": "hybrid_h2",
        "text": (
            "Pre-registered hybrid H2 across seeds: crack 1.6 mm 3687-3766/4014, "
            "crack 4.0 mm 899-2272/4078, crack 8.6 mm 3925/3925; false calls 7/778 scans."
        ),
        "evidence_file": "results/hybrid_metrics.json",
        "evidence_path": "folds",
        "scope_limit": (
            "pre-registered ADR 0010, one weld PAUT public dataset, "
            "3 physical cracks, seed-sensitive"
        ),
        "numbers": {
            "crack_1.6_min_k": 3687.0,
            "crack_1.6_max_k": 3766.0,
            "crack_1.6_n": 4014.0,
            "crack_4.0_min_k": 899.0,
            "crack_4.0_max_k": 2272.0,
            "crack_4.0_n": 4078.0,
            "crack_8.6_k": 3925.0,
            "crack_8.6_n": 3925.0,
            "n_false": 7.0,
            "n_scans": 778.0,
        },
    }


def _make_h3_claim() -> dict[str, Any]:
    """Build representative fake claim for physics H3 detector."""
    return {
        "id": "physics_h3",
        "text": (
            "Physics model H3: crack 1.6 mm 2661/4014, false calls 0/778; "
            "crack 4.0 mm 2578/4078, false calls 0/778; "
            "crack 8.6 mm 3925/3925, false calls 1/778."
        ),
        "evidence_file": "results/physics_model_metrics.json",
        "evidence_path": "folds",
        "scope_limit": "pre-registered ADR 0010 addendum, coefficients unstable across folds",
        "numbers": {
            "crack_1.6_k": 2661.0,
            "crack_1.6_n": 4014.0,
            "crack_1.6_n_false": 0.0,
            "crack_1.6_n_clean": 778.0,
            "crack_4.0_k": 2578.0,
            "crack_4.0_n": 4078.0,
            "crack_4.0_n_false": 0.0,
            "crack_4.0_n_clean": 778.0,
            "crack_8.6_k": 3925.0,
            "crack_8.6_n": 3925.0,
            "crack_8.6_n_false": 1.0,
            "crack_8.6_n_clean": 778.0,
        },
    }


def _make_hybrid_claims() -> list[dict[str, Any]]:
    """Build representative fake claims for hybrid H2 and physics H3 detectors."""
    return [_make_h2_claim(), _make_h3_claim()]


def _make_leakage_claim() -> dict[str, Any]:
    """Build fake leakage claim."""
    return {
        "id": "leakage",
        "text": "Audit found random split shared all 3 physical cracks between train and test.",
        "evidence_file": "results/leakage_audit.json",
        "evidence_path": "shared_cracks_train_test",
        "scope_limit": "public dataset",
        "numbers": {"shared_cracks": 3.0},
    }


def _make_loco_batch_claims() -> list[dict[str, Any]]:
    """Build fake leave-one-crack-out claims for batch early stopping."""
    return [
        {
            "id": "loco_crack_1.6",
            "text": "LOCO crack_1.6: detection 579/4014, false calls 0/778.",
            "evidence_file": "results/generalization_matrix.json",
            "evidence_path": "rows.supervised.mlndt_loco_crack_1.6",
            "scope_limit": "leave-one-crack-out",
            "numbers": {"k": 579.0, "n": 4014.0, "n_false": 0.0, "n_unflawed": 778.0},
        },
        {
            "id": "loco_crack_4.0",
            "text": "LOCO crack_4.0: detection 671/4078, false calls 0/778.",
            "evidence_file": "results/generalization_matrix.json",
            "evidence_path": "rows.supervised.mlndt_loco_crack_4.0",
            "scope_limit": "leave-one-crack-out",
            "numbers": {"k": 671.0, "n": 4078.0, "n_false": 0.0, "n_unflawed": 778.0},
        },
        {
            "id": "loco_crack_8.6",
            "text": "LOCO crack_8.6: detection 3925/3925, false calls 0/778.",
            "evidence_file": "results/generalization_matrix.json",
            "evidence_path": "rows.supervised.mlndt_loco_crack_8.6",
            "scope_limit": "leave-one-crack-out",
            "numbers": {"k": 3925.0, "n": 3925.0, "n_false": 0.0, "n_unflawed": 778.0},
        },
    ]


def _make_loco_crackval_claim() -> dict[str, Any]:
    """Build fake leave-one-crack-out claim for crack validation."""
    return {
        "id": "loco_crackval",
        "text": (
            "LOCO crackval: crack_1.6 4012/4014, false calls 753/778; "
            "crack_4.0 2449/4078, false calls 0/778; "
            "crack_8.6 3925/3925, false calls 11/778."
        ),
        "evidence_file": "results/loco_crackval_metrics.json",
        "evidence_path": "folds",
        "scope_limit": "leave-one-crack-out",
        "numbers": {
            "crack_1.6_k": 4012.0,
            "crack_1.6_n": 4014.0,
            "crack_1.6_n_false": 753.0,
            "crack_1.6_n_unflawed": 778.0,
            "crack_4.0_k": 2449.0,
            "crack_4.0_n": 4078.0,
            "crack_4.0_n_false": 0.0,
            "crack_4.0_n_unflawed": 778.0,
            "crack_8.6_k": 3925.0,
            "crack_8.6_n": 3925.0,
            "crack_8.6_n_false": 11.0,
            "crack_8.6_n_unflawed": 778.0,
        },
    }


def _make_loco_claims() -> list[dict[str, Any]]:
    """Build fake leave-one-crack-out claims."""
    return _make_loco_batch_claims() + [_make_loco_crackval_claim()]


def _make_anomaly_claims() -> list[dict[str, Any]]:
    """Build fake anomaly model claims."""
    return [
        {
            "id": "anomaly_crack_1.6",
            "text": "Anomaly model crack_1.6: detection 3649/4014, false calls 0.018 per scan.",
            "evidence_file": "results/generalization_matrix.json",
            "evidence_path": "rows.anomaly.mlndt_loco_crack_1.6",
            "scope_limit": "held-out cracks, trained on clean scans only",
            "numbers": {"k": 3649.0, "n": 4014.0, "false_call_rate": 0.018},
        },
        {
            "id": "anomaly_crack_4.0",
            "text": "Anomaly model crack_4.0: detection 4078/4078.",
            "evidence_file": "results/generalization_matrix.json",
            "evidence_path": "rows.anomaly.mlndt_loco_crack_4.0",
            "scope_limit": "held-out cracks, trained on clean scans only",
            "numbers": {"k": 4078.0, "n": 4078.0},
        },
    ]


def _make_sim_claims() -> list[dict[str, Any]]:
    """Build claims for simulator, union property, and latency."""
    return [
        {
            "id": "sim_regions",
            "text": "Simulator held-out regions: false calls exceed 0.3 per window.",
            "evidence_file": "results/generalization_matrix.json",
            "evidence_path": "rows",
            "scope_limit": "simulated",
            "numbers": {"threshold": 0.3},
        },
        {
            "id": "union_property",
            "text": "Property-tested union rule holds on 9 simulator columns with 0 violations.",
            "evidence_file": "results/generalization_matrix.json",
            "evidence_path": "union_property",
            "scope_limit": "advisory",
            "numbers": {"n_violations": 0.0, "n_columns": 9.0},
        },
        {
            "id": "latency",
            "text": "CPU latency on development workstation CPU: median 14.2 ms, p95 21.0 ms.",
            "evidence_file": "results/hardware.json",
            "evidence_path": "cpu_latency_ms.mlndt",
            "scope_limit": "advisory",
            "numbers": {"median_ms": 14.2, "p95_ms": 21.0},
        },
    ]


def _make_cross_claim() -> dict[str, Any]:
    """Build fake cross-dataset claim."""
    return {
        "id": "cross_dataset",
        "text": "Cross-dataset NDT_ML_Flaw: 3442/3442 detections and 3558/3558 false calls.",
        "evidence_file": "results/cross_dataset_metrics.json",
        "evidence_path": "summary",
        "scope_limit": "NDT_ML_Flaw, real weld data",
        "numbers": {
            "k": 3442.0,
            "n": 3442.0,
            "n_false": 3558.0,
            "n_scans": 3558.0,
        },
    }


def _make_fake_claims(
    *, include_cross: bool = True, include_hybrid: bool = False
) -> list[dict[str, Any]]:
    """Build representative fake claims.json data for unit testing."""
    claims: list[dict[str, Any]] = (
        [_make_leakage_claim()] + _make_loco_claims() + _make_anomaly_claims() + _make_sim_claims()
    )
    if include_cross:
        claims.append(_make_cross_claim())
    if include_hybrid:
        claims.extend(_make_hybrid_claims())
    return claims


def _make_fake_site_adaptation(*, underpowered: bool = False) -> dict[str, Any]:
    """Build representative fake site adaptation metrics data."""
    return {
        "n_calib_clean": 824,
        "n_test_clean": 43,
        "test_underpowered": underpowered,
        "shifts": {
            "gain 2.0": {
                "baseline": {
                    "false_calls": {"k": 43, "n": 43},
                    "detection": {"k": 57, "n": 57},
                },
                "adapted": {
                    "false_calls": {"k": 0, "n": 43},
                    "detection": {"k": 42, "n": 57},
                },
            },
            "gain 0.5": {
                "baseline": {
                    "false_calls": {"k": 0, "n": 43},
                    "detection": {"k": 19, "n": 57},
                },
                "adapted": {
                    "false_calls": {"k": 0, "n": 43},
                    "detection": {"k": 42, "n": 57},
                },
            },
        },
    }


def _make_fake_joint_training() -> dict[str, Any]:
    """Build representative fake joint training metrics data."""
    return {
        "seeds": [0],
        "folds": [
            {
                "held_out": "crack_1.6",
                "arms": {
                    "A": {"detections": {"k": 165, "n": 4014}},
                    "C": {"detections": {"k": 107, "n": 4014}},
                },
            },
            {
                "held_out": "crack_4.0",
                "arms": {
                    "A": {"detections": {"k": 0, "n": 4078}},
                    "C": {"detections": {"k": 2306, "n": 4078}},
                },
            },
            {
                "held_out": "crack_8.6",
                "arms": {
                    "A": {"detections": {"k": 3925, "n": 3925}},
                    "C": {"detections": {"k": 3925, "n": 3925}},
                },
            },
        ],
    }


def _make_fake_false_call_analysis() -> dict[str, Any]:
    """Build representative fake false call analysis metrics data."""
    return {
        "n_flagged": 43,
        "cause_counts": {"high_snr_structure": 43},
        "per_fold": {
            "crack_1.6": {"held_out": "crack_1.6", "n_clean": 778},
            "crack_4.0": {"held_out": "crack_4.0", "n_clean": 778},
            "crack_8.6": {"held_out": "crack_8.6", "n_clean": 778},
        },
    }


def _val_matches(val: Any, num_str: str) -> bool:
    """Check if claim numeric value matches string representation."""
    formatted = (str(val), f"{float(val):.3f}", f"{float(val):.1f}")
    return num_str in formatted


def _number_in_claims(num_str: str, claims: list[dict[str, Any]]) -> bool:
    """Check if number string appears in some claim's text or numbers."""
    for c in claims:
        if num_str in c.get("text", ""):
            return True
        nums = c.get("numbers", {}).values()
        if any(_val_matches(v, num_str) for v in nums):
            return True
    return False


def test_bullet_word_limit_and_sentence(tmp_path: Path) -> None:
    """Verify each bullet is at most 40 words and forms a single sentence."""
    claims = _make_fake_claims(include_cross=True, include_hybrid=True)
    _write_claims(tmp_path / "claims.json", claims)
    _write_json(tmp_path / "site_adaptation.json", _make_fake_site_adaptation())
    _write_json(tmp_path / "joint_train_metrics.json", _make_fake_joint_training())
    bullets = build_bullets(tmp_path)
    assert len(bullets) == 10
    for b in bullets:
        words = b.split()
        assert len(words) <= 40, f"Bullet exceeds 40 words ({len(words)}): {b}"
        assert b.count(". ") == 0, f"Bullet contains multiple sentences: {b}"
        assert b.endswith(")."), f"Bullet does not end with scope tag: {b}"


def test_story_order_and_verbs(tmp_path: Path) -> None:
    """Verify bullets follow story order with action verbs and scope tags."""
    claims = _make_fake_claims(include_cross=True)
    _write_claims(tmp_path / "claims.json", claims)
    bullets = build_bullets(tmp_path)
    assert len(bullets) == 7
    expected = [
        ("Built", "(advisory)."),
        ("Audited", "(public dataset)."),
        ("Detected", "(leave-one-crack-out, 3 physical cracks, one public weld dataset)."),
        ("Trained", "(held-out cracks, trained on clean scans only)."),
        ("Observed", "(simulated)."),
        ("Evaluated", "(NDT_ML_Flaw, real weld data)."),
        ("Property-tested", "(advisory)."),
    ]
    for b, (verb, tag) in zip(bullets, expected, strict=True):
        assert b.startswith(f"{verb} "), f"Expected verb {verb} in {b}"
        assert b.endswith(tag), f"Expected tag {tag} in {b}"


def test_every_number_in_claims(tmp_path: Path) -> None:
    """Verify every number in the bullets appears in some claim text or numbers."""
    claims = _make_fake_claims(include_cross=True)
    _write_claims(tmp_path / "claims.json", claims)
    bullets = build_bullets(tmp_path)
    for bullet in bullets:
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", bullet)
        for num in numbers:
            assert _number_in_claims(num, claims), f"Number {num} in {bullet!r} not in claims"


@pytest.mark.parametrize(
    "missing_id",
    [
        "leakage",
        "loco_crack_8.6",
        "loco_crackval",
        "anomaly_crack_1.6",
        "sim_regions",
        "union_property",
        "latency",
    ],
)
def test_skipping_missing_claim(tmp_path: Path, missing_id: str) -> None:
    """Verify skipping bullet when an underlying claim is missing."""
    claims = [c for c in _make_fake_claims(include_cross=True) if c["id"] != missing_id]
    _write_claims(tmp_path / "claims.json", claims)
    bullets = build_bullets(tmp_path)
    assert len(bullets) == 6


def test_missing_cross_dataset_claim(tmp_path: Path) -> None:
    """Verify omission of cross-dataset claim omits cross-dataset mention."""
    claims = _make_fake_claims(include_cross=False)
    _write_claims(tmp_path / "claims.json", claims)
    bullets = build_bullets(tmp_path)
    assert len(bullets) == 6
    assert not any("cross-dataset" in b.lower() for b in bullets)


@pytest.mark.parametrize(
    "phrase",
    [
        *FORBIDDEN_WORDS,
        *FORBIDDEN,
    ],
)
def test_no_forbidden_phrases(tmp_path: Path, phrase: str) -> None:
    """Verify no forbidden phrase appears in any generated bullet."""
    claims = _make_fake_claims(include_cross=True, include_hybrid=True)
    _write_claims(tmp_path / "claims.json", claims)
    _write_json(tmp_path / "site_adaptation.json", _make_fake_site_adaptation())
    _write_json(tmp_path / "joint_train_metrics.json", _make_fake_joint_training())
    bullets = build_bullets(tmp_path)
    combined = " ".join(bullets).lower()
    assert phrase.lower() not in combined, f"Forbidden phrase {phrase!r} found in bullets"


@pytest.mark.parametrize(
    "phrase",
    FORBIDDEN,
)
def test_forbidden_phrase_raises(tmp_path: Path, phrase: str) -> None:
    """Verify build_bullets raises ValueError if any claim contains a forbidden phrase."""
    claims = _make_fake_claims()
    claims[0]["text"] = f"Evaluation was {phrase} on real data"
    _write_claims(tmp_path / "claims.json", claims)
    with pytest.raises(ValueError, match="[Ff]orbidden"):
        build_bullets(tmp_path)


@pytest.mark.parametrize(
    "word",
    FORBIDDEN_WORDS,
)
def test_forbidden_word_in_bullets_raises(word: str) -> None:
    """Verify _check_bullets_forbidden raises ValueError if any bullet contains a forbidden word."""
    with pytest.raises(ValueError, match="[Ff]orbidden"):
        _check_bullets_forbidden([f"Model was {word} on scans (advisory)."])


def test_missing_claims_json(tmp_path: Path) -> None:
    """Verify missing claims.json raises FileNotFoundError with clear message."""
    with pytest.raises(FileNotFoundError, match="Claims file not found"):
        build_bullets(tmp_path)


def test_real_results() -> None:
    """Verify build_bullets succeeds on real repository results directory if present."""
    results_dir = Path(__file__).resolve().parents[1] / "results"
    claims_path = results_dir / "claims.json"
    if not claims_path.exists():
        pytest.skip(f"{claims_path} does not exist yet")
    bullets = build_bullets(results_dir)
    assert len(bullets) == 12  # includes the separability bullet
    for bullet in bullets:
        assert bullet.endswith(").")


def test_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify main function executes and prints bullets from CLI."""
    claims = _make_fake_claims(include_cross=True)
    _write_claims(tmp_path / "claims.json", claims)
    out_file = tmp_path / "bullets.txt"
    main(["--results", str(tmp_path), "--output", str(out_file)])
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.startswith("- ")]
    assert len(lines) == 7
    assert out_file.exists()
    assert len(out_file.read_text(encoding="utf-8").splitlines()) == len(lines)


def test_hybrid_bullet_word_limit_and_structure(tmp_path: Path) -> None:
    """Verify hybrid bullet is at most 40 words, single sentence, and ends with scope tag."""
    claims = _make_fake_claims(include_cross=True, include_hybrid=True)
    _write_claims(tmp_path / "claims.json", claims)
    bullets = build_bullets(tmp_path)
    assert len(bullets) == 8
    hybrid_b = bullets[4]
    words = hybrid_b.split()
    assert len(words) <= 40, f"Hybrid bullet exceeds 40 words ({len(words)}): {hybrid_b}"
    assert hybrid_b.count(". ") == 0, f"Hybrid bullet contains multiple sentences: {hybrid_b}"
    assert hybrid_b.startswith("Pre-registered ")
    assert "3925/3925 vs 3925/3925 on 8.6 mm" in hybrid_b
    assert hybrid_b.endswith("(public dataset, pre-registered).")
    assert "confirming" not in hybrid_b
    assert "vary across seeds" in hybrid_b


def test_hybrid_h3_false_calls_max_over_folds(tmp_path: Path) -> None:
    """Verify H3 false calls reports the maximum over folds (1/778) not first match."""
    claims = _make_fake_claims(include_cross=True, include_hybrid=True)
    _write_claims(tmp_path / "claims.json", claims)
    bullets = build_bullets(tmp_path)
    hybrid_b = bullets[4]
    assert "1/778" in hybrid_b


def test_site_adaptation_bullet(tmp_path: Path) -> None:
    """Verify site adaptation bullet reads shifts and flags underpowered tests."""
    claims = _make_fake_claims()
    _write_claims(tmp_path / "claims.json", claims)
    _write_json(tmp_path / "site_adaptation.json", _make_fake_site_adaptation(underpowered=True))
    bullets = build_bullets(tmp_path)
    site_b = [b for b in bullets if b.startswith("Adapted ")][0]
    assert "gain 2.0: detection 57/57 to 42/57, false calls 43/43 to 0/43" in site_b
    assert "gain 0.5: detection 19/57 to 42/57, false calls 0/43 to 0/43" in site_b
    assert "small test set" in site_b
    assert site_b.endswith(
        "(single seed; shifts are synthetic perturbations of real weld images, small test set)."
    )


def test_crack_held_out_bullet_contains_false_calls(tmp_path: Path) -> None:
    """Verify crack-held-out bullet contains the false-call count."""
    claims = _make_fake_claims()
    _write_claims(tmp_path / "claims.json", claims)
    bullets = build_bullets(tmp_path)
    loco_b = [b for b in bullets if b.startswith("Detected ")][0]
    assert "8.6 mm 3925/3925" in loco_b
    assert "1.6 mm 4012/4014" in loco_b
    assert "4.0 mm 2449/4078" in loco_b
    assert "753/778" in loco_b
    assert "11/778" in loco_b
    assert "0/778" in loco_b


def test_site_bullet_false_call_transition(tmp_path: Path) -> None:
    """Verify site adaptation bullet shows false-call transition for each shift."""
    claims = _make_fake_claims()
    _write_claims(tmp_path / "claims.json", claims)
    _write_json(tmp_path / "site_adaptation.json", _make_fake_site_adaptation())
    bullets = build_bullets(tmp_path)
    site_b = [b for b in bullets if b.startswith("Adapted ")][0]
    assert "false calls 43/43 to 0/43" in site_b
    assert "false calls 0/43 to 0/43" in site_b


def test_false_call_analysis_bullet_and_skip(tmp_path: Path) -> None:
    """Verify false call bullet is skipped when missing and generated when present."""
    claims = _make_fake_claims()
    _write_claims(tmp_path / "claims.json", claims)
    bullets_without = build_bullets(tmp_path)
    assert not any("Analyzed false calls" in b for b in bullets_without)

    _write_json(tmp_path / "false_call_analysis.json", _make_fake_false_call_analysis())
    bullets_with = build_bullets(tmp_path)
    fc_b = [b for b in bullets_with if b.startswith("Analyzed ")][0]
    assert "2334" in fc_b
    assert "43" in fc_b
    assert "high_snr_structure" in fc_b
    assert fc_b.endswith("(public dataset, pre-registered).")


def test_joint_training_bullet(tmp_path: Path) -> None:
    """Verify joint training bullet reports arm A vs C and computed improvement rule."""
    claims = _make_fake_claims()
    _write_claims(tmp_path / "claims.json", claims)
    _write_json(tmp_path / "joint_train_metrics.json", _make_fake_joint_training())
    bullets = build_bullets(tmp_path)
    joint_b = [b for b in bullets if "joint synthetic-real training" in b][0]
    assert "0/4078" in joint_b
    assert "2306/4078" in joint_b
    assert "no arm improved every crack" in joint_b
    assert "1 seed" in joint_b


@pytest.mark.parametrize(
    ("claim_id", "bad_numbers"),
    [
        ("leakage", {}),
        ("anomaly_crack_1.6", {"k": 3649.0, "n": 4014.0}),
        ("union_property", {"n_violations": 0.0}),
        ("latency", {"p95_ms": 21.0}),
    ],
)
def test_missing_values_skip_bullets(
    tmp_path: Path, claim_id: str, bad_numbers: dict[str, float], capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify bullets are skipped and warnings emitted when required values are missing."""
    claims = _make_fake_claims(include_cross=True)
    for c in claims:
        if c["id"] == claim_id:
            c["numbers"] = bad_numbers
            c["text"] = "No numbers in text here."
    _write_claims(tmp_path / "claims.json", claims)
    bullets = build_bullets(tmp_path)
    assert len(bullets) == 6
    captured = capsys.readouterr()
    assert "Warning:" in captured.err


def _make_separability_numbers(k: float, n: float, fk: float, fn: float) -> dict[str, float]:
    """Helper to build separability numbers dictionary."""
    return {
        "target_fcr": 0.05,
        "crack_1.6_auroc": 0.852,
        "crack_1.6_ap": 0.741,
        "crack_1.6_det_k": k,
        "crack_1.6_det_n": n,
        "crack_1.6_fc_k": fk,
        "crack_1.6_fc_n": fn,
        "crack_4.0_auroc": 0.921,
        "crack_4.0_ap": 0.850,
        "crack_8.6_auroc": 0.999,
        "crack_8.6_ap": 0.998,
        "default_threshold": 0.5,
    }


def _make_separability_claim(
    *, k: float = 3000.0, n: float = 4014.0, fk: float = 200.0, fn: float = 4000.0
) -> dict[str, Any]:
    """Build representative fake claim for separability_batchval."""
    text = (
        "Separability (batch-held-out early stopping): "
        "crack_1.6: AUROC 0.852, AP 0.741, at FCR 0.05 detection 3000/4014, false calls 200/4000; "
        "crack_4.0: AUROC 0.921, AP 0.850; crack_8.6: AUROC 0.999, AP 0.998. "
        "Thresholds were set on clean scans only."
    )
    return {
        "id": "separability_batchval",
        "text": text,
        "evidence_file": "results/separability_metrics.json",
        "evidence_path": "protocols.batchval.folds",
        "scope_limit": "ML-NDT weld, 3 physical cracks, leave-one-crack-out, clean test scans",
        "numbers": _make_separability_numbers(k, n, fk, fn),
    }


def test_separability_bullet_present_with_numbers(tmp_path: Path) -> None:
    """Verify separability bullet is generated right after LOCO with correct numbers."""
    claims = _make_fake_claims() + [_make_separability_claim()]
    _write_claims(tmp_path / "claims.json", claims)
    bullets = build_bullets(tmp_path)
    loco_idx = next(i for i, b in enumerate(bullets) if b.startswith("Detected "))
    sep_b = bullets[loco_idx + 1]
    assert sep_b.startswith(
        "Showed that the leave-one-crack-out classifier ranked held-out cracks "
    )
    assert "almost perfectly (AUROC 0.852-0.999 across the 3 folds)" in sep_b
    assert (
        "calibration recovered detection (3000/4014 on the 1.6 mm crack at a 5% "
        "false-call target, 200/4000 false calls)" in sep_b
    )
    assert ", while the default 0.5 threshold missed most of that crack" in sep_b
    assert sep_b.endswith(
        "(leave-one-crack-out, 3 physical cracks, clean scans from the same weld)."
    )


def test_separability_bullet_absent_when_missing(tmp_path: Path) -> None:
    """Verify separability bullet is omitted when separability_batchval claim is missing."""
    claims = _make_fake_claims()
    _write_claims(tmp_path / "claims.json", claims)
    bullets = build_bullets(tmp_path)
    assert not any("Showed that the leave-one-crack-out classifier" in b for b in bullets)


def test_separability_clause_dropped_when_above_half(tmp_path: Path) -> None:
    """Verify 'missed most' clause is dropped when batch LOCO detection is above half."""
    claims = _make_fake_claims()
    for c in claims:
        if c["id"] == "loco_crack_1.6":
            c["numbers"]["k"] = 3500.0
            c["text"] = "LOCO crack_1.6: detection 3500/4014, false calls 0/778."
    claims.append(_make_separability_claim())
    _write_claims(tmp_path / "claims.json", claims)
    bullets = build_bullets(tmp_path)
    sep_b = [b for b in bullets if b.startswith("Showed that ")][0]
    assert "missed most" not in sep_b
    assert "default 0.5 threshold" not in sep_b
    assert sep_b.endswith(
        "200/4000 false calls) "
        "(leave-one-crack-out, 3 physical cracks, clean scans from the same weld)."
    )


@pytest.mark.parametrize(
    "word",
    [
        "verified",
        "confirming",
        "establishing",
        "ensuring",
        "guarantee",
        "certif",
        "validated",
        "production",
        "deployed",
    ],
)
def test_separability_forbidden_check_passes(tmp_path: Path, word: str) -> None:
    """Verify separability bullet does not contain any forbidden word."""
    claims = _make_fake_claims() + [_make_separability_claim()]
    _write_claims(tmp_path / "claims.json", claims)
    bullets = build_bullets(tmp_path)
    sep_b = [b for b in bullets if b.startswith("Showed that ")][0]
    assert word.lower() not in sep_b.lower()


def test_separability_every_number_in_claims(tmp_path: Path) -> None:
    """Verify every number in the separability bullet appears in claims."""
    claims = _make_fake_claims(include_cross=True) + [_make_separability_claim()]
    _write_claims(tmp_path / "claims.json", claims)
    bullets = build_bullets(tmp_path)
    sep_b = [b for b in bullets if b.startswith("Showed that ")][0]
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", sep_b)
    for num in numbers:
        assert _number_in_claims(num, claims), f"Number {num} in {sep_b!r} not in claims"
