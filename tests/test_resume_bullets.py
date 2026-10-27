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


def _write_claims(path: Path, data: list[dict[str, Any]]) -> None:
    """Helper to write claims list to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh)


def _make_fake_claims(*, include_cross: bool = True) -> list[dict[str, Any]]:
    """Build representative fake claims.json data for unit testing."""
    claims: list[dict[str, Any]] = [
        {
            "id": "leakage",
            "text": (
                "The old random batch split shares all 3 physical cracks "
                "between train and test, so earlier test scores are superseded."
            ),
            "evidence_file": "results/leakage_audit.json",
            "evidence_path": "shared_cracks_train_test",
            "scope_limit": "public dataset",
            "numbers": {"shared_cracks": 3.0},
        },
        {
            "id": "loco_crack_1.6",
            "text": (
                "Leave-one-crack-out crack_1.6: detection 19/4014 "
                "(rate 0.005, 95% CI [0.003, 0.007]), false calls 0.02 per scan."
            ),
            "evidence_file": "results/generalization_matrix.json",
            "evidence_path": "rows.supervised.mlndt_loco_crack_1.6",
            "scope_limit": "public dataset, leave-one-crack-out",
            "numbers": {"k": 19.0, "n": 4014.0, "rate": 0.005},
        },
        {
            "id": "loco_crack_4.0",
            "text": (
                "Leave-one-crack-out crack_4.0: detection 4078/4078 "
                "(rate 1.000, 95% CI [0.999, 1.000]), false calls 0.00 per scan."
            ),
            "evidence_file": "results/generalization_matrix.json",
            "evidence_path": "rows.supervised.mlndt_loco_crack_4.0",
            "scope_limit": "public dataset, leave-one-crack-out",
            "numbers": {"k": 4078.0, "n": 4078.0, "rate": 1.000},
        },
        {
            "id": "loco_crack_8.6",
            "text": (
                "Leave-one-crack-out crack_8.6: detection 3925/3925 "
                "(rate 1.000, 95% CI [0.999, 1.000]), false calls 0.00 per scan."
            ),
            "evidence_file": "results/generalization_matrix.json",
            "evidence_path": "rows.supervised.mlndt_loco_crack_8.6",
            "scope_limit": "public dataset, leave-one-crack-out",
            "numbers": {"k": 3925.0, "n": 3925.0, "rate": 1.000},
        },
        {
            "id": "anomaly_crack_1.6",
            "text": (
                "Anomaly model crack_1.6: detection 3649/4014 "
                "(rate 0.909, 95% CI [0.900, 0.918]), false calls 0.018 per scan."
            ),
            "evidence_file": "results/generalization_matrix.json",
            "evidence_path": "rows.anomaly.mlndt_loco_crack_1.6",
            "scope_limit": "public dataset, leave-one-crack-out",
            "numbers": {"k": 3649.0, "n": 4014.0, "rate": 0.909, "false_call_rate": 0.018},
        },
        {
            "id": "anomaly_crack_4.0",
            "text": (
                "Anomaly model crack_4.0: detection 4078/4078 "
                "(rate 1.000, 95% CI [0.999, 1.000]), false calls 0.00 per scan."
            ),
            "evidence_file": "results/generalization_matrix.json",
            "evidence_path": "rows.anomaly.mlndt_loco_crack_4.0",
            "scope_limit": "public dataset, leave-one-crack-out",
            "numbers": {"k": 4078.0, "n": 4078.0, "rate": 1.000},
        },
        {
            "id": "sim_regions",
            "text": (
                "Simulator held-out regions (ML vs rules): false calls "
                "exceed 0.3 per window at unseen frequency and above-range noise."
            ),
            "evidence_file": "results/generalization_matrix.json",
            "evidence_path": "rows",
            "scope_limit": "simulated",
            "numbers": {"threshold": 0.3},
        },
        {
            "id": "union_property",
            "text": (
                "The union property holds across all columns with 0 violations "
                "(ML adds flags, never removes a rules flag)."
            ),
            "evidence_file": "results/generalization_matrix.json",
            "evidence_path": "union_property",
            "scope_limit": "advisory",
            "numbers": {"n_violations": 0.0},
        },
        {
            "id": "latency",
            "text": "CPU latency on development CPU: median 14.2 ms, p95 21.0 ms.",
            "evidence_file": "results/hardware.json",
            "evidence_path": "cpu_latency_ms.mlndt",
            "scope_limit": "advisory",
            "numbers": {"median_ms": 14.2, "p95_ms": 21.0},
        },
    ]
    if include_cross:
        claims.append(
            {
                "id": "cross_dataset",
                "text": "Cross-dataset evaluation: 0 false calls across unflawed specimens.",
                "evidence_file": "results/cross_dataset_metrics.json",
                "evidence_path": "summary",
                "scope_limit": "cross-dataset",
                "numbers": {"n_false": 0.0},
            }
        )
    return claims


def _number_in_claims(num_str: str, claims: list[dict[str, Any]]) -> bool:
    """Check if number string appears in some claim's text or numbers."""
    for c in claims:
        if num_str in c.get("text", ""):
            return True
        for val in c.get("numbers", {}).values():
            if str(val) == num_str or f"{float(val):.3f}" == num_str:
                return True
    return False


def test_bullet_word_limit_and_sentence(tmp_path: Path) -> None:
    """Verify each bullet is at most 40 words and forms a single sentence."""
    claims = _make_fake_claims(include_cross=True)
    _write_claims(tmp_path / "claims.json", claims)
    bullets = build_bullets(tmp_path)
    assert 5 <= len(bullets) <= 6
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
    assert len(bullets) == 6
    expected = [
        ("Built", "(advisory)."),
        ("Audited", "(public dataset)."),
        ("Detected", "(leave-one-crack-out)."),
        ("Trained", "(leave-one-crack-out)."),
        ("Observed", "(simulated)."),
        ("Verified", "(advisory)."),
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
    assert len(bullets) == 5


def test_missing_cross_dataset_claim(tmp_path: Path) -> None:
    """Verify omission of cross-dataset claim omits cross-dataset mention."""
    claims = _make_fake_claims(include_cross=False)
    _write_claims(tmp_path / "claims.json", claims)
    bullets = build_bullets(tmp_path)
    assert len(bullets) == 6
    assert not any("cross-dataset" in b.lower() for b in bullets)


def test_no_forbidden_phrases(tmp_path: Path) -> None:
    """Verify no forbidden phrase appears in any generated bullet."""
    claims = _make_fake_claims(include_cross=True)
    _write_claims(tmp_path / "claims.json", claims)
    bullets = build_bullets(tmp_path)
    combined = " ".join(bullets).lower()
    for phrase in FORBIDDEN:
        assert phrase.lower() not in combined, f"Forbidden phrase {phrase!r} found in bullets"


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
def test_forbidden_phrase_raises(tmp_path: Path, phrase: str) -> None:
    """Verify build_bullets raises ValueError if any bullet contains a forbidden phrase."""
    claims = _make_fake_claims()
    claims[0]["text"] = f"Evaluation was {phrase} on real data"
    _write_claims(tmp_path / "claims.json", claims)
    with pytest.raises(ValueError, match="[Ff]orbidden"):
        build_bullets(tmp_path)


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
    assert 5 <= len(bullets) <= 6
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
    assert len(lines) == 6
    assert out_file.exists()
    assert len(out_file.read_text(encoding="utf-8").splitlines()) == len(lines)
