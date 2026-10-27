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
                "Physical crack separation verified across batches; "
                "train and test share 0 physical cracks."
            ),
            "evidence_file": "results/leakage_audit.json",
            "evidence_path": "splits.test",
            "scope_limit": "public dataset",
            "numbers": {"shared_cracks": 0.0},
        },
        {
            "id": "loco_crack_1.6",
            "text": (
                "crack 1.6 mm: 19 of 4014 detected "
                "(rate 0.005, 95% CI [0.003, 0.007]), a90/95 = 2.47 mm"
            ),
            "evidence_file": "results/loco_metrics.json",
            "evidence_path": "folds[0]",
            "scope_limit": "public dataset, leave-one-crack-out",
            "numbers": {"rate": 0.005, "ci_low": 0.003, "ci_high": 0.007, "a90_95": 2.47},
        },
        {
            "id": "loco_crack_4.0",
            "text": (
                "crack 4.0 mm: 4078 of 4078 detected "
                "(rate 1.000, 95% CI [0.999, 1.000]), a90/95 = 0.72 mm"
            ),
            "evidence_file": "results/loco_metrics.json",
            "evidence_path": "folds[1]",
            "scope_limit": "public dataset, leave-one-crack-out",
            "numbers": {"rate": 1.000, "ci_low": 0.999, "ci_high": 1.000, "a90_95": 0.72},
        },
        {
            "id": "loco_crack_8.6",
            "text": (
                "crack 8.6 mm: 3925 of 3925 detected "
                "(rate 1.000, 95% CI [0.999, 1.000]), a90/95 = 0.72 mm"
            ),
            "evidence_file": "results/loco_metrics.json",
            "evidence_path": "folds[2]",
            "scope_limit": "public dataset, leave-one-crack-out",
            "numbers": {"rate": 1.000, "ci_low": 0.999, "ci_high": 1.000, "a90_95": 0.72},
        },
        {
            "id": "anomaly_crack_1.6",
            "text": (
                "crack 1.6 mm: 3649 of 4014 detected "
                "(rate 0.909, 95% CI [0.900, 0.918]) with false-call rate 0.018"
            ),
            "evidence_file": "results/anomaly_metrics.json",
            "evidence_path": "real.per_crack.crack_1.6",
            "scope_limit": "public dataset, leave-one-crack-out",
            "numbers": {"rate": 0.909, "ci_low": 0.900, "ci_high": 0.918, "fcr": 0.018},
        },
        {
            "id": "anomaly_crack_4.0",
            "text": "crack 4.0 mm: 4078 of 4078 detected (rate 1.000, 95% CI [0.999, 1.000])",
            "evidence_file": "results/anomaly_metrics.json",
            "evidence_path": "real.per_crack.crack_4.0",
            "scope_limit": "public dataset, leave-one-crack-out",
            "numbers": {"rate": 1.000, "ci_low": 0.999, "ci_high": 1.000},
        },
        {
            "id": "anomaly_crack_8.6",
            "text": "crack 8.6 mm: 3925 of 3925 detected (rate 1.000, 95% CI [0.999, 1.000])",
            "evidence_file": "results/anomaly_metrics.json",
            "evidence_path": "real.per_crack.crack_8.6",
            "scope_limit": "public dataset, leave-one-crack-out",
            "numbers": {"rate": 1.000, "ci_low": 0.999, "ci_high": 1.000},
        },
        {
            "id": "sim_regions",
            "text": (
                "Simulator held-out parameter regions: detection rate 0.772 "
                "(95% CI [0.728, 0.813]) at matched false-call rate 0.029"
            ),
            "evidence_file": "results/sim_generalization.json",
            "evidence_path": "test_sets.in_distribution",
            "scope_limit": "simulated",
            "numbers": {"rate": 0.772, "ci_low": 0.728, "ci_high": 0.813, "fcr": 0.029},
        },
        {
            "id": "union_property",
            "text": (
                "Rules union property holds across all indications: "
                "ML layer can only add flags, never remove them"
            ),
            "evidence_file": "results/generalization_matrix.json",
            "evidence_path": "union_property",
            "scope_limit": "independent prototype, advisory",
            "numbers": {},
        },
        {
            "id": "latency",
            "text": "CPU inference latency: median 14.2 ms, p95 21.0 ms across 500 scans",
            "evidence_file": "results/hardware.json",
            "evidence_path": "cpu_latency_ms.mlndt",
            "scope_limit": "independent prototype, advisory",
            "numbers": {"median_ms": 14.2, "p95_ms": 21.0, "n": 500.0},
        },
        {
            "id": "evidence_limit",
            "text": "Evidence base bounded by 3 distinct physical cracks in public research data",
            "evidence_file": "results/data_inventory.md",
            "evidence_path": "summary",
            "scope_limit": "public dataset",
            "numbers": {"distinct_cracks": 3.0},
        },
    ]
    if include_cross:
        claims.append(
            {
                "id": "cross_dataset",
                "text": (
                    "Cross-dataset test on external austenitic weld: "
                    "0 false calls across unflawed specimens"
                ),
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


def test_bullet_count_and_framing(tmp_path: Path) -> None:
    """Verify build_bullets returns 4-6 bullets with framing bullet first."""
    claims = _make_fake_claims(include_cross=True)
    _write_claims(tmp_path / "claims.json", claims)
    bullets = build_bullets(tmp_path)
    assert 4 <= len(bullets) <= 6
    assert bullets[0].startswith("Independent prototype: advisory, offline triage")
    assert "ML can only add flags to the rules engine" in bullets[0]
    assert bullets[0].endswith("(independent prototype, advisory).")


def test_every_number_in_claims(tmp_path: Path) -> None:
    """Verify every number in the bullets appears in some claim text or numbers."""
    claims = _make_fake_claims(include_cross=True)
    _write_claims(tmp_path / "claims.json", claims)
    bullets = build_bullets(tmp_path)
    for bullet in bullets:
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", bullet)
        for num in numbers:
            assert _number_in_claims(num, claims), f"Number {num} in {bullet!r} not in claims"


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
    claims[1]["text"] = f"Evaluation was {phrase} on real data"
    _write_claims(tmp_path / "claims.json", claims)
    with pytest.raises(ValueError, match="[Ff]orbidden"):
        build_bullets(tmp_path)


def test_missing_claims_json(tmp_path: Path) -> None:
    """Verify missing claims.json raises FileNotFoundError with clear message."""
    with pytest.raises(FileNotFoundError, match="Claims file not found"):
        build_bullets(tmp_path)


def test_missing_cross_dataset_claim(tmp_path: Path) -> None:
    """Verify omission of cross-dataset claim results in no cross-dataset bullet."""
    claims = _make_fake_claims(include_cross=False)
    _write_claims(tmp_path / "claims.json", claims)
    bullets = build_bullets(tmp_path)
    assert len(bullets) == 5
    assert not any("cross-dataset" in b.lower() for b in bullets)


def test_all_bullets_have_scope_tags(tmp_path: Path) -> None:
    """Verify every bullet ends with a short scope tag in parentheses."""
    claims = _make_fake_claims(include_cross=True)
    _write_claims(tmp_path / "claims.json", claims)
    bullets = build_bullets(tmp_path)
    for b in bullets:
        assert re.search(r"\([^)]+\)\.$", b) is not None, f"Bullet lacks scope tag: {b}"


def test_real_results() -> None:
    """Verify build_bullets succeeds on real repository results directory if present."""
    results_dir = Path(__file__).resolve().parents[1] / "results"
    claims_path = results_dir / "claims.json"
    if not claims_path.exists():
        pytest.skip(f"{claims_path} does not exist yet")
    bullets = build_bullets(results_dir)
    assert 4 <= len(bullets) <= 6
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
    assert 4 <= len(lines) <= 6
    assert out_file.exists()
    assert len(out_file.read_text(encoding="utf-8").splitlines()) == len(lines)
