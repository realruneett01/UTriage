"""Tests for ML experiment quantitative claims (project plan Phase 8)."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pytest  # noqa: E402

from utriage.report.claims import (  # noqa: E402
    build_claims,
    check_forbidden,
    check_forbidden_words,
)
from utriage.report.claims_ml import (  # noqa: E402
    SCOPE_LIMIT,
    collect_ml_claims,
)


def _make_fake_separability_data() -> dict[str, Any]:
    """Construct minimal payload matching separability_metrics.json."""
    return {
        "protocols": {
            "batchval": {
                "status": "evaluated",
                "folds": [
                    {
                        "held_out": "crack_1.6",
                        "auroc": 0.8524,
                        "average_precision": 0.7412,
                        "operating_points": {
                            "0.05": {
                                "detection": {"k": 3000, "n": 4014},
                                "false_calls": {"k": 200, "n": 4000},
                            }
                        },
                    }
                ],
            },
            "crackval": {
                "status": "evaluated",
                "folds": [
                    {
                        "held_out": "crack_4.0",
                        "auroc": 0.7101,
                        "average_precision": 0.6234,
                        "operating_points": {
                            "0.05": {
                                "detection": {"k": 2500, "n": 3800},
                                "false_calls": {"k": 350, "n": 3900},
                            }
                        },
                    }
                ],
            },
        }
    }


def _make_fake_loco_robust_data() -> dict[str, Any]:
    """Construct minimal payload matching loco_robust_metrics.json."""
    return {
        "config": {"seeds": [0, 1, 2, 3, 4]},
        "folds": [
            {
                "held_out": "crack_1.6",
                "ensemble_metrics": {
                    "auroc": 0.8912,
                    "average_precision": 0.8123,
                    "fcr_0.05": {
                        "detections": {"n_hit": 3600, "n_flawed": 4014},
                        "false_calls": {"n_false": 150, "n_clean": 4000},
                    },
                },
            }
        ],
    }


def _make_fake_lite_hybrid_data() -> dict[str, Any]:
    """Construct minimal payload matching lite_hybrid_metrics.json."""
    return {
        "folds": {
            "crack_1.6": {
                "held_out": "crack_1.6",
                "threshold_free": {"fused": {"auroc": 0.9234}},
                "operating_points": {
                    "fcr_0.05": {
                        "hybrid": {
                            "detection": {"k": 3800, "n": 4014},
                            "false_calls": {"k": 180, "n": 4000},
                        }
                    }
                },
            }
        },
        "latency": {
            "threads": 4,
            "note": (
                "measured on the development machine CPU limited to 4 threads, "
                "not on the target i5 laptop"
            ),
            "litenet": {
                "params": 54240,
                "median_ms": 1.25,
            },
        },
    }


def _assert_numbers_match_text(text: str, numbers: dict[str, float]) -> None:
    """Verify each number in dictionary appears formatted in claim text."""
    assert len(numbers) > 0
    for key, val in numbers.items():
        candidates = [f"{val:.3f}", f"{val:.2f}", f"{val:.1f}", str(int(val))]
        assert any(re.search(rf"\b{re.escape(cand)}\b", text) for cand in candidates), (
            f"Value {val} ({key}) not found in text: '{text}'"
        )


def _assert_no_forbidden_words(text: str, scope: str) -> None:
    """Verify forbidden phrases and individual words do not appear."""
    assert not check_forbidden(text)
    assert not check_forbidden(scope)
    assert not check_forbidden_words(text)
    assert not check_forbidden_words(scope)
    extra_forbidden = ("certif", "validated", "production", "deployed")
    for bad in extra_forbidden:
        assert bad not in text.lower(), f"Forbidden substring '{bad}' found in text: '{text}'"
        assert bad not in scope.lower(), f"Forbidden substring '{bad}' found in scope: '{scope}'"


def test_missing_files_skipped(tmp_path: Path) -> None:
    """Verify collect_ml_claims returns empty list when results files are absent."""
    claims = collect_ml_claims(tmp_path)
    assert claims == []


def test_separability_claims_built_correctly(tmp_path: Path) -> None:
    """Verify separability claims capture protocols, metrics, clean scans note, and scope."""
    payload = _make_fake_separability_data()
    (tmp_path / "separability_metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    claims = collect_ml_claims(tmp_path)
    assert len(claims) == 2

    c_batch = next(c for c in claims if c.id == "separability_batchval")
    assert "batch-held-out early stopping" in c_batch.text
    assert "crack_1.6: AUROC 0.852, AP 0.741" in c_batch.text
    assert "detection 3000/4014, false calls 200/4000 on clean_eval" in c_batch.text
    assert "Thresholds were set on clean scans only." in c_batch.text
    assert c_batch.scope_limit == SCOPE_LIMIT
    _assert_numbers_match_text(c_batch.text, c_batch.numbers)
    _assert_no_forbidden_words(c_batch.text, c_batch.scope_limit)

    c_crack = next(c for c in claims if c.id == "separability_crackval")
    assert "crack-held-out early stopping" in c_crack.text
    assert "crack_4.0: AUROC 0.710, AP 0.623" in c_crack.text
    _assert_numbers_match_text(c_crack.text, c_crack.numbers)
    _assert_no_forbidden_words(c_crack.text, c_crack.scope_limit)


def test_loco_robust_claim_built_correctly(tmp_path: Path) -> None:
    """Verify loco_robust claim formats seed ensemble, seed count, and fold metrics."""
    payload = _make_fake_loco_robust_data()
    (tmp_path / "loco_robust_metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    claims = collect_ml_claims(tmp_path)
    assert len(claims) == 1
    c = claims[0]
    assert c.id == "loco_robust"
    assert "Robust LOCO seed ensemble (5 seeds)" in c.text
    assert "crack_1.6: AUROC 0.891, AP 0.812" in c.text
    assert "detection 3600/4014, false calls 150/4000 on clean_eval" in c.text
    assert "Thresholds were set on clean scans only." in c.text
    assert c.scope_limit == SCOPE_LIMIT
    _assert_numbers_match_text(c.text, c.numbers)
    _assert_no_forbidden_words(c.text, c.scope_limit)


def test_lite_hybrid_claim_built_correctly(tmp_path: Path) -> None:
    """Verify lite_hybrid claim formats fused AUROC, params, and CPU latency with note."""
    payload = _make_fake_lite_hybrid_data()
    (tmp_path / "lite_hybrid_metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    claims = collect_ml_claims(tmp_path)
    assert len(claims) == 1
    c = claims[0]
    assert c.id == "lite_hybrid"
    assert "crack_1.6: fused AUROC 0.923" in c.text
    assert "detection 3800/4014, false calls 180/4000" in c.text
    assert "LiteNet 54240 params, CPU median latency 1.25 ms" in c.text
    assert "measured on the development machine CPU limited to 4 threads" in c.text
    assert "not on the target i5 laptop" in c.text
    assert c.scope_limit == SCOPE_LIMIT
    _assert_numbers_match_text(c.text, c.numbers)
    _assert_no_forbidden_words(c.text, c.scope_limit)


@pytest.mark.parametrize(
    ("claim_id", "verdict_word"),
    [
        ("loco_robust", "no improvement"),
        ("lite_hybrid", "inconclusive"),
    ],
)
def test_preregistration_verdict_appended_verbatim(
    tmp_path: Path, claim_id: str, verdict_word: str
) -> None:
    """Verify verdict word is appended verbatim to claim text when file exists."""
    (tmp_path / "loco_robust_metrics.json").write_text(
        json.dumps(_make_fake_loco_robust_data()), encoding="utf-8"
    )
    (tmp_path / "lite_hybrid_metrics.json").write_text(
        json.dumps(_make_fake_lite_hybrid_data()), encoding="utf-8"
    )
    verdicts = {
        "loco_robust": "no improvement",
        "lite_hybrid": "inconclusive",
    }
    (tmp_path / "preregistration_verdicts.json").write_text(json.dumps(verdicts), encoding="utf-8")

    claims = collect_ml_claims(tmp_path)
    c = next(claim for claim in claims if claim.id == claim_id)
    assert f"Verdict: {verdict_word}." in c.text
    _assert_no_forbidden_words(c.text, c.scope_limit)


def test_build_claims_integrates_ml_claims(tmp_path: Path) -> None:
    """Verify build_claims in claims.py includes ML claims from collect_ml_claims."""
    (tmp_path / "separability_metrics.json").write_text(
        json.dumps(_make_fake_separability_data()), encoding="utf-8"
    )
    (tmp_path / "loco_robust_metrics.json").write_text(
        json.dumps(_make_fake_loco_robust_data()), encoding="utf-8"
    )
    (tmp_path / "lite_hybrid_metrics.json").write_text(
        json.dumps(_make_fake_lite_hybrid_data()), encoding="utf-8"
    )

    claims = build_claims(tmp_path)
    ids = {c.id for c in claims}
    assert "separability_batchval" in ids
    assert "separability_crackval" in ids
    assert "loco_robust" in ids
    assert "lite_hybrid" in ids


@pytest.mark.parametrize(
    ("builder_name", "filename", "payload_func", "expected_id"),
    [
        (
            "build_separability_claims",
            "separability_metrics.json",
            _make_fake_separability_data,
            "separability_batchval",
        ),
        (
            "build_loco_robust_claim",
            "loco_robust_metrics.json",
            _make_fake_loco_robust_data,
            "loco_robust",
        ),
        (
            "build_lite_hybrid_claim",
            "lite_hybrid_metrics.json",
            _make_fake_lite_hybrid_data,
            "lite_hybrid",
        ),
    ],
)
def test_individual_builders(
    tmp_path: Path,
    builder_name: str,
    filename: str,
    payload_func: Callable[[], dict[str, Any]],
    expected_id: str,
) -> None:
    """Verify individual builder functions on isolated fake payloads."""
    import utriage.report.claims_ml as claims_ml

    builder = getattr(claims_ml, builder_name)
    (tmp_path / filename).write_text(json.dumps(payload_func()), encoding="utf-8")
    res = builder(tmp_path)
    claim_obj = res[0] if isinstance(res, list) else res
    assert claim_obj is not None
    assert claim_obj.id == expected_id


def test_scope_limit_names_same_weld_caveat() -> None:
    """ML claims disclose that clean test scans come from the same weld."""
    assert "same weld" in SCOPE_LIMIT
    assert "3 physical cracks" in SCOPE_LIMIT
