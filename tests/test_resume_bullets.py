"""Tests for resume bullets generator (scripts/make_resume_bullets.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

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


def _write_json(path: Path, data: dict) -> None:
    """Helper to write dictionary data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh)


def _make_dummy_baseline() -> dict:
    """Return minimal rules_baseline.json structure."""
    return {"label": "Synthetic data", "gate": {"passed": True}}


def _make_dummy_synthetic(
    *, a90_rules: float | None = 0.8301, a90_ml: float | None = 0.7182
) -> dict:
    """Return minimal bar_domain_metrics.json structure."""
    rules_pod = {"a90_95": a90_rules} if a90_rules is not None else None
    ml_pod = {"a90_95": a90_ml} if a90_ml is not None else None
    return {
        "label": "Synthetic data",
        "kept_model": {"choice": "ml_2d"},
        "per_method_per_noise": {
            "rules": {
                "2.0": {
                    "pod_summary": {"pod": rules_pod},
                    "false_call_rate": {"rate": 0.01785},
                    "detection_table": [
                        {"size_mm": 0.6, "detection_rate": 0.75},
                    ],
                }
            },
            "ml_2d": {
                "2.0": {
                    "pod_summary": {"pod": ml_pod},
                    "false_call_rate": {"rate": 0.01785},
                    "detection_table": [
                        {"size_mm": 0.6, "detection_rate": 0.9999},
                    ],
                }
            },
        },
    }


def test_pod_present_path(tmp_path: Path) -> None:
    """Test real data bullet when pod logistic fit is present."""
    _write_json(tmp_path / "rules_baseline.json", _make_dummy_baseline())
    _write_json(tmp_path / "bar_domain_metrics.json", _make_dummy_synthetic())
    mlndt_data = {
        "per_split": {
            "test": {
                "pod": {"a90_95": 0.7231},
                "false_calls": {"n_false": 0, "n": 778},
            }
        }
    }
    _write_json(tmp_path / "mlndt_metrics.json", mlndt_data)

    bullets = build_bullets(tmp_path)
    real_bullet = bullets[1]

    assert "(real data)" in real_bullet
    assert "0.73 mm" in real_bullet
    assert "0 false calls on 778 unflawed scans" in real_bullet


def test_pod_null_path(tmp_path: Path) -> None:
    """Test real data bullet when pod is null and binomial block is used."""
    _write_json(tmp_path / "rules_baseline.json", _make_dummy_baseline())
    _write_json(tmp_path / "bar_domain_metrics.json", _make_dummy_synthetic())
    mlndt_data = {
        "per_split": {
            "test": {
                "pod": None,
                "detection": {"n_hit": 1222, "n_flawed": 1222, "pod_lower95": 0.9999},
                "false_calls": {"n_false": 0, "n": 778},
            }
        }
    }
    _write_json(tmp_path / "mlndt_metrics.json", mlndt_data)

    bullets = build_bullets(tmp_path)
    real_bullet = bullets[1]

    assert "(real data)" in real_bullet
    assert "1222 of 1222 flaws detected" in real_bullet
    assert "0.99" in real_bullet
    assert "0 false calls on 778 unflawed scans" in real_bullet


def test_missing_triage_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test omission notice when triage_metrics.json is missing."""
    _write_json(tmp_path / "rules_baseline.json", _make_dummy_baseline())
    _write_json(tmp_path / "bar_domain_metrics.json", _make_dummy_synthetic())
    _write_json(
        tmp_path / "mlndt_metrics.json",
        {
            "per_split": {
                "test": {"pod": {"a90_95": 0.7231}, "false_calls": {"n_false": 0, "n": 778}}
            }
        },
    )

    bullets = build_bullets(tmp_path)
    captured = capsys.readouterr()

    assert len(bullets) == 3
    assert "triage_metrics.json" in captured.err
    assert "omitting triage bullet" in captured.err


def test_synthetic_validation_detection_fallback(tmp_path: Path) -> None:
    """Test synthetic bullet falls back to 0.6 mm detection rate when a90 is missing."""
    _write_json(tmp_path / "rules_baseline.json", _make_dummy_baseline())
    _write_json(
        tmp_path / "bar_domain_metrics.json", _make_dummy_synthetic(a90_rules=None, a90_ml=None)
    )
    _write_json(
        tmp_path / "mlndt_metrics.json",
        {
            "per_split": {
                "test": {"pod": {"a90_95": 0.7231}, "false_calls": {"n_false": 0, "n": 778}}
            }
        },
    )

    bullets = build_bullets(tmp_path)
    synth_bullet = bullets[2]

    assert "(synthetic validation)" in synth_bullet
    assert "detection rate at 0.6 mm" in synth_bullet
    assert "0.75" in synth_bullet
    assert "0.99" in synth_bullet
    assert "0.018" in synth_bullet


def test_labels_present(tmp_path: Path) -> None:
    """Verify labels (real data) and (synthetic validation) are both present."""
    _write_json(tmp_path / "rules_baseline.json", _make_dummy_baseline())
    _write_json(tmp_path / "bar_domain_metrics.json", _make_dummy_synthetic())
    _write_json(
        tmp_path / "mlndt_metrics.json",
        {
            "per_split": {
                "test": {"pod": {"a90_95": 0.7231}, "false_calls": {"n_false": 0, "n": 778}}
            }
        },
    )

    bullets = build_bullets(tmp_path)
    all_text = " ".join(bullets)

    assert "(real data)" in all_text
    assert "(synthetic validation)" in all_text


def test_real_results() -> None:
    """Verify build_bullets succeeds on real repository results directory."""
    results_dir = Path(__file__).resolve().parents[1] / "results"
    bullets = build_bullets(results_dir)

    assert len(bullets) == 3
    all_text = " ".join(bullets)
    assert "(real data)" in all_text
    assert "(synthetic validation)" in all_text
    assert "1222 of 1222 flaws detected" in all_text
    assert "0.99" in all_text
    assert "0.84 mm to 0.72 mm" in all_text
    assert "0.018" in all_text


def test_main(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify main function executes and prints bullets."""
    main()
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.startswith("- ")]
    assert len(lines) == 3
