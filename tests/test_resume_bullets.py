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
    _extract_a90,
    _find_det_at_size,
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


@pytest.fixture
def dummy_results_dir(tmp_path: Path) -> Path:
    """Prepare a temporary directory with default baseline, synthetic, and mlndt JSON files."""
    _write_json(tmp_path / "rules_baseline.json", _make_dummy_baseline())
    _write_json(tmp_path / "bar_domain_metrics.json", _make_dummy_synthetic())
    _write_json(
        tmp_path / "mlndt_metrics.json",
        {
            "per_split": {
                "test": {
                    "pod": {"a90_95": 0.7231},
                    "false_calls": {"n_false": 0, "n": 778},
                }
            }
        },
    )
    return tmp_path


def _assert_contains_all(text: str, substrings: tuple[str, ...]) -> None:
    """Assert all substrings are contained within text."""
    for s in substrings:
        assert s in text


def _assert_valid_bullet_labels(items: list[str]) -> None:
    """Assert all items except safety baseline contain a valid label."""
    valid_labels = ("(real data)", "(synthetic validation)", "(synthetic triage)")
    for item in items:
        if "safety baseline" not in item:
            assert any(label in item for label in valid_labels)


def test_pod_present_path(dummy_results_dir: Path) -> None:
    """Test real data bullet when pod logistic fit is present."""
    bullets = build_bullets(dummy_results_dir)
    _assert_contains_all(
        bullets[1],
        ("(real data)", "0.73 mm", "0 false calls on 778 unflawed scans"),
    )


def test_pod_null_path(dummy_results_dir: Path) -> None:
    """Test real data bullet when pod is null and binomial block is used."""
    mlndt_data = {
        "per_split": {
            "test": {
                "pod": None,
                "detection": {"n_hit": 1222, "n_flawed": 1222, "pod_lower95": 0.9999},
                "false_calls": {"n_false": 0, "n": 778},
            }
        }
    }
    _write_json(dummy_results_dir / "mlndt_metrics.json", mlndt_data)

    bullets = build_bullets(dummy_results_dir)
    _assert_contains_all(
        bullets[1],
        (
            "(real data)",
            "1222 of 1222 flaws detected",
            "0.99",
            "0 false calls on 778 unflawed scans",
        ),
    )


def test_missing_triage_file(dummy_results_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test omission notice when triage_metrics.json is missing."""
    bullets = build_bullets(dummy_results_dir)
    captured = capsys.readouterr()

    assert len(bullets) == 3
    _assert_contains_all(captured.err, ("triage_metrics.json", "omitting triage bullet"))


def test_synthetic_validation_detection_fallback(dummy_results_dir: Path) -> None:
    """Test synthetic bullet falls back to 0.6 mm detection rate when a90 is missing."""
    _write_json(
        dummy_results_dir / "bar_domain_metrics.json",
        _make_dummy_synthetic(a90_rules=None, a90_ml=None),
    )
    bullets = build_bullets(dummy_results_dir)
    _assert_contains_all(
        bullets[2],
        (
            "(synthetic validation)",
            "detection rate at 0.6 mm",
            "0.75",
            "0.99",
            "0.018",
        ),
    )


def test_labels_present(dummy_results_dir: Path) -> None:
    """Verify labels (real data) and (synthetic validation) are both present."""
    bullets = build_bullets(dummy_results_dir)
    _assert_contains_all(
        " ".join(bullets),
        ("(real data)", "(synthetic validation)"),
    )


@pytest.mark.parametrize(
    ("payload", "expected_snippets"),
    [
        ({"conformal_coverage": 0.85}, ("(synthetic validation)", "0.85")),
        ({"bullet": "Custom triage summary."}, ("(synthetic validation)",)),
    ],
    ids=["coverage_field", "custom_bullet_field"],
)
def test_triage_bullet_label(
    dummy_results_dir: Path,
    payload: dict,
    expected_snippets: tuple[str, ...],
) -> None:
    """Verify triage bullet contains expected label and content."""
    _write_json(dummy_results_dir / "triage_metrics.json", payload)
    bullets = build_bullets(dummy_results_dir)
    assert len(bullets) == 4
    _assert_contains_all(bullets[3], expected_snippets)


def _expected_real_snippets(results_dir: Path) -> tuple[str, ...]:
    """Derive expected metric substrings for real-data bullet."""
    with (results_dir / "mlndt_metrics.json").open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    test_sp = data.get("per_split", {}).get("test", {})
    pod = test_sp.get("pod")
    if pod is not None and "a90_95" in pod:
        return (f"{round_ceil(pod['a90_95'], 2)} mm",)
    det = test_sp.get("detection", {})
    n_hit = det.get("n_hit", 0)
    n_flawed = det.get("n_flawed", 0)
    lb = round_floor(det.get("pod_lower95", 0.0), 2)
    return (f"{n_hit} of {n_flawed} flaws detected", lb)


def _expected_synthetic_snippets(results_dir: Path) -> tuple[str, ...]:
    """Derive expected metric substrings for synthetic validation bullet."""
    with (results_dir / "bar_domain_metrics.json").open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    model = data.get("kept_model", {}).get("choice", "ml_2d")
    pmn = data.get("per_method_per_noise", {})
    rules_sp = pmn.get("rules", {}).get("2.0", {})
    ml_sp = pmn.get(model, {}).get("2.0", {})
    fcr_str = round_ceil(rules_sp.get("false_call_rate", {}).get("rate", 0.0), 3)
    r_a90, m_a90 = _extract_a90(rules_sp), _extract_a90(ml_sp)
    if r_a90 is not None and m_a90 is not None:
        comp = f"{round_ceil(r_a90, 2)} mm to {round_ceil(m_a90, 2)} mm"
        return (fcr_str, comp)
    r_det = _find_det_at_size(rules_sp.get("detection_table", []), 0.6) or 0.0
    m_det = _find_det_at_size(ml_sp.get("detection_table", []), 0.6) or 0.0
    comp = f"{round_floor(r_det, 2)} to {round_floor(m_det, 2)}"
    return (fcr_str, comp)


def test_real_results() -> None:
    """Verify build_bullets succeeds on real repository results directory."""
    results_dir = Path(__file__).resolve().parents[1] / "results"
    bullets = build_bullets(results_dir)

    assert len(bullets) >= 3
    expected = (
        "(real data)",
        "(synthetic validation)",
        *_expected_real_snippets(results_dir),
        *_expected_synthetic_snippets(results_dir),
    )
    _assert_contains_all(" ".join(bullets), expected)
    assert any("(real data)" in b for b in bullets)
    _assert_valid_bullet_labels(bullets)


def test_main(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify main function executes and prints bullets."""
    main()
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.startswith("- ")]
    assert len(lines) >= 3
    assert any("(real data)" in line for line in lines)
    assert any("(synthetic validation)" in line for line in lines)
    _assert_valid_bullet_labels(lines)
