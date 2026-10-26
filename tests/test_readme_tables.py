"""Tests for README results tables generator (scripts/make_readme_tables.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from make_readme_tables import (  # noqa: E402
    render_all_tables,
    round_ceil,
    round_floor,
    update_readme_file,
    update_readme_text,
)


@pytest.mark.parametrize(
    ("val", "decimals", "expected"),
    [
        (0.7231, 2, "0.73"),
        (0.7200, 2, "0.72"),
        (0.8301, 2, "0.84"),
        (0.00384, 3, "0.004"),
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
        (0.9975, 2, "0.99"),
        (0.9900, 2, "0.99"),
        (0.9166, 2, "0.91"),
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


def _make_dummy_mlndt() -> dict:
    """Return minimal mlndt_metrics.json structure."""
    return {
        "label": "Real data (ML-NDT)",
        "per_split": {
            "test": {
                "pod": None,
                "pod_reason": "Data is separable: only hits or only misses present.",
                "detection": {
                    "n_flawed": 100,
                    "n_hit": 100,
                    "pod_lower95": 0.9975,
                },
                "false_calls": {
                    "n": 50,
                    "n_false": 0,
                    "upper95": 0.00384,
                },
            }
        },
        "latency_ms": {
            "cpu_batch_1": 21.73,
            "gpu_batch_1": 0.99,
        },
    }


def _make_dummy_method_data(det_06: float, a90: float | None) -> dict:
    """Build single method entry for bar domain metrics."""
    pod = {"a90_95": a90} if a90 is not None else None
    return {
        "false_call_rate": {"rate": 0.0},
        "detection_table": [
            {"size_mm": 0.4, "detection_rate": 0.0 if det_06 < 1.0 else 1.0},
            {"size_mm": 0.6, "detection_rate": det_06},
            {"size_mm": 0.8, "detection_rate": 1.0},
        ],
        "pod_summary": {"pod": pod, "reason": "Separable" if pod is None else None},
    }


def _make_dummy_bar_domain() -> dict:
    """Return minimal bar_domain_metrics.json structure."""
    return {
        "label": "Synthetic data",
        "configs": {"sizes_mm": [0.4, 0.6, 0.8]},
        "per_method_per_noise": {
            "rules": {"0.5": _make_dummy_method_data(0.9166, None)},
            "ml_1d": {"0.5": _make_dummy_method_data(1.0, None)},
            "ml_2d": {"0.5": _make_dummy_method_data(1.0, 0.5887)},
        },
    }


def _make_dummy_triage() -> dict:
    """Return minimal triage_metrics.json structure."""
    return {
        "label": "Synthetic data",
        "conformal": {
            "alpha": 0.1,
            "coverage": 0.9136,
            "uncertain_fraction": 0.0625,
        },
        "operating_point": {
            "threshold": 0.9991,
            "pod_lower95_at_ref": 0.9037,
            "ref_size": 0.8,
            "false_call_rate": 0.0,
        },
        "ranking": {
            "operating_point": {
                "threshold": 0.9991,
                "mean_effort_ranked": 18.0,
                "mean_effort_random": 99.25,
                "flaws_found_rules_only": 0.7833,
                "flaws_found_union": 0.8166,
                "ml_false_calls_per_bar": 109.55,
            }
        },
    }


def _make_dummy_robustness() -> dict:
    """Return minimal robustness_metrics.json structure."""
    return {
        "label": "Synthetic data",
        "domain_shift": {
            "noise_2.5x": {
                "spec": {"material": "ti64", "probe": "p5", "noise_scale": 2.5},
                "rules": {
                    "false_call_rate": 0.1562,
                    "detection_per_size": {"0.6": 0.875, "0.8": 1.0},
                },
                "ml_2d_matched": {
                    "false_call_rate": 0.1562,
                    "detection_per_size": {"0.6": 1.0, "0.8": 1.0},
                },
            }
        },
        "sim_to_real": {
            "bar_1d_on_mlndt_val": {
                "label": "Real data (ML-NDT)",
                "model": "models/bar_1d.pt",
                "roc_auc": 0.711,
                "n_images": 200,
                "image_axis_meaning": "UNVERIFIED - Ruri to check",
                "domain_gap_note": "Expected domain gap note.",
            }
        },
    }


def _populate_dummy_results(results_dir: Path) -> None:
    """Populate directory with complete set of dummy result files."""
    _write_json(results_dir / "mlndt_metrics.json", _make_dummy_mlndt())
    _write_json(results_dir / "bar_domain_metrics.json", _make_dummy_bar_domain())
    _write_json(results_dir / "triage_metrics.json", _make_dummy_triage())
    _write_json(results_dir / "robustness_metrics.json", _make_dummy_robustness())


def test_missing_files_render_notices(tmp_path: Path) -> None:
    """Verify missing result files yield experiment not run messages."""
    empty_dir = tmp_path / "empty_results"
    empty_dir.mkdir()
    rendered = render_all_tables(empty_dir)

    assert "Experiment has not been run (mlndt_metrics.json not found)" in rendered
    assert "Experiment has not been run (bar_domain_metrics.json not found)" in rendered
    assert "Experiment has not been run (triage_metrics.json not found)" in rendered
    assert "Experiment has not been run (robustness_metrics.json not found)" in rendered


def test_labels_present_in_captions(tmp_path: Path) -> None:
    """Verify table captions state their data label."""
    res_dir = tmp_path / "results"
    _populate_dummy_results(res_dir)
    rendered = render_all_tables(res_dir)

    assert "(Real data" in rendered or "(Real data)" in rendered
    assert "(Synthetic data)" in rendered


def test_rounding_direction_in_rendered_tables(tmp_path: Path) -> None:
    """Verify detection is floored and false-call / a90 bounds are ceiled."""
    res_dir = tmp_path / "results"
    _populate_dummy_results(res_dir)
    rendered = render_all_tables(res_dir)

    assert "0.99" in rendered  # 0.9975 floored
    assert "0.004" in rendered  # 0.00384 ceiled
    assert "0.59 mm" in rendered  # 0.5887 ceiled
    assert "0.91" in rendered  # 0.9166 floored
    assert "UNVERIFIED - Ruri to check" in rendered


def test_update_readme_text_with_existing_markers() -> None:
    """Verify update_readme_text replaces content between existing markers."""
    text = "# Head\n\n<!-- RESULTS:BEGIN -->\nold\n<!-- RESULTS:END -->\n\n# Tail\n"
    res = update_readme_text(text, "new_tables")
    assert "new_tables" in res
    assert "old" not in res
    assert "# Head" in res
    assert "# Tail" in res


def test_markers_created_when_missing(tmp_path: Path) -> None:
    """Verify markers and Results section are created if missing."""
    readme_path = tmp_path / "README.md"
    readme_path.write_text("# Title\n\nSome body text.\n", encoding="utf-8")
    res_dir = tmp_path / "results"
    _populate_dummy_results(res_dir)

    update_readme_file(readme_path, res_dir)
    content = readme_path.read_text(encoding="utf-8")

    assert "## Results" in content
    assert "<!-- RESULTS:BEGIN -->" in content
    assert "<!-- RESULTS:END -->" in content
    assert "# Title\n\nSome body text." in content


def test_idempotent_rewrite(tmp_path: Path) -> None:
    """Verify running table update multiple times produces identical text."""
    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        "# Title\n\n## Results\n\n<!-- RESULTS:BEGIN -->\nold\n<!-- RESULTS:END -->\n\nFooter.\n",
        encoding="utf-8",
    )
    res_dir = tmp_path / "results"
    _populate_dummy_results(res_dir)

    update_readme_file(readme_path, res_dir)
    content_first = readme_path.read_text(encoding="utf-8")

    update_readme_file(readme_path, res_dir)
    content_second = readme_path.read_text(encoding="utf-8")

    assert content_first == content_second
    assert "Footer." in content_first
    assert "# Title" in content_first
