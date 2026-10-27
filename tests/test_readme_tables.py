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


def _make_cell(rate: float, ci_low: float, ci_high: float, fcr: float) -> dict:
    """Build single evaluated metrics cell."""
    return {
        "available": True,
        "detection": {
            "n": 100,
            "k": int(rate * 100),
            "rate": rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
        },
        "false_calls": {"rate": fcr, "n_false": int(fcr * 200), "n_scans": 200},
    }


def _make_fake_matrix() -> dict:
    """Build dummy generalization matrix with real and synthetic columns."""
    cols = [
        {"id": "real_loco", "group": "real", "data_label": "Real data (ML-NDT)"},
        {"id": "synth_noise", "group": "synthetic", "data_label": "Synthetic data"},
    ]
    rows = {
        "rules": {
            "real_loco": _make_cell(0.95, 0.887, 0.984, 0.005),
            "synth_noise": _make_cell(0.80, 0.663, 0.899, 0.012),
        },
        "supervised": {
            "real_loco": _make_cell(0.98, 0.930, 0.998, 0.002),
            "synth_noise": _make_cell(0.90, 0.782, 0.967, 0.008),
        },
        "anomaly": {
            "real_loco": {"available": False, "reason": "no flaw-free scans"},
            "synth_noise": _make_cell(0.70, 0.554, 0.821, 0.010),
        },
        "fused": {
            "real_loco": _make_cell(0.99, 0.946, 0.999, 0.006),
            "synth_noise": _make_cell(0.96, 0.863, 0.995, 0.015),
        },
    }
    return {"columns": cols, "rows": rows}


def _populate_fake_results(results_dir: Path, leaky: bool = True) -> None:
    """Populate directory with fake Phase 8 result files."""
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_json(results_dir / "generalization_matrix.json", _make_fake_matrix())
    _write_json(results_dir / "leakage_audit.json", {"leakage_by_physical_crack": leaky})
    inv_text = (
        "# Real-data evidence limits\n\n"
        "Across the 6 accepted datasets, 3 datasets have verified counts totaling "
        "10 distinct physical flaws.\n"
    )
    (results_dir / "data_inventory.md").write_text(inv_text, encoding="utf-8")


def test_missing_files_produce_missing_line(tmp_path: Path) -> None:
    """Verify missing result files yield missing file notices instead of crashing."""
    empty_dir = tmp_path / "empty_results"
    empty_dir.mkdir()
    rendered = render_all_tables(empty_dir)
    assert "data_inventory.md not found" in rendered
    assert "leakage_audit.json not found" in rendered
    assert "generalization_matrix.json not found" in rendered


def test_both_tables_present(tmp_path: Path) -> None:
    """Verify both Real data and Synthetic data generalization tables are rendered."""
    res_dir = tmp_path / "results"
    _populate_fake_results(res_dir)
    rendered = render_all_tables(res_dir)
    assert "(Real data" in rendered
    assert "(Synthetic data" in rendered
    for row_name in ("rules", "supervised", "anomaly", "fused"):
        assert f"| {row_name} |" in rendered


def test_na_cells_carry_reasons(tmp_path: Path) -> None:
    """Verify unavailable cells output n/a with their specific reason."""
    res_dir = tmp_path / "results"
    _populate_fake_results(res_dir)
    rendered = render_all_tables(res_dir)
    assert "n/a (no flaw-free scans)" in rendered


@pytest.mark.parametrize(
    ("leaky", "expect_superseded"),
    [
        (True, True),
        (False, False),
    ],
)
def test_superseded_note_leakage(tmp_path: Path, leaky: bool, expect_superseded: bool) -> None:
    """Verify superseded note is shown only when leakage_by_physical_crack is true."""
    res_dir = tmp_path / f"results_{leaky}"
    _populate_fake_results(res_dir, leaky=leaky)
    rendered = render_all_tables(res_dir)
    target_phrase = "superseded by the leave-one-crack-out results"
    if expect_superseded:
        assert target_phrase in rendered
    else:
        assert "superseded" not in rendered.lower()


def test_evidence_limits_box(tmp_path: Path) -> None:
    """Verify Evidence limits box contains required statements."""
    res_dir = tmp_path / "results"
    _populate_fake_results(res_dir)
    rendered = render_all_tables(res_dir)
    assert "> ### Evidence limits" in rendered
    assert "10" in rendered
    assert "weld PAUT datasets not bars" in rendered
    assert "UNVERIFIED" in rendered
    assert "advisory independent prototype" in rendered


def test_links_line(tmp_path: Path) -> None:
    """Verify links to claims and matrix figure are present."""
    res_dir = tmp_path / "results"
    _populate_fake_results(res_dir)
    rendered = render_all_tables(res_dir)
    assert "results/claims.md" in rendered
    assert "results/figures/generalization_matrix.png" in rendered


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
def test_no_forbidden_phrases(tmp_path: Path, phrase: str) -> None:
    """Verify forbidden phrases never appear in rendered results text."""
    res_dir = tmp_path / "results"
    _populate_fake_results(res_dir)
    rendered = render_all_tables(res_dir)
    assert phrase.lower() not in rendered.lower()


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
    _populate_fake_results(res_dir)

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
    _populate_fake_results(res_dir)

    update_readme_file(readme_path, res_dir)
    content_first = readme_path.read_text(encoding="utf-8")

    update_readme_file(readme_path, res_dir)
    content_second = readme_path.read_text(encoding="utf-8")

    assert content_first == content_second
    assert "Footer." in content_first
    assert "# Title" in content_first
