"""Unit tests for failure mode classification and artifact generation (project plan Phase 8).

Verifies classify_failure branches, failure case rendering, and markdown report synthesis
using synthetic hand-made dictionaries without touching real ML-NDT data or models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from utriage.report.failures import (  # noqa: E402
    build_failure_modes_md,
    classify_failure,
    save_failure_case,
)


@pytest.mark.parametrize(
    ("meta", "kind", "expected"),
    [
        (
            {
                "depth": 10.0,
                "max_amplitude": 5000,
                "noise_threshold": 1000,
                "equivalent_flawsize": 3.0,
            },
            "miss",
            "near_wall",
        ),
        (
            {
                "depth": 245.0,
                "max_amplitude": 5000,
                "noise_threshold": 1000,
                "equivalent_flawsize": 3.0,
            },
            "miss",
            "near_wall",
        ),
        (
            {
                "location": "10-20",
                "max_amplitude": 5000,
                "noise_threshold": 1000,
                "equivalent_flawsize": 3.0,
            },
            "miss",
            "near_wall",
        ),
        (
            {
                "location": 100.0,
                "max_amplitude": 1500,
                "noise_threshold": 1000,
                "equivalent_flawsize": 3.0,
            },
            "miss",
            "low_snr",
        ),
        (
            {
                "location": 100.0,
                "max_amplitude": 5000,
                "noise_threshold": 1000,
                "equivalent_flawsize": 1.0,
            },
            "miss",
            "small_flaw",
        ),
        (
            {
                "location": 100.0,
                "max_amplitude": 5000,
                "noise_threshold": 1000,
                "equivalent_flawsize": 3.0,
            },
            "miss",
            "unclassified",
        ),
        (
            {"max_amplitude": 2000, "noise_threshold": 1000},
            "false_call",
            "artefact_like",
        ),
        (
            {"image_max": 2500, "noise_threshold": 1000},
            "false_call",
            "artefact_like",
        ),
        (
            {"max_amplitude": 800, "noise_threshold": 1000},
            "false_call",
            "unclassified",
        ),
        (
            {"max_amplitude": 5000},
            "unknown_kind",
            "unclassified",
        ),
    ],
)
def test_classify_failure_branches(meta: dict, kind: str, expected: str) -> None:
    """Verify each discrete decision branch for misses and false calls."""
    assert classify_failure(meta, kind) == expected


def test_classify_failure_custom_p10() -> None:
    """Verify custom detected-size p10 threshold override."""
    meta = {
        "location": 100.0,
        "max_amplitude": 5000,
        "noise_threshold": 1000,
        "equivalent_flawsize": 1.8,
    }
    assert classify_failure(meta, "miss", p10_threshold=2.0) == "small_flaw"
    assert classify_failure(meta, "miss", p10_threshold=1.5) == "unclassified"


def test_save_failure_case_artifacts(tmp_path: Path) -> None:
    """Verify PNG plot and evidence JSON generation in destination directory."""
    img = np.zeros((256, 256), dtype=np.uint16)
    img[120, 80] = 4500
    meta = {"size": "2.5", "noise_threshold": "950"}

    saved_path = save_failure_case(img, meta, score=0.88, out_dir=tmp_path, name="case_001")
    assert saved_path == tmp_path / "case_001.png"
    assert saved_path.exists()
    assert saved_path.stat().st_size > 0

    json_path = tmp_path / "case_001.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["name"] == "case_001"
    assert payload["score"] == pytest.approx(0.88)
    assert payload["peak_amplitude"] == pytest.approx(4500.0)
    assert payload["peak_location"] == [120, 80]


def _fake_sim_data() -> dict:
    """Construct mock simulation generalization payload."""
    return {
        "test_sets": {
            "in_distribution": {
                "rules": {"false_calls": {"rate": 0.03}},
                "ml_1d": {"false_calls": {"rate": 0.13, "n_false": 55, "n_scans": 410}},
                "union": {"false_calls": {"rate": 0.15}},
            },
            "freq_10mhz": {
                "rules": {"false_calls": {"rate": 0.64}},
                "ml_1d": {"false_calls": {"rate": 0.47, "n_false": 197, "n_scans": 420}},
                "union": {"false_calls": {"rate": 0.69}},
            },
            "noise_above_range": {
                "rules": {"false_calls": {"rate": 0.53}},
                "ml_1d": {"false_calls": {"rate": 0.47, "n_false": 211, "n_scans": 450}},
                "union": {"false_calls": {"rate": 0.66}},
            },
        }
    }


def _make_fake_markdown_inputs() -> tuple[list[dict], list[dict], dict]:
    """Construct mock data structures for failure modes markdown generation."""
    summaries = [
        {
            "fold": "crack_1.6",
            "n_misses": 4,
            "miss_causes": {"small_flaw": 2, "low_snr": 1, "near_wall": 1, "unclassified": 0},
            "n_false_calls": 3,
            "false_call_causes": {"artefact_like": 2, "unclassified": 1},
        }
    ]
    cases = [
        {
            "name": "miss_001",
            "fold": "crack_1.6",
            "kind": "miss",
            "score": 0.12,
            "cause": "small_flaw",
            "evidence": "size=0.80mm, snr=3.2",
        }
    ]
    return summaries, cases, _fake_sim_data()


def test_build_failure_modes_md() -> None:
    """Verify markdown output contains required sections, counts, and descriptions."""
    summaries, cases, sim_data = _make_fake_markdown_inputs()
    md = build_failure_modes_md(summaries, cases, sim_data)

    assert "# Failure Modes and Generalization Boundaries" in md
    assert "## Plain-Language Summary" in md
    assert "## LOCO Folds Failure Breakdown" in md
    assert "| crack_1.6 | 4 | 1 | 1 | 2 | 0 | 3 | 2 | 1 |" in md
    assert "| miss_001 | crack_1.6 | miss | 0.120 | small_flaw | size=0.80mm, snr=3.2 |" in md
    assert "## Synthetic Held-Out Regions Generalization" in md
    assert "10 MHz Frequency Shift" in md
    assert "Noise Above Range" in md
