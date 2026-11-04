"""Tests for scripts/make_verification_checklist.py."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from make_verification_checklist import (  # noqa: E402
    collect_rows,
    extract_unit_hint,
    main,
    render_markdown,
)


@pytest.mark.parametrize(
    ("key_path", "expected_hint"),
    [
        ("velocity_m_s", "_m_s"),
        ("density_kg_m3", "_kg_m3"),
        ("front_dead_zone_mm", "_mm"),
        ("record_pct", "_pct"),
        ("frequency_mhz", "_mhz"),
        ("attenuation_db_mm_mhz", "_db_mm_mhz"),
        ("gain_db", ""),
        ("ranges.focal_depth_mm.min", "_mm"),
        ("ranges.frequency_mhz.max", "_mhz"),
        ("interface_threshold", ""),
    ],
)
def test_unit_hint_extraction(key_path: str, expected_hint: str) -> None:
    """Verify suffix extraction maps correctly or returns empty string."""
    assert extract_unit_hint(key_path) == expected_hint


def _write_yaml(dir_path: Path, filename: str, content: dict[str, Any]) -> Path:
    """Helper to write YAML content into a directory."""
    path = dir_path / filename
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(content, fh)
    return path


def test_nested_keys_produce_dot_paths(tmp_path: Path) -> None:
    """Nested mappings inside unverified config produce dot-separated key paths."""
    sample = {
        "source": "UNVERIFIED - Ruri to check",
        "parent": {
            "child": {
                "depth_mm": 12.5,
                "count": 4,
                "label": "ignore_string",
                "flag": True,
            }
        },
    }
    _write_yaml(tmp_path, "sample.yaml", sample)

    rows = collect_rows(tmp_path)
    paths = {r["key_path"]: r["current_value"] for r in rows}

    assert paths == {
        "parent.child.depth_mm": 12.5,
        "parent.child.count": 4,
    }
    assert all(r["config_file"] == "sample.yaml" for r in rows)


def test_verified_mappings_are_skipped(tmp_path: Path) -> None:
    """Mappings with source not containing UNVERIFIED or PLACEHOLDER are skipped."""
    sample = {
        "verified_section": {
            "source": "ASTM E2375 Class AA standard",
            "speed_m_s": 5900.0,
        },
        "unverified_section": {
            "source": "UNVERIFIED placeholder",
            "speed_m_s": 6100.0,
        },
    }
    _write_yaml(tmp_path, "materials.yaml", sample)

    rows = collect_rows(tmp_path)
    paths = [r["key_path"] for r in rows]

    assert "unverified_section.speed_m_s" in paths
    assert "verified_section.speed_m_s" not in paths


@pytest.mark.parametrize(
    ("marker_text", "should_include"),
    [
        ("UNVERIFIED - Ruri to check", True),
        ("PLACEHOLDER only", True),
        ("unverified lower", True),
        ("placeholder lower", True),
        ("Official manufacturer datasheet", False),
    ],
)
def test_marker_detection(tmp_path: Path, marker_text: str, should_include: bool) -> None:
    """Detect unverified markers across uppercase, lowercase and placeholders."""
    sample = {
        "item": {
            "source": marker_text,
            "threshold_pct": 50.0,
        }
    }
    _write_yaml(tmp_path, "test_marker.yaml", sample)

    rows = collect_rows(tmp_path)
    found = any(r["key_path"] == "item.threshold_pct" for r in rows)
    assert found is should_include


def test_render_contains_header_and_rows() -> None:
    """Markdown rendering must contain required headers and one row per entry."""
    test_rows = [
        {
            "config_file": "acceptance.yaml",
            "key_path": "gates.front_dead_zone_mm",
            "current_value": 2.0,
            "unit_hint": "_mm",
            "source_text": "UNVERIFIED - test",
            "verified_against": "",
            "checked_by": "",
        },
        {
            "config_file": "acceptance.yaml",
            "key_path": "amplitude.record_pct",
            "current_value": 50.0,
            "unit_hint": "_pct",
            "source_text": "UNVERIFIED - test",
            "verified_against": "",
            "checked_by": "",
        },
    ]
    md = render_markdown(test_rows)

    assert "scripts/make_verification_checklist.py" in md
    assert "Verified against" in md
    assert "applicable standard" in md
    assert "| acceptance.yaml | gates.front_dead_zone_mm | 2.0 | _mm |" in md
    assert "| acceptance.yaml | amplitude.record_pct | 50.0 | _pct |" in md


def test_main_writes_file(tmp_path: Path) -> None:
    """CLI main parses arguments and writes markdown table to target path."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    sample = {
        "status": "PLACEHOLDER",
        "speed_m_s": 1500.0,
    }
    _write_yaml(cfg_dir, "test.yaml", sample)

    out_file = tmp_path / "docs" / "checklist.md"
    main(["--configs", str(cfg_dir), "--out", str(out_file)])

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "test.yaml" in content
    assert "speed_m_s" in content
