"""Tests for template placeholder filling (scripts/fill_placeholders.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from fill_placeholders import (  # noqa: E402
    format_value,
    main,
    render,
    render_with_stats,
    resolve_key_path,
)


def _write_json(path: Path, data: Any) -> None:
    """Helper to write data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh)


def test_dotted_keys_with_dots_inside_key_names(tmp_path: Path) -> None:
    """Resolve key paths where dictionary keys contain dots."""
    payload = {
        "domain": {
            "noise_2.5x": {
                "rules": {
                    "detection": {
                        "0.8": 0.95,
                    }
                }
            }
        }
    }
    _write_json(tmp_path / "metrics.json", payload)
    template = "Rate: {{metrics.json: domain.noise_2.5x.rules.detection.0.8}}"
    assert render(template, tmp_path) == "Rate: 0.95"


def test_greedy_longest_matching_key(tmp_path: Path) -> None:
    """Verify greedy resolution picks the longest matching key first."""
    payload = {
        "a.b": {"c": 42},
        "a": {"b.c": 99},
    }
    _write_json(tmp_path / "greedy.json", payload)
    template = "Val: {{greedy.json: a.b.c}}"
    assert render(template, tmp_path) == "Val: 42"


def test_list_index(tmp_path: Path) -> None:
    """Resolve list indices and nested dicts inside lists."""
    payload = {
        "items": ["first", "second", "third"],
        "users": [{"name": "alice"}, {"name": "bob"}],
    }
    _write_json(tmp_path / "lists.json", payload)
    template = "Item: {{lists.json: items.1}}, User: {{lists.json: users.0.name}}"
    assert render(template, tmp_path) == "Item: second, User: alice"


def test_missing_key_message(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Missing file or key renders fallback and writes message to stderr."""
    _write_json(tmp_path / "present.json", {"known": 1})
    template = "A: {{present.json: missing_key}}, B: {{absent.json: any_key}}"
    out, filled, missing = render_with_stats(template, tmp_path)

    expected = (
        "A: [not available: present.json missing_key], B: [not available: absent.json any_key]"
    )
    assert out == expected
    assert filled == 0
    assert missing == 2

    stderr_out = capsys.readouterr().err
    assert "[not available: present.json missing_key]" in stderr_out
    assert "[not available: absent.json any_key]" in stderr_out


@pytest.mark.parametrize(
    ("val", "expected"),
    [
        (123.456, "123"),
        (12.3456, "12.3"),
        (1.23456, "1.23"),
        (0.123456, "0.123"),
        (0.017857, "0.0179"),
        (0.003843, "0.00384"),
        (0.8, "0.8"),
        (1.0, "1"),
        (0.0, "0"),
    ],
)
def test_format_floats(val: float, expected: str, tmp_path: Path) -> None:
    """Format floating point numbers to 3 significant digits."""
    assert format_value(val) == expected
    _write_json(tmp_path / "num.json", {"val": val})
    template = "{{num.json: val}}"
    assert render(template, tmp_path) == expected


@pytest.mark.parametrize(
    ("val", "expected"),
    [
        (True, "True"),
        (False, "False"),
        (42, "42"),
        ("abc", "abc"),
    ],
)
def test_format_non_floats(val: Any, expected: str, tmp_path: Path) -> None:
    """Format boolean, integer, and string types without float conversion."""
    assert format_value(val) == expected
    _write_json(tmp_path / "raw.json", {"val": val})
    template = "{{raw.json: val}}"
    assert render(template, tmp_path) == expected


def test_idempotence(tmp_path: Path) -> None:
    """Rendering twice on output produces the exact same string."""
    _write_json(tmp_path / "data.json", {"val": 1.2345})
    template = "Value: {{data.json: val}}, Missing: {{data.json: not_found}}"
    first_pass = render(template, tmp_path)
    second_pass = render(first_pass, tmp_path)
    fresh_pass = render(template, tmp_path)

    assert first_pass == second_pass
    assert first_pass == fresh_pass


def test_resolve_key_path_edge_cases() -> None:
    """Verify resolve_key_path handles invalid inputs and list out-of-bounds."""
    assert resolve_key_path(None, "any")[0] is False
    assert resolve_key_path(["a"], "5")[0] is False
    assert resolve_key_path(["a"], "invalid")[0] is False
    assert resolve_key_path({"k": 1}, "") == (True, {"k": 1})


def test_main_execution(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify main writes technical-note.md and reports stats."""
    repo_root = Path(__file__).resolve().parents[1]
    note_path = repo_root / "docs" / "technical-note.md"

    # Ensure clean state before running
    if note_path.exists():
        note_path.unlink()

    exit_code = main()
    assert exit_code == 0
    assert note_path.is_file()

    stdout_out = capsys.readouterr().out
    assert "Filled: 57, Missing: 0" in stdout_out
