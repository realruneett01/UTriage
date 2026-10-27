"""Fill placeholders in documentation templates from experiment results JSON files."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([^:]+?)\s*:\s*(.+?)\s*\}\}")


def _resolve_in_list(obj: list[Any] | tuple[Any, ...], path: str) -> tuple[bool, Any]:
    """Resolve remaining path against a list using integer index."""
    parts = path.split(".", 1)
    token = parts[0]
    rem = parts[1] if len(parts) > 1 else ""
    if not token.isdigit():
        return False, None
    idx = int(token)
    if not (0 <= idx < len(obj)):
        return False, None
    return resolve_key_path(obj[idx], rem)


def _match_candidate(ks: str, path: str) -> tuple[str, str] | None:
    """Return candidate key and remainder if key matches path or prefix."""
    if path == ks:
        return (ks, "")
    if path.startswith(ks + "."):
        return (ks, path[len(ks) + 1 :])
    return None


def _find_dict_candidates(keys: Iterable[Any], path: str) -> list[tuple[str, str]]:
    """Find candidate matching keys and remainders sorted by key length descending."""
    candidates: list[tuple[str, str]] = []
    for k in keys:
        cand = _match_candidate(str(k), path)
        if cand is not None:
            candidates.append(cand)
    candidates.sort(key=lambda item: len(item[0]), reverse=True)
    return candidates


def _resolve_in_dict(obj: dict[Any, Any], path: str) -> tuple[bool, Any]:
    """Resolve remaining path against a dict trying longest key first."""
    candidates = _find_dict_candidates(obj.keys(), path)
    for ks, rem in candidates:
        ok, val = resolve_key_path(obj[ks], rem)
        if ok:
            return True, val
    return False, None


def resolve_key_path(obj: Any, path: str) -> tuple[bool, Any]:
    """Recursively resolve a dotted path in dicts and lists."""
    if not path:
        return True, obj
    if isinstance(obj, dict):
        return _resolve_in_dict(obj, path)
    if isinstance(obj, (list, tuple)):
        return _resolve_in_list(obj, path)
    return False, None


def format_value(val: Any) -> str:
    """Format extracted value for template substitution."""
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, float):
        return f"{val:.3g}"
    return str(val)


def load_json_data(file_path: Path) -> Any | None:
    """Load JSON from file if it exists, else return None."""
    if not file_path.is_file():
        return None
    try:
        with file_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        # Return None if file cannot be read or parsed as JSON
        return None


@dataclass
class RenderContext:
    """Context and cache for rendering template placeholders."""

    results_dir: Path
    cache: dict[str, Any] = field(default_factory=dict)
    filled: int = 0
    missing: int = 0


def _handle_missing(filename: str, key_path: str, ctx: RenderContext) -> str:
    """Record missing placeholder, write to stderr, and return fallback string."""
    ctx.missing += 1
    msg = f"[not available: {filename} {key_path}]"
    print(msg, file=sys.stderr)
    return msg


def _replace_placeholder(match: re.Match[str], ctx: RenderContext) -> str:
    """Process a single placeholder match."""
    filename = match.group(1).strip()
    key_path = match.group(2).strip()

    if filename not in ctx.cache:
        ctx.cache[filename] = load_json_data(ctx.results_dir / filename)

    data = ctx.cache[filename]
    if data is None:
        return _handle_missing(filename, key_path, ctx)

    ok, val = resolve_key_path(data, key_path)
    if not ok:
        return _handle_missing(filename, key_path, ctx)

    ctx.filled += 1
    return format_value(val)


def render_with_stats(
    template_text: str,
    results_dir: Path | str,
) -> tuple[str, int, int]:
    """Render template text, returning (rendered_text, filled_count, missing_count)."""
    ctx = RenderContext(results_dir=Path(results_dir))
    rendered = PLACEHOLDER_PATTERN.sub(
        lambda m: _replace_placeholder(m, ctx),
        template_text,
    )
    return rendered, ctx.filled, ctx.missing


def render(template_text: str, results_dir: Path | str) -> str:
    """Render template text replacing placeholders with results values."""
    rendered, _, _ = render_with_stats(template_text, results_dir)
    return rendered


def main() -> int:
    """Render docs/technical-note.tmpl.md into docs/technical-note.md."""
    repo_root = Path(__file__).resolve().parents[1]
    template_path = repo_root / "docs" / "technical-note.tmpl.md"
    output_path = repo_root / "docs" / "technical-note.md"
    results_dir = repo_root / "results"

    if not template_path.is_file():
        print(f"Error: Template not found at {template_path}", file=sys.stderr)
        return 1

    template_text = template_path.read_text(encoding="utf-8")
    rendered, filled, missing = render_with_stats(template_text, results_dir)

    output_path.write_text(rendered, encoding="utf-8")
    print(f"Filled: {filled}, Missing: {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
