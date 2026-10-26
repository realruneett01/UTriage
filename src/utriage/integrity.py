"""Configuration control and file integrity verification for shop-floor use."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST: Path = Path(__file__).resolve().parents[2] / "configs" / "manifest.json"

TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {".yaml", ".yml", ".json", ".toml", ".md", ".txt", ".csv"}
)


class IntegrityError(RuntimeError):
    """Raised when configuration integrity verification fails under locked mode."""


@dataclass(frozen=True)
class IntegrityReport:
    """Report summarizing configuration file integrity verification."""

    ok: bool
    mismatched: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    unlisted_checked: tuple[str, ...] = ()


def is_locked() -> bool:
    """Check whether configuration lock mode is enabled via environment variable."""
    return os.environ.get("UTRIAGE_LOCKED") == "1"


def sha256_file(path: Path) -> str:
    """Compute sha256 digest of a file.

    Text files (.yaml, .yml, .json, .toml, .md, .txt, .csv) are hashed after
    normalising CRLF and CR to LF so that cross-platform line ending differences
    (such as git checkout on Windows converting LF to CRLF) do not alter the hash.
    Binary files (e.g. .pt, .bin) are hashed as raw bytes.
    """
    hasher = hashlib.sha256()
    if path.suffix.lower() in TEXT_EXTENSIONS:
        raw = path.read_bytes()
        normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        hasher.update(normalized)
        return hasher.hexdigest()

    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _relative_posix(path: Path, root: Path) -> str:
    """Return path relative to root formatted as a POSIX string."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        pass
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def build_manifest(paths: list[Path], root: Path) -> dict[str, Any]:
    """Build manifest dictionary containing SHA-256 digests of given paths."""
    files: dict[str, str] = {}
    for path in sorted(paths, key=lambda p: _relative_posix(p, root)):
        files[_relative_posix(path, root)] = sha256_file(path)
    return {
        "version": 1,
        "files": files,
    }


def write_manifest(paths: list[Path], root: Path, out: Path) -> Path:
    """Write manifest dictionary to output JSON file and return the output path."""
    manifest_dict = build_manifest(paths, root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest_dict, indent=2) + "\n", encoding="utf-8")
    return out


def verify(manifest_path: Path, root: Path) -> IntegrityReport:
    """Verify integrity of files listed in manifest against files under root."""
    if not manifest_path.is_file():
        rel = _relative_posix(manifest_path, root)
        return IntegrityReport(ok=False, missing=(rel,))

    content = json.loads(manifest_path.read_text(encoding="utf-8"))
    files_map: dict[str, str] = content.get("files", {})

    mismatched: list[str] = []
    missing: list[str] = []

    for rel_posix, expected_hash in sorted(files_map.items()):
        target = root / rel_posix
        if not target.is_file():
            missing.append(rel_posix)
        elif sha256_file(target) != expected_hash:
            mismatched.append(rel_posix)

    ok = not mismatched and not missing
    return IntegrityReport(
        ok=ok,
        mismatched=tuple(mismatched),
        missing=tuple(missing),
        unlisted_checked=(),
    )


def require_locked(manifest_path: Path, root: Path) -> None:
    """Enforce configuration integrity when UTRIAGE_LOCKED=1."""
    if not is_locked():
        return
    report = verify(manifest_path, root)
    if report.ok:
        return

    problems: list[str] = []
    if report.missing:
        problems.append(f"missing: {', '.join(report.missing)}")
    if report.mismatched:
        problems.append(f"mismatched: {', '.join(report.mismatched)}")

    detail = "; ".join(problems)
    raise IntegrityError(f"Locked configuration check failed: {detail}")
