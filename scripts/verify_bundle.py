"""Checksum creation and integrity checking for offline bundles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path if repository root is present
src_dir = Path(__file__).resolve().parent.parent / "src"
if src_dir.is_dir() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from utriage.integrity import sha256_file
except ImportError:
    import hashlib

    # Fallback when running in an uninstalled python environment
    _TEXT_EXTS = frozenset({".yaml", ".yml", ".json", ".toml", ".md", ".txt", ".csv"})

    def sha256_file(path: Path) -> str:
        """Compute sha256 digest with newline normalization for text files."""
        hasher = hashlib.sha256()
        if path.suffix.lower() in _TEXT_EXTS:
            raw = path.read_bytes()
            normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            hasher.update(normalized)
            return hasher.hexdigest()
        with path.open("rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()


def _is_unsafe(path_str: str) -> bool:
    """Check if relative path string contains unsafe components."""
    if path_str.startswith(("/", "\\")):
        return True
    path_obj = Path(path_str)
    return path_obj.is_absolute() or (".." in path_obj.parts)


def write_sums(root: Path, out: Path) -> Path:
    """Write SHA256 digests of all files under root to output path."""
    lines: list[str] = []
    out_path_abs = out.resolve()
    excluded = {"SHA256SUMS.txt", out.name}
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        if item.name in excluded or item.resolve() == out_path_abs:
            continue
        rel_posix = item.relative_to(root).as_posix()
        digest = sha256_file(item)
        lines.append(f"{digest}  {rel_posix}\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8")
    return out


def _check_entry(root: Path, rel_str: str, expected_hex: str) -> str | None:
    """Check single path entry against disk file and expected digest."""
    if _is_unsafe(rel_str):
        return f"unsafe path: {rel_str}"
    target = root / rel_str
    if not target.is_file():
        return f"missing: {rel_str}"
    if sha256_file(target).lower() != expected_hex.lower():
        return f"mismatch: {rel_str}"
    return None


def _find_unlisted(root: Path, sums: Path, seen: set[str]) -> list[str]:
    """Find files on disk that are not in checksum manifest."""
    unlisted: list[str] = []
    sums_abs = sums.resolve()
    excluded = {"SHA256SUMS.txt", sums.name}
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        if item.name in excluded or item.resolve() == sums_abs:
            continue
        rel = item.relative_to(root).as_posix()
        if rel not in seen:
            unlisted.append(f"unlisted: {rel}")
    return unlisted


def _process_manifest_line(root: Path, line: str, seen: set[str]) -> str | None:
    """Parse single manifest line and check corresponding file."""
    parts = line.strip().split(maxsplit=1)
    if len(parts) != 2:
        return None
    expected_hex, rel_str = parts[0], parts[1].strip()
    if not _is_unsafe(rel_str):
        seen.add(Path(rel_str).as_posix())
    return _check_entry(root, rel_str, expected_hex)


def verify_sums(root: Path, sums: Path) -> list[str]:
    """Check integrity of files under root against manifest."""
    if not sums.is_file():
        return [f"missing: {sums.as_posix()}"]
    problems: list[str] = []
    seen: set[str] = set()
    for raw_line in sums.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        if issue := _process_manifest_line(root, raw_line, seen):
            problems.append(issue)
    problems.extend(_find_unlisted(root, sums, seen))
    return problems


def _build_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser for bundle operations."""
    parser = argparse.ArgumentParser(description="Bundle checksum utility")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_write = subparsers.add_parser("write")
    p_write.add_argument("root", nargs="?", default=".", type=Path)
    p_write.add_argument("--out", default=None, type=Path)

    p_verify = subparsers.add_parser("verify")
    p_verify.add_argument("root", nargs="?", default=".", type=Path)
    p_verify.add_argument("--sums", default=None, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run CLI for writing or checking bundle checksums."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "write":
        out = args.out or (args.root / "SHA256SUMS.txt")
        write_sums(args.root, out)
        return 0
    sums = args.sums or (args.root / "SHA256SUMS.txt")
    problems = verify_sums(args.root, sums)
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
