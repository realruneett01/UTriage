"""Tests for offline bundle verification utility and packaging script."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from verify_bundle import main, verify_sums, write_sums  # noqa: E402


@dataclass(frozen=True)
class MainCase:
    """Settings for CLI main test case."""

    cmd: str
    corrupt: bool
    expected_code: int


@pytest.fixture
def sample_tree(tmp_path: Path) -> Path:
    """Create a temporary directory tree with sample files."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("hello world\n", encoding="utf-8")
    (tmp_path / "sub" / "b.bin").write_bytes(b"\x00\x01\x02\x03")
    return tmp_path


def test_write_and_verify_clean(sample_tree: Path) -> None:
    """Test write_sums followed by verify_sums returns no problems."""
    sums = sample_tree / "SHA256SUMS.txt"
    written = write_sums(sample_tree, sums)
    assert written == sums
    assert sums.is_file()
    problems = verify_sums(sample_tree, sums)
    assert problems == []


def _apply_mutation(root: Path, sums: Path, kind: str) -> None:
    """Apply a mutation to test tree or manifest."""
    if kind == "corrupt":
        (root / "a.txt").write_text("modified\n", encoding="utf-8")
    elif kind == "delete":
        (root / "a.txt").unlink()
    elif kind == "extra":
        (root / "new.txt").write_text("extra", encoding="utf-8")
    elif kind == "unsafe":
        with sums.open("a", encoding="utf-8") as f:
            f.write("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef  ../x\n")


@pytest.mark.parametrize(
    ("mutation_kind", "expected_prefix"),
    [
        ("corrupt", "mismatch: a.txt"),
        ("delete", "missing: a.txt"),
        ("extra", "unlisted: new.txt"),
        ("unsafe", "unsafe path: ../x"),
    ],
)
def test_verify_sums_anomalies(
    sample_tree: Path,
    mutation_kind: str,
    expected_prefix: str,
) -> None:
    """Test detection of various bundle integrity anomalies."""
    sums = sample_tree / "SHA256SUMS.txt"
    write_sums(sample_tree, sums)
    _apply_mutation(sample_tree, sums, mutation_kind)
    problems = verify_sums(sample_tree, sums)
    assert any(p.startswith(expected_prefix) for p in problems)


@pytest.mark.parametrize(
    "case",
    [
        MainCase(cmd="write", corrupt=False, expected_code=0),
        MainCase(cmd="verify", corrupt=False, expected_code=0),
        MainCase(cmd="verify", corrupt=True, expected_code=1),
    ],
)
def test_main_exit_codes(
    sample_tree: Path,
    case: MainCase,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test CLI subcommand exit codes and problem reporting."""
    write_sums(sample_tree, sample_tree / "SHA256SUMS.txt")
    if case.corrupt:
        (sample_tree / "a.txt").write_text("corrupted content", encoding="utf-8")
    code = main([case.cmd, str(sample_tree)])
    assert code == case.expected_code
    captured = capsys.readouterr()
    if case.expected_code == 1:
        assert "mismatch: a.txt" in captured.out


def test_ps1_offline_bundle_install_script_content() -> None:
    """Assert build_offline_bundle script embeds required offline install steps."""
    ps1_path = Path(__file__).resolve().parent.parent / "scripts" / "build_offline_bundle.ps1"
    content = ps1_path.read_text(encoding="utf-8")
    assert "--no-index" in content
    assert "verify_bundle.py verify" in content
