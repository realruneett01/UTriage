from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pytest  # noqa: E402

from utriage.integrity import (  # noqa: E402
    MANIFEST,
    IntegrityError,
    IntegrityReport,
    build_manifest,
    is_locked,
    require_locked,
    sha256_file,
    verify,
    write_manifest,
)


@pytest.fixture
def sample_workspace(tmp_path: Path) -> tuple[Path, list[Path]]:
    """Create a temporary workspace with two sample config files."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    file_a = cfg_dir / "materials.yaml"
    file_b = cfg_dir / "probe.yaml"
    file_a.write_text("density: 4430\n", encoding="utf-8")
    file_b.write_text("frequency_mhz: 5.0\n", encoding="utf-8")
    return tmp_path, [file_a, file_b]


@pytest.fixture
def locked_workspace(
    sample_workspace: tuple[Path, list[Path]],
) -> tuple[Path, list[Path], Path]:
    """Workspace with sample files and a written manifest.json."""
    root, files = sample_workspace
    manifest_path = write_manifest(files, root, root / "manifest.json")
    return root, files, manifest_path


def _assert_integrity_report(
    report: IntegrityReport,
    *,
    ok: bool,
    missing: tuple[str, ...] = (),
    mismatched: tuple[str, ...] = (),
) -> None:
    """Assert status, missing files, and mismatched files on an IntegrityReport."""
    assert isinstance(report, IntegrityReport)
    assert report.ok is ok
    assert report.missing == missing
    assert report.mismatched == mismatched


def test_sha256_binary_file(tmp_path: Path) -> None:
    data = b"ultrasonic testing shop floor integrity"
    target = tmp_path / "model.pt"
    target.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    assert sha256_file(target) == expected


def test_sha256_binary_preserves_cr_bytes(tmp_path: Path) -> None:
    bin_lf = tmp_path / "model_lf.pt"
    bin_crlf = tmp_path / "model_crlf.pt"
    bin_lf.write_bytes(b"model\ndata")
    bin_crlf.write_bytes(b"model\r\ndata")
    assert sha256_file(bin_lf) != sha256_file(bin_crlf)


@pytest.mark.parametrize(
    "extension",
    [".yaml", ".yml", ".json", ".toml", ".md", ".txt", ".csv"],
)
def test_sha256_text_line_ending_normalization(tmp_path: Path, extension: str) -> None:
    lf_path = tmp_path / f"test_lf{extension}"
    crlf_path = tmp_path / f"test_crlf{extension}"
    lf_path.write_bytes(b"key: value\nsecond: line\n")
    crlf_path.write_bytes(b"key: value\r\nsecond: line\r\n")
    assert sha256_file(lf_path) == sha256_file(crlf_path)


def test_sha256_text_real_content_change(tmp_path: Path) -> None:
    file_a = tmp_path / "base.yaml"
    file_b = tmp_path / "changed.yaml"
    file_a.write_text("density: 4430\n", encoding="utf-8")
    file_b.write_text("density: 4500\n", encoding="utf-8")
    assert sha256_file(file_a) != sha256_file(file_b)


def test_manifest_round_trip(sample_workspace: tuple[Path, list[Path]]) -> None:
    root, files = sample_workspace
    manifest_path = root / "manifest.json"
    written = write_manifest(files, root, manifest_path)
    assert written == manifest_path
    assert manifest_path.is_file()

    report = verify(manifest_path, root)
    _assert_integrity_report(report, ok=True)
    assert report.unlisted_checked == ()


@dataclass(frozen=True)
class TamperScenario:
    action: str
    file_index: int
    expected_missing: tuple[str, ...]
    expected_mismatched: tuple[str, ...]


@pytest.mark.parametrize(
    "case",
    [
        TamperScenario("edit", 0, (), ("configs/materials.yaml",)),
        TamperScenario("delete", 1, ("configs/probe.yaml",), ()),
    ],
    ids=["edited_file_mismatch", "deleted_file_missing"],
)
def test_verify_tampered_file(
    locked_workspace: tuple[Path, list[Path], Path],
    case: TamperScenario,
) -> None:
    root, files, manifest_path = locked_workspace
    if case.action == "edit":
        files[case.file_index].write_text("tampered_content: true\n", encoding="utf-8")
    else:
        files[case.file_index].unlink()

    report = verify(manifest_path, root)
    _assert_integrity_report(
        report,
        ok=False,
        missing=case.expected_missing,
        mismatched=case.expected_mismatched,
    )


def test_verify_missing_manifest_file(tmp_path: Path) -> None:
    missing_manifest = tmp_path / "nonexistent.json"
    report = verify(missing_manifest, tmp_path)
    _assert_integrity_report(report, ok=False, missing=("nonexistent.json",))


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        (None, False),
        ("", False),
        ("0", False),
        ("true", False),
        ("1", True),
    ],
)
def test_is_locked_env(
    monkeypatch: pytest.MonkeyPatch, env_value: str | None, expected: bool
) -> None:
    if env_value is None:
        monkeypatch.delenv("UTRIAGE_LOCKED", raising=False)
    else:
        monkeypatch.setenv("UTRIAGE_LOCKED", env_value)
    assert is_locked() is expected


@dataclass(frozen=True)
class LockFaultScenario:
    mutate_first: bool
    mutate_second: bool
    expected_missing: tuple[str, ...]
    expected_mismatched: tuple[str, ...]


@pytest.mark.parametrize(
    "case",
    [
        LockFaultScenario(False, True, (), ("configs/probe.yaml",)),
        LockFaultScenario(True, False, ("configs/materials.yaml",), ()),
        LockFaultScenario(True, True, ("configs/materials.yaml",), ("configs/probe.yaml",)),
    ],
)
def test_require_locked_raises_naming_files(
    locked_workspace: tuple[Path, list[Path], Path],
    monkeypatch: pytest.MonkeyPatch,
    case: LockFaultScenario,
) -> None:
    root, files, manifest_path = locked_workspace

    if case.mutate_first:
        files[0].unlink()
    if case.mutate_second:
        files[1].write_text("corrupted\n", encoding="utf-8")

    monkeypatch.setenv("UTRIAGE_LOCKED", "1")
    with pytest.raises(IntegrityError) as exc_info:
        require_locked(manifest_path, root)

    assert issubclass(IntegrityError, RuntimeError)
    err_msg = str(exc_info.value)
    for expected_path in case.expected_missing + case.expected_mismatched:
        assert expected_path in err_msg


@pytest.mark.parametrize("env_val", [None, "0"])
def test_require_locked_bypassed_when_not_locked(
    locked_workspace: tuple[Path, list[Path], Path],
    monkeypatch: pytest.MonkeyPatch,
    env_val: str | None,
) -> None:
    root, files, manifest_path = locked_workspace
    files[0].unlink()

    if env_val is None:
        monkeypatch.delenv("UTRIAGE_LOCKED", raising=False)
    else:
        monkeypatch.setenv("UTRIAGE_LOCKED", env_val)
    require_locked(manifest_path, root)


def test_require_locked_passes_when_intact(
    locked_workspace: tuple[Path, list[Path], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _, manifest_path = locked_workspace
    monkeypatch.setenv("UTRIAGE_LOCKED", "1")
    require_locked(manifest_path, root)


def test_repo_manifest_integrity(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = MANIFEST.parents[1]
    assert MANIFEST.is_file()

    report = verify(MANIFEST, repo_root)
    _assert_integrity_report(report, ok=True)

    manifest_data = build_manifest([], repo_root)
    assert isinstance(manifest_data, dict)

    monkeypatch.setenv("UTRIAGE_LOCKED", "1")
    require_locked(MANIFEST, repo_root)
