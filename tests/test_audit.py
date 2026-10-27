"""Tests for tamper-evident hash-chain audit log (audit.py and app_logic integration)."""

from __future__ import annotations

import csv
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        del sys.modules["utriage"]

import pytest  # noqa: E402

from utriage.app_logic import (  # noqa: E402
    Session,
    load_session,
    record_decision,
)
from utriage.audit import (  # noqa: E402
    GENESIS,
    AuditIntegrityError,
    ChainCheck,
    append_entry,
    export_csv,
    verify_chain,
)

SAMPLE_PATH = Path("data/samples/ti64/bar_0000.npz")


@pytest.fixture
def sample_session() -> Session:
    """Provide a Session initialized from the sample scan."""
    return load_session(SAMPLE_PATH, model_path=None)


@pytest.fixture
def nominal_chain_log(tmp_path: Path) -> Path:
    """Provide an audit.jsonl log populated with 5 nominal entries."""
    log_path = tmp_path / "audit.jsonl"
    for i in range(5):
        append_entry(log_path, {"step": i, "status": "nominal"})
    return log_path


def assert_valid_chain(check: ChainCheck, expected_n: int) -> None:
    """Assert that chain verification succeeded with expected entry count."""
    assert check.ok is True
    assert check.n == expected_n
    assert check.first_bad_line is None


def assert_tamper_detected(
    check: ChainCheck,
    expected_bad_line: int,
    reason_substr: str | None = None,
) -> None:
    """Assert that chain tampering was detected at the expected line."""
    assert check.ok is False
    assert check.first_bad_line == expected_bad_line
    if reason_substr:
        assert reason_substr in check.reason


def _tamper_edit(lines: list[str]) -> list[str]:
    bad_data = json.loads(lines[2])
    bad_data["status"] = "tampered"
    lines[2] = json.dumps(bad_data)
    return lines


def _tamper_delete(lines: list[str]) -> list[str]:
    del lines[2]
    return lines


def _tamper_swap(lines: list[str]) -> list[str]:
    lines[1], lines[2] = lines[2], lines[1]
    return lines


@pytest.mark.parametrize(
    ("filename", "create"),
    [("nonexistent.jsonl", False), ("empty.jsonl", True)],
    ids=["missing", "empty"],
)
def test_empty_or_missing_chain(tmp_path: Path, filename: str, create: bool) -> None:
    """Verify empty or non-existent file passes as an empty valid chain."""
    log_path = tmp_path / filename
    if create:
        log_path.touch()
    check = verify_chain(log_path)
    assert check == ChainCheck(ok=True, n=0, first_bad_line=None, reason="empty log")


def test_chain_of_5_verifies(tmp_path: Path) -> None:
    """Verify appending 5 sequential entries creates a valid hash chain."""
    log_path = tmp_path / "audit.jsonl"
    entries = [append_entry(log_path, {"action": f"evt_{i}", "val": i * 10}) for i in range(5)]

    assert len(entries) == 5
    assert entries[0]["prev"] == GENESIS
    assert entries[0]["seq"] == 0

    for i in range(1, 5):
        assert entries[i]["prev"] == entries[i - 1]["hash"]
        assert entries[i]["seq"] == i

    check = verify_chain(log_path)
    assert_valid_chain(check, expected_n=5)


@pytest.mark.parametrize(
    ("tamper_fn", "expected_bad_line", "reason_substr"),
    [
        (_tamper_edit, 3, "hash mismatch"),
        (_tamper_delete, 3, None),
        (_tamper_swap, 2, None),
    ],
    ids=["edited_field", "delete_middle_line", "swap_lines"],
)
def test_tamper_detection(
    nominal_chain_log: Path,
    tamper_fn: Callable[[list[str]], list[str]],
    expected_bad_line: int,
    reason_substr: str | None,
) -> None:
    """Verify tampering with chain entries fails verification on expected line."""
    lines = nominal_chain_log.read_text(encoding="utf-8").splitlines()
    modified = tamper_fn(lines)
    nominal_chain_log.write_text("\n".join(modified) + "\n", encoding="utf-8")

    check = verify_chain(nominal_chain_log)
    assert_tamper_detected(check, expected_bad_line, reason_substr)


def test_export_csv_rows_equal_entries(tmp_path: Path) -> None:
    """Verify export_csv outputs exactly one row per entry plus header."""
    log_path = tmp_path / "audit.jsonl"
    for i in range(5):
        append_entry(log_path, {"event": f"ev_{i}", "score": i * 1.5})

    csv_path = tmp_path / "audit.csv"
    res_path = export_csv(log_path, csv_path)
    assert res_path == csv_path
    assert res_path.is_file()

    with csv_path.open("r", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))

    # Header + 5 entries = 6 rows
    assert len(rows) == 6
    assert rows[0] == ["event", "score", "seq", "ts", "prev", "hash"]
    assert rows[1][0] == "ev_0"
    assert rows[5][0] == "ev_4"


def test_record_decision_writes_verifiable_chain(sample_session: Session, tmp_path: Path) -> None:
    """Verify record_decision updates audit CSV and writes a valid JSONL hash chain."""
    csv_path = tmp_path / "audit.csv"
    jsonl_path = tmp_path / "audit.jsonl"
    decisions = [("accepted", "First pass"), ("rejected", "Noise"), ("rescan", "Review")]

    for idx, (dec, comment) in enumerate(decisions):
        record_decision(
            session=sample_session,
            item_index=idx,
            decision=dec,
            comment=comment,
            inspector=f"Insp-{idx}",
            log_path=csv_path,
        )

    assert csv_path.is_file()
    assert jsonl_path.is_file()
    check = verify_chain(jsonl_path)
    assert_valid_chain(check, expected_n=len(decisions))


def test_record_decision_jsonl_content(sample_session: Session, tmp_path: Path) -> None:
    """Verify record_decision writes matching fields to JSONL entries."""
    csv_path = tmp_path / "audit.csv"
    record_decision(sample_session, 0, "accepted", "Approved", "Insp-0", csv_path)
    record_decision(sample_session, 1, "rejected", "Flaw", "Insp-1", csv_path)

    jsonl_path = csv_path.with_suffix(".jsonl")
    entries = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(entries) == 2
    assert entries[0]["decision"] == "accepted"
    assert entries[0]["inspector"] == "Insp-0"
    assert entries[1]["decision"] == "rejected"
    assert entries[1]["inspector"] == "Insp-1"


@pytest.mark.parametrize(
    "corrupt_tail",
    [
        "not-valid-json\n",
        "42\n",
        '{"seq": 0}\n',
        '{"seq": 0, "ts": "2026-01-01", "prev": "0000", "hash": "wrong"}\n',
    ],
    ids=["malformed_json", "non_dict", "missing_fields", "hash_mismatch"],
)
def test_malformed_tail_raises_and_never_appends(
    tmp_path: Path,
    corrupt_tail: str,
) -> None:
    """Verify append_entry raises AuditIntegrityError on malformed tail without appending."""
    log_path = tmp_path / "corrupt.jsonl"
    log_path.write_text(corrupt_tail, encoding="utf-8")
    initial_content = log_path.read_text(encoding="utf-8")

    with pytest.raises(AuditIntegrityError):
        append_entry(log_path, {"action": "should_fail"})

    assert log_path.read_text(encoding="utf-8") == initial_content


def test_truncating_last_line_detected(nominal_chain_log: Path) -> None:
    """Verify deleting the last entry is caught as truncation by verify_chain."""
    lines = nominal_chain_log.read_text(encoding="utf-8").splitlines()
    truncated = lines[:-1]
    nominal_chain_log.write_text("\n".join(truncated) + "\n", encoding="utf-8")

    check = verify_chain(nominal_chain_log)
    assert check.ok is False
    assert check.n == 4
    assert check.first_bad_line is None
    assert "truncation" in check.reason.lower()
    assert "count mismatch" in check.reason.lower()


def test_missing_anchor_fails_non_empty_log(nominal_chain_log: Path) -> None:
    """Verify a non-empty log without an anchor fails verification."""
    anchor_path = nominal_chain_log.with_name(f"{nominal_chain_log.name}.head.json")
    if anchor_path.is_file():
        anchor_path.unlink()

    check = verify_chain(nominal_chain_log)
    assert check.ok is False
    assert "missing anchor" in check.reason.lower()


def test_anchor_written_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify anchor is written atomically via os.replace after append."""
    log_path = tmp_path / "audit.jsonl"
    calls: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy_replace(src: str | Path, dst: str | Path) -> None:
        calls.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy_replace)
    append_entry(log_path, {"event": "init"})

    anchor_path = log_path.with_name(f"{log_path.name}.head.json")
    assert anchor_path.is_file()
    assert len(calls) == 1
    src_call, dst_call = calls[0]
    assert dst_call == str(anchor_path)
    assert src_call.endswith(".tmp")

    anchor_data = json.loads(anchor_path.read_text(encoding="utf-8"))
    assert anchor_data["count"] == 1
    assert "last_hash" in anchor_data
