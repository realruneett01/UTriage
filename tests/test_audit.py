"""Tests for tamper-evident hash-chain audit log (audit.py and app_logic integration)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        del sys.modules["utriage"]

from utriage.app_logic import load_session, record_decision  # noqa: E402
from utriage.audit import (  # noqa: E402
    GENESIS,
    ChainCheck,
    append_entry,
    export_csv,
    verify_chain,
)

SAMPLE_PATH = Path("data/samples/ti64/bar_0000.npz")


def test_empty_or_missing_chain(tmp_path: Path) -> None:
    """Verify empty or non-existent file passes as an empty valid chain."""
    missing_path = tmp_path / "nonexistent.jsonl"
    check = verify_chain(missing_path)
    assert check == ChainCheck(ok=True, n=0, first_bad_line=None, reason="empty log")

    empty_path = tmp_path / "empty.jsonl"
    empty_path.touch()
    check_empty = verify_chain(empty_path)
    assert check_empty.ok is True
    assert check_empty.n == 0


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
    assert check.ok is True
    assert check.n == 5
    assert check.first_bad_line is None


def test_tamper_edited_field(tmp_path: Path) -> None:
    """Verify editing a field in line 3 fails verification on line 3."""
    log_path = tmp_path / "audit.jsonl"
    for i in range(5):
        append_entry(log_path, {"step": i, "status": "nominal"})

    lines = log_path.read_text(encoding="utf-8").splitlines()
    bad_line_data = json.loads(lines[2])
    bad_line_data["status"] = "tampered"
    lines[2] = json.dumps(bad_line_data)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    check = verify_chain(log_path)
    assert check.ok is False
    assert check.first_bad_line == 3
    assert "hash mismatch" in check.reason


def test_tamper_delete_middle_line(tmp_path: Path) -> None:
    """Verify deleting line 3 causes line 3 (was line 4) to fail."""
    log_path = tmp_path / "audit.jsonl"
    for i in range(5):
        append_entry(log_path, {"step": i, "status": "nominal"})

    lines = log_path.read_text(encoding="utf-8").splitlines()
    del lines[2]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    check = verify_chain(log_path)
    assert check.ok is False
    assert check.first_bad_line == 3


def test_tamper_swap_lines(tmp_path: Path) -> None:
    """Verify swapping line 2 and line 3 fails verification on line 2."""
    log_path = tmp_path / "audit.jsonl"
    for i in range(5):
        append_entry(log_path, {"step": i, "status": "nominal"})

    lines = log_path.read_text(encoding="utf-8").splitlines()
    lines[1], lines[2] = lines[2], lines[1]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    check = verify_chain(log_path)
    assert check.ok is False
    assert check.first_bad_line == 2


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


def test_record_decision_writes_verifiable_chain(tmp_path: Path) -> None:
    """Verify record_decision updates audit CSV and writes a valid JSONL hash chain."""
    session = load_session(SAMPLE_PATH, model_path=None)
    csv_path = tmp_path / "audit.csv"
    jsonl_path = tmp_path / "audit.jsonl"
    decisions = [("accepted", "First pass"), ("rejected", "Noise"), ("rescan", "Review")]

    for idx, (dec, comment) in enumerate(decisions):
        record_decision(
            session=session,
            item_index=idx,
            decision=dec,
            comment=comment,
            inspector=f"Insp-{idx}",
            log_path=csv_path,
        )

    assert csv_path.is_file()
    assert jsonl_path.is_file()
    check = verify_chain(jsonl_path)
    assert check.ok is True
    assert check.n == len(decisions)
    assert check.first_bad_line is None


def test_record_decision_jsonl_content(tmp_path: Path) -> None:
    """Verify record_decision writes matching fields to JSONL entries."""
    session = load_session(SAMPLE_PATH, model_path=None)
    csv_path = tmp_path / "audit.csv"
    record_decision(session, 0, "accepted", "Approved", "Insp-0", csv_path)
    record_decision(session, 1, "rejected", "Flaw", "Insp-1", csv_path)

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
