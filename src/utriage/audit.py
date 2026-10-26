"""Tamper-evident, append-only audit log using a SHA-256 hash chain."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GENESIS: str = "0" * 64


@dataclass(frozen=True)
class ChainCheck:
    """Verification outcome of an audit hash chain."""

    ok: bool
    n: int
    first_bad_line: int | None
    reason: str


def _compute_hash(entry: dict[str, Any]) -> str:
    """Compute sha256 hex digest of canonical JSON excluding hash key."""
    payload = {k: v for k, v in entry.items() if k != "hash"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_last_entry(log_path: Path) -> dict[str, Any] | None:
    """Return the last parsed JSON line if file exists, else None."""
    if not log_path.is_file() or log_path.stat().st_size == 0:
        return None
    with log_path.open("r", encoding="utf-8") as fh:
        lines = [line.strip() for line in fh if line.strip()]
    if not lines:
        return None
    try:
        data = json.loads(lines[-1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        # File has corrupt or incomplete JSON on last line
        return None


def append_entry(log_path: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Append a tamper-evident audit record to a JSONL hash chain."""
    target = Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    last = _read_last_entry(target)
    prev = last["hash"] if (last and "hash" in last) else GENESIS
    seq = (int(last["seq"]) + 1) if (last and "seq" in last) else 0

    entry = dict(record)
    entry["seq"] = seq
    entry["ts"] = str(record.get("ts", datetime.now(UTC).isoformat()))
    entry["prev"] = prev
    entry["hash"] = _compute_hash(entry)

    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def _verify_entry(
    entry: Any,
    expected_seq: int,
    expected_prev: str,
) -> str | None:
    """Verify entry fields, sequence, prev link, and hash; return reason on error."""
    if not isinstance(entry, dict):
        return "entry is not a JSON object"
    required = ("seq", "ts", "prev", "hash")
    if not all(k in entry for k in required):
        return "missing required hash chain fields"
    if entry.get("seq") != expected_seq:
        return f"seq mismatch: expected {expected_seq}, got {entry.get('seq')}"
    if entry.get("prev") != expected_prev:
        return f"prev mismatch: expected {expected_prev}, got {entry.get('prev')}"
    if entry.get("hash") != _compute_hash(entry):
        return "hash mismatch (tampered content)"
    return None


def verify_chain(log_path: Path) -> ChainCheck:
    """Verify integrity of the append-only JSONL hash chain."""
    target = Path(log_path)
    if not target.is_file() or target.stat().st_size == 0:
        return ChainCheck(ok=True, n=0, first_bad_line=None, reason="empty log")

    with target.open("r", encoding="utf-8") as fh:
        raw_lines = [line.strip() for line in fh]

    expected_prev = GENESIS
    verified_n = 0
    for idx, line in enumerate(raw_lines):
        line_no = idx + 1
        if not line:
            return ChainCheck(
                ok=False, n=verified_n, first_bad_line=line_no, reason="empty line in JSONL"
            )
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as err:
            return ChainCheck(
                ok=False, n=verified_n, first_bad_line=line_no, reason=f"malformed JSON: {err}"
            )

        err_reason = _verify_entry(entry, expected_seq=idx, expected_prev=expected_prev)
        if err_reason:
            return ChainCheck(ok=False, n=verified_n, first_bad_line=line_no, reason=err_reason)

        expected_prev = str(entry["hash"])
        verified_n += 1

    return ChainCheck(ok=True, n=verified_n, first_bad_line=None, reason="")


def export_csv(log_path: Path, csv_path: Path) -> Path:
    """Export human-readable CSV copy of JSONL audit log for QA."""
    src = Path(log_path)
    dst = Path(csv_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.is_file() or src.stat().st_size == 0:
        dst.write_text("", encoding="utf-8")
        return dst

    with src.open("r", encoding="utf-8") as fh:
        entries = [json.loads(line.strip()) for line in fh if line.strip()]
    if not entries:
        dst.write_text("", encoding="utf-8")
        return dst

    fieldnames: list[str] = []
    for entry in entries:
        for key in entry:
            if key not in fieldnames:
                fieldnames.append(key)

    with dst.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entries)
    return dst
