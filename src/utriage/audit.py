"""Tamper-evident, append-only audit log using a SHA-256 hash chain."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GENESIS: str = "0" * 64


class AuditIntegrityError(Exception):
    """Raised when audit log integrity is compromised."""


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


def _anchor_path(log_path: Path) -> Path:
    """Return path to anchor sidecar file (<log>.head.json)."""
    p = Path(log_path)
    return p.with_name(f"{p.name}.head.json")


def _write_anchor(anchor_path: Path, count: int, last_hash: str) -> None:
    """Atomically write anchor sidecar with count and last hash."""
    payload = {"count": count, "last_hash": last_hash}
    tmp_path = anchor_path.with_name(f"{anchor_path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, anchor_path)


def _read_anchor(anchor_path: Path) -> tuple[int, str] | str:
    """Read anchor sidecar; returns (count, last_hash) or error reason string."""
    if not anchor_path.is_file():
        return "missing anchor file for non-empty log"
    try:
        data = json.loads(anchor_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        return f"malformed anchor JSON: {err}"
    if not isinstance(data, dict):
        return "anchor content is not a JSON object"
    count = data.get("count")
    last_hash = data.get("last_hash")
    if count is None or last_hash is None:
        return "anchor missing required count or last_hash"
    return (int(count), str(last_hash))


def _validate_tail_entry(data: Any) -> dict[str, Any]:
    """Validate tail entry structure and hash; raise AuditIntegrityError on corruption."""
    if not isinstance(data, dict):
        raise AuditIntegrityError("trailing entry is not a JSON object")
    required = ("seq", "ts", "prev", "hash")
    if not all(k in data for k in required):
        raise AuditIntegrityError("trailing entry missing required hash chain fields")
    if data["hash"] != _compute_hash(data):
        raise AuditIntegrityError("trailing entry hash mismatch")
    return data


def _read_last_entry(log_path: Path) -> dict[str, Any] | None:
    """Return last verified entry if log exists, None if empty; raise on corruption."""
    if not log_path.is_file() or log_path.stat().st_size == 0:
        return None
    with log_path.open("r", encoding="utf-8") as fh:
        lines = [line.strip() for line in fh if line.strip()]
    if not lines:
        raise AuditIntegrityError("log file is non-empty but contains no valid records")
    try:
        data = json.loads(lines[-1])
    except json.JSONDecodeError as err:
        raise AuditIntegrityError(f"malformed trailing JSONL entry: {err}") from err
    return _validate_tail_entry(data)


def append_entry(log_path: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Append a tamper-evident audit record to a JSONL hash chain."""
    target = Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    last = _read_last_entry(target)
    prev = last["hash"] if last else GENESIS
    seq = (int(last["seq"]) + 1) if last else 0

    entry = dict(record)
    entry["seq"] = seq
    entry["ts"] = str(record.get("ts", datetime.now(UTC).isoformat()))
    entry["prev"] = prev
    entry["hash"] = _compute_hash(entry)

    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    _write_anchor(_anchor_path(target), count=seq + 1, last_hash=entry["hash"])
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


def _scan_chain(target: Path) -> ChainCheck:
    """Scan and verify all JSONL records in file; reason holds last hash on success."""
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
        err = _verify_entry(entry, expected_seq=idx, expected_prev=expected_prev)
        if err:
            return ChainCheck(ok=False, n=verified_n, first_bad_line=line_no, reason=err)
        expected_prev = str(entry["hash"])
        verified_n += 1

    return ChainCheck(ok=True, n=verified_n, first_bad_line=None, reason=expected_prev)


def _verify_against_anchor(
    verified_n: int,
    last_hash: str,
    anchor_count: int,
    anchor_last_hash: str,
) -> ChainCheck:
    """Compare verified chain length and tail hash against anchor sidecar."""
    if verified_n != anchor_count:
        return ChainCheck(
            ok=False,
            n=verified_n,
            first_bad_line=None,
            reason=(
                f"truncation detected: log has {verified_n} entries, "
                f"anchor expects {anchor_count} (count mismatch)"
            ),
        )
    if last_hash != anchor_last_hash:
        return ChainCheck(
            ok=False,
            n=verified_n,
            first_bad_line=None,
            reason=(
                f"truncation detected: tail hash {last_hash} != "
                f"anchor hash {anchor_last_hash} (last hash mismatch)"
            ),
        )
    return ChainCheck(ok=True, n=verified_n, first_bad_line=None, reason="")


def verify_chain(log_path: Path) -> ChainCheck:
    """Verify integrity of the append-only JSONL hash chain and sidecar anchor.

    Note: The sidecar anchor (<log>.head.json) detects log truncation only
    if the anchor itself is not altered. A full defense against truncation
    requires an external, append-only, or cryptographically signed anchor.
    """
    target = Path(log_path)
    if not target.is_file() or target.stat().st_size == 0:
        return ChainCheck(ok=True, n=0, first_bad_line=None, reason="empty log")

    anchor_res = _read_anchor(_anchor_path(target))
    if isinstance(anchor_res, str):
        return ChainCheck(ok=False, n=0, first_bad_line=None, reason=anchor_res)

    anchor_count, anchor_last_hash = anchor_res
    check = _scan_chain(target)
    if not check.ok:
        return check

    return _verify_against_anchor(check.n, check.reason, anchor_count, anchor_last_hash)


verify = verify_chain


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
