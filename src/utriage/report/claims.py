"""Quantitative claim generation and evidence auditing (project plan Phase 8).

Constructs frozen Claim instances strictly from results files, verifies evidence paths,
and audits text and scopes against forbidden phrases.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FORBIDDEN: tuple[str, ...] = (
    "perfect",
    "100%",
    "for MIDHANI",
    "validated on plant data",
    "qualified",
    "deployed",
    "generalizes to all",
)


def check_forbidden(text: str) -> list[str]:
    """Find any forbidden phrases in text, matching case-insensitively."""
    lower_text = text.lower()
    return [p for p in FORBIDDEN if p.lower() in lower_text]


def _format_rate(rate: float) -> str:
    """Format rate with 3 decimals if below 0.05, else 2 decimals."""
    if rate < 0.05:
        return f"{rate:.3f}"
    return f"{rate:.2f}"


@dataclass(frozen=True)
class Claim:
    """A verified quantitative claim with evidence reference and scope limit."""

    id: str
    text: str
    evidence_file: str
    evidence_path: str
    scope_limit: str
    numbers: dict[str, float]

    def __post_init__(self) -> None:
        """Validate that text and scope_limit do not contain forbidden phrases."""
        bad_text = check_forbidden(self.text)
        if bad_text:
            raise ValueError(f"Forbidden phrase in claim text: {bad_text}")
        bad_scope = check_forbidden(self.scope_limit)
        if bad_scope:
            raise ValueError(f"Forbidden phrase in scope limit: {bad_scope}")


def _matching_keys(mapping: dict[str, Any], path: str) -> list[str]:
    """Return keys matching path or path prefix, ordered by length descending."""
    matches = [str(k) for k in mapping if path == str(k) or path.startswith(f"{k}.")]
    matches.sort(key=len, reverse=True)
    return matches


def _traverse_list(items: list[Any], path: str) -> Any:
    """Traverse into a list segment using an integer index."""
    parts = path.split(".", 1)
    idx_str = parts[0]
    is_valid_int = idx_str.isdigit() or (idx_str.startswith("-") and idx_str[1:].isdigit())
    if not is_valid_int:
        raise IndexError(f"Expected integer list index, got {idx_str} in path {path}")
    idx = int(idx_str)
    if not (-len(items) <= idx < len(items)):
        raise IndexError(f"List index {idx} out of range (length {len(items)})")
    remainder = parts[1] if len(parts) > 1 else ""
    return get_path(items[idx], remainder)


def _traverse_dict(mapping: dict[str, Any], path: str) -> Any:
    """Traverse into dict using matching keys ordered by length descending."""
    candidates = _matching_keys(mapping, path)
    for key in candidates:
        remainder = path[len(key) + 1 :] if len(path) > len(key) else ""
        try:
            return get_path(mapping[key], remainder)
        except KeyError:
            continue
    raise KeyError(f"Key path '{path}' not found in dict")


def get_path(data: Any, dotted: str) -> Any:
    """Retrieve value at dotted path, trying longest matching keys first."""
    if not dotted:
        return data
    if isinstance(data, list):
        return _traverse_list(data, dotted)
    if isinstance(data, dict):
        return _traverse_dict(data, dotted)
    raise KeyError(f"Cannot traverse into leaf type {type(data)} at '{dotted}'")


def render_markdown(claims: list[Claim]) -> str:
    """Render claims as a markdown table with Claim | Evidence | Scope limit."""
    lines = [
        "# Verified Claims and Evidence Limits",
        "",
        "| Claim | Evidence | Scope limit |",
        "|---|---|---|",
    ]
    for c in claims:
        evidence = f"{c.evidence_file} (`{c.evidence_path}`)"
        lines.append(f"| {c.text} | {evidence} | {c.scope_limit} |")
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load JSON from path, returning None on failure."""
    if not path.exists():
        print(f"Skipping claims requiring {path.name}: file not found", file=sys.stderr)
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as err:  # narrow reason: unreadable or corrupt JSON
        print(f"Skipping claims requiring {path.name}: {err}", file=sys.stderr)
        return None


def _parse_flaws_inventory(text: str) -> int:
    """Parse verified total distinct physical flaws from inventory text."""
    match = re.search(r"totaling\s+(\d+)\s+distinct physical flaws", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    total = 0
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 6 and parts[5].isdigit():
            total += int(parts[5])
    return total


def _parse_used_flaws(text: str) -> int | None:
    """Parse distinct physical flaws used in evaluation if present."""
    match = re.search(r"Used in evaluation:\s*(\d+)\s+distinct physical flaws", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _extract_cell_metrics(cell: dict[str, Any]) -> dict[str, float] | None:
    """Extract detection and false call numbers from a metric cell."""
    if cell.get("available") is False or "detection" not in cell or "false_calls" not in cell:
        return None
    det = cell["detection"]
    fc = cell["false_calls"]
    return {
        "k": float(det["k"]),
        "n": float(det["n"]),
        "rate": float(det["rate"]),
        "ci_low": float(det["ci_low"]),
        "ci_high": float(det["ci_high"]),
        "false_call_rate": float(fc["rate"]),
    }


def _build_leakage_claim(results_dir: Path) -> Claim | None:
    """Build claim for dataset leakage audit."""
    path = results_dir / "leakage_audit.json"
    data = _load_json(path)
    if data is None:
        return None
    shared = data.get("shared_cracks_train_test", [])
    n_shared = len(shared)
    text = (
        f"The old random batch split shares all {n_shared} physical cracks "
        "between train and test, so earlier test scores are superseded."
    )
    scope = "one austenitic weld PAUT public dataset (ML-NDT), historical benchmark superseded"
    return Claim(
        id="leakage",
        text=text,
        evidence_file="results/leakage_audit.json",
        evidence_path="shared_cracks_train_test",
        scope_limit=scope,
        numbers={"shared_cracks": float(n_shared)},
    )


def _build_loco_claim(matrix: dict[str, Any], crack: str) -> Claim | None:
    """Build leave-one-crack-out detection claim for a single held-out crack."""
    col_id = f"mlndt_loco_crack_{crack}"
    cell = matrix.get("rows", {}).get("supervised", {}).get(col_id)
    if not isinstance(cell, dict):
        print(f"Skipping loco_crack_{crack}: column {col_id} missing", file=sys.stderr)
        return None
    nums = _extract_cell_metrics(cell)
    if nums is None:
        print(f"Skipping loco_crack_{crack}: cell {col_id} unavailable", file=sys.stderr)
        return None
    k_int, n_int = int(nums["k"]), int(nums["n"])
    rate_s = _format_rate(nums["rate"])
    low_s = _format_rate(nums["ci_low"])
    high_s = _format_rate(nums["ci_high"])
    fc_s = _format_rate(nums["false_call_rate"])
    text = (
        f"Leave-one-crack-out crack_{crack}: detection {k_int}/{n_int} "
        f"(rate {rate_s}, 95% CI [{low_s}, {high_s}]), "
        f"false calls {fc_s} per scan."
    )
    scope = "one austenitic weld PAUT public dataset, 3 physical cracks, not validated on bars"
    return Claim(
        id=f"loco_crack_{crack}",
        text=text,
        evidence_file="results/generalization_matrix.json",
        evidence_path=f"rows.supervised.{col_id}",
        scope_limit=scope,
        numbers=nums,
    )


def _build_anomaly_claim(matrix: dict[str, Any], crack: str) -> Claim | None:
    """Build anomaly detection claim for a single held-out crack."""
    col_id = f"mlndt_loco_crack_{crack}"
    cell = matrix.get("rows", {}).get("anomaly", {}).get(col_id)
    if not isinstance(cell, dict):
        print(f"Skipping anomaly_crack_{crack}: column {col_id} missing", file=sys.stderr)
        return None
    nums = _extract_cell_metrics(cell)
    if nums is None:
        print(f"Skipping anomaly_crack_{crack}: cell {col_id} unavailable", file=sys.stderr)
        return None
    k_int, n_int = int(nums["k"]), int(nums["n"])
    rate_s = _format_rate(nums["rate"])
    low_s = _format_rate(nums["ci_low"])
    high_s = _format_rate(nums["ci_high"])
    fc_s = _format_rate(nums["false_call_rate"])
    text = (
        f"Anomaly model crack_{crack}: detection {k_int}/{n_int} "
        f"(rate {rate_s}, 95% CI [{low_s}, {high_s}]), "
        f"false calls {fc_s} per scan."
    )
    return Claim(
        id=f"anomaly_crack_{crack}",
        text=text,
        evidence_file="results/generalization_matrix.json",
        evidence_path=f"rows.anomaly.{col_id}",
        scope_limit="trained on flaw-free images only",
        numbers=nums,
    )


def _cross_high_fc_claim(nums: dict[str, float], fc: dict[str, Any], col_id: str) -> Claim:
    """Format cross-dataset claim when false-call rate is at or above 0.5."""
    k_int, n_int = int(nums["k"]), int(nums["n"])
    n_false, n_scans = int(fc.get("n_false", 0)), int(fc.get("n_scans", 0))
    text = (
        "Cross-dataset NDT_ML_Flaw: the model flagged (nearly) every image, "
        f"giving detections {k_int}/{n_int} and false calls {n_false}/{n_scans}; "
        "no cross-dataset transfer under the documented domain mapping."
    )
    claim_nums = {
        "k": float(k_int),
        "n": float(n_int),
        "n_false": float(n_false),
        "n_scans": float(n_scans),
    }
    return Claim(
        id="cross_dataset",
        text=text,
        evidence_file="results/generalization_matrix.json",
        evidence_path=f"rows.supervised.{col_id}",
        scope_limit="test only, different weld, domain-mapping assumptions",
        numbers=claim_nums,
    )


def _cross_low_fc_claim(nums: dict[str, float], col_id: str) -> Claim:
    """Format cross-dataset claim when false-call rate is below 0.5."""
    k_int, n_int = int(nums["k"]), int(nums["n"])
    rate_s = _format_rate(nums["rate"])
    low_s, high_s = _format_rate(nums["ci_low"]), _format_rate(nums["ci_high"])
    fc_s = _format_rate(nums["false_call_rate"])
    text = (
        f"Cross-dataset NDT_ML_Flaw: detection {k_int}/{n_int} "
        f"(rate {rate_s}, 95% CI [{low_s}, {high_s}]), "
        f"false calls {fc_s} per scan."
    )
    return Claim(
        id="cross_dataset",
        text=text,
        evidence_file="results/generalization_matrix.json",
        evidence_path=f"rows.supervised.{col_id}",
        scope_limit="test only, different weld, domain-mapping assumptions",
        numbers=nums,
    )


def _build_cross_dataset_claim(matrix: dict[str, Any]) -> Claim | None:
    """Build cross-dataset evaluation claim if available."""
    col_id = "cross_ndt_ml_flaw"
    cell = matrix.get("rows", {}).get("supervised", {}).get(col_id)
    if not isinstance(cell, dict):
        print("Skipping cross_dataset: cross_ndt_ml_flaw column missing", file=sys.stderr)
        return None
    nums = _extract_cell_metrics(cell)
    if nums is None:
        reason = cell.get("reason", "unavailable")
        print(f"Skipping cross_dataset: {reason}", file=sys.stderr)
        return None
    if nums["false_call_rate"] >= 0.5:
        return _cross_high_fc_claim(nums, cell.get("false_calls", {}), col_id)
    return _cross_low_fc_claim(nums, col_id)


def _build_sim_in_distribution_claim(matrix: dict[str, Any]) -> Claim | None:
    """Build in-distribution simulator baseline claim comparing ML and rules."""
    col_id = "sim_in_distribution"
    r_cell = matrix.get("rows", {}).get("rules", {}).get(col_id)
    m_cell = matrix.get("rows", {}).get("supervised", {}).get(col_id)
    if not (isinstance(r_cell, dict) and isinstance(m_cell, dict)):
        print(f"Skipping {col_id}: rules or supervised cell missing", file=sys.stderr)
        return None
    r_nums, m_nums = _extract_cell_metrics(r_cell), _extract_cell_metrics(m_cell)
    if r_nums is None or m_nums is None:
        print(f"Skipping {col_id}: cell metrics unavailable", file=sys.stderr)
        return None
    r_det, r_fc = r_nums["rate"], r_nums["false_call_rate"]
    m_det, m_fc = m_nums["rate"], m_nums["false_call_rate"]
    text = (
        f"Simulator in-distribution: ML det {_format_rate(m_det)} vs rules {_format_rate(r_det)}, "
        f"false calls {_format_rate(m_fc)} vs {_format_rate(r_fc)}."
    )
    numbers = {"ml_det": m_det, "rules_det": r_det, "ml_fc": m_fc, "rules_fc": r_fc}
    return Claim(
        id=col_id,
        text=text,
        evidence_file="results/generalization_matrix.json",
        evidence_path="rows",
        scope_limit="Synthetic data, UNVERIFIED constants",
        numbers=numbers,
    )


def _find_sim_columns(matrix: dict[str, Any]) -> list[str]:
    """Find held-out simulator parameter region column IDs."""
    cols = matrix.get("columns", [])
    if isinstance(cols, list) and cols:
        return [
            str(c.get("id"))
            for c in cols
            if str(c.get("id")).startswith("sim_") and str(c.get("id")) != "sim_in_distribution"
        ]
    sup = matrix.get("rows", {}).get("supervised", {})
    return [k for k in sup if k.startswith("sim_") and k != "sim_in_distribution"]


def _format_sim_region(
    name: str, r_cell: dict[str, Any], m_cell: dict[str, Any]
) -> tuple[str, bool, dict[str, float]]:
    """Format single simulator held-out region metrics."""
    r_det = float(r_cell["detection"]["rate"])
    r_fc = float(r_cell["false_calls"]["rate"])
    m_det = float(m_cell["detection"]["rate"])
    m_fc = float(m_cell["false_calls"]["rate"])
    part = (
        f"{name}: ML det {_format_rate(m_det)} vs rules {_format_rate(r_det)}, "
        f"false calls {_format_rate(m_fc)} vs {_format_rate(r_fc)}"
    )
    is_high = any([m_fc > 0.3, r_fc > 0.3])
    nums = {
        f"ml_det_{name}": m_det,
        f"rules_det_{name}": r_det,
        f"ml_fc_{name}": m_fc,
        f"rules_fc_{name}": r_fc,
    }
    return part, is_high, nums


def _collect_sim_regions(
    cols: list[str], matrix: dict[str, Any]
) -> tuple[list[str], list[str], dict[str, float]]:
    """Collect formatted region lines, high false-call flags, and numeric rates."""
    parts: list[str] = []
    high_fc: list[str] = []
    numbers: dict[str, float] = {}
    for col in cols:
        r_cell = matrix.get("rows", {}).get("rules", {}).get(col)
        m_cell = matrix.get("rows", {}).get("supervised", {}).get(col)
        if not (isinstance(r_cell, dict) and isinstance(m_cell, dict)):
            continue
        part, is_high, nums = _format_sim_region(col, r_cell, m_cell)
        parts.append(part)
        numbers.update(nums)
        if is_high:
            high_fc.append(col)
    return parts, high_fc, numbers


def _build_sim_regions_claim(matrix: dict[str, Any]) -> Claim | None:
    """Build claim comparing ML and rules across held-out simulator regions."""
    cols = _find_sim_columns(matrix)
    if not cols:
        print("Skipping sim_regions: no simulator held-out columns found", file=sys.stderr)
        return None
    parts, high_fc, numbers = _collect_sim_regions(cols, matrix)
    if not parts:
        return None
    high_str = f"exceeds 0.3 in: {', '.join(high_fc)}" if high_fc else "none exceed 0.3"
    text = f"Simulator held-out regions (ML vs rules): {'; '.join(parts)}; {high_str}."
    return Claim(
        id="sim_regions",
        text=text,
        evidence_file="results/generalization_matrix.json",
        evidence_path="rows",
        scope_limit="Synthetic data, UNVERIFIED constants",
        numbers=numbers,
    )


def _build_union_fallback(results_dir: Path) -> Claim | None:
    """Fallback union claim using triage_metrics.json."""
    triage_data = _load_json(results_dir / "triage_metrics.json")
    if triage_data is None or "union_violations" not in triage_data:
        print("Skipping union_property: no union evaluation data available", file=sys.stderr)
        return None
    viol = int(triage_data["union_violations"])
    text = (
        f"The union property holds with {viol} violations "
        "(ML adds flags, never removes a rules flag)."
    )
    scope = "synthetic simulator columns only (the real-image columns have no rules baseline)"
    return Claim(
        id="union_property",
        text=text,
        evidence_file="results/triage_metrics.json",
        evidence_path="union_violations",
        scope_limit=scope,
        numbers={"n_violations": float(viol)},
    )


def _filter_sim_union_cols(up_data: dict[str, Any]) -> dict[str, Any]:
    """Extract valid simulator columns from union property data."""
    valid: dict[str, Any] = {}
    for k, v in up_data.items():
        if not k.startswith("sim_") or not isinstance(v, dict):
            continue
        if v.get("available") is not False and "holds" in v:
            valid[k] = v
    return valid


def _build_union_claim(matrix: dict[str, Any], results_dir: Path) -> Claim | None:
    """Build safety claim verifying union property holds across computable columns."""
    up_data = matrix.get("union_property")
    if isinstance(up_data, dict) and up_data:
        computable = _filter_sim_union_cols(up_data)
        if computable:
            viol = sum(int(v.get("n_violations", 0)) for v in computable.values())
            n_cols = len(computable)
            text = (
                f"The union property holds in all {n_cols} synthetic simulator columns "
                f"with {viol} violations (ML adds flags, never removes a rules flag)."
            )
            scope = (
                "synthetic simulator columns only (the real-image columns have no rules baseline)"
            )
            return Claim(
                id="union_property",
                text=text,
                evidence_file="results/generalization_matrix.json",
                evidence_path="union_property",
                scope_limit=scope,
                numbers={"n_violations": float(viol), "n_columns": float(n_cols)},
            )
    return _build_union_fallback(results_dir)


def _build_latency_claim(results_dir: Path) -> Claim | None:
    """Build CPU inference latency claim."""
    data = _load_json(results_dir / "hardware.json")
    if data is None:
        return None
    mach = data.get("inference_machine", {})
    cpu_name = mach.get("cpu_name")
    proc = cpu_name if cpu_name else mach.get("processor", "development CPU")
    lat = data.get("cpu_latency_ms", {}).get("mlndt")
    if not isinstance(lat, dict):
        print("Skipping latency: mlndt cpu_latency_ms missing", file=sys.stderr)
        return None
    med = float(lat.get("median_ms", lat.get("median", 0.0)))
    p95 = float(lat.get("p95_ms", lat.get("p95", 0.0)))
    text = f"CPU latency on {proc}: median {med:.1f} ms, p95 {p95:.1f} ms."
    return Claim(
        id="latency",
        text=text,
        evidence_file="results/hardware.json",
        evidence_path="cpu_latency_ms.mlndt",
        scope_limit="this development machine, not the target i5 laptop",
        numbers={"median_ms": med, "p95_ms": p95},
    )


def _format_evidence_limit_text(listed: int, used: int | None) -> tuple[str, dict[str, float]]:
    """Format claim text and numbers for physical flaw inventory."""
    if used is not None:
        text = (
            f"Total distinct real physical flaws across accepted public datasets: "
            f"{listed} listed vs {used} used in evaluation."
        )
        return text, {
            "distinct_physical_flaws": float(listed),
            "distinct_physical_flaws_used": float(used),
        }
    text = f"Total distinct real physical flaws across accepted public datasets: {listed} listed."
    return text, {"distinct_physical_flaws": float(listed)}


def _build_evidence_limit_claim(results_dir: Path) -> Claim | None:
    """Build claim on total distinct real physical flaws from inventory."""
    path = results_dir / "data_inventory.md"
    if not path.exists():
        print(f"Skipping evidence_limit: {path.name} not found", file=sys.stderr)
        return None
    content = path.read_text(encoding="utf-8")
    listed = _parse_flaws_inventory(content)
    used = _parse_used_flaws(content)
    text, nums = _format_evidence_limit_text(listed, used)
    return Claim(
        id="evidence_limit",
        text=text,
        evidence_file="results/data_inventory.md",
        evidence_path="distinct_physical_flaws",
        scope_limit="evidence limit",
        numbers=nums,
    )


def _collect_crack_claims(matrix: dict[str, Any]) -> list[Claim]:
    """Collect leave-one-crack-out and anomaly crack claims."""
    claims: list[Claim] = []
    for crack in ("1.6", "4.0", "8.6"):
        c_loco = _build_loco_claim(matrix, crack)
        if c_loco is not None:
            claims.append(c_loco)
    for crack in ("1.6", "4.0", "8.6"):
        c_anom = _build_anomaly_claim(matrix, crack)
        if c_anom is not None:
            claims.append(c_anom)
    return claims


def _collect_matrix_claims(matrix: dict[str, Any]) -> list[Claim]:
    """Collect claims derived directly from generalization matrix."""
    claims = _collect_crack_claims(matrix)
    c_cross = _build_cross_dataset_claim(matrix)
    if c_cross is not None:
        claims.append(c_cross)
    c_sim_in_dist = _build_sim_in_distribution_claim(matrix)
    if c_sim_in_dist is not None:
        claims.append(c_sim_in_dist)
    c_sim = _build_sim_regions_claim(matrix)
    if c_sim is not None:
        claims.append(c_sim)
    return claims


def _collect_final_claims(r_dir: Path, matrix: dict[str, Any]) -> list[Claim]:
    """Collect trailing safety, hardware latency, and evidence limit claims."""
    trailing = [
        _build_union_claim(matrix, r_dir),
        _build_latency_claim(r_dir),
        _build_evidence_limit_claim(r_dir),
    ]
    return [c for c in trailing if c is not None]


def build_claims(results_dir: Path | str) -> list[Claim]:
    """Build all verified claims composed only from allowed results files."""
    r_dir = Path(results_dir)
    claims: list[Claim] = []

    c_leak = _build_leakage_claim(r_dir)
    if c_leak is not None:
        claims.append(c_leak)

    matrix = _load_json(r_dir / "generalization_matrix.json")
    if matrix is not None:
        claims.extend(_collect_matrix_claims(matrix))

    mat_dict = matrix if matrix is not None else {}
    claims.extend(_collect_final_claims(r_dir, mat_dict))
    return claims
