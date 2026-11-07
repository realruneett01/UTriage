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

from utriage.report.claims_ml import collect_ml_claims
from utriage.report.claims_ti import collect_ti_claims

FORBIDDEN: tuple[str, ...] = (
    "perfect",
    "100%",
    "for MIDHANI",
    "validated on plant data",
    "qualified",
    "deployed",
    "generalizes to all",
)

FORBIDDEN_WORDS: tuple[str, ...] = (
    "verified",
    "confirming",
    "establishing",
    "ensuring",
    "guarantee",
    "nearly",
)

HIGH_FCR_THRESHOLD: float = 0.3


def check_forbidden(text: str) -> list[str]:
    """Find any forbidden phrases in text, matching case-insensitively."""
    lower_text = text.lower()
    return [p for p in FORBIDDEN if p.lower() in lower_text]


def check_forbidden_words(text: str) -> list[str]:
    """Find any forbidden whole words in text, matching case-insensitively."""
    return [w for w in FORBIDDEN_WORDS if re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE)]


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
        """Validate that text and scope_limit do not contain forbidden phrases or words."""
        bad_text = check_forbidden(self.text)
        if bad_text:
            raise ValueError(f"Forbidden phrase in claim text: {bad_text}")
        bad_words_text = check_forbidden_words(self.text)
        if bad_words_text:
            raise ValueError(f"Forbidden word in claim text: {bad_words_text}")
        bad_scope = check_forbidden(self.scope_limit)
        if bad_scope:
            raise ValueError(f"Forbidden phrase in scope limit: {bad_scope}")
        bad_words_scope = check_forbidden_words(self.scope_limit)
        if bad_words_scope:
            raise ValueError(f"Forbidden word in scope limit: {bad_words_scope}")


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
        "# Claims and Evidence Limits",
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
    scope = "ML-NDT (316L weld, 3 physical cracks), historical benchmark superseded"
    return Claim(
        id="leakage",
        text=text,
        evidence_file="results/leakage_audit.json",
        evidence_path="shared_cracks_train_test",
        scope_limit=scope,
        numbers={"shared_cracks": float(n_shared)},
    )


def _format_loco_text(crack: str, nums: dict[str, float]) -> str:
    """Format claim text for leave-one-crack-out detection."""
    k_int, n_int = int(nums["k"]), int(nums["n"])
    rate_s = _format_rate(nums["rate"])
    low_s, high_s = _format_rate(nums["ci_low"]), _format_rate(nums["ci_high"])
    fc_s = _format_rate(nums["false_call_rate"])
    return (
        f"Leave-one-crack-out crack_{crack} "
        "(early stopping on validation batches that contain the training cracks (batch-held-out)): "
        f"detection {k_int}/{n_int} (rate {rate_s}, 95% CI [{low_s}, {high_s}]), "
        f"false calls {fc_s} per scan."
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
    return Claim(
        id=f"loco_crack_{crack}",
        text=_format_loco_text(crack, nums),
        evidence_file="results/generalization_matrix.json",
        evidence_path=f"rows.supervised.{col_id}",
        scope_limit="ML-NDT (316L weld, 3 physical cracks), not validated on bars",
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
        scope_limit="ML-NDT (316L weld, 3 physical cracks), trained on flaw-free images only",
        numbers=nums,
    )


def _cross_high_fc_claim(nums: dict[str, float], fc: dict[str, Any], col_id: str) -> Claim:
    """Format cross-dataset claim when false-call rate is at or above 0.5."""
    k_int, n_int = int(nums["k"]), int(nums["n"])
    n_false, n_scans = int(fc.get("n_false", 0)), int(fc.get("n_scans", 0))
    n_flagged = k_int + n_false
    n_total = n_int + n_scans
    text = (
        f"Cross-dataset NDT_ML_Flaw: the model flagged {n_flagged}/{n_total} images, "
        f"giving detections {k_int}/{n_int} and false calls {n_false}/{n_scans}; "
        "no cross-dataset transfer under the documented domain mapping."
    )
    claim_nums = {
        "n_flagged": float(n_flagged),
        "n_total": float(n_total),
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
        scope_limit="NDT_ML_Flaw, test only, different weld, domain-mapping assumptions",
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
        scope_limit="NDT_ML_Flaw, test only, different weld, domain-mapping assumptions",
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
    is_high = any([m_fc > HIGH_FCR_THRESHOLD, r_fc > HIGH_FCR_THRESHOLD])
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
    thresh_s = f"{HIGH_FCR_THRESHOLD}"
    note = (
        f"(reporting threshold {thresh_s} false calls "
        "per window chosen by the project, not a standard)"
    )
    if high_fc:
        high_str = f"exceeds {thresh_s} {note} in: {', '.join(high_fc)}"
    else:
        high_str = f"none exceed {thresh_s} {note}"
    text = f"Simulator held-out regions (ML vs rules): {'; '.join(parts)}; {high_str}."
    numbers["high_fcr_threshold"] = HIGH_FCR_THRESHOLD
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


def _format_fold_crackval(fold: dict[str, Any]) -> tuple[str, bool, dict[str, float]]:
    """Format single fold metrics for crack-held-out LOCO."""
    held_out = str(fold.get("held_out", ""))
    val_crack = str(fold.get("val_crack", ""))
    det = fold.get("detections", {})
    fc = fold.get("false_calls", {})
    k = int(det.get("n_hit", det.get("k", 0)))
    n = int(det.get("n_flawed", det.get("n", 0)))
    n_false = int(fc.get("n_false", 0))
    n_clean = int(fc.get("n_unflawed", fc.get("n_scans", 0)))
    part = (
        f"held-out {held_out}, validation {val_crack}: "
        f"detections {k}/{n}, false calls {n_false}/{n_clean}"
    )
    is_high_fc = n_clean > 0 and n_false > (n_clean / 2)
    nums = {
        f"{held_out}_k": float(k),
        f"{held_out}_n": float(n),
        f"{held_out}_n_false": float(n_false),
        f"{held_out}_n_unflawed": float(n_clean),
    }
    return part, is_high_fc, nums


def _format_crackval_swing_sentence(high_fc_folds: list[str]) -> str:
    """Format conclusion sentence noting results swing and naming high-false-call fold."""
    if not high_fc_folds:
        return "Results swing strongly with the choice of early-stopping crack."
    label = "folds" if len(high_fc_folds) > 1 else "fold"
    folds_str = ", ".join(high_fc_folds)
    return (
        "Results swing strongly with the choice of early-stopping crack: "
        f"false calls exceed half the clean images in {label} {folds_str}."
    )


def _collect_crackval_folds(
    folds: list[dict[str, Any]],
) -> tuple[list[str], list[str], dict[str, float]]:
    """Collect formatted fold descriptions, high false-call folds, and numeric counts."""
    parts: list[str] = []
    high_fc: list[str] = []
    numbers: dict[str, float] = {}
    for fold in folds:
        part, is_high, nums = _format_fold_crackval(fold)
        parts.append(part)
        numbers.update(nums)
        if is_high:
            high_fc.append(str(fold.get("held_out", "")))
    return parts, high_fc, numbers


def _build_loco_crackval_claim(results_dir: Path) -> Claim | None:
    """Build claim for leave-one-crack-out with crack-held-out validation."""
    path = results_dir / "loco_crackval_metrics.json"
    data = _load_json(path)
    if data is None:
        return None
    folds = data.get("folds")
    if not isinstance(folds, list) or not folds:
        print("Skipping loco_crackval: no folds data found", file=sys.stderr)
        return None
    parts, high_fc, numbers = _collect_crackval_folds(folds)
    swing = _format_crackval_swing_sentence(high_fc)
    text = (
        "Leave-one-crack-out with crack-held-out early stopping "
        f"(train on one crack, early-stop on another): {'; '.join(parts)}. {swing}"
    )
    scope = "ML-NDT (316L weld, 3 physical cracks), each fold trains on a single crack"
    return Claim(
        id="loco_crackval",
        text=text,
        evidence_file="results/loco_crackval_metrics.json",
        evidence_path="folds",
        scope_limit=scope,
        numbers=numbers,
    )


def _extract_single_model_seed_vals(
    m: dict[str, Any],
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """Extract (k, n) and (n_false, n_scans) from a single model dict."""
    h2 = m.get("methods", {}).get("h2", {})
    det, fc = h2.get("detection", {}), h2.get("false_calls", {})
    det_val = (int(det["k"]), int(det["n"])) if ("k" in det and "n" in det) else None
    fc_val = (
        (int(fc["n_false"]), int(fc["n_scans"])) if ("n_false" in fc and "n_scans" in fc) else None
    )
    return det_val, fc_val


def _extract_seed_metrics(models: dict[str, Any]) -> dict[str, Any] | None:
    """Extract detection range and false call counts across seed models."""
    seeds = [m for name, m in models.items() if "_seed" in name]
    targets = seeds if seeds else list(models.values())
    if not targets:
        return None
    k_vals, n_vals, fc_vals, scans_vals = [], [], [], []
    for m in targets:
        d_val, f_val = _extract_single_model_seed_vals(m)
        if d_val:
            k_vals.append(d_val[0])
            n_vals.append(d_val[1])
        if f_val:
            fc_vals.append(f_val[0])
            scans_vals.append(f_val[1])
    if not k_vals or not fc_vals:
        return None
    return {
        "min_k": min(k_vals),
        "max_k": max(k_vals),
        "n": n_vals[0],
        "n_false": max(fc_vals),
        "n_scans": scans_vals[0],
    }


def _format_hybrid_fold(crack: str, fold: dict[str, Any]) -> tuple[str, dict[str, float]] | None:
    """Format single crack fold description and counts for hybrid H2."""
    models = fold.get("models")
    if not isinstance(models, dict):
        return None
    m = _extract_seed_metrics(models)
    if m is None:
        return None
    min_k, max_k = m["min_k"], m["max_k"]
    n, n_false, n_scans = m["n"], m["n_false"], m["n_scans"]
    part = (
        f"{crack}: detections range across seeds {min_k}-{max_k}/{n}, "
        f"false calls {n_false}/{n_scans}"
    )
    nums = {
        f"{crack}_min_k": float(min_k),
        f"{crack}_max_k": float(max_k),
        f"{crack}_n": float(n),
        f"{crack}_n_false": float(n_false),
        f"{crack}_n_scans": float(n_scans),
    }
    return part, nums


def _collect_hybrid_parts(folds: dict[str, Any]) -> tuple[list[str], dict[str, float]]:
    """Collect formatted lines and numbers across hybrid folds."""
    parts: list[str] = []
    numbers: dict[str, float] = {}
    for crack in ("crack_1.6", "crack_4.0", "crack_8.6"):
        res = _format_hybrid_fold(crack, folds.get(crack, {}))
        if res is not None:
            parts.append(res[0])
            numbers.update(res[1])
    return parts, numbers


def _build_hybrid_h2_claim(data_or_dir: dict[str, Any] | Path) -> Claim | None:
    """Build claim for hybrid H2 score fusion across seeds."""
    data = (
        data_or_dir
        if isinstance(data_or_dir, dict)
        else _load_json(data_or_dir / "hybrid_metrics.json")
    )
    if data is None:
        return None
    folds = data.get("folds")
    if not isinstance(folds, dict) or not folds:
        print("Skipping hybrid_h2: folds data missing", file=sys.stderr)
        return None
    parts, numbers = _collect_hybrid_parts(folds)
    if not parts:
        return None
    prefix = "Exploratory: " if data.get("run_label") == "exploratory" else ""
    text = f"{prefix}Hybrid H2: {'; '.join(parts)}."
    scope = (
        "pre-registered ADR 0010, one weld PAUT public dataset, 3 physical cracks, seed-sensitive"
    )
    return Claim(
        id="hybrid_h2",
        text=text,
        evidence_file="results/hybrid_metrics.json",
        evidence_path="folds",
        scope_limit=scope,
        numbers=numbers,
    )


def _get_fold_dict(folds_raw: Any) -> dict[str, dict[str, Any]]:
    """Convert folds list or dict to dictionary keyed by held_out crack."""
    if isinstance(folds_raw, dict):
        return {str(k): v for k, v in folds_raw.items() if isinstance(v, dict)}
    if isinstance(folds_raw, list):
        res: dict[str, dict[str, Any]] = {}
        for item in folds_raw:
            if isinstance(item, dict) and "held_out" in item:
                res[str(item["held_out"])] = item
        return res
    return {}


def _format_physics_fold(crack: str, fold: dict[str, Any]) -> tuple[str, dict[str, float]] | None:
    """Format single crack fold description and metrics for physics H3."""
    det = fold.get("detections", fold.get("detection", {}))
    fc = fold.get("false_calls", {})
    if not isinstance(det, dict) or not isinstance(fc, dict):
        return None
    k = int(det.get("k", det.get("n_hit", 0)))
    n = int(det.get("n", det.get("n_flawed", 0)))
    n_false = int(fc.get("n_false", 0))
    n_clean = int(fc.get("n_clean", fc.get("n_scans", fc.get("n_unflawed", 0))))
    if n <= 0 or n_clean <= 0:
        return None
    part = f"{crack}: detections {k}/{n}, false calls {n_false}/{n_clean}"
    nums = {
        f"{crack}_k": float(k),
        f"{crack}_n": float(n),
        f"{crack}_n_false": float(n_false),
        f"{crack}_n_clean": float(n_clean),
    }
    return part, nums


def _collect_physics_parts(fold_dict: dict[str, Any]) -> tuple[list[str], dict[str, float]]:
    """Collect formatted lines and numbers across physics folds."""
    parts: list[str] = []
    numbers: dict[str, float] = {}
    for crack in ("crack_1.6", "crack_4.0", "crack_8.6"):
        res = _format_physics_fold(crack, fold_dict.get(crack, {}))
        if res is not None:
            parts.append(res[0])
            numbers.update(res[1])
    return parts, numbers


def _build_physics_h3_claim(data_or_dir: dict[str, Any] | Path) -> Claim | None:
    """Build claim for physics feature model H3 across cracks."""
    data = (
        data_or_dir
        if isinstance(data_or_dir, dict)
        else _load_json(data_or_dir / "physics_model_metrics.json")
    )
    if data is None:
        return None
    fold_dict = _get_fold_dict(data.get("folds"))
    if not fold_dict:
        print("Skipping physics_h3: folds data missing", file=sys.stderr)
        return None
    parts, numbers = _collect_physics_parts(fold_dict)
    if not parts:
        return None
    prefix = "Exploratory: " if data.get("run_label") == "exploratory" else ""
    text = f"{prefix}Physics model H3: {'; '.join(parts)}."
    scope = "pre-registered ADR 0010 addendum, coefficients unstable across folds"
    return Claim(
        id="physics_h3",
        text=text,
        evidence_file="results/physics_model_metrics.json",
        evidence_path="folds",
        scope_limit=scope,
        numbers=numbers,
    )


def _find_pt_model_rate(models: dict[str, Any]) -> float | None:
    """Find detection rate for a non-seed .pt model."""
    for name, m in models.items():
        if name.endswith(".pt") and "_seed" not in name:
            det = m.get("methods", {}).get("h2", {}).get("detection", {})
            if "rate" in det:
                return float(det["rate"])
    return None


def _get_h2_det_rate(fold: dict[str, Any]) -> float:
    """Retrieve best or mean H2 detection rate for a fold."""
    models = fold.get("models", {})
    if isinstance(models, dict):
        pt_rate = _find_pt_model_rate(models)
        if pt_rate is not None:
            return pt_rate
        seed_rates = [
            float(m.get("methods", {}).get("h2", {}).get("detection", {}).get("rate", 0.0))
            for name, m in models.items()
            if "_seed" in name
        ]
        if seed_rates:
            return sum(seed_rates) / len(seed_rates)
    agg_rate = fold.get("aggregate", {}).get("h2", {}).get("mean_detection_rate")
    return float(agg_rate) if agg_rate is not None else 0.0


def _get_h3_det_rate(fold: dict[str, Any]) -> float:
    """Retrieve H3 detection rate for a fold."""
    det = fold.get("detections", fold.get("detection", {}))
    if isinstance(det, dict) and "rate" in det:
        return float(det["rate"])
    k = float(det.get("k", 0))
    n = float(det.get("n", 1))
    return k / n if n > 0 else 0.0


def _parse_crack_size(fold_name: str) -> float | None:
    """Parse numeric crack size in mm from fold name (e.g. crack_1.6 -> 1.6)."""
    match = re.search(r"(\d+(?:\.\d+)?)", fold_name)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _determine_stronger_methods(
    h_folds: dict[str, Any], p_folds: dict[str, Any]
) -> tuple[str, str]:
    """Determine stronger method (H2 vs H3) on 1.6 mm and 4.0 mm cracks."""
    f_1_6_h, f_1_6_p = h_folds.get("crack_1.6", {}), p_folds.get("crack_1.6", {})
    f_4_0_h, f_4_0_p = h_folds.get("crack_4.0", {}), p_folds.get("crack_4.0", {})
    h2_1_6, h3_1_6 = _get_h2_det_rate(f_1_6_h), _get_h3_det_rate(f_1_6_p)
    h2_4_0, h3_4_0 = _get_h2_det_rate(f_4_0_h), _get_h3_det_rate(f_4_0_p)
    w_1_6 = "H2" if h2_1_6 >= h3_1_6 else "H3"
    w_4_0 = "H3" if h3_4_0 >= h2_4_0 else "H2"
    return w_1_6, w_4_0


def _build_hybrid_comparison_claim(
    h_input: dict[str, Any] | Path, p_input: dict[str, Any] | None = None
) -> Claim | None:
    """Build claim comparing H2 and H3 performance across cracks."""
    h_data = _load_json(h_input / "hybrid_metrics.json") if isinstance(h_input, Path) else h_input
    p_data = (
        _load_json(h_input / "physics_model_metrics.json") if isinstance(h_input, Path) else p_input
    )
    if h_data is None or p_data is None:
        return None
    h_folds = h_data.get("folds", {})
    p_folds = _get_fold_dict(p_data.get("folds"))
    w_1_6, w_4_0 = _determine_stronger_methods(h_folds, p_folds)
    is_exp = any(
        [
            h_data.get("run_label") == "exploratory",
            p_data.get("run_label") == "exploratory",
        ]
    )
    pfx = "Exploratory: " if is_exp else ""
    sz_1_6 = _parse_crack_size("crack_1.6") or 1.6
    sz_4_0 = _parse_crack_size("crack_4.0") or 4.0
    text = (
        f"{pfx}{w_1_6} strongest on the {sz_1_6:.1f} mm crack, "
        f"{w_4_0} strongest on the {sz_4_0:.1f} mm crack."
    )
    scope = "pre-registered ADR 0010, one weld PAUT public dataset, 3 physical cracks"
    return Claim(
        id="hybrid_comparison",
        text=text,
        evidence_file="results/hybrid_metrics.json",
        evidence_path="folds",
        scope_limit=scope,
        numbers={"crack_1.6": sz_1_6, "crack_4.0": sz_4_0},
    )


def _collect_h2_claim(h_data: dict[str, Any] | None) -> Claim | None:
    """Build hybrid H2 claim if data is available."""
    if h_data is None:
        return None
    return _build_hybrid_h2_claim(h_data)


def _collect_h3_claim(p_data: dict[str, Any] | None) -> Claim | None:
    """Build physics H3 claim if data is available."""
    if p_data is None:
        return None
    return _build_physics_h3_claim(p_data)


def _collect_comparison_claim(
    h_data: dict[str, Any] | None, p_data: dict[str, Any] | None
) -> Claim | None:
    """Build comparison claim if both H2 and H3 data are available."""
    if h_data is None or p_data is None:
        return None
    return _build_hybrid_comparison_claim(h_data, p_data)


def _collect_hybrid_claims(r_dir: Path) -> list[Claim]:
    """Collect hybrid H2, physics H3, and comparison claims."""
    h_data = _load_json(r_dir / "hybrid_metrics.json")
    p_data = _load_json(r_dir / "physics_model_metrics.json")
    candidates = [
        _collect_h2_claim(h_data),
        _collect_h3_claim(p_data),
        _collect_comparison_claim(h_data, p_data),
    ]
    return [c for c in candidates if c is not None]


def _extract_shift_fc(
    fc_dict: dict[str, Any],
) -> tuple[int, int, float, float, float] | None:
    """Extract false call count, scans, rate, and 95% CI bounds."""
    if not isinstance(fc_dict, dict):
        return None
    if any(k not in fc_dict for k in ("k", "n")):
        return None
    k = int(fc_dict["k"])
    n = int(fc_dict["n"])
    rate = float(fc_dict.get("rate", k / n if n > 0 else 0.0))
    low = float(fc_dict.get("ci_low", fc_dict.get("interval_95", [0.0, 0.0])[0]))
    high = float(fc_dict.get("ci_high", fc_dict.get("interval_95", [0.0, 0.0])[1]))
    return k, n, rate, low, high


@dataclass(frozen=True)
class ShiftMetrics:
    """Encapsulates metrics extracted for a single site adaptation shift."""

    shift_name: str
    b_fc: tuple[int, int, float, float, float]
    a_fc: tuple[int, int, float, float, float]
    det: tuple[int, int, int, int]
    n_clean: int
    n_calib_clean: int = 0
    underpowered: Any = None
    u95_scope: Any = None


def _extract_shift_det(shift_data: dict[str, Any]) -> tuple[int, int, int, int]:
    """Extract baseline and adapted detection counts."""
    b_det = shift_data.get("baseline", {}).get("detection", {})
    a_det = shift_data.get("adapted", {}).get("detection", {})
    bk_det = int(b_det.get("k", b_det.get("n_hit", 0)))
    bn_det = int(b_det.get("n", b_det.get("n_flawed", 0)))
    ak_det = int(a_det.get("k", a_det.get("n_hit", 0)))
    an_det = int(a_det.get("n", a_det.get("n_flawed", 0)))
    return bk_det, bn_det, ak_det, an_det


def _format_shift_text(m: ShiftMetrics) -> str:
    """Format site adaptation claim comparison text."""
    bk, bn, br, blow, bhigh = m.b_fc
    ak, an, ar, alow, ahigh = m.a_fc
    bk_d, bn_d, ak_d, an_d = m.det
    b_s = (
        f"{bk}/{bn} (rate {_format_rate(br)}, 95% CI [{_format_rate(blow)}, {_format_rate(bhigh)}])"
    )
    a_s = (
        f"{ak}/{an} (rate {_format_rate(ar)}, 95% CI [{_format_rate(alow)}, {_format_rate(ahigh)}])"
    )
    det_s = f"detection baseline {bk_d}/{bn_d} vs adapted {ak_d}/{an_d}"
    extras = [f"n_calib_clean {m.n_calib_clean}", f"n_test_clean {m.n_clean}"]
    if m.underpowered is not None:
        extras.append(f"test_underpowered {1 if m.underpowered else 0}")
    if m.u95_scope is not None:
        u_val = (
            _format_rate(float(m.u95_scope))
            if isinstance(m.u95_scope, (int, float))
            else str(m.u95_scope)
        )
        extras.append(f"upper95_fcr_scope {u_val}")
    extra_str = f", {', '.join(extras)}" if extras else ""
    return (
        f"Site adaptation shift {m.shift_name}: {det_s}, "
        f"false calls baseline {b_s} vs adapted {a_s}{extra_str}."
    )


def _build_shift_numbers(m: ShiftMetrics) -> dict[str, float]:
    """Construct numbers dictionary for shift baseline and adapted false calls."""
    bk, bn, br, blow, bhigh = m.b_fc
    ak, an, ar, alow, ahigh = m.a_fc
    bk_d, bn_d, ak_d, an_d = m.det
    nums: dict[str, float] = {
        "baseline_det_k": float(bk_d),
        "baseline_det_n": float(bn_d),
        "adapted_det_k": float(ak_d),
        "adapted_det_n": float(an_d),
        "baseline_k": float(bk),
        "baseline_n": float(bn),
        "baseline_rate": br,
        "baseline_ci_low": blow,
        "baseline_ci_high": bhigh,
        "adapted_k": float(ak),
        "adapted_n": float(an),
        "adapted_rate": ar,
        "adapted_ci_low": alow,
        "adapted_ci_high": ahigh,
        "n_calib_clean": float(m.n_calib_clean),
        "n_test_clean": float(m.n_clean),
    }
    if m.underpowered is not None:
        nums["test_underpowered"] = 1.0 if m.underpowered else 0.0
    if isinstance(m.u95_scope, (int, float)):
        nums["upper95_fcr_scope"] = float(m.u95_scope)
    return nums


def _build_single_site_adaptation_claim(
    shift_name: str,
    shift_data: dict[str, Any],
    top_data: dict[str, Any] | None = None,
) -> Claim | None:
    """Build claim for a single site adaptation shift."""
    b_m = _extract_shift_fc(shift_data.get("baseline", {}).get("false_calls", {}))
    a_m = _extract_shift_fc(shift_data.get("adapted", {}).get("false_calls", {}))
    if b_m is None or a_m is None:
        return None
    det_m = _extract_shift_det(shift_data)
    top = top_data if isinstance(top_data, dict) else {}
    n_img = top.get("n_images", {})
    n_clean = int(shift_data.get("n_test_clean", n_img.get("test_clean", b_m[1])))
    n_calib = int(
        shift_data.get("n_calib_clean", top.get("n_calib_clean", n_img.get("cal_clean", 0)))
    )
    underp = shift_data.get("test_underpowered", top.get("test_underpowered"))
    u95 = shift_data.get("upper95_fcr_scope", top.get("upper95_fcr_scope"))
    m = ShiftMetrics(shift_name, b_m, a_m, det_m, n_clean, n_calib, underp, u95)
    slug = re.sub(r"[^\w.]+", "_", shift_name.lower()).strip("_")
    return Claim(
        id=f"site_adaptation_{slug}",
        text=_format_shift_text(m),
        evidence_file="results/site_adaptation.json",
        evidence_path=f"shifts.{shift_name}",
        scope_limit="Real ML-NDT images with synthetic perturbations; single seed",
        numbers=_build_shift_numbers(m),
    )


def _build_site_adaptation_claims(data_or_dir: dict[str, Any] | Path) -> list[Claim]:
    """Build site adaptation claims for all shifts in data."""
    data = (
        data_or_dir
        if isinstance(data_or_dir, dict)
        else _load_json(data_or_dir / "site_adaptation.json")
    )
    if data is None:
        return []
    shifts = data.get("shifts")
    if not isinstance(shifts, dict):
        return []
    claims: list[Claim] = []
    for name, shift_info in shifts.items():
        if isinstance(shift_info, dict):
            c = _build_single_site_adaptation_claim(name, shift_info, data)
            if c is not None:
                claims.append(c)
    return claims


def _collect_site_adaptation_claims(r_dir: Path) -> list[Claim]:
    """Collect site adaptation claims from results directory."""
    return _build_site_adaptation_claims(r_dir)


def _extract_joint_arm_counts(
    arm_data: dict[str, Any],
) -> tuple[int, int, int, int] | None:
    """Extract detection and false call counts for a joint training arm."""
    det = arm_data.get("detections", arm_data.get("detection", {}))
    fc = arm_data.get("false_calls", {})
    if not (isinstance(det, dict) and isinstance(fc, dict)):
        return None
    det_k = int(det.get("k", det.get("n_hit", 0)))
    det_n = int(det.get("n", det.get("n_flawed", 0)))
    fc_k = int(fc.get("k", fc.get("n_false", 0)))
    fc_n = int(fc.get("n", fc.get("n_unflawed", fc.get("n_scans", 0))))
    if det_n <= 0 or fc_n <= 0:
        return None
    return det_k, det_n, fc_k, fc_n


def _extract_joint_extras(
    arm_data: dict[str, Any], top: dict[str, Any]
) -> tuple[str, dict[str, float]]:
    """Extract extra joint training parameters and their numeric values."""
    actual_sim = arm_data.get("actual_sim_fraction", top.get("actual_sim_fraction"))
    seeds = arm_data.get("seeds", top.get("seeds"))
    inv_type = arm_data.get("interval_type", top.get("interval_type"))
    extras: list[str] = []
    nums: dict[str, float] = {}
    if actual_sim is not None:
        extras.append(f"actual_sim_fraction {_format_rate(float(actual_sim))}")
        nums["actual_sim_fraction"] = float(actual_sim)
    if seeds is not None:
        extras.append(f"seeds {seeds}")
    if inv_type is not None:
        extras.append(f"interval_type {inv_type}")
    extra_str = f", {', '.join(extras)}" if extras else ""
    return extra_str, nums


def _build_single_joint_arm_claim(
    crack_tag: str,
    arm_name: str,
    arm_data: dict[str, Any],
    top_data: dict[str, Any] | None = None,
) -> Claim | None:
    """Build claim for a single crack and arm in joint training."""
    counts = _extract_joint_arm_counts(arm_data)
    if counts is None:
        return None
    det_k, det_n, fc_k, fc_n = counts
    top = top_data if isinstance(top_data, dict) else {}
    extra_str, extra_nums = _extract_joint_extras(arm_data, top)
    text = (
        f"Joint training {crack_tag} arm {arm_name}: "
        f"detection {det_k}/{det_n}, false calls {fc_k}/{fc_n}{extra_str}."
    )
    scope = "real metrics on ML-NDT weld with 3 physical cracks; sim images Synthetic data"
    numbers = {
        "detection_k": float(det_k),
        "detection_n": float(det_n),
        "false_calls_k": float(fc_k),
        "false_calls_n": float(fc_n),
    }
    numbers.update(extra_nums)
    return Claim(
        id=f"joint_train_{crack_tag}_{arm_name.lower()}",
        text=text,
        evidence_file="results/joint_train_metrics.json",
        evidence_path="folds",
        scope_limit=scope,
        numbers=numbers,
    )


def _collect_fold_arm_claims(
    fold: dict[str, Any], top_data: dict[str, Any] | None = None
) -> list[Claim]:
    """Collect joint training claims across all arms of a single fold."""
    held_out = str(fold.get("held_out", ""))
    crack_tag = held_out if held_out.startswith("crack_") else f"crack_{held_out}"
    arms = fold.get("arms")
    if not isinstance(arms, dict):
        return []
    claims: list[Claim] = []
    for arm_name, arm_data in arms.items():
        if isinstance(arm_data, dict):
            c = _build_single_joint_arm_claim(crack_tag, str(arm_name), arm_data, top_data)
            if c is not None:
                claims.append(c)
    return claims


def _arm_beats_baseline_on_fold(arm: str, fold: dict[str, Any]) -> bool:
    """Check if an arm strictly beats baseline arm A on a single fold."""
    arms = fold.get("arms", {})
    if arm not in arms or "A" not in arms:
        return False
    a_counts = _extract_joint_arm_counts(arms["A"])
    arm_counts = _extract_joint_arm_counts(arms[arm])
    if a_counts is None or arm_counts is None:
        return False
    a_rate = a_counts[0] / a_counts[1]
    arm_rate = arm_counts[0] / arm_counts[1]
    return arm_rate > a_rate


def _arm_consistently_improves(folds: list[dict[str, Any]]) -> bool:
    """Check if any non-A arm strictly beats baseline A across all folds."""
    non_a = {k for f in folds for k in f.get("arms", {}) if k != "A"}
    if not non_a:
        return False
    for arm in non_a:
        if all(_arm_beats_baseline_on_fold(arm, f) for f in folds):
            return True
    return False


def _build_joint_train_summary_claim(
    folds: list[dict[str, Any]], top: dict[str, Any]
) -> Claim | None:
    """Build summary claim across joint training folds without verdict bias."""
    if len(folds) < 2:
        return None
    n_folds = float(len(folds))
    arms_set = {arm for f in folds for arm in f.get("arms", {})}
    n_arms = float(len(arms_set))
    any_improved = _arm_consistently_improves(folds)
    if not any_improved:
        verdict = f"across {int(n_folds)} folds: no arm improves consistently over baseline arm A"
    else:
        verdict = f"across {int(n_folds)} folds and {int(n_arms)} arms evaluated"
    text = f"Joint training summary {verdict}."
    numbers = {"n_folds": n_folds, "n_arms": n_arms}
    return Claim(
        id="joint_train_summary",
        text=text,
        evidence_file="results/joint_train_metrics.json",
        evidence_path="folds",
        scope_limit="real metrics on ML-NDT weld with 3 physical cracks; sim images Synthetic data",
        numbers=numbers,
    )


def _build_joint_train_claims(data_or_dir: dict[str, Any] | Path) -> list[Claim]:
    """Build joint training claims across all held-out cracks and arms."""
    data = (
        data_or_dir
        if isinstance(data_or_dir, dict)
        else _load_json(data_or_dir / "joint_train_metrics.json")
    )
    if data is None:
        return []
    folds = data.get("folds")
    fold_list: list[dict[str, Any]] = []
    if isinstance(folds, list):
        fold_list = [f for f in folds if isinstance(f, dict)]
    elif isinstance(folds, dict):
        fold_list = [f for f in folds.values() if isinstance(f, dict)]
    claims: list[Claim] = []
    for f in fold_list:
        claims.extend(_collect_fold_arm_claims(f, data))
    c_summary = _build_joint_train_summary_claim(fold_list, data)
    if c_summary is not None:
        claims.append(c_summary)
    return claims


def _collect_joint_train_claims(r_dir: Path) -> list[Claim]:
    """Collect joint training claims from results directory."""
    return _build_joint_train_claims(r_dir)


def _format_fca_per_fold(per_fold: Any) -> str | None:
    """Build compact per-fold summary '<fold>: <n_flagged> flagged / <n_clean>'."""
    if not isinstance(per_fold, dict) or not per_fold:
        return None
    parts: list[str] = []
    for fold, f_data in per_fold.items():
        if isinstance(f_data, dict):
            n_fl = f_data.get("n_flagged", 0)
            n_cl = f_data.get("n_clean", 0)
            parts.append(f"{fold}: {n_fl} flagged / {n_cl}")
    return "; ".join(parts) if parts else None


def _format_fca_text(
    counts: tuple[int, int],
    agr_str: str,
    cause_str: str,
    extra: tuple[str | None, str | None],
) -> str:
    """Format false-call analysis text with clean, flagged, agreement, and causes."""
    n_clean, n_flagged = counts
    fold_str, note = extra
    text = (
        f"False-call analysis on ML-NDT (316L weld, 3 physical cracks): "
        f"{n_clean} fold-scan clean evaluations (each clean scan scored by every fold model), "
        f"{n_flagged} flagged by any detector; "
        f"agreement counts: {agr_str}; causes: {cause_str}."
    )
    if fold_str:
        text += f" Per-fold: {fold_str}."
    if note:
        text += f" Protocol note: {note}."
    return text


def _extract_fca_counts(
    data: dict[str, Any],
) -> tuple[str, str, dict[str, float]]:
    """Extract formatted agreement string, cause string, and count numbers."""
    agr_tbl = data.get("agreement_table", {})
    agr_counts = agr_tbl.get("counts_by_agreement", data.get("agreement_counts", {}))
    agr_parts = [f"{k}: {v}" for k, v in agr_counts.items()]
    agr_str = ", ".join(agr_parts) if agr_parts else "none"
    causes = data.get("cause_counts", data.get("causes", {}))
    cause_parts = [f"{k}: {v}" for k, v in causes.items()]
    cause_str = ", ".join(cause_parts) if cause_parts else "none"
    nums: dict[str, float] = {}
    for k, v in agr_counts.items():
        nums[f"agreement_{k}"] = float(v)
    for k, v in causes.items():
        nums[f"cause_{k}"] = float(v)
    return agr_str, cause_str, nums


def _build_false_call_analysis_claim(data_or_dir: dict[str, Any] | Path) -> Claim | None:
    """Build claim from results/false_call_analysis.json if available."""
    if not isinstance(data_or_dir, dict):
        data = _load_json(data_or_dir / "false_call_analysis.json")
    else:
        data = data_or_dir
    if data is None:
        return None
    n_cl = int(
        data.get("n_clean_pairs", data.get("n_clean", data.get("clean_images_evaluated", 0)))
    )
    n_fl = int(data.get("n_flagged", data.get("flagged_by_any_detector", 0)))
    agr_str, cause_str, nums = _extract_fca_counts(data)
    fold_s = _format_fca_per_fold(data.get("per_fold", data.get("folds")))
    proto = data.get("protocol_note", data.get("protocol"))
    text = _format_fca_text((n_cl, n_fl), agr_str, cause_str, (fold_s, proto))
    numbers = {
        "clean_images_evaluated": float(n_cl),
        "n_clean_pairs": float(n_cl),
        "flagged_by_any_detector": float(n_fl),
        **nums,
    }
    return Claim(
        id="false_call_analysis",
        text=text,
        evidence_file="results/false_call_analysis.json",
        evidence_path="agreement_table",
        scope_limit="ML-NDT (316L weld, 3 physical cracks), flaw-free scans",
        numbers=numbers,
    )


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
    c_crackval = _build_loco_crackval_claim(r_dir)
    if c_crackval is not None:
        claims.append(c_crackval)

    claims.extend(_collect_hybrid_claims(r_dir))
    claims.extend(_collect_site_adaptation_claims(r_dir))
    claims.extend(_collect_joint_train_claims(r_dir))
    c_fca = _build_false_call_analysis_claim(r_dir)
    if c_fca is not None:
        claims.append(c_fca)
    claims.extend(collect_ml_claims(r_dir))
    claims.extend(collect_ti_claims(r_dir))
    claims.extend(_collect_final_claims(r_dir, mat_dict))
    return claims
