"""Render markdown results tables from results/*.json into README.md."""

from __future__ import annotations

import json
import re
import sys
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path

BEGIN_MARKER = "<!-- RESULTS:BEGIN -->"
END_MARKER = "<!-- RESULTS:END -->"
ROW_KEYS = ("rules", "supervised", "anomaly", "fused")


def round_ceil(val: float | int | str, decimals: int = 2) -> str:
    """Format number rounded up (ceiling) to avoid project-favour rounding."""
    try:
        step = Decimal("10") ** -decimals
        d = Decimal(str(val))
        return str(d.quantize(step, rounding=ROUND_CEILING))
    except (ArithmeticError, ValueError):
        # Fallback on invalid numerical representations
        return "nan"


def round_floor(val: float | int | str, decimals: int = 2) -> str:
    """Format number rounded down (floor) to avoid project-favour rounding."""
    try:
        step = Decimal("10") ** -decimals
        d = Decimal(str(val))
        return str(d.quantize(step, rounding=ROUND_FLOOR))
    except (ArithmeticError, ValueError):
        # Fallback on invalid numerical representations
        return "nan"


def round_rate(val: float | int | str | None, decimals: int = 3) -> str:
    """Format rate rounded half to even via round(x, decimals)."""
    if val is None:
        return "n/a"
    try:
        rounded = round(float(val), decimals)
        return f"{rounded:.{decimals}f}"
    except (ArithmeticError, ValueError, TypeError):
        # Fallback on invalid numerical representations
        return "nan"


def load_json(path: Path) -> dict | None:
    """Load JSON from path if file exists, else return None."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_flaw_counts(content: str) -> tuple[str, str]:
    """Parse accepted and evaluated flaw counts from data inventory text."""
    acc_m = re.search(r"totaling\s+(\d+)\s+distinct physical flaws", content, re.IGNORECASE)
    accepted = acc_m.group(1) if acc_m else "UNVERIFIED"
    if accepted == "UNVERIFIED":
        fb = re.search(r"(\d+)\s+distinct(?: real)? physical flaws", content, re.IGNORECASE)
        accepted = fb.group(1) if fb else "UNVERIFIED"

    used_m = re.search(
        r"Used in evaluation:\s*(\d+)\s+distinct physical flaws", content, re.IGNORECASE
    )
    used = used_m.group(1) if used_m else "UNVERIFIED"
    return accepted, used


def render_evidence_limits(results_dir: Path) -> str:
    """Render evidence limits blockquote from data_inventory.md."""
    path = results_dir / "data_inventory.md"
    if not path.exists():
        return "*data_inventory.md not found.*\n"
    content = path.read_text(encoding="utf-8")
    acc, used = parse_flaw_counts(content)
    line_flaws = (
        f"> - Total distinct real physical flaws: {acc} accepted, "
        f"{used} used in evaluation (across accepted real-data sources)."
    )
    lines = [
        "> ### Evidence limits",
        line_flaws,
        "> - Real data are weld PAUT datasets not bars.",
        "> - Simulator constants are UNVERIFIED - Ruri to check.",
        "> - UTriage is an advisory independent prototype; inspectors make all decisions.",
    ]
    return "\n".join(lines) + "\n"


def render_superseded_note(results_dir: Path) -> str:
    """Render superseded notice if leakage audit indicates crack leakage."""
    path = results_dir / "leakage_audit.json"
    if not path.exists():
        return "*leakage_audit.json not found.*\n"
    data = load_json(path)
    if not isinstance(data, dict) or not data.get("leakage_by_physical_crack"):
        return ""
    return (
        "> [!NOTE]\n"
        "> **Superseded:** Earlier test-split scores measured interpolation "
        "within known cracks and are superseded by the leave-one-crack-out results.\n"
    )


def _format_cell(cell_data: dict | None) -> str:
    """Format single cell metrics or return n/a with reason."""
    if not cell_data or not cell_data.get("available", True):
        reason = "not evaluated" if not cell_data else cell_data.get("reason", "unavailable")
        return f"n/a ({reason})"
    det = cell_data.get("detection", {})
    rate, ci_low, ci_high = det.get("rate"), det.get("ci_low"), det.get("ci_high")
    fc = cell_data.get("false_calls", {})
    fc_rate = fc.get("rate")
    if any([rate is None, fc_rate is None]):
        reason = cell_data.get("reason", "incomplete")
        return f"n/a ({reason})"
    k, n = det.get("k", 0), det.get("n", 0)
    r_str = round_floor(rate, 2)
    low_str = round_floor(0.0 if ci_low is None else ci_low, 2)
    high_str = round_ceil(0.0 if ci_high is None else ci_high, 2)
    fcr_str = round_rate(fc_rate, 3)
    return f"{r_str} [{low_str}, {high_str}] {k}/{n}, FCR: {fcr_str}"


def _is_real_column(col: dict) -> bool:
    """Check whether a column definition belongs to real data."""
    group = str(col.get("group", "")).lower()
    label = str(col.get("data_label", "")).lower()
    return "real" in group or "real" in label


def render_matrix_table(title: str, cols: list[dict], rows: dict) -> str:
    """Render single markdown table for given columns and rows."""
    header_block = f"### {title}\n\n"
    if not cols:
        return header_block + "*No columns available.*\n"
    col_ids = [str(c.get("id", "")) for c in cols]
    lines = [
        "| Method | " + " | ".join(col_ids) + " |",
        "|---| " + " | ".join(["---"] * len(col_ids)) + " |",
    ]
    for rk in ROW_KEYS:
        row_dict = rows.get(rk, {})
        cells = [_format_cell(row_dict.get(cid)) for cid in col_ids]
        lines.append(f"| {rk} | " + " | ".join(cells) + " |")
    return header_block + "\n".join(lines) + "\n"


def render_matrix_section(results_dir: Path) -> str:
    """Render real and synthetic generalization matrix markdown tables."""
    path = results_dir / "generalization_matrix.json"
    if not path.exists():
        return "*generalization_matrix.json not found.*\n"
    data = load_json(path)
    if not isinstance(data, dict):
        return "*generalization_matrix.json invalid.*\n"
    cols = data.get("columns", [])
    rows = data.get("rows", {})
    real_cols = [c for c in cols if _is_real_column(c)]
    synth_cols = [c for c in cols if not _is_real_column(c)]
    t1 = render_matrix_table("Table 1: Generalization matrix (Real data columns)", real_cols, rows)
    t2 = render_matrix_table(
        "Table 2: Generalization matrix (Synthetic data columns)", synth_cols, rows
    )
    return f"{t1}\n{t2}"


def _format_shift_metric(m: dict | None) -> str:
    """Format single shift metric as k/n (rate)."""
    if not m:
        return "n/a"
    k = m.get("k", 0)
    n = m.get("n", 0)
    rate_str = round_rate(m.get("rate"), 3)
    return f"{k}/{n} ({rate_str})"


def render_site_adaptation_table(results_dir: Path) -> str:
    """Render site adaptation results table if JSON exists."""
    path = results_dir / "site_adaptation.json"
    data = load_json(path)
    if not isinstance(data, dict):
        return ""
    shifts = data.get("shifts", {})
    if not shifts:
        return ""
    n_calib = data.get("n_calib_clean", "unknown")
    tbl_cols = (
        "| Shift | Baseline False Calls | Baseline Detection | "
        "Adapted False Calls | Adapted Detection | n_calib_clean |"
    )
    lines = [
        "### Table 3: Site adaptation",
        "",
        "*Real ML-NDT weld images with synthetic perturbations, single seed.*",
        "",
        tbl_cols,
        "|---|---|---|---|---|---|",
    ]
    for name, s_data in shifts.items():
        b_fc = _format_shift_metric(s_data.get("baseline", {}).get("false_calls"))
        b_det = _format_shift_metric(s_data.get("baseline", {}).get("detection"))
        a_fc = _format_shift_metric(s_data.get("adapted", {}).get("false_calls"))
        a_det = _format_shift_metric(s_data.get("adapted", {}).get("detection"))
        lines.append(f"| {name} | {b_fc} | {b_det} | {a_fc} | {a_det} | {n_calib} |")
    return "\n".join(lines) + "\n"


def render_joint_train_table(results_dir: Path) -> str:
    """Render joint training metrics table if JSON exists."""
    path = results_dir / "joint_train_metrics.json"
    data = load_json(path)
    if not isinstance(data, dict):
        return ""
    arms = data.get("arms", {})
    if not arms:
        return ""
    lookup = {
        (f.get("held_out"), arm_id): f for arm_id, arm in arms.items() for f in arm.get("folds", [])
    }
    lines = [
        "### Table 4: Joint training metrics",
        "",
        "*sim images are Synthetic data*",
        "",
        "| Held-out Crack | Arm | Detections | False Calls |",
        "|---|---|---|---|",
    ]
    for crack in ("crack_1.6", "crack_4.0", "crack_8.6"):
        for arm_id in ("A", "B", "C", "D"):
            fold = lookup.get((crack, arm_id), {})
            det, fc = fold.get("detections", {}), fold.get("false_calls", {})
            det_s = f"{det.get('k', 0)}/{det.get('n', 0)}"
            fc_s = f"{fc.get('k', 0)}/{fc.get('n', 0)}"
            lines.append(f"| {crack} | {arm_id} | {det_s} | {fc_s} |")
    return "\n".join(lines) + "\n"


def render_false_call_table(results_dir: Path) -> str:
    """Render false call analysis table if JSON exists."""
    path = results_dir / "false_call_analysis.json"
    data = load_json(path)
    if not isinstance(data, dict):
        return ""
    agr = data.get("agreement_table", {})
    counts = agr.get("counts_by_agreement", {})
    if not counts:
        return ""
    n_clean = data.get("n_clean", "unknown")
    n_flagged = data.get("n_flagged", agr.get("total_flagged", "unknown"))
    lines = [
        "### Table 5: False-call analysis",
        "",
        f"- Clean images evaluated: {n_clean}",
        f"- Flagged by any detector: {n_flagged}",
        "",
        "| Agreement Count | Scans Flagged |",
        "|---|---|",
    ]
    for count_key in sorted(counts.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        lines.append(f"| {count_key} | {counts[count_key]} |")
    return "\n".join(lines) + "\n"


def render_links_line() -> str:
    """Render markdown link line to claims and matrix figure."""
    return (
        "Claims and evidence: [results/claims.md](results/claims.md) | "
        "Matrix figure: [results/figures/generalization_matrix.png]"
        "(results/figures/generalization_matrix.png)\n"
    )


def render_all_tables(results_dir: Path) -> str:
    """Render all Phase 8 results markdown blocks for README.md."""
    parts = [
        render_evidence_limits(results_dir),
        render_superseded_note(results_dir),
        render_matrix_section(results_dir),
        render_site_adaptation_table(results_dir),
        render_joint_train_table(results_dir),
        render_false_call_table(results_dir),
        render_links_line(),
    ]
    return "\n\n".join(p.strip() for p in parts if p.strip()) + "\n"


def update_readme_text(readme_text: str, tables_markdown: str) -> str:
    """Update markdown text between RESULTS markers, adding markers if missing."""
    clean_tables = tables_markdown.strip()
    block = f"{BEGIN_MARKER}\n\n{clean_tables}\n\n{END_MARKER}"

    if BEGIN_MARKER in readme_text and END_MARKER in readme_text:
        start = readme_text.index(BEGIN_MARKER)
        end = readme_text.index(END_MARKER) + len(END_MARKER)
        return readme_text[:start] + block + readme_text[end:]

    if "## Results" in readme_text:
        pos = readme_text.index("## Results") + len("## Results")
        return readme_text[:pos] + f"\n\n{block}\n" + readme_text[pos:]

    target = "## Licence"
    if target in readme_text:
        pos = readme_text.index(target)
        return readme_text[:pos] + f"## Results\n\n{block}\n\n" + readme_text[pos:]

    return readme_text.rstrip() + f"\n\n## Results\n\n{block}\n"


def update_readme_file(readme_path: Path, results_dir: Path) -> None:
    """Render tables from results and update README.md file in place."""
    text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    tables_md = render_all_tables(results_dir)
    new_text = update_readme_text(text, tables_md)
    readme_path.write_text(new_text, encoding="utf-8")


def main() -> None:
    """Command line entry point to update README tables from results."""
    results_dir = Path("results")
    readme_path = Path("README.md")
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        readme_path = Path(sys.argv[2])
    update_readme_file(readme_path, results_dir)
    print(f"Updated {readme_path} with results tables from {results_dir}.")


if __name__ == "__main__":
    main()
