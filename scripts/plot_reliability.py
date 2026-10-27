"""Generalization matrix calibration curves and reliability diagrams.

Reads results/generalization_matrix.json, extracts reliability curves and ECE
metrics across models, and renders calibration plots per row and column group.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


@dataclass(frozen=True)
class CurveData:
    """Extracted valid calibration curve points and ECE for one column."""

    col_id: str
    mean_conf: tuple[float, ...]
    frac_pos: tuple[float, ...]
    ece: float | None


@dataclass(frozen=True)
class PlotTarget:
    """Rendering target configuration for a row and data group."""

    row_name: str
    group_name: str
    data_label: str
    curves: tuple[CurveData, ...]
    out_path: Path


def classify_column_group(col_meta: dict[str, Any]) -> str:
    """Classify column metadata into 'real' or 'synthetic' group."""
    label = str(col_meta.get("data_label", "")).lower()
    if "real" in label:
        return "real"
    if "synth" in label:
        return "synthetic"
    cid = str(col_meta.get("id", "")).lower()
    grp = str(col_meta.get("group", "")).lower()
    if cid.startswith("sim_") or "sim" in grp:
        return "synthetic"
    return "real"


def resolve_group_for_col(col_id: str, col_groups: dict[str, str]) -> str:
    """Resolve column group from precomputed mapping or column ID."""
    if col_id in col_groups:
        return col_groups[col_id]
    return classify_column_group({"id": col_id})


def get_column_groups(matrix: dict[str, Any]) -> dict[str, str]:
    """Map column IDs to group name ('real' or 'synthetic')."""
    mapping: dict[str, str] = {}
    for col in matrix.get("columns", []):
        cid = col.get("id")
        if cid:
            mapping[cid] = classify_column_group(col)
    return mapping


def _parse_ece(val: Any) -> float | None:
    """Safely extract float ECE if numeric and non-nan."""
    if isinstance(val, (int, float)) and not np.isnan(val):
        return float(val)
    return None


def _extract_valid_points(rel: Any) -> list[tuple[float, float]]:
    """Extract non-null confidence and empirical fraction pairs."""
    if not isinstance(rel, dict):
        return []
    confs = rel.get("mean_conf")
    fracs = rel.get("frac_pos")
    if not (isinstance(confs, list) and isinstance(fracs, list)):
        return []
    return [
        (float(c), float(f))
        for c, f in zip(confs, fracs, strict=False)
        if c is not None and f is not None
    ]


def extract_curve(col_id: str, cell: Any) -> CurveData | None:
    """Extract valid reliability curve points and ECE from cell dict."""
    if not isinstance(cell, dict) or cell.get("available") is False:
        return None
    pts = _extract_valid_points(cell.get("reliability"))
    if not pts:
        return None
    return CurveData(
        col_id=col_id,
        mean_conf=tuple(p[0] for p in pts),
        frac_pos=tuple(p[1] for p in pts),
        ece=_parse_ece(cell.get("ece")),
    )


def extract_row_curves(
    row_dict: dict[str, Any], col_groups: dict[str, str]
) -> dict[str, list[CurveData]]:
    """Extract valid curves grouped by 'real' and 'synthetic'."""
    grouped: dict[str, list[CurveData]] = {"real": [], "synthetic": []}
    for col_id, cell in row_dict.items():
        curve = extract_curve(col_id, cell)
        if curve is None:
            continue
        grp = resolve_group_for_col(col_id, col_groups)
        target_list = grouped.setdefault(grp, [])
        target_list.append(curve)
    return grouped


def _make_target(row: str, grp: str, pts: tuple[CurveData, ...], out_path: Path) -> PlotTarget:
    """Build single PlotTarget for real or synthetic group."""
    lbl = "Real data" if grp == "real" else "Synthetic data"
    return PlotTarget(row_name=row, group_name=grp, data_label=lbl, curves=pts, out_path=out_path)


def build_plot_targets(
    row_name: str, group_curves: dict[str, list[CurveData]], out_dir: Path
) -> list[PlotTarget]:
    """Determine output paths and plot configurations for a model row."""
    real_pts = tuple(group_curves.get("real", []))
    syn_pts = tuple(group_curves.get("synthetic", []))
    both = bool(real_pts and syn_pts)
    targets: list[PlotTarget] = []
    if real_pts:
        p = out_dir / f"reliability_{row_name}{'_real' if both else ''}.png"
        targets.append(_make_target(row_name, "real", real_pts, p))
    if syn_pts:
        p = out_dir / f"reliability_{row_name}{'_synthetic' if both else ''}.png"
        targets.append(_make_target(row_name, "synthetic", syn_pts, p))
    return targets


def _format_legend_label(col_id: str, ece: float | None) -> str:
    """Format legend string with column id and ECE value."""
    if ece is not None:
        return f"{col_id} (ECE: {ece:.3f})"
    return col_id


def _draw_diagram_axes(ax: plt.Axes, target: PlotTarget) -> None:
    """Draw diagonal reference and column curves onto axes."""
    ax.plot([0.0, 1.0], [0.0, 1.0], "k--", label="Ideal", alpha=0.7)
    for c in target.curves:
        lbl = _format_legend_label(c.col_id, c.ece)
        ax.plot(c.mean_conf, c.frac_pos, marker="o", markersize=4, label=lbl)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Reliability Diagram — {target.row_name.capitalize()} ({target.data_label})")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)


def draw_reliability_diagram(target: PlotTarget) -> Path:
    """Draw and save single reliability diagram to target.out_path."""
    target.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    _draw_diagram_axes(ax, target)
    fig.tight_layout()
    fig.savefig(target.out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target.out_path


def plot_all_reliability(matrix: dict[str, Any], out_dir: Path) -> list[Path]:
    """Generate reliability diagram figures for each row with valid curves."""
    col_groups = get_column_groups(matrix)
    rows = matrix.get("rows", {})
    written: list[Path] = []
    for row_name, row_dict in rows.items():
        if not isinstance(row_dict, dict):
            continue
        group_curves = extract_row_curves(row_dict, col_groups)
        targets = build_plot_targets(row_name, group_curves, out_dir)
        for target in targets:
            saved_path = draw_reliability_diagram(target)
            written.append(saved_path)
    return written


def load_matrix(path: Path) -> dict[str, Any]:
    """Load generalization matrix from JSON file or build from results directory."""
    if path.is_file():
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    results_dir = path.parent if path.suffix == ".json" else path
    if results_dir.is_dir():
        from utriage.report.matrix import build_matrix

        return build_matrix(results_dir)
    raise FileNotFoundError(f"Matrix file not found: {path}")


def main(argv: list[str] | None = None) -> list[Path]:
    """Main CLI entrypoint to load matrix, generate diagrams, and print files."""
    args = sys.argv[1:] if argv is None else argv
    matrix_path = Path(args[0]) if len(args) > 0 else Path("results/generalization_matrix.json")
    out_dir = Path(args[1]) if len(args) > 1 else Path("results/figures")
    matrix = load_matrix(matrix_path)
    written = plot_all_reliability(matrix, out_dir)
    for p in written:
        print(p)
    return written


if __name__ == "__main__":
    main()
