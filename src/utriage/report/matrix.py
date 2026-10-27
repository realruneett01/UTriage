"""Generalization matrix assembly and visualization (project plan Phase 8).

Synthesizes model evaluation results across LOCO, cross-dataset, and simulation
stress conditions into a unified generalization matrix artifact and heatmap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from utriage.pod.metrics import clopper_pearson, false_calls_per_scan  # noqa: E402

LABEL_NOTE: str = (
    "Real and synthetic results reported separately; synthetic results are "
    "uncalibrated physical proxies."
)
RULES_UNAVAIL_REAL: str = (
    "rules engine operates on bar immersion A-scans, not on weld PAUT B-scan images"
)
FUSED_UNAVAIL_REAL: str = "per-image fused scores not computed"

DEFAULT_SIM_SETS: tuple[str, ...] = (
    "in_distribution",
    "freq_10mhz",
    "material_third",
    "noise_above_range",
    "artefact_bubble",
    "artefact_lost_backwall",
    "artefact_waterpath_drift",
    "artefact_roughness",
    "artefact_missing_coverage",
)


@dataclass(frozen=True)
class MatrixContext:
    """Loaded results data for generalization matrix construction."""

    loco: dict[str, Any] | None
    anomaly: dict[str, Any] | None
    sim: dict[str, Any] | None
    cross: dict[str, Any] | None


@dataclass(frozen=True)
class CellCounts:
    """Aggregated counts for initializing a metrics cell."""

    k: int
    n: int
    n_false: int
    n_scans: int
    n_flaws: int


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    """Load JSON from disk if file exists; return None otherwise."""
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):  # narrow reason: corrupt or unreadable results file
        return None


def _format_cell_from_counts(counts: CellCounts) -> dict[str, Any]:
    """Construct standardized metrics cell from detection and false call counts."""
    rate = float(counts.k / counts.n) if counts.n > 0 else float("nan")
    ci_low, ci_high = clopper_pearson(counts.k, counts.n)
    n_images = counts.n + counts.n_scans
    return {
        "detection": {
            "n": counts.n,
            "k": counts.k,
            "rate": rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "method": "Clopper-Pearson two-sided 95",
        },
        "false_calls": false_calls_per_scan(counts.n_false, counts.n_scans),
        "ece": float("nan"),
        "reliability": {"counts": [counts.n]},
        "pod": {"fittable": False, "a90_95": None},
        "n_images": n_images,
        "n_physical_flaws": counts.n_flaws,
        "counts": {
            "images": n_images,
            "distinct_physical_flaws": counts.n_flaws,
            "hits": counts.k,
            "total_flaws_tested": counts.n,
        },
    }


def _normalize_loco_record(rec: dict[str, Any], n_flaws: int) -> dict[str, Any]:
    """Ensure LOCO fold or validation record satisfies cell schema."""
    if "detection" in rec and "false_calls" in rec:
        return rec
    dets = rec.get("detections", {})
    fcs = rec.get("false_calls", {})
    counts = CellCounts(
        k=int(dets.get("n_hit", 0)),
        n=int(dets.get("n_flawed", 0)),
        n_false=int(fcs.get("n_false", 0)),
        n_scans=int(fcs.get("n_unflawed", 0)),
        n_flaws=n_flaws,
    )
    cell_dict = _format_cell_from_counts(counts)
    if "pod" in rec:
        cell_dict["pod"] = rec["pod"]
    return cell_dict


def _get_loco_fold_cell(loco: dict[str, Any] | None, crack_id: str) -> dict[str, Any]:
    """Extract and format LOCO fold cell for a specific crack."""
    if loco is None:
        return {"available": False, "reason": "results/loco_metrics.json missing"}
    for fold in loco.get("folds", []):
        held = str(fold.get("held_out", ""))
        if crack_id in held:
            return _normalize_loco_record(fold, n_flaws=1)
    return {"available": False, "reason": f"LOCO fold for {crack_id} not found"}


def _get_loco_val_cell(loco: dict[str, Any] | None) -> dict[str, Any]:
    """Extract LOCO official validation cell."""
    if loco is None:
        return {"available": False, "reason": "results/loco_metrics.json missing"}
    val = loco.get("official_validation") or loco.get("part_a")
    if isinstance(val, dict):
        return _normalize_loco_record(val, n_flaws=3)
    return {"available": False, "reason": "LOCO official validation metrics missing"}


def _aggregate_loco_models(models: dict[str, Any]) -> dict[str, Any] | None:
    """Aggregate predictions across multiple LOCO model checkpoints."""
    loco_keys = [k for k in models if k.startswith("loco_crack_")]
    if not loco_keys:
        return None
    hits_list: list[int] = []
    n_list: list[int] = []
    false_list: list[int] = []
    scans_list: list[int] = []
    for k in loco_keys:
        ov = models[k].get("overall", models[k])
        det, fc = ov.get("detection", {}), ov.get("false_calls", {})
        hits_list.append(det.get("k", 0))
        n_list.append(det.get("n", 0))
        false_list.append(fc.get("n_false", 0))
        scans_list.append(fc.get("n_scans", 0))
    counts = CellCounts(
        k=int(round(float(np.mean(hits_list)))),
        n=int(round(float(np.mean(n_list)))),
        n_false=int(round(float(np.mean(false_list)))),
        n_scans=int(round(float(np.mean(scans_list)))),
        n_flaws=6,
    )
    return _format_cell_from_counts(counts)


def _get_cross_supervised(cross: dict[str, Any] | None) -> tuple[dict[str, Any], str]:
    """Extract cross-dataset supervised cell, preferring LOCO aggregate."""
    if cross is None:
        return (
            {"available": False, "reason": "results/cross_dataset_metrics.json missing"},
            "NDT_ML_Flaw test split (not yet available)",
        )
    models = cross.get("models", {})
    if "loco_aggregate" in models:
        return models["loco_aggregate"].get("overall", models["loco_aggregate"]), "LOCO aggregate"
    agg = _aggregate_loco_models(models)
    if agg is not None:
        return agg, "LOCO models aggregate on NDT_ML_Flaw"
    if "mlndt_best.pt" in models:
        return models["mlndt_best.pt"].get("overall", models["mlndt_best.pt"]), "mlndt_best"
    return (
        {"available": False, "reason": "No supervised model in cross-dataset metrics"},
        "No supervised model evaluated on NDT_ML_Flaw",
    )


def _get_cross_anomaly(cross: dict[str, Any] | None) -> dict[str, Any]:
    """Extract anomaly detection metrics from cross-dataset evaluation."""
    if cross is None:
        return {"available": False, "reason": "results/cross_dataset_metrics.json missing"}
    models = cross.get("models", {})
    for cand in ("anomaly_2d.pt", "anomaly_2d", "anomaly"):
        if cand in models:
            return models[cand].get("overall", models[cand])
    return {"available": False, "reason": "Anomaly model not evaluated on NDT_ML_Flaw"}


def _get_sim_cell(sim: dict[str, Any] | None, ts: str, model_key: str) -> dict[str, Any]:
    """Extract cell from simulation results for specified test set and model."""
    if sim is None:
        return {"available": False, "reason": "results/sim_generalization.json missing"}
    t_sets = sim.get("test_sets", {})
    if ts not in t_sets or model_key not in t_sets[ts]:
        return {"available": False, "reason": f"Simulation {model_key} missing for {ts}"}
    return t_sets[ts][model_key]


def _make_loco_col(size: str) -> dict[str, Any]:
    """Create metadata for single LOCO crack column."""
    return {
        "id": f"mlndt_loco_crack_{size}",
        "group": "mlndt_loco",
        "data_label": "Real data (ML-NDT)",
        "n_physical_flaws": 1,
        "leaky": False,
        "note": f"Leave-one-crack-out fold: {size} mm crack held out",
    }


def _make_real_columns(cross_note: str) -> list[dict[str, Any]]:
    """Create metadata specifications for real ML-NDT and NDT_ML_Flaw columns."""
    cols = [_make_loco_col(s) for s in ("1.6", "4.0", "8.6")]
    cols.append(
        {
            "id": "mlndt_validation",
            "group": "mlndt_validation",
            "data_label": "Real data (ML-NDT)",
            "n_physical_flaws": 3,
            "leaky": True,
            "note": "Official validation set (leaky: shares 3 physical cracks with training)",
        }
    )
    cols.append(
        {
            "id": "cross_ndt_ml_flaw",
            "group": "cross_dataset",
            "data_label": "Real data (NDT_ML_Flaw)",
            "n_physical_flaws": 6,
            "leaky": False,
            "note": cross_note,
        }
    )
    return cols


def _make_sim_columns(sim: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Create metadata specifications for simulation stress columns."""
    test_set_names = list(sim.get("test_sets", {}).keys()) if sim else list(DEFAULT_SIM_SETS)
    cols: list[dict[str, Any]] = []
    for ts in test_set_names:
        grp = "sim_artefact" if ts.startswith("artefact_") else "sim_region"
        cols.append(
            {
                "id": f"sim_{ts}",
                "group": grp,
                "data_label": "Synthetic data",
                "n_physical_flaws": None,
                "leaky": False,
                "note": f"Simulated test condition: {ts}",
            }
        )
    return cols


def _build_column_specs(sim: dict[str, Any] | None, cross_note: str) -> list[dict[str, Any]]:
    """Define metadata for all evaluation columns."""
    return _make_real_columns(cross_note) + _make_sim_columns(sim)


def _eval_sim_union_prop(
    rules_c: dict[str, Any], ml_c: dict[str, Any], u_c: dict[str, Any]
) -> dict[str, Any]:
    """Check union detection monotonicity: union >= max(rules, ml)."""
    cells = (rules_c, ml_c, u_c)
    if any(c.get("available") is False or "detection" not in c for c in cells):
        return {"available": False, "reason": "Component models unavailable"}
    if any("false_calls" not in c for c in cells):
        return {"available": False, "reason": "missing false_calls"}
    r_k, ml_k, u_k = (
        rules_c["detection"]["k"],
        ml_c["detection"]["k"],
        u_c["detection"]["k"],
    )
    r_f, ml_f, u_f = (
        rules_c["false_calls"]["n_false"],
        ml_c["false_calls"]["n_false"],
        u_c["false_calls"]["n_false"],
    )
    viol_det = u_k < max(r_k, ml_k)
    viol_fc = u_f < max(r_f, ml_f)
    n_viol = (1 if viol_det else 0) + (1 if viol_fc else 0)
    return {"holds": n_viol == 0, "n_violations": n_viol}


def _build_row_rules(columns: list[dict[str, Any]], sim: dict[str, Any] | None) -> dict[str, Any]:
    """Build rules engine evaluation row."""
    row: dict[str, Any] = {}
    for col in columns:
        cid = col["id"]
        if cid.startswith("sim_"):
            row[cid] = _get_sim_cell(sim, cid.removeprefix("sim_"), "rules")
        else:
            row[cid] = {"available": False, "reason": RULES_UNAVAIL_REAL}
    return row


def _build_row_fused(columns: list[dict[str, Any]], sim: dict[str, Any] | None) -> dict[str, Any]:
    """Build fused rules-union-ML evaluation row."""
    row: dict[str, Any] = {}
    for col in columns:
        cid = col["id"]
        if cid.startswith("sim_"):
            row[cid] = _get_sim_cell(sim, cid.removeprefix("sim_"), "union")
        else:
            row[cid] = {"available": False, "reason": FUSED_UNAVAIL_REAL}
    return row


def _supervised_cell_for_col(
    col_id: str, ctx: MatrixContext, cross_cell: dict[str, Any]
) -> dict[str, Any]:
    """Retrieve supervised metrics cell for given column without nested conditions."""
    if col_id.startswith("mlndt_loco_crack_"):
        return _get_loco_fold_cell(ctx.loco, col_id.removeprefix("mlndt_loco_"))
    if col_id == "mlndt_validation":
        return _get_loco_val_cell(ctx.loco)
    if col_id == "cross_ndt_ml_flaw":
        return cross_cell
    return _get_sim_cell(ctx.sim, col_id.removeprefix("sim_"), "ml_1d")


def _build_row_supervised(
    columns: list[dict[str, Any]], ctx: MatrixContext, cross_cell: dict[str, Any]
) -> dict[str, Any]:
    """Build supervised model evaluation row."""
    return {col["id"]: _supervised_cell_for_col(col["id"], ctx, cross_cell) for col in columns}


def _anomaly_cell_for_col(col_id: str, ctx: MatrixContext) -> dict[str, Any]:
    """Retrieve anomaly metrics cell for given column without nested conditions."""
    if col_id.startswith("mlndt_loco_crack_"):
        crack_id = col_id.removeprefix("mlndt_loco_")
        anom_real = ctx.anomaly.get("real", {}).get("per_crack", {}) if ctx.anomaly else {}
        return anom_real.get(
            crack_id,
            {"available": False, "reason": f"Anomaly per-crack metrics missing for {crack_id}"},
        )
    if col_id == "mlndt_validation":
        return {
            "available": False,
            "reason": "Anomaly model uses validation split for threshold calibration only",
        }
    if col_id == "cross_ndt_ml_flaw":
        return _get_cross_anomaly(ctx.cross)
    if col_id == "sim_in_distribution":
        syn_c = ctx.anomaly.get("synthetic", {}).get("cell") if ctx.anomaly else None
        return syn_c if syn_c else {"available": False, "reason": "Synthetic anomaly cell missing"}
    return {
        "available": False,
        "reason": "Anomaly model only evaluated on synthetic in-distribution data",
    }


def _build_row_anomaly(columns: list[dict[str, Any]], ctx: MatrixContext) -> dict[str, Any]:
    """Build anomaly detector evaluation row."""
    return {col["id"]: _anomaly_cell_for_col(col["id"], ctx) for col in columns}


def _source_for_col(col_id: str) -> str:
    """Map single column ID to its relative source results file."""
    if col_id.startswith("mlndt_"):
        return "results/loco_metrics.json"
    if col_id == "cross_ndt_ml_flaw":
        return "results/cross_dataset_metrics.json"
    return "results/sim_generalization.json"


def _build_sources(columns: list[dict[str, Any]]) -> dict[str, str]:
    """Map each column ID to its relative source results file."""
    return {col["id"]: _source_for_col(col["id"]) for col in columns}


def _build_union_property(
    columns: list[dict[str, Any]],
    rules_row: dict[str, Any],
    sup_row: dict[str, Any],
    fused_row: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate union property satisfaction across all columns."""
    u_prop: dict[str, Any] = {}
    for col in columns:
        cid = col["id"]
        if cid.startswith("sim_"):
            u_prop[cid] = _eval_sim_union_prop(rules_row[cid], sup_row[cid], fused_row[cid])
        else:
            u_prop[cid] = {"available": False, "reason": FUSED_UNAVAIL_REAL}
    return u_prop


def build_matrix(results_dir: Path) -> dict[str, Any]:
    """Build complete generalization matrix dictionary per shared contract."""
    ctx = MatrixContext(
        loco=_load_optional_json(results_dir / "loco_metrics.json"),
        anomaly=_load_optional_json(results_dir / "anomaly_metrics.json"),
        sim=_load_optional_json(results_dir / "sim_generalization.json"),
        cross=_load_optional_json(results_dir / "cross_dataset_metrics.json"),
    )
    cross_cell, cross_note = _get_cross_supervised(ctx.cross)
    columns = _build_column_specs(ctx.sim, cross_note)

    rules_row = _build_row_rules(columns, ctx.sim)
    fused_row = _build_row_fused(columns, ctx.sim)
    sup_row = _build_row_supervised(columns, ctx, cross_cell)
    anom_row = _build_row_anomaly(columns, ctx)

    return {
        "label_note": LABEL_NOTE,
        "columns": columns,
        "rows": {
            "rules": rules_row,
            "supervised": sup_row,
            "anomaly": anom_row,
            "fused": fused_row,
        },
        "sources": _build_sources(columns),
        "union_property": _build_union_property(columns, rules_row, sup_row, fused_row),
    }


def _cell_rate(row_dict: dict[str, Any], col_id: str) -> float:
    """Safely extract detection rate from row cell or return NaN."""
    cell_data = row_dict.get(col_id, {})
    if not isinstance(cell_data, dict) or cell_data.get("available") is False:
        return np.nan
    return float(cell_data.get("detection", {}).get("rate", np.nan))


def _extract_grid_data(
    matrix: dict[str, Any], row_keys: list[str], cols: list[dict[str, Any]]
) -> np.ndarray:
    """Extract detection rates into 2D float matrix with NaNs for unavailables."""
    return np.array(
        [[_cell_rate(matrix["rows"].get(r, {}), c["id"]) for c in cols] for r in row_keys],
        dtype=float,
    )


def _draw_single_cell_text(ax: plt.Axes, i: int, j: int, val: float) -> None:
    """Render detection rate percentage inside a single heatmap cell."""
    if np.isnan(val):
        return
    txt_color = "white" if val > 0.6 else "black"
    ax.text(
        j,
        i,
        f"{val * 100.0:.1f}%",
        ha="center",
        va="center",
        color=txt_color,
        fontsize=9,
        fontweight="bold",
    )


def _annotate_heatmap(ax: plt.Axes, grid: np.ndarray) -> None:
    """Draw detection rate percentage labels inside valid heatmap cells."""
    for i, j in np.ndindex(grid.shape):
        _draw_single_cell_text(ax, i, j, grid[i, j])


def plot_matrix(matrix: dict[str, Any], out_png: Path) -> Path:
    """Plot detection rate heatmap across models and evaluation conditions."""
    row_keys = ["rules", "supervised", "anomaly", "fused"]
    row_labels = ["Rules", "Supervised", "Anomaly", "Fused"]
    cols = matrix["columns"]
    grid = _extract_grid_data(matrix, row_keys, cols)

    fig, ax = plt.subplots(figsize=(max(12, len(cols) * 0.95), 4.5))
    cmap = plt.colormaps["Blues"].with_extremes(bad="#dcdcdc")
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    _annotate_heatmap(ax, grid)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(
        [f"{c['id']}\n[{c['data_label']}]" for c in cols], rotation=35, ha="right", fontsize=8
    )
    ax.set_yticks(range(len(row_keys)))
    ax.set_yticklabels(row_labels, fontsize=10, fontweight="bold")
    ax.set_title("Generalization Matrix — Detection Rate Across Domains", fontsize=12, pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Detection Rate", fontsize=9)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png
