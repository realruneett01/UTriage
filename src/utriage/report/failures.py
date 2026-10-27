"""Model failure mode classification and failure case reporting (project plan Phase 8).

Categorises LOCO detection misses and false calls using measurable rules and
analysis heuristics, and persists failure case artifacts (plots and metadata).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Analysis heuristics (not physics constants)
NEAR_EDGE_MARGIN_PIXELS: float = 25.0
NEAR_WALL_MARGIN_PIXELS: float = NEAR_EDGE_MARGIN_PIXELS
IMAGE_DEPTH_PIXELS: float = 256.0
LOW_SNR_RATIO: float = 2.0
SMALL_FLAW_P10_DEFAULT: float = 1.5


def _extract_float(val: Any) -> float | None:
    """Safely convert value to float or return None."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):  # narrow reason: non-numeric metadata representation
        return None


def compute_image_snr(img: np.ndarray) -> float:
    """Compute image SNR as image maximum divided by robust noise estimate.

    Calculates SNR as image max divided by the median absolute deviation (MAD)
    of the image values. If MAD is zero, returns 0.0 for non-positive peak or
    float('inf') for positive peak.
    Note: this is an image-space analysis heuristic.
    """
    arr = np.asarray(img, dtype=float)
    if arr.ndim > 2:
        arr = arr.squeeze()
    if arr.size == 0:
        return 0.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    peak = float(np.max(arr))
    if mad <= 0.0:
        return 0.0 if peak <= 0.0 else float("inf")
    return peak / mad


def is_near_edge(img: np.ndarray, margin: float = NEAR_EDGE_MARGIN_PIXELS) -> bool:
    """Check if the image peak row is within margin of the first or last row.

    Finds the row (depth axis) of the image maximum inside the image itself and
    evaluates whether it falls within the specified margin of the first (0) or
    last row.
    Note: this is an image-space heuristic.
    """
    arr = np.asarray(img)
    if arr.ndim > 2:
        arr = arr.squeeze()
    if arr.size == 0 or arr.ndim < 2:
        return False
    r_max = int(np.unravel_index(np.argmax(arr), arr.shape)[0])
    last_row = arr.shape[0] - 1
    return any((r_max <= margin, r_max >= (last_row - margin)))


def _is_near_edge(
    meta: dict,
    img: np.ndarray | None = None,
    margin: float = NEAR_EDGE_MARGIN_PIXELS,
) -> bool:
    """Check if flaw peak is within margin of the depth axis edges (image-space heuristic)."""
    if img is not None:
        return is_near_edge(img, margin)
    if meta.get("near_edge") is True or meta.get("near_wall") is True:
        return True
    row = _extract_float(meta.get("peak_row"))
    if row is None:
        row = _extract_float(meta.get("row"))
    if row is not None:
        depth_len = _extract_float(meta.get("image_depth")) or IMAGE_DEPTH_PIXELS
        return any((row <= margin, row >= (depth_len - 1.0 - margin)))
    return False


def _is_low_snr(
    meta: dict,
    img: np.ndarray | None = None,
    ratio_threshold: float = LOW_SNR_RATIO,
) -> bool:
    """Check if image SNR is below LOW_SNR_RATIO heuristic constant."""
    if img is not None:
        return compute_image_snr(img) < ratio_threshold
    snr = _extract_float(meta.get("snr"))
    if snr is None:
        snr = _extract_float(meta.get("image_snr"))
    if snr is not None:
        return snr < ratio_threshold
    return False


def _is_small_flaw(meta: dict, p10_threshold: float) -> bool:
    """Check if equivalent flaw size is below fold detected-size p10 threshold."""
    size = _extract_float(meta.get("equivalent_flawsize"))
    if size is None:
        size = _extract_float(meta.get("size"))
    if size is None or size <= 0.0:
        return False
    return size < p10_threshold


def _classify_miss(
    meta: dict,
    p10_thresh: float,
    img: np.ndarray | None = None,
) -> str:
    """Classify false negative (miss) failure cause in priority order."""
    if _is_near_edge(meta, img):
        return "near_edge"
    if _is_low_snr(meta, img):
        return "low_snr"
    if _is_small_flaw(meta, p10_thresh):
        return "small_flaw"
    return "unclassified"


def _classify_false_call(meta: dict, img: np.ndarray | None = None) -> str:
    """Classify false positive (false call) failure cause."""
    img_max = _extract_float(meta.get("image_max"))
    if img_max is None and img is not None and getattr(img, "size", 0) > 0:
        img_max = float(np.max(img))
    if img_max is None:
        img_max = _extract_float(meta.get("max_amplitude"))
    thresh = _extract_float(meta.get("noise_threshold"))
    if img_max is not None and thresh is not None and img_max > thresh:
        return "artefact_like"
    return "unclassified"


def classify_failure(
    meta: dict,
    kind: str,
    p10_threshold: float = SMALL_FLAW_P10_DEFAULT,
    img: np.ndarray | None = None,
) -> str:
    """Classify failure mode from measurable evidence only.

    All thresholds are analysis heuristics, not physics constants.

    Miss rules (kind='miss'):
    - near_edge: peak row within NEAR_EDGE_MARGIN_PIXELS (25 px) of depth edges
      (image-space heuristic)
    - low_snr: image max / MAD noise < LOW_SNR_RATIO (2.0)
    - small_flaw: equivalent_flawsize < fold's detected-size p10
    - else: unclassified

    False call rules (kind='false_call'):
    - artefact_like: image max amplitude > noise_threshold
    - else: unclassified
    """
    k = kind.strip().lower()
    target_img = img if img is not None else meta.get("image")
    if target_img is None and isinstance(meta.get("img"), np.ndarray):
        target_img = meta.get("img")
    if k in ("false_call", "false_alarm", "fp"):
        return _classify_false_call(meta, target_img)
    if k in ("miss", "missed", "fn"):
        thresh = _extract_float(meta.get("p10_threshold")) or p10_threshold
        return _classify_miss(meta, thresh, target_img)
    return "unclassified"


def _plot_failure_case(
    img: np.ndarray,
    png_path: Path,
    title: str,
    c_max: int,
) -> None:
    """Render and save 2D B-scan and A-scan column figure."""
    r_max = int(np.argmax(img[:, c_max]))
    fig, (ax_img, ax_scan) = plt.subplots(1, 2, figsize=(10, 4.5))
    im = ax_img.imshow(img, cmap="viridis", aspect="auto")
    ax_img.axvline(x=c_max, color="red", linestyle="--", alpha=0.7)
    ax_img.scatter([c_max], [r_max], color="red", marker="x", s=40)
    ax_img.set_title(title, fontsize=10)
    ax_img.set_xlabel("Scan column")
    ax_img.set_ylabel("Depth sample")
    fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

    ax_scan.plot(img[:, c_max], color="navy", lw=1.2)
    ax_scan.scatter([r_max], [img[r_max, c_max]], color="red", s=40, zorder=3)
    ax_scan.set_title(f"A-scan column {c_max} (max={img[r_max, c_max]:.1f})", fontsize=10)
    ax_scan.set_xlabel("Depth sample")
    ax_scan.set_ylabel("Amplitude")
    ax_scan.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def save_failure_case(
    img: np.ndarray,
    meta: dict,
    score: float,
    out_dir: Path | str,
    name: str = "",
    save_png: bool = True,
) -> Path:
    """Save failure case PNG plot (optional) and evidence JSON to output directory."""
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(img)
    if arr.ndim > 2:
        arr = arr.squeeze()

    r_max, c_max = np.unravel_index(np.argmax(arr), arr.shape)
    png_path = target_dir / f"{name}.png"
    if save_png:
        _plot_failure_case(arr, png_path, f"{name} (score={score:.3f})", int(c_max))

    snr_val = compute_image_snr(arr)
    evidence = {
        "name": name,
        "score": float(score),
        "peak_amplitude": float(arr[r_max, c_max]),
        "peak_location": [int(r_max), int(c_max)],
        "snr": float(snr_val) if np.isfinite(snr_val) else None,
        "near_edge": bool(is_near_edge(arr)),
        "png_saved": bool(save_png),
        "meta": {k: str(v) for k, v in meta.items()},
    }
    json_path = target_dir / f"{name}.json"
    json_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return png_path if save_png else json_path


def _top_cause(causes: dict[str, int], total: int) -> tuple[str, int, float]:
    """Find dominant cause, its count, and percentage share."""
    if not causes or total <= 0:
        return "none", 0, 0.0
    top_name, top_cnt = max(causes.items(), key=lambda item: item[1])
    return top_name, top_cnt, (top_cnt / total) * 100.0


def _format_fold_summary(s: dict) -> str:
    """Format count-driven summary bullet for a single fold."""
    fold = s.get("fold", "unknown")
    n_m = s.get("n_misses", 0)
    m_cause, m_cnt, m_pct = _top_cause(s.get("miss_causes", {}), n_m)
    n_fc = s.get("n_false_calls", 0)
    fc_cause, fc_cnt, fc_pct = _top_cause(s.get("false_call_causes", {}), n_fc)
    return (
        f"- **Fold {fold}**: {n_m} misses (top cause: {m_cause} "
        f"with {m_cnt} cases, {m_pct:.1f}%), {n_fc} false calls "
        f"(top cause: {fc_cause} with {fc_cnt} cases, {fc_pct:.1f}%)."
    )


def _format_synthetic_summary(sim_gen: dict) -> str:
    """Format synthetic generalization summary from computed rates."""
    tsets = sim_gen.get("test_sets", {})
    if not tsets:
        return "- **Synthetic Shift Summary**: No synthetic evaluation data available."
    id_fcr = (
        tsets.get("in_distribution", {}).get("ml_1d", {}).get("false_calls", {}).get("rate", 0.0)
    )
    f10_fcr = tsets.get("freq_10mhz", {}).get("ml_1d", {}).get("false_calls", {}).get("rate", 0.0)
    noise_fcr = (
        tsets.get("noise_above_range", {}).get("ml_1d", {}).get("false_calls", {}).get("rate", 0.0)
    )
    return (
        f"- **Synthetic Shift Summary**: ML 1D false-call rate shifts from "
        f"{id_fcr * 100:.1f}% in-distribution to {f10_fcr * 100:.1f}% at 10 MHz "
        f"and {noise_fcr * 100:.1f}% under elevated noise."
    )


def _format_summary_section(
    fold_summaries: list[dict],
    sim_gen: dict,
    png_count: int,
) -> str:
    """Construct plain-language summary strictly from counts and computed rates."""
    lines = [
        "## Plain-Language Summary\n",
        "Evaluation summary derived strictly from measurable counts and evaluation metrics:\n",
    ]
    for s in fold_summaries:
        lines.append(_format_fold_summary(s))
    lines.append(_format_synthetic_summary(sim_gen))
    lines.append(
        f"- **Artifact Storage**: {png_count} PNG plots saved; "
        "full JSON evidence retained for all evaluated failure cases."
    )
    return "\n".join(lines)


def _format_counts_table(fold_summaries: list[dict]) -> str:
    """Format markdown table for per-fold failure counts by cause."""
    lines = [
        "## LOCO Folds Failure Breakdown\n",
        "| Fold | Misses | Near Edge | Low SNR | Small Flaw | Miss Unclass "
        "| False Calls | Artefact-like | FC Unclass |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in fold_summaries:
        m_causes = s.get("miss_causes", {})
        fc_causes = s.get("false_call_causes", {})
        edge_cnt = m_causes.get("near_edge", m_causes.get("near_wall", 0))
        lines.append(
            f"| {s['fold']} | {s['n_misses']} "
            f"| {edge_cnt} "
            f"| {m_causes.get('low_snr', 0)} "
            f"| {m_causes.get('small_flaw', 0)} "
            f"| {m_causes.get('unclassified', 0)} "
            f"| {s['n_false_calls']} "
            f"| {fc_causes.get('artefact_like', 0)} "
            f"| {fc_causes.get('unclassified', 0)} |"
        )
    return "\n".join(lines)


def _format_cases_table(cases: list[dict]) -> str:
    """Format markdown table of individual failure cases."""
    plotted = [c for c in cases if c.get("has_png", True)]
    display_cases = plotted if plotted else cases
    lines = [
        "## Evaluated Failure Cases\n",
        "| Case ID | Fold | Kind | Score | Cause | Evidence |",
        "|---|---|---|---|---|---|",
    ]
    for c in display_cases:
        ev = str(c.get("evidence", "-")).replace("|", "/")
        lines.append(
            f"| {c['name']} | {c['fold']} | {c['kind']} | {c['score']:.3f} | {c['cause']} | {ev} |"
        )
    return "\n".join(lines)


def _format_synthetic_notes(tsets: dict) -> list[str]:
    """Format count-driven notes for synthetic parameter shifts."""
    lines = [""]
    if "freq_10mhz" in tsets:
        dat = tsets["freq_10mhz"]
        ml_fc = dat.get("ml_1d", {}).get("false_calls", {})
        u_fcr = dat.get("union", {}).get("false_calls", {}).get("rate", 0.0)
        ml_rate = ml_fc.get("rate", 0.0) * 100.0
        n_false, n_scans = ml_fc.get("n_false", 0), ml_fc.get("n_scans", 0)
        lines.append(
            f"- **10 MHz Frequency Shift**: ML false-call rate = {ml_rate:.1f}% "
            f"({n_false}/{n_scans} scans), union FCR = {u_fcr * 100:.1f}%."
        )
    if "noise_above_range" in tsets:
        dat = tsets["noise_above_range"]
        ml_fc = dat.get("ml_1d", {}).get("false_calls", {})
        u_fcr = dat.get("union", {}).get("false_calls", {}).get("rate", 0.0)
        ml_rate = ml_fc.get("rate", 0.0) * 100.0
        n_false, n_scans = ml_fc.get("n_false", 0), ml_fc.get("n_scans", 0)
        lines.append(
            f"- **Noise Above Range**: ML false-call rate = {ml_rate:.1f}% "
            f"({n_false}/{n_scans} scans), union FCR = {u_fcr * 100:.1f}%."
        )
    return lines


def _format_synthetic_section(sim_gen: dict) -> str:
    """Format markdown section describing synthetic held-out region behavior."""
    tsets = sim_gen.get("test_sets", {})
    lines = [
        "## Synthetic Held-Out Regions Generalization\n",
        "Evaluation results extracted from `results/sim_generalization.json` demonstrate "
        "false-call blow-up under parameter shifts without re-running simulation:\n",
        "| Region | Rules FCR | ML 1D FCR | Union FCR | ML False / Total Scans |",
        "|---|---|---|---|---|",
    ]
    for key in ("in_distribution", "freq_10mhz", "material_third", "noise_above_range"):
        if key not in tsets:
            continue
        dat = tsets[key]
        r_fcr = dat.get("rules", {}).get("false_calls", {}).get("rate", 0.0)
        ml_fc = dat.get("ml_1d", {}).get("false_calls", {})
        u_fcr = dat.get("union", {}).get("false_calls", {}).get("rate", 0.0)
        lines.append(
            f"| {key} | {r_fcr * 100:.2f}% | {ml_fc.get('rate', 0.0) * 100:.2f}% "
            f"| {u_fcr * 100:.2f}% | {ml_fc.get('n_false', 0)} / {ml_fc.get('n_scans', 0)} |"
        )
    lines.extend(_format_synthetic_notes(tsets))
    return "\n".join(lines)


def build_failure_modes_md(
    fold_summaries: list[dict],
    cases: list[dict],
    sim_gen_data: dict | None = None,
    png_count: int | None = None,
) -> str:
    """Construct complete markdown report for failure modes and generalization."""
    sim_data = sim_gen_data or {}
    plotted = [c for c in cases if c.get("has_png", True)]
    n_pngs = len(plotted) if png_count is None else png_count
    sections = [
        "# Failure Modes and Generalization Boundaries\n",
        _format_summary_section(fold_summaries, sim_data, n_pngs),
        _format_counts_table(fold_summaries),
        _format_cases_table(cases),
        _format_synthetic_section(sim_data),
    ]
    return "\n\n".join(sections) + "\n"
