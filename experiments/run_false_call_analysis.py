"""LOCO false call analysis, cause classification, and trade-off curves (ADR 0010).

Re-scores clean test and calibration images using pre-registered detector thresholds,
classifies false call causes, produces 6x3 diagnostic PNGs, and plots trade-off curves.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from utriage.detector_config import (  # noqa: E402
    DetectorConfig,
    config_fingerprint,
    load_detector_config,
)
from utriage.ml.anomaly import ConvAE2D, _smooth_error, anomaly_score  # noqa: E402
from utriage.ml.datasets import (  # noqa: E402
    MLNDT_ROOT,
    list_batches,
    read_image,
    standardise,
)
from utriage.ml.infer import load_model  # noqa: E402
from utriage.ml.loco import IndexedMlNdtDataset, LocoFold, make_folds  # noqa: E402
from utriage.ml.mlndt_meta import image_index  # noqa: E402
from utriage.ml.physics_features import score_physics_classifier  # noqa: E402
from utriage.ml.train import TrainConfig, pick_device, predict, set_seed  # noqa: E402
from utriage.report.failures import compute_image_snr, is_near_edge  # noqa: E402
from utriage.report.false_calls import (  # noqa: E402
    build_agreement_table,
    classify_false_call_cause,
    compute_tradeoff_curve,
    is_near_miss_supervised,
    plot_flagged_image,
    plot_tradeoff_curve,
    save_false_call_analysis,
)
from utriage.triage.hybrid import ecdf, fused_score, quantile_threshold  # noqa: E402


@dataclass(frozen=True)
class FalseCallConfig:
    """Configuration options for false call analysis and trade-off curve generation."""

    root: Path = MLNDT_ROOT
    models_dir: Path = Path("models")
    anomaly_metrics_path: Path = Path("results/anomaly_metrics.json")
    out_json: Path = Path("results/false_call_analysis.json")
    out_md: Path = Path("results/false_call_analysis.md")
    figures_dir: Path = Path("results/figures")
    cases_dir: Path = Path("results/false_calls")
    target_fcr: float = 0.01
    device: str = "auto"
    quick: bool = False
    seed: int = 0
    config_path: Path | None = None
    overrides: tuple[str, ...] | list[str] | None = None
    detector_config: DetectorConfig | None = None


@dataclass(frozen=True)
class CalibContext:
    """Clean calibration items, anomaly probabilities, and empirical CDF."""

    items: list[tuple[Path, int]]
    p_anom_calib: np.ndarray
    ecdf_fn: Callable[[np.ndarray | float], np.ndarray]
    target_fcr: float


@dataclass(frozen=True)
class CalibSetup:
    """Inputs required for calibration context creation."""

    anom_model: nn.Module
    dev: torch.device
    target_fcr: float
    quick: bool
    metrics_path: Path


@dataclass(frozen=True)
class ThresholdConfig:
    """Pre-registered and configured detector threshold parameters."""

    supervised: float = 0.5
    anomaly_fcr: float = 0.01
    h2_fcr: float = 0.01
    physics: float = 0.5


@dataclass(frozen=True)
class CleanInspectionContext:
    """Artifact saving context for clean image inspection."""

    anom_model: nn.Module
    dev: torch.device
    cases_dir: Path


PROTOCOL_NOTE: str = (
    "Scoring protocol: Clean scans are scored with every LOCO fold model. "
    "Trade-off curves and false-call counts are computed per fold (each fold evaluates "
    "its own clean scores against its own held-out flawed scores). Totals and pooled summaries "
    "pool (fold, scan) pairs across all folds at pre-registered operating thresholds."
)


@dataclass(frozen=True)
class FoldScoringContext:
    """Context and models needed for fold scoring."""

    cal_ctx: CalibContext
    anom_model: nn.Module
    dev: torch.device
    cfg: FalseCallConfig
    thresholds: ThresholdConfig
    t_anom: float


@dataclass(frozen=True)
class FoldResult:
    """Evaluation results for a single LOCO fold."""

    held_out: str
    clean_items: list[tuple[Path, int]]
    flawed_items: list[tuple[Path, int]]
    clean_scores: dict[str, np.ndarray]
    flawed_scores: dict[str, np.ndarray]
    thresholds: dict[str, float]
    t_s: float


@dataclass(frozen=True)
class PooledMetrics:
    """Aggregated metrics across all evaluated LOCO folds."""

    all_flagged: list[dict[str, Any]]
    total_clean_pairs: int
    causes_cnt: dict[str, int]
    agr_table: dict[str, Any]


@dataclass(frozen=True)
class PayloadContext:
    """Context metadata for final payload assembly."""

    per_fold: dict[str, dict[str, Any]]
    pooled_curves: dict[str, Any]
    pooled_ops: dict[str, Any]
    threshs: dict[str, float]
    fp: dict[str, Any]
    run_label: str


def _repo_parent_root() -> Path | None:
    """Locate root directory of parent repository across worktrees."""
    try:
        cmd = ["git", "rev-parse", "--git-common-dir"]
        common = subprocess.check_output(  # noqa: S603
            cmd, text=True, stderr=subprocess.DEVNULL
        ).strip()
        return Path(common).resolve().parent
    except (subprocess.CalledProcessError, FileNotFoundError):  # git query unavailable
        return None


def _find_data_root(provided: Path) -> Path:
    """Locate ML-NDT dataset root across standalone clone or git worktree."""
    if provided.exists():
        return provided
    parent = _repo_parent_root()
    if parent is not None:
        cand = parent / "data" / "external" / "ML-NDT" / "data"
        if cand.exists():
            return cand
    return provided


def _resolve_model_path(filename: str, models_dir: Path) -> Path:
    """Locate model checkpoint in models_dir or parent fallback directory."""
    local_p = models_dir / filename
    if local_p.exists() or models_dir.resolve() != Path("models").resolve():
        return local_p
    parent = _repo_parent_root()
    if parent is not None:
        fallback = parent / "models" / filename
        if fallback.exists():
            return fallback
    fallback = MLNDT_ROOT.parents[3] / "models" / filename
    return fallback if fallback.exists() else local_p


def _load_anomaly_model(path: Path, dev: torch.device) -> ConvAE2D:
    """Load ConvAE2D model from checkpoint onto target compute device."""
    model = ConvAE2D()
    loaded = torch.load(path, map_location=dev, weights_only=True)
    state = loaded["state_dict"] if isinstance(loaded, dict) and "state_dict" in loaded else loaded
    model.load_state_dict(state)
    model.to(dev)
    model.eval()
    return model


def _resolve_detector_config(cfg: FalseCallConfig) -> DetectorConfig:
    """Resolve DetectorConfig from cfg instance, config file path, or overrides."""
    if cfg.detector_config is not None:
        return cfg.detector_config
    overrides_list = list(cfg.overrides) if cfg.overrides is not None else []
    if cfg.target_fcr != 0.01 and not any("target_fcr" in o for o in overrides_list):
        overrides_list.append(f"anomaly.target_fcr={cfg.target_fcr}")
    return load_detector_config(
        path=cfg.config_path, overrides=overrides_list if overrides_list else None
    )


def _load_threshold_config(det_cfg: DetectorConfig) -> ThresholdConfig:
    """Load detector thresholds from DetectorConfig object."""
    return ThresholdConfig(
        supervised=float(det_cfg.supervised.threshold),
        anomaly_fcr=float(det_cfg.anomaly.target_fcr),
        h2_fcr=float(det_cfg.hybrid.fused_target_fcr),
        physics=float(det_cfg.physics.threshold),
    )


def _read_metric_threshold(path: Path, target_fcr: float) -> float | None:
    """Read threshold from metrics file if it exists and matches target FCR."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        real_block = data.get("real", {})
        file_fcr = float(real_block.get("target_fcr", 0.01))
        match = "threshold" in real_block and abs(file_fcr - target_fcr) < 1e-6
        return float(real_block["threshold"]) if match else None
    except (json.JSONDecodeError, KeyError, OSError):  # corrupted or invalid metrics json file
        return None


def _load_or_fit_t_anom(
    path: Path,
    cal_scores: np.ndarray,
    target_fcr: float,
) -> float:
    """Load real threshold from anomaly_metrics.json or fit on calibration scores."""
    if path.exists():
        val = _read_metric_threshold(path, target_fcr)
        if val is not None:
            return val
    return float(quantile_threshold(cal_scores, target_fcr))


def _get_calibration_items(val_batches: list[Path], quick: bool) -> list[tuple[Path, int]]:
    """Extract flaw-free validation items for calibration."""
    val_idx = image_index(val_batches)
    clean = [(b, idx) for b, idx, crack in val_idx if crack is None]
    return clean[:32] if quick else clean


def _prepare_test_items(
    test_items: list[tuple[Path, int]],
    quick: bool,
) -> list[tuple[Path, int]]:
    """Cap test items if quick mode is active, preserving flawed and unflawed."""
    if not quick:
        return test_items
    ds = IndexedMlNdtDataset(test_items)
    y_test, _ = ds.get_labels()
    flaw_idx = np.where(y_test == 1.0)[0][:32]
    clean_idx = np.where(y_test == 0.0)[0][:32]
    keep = np.sort(np.concatenate([flaw_idx, clean_idx]))
    return [test_items[i] for i in keep]


@torch.no_grad()
def _score_anomaly(
    model: nn.Module,
    items: list[tuple[Path, int]],
    dev: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute ConvAE2D anomaly scores for indexed items."""
    if not items:
        return np.empty(0, dtype=np.float32)
    loader = DataLoader(IndexedMlNdtDataset(items), batch_size=batch_size, shuffle=False)
    scores: list[float] = []
    for batch in loader:
        x = batch[0].to(dev)
        s = anomaly_score(model, x)
        scores.extend(s.detach().cpu().numpy().tolist())
    return np.array(scores, dtype=np.float32)


def _score_supervised(
    model: nn.Module,
    items: list[tuple[Path, int]],
    dev: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute flaw probabilities from supervised MlNdtNet model."""
    if not items:
        return np.empty(0, dtype=np.float32)
    ds = IndexedMlNdtDataset(items)
    cfg = TrainConfig(device=str(dev), batch_size=batch_size)
    probs, _ = predict(model, ds, cfg)
    return probs


def _build_calib_context(
    val_batches: list[Path],
    setup: CalibSetup,
) -> tuple[CalibContext, float]:
    """Prepare clean calibration items, anomaly scores, and threshold."""
    cal_items = _get_calibration_items(val_batches, setup.quick)
    a_score_calib = _score_anomaly(setup.anom_model, cal_items, setup.dev)
    t_anom = _load_or_fit_t_anom(setup.metrics_path, a_score_calib, setup.target_fcr)
    ecdf_fn = ecdf(a_score_calib)
    p_anom_calib = ecdf_fn(a_score_calib)
    ctx = CalibContext(
        items=cal_items,
        p_anom_calib=p_anom_calib,
        ecdf_fn=ecdf_fn,
        target_fcr=setup.target_fcr,
    )
    return ctx, t_anom


def _compute_reconstruction_error(
    model: nn.Module,
    img: np.ndarray,
    dev: torch.device,
) -> tuple[np.ndarray, int, int]:
    """Compute 2D smoothed error map and find peak row and column coordinates."""
    img_std = standardise(img)
    x = torch.from_numpy(img_std).unsqueeze(0).unsqueeze(0).to(dev)
    with torch.no_grad():
        recon = model(x)
        sq_err = (x - recon) ** 2
        smoothed = _smooth_error(sq_err)
        err_map = smoothed.squeeze().detach().cpu().numpy()
    r_peak, c_peak = np.unravel_index(np.argmax(err_map), err_map.shape)
    return err_map, int(r_peak), int(c_peak)


def _score_physics_classifier(
    fold: LocoFold,
    items: list[tuple[Path, int]],
) -> np.ndarray:
    """Train logistic regression on fold training features and score items."""
    return score_physics_classifier(fold, items)


def _diagnose_image(
    ctx: CleanInspectionContext,
    bins_path: Path,
    idx: int,
) -> dict[str, Any]:
    """Read image, compute SNR and error map, and save PNG artifact."""
    img = read_image(bins_path, idx)
    snr, edge = float(compute_image_snr(img)), bool(is_near_edge(img))
    r_p, c_p = np.unravel_index(np.argmax(img), img.shape)
    err_map, a_r, a_c = _compute_reconstruction_error(ctx.anom_model, img, ctx.dev)
    png = ctx.cases_dir / f"{bins_path.stem}_{idx}.png"
    plot_flagged_image(img, err_map, png, title=f"{bins_path.stem}_{idx}")
    return {
        "image_snr": snr,
        "near_edge": edge,
        "peak_row": int(r_p),
        "peak_col": int(c_p),
        "anomaly_peak_row": a_r,
        "anomaly_peak_col": a_c,
        "png_path": str(png),
    }


def _format_clean_metadata(
    bins_path: Path,
    idx: int,
    flagged: list[str],
    cause: str,
) -> dict[str, Any]:
    """Format categorical and indexing attributes for a clean record."""
    return {
        "batch_stem": bins_path.stem,
        "image_idx": int(idx),
        "flagged_detectors": flagged,
        "agreement_count": len(flagged),
        "cause": cause,
    }


def _build_clean_record(
    item: tuple[Path, int],
    scores: dict[str, float],
    threshs: dict[str, float],
    ctx: CleanInspectionContext,
) -> dict[str, Any]:
    """Inspect and save diagnostic artifacts for a flagged clean test image."""
    bins_path, idx = item
    flagged = [d for d in ("supervised", "anomaly", "h2", "h3") if scores[d] >= threshs[d]]
    rec = _diagnose_image(ctx, bins_path, idx)
    ev = {
        "near_edge": rec["near_edge"],
        "image_snr": rec["image_snr"],
        "agreement_count": len(flagged),
        "scores": scores,
        "thresholds": threshs,
    }
    rec.update(_format_clean_metadata(bins_path, idx, flagged, classify_false_call_cause(ev)))
    rec["scores"], rec["thresholds"] = scores, threshs
    rec["margins"] = {d: float(scores[d] - threshs[d]) for d in scores}
    rec["is_near_miss_supervised"] = is_near_miss_supervised(
        scores["supervised"], threshs["supervised"]
    )
    return rec


def _evaluate_clean_images(
    fold_res: FoldResult,
    ctx: CleanInspectionContext,
) -> list[dict[str, Any]]:
    """Identify and record all clean images flagged by any detector in fold."""
    flagged_records: list[dict[str, Any]] = []
    threshs = fold_res.thresholds
    for i, item in enumerate(fold_res.clean_items):
        scores = {
            d: float(fold_res.clean_scores[d][i]) for d in ("supervised", "anomaly", "h2", "h3")
        }
        is_hit = any(scores[d] >= threshs[d] for d in scores)
        if is_hit:
            rec = _build_clean_record(item, scores, threshs, ctx)
            rec["fold"] = fold_res.held_out
            flagged_records.append(rec)
    return flagged_records


def _compute_fold_curves(fold_res: FoldResult) -> dict[str, Any]:
    """Compute detection vs false call curves for a single fold."""
    curves: dict[str, Any] = {}
    for det in ("supervised", "anomaly", "h2", "h3"):
        f_arr = np.array(fold_res.flawed_scores[det], dtype=np.float32)
        c_arr = np.array(fold_res.clean_scores[det], dtype=np.float32)
        y_true = np.concatenate([np.ones(len(f_arr), dtype=int), np.zeros(len(c_arr), dtype=int)])
        all_s = np.concatenate([f_arr, c_arr])
        curve = compute_tradeoff_curve(y_true, all_s, operating_threshold=fold_res.thresholds[det])
        curve["operating_point"]["flaw_hits"] = int(np.sum(f_arr >= fold_res.thresholds[det]))
        curve["operating_point"]["clean_hits"] = int(np.sum(c_arr >= fold_res.thresholds[det]))
        curves[det] = curve
    return curves


def _assemble_fold_entry(
    fold_res: FoldResult,
    flagged: list[dict[str, Any]],
    curves: dict[str, Any],
) -> dict[str, Any]:
    """Assemble dictionary of metrics, curves, and causes for one fold."""
    causes_cnt: dict[str, int] = {}
    for rec in flagged:
        c = rec["cause"]
        causes_cnt[c] = causes_cnt.get(c, 0) + 1
    agr_table = build_agreement_table(flagged)
    agr_table["total_pairs_counted"] = len(fold_res.clean_items)
    agr_table["scan_pairs_evaluated"] = len(fold_res.clean_items)
    return {
        "held_out": fold_res.held_out,
        "n_clean": len(fold_res.clean_items),
        "n_flawed": len(fold_res.flawed_items),
        "n_flagged": len(flagged),
        "thresholds": fold_res.thresholds,
        "cause_counts": causes_cnt,
        "agreement_table": agr_table,
        "tradeoff_curves": curves,
        "operating_points": {d: curves[d]["operating_point"] for d in curves},
        "flagged_cases": flagged,
        "clean_scores": {
            d: [float(s) for s in fold_res.clean_scores[d]] for d in fold_res.clean_scores
        },
    }


def _calibrate_fold_h2(sup_model: nn.Module, ctx: FoldScoringContext) -> float:
    """Calibrate h2 fused score threshold on validation items."""
    p_sup_cal = _score_supervised(sup_model, ctx.cal_ctx.items, ctx.dev)
    s_calib = fused_score(p_sup_cal, ctx.cal_ctx.p_anom_calib)
    return float(quantile_threshold(s_calib, ctx.thresholds.h2_fcr))


def _score_fold_detectors(
    fold: LocoFold,
    sup_model: nn.Module,
    items: list[tuple[Path, int]],
    ctx: FoldScoringContext,
) -> dict[str, np.ndarray]:
    """Score test items using supervised, anomaly, h2, and physics detectors."""
    a_test = _score_anomaly(ctx.anom_model, items, ctx.dev)
    p_sup = _score_supervised(sup_model, items, ctx.dev)
    s_test = fused_score(p_sup, ctx.cal_ctx.ecdf_fn(a_test))
    p_phys = _score_physics_classifier(fold, items)
    return {"supervised": p_sup, "anomaly": a_test, "h2": s_test, "h3": p_phys}


def _run_fold_scoring(
    fold: LocoFold,
    ctx: FoldScoringContext,
) -> FoldResult:
    """Evaluate detector candidates on fold test partition."""
    weights_path = _resolve_model_path(f"loco_{fold.held_out}.pt", ctx.cfg.models_dir)
    sup_model, _ = load_model(weights_path, device=str(ctx.dev))
    t_s = _calibrate_fold_h2(sup_model, ctx)

    test_items = _prepare_test_items(fold.test, ctx.cfg.quick)
    ds = IndexedMlNdtDataset(test_items)
    y_test = ds.get_labels()[0] if test_items else np.empty(0)
    scores = _score_fold_detectors(fold, sup_model, test_items, ctx)

    flaw_idx, clean_idx = np.where(y_test == 1.0)[0], np.where(y_test == 0.0)[0]
    threshs = {
        "supervised": ctx.thresholds.supervised,
        "anomaly": ctx.t_anom,
        "h2": t_s,
        "h3": ctx.thresholds.physics,
    }
    return FoldResult(
        held_out=fold.held_out,
        clean_items=[test_items[i] for i in clean_idx],
        flawed_items=[test_items[i] for i in flaw_idx],
        clean_scores={d: scores[d][clean_idx] for d in scores},
        flawed_scores={d: scores[d][flaw_idx] for d in scores},
        thresholds=threshs,
        t_s=t_s,
    )


def _evaluate_all_folds(
    folds: list[LocoFold],
    ctx: FoldScoringContext,
) -> tuple[dict[str, FoldResult], float]:
    """Score all LOCO folds with their respective models."""
    results: dict[str, FoldResult] = {}
    t_s_folds: list[float] = []
    for fold in folds:
        res = _run_fold_scoring(fold, ctx)
        results[fold.held_out] = res
        t_s_folds.append(res.t_s)
    mean_ts = float(np.mean(t_s_folds)) if t_s_folds else 0.5
    return results, mean_ts


def _process_per_fold_results(
    fold_results: dict[str, FoldResult],
    insp_ctx: CleanInspectionContext,
) -> dict[str, dict[str, Any]]:
    """Evaluate clean false calls and tradeoff curves for each fold."""
    per_fold: dict[str, dict[str, Any]] = {}
    for fold_name, fold_res in fold_results.items():
        flagged = _evaluate_clean_images(fold_res, insp_ctx)
        curves = _compute_fold_curves(fold_res)
        per_fold[fold_name] = _assemble_fold_entry(fold_res, flagged, curves)
    return per_fold


def _pool_operating_points(
    fold_entries: dict[str, dict[str, Any]],
    threshs: dict[str, float],
) -> dict[str, Any]:
    """Pool operating points across folds and compute micro and macro averages."""
    detectors = ("supervised", "anomaly", "h2", "h3")
    total_flawed = sum(e["n_flawed"] for e in fold_entries.values())
    total_clean = sum(e["n_clean"] for e in fold_entries.values())
    pooled_ops: dict[str, Any] = {}
    for det in detectors:
        ops = [e["operating_points"][det] for e in fold_entries.values()]
        mean_dr = float(np.mean([op["detection_rate"] for op in ops]))
        mean_fcr = float(np.mean([op["false_call_rate"] for op in ops]))
        flaw_hits = sum(int(op.get("flaw_hits", 0)) for op in ops)
        clean_hits = sum(int(op.get("clean_hits", 0)) for op in ops)
        pooled_dr = float(flaw_hits / total_flawed) if total_flawed > 0 else 0.0
        pooled_fcr = float(clean_hits / total_clean) if total_clean > 0 else 0.0
        pooled_ops[det] = {
            "threshold": threshs[det],
            "detection_rate": pooled_dr,
            "false_call_rate": pooled_fcr,
            "mean_detection_rate": mean_dr,
            "mean_false_call_rate": mean_fcr,
            "pooled_detection_rate": pooled_dr,
            "pooled_false_call_rate": pooled_fcr,
            "flaw_hits": flaw_hits,
            "clean_hits": clean_hits,
        }
    return pooled_ops


def _generate_pooled_tradeoff_curves(
    fold_results: dict[str, FoldResult],
    threshs: dict[str, float],
    figures_dir: Path,
) -> dict[str, Any]:
    """Compute pooled tradeoff curves and render plot figures."""
    curves: dict[str, Any] = {}
    for det in ("supervised", "anomaly", "h2", "h3"):
        f_arr = np.concatenate([res.flawed_scores[det] for res in fold_results.values()])
        c_arr = np.concatenate([res.clean_scores[det] for res in fold_results.values()])
        y_true = np.concatenate([np.ones(len(f_arr), dtype=int), np.zeros(len(c_arr), dtype=int)])
        all_s = np.concatenate([f_arr, c_arr])
        curve = compute_tradeoff_curve(y_true, all_s, operating_threshold=threshs[det])
        curves[det] = curve
        plot_tradeoff_curve(curve, det, figures_dir / f"tradeoff_{det}.png")
    return curves


def _setup_analysis_context(
    cfg: FalseCallConfig,
    det_cfg: DetectorConfig,
) -> tuple[torch.device, ThresholdConfig, nn.Module, CalibContext, float, list[LocoFold]]:
    """Initialize compute device, thresholds, models, and LOCO fold splits."""
    set_seed(cfg.seed)
    dev = pick_device(cfg.device)
    thresholds = _load_threshold_config(det_cfg)
    anom_path = _resolve_model_path("anomaly_2d.pt", cfg.models_dir)
    anom_model = _load_anomaly_model(anom_path, dev)

    root = _find_data_root(cfg.root)
    tr_batches = list_batches("training", root=root)
    val_batches = list_batches("validation", root=root)
    cal_setup = CalibSetup(
        anom_model=anom_model,
        dev=dev,
        target_fcr=thresholds.anomaly_fcr,
        quick=cfg.quick,
        metrics_path=cfg.anomaly_metrics_path,
    )
    cal_ctx, t_anom = _build_calib_context(val_batches, cal_setup)
    folds = make_folds(tr_batches, val_batches, seed=cfg.seed, val_mode="batch")
    return dev, thresholds, anom_model, cal_ctx, float(t_anom), folds


def _aggregate_pooled_metrics(
    per_fold: dict[str, dict[str, Any]],
) -> PooledMetrics:
    """Aggregate flagged records, cause counts, and agreement across folds."""
    all_flagged: list[dict[str, Any]] = []
    for entry in per_fold.values():
        all_flagged.extend(entry["flagged_cases"])

    total_clean_pairs = sum(e["n_clean"] for e in per_fold.values())
    causes_cnt: dict[str, int] = {}
    for rec in all_flagged:
        c = rec["cause"]
        causes_cnt[c] = causes_cnt.get(c, 0) + 1

    agr_table = build_agreement_table(all_flagged)
    agr_table["total_pairs_counted"] = total_clean_pairs
    agr_table["scan_pairs_evaluated"] = total_clean_pairs
    return PooledMetrics(
        all_flagged=all_flagged,
        total_clean_pairs=total_clean_pairs,
        causes_cnt=causes_cnt,
        agr_table=agr_table,
    )


def _build_pooled_summary(
    metrics: PooledMetrics,
    pooled_ops: dict[str, Any],
) -> dict[str, Any]:
    """Build pooled summary block recording operating points and evaluated pairs."""
    return {
        "description": "Pooled across all LOCO folds; pools per-fold operating points.",
        "total_scan_pairs_counted": metrics.total_clean_pairs,
        "total_scan_pairs_evaluated": metrics.total_clean_pairs,
        "total_flagged_pairs": len(metrics.all_flagged),
        "operating_points": pooled_ops,
        "cause_counts": metrics.causes_cnt,
        "cause_counts_totals": {
            "total_pairs_counted": metrics.total_clean_pairs,
            "total_flagged": len(metrics.all_flagged),
            "counts": metrics.causes_cnt,
        },
        "agreement_table": metrics.agr_table,
    }


def _assemble_payload(
    ctx: PayloadContext,
    metrics: PooledMetrics,
) -> dict[str, Any]:
    """Assemble final analysis dictionary for JSON and Markdown persistence."""
    pooled_summary = _build_pooled_summary(metrics, ctx.pooled_ops)
    return {
        "label": "Real data (ML-NDT)",
        "run_label": ctx.run_label,
        "detector_config": ctx.fp,
        "config_fingerprint": ctx.fp,
        "protocol": "Pre-registered threshold false call analysis and trade-off curves.",
        "protocol_note": PROTOCOL_NOTE,
        "n_clean": metrics.total_clean_pairs,
        "n_clean_pairs": metrics.total_clean_pairs,
        "n_flagged": len(metrics.all_flagged),
        "total_scan_pairs_counted": metrics.total_clean_pairs,
        "thresholds": ctx.threshs,
        "cause_counts": metrics.causes_cnt,
        "cause_counts_totals": pooled_summary["cause_counts_totals"],
        "agreement_table": metrics.agr_table,
        "flagged_cases": metrics.all_flagged,
        "tradeoff_curves": ctx.pooled_curves,
        "operating_points": {d: ctx.pooled_ops[d] for d in ctx.pooled_ops},
        "per_fold": ctx.per_fold,
        "pooled_summary": pooled_summary,
    }


def run_false_call_analysis(cfg: FalseCallConfig) -> dict[str, Any]:
    """Execute complete false call analysis and trade-off evaluation."""
    det_cfg = _resolve_detector_config(cfg)
    fp = config_fingerprint(det_cfg)
    run_label = "pre-registered" if fp["is_default"] else "exploratory"

    dev, thresholds, anom_model, cal_ctx, t_anom, folds = _setup_analysis_context(cfg, det_cfg)
    fold_ctx = FoldScoringContext(cal_ctx, anom_model, dev, cfg, thresholds, t_anom)
    fold_results, t_s_mean = _evaluate_all_folds(folds, fold_ctx)

    threshs = {
        "supervised": thresholds.supervised,
        "anomaly": t_anom,
        "h2": t_s_mean,
        "h3": thresholds.physics,
    }
    insp_ctx = CleanInspectionContext(anom_model, dev, cfg.cases_dir)
    per_fold = _process_per_fold_results(fold_results, insp_ctx)
    pooled_ops = _pool_operating_points(per_fold, threshs)
    pooled_curves = _generate_pooled_tradeoff_curves(fold_results, threshs, cfg.figures_dir)

    metrics = _aggregate_pooled_metrics(per_fold)
    p_ctx = PayloadContext(per_fold, pooled_curves, pooled_ops, threshs, fp, run_label)
    payload = _assemble_payload(p_ctx, metrics)
    save_false_call_analysis(payload, cfg.out_json, cfg.out_md)
    return payload


def main() -> None:
    """CLI entry point for running false call analysis."""
    parser = argparse.ArgumentParser(description="False call analysis runner")
    parser.add_argument("--root", type=Path, default=MLNDT_ROOT)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--out-json", type=Path, default=Path("results/false_call_analysis.json"))
    parser.add_argument("--out-md", type=Path, default=Path("results/false_call_analysis.md"))
    parser.add_argument("--config", type=Path, default=None, help="Custom detectors.yaml")
    parser.add_argument("--set", dest="overrides", action="append", help="Config override key=val")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = FalseCallConfig(
        root=args.root,
        models_dir=args.models_dir,
        out_json=args.out_json,
        out_md=args.out_md,
        config_path=args.config,
        overrides=args.overrides,
        device=args.device,
        quick=args.quick,
    )
    run_false_call_analysis(cfg)


if __name__ == "__main__":
    main()
