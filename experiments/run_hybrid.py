"""Hybrid detection experiment runner combining supervised LOCO and ConvAE2D (ADR 0010).

Evaluates supervised-only, unsupervised anomaly-only, H1 OR-union, and H2 score fusion
across 3 physical crack LOCO folds without test-set tuning.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        del sys.modules["utriage"]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from utriage.integrity import sha256_file  # noqa: E402
from utriage.ml.anomaly import ConvAE2D, anomaly_score  # noqa: E402
from utriage.ml.datasets import MLNDT_ROOT, list_batches  # noqa: E402
from utriage.ml.infer import load_model  # noqa: E402
from utriage.ml.loco import IndexedMlNdtDataset, make_folds  # noqa: E402
from utriage.ml.mlndt_meta import image_index  # noqa: E402
from utriage.ml.train import TrainConfig, pick_device, predict, set_seed  # noqa: E402
from utriage.triage.hybrid import (  # noqa: E402
    ecdf,
    evaluate_detector,
    fused_score,
    or_union,
    quantile_threshold,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HybridConfig:
    """Configuration for running hybrid detection evaluation."""

    root: Path = MLNDT_ROOT
    models_dir: Path = Path("models")
    anomaly_metrics_path: Path = Path("results/anomaly_metrics.json")
    out_json: Path = Path("results/hybrid_metrics.json")
    target_fcr: float = 0.01
    device: str = "auto"
    quick: bool = False
    seed: int = 0


@dataclass(frozen=True)
class CalibContext:
    """Precomputed calibration items, anomaly probabilities, and ecdf."""

    items: list[tuple[Path, int]]
    p_anom_calib: np.ndarray
    ecdf_fn: Callable[[np.ndarray | float], np.ndarray]
    target_fcr: float


@dataclass(frozen=True)
class AnomalyModelInfo:
    """Loaded anomaly model, threshold, sha256, and compute device."""

    model: nn.Module
    t_anom: float
    sha256: str
    dev: torch.device


@dataclass(frozen=True)
class FoldAnomalyContext:
    """Precomputed anomaly scores and probabilities for a fold."""

    held_out: str
    test_items: list[tuple[Path, int]]
    a_score_test: np.ndarray
    p_anom_test: np.ndarray
    y_test: np.ndarray
    t_anom: float
    anomaly_sha: str


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


def _find_data_root(root: Path) -> Path:
    """Locate ML-NDT dataset root across standalone clone or git worktree."""
    if root.exists():
        return root
    parent = _repo_parent_root()
    if parent is not None:
        target = parent / "data" / "external" / "ML-NDT" / "data"
        if target.exists():
            return target
    return root


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
    state_dict = torch.load(path, map_location=dev, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()
    return model


def _load_or_fit_t_anom(
    path: Path,
    cal_scores: np.ndarray,
    target_fcr: float,
) -> float:
    """Load real threshold from anomaly_metrics.json or fit on calibration scores."""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        real_block = data.get("real", {})
        if "threshold" in real_block:
            return float(real_block["threshold"])
    return quantile_threshold(cal_scores, target_fcr)


def _get_calibration_items(val_batches: list[Path], quick: bool) -> list[tuple[Path, int]]:
    """Extract flaw-free validation items for calibration."""
    val_idx = image_index(val_batches)
    clean = [(b, idx) for b, idx, c in val_idx if c is None]
    return clean[:32] if quick else clean


def _prepare_test_items(
    test_items: list[tuple[Path, int]],
    quick: bool,
) -> list[tuple[Path, int]]:
    """Cap test items if quick mode is active, preserving both flawed and unflawed."""
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
    """Compute ConvAE2D anomaly scores for indexed ML-NDT items."""
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


def _eval_cell(flags: np.ndarray, y: np.ndarray, sha: str) -> dict[str, Any]:
    """Evaluate detector flags and attach model sha256."""
    res = evaluate_detector(flags, y)
    res["model_sha256"] = sha
    return res


def _evaluate_weights(
    weights_path: Path,
    fold_ctx: FoldAnomalyContext,
    cal_ctx: CalibContext,
    dev: torch.device,
) -> dict[str, Any]:
    """Score fold test set with supervised model and evaluate hybrid detectors."""
    sup_model, _ = load_model(weights_path, device=str(dev))
    model_sha = sha256_file(weights_path)
    p_sup_test = _score_supervised(sup_model, fold_ctx.test_items, dev)
    p_sup_calib = _score_supervised(sup_model, cal_ctx.items, dev)
    s_calib = fused_score(p_sup_calib, cal_ctx.p_anom_calib)
    t_s = quantile_threshold(s_calib, cal_ctx.target_fcr)
    s_test = fused_score(p_sup_test, fold_ctx.p_anom_test)

    flags_sup = p_sup_test >= 0.5
    flags_anom = fold_ctx.a_score_test >= fold_ctx.t_anom
    flags_h1 = or_union(p_sup_test, fold_ctx.a_score_test, fold_ctx.t_anom)
    flags_h2 = s_test >= t_s
    y = fold_ctx.y_test
    methods = {
        "supervised": _eval_cell(flags_sup, y, model_sha),
        "anomaly": _eval_cell(flags_anom, y, fold_ctx.anomaly_sha),
        "h1": _eval_cell(flags_h1, y, model_sha),
        "h2": _eval_cell(flags_h2, y, model_sha),
    }
    return {
        "model_name": weights_path.name,
        "model_sha256": model_sha,
        "t_s": float(t_s),
        "methods": methods,
    }


def _compute_seed_summary(
    seed_cells: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compute mean/min/max detection and false call rates across seeds."""
    summary: dict[str, dict[str, float]] = {}
    for method in ("supervised", "anomaly", "h1", "h2"):
        d_rates = [c[method]["detection"]["rate"] for c in seed_cells if method in c]
        fc_rates = [c[method]["false_calls"]["rate"] for c in seed_cells if method in c]
        valid_d = [r for r in d_rates if not np.isnan(r)]
        valid_fc = [r for r in fc_rates if not np.isnan(r)]
        summary[method] = {
            "mean_detection_rate": float(np.mean(valid_d)) if valid_d else 0.0,
            "min_detection_rate": float(np.min(valid_d)) if valid_d else 0.0,
            "max_detection_rate": float(np.max(valid_d)) if valid_d else 0.0,
            "mean_false_call_rate": float(np.mean(valid_fc)) if valid_fc else 0.0,
            "min_false_call_rate": float(np.min(valid_fc)) if valid_fc else 0.0,
            "max_false_call_rate": float(np.max(valid_fc)) if valid_fc else 0.0,
        }
    return summary


def _eval_candidate_models(
    fold_ctx: FoldAnomalyContext,
    cal_ctx: CalibContext,
    anom_info: AnomalyModelInfo,
    cfg: HybridConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate seed models and single model for a fold."""
    crack = fold_ctx.held_out
    candidates = [f"loco_{crack}_seed{s}.pt" for s in range(3)] + [f"loco_{crack}.pt"]
    models_dict: dict[str, Any] = {}
    seed_cells: list[dict[str, Any]] = []

    for fname in candidates:
        m_path = _resolve_model_path(fname, cfg.models_dir)
        if not m_path.exists():
            continue
        res = _evaluate_weights(m_path, fold_ctx, cal_ctx, anom_info.dev)
        models_dict[fname] = res
        if "_seed" in fname:
            seed_cells.append(res["methods"])

    return models_dict, seed_cells


def _run_fold_evaluation(
    fold: Any,
    cal_ctx: CalibContext,
    anom_info: AnomalyModelInfo,
    cfg: HybridConfig,
) -> dict[str, Any]:
    """Evaluate all available models on fold test set."""
    test_items = _prepare_test_items(fold.test, cfg.quick)
    y_test, _ = IndexedMlNdtDataset(test_items).get_labels()
    a_score_test = _score_anomaly(anom_info.model, test_items, anom_info.dev)
    p_anom_test = cal_ctx.ecdf_fn(a_score_test)
    fold_ctx = FoldAnomalyContext(
        held_out=fold.held_out,
        test_items=test_items,
        a_score_test=a_score_test,
        p_anom_test=p_anom_test,
        y_test=y_test,
        t_anom=anom_info.t_anom,
        anomaly_sha=anom_info.sha256,
    )
    models_dict, seed_cells = _eval_candidate_models(fold_ctx, cal_ctx, anom_info, cfg)
    return {
        "held_out": fold.held_out,
        "models": models_dict,
        "summary_3seeds": _compute_seed_summary(seed_cells),
    }


def _build_calib_context(
    val_batches: list[Path],
    anom_model: nn.Module,
    dev: torch.device,
    cfg: HybridConfig,
) -> tuple[CalibContext, float]:
    """Prepare clean calibration items, anomaly scores, and threshold."""
    cal_items = _get_calibration_items(val_batches, cfg.quick)
    a_score_calib = _score_anomaly(anom_model, cal_items, dev)
    t_anom = _load_or_fit_t_anom(cfg.anomaly_metrics_path, a_score_calib, cfg.target_fcr)
    ecdf_fn = ecdf(a_score_calib)
    p_anom_calib = ecdf_fn(a_score_calib)
    ctx = CalibContext(
        items=cal_items,
        p_anom_calib=p_anom_calib,
        ecdf_fn=ecdf_fn,
        target_fcr=cfg.target_fcr,
    )
    return ctx, t_anom


def _build_payload(
    fold_results: dict[str, Any],
    anom_info: AnomalyModelInfo,
    dev: torch.device,
    cfg: HybridConfig,
) -> dict[str, Any]:
    """Construct top-level metrics dictionary citing ADR 0010."""
    dev_name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"
    return {
        "label": "Real data (ML-NDT)",
        "protocol": (
            "Pre-registered design per docs/decisions/0010: H1 OR-union (flag = (p_sup >= 0.5) "
            "OR (anomaly_score >= T_anom)) and H2 score fusion (p_anom = empirical CDF of "
            "anomaly score on clean calibration images, s = (p_sup + p_anom) / 2, T_s = (1 - 0.01) "
            "quantile of s on clean calibration images). Evaluated on leave-one-crack-out (LOCO) "
            "folds across 3 physical cracks without test-set tuning."
        ),
        "target_fcr": cfg.target_fcr,
        "t_anom": float(anom_info.t_anom),
        "anomaly_model_sha256": anom_info.sha256,
        "device": str(dev),
        "device_name": dev_name,
        "folds": fold_results,
    }


def _save_results(path: Path, payload: dict[str, Any]) -> None:
    """Save dictionary to JSON file ensuring parent directories exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_hybrid(cfg: HybridConfig) -> dict[str, Any]:
    """Execute hybrid detection experiment across LOCO folds and write results."""
    set_seed(cfg.seed)
    dev = pick_device(cfg.device)
    dev_name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"
    logger.info("Running hybrid evaluation on %s (%s)", dev, dev_name)

    anom_path = _resolve_model_path("anomaly_2d.pt", cfg.models_dir)
    if not anom_path.exists():
        raise FileNotFoundError(f"Anomaly model checkpoint not found: {anom_path}")
    anom_sha = sha256_file(anom_path)
    anom_model = _load_anomaly_model(anom_path, dev)

    root = _find_data_root(cfg.root)
    tr_batches = list_batches("training", root=root)
    val_batches = list_batches("validation", root=root)
    cal_ctx, t_anom = _build_calib_context(val_batches, anom_model, dev, cfg)
    anom_info = AnomalyModelInfo(model=anom_model, t_anom=t_anom, sha256=anom_sha, dev=dev)

    folds = make_folds(tr_batches, val_batches, seed=cfg.seed, val_mode="batch")
    fold_results: dict[str, Any] = {}
    for fold in folds:
        fold_results[fold.held_out] = _run_fold_evaluation(fold, cal_ctx, anom_info, cfg)

    payload = _build_payload(fold_results, anom_info, dev, cfg)
    _save_results(cfg.out_json, payload)
    return payload


def main() -> None:
    """Parse command-line arguments and run hybrid evaluation."""
    parser = argparse.ArgumentParser(description="Run hybrid detection evaluation.")
    parser.add_argument("--root", type=Path, default=MLNDT_ROOT)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--anomaly-metrics", type=Path, default=Path("results/anomaly_metrics.json")
    )
    parser.add_argument("--out-json", type=Path, default=Path("results/hybrid_metrics.json"))
    parser.add_argument("--target-fcr", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = HybridConfig(
        root=args.root,
        models_dir=args.models_dir,
        anomaly_metrics_path=args.anomaly_metrics,
        out_json=args.out_json,
        target_fcr=args.target_fcr,
        device=args.device,
        quick=args.quick,
        seed=args.seed,
    )
    run_hybrid(cfg)


if __name__ == "__main__":
    main()
