"""Unsupervised anomaly detection experiment runner (project plan Phase 8).

Trains ConvAE2D on flaw-free real data and ConvAE1D on flaw-free synthetic windows.
Evaluates detection cells, empirical CDF probability calibration, and false call rates.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utriage.ml.anomaly import (
    ConvAE1D,
    ConvAE2D,
    anomaly_score,
    fit_threshold,
    train_autoencoder,
)
from utriage.ml.bar_data import WindowConfig, make_test_set, make_training_set
from utriage.ml.datasets import MLNDT_ROOT, list_batches, split_holdout
from utriage.ml.diagnostics import learning_curve
from utriage.ml.loco import PHYSICAL_CRACKS, IndexedMlNdtDataset
from utriage.ml.mlndt_meta import image_index
from utriage.ml.train import pick_device, set_seed
from utriage.pod.metrics import cell

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnomalyConfig:
    """Execution configuration for anomaly detection experiments."""

    root: Path = MLNDT_ROOT
    out_json: Path = Path("results/anomaly_metrics.json")
    model_2d_path: Path = Path("models/anomaly_2d.pt")
    model_1d_path: Path = Path("models/anomaly_1d.pt")
    epochs: int = 15
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    target_fcr: float = 0.01
    device: str = "auto"
    quick: bool = False
    seed: int = 0


@dataclass(frozen=True)
class RealSplits:
    """Partitions for flaw-free training, calibration, and crack evaluation."""

    train_items: list[tuple[Path, int]]
    cal_items: list[tuple[Path, int]]
    test_unflawed_items: list[tuple[Path, int]]
    crack_flawed_items: dict[str, list[tuple[Path, int]]]
    all_flawed_items: list[tuple[Path, int]]
    train_flaw_count: int


@dataclass(frozen=True)
class SyntheticData:
    """Synthetic window partitions for ConvAE1D anomaly detection."""

    tr_clean: np.ndarray
    cal_clean: np.ndarray
    test_flaw: np.ndarray
    test_clean: np.ndarray
    flaw_sizes: np.ndarray

    def __iter__(self):
        """Allow unpacking into 4 primary window arrays for testing."""
        return iter((self.tr_clean, self.cal_clean, self.test_flaw, self.test_clean))


@dataclass(frozen=True)
class CellInput:
    """Inputs required to compute a pod.metrics evaluation cell."""

    flaw_scores: np.ndarray
    unflawed_scores: np.ndarray
    cal_scores: np.ndarray
    flaw_sizes: np.ndarray
    threshold: float
    n_physical_flaws: int


@dataclass(frozen=True)
class EvalContext:
    """Evaluation context for real crack scoring."""

    cal_scores: np.ndarray
    unflawed_scores: np.ndarray
    threshold: float
    dev: torch.device


def map_scores_to_cdf(scores: np.ndarray, cal_scores: np.ndarray) -> np.ndarray:
    """Map anomaly scores to [0, 1] using empirical CDF of calibration clean scores."""
    sorted_cal = np.sort(np.asarray(cal_scores, dtype=float).ravel())
    if sorted_cal.size == 0:
        return np.zeros_like(scores, dtype=float)
    ranks = np.searchsorted(sorted_cal, np.asarray(scores, dtype=float), side="right")
    return np.clip(ranks / float(sorted_cal.size), 0.0, 1.0)


def _partition_crack_flaws(
    all_tr_idx: list[tuple[Path, int, str | None]],
    quick: bool,
) -> tuple[dict[str, list[tuple[Path, int]]], list[tuple[Path, int]]]:
    """Group flawed images by physical crack type."""
    crack_flawed: dict[str, list[tuple[Path, int]]] = {}
    all_flawed: list[tuple[Path, int]] = []
    for crack in PHYSICAL_CRACKS:
        items = [(b, idx) for b, idx, c in all_tr_idx if c == crack]
        subset = items[:16] if quick else items
        crack_flawed[crack] = subset
        all_flawed.extend(subset)
    return crack_flawed, all_flawed


def prepare_real_splits(cfg: AnomalyConfig) -> RealSplits:
    """Partition real ML-NDT data into flaw-free train, calibration, and test sets."""
    tr_batches = list_batches("training", root=cfg.root)
    val_batches = list_batches("validation", root=cfg.root)
    non_ho, ho = split_holdout(tr_batches, test_frac=0.1, seed=cfg.seed)

    non_ho_idx = image_index(non_ho)
    ho_idx = image_index(ho)
    val_idx = image_index(val_batches)

    train_items = [(b, idx) for b, idx, c in non_ho_idx if c is None]
    train_set = set(train_items)
    flaw_in_train = sum(1 for b, idx, c in non_ho_idx if (b, idx) in train_set and c is not None)

    cal_items = [(b, idx) for b, idx, c in val_idx if c is None]
    test_unflawed = [(b, idx) for b, idx, c in ho_idx if c is None]
    crack_flawed, all_flawed = _partition_crack_flaws(non_ho_idx + ho_idx, cfg.quick)

    return RealSplits(
        train_items=train_items[:64] if cfg.quick else train_items,
        cal_items=cal_items[:32] if cfg.quick else cal_items,
        test_unflawed_items=test_unflawed[:32] if cfg.quick else test_unflawed,
        crack_flawed_items=crack_flawed,
        all_flawed_items=all_flawed,
        train_flaw_count=flaw_in_train,
    )


def prepare_synthetic_data(quick: bool = False, seed: int = 0) -> SyntheticData:
    """Prepare clean train/cal windows and test windows from bar simulator."""
    n_tr_bars = 2 if quick else 20
    per_bar = 20 if quick else 50
    wcfg = WindowConfig(patch=False, seed=seed)
    wset_tr = make_training_set(wcfg, n_bars=n_tr_bars, per_bar=per_bar)
    clean_all = wset_tr.x[wset_tr.y == 0]

    n_clean = len(clean_all)
    split_idx = int(0.7 * n_clean)
    tr_clean = clean_all[:split_idx]
    cal_clean = clean_all[split_idx:]

    n_te_bars = 1 if quick else 4
    sizes = (0.5, 1.0) if quick else (0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2)
    scales = (1.0,) if quick else (0.5, 1.0, 1.5, 2.0)
    wset_te = make_test_set(wcfg, n_bars=n_te_bars, sizes_mm=sizes, noise_scales=scales)

    flaw_mask = wset_te.y == 1
    clean_mask = wset_te.y == 0
    return SyntheticData(
        tr_clean=tr_clean,
        cal_clean=cal_clean,
        test_flaw=wset_te.x[flaw_mask],
        test_clean=wset_te.x[clean_mask],
        flaw_sizes=wset_te.size_mm[flaw_mask],
    )


def _score_indexed_dataset(
    model: nn.Module,
    items: list[tuple[Path, int]],
    dev: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute anomaly scores for a list of ML-NDT indexed image items."""
    if not items:
        return np.empty(0, dtype=np.float32)
    ds = IndexedMlNdtDataset(items)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    scores: list[float] = []
    for x, _, _ in loader:
        s = anomaly_score(model, x.to(dev))
        scores.extend(s.detach().cpu().numpy().tolist())
    return np.array(scores, dtype=np.float32)


def _score_windows(
    model: nn.Module,
    windows: np.ndarray,
    dev: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute anomaly scores for synthetic 1D A-scan windows."""
    if len(windows) == 0:
        return np.empty(0, dtype=np.float32)
    tensor_ds = TensorDataset(torch.from_numpy(windows))
    loader = DataLoader(tensor_ds, batch_size=batch_size, shuffle=False)
    scores: list[float] = []
    for (x,) in loader:
        s = anomaly_score(model, x.to(dev))
        scores.extend(s.detach().cpu().numpy().tolist())
    return np.array(scores, dtype=np.float32)


def _build_eval_cell(data: CellInput) -> dict[str, Any]:
    """Compute cell metrics combining flaw detections and false calls."""
    hits = (data.flaw_scores >= data.threshold).astype(float)
    f_probs = map_scores_to_cdf(data.flaw_scores, data.cal_scores)
    u_probs = map_scores_to_cdf(data.unflawed_scores, data.cal_scores)
    all_probs = np.concatenate([f_probs, u_probs])

    n_f, n_u = len(data.flaw_scores), len(data.unflawed_scores)
    all_labels = np.concatenate([np.ones(n_f, dtype=float), np.zeros(n_u, dtype=float)])
    n_false = int(np.sum(data.unflawed_scores >= data.threshold))

    return cell(
        hits=hits,
        sizes=data.flaw_sizes,
        probs=all_probs,
        labels=all_labels,
        n_false=n_false,
        n_scans=n_u,
        n_physical_flaws=data.n_physical_flaws,
    )


def _train_2d_model(
    splits: RealSplits,
    cfg: AnomalyConfig,
    dev: torch.device,
) -> tuple[ConvAE2D, list[float]]:
    """Train ConvAE2D autoencoder on flaw-free real data."""
    train_ds = IndexedMlNdtDataset(splits.train_items)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    model = ConvAE2D()
    epochs = 1 if cfg.quick else cfg.epochs
    losses = train_autoencoder(
        model,
        loader,
        epochs=epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        device=dev,
    )
    return model, losses


def _train_1d_model(
    synth: SyntheticData,
    cfg: AnomalyConfig,
    dev: torch.device,
) -> tuple[ConvAE1D, list[float]]:
    """Train ConvAE1D autoencoder on simulated clean windows."""
    train_ds = TensorDataset(torch.from_numpy(synth.tr_clean))
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    model = ConvAE1D()
    epochs = 1 if cfg.quick else cfg.epochs
    losses = train_autoencoder(
        model,
        loader,
        epochs=epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        device=dev,
    )
    return model, losses


def _eval_crack_cells(
    model: nn.Module, splits: RealSplits, ctx: EvalContext
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate individual physical crack cells and aggregated overall cell."""
    per_crack: dict[str, Any] = {}
    all_scores: list[np.ndarray] = []
    all_sizes: list[np.ndarray] = []
    for crack in PHYSICAL_CRACKS:
        items = splits.crack_flawed_items[crack]
        s = _score_indexed_dataset(model, items, ctx.dev)
        sizes = IndexedMlNdtDataset(items).get_labels()[1]
        all_scores.append(s)
        all_sizes.append(sizes)
        inp = CellInput(s, ctx.unflawed_scores, ctx.cal_scores, sizes, ctx.threshold, 1)
        per_crack[crack] = _build_eval_cell(inp)

    comb_s = np.concatenate(all_scores) if all_scores else np.empty(0, dtype=np.float32)
    comb_sz = np.concatenate(all_sizes) if all_sizes else np.empty(0, dtype=np.float32)
    overall_inp = CellInput(comb_s, ctx.unflawed_scores, ctx.cal_scores, comb_sz, ctx.threshold, 3)
    return per_crack, _build_eval_cell(overall_inp)


def _format_learning_curve(history: Any) -> dict[str, Any] | None:
    """Format learning curve if history matches expected schema."""
    if not isinstance(history, list) or not history:
        return None
    first = history[0]
    if (
        isinstance(first, dict)
        and "epoch" in first
        and "train_loss" in first
        and "val_loss" in first
    ):
        return learning_curve(history)
    return None


def _run_real_anomaly(
    splits: RealSplits,
    cfg: AnomalyConfig,
    dev: torch.device,
) -> tuple[dict[str, Any], ConvAE2D]:
    """Train ConvAE2D on flaw-free data and evaluate on cracks and holdout unflawed."""
    model, losses = _train_2d_model(splits, cfg, dev)
    cal_scores = _score_indexed_dataset(model, splits.cal_items, dev, cfg.batch_size)
    threshold = fit_threshold(cal_scores, false_call_rate=cfg.target_fcr)
    unflawed_scores = _score_indexed_dataset(model, splits.test_unflawed_items, dev, cfg.batch_size)

    ctx = EvalContext(cal_scores, unflawed_scores, threshold, dev)
    per_crack, overall = _eval_crack_cells(model, splits, ctx)

    dev_name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
    metrics = {
        "label": "Real data (ML-NDT)",
        "protocol": (
            "ConvAE2D trained ONLY on flaw-free images of non-holdout batches; "
            "threshold calibration on flaw-free validation images; "
            "tested on physical cracks plus holdout unflawed images"
        ),
        "threshold": float(threshold),
        "target_fcr": float(cfg.target_fcr),
        "device": str(dev),
        "device_name": dev_name,
        "probability_mapping": "empirical CDF of calibration clean scores",
        "n_images": {
            "training_clean": len(splits.train_items),
            "calibration_clean": len(splits.cal_items),
            "test_unflawed": len(splits.test_unflawed_items),
            "test_flawed": len(splits.all_flawed_items),
            "total_test": len(splits.test_unflawed_items) + len(splits.all_flawed_items),
        },
        "per_crack": per_crack,
        "overall": overall,
        "learning_curve": _format_learning_curve(losses),
    }
    return metrics, model


def _run_synthetic_anomaly(
    synth: SyntheticData,
    cfg: AnomalyConfig,
    dev: torch.device,
) -> tuple[dict[str, Any], ConvAE1D]:
    """Train ConvAE1D on simulated clean windows and evaluate on test windows."""
    model, losses = _train_1d_model(synth, cfg, dev)
    cal_scores = _score_windows(model, synth.cal_clean, dev, cfg.batch_size)
    threshold = fit_threshold(cal_scores, false_call_rate=cfg.target_fcr)

    flaw_scores = _score_windows(model, synth.test_flaw, dev, cfg.batch_size)
    clean_scores = _score_windows(model, synth.test_clean, dev, cfg.batch_size)
    n_phys = len(np.unique(synth.flaw_sizes)) if len(synth.flaw_sizes) > 0 else 0

    inp = CellInput(flaw_scores, clean_scores, cal_scores, synth.flaw_sizes, threshold, n_phys)
    cell_data = _build_eval_cell(inp)

    dev_name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
    metrics = {
        "label": "Synthetic data",
        "protocol": (
            "ConvAE1D trained on simulated clean windows; "
            "calibrated on separate clean windows; "
            "tested on simulated flawed and clean windows"
        ),
        "threshold": float(threshold),
        "target_fcr": float(cfg.target_fcr),
        "device": str(dev),
        "device_name": dev_name,
        "probability_mapping": "empirical CDF of calibration clean scores",
        "n_windows": {
            "training_clean": len(synth.tr_clean),
            "calibration_clean": len(synth.cal_clean),
            "test_flaw": len(synth.test_flaw),
            "test_clean": len(synth.test_clean),
            "total_test": len(synth.test_flaw) + len(synth.test_clean),
        },
        "cell": cell_data,
        "learning_curve": _format_learning_curve(losses),
    }
    return metrics, model


def _make_serializable(obj: Any) -> Any:
    """Convert numpy scalar types and arrays to native Python primitives."""
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return None if np.isnan(val) else val
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj


def run_anomaly(cfg: AnomalyConfig) -> dict[str, Any]:
    """Execute complete unsupervised anomaly detection pipeline and save artifacts."""
    set_seed(cfg.seed)
    dev = pick_device(cfg.device)

    splits = prepare_real_splits(cfg)
    real_metrics, model_2d = _run_real_anomaly(splits, cfg, dev)

    synth = prepare_synthetic_data(quick=cfg.quick, seed=cfg.seed)
    synth_metrics, model_1d = _run_synthetic_anomaly(synth, cfg, dev)

    cfg.model_2d_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_2d.state_dict(), cfg.model_2d_path)
    cfg.model_1d_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_1d.state_dict(), cfg.model_1d_path)

    payload = {"real": real_metrics, "synthetic": synth_metrics}
    cfg.out_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_json.write_text(
        json.dumps(_make_serializable(payload), indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments for anomaly experiment."""
    parser = argparse.ArgumentParser(description="Unsupervised anomaly detection experiment")
    parser.add_argument("--quick", action="store_true", help="Tiny subsets, 1 epoch")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument(
        "--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')"
    )
    parser.add_argument("--root", type=Path, default=MLNDT_ROOT, help="ML-NDT root directory")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("results/anomaly_metrics.json"),
        help="Output metrics path",
    )
    parser.add_argument(
        "--model-2d", type=Path, default=Path("models/anomaly_2d.pt"), help="2D model output path"
    )
    parser.add_argument(
        "--model-1d", type=Path, default=Path("models/anomaly_1d.pt"), help="1D model output path"
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for anomaly detection experiment."""
    args = _parse_args()
    cfg = AnomalyConfig(
        root=args.root,
        out_json=args.out_json,
        model_2d_path=args.model_2d,
        model_1d_path=args.model_1d,
        epochs=args.epochs,
        device=args.device,
        quick=args.quick,
    )
    run_anomaly(cfg)


if __name__ == "__main__":
    main()
