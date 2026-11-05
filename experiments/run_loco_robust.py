"""Robust leave-one-crack-out (LOCO) training and evaluation experiment.

Pre-registered in docs/decisions/0012-robust-loco-preregistration.md.
Trains MlNdtNet with GPU-resident float16 caching and physical augmentations,
evaluating on held-out crack test sets at clean-calibrated FCR thresholds.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from utriage.integrity import sha256_file  # noqa: E402
from utriage.ml.datasets import list_batches  # noqa: E402
from utriage.ml.infer import save_model  # noqa: E402
from utriage.ml.loco import IndexedMlNdtDataset, LocoFold, make_folds  # noqa: E402
from utriage.ml.models import MlNdtNet  # noqa: E402
from utriage.ml.train import loss_fn, pick_device, set_seed  # noqa: E402
from utriage.pod.metrics import clopper_pearson  # noqa: E402


@dataclass(frozen=True)
class RobustConfig:
    """Configuration options for robust LOCO training and evaluation."""

    device: str = "auto"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 40
    batch_size: int = 256
    patience: int = 6
    quick: bool = False
    out_path: Path = Path("results/loco_robust_metrics.json")
    models_dir: Path = Path("models")


@dataclass(frozen=True)
class GpuTrainBatch:
    """Context holding GPU training loop components."""

    scaler: torch.amp.GradScaler | None
    gen: torch.Generator
    batch_size: int


def apply_gain(
    x: torch.Tensor,
    gen: torch.Generator,
    gain_min: float = 0.5,
    gain_max: float = 2.0,
) -> torch.Tensor:
    """Scale input amplitudes by uniform random gain."""
    shape = (x.shape[0], 1, 1, 1) if x.ndim == 4 else (1, 1, 1)
    gain = torch.empty(shape, device=x.device, dtype=x.dtype).uniform_(
        gain_min, gain_max, generator=gen
    )
    return x * gain


def apply_offset(
    x: torch.Tensor,
    gen: torch.Generator,
    scale: float = 0.2,
) -> torch.Tensor:
    """Add uniform DC baseline shift scaled by per-image standard deviation."""
    shape = (x.shape[0], 1, 1, 1) if x.ndim == 4 else (1, 1, 1)
    std = torch.clamp(x.float().std(dim=(-2, -1), keepdim=True).to(x.dtype), min=1e-5)
    offset = (
        torch.empty(shape, device=x.device, dtype=x.dtype).uniform_(-scale, scale, generator=gen)
        * std
    )
    return x + offset


def apply_noise(
    x: torch.Tensor,
    gen: torch.Generator,
    scale: float = 0.1,
) -> torch.Tensor:
    """Add Gaussian electronic/grain scatter noise scaled by image standard deviation."""
    shape = (x.shape[0], 1, 1, 1) if x.ndim == 4 else (1, 1, 1)
    std = torch.clamp(x.float().std(dim=(-2, -1), keepdim=True).to(x.dtype), min=1e-5)
    noise_std = (
        torch.empty(shape, device=x.device, dtype=x.dtype).uniform_(0.0, scale, generator=gen) * std
    )
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen) * noise_std
    return x + noise


def apply_time_shift(
    x: torch.Tensor,
    gen: torch.Generator,
    max_shift: int = 16,
) -> torch.Tensor:
    """Random circular time roll along fast-time axis (rows, dim -2)."""
    shift = int(
        torch.randint(-max_shift, max_shift + 1, (1,), device=x.device, generator=gen).item()
    )
    return torch.roll(x, shifts=shift, dims=-2)


def apply_scan_flip(
    x: torch.Tensor,
    gen: torch.Generator,
) -> torch.Tensor:
    """Randomly flip images along scan axis (columns, dim -1)."""
    shape = (x.shape[0], 1, 1, 1) if x.ndim == 4 else (1, 1, 1)
    mask = torch.rand(shape, device=x.device, generator=gen) < 0.5
    flipped = torch.flip(x, dims=[-1])
    return torch.where(mask, flipped, x)


def apply_augmentation(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Apply full chain of physical data augmentations to input batch."""
    out = apply_gain(x, gen)
    out = apply_offset(out, gen)
    out = apply_noise(out, gen)
    out = apply_time_shift(out, gen)
    return apply_scan_flip(out, gen)


def _copy_chunk_to_gpu(
    ds: IndexedMlNdtDataset,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    start: int,
    end: int,
) -> None:
    """Copy a slice of dataset items to pre-allocated GPU tensors."""
    imgs, flaws, sizes = tensors
    dev = imgs.device
    chunk_x: list[torch.Tensor] = []
    chunk_f: list[torch.Tensor] = []
    chunk_s: list[torch.Tensor] = []
    for i in range(start, end):
        x, f, s = ds[i]
        chunk_x.append(x)
        chunk_f.append(f)
        chunk_s.append(s)
    imgs[start:end] = torch.stack(chunk_x).to(device=dev, dtype=torch.float16)
    flaws[start:end] = torch.stack(chunk_f).to(device=dev, dtype=torch.float32)
    sizes[start:end] = torch.stack(chunk_s).to(device=dev, dtype=torch.float32)


def load_dataset_to_gpu(
    ds: IndexedMlNdtDataset,
    device: torch.device,
    chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load dataset into GPU float16 image tensor and float32 label tensors."""
    n = len(ds)
    imgs = torch.empty((n, 1, 256, 256), dtype=torch.float16, device=device)
    flaws = torch.empty((n,), dtype=torch.float32, device=device)
    sizes = torch.empty((n,), dtype=torch.float32, device=device)
    target = (imgs, flaws, sizes)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        _copy_chunk_to_gpu(ds, target, start, end)
    return imgs, flaws, sizes


def _step_gpu_batch(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
) -> float:
    """Execute single augmented forward-backward step on GPU batch."""
    bx, bf, bs = batch
    opt.zero_grad()
    if scaler is not None:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logit, size_pred = model(bx)
            loss = loss_fn(logit, size_pred, bf, bs, w=0.1)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    else:
        logit, size_pred = model(bx)
        loss = loss_fn(logit, size_pred, bf, bs, w=0.1)
        loss.backward()
        opt.step()
    return float(loss.item())


def _run_gpu_train_epoch(
    model: nn.Module,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    opt: torch.optim.Optimizer,
    batch_ctx: GpuTrainBatch,
) -> float:
    """Run one training epoch on GPU-resident tensors with augmentation and AMP."""
    imgs, flaws, sizes = tensors
    n = len(imgs)
    perm = torch.randperm(n, device=imgs.device, generator=batch_ctx.gen)
    model.train()
    total_loss, total_count = 0.0, 0
    for start in range(0, n, batch_ctx.batch_size):
        end = min(start + batch_ctx.batch_size, n)
        idx = perm[start:end]
        bx = apply_augmentation(imgs[idx], batch_ctx.gen)
        batch = (bx, flaws[idx], sizes[idx])
        loss_val = _step_gpu_batch(model, batch, opt, batch_ctx.scaler)
        total_loss += loss_val * len(bx)
        total_count += len(bx)
    return total_loss / max(1, total_count)


def _run_cpu_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    gen: torch.Generator,
) -> float:
    """Run one training epoch on CPU using DataLoader with augmentation."""
    model.train()
    total_loss, total_count = 0.0, 0
    for bx, bf, bs in loader:
        bx = apply_augmentation(bx, gen)
        opt.zero_grad()
        logit, size_pred = model(bx)
        loss = loss_fn(logit, size_pred, bf, bs, w=0.1)
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(bx)
        total_count += len(bx)
    return total_loss / max(1, total_count)


@torch.no_grad()
def _evaluate_gpu_batch(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    is_cuda: bool,
) -> float:
    """Forward pass for evaluation batch."""
    bx, bf, bs = batch
    if is_cuda:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logit, size_pred = model(bx)
            loss = loss_fn(logit, size_pred, bf, bs, w=0.1)
    else:
        logit, size_pred = model(bx)
        loss = loss_fn(logit, size_pred, bf, bs, w=0.1)
    return float(loss.item()) * len(bx)


def _evaluate_gpu_val(
    model: nn.Module,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch_size: int,
) -> float:
    """Compute validation loss on GPU-resident tensors."""
    imgs, flaws, sizes = tensors
    n, is_cuda = len(imgs), imgs.device.type == "cuda"
    model.eval()
    total_loss, total_count = 0.0, 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = (imgs[start:end], flaws[start:end], sizes[start:end])
        total_loss += _evaluate_gpu_batch(model, batch, is_cuda)
        total_count += len(batch[0])
    return total_loss / max(1, total_count)


def _evaluate_cpu_val(model: nn.Module, loader: DataLoader) -> float:
    """Compute validation loss on CPU using DataLoader."""
    model.eval()
    total_loss, total_count = 0.0, 0
    with torch.no_grad():
        for bx, bf, bs in loader:
            logit, size_pred = model(bx)
            loss = loss_fn(logit, size_pred, bf, bs, w=0.1)
            total_loss += loss.item() * len(bx)
            total_count += len(bx)
    return total_loss / max(1, total_count)


def _train_seed_gpu(
    model: nn.Module,
    gpu_data: tuple[Any, Any],
    cfg: RobustConfig,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train single seed on GPU-resident data with early stopping."""
    gpu_tr, gpu_val = gpu_data
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda")
    gen = torch.Generator(device="cuda").manual_seed(seed)
    batch_ctx = GpuTrainBatch(scaler, gen, cfg.batch_size)

    best_loss, best_ep, no_improve = float("inf"), 1, 0
    best_weights = copy.deepcopy(model.state_dict())

    for ep in range(1, cfg.epochs + 1):
        _ = _run_gpu_train_epoch(model, gpu_tr, opt, batch_ctx)
        vl_loss = _evaluate_gpu_val(model, gpu_val, cfg.batch_size)
        sched.step()
        if vl_loss < best_loss:
            best_loss, best_ep, no_improve = vl_loss, ep, 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break
    return best_weights, {"seed": seed, "best_epoch": best_ep, "best_val_loss": best_loss}


def _train_seed_cpu(
    model: nn.Module,
    datasets: tuple[IndexedMlNdtDataset, IndexedMlNdtDataset],
    cfg: RobustConfig,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train single seed on CPU with DataLoaders and early stopping."""
    train_ds, val_ds = datasets
    t_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    v_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    best_loss, best_ep, no_improve = float("inf"), 1, 0
    best_weights = copy.deepcopy(model.state_dict())

    for ep in range(1, cfg.epochs + 1):
        _ = _run_cpu_train_epoch(model, t_loader, opt, gen)
        vl_loss = _evaluate_cpu_val(model, v_loader)
        sched.step()
        if vl_loss < best_loss:
            best_loss, best_ep, no_improve = vl_loss, ep, 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break
    return best_weights, {"seed": seed, "best_epoch": best_ep, "best_val_loss": best_loss}


@torch.no_grad()
def predict_dataset(
    model: nn.Module,
    ds: IndexedMlNdtDataset,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Generate flaw probabilities on dataset in order."""
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs: list[np.ndarray] = []
    for batch in loader:
        x = batch[0].to(device)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logit, _ = model(x)
        else:
            logit, _ = model(x)
        probs.append(torch.sigmoid(logit).view(-1).cpu().numpy())
    if not probs:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(probs, axis=0)


def split_clean_scores(clean_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Partition clean scores into 50/50 calibration and evaluation halves."""
    n = len(clean_scores)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    half = n // 2
    return clean_scores[perm[:half]], clean_scores[perm[half:]]


def calibrate_threshold(clean_calib: np.ndarray, target_fcr: float) -> float:
    """Calculate clean-calibrated threshold at (1 - target_fcr) quantile."""
    if len(clean_calib) == 0:
        return 0.5
    q = 1.0 - target_fcr
    return float(np.quantile(clean_calib, q, method="higher"))


def _calc_binomial_metric(count: int, total: int) -> dict[str, Any]:
    """Compute empirical rate and Clopper-Pearson 95% confidence interval."""
    rate = float(count / total) if total > 0 else 0.0
    low, high = clopper_pearson(count, total, conf=0.95)
    return {"rate": rate, "ci_95": [low, high]}


def evaluate_detection_and_fcr(
    flawed_scores: np.ndarray,
    clean_eval: np.ndarray,
    threshold: float,
    target_fcr: float,
) -> dict[str, Any]:
    """Compute detection hit rate and false call rate at fixed threshold with Clopper-Pearson CI."""
    n_flawed = len(flawed_scores)
    hits = int(np.sum(flawed_scores >= threshold)) if n_flawed > 0 else 0
    det_stats = _calc_binomial_metric(hits, n_flawed)
    det_stats.update({"n_hit": hits, "n_flawed": n_flawed})

    n_clean = len(clean_eval)
    false_calls = int(np.sum(clean_eval >= threshold)) if n_clean > 0 else 0
    fc_stats = _calc_binomial_metric(false_calls, n_clean)
    fc_stats.update({"n_false": false_calls, "n_clean": n_clean})

    return {
        "target_fcr": target_fcr,
        "threshold": threshold,
        "detections": det_stats,
        "false_calls": fc_stats,
    }


def compute_ensemble_metrics(
    ensemble_probs: np.ndarray,
    flaw_labels: np.ndarray,
) -> dict[str, Any]:
    """Compute AUROC, AP, and clean-calibrated metrics for FCR 0.01 and 0.05."""
    y_true = (flaw_labels > 0.5).astype(int)
    flawed_scores = ensemble_probs[y_true == 1]
    clean_scores = ensemble_probs[y_true == 0]

    unique_classes = len(np.unique(y_true))
    auroc = float(roc_auc_score(y_true, ensemble_probs)) if unique_classes > 1 else float("nan")
    ap = (
        float(average_precision_score(y_true, ensemble_probs))
        if unique_classes > 1
        else float("nan")
    )

    clean_calib, clean_eval = split_clean_scores(clean_scores)
    th_01 = calibrate_threshold(clean_calib, 0.01)
    th_05 = calibrate_threshold(clean_calib, 0.05)

    m_01 = evaluate_detection_and_fcr(flawed_scores, clean_eval, th_01, 0.01)
    m_05 = evaluate_detection_and_fcr(flawed_scores, clean_eval, th_05, 0.05)

    return {
        "auroc": auroc,
        "average_precision": ap,
        "n_clean": int(len(clean_scores)),
        "fcr_0.01": m_01,
        "fcr_0.05": m_05,
    }


def _calc_summary(metrics: list[dict[str, Any]]) -> dict[str, float]:
    """Calculate mean, min, and max across seed metric evaluations."""
    aucs = [m["auroc"] for m in metrics]
    aps = [m["average_precision"] for m in metrics]
    return {
        "auroc_mean": float(np.mean(aucs)),
        "auroc_min": float(np.min(aucs)),
        "auroc_max": float(np.max(aucs)),
        "ap_mean": float(np.mean(aps)),
        "ap_min": float(np.min(aps)),
        "ap_max": float(np.max(aps)),
    }


def _evaluate_seed_predictions(
    seed_models: list[nn.Module],
    test_ds: IndexedMlNdtDataset,
    dev: torch.device,
    flaw_labels: np.ndarray,
) -> tuple[list[np.ndarray], list[dict[str, Any]], dict[str, float]]:
    """Generate seed predictions and summarize individual seed metrics."""
    probs_list = [predict_dataset(m, test_ds, dev) for m in seed_models]
    y_true = (flaw_labels > 0.5).astype(int)
    unique_classes = len(np.unique(y_true))
    seed_metrics = []
    for p in probs_list:
        auc = float(roc_auc_score(y_true, p)) if unique_classes > 1 else float("nan")
        ap = float(average_precision_score(y_true, p)) if unique_classes > 1 else float("nan")
        seed_metrics.append({"auroc": auc, "average_precision": ap})
    summary = _calc_summary(seed_metrics)
    return probs_list, seed_metrics, summary


def _save_seed_models(
    models: list[nn.Module],
    held_out: str,
    seeds: tuple[int, ...],
    models_dir: Path,
) -> list[dict[str, Any]]:
    """Save trained seed models and return paths with sha256 digests."""
    models_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    meta = {"width": 32, "first_pool": (2, 2), "size": 256}
    for m, s in zip(models, seeds, strict=True):
        p = models_dir / f"robust_{held_out}_s{s}.pt"
        save_model(m, p, meta)
        saved.append({"seed": s, "path": str(p), "sha256": sha256_file(p)})
    return saved


def _train_fold_models(
    fold: LocoFold,
    cfg: RobustConfig,
    dev: torch.device,
) -> tuple[list[nn.Module], list[dict[str, Any]]]:
    """Train MlNdtNet models across all seeds for a single fold."""
    train_ds = IndexedMlNdtDataset(fold.train)
    val_ds = IndexedMlNdtDataset(fold.val)
    use_gpu = dev.type == "cuda"
    gpu_data = (
        (load_dataset_to_gpu(train_ds, dev), load_dataset_to_gpu(val_ds, dev)) if use_gpu else None
    )

    models, summaries = [], []
    for s in cfg.seeds:
        set_seed(s)
        m = MlNdtNet(width=32, first_pool=(2, 2)).to(dev)
        if use_gpu and gpu_data is not None:
            w, sm = _train_seed_gpu(m, gpu_data, cfg, s)
        else:
            w, sm = _train_seed_cpu(m, (train_ds, val_ds), cfg, s)
        m.load_state_dict(w)
        models.append(m)
        summaries.append(sm)
    return models, summaries


def _format_fold_record(
    fold: LocoFold,
    artifacts: tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]],
    ens_metrics: dict[str, Any],
    wall_sec: float,
) -> dict[str, Any]:
    """Construct fold dictionary payload."""
    saved_models, seed_metrics, seed_summary = artifacts
    n_flawed = ens_metrics["fcr_0.01"]["detections"]["n_flawed"]
    n_unflawed = ens_metrics["n_clean"]
    return {
        "held_out": fold.held_out,
        "seeds": [m["seed"] for m in saved_models],
        "seed_history": seed_metrics,
        "seed_metrics": seed_metrics,
        "seed_summary": seed_summary,
        "ensemble_metrics": ens_metrics,
        "models": saved_models,
        "n_train": len(fold.train),
        "n_val": len(fold.val),
        "n_test_flawed": n_flawed,
        "n_test_unflawed": n_unflawed,
        "wall_time_seconds": round(wall_sec, 2),
    }


def _run_fold(
    fold: LocoFold,
    cfg: RobustConfig,
    dev: torch.device,
) -> dict[str, Any]:
    """Train seeds, evaluate ensemble metrics, and save checkpoints for a fold."""
    t0 = time.time()
    models, train_summaries = _train_fold_models(fold, cfg, dev)
    saved_models = _save_seed_models(models, fold.held_out, cfg.seeds, cfg.models_dir)

    test_ds = IndexedMlNdtDataset(fold.test)
    flaw_labels, _ = test_ds.get_labels()
    probs_list, seed_metrics, seed_summary = _evaluate_seed_predictions(
        models, test_ds, dev, flaw_labels
    )
    for sm, tr in zip(seed_metrics, train_summaries, strict=True):
        sm.update(tr)

    ens_probs = np.mean(probs_list, axis=0) if probs_list else np.empty(0, dtype=np.float32)
    ens_metrics = compute_ensemble_metrics(ens_probs, flaw_labels)
    artifacts = (saved_models, seed_metrics, seed_summary)
    return _format_fold_record(fold, artifacts, ens_metrics, time.time() - t0)


def _resolve_batches(quick: bool) -> tuple[list[Path], list[Path]]:
    """Partition dataset into training and validation batch lists."""
    tr_all = list_batches("training")
    val_all = list_batches("validation")
    if quick:
        return tr_all[:6], val_all[:1]
    return tr_all, val_all


def _build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(description="Robust LOCO experiment")
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Random seeds"
    )
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--quick", action="store_true", help="Quick run with subset")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/loco_robust_metrics.json"),
        help="Output metrics path",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=Path("models"), help="Directory to save models"
    )
    return parser


def _dump_json(path: Path, data: dict[str, Any]) -> None:
    """Save dictionary to JSON file creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _resolve_config(args: argparse.Namespace) -> RobustConfig:
    """Resolve CLI options into typed RobustConfig."""
    epochs = 1 if args.quick and args.epochs == 40 else args.epochs
    seeds = (0,) if args.quick and args.seeds == [0, 1, 2, 3, 4] else tuple(args.seeds)
    return RobustConfig(
        device=args.device,
        seeds=seeds,
        epochs=epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        quick=args.quick,
        out_path=args.out,
        models_dir=args.models_dir,
    )


def _build_payload(
    cfg: RobustConfig,
    fold_records: list[dict[str, Any]],
    dev: torch.device,
    t_start: float,
) -> dict[str, Any]:
    """Construct JSON output payload dictionary."""
    gpu_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else None
    return {
        "label": "Real data (ML-NDT weld, 3 physical cracks)",
        "preregistration": "docs/decisions/0012-robust-loco-preregistration.md",
        "gpu_name": gpu_name,
        "device_name": str(dev),
        "wall_time_seconds": round(time.time() - t_start, 2),
        "config": {
            "device": cfg.device,
            "seeds": list(cfg.seeds),
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "patience": cfg.patience,
            "quick": cfg.quick,
            "models_dir": str(cfg.models_dir),
            "out": str(cfg.out_path),
        },
        "folds": fold_records,
    }


def main(argv: list[str] | None = None) -> None:
    """Execute complete robust LOCO training, evaluation, and export."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    cfg = _resolve_config(args)

    t_start = time.time()
    dev = pick_device(cfg.device)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True

    tr_b, val_b = _resolve_batches(cfg.quick)
    folds = make_folds(tr_b, val_b, seed=0, val_mode="batch")
    fold_records = [_run_fold(f, cfg, dev) for f in folds]

    payload = _build_payload(cfg, fold_records, dev, t_start)
    _dump_json(cfg.out_path, payload)
    print(f"Robust LOCO metrics saved to {cfg.out_path}", flush=True)


if __name__ == "__main__":
    main()
