"""Lightweight hybrid detector training and evaluation on ML-NDT LOCO folds.

Trains LiteNet CNN ensemble and CleanFeatureModel anomaly detector across
leave-one-crack-out folds, evaluates hybrid operating points and ranking metrics,
and benchmarks CPU latency against baseline MlNdtNet.
"""

# ruff: noqa: E402  (repo root is added to sys.path before the local imports)
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# repo root on sys.path so `experiments.*` imports work when run as a script
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
from experiments.run_hardware import get_cpu_name
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from utriage.integrity import sha256_file
from utriage.ml.datasets import list_batches
from utriage.ml.lite_hybrid import (
    CleanFeatureModel,
    LiteNet,
    augment_batch,
    clean_threshold,
    count_params,
    hybrid_flags,
    image_features,
)
from utriage.ml.loco import IndexedMlNdtDataset, LocoFold, make_folds
from utriage.ml.models import MlNdtNet
from utriage.ml.models import count_params as count_params_mlndt
from utriage.ml.train import pick_device, set_seed
from utriage.pod.metrics import clopper_pearson


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment configuration parameters."""

    device: str = "auto"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 40
    batch_size: int = 512
    patience: int = 6
    width: int = 16
    threads: int = 4
    quick: bool = False
    out: Path = Path("results/lite_hybrid_metrics.json")
    models_dir: Path = Path("models")


@dataclass(frozen=True)
class SeedTrainResult:
    """Summary of training run for a single seed."""

    held_out: str
    seed: int
    best_epoch: int
    best_val_loss: float
    seconds: float
    model_path: str
    model_sha256: str


@dataclass(frozen=True)
class TrainLoopArgs:
    """Arguments bundling execution steps for early-stopping training loop."""

    epochs: int
    patience: int
    train_step: Callable[[], float]
    val_step: Callable[[], float]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for lite hybrid experiment."""
    parser = argparse.ArgumentParser(description="Run lightweight hybrid LOCO experiment.")
    parser.add_argument(
        "--device", type=str, default="auto", help="Compute device (auto, cuda, cpu)"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Random seeds")
    parser.add_argument("--epochs", type=int, default=None, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--width", type=int, default=16, help="LiteNet channel multiplier")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for latency benchmark")
    parser.add_argument(
        "--quick", action="store_true", help="Quick run with subset and single seed"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/lite_hybrid_metrics.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=Path("models"), help="Directory to save models"
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Resolve argument defaults and construct ExperimentConfig."""
    epochs = args.epochs if args.epochs is not None else (1 if args.quick else 40)
    if args.seeds is not None:
        seeds = tuple(args.seeds)
    else:
        seeds = (0,) if args.quick else (0, 1, 2, 3, 4)
    return ExperimentConfig(
        device=args.device,
        seeds=seeds,
        epochs=epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        width=args.width,
        threads=args.threads,
        quick=args.quick,
        out=Path(args.out),
        models_dir=Path(args.models_dir),
    )


def resolve_folds(quick: bool = False) -> list[LocoFold]:
    """Load batch paths and partition into 3 leave-one-crack-out folds."""
    tr_all = list_batches("training")
    val_all = list_batches("validation")
    if quick:
        tr_all, val_all = tr_all[:6], val_all[:1]
    return make_folds(tr_all, val_all, seed=0, val_mode="batch")


def load_gpu_tensors(
    items: list[tuple[Path, int]],
    device: torch.device,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked loading of dataset items directly into float16 GPU tensors."""
    n = len(items)
    x_gpu = torch.empty((n, 1, 256, 256), dtype=torch.float16, device=device)
    y_gpu = torch.empty((n,), dtype=torch.float32, device=device)
    ds = IndexedMlNdtDataset(items)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        batch_x = [ds[i][0] for i in range(start, end)]
        batch_y = [ds[i][1] for i in range(start, end)]
        x_gpu[start:end] = torch.stack(batch_x).to(device=device, dtype=torch.float16)
        y_gpu[start:end] = torch.stack(batch_y).to(device=device, dtype=torch.float32)
    return x_gpu, y_gpu


def make_data_loader(
    items: list[tuple[Path, int]],
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    """Build PyTorch DataLoader streaming images from IndexedMlNdtDataset."""
    ds = IndexedMlNdtDataset(items)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_gpu_epoch(
    model: nn.Module,
    tensors: tuple[torch.Tensor, torch.Tensor],
    opt_scaler: tuple[torch.optim.Optimizer, torch.amp.GradScaler],
    gen_and_bs: tuple[torch.Generator, int],
) -> float:
    """Run one training epoch over preloaded GPU float16 tensors."""
    model.train()
    x_tr, y_tr = tensors
    opt, scaler = opt_scaler
    gen, bs = gen_and_bs
    n = x_tr.size(0)
    perm = torch.randperm(n, generator=gen, device=x_tr.device)
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss, steps = 0.0, 0
    for i in range(0, n, bs):
        idx = perm[i : i + bs]
        bx = augment_batch(x_tr[idx].float(), gen).to(dtype=torch.float16)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            loss = loss_fn(model(bx), y_tr[idx])
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total_loss += float(loss.item())
        steps += 1
    return total_loss / max(steps, 1)


def train_cpu_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    gen: torch.Generator,
) -> float:
    """Run one training epoch using CPU DataLoader."""
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss, steps = 0.0, 0
    for bx, by, _ in loader:
        bx_aug = augment_batch(bx, gen)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(bx_aug), by)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        steps += 1
    return total_loss / max(steps, 1)


@torch.inference_mode()
def eval_val_loss_gpu(
    model: nn.Module,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    batch_size: int,
) -> float:
    """Compute validation BCE loss across GPU float16 tensors."""
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total, steps = 0.0, 0
    for i in range(0, x_val.size(0), batch_size):
        bx = x_val[i : i + batch_size]
        by = y_val[i : i + batch_size]
        with torch.amp.autocast("cuda"):
            loss = loss_fn(model(bx), by)
        total += float(loss.item())
        steps += 1
    return total / max(steps, 1)


@torch.inference_mode()
def eval_val_loss_cpu(model: nn.Module, loader: DataLoader) -> float:
    """Compute validation BCE loss across CPU DataLoader."""
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total, steps = 0.0, 0
    for bx, by, _ in loader:
        loss = loss_fn(model(bx), by)
        total += float(loss.item())
        steps += 1
    return total / max(steps, 1)


def save_model_checkpoint(model: nn.Module, path: Path) -> str:
    """Save model state dict to disk and return computed SHA256 hex digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    sha = sha256_file(path)
    sha_path = path.with_suffix(".pt.sha256")
    sha_path.write_text(f"{sha}  {path.name}\n", encoding="utf-8")
    return sha


def _execute_training_loop(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    sched: Any,
    args: TrainLoopArgs,
) -> tuple[dict[str, Any], int, float]:
    """Execute training epochs with early stopping on validation loss."""
    best_weights = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 1
    no_improve = 0
    for ep in range(1, args.epochs + 1):
        _ = args.train_step()
        val_loss = args.val_step()
        sched.step()
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = ep
            best_weights = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break
    return best_weights, best_epoch, best_loss


def _prepare_gpu_loop_args(
    fold: LocoFold,
    cfg: ExperimentConfig,
    dev: torch.device,
    model_opt_gen: tuple[nn.Module, torch.optim.Optimizer, torch.Generator],
) -> TrainLoopArgs:
    """Build GPU training loop closures."""
    model, opt, gen = model_opt_gen
    torch.backends.cudnn.benchmark = True
    x_tr, y_tr = load_gpu_tensors(fold.train, dev, cfg.batch_size)
    x_va, y_va = load_gpu_tensors(fold.val, dev, cfg.batch_size)
    scaler = torch.amp.GradScaler("cuda")

    def t_fn_gpu() -> float:
        return train_gpu_epoch(model, (x_tr, y_tr), (opt, scaler), (gen, cfg.batch_size))

    def v_fn_gpu() -> float:
        return eval_val_loss_gpu(model, x_va, y_va, cfg.batch_size)

    return TrainLoopArgs(cfg.epochs, cfg.patience, t_fn_gpu, v_fn_gpu)


def _prepare_cpu_loop_args(
    fold: LocoFold,
    cfg: ExperimentConfig,
    model_opt_gen: tuple[nn.Module, torch.optim.Optimizer, torch.Generator],
) -> TrainLoopArgs:
    """Build CPU training loop closures."""
    model, opt, gen = model_opt_gen
    tr_ld = make_data_loader(fold.train, cfg.batch_size, shuffle=True)
    va_ld = make_data_loader(fold.val, cfg.batch_size, shuffle=False)

    def t_fn_cpu() -> float:
        return train_cpu_epoch(model, tr_ld, opt, gen)

    def v_fn_cpu() -> float:
        return eval_val_loss_cpu(model, va_ld)

    return TrainLoopArgs(cfg.epochs, cfg.patience, t_fn_cpu, v_fn_cpu)


def _prepare_train_loop_args(
    fold: LocoFold,
    cfg: ExperimentConfig,
    dev: torch.device,
    model_opt_gen: tuple[nn.Module, torch.optim.Optimizer, torch.Generator],
) -> TrainLoopArgs:
    """Build training and validation closures for GPU or CPU execution."""
    if dev.type == "cuda":
        return _prepare_gpu_loop_args(fold, cfg, dev, model_opt_gen)
    return _prepare_cpu_loop_args(fold, cfg, model_opt_gen)


def _make_seed_result(
    held_out: str,
    seed: int,
    stats: tuple[int, float, float],
    ckpt_info: tuple[Path, str],
) -> SeedTrainResult:
    """Construct SeedTrainResult record."""
    best_ep, best_loss, wall_s = stats
    ckpt_path, sha = ckpt_info
    return SeedTrainResult(
        held_out=held_out,
        seed=seed,
        best_epoch=best_ep,
        best_val_loss=best_loss,
        seconds=wall_s,
        model_path=str(ckpt_path),
        model_sha256=sha,
    )


def train_single_seed(
    fold: LocoFold,
    seed: int,
    cfg: ExperimentConfig,
    dev: torch.device,
) -> tuple[nn.Module, SeedTrainResult]:
    """Train LiteNet on GPU/CPU for a single seed with AdamW and cosine LR."""
    set_seed(seed)
    gen = torch.Generator(device=dev if dev.type == "cuda" else "cpu")
    gen.manual_seed(seed)
    model = LiteNet(width=cfg.width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=1e-6)
    t0 = time.perf_counter()

    loop_args = _prepare_train_loop_args(fold, cfg, dev, (model, opt, gen))
    best_w, best_ep, best_loss = _execute_training_loop(model, opt, sched, loop_args)
    model.load_state_dict(best_w)
    wall_s = float(time.perf_counter() - t0)

    ckpt_path = cfg.models_dir / f"lite_{fold.held_out}_s{seed}.pt"
    sha = save_model_checkpoint(model, ckpt_path)
    res = _make_seed_result(fold.held_out, seed, (best_ep, best_loss, wall_s), (ckpt_path, sha))
    return model, res


@torch.inference_mode()
def ensemble_predict_cnn(
    models: list[nn.Module],
    items: list[tuple[Path, int]],
    dev: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Compute mean sigmoid probability across seed models for dataset items."""
    loader = make_data_loader(items, batch_size=batch_size, shuffle=False)
    seed_probs: list[np.ndarray] = []
    for model in models:
        model.eval()
        batches: list[np.ndarray] = []
        for bx, _, _ in loader:
            p = torch.sigmoid(model(bx.to(dev))).detach().cpu().numpy()
            batches.append(np.atleast_1d(p))
        seed_probs.append(np.concatenate(batches))
    return np.mean(seed_probs, axis=0)


def extract_dataset_features(
    items: list[tuple[Path, int]],
    chunk_size: int = 512,
) -> np.ndarray:
    """Extract physics features from items in chunks to respect host RAM limits."""
    ds = IndexedMlNdtDataset(items)
    n = len(items)
    chunks: list[np.ndarray] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_imgs = np.stack([ds[i][0].squeeze(0).numpy() for i in range(start, end)])
        chunks.append(image_features(chunk_imgs))
    return np.vstack(chunks)


def extract_clean_train_features(
    train_items: list[tuple[Path, int]],
) -> np.ndarray:
    """Extract physics features for clean training items only."""
    ds = IndexedMlNdtDataset(train_items)
    flaws, _ = ds.get_labels()
    clean_items = [item for item, fl in zip(train_items, flaws, strict=True) if fl == 0.0]
    if not clean_items:
        raise ValueError("No clean training items available.")
    return extract_dataset_features(clean_items)


def split_clean_test_scans(
    flaws: np.ndarray,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split clean test scans 50/50 into disjoint clean_calib and clean_eval splits."""
    flawed_idx = np.where(flaws == 1.0)[0]
    clean_idx = np.where(flaws == 0.0)[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(clean_idx)
    half = len(perm) // 2
    calib = perm[:half]
    ev = perm[half:]
    if set(calib).intersection(set(ev)):
        raise AssertionError("clean_calib and clean_eval are not disjoint.")
    return flawed_idx, calib, ev


def _rate_and_ci(k: int, n: int) -> dict[str, Any]:
    """Compute rate and Clopper-Pearson 95% confidence interval."""
    rate = float(k / n) if n > 0 else 0.0
    low, high = clopper_pearson(k, n, conf=0.95)
    return {
        "k": k,
        "n": n,
        "rate": rate,
        "ci_low": low,
        "ci_high": high,
        "method": "Clopper-Pearson two-sided 95",
    }


def compute_detection_and_false_calls(
    flags: np.ndarray,
    flawed_idx: np.ndarray,
    clean_eval: np.ndarray,
) -> dict[str, Any]:
    """Compute detection and false call counts, rates, and 95% Clopper-Pearson CIs."""
    k_det, n_det = int(np.sum(flags[flawed_idx])), int(len(flawed_idx))
    k_fc, n_fc = int(np.sum(flags[clean_eval])), int(len(clean_eval))
    return {
        "detection": _rate_and_ci(k_det, n_det),
        "false_calls": _rate_and_ci(k_fc, n_fc),
    }


def evaluate_target_fcr(
    scores: tuple[np.ndarray, np.ndarray],
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    target: float,
) -> dict[str, Any]:
    """Evaluate CNN, Anomaly, and Hybrid detectors at a target false call rate."""
    p_cnn, anom = scores
    flawed_idx, calib, ev = indices
    t_cnn = clean_threshold(p_cnn[calib], target)
    m_cnn = compute_detection_and_false_calls(p_cnn >= t_cnn, flawed_idx, ev)
    m_cnn["threshold"] = t_cnn

    t_anom = clean_threshold(anom[calib], target)
    m_anom = compute_detection_and_false_calls(anom >= t_anom, flawed_idx, ev)
    m_anom["threshold"] = t_anom

    t_cnn_h = clean_threshold(p_cnn[calib], target / 2.0)
    t_anom_h = clean_threshold(anom[calib], target / 2.0)
    flags_hyb = hybrid_flags(p_cnn, anom, t_cnn_h, t_anom_h)
    m_hyb = compute_detection_and_false_calls(flags_hyb, flawed_idx, ev)
    m_hyb["threshold_cnn"] = t_cnn_h
    m_hyb["threshold_anom"] = t_anom_h

    return {
        "target_fcr": target,
        "cnn_only": m_cnn,
        "anomaly_only": m_anom,
        "hybrid": m_hyb,
    }


def map_scores_to_cdf(scores: np.ndarray, cal_scores: np.ndarray) -> np.ndarray:
    """Map scores to [0, 1] using empirical CDF of calibration clean scores."""
    sorted_cal = np.sort(np.asarray(cal_scores, dtype=float).ravel())
    if sorted_cal.size == 0:
        return np.zeros_like(scores, dtype=float)
    ranks = np.searchsorted(sorted_cal, np.asarray(scores, dtype=float), side="right")
    return np.clip(ranks / float(sorted_cal.size), 0.0, 1.0)


def compute_ranking_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Compute AUROC and average precision with binary class check."""
    if len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "average_precision": float("nan")}
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
    }


def evaluate_threshold_free(
    scores: tuple[np.ndarray, np.ndarray],
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """Compute AUROC and AP for CNN, Anomaly, and fused scores on evaluation set."""
    p_cnn, anom = scores
    flawed_idx, calib, ev = indices
    eval_idx = np.concatenate([flawed_idx, ev])
    y_true = np.concatenate([np.ones(len(flawed_idx), dtype=int), np.zeros(len(ev), dtype=int)])
    ecdf_c = map_scores_to_cdf(p_cnn, p_cnn[calib])
    ecdf_a = map_scores_to_cdf(anom, anom[calib])
    fused = np.maximum(ecdf_c, ecdf_a)
    return {
        "cnn": compute_ranking_metrics(y_true, p_cnn[eval_idx]),
        "anomaly": compute_ranking_metrics(y_true, anom[eval_idx]),
        "fused": compute_ranking_metrics(y_true, fused[eval_idx]),
    }


def _timed_loop(fn: Callable[[], Any], warmup: int, runs: int) -> dict[str, float]:
    """Run warmup and timed iterations, returning median and p95 latency in ms."""
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
    }


def _benchmark_models(
    lite: nn.Module,
    mlndt: nn.Module,
    x: torch.Tensor,
    spec: tuple[int, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Benchmark forward pass latency for LiteNet and baseline MlNdtNet."""
    warmup, runs = spec

    def _bench_lite() -> None:
        with torch.inference_mode():
            _ = lite(x)

    def _bench_mlndt() -> None:
        with torch.inference_mode():
            _ = mlndt(x)

    m_lite = _timed_loop(_bench_lite, warmup, runs)
    m_lite["params"] = count_params(lite)
    m_mlndt = _timed_loop(_bench_mlndt, warmup, runs)
    m_mlndt["params"] = count_params_mlndt(mlndt)
    return m_lite, m_mlndt


def _benchmark_anomaly(
    lite: nn.Module,
    anom_model: CleanFeatureModel,
    inputs: tuple[torch.Tensor, np.ndarray],
    spec: tuple[int, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Benchmark anomaly scoring and combined hybrid forward latency."""
    x, img = inputs
    warmup, runs = spec

    def _bench_anom() -> None:
        _ = anom_model.score(image_features(img[np.newaxis, ...]))

    def _bench_total() -> None:
        with torch.inference_mode():
            _ = lite(x)
        _ = anom_model.score(image_features(img[np.newaxis, ...]))

    m_anom = _timed_loop(_bench_anom, warmup, runs)
    m_total = _timed_loop(_bench_total, warmup, runs)
    return m_anom, m_total


def run_latency_benchmarks(
    anom_model: CleanFeatureModel,
    width: int,
    threads: int,
    quick: bool = False,
) -> dict[str, Any]:
    """Measure single-image CPU latency for LiteNet, Anomaly, Total, and MlNdtNet."""
    torch.set_num_threads(threads)
    warmup, runs = (2, 5) if quick else (20, 300)
    lite = LiteNet(width=width).to("cpu").eval()
    mlndt = MlNdtNet(width=32, first_pool=(2, 2)).to("cpu").eval()
    x = torch.randn(1, 1, 256, 256, dtype=torch.float32)
    img = np.random.default_rng(0).normal(size=(256, 256)).astype(np.float32)

    m_lite, m_mlndt = _benchmark_models(lite, mlndt, x, (warmup, runs))
    m_anom, m_total = _benchmark_anomaly(lite, anom_model, (x, img), (warmup, runs))
    note = (
        f"measured on the development machine CPU limited to {threads} threads, "
        "not on the target i5 laptop"
    )
    return {
        "cpu_name": get_cpu_name() or "Unknown CPU",
        "threads": threads,
        "note": note,
        "litenet": m_lite,
        "anomaly": m_anom,
        "total": m_total,
        "mlndtnet": m_mlndt,
    }


def evaluate_single_fold(
    fold: LocoFold,
    models: list[nn.Module],
    anom_model: CleanFeatureModel,
    dev: torch.device,
) -> dict[str, Any]:
    """Evaluate ensemble LiteNet and CleanFeatureModel on held-out test fold."""
    flaws, _ = IndexedMlNdtDataset(fold.test).get_labels()
    flawed_idx, calib, ev = split_clean_test_scans(flaws, seed=0)

    p_cnn = ensemble_predict_cnn(models, fold.test, dev)
    feats_test = extract_dataset_features(fold.test)
    anom = anom_model.score(feats_test)
    scores, indices = (p_cnn, anom), (flawed_idx, calib, ev)

    return {
        "held_out": fold.held_out,
        "n_train": len(fold.train),
        "n_val": len(fold.val),
        "n_test_flawed": len(flawed_idx),
        "n_clean_calib": len(calib),
        "n_clean_eval": len(ev),
        "operating_points": {
            "fcr_0.01": evaluate_target_fcr(scores, indices, target=0.01),
            "fcr_0.05": evaluate_target_fcr(scores, indices, target=0.05),
        },
        "threshold_free": evaluate_threshold_free(scores, indices),
    }


def _process_all_folds(
    folds: list[LocoFold],
    cfg: ExperimentConfig,
    dev: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], CleanFeatureModel]:
    """Train models, fit anomaly detector, and evaluate metrics across folds."""
    all_summary: list[dict[str, Any]] = []
    fold_metrics: dict[str, Any] = {}
    last_anom: CleanFeatureModel | None = None
    for fold in folds:
        models: list[nn.Module] = []
        for s in cfg.seeds:
            m, res = train_single_seed(fold, s, cfg, dev)
            models.append(m)
            all_summary.append(asdict(res))
        clean_feats = extract_clean_train_features(fold.train)
        anom_model = CleanFeatureModel().fit(clean_feats)
        last_anom = anom_model
        fold_metrics[fold.held_out] = evaluate_single_fold(fold, models, anom_model, dev)
    if last_anom is None:
        raise ValueError("No LOCO folds to evaluate.")
    return fold_metrics, all_summary, last_anom


def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    """Execute full leave-one-crack-out lightweight hybrid pipeline."""
    t_start = time.perf_counter()
    dev = pick_device(cfg.device)
    folds = resolve_folds(cfg.quick)
    fold_metrics, all_summary, last_anom = _process_all_folds(folds, cfg, dev)
    lat_block = run_latency_benchmarks(last_anom, cfg.width, cfg.threads, cfg.quick)
    wall_s = float(time.perf_counter() - t_start)
    gpu_name = str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu"

    cfg_dict = asdict(cfg)
    cfg_dict["out"] = str(cfg.out)
    cfg_dict["models_dir"] = str(cfg.models_dir)
    return {
        "label": "Real data (ML-NDT weld, 3 physical cracks)",
        "preregistration": "docs/decisions/0013-lite-hybrid-preregistration.md",
        "config": cfg_dict,
        "gpu_name": gpu_name,
        "wall_time_s": wall_s,
        "training_summary": all_summary,
        "folds": fold_metrics,
        "latency": lat_block,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for lightweight hybrid experiment."""
    args = parse_args(argv)
    cfg = build_config(args)
    payload = run_experiment(cfg)
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    cfg.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
