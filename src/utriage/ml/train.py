"""Phase 3 ML-NDT training loop, loss functions, and evaluation helpers."""

from __future__ import annotations

import contextlib
import copy
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters and execution settings for model training."""

    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    seed: int = 0
    size_loss_weight: float = 0.1
    amp: bool = True
    device: str = "auto"
    num_workers: int = 0


@dataclass
class TrainResult:
    """Summary of training run, history, and selected best checkpoint."""

    best_epoch: int
    best_val_loss: float
    history: list[dict]
    seconds: float
    device: str


@dataclass
class _TrainState:
    """Tracks best checkpoint and early-stopping counter."""

    best_loss: float = float("inf")
    best_ep: int = 1
    best_weights: dict | None = None
    no_improve: int = 0

    def update(self, loss: float, epoch: int, weights: dict) -> None:
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_ep = epoch
            self.best_weights = copy.deepcopy(weights)
            self.no_improve = 0
            return
        self.no_improve += 1


def set_seed(seed: int) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def pick_device(pref: str = "auto") -> torch.device:
    """Resolve target device ('auto' selects CUDA when available, otherwise CPU)."""
    clean_pref = pref.strip().lower()
    if clean_pref == "auto":
        dev_str = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(dev_str)
    return torch.device(clean_pref)


def loss_fn(
    logit: torch.Tensor,
    size_pred: torch.Tensor,
    flaw: torch.Tensor,
    size: torch.Tensor,
    w: float,
) -> torch.Tensor:
    """Calculate combined flaw classification and conditional size regression loss."""
    bce = F.binary_cross_entropy_with_logits(logit.view(-1), flaw.view(-1).float())
    mask = flaw.view(-1) > 0.5
    if mask.any():
        size_loss = F.smooth_l1_loss(size_pred.view(-1)[mask], size.view(-1).float()[mask])
    else:
        size_loss = torch.zeros((), device=logit.device, dtype=logit.dtype)
    return bce + float(w) * size_loss


def _make_loaders(train_ds: Any, val_ds: Any, cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    """Build training and validation DataLoaders from datasets."""
    t_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    v_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    return t_loader, v_loader


def _make_opt_sched(model: nn.Module, cfg: TrainConfig) -> tuple[torch.optim.Optimizer, Any]:
    """Create AdamW optimizer and CosineAnnealingLR scheduler."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    return opt, sched


def _run_epoch(
    model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, cfg: TrainConfig
) -> float:
    """Execute one training epoch with optional mixed-precision autocast."""
    model.train()
    dev = next(model.parameters()).device
    use_amp = (dev.type == "cuda") and cfg.amp
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp
        else contextlib.nullcontext()
    )
    total_loss, total_count = 0.0, 0
    for x, flaw, size in loader:
        x, flaw, size = x.to(dev), flaw.to(dev), size.to(dev)
        opt.zero_grad()
        with amp_ctx:
            logit, size_pred = model(x)
            loss = loss_fn(logit, size_pred, flaw, size, cfg.size_loss_weight)
        if loss.requires_grad:
            loss.backward()
            opt.step()
        total_loss += loss.item() * len(x)
        total_count += len(x)
    return total_loss / max(1, total_count)


def _evaluate(
    model: nn.Module, loader: DataLoader, cfg: TrainConfig, dev: torch.device
) -> tuple[float, float]:
    """Compute validation loss and classification accuracy."""
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    with torch.no_grad():
        for x, flaw, size in loader:
            x, flaw, size = x.to(dev), flaw.to(dev), size.to(dev)
            logit, size_pred = model(x)
            loss = loss_fn(logit, size_pred, flaw, size, cfg.size_loss_weight)
            total_loss += loss.item() * len(x)
            pred_flaw = logit.view(-1) >= 0.0
            actual_flaw = flaw.view(-1) > 0.5
            total_correct += int((pred_flaw == actual_flaw).sum().item())
            total_count += len(x)
    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_correct / total_count


def _make_row(ep: int, tr: float, vl: float, acc: float, sec: float) -> dict:
    """Construct one epoch history row."""
    return {"epoch": ep, "train_loss": tr, "val_loss": vl, "val_acc": acc, "seconds": sec}


def _log_epoch(row: dict, total_epochs: int, device_str: str) -> None:
    """Print standard one-line summary for completed training epoch."""
    msg = (
        f"Epoch {row['epoch']:2d}/{total_epochs} [{device_str}]: "
        f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
        f"val_acc={row['val_acc']:.4f} ({row['seconds']:.2f}s)"
    )
    print(msg, flush=True)


def train_model(model: nn.Module, train_ds: Any, val_ds: Any, cfg: TrainConfig) -> TrainResult:
    """Train model using AdamW, CosineAnnealingLR, autocast, and early stopping."""
    set_seed(cfg.seed)
    dev = pick_device(cfg.device)
    model.to(dev)
    t_loader, v_loader = _make_loaders(train_ds, val_ds, cfg)
    opt, sched = _make_opt_sched(model, cfg)
    st = _TrainState(best_weights=copy.deepcopy(model.state_dict()))
    hist: list[dict] = []
    t_start = time.perf_counter()
    for ep in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        tr_loss = _run_epoch(model, t_loader, opt, cfg)
        vl_loss, vl_acc = _evaluate(model, v_loader, cfg, dev)
        sched.step()
        row = _make_row(ep, tr_loss, vl_loss, vl_acc, time.perf_counter() - t0)
        hist.append(row)
        _log_epoch(row, cfg.epochs, str(dev))
        st.update(vl_loss, ep, model.state_dict())
        if st.no_improve >= cfg.patience:
            break
    if st.best_weights is not None:
        model.load_state_dict(st.best_weights)
    return TrainResult(st.best_ep, st.best_loss, hist, time.perf_counter() - t_start, str(dev))


@torch.no_grad()
def predict(model: nn.Module, ds: Any, cfg: TrainConfig) -> tuple[np.ndarray, np.ndarray]:
    """Generate flaw probabilities and size estimates in dataset order."""
    model.eval()
    dev = pick_device(cfg.device)
    model.to(dev)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    prob_list: list[np.ndarray] = []
    size_list: list[np.ndarray] = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        logit, size_pred = model(x.to(dev))
        prob_list.append(torch.sigmoid(logit).view(-1).cpu().numpy())
        size_list.append(size_pred.view(-1).cpu().numpy())
    if not prob_list:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    return np.concatenate(prob_list, axis=0), np.concatenate(size_list, axis=0)
