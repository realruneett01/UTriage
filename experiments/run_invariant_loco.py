"""Two-stage training of an invariant crack representation on ML-NDT LOCO folds.

Stage 1: Pretrains LiteEncoder and ProjectionHead with SimCLR NT-Xent and SupCon losses.
Stage 2: Fine-tunes InvariantClassifier with two learning rates (encoder 1e-4, head 1e-3).
Anomaly: CleanFeatureModel fit on clean training embeddings (seed 0).
Evaluation: Evaluates ensemble probability, anomaly, and fused scores on held-out test fold.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Configure local workspace paths
_EXP_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _EXP_DIR.parent
_SRC_DIR = _ROOT_DIR / "src"
for _p in (_ROOT_DIR, _SRC_DIR, _EXP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_lite_hybrid import (  # noqa: E402
    compute_detection_and_false_calls,
    compute_ranking_metrics,
    eval_val_loss_cpu,
    eval_val_loss_gpu,
    load_gpu_tensors,
    make_data_loader,
    map_scores_to_cdf,
    split_clean_test_scans,
)

from utriage.integrity import sha256_file  # noqa: E402
from utriage.ml.datasets import list_batches  # noqa: E402
from utriage.ml.lite_hybrid import CleanFeatureModel, clean_threshold  # noqa: E402
from utriage.ml.loco import IndexedMlNdtDataset, LocoFold, make_folds  # noqa: E402
from utriage.ml.train import pick_device, set_seed  # noqa: E402

try:
    from utriage.ml.invariance import (
        InvariantClassifier,
        LiteEncoder,
        ProjectionHead,
        nt_xent_loss,
        physics_view,
        supcon_loss,
    )
except ImportError:  # module written in parallel; imported in tests or callers
    pass


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment configuration parameters."""

    device: str = "auto"
    seeds: tuple[int, ...] = (0, 1, 2)
    pretrain_epochs: int = 30
    finetune_epochs: int = 30
    batch_size: int = 256
    patience: int = 6
    width: int = 16
    temperature: float = 0.2
    quick: bool = False
    out: Path = Path("results/invariant_loco_metrics.json")
    models_dir: Path = Path("models")


@dataclass(frozen=True)
class TrainLoopArgs:
    """Arguments bundling execution steps for early-stopping training loop."""

    epochs: int
    patience: int
    train_step: Any
    val_step: Any


class PretrainWrapper(nn.Module):
    """Container bundling LiteEncoder and ProjectionHead for Stage 1 pretraining."""

    def __init__(self, encoder: LiteEncoder, head: ProjectionHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for invariant LOCO experiment."""
    parser = argparse.ArgumentParser(description="Run invariant representation LOCO experiment.")
    parser.add_argument(
        "--device", type=str, default="auto", help="Compute device (auto, cuda, cpu)"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Random seeds")
    parser.add_argument("--pretrain-epochs", type=int, default=None, help="Stage 1 pretrain epochs")
    parser.add_argument("--finetune-epochs", type=int, default=None, help="Stage 2 finetune epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--width", type=int, default=16, help="LiteNet channel multiplier")
    parser.add_argument("--temperature", type=float, default=0.2, help="Loss temperature")
    parser.add_argument("--quick", action="store_true", help="Quick run with subset")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/invariant_loco_metrics.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=Path("models"), help="Directory to save models"
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Resolve argument defaults and construct ExperimentConfig."""
    pre_epochs = (
        args.pretrain_epochs if args.pretrain_epochs is not None else (1 if args.quick else 30)
    )
    ft_epochs = (
        args.finetune_epochs if args.finetune_epochs is not None else (1 if args.quick else 30)
    )
    seeds = tuple(args.seeds) if args.seeds is not None else ((0,) if args.quick else (0, 1, 2))
    return ExperimentConfig(
        device=args.device,
        seeds=seeds,
        pretrain_epochs=pre_epochs,
        finetune_epochs=ft_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        width=args.width,
        temperature=args.temperature,
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
    folds = make_folds(tr_all, val_all, seed=0, val_mode="batch")
    for f in folds:
        s_tr, s_val, s_te = set(f.train), set(f.val), set(f.test)
        if not s_te.isdisjoint(s_tr) or not s_te.isdisjoint(s_val):
            raise AssertionError(f"Held-out test overlap in fold {f.held_out}")
    return folds


def build_finetune_optimizer(model: InvariantClassifier) -> torch.optim.AdamW:
    """Build AdamW optimizer with two parameter groups: encoder 1e-4, head 1e-3."""
    groups = [
        {"params": list(model.encoder.parameters()), "lr": 1e-4},
        {"params": list(model.head.parameters()), "lr": 1e-3},
    ]
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def save_invariant_checkpoint(model: nn.Module, path: Path, width: int) -> str:
    """Save model checkpoint dictionary and computed SHA256 hex digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "width": width}
    torch.save(payload, path)
    sha = sha256_file(path)
    sha_path = path.with_suffix(".pt.sha256")
    sha_path.write_text(f"{sha}  {path.name}\n", encoding="utf-8")
    return sha


def _execute_early_stopping(
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
            continue
        no_improve += 1
        if no_improve >= args.patience:
            break
    return best_weights, best_epoch, best_loss


def _pretrain_gpu_step(
    wrapper: PretrainWrapper,
    views_labels: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    opt_scaler: tuple[torch.optim.Optimizer, torch.amp.GradScaler],
    temp: float,
) -> float:
    """Execute one AMP forward-backward step for pretraining."""
    v1, v2, y = views_labels
    opt, scaler = opt_scaler
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda"):
        z1, z2 = wrapper(v1), wrapper(v2)
        z_cat = torch.cat([z1, z2], dim=0)
        y_cat = torch.cat([y, y], dim=0).long()
        loss = nt_xent_loss(z1, z2, temp) + supcon_loss(z_cat, y_cat, temp)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    return float(loss.item())


def _train_pretrain_gpu_epoch(
    wrapper: PretrainWrapper,
    tensors: tuple[torch.Tensor, torch.Tensor],
    opt_scaler: tuple[torch.optim.Optimizer, torch.amp.GradScaler],
    ctx: tuple[torch.Generator, int, float],
) -> float:
    """Run one Stage 1 pretraining epoch over GPU float16 tensors."""
    wrapper.train()
    x_tr, y_tr = tensors
    gen, bs, temp = ctx
    perm = torch.randperm(x_tr.size(0), generator=gen, device=x_tr.device)
    total_loss, steps = 0.0, 0
    for i in range(0, x_tr.size(0), bs):
        idx = perm[i : i + bs]
        bx = x_tr[idx].float()
        v1 = physics_view(bx, gen).to(dtype=torch.float16)
        v2 = physics_view(bx, gen).to(dtype=torch.float16)
        step_loss = _pretrain_gpu_step(wrapper, (v1, v2, y_tr[idx]), opt_scaler, temp)
        total_loss += step_loss
        steps += 1
    return total_loss / max(steps, 1)


def _train_pretrain_cpu_epoch(
    wrapper: PretrainWrapper,
    loader: Any,
    opt: torch.optim.Optimizer,
    gen_temp: tuple[torch.Generator, float],
) -> float:
    """Run one Stage 1 pretraining epoch using CPU DataLoader."""
    wrapper.train()
    gen, temp = gen_temp
    total_loss, steps = 0.0, 0
    for bx, by, _ in loader:
        v1 = physics_view(bx, gen)
        v2 = physics_view(bx, gen)
        opt.zero_grad(set_to_none=True)
        z1 = wrapper(v1)
        z2 = wrapper(v2)
        z_cat = torch.cat([z1, z2], dim=0)
        y_cat = torch.cat([by, by], dim=0).long()
        loss = nt_xent_loss(z1, z2, temp) + supcon_loss(z_cat, y_cat, temp)
        loss.backward()
        opt.step()
        total_loss += float(loss.item())
        steps += 1
    return total_loss / max(steps, 1)


@torch.inference_mode()
def _eval_pretrain_loss_gpu(
    wrapper: PretrainWrapper,
    tensors: tuple[torch.Tensor, torch.Tensor],
    ctx: tuple[torch.Generator, int, float],
) -> float:
    """Compute validation contrastive loss on GPU float16 tensors."""
    wrapper.eval()
    x_va, y_va = tensors
    gen, bs, temp = ctx
    gen.manual_seed(0)
    total, steps = 0.0, 0
    for i in range(0, x_va.size(0), bs):
        bx = x_va[i : i + bs].float()
        v1 = physics_view(bx, gen).to(dtype=torch.float16)
        v2 = physics_view(bx, gen).to(dtype=torch.float16)
        with torch.amp.autocast("cuda"):
            z1 = wrapper(v1)
            z2 = wrapper(v2)
            z_cat = torch.cat([z1, z2], dim=0)
            y_cat = torch.cat([y_va[i : i + bs], y_va[i : i + bs]], dim=0).long()
            loss = nt_xent_loss(z1, z2, temp) + supcon_loss(z_cat, y_cat, temp)
        total += float(loss.item())
        steps += 1
    return total / max(steps, 1)


@torch.inference_mode()
def _eval_pretrain_loss_cpu(
    wrapper: PretrainWrapper,
    loader: Any,
    gen_temp: tuple[torch.Generator, float],
) -> float:
    """Compute validation contrastive loss on CPU DataLoader."""
    wrapper.eval()
    gen, temp = gen_temp
    gen.manual_seed(0)
    total, steps = 0.0, 0
    for bx, by, _ in loader:
        v1 = physics_view(bx, gen)
        v2 = physics_view(bx, gen)
        z1 = wrapper(v1)
        z2 = wrapper(v2)
        z_cat = torch.cat([z1, z2], dim=0)
        y_cat = torch.cat([by, by], dim=0).long()
        loss = nt_xent_loss(z1, z2, temp) + supcon_loss(z_cat, y_cat, temp)
        total += float(loss.item())
        steps += 1
    return total / max(steps, 1)


def _train_finetune_gpu_epoch(
    model: nn.Module,
    tensors: tuple[torch.Tensor, torch.Tensor],
    opt_scaler: tuple[torch.optim.Optimizer, torch.amp.GradScaler],
    gen_bs: tuple[torch.Generator, int],
) -> float:
    """Run one Stage 2 fine-tuning epoch over GPU float16 tensors."""
    model.train()
    x_tr, y_tr = tensors
    opt, scaler = opt_scaler
    gen, bs = gen_bs
    n = x_tr.size(0)
    perm = torch.randperm(n, generator=gen, device=x_tr.device)
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss, steps = 0.0, 0
    for i in range(0, n, bs):
        idx = perm[i : i + bs]
        bx = x_tr[idx].float()
        bx_aug = physics_view(bx, gen).to(dtype=torch.float16)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            loss = loss_fn(model(bx_aug), y_tr[idx])
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total_loss += float(loss.item())
        steps += 1
    return total_loss / max(steps, 1)


def _train_finetune_cpu_epoch(
    model: nn.Module,
    loader: Any,
    opt: torch.optim.Optimizer,
    gen: torch.Generator,
) -> float:
    """Run one Stage 2 fine-tuning epoch using CPU DataLoader."""
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss, steps = 0.0, 0
    for bx, by, _ in loader:
        bx_aug = physics_view(bx, gen)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(bx_aug), by)
        loss.backward()
        opt.step()
        total_loss += float(loss.item())
        steps += 1
    return total_loss / max(steps, 1)


def _prep_pretrain_gpu_loop(
    wrapper: PretrainWrapper,
    fold_cfg: tuple[LocoFold, ExperimentConfig],
    dev: torch.device,
    opt_gen: tuple[torch.optim.Optimizer, torch.Generator],
) -> TrainLoopArgs:
    """Build GPU closures for Stage 1 training loop."""
    fold, cfg = fold_cfg
    opt, gen = opt_gen
    torch.backends.cudnn.benchmark = True
    x_tr, y_tr = load_gpu_tensors(fold.train, dev, cfg.batch_size)
    x_va, y_va = load_gpu_tensors(fold.val, dev, cfg.batch_size)
    scaler = torch.amp.GradScaler("cuda")
    ctx = (gen, cfg.batch_size, cfg.temperature)

    def t_fn() -> float:
        return _train_pretrain_gpu_epoch(wrapper, (x_tr, y_tr), (opt, scaler), ctx)

    def v_fn() -> float:
        return _eval_pretrain_loss_gpu(wrapper, (x_va, y_va), ctx)

    return TrainLoopArgs(cfg.pretrain_epochs, cfg.patience, t_fn, v_fn)


def _prep_pretrain_cpu_loop(
    wrapper: PretrainWrapper,
    fold_cfg: tuple[LocoFold, ExperimentConfig],
    opt_gen: tuple[torch.optim.Optimizer, torch.Generator],
) -> TrainLoopArgs:
    """Build CPU closures for Stage 1 training loop."""
    fold, cfg = fold_cfg
    opt, gen = opt_gen
    tr_ld = make_data_loader(fold.train, cfg.batch_size, shuffle=True)
    va_ld = make_data_loader(fold.val, cfg.batch_size, shuffle=False)
    ctx_c = (gen, cfg.temperature)

    def t_fn() -> float:
        return _train_pretrain_cpu_epoch(wrapper, tr_ld, opt, ctx_c)

    def v_fn() -> float:
        return _eval_pretrain_loss_cpu(wrapper, va_ld, ctx_c)

    return TrainLoopArgs(cfg.pretrain_epochs, cfg.patience, t_fn, v_fn)


def _prepare_pretrain_loop_args(
    fold: LocoFold,
    cfg: ExperimentConfig,
    dev: torch.device,
    model_opt_gen: tuple[PretrainWrapper, torch.optim.Optimizer, torch.Generator],
) -> TrainLoopArgs:
    """Prepare training and validation closures for Stage 1."""
    wrapper, opt, gen = model_opt_gen
    if dev.type == "cuda":
        return _prep_pretrain_gpu_loop(wrapper, (fold, cfg), dev, (opt, gen))
    return _prep_pretrain_cpu_loop(wrapper, (fold, cfg), (opt, gen))


def _prep_finetune_gpu_loop(
    model: nn.Module,
    fold_cfg: tuple[LocoFold, ExperimentConfig],
    dev: torch.device,
    opt_gen: tuple[torch.optim.Optimizer, torch.Generator],
) -> TrainLoopArgs:
    """Build GPU closures for Stage 2 fine-tuning loop."""
    fold, cfg = fold_cfg
    opt, gen = opt_gen
    torch.backends.cudnn.benchmark = True
    x_tr, y_tr = load_gpu_tensors(fold.train, dev, cfg.batch_size)
    x_va, y_va = load_gpu_tensors(fold.val, dev, cfg.batch_size)
    scaler = torch.amp.GradScaler("cuda")

    def t_fn() -> float:
        return _train_finetune_gpu_epoch(model, (x_tr, y_tr), (opt, scaler), (gen, cfg.batch_size))

    def v_fn() -> float:
        return eval_val_loss_gpu(model, x_va, y_va, cfg.batch_size)

    return TrainLoopArgs(cfg.finetune_epochs, cfg.patience, t_fn, v_fn)


def _prep_finetune_cpu_loop(
    model: nn.Module,
    fold_cfg: tuple[LocoFold, ExperimentConfig],
    opt_gen: tuple[torch.optim.Optimizer, torch.Generator],
) -> TrainLoopArgs:
    """Build CPU closures for Stage 2 fine-tuning loop."""
    fold, cfg = fold_cfg
    opt, gen = opt_gen
    tr_ld = make_data_loader(fold.train, cfg.batch_size, shuffle=True)
    va_ld = make_data_loader(fold.val, cfg.batch_size, shuffle=False)

    def t_fn() -> float:
        return _train_finetune_cpu_epoch(model, tr_ld, opt, gen)

    def v_fn() -> float:
        return eval_val_loss_cpu(model, va_ld)

    return TrainLoopArgs(cfg.finetune_epochs, cfg.patience, t_fn, v_fn)


def _prepare_finetune_loop_args(
    fold: LocoFold,
    cfg: ExperimentConfig,
    dev: torch.device,
    model_opt_gen: tuple[nn.Module, torch.optim.Optimizer, torch.Generator],
) -> TrainLoopArgs:
    """Prepare training and validation closures for Stage 2."""
    model, opt, gen = model_opt_gen
    if dev.type == "cuda":
        return _prep_finetune_gpu_loop(model, (fold, cfg), dev, (opt, gen))
    return _prep_finetune_cpu_loop(model, (fold, cfg), (opt, gen))


def train_pretrain_seed(
    fold: LocoFold,
    seed: int,
    cfg: ExperimentConfig,
    dev: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Stage 1: Pretrain LiteEncoder + ProjectionHead on invariant loss."""
    set_seed(seed)
    gen = torch.Generator(device=dev if dev.type == "cuda" else "cpu")
    gen.manual_seed(seed)

    encoder = LiteEncoder(width=cfg.width).to(dev)
    head = ProjectionHead(in_dim=encoder.out_dim, out_dim=64).to(dev)
    wrapper = PretrainWrapper(encoder, head)
    opt = torch.optim.AdamW(wrapper.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.pretrain_epochs, eta_min=1e-6)

    t0 = time.perf_counter()
    loop_args = _prepare_pretrain_loop_args(fold, cfg, dev, (wrapper, opt, gen))
    best_w, best_ep, best_loss = _execute_early_stopping(wrapper, opt, sched, loop_args)
    wrapper.load_state_dict(best_w)
    wall_s = float(time.perf_counter() - t0)

    summary = {"best_epoch": best_ep, "best_loss": float(best_loss), "seconds": wall_s}
    return encoder, summary


def _format_finetune_summary(
    stats: tuple[int, float, float],
    ckpt_info: tuple[Path, str],
) -> dict[str, Any]:
    """Construct summary record for fine-tuning stage."""
    best_ep, best_loss, wall_s = stats
    ckpt_path, sha = ckpt_info
    return {
        "best_epoch": best_ep,
        "best_loss": float(best_loss),
        "seconds": wall_s,
        "model_path": str(ckpt_path),
        "model_sha256": sha,
    }


def train_finetune_seed(
    fold: LocoFold,
    encoder: nn.Module,
    seed: int,
    cfg_dev: tuple[ExperimentConfig, torch.device],
) -> tuple[nn.Module, dict[str, Any]]:
    """Stage 2: Fine-tune InvariantClassifier with two learning rates."""
    cfg, dev = cfg_dev
    set_seed(seed)
    gen = torch.Generator(device=dev if dev.type == "cuda" else "cpu")
    gen.manual_seed(seed)

    classifier = InvariantClassifier(encoder).to(dev)
    opt = build_finetune_optimizer(classifier)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.finetune_epochs, eta_min=1e-6)

    t0 = time.perf_counter()
    loop_args = _prepare_finetune_loop_args(fold, cfg, dev, (classifier, opt, gen))
    best_w, best_ep, best_loss = _execute_early_stopping(classifier, opt, sched, loop_args)
    classifier.load_state_dict(best_w)
    wall_s = float(time.perf_counter() - t0)

    ckpt_path = cfg.models_dir / f"invariant_{fold.held_out}_s{seed}.pt"
    sha = save_invariant_checkpoint(classifier, ckpt_path, cfg.width)
    summary = _format_finetune_summary((best_ep, best_loss, wall_s), (ckpt_path, sha))
    return classifier, summary


@torch.inference_mode()
def extract_embeddings(
    model: nn.Module,
    items: list[tuple[Path, int]],
    dev: torch.device,
    bs: int = 256,
) -> np.ndarray:
    """Extract latent embeddings for dataset items using encoder."""
    loader = make_data_loader(items, batch_size=bs, shuffle=False)
    model.eval()
    chunks: list[np.ndarray] = []
    for bx, _, _ in loader:
        embs = model.embed(bx.to(dev)).detach().cpu().numpy()
        chunks.append(embs)
    return np.vstack(chunks)


def extract_clean_train_embeddings(
    train_items: list[tuple[Path, int]],
    classifier: nn.Module,
    dev: torch.device,
) -> np.ndarray:
    """Extract embeddings of clean training images only."""
    ds = IndexedMlNdtDataset(train_items)
    flaws, _ = ds.get_labels()
    clean_items = [item for item, fl in zip(train_items, flaws, strict=True) if fl == 0.0]
    if not clean_items:
        raise ValueError("No clean training items available for anomaly fitting.")
    return extract_embeddings(classifier, clean_items, dev)


@torch.inference_mode()
def predict_probabilities(
    models: list[nn.Module],
    items: list[tuple[Path, int]],
    dev: torch.device,
    bs: int = 256,
) -> np.ndarray:
    """Compute ensemble mean sigmoid probabilities across seed models."""
    loader = make_data_loader(items, batch_size=bs, shuffle=False)
    seed_probs: list[np.ndarray] = []
    for model in models:
        model.eval()
        batches = [torch.sigmoid(model(bx.to(dev))).cpu().numpy() for bx, _, _ in loader]
        seed_probs.append(np.concatenate(batches))
    return np.mean(seed_probs, axis=0)


def evaluate_target_fcr_points(
    scores: tuple[np.ndarray, np.ndarray],
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    target: float,
) -> dict[str, Any]:
    """Evaluate p, emb_anom, and fused detectors at target FCR."""
    p_arr, anom_arr = scores
    flawed_idx, calib, ev = indices
    t_p = clean_threshold(p_arr[calib], target)
    m_p = compute_detection_and_false_calls(p_arr >= t_p, flawed_idx, ev)
    m_p["threshold"] = t_p

    t_anom = clean_threshold(anom_arr[calib], target)
    m_anom = compute_detection_and_false_calls(anom_arr >= t_anom, flawed_idx, ev)
    m_anom["threshold"] = t_anom

    t_p_h = clean_threshold(p_arr[calib], target / 2.0)
    t_anom_h = clean_threshold(anom_arr[calib], target / 2.0)
    flags_hyb = (p_arr >= t_p_h) | (anom_arr >= t_anom_h)
    m_hyb = compute_detection_and_false_calls(flags_hyb, flawed_idx, ev)
    m_hyb["threshold_p"] = t_p_h
    m_hyb["threshold_anom"] = t_anom_h

    return {"target_fcr": target, "p": m_p, "emb_anom": m_anom, "fused": m_hyb}


def evaluate_ranking(
    scores: tuple[np.ndarray, np.ndarray, np.ndarray],
    indices: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """Compute AUROC and AP on evaluation partition (flawed + clean_eval)."""
    p_arr, anom_arr, fused_arr = scores
    flawed_idx, ev = indices
    eval_idx = np.concatenate([flawed_idx, ev])
    y_eval = np.concatenate([np.ones(len(flawed_idx), dtype=int), np.zeros(len(ev), dtype=int)])
    return {
        "p": compute_ranking_metrics(y_eval, p_arr[eval_idx]),
        "emb_anom": compute_ranking_metrics(y_eval, anom_arr[eval_idx]),
        "fused": compute_ranking_metrics(y_eval, fused_arr[eval_idx]),
    }


def _make_branch_metrics(rank: dict[str, Any], ops: dict[str, Any], branch: str) -> dict[str, Any]:
    """Extract ranking and operating points for a single detector branch."""
    return {
        "auroc": rank[branch]["auroc"],
        "average_precision": rank[branch]["average_precision"],
        "operating_points": {k: v[branch] for k, v in ops.items()},
    }


def build_fold_record(
    held_out: str,
    ranking: dict[str, Any],
    ops: dict[str, Any],
    meta: tuple[int, int, int, list[str]],
) -> dict[str, Any]:
    """Assemble metrics record for a single LOCO fold."""
    n_flawed, n_calib, n_eval, sha_list = meta
    return {
        "held_out": held_out,
        "n_flawed": n_flawed,
        "n_clean_calib": n_calib,
        "n_clean_eval": n_eval,
        "ranking": ranking,
        "operating_points": ops,
        "p": _make_branch_metrics(ranking, ops, "p"),
        "emb_anom": _make_branch_metrics(ranking, ops, "emb_anom"),
        "fused": _make_branch_metrics(ranking, ops, "fused"),
        "model_sha256": sha_list,
    }


def _evaluate_fold_metrics(
    fold: LocoFold,
    models: list[nn.Module],
    anom_model: CleanFeatureModel,
    dev_sha: tuple[torch.device, list[str]],
) -> dict[str, Any]:
    """Evaluate ensemble, anomaly, and fused scores on held-out test fold."""
    dev, sha_list = dev_sha
    p_cnn = predict_probabilities(models, fold.test, dev)
    test_embs = extract_embeddings(models[0], fold.test, dev)
    emb_anom = anom_model.score(test_embs)

    ds_test = IndexedMlNdtDataset(fold.test)
    flaws, _ = ds_test.get_labels()
    y_true = (flaws == 1.0).astype(int)
    flawed_idx, calib, ev = split_clean_test_scans(y_true, seed=0)

    ecdf_p = map_scores_to_cdf(p_cnn, p_cnn[calib])
    ecdf_a = map_scores_to_cdf(emb_anom, emb_anom[calib])
    fused = np.maximum(ecdf_p, ecdf_a)

    ranking = evaluate_ranking((p_cnn, emb_anom, fused), (flawed_idx, ev))
    ops = {
        "0.01": evaluate_target_fcr_points((p_cnn, emb_anom), (flawed_idx, calib, ev), 0.01),
        "0.05": evaluate_target_fcr_points((p_cnn, emb_anom), (flawed_idx, calib, ev), 0.05),
    }
    meta = (len(flawed_idx), len(calib), len(ev), sha_list)
    return build_fold_record(fold.held_out, ranking, ops, meta)


def _process_single_fold(
    fold: LocoFold,
    cfg: ExperimentConfig,
    dev: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Train pretrain/finetune models and evaluate metrics for one fold."""
    models: list[nn.Module] = []
    summaries: list[dict[str, Any]] = []
    sha_list: list[str] = []

    for seed in cfg.seeds:
        encoder, pre_summary = train_pretrain_seed(fold, seed, cfg, dev)
        classifier, ft_summary = train_finetune_seed(fold, encoder, seed, (cfg, dev))
        models.append(classifier)
        summaries.append(
            {
                "held_out": fold.held_out,
                "seed": seed,
                "pretrain": pre_summary,
                "finetune": ft_summary,
            }
        )
        sha_list.append(ft_summary["model_sha256"])

    clean_embs = extract_clean_train_embeddings(fold.train, models[0], dev)
    anom_model = CleanFeatureModel().fit(clean_embs)

    fold_rec = _evaluate_fold_metrics(fold, models, anom_model, (dev, sha_list))
    return fold_rec, summaries


def build_payload(
    cfg: ExperimentConfig,
    fold_metrics: list[dict[str, Any]],
    training_summary: list[dict[str, Any]],
    timing_gpu: tuple[float, str],
) -> dict[str, Any]:
    """Construct complete output JSON payload for invariant LOCO experiment."""
    wall_s, gpu_name = timing_gpu
    all_sha: list[str] = []
    for f in fold_metrics:
        all_sha.extend(f.get("model_sha256", []))
    unique_sha = sorted(set(all_sha))
    cfg_dict = asdict(cfg)
    cfg_dict["out"] = str(cfg.out)
    cfg_dict["models_dir"] = str(cfg.models_dir)
    note_msg = (
        "transforms are approximations of physical variation; the held-out crack is never seen"
    )
    return {
        "label": "Real data (ML-NDT weld, 3 physical cracks)",
        "preregistration": "docs/decisions/0014-invariant-representation-preregistration.md",
        "note": note_msg,
        "config": cfg_dict,
        "gpu_name": gpu_name,
        "wall_time_s": wall_s,
        "training_summary": training_summary,
        "folds": fold_metrics,
        "model_sha256": unique_sha,
    }


def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    """Execute complete invariant LOCO cross-validation experiment."""
    t_start = time.perf_counter()
    dev = pick_device(cfg.device)
    folds = resolve_folds(cfg.quick)
    fold_metrics: list[dict[str, Any]] = []
    training_summary: list[dict[str, Any]] = []

    for fold in folds:
        fold_rec, f_summary = _process_single_fold(fold, cfg, dev)
        fold_metrics.append(fold_rec)
        training_summary.extend(f_summary)

    wall_s = float(time.perf_counter() - t_start)
    gpu_name = str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu"
    return build_payload(cfg, fold_metrics, training_summary, (wall_s, gpu_name))


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entrypoint for invariant representation LOCO experiment."""
    args = parse_args(argv)
    cfg = build_config(args)
    payload = run_experiment(cfg)
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    cfg.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    main()
