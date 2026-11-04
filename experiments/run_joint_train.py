"""Joint simulated and real training with self-supervised pretraining (project plan Step 3.1 & 3.3).

Arms per leave-one-crack-out (LOCO) fold:
(A) Real only (baseline, same as run_loco training).
(B) Real + SimBscanDataset mixed via ConcatDataset at sim_fraction.
(C) Masked pretraining on unlabelled real+sim images then fine-tune on real.
(D) B + C (pretrain on real+sim, then fine-tune on real+sim).

Evaluated on held-out crack test split with detection k/n, false calls k/n
(Clopper-Pearson intervals), and POD hit/miss analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from utriage.integrity import sha256_file
from utriage.ml.datasets import MLNDT_ROOT, list_batches
from utriage.ml.infer import save_model
from utriage.ml.loco import IndexedMlNdtDataset, LocoFold, make_folds
from utriage.ml.models import BasicBlock, MlNdtNet, conv3x3
from utriage.ml.pretrain import PretrainConfig, encoder_state, pretrain
from utriage.ml.sim_to_2d import SimBscanConfig, SimBscanDataset
from utriage.ml.train import TrainConfig, pick_device, predict, set_seed, train_model
from utriage.pod.hitmiss import binomial_pod_bound, false_call_rate, fit_hitmiss, fit_to_dict

ARM_NAMES: dict[str, str] = {
    "A": "real_only",
    "B": "real_plus_sim",
    "C": "pretrain_real",
    "D": "pretrain_real_plus_sim",
}

ARM_DESCRIPTIONS: dict[str, str] = {
    "A": "real only (baseline, same as run_loco training)",
    "B": "real + SimBscanDataset mixed via ConcatDataset at sim_fraction",
    "C": "masked pretraining on unlabelled real+sim images then fine-tune on real",
    "D": "B+C",
}

TRANSFER_DESC: str = (
    "encoder weight transfer for matching-shape layers (load_state_dict strict=False)"
)


def _apply_quick_defaults(cfg: JointConfig) -> None:
    """Adjust default hyperparameter values for fast testing mode."""
    if cfg.epochs == 20:
        object.__setattr__(cfg, "epochs", 1)
    if cfg.pretrain_epochs == 5:
        object.__setattr__(cfg, "pretrain_epochs", 1)
    if cfg.n_sim == 100:
        object.__setattr__(cfg, "n_sim", 10)


@dataclass(frozen=True)
class JointConfig:
    """Execution configuration for joint synthetic-real training experiments."""

    sim_fraction: float = 0.5
    n_sim: int = 100
    pretrain_epochs: int = 5
    seed: int = 0
    quick: bool = False
    out_dir: Path = Path("results")
    epochs: int = 20
    arms: tuple[str, ...] = ("A", "B", "C", "D")
    data_root: Path | None = None
    models_dir: Path = Path("models")
    device: str = "auto"

    def __post_init__(self) -> None:
        if isinstance(self.out_dir, str):
            object.__setattr__(self, "out_dir", Path(self.out_dir))
        if isinstance(self.models_dir, str):
            object.__setattr__(self, "models_dir", Path(self.models_dir))
        if isinstance(self.arms, list):
            object.__setattr__(self, "arms", tuple(self.arms))
        if self.quick:
            _apply_quick_defaults(self)


@dataclass(frozen=True)
class PretrainArtifacts:
    """Pretraining state and transferred parameter keys."""

    enc_state: dict[str, torch.Tensor]
    losses: list[float]
    transferred_keys: list[str]


@dataclass(frozen=True)
class ArmContext:
    """Context and dependencies for evaluating a single arm on a fold."""

    arm: str
    fold: LocoFold
    cfg: JointConfig
    device: torch.device
    artifacts: PretrainArtifacts | None = None


class MlNdtEncoder(nn.Module):
    """Backbone encoder matching MlNdtNet convolutional layers."""

    def __init__(self, width: int = 32, first_pool: tuple[int, int] = (2, 2)) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            conv3x3(1, width),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=first_pool),
        )
        self.stage1 = BasicBlock(width, width, stride=1)
        self.stage2 = BasicBlock(width, 2 * width, stride=2)
        self.stage3 = BasicBlock(2 * width, 4 * width, stride=2)
        self.stage4 = BasicBlock(4 * width, 8 * width, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract multi-stage convolutional feature maps."""
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        return self.stage4(feat)


class MlNdtAE(nn.Module):
    """Autoencoder with encoder matching MlNdtNet convolutional backbone."""

    def __init__(self, width: int = 32, first_pool: tuple[int, int] = (2, 2)) -> None:
        super().__init__()
        self.encoder = MlNdtEncoder(width=width, first_pool=first_pool)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8 * width, 4 * width, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4 * width),
            nn.ReLU(),
            nn.ConvTranspose2d(4 * width, 2 * width, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * width, width, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.ConvTranspose2d(width, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct 2D B-scans from latent encoder features."""
        return self.decoder(self.encoder(x))


class SimDatasetAdapter(Dataset):
    """Adapts SimBscanDataset yielding (x, flaw, eq_size) for training compatibility."""

    def __init__(self, sim_ds: SimBscanDataset, flaw_size: float = 1.0) -> None:
        self.sim_ds = sim_ds
        self.flaw_size = flaw_size

    def __len__(self) -> int:
        return len(self.sim_ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, flaw = self.sim_ds[idx]
        size_val = self.flaw_size if float(flaw) > 0.5 else 0.0
        return x, flaw, torch.tensor(size_val, dtype=torch.float32)


def resolve_sim_count(sim_fraction: float, n_sim: int, n_real: int | None = None) -> int:
    """Calculate number of simulated samples to generate."""
    if sim_fraction <= 0.0 or n_sim <= 0:
        return 0
    if n_real is not None and n_real > 0:
        count = int(round(n_real * sim_fraction))
        return max(1, min(n_sim, count))
    return max(1, min(n_sim, int(round(n_sim * sim_fraction))))


def _clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate exact two-sided Clopper-Pearson binomial confidence interval."""
    alpha = 1.0 - confidence
    lower = 0.0 if k == 0 else float(scipy.stats.beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(scipy.stats.beta.ppf(1.0 - alpha / 2, k + 1, n - k))
    return lower, upper


def calc_detection(
    hits: np.ndarray, flawed: np.ndarray, confidence: float = 0.95
) -> dict[str, Any]:
    """Calculate detection counts, rate, and Clopper-Pearson confidence bounds."""
    n_flawed = int(np.sum(flawed))
    if n_flawed == 0:
        return {
            "k": 0,
            "n": 0,
            "n_hit": 0,
            "n_flawed": 0,
            "rate": 0.0,
            "lower95": 0.0,
            "upper95": 0.0,
        }
    n_hit = int(np.sum(hits[flawed]))
    rate = float(n_hit / n_flawed)
    lower, upper = _clopper_pearson(n_hit, n_flawed, confidence)
    return {
        "k": n_hit,
        "n": n_flawed,
        "n_hit": n_hit,
        "n_flawed": n_flawed,
        "rate": rate,
        "lower95": lower,
        "upper95": upper,
    }


def calc_false_calls(
    hits: np.ndarray, unflawed: np.ndarray, confidence: float = 0.95
) -> dict[str, Any]:
    """Calculate false call count, rate, and Clopper-Pearson upper bound."""
    n_unflawed = int(np.sum(unflawed))
    if n_unflawed == 0:
        return {"k": 0, "n": 0, "n_false": 0, "n_unflawed": 0, "rate": 0.0, "upper95": 0.0}
    fc = false_call_rate(hits[unflawed])
    n_false, n = int(fc["n_false"]), int(fc["n"])
    upper = (
        float(fc["upper95"])
        if confidence == 0.95
        else (
            1.0
            if n_false == n
            else float(scipy.stats.beta.ppf(confidence, n_false + 1, n - n_false))
        )
    )
    return {
        "k": n_false,
        "n": n,
        "n_false": n_false,
        "n_unflawed": n,
        "rate": float(fc["rate"]),
        "upper95": upper,
    }


def eval_pod(sizes: np.ndarray, hits: np.ndarray) -> dict[str, Any]:
    """Fit logistic POD if possible, otherwise return binomial Clopper-Pearson bound."""
    if len(sizes) == 0:
        res = binomial_pod_bound(sizes, hits)
        res["reason"] = "No flawed samples"
        res["a90_95"] = None
        return res
    try:
        fit = fit_hitmiss(sizes, hits)
        d = fit_to_dict(fit)
        d["reason"] = None
        return d
    except (ValueError, RuntimeError) as exc:  # Narrow exception for non-convergent fit
        res = binomial_pod_bound(sizes, hits)
        res["reason"] = str(exc)
        res["a90_95"] = None
        return res


def make_sim_dataset(sim_count: int, seed: int) -> SimDatasetAdapter:
    """Build synthetic B-scan dataset with adapted 3-tuple interface."""
    sim_ds = SimBscanDataset(length=sim_count, base_seed=seed, cfg=SimBscanConfig())
    return SimDatasetAdapter(sim_ds)


def resolve_train_dataset(
    real_train_ds: IndexedMlNdtDataset,
    sim_adapter: SimDatasetAdapter,
    arm: str,
) -> Dataset:
    """Select training dataset based on experimental arm specification."""
    if arm in ("B", "D") and len(sim_adapter) > 0:
        return ConcatDataset([real_train_ds, sim_adapter])
    return real_train_ds


def run_pretrain(
    pretrain_ds: Dataset,
    cfg: JointConfig,
    device: torch.device,
) -> PretrainArtifacts:
    """Execute masked autoencoder pretraining and identify transferred keys."""
    set_seed(cfg.seed)
    ae = MlNdtAE(width=32, first_pool=(2, 2))
    ae.to(device)
    pre_cfg = PretrainConfig(epochs=cfg.pretrain_epochs, seed=cfg.seed)
    loader = DataLoader(pretrain_ds, batch_size=32, shuffle=True, drop_last=False)
    losses = pretrain(ae, loader, pre_cfg, device=device)
    enc_state = encoder_state(ae)
    ref_net = MlNdtNet(width=32, first_pool=(2, 2))
    net_state = ref_net.state_dict()
    transferred_keys = [
        k for k, v in enc_state.items() if k in net_state and v.shape == net_state[k].shape
    ]
    return PretrainArtifacts(enc_state, losses, transferred_keys)


def init_model(arm: str, artifacts: PretrainArtifacts | None) -> MlNdtNet:
    """Instantiate MlNdtNet with optional pretrained encoder weights."""
    model = MlNdtNet(width=32, first_pool=(2, 2))
    if arm in ("C", "D") and artifacts is not None:
        model.load_state_dict(artifacts.enc_state, strict=False)
    return model


def compute_model_sha(model: MlNdtNet, path: Path, save_disk: bool) -> str:
    """Save model checkpoint to disk or compute sha256 digest in memory."""
    meta = {"width": 32, "first_pool": (2, 2)}
    if save_disk:
        save_model(model, path, meta)
        return sha256_file(path)
    buf = io.BytesIO()
    torch.save({"state_dict": model.state_dict(), "meta": meta}, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def eval_model_on_test(
    model: nn.Module, fold: LocoFold, dev: torch.device
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int, int]:
    """Evaluate trained model on fold test split returning detection and POD metrics."""
    test_ds = IndexedMlNdtDataset(fold.test)
    eval_cfg = TrainConfig(batch_size=64, device=str(dev))
    probs, _ = predict(model, test_ds, eval_cfg)
    flaw_labels, eq_sizes = test_ds.get_labels()
    hits = (probs >= 0.5).astype(float)
    flawed = (flaw_labels == 1.0) & (eq_sizes > 0.0)
    unflawed = flaw_labels == 0.0
    det = calc_detection(hits, flawed)
    fc = calc_false_calls(hits, unflawed)
    pod = eval_pod(eq_sizes[flawed], hits[flawed])
    return det, fc, pod, int(np.sum(flawed)), int(np.sum(unflawed))


def _build_arm_record(
    ctx: ArmContext,
    counts: tuple[int, int, int, int],
    eval_res: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    model_stats: tuple[float, str],
) -> dict[str, Any]:
    """Assemble individual arm metrics dictionary."""
    n_tr, n_val, n_flaw, n_unflaw = counts
    det, fc, pod = eval_res
    val_loss, sha = model_stats
    rec: dict[str, Any] = {
        "arm": ctx.arm,
        "held_out": ctx.fold.held_out,
        "n_train": n_tr,
        "n_val": n_val,
        "n_test_flawed": n_flaw,
        "n_test_unflawed": n_unflaw,
        "detections": det,
        "false_calls": fc,
        "pod": pod,
        "best_val_loss": val_loss,
        "model_sha256": sha,
    }
    if ctx.arm in ("C", "D") and ctx.artifacts is not None:
        rec["transferred_keys"] = ctx.artifacts.transferred_keys
        rec["transfer_method"] = TRANSFER_DESC
        rec["pretrain_losses"] = ctx.artifacts.losses
    return rec


def run_arm_on_fold(ctx: ArmContext) -> dict[str, Any]:
    """Execute training, evaluation, and artifact tracking for one arm on a fold."""
    set_seed(ctx.cfg.seed)
    real_tr_ds = IndexedMlNdtDataset(ctx.fold.train)
    val_ds = IndexedMlNdtDataset(ctx.fold.val)
    sim_count = resolve_sim_count(ctx.cfg.sim_fraction, ctx.cfg.n_sim, len(real_tr_ds))
    sim_ds = make_sim_dataset(sim_count, ctx.cfg.seed)
    train_ds = resolve_train_dataset(real_tr_ds, sim_ds, ctx.arm)

    model = init_model(ctx.arm, ctx.artifacts)
    t_cfg = TrainConfig(
        epochs=ctx.cfg.epochs, batch_size=64, seed=ctx.cfg.seed, device=str(ctx.device)
    )
    train_res = train_model(model, train_ds, val_ds, t_cfg)

    det, fc, pod, n_flaw, n_unflaw = eval_model_on_test(model, ctx.fold, ctx.device)
    m_path = ctx.cfg.models_dir / f"joint_{ctx.arm}_{ctx.fold.held_out}.pt"
    sha = compute_model_sha(model, m_path, not ctx.cfg.quick)
    counts = (len(train_ds), len(val_ds), n_flaw, n_unflaw)
    stats = (float(train_res.best_val_loss), sha)
    return _build_arm_record(ctx, counts, (det, fc, pod), stats)


def run_fold(fold: LocoFold, cfg: JointConfig, dev: torch.device) -> dict[str, Any]:
    """Execute all configured arms on a single leave-one-crack-out fold."""
    artifacts: PretrainArtifacts | None = None
    if any(a in cfg.arms for a in ("C", "D")):
        real_tr_ds = IndexedMlNdtDataset(fold.train)
        sim_count = resolve_sim_count(cfg.sim_fraction, cfg.n_sim, len(real_tr_ds))
        sim_ds = make_sim_dataset(sim_count, cfg.seed)
        pre_ds = ConcatDataset([real_tr_ds, sim_ds]) if sim_count > 0 else real_tr_ds
        artifacts = run_pretrain(pre_ds, cfg, dev)

    arm_recs = {}
    for arm in cfg.arms:
        ctx = ArmContext(arm, fold, cfg, dev, artifacts)
        arm_recs[arm] = run_arm_on_fold(ctx)
    return {"held_out": fold.held_out, "arms": arm_recs}


def resolve_batches(quick: bool, root: Path | None = None) -> tuple[list[Path], list[Path]]:
    """Partition dataset into training and validation batch lists."""
    base_root = root if root is not None else MLNDT_ROOT
    tr_all = list_batches("training", root=base_root)
    val_all = list_batches("validation", root=base_root)
    if quick:
        return tr_all[:6], val_all[:1]
    return tr_all, val_all


def _extract_transferred_keys(fold_records: list[dict[str, Any]]) -> list[str]:
    """Retrieve transferred parameter keys from first available fold record."""
    arm_records = [rec for f in fold_records for rec in f["arms"].values()]
    for arm_data in arm_records:
        if "transferred_keys" in arm_data:
            return list(arm_data["transferred_keys"])
    return []


def _build_arm_summaries(
    arms: tuple[str, ...], fold_records: list[dict[str, Any]], arm_shas: dict[str, Any]
) -> dict[str, Any]:
    """Assemble per-arm summary records across completed folds."""
    return {
        arm: {
            "name": ARM_NAMES.get(arm, arm),
            "description": ARM_DESCRIPTIONS.get(arm, arm),
            "folds": [f["arms"][arm] for f in fold_records if arm in f["arms"]],
            "model_sha256": arm_shas[arm],
        }
        for arm in arms
    }


def build_experiment_summary(
    cfg: JointConfig, fold_records: list[dict[str, Any]], dev_str: str
) -> dict[str, Any]:
    """Assemble complete experiment metrics dictionary and summary structures."""
    arm_shas: dict[str, Any] = {}
    for arm in cfg.arms:
        shas = [f["arms"][arm]["model_sha256"] for f in fold_records if arm in f["arms"]]
        arm_shas[arm] = shas[0] if len(shas) == 1 else shas

    return {
        "label": "real metrics on ML-NDT (weld, 3 physical cracks); sim images are Synthetic data",
        "protocol": "Leave-one-crack-out joint synthetic-real training",
        "device": dev_str,
        "seeds": [cfg.seed],
        "sim_fraction": cfg.sim_fraction,
        "n_sim": cfg.n_sim,
        "pretrain_epochs": cfg.pretrain_epochs,
        "epochs": cfg.epochs,
        "quick": cfg.quick,
        "transfer_method": TRANSFER_DESC,
        "transferred_keys": _extract_transferred_keys(fold_records),
        "model_sha256": arm_shas,
        "arms": _build_arm_summaries(cfg.arms, fold_records, arm_shas),
        "folds": fold_records,
    }


def run_joint_experiment(cfg: JointConfig) -> dict[str, Any]:
    """Execute complete joint training experiment across folds and export JSON."""
    set_seed(cfg.seed)
    dev = pick_device("cpu" if (cfg.quick and cfg.device == "cpu") else cfg.device)
    tr_b, val_b = resolve_batches(cfg.quick, root=cfg.data_root)
    folds = make_folds(tr_b, val_b, seed=cfg.seed)
    if cfg.quick:
        folds = folds[:1]

    fold_records = [run_fold(f, cfg, dev) for f in folds]
    summary = build_experiment_summary(cfg, fold_records, str(dev))

    out_file = (
        cfg.out_dir if cfg.out_dir.suffix == ".json" else cfg.out_dir / "joint_train_metrics.json"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for joint training experiment."""
    p = argparse.ArgumentParser(description="Joint training and self-supervised pretraining")
    p.add_argument("--quick", action="store_true", help="Quick run with subset, 1 epoch, 1 fold")
    p.add_argument("--sim-fraction", type=float, default=0.5, help="Simulated sample fraction")
    p.add_argument("--n-sim", type=int, default=100, help="Base synthetic pool size")
    p.add_argument("--pretrain-epochs", type=int, default=None, help="Pretraining epoch count")
    p.add_argument("--epochs", type=int, default=None, help="Supervised epoch count")
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    p.add_argument("--out-dir", type=Path, default=Path("results"), help="Metrics output dir")
    p.add_argument("--arms", nargs="+", default=["A", "B", "C", "D"], help="Arms to run")
    p.add_argument("--data-root", type=Path, default=None, help="ML-NDT dataset root")
    return p


def parse_cli_args(args: argparse.Namespace) -> JointConfig:
    """Translate parsed CLI arguments into immutable JointConfig."""
    raw_arms = args.arms
    if len(raw_arms) == 1 and "," in raw_arms[0]:
        arms = tuple(a.strip().upper() for a in raw_arms[0].split(","))
    else:
        arms = tuple(str(a).strip().upper() for a in raw_arms)

    kw: dict[str, Any] = {
        "sim_fraction": args.sim_fraction,
        "n_sim": args.n_sim,
        "seed": args.seed,
        "quick": args.quick,
        "out_dir": args.out_dir,
        "arms": arms,
        "data_root": args.data_root,
    }
    if args.epochs is not None:
        kw["epochs"] = args.epochs
    if args.pretrain_epochs is not None:
        kw["pretrain_epochs"] = args.pretrain_epochs
    return JointConfig(**kw)


def main() -> None:
    """Execute complete joint training CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = parse_cli_args(args)
    run_joint_experiment(cfg)


if __name__ == "__main__":
    main()
