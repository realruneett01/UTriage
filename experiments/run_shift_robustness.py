"""Site shift robustness evaluation experiment (project plan Phase 8 / ADR 0015).

Evaluates whether flaw/clean separation on held-out physical cracks survives
synthetic inspection environment shifts (gain, offset, noise) across model families.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Configure local workspace paths
_EXP_DIR = Path(__file__).resolve().parent
_SRC_DIR = _EXP_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from run_separability import (  # noqa: E402
    compute_auroc,
    split_clean_scans,
)
from run_site_adaptation import FIXED_SHIFTS, SiteShift, apply_shift  # noqa: E402

from utriage.integrity import sha256_file  # noqa: E402
from utriage.ml.datasets import (  # noqa: E402
    MLNDT_ROOT,
    list_batches,
    read_image,
    standardise,
)
from utriage.ml.infer import load_model  # noqa: E402
from utriage.ml.invariance import InvariantClassifier, LiteEncoder  # noqa: E402
from utriage.ml.lite_hybrid import LiteNet  # noqa: E402
from utriage.ml.loco import IndexedMlNdtDataset, LocoFold, make_folds  # noqa: E402
from utriage.ml.train import pick_device  # noqa: E402
from utriage.pod.metrics import clopper_pearson  # noqa: E402

EXPERIMENT_LABEL: str = (
    "Real ML-NDT images with synthetic gain/offset/noise shifts (3 physical cracks)"
)
PREREGISTRATION_PATH: str = "docs/decisions/0015-shift-robustness-preregistration.md"
EXPERIMENT_NOTE: str = "threshold fixed on unshifted clean scans; shifts are synthetic"
SUPPORTED_FAMILIES: tuple[str, ...] = ("loco", "robust", "lite", "invariant")


@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters governing shift robustness execution."""

    families: tuple[str, ...] = SUPPORTED_FAMILIES
    device: str = "auto"
    seed: int = 0
    models_dir: Path = Path("models")
    out: Path = Path("results/shift_robustness_metrics.json")
    quick: bool = False
    lite_width: int = 16
    chunk_size: int = 64


@dataclass(frozen=True)
class ShiftRunContext:
    """Execution context bundling experiment settings and target compute device."""

    cfg: ExperimentConfig
    device: torch.device


@dataclass(frozen=True)
class FoldThresholdData:
    """Partition indices and unshifted calibration operating thresholds."""

    flawed_indices: np.ndarray
    eval_indices: np.ndarray
    th_01: float
    th_05: float


def _find_repo_parent() -> Path | None:
    """Locate common root of parent repository across worktrees."""
    try:
        cmd = ["git", "rev-parse", "--git-common-dir"]
        common = subprocess.check_output(  # noqa: S603
            cmd, text=True, stderr=subprocess.DEVNULL
        ).strip()
        return Path(common).resolve().parent
    except (subprocess.CalledProcessError, FileNotFoundError):  # git error
        return None


def _resolve_models_directory(models_dir: Path) -> Path:
    """Resolve models directory looking locally then in parent repository."""
    if Path(models_dir).resolve() != Path("models").resolve():
        return models_dir
    if models_dir.exists():
        return models_dir
    parent = _find_repo_parent()
    if parent is not None:
        target = parent / "models"
        if target.exists():
            return target
    return models_dir


def _resolve_data_directory(root: Path | None = None) -> Path:
    """Locate ML-NDT data directory locally or in parent repository."""
    if root is not None:
        return root
    if MLNDT_ROOT.exists():
        return MLNDT_ROOT
    parent = _find_repo_parent()
    if parent is not None:
        target = parent / "data" / "external" / "ML-NDT" / "data"
        if target.exists():
            return target
    return MLNDT_ROOT


def resolve_family_model_paths(
    family: str,
    crack: str,
    models_dir: Path,
) -> list[Path]:
    """Find existing model checkpoint paths for a given family and held-out crack."""
    if not models_dir.exists():
        return []
    if family == "loco":
        seed_paths = [models_dir / f"loco_{crack}_seed{s}.pt" for s in (0, 1, 2)]
        return [p for p in seed_paths if p.exists()]
    pattern = rf"^{re.escape(family)}_{re.escape(crack)}_s(\d+)\.pt$"
    matched: list[tuple[int, Path]] = []
    for p in models_dir.glob(f"{family}_{crack}_s*.pt"):
        m = re.match(pattern, p.name)
        if m:
            matched.append((int(m.group(1)), p))
    matched.sort(key=lambda t: t[0])
    return [p for _, p in matched]


def _load_lite_checkpoint(path: Path, device: torch.device, width: int) -> nn.Module:
    """Load LiteNet module weights from checkpoint file."""
    model = LiteNet(width=width)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    st = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(st)
    model.to(device)
    model.eval()
    return model


def _load_invariant_checkpoint(path: Path, device: torch.device, width: int) -> nn.Module:
    """Load InvariantClassifier module weights from checkpoint file."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    w = int(ckpt.get("width", width)) if isinstance(ckpt, dict) else width
    st = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model = InvariantClassifier(LiteEncoder(width=w))
    model.load_state_dict(st)
    model.to(device)
    model.eval()
    return model


def load_single_model(
    family: str,
    path: Path,
    device: torch.device,
    lite_width: int = 16,
) -> nn.Module:
    """Load neural network module for given family onto target device."""
    if family in ("loco", "robust"):
        model, _ = load_model(path, device=str(device))
        return model
    if family == "lite":
        return _load_lite_checkpoint(path, device, lite_width)
    if family == "invariant":
        return _load_invariant_checkpoint(path, device, lite_width)
    raise ValueError(f"Unknown family: {family}")


def load_family_models(
    family: str,
    fold: LocoFold,
    cfg: ExperimentConfig,
    device: torch.device,
) -> list[nn.Module]:
    """Load all seed models for a family and fold."""
    paths = resolve_family_model_paths(family, fold.held_out, cfg.models_dir)
    return [load_single_model(family, p, device, cfg.lite_width) for p in paths]


def get_fold_labels(fold: LocoFold) -> np.ndarray:
    """Retrieve flaw binary labels for items in fold test partition."""
    ds = IndexedMlNdtDataset(fold.test)
    flaws, _ = ds.get_labels()
    return flaws


def read_item_image(item: tuple[Path, int]) -> np.ndarray:
    """Read and standardise single ML-NDT image array."""
    bins, idx = item
    return standardise(read_image(bins, idx))


@torch.inference_mode()
def model_forward_sigmoid(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """Execute model forward pass and return 1D numpy array of sigmoid probabilities."""
    out = model(x)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    if logits.ndim > 1:
        logits = logits.squeeze(-1)
    probs = torch.sigmoid(logits)
    return probs.detach().cpu().numpy().astype(np.float32).reshape(-1)


def score_chunk_ensemble(
    models: list[nn.Module],
    chunk_tensor: torch.Tensor,
) -> np.ndarray:
    """Compute mean sigmoid score across all seed models for a chunk tensor."""
    if not models:
        return np.empty(chunk_tensor.shape[0], dtype=np.float32)
    scores = [model_forward_sigmoid(m, chunk_tensor) for m in models]
    return np.mean(np.stack(scores, axis=0), axis=0).astype(np.float32)


def compute_unshifted_thresholds(calib_scores: np.ndarray) -> tuple[float, float]:
    """Calculate fixed decision thresholds at target FCR 0.01 and 0.05 from clean calib."""
    if len(calib_scores) == 0:
        return 0.5, 0.5
    th_01 = float(np.quantile(calib_scores, 1.0 - 0.01, method="higher"))
    th_05 = float(np.quantile(calib_scores, 1.0 - 0.05, method="higher"))
    return th_01, th_05


def score_items_unshifted(
    models: list[nn.Module],
    items: list[tuple[Path, int]],
    device: torch.device,
    chunk_size: int = 64,
) -> np.ndarray:
    """Score sequence of items without perturbation in memory-safe chunks."""
    if not items or not models:
        return np.empty(0, dtype=np.float32)
    scores_list: list[np.ndarray] = []
    for start in range(0, len(items), chunk_size):
        end = min(start + chunk_size, len(items))
        imgs = [read_item_image(items[i]) for i in range(start, end)]
        t = torch.from_numpy(np.stack(imgs)).unsqueeze(1).to(device)
        scores_list.append(score_chunk_ensemble(models, t))
    return np.concatenate(scores_list, axis=0)


def compute_shift_margin(flawed_scores: np.ndarray, threshold_01: float) -> float:
    """Calculate log10 flawed score margin over unshifted FCR 0.01 threshold."""
    if len(flawed_scores) == 0:
        return float("nan")
    clipped_flawed = np.clip(flawed_scores, 1e-12, 1.0)
    med_flawed = float(np.median(clipped_flawed))
    th_clipped = float(np.clip(threshold_01, 1e-12, 1.0))
    return float(np.log10(med_flawed) - np.log10(th_clipped))


def evaluate_counts_at_threshold(scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """Calculate hits k, trials n, rate, and Clopper-Pearson 95% interval."""
    n = int(len(scores))
    k = int(np.sum(scores >= threshold))
    rate = float(k / n) if n > 0 else float("nan")
    ci_low, ci_high = clopper_pearson(k, n, conf=0.95)
    return {
        "k": k,
        "n": n,
        "rate": rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "interval_95": [ci_low, ci_high],
    }


def evaluate_fcr_operating_points(
    flawed: np.ndarray,
    clean_eval: np.ndarray,
    th_01: float,
    th_05: float,
) -> dict[str, Any]:
    """Compute detection and false calls at fixed unshifted FCR thresholds."""
    return {
        "0.01": {
            "target_fcr": 0.01,
            "threshold": float(th_01),
            "detection": evaluate_counts_at_threshold(flawed, th_01),
            "false_calls": evaluate_counts_at_threshold(clean_eval, th_01),
        },
        "0.05": {
            "target_fcr": 0.05,
            "threshold": float(th_05),
            "detection": evaluate_counts_at_threshold(flawed, th_05),
            "false_calls": evaluate_counts_at_threshold(clean_eval, th_05),
        },
    }


def compute_worst_case_summary(shift_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract worst-case detection and false call rates across shifts."""
    if not shift_results:
        return {}
    det_01_vals = [
        float(r["operating_points"]["0.01"]["detection"]["rate"])
        for r in shift_results
        if not np.isnan(r["operating_points"]["0.01"]["detection"]["rate"])
    ]
    fc_01_vals = [
        float(r["operating_points"]["0.01"]["false_calls"]["rate"])
        for r in shift_results
        if not np.isnan(r["operating_points"]["0.01"]["false_calls"]["rate"])
    ]
    worst_det = min(det_01_vals) if det_01_vals else float("nan")
    worst_fc = max(fc_01_vals) if fc_01_vals else float("nan")
    return {
        "worst_case_detection_rate": worst_det,
        "worst_case_false_call_rate": worst_fc,
        "worst_case_detection_rate_0.01": worst_det,
        "worst_case_false_call_rate_0.01": worst_fc,
    }


def score_items_under_shift(
    models: list[nn.Module],
    items: list[tuple[Path, int]],
    shift: SiteShift,
    ctx: ShiftRunContext,
) -> np.ndarray:
    """Apply shift and score items in chunks with fixed random generator sequence."""
    rng = np.random.default_rng(ctx.cfg.seed)
    scores_list: list[np.ndarray] = []
    chunk_sz = ctx.cfg.chunk_size
    for start in range(0, len(items), chunk_sz):
        end = min(start + chunk_sz, len(items))
        raw_imgs = [read_item_image(items[i]) for i in range(start, end)]
        shifted = [apply_shift(img, shift, rng) for img in raw_imgs]
        t = torch.from_numpy(np.stack(shifted)).unsqueeze(1).to(ctx.device)
        scores_list.append(score_chunk_ensemble(models, t))
    return np.concatenate(scores_list, axis=0) if scores_list else np.empty(0, dtype=np.float32)


def evaluate_shift_metrics(
    scores: np.ndarray,
    split_indices: tuple[np.ndarray, np.ndarray],
    thresholds: tuple[float, float],
) -> tuple[float, float, dict[str, Any]]:
    """Compute AUROC, margin, and operating points for a shift."""
    flawed_idx, eval_idx = split_indices
    th_01, th_05 = thresholds
    flawed_scores = scores[flawed_idx]
    eval_scores = scores[eval_idx]
    y_true = np.concatenate(
        [np.ones(len(flawed_scores), dtype=int), np.zeros(len(eval_scores), dtype=int)]
    )
    y_score = np.concatenate([flawed_scores, eval_scores])
    auroc = compute_auroc(y_true, y_score)
    margin = compute_shift_margin(flawed_scores, th_01)
    ops = evaluate_fcr_operating_points(flawed_scores, eval_scores, th_01, th_05)
    return auroc, margin, ops


def _build_not_run_fold_record(held_out: str) -> dict[str, Any]:
    """Build summary dictionary for skipped fold with no checkpoints."""
    return {
        "held_out": held_out,
        "status": "not run",
        "model_paths": [],
        "model_sha256": [],
        "shifts": [],
        "summary": None,
    }


def _prepare_fold_threshold_data(
    models: list[nn.Module],
    fold: LocoFold,
    ctx: ShiftRunContext,
) -> FoldThresholdData:
    """Calculate unshifted calibration thresholds and resolve evaluation indices."""
    flaws = get_fold_labels(fold)
    clean_indices = np.where(flaws == 0.0)[0]
    flawed_indices = np.where(flaws == 1.0)[0]
    clean_split = split_clean_scans(clean_indices, seed=0)
    cal_items = [fold.test[i] for i in clean_split.calib_indices]
    cal_scores = score_items_unshifted(models, cal_items, ctx.device, ctx.cfg.chunk_size)
    th_01, th_05 = compute_unshifted_thresholds(cal_scores)
    return FoldThresholdData(
        flawed_indices=flawed_indices,
        eval_indices=clean_split.eval_indices,
        th_01=th_01,
        th_05=th_05,
    )


def _run_shifts_for_fold(
    models: list[nn.Module],
    fold: LocoFold,
    th_data: FoldThresholdData,
    ctx: ShiftRunContext,
) -> list[dict[str, Any]]:
    """Evaluate all fixed shifts on fold test items against unshifted thresholds."""
    shift_results: list[dict[str, Any]] = []
    split_idx = (th_data.flawed_indices, th_data.eval_indices)
    thresholds = (th_data.th_01, th_data.th_05)
    for shift in FIXED_SHIFTS:
        scores = score_items_under_shift(models, fold.test, shift, ctx)
        auroc, margin, ops = evaluate_shift_metrics(scores, split_idx, thresholds)
        shift_results.append(
            {
                "shift_name": shift.name,
                "shift_params": {
                    "gain": float(shift.gain),
                    "offset": float(shift.offset),
                    "noise_std": float(shift.noise_std),
                },
                "auroc": auroc,
                "margin": margin,
                "operating_points": ops,
            }
        )
    return shift_results


def evaluate_family_fold(
    family: str,
    fold: LocoFold,
    cfg: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate single model family across site shifts on a single fold."""
    ctx = ShiftRunContext(cfg=cfg, device=device)
    models = load_family_models(family, fold, cfg, device)
    paths = resolve_family_model_paths(family, fold.held_out, cfg.models_dir)
    if not models:
        return _build_not_run_fold_record(fold.held_out)
    th_data = _prepare_fold_threshold_data(models, fold, ctx)
    shift_results = _run_shifts_for_fold(models, fold, th_data, ctx)
    sha_list = [sha256_file(p) for p in paths if p.exists()]
    return {
        "held_out": fold.held_out,
        "status": "evaluated",
        "model_paths": [str(p) for p in paths],
        "model_sha256": sha_list,
        "unshifted_thresholds": {"0.01": th_data.th_01, "0.05": th_data.th_05},
        "shifts": shift_results,
        "summary": compute_worst_case_summary(shift_results),
    }


def build_payload(
    families_data: dict[str, Any],
    model_sha_map: dict[str, str],
    seed: int = 0,
) -> dict[str, Any]:
    """Construct top-level metrics payload dictionary."""
    shifts_list = [
        {
            "name": s.name,
            "gain": float(s.gain),
            "offset": float(s.offset),
            "noise_std": float(s.noise_std),
        }
        for s in FIXED_SHIFTS
    ]
    return {
        "label": EXPERIMENT_LABEL,
        "preregistration": PREREGISTRATION_PATH,
        "note": EXPERIMENT_NOTE,
        "seed": seed,
        "shifts": shifts_list,
        "families": families_data,
        "model_sha256": model_sha_map,
    }


def resolve_experiment_folds(root: Path | None = None, quick: bool = False) -> list[LocoFold]:
    """Discover batches and build LOCO evaluation folds."""
    data_root = _resolve_data_directory(root)
    if not data_root.exists():
        return []
    tr_all = list_batches("training", root=data_root)
    val_all = list_batches("validation", root=data_root)
    if quick:
        tr_all, val_all = tr_all[:6], val_all[:1]
    return make_folds(tr_all, val_all, seed=0, val_mode="batch")


def _add_core_arguments(parser: argparse.ArgumentParser) -> None:
    """Register core execution arguments on argument parser."""
    parser.add_argument(
        "--families",
        type=str,
        nargs="+",
        default=list(SUPPORTED_FAMILIES),
        choices=SUPPORTED_FAMILIES,
        help="Model families to evaluate (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for clean scan split and shift perturbations",
    )


def _add_path_arguments(parser: argparse.ArgumentParser) -> None:
    """Register path and architectural arguments on argument parser."""
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing trained model checkpoints",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/shift_robustness_metrics.json"),
        help="Output path for evaluation JSON metrics",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast run on subset of batches",
    )
    parser.add_argument(
        "--lite-width",
        type=int,
        default=16,
        help="Channel width multiplier for LiteNet (default: 16)",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for shift robustness experiment."""
    parser = argparse.ArgumentParser(
        description="Evaluate site shift robustness across model families."
    )
    _add_core_arguments(parser)
    _add_path_arguments(parser)
    return parser.parse_args(argv)


def _collect_family_sha(records: list[dict[str, Any]], sha_map: dict[str, str]) -> None:
    """Populate model sha256 map from evaluated fold records."""
    for r in records:
        paths = r.get("model_paths", [])
        shas = r.get("model_sha256", [])
        for p_str, sha in zip(paths, shas, strict=True):
            sha_map[Path(p_str).name] = sha


def _evaluate_all_families(
    cfg: ExperimentConfig,
    folds: list[LocoFold],
    dev: torch.device,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Evaluate each family across all folds and gather model hashes."""
    families_data: dict[str, Any] = {}
    all_sha: dict[str, str] = {}
    for fam in cfg.families:
        records = [evaluate_family_fold(fam, f, cfg, dev) for f in folds]
        has_run = any(r["status"] == "evaluated" for r in records)
        families_data[fam] = {
            "status": "evaluated" if has_run else "not run",
            "folds": records,
        }
        _collect_family_sha(records, all_sha)
    return families_data, all_sha


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Execute shift robustness experiment across configured families and folds."""
    args = parse_args(argv)
    cfg = ExperimentConfig(
        families=tuple(args.families),
        device=args.device,
        seed=args.seed,
        models_dir=_resolve_models_directory(Path(args.models_dir)),
        out=Path(args.out),
        quick=args.quick,
        lite_width=args.lite_width,
    )
    dev = pick_device(cfg.device)
    folds = resolve_experiment_folds(quick=cfg.quick)
    families_data, all_sha = _evaluate_all_families(cfg, folds, dev)
    payload = build_payload(families_data, all_sha, seed=cfg.seed)
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    cfg.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    main()
