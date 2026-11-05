"""Leave-one-crack-out (LOCO) separability evaluation experiment (Phase 8).

Measures, threshold-free, whether existing trained LOCO models separate flawed
from clean ML-NDT scans on the held-out physical crack, reporting AUROC, AP,
and calibrated operating points at target FCR thresholds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
import torch.nn as nn  # noqa: E402
from run_loco import (  # noqa: E402
    PROTOCOL_DESC_CRACKVAL,
    PROTOCOL_DESC_ENSEMBLE,
    _ensemble_probabilities,
    _file_sha256,
    _resolve_seed_model_path,
    _resolve_single_model_path,
)
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from utriage.ml.datasets import MLNDT_ROOT, list_batches  # noqa: E402
from utriage.ml.infer import load_model  # noqa: E402
from utriage.ml.loco import IndexedMlNdtDataset, LocoFold, make_folds  # noqa: E402
from utriage.ml.train import TrainConfig, pick_device, predict  # noqa: E402
from utriage.pod.metrics import clopper_pearson  # noqa: E402

LABEL_REAL: str = "Real data (ML-NDT weld, 3 physical cracks)"
READING_NOTE: str = (
    "AUROC 0.5 = no separation, 1.0 = perfect ranking; thresholds set on clean scans only"
)


@dataclass(frozen=True)
class CleanSplit:
    """Disjoint calibration and evaluation splits for clean scans."""

    calib_indices: np.ndarray
    eval_indices: np.ndarray


@dataclass(frozen=True)
class EvalContext:
    """Execution context for separability evaluation."""

    device: str = "auto"
    protocols: tuple[str, ...] = ("batchval", "crackval")
    seed: int = 0
    models_dir: Path = Path("models")
    out_path: Path = Path("results/separability_metrics.json")
    quick: bool = False
    seeds: tuple[int, ...] = (0, 1, 2)
    root: Path | None = None


def split_clean_scans(clean_indices: np.ndarray, seed: int = 0) -> CleanSplit:
    """Split clean indices 50/50 into disjoint calib and eval sets using default_rng."""
    arr = np.asarray(clean_indices)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(arr))
    n_half = len(arr) // 2
    calib = arr[perm[:n_half]]
    eval_set = arr[perm[n_half:]]
    if not set(calib).isdisjoint(set(eval_set)):
        raise AssertionError("calib and eval splits must be disjoint")
    assert set(calib).isdisjoint(set(eval_set))  # noqa: S101
    return CleanSplit(calib_indices=calib, eval_indices=eval_set)


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Calculate ROC AUC score or NaN if fewer than two classes present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def compute_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Calculate average precision score or NaN if fewer than two classes present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def compute_operating_point(
    calib_scores: np.ndarray,
    eval_scores: np.ndarray,
    flawed_scores: np.ndarray,
    target_fcr: float,
) -> dict[str, Any]:
    """Compute detection and false calls at calibrated target FCR threshold."""
    th = float(np.quantile(calib_scores, 1.0 - target_fcr, method="higher"))
    n_fl = len(flawed_scores)
    k_fl = int(np.sum(flawed_scores >= th))
    fl_low, fl_high = clopper_pearson(k_fl, n_fl, conf=0.95)
    fl_rate = float(k_fl / n_fl) if n_fl > 0 else float("nan")

    n_ev = len(eval_scores)
    k_ev = int(np.sum(eval_scores >= th))
    ev_low, ev_high = clopper_pearson(k_ev, n_ev, conf=0.95)
    ev_rate = float(k_ev / n_ev) if n_ev > 0 else float("nan")

    return {
        "target_fcr": float(target_fcr),
        "threshold": th,
        "detection": {
            "k": k_fl,
            "n": n_fl,
            "rate": fl_rate,
            "ci_low": fl_low,
            "ci_high": fl_high,
        },
        "false_calls": {
            "k": k_ev,
            "n": n_ev,
            "rate": ev_rate,
            "ci_low": ev_low,
            "ci_high": ev_high,
        },
    }


def compute_operating_points(
    calib_scores: np.ndarray,
    eval_scores: np.ndarray,
    flawed_scores: np.ndarray,
    target_fcrs: tuple[float, ...] = (0.01, 0.05),
) -> dict[str, Any]:
    """Compute operating points for multiple target false call rates."""
    results: dict[str, Any] = {}
    for fcr in target_fcrs:
        results[str(fcr)] = compute_operating_point(calib_scores, eval_scores, flawed_scores, fcr)
    return results


def evaluate_fold_separability(
    y_true: np.ndarray,
    y_score: np.ndarray,
    seed: int = 0,
) -> dict[str, Any]:
    """Calculate threshold-free and calibrated operating point metrics for a fold."""
    clean_indices = np.where(y_true == 0)[0]
    flawed_indices = np.where(y_true == 1)[0]
    split = split_clean_scans(clean_indices, seed=seed)

    calib_scores = y_score[split.calib_indices]
    eval_scores = y_score[split.eval_indices]
    flawed_scores = y_score[flawed_indices]

    ops = compute_operating_points(calib_scores, eval_scores, flawed_scores)
    return {
        "n_flawed": int(len(flawed_indices)),
        "n_clean": int(len(clean_indices)),
        "auroc": compute_auroc(y_true, y_score),
        "average_precision": compute_average_precision(y_true, y_score),
        "operating_points": ops,
    }


def _repo_parent_root() -> Path | None:
    """Locate root directory of parent repository across worktrees."""
    try:
        cmd = ["git", "rev-parse", "--git-common-dir"]
        common = subprocess.check_output(  # noqa: S603
            cmd, text=True, stderr=subprocess.DEVNULL
        ).strip()
        return Path(common).resolve().parent
    except (subprocess.CalledProcessError, FileNotFoundError):  # git invocation failure
        return None


def _find_data_root() -> Path:
    """Locate ML-NDT dataset root across standalone clone or git worktree."""
    if MLNDT_ROOT.exists():
        return MLNDT_ROOT
    parent = _repo_parent_root()
    if parent is not None:
        target = parent / "data" / "external" / "ML-NDT" / "data"
        if target.exists():
            return target
    return MLNDT_ROOT


def _find_models_dir(models_dir: Path = Path("models")) -> Path:
    """Resolve models directory looking in local dir then parent repo if in worktree."""
    if models_dir.exists() or Path(models_dir).resolve() != Path("models").resolve():
        return models_dir
    parent = _repo_parent_root()
    if parent is not None:
        target = parent / "models"
        if target.exists():
            return target
    return models_dir


def _resolve_fold_model_paths(
    held_out: str,
    protocol: str,
    models_dir: Path,
    seeds: list[int],
) -> list[Path]:
    """Resolve model paths for a fold under batchval or crackval protocol."""
    if protocol == "batchval":
        seed_paths = [_resolve_seed_model_path(held_out, s, models_dir) for s in seeds]
        if all(p.exists() for p in seed_paths):
            return seed_paths
        single_p = _resolve_single_model_path(held_out, models_dir, val_mode="batch")
        return [single_p] if single_p.exists() else []
    if protocol == "crackval":
        seed_paths = [models_dir / f"loco_crackval_{held_out}_seed{s}.pt" for s in seeds]
        if all(p.exists() for p in seed_paths):
            return seed_paths
        single_p = _resolve_single_model_path(held_out, models_dir, val_mode="crack")
        return [single_p] if single_p.exists() else []
    return []


def _load_fold_models(paths: list[Path], device: str) -> list[nn.Module]:
    """Load model modules from list of paths onto target device."""
    dev = str(pick_device(device))
    loaded: list[nn.Module] = []
    for p in paths:
        m, _ = load_model(p, device=dev)
        loaded.append(m)
    return loaded


def _evaluate_fold(
    fold: LocoFold,
    models: list[nn.Module],
    paths: list[Path],
    ctx: EvalContext,
) -> dict[str, Any]:
    """Evaluate separability metrics on test split of a single fold."""
    test_ds = IndexedMlNdtDataset(fold.test)
    eval_cfg = TrainConfig(batch_size=64, device=str(pick_device(ctx.device)))
    probs_list = [predict(m, test_ds, eval_cfg)[0] for m in models]
    mean_probs = _ensemble_probabilities(probs_list)

    flaw_labels, _ = test_ds.get_labels()
    y_true = (flaw_labels == 1.0).astype(int)
    metrics = evaluate_fold_separability(y_true, mean_probs, seed=ctx.seed)

    sha_list = [_file_sha256(p) for p in paths]
    rec: dict[str, Any] = {"held_out": fold.held_out, **metrics, "model_sha256": sha_list}
    if fold.val_crack is not None:
        rec["val_crack"] = fold.val_crack
    return rec


def _evaluate_protocol(
    protocol: str,
    folds: list[LocoFold],
    ctx: EvalContext,
) -> dict[str, Any]:
    """Evaluate separability for all folds under a specific protocol."""
    desc = PROTOCOL_DESC_CRACKVAL if protocol == "crackval" else PROTOCOL_DESC_ENSEMBLE
    fold_recs: list[dict[str, Any]] = []
    missing_folds: list[str] = []

    for fold in folds:
        paths = _resolve_fold_model_paths(fold.held_out, protocol, ctx.models_dir, list(ctx.seeds))
        if not paths:
            missing_folds.append(fold.held_out)
            continue
        models = _load_fold_models(paths, ctx.device)
        rec = _evaluate_fold(fold, models, paths, ctx)
        fold_recs.append(rec)

    if not fold_recs:
        note = f"Model files for {protocol} not found in {ctx.models_dir}"
        return {"description": desc, "status": "skipped", "note": note, "folds": []}

    status = "partial" if missing_folds else "evaluated"
    return {"description": desc, "status": status, "folds": fold_recs}


def _build_payload(
    protocols_data: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Construct complete separability JSON payload."""
    all_sha: list[str] = []
    for proto_dict in protocols_data.values():
        for fold_rec in proto_dict.get("folds", []):
            all_sha.extend(fold_rec.get("model_sha256", []))
    unique_sha = sorted(set(all_sha))

    return {
        "label": LABEL_REAL,
        "reading": READING_NOTE,
        "seed": seed,
        "protocol_descriptions": {
            "batchval": PROTOCOL_DESC_ENSEMBLE,
            "crackval": PROTOCOL_DESC_CRACKVAL,
        },
        "protocols": protocols_data,
        "model_sha256": unique_sha,
    }


def _resolve_batches(quick: bool, root: Path | None = None) -> tuple[list[Path], list[Path]]:
    """Partition dataset into training and validation batch lists."""
    data_root = root if root is not None else _find_data_root()
    tr_all = list_batches("training", root=data_root)
    val_all = list_batches("validation", root=data_root)
    if quick:
        return tr_all[:6], val_all[:1]
    return tr_all, val_all


def _dump_json(path: Path, data: dict[str, Any]) -> None:
    """Save dictionary to JSON file creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def run_separability(ctx: EvalContext) -> dict[str, Any]:
    """Execute separability evaluation across specified protocols and export payload."""
    tr_b, val_b = _resolve_batches(ctx.quick, ctx.root)
    protocols_data: dict[str, Any] = {}

    for proto in ctx.protocols:
        val_mode = "crack" if proto == "crackval" else "batch"
        folds = make_folds(tr_b, val_b, seed=0, val_mode=val_mode)
        protocols_data[proto] = _evaluate_protocol(proto, folds, ctx)

    payload = _build_payload(protocols_data, ctx.seed)
    _dump_json(ctx.out_path, payload)
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for separability experiment."""
    parser = argparse.ArgumentParser(description="LOCO separability evaluation")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute device for neural inference",
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=["batchval", "crackval"],
        default=["batchval", "crackval"],
        help="LOCO protocols to evaluate",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for calib split")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/separability_metrics.json"),
        help="Path for output metrics JSON",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing trained model weights",
    )
    parser.add_argument("--quick", action="store_true", help="Quick run with subset")
    parser.add_argument("--root", type=Path, default=None, help="ML-NDT root override")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """CLI entrypoint for LOCO separability evaluation experiment."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    models_dir = _find_models_dir(args.models_dir)
    ctx = EvalContext(
        device=args.device,
        protocols=tuple(args.protocols),
        seed=args.seed,
        models_dir=models_dir,
        out_path=args.out,
        quick=args.quick,
        root=args.root,
    )
    return run_separability(ctx)


if __name__ == "__main__":
    main()
