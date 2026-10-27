"""Leave-one-crack-out (LOCO) training and evaluation experiment (project plan Phase 8.0).

Trains on 2 physical cracks, early-stops on validation without the held-out crack,
and evaluates detection, false calls, and POD on the held-out crack test split.
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
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        del sys.modules["utriage"]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

from utriage.ml.datasets import (  # noqa: E402
    MLNDT_ROOT,
    list_batches,
)
from utriage.ml.infer import load_model, save_model  # noqa: E402
from utriage.ml.loco import IndexedMlNdtDataset, LocoFold, make_folds  # noqa: E402
from utriage.ml.mlndt_meta import image_index  # noqa: E402
from utriage.ml.models import MlNdtNet  # noqa: E402
from utriage.ml.train import (  # noqa: E402
    TrainConfig,
    TrainResult,
    predict,
    train_model,
)
from utriage.pod.hitmiss import (  # noqa: E402
    binomial_pod_bound,
    false_call_rate,
    fit_hitmiss,
    fit_to_dict,
)


@dataclass(frozen=True)
class EnsembleFoldContext:
    """Context and execution parameters for ensemble fold workflow."""

    epochs: int
    seeds: list[int]
    dev_name: str
    eval_only: bool = False
    models_dir: Path = Path("models")


def _parse_args() -> argparse.Namespace:
    """Parse CLI options for LOCO training and evaluation experiment."""
    parser = argparse.ArgumentParser(description="Leave-one-crack-out (LOCO) experiment")
    parser.add_argument("--quick", action="store_true", help="Quick run with subset")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count override")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Restart seeds")
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Train and evaluate deep ensemble across seeds",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and load existing seed weights",
    )
    return parser.parse_args()


def _find_best_model_path() -> Path:
    """Locate models/mlndt_best.pt in local folder or parent repo."""
    local_p = Path("models/mlndt_best.pt")
    if local_p.exists():
        return local_p
    fallback = MLNDT_ROOT.parents[3] / "models" / "mlndt_best.pt"
    return fallback if fallback.exists() else local_p


def _resolve_batches(quick: bool) -> tuple[list[Path], list[Path]]:
    """Partition dataset into training and validation batch lists."""
    tr_all = list_batches("training")
    val_all = list_batches("validation")
    if quick:
        return tr_all[:6], val_all[:1]
    return tr_all, val_all


def _run_single_seed(
    seed: int, train_ds: Dataset, val_ds: Dataset, epochs: int
) -> tuple[dict, TrainResult]:
    """Train single restart seed and return state dict and TrainResult."""
    model = MlNdtNet(width=32, first_pool=(2, 2))
    cfg = TrainConfig(
        epochs=epochs,
        batch_size=64,
        patience=4,
        amp=True,
        seed=seed,
        device="auto",
    )
    res = train_model(model, train_ds, val_ds, cfg)
    return copy.deepcopy(model.state_dict()), res


def _train_fold(fold: LocoFold, epochs: int, seeds: list[int]) -> tuple[nn.Module, TrainResult]:
    """Execute training restarts and select weights with lowest validation loss."""
    train_ds = IndexedMlNdtDataset(fold.train)
    val_ds = IndexedMlNdtDataset(fold.val)
    best_weights: dict = {}
    best_loss = float("inf")
    best_res: TrainResult | None = None

    for s in seeds:
        weights, res = _run_single_seed(s, train_ds, val_ds, epochs)
        if res.best_val_loss < best_loss:
            best_loss = res.best_val_loss
            best_weights = weights
            best_res = res

    model = MlNdtNet(width=32, first_pool=(2, 2))
    model.load_state_dict(best_weights)
    if best_res is None:
        raise RuntimeError("Training failed to produce valid TrainResult")
    return model, best_res


def _resolve_seed_model_path(held_out: str, seed: int, models_dir: Path = Path("models")) -> Path:
    """Locate seed model checkpoint in local folder or parent repo."""
    local_p = models_dir / f"loco_{held_out}_seed{seed}.pt"
    if local_p.exists():
        return local_p
    fallback = MLNDT_ROOT.parents[3] / "models" / f"loco_{held_out}_seed{seed}.pt"
    return fallback if fallback.exists() else local_p


def _load_fold_seed_models(held_out: str, seeds: list[int], models_dir: Path) -> list[nn.Module]:
    """Load existing seed models for a fold."""
    models: list[nn.Module] = []
    for s in seeds:
        p = _resolve_seed_model_path(held_out, s, models_dir)
        m, _ = load_model(p, device="auto")
        models.append(m)
    return models


def _train_and_save_seed_models(
    fold: LocoFold, epochs: int, seeds: list[int], models_dir: Path
) -> tuple[list[nn.Module], TrainResult]:
    """Train restart seeds, save each seed model plus best single model."""
    train_ds, val_ds = IndexedMlNdtDataset(fold.train), IndexedMlNdtDataset(fold.val)
    seed_models: list[nn.Module] = []
    best_weights: dict[str, Any] = {}
    best_loss, best_res = float("inf"), None
    meta = {"width": 32, "first_pool": (2, 2), "size": 256}
    for s in seeds:
        weights, res = _run_single_seed(s, train_ds, val_ds, epochs)
        m = MlNdtNet(width=32, first_pool=(2, 2))
        m.load_state_dict(weights)
        save_model(m, models_dir / f"loco_{fold.held_out}_seed{s}.pt", meta)
        seed_models.append(m)
        if res.best_val_loss < best_loss:
            best_loss, best_weights, best_res = res.best_val_loss, weights, res
    if best_res is None:
        raise RuntimeError("Training failed to produce valid TrainResult")
    best_m = MlNdtNet(width=32, first_pool=(2, 2))
    best_m.load_state_dict(best_weights)
    save_model(best_m, models_dir / f"loco_{fold.held_out}.pt", meta)
    return seed_models, best_res


def _ensemble_probabilities(probs_list: list[np.ndarray]) -> np.ndarray:
    """Compute mean sigmoid probability across ensemble members."""
    if not probs_list:
        return np.empty(0, dtype=np.float32)
    return np.mean(np.asarray(probs_list, dtype=np.float32), axis=0)


def _eval_pod(sizes: np.ndarray, hits: np.ndarray) -> dict[str, Any]:
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
    except ValueError as exc:
        res = binomial_pod_bound(sizes, hits)
        res["reason"] = str(exc)
        res["a90_95"] = None
        return res


def _calc_detection(hits: np.ndarray, flawed: np.ndarray) -> dict[str, Any]:
    """Calculate detection counts and empirical hit rate for flawed samples."""
    n_flawed = int(np.sum(flawed))
    n_hit = int(np.sum(hits[flawed])) if n_flawed > 0 else 0
    rate = float(n_hit / n_flawed) if n_flawed > 0 else 0.0
    return {"n_hit": n_hit, "n_flawed": n_flawed, "rate": rate}


def _calc_false_calls(hits: np.ndarray, unflawed: np.ndarray) -> dict[str, Any]:
    """Calculate false call counts, empirical rate, and 95% upper confidence bound."""
    n_unflawed = int(np.sum(unflawed))
    if n_unflawed == 0:
        return {"n_false": 0, "n_unflawed": 0, "rate": 0.0, "upper95": 0.0}
    fc = false_call_rate(hits[unflawed])
    return {
        "n_false": int(fc["n_false"]),
        "n_unflawed": int(fc["n"]),
        "rate": float(fc["rate"]),
        "upper95": float(fc["upper95"]),
    }


def _evaluate_predictions(
    probs: np.ndarray, flaw_labels: np.ndarray, eq_sizes: np.ndarray
) -> dict[str, Any]:
    """Calculate detection, false call, and POD metrics from predictions."""
    hits = (probs >= 0.5).astype(float)
    flawed = (flaw_labels == 1.0) & (eq_sizes > 0.0)
    unflawed = flaw_labels == 0.0
    return {
        "n_flawed": int(np.sum(flawed)),
        "n_unflawed": int(np.sum(unflawed)),
        "detections": _calc_detection(hits, flawed),
        "false_calls": _calc_false_calls(hits, unflawed),
        "pod": _eval_pod(eq_sizes[flawed], hits[flawed]),
    }


def _eval_fold_test(model: nn.Module, fold: LocoFold) -> dict[str, Any]:
    """Evaluate trained model on fold test split: detections, false calls, and POD."""
    test_ds = IndexedMlNdtDataset(fold.test)
    eval_cfg = TrainConfig(batch_size=64, device="auto")
    probs, _ = predict(model, test_ds, eval_cfg)
    flaw_labels, eq_sizes = test_ds.get_labels()
    ev = _evaluate_predictions(probs, flaw_labels, eq_sizes)
    return {
        "n_test_flawed": ev["n_flawed"],
        "n_test_unflawed": ev["n_unflawed"],
        "detections": ev["detections"],
        "false_calls": ev["false_calls"],
        "pod": ev["pod"],
    }


def _eval_ensemble_fold(models: list[nn.Module], fold: LocoFold) -> dict[str, Any]:
    """Evaluate ensemble of models on fold test split via average probability."""
    test_ds = IndexedMlNdtDataset(fold.test)
    eval_cfg = TrainConfig(batch_size=64, device="auto")
    probs_list = [predict(m, test_ds, eval_cfg)[0] for m in models]
    mean_probs = _ensemble_probabilities(probs_list)
    flaw_labels, eq_sizes = test_ds.get_labels()
    ev = _evaluate_predictions(mean_probs, flaw_labels, eq_sizes)
    return {
        "n_test_flawed": ev["n_flawed"],
        "n_test_unflawed": ev["n_unflawed"],
        "detections": ev["detections"],
        "false_calls": ev["false_calls"],
        "pod": ev["pod"],
    }


def _load_single_best(path: Path = Path("results/loco_metrics.json")) -> dict[str, Any] | None:
    """Load existing loco_metrics.json if present."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):  # Narrow exception for corrupt/unreadable file
        return None


def _find_fold_record(data: dict[str, Any] | None, held_out: str) -> dict[str, Any] | None:
    """Find matching fold entry from previous loco_metrics.json."""
    if not data or "folds" not in data:
        return None
    for entry in data["folds"]:
        if entry.get("held_out") == held_out:
            return entry
    return None


def _eval_part_a(val_batches: list[Path]) -> dict[str, Any]:
    """Evaluate existing models/mlndt_best.pt on official validation folder only."""
    flag = (
        "leaky: validation shares all 3 physical cracks with training "
        "(see results/leakage_audit.json)"
    )
    model_path = _find_best_model_path()
    if not model_path.exists():
        return {"flag": flag, "status": "skipped", "reason": "mlndt_best.pt not found"}

    model, _ = load_model(model_path, device="auto")
    val_items = [(b, i) for b, i, _ in image_index(val_batches)]
    val_ds = IndexedMlNdtDataset(val_items)
    eval_cfg = TrainConfig(batch_size=64, device="auto")
    probs, _ = predict(model, val_ds, eval_cfg)

    flaw_labels, eq_sizes = val_ds.get_labels()
    ev = _evaluate_predictions(probs, flaw_labels, eq_sizes)
    return {
        "flag": flag,
        "model_path": str(model_path),
        "n_flawed": ev["n_flawed"],
        "n_unflawed": ev["n_unflawed"],
        "detections": ev["detections"],
        "false_calls": ev["false_calls"],
        "pod": ev["pod"],
    }


def _run_fold_workflow(
    fold: LocoFold, epochs: int, seeds: list[int], dev_name: str
) -> dict[str, Any]:
    """Train single fold, save checkpoint, and evaluate test performance."""
    t0 = time.time()
    model, best_res = _train_fold(fold, epochs, seeds)
    sec = time.time() - t0

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    save_model(
        model,
        models_dir / f"loco_{fold.held_out}.pt",
        {"width": 32, "first_pool": (2, 2), "size": 256},
    )

    ev = _eval_fold_test(model, fold)
    return {
        "held_out": fold.held_out,
        "n_train": len(fold.train),
        "n_val": len(fold.val),
        "n_test_flawed": ev["n_test_flawed"],
        "n_test_unflawed": ev["n_test_unflawed"],
        "detections": ev["detections"],
        "false_calls": ev["false_calls"],
        "pod": ev["pod"],
        "best_val_loss": float(best_res.best_val_loss),
        "seconds": round(sec, 2),
        "device_name": dev_name,
    }


def _run_ensemble_fold_workflow(
    fold: LocoFold, ctx: EnsembleFoldContext, single_best_data: dict[str, Any] | None
) -> dict[str, Any]:
    """Execute training or loading and ensemble evaluation for single fold."""
    t0, best_loss = time.time(), None
    if ctx.eval_only:
        models = _load_fold_seed_models(fold.held_out, ctx.seeds, ctx.models_dir)
    else:
        ctx.models_dir.mkdir(parents=True, exist_ok=True)
        models, b_res = _train_and_save_seed_models(fold, ctx.epochs, ctx.seeds, ctx.models_dir)
        best_loss = float(b_res.best_val_loss)
    ev = _eval_ensemble_fold(models, fold)
    single_best = _find_fold_record(single_best_data, fold.held_out)
    if best_loss is None and single_best is not None:
        best_loss = single_best.get("best_val_loss")
    return {
        "held_out": fold.held_out,
        "n_train": len(fold.train),
        "n_val": len(fold.val),
        "n_test_flawed": ev["n_test_flawed"],
        "n_test_unflawed": ev["n_test_unflawed"],
        "detections": ev["detections"],
        "false_calls": ev["false_calls"],
        "pod": ev["pod"],
        "best_val_loss": best_loss,
        "seconds": round(time.time() - t0, 2),
        "device_name": ctx.dev_name,
        "seeds": ctx.seeds,
        "seeds_used": ctx.seeds,
        "single_best": single_best,
        "single_best_comparison": single_best,
    }


def _run_ensemble_workflow(
    folds: list[LocoFold],
    ctx: EnsembleFoldContext,
    val_batches: list[Path],
    out_path: Path = Path("results/loco_ensemble_metrics.json"),
) -> dict[str, Any]:
    """Execute complete ensemble training/evaluation and write output metrics."""
    single_best_data = _load_single_best()
    fold_records = [_run_ensemble_fold_workflow(f, ctx, single_best_data) for f in folds]
    part_a_rec = _eval_part_a(val_batches)
    protocol_desc = (
        "Leave-one-crack-out (LOCO) deep ensemble cross-validation across 3 physical cracks. "
        "Train on 2 cracks and non-holdout unflawed, validation on official "
        "validation batches without held-out crack, test on held-out crack and "
        "holdout-batch unflawed only."
    )
    metrics = {
        "label": "Real data (ML-NDT)",
        "protocol": protocol_desc,
        "seeds": ctx.seeds,
        "seeds_used": ctx.seeds,
        "folds": fold_records,
        "official_validation": part_a_rec,
        "part_a": part_a_rec,
    }
    if single_best_data is not None:
        metrics["single_best"] = single_best_data.get("folds")
        metrics["single_best_comparison"] = single_best_data.get("folds")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"LOCO ensemble metrics saved to {out_path}", flush=True)
    return metrics


def _resolve_epochs(args: argparse.Namespace) -> int:
    """Resolve epoch count from CLI options."""
    if args.epochs is not None:
        return args.epochs
    return 1 if args.quick else 20


def _resolve_seeds(args: argparse.Namespace) -> list[int]:
    """Resolve restart seeds from CLI options."""
    if args.seeds is not None:
        return args.seeds
    return [0] if args.quick else [0, 1, 2]


def _device_name() -> str:
    """Return active GPU device name or cpu."""
    if torch.cuda.is_available():
        return str(torch.cuda.get_device_name(0))
    return "cpu"


def main() -> None:
    """Execute complete leave-one-crack-out training, evaluation, and export."""
    args = _parse_args()
    epochs = _resolve_epochs(args)
    seeds = _resolve_seeds(args)
    dev_name = _device_name()
    tr_batches, val_batches = _resolve_batches(args.quick)
    folds = make_folds(tr_batches, val_batches, seed=0)

    if args.ensemble or args.eval_only:
        ctx = EnsembleFoldContext(
            epochs=epochs,
            seeds=seeds,
            dev_name=dev_name,
            eval_only=args.eval_only,
        )
        _run_ensemble_workflow(folds, ctx, val_batches)
        return

    fold_records = [_run_fold_workflow(f, epochs, seeds, dev_name) for f in folds]
    part_a_rec = _eval_part_a(val_batches)
    protocol_desc = (
        "Leave-one-crack-out (LOCO) cross-validation across 3 physical cracks. "
        "Train on 2 cracks and non-holdout unflawed, validation on official "
        "validation batches without held-out crack, test on held-out crack and "
        "holdout-batch unflawed only."
    )
    metrics = {
        "label": "Real data (ML-NDT)",
        "protocol": protocol_desc,
        "folds": fold_records,
        "official_validation": part_a_rec,
        "part_a": part_a_rec,
    }
    out_path = Path("results/loco_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"LOCO metrics saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
