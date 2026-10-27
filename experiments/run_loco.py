"""Leave-one-crack-out (LOCO) training and evaluation experiment (project plan Phase 8.0).

Trains on 2 physical cracks, early-stops on validation without the held-out crack,
and evaluates detection, false calls, and POD on the held-out crack test split.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
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
    MlNdtDataset,
    list_batches,
    read_labels,
)
from utriage.ml.infer import load_model, save_model  # noqa: E402
from utriage.ml.loco import IndexedMlNdtDataset, LocoFold, make_folds  # noqa: E402
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


def _parse_args() -> argparse.Namespace:
    """Parse CLI options for LOCO training and evaluation experiment."""
    parser = argparse.ArgumentParser(description="Leave-one-crack-out (LOCO) experiment")
    parser.add_argument("--quick", action="store_true", help="Quick run with subset")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count override")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Restart seeds")
    return parser.parse_args()


def _find_data_root() -> Path:
    """Locate ML-NDT dataset root across standalone clone or git worktree."""
    if MLNDT_ROOT.exists():
        return MLNDT_ROOT
    try:
        cmd = ["git", "rev-parse", "--git-common-dir"]
        common = subprocess.check_output(  # noqa: S603
            cmd, text=True, stderr=subprocess.DEVNULL
        ).strip()
        fallback = Path(common).resolve().parent / "data" / "external" / "ML-NDT" / "data"
        if fallback.exists():
            return fallback
    except (subprocess.SubprocessError, OSError):
        # Fallback to MLNDT_ROOT default if git inspection fails
        pass
    return MLNDT_ROOT


def _find_best_model_path() -> Path:
    """Locate models/mlndt_best.pt in local folder or parent git tree."""
    local_p = Path("models/mlndt_best.pt")
    if local_p.exists():
        return local_p
    try:
        cmd = ["git", "rev-parse", "--git-common-dir"]
        common = subprocess.check_output(  # noqa: S603
            cmd, text=True, stderr=subprocess.DEVNULL
        ).strip()
        fallback = Path(common).resolve().parent / "models" / "mlndt_best.pt"
        if fallback.exists():
            return fallback
    except (subprocess.SubprocessError, OSError):
        # Return local path if git lookup fails
        pass
    return local_p


def _resolve_batches(quick: bool) -> tuple[list[Path], list[Path]]:
    """Partition dataset into training and validation batch lists."""
    root = _find_data_root()
    tr_all = list_batches("training", root=root)
    val_all = list_batches("validation", root=root)
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


def _eval_fold_test(model: nn.Module, fold: LocoFold) -> dict[str, Any]:
    """Evaluate trained model on fold test split: detections, false calls, and POD."""
    test_ds = IndexedMlNdtDataset(fold.test)
    eval_cfg = TrainConfig(batch_size=64, device="auto")
    probs, _ = predict(model, test_ds, eval_cfg)
    flaw_labels, eq_sizes = test_ds.get_labels()
    hits = (probs >= 0.5).astype(float)

    flawed = (flaw_labels == 1.0) & (eq_sizes > 0.0)
    unflawed = flaw_labels == 0.0

    return {
        "n_test_flawed": int(np.sum(flawed)),
        "n_test_unflawed": int(np.sum(unflawed)),
        "detections": _calc_detection(hits, flawed),
        "false_calls": _calc_false_calls(hits, unflawed),
        "pod": _eval_pod(eq_sizes[flawed], hits[flawed]),
    }


def _collect_labels(batches: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate flaw labels and equivalent sizes across batch files."""
    flaw_list: list[np.ndarray] = []
    size_list: list[np.ndarray] = []
    for b in batches:
        lbl = read_labels(b)
        flaw_list.append(lbl[:, 0])
        size_list.append(lbl[:, 1])
    if not flaw_list:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    return np.concatenate(flaw_list, axis=0), np.concatenate(size_list, axis=0)


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
    val_ds = MlNdtDataset(val_batches)
    eval_cfg = TrainConfig(batch_size=64, device="auto")
    probs, _ = predict(model, val_ds, eval_cfg)

    flaw_labels, eq_sizes = _collect_labels(val_batches)
    hits = (probs >= 0.5).astype(float)
    flawed = (flaw_labels == 1.0) & (eq_sizes > 0.0)
    unflawed = flaw_labels == 0.0

    return {
        "flag": flag,
        "model_path": str(model_path),
        "n_flawed": int(np.sum(flawed)),
        "n_unflawed": int(np.sum(unflawed)),
        "detections": _calc_detection(hits, flawed),
        "false_calls": _calc_false_calls(hits, unflawed),
        "pod": _eval_pod(eq_sizes[flawed], hits[flawed]),
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


def main() -> None:
    """Execute complete leave-one-crack-out training, evaluation, and export."""
    args = _parse_args()
    epochs = args.epochs if args.epochs is not None else (1 if args.quick else 20)
    seeds = args.seeds if args.seeds is not None else ([0] if args.quick else [0, 1, 2])
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    tr_batches, val_batches = _resolve_batches(args.quick)
    folds = make_folds(tr_batches, val_batches, seed=0)
    fold_records: list[dict[str, Any]] = []

    for fold in folds:
        rec = _run_fold_workflow(fold, epochs, seeds, dev_name)
        fold_records.append(rec)

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
