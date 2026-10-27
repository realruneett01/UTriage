"""Model fit diagnostics and learning curve analysis (project plan Phase 8).

Evaluates learning curves, data-size scaling on synthetic supervised model,
and train-vs-test gaps with FitRule(target_train_metric=0.9, max_gap=0.1).
"""

from __future__ import annotations

import argparse
import functools
import json
import subprocess
import sys
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from utriage.ml.bar_data import WindowSet  # noqa: E402
from utriage.ml.bar_models import TrainerAdapter, WindowDataset, WindowNet  # noqa: E402
from utriage.ml.datasets import list_batches  # noqa: E402
from utriage.ml.diagnostics import (  # noqa: E402
    FitRule,
    data_size_curve,
    diagnose,
    learning_curve,
    plot_learning_curve,
)
from utriage.ml.infer import load_model  # noqa: E402
from utriage.ml.loco import IndexedMlNdtDataset, LocoFold, make_folds  # noqa: E402
from utriage.ml.sim_windows import (  # noqa: E402
    SimGenSettings,
    build_in_distribution_test,
    build_region_test,
    build_train,
)
from utriage.ml.train import (  # noqa: E402
    TrainConfig,
    pick_device,
    predict,
    set_seed,
    train_model,
)
from utriage.physics.randomize import DEFAULT_HOLDOUT_REGIONS  # noqa: E402


@dataclass(frozen=True)
class FitDiagnosticsConfig:
    """Execution configuration for fit diagnostics."""

    mlndt_metrics_path: Path = Path("results/mlndt_metrics.json")
    loco_metrics_path: Path = Path("results/loco_metrics.json")
    anomaly_metrics_path: Path = Path("results/anomaly_metrics.json")
    out_json: Path = Path("results/fit_diagnostics.json")
    figures_dir: Path = Path("results/figures")
    models_dir: Path = Path("models")
    fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 1.0)
    epochs: int = 10
    quick: bool = False
    device: str = "auto"
    seed: int = 42


@dataclass(frozen=True)
class LocoEvalContext:
    """Execution context for LOCO fold scoring."""

    models_dir: Path = Path("models")
    device: str = "auto"
    seed: int = 42
    max_samples: int = 2000


@dataclass(frozen=True)
class WindowEvalBundle:
    """Evaluation sets container for data-size benchmark."""

    in_dist: WindowSet
    regions: dict[str, WindowSet]


@dataclass(frozen=True)
class FractionEvalContext:
    """Context holding datasets and hardware settings for data-size evaluation."""

    full_ws: WindowSet
    bundles: WindowEvalBundle
    cfg: FitDiagnosticsConfig
    dev: torch.device


@dataclass(frozen=True)
class FitPayloadComponents:
    """Diagnostic outputs and curve artifacts to serialize into payload."""

    lc_data: dict[str, Any]
    ds_records: list[dict[str, Any]]
    synth_diag: dict[str, Any]
    rule: FitRule
    fig_paths: list[str]
    loco_diag: dict[str, Any]
    mlndt_diag: dict[str, Any]


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    """Safely load JSON dictionary from path if file exists."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        # Return None if file is unreadable or malformed
        return None


def _select_mlndt_restart(restarts: list[dict[str, Any]], kept_seed: Any) -> dict[str, Any]:
    """Select kept restart matching kept_seed or lowest best_val_loss."""
    matches = [r for r in restarts if r.get("seed") == kept_seed]
    if matches:
        return matches[0]
    return min(restarts, key=lambda r: float(r.get("best_val_loss", float("inf"))))


def extract_mlndt_learning_curve(path: Path) -> dict[str, Any]:
    """Extract learning curve from kept restart of ML-NDT results."""
    data = _load_json_if_exists(path)
    if data is None:
        return {"available": False, "reason": f"{path} not found"}
    restarts = data.get("restarts", [])
    if not restarts:
        return {"available": False, "reason": "No restarts recorded in mlndt_metrics"}
    restart = _select_mlndt_restart(restarts, data.get("kept_seed"))
    hist = restart.get("history", [])
    if not hist:
        return {"available": False, "reason": "No training history recorded in kept restart"}
    return {
        "available": True,
        "curve": learning_curve(hist),
        "note": "leaky: validation shares all 3 physical cracks with training",
        "leaky": True,
        "kept_seed": restart.get("seed"),
    }


def _extract_fold_entry(f: dict[str, Any]) -> tuple[str, dict[str, Any], bool]:
    """Extract curve or reason for a single LOCO fold."""
    name = str(f.get("held_out", "unknown"))
    hist = f.get("history")
    if isinstance(hist, list) and hist:
        return name, {"available": True, "curve": learning_curve(hist)}, True
    return (
        name,
        {
            "available": False,
            "reason": "No epoch history recorded for fold (only best_val_loss recorded)",
        },
        False,
    )


def extract_loco_learning_curves(path: Path) -> dict[str, Any]:
    """Extract per-fold learning curves from LOCO experiment results."""
    data = _load_json_if_exists(path)
    if data is None:
        return {"available": False, "reason": f"{path} not found"}
    folds = data.get("folds", [])
    if not folds:
        return {"available": False, "reason": "No folds recorded in loco_metrics"}
    fold_curves: dict[str, Any] = {}
    any_history = False
    for f in folds:
        name, entry, has_hist = _extract_fold_entry(f)
        fold_curves[name] = entry
        if has_hist:
            any_history = True
    if not any_history:
        return {
            "available": False,
            "reason": "LOCO folds do not record epoch history (only best_val_loss recorded)",
            "folds": fold_curves,
        }
    return {"available": True, "folds": fold_curves}


def _extract_raw_curve(sub: dict[str, Any]) -> Any:
    """Extract raw curve payload from learning_curve or history key."""
    return sub.get("learning_curve") or sub.get("history")


def _curve_from_dict(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Validate and return curve dictionary if val_loss is present."""
    return raw if raw.get("val_loss") else None


def _curve_from_list(raw: list[Any]) -> dict[str, Any] | None:
    """Validate and return curve from history list if val_loss is present."""
    if not raw:
        return None
    first = raw[0]
    if isinstance(first, dict) and "val_loss" in first:
        return learning_curve(raw)
    return None


def _curve_from_raw(raw: Any) -> dict[str, Any] | None:
    """Convert dict or list history into a validated curve dict."""
    if isinstance(raw, dict):
        return _curve_from_dict(raw)
    if isinstance(raw, list):
        return _curve_from_list(raw)
    return None


def _anomaly_missing_reason(sub: dict[str, Any], key: str) -> str:
    """Describe why anomaly model validation history is missing."""
    if sub.get("learning_curve") is None:
        return f"Anomaly {key} model has key 'learning_curve' is null"
    return f"Anomaly {key} model has no validation loss history"


def _extract_anomaly_model_curve(sub: dict[str, Any], key: str) -> dict[str, Any]:
    """Extract single model curve or reason for missing validation history."""
    raw = _extract_raw_curve(sub)
    curve = _curve_from_raw(raw)
    if curve is not None:
        return {"available": True, "curve": curve}
    return {"available": False, "reason": _anomaly_missing_reason(sub, key)}


def extract_anomaly_learning_curves(path: Path) -> dict[str, Any]:
    """Extract learning curves for unsupervised anomaly detection models."""
    data = _load_json_if_exists(path)
    if data is None:
        return {"available": False, "reason": f"{path} not found"}
    models = {
        key: _extract_anomaly_model_curve(data.get(key, {}), key) for key in ("real", "synthetic")
    }
    available = any(m.get("available") is True for m in models.values())
    reason = (
        None
        if available
        else "Anomaly models do not record validation loss history (learning_curve is null)"
    )
    return {"available": available, "models": models, "reason": reason}


@functools.lru_cache(maxsize=1)
def _repo_parent_root() -> Path | None:
    """Locate root directory of parent repository across worktrees."""
    try:
        cmd = ["git", "rev-parse", "--git-common-dir"]
        common = subprocess.check_output(  # noqa: S603
            cmd, text=True, stderr=subprocess.DEVNULL
        ).strip()
        return Path(common).resolve().parent
    except (subprocess.SubprocessError, OSError):
        # Fall back if git directory inspection fails
        return None


def _find_loco_model_path(held_out: str, models_dir: Path | None = None) -> Path | None:
    """Locate loco checkpoint in local models_dir or parent repository."""
    m_dir = models_dir if models_dir is not None else Path("models")
    clean = held_out if held_out.startswith("crack_") else held_out.replace("loco_", "")
    filename = f"loco_{clean}.pt"
    local_p = m_dir / filename
    if local_p.exists():
        return local_p
    parent = _repo_parent_root()
    if parent is not None:
        cand = parent / "models" / filename
        if cand.exists():
            return cand
    return None


def _rebuild_loco_folds() -> dict[str, LocoFold] | None:
    """Rebuild LOCO folds from ML-NDT dataset exactly as run_loco does."""
    try:
        tr_batches = list_batches("training")
        val_batches = list_batches("validation")
        if not tr_batches:
            return None
        folds = make_folds(tr_batches, val_batches, seed=0)
        return {f.held_out: f for f in folds}
    except (OSError, ValueError, RuntimeError):
        # Narrow reason: ML-NDT dataset missing or unreadable
        return None


def _sample_flawed_train_items(
    fold: LocoFold, max_samples: int = 2000, seed: int = 42
) -> list[tuple[Path, int]]:
    """Sample up to max_samples flawed training images from fold with a seed."""
    ds = IndexedMlNdtDataset(fold.train)
    flaw_labels, eq_sizes = ds.get_labels()
    flawed_mask = (flaw_labels == 1.0) & (eq_sizes > 0.0)
    flawed_indices = np.where(flawed_mask)[0]
    if len(flawed_indices) == 0:
        return []
    if len(flawed_indices) <= max_samples:
        return [fold.train[i] for i in flawed_indices]
    rng = np.random.default_rng(seed)
    sampled_idx = rng.permutation(len(flawed_indices))[:max_samples]
    return [fold.train[flawed_indices[i]] for i in sampled_idx]


def _score_fold_train_sample(
    model_path: Path,
    sampled_items: list[tuple[Path, int]],
    device: str = "auto",
    threshold: float = 0.5,
) -> float:
    """Score flawed training images on GPU and return empirical detection rate."""
    if not sampled_items:
        return 0.0
    dev = pick_device(device)
    model, _ = load_model(model_path, device=str(dev))
    model.eval()
    ds = IndexedMlNdtDataset(sampled_items)
    cfg = TrainConfig(batch_size=64, device=str(dev))
    probs, _ = predict(model, ds, cfg)
    hits = (probs >= threshold).astype(int)
    return float(np.sum(hits) / len(sampled_items))


def _compute_fold_train_metric(
    held_out: str, fold: LocoFold | None, ctx: LocoEvalContext
) -> tuple[float | None, int, str | None]:
    """Score flawed training images of fold to obtain training detection rate."""
    model_p = _find_loco_model_path(held_out, ctx.models_dir)
    if model_p is None:
        return None, 0, f"Model loco_{held_out}.pt not found"
    if fold is None:
        return None, 0, "Could not rebuild LOCO folds (ML-NDT data not available)"
    flawed_samples = _sample_flawed_train_items(fold, max_samples=ctx.max_samples, seed=ctx.seed)
    if not flawed_samples:
        return None, 0, "No flawed training samples found in fold"
    rate = _score_fold_train_sample(model_p, flawed_samples, device=ctx.device)
    return rate, len(flawed_samples), None


def _extract_train_detection_rate(fold_dict: dict[str, Any]) -> float | None:
    """Extract training detection rate from fold record if present."""
    tr = fold_dict.get("train_detection")
    if isinstance(tr, dict) and "rate" in tr:
        return float(tr["rate"])
    if isinstance(tr, (int, float)):
        return float(tr)
    tr_rate = fold_dict.get("train_detection_rate")
    return float(tr_rate) if isinstance(tr_rate, (int, float)) else None


def _resolve_fold_train_metric(
    f: dict[str, Any],
    name: str,
    fold: LocoFold | None,
    ctx: LocoEvalContext,
) -> tuple[float | None, int, str | None]:
    """Retrieve recorded train detection rate or compute from fold items."""
    tr_det = _extract_train_detection_rate(f)
    if tr_det is not None:
        tr_info = f.get("train_detection")
        n_samples = int(tr_info.get("n_flawed", 0)) if isinstance(tr_info, dict) else 0
        return tr_det, n_samples, None
    return _compute_fold_train_metric(name, fold, ctx)


def _diagnose_single_loco_fold(
    f: dict[str, Any],
    rule: FitRule,
    fold: LocoFold | None = None,
    ctx: LocoEvalContext | None = None,
) -> tuple[str, dict[str, Any], bool]:
    """Diagnose single fold using recorded or computed train detection rate."""
    eval_ctx = ctx if ctx is not None else LocoEvalContext()
    name = str(f.get("held_out", "unknown"))
    det_info = f.get("detections")
    test_det = det_info.get("rate") if isinstance(det_info, dict) else None
    if test_det is None:
        return name, {"available": False, "reason": "Test detection rate not recorded"}, False

    tr_det, n_sampled, err = _resolve_fold_train_metric(f, name, fold, eval_ctx)
    if tr_det is None:
        return name, {"available": False, "reason": err or "Train metric unavailable"}, False

    diag = diagnose(float(tr_det), {name: float(test_det)}, rule)
    res = {
        "available": True,
        "train_detection_rate": float(tr_det),
        "heldout_detection_rate": float(test_det),
        "n_sampled": n_sampled,
        "seed": eval_ctx.seed,
        "diagnosis": diag,
    }
    return name, res, True


def _evaluate_all_loco_folds(
    folds: list[dict[str, Any]],
    rule: FitRule,
    rebuilt_folds: dict[str, LocoFold] | None,
    ctx: LocoEvalContext,
) -> tuple[dict[str, Any], bool]:
    """Evaluate each LOCO fold and track if any was recorded."""
    fold_diags: dict[str, Any] = {}
    any_recorded = False
    for f in folds:
        name = str(f.get("held_out", "unknown"))
        rebuilt = rebuilt_folds.get(name) if rebuilt_folds else None
        fname, res, recorded = _diagnose_single_loco_fold(f, rule, rebuilt, ctx)
        fold_diags[fname] = res
        if recorded:
            any_recorded = True
    return fold_diags, any_recorded


def diagnose_loco_folds(
    loco_data: dict[str, Any] | None,
    rule: FitRule,
    ctx: LocoEvalContext | None = None,
) -> dict[str, Any]:
    """Diagnose LOCO fold train-vs-test gap using computed or recorded train detection."""
    note = (
        "early stopping used the official validation images without the "
        "held-out crack, i.e. batch-held-out, not crack-held-out."
    )
    if not loco_data:
        return {
            "available": False,
            "reason": "results/loco_metrics.json not found",
            "rule": rule.rule_text,
            "note": note,
        }
    folds = loco_data.get("folds", [])
    if not folds:
        return {
            "available": False,
            "reason": "No folds recorded in loco_metrics",
            "rule": rule.rule_text,
            "note": note,
        }
    eval_ctx = ctx or LocoEvalContext()
    rebuilt = _rebuild_loco_folds()
    fold_diags, any_rec = _evaluate_all_loco_folds(folds, rule, rebuilt, eval_ctx)
    reason = None if any_rec else "Fold training detection rate could not be computed or found"
    return {
        "available": any_rec,
        "reason": reason,
        "folds": fold_diags,
        "rule": rule.rule_text,
        "note": note,
    }


def _extract_mlndt_heldout(per_split: dict[str, Any]) -> dict[str, float]:
    """Extract validation and test detection rates for ML-NDT diagnosis."""
    heldout: dict[str, float] = {}
    for name in ("validation", "test"):
        sp = per_split.get(name, {})
        det = sp.get("detection")
        if isinstance(det, dict) and "rate" in det:
            heldout[name] = float(det["rate"])
    return heldout


def diagnose_mlndt(mlndt_data: dict[str, Any] | None, rule: FitRule) -> dict[str, Any]:
    """Diagnose ML-NDT train-vs-test gap if training detection was recorded."""
    if not mlndt_data:
        return {
            "available": False,
            "reason": "results/mlndt_metrics.json not found",
            "rule": rule.rule_text,
        }
    per_split = mlndt_data.get("per_split", {})
    tr_det = (
        per_split.get("train", {}).get("detection", {}).get("rate")
        if isinstance(per_split.get("train"), dict)
        else None
    )
    if tr_det is None:
        return {
            "available": False,
            "reason": (
                "Training detection rate not recorded in mlndt_metrics.json "
                "(only test and validation splits evaluated)"
            ),
            "rule": rule.rule_text,
        }
    heldout = _extract_mlndt_heldout(per_split)
    return {
        "available": True,
        "diagnosis": diagnose(float(tr_det), heldout, rule),
        "rule": rule.rule_text,
    }


def _count_prediction_hits(hits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Tally detection counts, rates, and false calls from predicted hits."""
    flaw_mask = labels == 1
    clean_mask = labels == 0
    n_flawed = int(np.sum(flaw_mask))
    n_hit = int(np.sum(hits[flaw_mask]))
    n_clean = int(np.sum(clean_mask))
    n_false = int(np.sum(hits[clean_mask]))
    return {
        "detection_rate": float(n_hit / n_flawed) if n_flawed > 0 else 0.0,
        "false_call_rate": float(n_false / n_clean) if n_clean > 0 else 0.0,
        "n_flawed": n_flawed,
        "n_hit": n_hit,
        "n_clean": n_clean,
        "n_false": n_false,
    }


def evaluate_window_predictions(
    adapter: nn.Module,
    wset: WindowSet,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Calculate detection rate and false call rate on a window set."""
    if len(wset.x) == 0:
        return {
            "detection_rate": 0.0,
            "false_call_rate": 0.0,
            "n_flawed": 0,
            "n_hit": 0,
            "n_clean": 0,
            "n_false": 0,
        }
    ds = WindowDataset(wset.x, wset.y, wset.size_mm)
    cfg = TrainConfig(batch_size=256, device=str(device))
    probs, _ = predict(adapter, ds, cfg)
    hits = (probs >= threshold).astype(int)
    return _count_prediction_hits(hits, wset.y)


def _make_subset_datasets(
    full_ws: WindowSet, n_sub: int, seed: int
) -> tuple[WindowDataset, WindowDataset, int]:
    """Permute full window set and construct train and validation datasets."""
    n_total = len(full_ws.x)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)[:n_sub]
    sub_x = full_ws.x[indices]
    sub_y = full_ws.y[indices]
    sub_sz = full_ws.size_mm[indices]
    n_split = max(1, int(0.9 * n_sub))
    t_ds = WindowDataset(sub_x[:n_split], sub_y[:n_split], sub_sz[:n_split])
    v_ds = WindowDataset(sub_x[n_split:], sub_y[n_split:], sub_sz[n_split:])
    return t_ds, v_ds, n_split


def _train_window_subset(
    full_ws: WindowSet,
    fraction: float,
    cfg: FitDiagnosticsConfig,
    dev: torch.device,
) -> TrainerAdapter:
    """Subsample training windows and fit WindowNet 1D adapter."""
    n_sub = max(8, int(round(len(full_ws.x) * fraction)))
    t_ds, v_ds, n_split = _make_subset_datasets(full_ws, n_sub, cfg.seed)

    adapter = TrainerAdapter(WindowNet(in_channels=1, width=32))
    train_cfg = TrainConfig(
        epochs=1 if cfg.quick else cfg.epochs,
        batch_size=min(64, max(8, n_split)),
        patience=3,
        seed=cfg.seed,
        device=str(dev),
    )
    train_model(adapter, t_ds, v_ds, train_cfg)
    return adapter


def _eval_one_fraction(
    fraction: float,
    ctx: FractionEvalContext,
) -> dict[str, Any]:
    """Train on fraction and evaluate on in-dist and held-out regions."""
    adapter = _train_window_subset(ctx.full_ws, fraction, ctx.cfg, ctx.dev)
    tr_eval = evaluate_window_predictions(adapter, ctx.full_ws, ctx.dev)
    in_dist_eval = evaluate_window_predictions(adapter, ctx.bundles.in_dist, ctx.dev)

    reg_evals = {
        name: evaluate_window_predictions(adapter, wset, ctx.dev)
        for name, wset in ctx.bundles.regions.items()
    }
    test_sets: dict[str, Any] = {"in_distribution": in_dist_eval}
    test_sets.update(reg_evals)

    n_samples = max(8, int(round(len(ctx.full_ws.x) * fraction)))
    return {"n_train_samples": n_samples, "train": tr_eval, "test_sets": test_sets}


def _run_synthetic_data_size_curve(
    cfg: FitDiagnosticsConfig,
    dev: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build simulator datasets, run data-size curve, and return curve + diagnosis."""
    n_tr = 4 if cfg.quick else 40
    n_te = 2 if cfg.quick else 10
    sim_settings = SimGenSettings(n_train_bars=n_tr, n_test_bars=n_te, seed=cfg.seed)

    print("Generating simulated training windows...", flush=True)
    full_train_ws = build_train(sim_settings)
    print("Generating simulated test windows...", flush=True)
    in_dist_ws = build_in_distribution_test(sim_settings)
    reg_sets = {r.name: build_region_test(sim_settings, r) for r in DEFAULT_HOLDOUT_REGIONS}
    bundle = WindowEvalBundle(in_dist=in_dist_ws, regions=reg_sets)
    eval_ctx = FractionEvalContext(full_ws=full_train_ws, bundles=bundle, cfg=cfg, dev=dev)

    def _runner(f: float) -> dict[str, Any]:
        return _eval_one_fraction(f, eval_ctx)

    curve_records = data_size_curve(cfg.fractions, _runner)
    full_rec = curve_records[-1]
    train_det = float(full_rec["train"]["detection_rate"])
    heldout_det = {
        name: float(info["detection_rate"]) for name, info in full_rec["test_sets"].items()
    }

    rule = FitRule(target_train_metric=0.9, max_gap=0.1)
    synth_diag = diagnose(train_det, heldout_det, rule)
    synth_diag["available"] = True
    return curve_records, synth_diag


def _format_data_size_axes(ax: plt.Axes, title: str, ylabel: str) -> None:
    """Apply styling, grid, limits, and legends to data size axes."""
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Training Data Fraction (%)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=8)


def plot_data_size_curve(results: list[dict[str, Any]], out_path: Path) -> Path:
    """Render detection rate and false call rate vs training fraction."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fractions = [float(r["fraction"]) * 100.0 for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    test_keys = list(results[0].get("test_sets", {}).keys())

    markers = ["o", "s", "^", "v", "D"]
    for idx, key in enumerate(test_keys):
        m = markers[idx % len(markers)]
        det_rates = [float(r["test_sets"][key]["detection_rate"]) for r in results]
        fc_rates = [float(r["test_sets"][key]["false_call_rate"]) for r in results]
        label = key.replace("_", " ").title()
        ax1.plot(fractions, det_rates, label=label, marker=m, linewidth=1.5)
        ax2.plot(fractions, fc_rates, label=label, marker=m, linewidth=1.5)

    _format_data_size_axes(ax1, "Detection Rate vs Training Size", "Detection Rate")
    _format_data_size_axes(ax2, "False Call Rate vs Training Size", "False Call Rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_named_curves(curves_dict: dict[str, Any], figures_dir: Path, prefix: str) -> list[str]:
    """Plot curves from a name-to-entry dictionary."""
    paths: list[str] = []
    for name, data in curves_dict.items():
        if data.get("available") and "curve" in data:
            fp = figures_dir / f"{prefix}_{name}.png"
            title = f"{prefix.replace('_', ' ').title()} {name.title()}"
            plot_learning_curve(data["curve"], fp, title)
            data["figure"] = str(fp)
            paths.append(str(fp))
    return paths


def _save_learning_figures(learning_curves: dict[str, Any], figures_dir: Path) -> list[str]:
    """Plot available learning curves and return generated file paths."""
    fig_paths: list[str] = []
    ml_res = learning_curves.get("mlndt", {})
    if ml_res.get("available") and "curve" in ml_res:
        p = figures_dir / "fit_learning_curve_mlndt.png"
        plot_learning_curve(ml_res["curve"], p, "ML-NDT Learning Curve (Leaky Val)")
        ml_res["figure"] = str(p)
        fig_paths.append(str(p))

    loco_folds = learning_curves.get("loco", {}).get("folds", {})
    fig_paths.extend(_plot_named_curves(loco_folds, figures_dir, "fit_learning_curve_loco"))
    anom_models = learning_curves.get("anomaly", {}).get("models", {})
    fig_paths.extend(_plot_named_curves(anom_models, figures_dir, "fit_learning_curve_anomaly"))
    return fig_paths


def _collect_learning_curves(
    cfg: FitDiagnosticsConfig,
) -> tuple[dict[str, Any], list[str]]:
    """Extract learning curves and render corresponding figures."""
    lc_data = {
        "mlndt": extract_mlndt_learning_curve(cfg.mlndt_metrics_path),
        "loco": extract_loco_learning_curves(cfg.loco_metrics_path),
        "anomaly": extract_anomaly_learning_curves(cfg.anomaly_metrics_path),
    }
    return lc_data, _save_learning_figures(lc_data, cfg.figures_dir)


def _write_json_payload(out_json: Path, payload: dict[str, Any]) -> None:
    """Save formatted JSON payload to disk."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _assemble_payload(comp: FitPayloadComponents) -> dict[str, Any]:
    """Assemble final JSON payload dictionary."""
    return {
        "rule": comp.rule.rule_text,
        "fit_rule": {
            "target_train_metric": comp.rule.target_train_metric,
            "max_gap": comp.rule.max_gap,
            "text": comp.rule.rule_text,
        },
        "learning_curves": comp.lc_data,
        "data_size_curve": comp.ds_records,
        "diagnostics": {
            "synthetic_supervised": comp.synth_diag,
            "loco": comp.loco_diag,
            "mlndt": comp.mlndt_diag,
        },
        "figures": comp.fig_paths,
    }


def run_fit_diagnostics(cfg: FitDiagnosticsConfig) -> dict[str, Any]:
    """Execute complete fit diagnostics, generate figures, and export metrics."""
    set_seed(cfg.seed)
    dev = pick_device(cfg.device)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    lc_data, fig_paths = _collect_learning_curves(cfg)
    ds_records, synth_diag = _run_synthetic_data_size_curve(cfg, dev)
    ds_fig = cfg.figures_dir / "fit_data_size_curve.png"
    plot_data_size_curve(ds_records, ds_fig)
    fig_paths.append(str(ds_fig))

    rule = FitRule(target_train_metric=0.9, max_gap=0.1)
    loco_ctx = LocoEvalContext(models_dir=cfg.models_dir, device=cfg.device, seed=cfg.seed)
    loco_diag = diagnose_loco_folds(_load_json_if_exists(cfg.loco_metrics_path), rule, loco_ctx)
    mlndt_diag = diagnose_mlndt(_load_json_if_exists(cfg.mlndt_metrics_path), rule)

    payload = _assemble_payload(
        FitPayloadComponents(
            lc_data=lc_data,
            ds_records=ds_records,
            synth_diag=synth_diag,
            rule=rule,
            fig_paths=fig_paths,
            loco_diag=loco_diag,
            mlndt_diag=mlndt_diag,
        )
    )
    _write_json_payload(cfg.out_json, payload)
    return payload


def _parse_args() -> argparse.Namespace:
    """Parse CLI options for fit diagnostics experiment."""
    parser = argparse.ArgumentParser(description="Model fit diagnostics (project plan Phase 8)")
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer bars/epochs")
    parser.add_argument("--epochs", type=int, default=10, help="Epoch count for synthetic training")
    parser.add_argument(
        "--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing model checkpoints",
    )
    parser.add_argument("--out-json", type=Path, default=Path("results/fit_diagnostics.json"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    return parser.parse_args()


def main() -> None:
    """Entrypoint for CLI execution."""
    args = _parse_args()
    cfg = FitDiagnosticsConfig(
        epochs=args.epochs,
        quick=args.quick,
        device=args.device,
        models_dir=args.models_dir,
        out_json=args.out_json,
        figures_dir=args.figures_dir,
    )
    res = run_fit_diagnostics(cfg)
    print(f"Fit diagnostics saved to {cfg.out_json}", flush=True)
    print(f"Verdict: {res['diagnostics']['synthetic_supervised']['verdict']}")


if __name__ == "__main__":
    main()
