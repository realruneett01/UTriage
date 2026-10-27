"""Model fit diagnostics and learning curve analysis (project plan Phase 8).

Evaluates learning curves, data-size scaling on synthetic supervised model,
and train-vs-test gaps with FitRule(target_train_metric=0.9, max_gap=0.1).
"""

from __future__ import annotations

import argparse
import json
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
from utriage.ml.diagnostics import (  # noqa: E402
    FitRule,
    data_size_curve,
    diagnose,
    learning_curve,
    plot_learning_curve,
)
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
    fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 1.0)
    epochs: int = 10
    quick: bool = False
    device: str = "auto"
    seed: int = 42


@dataclass(frozen=True)
class WindowEvalBundle:
    """Evaluation sets container for data-size benchmark."""

    in_dist: WindowSet
    regions: dict[str, WindowSet]


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


def extract_mlndt_learning_curve(path: Path) -> dict[str, Any]:
    """Extract learning curve from kept restart of ML-NDT results."""
    data = _load_json_if_exists(path)
    if data is None:
        return {"available": False, "reason": f"{path} not found"}
    restarts = data.get("restarts", [])
    if not restarts:
        return {"available": False, "reason": "No restarts recorded in mlndt_metrics"}
    kept_seed = data.get("kept_seed")
    matches = [r for r in restarts if r.get("seed") == kept_seed]
    restart = (
        matches[0]
        if matches
        else min(restarts, key=lambda r: float(r.get("best_val_loss", float("inf"))))
    )
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
        name = str(f.get("held_out", "unknown"))
        hist = f.get("history")
        if hist and isinstance(hist, list):
            any_history = True
            fold_curves[name] = {"available": True, "curve": learning_curve(hist)}
        else:
            fold_curves[name] = {
                "available": False,
                "reason": "No epoch history recorded for fold (only best_val_loss recorded)",
            }
    if not any_history:
        return {
            "available": False,
            "reason": "LOCO folds do not record epoch history (only best_val_loss recorded)",
            "folds": fold_curves,
        }
    return {"available": True, "folds": fold_curves}


def _extract_anomaly_model_curve(sub: dict[str, Any], key: str) -> dict[str, Any]:
    """Extract single model curve or reason for missing validation history."""
    lc = sub.get("learning_curve")
    if isinstance(lc, dict) and lc.get("val_loss"):
        return {"available": True, "curve": lc}
    if isinstance(lc, list) and lc and isinstance(lc[0], dict) and "val_loss" in lc[0]:
        return {"available": True, "curve": learning_curve(lc)}
    return {
        "available": False,
        "reason": f"Anomaly {key} model has no validation loss history",
    }


def extract_anomaly_learning_curves(path: Path) -> dict[str, Any]:
    """Extract learning curves for unsupervised anomaly detection models."""
    data = _load_json_if_exists(path)
    if data is None:
        return {"available": False, "reason": f"{path} not found"}
    models = {
        key: _extract_anomaly_model_curve(data.get(key, {}), key) for key in ("real", "synthetic")
    }
    available = any(m.get("available") is True for m in models.values())
    reason = None if available else "Anomaly models do not record validation loss history"
    return {"available": available, "models": models, "reason": reason}


def _extract_train_detection_rate(fold_dict: dict[str, Any]) -> float | None:
    """Extract training detection rate from fold record if present."""
    tr = fold_dict.get("train_detection")
    if isinstance(tr, dict) and "rate" in tr:
        return float(tr["rate"])
    if isinstance(tr, (int, float)):
        return float(tr)
    tr_rate = fold_dict.get("train_detection_rate")
    return float(tr_rate) if isinstance(tr_rate, (int, float)) else None


def _diagnose_single_loco_fold(
    f: dict[str, Any], rule: FitRule
) -> tuple[str, dict[str, Any], bool]:
    """Diagnose single fold if train detection was recorded."""
    name = str(f.get("held_out", "unknown"))
    tr_det = _extract_train_detection_rate(f)
    det_info = f.get("detections")
    test_det = det_info.get("rate") if isinstance(det_info, dict) else None
    if tr_det is not None and test_det is not None:
        diag = diagnose(float(tr_det), {name: float(test_det)}, rule)
        return name, {"available": True, "diagnosis": diag}, True
    return name, {"available": False, "reason": "Fold training detection rate not recorded"}, False


def diagnose_loco_folds(loco_data: dict[str, Any] | None, rule: FitRule) -> dict[str, Any]:
    """Diagnose LOCO fold train-vs-test gap if training detection was recorded."""
    if not loco_data:
        return {
            "available": False,
            "reason": "results/loco_metrics.json not found",
            "rule": rule.rule_text,
        }
    folds = loco_data.get("folds", [])
    if not folds:
        return {
            "available": False,
            "reason": "No folds recorded in loco_metrics",
            "rule": rule.rule_text,
        }
    fold_diags: dict[str, Any] = {}
    any_recorded = False
    for f in folds:
        name, res, recorded = _diagnose_single_loco_fold(f, rule)
        fold_diags[name] = res
        if recorded:
            any_recorded = True
    reason = (
        None
        if any_recorded
        else (
            "Fold training detection rate not recorded in loco_metrics.json "
            "(only test detections recorded)"
        )
    )
    return {
        "available": any_recorded,
        "reason": reason,
        "folds": fold_diags,
        "rule": rule.rule_text,
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
    full_ws: WindowSet,
    bundles: WindowEvalBundle,
    cfg: FitDiagnosticsConfig,
    dev: torch.device,
) -> dict[str, Any]:
    """Train on fraction and evaluate on in-dist and held-out regions."""
    adapter = _train_window_subset(full_ws, fraction, cfg, dev)
    tr_eval = evaluate_window_predictions(adapter, full_ws, dev)
    in_dist_eval = evaluate_window_predictions(adapter, bundles.in_dist, dev)

    reg_evals = {
        name: evaluate_window_predictions(adapter, wset, dev)
        for name, wset in bundles.regions.items()
    }
    test_sets: dict[str, Any] = {"in_distribution": in_dist_eval}
    test_sets.update(reg_evals)

    n_samples = max(8, int(round(len(full_ws.x) * fraction)))
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

    def _runner(f: float) -> dict[str, Any]:
        return _eval_one_fraction(f, full_train_ws, bundle, cfg, dev)

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


def _save_learning_figures(learning_curves: dict[str, Any], figures_dir: Path) -> list[str]:
    """Plot available learning curves and return generated file paths."""
    fig_paths: list[str] = []
    ml_res = learning_curves.get("mlndt", {})
    if ml_res.get("available") and "curve" in ml_res:
        p = figures_dir / "fit_learning_curve_mlndt.png"
        plot_learning_curve(ml_res["curve"], p, "ML-NDT Learning Curve (Leaky Val)")
        ml_res["figure"] = str(p)
        fig_paths.append(str(p))

    loco_res = learning_curves.get("loco", {})
    if loco_res.get("available") and "folds" in loco_res:
        for f_name, f_data in loco_res["folds"].items():
            if f_data.get("available") and "curve" in f_data:
                fp = figures_dir / f"fit_learning_curve_loco_{f_name}.png"
                plot_learning_curve(f_data["curve"], fp, f"LOCO Fold {f_name}")
                f_data["figure"] = str(fp)
                fig_paths.append(str(fp))
    return fig_paths


def _assemble_payload(
    lc_data: dict[str, Any],
    ds_records: list[dict[str, Any]],
    synth_diag: dict[str, Any],
    rule: FitRule,
    diagnostics_bundle: tuple[list[str], dict[str, Any], dict[str, Any]],
) -> dict[str, Any]:
    """Assemble final JSON payload dictionary."""
    fig_paths, loco_diag, mlndt_diag = diagnostics_bundle
    return {
        "rule": rule.rule_text,
        "fit_rule": {
            "target_train_metric": rule.target_train_metric,
            "max_gap": rule.max_gap,
            "text": rule.rule_text,
        },
        "learning_curves": lc_data,
        "data_size_curve": ds_records,
        "diagnostics": {
            "synthetic_supervised": synth_diag,
            "loco": loco_diag,
            "mlndt": mlndt_diag,
        },
        "figures": fig_paths,
    }


def run_fit_diagnostics(cfg: FitDiagnosticsConfig) -> dict[str, Any]:
    """Execute complete fit diagnostics, generate figures, and export metrics."""
    set_seed(cfg.seed)
    dev = pick_device(cfg.device)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    lc_data = {
        "mlndt": extract_mlndt_learning_curve(cfg.mlndt_metrics_path),
        "loco": extract_loco_learning_curves(cfg.loco_metrics_path),
        "anomaly": extract_anomaly_learning_curves(cfg.anomaly_metrics_path),
    }
    fig_paths = _save_learning_figures(lc_data, cfg.figures_dir)

    ds_records, synth_diag = _run_synthetic_data_size_curve(cfg, dev)
    ds_fig = cfg.figures_dir / "fit_data_size_curve.png"
    plot_data_size_curve(ds_records, ds_fig)
    fig_paths.append(str(ds_fig))

    rule = FitRule(target_train_metric=0.9, max_gap=0.1)
    loco_diag = diagnose_loco_folds(_load_json_if_exists(cfg.loco_metrics_path), rule)
    mlndt_diag = diagnose_mlndt(_load_json_if_exists(cfg.mlndt_metrics_path), rule)

    payload = _assemble_payload(
        lc_data, ds_records, synth_diag, rule, (fig_paths, loco_diag, mlndt_diag)
    )
    cfg.out_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    """Parse CLI options for fit diagnostics experiment."""
    parser = argparse.ArgumentParser(description="Model fit diagnostics (project plan Phase 8)")
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer bars/epochs")
    parser.add_argument("--epochs", type=int, default=10, help="Epoch count for synthetic training")
    parser.add_argument(
        "--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')"
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
        out_json=args.out_json,
        figures_dir=args.figures_dir,
    )
    res = run_fit_diagnostics(cfg)
    print(f"Fit diagnostics saved to {cfg.out_json}", flush=True)
    print(f"Verdict: {res['diagnostics']['synthetic_supervised']['verdict']}")


if __name__ == "__main__":
    main()
