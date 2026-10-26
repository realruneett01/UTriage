"""Phase 4 experiment: Bar-domain ML vs rules engine under grain noise (Synthetic data)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Ensure local workspace src is prioritized over external paths
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        sys.modules["utriage"].__path__.insert(0, _pkg_dir)
if "utriage.ml" in sys.modules:
    _ml_dir = str(Path(_SRC) / "utriage" / "ml")
    if _ml_dir not in sys.modules["utriage.ml"].__path__:
        sys.modules["utriage.ml"].__path__.insert(0, _ml_dir)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from utriage.ml.bar_data import (  # noqa: E402
    WindowConfig,
    WindowSet,
    make_test_set,
    make_training_set,
)
from utriage.ml.bar_models import TrainerAdapter, WindowDataset, WindowNet  # noqa: E402
from utriage.ml.compare import (  # noqa: E402
    detection_table,
    matched_threshold,
    pod_summary,
    rules_window_score,
)
from utriage.ml.infer import save_model  # noqa: E402
from utriage.ml.train import TrainConfig, predict, set_seed, train_model  # noqa: E402
from utriage.pod.ahat import Censoring, ahat_to_dict, fit_ahat  # noqa: E402
from utriage.pod.hitmiss import false_call_rate  # noqa: E402
from utriage.rules.config import load_acceptance  # noqa: E402


@dataclass(frozen=True)
class TestSettings:
    """Fixed flaw sizes and noise scales for test bars."""

    sizes: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2)
    scales: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)


@dataclass(frozen=True)
class TrainRunData:
    """Packaged datasets for 1D and 2D models."""

    ds_train_1d: WindowDataset
    ds_val_1d: WindowDataset
    ds_train_2d: WindowDataset
    ds_val_2d: WindowDataset
    wset_1d: WindowSet
    wset_2d: WindowSet


@dataclass(frozen=True)
class ModelPair:
    """Pair of 1D and 2D trainer adapters."""

    adapter_1d: TrainerAdapter
    adapter_2d: TrainerAdapter


@dataclass(frozen=True)
class TestRunData:
    """Packaged test window sets and test parameters."""

    test_1d: WindowSet
    test_2d: WindowSet
    settings: TestSettings


@dataclass(frozen=True)
class EvalContext:
    """Summary of all evaluation metrics and artifacts."""

    eval_by_method: dict[str, dict[float, Any]]
    thresholds: dict[str, Any]
    ahat_info: dict[str, Any]
    latency: dict[str, Any]
    kept_model: dict[str, str]


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Phase 4 experiment."""
    parser = argparse.ArgumentParser(description="Phase 4 bar-domain ML experiment")
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer bars")
    return parser.parse_args()


def _load_or_create_train(
    cfg: WindowConfig,
    n_bars: int,
    per_bar: int,
    cache_path: Path,
) -> WindowSet:
    """Load cached training set or generate and cache it."""
    if cache_path.exists():
        print(f"Loading cached training set from {cache_path}", flush=True)
        return WindowSet.load(cache_path)
    print(f"Generating training set (patch={cfg.patch}, bars={n_bars})...", flush=True)
    wset = make_training_set(cfg, n_bars=n_bars, per_bar=per_bar)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    wset.save(cache_path)
    print(f"Saved {len(wset.x)} windows to {cache_path}", flush=True)
    return wset


def _split_train_val(
    wset: WindowSet,
    val_frac: float = 0.1,
) -> tuple[WindowDataset, WindowDataset]:
    """Hold out 10% of windows by index for validation."""
    n = len(wset.x)
    split_idx = int((1.0 - val_frac) * n)
    t_ds = WindowDataset(wset.x[:split_idx], wset.y[:split_idx], wset.size_mm[:split_idx])
    v_ds = WindowDataset(wset.x[split_idx:], wset.y[split_idx:], wset.size_mm[split_idx:])
    return t_ds, v_ds


def _get_training_data(quick: bool) -> TrainRunData:
    """Prepare 1D and 2D training sets with validation splits."""
    n_bars = 20 if quick else 400
    per_bar = 40 if quick else 100
    tag = "quick" if quick else "full"
    c_dir = Path("data/synthetic/bar_windows")
    cfg_1d = WindowConfig(patch=False, seed=0)
    cfg_2d = WindowConfig(patch=True, seed=0)
    wset_1d = _load_or_create_train(cfg_1d, n_bars, per_bar, c_dir / f"train_1d_{tag}.npz")
    wset_2d = _load_or_create_train(cfg_2d, n_bars, per_bar, c_dir / f"train_2d_{tag}.npz")
    t_1d, v_1d = _split_train_val(wset_1d)
    t_2d, v_2d = _split_train_val(wset_2d)
    return TrainRunData(t_1d, v_1d, t_2d, v_2d, wset_1d, wset_2d)


def _train_bar_model(
    in_channels: int,
    train_ds: WindowDataset,
    val_ds: WindowDataset,
    out_path: Path,
) -> tuple[WindowNet, TrainerAdapter]:
    """Train WindowNet via TrainerAdapter and save checkpoint."""
    net = WindowNet(in_channels=in_channels, width=32)
    adapter = TrainerAdapter(net)
    cfg = TrainConfig(epochs=15, batch_size=256, patience=3, seed=0, device="auto")
    print(f"Training WindowNet(in_channels={in_channels}) on {cfg.device}...", flush=True)
    train_model(adapter, train_ds, val_ds, cfg)
    save_model(adapter.net, out_path, {"in_channels": in_channels, "width": 32})
    print(f"Saved model to {out_path}", flush=True)
    return net, adapter


def _train_both_models(tr_data: TrainRunData) -> ModelPair:
    """Train both 1D and 2D bar models and return their adapters."""
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    _, a_1d = _train_bar_model(1, tr_data.ds_train_1d, tr_data.ds_val_1d, models_dir / "bar_1d.pt")
    _, a_2d = _train_bar_model(9, tr_data.ds_train_2d, tr_data.ds_val_2d, models_dir / "bar_2d.pt")
    return ModelPair(a_1d, a_2d)


def _load_or_create_test(
    cfg: WindowConfig,
    n_bars: int,
    cache_path: Path,
    settings: TestSettings,
) -> WindowSet:
    """Load cached test set or simulate full test bars."""
    if cache_path.exists():
        print(f"Loading cached test set from {cache_path}", flush=True)
        return WindowSet.load(cache_path)
    print(f"Generating test set (patch={cfg.patch}, bars={n_bars})...", flush=True)
    wset = make_test_set(cfg, n_bars=n_bars, sizes_mm=settings.sizes, noise_scales=settings.scales)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    wset.save(cache_path)
    print(f"Saved {len(wset.x)} test windows to {cache_path}", flush=True)
    return wset


def _get_test_data(quick: bool) -> TestRunData:
    """Prepare 1D and 2D test window sets."""
    settings = TestSettings()
    n_bars = 2 if quick else 8
    tag = "quick" if quick else "full"
    c_dir = Path("data/synthetic/bar_windows")
    cfg_1d = WindowConfig(patch=False, seed=0)
    cfg_2d = WindowConfig(patch=True, seed=0)
    t_1d = _load_or_create_test(cfg_1d, n_bars, c_dir / f"test_1d_{tag}.npz", settings)
    t_2d = _load_or_create_test(cfg_2d, n_bars, c_dir / f"test_2d_{tag}.npz", settings)
    return TestRunData(t_1d, t_2d, settings)


def _sync_cuda(device: torch.device) -> None:
    """Synchronize CUDA stream if device is a CUDA GPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_window_pass(model: nn.Module, dummy: torch.Tensor, device: torch.device) -> float:
    """Time a single forward pass with CUDA sync."""
    _sync_cuda(device)
    t0 = time.perf_counter()
    _ = model(dummy)
    _sync_cuda(device)
    return time.perf_counter() - t0


def _bench_latency(
    model: nn.Module,
    in_channels: int,
    device: torch.device,
    batch: int,
) -> float:
    """Measure median milliseconds per window after warmup."""
    model.eval()
    model.to(device)
    dummy = torch.zeros((batch, in_channels, 128), dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
        _sync_cuda(device)
        timings = [(_time_window_pass(model, dummy, device) * 1000.0) / batch for _ in range(50)]
    return float(np.median(timings))


def _model_latencies(
    model: nn.Module,
    in_channels: int,
    dev_gpu: torch.device,
    dev_cpu: torch.device,
) -> dict[str, float]:
    """Measure latencies for one model across GPU and CPU."""
    return {
        "gpu_batch_1_ms": _bench_latency(model, in_channels, dev_gpu, 1),
        "gpu_batch_256_ms": _bench_latency(model, in_channels, dev_gpu, 256),
        "cpu_batch_1_ms": _bench_latency(model, in_channels, dev_cpu, 1),
    }


def _measure_all_latencies(models: ModelPair) -> dict[str, dict[str, float]]:
    """Measure latencies for 1D and 2D models across GPU and CPU."""
    has_cuda = torch.cuda.is_available()
    dev_gpu = torch.device("cuda") if has_cuda else torch.device("cpu")
    dev_cpu = torch.device("cpu")
    return {
        "ml_1d": _model_latencies(models.adapter_1d.net, 1, dev_gpu, dev_cpu),
        "ml_2d": _model_latencies(models.adapter_2d.net, 9, dev_gpu, dev_cpu),
    }


def _eval_method_at_noise(
    hits: np.ndarray,
    noise_hits: np.ndarray,
    sizes: np.ndarray,
    noise_scale: float,
) -> dict[str, Any]:
    """Calculate detection table, POD summary, and FCR for one method at one noise scale."""
    det_tab = detection_table(hits, sizes, np.full(len(sizes), noise_scale))
    pod_res = pod_summary(sizes, hits)
    fcr_res = false_call_rate(noise_hits)
    return {
        "detection_table": det_tab,
        "pod_summary": pod_res,
        "false_call_rate": fcr_res,
    }


def _fit_rules_ahat(
    sizes: np.ndarray,
    scores: np.ndarray,
    decision: float,
) -> dict[str, Any]:
    """Fit Tobit ahat model on rules scores."""
    censor = Censoring(low=None, high=None)
    try:
        fit = fit_ahat(sizes, scores, decision=decision, censor=censor)
        return {"fit": ahat_to_dict(fit), "reason": None}
    except (
        ValueError,
        RuntimeError,
    ) as exc:  # MLE optimization failure, singular Hessian, or invalid sample data
        return {"fit": None, "reason": str(exc)}


def _decide_kept_model(
    results_1d: dict[float, Any],
    results_2d: dict[float, Any],
    highest_noise: float,
) -> dict[str, str]:
    """Select between 1D and 2D models using POD metric at highest noise scale."""
    rule = (
        "Keep simpler 1D model unless 2D a90_95 (or binomial bound) is clearly better (>= 0.1 mm)."
    )
    pod_1d = results_1d[highest_noise]["pod_summary"]
    pod_2d = results_2d[highest_noise]["pod_summary"]
    val_1d = pod_1d["pod"]["a90_95"] if pod_1d.get("pod") else pod_1d["binomial"]["pod_lower95"]
    val_2d = pod_2d["pod"]["a90_95"] if pod_2d.get("pod") else pod_2d["binomial"]["pod_lower95"]
    better = (val_2d < val_1d - 0.1) if pod_1d.get("pod") else (val_2d > val_1d + 0.1)
    choice = "ml_2d" if better else "ml_1d"
    reason = (
        f"At noise scale {highest_noise}, 1D metric is {val_1d:.4f} and 2D metric is {val_2d:.4f}; "
        f"2D {'is' if better else 'is not'} clearly better."
    )
    return {"choice": choice, "rule": rule, "reason": reason}


def _compute_matched_thresholds(
    t_data: TestRunData,
    scores_1d: np.ndarray,
    scores_2d: np.ndarray,
    rules_hits: np.ndarray,
) -> tuple[dict[str, float], dict[str, float], dict[float, float]]:
    """Compute matched false-call rate thresholds per noise scale and overall."""
    fcr_dict: dict[float, float] = {}
    thresh_1d: dict[str, float] = {}
    thresh_2d: dict[str, float] = {}
    for ns in t_data.settings.scales:
        m = (t_data.test_1d.y == 0) & np.isclose(t_data.test_1d.noise_scale, ns)
        fcr = float(false_call_rate(rules_hits[m])["rate"])
        fcr_dict[ns] = fcr
        thresh_1d[str(ns)] = matched_threshold(scores_1d[m], fcr)
        m_2d = (t_data.test_2d.y == 0) & np.isclose(t_data.test_2d.noise_scale, ns)
        thresh_2d[str(ns)] = matched_threshold(scores_2d[m_2d], fcr)
    ov_m = t_data.test_1d.y == 0
    ov_fcr = float(false_call_rate(rules_hits[ov_m])["rate"])
    thresh_1d["overall"] = matched_threshold(scores_1d[ov_m], ov_fcr)
    ov_m2 = t_data.test_2d.y == 0
    thresh_2d["overall"] = matched_threshold(scores_2d[ov_m2], ov_fcr)
    return thresh_1d, thresh_2d, fcr_dict


def _evaluate_all_methods(
    t_data: TestRunData,
    scores_1d: np.ndarray,
    scores_2d: np.ndarray,
    rules_scores: np.ndarray,
) -> EvalContext:
    """Evaluate rules, ML 1D, and ML 2D at matched false-call rates."""
    rec_pct = float(load_acceptance().amplitude.record_pct)
    rules_hits = (rules_scores >= rec_pct).astype(float)
    t_1d, t_2d, r_fcr = _compute_matched_thresholds(t_data, scores_1d, scores_2d, rules_hits)
    res_r: dict[float, Any] = {}
    res_1: dict[float, Any] = {}
    res_2: dict[float, Any] = {}
    for ns in t_data.settings.scales:
        f_m = (t_data.test_1d.y == 1) & np.isclose(t_data.test_1d.noise_scale, ns)
        n_m = (t_data.test_1d.y == 0) & np.isclose(t_data.test_1d.noise_scale, ns)
        sz = t_data.test_1d.size_mm[f_m]
        res_r[ns] = _eval_method_at_noise(rules_hits[f_m], rules_hits[n_m], sz, ns)
        h1 = (scores_1d >= t_1d[str(ns)]).astype(float)
        res_1[ns] = _eval_method_at_noise(h1[f_m], h1[n_m], sz, ns)
        f_m2 = (t_data.test_2d.y == 1) & np.isclose(t_data.test_2d.noise_scale, ns)
        n_m2 = (t_data.test_2d.y == 0) & np.isclose(t_data.test_2d.noise_scale, ns)
        h2 = (scores_2d >= t_2d[str(ns)]).astype(float)
        res_2[ns] = _eval_method_at_noise(h2[f_m2], h2[n_m2], t_data.test_2d.size_mm[f_m2], ns)
    flaw_all = t_data.test_1d.y == 1
    ahat_info = {
        "overall": _fit_rules_ahat(
            t_data.test_1d.size_mm[flaw_all], rules_scores[flaw_all], rec_pct
        ),
        "per_noise": {
            str(ns): _fit_rules_ahat(
                t_data.test_1d.size_mm[flaw_all & np.isclose(t_data.test_1d.noise_scale, ns)],
                rules_scores[flaw_all & np.isclose(t_data.test_1d.noise_scale, ns)],
                rec_pct,
            )
            for ns in t_data.settings.scales
        },
    }
    kept = _decide_kept_model(res_1, res_2, max(t_data.settings.scales))
    all_eval = {"rules": res_r, "ml_1d": res_1, "ml_2d": res_2}
    thresh_all = {"ml_1d": t_1d, "ml_2d": t_2d, "rules_fcr": {str(k): v for k, v in r_fcr.items()}}
    return EvalContext(all_eval, thresh_all, ahat_info, {}, kept)


_METHOD_STYLES: tuple[tuple[str, str, str, str], ...] = (
    ("rules", "Rules", "black", "s"),
    ("ml_1d", "ML 1D", "blue", "o"),
    ("ml_2d", "ML 2D", "red", "^"),
)


def _style_detection_axis(ax: plt.Axes, title: str, xlabel: str) -> None:
    """Apply standard formatting, labels, limits, and grid to a rate axis."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()


def _finish_detection_plot(
    fig: plt.Figure,
    axes: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """Apply overall labels, save figure to disk, and close figure."""
    axes[0].set_ylabel("Detection Rate")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _rate_at_size(table: list[dict[str, Any]], target_size: float) -> float:
    """Find detection rate for a target flaw size from detection table."""
    matches = [float(row["rate"]) for row in table if np.isclose(row["size"], target_size)]
    return matches[0] if matches else 0.0


def _method_rates_for_size(
    eval_by_scale: dict[float, Any],
    scales: tuple[float, ...],
    size: float,
) -> list[float]:
    """Collect detection rates for one method at a target size across noise scales."""
    return [_rate_at_size(eval_by_scale[sc]["detection_table"], size) for sc in scales]


def _plot_size_detection_axis(
    ax: plt.Axes,
    eval_by_method: dict[str, dict[float, Any]],
    scales: tuple[float, ...],
    size: float,
) -> None:
    """Plot detection rate curves for all methods on one flaw size axis."""
    for m_key, m_lbl, col, mk in _METHOD_STYLES:
        rates = _method_rates_for_size(eval_by_method[m_key], scales, size)
        ax.plot(scales, rates, label=m_lbl, color=col, marker=mk, linewidth=2)
    _style_detection_axis(ax, f"Flaw Size {size:.1f} mm", "Noise Scale")


def _plot_detection_vs_noise(
    eval_by_method: dict[str, dict[float, Any]],
    scales: tuple[float, ...],
    out_path: Path,
) -> None:
    """Plot detection rate vs noise scale at matched FCR for 0.4 and 0.8 mm flaws."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    target_sizes = (0.4, 0.8)
    for ax, sz in zip(axes, target_sizes, strict=True):
        _plot_size_detection_axis(ax, eval_by_method, scales, sz)
    title = "Detection Rate vs Noise Scale at Matched FCR (Synthetic data)"
    _finish_detection_plot(fig, axes, title, out_path)


def _plot_scale_pod_axis(
    ax: plt.Axes,
    eval_by_method: dict[str, dict[float, Any]],
    scale: float,
    sizes: tuple[float, ...],
) -> None:
    """Plot detection rate vs flaw size for all methods at one noise scale."""
    for m_key, m_lbl, col, mk in _METHOD_STYLES:
        tbl = eval_by_method[m_key][scale]["detection_table"]
        sz_map = {row["size"]: row["rate"] for row in tbl}
        rates = [sz_map.get(s, 0.0) for s in sizes]
        ax.plot(sizes, rates, label=m_lbl, color=col, marker=mk, linewidth=2)
    _style_detection_axis(ax, f"Noise Scale {scale:.1f}x", "Flaw Diameter (mm)")


def _plot_pod_curves(
    eval_by_method: dict[str, dict[float, Any]],
    sizes: tuple[float, ...],
    out_path: Path,
) -> None:
    """Plot detection rate vs flaw size at noise scales 1.5 and 2.0."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    target_scales = (1.5, 2.0)
    for ax, sc in zip(axes, target_scales, strict=True):
        _plot_scale_pod_axis(ax, eval_by_method, sc, sizes)
    title = "Detection Rate vs Flaw Size by Method (Synthetic data)"
    _finish_detection_plot(fig, axes, title, out_path)


def _save_metrics_and_figures(
    tr_data: TrainRunData,
    ctx: EvalContext,
    settings: TestSettings,
) -> None:
    """Save metrics JSON and output figures with Synthetic data labels."""
    metrics_path = Path("results/bar_domain_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    record_pct = float(load_acceptance().amplitude.record_pct)
    payload = {
        "label": "Synthetic data",
        "configs": {
            "record_pct": record_pct,
            "noise_scales": list(settings.scales),
            "sizes_mm": list(settings.sizes),
        },
        "train_sizes": {
            "n_train_1d": len(tr_data.ds_train_1d),
            "n_val_1d": len(tr_data.ds_val_1d),
            "n_train_2d": len(tr_data.ds_train_2d),
            "n_val_2d": len(tr_data.ds_val_2d),
        },
        "thresholds": ctx.thresholds,
        "per_method_per_noise": {
            m: {str(k): v for k, v in res.items()} for m, res in ctx.eval_by_method.items()
        },
        "ahat_rules": ctx.ahat_info,
        "latency": ctx.latency,
        "kept_model": ctx.kept_model,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved metrics to {metrics_path}", flush=True)

    fig_dir = Path("results/figures")
    _plot_detection_vs_noise(
        ctx.eval_by_method, settings.scales, fig_dir / "bar_detection_vs_noise.png"
    )
    _plot_pod_curves(ctx.eval_by_method, settings.sizes, fig_dir / "bar_pod_curves.png")
    print("Saved evaluation figures to results/figures/", flush=True)


def main() -> None:
    """Execute Phase 4 bar-domain ML experiment."""
    args = _parse_args()
    set_seed(0)
    dev_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Device: {dev_name} (NVIDIA GeForce RTX 5070 Ti Laptop GPU)"
        if dev_name == "cuda"
        else "CUDA unavailable, using CPU",
        flush=True,
    )
    tr_data = _get_training_data(args.quick)
    models = _train_both_models(tr_data)
    t_data = _get_test_data(args.quick)

    rules_scores = rules_window_score(t_data.test_1d.x)
    p_cfg = TrainConfig(batch_size=256, device="auto")
    ds_t1 = WindowDataset(t_data.test_1d.x, t_data.test_1d.y, t_data.test_1d.size_mm)
    ds_t2 = WindowDataset(t_data.test_2d.x, t_data.test_2d.y, t_data.test_2d.size_mm)
    scores_1d, _ = predict(models.adapter_1d, ds_t1, p_cfg)
    scores_2d, _ = predict(models.adapter_2d, ds_t2, p_cfg)

    ctx = _evaluate_all_methods(t_data, scores_1d, scores_2d, rules_scores)
    lat = _measure_all_latencies(models)
    full_ctx = EvalContext(ctx.eval_by_method, ctx.thresholds, ctx.ahat_info, lat, ctx.kept_model)
    _save_metrics_and_figures(tr_data, full_ctx, t_data.settings)
    print("Phase 4 completed successfully.", flush=True)


if __name__ == "__main__":
    main()
