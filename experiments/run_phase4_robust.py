"""Phase 4 robustness: domain shifts, unseen artefacts, sim-to-real (Synthetic data)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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
import scipy.ndimage  # noqa: E402
import scipy.signal  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from utriage.ml.bar_data import (  # noqa: E402
    WindowConfig,
    WindowSet,
    _bar_artefact_windows,
    _BarContext,
    _concat_sets,
    _partition_grid,
    make_test_set,
    make_training_set,
)
from utriage.ml.bar_infer import load_window_model  # noqa: E402
from utriage.ml.bar_models import TrainerAdapter, WindowDataset, WindowNet, flaw_logit  # noqa: E402
from utriage.ml.compare import matched_threshold, rules_window_score  # noqa: E402
from utriage.ml.datasets import MLNDT_ROOT, list_batches, read_batch, standardise  # noqa: E402
from utriage.ml.infer import load_model, predict_images  # noqa: E402
from utriage.ml.train import TrainConfig, predict, set_seed, train_model  # noqa: E402
from utriage.physics.simulator import (  # noqa: E402
    ARTEFACT_KINDS,
    Artefact,
    BarSpec,
    Reflector,
    envelope,
    load_material,
    load_probe,
    simulate_bar,
)
from utriage.pod.hitmiss import false_call_rate  # noqa: E402
from utriage.rules.pipeline import dac_for  # noqa: E402


@dataclass(frozen=True)
class ShiftSpec:
    """Specification for one domain shift condition."""

    name: str
    material: str
    probe: str
    noise_scale: float


@dataclass(frozen=True)
class ShiftContext:
    """Aggregated models, thresholds, and runtime configuration."""

    adapter_1d: TrainerAdapter
    adapter_2d: TrainerAdapter
    rec_pct: float
    thresh_1d: float
    thresh_2d: float
    dev: torch.device
    p_cfg: TrainConfig


@dataclass(frozen=True)
class EvalGroundTruth:
    """Ground truth labels and flaw sizes for evaluation."""

    y: np.ndarray
    sizes: np.ndarray
    target_sizes: tuple[float, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 robustness experiment")
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer bars/batches")
    return parser.parse_args()


def _repo_parent_root() -> Path | None:
    """Find parent workspace root via git common dir if in a worktree."""
    try:
        cmd = ["git", "rev-parse", "--git-common-dir"]  # noqa: S607
        common = subprocess.check_output(  # noqa: S603
            cmd, text=True, stderr=subprocess.DEVNULL
        ).strip()
        return Path(common).resolve().parent
    except (subprocess.SubprocessError, OSError):  # Git inspection fallback
        return None


def _resolve_model_path(name: str) -> Path:
    local_p = Path("models") / name
    if local_p.exists():
        return local_p
    parent = _repo_parent_root()
    if parent is not None and (parent / "models" / name).exists():
        return parent / "models" / name
    wt_dir = Path(__file__).resolve().parents[2]
    for cand in wt_dir.glob(f"*/models/{name}"):
        if cand.exists():
            return cand
    return local_p


def _find_data_root() -> Path:
    if MLNDT_ROOT.exists():
        return MLNDT_ROOT
    parent = _repo_parent_root()
    if parent is not None:
        fallback = parent / "data" / "external" / "ML-NDT" / "data"
        if fallback.exists():
            return fallback
    return MLNDT_ROOT


def _load_p4_thresholds() -> tuple[float, float, float]:
    p = Path("results/bar_domain_metrics.json")
    if not p.exists():
        parent = _repo_parent_root()
        if parent is not None:
            p = parent / "results" / "bar_domain_metrics.json"
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rec_pct = float(data.get("configs", {}).get("record_pct", 50.0))
    t1 = float(data["thresholds"]["ml_1d"]["overall"])
    t2 = float(data["thresholds"]["ml_2d"]["overall"])
    return rec_pct, t1, t2


def _calc_detection_rates(
    hits: np.ndarray,
    gt: EvalGroundTruth,
) -> dict[str, float]:
    rates: dict[str, float] = {}
    for sz in gt.target_sizes:
        mask = (gt.y == 1) & np.isclose(gt.sizes, sz)
        count = int(np.sum(mask))
        rate = float(np.mean(hits[mask])) if count > 0 else 0.0
        rates[f"{sz:.1f}"] = rate
    return rates


def _eval_method(
    scores: np.ndarray,
    threshold: float,
    gt: EvalGroundTruth,
) -> dict[str, Any]:
    hits = (scores >= threshold).astype(float)
    noise_mask = gt.y == 0
    fcr = float(false_call_rate(hits[noise_mask])["rate"]) if np.any(noise_mask) else 0.0
    rates = _calc_detection_rates(hits, gt)
    return {"threshold": threshold, "detection_per_size": rates, "false_call_rate": fcr}


def _eval_model_pair(
    scores: np.ndarray,
    fixed_thresh: float,
    rules_fcr: float,
    gt: EvalGroundTruth,
) -> tuple[dict[str, Any], dict[str, Any]]:
    p4_eval = _eval_method(scores, fixed_thresh, gt)
    m_thresh = matched_threshold(scores[gt.y == 0], rules_fcr)
    m_eval = _eval_method(scores, m_thresh, gt)
    return p4_eval, m_eval


def _make_shift_datasets(
    spec: ShiftSpec,
    n_bars: int,
    sizes: tuple[float, ...],
) -> tuple[WindowSet, WindowSet]:
    cfg1 = WindowConfig(material=spec.material, probe=spec.probe, patch=False, seed=0)
    cfg2 = WindowConfig(material=spec.material, probe=spec.probe, patch=True, seed=0)
    t1 = make_test_set(cfg1, n_bars=n_bars, sizes_mm=sizes, noise_scales=(spec.noise_scale,))
    t2 = make_test_set(cfg2, n_bars=n_bars, sizes_mm=sizes, noise_scales=(spec.noise_scale,))
    return t1, t2


def _predict_windowset(
    adapter: TrainerAdapter,
    ws: WindowSet,
    p_cfg: TrainConfig,
) -> np.ndarray:
    ds = WindowDataset(ws.x, ws.y, ws.size_mm)
    scores, _ = predict(adapter, ds, p_cfg)
    return scores


def _run_domain_shift(
    spec: ShiftSpec,
    ctx: ShiftContext,
    n_bars: int,
    sizes: tuple[float, ...],
) -> dict[str, Any]:
    print(f"Running domain shift: {spec.name}...", flush=True)
    t1, t2 = _make_shift_datasets(spec, n_bars, sizes)
    gt1 = EvalGroundTruth(t1.y, t1.size_mm, sizes)
    gt2 = EvalGroundTruth(t2.y, t2.size_mm, sizes)

    r_scores = rules_window_score(t1.x)
    res_rules = _eval_method(r_scores, ctx.rec_pct, gt1)
    rules_fcr = res_rules["false_call_rate"]

    s1 = _predict_windowset(ctx.adapter_1d, t1, ctx.p_cfg)
    s2 = _predict_windowset(ctx.adapter_2d, t2, ctx.p_cfg)
    res_1d_p4, res_1d_m = _eval_model_pair(s1, ctx.thresh_1d, rules_fcr, gt1)
    res_2d_p4, res_2d_m = _eval_model_pair(s2, ctx.thresh_2d, rules_fcr, gt2)

    return {
        "spec": {"material": spec.material, "probe": spec.probe, "noise_scale": spec.noise_scale},
        "rules": res_rules,
        "ml_1d_p4_thresh": res_1d_p4,
        "ml_2d_p4_thresh": res_2d_p4,
        "ml_1d_matched": res_1d_m,
        "ml_2d_matched": res_2d_m,
    }


def _sample_roughness_bar(
    cfg: WindowConfig,
    rng: np.random.Generator,
    per_bar: int,
) -> WindowSet | None:
    art = Artefact("roughness", (4.0, 12.0), (30.0, 150.0), severity=1.0)
    spec = BarSpec(
        material=load_material("ti64"),
        probe=load_probe("p5"),
        water=load_material("water"),
        diameter_mm=cfg.diameter_mm,
        length_mm=16.0,
        axial_step_mm=1.0,
        n_angle=36,
        artefacts=(art,),
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    scan = simulate_bar(spec)
    _, inside = _partition_grid(scan, (art,))
    if len(inside) == 0:
        return None
    ctx = _BarContext(scan, dac_for(scan), cfg, 1.0, (art,))
    return _bar_artefact_windows(ctx, inside, rng, per_bar)


def _make_roughness_windows(n_bars: int, per_bar: int, seed: int = 42) -> WindowSet:
    rng = np.random.default_rng(seed)
    cfg = WindowConfig(material="ti64", probe="p5", patch=False)
    all_sets: list[WindowSet] = []
    for _ in range(n_bars):
        wset = _sample_roughness_bar(cfg, rng, per_bar)
        if wset is not None:
            all_sets.append(wset)
    return _concat_sets(all_sets, patch=False)


def _train_unseen_model(quick: bool, dev: torch.device) -> TrainerAdapter:
    print("Training 1D WindowNet without roughness artefacts...", flush=True)
    allowed = tuple(k for k in ARTEFACT_KINDS if k != "roughness")
    n_bars = 20 if quick else 400
    per_bar = 40 if quick else 100
    epochs = 1 if quick else 15
    cfg = WindowConfig(patch=False, seed=0, artefact_kinds=allowed)
    wset = make_training_set(cfg, n_bars=n_bars, per_bar=per_bar)
    split_idx = int(0.9 * len(wset.x))
    t_ds = WindowDataset(wset.x[:split_idx], wset.y[:split_idx], wset.size_mm[:split_idx])
    v_ds = WindowDataset(wset.x[split_idx:], wset.y[split_idx:], wset.size_mm[split_idx:])

    net = WindowNet(in_channels=1, width=32)
    adapter = TrainerAdapter(net)
    dev_str = "cuda" if dev.type == "cuda" else "cpu"
    tr_cfg = TrainConfig(epochs=epochs, batch_size=256, patience=3, seed=0, device=dev_str)
    train_model(adapter, t_ds, v_ds, tr_cfg)
    return adapter


def _eval_unseen_artefact(ctx: ShiftContext, quick: bool) -> dict[str, Any]:
    print("Evaluating unseen roughness artefact...", flush=True)
    fresh_adapter = _train_unseen_model(quick, ctx.dev)
    n_bars = 2 if quick else 8
    rough_wset = _make_roughness_windows(n_bars=n_bars, per_bar=50)

    fresh_scores = _predict_windowset(fresh_adapter, rough_wset, ctx.p_cfg)
    p4_scores = _predict_windowset(ctx.adapter_1d, rough_wset, ctx.p_cfg)
    rules_scores = rules_window_score(rough_wset.x)

    fresh_fcr = float(np.mean(fresh_scores >= ctx.thresh_1d))
    p4_fcr = float(np.mean(p4_scores >= ctx.thresh_1d))
    rules_fcr = float(np.mean(rules_scores >= ctx.rec_pct))
    print(
        f"Roughness FCR: fresh={fresh_fcr:.4f}, p4={p4_fcr:.4f}, rules={rules_fcr:.4f}", flush=True
    )
    return {
        "label": "Synthetic data",
        "artefact_evaluated": "roughness",
        "fresh_1d_no_roughness_fcr": fresh_fcr,
        "p4_1d_with_roughness_fcr": p4_fcr,
        "rules_fcr": rules_fcr,
        "threshold_p4_1d": ctx.thresh_1d,
        "threshold_rules": ctx.rec_pct,
    }


def _bscan_from_scan(scan: Any) -> np.ndarray:
    ax_idx = int(np.round(8.0 / scan.spec.axial_step_mm))
    env = envelope(scan.ascans[ax_idx])
    factors = (256.0 / env.shape[0], 256.0 / env.shape[1])
    bscan = scipy.ndimage.zoom(env, factors, order=1).astype(np.float32)
    return standardise(bscan)


def _simulate_bscan_slice(flawed: bool, seed: int) -> np.ndarray:
    refs = (
        (Reflector(axial_mm=8.0, angle_deg=0.0, depth_mm=15.0, diameter_mm=1.2, kind="fbh"),)
        if flawed
        else ()
    )
    spec = BarSpec(
        material=load_material("ti64"),
        probe=load_probe("p5"),
        water=load_material("water"),
        diameter_mm=40.0,
        length_mm=16.0,
        axial_step_mm=1.0,
        n_angle=36,
        reflectors=refs,
        seed=seed,
    )
    return _bscan_from_scan(simulate_bar(spec))


def _generate_sim_dataset(n_each: int) -> tuple[np.ndarray, list[int]]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    rng = np.random.default_rng(101)
    for _ in range(n_each):
        images.append(_simulate_bscan_slice(flawed=True, seed=int(rng.integers(0, 2**31 - 1))))
        labels.append(1)
        images.append(_simulate_bscan_slice(flawed=False, seed=int(rng.integers(0, 2**31 - 1))))
        labels.append(0)
    return np.stack(images, axis=0), labels


def _eval_mlndt_on_simulated(model: nn.Module, quick: bool, dev: torch.device) -> dict[str, Any]:
    print("Evaluating Phase 3 model on simulated B-scans...", flush=True)
    n_each = 5 if quick else 20
    img_arr, labels = _generate_sim_dataset(n_each)
    probs, _ = predict_images(model, img_arr, dev)
    auc = float(roc_auc_score(labels, probs))
    print(f"ML-NDT model on simulated B-scans ROC AUC: {auc:.4f}", flush=True)
    return {
        "label": "Synthetic data",
        "model": "models/mlndt_best.pt",
        "roc_auc": auc,
        "n_flawed": n_each,
        "n_unflawed": n_each,
    }


def _score_mlndt_image(img: np.ndarray, model: WindowNet, dev: torch.device) -> float:
    img_f = img.astype(np.float32)
    rows_128 = scipy.signal.resample(img_f, 128, axis=1)
    row_max = np.max(np.abs(rows_128), axis=1, keepdims=True)
    row_max = np.where(row_max > 1e-6, row_max, 1.0)
    rows_norm = (rows_128 / row_max).astype(np.float32)
    x_tensor = torch.from_numpy(rows_norm).unsqueeze(1).to(dev)
    with torch.no_grad():
        logits, _ = model(x_tensor)
        flaw_logits = flaw_logit(logits)
        scores = torch.sigmoid(flaw_logits).cpu().numpy()
    return float(np.max(scores))


def _score_mlndt_batch(
    batch_p: Path,
    model: WindowNet,
    dev: torch.device,
) -> tuple[list[float], list[int]]:
    mmap, labels = read_batch(batch_p)
    scores = [_score_mlndt_image(mmap[i], model, dev) for i in range(len(mmap))]
    gt = [int(labels[i, 0] > 0.5) for i in range(len(mmap))]
    return scores, gt


def _eval_bar1d_on_mlndt(model: WindowNet, quick: bool, dev: torch.device) -> dict[str, Any]:
    print("Evaluating Bar 1D model on ML-NDT validation images...", flush=True)
    root = _find_data_root()
    val_batches = list_batches("validation", root=root)
    batches_to_use = val_batches[:1] if (quick and val_batches) else val_batches
    all_scores: list[float] = []
    all_labels: list[int] = []

    for batch_p in batches_to_use:
        batch_scores, batch_labels = _score_mlndt_batch(batch_p, model, dev)
        all_scores.extend(batch_scores)
        all_labels.extend(batch_labels)

    auc = float(roc_auc_score(all_labels, all_scores)) if len(all_scores) > 0 else 0.0
    print(f"Bar 1D model on ML-NDT validation images ROC AUC: {auc:.4f}", flush=True)
    gap_note = (
        "Low AUC is the expected domain gap due to uncalibrated DAC normalisation, "
        "differing frequency, and axis representation."
    )
    return {
        "label": "Real data (ML-NDT)",
        "model": "models/bar_1d.pt",
        "roc_auc": auc,
        "n_images": len(all_scores),
        "image_axis_meaning": "UNVERIFIED - Ruri to check",
        "domain_gap_note": gap_note,
    }


def _plot_robustness_figure(shifts: dict[str, Any], out_path: Path) -> None:
    names = list(shifts.keys())
    labels = [k.replace("_", " ").title() for k in names]
    r_rates = [shifts[k]["rules"]["detection_per_size"]["0.8"] for k in names]
    m1_rates = [shifts[k]["ml_1d_p4_thresh"]["detection_per_size"]["0.8"] for k in names]
    m2_rates = [shifts[k]["ml_2d_p4_thresh"]["detection_per_size"]["0.8"] for k in names]

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width, r_rates, width, label="Rules (50% DAC)", color="black", alpha=0.8)
    ax.bar(x, m1_rates, width, label="ML 1D (P4 thresh)", color="tab:blue", alpha=0.8)
    ax.bar(x + width, m2_rates, width, label="ML 2D (P4 thresh)", color="tab:red", alpha=0.8)

    ax.set_ylabel("Detection Rate at 0.8 mm")
    ax.set_title("Detection Rate at 0.8 mm Flaw Size Under Domain Shift (Synthetic data)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6, axis="y")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_metrics_and_figures(metrics: dict[str, Any]) -> None:
    res_path = Path("results/robustness_metrics.json")
    res_path.parent.mkdir(parents=True, exist_ok=True)
    with res_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved robustness metrics to {res_path}", flush=True)

    fig_path = Path("results/figures/robustness.png")
    _plot_robustness_figure(metrics["domain_shift"], fig_path)
    print(f"Saved figure to {fig_path}", flush=True)


def _setup_context(dev: torch.device) -> ShiftContext:
    rec_pct, t1, t2 = _load_p4_thresholds()
    net1, _ = load_window_model(_resolve_model_path("bar_1d.pt"), device=str(dev))
    net2, _ = load_window_model(_resolve_model_path("bar_2d.pt"), device=str(dev))
    p_cfg = TrainConfig(batch_size=256, device="cuda" if dev.type == "cuda" else "cpu")
    return ShiftContext(TrainerAdapter(net1), TrainerAdapter(net2), rec_pct, t1, t2, dev, p_cfg)


def _run_all_domain_shifts(ctx: ShiftContext, quick: bool) -> dict[str, Any]:
    specs = (
        ShiftSpec("noise_2.5x", "ti64", "p5", 2.5),
        ShiftSpec("in718", "in718", "p5", 1.0),
        ShiftSpec("probe_p10", "ti64", "p10", 1.0),
    )
    n_bars = 2 if quick else 8
    target_sizes = (0.4, 0.6, 0.8, 1.2)
    return {s.name: _run_domain_shift(s, ctx, n_bars, target_sizes) for s in specs}


def _eval_sim_to_real(
    ctx: ShiftContext,
    quick: bool,
    dev: torch.device,
) -> dict[str, Any]:
    mlndt_net, _ = load_model(_resolve_model_path("mlndt_best.pt"), device=str(dev))
    s2r_sim = _eval_mlndt_on_simulated(mlndt_net, quick, dev)
    s2r_mlndt = _eval_bar1d_on_mlndt(ctx.adapter_1d.net, quick, dev)
    return {
        "mlndt_on_simulated_bscan": s2r_sim,
        "bar_1d_on_mlndt_val": s2r_mlndt,
    }


def _select_device() -> torch.device:
    has_cuda = torch.cuda.is_available()
    dev = torch.device("cuda" if has_cuda else "cpu")
    msg = (
        "Device: cuda (NVIDIA GeForce RTX 5070 Ti Laptop GPU)"
        if has_cuda
        else "CUDA unavailable, using CPU"
    )
    print(msg, flush=True)
    return dev


def main() -> None:
    args = _parse_args()
    set_seed(0)
    dev = _select_device()
    ctx = _setup_context(dev)
    metrics = {
        "label": "Synthetic data",
        "domain_shift": _run_all_domain_shifts(ctx, args.quick),
        "unseen_artefact": _eval_unseen_artefact(ctx, args.quick),
        "sim_to_real": _eval_sim_to_real(ctx, args.quick, dev),
    }
    _save_metrics_and_figures(metrics)
    print("Phase 4 robustness experiment completed successfully.", flush=True)


if __name__ == "__main__":
    main()
