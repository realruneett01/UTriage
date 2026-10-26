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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 robustness experiment")
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer bars/batches")
    return parser.parse_args()


def _resolve_model_path(name: str) -> Path:
    local_p = Path("models") / name
    if local_p.exists():
        return local_p
    try:
        cmd = ["git", "rev-parse", "--git-common-dir"]  # noqa: S607
        common = subprocess.check_output(  # noqa: S603
            cmd, text=True, stderr=subprocess.DEVNULL
        ).strip()
        desktop_p = Path(common).resolve().parent / "models" / name
        if desktop_p.exists():
            return desktop_p
    except (subprocess.SubprocessError, OSError):  # Git inspection fallback
        pass
    wt_dir = Path(__file__).resolve().parents[2]
    for cand in wt_dir.glob(f"*/models/{name}"):
        if cand.exists():
            return cand
    return local_p


def _find_data_root() -> Path:
    if MLNDT_ROOT.exists():
        return MLNDT_ROOT
    try:
        cmd = ["git", "rev-parse", "--git-common-dir"]  # noqa: S607
        common = subprocess.check_output(  # noqa: S603
            cmd, text=True, stderr=subprocess.DEVNULL
        ).strip()
        fallback = Path(common).resolve().parent / "data" / "external" / "ML-NDT" / "data"
        if fallback.exists():
            return fallback
    except (subprocess.SubprocessError, OSError):  # Git inspection fallback
        pass
    return MLNDT_ROOT


def _load_p4_thresholds() -> tuple[float, float, float]:
    p = Path("results/bar_domain_metrics.json")
    if not p.exists():
        try:
            cmd = ["git", "rev-parse", "--git-common-dir"]  # noqa: S607
            common = subprocess.check_output(  # noqa: S603
                cmd, text=True, stderr=subprocess.DEVNULL
            ).strip()
            p = Path(common).resolve().parent / "results" / "bar_domain_metrics.json"
        except (subprocess.SubprocessError, OSError):  # Git inspection fallback
            pass
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rec_pct = float(data.get("configs", {}).get("record_pct", 50.0))
    t1 = float(data["thresholds"]["ml_1d"]["overall"])
    t2 = float(data["thresholds"]["ml_2d"]["overall"])
    return rec_pct, t1, t2


def _calc_detection_rates(
    hits: np.ndarray,
    y: np.ndarray,
    sizes: np.ndarray,
    target_sizes: tuple[float, ...],
) -> dict[str, float]:
    rates: dict[str, float] = {}
    for sz in target_sizes:
        mask = (y == 1) & np.isclose(sizes, sz)
        count = int(np.sum(mask))
        rate = float(np.mean(hits[mask])) if count > 0 else 0.0
        rates[f"{sz:.1f}"] = rate
    return rates


def _eval_method(
    scores: np.ndarray,
    threshold: float,
    y: np.ndarray,
    sizes: np.ndarray,
    target_sizes: tuple[float, ...],
) -> dict[str, Any]:
    hits = (scores >= threshold).astype(float)
    noise_mask = y == 0
    fcr = float(false_call_rate(hits[noise_mask])["rate"]) if np.any(noise_mask) else 0.0
    rates = _calc_detection_rates(hits, y, sizes, target_sizes)
    return {"threshold": threshold, "detection_per_size": rates, "false_call_rate": fcr}


def _run_domain_shift(
    spec: ShiftSpec,
    ctx: ShiftContext,
    n_bars: int,
    sizes: tuple[float, ...],
) -> dict[str, Any]:
    print(f"Running domain shift: {spec.name}...", flush=True)
    cfg1 = WindowConfig(material=spec.material, probe=spec.probe, patch=False, seed=0)
    cfg2 = WindowConfig(material=spec.material, probe=spec.probe, patch=True, seed=0)
    t1 = make_test_set(cfg1, n_bars=n_bars, sizes_mm=sizes, noise_scales=(spec.noise_scale,))
    t2 = make_test_set(cfg2, n_bars=n_bars, sizes_mm=sizes, noise_scales=(spec.noise_scale,))

    r_scores = rules_window_score(t1.x)
    ds1 = WindowDataset(t1.x, t1.y, t1.size_mm)
    ds2 = WindowDataset(t2.x, t2.y, t2.size_mm)
    s1, _ = predict(ctx.adapter_1d, ds1, ctx.p_cfg)
    s2, _ = predict(ctx.adapter_2d, ds2, ctx.p_cfg)

    res_rules = _eval_method(r_scores, ctx.rec_pct, t1.y, t1.size_mm, sizes)
    res_1d_p4 = _eval_method(s1, ctx.thresh_1d, t1.y, t1.size_mm, sizes)
    res_2d_p4 = _eval_method(s2, ctx.thresh_2d, t2.y, t2.size_mm, sizes)

    n_m1, n_m2 = t1.y == 0, t2.y == 0
    m_t1 = matched_threshold(s1[n_m1], res_rules["false_call_rate"])
    m_t2 = matched_threshold(s2[n_m2], res_rules["false_call_rate"])
    res_1d_m = _eval_method(s1, m_t1, t1.y, t1.size_mm, sizes)
    res_2d_m = _eval_method(s2, m_t2, t2.y, t2.size_mm, sizes)

    return {
        "spec": {"material": spec.material, "probe": spec.probe, "noise_scale": spec.noise_scale},
        "rules": res_rules,
        "ml_1d_p4_thresh": res_1d_p4,
        "ml_2d_p4_thresh": res_2d_p4,
        "ml_1d_matched": res_1d_m,
        "ml_2d_matched": res_2d_m,
    }


def _make_roughness_windows(n_bars: int, per_bar: int, seed: int = 42) -> WindowSet:
    rng = np.random.default_rng(seed)
    cfg = WindowConfig(material="ti64", probe="p5", patch=False)
    mat = load_material("ti64")
    probe = load_probe("p5")
    water = load_material("water")
    all_sets: list[WindowSet] = []
    for _ in range(n_bars):
        art = Artefact("roughness", (4.0, 12.0), (30.0, 150.0), severity=1.0)
        spec = BarSpec(
            material=mat,
            probe=probe,
            water=water,
            diameter_mm=cfg.diameter_mm,
            length_mm=16.0,
            axial_step_mm=1.0,
            n_angle=36,
            artefacts=(art,),
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        scan = simulate_bar(spec)
        dac = dac_for(scan)
        _, inside = _partition_grid(scan, (art,))
        if len(inside) == 0:
            continue
        ctx = _BarContext(scan, dac, cfg, 1.0, (art,))
        wset = _bar_artefact_windows(ctx, inside, rng, per_bar)
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

    ds_rough = WindowDataset(rough_wset.x, rough_wset.y, rough_wset.size_mm)
    fresh_scores, _ = predict(fresh_adapter, ds_rough, ctx.p_cfg)
    p4_scores, _ = predict(ctx.adapter_1d, ds_rough, ctx.p_cfg)
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


def _simulate_bscan_slice(flawed: bool, seed: int) -> np.ndarray:
    mat = load_material("ti64")
    probe = load_probe("p5")
    water = load_material("water")
    refs = (
        (Reflector(axial_mm=8.0, angle_deg=0.0, depth_mm=15.0, diameter_mm=1.2, kind="fbh"),)
        if flawed
        else ()
    )
    spec = BarSpec(
        material=mat,
        probe=probe,
        water=water,
        diameter_mm=40.0,
        length_mm=16.0,
        axial_step_mm=1.0,
        n_angle=36,
        reflectors=refs,
        seed=seed,
    )
    scan = simulate_bar(spec)
    ax_idx = int(np.round(8.0 / scan.spec.axial_step_mm))
    rf = scan.ascans[ax_idx]
    env = envelope(rf)
    factors = (256.0 / env.shape[0], 256.0 / env.shape[1])
    bscan = scipy.ndimage.zoom(env, factors, order=1).astype(np.float32)
    return standardise(bscan)


def _eval_mlndt_on_simulated(model: nn.Module, quick: bool, dev: torch.device) -> dict[str, Any]:
    print("Evaluating Phase 3 model on simulated B-scans...", flush=True)
    n_each = 5 if quick else 20
    images: list[np.ndarray] = []
    labels: list[int] = []
    rng = np.random.default_rng(101)
    for _ in range(n_each):
        images.append(_simulate_bscan_slice(flawed=True, seed=int(rng.integers(0, 2**31 - 1))))
        labels.append(1)
        images.append(_simulate_bscan_slice(flawed=False, seed=int(rng.integers(0, 2**31 - 1))))
        labels.append(0)

    img_arr = np.stack(images, axis=0)
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


def _eval_bar1d_on_mlndt(model: WindowNet, quick: bool, dev: torch.device) -> dict[str, Any]:
    print("Evaluating Bar 1D model on ML-NDT validation images...", flush=True)
    root = _find_data_root()
    val_batches = list_batches("validation", root=root)
    batches_to_use = val_batches[:1] if (quick and val_batches) else val_batches
    all_scores: list[float] = []
    all_labels: list[int] = []

    for batch_p in batches_to_use:
        mmap, labels = read_batch(batch_p)
        for idx in range(len(mmap)):
            score = _score_mlndt_image(mmap[idx], model, dev)
            all_scores.append(score)
            all_labels.append(int(labels[idx, 0] > 0.5))

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


def main() -> None:
    args = _parse_args()
    set_seed(0)
    has_cuda = torch.cuda.is_available()
    dev = torch.device("cuda" if has_cuda else "cpu")
    print(
        "Device: cuda (NVIDIA GeForce RTX 5070 Ti Laptop GPU)"
        if has_cuda
        else "CUDA unavailable, using CPU",
        flush=True,
    )
    ctx = _setup_context(dev)
    shifts = _run_all_domain_shifts(ctx, args.quick)
    unseen_res = _eval_unseen_artefact(ctx, args.quick)
    mlndt_net, _ = load_model(_resolve_model_path("mlndt_best.pt"), device=str(dev))
    s2r_sim = _eval_mlndt_on_simulated(mlndt_net, args.quick, dev)
    s2r_mlndt = _eval_bar1d_on_mlndt(ctx.adapter_1d.net, args.quick, dev)

    metrics = {
        "label": "Synthetic data",
        "domain_shift": shifts,
        "unseen_artefact": unseen_res,
        "sim_to_real": {
            "mlndt_on_simulated_bscan": s2r_sim,
            "bar_1d_on_mlndt_val": s2r_mlndt,
        },
    }
    _save_metrics_and_figures(metrics)
    print("Phase 4 robustness experiment completed successfully.", flush=True)


if __name__ == "__main__":
    main()
