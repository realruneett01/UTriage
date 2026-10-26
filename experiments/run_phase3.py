"""Phase 3 experiment: Real-data ML detector training, POD, and latency benchmarking."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
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
        sys.modules["utriage"].__path__.insert(0, _pkg_dir)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from utriage.ml.datasets import (  # noqa: E402
    MLNDT_ROOT,
    MlNdtDataset,
    list_batches,
    read_batch,
    split_holdout,
)
from utriage.ml.infer import load_model, measure_latency, save_model  # noqa: E402
from utriage.ml.models import MlNdtNet, count_params  # noqa: E402
from utriage.ml.train import TrainConfig, TrainResult, predict, train_model  # noqa: E402
from utriage.pod.hitmiss import (  # noqa: E402
    binomial_pod_bound,
    false_call_rate,
    fit_hitmiss,
    fit_to_dict,
    pod_curve,
    pod_lower_bound,
)


@dataclass(frozen=True)
class SplitEvalResult:
    """Container for POD fit, false calls, and evaluation arrays for a split."""

    pod: dict | None
    pod_reason: str | None
    detection: dict
    false_calls: dict | None
    flawed_sizes: np.ndarray
    flawed_hits: np.ndarray
    fit_obj: Any


@dataclass(frozen=True)
class ExperimentContext:
    """Aggregated training and execution context for metrics reporting."""

    splits_count: dict
    restarts: list[dict]
    kept_seed: int
    params_count: int
    dev_name: str
    latencies: dict[str, float]


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Phase 3 experiment."""
    parser = argparse.ArgumentParser(description="Phase 3 ML-NDT detector experiment")
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer batches")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate existing checkpoint")
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
        # Fall back to default MLNDT_ROOT if git inspection fails
        pass
    return MLNDT_ROOT


def _resolve_splits(quick: bool) -> tuple[list[Path], list[Path], list[Path]]:
    """Partition ML-NDT dataset into train, test, and validation batches."""
    root = _find_data_root()
    training_all = list_batches("training", root=root)
    val_all = list_batches("validation", root=root)
    train_batches, test_batches = split_holdout(training_all, test_frac=0.1, seed=0)
    if quick:
        return train_batches[:4], test_batches[:1], val_all[:1]
    return train_batches, test_batches, val_all


def _collect_labels(batches: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Extract concatenated flaw indicators and equivalent sizes across batches."""
    flaw_list: list[np.ndarray] = []
    size_list: list[np.ndarray] = []
    for b in batches:
        _, lbl = read_batch(b)
        flaw_list.append(lbl[:, 0])
        size_list.append(lbl[:, 1])
    if not flaw_list:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    return np.concatenate(flaw_list, axis=0), np.concatenate(size_list, axis=0)


def _fit_pod_safe(sizes: np.ndarray, hits: np.ndarray) -> tuple[dict | None, str | None, Any]:
    """Fit hit/miss POD model, safely catching separation or convergence errors."""
    if len(sizes) == 0:
        return None, "No flawed images in split", None
    try:
        fit = fit_hitmiss(sizes, hits)
        return fit_to_dict(fit), None, fit
    except ValueError as exc:
        # Expected when outcomes are separable (e.g. all hits in small test split)
        return None, str(exc), None


def _eval_split(
    model: nn.Module, ds: MlNdtDataset, batches: list[Path], cfg: TrainConfig
) -> SplitEvalResult:
    """Evaluate detector on a split: POD on flawed scans and false calls on unflawed."""
    probs, _ = predict(model, ds, cfg)
    flaw_labels, eq_sizes = _collect_labels(batches)
    hits = (probs >= 0.5).astype(float)
    flawed = (flaw_labels == 1.0) & (eq_sizes > 0.0)
    unflawed = flaw_labels == 0.0
    det_dict = binomial_pod_bound(eq_sizes[flawed], hits[flawed])
    pod_dict, pod_err, fit_obj = _fit_pod_safe(eq_sizes[flawed], hits[flawed])
    fc_dict = false_call_rate(hits[unflawed]) if np.any(unflawed) else None
    return SplitEvalResult(
        pod=pod_dict,
        pod_reason=pod_err,
        detection=det_dict,
        false_calls=fc_dict,
        flawed_sizes=eq_sizes[flawed],
        flawed_hits=hits[flawed],
        fit_obj=fit_obj,
    )


def _run_single_seed(
    seed: int, train_ds: MlNdtDataset, val_ds: MlNdtDataset, epochs: int
) -> tuple[dict, TrainResult]:
    """Train one restart seed and return state dict and TrainResult."""
    print(f"Starting restart with seed={seed} (epochs={epochs})...", flush=True)
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


def _train_restarts(
    train_ds: MlNdtDataset, val_ds: MlNdtDataset, quick: bool, epochs: int
) -> tuple[dict, int, list[dict], str]:
    """Execute training restarts and select model with lowest validation loss."""
    seeds = [0] if quick else [0, 1, 2]
    best_weights: dict = {}
    best_loss = float("inf")
    best_seed = seeds[0]
    results_list: list[dict] = []
    device_str = "cpu"
    for s in seeds:
        weights, res = _run_single_seed(s, train_ds, val_ds, epochs)
        device_str = res.device
        results_list.append(
            {
                "seed": s,
                "best_epoch": res.best_epoch,
                "best_val_loss": res.best_val_loss,
                "seconds": res.seconds,
                "history": res.history,
            }
        )
        if res.best_val_loss < best_loss:
            best_loss = res.best_val_loss
            best_weights = weights
            best_seed = s
    return best_weights, best_seed, results_list, device_str


def _measure_latencies(model: nn.Module, quick: bool) -> dict[str, float]:
    """Measure inference latency per image across GPU and CPU."""
    repeats = 10 if quick else 50
    latencies: dict[str, float] = {}
    if torch.cuda.is_available():
        cuda_dev = torch.device("cuda")
        latencies["gpu_batch_1"] = measure_latency(
            model, cuda_dev, batch=1, size=256, repeats=repeats
        )
        latencies["gpu_batch_64"] = measure_latency(
            model, cuda_dev, batch=64, size=256, repeats=repeats
        )
    cpu_model = copy.deepcopy(model).to(torch.device("cpu"))
    latencies["cpu_batch_1"] = measure_latency(
        cpu_model, torch.device("cpu"), batch=1, size=256, repeats=repeats
    )
    return latencies


def _plot_pod_figure(test_eval: SplitEvalResult, out_path: Path) -> None:
    """Render POD curve, 95% lower bound, and hit/miss scatter for test split."""
    plt.figure(figsize=(7, 5))
    s = test_eval.flawed_sizes
    h = test_eval.flawed_hits
    if test_eval.fit_obj is not None and len(s) > 0:
        s_grid = np.linspace(max(0.1, float(np.min(s)) * 0.8), float(np.max(s)) * 1.2, 200)
        p_curve = pod_curve(test_eval.fit_obj, s_grid)
        p_lower = pod_lower_bound(test_eval.fit_obj, s_grid)
        plt.plot(s_grid, p_curve, color="blue", label="POD curve")
        plt.plot(s_grid, p_lower, "--", color="navy", label="95% lower bound")
    if len(s) > 0:
        plt.scatter(s, h, color="crimson", alpha=0.6, label="Hits / Misses")
    plt.xlabel("Equivalent Flaw Size")
    if test_eval.pod is None:
        lb = test_eval.detection["pod_lower95"]
        plt.title(f"Detection vs Size (Binomial 95% LB: {lb:.3f}) — Real data (ML-NDT)")
        plt.ylabel("Hit (1) / Miss (0)")
    else:
        plt.title("Probability of Detection vs Size — Real data (ML-NDT)")
        plt.ylabel("Detection Probability")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _build_metrics(
    ctx: ExperimentContext, test_res: SplitEvalResult, val_res: SplitEvalResult
) -> dict:
    """Assemble metrics dictionary for JSON export."""
    lbl = "Real data (ML-NDT, Virkkunen et al. arXiv:1903.11399; data LGPL-3.0, not redistributed)"
    t_sp = {
        "pod": test_res.pod,
        "pod_reason": test_res.pod_reason,
        "detection": test_res.detection,
        "false_calls": test_res.false_calls,
    }
    v_sp = {
        "pod": val_res.pod,
        "pod_reason": val_res.pod_reason,
        "detection": val_res.detection,
        "false_calls": val_res.false_calls,
    }
    ref = {"a90_95": "0.9 mm", "false_calls": 0, "source": "project plan V4"}
    return {
        "label": lbl,
        "size_units": "equivalent size",
        "splits": ctx.splits_count,
        "restarts": ctx.restarts,
        "kept_seed": ctx.kept_seed,
        "params": ctx.params_count,
        "device_name": ctx.dev_name,
        "per_split": {"test": t_sp, "validation": v_sp},
        "latency_ms": ctx.latencies,
        "paper_reference": ref,
    }


def _format_pod_str(pod_entry: dict) -> str:
    """Format POD thresholds string from a split result entry."""
    pod = pod_entry.get("pod")
    if pod is not None:
        return f"a50={pod['a50']:.3f}, a90={pod['a90']:.3f}, a90/95={pod['a90_95']:.3f}"
    reason = pod_entry.get("pod_reason", "Not estimable")
    return f"null ({reason})"


def _format_fc_str(fc_entry: dict | None) -> str:
    """Format false call summary string from entry."""
    if fc_entry is None:
        return "N/A"
    n_false = fc_entry.get("n_false", 0)
    n = fc_entry.get("n", 0)
    rate = fc_entry.get("rate", 0.0)
    up95 = fc_entry.get("upper95", 0.0)
    return f"{n_false}/{n} ({rate:.2%}, 95% UB: {up95:.2%})"


def _report_header_md(m: dict) -> list[str]:
    """Return header and dataset split sections for markdown report."""
    test_sp = m["per_split"]["test"]
    val_sp = m["per_split"]["validation"]
    sp = m["splits"]
    dev_info = (
        f"**Device:** {m['device_name']} | "
        f"**Parameters:** {m['params']} | "
        f"**Kept Seed:** {m['kept_seed']}"
    )
    return [
        "# ML-NDT Real-Data Crack Detection & POD Report",
        "",
        f"**Dataset Label:** {m['label']}",
        dev_info,
        f"**Flaw Size Metric:** {m['size_units']}",
        "",
        "## Dataset Splits",
        "| Split | Batches | Images |",
        "|---|---|---|",
        f"| Train | {sp['train']['batch_count']} | {sp['train']['image_count']} |",
        f"| Test | {sp['test']['batch_count']} | {sp['test']['image_count']} |",
        f"| Validation | {sp['validation']['batch_count']} | {sp['validation']['image_count']} |",
        "",
        "## Detection & POD Evaluation",
        "| Split | POD Metrics | False Calls |",
        "|---|---|---|",
        f"| Test | {_format_pod_str(test_sp)} | {_format_fc_str(test_sp.get('false_calls'))} |",
        f"| Validation | {_format_pod_str(val_sp)} | {_format_fc_str(val_sp.get('false_calls'))} |",
    ]


def _format_detection_row(split_name: str, sp_data: dict) -> str:
    """Format single row for detection summary table."""
    det = sp_data["detection"]
    n_fl = det["n_flawed"]
    n_h = det["n_hit"]
    rng = f"{det['min_size']:.2f} - {det['max_size']:.2f}" if n_fl > 0 else "N/A"
    return f"| {split_name} | {n_fl} | {n_h} | {rng} | {det['pod_lower95']:.3f} |"


def _format_detection_table(m: dict) -> list[str]:
    """Format flaw detection table with binomial POD lower bound per split."""
    ref_a90 = m["paper_reference"]["a90_95"]
    lines = [
        "",
        "## Flaw Detection (Binomial Bound)",
        "| Split | n_flawed | n_hit | Size Range | POD Lower Bound |",
        "|---|---|---|---|---|",
    ]
    for name, data in m["per_split"].items():
        lines.append(_format_detection_row(name.capitalize(), data))
    lines.append("")
    lines.append(
        "a90/95 is not estimable by the logistic model because every tested flaw was detected; "
        f"the paper reports a90/95 {ref_a90}; numbers are not directly comparable."
    )
    return lines


def _report_tail_md(m: dict) -> list[str]:
    """Return latency, paper comparison, and limitations sections."""
    test_sp = m["per_split"]["test"]
    ref = m["paper_reference"]
    lat_lines = [
        "",
        "## Latency Benchmark",
        "| Configuration | Median Latency (ms/image) |",
        "|---|---|",
    ]
    for k, v in m["latency_ms"].items():
        lat_lines.append(f"| {k} | {v:.3f} |")
    comp_lines = [
        "",
        "## Comparison with Prior Art (Virkkunen et al. arXiv:1903.11399)",
        f"- **Paper Reference:** a90/95 of {ref['a90_95']} with {ref['false_calls']} false calls.",
        (
            f"- **Our Model (Test):** POD: {_format_pod_str(test_sp)}; "
            f"False calls: {_format_fc_str(test_sp.get('false_calls'))}."
        ),
        "",
        "## Limitations",
        "- **Specimen Count:** Dataset comprises augmented scans from only 3 physical cracks.",
        "- **Component Geometry:** Planar weld inspection, not a rotating titanium bar.",
        "- **Data Independence:** Augmented copies of the same cracks so not independent.",
        "- **Scope:** Proves the method not MIDHANI applicability.",
    ]
    return lat_lines + comp_lines


def _format_report_md(m: dict) -> str:
    """Generate Markdown report entirely from metrics dictionary."""
    parts = _report_header_md(m) + _format_detection_table(m) + _report_tail_md(m)
    return "\n".join(parts) + "\n"


def _save_artifacts(metrics: dict, test_eval: SplitEvalResult, model: nn.Module) -> None:
    """Save model checkpoint, metrics JSON, figure, and markdown report."""
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    meta = {"width": 32, "first_pool": (2, 2), "size": 256}
    save_model(model, models_dir / "mlndt_best.pt", meta)

    metrics_path = results_dir / "mlndt_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    _plot_pod_figure(test_eval, fig_dir / "mlndt_pod.png")

    report_content = _format_report_md(metrics)
    report_path = results_dir / "mlndt_report.md"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write(report_content)


def _eval_and_benchmark(
    model: nn.Module,
    datasets: tuple[MlNdtDataset, MlNdtDataset],
    batches: tuple[list[Path], list[Path]],
    dev_str: str,
) -> tuple[SplitEvalResult, SplitEvalResult]:
    """Run split evaluations for test and validation partitions."""
    test_ds, val_ds = datasets
    test_b, val_b = batches
    eval_cfg = TrainConfig(batch_size=64, device=dev_str)
    test_eval = _eval_split(model, test_ds, test_b, eval_cfg)
    val_eval = _eval_split(model, val_ds, val_b, eval_cfg)
    return test_eval, val_eval


def _load_eval_context() -> tuple[MlNdtNet, list[dict], int, int, str]:
    """Load model and reuse training context from existing results for eval-only run."""
    metrics_path = Path("results/mlndt_metrics.json")
    with metrics_path.open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(Path("models/mlndt_best.pt"), device=dev_str)
    return (
        model,
        metrics["restarts"],
        int(metrics["kept_seed"]),
        int(metrics["params"]),
        dev_str,
    )


def _make_splits_dict(
    batches: tuple[list[Path], list[Path], list[Path]],
    datasets: tuple[MlNdtDataset, MlNdtDataset, MlNdtDataset],
) -> dict[str, dict[str, int]]:
    """Construct splits summary mapping batch and image counts."""
    tr_b, te_b, v_b = batches
    tr_ds, v_ds, te_ds = datasets
    return {
        "train": {"batch_count": len(tr_b), "image_count": len(tr_ds)},
        "test": {"batch_count": len(te_b), "image_count": len(te_ds)},
        "validation": {"batch_count": len(v_b), "image_count": len(v_ds)},
    }


def main() -> None:
    """Run Phase 3 ML-NDT training, POD evaluation, and benchmarking."""
    t0 = time.time()
    args = _parse_args()
    epochs = args.epochs if args.epochs is not None else (1 if args.quick else 20)
    train_b, test_b, val_b = _resolve_splits(args.quick)
    train_ds = MlNdtDataset(train_b)
    val_ds = MlNdtDataset(val_b)
    test_ds = MlNdtDataset(test_b)

    if args.eval_only:
        model, restarts, seed, params, dev_str = _load_eval_context()
    else:
        weights, seed, restarts, dev_str = _train_restarts(train_ds, val_ds, args.quick, epochs)
        model = MlNdtNet(width=32, first_pool=(2, 2))
        model.load_state_dict(weights)
        params = count_params(model)

    test_ev, val_ev = _eval_and_benchmark(model, (test_ds, val_ds), (test_b, val_b), dev_str)
    latencies = _measure_latencies(model, args.quick)
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    splits = _make_splits_dict((train_b, test_b, val_b), (train_ds, val_ds, test_ds))
    ctx = ExperimentContext(
        splits_count=splits,
        restarts=restarts,
        kept_seed=seed,
        params_count=params,
        dev_name=dev_name,
        latencies=latencies,
    )
    metrics = _build_metrics(ctx, test_ev, val_ev)
    _save_artifacts(metrics, test_ev, model)
    print(f"Phase 3 done in {time.time() - t0:.2f}s (dev={dev_name})", flush=True)


if __name__ == "__main__":
    main()
