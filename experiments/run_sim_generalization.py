"""Held-out parameter-region generalization experiment (project plan Phase 8).

Synthetic data.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure local workspace src is prioritized
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from utriage.ml.bar_data import WindowSet  # noqa: E402
from utriage.ml.bar_models import TrainerAdapter, WindowDataset, WindowNet  # noqa: E402
from utriage.ml.compare import matched_threshold, rules_window_score  # noqa: E402
from utriage.ml.infer import save_model  # noqa: E402
from utriage.ml.sim_windows import (  # noqa: E402
    SimGenSettings,
    build_artefact_stress,
    build_in_distribution_test,
    build_region_test,
    build_train,
)
from utriage.ml.train import TrainConfig, pick_device, predict, set_seed, train_model  # noqa: E402
from utriage.physics.randomize import DEFAULT_HOLDOUT_REGIONS  # noqa: E402
from utriage.pod.metrics import cell  # noqa: E402
from utriage.rules.config import load_acceptance  # noqa: E402


@dataclass(frozen=True)
class EvalContext:
    """Evaluation parameters and threshold configuration."""

    rec_pct: float
    r_fcr: float
    ml_thresh: float
    device: torch.device


@dataclass(frozen=True)
class EvalRunInput:
    """Input datasets for generalization evaluation."""

    train_ws: WindowSet
    in_dist_ws: WindowSet
    region_sets: dict[str, WindowSet]
    artefact_sets: dict[str, WindowSet]


@dataclass(frozen=True)
class EvalRunOutput:
    """Output configuration for evaluation run."""

    out_json_path: Path
    model_save_path: Path | None = None
    device: torch.device | None = None


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Simulator generalization experiment")
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer bars/epochs")
    return parser.parse_args()


def _make_settings(quick: bool) -> SimGenSettings:
    """Construct experiment settings based on quick flag."""
    if quick:
        return SimGenSettings(n_train_bars=4, n_test_bars=2, seed=42, epochs=1)
    return SimGenSettings(n_train_bars=40, n_test_bars=10, seed=42, epochs=15)


def _train_simgen_model(
    train_ws: WindowSet,
    settings: SimGenSettings,
    device: torch.device,
) -> TrainerAdapter:
    """Train WindowNet 1D on generated training windows."""
    split = int(0.9 * len(train_ws.x))
    t_ds = WindowDataset(train_ws.x[:split], train_ws.y[:split], train_ws.size_mm[:split])
    v_ds = WindowDataset(train_ws.x[split:], train_ws.y[split:], train_ws.size_mm[split:])

    net = WindowNet(in_channels=1, width=32)
    adapter = TrainerAdapter(net)
    cfg = TrainConfig(
        epochs=settings.epochs,
        batch_size=256,
        patience=3,
        seed=settings.seed,
        device=str(device),
    )
    print(f"Training WindowNet 1D on device: {device}", flush=True)
    train_model(adapter, t_ds, v_ds, cfg)
    return adapter


def _compute_thresholds(
    adapter: nn.Module,
    train_ws: WindowSet,
    device: torch.device,
) -> EvalContext:
    """Derive rules threshold, training noise false-call rate, and matched ML threshold."""
    noise_mask = train_ws.y == 0
    rec_pct = float(load_acceptance().amplitude.record_pct)
    r_scores = rules_window_score(train_ws.x[noise_mask])
    noise_hits = (r_scores >= rec_pct).astype(float)
    r_fcr = float(np.mean(noise_hits)) if len(noise_hits) > 0 else 0.0

    noise_ds = WindowDataset(
        train_ws.x[noise_mask],
        train_ws.y[noise_mask],
        train_ws.size_mm[noise_mask],
    )
    p_cfg = TrainConfig(batch_size=256, device=str(device))
    ml_noise_probs, _ = predict(adapter, noise_ds, p_cfg)
    ml_thresh = matched_threshold(ml_noise_probs, r_fcr)
    return EvalContext(rec_pct=rec_pct, r_fcr=r_fcr, ml_thresh=ml_thresh, device=device)


def _build_cell(
    hits_all: np.ndarray,
    wset: WindowSet,
    probs_all: np.ndarray,
) -> dict[str, Any]:
    """Compute pod.metrics cell dictionary for one evaluation method."""
    flaw_mask = wset.y == 1
    non_flaw_mask = wset.y != 1
    hits = hits_all[flaw_mask]
    sizes = wset.size_mm[flaw_mask]
    probs = probs_all.astype(float)
    labels = flaw_mask.astype(int)

    n_false = int(np.sum(hits_all[non_flaw_mask]))
    n_scans = int(np.sum(non_flaw_mask))
    n_physical_flaws = int(np.sum(flaw_mask))

    return cell(
        hits=hits,
        sizes=sizes,
        probs=probs,
        labels=labels,
        n_false=n_false,
        n_scans=n_scans,
        n_physical_flaws=n_physical_flaws,
    )


def _eval_wset_methods(
    adapter: nn.Module,
    test_ws: WindowSet,
    ctx: EvalContext,
) -> dict[str, Any]:
    """Evaluate rules, ML 1D, and union on a given window set."""
    r_scores = rules_window_score(test_ws.x)
    rules_hits = (r_scores >= ctx.rec_pct).astype(int)

    ds = WindowDataset(test_ws.x, test_ws.y, test_ws.size_mm)
    p_cfg = TrainConfig(batch_size=256, device=str(ctx.device))
    ml_probs, _ = predict(adapter, ds, p_cfg)
    ml_hits = (ml_probs >= ctx.ml_thresh).astype(int)
    union_hits = (rules_hits | ml_hits).astype(int)

    return {
        "rules": _build_cell(rules_hits, test_ws, rules_hits.astype(float)),
        "ml_1d": _build_cell(ml_hits, test_ws, ml_probs.astype(float)),
        "union": _build_cell(union_hits, test_ws, union_hits.astype(float)),
    }


def _collect_all_evaluations(
    adapter: nn.Module,
    all_sets: dict[str, WindowSet],
    ctx: EvalContext,
) -> dict[str, dict[str, Any]]:
    """Run method evaluations across all test sets."""
    return {name: _eval_wset_methods(adapter, wset, ctx) for name, wset in all_sets.items()}


def _build_payload(
    eval_results: dict[str, dict[str, Any]],
    ctx: EvalContext,
) -> dict[str, Any]:
    """Assemble final JSON payload (Synthetic data)."""
    protocol = (
        "Held-out parameter-region generalization protocol (project plan Phase 8). "
        "WindowNet 1D trained on in-distribution bars sampled with rejection of all held-out "
        "regions. Evaluated at matched false-call rate against immersion rules baseline and union."
    )
    regions_def = [
        {"name": r.name, "param": r.param, "low": r.low, "high": r.high}
        for r in DEFAULT_HOLDOUT_REGIONS
    ]
    payload: dict[str, Any] = {
        "label": "Synthetic data",
        "protocol": protocol,
        "regions": regions_def,
        "thresholds": {
            "rules_record_pct": ctx.rec_pct,
            "rules_noise_fcr": ctx.r_fcr,
            "ml_matched_threshold": ctx.ml_thresh,
        },
        "test_sets": eval_results,
    }
    for key, val in eval_results.items():
        payload[key] = val
    return payload


def _save_model_if_requested(adapter: nn.Module, path: Path | None) -> None:
    """Save model checkpoint if output path is provided."""
    if path is not None:
        model_to_save = adapter.net if hasattr(adapter, "net") else adapter
        save_model(model_to_save, path, {"in_channels": 1, "width": 32})


def evaluate_and_write_json(
    adapter: nn.Module,
    inp: EvalRunInput,
    out: EvalRunOutput,
) -> dict[str, Any]:
    """Evaluate methods, optionally save model, and serialize metrics to JSON."""
    dev = out.device if out.device is not None else torch.device("cpu")
    ctx = _compute_thresholds(adapter, inp.train_ws, dev)
    _save_model_if_requested(adapter, out.model_save_path)

    all_sets: dict[str, WindowSet] = {"in_distribution": inp.in_dist_ws}
    all_sets.update(inp.region_sets)
    for k, ws in inp.artefact_sets.items():
        all_sets[f"artefact_{k}"] = ws

    eval_results = _collect_all_evaluations(adapter, all_sets, ctx)
    payload = _build_payload(eval_results, ctx)

    out_p = Path(out.out_json_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    return payload


def _run_experiment(settings: SimGenSettings, dev: torch.device) -> None:
    """Execute complete training and evaluation flow."""
    print("Generating in-distribution training windows...", flush=True)
    train_ws = build_train(settings)
    adapter = _train_simgen_model(train_ws, settings, dev)

    print("Generating in-distribution and held-out test windows...", flush=True)
    inp = EvalRunInput(
        train_ws=train_ws,
        in_dist_ws=build_in_distribution_test(settings),
        region_sets={r.name: build_region_test(settings, r) for r in DEFAULT_HOLDOUT_REGIONS},
        artefact_sets=build_artefact_stress(settings),
    )
    out = EvalRunOutput(
        out_json_path=Path("results/sim_generalization.json"),
        model_save_path=Path("models/simgen_1d.pt"),
        device=dev,
    )
    evaluate_and_write_json(adapter, inp, out)
    print(f"Results written to {out.out_json_path}", flush=True)
    print(f"Model saved to {out.model_save_path}", flush=True)


def main() -> None:
    """Run simulator generalization pipeline."""
    args = _parse_args()
    settings = _make_settings(args.quick)
    set_seed(settings.seed)

    dev = pick_device("auto")
    print(f"Executing generalization pipeline on device: {dev}", flush=True)
    _run_experiment(settings, dev)


if __name__ == "__main__":
    main()
