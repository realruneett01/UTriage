"""Over/underfitting diagnosis helpers and learning curve analysis (Phase 8)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class FitRule:
    """Diagnostic criteria for evaluating model fit.

    Rule: underfitting if train metric < target;
    overfitting if (train metric - group-held-out metric) > max_gap; else ok.
    """

    target_train_metric: float
    max_gap: float

    @property
    def rule_text(self) -> str:
        """Formatted description of the evaluation rule."""
        return (
            f"underfitting if train metric < {self.target_train_metric}; "
            f"overfitting if (train metric - group-held-out metric) > {self.max_gap}; "
            "else ok."
        )


def learning_curve(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract epoch, train_loss, val_loss trajectories, best_epoch, and final_gap."""
    if not history:
        return {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "best_epoch": 0,
            "final_gap": 0.0,
        }
    epochs = [int(h["epoch"]) for h in history]
    tr_loss = [float(h["train_loss"]) for h in history]
    vl_loss = [float(h["val_loss"]) for h in history]
    best_row = min(history, key=lambda h: float(h["val_loss"]))
    best_ep = int(best_row["epoch"])
    final_gap = float(vl_loss[-1] - tr_loss[-1])
    return {
        "epoch": epochs,
        "train_loss": tr_loss,
        "val_loss": vl_loss,
        "best_epoch": best_ep,
        "final_gap": final_gap,
    }


def data_size_curve(
    fractions: tuple[float, ...],
    run: Callable[[float], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate training at specified data fractions and record metrics."""
    results: list[dict[str, Any]] = []
    for frac in fractions:
        f = float(frac)
        metrics = run(f)
        entry: dict[str, Any] = {"fraction": f}
        entry.update(metrics)
        results.append(entry)
    return results


def _evaluate_set(train_metric: float, heldout_val: float, rule: FitRule) -> dict[str, Any]:
    """Compute gap and verdict for a single evaluation set."""
    gap = float(train_metric - heldout_val)
    if train_metric < rule.target_train_metric:
        verdict = "underfitting"
    elif gap > rule.max_gap:
        verdict = "overfitting"
    else:
        verdict = "ok"
    return {"heldout_metric": float(heldout_val), "gap": gap, "verdict": verdict}


def _worst_case_verdict(verdicts: list[str], default: str) -> str:
    """Determine the worst-case verdict from evaluated sets."""
    if any(v == "underfitting" for v in verdicts):
        return "underfitting"
    if any(v == "overfitting" for v in verdicts):
        return "overfitting"
    return default if not verdicts else "ok"


def diagnose(
    train_metric: float,
    heldout_metrics: dict[str, float],
    rule: FitRule,
) -> dict[str, Any]:
    """Diagnose underfitting or overfitting against target metric and max gap."""
    tr = float(train_metric)
    sets: dict[str, dict[str, Any]] = {
        name: _evaluate_set(tr, val, rule) for name, val in heldout_metrics.items()
    }
    default = "underfitting" if tr < rule.target_train_metric else "ok"
    verdict_list = [s["verdict"] for s in sets.values()]
    overall = _worst_case_verdict(verdict_list, default=default)
    return {
        "train_metric": tr,
        "sets": sets,
        "evaluations": sets,
        "verdict": overall,
        "overall_verdict": overall,
        "rule": rule.rule_text,
    }


def _draw_learning_curve(ax: Any, curve: dict[str, Any], title: str) -> None:
    """Render loss trajectories and best checkpoint marker onto axes."""
    epochs = curve.get("epoch", [])
    ax.plot(epochs, curve.get("train_loss", []), label="Train Loss", color="#1f77b4")
    ax.plot(epochs, curve.get("val_loss", []), label="Val Loss", color="#ff7f0e")
    best_ep = curve.get("best_epoch")
    if best_ep in epochs:
        idx = epochs.index(best_ep)
        val_losses = curve.get("val_loss", [])
        if idx < len(val_losses):
            best_val = val_losses[idx]
            ax.scatter([best_ep], [best_val], color="crimson", zorder=5, label=f"Best ({best_ep})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")


def plot_learning_curve(curve: dict[str, Any], out: Path, title: str) -> Path:
    """Plot training and validation loss curves and save figure to disk."""
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    _draw_learning_curve(ax, curve, title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
