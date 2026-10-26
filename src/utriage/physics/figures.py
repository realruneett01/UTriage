"""Plotting utilities for immersion UT scans. Output is Synthetic data."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utriage.physics.simulator import BarScan, cscan_peak, envelope  # noqa: E402


def plot_cscan(
    scan: BarScan,
    out: Path,
    gate_mm: tuple[float, float] | None = None,
) -> Path:
    """Plot peak envelope C-scan with planted reflectors (Synthetic data)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    default_gate = (2.0, max(2.0, scan.spec.diameter_mm - 2.0))
    g_start, g_end = gate_mm if gate_mm is not None else default_gate
    peak = cscan_peak(scan, g_start, g_end)

    fig, ax = plt.subplots(figsize=(8, 5))
    extent = [0.0, 360.0, float(scan.spec.length_mm), 0.0]
    im = ax.imshow(peak, aspect="auto", extent=extent, cmap="viridis")
    fig.colorbar(im, ax=ax, label="Peak Amplitude")

    for r in scan.spec.reflectors:
        ax.plot(r.angle_deg, r.axial_mm, "rx", markersize=8, markeredgewidth=2)

    ax.set_xlabel("Angle (°)")
    ax.set_ylabel("Axial (mm)")
    ax.set_title(f"C-scan — {scan.spec.material.name} (Synthetic data)")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_ascan(scan: BarScan, ia: int, ja: int, out: Path) -> Path:
    """Plot RF A-scan and envelope vs time (Synthetic data)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rf = scan.ascans[ia, ja, :]
    env = envelope(rf)
    t = scan.time_us

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, rf, label="RF", alpha=0.7)
    ax.plot(t, env, "r--", label="Envelope")
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"A-scan ({scan.spec.material.name}, ia={ia}, ja={ja}) (Synthetic data)")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
