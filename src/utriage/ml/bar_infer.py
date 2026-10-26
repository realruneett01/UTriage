"""Phase 6 bar-domain ML inference, MC-dropout uncertainty, and NMS."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from utriage.ml.bar_data import cut_windows
from utriage.ml.bar_models import WindowNet, flaw_logit
from utriage.ml.infer import save_model
from utriage.ml.train import pick_device
from utriage.physics.simulator import BarScan
from utriage.rules.pipeline import dac_for
from utriage.triage.fusion import MlFinding
from utriage.triage.uncertainty import ensemble_stats

_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass(frozen=True)
class ScanSettings:
    """Settings for scanning a bar and extracting ML findings."""

    depth_step_mm: float = 0.5
    front_mm: float = 2.0
    back_margin_mm: float = 2.0
    threshold: float = 0.5
    batch: int = 2048
    mc_passes: int = 8


_DEFAULT_SETTINGS = ScanSettings()


@dataclass(frozen=True)
class _Candidate:
    axial_idx: int
    angle_idx: int
    depth_mm: float
    score: float
    uncertainty: float
    std: float


@dataclass(frozen=True)
class _InferStats:
    score: np.ndarray
    uncertainty: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class _GraphContext:
    cands: list[_Candidate]
    n_angle: int
    grid: dict[tuple[int, int], list[int]]
    adj: list[list[int]]


def save_window_model(model: WindowNet, path: Path | str, meta: dict) -> Path:
    """Save WindowNet state dict and metadata to disk."""
    return save_model(model, Path(path), meta)


def load_window_model(path: Path | str, device: str = "auto") -> tuple[WindowNet, dict]:
    """Load WindowNet model and metadata from checkpoint."""
    ckpt_path = Path(path)
    dev = pick_device(device)
    ckpt = torch.load(ckpt_path, map_location=dev)
    meta = dict(ckpt.get("meta", {}))
    in_channels = int(meta.get("in_channels", 1))
    width = int(meta.get("width", 32))
    dropout = float(meta.get("dropout", 0.1))
    n_classes = int(meta.get("n_classes", 3))
    model = WindowNet(
        in_channels=in_channels,
        width=width,
        n_classes=n_classes,
        dropout=dropout,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev)
    model.eval()
    return model, meta


def _make_grid_sites(scan: BarScan, settings: ScanSettings) -> np.ndarray:
    n_ax, n_an = scan.ascans.shape[0], scan.ascans.shape[1]
    d_min = settings.front_mm
    d_max = scan.spec.diameter_mm - settings.back_margin_mm
    if d_max < d_min or n_ax == 0 or n_an == 0:
        return np.zeros((0, 3), dtype=np.float64)
    depths = np.arange(d_min, d_max + 0.5 * settings.depth_step_mm, settings.depth_step_mm)
    depths = depths[depths <= d_max + 1e-6]
    if len(depths) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    ax_grid, an_grid, d_grid = np.meshgrid(
        np.arange(n_ax, dtype=np.float64),
        np.arange(n_an, dtype=np.float64),
        depths,
        indexing="ij",
    )
    return np.column_stack([ax_grid.ravel(), an_grid.ravel(), d_grid.ravel()])


def _activate_mc_dropout(model: nn.Module) -> None:
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
            m.train()


@torch.no_grad()
def _eval_batches(
    model: nn.Module,
    windows: np.ndarray,
    settings: ScanSettings,
    dev: torch.device,
) -> _InferStats:
    n_samples = len(windows)
    passes = max(1, settings.mc_passes)
    all_probs = np.empty((passes, n_samples), dtype=np.float32)
    _activate_mc_dropout(model)
    model.to(dev)
    for start in range(0, n_samples, settings.batch):
        end = min(start + settings.batch, n_samples)
        batch = torch.from_numpy(windows[start:end]).to(dev)
        for p_idx in range(passes):
            logits, _ = model(batch)
            lg = flaw_logit(logits)
            all_probs[p_idx, start:end] = torch.sigmoid(lg).cpu().numpy()
    model.eval()
    mean, std = ensemble_stats(all_probs)
    unc = np.minimum(1.0, 2.0 * std)
    return _InferStats(
        score=mean.astype(np.float32),
        uncertainty=unc.astype(np.float32),
        std=std.astype(np.float32),
    )


def _build_candidates(
    sites: np.ndarray,
    stats: _InferStats,
    threshold: float,
) -> list[_Candidate]:
    indices = np.where(stats.score >= threshold)[0]
    cands: list[_Candidate] = []
    for idx in indices:
        cands.append(
            _Candidate(
                axial_idx=int(sites[idx, 0]),
                angle_idx=int(sites[idx, 1]),
                depth_mm=float(sites[idx, 2]),
                score=float(stats.score[idx]),
                uncertainty=float(stats.uncertainty[idx]),
                std=float(stats.std[idx]),
            )
        )
    return cands


def _connect_cell(
    i: int,
    c: _Candidate,
    cell: tuple[int, int],
    ctx: _GraphContext,
) -> None:
    for j in ctx.grid.get(cell, ()):
        if j > i and abs(c.depth_mm - ctx.cands[j].depth_mm) <= 1.5 + 1e-6:
            ctx.adj[i].append(j)
            ctx.adj[j].append(i)


def _connect_candidate(i: int, c: _Candidate, ctx: _GraphContext) -> None:
    for dx, dy in _OFFSETS:
        cell = (c.axial_idx + dx, (c.angle_idx + dy) % ctx.n_angle)
        _connect_cell(i, c, cell, ctx)


def _build_adj(cands: list[_Candidate], n_angle: int) -> list[list[int]]:
    grid: dict[tuple[int, int], list[int]] = {}
    for i, c in enumerate(cands):
        grid.setdefault((c.axial_idx, c.angle_idx), []).append(i)
    adj: list[list[int]] = [[] for _ in range(len(cands))]
    ctx = _GraphContext(cands=cands, n_angle=n_angle, grid=grid, adj=adj)
    for i, c in enumerate(cands):
        _connect_candidate(i, c, ctx)
    return adj


def _greedy_nms(cands: list[_Candidate], adj: list[list[int]]) -> list[_Candidate]:
    order = sorted(range(len(cands)), key=lambda idx: cands[idx].score, reverse=True)
    suppressed = [False] * len(cands)
    kept: list[_Candidate] = []
    for idx in order:
        if suppressed[idx]:
            continue
        kept.append(cands[idx])
        for nbr in adj[idx]:
            suppressed[nbr] = True
    return kept


def _suppress_candidates(cands: list[_Candidate], n_angle: int) -> list[_Candidate]:
    if not cands:
        return []
    adj = _build_adj(cands, n_angle)
    return _greedy_nms(cands, adj)


def _candidate_to_finding(c: _Candidate) -> MlFinding:
    return MlFinding(
        axial_idx=c.axial_idx,
        angle_idx=c.angle_idx,
        depth_mm=c.depth_mm,
        score=c.score,
        uncertainty=c.uncertainty,
        grain_noise_prob=float(1.0 - c.score),
        evidence=(
            f"ML window score {c.score:.2f}",
            f"MC-dropout std {c.std:.2f}",
        ),
    )


def _scan_candidates(
    model: WindowNet,
    scan: BarScan,
    settings: ScanSettings,
    device: str,
) -> list[_Candidate]:
    sites = _make_grid_sites(scan, settings)
    if len(sites) == 0:
        return []
    dac = dac_for(scan)
    is_patch = getattr(model.stem[0], "in_channels", 1) == 9
    windows = cut_windows(scan, dac, sites, patch=is_patch)
    dev = pick_device(device)
    stats = _eval_batches(model, windows, settings, dev)
    cands = _build_candidates(sites, stats, settings.threshold)
    return _suppress_candidates(cands, scan.spec.n_angle)


def scan_bar(
    model: WindowNet,
    scan: BarScan,
    settings: ScanSettings = _DEFAULT_SETTINGS,
    device: str = "auto",
) -> list[MlFinding]:
    """Scan a bar with WindowNet, returning non-maximum suppressed findings."""
    kept = _scan_candidates(model, scan, settings, device)
    return [_candidate_to_finding(c) for c in kept]


def candidate_scores(
    model: WindowNet,
    scan: BarScan,
    settings: ScanSettings = _DEFAULT_SETTINGS,
    device: str = "auto",
) -> np.ndarray:
    """Return the scores of all NMS-surviving local maxima with threshold 0."""
    zero_settings = dataclasses.replace(settings, threshold=0.0)
    kept = _scan_candidates(model, scan, zero_settings, device)
    if not kept:
        return np.zeros(0, dtype=np.float64)
    return np.array([c.score for c in kept], dtype=np.float64)


def _collect_clean_scores(
    model: WindowNet,
    scans: list[BarScan],
    settings: ScanSettings,
) -> list[float]:
    all_scores: list[float] = []
    for scan in scans:
        sc = candidate_scores(model, scan, settings)
        all_scores.extend(float(x) for x in sc)
    return all_scores


def calibrate_bar_threshold(
    model: WindowNet,
    clean_scans: list[BarScan],
    max_flags_per_bar: float = 1.0,
    settings: ScanSettings = _DEFAULT_SETTINGS,
) -> float:
    """Calibrate the smallest threshold giving <= max_flags_per_bar on clean bars."""
    if not clean_scans:
        return 1.0
    all_scores = _collect_clean_scores(model, clean_scans, settings)
    if not all_scores:
        return 0.0
    max_total = max_flags_per_bar * len(clean_scans)
    arr = np.sort(np.asarray(all_scores, dtype=np.float64))
    if len(arr) <= max_total:
        return 0.0
    for val in np.unique(arr):
        if int(np.sum(arr >= val)) <= max_total:
            return float(val)
    return 1.0


def finding_to_dict(finding: MlFinding) -> dict:
    """Convert a single MlFinding into a JSON-safe dictionary."""
    return {
        "axial_idx": int(finding.axial_idx),
        "angle_idx": int(finding.angle_idx),
        "depth_mm": float(finding.depth_mm),
        "score": float(finding.score),
        "uncertainty": float(finding.uncertainty),
        "grain_noise_prob": (
            float(finding.grain_noise_prob) if finding.grain_noise_prob is not None else None
        ),
        "evidence": list(finding.evidence),
    }


def findings_to_dicts(findings: Sequence[MlFinding]) -> list[dict]:
    """Convert findings to a list of JSON-safe dictionaries."""
    return [finding_to_dict(f) for f in findings]


__all__ = [
    "ScanSettings",
    "calibrate_bar_threshold",
    "candidate_scores",
    "findings_to_dicts",
    "load_window_model",
    "save_window_model",
    "scan_bar",
]
