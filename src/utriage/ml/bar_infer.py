"""Phase 6 bar-domain ML inference, MC-dropout uncertainty, and NMS."""

from __future__ import annotations

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


def _expand_node(
    curr: int,
    adj: list[list[int]],
    visited: list[bool],
    queue: list[int],
) -> None:
    for nbr in adj[curr]:
        if not visited[nbr]:
            visited[nbr] = True
            queue.append(nbr)


def _bfs_component(start: int, adj: list[list[int]], visited: list[bool]) -> list[int]:
    comp: list[int] = []
    queue = [start]
    visited[start] = True
    while queue:
        curr = queue.pop()
        comp.append(curr)
        _expand_node(curr, adj, visited, queue)
    return comp


def _connected_groups(adj: list[list[int]]) -> list[list[int]]:
    visited = [False] * len(adj)
    groups: list[list[int]] = []
    for i in range(len(adj)):
        if visited[i]:
            continue
        groups.append(_bfs_component(i, adj, visited))
    return groups


def _pick_best(group: list[int], cands: list[_Candidate]) -> MlFinding:
    best_idx = max(group, key=lambda idx: (cands[idx].score, -idx))
    c = cands[best_idx]
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


def _suppress_candidates(cands: list[_Candidate], n_angle: int) -> list[MlFinding]:
    if not cands:
        return []
    adj = _build_adj(cands, n_angle)
    groups = _connected_groups(adj)
    findings = [_pick_best(grp, cands) for grp in groups]
    findings.sort(key=lambda f: f.score, reverse=True)
    return findings


def scan_bar(
    model: WindowNet,
    scan: BarScan,
    settings: ScanSettings = _DEFAULT_SETTINGS,
    device: str = "auto",
) -> list[MlFinding]:
    """Scan a bar with WindowNet, returning non-maximum suppressed findings."""
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
    "findings_to_dicts",
    "load_window_model",
    "save_window_model",
    "scan_bar",
]
