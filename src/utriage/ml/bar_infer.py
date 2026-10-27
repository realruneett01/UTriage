"""Phase 6 bar-domain ML inference, MC-dropout uncertainty, and NMS."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn

from utriage.ml.bar_data import WINDOW, cut_windows
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
    threshold: float = 0.0
    batch: int = 2048
    mc_passes: int = 8


_DEFAULT_SETTINGS = ScanSettings()


@dataclass(frozen=True)
class _Candidate:
    axial_idx: int
    angle_idx: int
    depth_mm: float
    score: float
    logit: float
    uncertainty: float
    std: float


@dataclass(frozen=True)
class _InferStats:
    score: np.ndarray
    logit: np.ndarray
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


def site_depths(scan: BarScan, settings: ScanSettings = _DEFAULT_SETTINGS) -> np.ndarray:
    """Return gated depth evaluation points centered inside the bar flaw gate.

    Near-surface and near-back-wall zones are covered by the rules engine only
    (the ML adds nothing there) - a stated limitation.
    """
    probe = scan.spec.probe
    c_metal_mm_per_us = scan.spec.material.velocity_m_s * 1e-3
    half_window_mm = (WINDOW / 2) / probe.sampling_mhz * c_metal_mm_per_us / 2
    d_min = settings.front_mm + half_window_mm
    d_max = scan.spec.diameter_mm - settings.back_margin_mm - half_window_mm
    if d_max < d_min:
        return np.zeros(0, dtype=np.float64)
    depths = np.arange(d_min, d_max + 0.5 * settings.depth_step_mm, settings.depth_step_mm)
    return depths[depths <= d_max + 1e-6]


def _make_grid_sites(
    scan: BarScan,
    settings: ScanSettings,
) -> tuple[np.ndarray, np.ndarray]:
    n_ax, n_an = scan.ascans.shape[0], scan.ascans.shape[1]
    depths = site_depths(scan, settings)
    if any((len(depths) == 0, n_ax == 0, n_an == 0)):
        return np.zeros((0, 3), dtype=np.float64), depths
    ax_grid, an_grid, d_grid = np.meshgrid(
        np.arange(n_ax, dtype=np.float64),
        np.arange(n_an, dtype=np.float64),
        depths,
        indexing="ij",
    )
    sites = np.column_stack([ax_grid.ravel(), an_grid.ravel(), d_grid.ravel()])
    return sites, depths


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
    all_logits = np.empty((passes, n_samples), dtype=np.float64)
    _activate_mc_dropout(model)
    model.to(dev)
    for start in range(0, n_samples, settings.batch):
        end = min(start + settings.batch, n_samples)
        batch = torch.from_numpy(windows[start:end]).to(dev)
        for p_idx in range(passes):
            logits, _ = model(batch)
            lg = flaw_logit(logits)
            all_logits[p_idx, start:end] = lg.detach().cpu().to(torch.float64).numpy()
    model.eval()
    all_probs = 1.0 / (1.0 + np.exp(-np.clip(all_logits, -500.0, 500.0)))
    mean_prob, std = ensemble_stats(all_probs)
    mean_logit = np.mean(all_logits, axis=0)
    unc = np.minimum(1.0, 2.0 * std)
    return _InferStats(
        score=mean_prob.astype(np.float64),
        logit=mean_logit.astype(np.float64),
        uncertainty=unc.astype(np.float64),
        std=std.astype(np.float64),
    )


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


def _pick_best(group: list[int], cands: list[_Candidate]) -> _Candidate:
    best_idx = max(group, key=lambda idx: (cands[idx].logit, -idx))
    return cands[best_idx]


def _suppress_candidates(cands: list[_Candidate], n_angle: int) -> list[_Candidate]:
    if not cands:
        return []
    adj = _build_adj(cands, n_angle)
    groups = _connected_groups(adj)
    kept = [_pick_best(grp, cands) for grp in groups]
    kept.sort(key=lambda c: c.logit, reverse=True)
    return kept


def _find_root(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _union(parent: list[int], i: int, j: int) -> None:
    ri, rj = _find_root(parent, i), _find_root(parent, j)
    if ri != rj:
        parent[ri] = rj


def _merge_angle_boundary(
    labeled: np.ndarray,
    parent: list[int],
    n_ax: int,
    n_d: int,
) -> None:
    s0 = labeled[:, 0, :]
    slast = labeled[:, -1, :]
    for ax, d in np.argwhere(s0 > 0):
        l0 = s0[ax, d]
        ax_min = max(0, ax - 1)
        ax_max = min(n_ax, ax + 2)
        d_min = max(0, d - 1)
        d_max = min(n_d, d + 2)
        sub = slast[ax_min:ax_max, d_min:d_max]
        for l1 in sub[sub > 0]:
            _union(parent, int(l0), int(l1))


def _build_component_grid(
    labeled: np.ndarray,
    parent: list[int],
    num_features: int,
) -> tuple[int, np.ndarray]:
    roots: dict[int, int] = {}
    for i in range(1, num_features + 1):
        r = _find_root(parent, i)
        if r not in roots:
            roots[r] = len(roots) + 1
    remap = np.zeros(num_features + 1, dtype=np.int32)
    for i in range(1, num_features + 1):
        remap[i] = roots[_find_root(parent, i)]
    return len(roots), remap[labeled]


def _label_connected(mask: np.ndarray) -> tuple[int, np.ndarray]:
    if not np.any(mask):
        return 0, np.zeros_like(mask, dtype=np.int32)
    structure = np.ones((3, 3, 3), dtype=bool)
    labeled, num_features = ndi.label(mask, structure=structure)
    if num_features <= 1 or mask.shape[1] < 3:
        return num_features, labeled
    parent = list(range(num_features + 1))
    _merge_angle_boundary(labeled, parent, mask.shape[0], mask.shape[2])
    return _build_component_grid(labeled, parent, num_features)


def count_findings(logits_grid: np.ndarray, logit_threshold: float) -> int:
    """Return number of connected groups of sites >= logit_threshold."""
    if logits_grid.size == 0:
        return 0
    mask = logits_grid >= logit_threshold
    n_comp, _ = _label_connected(mask)
    return n_comp


def _find_component_peaks(
    logits_grid: np.ndarray,
    comp_grid: np.ndarray,
) -> list[tuple[float, tuple[int, int, int]]]:
    best: dict[int, tuple[float, tuple[int, int, int]]] = {}
    for ax, an, d_idx in np.argwhere(comp_grid > 0):
        cid = int(comp_grid[ax, an, d_idx])
        lg = float(logits_grid[ax, an, d_idx])
        pos = (int(ax), int(an), int(d_idx))
        if cid not in best or lg > best[cid][0]:
            best[cid] = (lg, pos)
    peaks = list(best.values())
    peaks.sort(key=lambda p: (p[0], -p[1][0], -p[1][1], -p[1][2]), reverse=True)
    return peaks


def _peaks_to_findings(
    peaks: list[tuple[float, tuple[int, int, int]]],
    depths: np.ndarray,
) -> list[MlFinding]:
    findings: list[MlFinding] = []
    for lg, (ax, an, d_idx) in peaks:
        score = float(1.0 / (1.0 + np.exp(-np.clip(lg, -500.0, 500.0))))
        depth_val = float(depths[d_idx]) if d_idx < len(depths) else 0.0
        findings.append(
            MlFinding(
                axial_idx=ax,
                angle_idx=an,
                depth_mm=depth_val,
                score=score,
                uncertainty=0.0,
                grain_noise_prob=float(1.0 - score),
                evidence=(
                    f"ML window score {score:.2f}",
                    "MC-dropout std 0.00",
                ),
            )
        )
    return findings


def site_logits(
    model: WindowNet,
    scan: BarScan,
    settings: ScanSettings = _DEFAULT_SETTINGS,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-site flaw logits on the (axial, angle, depth) grid and depth axis."""
    n_ax, n_an = scan.ascans.shape[0], scan.ascans.shape[1]
    sites, depths = _make_grid_sites(scan, settings)
    if len(sites) == 0:
        return np.zeros((n_ax, n_an, len(depths)), dtype=np.float64), depths
    dac = dac_for(scan)
    is_patch = getattr(model.stem[0], "in_channels", 1) == 9
    windows = cut_windows(scan, dac, sites, patch=is_patch)
    dev = pick_device(device)
    stats = _eval_batches(model, windows, settings, dev)
    logits_grid = stats.logit.reshape(n_ax, n_an, len(depths))
    return logits_grid, depths


def scan_bar(
    model: WindowNet,
    scan: BarScan,
    settings: ScanSettings = _DEFAULT_SETTINGS,
    device: str = "auto",
) -> list[MlFinding]:
    """Scan a bar with WindowNet, returning non-maximum suppressed findings."""
    depths_check = site_depths(scan, settings)
    if len(depths_check) == 0:
        return []
    logits_grid, depths = site_logits(model, scan, settings, device)
    if logits_grid.size == 0:
        return []
    mask = logits_grid >= settings.threshold
    n_comp, comp_grid = _label_connected(mask)
    if n_comp == 0:
        return []
    peaks = _find_component_peaks(logits_grid, comp_grid)
    return _peaks_to_findings(peaks, depths)


def _extract_grid_maxima(grid: np.ndarray) -> np.ndarray:
    if grid.size == 0:
        return np.zeros(0, dtype=np.float64)
    mf = ndi.maximum_filter(grid, size=3)
    return np.unique(grid[grid == mf])


def _pool_candidate_logits(grids: list[np.ndarray]) -> np.ndarray:
    maxima_list = [_extract_grid_maxima(g) for g in grids if g.size > 0]
    if not maxima_list:
        return np.zeros(0, dtype=np.float64)
    pooled = np.concatenate(maxima_list)
    return np.sort(np.unique(pooled))[::-1]


def calibrate_bar_threshold(
    model: WindowNet,
    clean_scans: list[BarScan],
    max_flags_per_bar: float = 1.0,
    settings: ScanSettings = _DEFAULT_SETTINGS,
) -> float:
    """Calibrate the smallest threshold giving <= max_flags_per_bar on clean bars."""
    if not clean_scans:
        return 0.0
    if any(len(site_depths(s, settings)) == 0 for s in clean_scans):
        return 0.0
    grids = [site_logits(model, scan, settings)[0] for scan in clean_scans]
    cands = _pool_candidate_logits(grids)
    if len(cands) == 0:
        return 0.0
    best_val = float(cands[0]) + 0.1
    for val in cands:
        mean_cnt = float(np.mean([count_findings(g, float(val)) for g in grids]))
        if mean_cnt <= max_flags_per_bar:
            best_val = float(val)
        else:
            break
    return float(best_val)


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
    "count_findings",
    "findings_to_dicts",
    "load_window_model",
    "save_window_model",
    "scan_bar",
    "site_depths",
    "site_logits",
]
