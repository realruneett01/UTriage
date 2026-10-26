"""Phase 3 inference utilities, model checkpointing, and latency benchmarking."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from utriage.ml.datasets import standardise
from utriage.ml.models import MlNdtNet
from utriage.ml.train import pick_device


@dataclass(frozen=True)
class LatencySpec:
    """Benchmark parameters for latency measurement."""

    batch: int = 1
    size: int = 256
    repeats: int = 50


def save_model(model: nn.Module, path: Path, meta: dict) -> Path:
    """Save model state_dict and architecture metadata to disk."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "meta": dict(meta),
    }
    torch.save(payload, out_path)
    return out_path


def load_model(path: Path, device: str = "auto") -> tuple[MlNdtNet, dict]:
    """Load model weights and metadata from checkpoint."""
    ckpt_path = Path(path)
    dev = pick_device(device)
    ckpt = torch.load(ckpt_path, map_location=dev)
    meta = dict(ckpt.get("meta", {}))
    width = int(meta.get("width", 32))
    pool_raw = meta.get("first_pool", (2, 2))
    first_pool = (int(pool_raw[0]), int(pool_raw[1]))
    model = MlNdtNet(width=width, first_pool=first_pool)
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev)
    model.eval()
    return model, meta


@torch.no_grad()
def predict_images(
    model: nn.Module, images: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Predict flaw probability and size for raw uint16 or float images."""
    if len(images) == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    model.eval()
    model.to(device)
    probs_list: list[np.ndarray] = []
    sizes_list: list[np.ndarray] = []
    batch_size = 64
    for i in range(0, len(images), batch_size):
        chunk = images[i : i + batch_size]
        std_chunk = np.stack([standardise(img) for img in chunk], axis=0)
        x = torch.from_numpy(std_chunk).unsqueeze(1).to(device)
        logit, size_pred = model(x)
        probs_list.append(torch.sigmoid(logit).view(-1).cpu().numpy().astype(np.float32))
        sizes_list.append(size_pred.view(-1).cpu().numpy().astype(np.float32))
    return np.concatenate(probs_list, axis=0), np.concatenate(sizes_list, axis=0)


def _sync_cuda(device: torch.device) -> None:
    """Synchronize CUDA stream if device is a CUDA GPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_one_pass(model: nn.Module, dummy: torch.Tensor, device: torch.device) -> float:
    """Time a single forward pass in seconds with proper CUDA synchronization."""
    _sync_cuda(device)
    t0 = time.perf_counter()
    _ = model(dummy)
    _sync_cuda(device)
    return time.perf_counter() - t0


def _resolve_spec(spec: LatencySpec | int | None, kwargs: dict[str, Any]) -> LatencySpec:
    """Resolve LatencySpec supporting backward-compatible positional or keyword arguments."""
    if isinstance(spec, LatencySpec):
        return spec
    batch = spec if isinstance(spec, int) else int(kwargs.get("batch", 1))
    size = int(kwargs.get("size", 256))
    repeats = int(kwargs.get("repeats", 50))
    return LatencySpec(batch=batch, size=size, repeats=repeats)


def measure_latency(
    model: nn.Module,
    device: torch.device,
    spec: LatencySpec | int | None = None,
    **kwargs: Any,
) -> float:
    """Measure median milliseconds per image after 10 warmup runs."""
    cfg = _resolve_spec(spec, kwargs)
    model.eval()
    model.to(device)
    dummy = torch.zeros((cfg.batch, 1, cfg.size, cfg.size), dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
        _sync_cuda(device)
        timings = [
            (_time_one_pass(model, dummy, device) * 1000.0) / cfg.batch for _ in range(cfg.repeats)
        ]
    return float(np.median(timings))
