"""Phase 3 inference utilities, model checkpointing, and latency benchmarking."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from utriage.ml.datasets import standardise
from utriage.ml.models import MlNdtNet
from utriage.ml.train import pick_device


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


def measure_latency(
    model: nn.Module,
    device: torch.device,
    batch: int,
    size: int = 256,
    repeats: int = 50,
) -> float:
    """Measure median milliseconds per image after 10 warmup runs."""
    model.eval()
    model.to(device)
    dummy = torch.zeros((batch, 1, size, size), dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings: list[float] = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            timings.append(((t1 - t0) * 1000.0) / batch)
    return float(np.median(timings))
