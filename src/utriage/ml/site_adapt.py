"""Unsupervised site adaptation for new inspection environments (project plan Phase 0 / ADR 0010).

Enables a trained UTriage model to adapt to a new plant, probe, material, or gain
using only inspector-confirmed clean reference scans with zero labelled flaws.
Advisory only: the rules engine stays in charge; UTriage never accepts or passes a part.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.stats
import torch
import torch.nn as nn

from utriage.triage.hybrid import quantile_threshold

_VALID_LABELS: frozenset[str] = frozenset({"Synthetic", "Real"})


@dataclass(frozen=True)
class SiteStats:
    """Per-site acoustic baseline statistics.

    Attributes
    ----------
    mean : float
        Overall reference scan pixel intensity mean.
    std : float
        Overall reference scan pixel intensity standard deviation.
    n_scans : int
        Number of clean scans used to estimate the statistics.
    label : str
        Data provenance ('Synthetic' or 'Real').
    """

    mean: float
    std: float
    n_scans: int
    label: str = "Synthetic"

    def __post_init__(self) -> None:
        if self.label not in _VALID_LABELS:
            msg = f"label must be one of {sorted(_VALID_LABELS)}, got {self.label!r}"
            raise ValueError(msg)
        if self.std < 0.0:
            msg = f"std must be non-negative, got {self.std}"
            raise ValueError(msg)
        if self.n_scans < 0:
            msg = f"n_scans must be non-negative, got {self.n_scans}"
            raise ValueError(msg)


@dataclass(frozen=True)
class SiteCalibration:
    """Per-site detection threshold calibration and confidence bound.

    Attributes
    ----------
    threshold : float
        Empirical anomaly score threshold corresponding to target false call rate.
    false_call_rate : float
        Target false call rate in [0.0, 1.0].
    n_clean : int
        Number of clean calibration scans evaluated.
    upper95_fcr : float
        One-sided 95% Clopper-Pearson upper confidence bound on false call rate.
    """

    threshold: float
    false_call_rate: float
    n_clean: int
    upper95_fcr: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.false_call_rate <= 1.0):
            msg = f"false_call_rate must be in [0.0, 1.0], got {self.false_call_rate}"
            raise ValueError(msg)
        if self.n_clean < 0:
            msg = f"n_clean must be non-negative, got {self.n_clean}"
            raise ValueError(msg)
        if not (0.0 <= self.upper95_fcr <= 1.0):
            msg = f"upper95_fcr must be in [0.0, 1.0], got {self.upper95_fcr}"
            raise ValueError(msg)


@dataclass(frozen=True)
class SiteProfile:
    """Container for loaded site profile with attribute and mapping access."""

    stats: SiteStats
    calib: SiteCalibration
    label: str
    created: str

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)

    def __contains__(self, item: object) -> bool:
        return hasattr(self, str(item))

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def keys(self) -> list[str]:
        return ["stats", "calib", "label", "created"]


def _extract_batch_tensor(batch: Any) -> torch.Tensor:
    """Extract input tensor from data loader batch representation."""
    if isinstance(batch, (tuple, list)):
        return batch[0]
    if isinstance(batch, dict):
        if "image" in batch:
            return batch["image"]
        if "x" in batch:
            return batch["x"]
        return next(iter(batch.values()))
    return batch


def _normalize_images(images: Sequence[np.ndarray] | np.ndarray) -> list[np.ndarray]:
    """Ensure images are represented as a list of numpy arrays."""
    if isinstance(images, np.ndarray) and images.ndim <= 2:
        return [images]
    return list(images)


def _accumulate_image_stats(imgs: list[np.ndarray]) -> tuple[float, float, int]:
    """Compute (total_sum, total_sq, total_pixels) across all image arrays."""
    total_pixels = sum(img.size for img in imgs)
    total_sum = sum(float(np.sum(img, dtype=np.float64)) for img in imgs)
    total_sq = sum(float(np.sum(np.square(img, dtype=np.float64))) for img in imgs)
    return total_sum, total_sq, total_pixels


def fit_site_stats(
    images: Sequence[np.ndarray] | np.ndarray,
    label: str = "Synthetic",
) -> SiteStats:
    """Estimate global pixel mean and standard deviation from clean scans."""
    if label not in _VALID_LABELS:
        raise ValueError(f"label must be one of {sorted(_VALID_LABELS)}, got {label!r}")
    if len(images) == 0:
        raise ValueError("images cannot be empty")

    imgs = _normalize_images(images)
    total_sum, total_sq, total_pixels = _accumulate_image_stats(imgs)
    mean_val = float(total_sum / total_pixels)
    var_val = max(0.0, float(total_sq / total_pixels - mean_val**2))
    return SiteStats(mean=mean_val, std=float(np.sqrt(var_val)), n_scans=len(imgs), label=label)


def apply_site_stats(
    img: np.ndarray | torch.Tensor,
    stats: SiteStats,
) -> np.ndarray | torch.Tensor:
    """Z-score image using site baseline statistics: (x - mean) / (std + 1e-5)."""
    denom = stats.std + 1e-5
    if isinstance(img, torch.Tensor):
        return (img.float() - stats.mean) / denom
    arr = np.asarray(img, dtype=np.float32)
    return ((arr - stats.mean) / denom).astype(np.float32)


def _prepare_batchnorm_modules(model: nn.Module) -> list[nn.Module]:
    """Find and configure BatchNorm modules for cumulative statistics collection."""
    bn_list = []
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None
            bn_list.append(m)
    return bn_list


def _forward_loader(model: nn.Module, loader: Iterable[Any], dev: torch.device) -> None:
    """Run loader through model without gradients to update BatchNorm running stats."""
    with torch.no_grad():
        for batch in loader:
            x = _extract_batch_tensor(batch).to(dev)
            model(x)


def recalibrate_batchnorm(
    model: nn.Module,
    loader: Iterable[Any],
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Adaptive Batch Normalization (AdaBN) using site clean scans."""
    bn_modules = _prepare_batchnorm_modules(model)
    if not bn_modules:
        return model

    dev = torch.device(device)
    model.to(dev)
    model.eval()
    for bn in bn_modules:
        bn.train()

    _forward_loader(model, loader, dev)
    model.eval()
    return model


def min_clean_scans(false_call_rate: float, confidence: float = 0.95) -> int:
    """Clean scans needed to bound false call rate at target confidence with zero false calls."""
    if not (0.0 < false_call_rate < 1.0):
        msg = f"false_call_rate must be in (0.0, 1.0), got {false_call_rate}"
        raise ValueError(msg)
    if not (0.0 < confidence < 1.0):
        msg = f"confidence must be in (0.0, 1.0), got {confidence}"
        raise ValueError(msg)
    return int(math.ceil(math.log(1.0 - confidence) / math.log(1.0 - false_call_rate)))


def _clopper_pearson_upper95(k: int, n: int) -> float:
    """One-sided 95% Clopper-Pearson upper bound for k false calls in n trials."""
    if k >= n:
        return 1.0
    return float(scipy.stats.beta.ppf(0.95, k + 1, n - k))


def _check_clean_scans(n_clean: int, fcr: float) -> None:
    """Warn if clean scan count is below minimum needed for target FCR at 95% confidence."""
    min_scans = min_clean_scans(fcr, confidence=0.95)
    if n_clean < min_scans:
        warnings.warn(
            f"Clean scans ({n_clean}) below recommended minimum ({min_scans}) "
            f"for false call rate {fcr} at 95% confidence.",
            UserWarning,
            stacklevel=3,
        )


def calibrate_site_threshold(
    clean_scores: np.ndarray | Sequence[float],
    false_call_rate: float,
) -> SiteCalibration:
    """Calibrate detection threshold on clean scores and calculate 95% upper bound."""
    arr = np.asarray(clean_scores, dtype=np.float64).ravel()
    n_clean = int(arr.size)
    if n_clean == 0:
        raise ValueError("clean_scores cannot be empty")
    if not (0.0 < false_call_rate < 1.0):
        raise ValueError(f"false_call_rate must be in (0.0, 1.0), got {false_call_rate}")

    _check_clean_scans(n_clean, false_call_rate)
    thresh = float(quantile_threshold(arr, fcr=false_call_rate))
    k_false = int(np.sum(arr > thresh))
    upper95 = _clopper_pearson_upper95(k_false, n_clean)
    return SiteCalibration(
        threshold=thresh,
        false_call_rate=float(false_call_rate),
        n_clean=n_clean,
        upper95_fcr=upper95,
    )


def save_site_profile(
    path: Path | str,
    stats: SiteStats,
    calib: SiteCalibration,
    label: str,
) -> Path:
    """Serialize site adaptation profile to JSON with ISO timestamp and provenance label."""
    if label not in _VALID_LABELS:
        msg = f"label must be one of {sorted(_VALID_LABELS)}, got {label!r}"
        raise ValueError(msg)
    if stats.label not in _VALID_LABELS:
        msg = f"stats.label must be one of {sorted(_VALID_LABELS)}, got {stats.label!r}"
        raise ValueError(msg)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created": datetime.now(UTC).isoformat(),
        "label": label,
        "advisory_only": True,
        "disclaimer": "Advisory only. UTriage never accepts or passes a part.",
        "stats": asdict(stats),
        "calib": asdict(calib),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def load_site_profile(path: Path | str) -> SiteProfile:
    """Load and reconstruct SiteProfile from serialized JSON file."""
    in_path = Path(path)
    data = json.loads(in_path.read_text(encoding="utf-8"))
    label = data.get("label", "")
    if label not in _VALID_LABELS:
        msg = f"Loaded profile has invalid label: {label!r}"
        raise ValueError(msg)

    stats_dict = data.get("stats", {})
    calib_dict = data.get("calib", {})
    stats = SiteStats(
        mean=float(stats_dict["mean"]),
        std=float(stats_dict["std"]),
        n_scans=int(stats_dict["n_scans"]),
        label=str(stats_dict.get("label", label)),
    )
    calib = SiteCalibration(
        threshold=float(calib_dict["threshold"]),
        false_call_rate=float(calib_dict["false_call_rate"]),
        n_clean=int(calib_dict["n_clean"]),
        upper95_fcr=float(calib_dict["upper95_fcr"]),
    )
    return SiteProfile(stats=stats, calib=calib, label=label, created=str(data.get("created", "")))
