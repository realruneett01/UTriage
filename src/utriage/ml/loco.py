"""Leave-one-crack-out (LOCO) dataset and fold partitioning (project plan Phase 8.0).

Streams images from explicit (bins, idx) coordinates without caching images in RAM.
Partitions train, val, and test partitions to ensure complete crack independence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utriage.ml.datasets import (
    read_image,
    read_labels,
    split_holdout,
    standardise,
)
from utriage.ml.mlndt_meta import image_index

PHYSICAL_CRACKS: tuple[str, ...] = ("crack_1.6", "crack_4.0", "crack_8.6")


@dataclass(frozen=True)
class LocoFold:
    """Dataset partition indices for leave-one-crack-out cross-validation."""

    held_out: str
    train: list[tuple[Path, int]]
    val: list[tuple[Path, int]]
    test: list[tuple[Path, int]]
    val_crack: str | None = None


class IndexedMlNdtDataset(Dataset):
    """PyTorch Dataset streaming ML-NDT images from explicit (bins, idx) index."""

    def __init__(self, items: list[tuple[Path, int]], size: int = 256) -> None:
        if size not in (128, 256):
            raise ValueError(f"Unsupported size: {size}. Expected 128 or 256.")
        self.items: list[tuple[Path, int]] = [(Path(b), int(idx)) for b, idx in items]
        self.size: int = size
        self._labels_cache: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _get_labels(self, bins: Path) -> np.ndarray:
        if bins not in self._labels_cache:
            self._labels_cache[bins] = read_labels(bins)
        return self._labels_cache[bins]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self.items)
        if index < 0 or index >= len(self.items):
            raise IndexError(f"Index {index} out of range for length {len(self.items)}")
        bins, image_idx = self.items[index]
        img_std = standardise(read_image(bins, image_idx))
        x = torch.from_numpy(img_std).unsqueeze(0)
        if self.size == 128:
            x = F.avg_pool2d(x.unsqueeze(0), kernel_size=2, stride=2).squeeze(0)
        labels = self._get_labels(bins)
        flaw = torch.tensor(float(labels[image_idx, 0]), dtype=torch.float32)
        eq_size = torch.tensor(float(labels[image_idx, 1]), dtype=torch.float32)
        return x, flaw, eq_size

    def get_labels(self) -> tuple[np.ndarray, np.ndarray]:
        """Extract flaw indicators and equivalent sizes for all indexed items."""
        flaw_list: list[float] = []
        size_list: list[float] = []
        for bins, idx in self.items:
            lbl = self._get_labels(bins)
            flaw_list.append(float(lbl[idx, 0]))
            size_list.append(float(lbl[idx, 1]))
        return np.array(flaw_list, dtype=np.float32), np.array(size_list, dtype=np.float32)


def _next_crack(crack: str) -> str:
    """Return next physical crack in wrapping order."""
    idx = PHYSICAL_CRACKS.index(crack)
    return PHYSICAL_CRACKS[(idx + 1) % len(PHYSICAL_CRACKS)]


def _build_test_split(
    crack: str,
    all_tr_idx: list[tuple[Path, int, str | None]],
    holdout_idx: list[tuple[Path, int, str | None]],
) -> list[tuple[Path, int]]:
    """Build held-out test split containing crack flaw and holdout unflawed images."""
    test_flawed = [(b, idx) for b, idx, c in all_tr_idx if c == crack]
    test_unflawed = [(b, idx) for b, idx, c in holdout_idx if c is None]
    return test_flawed + test_unflawed


def _build_single_fold(
    crack: str,
    non_holdout_idx: list[tuple[Path, int, str | None]],
    holdout_idx: list[tuple[Path, int, str | None]],
    val_idx: list[tuple[Path, int, str | None]],
) -> LocoFold:
    """Build a single LocoFold partitioning for the target held-out crack."""
    train = [(b, idx) for b, idx, c in non_holdout_idx if c != crack]
    val = [(b, idx) for b, idx, c in val_idx if c != crack]
    test = _build_test_split(crack, non_holdout_idx + holdout_idx, holdout_idx)
    return LocoFold(held_out=crack, train=train, val=val, test=test)


def _build_crackval_fold(
    crack: str,
    non_holdout_idx: list[tuple[Path, int, str | None]],
    holdout_idx: list[tuple[Path, int, str | None]],
    val_idx: list[tuple[Path, int, str | None]],
) -> LocoFold:
    """Build fold with crack-held-out early-stopping validation split."""
    val_c = _next_crack(crack)
    train = [(b, idx) for b, idx, c in non_holdout_idx if c not in (crack, val_c)]
    all_idx = non_holdout_idx + holdout_idx + val_idx
    val_flawed = [(b, idx) for b, idx, c in all_idx if c == val_c]
    val_unflawed = [(b, idx) for b, idx, c in val_idx if c is None]
    val = val_flawed + val_unflawed
    test = _build_test_split(crack, non_holdout_idx + holdout_idx, holdout_idx)
    return LocoFold(held_out=crack, train=train, val=val, test=test, val_crack=val_c)


def _validate_val_crack(fold: LocoFold, crack_map: dict[tuple[Path, int], str | None]) -> None:
    """Assert that validation crack is absent from train and test splits."""
    if fold.val_crack is None:
        return
    if any(crack_map[it] == fold.val_crack for it in fold.train):
        raise AssertionError(f"Validation crack {fold.val_crack} present in train fold")
    if any(crack_map[it] == fold.val_crack for it in fold.test):
        raise AssertionError(f"Validation crack {fold.val_crack} present in test fold")


def _validate_fold(fold: LocoFold, crack_map: dict[tuple[Path, int], str | None]) -> None:
    """Assert mutual disjointness across splits and absence of held-out cracks."""
    s_tr, s_val, s_te = set(fold.train), set(fold.val), set(fold.test)
    disjoint_checks = [s_tr.isdisjoint(s_val), s_tr.isdisjoint(s_te), s_val.isdisjoint(s_te)]
    if not all(disjoint_checks):
        raise AssertionError(f"Folds overlap for {fold.held_out}")
    if any(crack_map[it] == fold.held_out for it in fold.train):
        raise AssertionError(f"Held-out crack {fold.held_out} present in train fold")
    if any(crack_map[it] == fold.held_out for it in fold.val):
        raise AssertionError(f"Held-out crack {fold.held_out} present in val fold")
    _validate_val_crack(fold, crack_map)


def make_folds(
    training_batches: list[Path],
    validation_batches: list[Path],
    seed: int = 0,
    val_mode: str = "batch",
) -> list[LocoFold]:
    """Generate 3 leave-one-crack-out folds for physical crack evaluation."""
    if val_mode not in ("batch", "crack"):
        raise ValueError(f"Unsupported val_mode: {val_mode}. Expected 'batch' or 'crack'.")
    non_ho_batches, ho_batches = split_holdout(training_batches, test_frac=0.1, seed=seed)
    non_ho_idx = image_index(non_ho_batches)
    ho_idx = image_index(ho_batches)
    val_idx = image_index(validation_batches)

    all_idx = non_ho_idx + ho_idx + val_idx
    crack_map = {(b, idx): c for b, idx, c in all_idx}

    folds: list[LocoFold] = []
    for crack in PHYSICAL_CRACKS:
        if val_mode == "crack":
            fold = _build_crackval_fold(crack, non_ho_idx, ho_idx, val_idx)
        else:
            fold = _build_single_fold(crack, non_ho_idx, ho_idx, val_idx)
        _validate_fold(fold, crack_map)
        folds.append(fold)
    return folds
