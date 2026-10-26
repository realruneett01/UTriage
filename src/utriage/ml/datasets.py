"""ML-NDT streaming dataset and batch utilities (project plan Phase 3.1).

Streams batches from disk using np.memmap without loading entire dataset into RAM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

MLNDT_ROOT: Path = Path(__file__).resolve().parents[3] / "data" / "external" / "ML-NDT" / "data"
_VALID_SPLITS: frozenset[str] = frozenset({"training", "validation"})


def list_batches(split: str, root: Path = MLNDT_ROOT) -> list[Path]:
    """Return sorted .bins paths of split 'training' or 'validation'."""
    if split not in _VALID_SPLITS:
        raise ValueError(f"Unknown split: {split!r}. Expected one of {sorted(_VALID_SPLITS)}.")
    split_dir = root / split
    if not split_dir.exists():
        return []
    return sorted(split_dir.glob("*.bins"))


def split_holdout(
    batches: list[Path], test_frac: float = 0.1, seed: int = 0
) -> tuple[list[Path], list[Path]]:
    """Deterministic (train, test) split by batch."""
    if not batches:
        return [], []
    if test_frac <= 0.0:
        return list(batches), []
    if test_frac >= 1.0:
        return [], list(batches)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(batches))
    n_test = max(1, int(round(len(batches) * test_frac)))
    if n_test >= len(batches):
        n_test = len(batches) - 1

    shuffled = [batches[i] for i in perm]
    return shuffled[n_test:], shuffled[:n_test]


_IMG_PIXELS = 256 * 256
_IMG_BYTES = _IMG_PIXELS * np.dtype(np.uint16).itemsize


def n_images(bins: Path) -> int:
    """Number of 256x256 uint16 images in a .bins file, from its size."""
    return Path(bins).stat().st_size // _IMG_BYTES


def read_batch(bins: Path) -> tuple[np.memmap, np.ndarray]:
    """Read uint16 memmap shape (n, 256, 256) and labels float32 shape (n, 2).

    Most batches hold 100 images, but not all (one training batch holds 10), so ``n`` comes
    from the file size.
    """
    bins_path = Path(bins)
    shape = (n_images(bins_path), 256, 256)
    mmap = np.memmap(bins_path, dtype=np.uint16, mode="r", shape=shape)
    labels_path = bins_path.with_suffix(".labels")
    labels = np.loadtxt(labels_path, dtype=np.float32)
    if labels.ndim == 1:
        labels = labels.reshape(1, -1)
    return mmap, labels


def standardise(img: np.ndarray) -> np.ndarray:
    """Per-image standardisation: (x - mean) / (std + 1e-5) as float32."""
    x = img.astype(np.float32)
    mean = np.mean(x)
    std = np.std(x)
    return ((x - mean) / (std + 1e-5)).astype(np.float32)


def read_image(bins: Path, image_idx: int) -> np.ndarray:
    """Read one uint16 256x256 image straight from disk (no memory map kept open)."""
    offset = image_idx * _IMG_PIXELS * np.dtype(np.uint16).itemsize
    flat = np.fromfile(bins, dtype=np.uint16, count=_IMG_PIXELS, offset=offset)
    return flat.reshape(256, 256)


def read_labels(bins: Path) -> np.ndarray:
    """Labels (100, 2) float32 for the batch whose .bins path is given."""
    labels = np.loadtxt(Path(bins).with_suffix(".labels"), dtype=np.float32)
    return labels.reshape(-1, 2)


class MlNdtDataset(Dataset):
    """PyTorch Dataset streaming ML-NDT images from disk one at a time (bounded RAM)."""

    def __init__(self, batches: list[Path], size: int = 256) -> None:
        if size not in (128, 256):
            raise ValueError(f"Unsupported size: {size}. Expected 128 or 256.")
        self.batches: list[Path] = [Path(b) for b in batches]
        self.size: int = size
        self._labels: dict[int, np.ndarray] = {}
        self._starts = np.cumsum([0] + [n_images(b) for b in self.batches])

    def __len__(self) -> int:
        return int(self._starts[-1])

    def _get_labels(self, batch_idx: int) -> np.ndarray:
        """Labels of one batch, cached (small). Images are never cached."""
        if batch_idx not in self._labels:
            self._labels[batch_idx] = read_labels(self.batches[batch_idx])
        return self._labels[batch_idx]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for length {len(self)}")

        batch_idx = int(np.searchsorted(self._starts, index, side="right")) - 1
        image_idx = index - int(self._starts[batch_idx])
        labels = self._get_labels(batch_idx)

        img_std = standardise(read_image(self.batches[batch_idx], image_idx))
        x = torch.from_numpy(img_std).unsqueeze(0)
        if self.size == 128:
            x = F.avg_pool2d(x.unsqueeze(0), kernel_size=2, stride=2).squeeze(0)

        flaw = torch.tensor(float(labels[image_idx, 0]), dtype=torch.float32)
        eq_size = torch.tensor(float(labels[image_idx, 1]), dtype=torch.float32)
        return x, flaw, eq_size
