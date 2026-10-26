"""Tests for ML-NDT dataset loading, standardisation, and batch splitting."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure local worktree src/ takes precedence over any installed package
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from utriage.ml.datasets import (  # noqa: E402
    MLNDT_ROOT,
    MlNdtDataset,
    list_batches,
    read_batch,
    read_image,
    read_labels,
    split_holdout,
    standardise,
)


def _create_fake_batch(dir_path: Path, stem: str, seed: int) -> Path:
    """Create a single fake .bins and .labels batch file pair."""
    rng = np.random.default_rng(seed)
    bins_path = dir_path / f"{stem}.bins"
    labels_path = dir_path / f"{stem}.labels"

    raw_images = rng.integers(100, 3000, size=(100, 256, 256), dtype=np.uint16)
    raw_images.tofile(bins_path)

    flaws = rng.integers(0, 2, size=100)
    sizes = np.where(flaws == 1, rng.uniform(0.5, 8.0, size=100), 0.0)
    lines = [f"{f}\t{s:.6f}\n" for f, s in zip(flaws, sizes, strict=True)]
    labels_path.write_text("".join(lines), encoding="utf-8")
    return bins_path


@pytest.fixture
def fake_dataset_root(tmp_path: Path) -> Path:
    """Create a fake ML-NDT root with training and validation splits."""
    root = tmp_path / "ML-NDT" / "data"
    train_dir = root / "training"
    val_dir = root / "validation"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    _create_fake_batch(train_dir, "batch_t1", seed=10)
    _create_fake_batch(train_dir, "batch_t2", seed=20)
    _create_fake_batch(train_dir, "batch_t3", seed=30)
    _create_fake_batch(val_dir, "batch_v1", seed=40)
    return root


def test_standardise_distribution() -> None:
    """Standardise normalises mean to ~0 and standard deviation to ~1."""
    rng = np.random.default_rng(42)
    img = rng.integers(200, 2500, size=(256, 256), dtype=np.uint16)
    norm = standardise(img)

    assert norm.dtype == np.float32
    assert abs(float(np.mean(norm))) < 1e-4
    assert abs(float(np.std(norm)) - 1.0) < 1e-3


def test_standardise_constant_image() -> None:
    """Standardise handles constant arrays safely without division by zero."""
    img = np.full((256, 256), fill_value=500, dtype=np.uint16)
    norm = standardise(img)
    assert norm.dtype == np.float32
    assert np.allclose(norm, 0.0, atol=1e-4)


def test_read_batch_shapes_and_types(fake_dataset_root: Path) -> None:
    """Read_batch returns a uint16 memmap and float32 labels of proper shapes."""
    bins_file = fake_dataset_root / "training" / "batch_t1.bins"
    mmap, labels = read_batch(bins_file)

    assert isinstance(mmap, np.memmap)
    assert mmap.dtype == np.uint16
    assert mmap.shape == (100, 256, 256)
    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.float32
    assert labels.shape == (100, 2)


def test_list_batches(fake_dataset_root: Path) -> None:
    """List_batches returns sorted paths for valid splits and raises on invalid."""
    train_batches = list_batches("training", root=fake_dataset_root)
    assert len(train_batches) == 3
    assert all(b.suffix == ".bins" for b in train_batches)
    assert train_batches == sorted(train_batches)

    val_batches = list_batches("validation", root=fake_dataset_root)
    assert len(val_batches) == 1

    with pytest.raises(ValueError, match="Unknown split"):
        list_batches("testing", root=fake_dataset_root)


def test_split_holdout_determinism_and_disjointness(fake_dataset_root: Path) -> None:
    """Split_holdout deterministically partitions batches into disjoint sets."""
    train_batches = list_batches("training", root=fake_dataset_root)
    train1, test1 = split_holdout(train_batches, test_frac=0.33, seed=42)
    train2, test2 = split_holdout(train_batches, test_frac=0.33, seed=42)

    assert train1 == train2
    assert test1 == test2
    assert set(train1).isdisjoint(set(test1))
    assert set(train1) | set(test1) == set(train_batches)
    assert len(train1) == 2
    assert len(test1) == 1


def test_dataset_len_and_indexing(fake_dataset_root: Path) -> None:
    """MlNdtDataset reports length 100 * batch_count and retrieves samples."""
    train_batches = list_batches("training", root=fake_dataset_root)
    dataset = MlNdtDataset(train_batches, size=256)
    _, expected_labels = read_batch(train_batches[0])

    assert len(dataset) == 300
    x, flaw, eq_size = dataset[0]

    assert isinstance(x, torch.Tensor)
    assert x.shape == torch.Size([1, 256, 256])
    assert x.dtype == torch.float32
    assert isinstance(flaw, torch.Tensor)
    assert flaw.shape == torch.Size([])
    assert flaw.item() == pytest.approx(float(expected_labels[0, 0]))
    assert isinstance(eq_size, torch.Tensor)
    assert eq_size.shape == torch.Size([])
    assert eq_size.item() == pytest.approx(float(expected_labels[0, 1]))


def test_dataset_invalid_size(fake_dataset_root: Path) -> None:
    """MlNdtDataset raises ValueError for unsupported sizes."""
    train_batches = list_batches("training", root=fake_dataset_root)
    with pytest.raises(ValueError, match="Unsupported size"):
        MlNdtDataset(train_batches, size=64)


def test_dataset_size_128_pooling(fake_dataset_root: Path) -> None:
    """Size 128 mode performs exact 2x2 average pooling on standardised image."""
    train_batches = list_batches("training", root=fake_dataset_root)
    ds_256 = MlNdtDataset(train_batches, size=256)
    ds_128 = MlNdtDataset(train_batches, size=128)

    x_256, _, _ = ds_256[0]
    x_128, _, _ = ds_128[0]

    assert x_128.shape == torch.Size([1, 128, 128])
    expected = torch.nn.functional.avg_pool2d(x_256.unsqueeze(0), kernel_size=2, stride=2).squeeze(
        0
    )
    assert torch.allclose(x_128, expected, atol=1e-6)


def test_dataset_lazy_label_cache(fake_dataset_root: Path) -> None:
    """Labels are read lazily and cached per batch; images are never cached."""
    train_batches = list_batches("training", root=fake_dataset_root)
    dataset = MlNdtDataset(train_batches, size=256)

    assert len(dataset._labels) == 0
    _ = dataset[5]
    assert set(dataset._labels.keys()) == {0}
    _ = dataset[95]
    assert set(dataset._labels.keys()) == {0}
    _ = dataset[105]
    assert set(dataset._labels.keys()) == {0, 1}
    assert not hasattr(dataset, "_cache")


def test_read_image_matches_batch(fake_dataset_root: Path) -> None:
    """Offset reads return exactly the image the memmap batch view returns."""
    bins = list_batches("training", root=fake_dataset_root)[0]
    mmap, labels = read_batch(bins)
    for idx in (0, 37, 99):
        np.testing.assert_array_equal(read_image(bins, idx), np.asarray(mmap[idx]))
    np.testing.assert_array_equal(read_labels(bins), labels)


@pytest.mark.skipif(not MLNDT_ROOT.exists(), reason="MLNDT_ROOT does not exist")
def test_real_mlndt_batches() -> None:
    """When MLNDT_ROOT is present, verify 199 training and 2 validation batches."""
    train_batches = list_batches("training")
    val_batches = list_batches("validation")
    assert len(train_batches) == 199
    assert len(val_batches) == 2
