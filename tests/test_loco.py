"""Tests for leave-one-crack-out (LOCO) dataset and fold partitioning."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        del sys.modules["utriage"]

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from utriage.ml.datasets import (  # noqa: E402
    MlNdtDataset,
    list_batches,
    split_holdout,
)
from utriage.ml.loco import (  # noqa: E402
    PHYSICAL_CRACKS,
    IndexedMlNdtDataset,
    make_folds,
)
from utriage.ml.mlndt_meta import image_index  # noqa: E402


def _create_fake_batch(
    batch_path: Path,
    flaws: list[dict[str, Any]],
    labels: list[tuple[float, float]],
    seed: int,
) -> Path:
    """Write fake .bins, .jsons, and .labels files for a single batch."""
    rng = np.random.default_rng(seed)
    n = len(flaws)
    raw = rng.integers(100, 3000, size=(n, 256, 256), dtype=np.uint16)
    raw.tofile(batch_path)

    jsons_text = "".join(json.dumps({"flaws": [f]}) for f in flaws)
    batch_path.with_suffix(".jsons").write_text(jsons_text, encoding="utf-8")

    labels_arr = np.array(labels, dtype=np.float32)
    np.savetxt(batch_path.with_suffix(".labels"), labels_arr, delimiter="\t")
    return batch_path


@pytest.fixture
def fake_mlndt_root(tmp_path: Path) -> Path:
    """Populate fake ML-NDT root with training and validation batches."""
    root = tmp_path / "fake_mlndt"
    tr_dir = root / "training"
    val_dir = root / "validation"
    tr_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    flaws = [{"size": "1.6"}, {"size": "4.0"}, {"size": "8.6"}, {"size": "0.0"}]
    labels = [(1.0, 1.6), (1.0, 4.0), (1.0, 8.6), (0.0, 0.0)]

    for i in range(10):
        _create_fake_batch(tr_dir / f"batch_tr_{i:02d}.bins", flaws, labels, seed=i)
    for j in range(2):
        _create_fake_batch(val_dir / f"batch_val_{j:02d}.bins", flaws, labels, seed=100 + j)
    return root


@pytest.mark.parametrize("size", [256, 128])
def test_indexed_dataset_matches_mlndt_dataset(fake_mlndt_root: Path, size: int) -> None:
    """Verify IndexedMlNdtDataset produces identical tensors to MlNdtDataset."""
    bpath = fake_mlndt_root / "training" / "batch_tr_00.bins"
    full_ds = MlNdtDataset([bpath], size=size)
    items = [(bpath, 0), (bpath, 3)]
    sub_ds = IndexedMlNdtDataset(items, size=size)

    assert len(sub_ds) == 2
    for sub_idx, full_idx in enumerate([0, 3]):
        x_sub, flaw_sub, size_sub = sub_ds[sub_idx]
        x_full, flaw_full, size_full = full_ds[full_idx]
        assert torch.allclose(x_sub, x_full)
        assert torch.equal(flaw_sub, flaw_full)
        assert torch.equal(size_sub, size_full)


def test_indexed_dataset_indexing_bounds(fake_mlndt_root: Path) -> None:
    """Verify negative indexing and IndexError on out-of-bounds access."""
    bpath = fake_mlndt_root / "training" / "batch_tr_00.bins"
    items = [(bpath, 1), (bpath, 2)]
    ds = IndexedMlNdtDataset(items)

    assert len(ds) == 2
    x_pos, _, _ = ds[1]
    x_neg, _, _ = ds[-1]
    assert torch.equal(x_pos, x_neg)

    with pytest.raises(IndexError):
        _ = ds[2]
    with pytest.raises(IndexError):
        _ = ds[-3]


@pytest.mark.parametrize("crack", PHYSICAL_CRACKS)
def test_folds_disjoint_and_held_out_absent(fake_mlndt_root: Path, crack: str) -> None:
    """Verify fold train, val, and test are mutually disjoint and crack is held out."""
    tr_batches = list_batches("training", root=fake_mlndt_root)
    val_batches = list_batches("validation", root=fake_mlndt_root)
    folds = make_folds(tr_batches, val_batches, seed=0)

    fold = next(f for f in folds if f.held_out == crack)
    crack_map = {(b, idx): c for b, idx, c in image_index(tr_batches + val_batches)}

    s_tr = set(fold.train)
    s_val = set(fold.val)
    s_te = set(fold.test)

    assert s_tr.isdisjoint(s_val)
    assert s_tr.isdisjoint(s_te)
    assert s_val.isdisjoint(s_te)
    assert all(crack_map[it] != crack for it in fold.train)
    assert all(crack_map[it] != crack for it in fold.val)


@pytest.mark.parametrize("crack", PHYSICAL_CRACKS)
def test_unflawed_test_only_from_holdout_batches(fake_mlndt_root: Path, crack: str) -> None:
    """Verify unflawed test images originate strictly from holdout training batches."""
    tr_batches = list_batches("training", root=fake_mlndt_root)
    val_batches = list_batches("validation", root=fake_mlndt_root)
    _, holdout_batches = split_holdout(tr_batches, test_frac=0.1, seed=0)
    holdout_set = set(holdout_batches)

    folds = make_folds(tr_batches, val_batches, seed=0)
    fold = next(f for f in folds if f.held_out == crack)
    crack_map = {(b, idx): c for b, idx, c in image_index(tr_batches + val_batches)}

    unflawed_test = [it for it in fold.test if crack_map[it] is None]
    assert len(unflawed_test) > 0
    assert all(bins in holdout_set for bins, _ in unflawed_test)
