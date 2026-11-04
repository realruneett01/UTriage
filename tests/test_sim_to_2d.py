"""Tests for 2D B-scan generation from simulated immersion A-scans (Synthetic data)."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import ConcatDataset, DataLoader, Dataset  # noqa: E402

from utriage.ml.sim_to_2d import (  # noqa: E402
    SimBscanConfig,
    SimBscanDataset,
    build_bscan,
    flaw_region_slice,
    simulate_bscan,
)


def _assert_image_properties(img: np.ndarray) -> None:
    """Validate 2D image shape, dtype, and standardised statistics."""
    assert isinstance(img, np.ndarray)
    assert img.shape == (256, 256)
    assert img.dtype == np.float32
    assert float(np.mean(img)) == pytest.approx(0.0, abs=1e-3)
    assert float(np.std(img)) == pytest.approx(1.0, abs=1e-3)


def _assert_tensor_sample(tensor: torch.Tensor, label: torch.Tensor, expected: float) -> None:
    """Validate PyTorch dataset sample shape, dtype, and label."""
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 256, 256)
    assert tensor.dtype == torch.float32
    assert isinstance(label, torch.Tensor)
    assert label.item() == pytest.approx(expected)


class _DummyDataset(Dataset):
    """Minimal dummy dataset yielding (tensor (1, 256, 256), label)."""

    def __init__(self, length: int = 2) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tensor = torch.zeros((1, 256, 256), dtype=torch.float32)
        label = torch.tensor(float(idx % 2), dtype=torch.float32)
        return tensor, label


@pytest.mark.parametrize("shape", [(16, 200), (32, 500)])
def test_build_bscan_shape_and_dtype(shape: tuple[int, int]) -> None:
    """build_bscan converts arbitrary (n_pos, n_samp) RF into 256x256 float32."""
    rng = np.random.default_rng(42)
    raw_ascans = rng.standard_normal(shape).astype(np.float32)
    img = build_bscan(raw_ascans)
    _assert_image_properties(img)


def test_build_bscan_invalid_dimensions() -> None:
    """build_bscan rejects 1D or unsupported input dimensions."""
    with pytest.raises(ValueError, match="Expected 2D"):
        build_bscan(np.zeros(256, dtype=np.float32))


@pytest.mark.parametrize("with_flaw, expected_label", [(True, 1), (False, 0)])
def test_simulate_bscan_shape_and_label(with_flaw: bool, expected_label: int) -> None:
    """simulate_bscan produces 256x256 float32 image and matched integer label."""
    img, label = simulate_bscan(seed=7, with_flaw=with_flaw)
    _assert_image_properties(img)
    assert label == expected_label


@pytest.mark.parametrize("seed", [0, 42])
def test_simulate_bscan_deterministic_per_seed(seed: int) -> None:
    """simulate_bscan outputs identical results for identical seeds."""
    img1, lbl1 = simulate_bscan(seed=seed, with_flaw=True)
    img2, lbl2 = simulate_bscan(seed=seed, with_flaw=True)
    np.testing.assert_array_equal(img1, img2)
    assert lbl1 == lbl2


def test_simulate_bscan_distinct_seeds_differ() -> None:
    """Different simulation seeds produce distinct grain noise patterns."""
    img_a, _ = simulate_bscan(seed=1, with_flaw=False)
    img_b, _ = simulate_bscan(seed=2, with_flaw=False)
    assert not np.array_equal(img_a, img_b)


@pytest.mark.parametrize("seed", [0, 42, 99])
def test_flaw_peak_exceeds_clean_in_flaw_region(seed: int) -> None:
    """Peak response in flaw region exceeds clean background by a wide margin."""
    cfg = SimBscanConfig(length_mm=16.0, axial_step_mm=0.25)
    img_flaw, _ = simulate_bscan(seed=seed, with_flaw=True, cfg=cfg)
    img_clean, _ = simulate_bscan(seed=seed, with_flaw=False, cfg=cfg)
    r_slice, c_slice = flaw_region_slice(cfg)
    peak_flaw = float(img_flaw[r_slice, c_slice].max())
    peak_clean = float(img_clean[r_slice, c_slice].max())
    assert peak_flaw > peak_clean + 0.5


def test_dataset_len_and_indexing() -> None:
    """SimBscanDataset yields expected tensor samples and labels."""
    cfg = SimBscanConfig(length_mm=16.0, axial_step_mm=0.25)
    ds = SimBscanDataset(length=2, base_seed=10, cfg=cfg)
    assert len(ds) == 2
    x0, y0 = ds[0]
    _assert_tensor_sample(x0, y0, 0.0)
    x1, y1 = ds[1]
    _assert_tensor_sample(x1, y1, 1.0)


@pytest.mark.parametrize("bad_idx", [-3, 5])
def test_dataset_index_out_of_bounds(bad_idx: int) -> None:
    """IndexError is raised when accessing indices outside dataset bounds."""
    ds = SimBscanDataset(length=2, base_seed=0)
    with pytest.raises(IndexError):
        _ = ds[bad_idx]


def test_dataset_deterministic_per_seed() -> None:
    """Datasets with matching base_seed yield identical tensor values."""
    ds1 = SimBscanDataset(length=2, base_seed=42)
    ds2 = SimBscanDataset(length=2, base_seed=42)
    x1, y1 = ds1[0]
    x2, y2 = ds2[0]
    assert torch.equal(x1, x2)
    assert y1 == y2


def test_concat_dataset_with_dummy_dataset() -> None:
    """SimBscanDataset mixes with dummy dataset in ConcatDataset and DataLoader."""
    cfg = SimBscanConfig(length_mm=16.0, axial_step_mm=0.25)
    sim_ds = SimBscanDataset(length=2, base_seed=0, cfg=cfg)
    dummy_ds = _DummyDataset(length=2)
    concat = ConcatDataset([sim_ds, dummy_ds])
    assert len(concat) == 4

    loader = DataLoader(concat, batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(batches) == 2
    for x_batch, y_batch in batches:
        assert x_batch.shape == (2, 1, 256, 256)
        assert x_batch.dtype == torch.float32
        assert y_batch.shape == (2,)
        assert y_batch.dtype == torch.float32
