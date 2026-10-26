"""Tests for bar-domain ML window generation (Synthetic data)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Ensure local workspace src is prioritized over external paths
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        sys.modules["utriage"].__path__.insert(0, _pkg_dir)
if "utriage.ml" in sys.modules:
    _ml_dir = str(Path(_SRC) / "utriage" / "ml")
    if _ml_dir not in sys.modules["utriage.ml"].__path__:
        sys.modules["utriage.ml"].__path__.insert(0, _ml_dir)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from utriage.ml.bar_data import (  # noqa: E402
    CLASSES,
    WINDOW,
    WindowConfig,
    WindowSet,
    cut_windows,
    flaw_signal,
    implant,
    make_test_set,
    make_training_set,
)
from utriage.physics.simulator import (  # noqa: E402
    BarSpec,
    Reflector,
    envelope,
    load_material,
    load_probe,
    simulate_bar,
)
from utriage.rules.pipeline import dac_for  # noqa: E402


@pytest.fixture
def base_scan():
    mat = load_material("ti64")
    probe = load_probe("p5")
    water = load_material("water")
    spec = BarSpec(
        material=mat,
        probe=probe,
        water=water,
        diameter_mm=40.0,
        length_mm=16.0,
        axial_step_mm=1.0,
        n_angle=36,
        seed=1,
    )
    return simulate_bar(spec)


@pytest.mark.parametrize(("patch", "channels"), [(False, 1), (True, 9)])
def test_shapes_and_dtypes(patch: bool, channels: int) -> None:
    cfg = WindowConfig(patch=patch, seed=42)
    wset = make_training_set(cfg, n_bars=1, per_bar=4)

    assert wset.x.ndim == 3
    assert wset.x.shape[1] == channels
    assert wset.x.shape[2] == WINDOW
    assert wset.x.dtype == np.float32
    assert wset.y.dtype == np.int64
    assert wset.size_mm.dtype == np.float32
    assert wset.depth_mm.dtype == np.float32
    assert wset.noise_scale.dtype == np.float32


def test_implant_zeros() -> None:
    noise = np.ones((2, 1, WINDOW), dtype=np.float32)
    zeros = np.zeros_like(noise)
    res = implant(noise, zeros)
    np.testing.assert_array_equal(res, noise)
    assert res.dtype == np.float32


def test_flaw_signal_peak_dac(base_scan) -> None:
    dac = dac_for(base_scan)
    ref = Reflector(
        axial_mm=8.0,
        angle_deg=0.0,
        depth_mm=15.0,
        diameter_mm=0.8,
        kind="fbh",
    )
    site = np.array([[8, 0, 15.0]])
    sig = flaw_signal(base_scan, dac, ref, site, patch=False)
    env = envelope(sig[0, 0])
    peak = float(np.max(env))
    assert np.isclose(peak, 1.0, atol=0.10)


def test_class_counts() -> None:
    cfg = WindowConfig(patch=False, seed=7)
    wset = make_training_set(cfg, n_bars=1, per_bar=4)
    noise_count = int(np.sum(wset.y == CLASSES.index("noise")))
    flaw_count = int(np.sum(wset.y == CLASSES.index("flaw")))
    assert noise_count > 0
    assert flaw_count > 0


def test_save_load_round_trip(tmp_path: Path) -> None:
    cfg = WindowConfig(patch=False, seed=9)
    wset = make_training_set(cfg, n_bars=1, per_bar=2)
    save_path = tmp_path / "windows.npz"
    wset.save(save_path)
    loaded = WindowSet.load(save_path)

    np.testing.assert_array_equal(loaded.x, wset.x)
    np.testing.assert_array_equal(loaded.y, wset.y)
    np.testing.assert_array_equal(loaded.size_mm, wset.size_mm)
    np.testing.assert_array_equal(loaded.depth_mm, wset.depth_mm)
    np.testing.assert_array_equal(loaded.noise_scale, wset.noise_scale)


def test_determinism() -> None:
    cfg = WindowConfig(patch=False, seed=99)
    set1 = make_training_set(cfg, n_bars=1, per_bar=3)
    set2 = make_training_set(cfg, n_bars=1, per_bar=3)
    np.testing.assert_array_equal(set1.x, set2.x)
    np.testing.assert_array_equal(set1.y, set2.y)


def test_make_test_set_basic() -> None:
    cfg = WindowConfig(patch=False, seed=12)
    test_set = make_test_set(
        cfg,
        n_bars=1,
        sizes_mm=(0.5, 0.8),
        noise_scales=(1.0,),
    )
    assert test_set.x.shape[0] == 4  # 2 flaws + 2 noise
    assert np.sum(test_set.y == CLASSES.index("flaw")) == 2
    assert np.sum(test_set.y == CLASSES.index("noise")) == 2


def test_cut_windows_empty(base_scan) -> None:
    dac = dac_for(base_scan)
    empty = cut_windows(base_scan, dac, np.zeros((0, 3)), patch=False)
    assert empty.shape == (0, 1, WINDOW)
    empty_patch = cut_windows(base_scan, dac, np.zeros((0, 3)), patch=True)
    assert empty_patch.shape == (0, 9, WINDOW)


def test_flaw_signal_inclusion(base_scan) -> None:
    dac = dac_for(base_scan)
    ref = Reflector(
        axial_mm=8.0,
        angle_deg=0.0,
        depth_mm=15.0,
        diameter_mm=0.8,
        kind="inclusion",
        amplitude_factor=0.5,
    )
    site = np.array([[8, 0, 15.0]])
    sig = flaw_signal(base_scan, dac, ref, site, patch=True)
    assert sig.shape == (1, 9, WINDOW)
