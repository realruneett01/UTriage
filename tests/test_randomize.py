"""Tests for domain randomization and streaming synthetic bar dataset generation.

Synthetic data.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure local workspace src is prioritized over external paths
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        sys.modules["utriage"].__path__.insert(0, _pkg_dir)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

from utriage.physics.dataset import DatasetConfig  # noqa: E402
from utriage.physics.randomize import (  # noqa: E402
    DEFAULT_HOLDOUT_REGIONS,
    RandomBarStream,
    RandomizationConfig,
    Region,
    _get_worker_seed,
    load_randomization,
    make_bar_spec,
    sample_in_region,
    sample_params,
)
from utriage.physics.simulator import BarScan, BarSpec, simulate_bar  # noqa: E402


@pytest.fixture
def cfg() -> RandomizationConfig:
    """Fixture providing loaded default RandomizationConfig."""
    return load_randomization()


@pytest.fixture
def tiny_cfg() -> DatasetConfig:
    """Fixture providing minimal dataset configuration for fast simulation."""
    return DatasetConfig(length_mm=4.0, n_angle=8, diameter_mm=20.0, mean_reflectors=1.0)


def test_yaml_loads_and_every_entry_has_source() -> None:
    """YAML loads correctly and every entry in ranges and materials specifies a source."""
    yaml_path = Path(__file__).resolve().parents[1] / "configs" / "randomization.yaml"
    assert yaml_path.exists()
    with yaml_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    assert "ranges" in data
    assert "not_supported" in data
    assert "materials" in data

    for key, entry in data["ranges"].items():
        assert "min" in entry, f"Missing min in {key}"
        assert "max" in entry, f"Missing max in {key}"
        assert "source" in entry, f"Missing source in {key}"
        assert entry["source"].strip() != "", f"Empty source in {key}"

    assert "source" in data["materials"]
    for _key, reason in data["not_supported"].items():
        assert isinstance(reason, str)
        assert len(reason) > 5


def test_sample_params_never_lands_in_excluded_region(cfg: RandomizationConfig) -> None:
    """sample_params never lands in an excluded region across 1,000 draws."""
    rng = np.random.default_rng(20261027)
    for _ in range(1000):
        params = sample_params(cfg, rng, exclude=DEFAULT_HOLDOUT_REGIONS)
        for reg in DEFAULT_HOLDOUT_REGIONS:
            assert not reg.matches(params), f"Sample landed in excluded region {reg.name}: {params}"


@pytest.mark.parametrize("region", DEFAULT_HOLDOUT_REGIONS)
def test_sample_in_region_always_inside(cfg: RandomizationConfig, region: Region) -> None:
    """sample_in_region always produces parameters matching the target region."""
    rng = np.random.default_rng(42)
    for _ in range(50):
        params = sample_in_region(cfg, rng, region)
        assert region.matches(params), f"Failed to sample inside region {region.name}: {params}"


def test_same_seed_gives_same_params(cfg: RandomizationConfig) -> None:
    """Identical RNG state produces identical sampled parameters."""
    rng1 = np.random.default_rng(999)
    rng2 = np.random.default_rng(999)
    p1 = sample_params(cfg, rng1)
    p2 = sample_params(cfg, rng2)
    assert p1 == p2


def test_two_workers_different_ids_give_different_streams(
    cfg: RandomizationConfig, tiny_cfg: DatasetConfig
) -> None:
    """Two workers with different ids yield different seeds and distinct streams."""
    base_seed = 42
    s0 = _get_worker_seed(base_seed=base_seed, worker_id=0)
    s1 = _get_worker_seed(base_seed=base_seed, worker_id=1)
    assert s0 != s1

    stream0 = RandomBarStream(cfg=cfg, base_cfg=tiny_cfg, base_seed=base_seed, worker_id=0)
    stream1 = RandomBarStream(cfg=cfg, base_cfg=tiny_cfg, base_seed=base_seed, worker_id=1)

    _, params0 = next(iter(stream0))
    _, params1 = next(iter(stream1))
    assert params0 != params1


def test_small_simulate_bar_runs_under_10_seconds(
    cfg: RandomizationConfig, tiny_cfg: DatasetConfig
) -> None:
    """Simulating one bar from sampled parameters completes in well under 10 seconds."""
    rng = np.random.default_rng(7)
    params = sample_params(cfg, rng)
    spec = make_bar_spec(params, tiny_cfg, rng)

    assert isinstance(spec, BarSpec)
    t0 = time.perf_counter()
    scan = simulate_bar(spec)
    elapsed = time.perf_counter() - t0

    assert elapsed < 10.0
    assert isinstance(scan, BarScan)
    assert scan.ascans.shape[0] > 0
    assert scan.ascans.shape[1] == tiny_cfg.n_angle


def test_random_bar_stream_transform_and_limit(
    cfg: RandomizationConfig, tiny_cfg: DatasetConfig
) -> None:
    """RandomBarStream supports user transforms and limits output when max_items is set."""
    stream = RandomBarStream(
        cfg=cfg,
        base_cfg=tiny_cfg,
        base_seed=123,
        transform=lambda scan, p: (scan.ascans.shape, p["material"]),
        max_items=2,
    )
    items = list(stream)
    assert len(items) == 2
    for shape, mat in items:
        assert len(shape) == 3
        assert isinstance(mat, str)
