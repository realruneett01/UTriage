"""Tests for dataset generation, saving, and loading (Synthetic data)."""

import json
import sys
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

from utriage.physics.dataset import (  # noqa: E402
    DatasetConfig,
    generate_dataset,
    load_scan,
    random_bar_spec,
    save_scan,
)
from utriage.physics.simulator import SIM_VERSION  # noqa: E402


@pytest.fixture
def tiny_cfg() -> DatasetConfig:
    return DatasetConfig(n_bars=2, length_mm=6.0, n_angle=8, diameter_mm=20.0)


def test_generate_writes_files_and_manifest(tmp_path: Path, tiny_cfg: DatasetConfig) -> None:
    npz_paths = generate_dataset(tiny_cfg, tmp_path)
    assert len(npz_paths) == 2

    for path in npz_paths:
        assert path.exists()
        assert path.with_suffix(".json").exists()

    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    with manifest_path.open(encoding="utf-8") as fh:
        manifest = json.load(fh)

    assert manifest["label"] == "Synthetic data"
    assert manifest["sim_version"] == SIM_VERSION
    assert len(manifest["files"]) == 2
    assert manifest["config"]["n_bars"] == 2


def test_load_scan_round_trips(tmp_path: Path, tiny_cfg: DatasetConfig) -> None:
    npz_paths = generate_dataset(tiny_cfg, tmp_path)
    loaded = load_scan(npz_paths[0])

    assert isinstance(loaded.spec.reflectors, tuple)
    assert isinstance(loaded.spec.artefacts, tuple)
    assert loaded.ascans.dtype == np.float32

    stem = tmp_path / "custom_bar"
    npz_p, json_p = save_scan(loaded, stem)
    loaded2 = load_scan(npz_p)

    assert loaded2.spec == loaded.spec
    np.testing.assert_allclose(loaded2.ascans, loaded.ascans, atol=1e-3, rtol=1e-3)


def test_same_cfg_twice_gives_identical_truth(tmp_path: Path, tiny_cfg: DatasetConfig) -> None:
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"
    generate_dataset(tiny_cfg, dir1)
    generate_dataset(tiny_cfg, dir2)

    with (dir1 / "bar_0000.json").open() as f1, (dir2 / "bar_0000.json").open() as f2:
        assert json.load(f1) == json.load(f2)

    with (dir1 / "manifest.json").open() as f1, (dir2 / "manifest.json").open() as f2:
        assert json.load(f1) == json.load(f2)


@pytest.mark.parametrize("seed", [0, 1, 42, 99])
def test_reflector_depths_within_bounds(tiny_cfg: DatasetConfig, seed: int) -> None:
    cfg = DatasetConfig(
        n_bars=2,
        length_mm=6.0,
        n_angle=8,
        diameter_mm=20.0,
        mean_reflectors=4.0,
        seed=seed,
    )
    spec = random_bar_spec(cfg, seed)
    for ref in spec.reflectors:
        assert 3.0 <= ref.depth_mm <= cfg.diameter_mm - 5.0


@pytest.mark.parametrize("file_name", ["manifest.json", "bar_0000.json", "bar_0001.json"])
def test_label_is_synthetic_data(tmp_path: Path, tiny_cfg: DatasetConfig, file_name: str) -> None:
    generate_dataset(tiny_cfg, tmp_path)
    with (tmp_path / file_name).open(encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["label"] == "Synthetic data"
