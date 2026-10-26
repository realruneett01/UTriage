"""Tests for WindowConfig artefact_kinds filtering and robustness (Synthetic data)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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
    WindowConfig,
    _random_artefact,
    _simulate_training_bar,
    make_training_set,
)
from utriage.physics.simulator import ARTEFACT_KINDS  # noqa: E402


def test_default_window_config_artefact_kinds() -> None:
    """Default WindowConfig artefact_kinds equals ARTEFACT_KINDS."""
    cfg = WindowConfig()
    assert cfg.artefact_kinds == ARTEFACT_KINDS


def test_config_excluding_roughness_never_produces_roughness() -> None:
    """A config excluding roughness never produces roughness artefacts."""
    allowed = tuple(k for k in ARTEFACT_KINDS if k != "roughness")
    cfg = WindowConfig(artefact_kinds=allowed, seed=42)
    rng = np.random.default_rng(42)
    observed_kinds: set[str] = set()
    for _ in range(50):
        _, _, _, artefacts = _simulate_training_bar(cfg, rng)
        for art in artefacts:
            observed_kinds.add(art.kind)
    assert "roughness" not in observed_kinds
    assert observed_kinds.issubset(set(allowed))


def test_random_artefact_respects_custom_kinds() -> None:
    """_random_artefact draws only from provided kinds."""
    rng = np.random.default_rng(123)
    kinds = ("bubble", "lost_backwall")
    samples = {_random_artefact(rng, kinds)[0].kind for _ in range(30)}
    assert samples.issubset(set(kinds))


@pytest.mark.parametrize("seed", [0, 42, 99])
def test_same_seed_same_output(seed: int) -> None:
    """WindowSet outputs are identical for the same seed."""
    cfg1 = WindowConfig(seed=seed)
    cfg2 = WindowConfig(seed=seed, artefact_kinds=ARTEFACT_KINDS)
    set1 = make_training_set(cfg1, n_bars=2, per_bar=6)
    set2 = make_training_set(cfg2, n_bars=2, per_bar=6)
    np.testing.assert_array_equal(set1.x, set2.x)
    np.testing.assert_array_equal(set1.y, set2.y)
    np.testing.assert_array_equal(set1.size_mm, set2.size_mm)
    np.testing.assert_array_equal(set1.depth_mm, set2.depth_mm)
    np.testing.assert_array_equal(set1.noise_scale, set2.noise_scale)
