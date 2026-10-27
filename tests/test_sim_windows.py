"""Tests for simulation held-out parameter-region generalization (Synthetic data)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Prioritize local workspace src/
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from utriage.ml.bar_data import WindowConfig  # noqa: E402
from utriage.ml.sim_windows import (  # noqa: E402
    SimGenSettings,
    build_artefact_stress,
    build_region_test,
    build_train,
    sample_artefact_params,
    sample_region_params,
    sample_train_params,
    windows_from_params,
)
from utriage.physics.randomize import DEFAULT_HOLDOUT_REGIONS, Region  # noqa: E402
from utriage.physics.simulator import ARTEFACT_KINDS  # noqa: E402


class StubWindowAdapter(nn.Module):
    """CPU-safe stub model returning synthetic flaw logit and size."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        logit = torch.zeros(batch_size, dtype=torch.float32)
        size = torch.full((batch_size,), 1.0, dtype=torch.float32)
        return logit, size


@pytest.fixture
def tiny_settings() -> SimGenSettings:
    """Fixture providing minimal settings for fast testing."""
    return SimGenSettings(n_train_bars=2, n_test_bars=1, seed=42, epochs=1)


def test_training_params_never_inside_heldout_region() -> None:
    """Sampled training params are never inside any default held-out region."""
    settings = SimGenSettings(n_train_bars=50, n_test_bars=1, seed=123, epochs=1)
    params = sample_train_params(settings, DEFAULT_HOLDOUT_REGIONS)
    assert len(params) == 50
    for p in params:
        for reg in DEFAULT_HOLDOUT_REGIONS:
            assert not reg.matches(p), f"Param {p} matched excluded region {reg.name}"


@pytest.mark.parametrize("region", DEFAULT_HOLDOUT_REGIONS, ids=lambda r: r.name)
def test_region_test_params_always_inside(region: Region) -> None:
    """Sampled region test params are strictly inside target held-out region."""
    settings = SimGenSettings(n_train_bars=1, n_test_bars=10, seed=456, epochs=1)
    params = sample_region_params(settings, region)
    assert len(params) == 10
    for p in params:
        assert region.matches(p), f"Param {p} outside expected region {region.name}"


@pytest.mark.parametrize("kind", ARTEFACT_KINDS)
def test_artefact_stress_params_and_windows_contain_artefact(
    kind: str,
    tiny_settings: SimGenSettings,
) -> None:
    """Artefact stress parameters and sets contain forced artefact."""
    params = sample_artefact_params(tiny_settings, kind)
    assert len(params) == tiny_settings.n_test_bars
    assert all(p.get("forced_artefact") == kind for p in params)

    stress_dict = build_artefact_stress(tiny_settings, kinds=(kind,))
    wset = stress_dict[kind]
    assert 2 in wset.y, f"Expected artefact windows (y=2) in stress set for {kind}"


def test_windowset_shapes(tiny_settings: SimGenSettings) -> None:
    """WindowSet built from params has expected 1D shape and consistent fields."""
    params = sample_train_params(tiny_settings, DEFAULT_HOLDOUT_REGIONS)
    wcfg = WindowConfig(patch=False, seed=tiny_settings.seed)
    rng = np.random.default_rng(tiny_settings.seed)
    wset = windows_from_params(params, wcfg, rng)

    n_samples = len(wset.x)
    assert n_samples > 0
    assert wset.x.shape == (n_samples, 1, 128)
    assert wset.y.shape == (n_samples,)
    assert wset.size_mm.shape == (n_samples,)
    assert wset.depth_mm.shape == (n_samples,)
    assert wset.noise_scale.shape == (n_samples,)
    assert 0 in wset.y  # Contains noise windows
    assert 1 in wset.y  # Contains flaw windows


def _check_cell_structure(cell_dict: dict[str, Any]) -> None:
    """Verify standard pod.metrics cell keys."""
    expected = ("detection", "false_calls", "ece", "reliability", "pod", "counts")
    for key in expected:
        assert key in cell_dict, f"Missing key '{key}' in metric cell"


def _assert_json_payload_valid(data: dict[str, Any]) -> None:
    """Validate top-level and method cell integrity."""
    assert data["label"] == "Synthetic data"
    assert "protocol" in data and len(data["protocol"]) > 0
    assert "regions" in data and len(data["regions"]) == len(DEFAULT_HOLDOUT_REGIONS)

    for test_key in ("in_distribution", "freq_10mhz", "artefact_bubble"):
        assert test_key in data
        for method in ("rules", "ml_1d", "union"):
            assert method in data[test_key]
            _check_cell_structure(data[test_key][method])


def test_json_writer_tiny_run_with_stub_model(
    tmp_path: Path,
    tiny_settings: SimGenSettings,
) -> None:
    """JSON output contains synthetic label, protocol, regions, and metric cells."""
    from experiments.run_sim_generalization import (
        EvalRunInput,
        EvalRunOutput,
        evaluate_and_write_json,
    )

    out_file = tmp_path / "sim_generalization.json"
    model_file = tmp_path / "simgen_1d.pt"
    train_ws = build_train(tiny_settings)
    test_ws = build_region_test(tiny_settings, DEFAULT_HOLDOUT_REGIONS[0])
    artefact_sets = build_artefact_stress(tiny_settings, kinds=("bubble",))

    inp = EvalRunInput(train_ws, test_ws, {"freq_10mhz": test_ws}, artefact_sets)
    out = EvalRunOutput(out_file, model_file, torch.device("cpu"))
    evaluate_and_write_json(StubWindowAdapter(), inp, out)

    assert out_file.exists()
    assert model_file.exists()
    with out_file.open(encoding="utf-8") as fh:
        data = json.load(fh)
    _assert_json_payload_valid(data)
