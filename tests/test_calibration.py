"""Tests for ASTM E127-style flat calibration block simulation and DAC measurement."""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure local worktree src is prioritized over any site-packages editable install
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from utriage.physics.calibration import (  # noqa: E402
    CalibrationBlock,
    DacPoint,
    dac_curve,
    measure_dac,
    simulate_calibration_block,
)
from utriage.physics.simulator import fbh_amplitude, load_material, load_probe  # noqa: E402


@dataclass(frozen=True)
class DacInterpolationCase:
    z1: float
    a1: float
    z2: float
    a2: float
    eval_z: float
    expected: float


def _simulate_and_measure(block: CalibrationBlock) -> list[DacPoint]:
    """Helper to simulate calibration block and measure DAC points."""
    time_us, ascans, t_if_us = simulate_calibration_block(block)
    return measure_dac(time_us, ascans, t_if_us, block)


@pytest.fixture
def default_block() -> CalibrationBlock:
    """Default calibration block specification."""
    return CalibrationBlock(
        material=load_material("ti64"),
        probe=load_probe("p5"),
        water=load_material("water"),
    )


@pytest.fixture
def clean_block(default_block: CalibrationBlock) -> CalibrationBlock:
    """Calibration block with noise disabled for theoretical checks."""
    ti64 = dataclasses.replace(default_block.material, grain_noise=0.0)
    p5 = dataclasses.replace(default_block.probe, electronic_noise=0.0)
    return dataclasses.replace(default_block, material=ti64, probe=p5)


def test_dac_amplitudes_match_fbh_theory(clean_block: CalibrationBlock) -> None:
    """With zero noise, measured DAC amplitudes match fbh_amplitude within 3%."""
    points = _simulate_and_measure(clean_block)
    for pt in points:
        expected = abs(fbh_amplitude(clean_block.fbh_diameter_mm, pt.depth_mm, clean_block))
        assert np.isclose(pt.amplitude, expected, rtol=0.03)


def test_zero_attenuation_focal_peak(clean_block: CalibrationBlock) -> None:
    """With zero attenuation, the DAC point nearest the focal depth is maximal."""
    lossless_mat = dataclasses.replace(clean_block.material, attenuation_db_mm_mhz=0.0)
    block = dataclasses.replace(clean_block, material=lossless_mat)
    points = _simulate_and_measure(block)

    focal_depth = block.probe.focal_depth_mm
    nearest_pt = min(points, key=lambda p: abs(p.depth_mm - focal_depth))
    max_pt = max(points, key=lambda p: p.amplitude)
    assert max_pt.depth_mm == nearest_pt.depth_mm


@pytest.mark.parametrize(
    "case",
    [
        DacInterpolationCase(10.0, 0.4, 20.0, 0.8, 15.0, 0.6),
        DacInterpolationCase(5.0, 0.2, 15.0, 1.0, 10.0, 0.6),
        DacInterpolationCase(0.0, 0.0, 10.0, 1.0, 2.5, 0.25),
    ],
)
def test_dac_curve_interpolates_midpoints(case: DacInterpolationCase) -> None:
    """Linear DAC interpolation produces exact midpoints."""
    points = [
        DacPoint(depth_mm=case.z1, amplitude=case.a1),
        DacPoint(depth_mm=case.z2, amplitude=case.a2),
    ]
    curve = dac_curve(points)
    result = curve(np.array([case.eval_z]))
    assert np.isclose(result[0], case.expected)


def test_dac_curve_clamped_ends() -> None:
    """DAC curve clamps to boundary amplitudes outside depth range."""
    points = [DacPoint(depth_mm=10.0, amplitude=0.5), DacPoint(depth_mm=20.0, amplitude=0.9)]
    curve = dac_curve(points)
    assert np.isclose(curve(np.array([5.0]))[0], 0.5)
    assert np.isclose(curve(np.array([25.0]))[0], 0.9)


def test_dac_curve_empty_points_raises() -> None:
    """Constructing a DAC curve from an empty list raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        dac_curve([])


def test_simulation_determinism_and_shapes(default_block: CalibrationBlock) -> None:
    """Simulation is deterministic with seed and returns proper shapes and types."""
    block = dataclasses.replace(default_block, seed=123)

    t1, a1, if1 = simulate_calibration_block(block)
    t2, a2, if2 = simulate_calibration_block(block)

    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(a1, a2)
    assert if1 == if2
    assert a1.dtype == np.float32
    assert a1.shape == (len(block.metal_paths_mm), len(t1))

    diff_block = dataclasses.replace(block, seed=999)
    _, a_diff, _ = simulate_calibration_block(diff_block)
    assert not np.array_equal(a1, a_diff)
