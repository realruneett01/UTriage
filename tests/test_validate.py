"""Tests for scan validation and logging configuration."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from utriage.io.validate import (  # noqa: E402
    ERR_ANGLE_LENGTH_MISMATCH,
    ERR_ASCANS_NOT_3D,
    ERR_AXIAL_LENGTH_MISMATCH,
    ERR_DTYPE_NOT_FLOATING,
    ERR_FEWER_THAN_32_SAMPLES,
    ERR_NON_POSITIVE_DIAMETER,
    ERR_NON_POSITIVE_FS,
    ERR_SAMPLES_NAN_OR_INF,
    WARN_HIGH_CLIPPED_SAMPLES,
    WARN_HIGH_NAN_INTERFACE_TIMES,
    WARN_HIGH_ZERO_ASCANS,
    WARN_NON_UNIFORM_AXIAL_STEP,
    ScanValidationError,
    load_validated,
    validate_scan,
)
from utriage.logging_setup import configure_logging  # noqa: E402
from utriage.physics.dataset import save_scan  # noqa: E402
from utriage.physics.simulator import (  # noqa: E402
    BarScan,
    BarSpec,
    load_material,
    load_probe,
    simulate_bar,
)


@pytest.fixture
def tiny_scan() -> BarScan:
    """Create a tiny valid simulated bar scan."""
    spec = BarSpec(
        material=load_material("ti64"),
        probe=load_probe("p5"),
        water=load_material("water"),
        diameter_mm=20.0,
        length_mm=4.0,
        axial_step_mm=1.0,
        n_angle=6,
        seed=42,
    )
    return simulate_bar(spec)


@pytest.fixture
def saved_scan_npz(tiny_scan: BarScan, tmp_path: Path) -> Path:
    """Save a tiny valid scan to disk and return the npz file path."""
    npz_path, _ = save_scan(tiny_scan, tmp_path / "valid")
    return npz_path


def _mutate_ascan_val(scan: BarScan, val: float) -> BarScan:
    ascans = scan.ascans.copy()
    ascans[0, 0, 0] = val
    return replace(scan, ascans=ascans)


def _mutate_all_zeros(scan: BarScan) -> BarScan:
    ascans = scan.ascans.copy()
    ascans[0, 0, :] = 0.0
    ascans[0, 1, :] = 0.0
    return replace(scan, ascans=ascans)


def _mutate_clipped(scan: BarScan) -> BarScan:
    ascans = scan.ascans.copy()
    ascans.flat[: max(5, int(ascans.size * 0.05))] = 1.0
    return replace(scan, ascans=ascans)


def _mutate_nan_tif(scan: BarScan) -> BarScan:
    t_if = scan.t_if_us.copy()
    t_if.flat[: max(2, int(t_if.size * 0.1))] = np.nan
    return replace(scan, t_if_us=t_if)


def _assert_issue_code(issues: list, level: str, expected_code: str) -> None:
    """Assert that expected issue code is present in issues for given level."""
    codes = [i.code for i in issues if i.level == level]
    assert expected_code in codes


def _read_json_lines(path: Path) -> list[dict]:
    """Read non-empty lines from file and parse each as JSON."""
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line.strip()) for line in fh if line.strip()]


def test_valid_scan_passes(tiny_scan: BarScan) -> None:
    """A valid tiny simulated scan has no errors."""
    issues = validate_scan(tiny_scan)
    errors = [i for i in issues if i.level == "error"]
    assert errors == []


@pytest.mark.parametrize(
    ("mutator", "expected_code"),
    [
        (lambda s: replace(s, ascans=s.ascans[0]), ERR_ASCANS_NOT_3D),
        (lambda s: replace(s, ascans=s.ascans.astype(np.int32)), ERR_DTYPE_NOT_FLOATING),
        (lambda s: _mutate_ascan_val(s, np.nan), ERR_SAMPLES_NAN_OR_INF),
        (lambda s: _mutate_ascan_val(s, np.inf), ERR_SAMPLES_NAN_OR_INF),
        (lambda s: replace(s, axial_mm=s.axial_mm[:-1]), ERR_AXIAL_LENGTH_MISMATCH),
        (lambda s: replace(s, angle_deg=s.angle_deg[:-1]), ERR_ANGLE_LENGTH_MISMATCH),
        (lambda s: replace(s, fs_mhz=0.0), ERR_NON_POSITIVE_FS),
        (lambda s: replace(s, ascans=s.ascans[:, :, :16]), ERR_FEWER_THAN_32_SAMPLES),
        (lambda s: replace(s, spec=replace(s.spec, diameter_mm=0.0)), ERR_NON_POSITIVE_DIAMETER),
    ],
)
def test_validation_errors(
    tiny_scan: BarScan,
    mutator: Callable[[BarScan], BarScan],
    expected_code: str,
) -> None:
    """Each error case built by mutating a copy is reported with its code."""
    bad_scan = mutator(tiny_scan)
    issues = validate_scan(bad_scan)
    _assert_issue_code(issues, "error", expected_code)


@pytest.mark.parametrize(
    ("mutator", "expected_code"),
    [
        (_mutate_all_zeros, WARN_HIGH_ZERO_ASCANS),
        (_mutate_clipped, WARN_HIGH_CLIPPED_SAMPLES),
        (
            lambda s: replace(s, axial_mm=np.array([0.0, 1.0, 3.5, 4.0])),
            WARN_NON_UNIFORM_AXIAL_STEP,
        ),
        (_mutate_nan_tif, WARN_HIGH_NAN_INTERFACE_TIMES),
    ],
)
def test_validation_warnings(
    tiny_scan: BarScan,
    mutator: Callable[[BarScan], BarScan],
    expected_code: str,
) -> None:
    """Each warning condition is correctly detected and reported."""
    warn_scan = mutator(tiny_scan)
    issues = validate_scan(warn_scan)
    _assert_issue_code(issues, "warning", expected_code)


def test_load_validated_corrupt_scan(saved_scan_npz: Path) -> None:
    """load_validated raises ScanValidationError on corrupt npz written to tmp_path."""
    with np.load(saved_scan_npz) as data:
        arrays = dict(data)
    arrays["fs_mhz"] = np.array(-5.0)
    np.savez_compressed(saved_scan_npz, **arrays)

    with pytest.raises(ScanValidationError) as exc_info:
        load_validated(saved_scan_npz)
    assert issubclass(ScanValidationError, ValueError)
    _assert_issue_code(exc_info.value.issues, "error", ERR_NON_POSITIVE_FS)


def test_load_validated_valid_scan(saved_scan_npz: Path, tiny_scan: BarScan) -> None:
    """load_validated successfully returns valid BarScan from disk."""
    loaded = load_validated(saved_scan_npz)
    assert isinstance(loaded, BarScan)
    assert loaded.ascans.shape == tiny_scan.ascans.shape


def test_configure_logging(tmp_path: Path) -> None:
    """configure_logging writes JSON lines to tmp_path and is idempotent."""
    logger = configure_logging(log_dir=tmp_path, level="INFO")
    logger.info("First event", extra={"scan_id": "test_scan", "step": 1})

    log_file = tmp_path / "utriage.log"
    assert log_file.exists()

    records = _read_json_lines(log_file)
    assert len(records) == 1
    rec1 = records[0]
    assert rec1["level"] == "INFO"
    assert rec1["name"] == "utriage"
    assert rec1["message"] == "First event"
    assert rec1["scan_id"] == "test_scan"
    assert rec1["step"] == 1
    assert "ts" in rec1

    # Test idempotence: calling again retains 2 handlers and does not duplicate lines
    logger2 = configure_logging(log_dir=tmp_path, level="INFO")
    assert len(logger2.handlers) == 2
    logger2.warning("Second event")

    records2 = _read_json_lines(log_file)
    assert len(records2) == 2
    assert records2[1]["level"] == "WARNING"
    assert records2[1]["message"] == "Second event"
