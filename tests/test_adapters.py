"""Tests for file format adapters (NPZ, CSV, ML-NDT)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from utriage.io.adapters import (  # noqa: E402
    ADAPTERS,
    CsvAscanAdapter,
    MlNdtAdapter,
    NpzAdapter,
    ScanAdapter,
    read_scan,
    write_csv,
)
from utriage.physics.dataset import save_scan  # noqa: E402
from utriage.physics.simulator import (  # noqa: E402
    BarScan,
    BarSpec,
    load_material,
    load_probe,
    simulate_bar,
)


@dataclass(frozen=True)
class RoundTripCase:
    fmt: str
    atol: float
    check_missing: bool


@pytest.fixture
def tiny_scan() -> BarScan:
    """Simulate a tiny bar scan for adapter tests."""
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
def default_csv_meta() -> dict[str, str]:
    """Default metadata dictionary for CSV adapter testing."""
    return {
        "fs_mhz": "50.0",
        "t0_us": "12.0",
        "material": "ti64",
        "probe": "p5",
        "diameter_mm": "20.0",
    }


def _build_csv_content(meta: dict[str, str], rows: str) -> str:
    """Assemble commented metadata header and data rows."""
    header = "\n".join(f"# {k}={v}" for k, v in meta.items())
    return f"{header}\n{rows}"


def test_adapters_registered() -> None:
    """ADAPTERS tuple contains expected adapter instances implementing ScanAdapter."""
    names = [a.name for a in ADAPTERS]
    assert names == ["npz", "csv", "mlndt"]
    assert any(isinstance(a, NpzAdapter) for a in ADAPTERS)
    assert any(isinstance(a, CsvAscanAdapter) for a in ADAPTERS)
    assert any(isinstance(a, MlNdtAdapter) for a in ADAPTERS)
    for adapter in ADAPTERS:
        assert isinstance(adapter, ScanAdapter)


@pytest.mark.parametrize(
    "case",
    [
        RoundTripCase(fmt="npz", atol=1e-3, check_missing=False),
        RoundTripCase(fmt="csv", atol=1e-6, check_missing=True),
    ],
)
def test_scan_round_trip(tmp_path: Path, tiny_scan: BarScan, case: RoundTripCase) -> None:
    """Save via appropriate format writer and read back via read_scan."""
    if case.fmt == "npz":
        scan_path, _ = save_scan(tiny_scan, tmp_path / "tiny_bar")
    else:
        scan_path = tmp_path / "tiny_bar.csv"
        returned = write_csv(tiny_scan, scan_path, material_key="ti64", probe_key="p5")
        assert returned == scan_path

    loaded = read_scan(scan_path)
    assert loaded.ascans.shape == tiny_scan.ascans.shape
    assert loaded.axial_mm.shape == tiny_scan.axial_mm.shape
    assert loaded.angle_deg.shape == tiny_scan.angle_deg.shape
    assert loaded.missing.shape == tiny_scan.missing.shape
    np.testing.assert_allclose(loaded.ascans, tiny_scan.ascans, atol=case.atol)
    if case.check_missing:
        assert not np.any(loaded.missing)


@pytest.mark.parametrize(
    "missing_key",
    ["fs_mhz", "t0_us", "material", "probe", "diameter_mm"],
)
def test_csv_missing_key_raises(
    tmp_path: Path, missing_key: str, default_csv_meta: dict[str, str]
) -> None:
    """Missing required metadata key in CSV raises ValueError."""
    meta = dict(default_csv_meta)
    del meta[missing_key]

    csv_path = tmp_path / f"missing_{missing_key}.csv"
    content = _build_csv_content(meta, "axial_mm,angle_deg,s0,s1\n0.0,0.0,0.1,0.2\n")
    csv_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required metadata"):
        read_scan(csv_path)


def test_csv_ragged_row_raises(tmp_path: Path, default_csv_meta: dict[str, str]) -> None:
    """CSV with ragged row raises ValueError with clear message."""
    rows = "axial_mm,angle_deg,s0,s1\n0.0,0.0,0.1,0.2\n1.0,0.0,0.1\n"
    csv_path = tmp_path / "ragged.csv"
    csv_path.write_text(_build_csv_content(default_csv_meta, rows), encoding="utf-8")

    with pytest.raises(ValueError, match="Ragged row"):
        read_scan(csv_path)


def test_unknown_extension_raises(tmp_path: Path) -> None:
    """Unknown file extension raises ValueError listing supported formats."""
    dummy_path = tmp_path / "scan.unsupported"
    dummy_path.write_text("content", encoding="utf-8")

    with pytest.raises(ValueError, match="Supported formats: npz, csv, mlndt"):
        read_scan(dummy_path)


def test_mlndt_raises_not_implemented(tmp_path: Path) -> None:
    """MlNdtAdapter.read raises NotImplementedError with exact message."""
    bins_path = tmp_path / "weld.bins"
    bins_path.write_text("raw binary", encoding="utf-8")

    expected_msg = "ML-NDT images are not bar scans; use utriage.ml.datasets instead"
    with pytest.raises(NotImplementedError, match=expected_msg):
        read_scan(bins_path)

    adapter = MlNdtAdapter()
    assert adapter.can_read(bins_path)
    with pytest.raises(NotImplementedError, match=expected_msg):
        adapter.read(bins_path)


def test_csv_missing_combination_sets_missing_and_nan(
    tmp_path: Path, default_csv_meta: dict[str, str]
) -> None:
    """Omitted (axial, angle) pair results in zeros, missing=True, and NaN t_if_us."""
    meta = dict(default_csv_meta)
    meta["t0_us"] = "10.0"
    meta["axial_step_mm"] = "1.0"
    rows = (
        "axial_mm,angle_deg,s0,s1,s2\n"
        "0.0,0.0,0.5,0.6,0.1\n"
        "0.0,60.0,0.4,0.5,0.1\n"
        "1.0,0.0,0.5,0.5,0.2\n"
    )
    csv_path = tmp_path / "missing_coord.csv"
    csv_path.write_text(_build_csv_content(meta, rows), encoding="utf-8")

    loaded = read_scan(csv_path)
    assert loaded.ascans.shape == (2, 2, 3)
    assert loaded.missing.shape == (2, 2)
    assert not loaded.missing[0, 0]
    assert not loaded.missing[0, 1]
    assert not loaded.missing[1, 0]
    assert loaded.missing[1, 1]
    assert np.all(loaded.ascans[1, 1, :] == 0.0)
    assert np.isnan(loaded.t_if_us[1, 1])
    assert not np.isnan(loaded.t_if_us[0, 0])
