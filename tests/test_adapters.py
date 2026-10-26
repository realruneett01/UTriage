"""Tests for file format adapters (NPZ, CSV, ML-NDT)."""

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


def test_adapters_registered() -> None:
    """ADAPTERS tuple contains expected adapter instances implementing ScanAdapter."""
    names = [a.name for a in ADAPTERS]
    assert names == ["npz", "csv", "mlndt"]
    assert any(isinstance(a, NpzAdapter) for a in ADAPTERS)
    assert any(isinstance(a, CsvAscanAdapter) for a in ADAPTERS)
    assert any(isinstance(a, MlNdtAdapter) for a in ADAPTERS)
    for adapter in ADAPTERS:
        assert isinstance(adapter, ScanAdapter)


def test_npz_round_trip(tmp_path: Path, tiny_scan: BarScan) -> None:
    """Save via dataset.save_scan and read back via read_scan."""
    stem = tmp_path / "tiny_bar"
    npz_path, _ = save_scan(tiny_scan, stem)
    loaded = read_scan(npz_path)

    assert loaded.ascans.shape == tiny_scan.ascans.shape
    assert loaded.axial_mm.shape == tiny_scan.axial_mm.shape
    assert loaded.angle_deg.shape == tiny_scan.angle_deg.shape
    assert loaded.missing.shape == tiny_scan.missing.shape
    np.testing.assert_allclose(loaded.ascans, tiny_scan.ascans, atol=1e-3)


def test_csv_round_trip(tmp_path: Path, tiny_scan: BarScan) -> None:
    """write_csv then read_scan round-trips ascans within 1e-6 and shapes."""
    csv_path = tmp_path / "tiny_bar.csv"
    returned_path = write_csv(tiny_scan, csv_path, material_key="ti64", probe_key="p5")
    assert returned_path == csv_path

    loaded = read_scan(csv_path)
    assert loaded.ascans.shape == tiny_scan.ascans.shape
    assert loaded.axial_mm.shape == tiny_scan.axial_mm.shape
    assert loaded.angle_deg.shape == tiny_scan.angle_deg.shape
    assert loaded.missing.shape == tiny_scan.missing.shape
    np.testing.assert_allclose(loaded.ascans, tiny_scan.ascans, atol=1e-6)
    assert not np.any(loaded.missing)


@pytest.mark.parametrize(
    "missing_key",
    ["fs_mhz", "t0_us", "material", "probe", "diameter_mm"],
)
def test_csv_missing_key_raises(tmp_path: Path, missing_key: str) -> None:
    """Missing required metadata key in CSV raises ValueError."""
    meta = {
        "fs_mhz": "50.0",
        "t0_us": "12.0",
        "material": "ti64",
        "probe": "p5",
        "diameter_mm": "20.0",
    }
    del meta[missing_key]

    csv_path = tmp_path / f"missing_{missing_key}.csv"
    content = "\n".join(f"# {k}={v}" for k, v in meta.items())
    content += "\naxial_mm,angle_deg,s0,s1\n0.0,0.0,0.1,0.2\n"
    csv_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required metadata"):
        read_scan(csv_path)


def test_csv_ragged_row_raises(tmp_path: Path) -> None:
    """CSV with ragged row raises ValueError with clear message."""
    content = (
        "# fs_mhz=50.0\n"
        "# t0_us=12.0\n"
        "# material=ti64\n"
        "# probe=p5\n"
        "# diameter_mm=20.0\n"
        "axial_mm,angle_deg,s0,s1\n"
        "0.0,0.0,0.1,0.2\n"
        "1.0,0.0,0.1\n"  # ragged row
    )
    csv_path = tmp_path / "ragged.csv"
    csv_path.write_text(content, encoding="utf-8")

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


def test_csv_missing_combination_sets_missing_and_nan(tmp_path: Path) -> None:
    """Omitted (axial, angle) pair results in zeros, missing=True, and NaN t_if_us."""
    content = (
        "# fs_mhz=50.0\n"
        "# t0_us=10.0\n"
        "# material=ti64\n"
        "# probe=p5\n"
        "# diameter_mm=20.0\n"
        "# axial_step_mm=1.0\n"
        "axial_mm,angle_deg,s0,s1,s2\n"
        "0.0,0.0,0.5,0.6,0.1\n"  # envelope >= 0.3
        "0.0,60.0,0.4,0.5,0.1\n"
        "1.0,0.0,0.5,0.5,0.2\n"
        # 1.0, 60.0 is missing
    )
    csv_path = tmp_path / "missing_coord.csv"
    csv_path.write_text(content, encoding="utf-8")

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
