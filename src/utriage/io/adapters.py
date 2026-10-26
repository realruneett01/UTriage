"""File format adapters for reading and writing ultrasonic bar scans.

Read-only adapters for external formats; nothing ever connects to a machine or network.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from utriage.physics.dataset import load_scan
from utriage.physics.simulator import (
    BarScan,
    BarSpec,
    envelope,
    load_material,
    load_probe,
)

REQUIRED_METADATA_KEYS = ("fs_mhz", "t0_us", "material", "probe", "diameter_mm")


@runtime_checkable
class ScanAdapter(Protocol):
    """Protocol for UT scan file adapters."""

    name: str

    def can_read(self, path: Path) -> bool:
        """Check whether this adapter can read the given file path."""
        ...

    def read(self, path: Path) -> BarScan:
        """Read the scan file and return a BarScan instance."""
        ...


class NpzAdapter:
    """Adapter for reading native UTriage NPZ bar scan files."""

    name: str = "npz"

    def can_read(self, path: Path) -> bool:
        """Return True if path has .npz extension."""
        return Path(path).suffix.lower() == ".npz"

    def read(self, path: Path) -> BarScan:
        """Read .npz and sibling .json via load_scan."""
        return load_scan(Path(path))


def _parse_comment_line(line: str, meta: dict[str, str]) -> None:
    content = line.lstrip("#").strip()
    if "=" in content:
        key, val = content.split("=", 1)
        meta[key.strip()] = val.strip()


def _parse_metadata(lines: list[str]) -> tuple[dict[str, str], int]:
    meta: dict[str, str] = {}
    header_idx = -1
    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            _parse_comment_line(line, meta)
            continue
        header_idx = idx
        break

    if header_idx < 0:
        raise ValueError("CSV contains no header row or data")

    missing = [k for k in REQUIRED_METADATA_KEYS if k not in meta]
    if missing:
        raise ValueError(f"Missing required metadata keys in CSV: {missing}")
    return meta, header_idx


def _parse_header(line: str) -> int:
    parts = [col.strip() for col in line.split(",")]
    if len(parts) < 3:
        raise ValueError("Header row must have at least 3 columns (axial_mm, angle_deg, s0, ...)")
    if parts[0] != "axial_mm" or parts[1] != "angle_deg":
        raise ValueError(
            f"Header row must begin with 'axial_mm,angle_deg', got {parts[0]},{parts[1]}"
        )
    return len(parts)


def _parse_data_row(
    line: str, expected_cols: int, line_no: int
) -> tuple[float, float, list[float]]:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != expected_cols:
        raise ValueError(
            f"Ragged row at line {line_no}: expected {expected_cols} columns, got {len(parts)}"
        )
    try:
        return float(parts[0]), float(parts[1]), [float(p) for p in parts[2:]]
    except ValueError as err:
        raise ValueError(f"Invalid numeric value in row at line {line_no}") from err


def _read_rows(
    lines: list[str], header_idx: int, expected_cols: int
) -> tuple[list[float], list[float], list[list[float]]]:
    axials: list[float] = []
    angles: list[float] = []
    ascans: list[list[float]] = []
    for offset, raw_line in enumerate(lines[header_idx + 1 :]):
        line = raw_line.strip()
        if not line:
            continue
        line_no = header_idx + offset + 2
        ax, an, samples = _parse_data_row(line, expected_cols, line_no)
        axials.append(ax)
        angles.append(an)
        ascans.append(samples)
    if not axials:
        raise ValueError("CSV contains no A-scan data rows")
    return axials, angles, ascans


def _assemble_cube(
    axials: list[float],
    angles: list[float],
    ascans: list[list[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ax_unique = np.unique(axials)
    an_unique = np.unique(angles)
    n_ax, n_an, n_t = len(ax_unique), len(an_unique), len(ascans[0])
    cube = np.zeros((n_ax, n_an, n_t), dtype=np.float32)
    missing = np.ones((n_ax, n_an), dtype=bool)

    ax_idx = {val: idx for idx, val in enumerate(ax_unique)}
    an_idx = {val: idx for idx, val in enumerate(an_unique)}
    for ax, an, ascan in zip(axials, angles, ascans, strict=False):
        i = ax_idx[ax]
        j = an_idx[an]
        cube[i, j, :] = ascan
        missing[i, j] = False
    return cube, ax_unique, an_unique, missing


def _compute_t_if_us(cube: np.ndarray, t0_us: float, fs_mhz: float) -> np.ndarray:
    env = envelope(cube)
    reached = env >= 0.3
    any_reached = np.any(reached, axis=-1)
    first_indices = np.argmax(reached, axis=-1)
    sample_times = t0_us + first_indices / fs_mhz
    return np.where(any_reached, sample_times, np.nan)


def _build_spec(meta: dict[str, str], n_angle: int, n_axial: int) -> BarSpec:
    axial_step = float(meta.get("axial_step_mm", 1.0))
    length = float(meta.get("length_mm", n_axial * axial_step))
    return BarSpec(
        material=load_material(meta["material"]),
        probe=load_probe(meta["probe"]),
        water=load_material("water"),
        diameter_mm=float(meta["diameter_mm"]),
        length_mm=length,
        axial_step_mm=axial_step,
        n_angle=n_angle,
        reflectors=(),
        artefacts=(),
        seed=0,
    )


class CsvAscanAdapter:
    """Adapter for CSV A-scan exports.

    Format:
    - Leading comment lines '# key=value' with required keys:
      fs_mhz, t0_us, material, probe, diameter_mm (optional axial_step_mm default 1.0).
    - Header row: axial_mm,angle_deg,s0,s1,...
    - One row per A-scan.
    Builds the (n_axial, n_angle, n_t) cube from unique sorted axial and angle values
    (missing combinations -> zeros and missing=True), spec from load_material/load_probe/water
    with no reflectors, t_if_us from the first sample where the envelope reaches 0.3 (NaN if none).
    Raises ValueError with a clear message on missing keys or ragged rows.
    """

    name: str = "csv"

    def can_read(self, path: Path) -> bool:
        """Return True if path has .csv extension."""
        return Path(path).suffix.lower() == ".csv"

    def read(self, path: Path) -> BarScan:
        """Read CSV A-scan export file into BarScan."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()

        meta, header_idx = _parse_metadata(lines)
        num_cols = _parse_header(lines[header_idx].strip())
        axials, angles, ascans = _read_rows(lines, header_idx, num_cols)
        cube, ax_unique, an_unique, missing = _assemble_cube(axials, angles, ascans)
        t0_us = float(meta["t0_us"])
        fs_mhz = float(meta["fs_mhz"])
        t_if_us = _compute_t_if_us(cube, t0_us, fs_mhz)
        spec = _build_spec(meta, len(an_unique), len(ax_unique))
        return BarScan(cube, t0_us, fs_mhz, ax_unique, an_unique, t_if_us, spec, missing)


class MlNdtAdapter:
    """Adapter placeholder for ML-NDT dataset files.

    Exists in the registry to explain the ML-NDT format.
    """

    name: str = "mlndt"

    def can_read(self, path: Path) -> bool:
        """Return True if path has .bins extension."""
        return Path(path).suffix.lower() == ".bins"

    def read(self, path: Path) -> BarScan:
        """Raise NotImplementedError explaining that ML-NDT files are not bar scans."""
        raise NotImplementedError(
            "ML-NDT images are not bar scans; use utriage.ml.datasets instead"
        )


ADAPTERS: tuple[ScanAdapter, ...] = (
    NpzAdapter(),
    CsvAscanAdapter(),
    MlNdtAdapter(),
)


def read_scan(path: Path) -> BarScan:
    """Read a scan file using the first matching adapter in ADAPTERS."""
    path = Path(path)
    for adapter in ADAPTERS:
        if adapter.can_read(path):
            return adapter.read(path)
    formats = ", ".join(a.name for a in ADAPTERS)
    raise ValueError(f"Unsupported file format for '{path.name}'. Supported formats: {formats}")


def _iter_csv_rows(scan: BarScan) -> Iterator[str]:
    n_ax = len(scan.axial_mm)
    n_an = len(scan.angle_deg)
    for i, j in itertools.product(range(n_ax), range(n_an)):
        if scan.missing.size > 0 and scan.missing[i, j]:
            continue
        ax, an = scan.axial_mm[i], scan.angle_deg[j]
        samples = ",".join(f"{float(v):.8g}" for v in scan.ascans[i, j])
        yield f"{float(ax):.8g},{float(an):.8g},{samples}\n"


def write_csv(scan: BarScan, path: Path, material_key: str, probe_key: str) -> Path:
    """Write BarScan A-scans to CSV format with metadata comments."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_t = scan.ascans.shape[-1]
    header = ",".join(["axial_mm", "angle_deg", *(f"s{k}" for k in range(n_t))])

    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# fs_mhz={scan.fs_mhz}\n")
        fh.write(f"# t0_us={scan.t0_us}\n")
        fh.write(f"# material={material_key}\n")
        fh.write(f"# probe={probe_key}\n")
        fh.write(f"# diameter_mm={scan.spec.diameter_mm}\n")
        fh.write(f"# axial_step_mm={scan.spec.axial_step_mm}\n")
        fh.write(f"# length_mm={scan.spec.length_mm}\n")
        fh.write(f"{header}\n")
        for row in _iter_csv_rows(scan):
            fh.write(row)
    return path
