"""Validation for ultrasonic bar scans."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utriage.io.adapters import read_scan
from utriage.physics.simulator import BarScan

ERR_ASCANS_NOT_3D = "ASCANS_NOT_3D"
ERR_DTYPE_NOT_FLOATING = "DTYPE_NOT_FLOATING"
ERR_SAMPLES_NAN_OR_INF = "SAMPLES_NAN_OR_INF"
ERR_AXIAL_LENGTH_MISMATCH = "AXIAL_LENGTH_MISMATCH"
ERR_ANGLE_LENGTH_MISMATCH = "ANGLE_LENGTH_MISMATCH"
ERR_NON_POSITIVE_FS = "NON_POSITIVE_FS"
ERR_FEWER_THAN_32_SAMPLES = "FEWER_THAN_32_SAMPLES"
ERR_NON_POSITIVE_DIAMETER = "NON_POSITIVE_DIAMETER"

WARN_HIGH_ZERO_ASCANS = "HIGH_ZERO_ASCANS"
WARN_HIGH_CLIPPED_SAMPLES = "HIGH_CLIPPED_SAMPLES"
WARN_NON_UNIFORM_AXIAL_STEP = "NON_UNIFORM_AXIAL_STEP"
WARN_HIGH_NAN_INTERFACE_TIMES = "HIGH_NAN_INTERFACE_TIMES"


@dataclass(frozen=True)
class Issue:
    """A validation issue representing an error or a warning."""

    level: str
    code: str
    message: str


class ScanValidationError(ValueError):
    """Raised when BarScan validation fails with one or more errors."""

    def __init__(self, issues: list[Issue] | str) -> None:
        if isinstance(issues, list):
            self.issues: list[Issue] = issues
            msg = "; ".join(f"[{i.code}] {i.message}" for i in issues)
            super().__init__(msg)
        else:
            self.issues = []
            super().__init__(issues)


def _check_cube_dimensions(scan: BarScan) -> list[Issue]:
    if scan.ascans.ndim != 3:
        return [
            Issue(
                "error",
                ERR_ASCANS_NOT_3D,
                f"A-scans must be 3-D (axial, angle, time), got ndim={scan.ascans.ndim}",
            )
        ]
    issues: list[Issue] = []
    if scan.ascans.shape[2] < 32:
        issues.append(
            Issue(
                "error",
                ERR_FEWER_THAN_32_SAMPLES,
                f"A-scan has fewer than 32 samples ({scan.ascans.shape[2]})",
            )
        )
    return issues


def _check_cube_values(scan: BarScan) -> list[Issue]:
    issues: list[Issue] = []
    if not np.issubdtype(scan.ascans.dtype, np.floating):
        issues.append(
            Issue(
                "error",
                ERR_DTYPE_NOT_FLOATING,
                f"A-scans dtype must be floating point, got {scan.ascans.dtype}",
            )
        )
        return issues
    has_nan_or_inf = bool(np.isnan(scan.ascans).any() or np.isinf(scan.ascans).any())
    if has_nan_or_inf:
        issues.append(
            Issue("error", ERR_SAMPLES_NAN_OR_INF, "A-scans array contains NaN or Inf samples")
        )
    return issues


def _check_axis_lengths(scan: BarScan) -> list[Issue]:
    if scan.ascans.ndim != 3:
        return []
    issues: list[Issue] = []
    if len(scan.axial_mm) != scan.ascans.shape[0]:
        issues.append(
            Issue(
                "error",
                ERR_AXIAL_LENGTH_MISMATCH,
                f"axial_mm len {len(scan.axial_mm)} != cube {scan.ascans.shape[0]}",
            )
        )
    if len(scan.angle_deg) != scan.ascans.shape[1]:
        issues.append(
            Issue(
                "error",
                ERR_ANGLE_LENGTH_MISMATCH,
                f"angle_deg len {len(scan.angle_deg)} != cube {scan.ascans.shape[1]}",
            )
        )
    return issues


def _check_spec_errors(scan: BarScan) -> list[Issue]:
    issues: list[Issue] = []
    if scan.fs_mhz <= 0:
        issues.append(
            Issue("error", ERR_NON_POSITIVE_FS, f"fs_mhz must be positive, got {scan.fs_mhz}")
        )
    if scan.spec.diameter_mm <= 0:
        issues.append(
            Issue(
                "error",
                ERR_NON_POSITIVE_DIAMETER,
                f"diameter_mm must be positive, got {scan.spec.diameter_mm}",
            )
        )
    return issues


def _check_ascan_warnings(scan: BarScan) -> list[Issue]:
    if scan.ascans.ndim != 3 or scan.ascans.size == 0:
        return []
    issues: list[Issue] = []
    zero_frac = float(np.mean(np.all(scan.ascans == 0.0, axis=-1)))
    if zero_frac > 0.05:
        issues.append(
            Issue(
                "warning",
                WARN_HIGH_ZERO_ASCANS,
                f"Fraction of all-zero A-scans ({zero_frac:.2%}) exceeds 5%",
            )
        )
    with np.errstate(invalid="ignore"):
        clipped_frac = float(np.mean(np.abs(scan.ascans) >= 0.999))
    if clipped_frac > 0.01:
        issues.append(
            Issue(
                "warning",
                WARN_HIGH_CLIPPED_SAMPLES,
                f"Fraction of clipped samples ({clipped_frac:.2%}) exceeds 1%",
            )
        )
    return issues


def _check_geometry_warnings(scan: BarScan) -> list[Issue]:
    issues: list[Issue] = []
    if len(scan.axial_mm) > 1:
        diffs = np.diff(scan.axial_mm)
        if not np.allclose(diffs, diffs[0], rtol=1e-3, atol=1e-3):
            issues.append(
                Issue("warning", WARN_NON_UNIFORM_AXIAL_STEP, "Axial step is non-uniform")
            )
    if scan.t_if_us.size > 0:
        nan_frac = float(np.mean(np.isnan(scan.t_if_us)))
        if nan_frac > 0.05:
            issues.append(
                Issue(
                    "warning",
                    WARN_HIGH_NAN_INTERFACE_TIMES,
                    f"Fraction of NaN interface times ({nan_frac:.2%}) exceeds 5%",
                )
            )
    return issues


def validate_scan(scan: BarScan) -> list[Issue]:
    """Validate a BarScan instance for shape, dtype, values, and signal quality."""
    return [
        *_check_cube_dimensions(scan),
        *_check_cube_values(scan),
        *_check_axis_lengths(scan),
        *_check_spec_errors(scan),
        *_check_ascan_warnings(scan),
        *_check_geometry_warnings(scan),
    ]


def load_validated(path: Path) -> BarScan:
    """Read scan from file, validate it, log warnings, and return BarScan or raise."""
    scan = read_scan(Path(path))
    issues = validate_scan(scan)
    errors = [i for i in issues if i.level == "error"]
    warnings = [i for i in issues if i.level == "warning"]

    logger = logging.getLogger("utriage")
    for w in warnings:
        logger.warning("[%s] %s", w.code, w.message)

    if errors:
        raise ScanValidationError(errors)

    return scan


_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)

MAX_SCAN_BYTES = 512 * 1024 * 1024  # operational limit, adjustable


def safe_filename(name: str) -> str:
    """Sanitize string for use as a safe, cross-platform filename component."""
    norm = str(name).replace("\\", "/").rstrip("/")
    base = norm.rsplit("/", 1)[-1]
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", base).lstrip(".")[:100]
    if not cleaned:
        raise ValueError("Filename is empty or contains only invalid characters")
    stem = cleaned.split(".")[0].upper()
    if stem in _RESERVED_NAMES:
        raise ValueError(f"Reserved Windows filename: {cleaned}")
    return cleaned


def check_file_size(path: Path) -> None:
    """Ensure file size does not exceed MAX_SCAN_BYTES operational limit."""
    p = Path(path)
    size = p.stat().st_size
    if size > MAX_SCAN_BYTES:
        raise ValueError(
            f"Scan file {p.name} ({size} bytes) exceeds limit of {MAX_SCAN_BYTES} bytes"
        )
