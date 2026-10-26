"""Typed loader for ``configs/acceptance.yaml``. Every limit in that file is a PLACEHOLDER."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

DEFAULT_ACCEPTANCE = Path(__file__).resolve().parents[3] / "configs" / "acceptance.yaml"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Gates(_Frozen):
    """Gate set-up relative to the interface echo."""

    interface_threshold: float
    front_dead_zone_mm: float
    back_margin_mm: float


class Amplitude(_Frozen):
    """Recording and rejection levels in % of reference (DAC)."""

    record_pct: float
    reject_pct: float


class Multiple(_Frozen):
    """Limit on several indications close together."""

    window_mm: float
    max_count: int


class Linear(_Frozen):
    """Limit on long (linear) indications."""

    max_length_mm: float


class Snr(_Frozen):
    """Local SNR measurement windows and the borderline threshold."""

    noise_window_mm: float
    exclude_mm: float
    min_snr_db: float


class Quality(_Frozen):
    """Data-quality watchdog limits."""

    waterpath_tolerance_mm: float
    backwall_drop_ratio: float
    noise_limit_ratio: float
    coverage_floor: float
    saturation_level: float
    early_echo_threshold: float


class AcceptanceConfig(_Frozen):
    """All rules-engine settings. ``status`` says whether the limits are verified."""

    status: str
    source: str
    gates: Gates
    amplitude: Amplitude
    multiple: Multiple
    linear: Linear
    snr: Snr
    quality: Quality

    @property
    def is_placeholder(self) -> bool:
        """True while the limits have not been checked against the real standard."""
        return "PLACEHOLDER" in self.status.upper()


def load_acceptance(path: Path | None = None) -> AcceptanceConfig:
    """Load and validate the acceptance configuration."""
    with (path or DEFAULT_ACCEPTANCE).open(encoding="utf-8") as fh:
        return AcceptanceConfig.model_validate(yaml.safe_load(fh))
