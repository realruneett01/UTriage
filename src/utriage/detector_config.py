"""Typed soft-coded configuration for UTriage flaw detectors.

Supports pre-registered defaults from ``configs/detectors.yaml``, runtime
overrides with dotted-path syntax, and SHA-256 fingerprinting for auditability.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "detectors.yaml"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SupervisedConfig(_Section):
    """Supervised CNN classifier configuration."""

    threshold: float = Field(default=0.5, gt=0.0, lt=1.0)


class AnomalyConfig(_Section):
    """Unsupervised anomaly autoencoder configuration."""

    target_fcr: float = Field(default=0.01, gt=0.0, lt=1.0)
    smooth_window: int = Field(default=5, ge=1)
    wall_guard: int = Field(default=0, ge=0)


class HybridConfig(_Section):
    """Hybrid detector fusion configuration."""

    mode: Literal["or_union", "fused", "both"] = "or_union"
    fused_target_fcr: float = Field(default=0.01, gt=0.0, lt=1.0)


class PhysicsConfig(_Section):
    """Physics-guided linear classifier configuration."""

    threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    C: float = Field(default=1.0, gt=0.0)
    class_weight: str = "balanced"
    max_iter: int = Field(default=1000, ge=1)
    noise_multiplier: float = Field(default=6.0, gt=0.0)
    energy_window: int = Field(default=8, ge=1)


class EvaluationConfig(_Section):
    """Statistical evaluation and validation protocol configuration."""

    confidence: float = Field(default=0.95, gt=0.0, lt=1.0)
    loco_val_mode: Literal["batch", "crack"] = "batch"
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2], min_length=1)


class DetectorConfig(_Section):
    """Root configuration for all UTriage detectors and evaluation protocol."""

    supervised: SupervisedConfig = Field(default_factory=SupervisedConfig)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


def _apply_override(data: dict[str, Any], override: str) -> None:
    """Apply a dotted key=value override to a nested configuration dictionary."""
    if "=" not in override:
        msg = f"Invalid override format: {override!r}, expected 'key=value'"
        raise ValueError(msg)
    key, val_str = override.split("=", 1)
    val = yaml.safe_load(val_str)
    parts = key.strip().split(".")
    target = data
    for part in parts[:-1]:
        if not isinstance(target.get(part), dict):
            target[part] = {}
        target = target[part]
    target[parts[-1]] = val


def _canonical_json(cfg: DetectorConfig) -> str:
    """Format canonical sorted compact JSON string for deterministic hashing."""
    return json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def load_detector_config(
    path: Path | None = None,
    overrides: list[str] | None = None,
) -> DetectorConfig:
    """Load detector settings from YAML and apply optional dotted overrides."""
    config_path = path or DEFAULT_PATH
    data: dict[str, Any] = {}
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    elif path is not None:
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    if overrides:
        data = copy.deepcopy(data)
        for item in overrides:
            _apply_override(data, item)

    return DetectorConfig.model_validate(data)


def config_fingerprint(cfg: DetectorConfig) -> dict[str, Any]:
    """Compute sha256 hash and is_default status for a detector configuration."""
    canonical = _canonical_json(cfg)
    sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    default_cfg = load_detector_config(DEFAULT_PATH) if DEFAULT_PATH.is_file() else DetectorConfig()
    is_default = _canonical_json(default_cfg) == canonical
    return {"sha256": sha, "is_default": is_default}
