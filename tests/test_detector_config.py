"""Tests for soft-coded detector configuration and validation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from utriage.detector_config import (  # noqa: E402
    AnomalyConfig,
    DetectorConfig,
    EvaluationConfig,
    HybridConfig,
    PhysicsConfig,
    SupervisedConfig,
    config_fingerprint,
    load_detector_config,
)


def test_default_config_loads_expected_values() -> None:
    """Verify that shipped YAML detector configuration loads pre-registered defaults."""
    cfg = load_detector_config()

    assert cfg.supervised.threshold == 0.5
    assert cfg.anomaly.target_fcr == 0.01
    assert cfg.anomaly.smooth_window == 5
    assert cfg.anomaly.wall_guard == 0
    assert cfg.hybrid.mode == "or_union"
    assert cfg.hybrid.fused_target_fcr == 0.01
    assert cfg.physics.threshold == 0.5
    assert cfg.physics.C == 1.0
    assert cfg.physics.class_weight == "balanced"
    assert cfg.physics.max_iter == 1000
    assert cfg.physics.noise_multiplier == 6.0
    assert cfg.physics.energy_window == 8
    assert cfg.evaluation.confidence == 0.95
    assert cfg.evaluation.loco_val_mode == "batch"
    assert cfg.evaluation.seeds == [0, 1, 2]


def test_detector_config_direct_defaults() -> None:
    """Direct instantiation of DetectorConfig yields pre-registered defaults."""
    cfg = DetectorConfig()
    loaded = load_detector_config()
    assert cfg == loaded


@pytest.mark.parametrize(
    ("model_cls", "field_name", "invalid_value"),
    [
        (SupervisedConfig, "threshold", 1.5),
        (SupervisedConfig, "threshold", 0.0),
        (AnomalyConfig, "target_fcr", 1.5),
        (AnomalyConfig, "target_fcr", 0.0),
        (AnomalyConfig, "smooth_window", 0),
        (HybridConfig, "mode", "invalid_mode"),
        (HybridConfig, "fused_target_fcr", 1.0),
        (PhysicsConfig, "threshold", 1.5),
        (PhysicsConfig, "C", 0.0),
        (PhysicsConfig, "C", -1.0),
        (PhysicsConfig, "energy_window", 0),
        (EvaluationConfig, "confidence", 1.0),
        (EvaluationConfig, "loco_val_mode", "bad_mode"),
        (EvaluationConfig, "seeds", []),
    ],
)
def test_field_validation_errors(
    model_cls: type[Any],
    field_name: str,
    invalid_value: Any,
) -> None:
    """Individual section models enforce validation constraints on invalid values."""
    with pytest.raises(ValidationError):
        model_cls(**{field_name: invalid_value})


@pytest.mark.parametrize(
    "model_cls",
    [
        SupervisedConfig,
        AnomalyConfig,
        HybridConfig,
        PhysicsConfig,
        EvaluationConfig,
        DetectorConfig,
    ],
)
def test_extra_forbidden(model_cls: type[Any]) -> None:
    """Unknown keys are rejected across all configuration models."""
    with pytest.raises(ValidationError, match="extra"):
        model_cls(unknown_field_123=999)


@pytest.mark.parametrize(
    ("override", "expected_check"),
    [
        ("hybrid.mode=fused", lambda cfg: cfg.hybrid.mode == "fused"),
        ("supervised.threshold=0.6", lambda cfg: cfg.supervised.threshold == 0.6),
        ("anomaly.smooth_window=7", lambda cfg: cfg.anomaly.smooth_window == 7),
        ("physics.C=2.5", lambda cfg: cfg.physics.C == 2.5),
        ("evaluation.seeds=[10, 20]", lambda cfg: cfg.evaluation.seeds == [10, 20]),
    ],
)
def test_overrides_apply_successfully(override: str, expected_check: Any) -> None:
    """Valid dotted overrides correctly update configuration values."""
    cfg = load_detector_config(overrides=[override])
    assert expected_check(cfg)


@pytest.mark.parametrize(
    "bad_override",
    [
        "supervised.threshold=1.5",
        "physics.C=0",
        "hybrid.mode=unsupported_mode",
        "supervised.unknown_key=10",
        "unknown_section.key=10",
        "evaluation.seeds=[]",
    ],
)
def test_overrides_validation_errors(bad_override: str) -> None:
    """Invalid overrides raise ValidationError during validation."""
    with pytest.raises(ValidationError):
        load_detector_config(overrides=[bad_override])


def test_invalid_override_format_raises() -> None:
    """Overrides without key=value syntax raise ValueError."""
    with pytest.raises(ValueError, match="Invalid override format"):
        load_detector_config(overrides=["not_a_key_value"])


def test_config_fingerprint_default() -> None:
    """Default configuration returns valid sha256 and is_default=True."""
    cfg = load_detector_config()
    fp = config_fingerprint(cfg)
    assert fp["is_default"] is True
    assert isinstance(fp["sha256"], str)
    assert len(fp["sha256"]) == 64


def test_config_fingerprint_modified() -> None:
    """Modified configuration returns different sha256 and is_default=False."""
    default_fp = config_fingerprint(load_detector_config())
    cfg_mod = load_detector_config(overrides=["supervised.threshold=0.7"])
    mod_fp = config_fingerprint(cfg_mod)

    assert mod_fp["is_default"] is False
    assert mod_fp["sha256"] != default_fp["sha256"]
