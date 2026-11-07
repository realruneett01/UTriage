"""Tests for titanium zone evaluation quantitative claims.

Checks claim construction, evidence references, and wording rules.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pytest  # noqa: E402

from utriage.report.claims import (  # noqa: E402
    Claim,
    build_claims,
    check_forbidden,
    check_forbidden_words,
)
from utriage.report.claims_ti import (  # noqa: E402
    TI_SCOPE_LIMIT,
    collect_ti_claims,
)

DEFAULT_SWEEP: tuple[tuple[str, float, int], ...] = (
    ("flat", 0.0, 0),
    ("structured", 0.1, 0),
)


def _make_fake_zone_cal(
    zone: int,
    snr: float | None,
    inspectable: bool,
) -> dict[str, Any]:
    """Build fake zone calibration dictionary."""
    return {
        "zone": zone,
        "snr": snr,
        "inspectable": inspectable,
        "alarm_level": 0.5,
        "depth_mm": [0.0, 10.0],
    }


def _make_fake_zone_flaw(
    zone: int,
    k: int,
    n: int,
    inspectable: bool = True,
) -> dict[str, Any]:
    """Build fake zone flaw detection dictionary."""
    return {
        "zone": zone,
        "k": k,
        "n": n,
        "rate": k / n if n > 0 else 0.0,
        "inspectable": inspectable,
    }


def _make_fake_arm_data(
    far: float,
    k_clean: int,
    flaws: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build fake arm evaluation dictionary."""
    return {
        "clean": {
            "false_alarms_per_bar": far,
            "bars_with_alarm": {"k": k_clean, "n": 20},
        },
        "flawed": {"per_zone": flaws, "zones_not_inspectable": 0},
        "zones_not_inspectable": 0,
    }


def _make_fake_structured_model() -> dict[str, Any]:
    """Build fake structured noise model data."""
    z_cals = [
        _make_fake_zone_cal(0, 12.34, True),
        _make_fake_zone_cal(1, 10.50, True),
    ]
    fz_zones = [_make_fake_zone_flaw(0, 20, 20), _make_fake_zone_flaw(1, 19, 20)]
    fg_zones = [_make_fake_zone_flaw(0, 18, 20), _make_fake_zone_flaw(1, 15, 20)]
    return {
        "calibration": {"zones": z_cals, "ref_fbh_mm": 0.8},
        "arms": {
            "zone": _make_fake_arm_data(0.05, 1, fz_zones),
            "global": _make_fake_arm_data(0.10, 2, fg_zones),
        },
    }


def _make_fake_sweep_model(uninspect: int, k: int, n: int) -> dict[str, Any]:
    """Build fake model arm data for sweep entry."""
    half_k = k // 2
    half_n = n // 2
    fz = [
        _make_fake_zone_flaw(0, half_k, half_n, inspectable=(uninspect < 2)),
        _make_fake_zone_flaw(1, k - half_k, n - half_n, inspectable=(uninspect < 1)),
    ]
    return {
        "arms": {
            "zone": {
                "zones_not_inspectable": uninspect,
                "flawed": {"per_zone": fz},
            }
        }
    }


def _make_fake_ti_payload(
    sweep_entries: Sequence[tuple[str, float, int]] = DEFAULT_SWEEP,
) -> dict[str, Any]:
    """Build complete fake ti_zone_metrics payload."""
    models: dict[str, Any] = {"structured": _make_fake_structured_model()}
    sweep_list = []
    for name, sigma, uninsp in sweep_entries:
        sweep_list.append({"name": name, "field_sigma": sigma})
        if name != "structured":
            models[name] = _make_fake_sweep_model(uninsp, 38, 40)
    return {
        "sweep": sweep_list,
        "noise_models": models,
        "counts": {"n_clean": 20, "n_flaw": 20, "n_zones": 2},
        "notes": [
            "Synthetic data only.",
            "Default field_sigma was lowered from 0.4 to 0.1 after the first calibration run.",
        ],
    }


def _payload(
    sweep: Sequence[tuple[str, float, int]] = DEFAULT_SWEEP,
    **overrides: Any,
) -> dict[str, Any]:
    """Build fake ti_zone_metrics payload with optional overrides."""
    data = _make_fake_ti_payload(sweep)
    data.update(overrides)
    return data


def _write_payload(tmp_path: Path, payload: dict[str, Any]) -> None:
    """Write metrics payload JSON to the test directory."""
    (tmp_path / "ti_zone_metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def _collect_for_payload(
    tmp_path: Path,
    payload: dict[str, Any] | None = None,
) -> list[Claim]:
    """Write metrics payload and return collected titanium claims."""
    data = payload if payload is not None else _payload()
    _write_payload(tmp_path, data)
    return collect_ti_claims(tmp_path)


def _assert_numbers_match_text(text: str, numbers: dict[str, float]) -> None:
    """Check each number in dictionary appears formatted in claim text."""
    assert len(numbers) > 0
    for key, val in numbers.items():
        candidates = [f"{val:.3f}", f"{val:.2f}", f"{val:.1f}", str(int(val))]
        assert any(re.search(rf"\b{re.escape(cand)}\b", text) for cand in candidates), (
            f"Value {val} ({key}) not found in text: '{text}'"
        )


def _assert_no_forbidden_phrases(text: str, scope: str) -> None:
    """Check that forbidden phrases and words are absent."""
    assert not check_forbidden(text)
    assert not check_forbidden(scope)
    assert not check_forbidden_words(text)
    assert not check_forbidden_words(scope)


def _assert_numbers_in_text(claim: Claim) -> None:
    """Check claim numbers match text and forbidden phrases are absent."""
    _assert_numbers_match_text(claim.text, claim.numbers)
    _assert_no_forbidden_phrases(claim.text, claim.scope_limit)


def _assert_no_forbidden(claim: Claim) -> None:
    """Check that claim text and scope limit are free of forbidden words."""
    _assert_no_forbidden_phrases(claim.text, claim.scope_limit)


def test_missing_file_returns_empty(tmp_path: Path) -> None:
    """Check empty list returned when results file is absent."""
    claims = collect_ti_claims(tmp_path)
    assert claims == []


def test_both_claims_built_from_tiny_fake_payload(tmp_path: Path) -> None:
    """Check both claims built with numbers, scope, and text elements."""
    sweep = [("flat", 0.0, 0), ("structured", 0.1, 0), ("sigma_0.2", 0.2, 1)]
    claims = _collect_for_payload(tmp_path, _payload(sweep=sweep))
    assert len(claims) == 2

    c_cal = next(c for c in claims if c.id == "ti_zone_calibration")
    assert c_cal.evidence_file == "results/ti_zone_metrics.json"
    assert c_cal.evidence_path == "noise_models.structured"
    assert c_cal.scope_limit == TI_SCOPE_LIMIT
    assert "Synthetic data (titanium-mode simulator)" in c_cal.text
    assert "placeholders (UNVERIFIED)" in c_cal.text
    assert "20 clean bars" in c_cal.text
    assert "0.8 mm" in c_cal.text
    _assert_numbers_in_text(c_cal)

    c_sweep = next(c for c in claims if c.id == "ti_noise_sweep")
    assert c_sweep.evidence_file == "results/ti_zone_metrics.json"
    assert c_sweep.evidence_path == "sweep"
    assert c_sweep.scope_limit == TI_SCOPE_LIMIT
    sentence = "as the synthetic noise-strength variation grows, zones lose inspectability"
    assert sentence in c_sweep.text
    assert "lowered after the first calibration run" in c_sweep.text
    _assert_numbers_in_text(c_sweep)


def test_missing_structured_model_skips_claim_1(tmp_path: Path) -> None:
    """Check calibration claim skipped when structured model data is missing."""
    payload = _payload(sweep=[("flat", 0.0, 0)])
    del payload["noise_models"]["structured"]
    claims = _collect_for_payload(tmp_path, payload)
    assert len(claims) == 1
    assert claims[0].id == "ti_noise_sweep"


@pytest.mark.parametrize(
    ("sweep_spec", "expect_sentence"),
    [
        ([("flat", 0.0, 0), ("s1", 0.1, 0), ("s2", 0.2, 1)], True),
        ([("flat", 0.0, 1), ("s1", 0.1, 0), ("s2", 0.2, 2)], False),
    ],
)
def test_sweep_sentence_monotone_condition(
    tmp_path: Path,
    sweep_spec: list[tuple[str, float, int]],
    expect_sentence: bool,
) -> None:
    """Check monotonicity clause included only when counts are non-decreasing."""
    sentence = "as the synthetic noise-strength variation grows, zones lose inspectability"
    claims = _collect_for_payload(tmp_path, _payload(sweep=sweep_spec))
    c_sweep = next(c for c in claims if c.id == "ti_noise_sweep")
    if expect_sentence:
        assert sentence in c_sweep.text
    else:
        assert sentence not in c_sweep.text


def test_build_claims_includes_ti_claims(tmp_path: Path) -> None:
    """Check build_claims attaches titanium claims when file exists."""
    _write_payload(tmp_path, _payload())
    claims = build_claims(tmp_path)
    ti_ids = {c.id for c in claims if c.id.startswith("ti_")}
    assert ti_ids == {"ti_zone_calibration", "ti_noise_sweep"}


def test_missing_ref_fbh_skips_calibration_claim(tmp_path: Path) -> None:
    """Check calibration claim skipped when ref_fbh_mm is missing."""
    payload = _payload()
    del payload["noise_models"]["structured"]["calibration"]["ref_fbh_mm"]
    claims = _collect_for_payload(tmp_path, payload)
    cal_claims = [c for c in claims if c.id == "ti_zone_calibration"]
    assert len(cal_claims) == 0
    for claim in claims:
        assert "0.8" not in claim.text
        assert 0.8 not in claim.numbers.values()


def test_missing_zone_k_skips_sweep_claim(tmp_path: Path) -> None:
    """Check sweep claim skipped when a per_zone entry misses k."""
    payload = _payload()
    del payload["noise_models"]["flat"]["arms"]["zone"]["flawed"]["per_zone"][0]["k"]
    claims = _collect_for_payload(tmp_path, payload)
    sweep_claims = [c for c in claims if c.id == "ti_noise_sweep"]
    assert len(sweep_claims) == 0
    cal_claims = [c for c in claims if c.id == "ti_zone_calibration"]
    assert len(cal_claims) == 1


def test_snr_none_shows_na_and_omits_key(tmp_path: Path) -> None:
    """Check snr None shows SNR n/a in text and leaves key out of numbers."""
    payload = _payload()
    payload["noise_models"]["structured"]["calibration"]["zones"][0]["snr"] = None
    claims = _collect_for_payload(tmp_path, payload)
    c_cal = next(c for c in claims if c.id == "ti_zone_calibration")
    assert "SNR n/a" in c_cal.text
    assert "structured_zone0_snr" not in c_cal.numbers
    assert "structured_zone1_snr" in c_cal.numbers
    _assert_numbers_in_text(c_cal)


def test_global_arm_not_inspectable(tmp_path: Path) -> None:
    """Check claim text when global arm is flagged not inspectable."""
    payload = _payload()
    arm_g = payload["noise_models"]["structured"]["arms"]["global"]
    arm_g["zones_not_inspectable"] = 1
    for z in arm_g["flawed"]["per_zone"]:
        z["inspectable"] = False
        z["k"] = 0
    claims = _collect_for_payload(tmp_path, payload)
    c_cal = next(c for c in claims if c.id == "ti_zone_calibration")
    assert "not inspectable (whole depth range flagged to the inspector)" in c_cal.text
    assert "global arm 0/" not in c_cal.text
    assert c_cal.numbers["global_zones_not_inspectable"] == 1.0
    _assert_no_forbidden(c_cal)


def test_sweep_all_zones_not_inspectable(tmp_path: Path) -> None:
    """Check claim text when all zones of sweep model are not inspectable."""
    payload = _payload(sweep=[("flat", 0.0, 0), ("s_all", 0.3, 3)])
    fz = [_make_fake_zone_flaw(i, 0, 20, inspectable=False) for i in range(3)]
    payload["noise_models"]["s_all"]["arms"]["zone"]["flawed"]["per_zone"] = fz
    payload["noise_models"]["s_all"]["arms"]["zone"]["zones_not_inspectable"] = 3
    claims = _collect_for_payload(tmp_path, payload)
    c_sweep = next(c for c in claims if c.id == "ti_noise_sweep")
    assert "no inspectable zone" in c_sweep.text
    assert "0/60" not in c_sweep.text
    assert c_sweep.numbers["s_all_zone_arm_total_k"] == 0.0
    assert c_sweep.numbers["s_all_zone_arm_total_n"] == 60.0
    _assert_no_forbidden(c_sweep)


def test_sweep_mixed_inspectable_zones(tmp_path: Path) -> None:
    """Check detection k/n kept only over inspectable zones in mixed case."""
    payload = _payload(sweep=[("flat", 0.0, 0), ("s_mix", 0.2, 1)])
    fz = [
        _make_fake_zone_flaw(0, 19, 20, inspectable=True),
        _make_fake_zone_flaw(1, 0, 20, inspectable=False),
    ]
    payload["noise_models"]["s_mix"]["arms"]["zone"]["flawed"]["per_zone"] = fz
    payload["noise_models"]["s_mix"]["arms"]["zone"]["zones_not_inspectable"] = 1
    claims = _collect_for_payload(tmp_path, payload)
    c_sweep = next(c for c in claims if c.id == "ti_noise_sweep")
    assert "1 zones not inspectable (flagged to the inspector)" in c_sweep.text
    assert "zone arm detection 19/20" in c_sweep.text
    assert c_sweep.numbers["s_mix_zone_arm_total_k"] == 19.0
    assert c_sweep.numbers["s_mix_zone_arm_total_n"] == 20.0
    _assert_no_forbidden(c_sweep)


def test_zone_arm_not_inspectable(tmp_path: Path) -> None:
    """Check claim text when zone arm has an uninspectable zone."""
    payload = _payload()
    arm_z = payload["noise_models"]["structured"]["arms"]["zone"]
    arm_z["zones_not_inspectable"] = 1
    arm_z["flawed"]["per_zone"][0]["inspectable"] = False
    claims = _collect_for_payload(tmp_path, payload)
    c_cal = next(c for c in claims if c.id == "ti_zone_calibration")
    assert "zone arm: not inspectable (zone flagged to the inspector)" in c_cal.text
    assert c_cal.numbers["zone_zones_not_inspectable"] == 1.0
    _assert_no_forbidden(c_cal)
