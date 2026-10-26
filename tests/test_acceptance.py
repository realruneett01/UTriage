"""Tests for the acceptance rules engine and indication categorisation."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

# Ensure local src/ is imported in worktrees
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from utriage.rules.acceptance import categorise, summarise
from utriage.rules.config import load_acceptance
from utriage.rules.types import CATEGORIES, Indication


@dataclass(frozen=True)
class SingleRuleCase:
    """Test case for a single indication rule."""

    name: str
    amplitude_pct: float
    axial_len: float
    circ_len: float
    snr_db: float
    on_fault: bool
    expected_cat: str
    expected_reason: str


SINGLE_CASES = (
    SingleRuleCase(
        name="insignificant",
        amplitude_pct=30.0,
        axial_len=1.0,
        circ_len=1.0,
        snr_db=12.0,
        on_fault=False,
        expected_cat="insignificant",
        expected_reason="",
    ),
    SingleRuleCase(
        name="reportable",
        amplitude_pct=60.0,
        axial_len=1.0,
        circ_len=1.0,
        snr_db=12.0,
        on_fault=False,
        expected_cat="reportable",
        expected_reason="",
    ),
    SingleRuleCase(
        name="reject_amplitude",
        amplitude_pct=110.0,
        axial_len=1.0,
        circ_len=1.0,
        snr_db=12.0,
        on_fault=False,
        expected_cat="rejectable",
        expected_reason="amplitude 110.0% >= reject 100.0%",
    ),
    SingleRuleCase(
        name="reject_linear_axial",
        amplitude_pct=40.0,
        axial_len=8.0,
        circ_len=1.0,
        snr_db=12.0,
        on_fault=False,
        expected_cat="rejectable",
        expected_reason="linear indication",
    ),
    SingleRuleCase(
        name="reject_linear_circ",
        amplitude_pct=40.0,
        axial_len=1.0,
        circ_len=7.0,
        snr_db=12.0,
        on_fault=False,
        expected_cat="rejectable",
        expected_reason="linear indication",
    ),
    SingleRuleCase(
        name="data_quality_fault",
        amplitude_pct=30.0,
        axial_len=1.0,
        circ_len=1.0,
        snr_db=12.0,
        on_fault=True,
        expected_cat="data_quality_fault",
        expected_reason="data quality fault at this location (re-scan)",
    ),
    SingleRuleCase(
        name="low_snr_insignificant",
        amplitude_pct=30.0,
        axial_len=1.0,
        circ_len=1.0,
        snr_db=4.0,
        on_fault=False,
        expected_cat="insignificant",
        expected_reason="low SNR (borderline)",
    ),
)


def _make_ind(ind_id: int, axial_mm: float, amp_pct: float, snr_db: float) -> Indication:
    """Create a minimal Indication for tests with at most 4 parameters."""
    return Indication(
        id=ind_id,
        axial_mm=axial_mm,
        angle_deg=0.0,
        depth_mm=10.0,
        amplitude_pct=amp_pct,
        snr_db=snr_db,
        axial_length_mm=1.0,
        circ_length_mm=1.0,
        n_ascans=5,
        axial_idx=0,
        angle_idx=0,
    )


@pytest.mark.parametrize("case", SINGLE_CASES, ids=lambda c: c.name)
def test_single_indication_rules(case: SingleRuleCase) -> None:
    cfg = load_acceptance()
    ind = _make_ind(1, 10.0, case.amplitude_pct, case.snr_db)
    ind = replace(ind, axial_length_mm=case.axial_len, circ_length_mm=case.circ_len)
    mask = np.zeros((1, 1), dtype=bool)
    if case.on_fault:
        mask[0, 0] = True
    result = categorise([ind], cfg, fault_mask=mask)
    assert len(result) == 1
    assert result[0].category == case.expected_cat
    if case.expected_reason:
        assert case.expected_reason in result[0].reasons


@dataclass(frozen=True)
class MultipleCase:
    """Test case for multiple indications clustering."""

    name: str
    axial_positions: tuple[float, ...]
    expected_cat: str
    should_have_reason: bool


MULTIPLE_CASES = (
    MultipleCase(
        name="four_in_window_triggers",
        axial_positions=(10.0, 15.0, 20.0, 25.0),
        expected_cat="rejectable",
        should_have_reason=True,
    ),
    MultipleCase(
        name="three_in_window_does_not_trigger",
        axial_positions=(10.0, 15.0, 20.0),
        expected_cat="reportable",
        should_have_reason=False,
    ),
    MultipleCase(
        name="four_spread_out_does_not_trigger",
        axial_positions=(0.0, 50.0, 100.0, 150.0),
        expected_cat="reportable",
        should_have_reason=False,
    ),
)


@pytest.mark.parametrize("case", MULTIPLE_CASES, ids=lambda c: c.name)
def test_multiple_rule_behaviour(case: MultipleCase) -> None:
    cfg = load_acceptance()
    inds = [_make_ind(idx + 1, pos, 60.0, 12.0) for idx, pos in enumerate(case.axial_positions)]
    result = categorise(inds, cfg)
    for ind in result:
        assert ind.category == case.expected_cat
        has_reason = "multiple indications" in ind.reasons
        assert has_reason == case.should_have_reason


def test_fault_never_downgrades_rejectable() -> None:
    """Safety: a data-quality fault must not hide a rejectable indication."""
    cfg = load_acceptance()
    ind = _make_ind(1, 10.0, 150.0, 12.0)
    mask = np.ones((1, 1), dtype=bool)
    result = categorise([ind], cfg, fault_mask=mask)
    assert result[0].category == "rejectable"
    assert "re-scan" in result[0].reasons[0]
    assert "amplitude" in result[0].reasons[0]


def test_categorise_preserves_length_and_ids() -> None:
    cfg = load_acceptance()
    expected_ids = [10, 20, 30, 40, 50]
    inds = [_make_ind(i, float(i), 30.0, 12.0) for i in expected_ids]
    result = categorise(inds, cfg)
    assert len(result) == len(inds)
    assert [ind.id for ind in result] == expected_ids
    assert categorise([], cfg) == []


def test_placeholder_reason_lifecycle() -> None:
    cfg = load_acceptance()
    ind = _make_ind(1, 10.0, 30.0, 12.0)
    result = categorise([ind], cfg)
    assert result[0].reasons[-1] == "limits are PLACEHOLDERS"

    verified_cfg = cfg.model_copy(update={"status": "VERIFIED - inspected standard"})
    verified_result = categorise([ind], verified_cfg)
    assert "limits are PLACEHOLDERS" not in verified_result[0].reasons


def test_summarise_all_categories() -> None:
    cfg = load_acceptance()
    inds = [
        _make_ind(1, 10.0, 120.0, 12.0),
        _make_ind(2, 20.0, 60.0, 12.0),
        _make_ind(3, 30.0, 30.0, 12.0),
        replace(_make_ind(4, 40.0, 30.0, 12.0), axial_idx=1),
    ]
    mask = np.zeros((2, 1), dtype=bool)
    mask[1, 0] = True
    categorised = categorise(inds, cfg, fault_mask=mask)
    summary = summarise(categorised)

    assert set(summary.keys()) == set(CATEGORIES)
    assert summary["rejectable"] == 1
    assert summary["reportable"] == 1
    assert summary["insignificant"] == 1
    assert summary["data_quality_fault"] == 1

    empty_summary = summarise([])
    assert empty_summary == {cat: 0 for cat in CATEGORIES}
