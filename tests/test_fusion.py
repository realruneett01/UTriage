"""Tests for Phase 5 triage fusion, ranking, and safety preservation."""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

# Ensure local src/ is imported in worktrees
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from utriage.rules.types import CATEGORIES, Indication
from utriage.triage.fusion import (
    FusionConfig,
    MlFinding,
    TriageItem,
    fuse,
    rank,
    rules_flags_preserved,
)

_BASE_IND = Indication(
    id=1,
    axial_mm=10.0,
    angle_deg=45.0,
    depth_mm=12.0,
    amplitude_pct=60.0,
    snr_db=15.0,
    axial_length_mm=2.0,
    circ_length_mm=2.0,
    n_ascans=4,
    axial_idx=5,
    angle_idx=5,
    category="reportable",
    reasons=(),
)

_BASE_FINDING = MlFinding(
    axial_idx=5,
    angle_idx=5,
    depth_mm=12.0,
    score=0.8,
    uncertainty=0.1,
    grain_noise_prob=None,
    evidence=(),
)


def test_fuse_empty() -> None:
    """Empty inputs yield an empty triage list."""
    assert fuse([], []) == []


def test_fuse_rules_only() -> None:
    """Rule indications without ML findings preserve category and order."""
    ind1 = replace(_BASE_IND, id=1, category="reportable")
    ind2 = replace(_BASE_IND, id=2, category="rejectable")
    items = fuse([ind1, ind2], [])
    assert len(items) == 2
    assert [it.category for it in items] == ["reportable", "rejectable"]
    assert all(it.source == "rules" for it in items)
    assert all(it.ml is None for it in items)
    assert rules_flags_preserved([ind1, ind2], items)


@pytest.mark.parametrize(
    ("d_ax", "d_ang", "d_dep", "expected_match"),
    [
        (0, 0, 0.0, True),
        (1, 1, 2.0, True),
        (2, 0, 0.0, False),
        (0, 2, 0.0, False),
        (0, 0, 2.1, False),
    ],
)
def test_fuse_matching_boundary(
    d_ax: int,
    d_ang: int,
    d_dep: float,
    expected_match: bool,
) -> None:
    """Matching respects cell boundaries on axial/angle and depth window."""
    ind = replace(_BASE_IND, axial_idx=10, angle_idx=10, depth_mm=20.0)
    finding = replace(
        _BASE_FINDING,
        axial_idx=10 + d_ax,
        angle_idx=10 + d_ang,
        depth_mm=20.0 + d_dep,
        score=0.85,
    )
    items = fuse([ind], [finding])
    rule_item = items[0]
    if expected_match:
        assert rule_item.ml == finding
    else:
        assert rule_item.ml is None


def test_fuse_best_scoring_match() -> None:
    """Indication attaches the best-scoring matching ML finding."""
    ind = replace(_BASE_IND, axial_idx=5, angle_idx=5, depth_mm=10.0)
    f_low = replace(_BASE_FINDING, axial_idx=5, angle_idx=5, depth_mm=10.0, score=0.6)
    f_high = replace(_BASE_FINDING, axial_idx=5, angle_idx=5, depth_mm=10.0, score=0.95)
    f_mid = replace(_BASE_FINDING, axial_idx=5, angle_idx=5, depth_mm=10.0, score=0.75)
    items = fuse([ind], [f_low, f_high, f_mid])
    assert items[0].ml == f_high


def test_fuse_custom_config_thresholds() -> None:
    """Custom FusionConfig tolerances and thresholds are respected."""
    cfg = FusionConfig(ml_threshold=0.8, match_cells=2, match_depth_mm=4.0)
    ind = replace(_BASE_IND, axial_idx=10, angle_idx=10, depth_mm=20.0)
    f_matched = replace(_BASE_FINDING, axial_idx=12, angle_idx=12, depth_mm=24.0, score=0.7)
    f_unmatched = replace(_BASE_FINDING, axial_idx=50, angle_idx=50, depth_mm=50.0, score=0.75)
    items = fuse([ind], [f_matched, f_unmatched], cfg=cfg)
    assert items[0].ml == f_matched
    assert len(items) == 1  # f_unmatched score 0.75 < ml_threshold 0.8 is discarded


@pytest.mark.parametrize(
    ("uncertainty", "grain_prob", "reasons", "expected_tags"),
    [
        (0.1, None, (), ()),
        (0.2, None, (), ("uncertain",)),
        (0.05, 0.85, (), ("likely_grain_noise_opinion",)),
        (0.3, 0.9, (), ("uncertain", "likely_grain_noise_opinion")),
        (0.0, None, ("low SNR (borderline)",), ("borderline_low_snr",)),
        (
            0.25,
            0.8,
            ("low SNR",),
            ("uncertain", "likely_grain_noise_opinion", "borderline_low_snr"),
        ),
    ],
)
def test_fuse_review_tags(
    uncertainty: float,
    grain_prob: float | None,
    reasons: tuple[str, ...],
    expected_tags: tuple[str, ...],
) -> None:
    """Review tags reflect uncertainty, grain noise opinion, and low SNR."""
    ind = replace(_BASE_IND, category="reportable", reasons=reasons)
    finding = replace(_BASE_FINDING, uncertainty=uncertainty, grain_noise_prob=grain_prob)
    items = fuse([ind], [finding])
    assert items[0].review == expected_tags
    assert items[0].category == "reportable"


def test_fuse_ml_flag_created_unmatched() -> None:
    """High-scoring unmatched finding generates an ml_flag item."""
    f_pass = replace(_BASE_FINDING, axial_idx=50, score=0.6, uncertainty=0.1)
    f_drop = replace(_BASE_FINDING, axial_idx=60, score=0.4, uncertainty=0.1)
    f_unc = replace(_BASE_FINDING, axial_idx=70, score=0.7, uncertainty=0.3)
    items = fuse([], [f_pass, f_drop, f_unc])
    assert len(items) == 2
    assert [it.category for it in items] == ["ml_flag", "ml_flag"]
    assert items[0].review == ()
    assert items[1].review == ("uncertain",)


def test_fuse_matched_finding_not_duplicated_as_flag() -> None:
    """A finding that matched a rule indication does not become an ml_flag."""
    ind = replace(_BASE_IND, axial_idx=5, angle_idx=5, depth_mm=10.0)
    finding = replace(_BASE_FINDING, axial_idx=5, angle_idx=5, depth_mm=10.0, score=0.9)
    items = fuse([ind], [finding])
    assert len(items) == 1
    assert items[0].source == "rules"
    assert items[0].ml == finding


@pytest.mark.parametrize(
    ("mutate_kind", "expected"),
    [
        ("none", True),
        ("category", False),
        ("drop", False),
        ("extra", False),
    ],
)
def test_rules_flags_preserved_validation(mutate_kind: str, expected: bool) -> None:
    """rules_flags_preserved catches category changes, drops, and extra items."""
    ind = replace(_BASE_IND, category="reportable")
    items = fuse([ind], [])
    if mutate_kind == "category":
        items = [replace(items[0], category="rejectable")]
    elif mutate_kind == "drop":
        items = []
    elif mutate_kind == "extra":
        extra = TriageItem("rules", "reportable", replace(ind, id=99), None, ())
        items = [items[0], extra]
    assert rules_flags_preserved([ind], items) is expected


def test_rank_priority_order() -> None:
    """Ranking follows the strict safety priority hierarchy."""
    it_rej = TriageItem("rules", "rejectable", replace(_BASE_IND, amplitude_pct=90.0), None, ())
    it_unc = TriageItem(
        "rules",
        "reportable",
        replace(_BASE_IND, amplitude_pct=70.0),
        None,
        ("uncertain",),
    )
    it_bord = TriageItem(
        "rules",
        "reportable",
        replace(_BASE_IND, amplitude_pct=65.0),
        None,
        ("borderline_low_snr",),
    )
    it_rep = TriageItem("rules", "reportable", replace(_BASE_IND, amplitude_pct=60.0), None, ())
    it_ml = TriageItem("ml", "ml_flag", None, replace(_BASE_FINDING, score=0.7), ())
    it_dq = TriageItem(
        "rules", "data_quality_fault", replace(_BASE_IND, amplitude_pct=20.0), None, ()
    )
    it_insig = TriageItem(
        "rules", "insignificant", replace(_BASE_IND, amplitude_pct=10.0), None, ()
    )

    ordered = rank([it_insig, it_dq, it_ml, it_rep, it_bord, it_unc, it_rej])
    assert ordered == [it_rej, it_unc, it_bord, it_rep, it_ml, it_dq, it_insig]


def test_rank_rejectable_with_uncertainty_stays_tier_zero() -> None:
    """Rejectable indication with uncertain review remains top priority."""
    rej_unc = TriageItem(
        "rules",
        "rejectable",
        replace(_BASE_IND, amplitude_pct=110.0),
        None,
        ("uncertain",),
    )
    rep_unc = TriageItem(
        "rules",
        "reportable",
        replace(_BASE_IND, amplitude_pct=80.0),
        None,
        ("uncertain",),
    )
    assert rank([rep_unc, rej_unc]) == [rej_unc, rep_unc]


def test_rank_within_group_metric_and_stability() -> None:
    """Within a group, higher metric ranks first; identical ranks preserve order."""
    it1 = TriageItem("rules", "reportable", replace(_BASE_IND, id=1, amplitude_pct=50.0), None, ())
    it2 = TriageItem("rules", "reportable", replace(_BASE_IND, id=2, amplitude_pct=80.0), None, ())
    it3 = TriageItem("rules", "reportable", replace(_BASE_IND, id=3, amplitude_pct=80.0), None, ())
    ranked = rank([it1, it2, it3])
    assert [it.indication.id for it in ranked if it.indication is not None] == [2, 3, 1]


indication_strategy = st.builds(
    Indication,
    id=st.integers(min_value=0, max_value=1000),
    axial_mm=st.floats(min_value=0.0, max_value=500.0, allow_nan=False),
    angle_deg=st.floats(min_value=0.0, max_value=360.0, allow_nan=False),
    depth_mm=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
    amplitude_pct=st.floats(min_value=0.0, max_value=200.0, allow_nan=False),
    snr_db=st.floats(min_value=0.0, max_value=50.0, allow_nan=False),
    axial_length_mm=st.floats(min_value=0.1, max_value=50.0, allow_nan=False),
    circ_length_mm=st.floats(min_value=0.1, max_value=50.0, allow_nan=False),
    n_ascans=st.integers(min_value=1, max_value=100),
    axial_idx=st.integers(min_value=0, max_value=50),
    angle_idx=st.integers(min_value=0, max_value=50),
    category=st.sampled_from(CATEGORIES),
    reasons=st.lists(
        st.sampled_from(["low SNR (borderline)", "amplitude high", "linear", "data quality"]),
        max_size=3,
    ).map(tuple),
)

finding_strategy = st.builds(
    MlFinding,
    axial_idx=st.integers(min_value=0, max_value=50),
    angle_idx=st.integers(min_value=0, max_value=50),
    depth_mm=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
    score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    uncertainty=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    grain_noise_prob=st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    ),
    evidence=st.lists(
        st.sampled_from(["high_grain_noise", "sharp_peak"]),
        max_size=2,
    ).map(tuple),
)


@settings(max_examples=10000, deadline=None)
@given(
    rule_inds=st.lists(indication_strategy, min_size=0, max_size=15),
    findings=st.lists(finding_strategy, min_size=0, max_size=20),
)
def test_property_fusion_safety(
    rule_inds: list[Indication],
    findings: list[MlFinding],
) -> None:
    """Property test (10,000 cases): union preservation and permutation safety."""
    items = fuse(rule_inds, findings)
    assert rules_flags_preserved(rule_inds, items)
    assert len(items) >= len(rule_inds)
    assert all(it.category == "ml_flag" for it in items if it.source != "rules")

    ranked = rank(items)
    assert rules_flags_preserved(rule_inds, ranked)
    assert len(ranked) == len(items)
    assert Counter(id(it) for it in ranked) == Counter(id(it) for it in items)
