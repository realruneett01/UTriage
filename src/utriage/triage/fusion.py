"""Phase 5 triage fusion: union of rules indications and ML findings.

Safety architecture:
- Rules indications are never removed or changed by ML.
- ML findings can only add extra advisory flags (source='ml', category='ml_flag').
- Borderline rule indications receive advisory review tags without category changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from utriage.rules.types import Indication

_BORDERLINE_TAGS = frozenset({"likely_grain_noise_opinion", "borderline_low_snr"})
_CATEGORY_PRIORITY = {
    "reportable": 3,
    "ml_flag": 4,
    "data_quality_fault": 5,
    "insignificant": 6,
}


@dataclass(frozen=True)
class MlFinding:
    """An ML detection finding across A-scan or C-scan grids."""

    axial_idx: int
    angle_idx: int
    depth_mm: float
    score: float
    uncertainty: float
    grain_noise_prob: float | None = None
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class FusionConfig:
    """Thresholds and matching tolerances for rules and ML fusion."""

    ml_threshold: float = 0.5
    uncertain_threshold: float = 0.2
    grain_opinion_threshold: float = 0.8
    match_cells: int = 1
    match_depth_mm: float = 2.0


_DEFAULT_CONFIG = FusionConfig()


@dataclass(frozen=True)
class TriageItem:
    """One fused indication item for inspector review."""

    source: str
    category: str
    indication: Indication | None
    ml: MlFinding | None
    review: tuple[str, ...] = ()


def _matches(ind: Indication, finding: MlFinding, cfg: FusionConfig) -> bool:
    """Return True if an ML finding matches a rule indication within tolerance."""
    return all(
        (
            abs(ind.axial_idx - finding.axial_idx) <= cfg.match_cells,
            abs(ind.angle_idx - finding.angle_idx) <= cfg.match_cells,
            abs(ind.depth_mm - finding.depth_mm) <= cfg.match_depth_mm,
        )
    )


def _best_match(
    ind: Indication,
    findings: list[MlFinding],
    cfg: FusionConfig,
) -> MlFinding | None:
    """Attach the best-scoring matching finding to a rule indication."""
    matches = [f for f in findings if _matches(ind, f, cfg)]
    if not matches:
        return None
    return max(matches, key=lambda f: f.score)


def _is_low_snr(reasons: tuple[str, ...]) -> bool:
    """Return True if any reason indicates low SNR."""
    return any("low snr" in r.lower() for r in reasons)


def _rule_review(
    ind: Indication,
    ml: MlFinding | None,
    cfg: FusionConfig,
) -> tuple[str, ...]:
    """Assemble review tags for a rules indication."""
    tags: list[str] = []
    if ml is not None:
        if ml.uncertainty >= cfg.uncertain_threshold:
            tags.append("uncertain")
        if ml.grain_noise_prob is not None and ml.grain_noise_prob >= cfg.grain_opinion_threshold:
            tags.append("likely_grain_noise_opinion")
    if _is_low_snr(ind.reasons):
        tags.append("borderline_low_snr")
    return tuple(tags)


def _fuse_rule_item(
    ind: Indication,
    findings: list[MlFinding],
    cfg: FusionConfig,
) -> TriageItem:
    """Construct one rules TriageItem with attached best-scoring ML finding."""
    matched_ml = _best_match(ind, findings, cfg)
    review = _rule_review(ind, matched_ml, cfg)
    return TriageItem(
        source="rules",
        category=ind.category,
        indication=ind,
        ml=matched_ml,
        review=review,
    )


def _ml_item(finding: MlFinding, cfg: FusionConfig) -> TriageItem:
    """Construct an ML-only triage item for an unmatched finding."""
    review = ("uncertain",) if finding.uncertainty >= cfg.uncertain_threshold else ()
    return TriageItem(
        source="ml",
        category="ml_flag",
        indication=None,
        ml=finding,
        review=review,
    )


def _find_unmatched_ml(
    rule_inds: list[Indication],
    findings: list[MlFinding],
    cfg: FusionConfig,
) -> list[TriageItem]:
    """Create ML triage items for high-scoring findings matching no rules."""
    items: list[TriageItem] = []
    for f in findings:
        if f.score < cfg.ml_threshold:
            continue
        if any(_matches(ind, f, cfg) for ind in rule_inds):
            continue
        items.append(_ml_item(f, cfg))
    return items


def fuse(
    rule_inds: list[Indication],
    findings: list[MlFinding],
    cfg: FusionConfig = _DEFAULT_CONFIG,
) -> list[TriageItem]:
    """Fuse rules indications with ML findings: rules union ML flags."""
    rule_items = [_fuse_rule_item(ind, findings, cfg) for ind in rule_inds]
    ml_items = _find_unmatched_ml(rule_inds, findings, cfg)
    return rule_items + ml_items


def _group_priority(item: TriageItem) -> int:
    """Return ranking group priority (0 = highest priority)."""
    if item.category == "rejectable":
        return 0
    if "uncertain" in item.review:
        return 1
    if any(tag in _BORDERLINE_TAGS for tag in item.review):
        return 2
    return _CATEGORY_PRIORITY.get(item.category, 7)


def _item_metric(item: TriageItem) -> float:
    """Return numeric metric for ranking within a group."""
    if item.indication is not None:
        return item.indication.amplitude_pct
    if item.ml is not None:
        return item.ml.score
    return 0.0


def rank(items: list[TriageItem]) -> list[TriageItem]:
    """Sort triage items in stable review priority order."""
    return sorted(items, key=lambda it: (_group_priority(it), -_item_metric(it)))


def _match_and_pop(
    ind: Indication,
    remaining: list[TriageItem],
) -> bool:
    """Find and pop the matching rule item from remaining items."""
    for idx, item in enumerate(remaining):
        if item.indication == ind and item.category == ind.category:
            remaining.pop(idx)
            return True
    return False


def rules_flags_preserved(
    rule_inds: list[Indication],
    items: list[TriageItem],
) -> bool:
    """Return True iff every rule indication appears exactly once unchanged."""
    rule_items = [it for it in items if it.source == "rules"]
    indications = list(rule_inds)
    if len(rule_items) != len(indications):
        return False
    if len(rule_items) != len([it for it in items if it.indication is not None]):
        return False
    remaining = list(rule_items)
    return all(_match_and_pop(ind, remaining) for ind in indications)
