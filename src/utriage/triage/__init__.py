"""Phase 5 triage fusion: union of rules indications and ML extra findings."""

from utriage.triage.fusion import (
    FusionConfig,
    MlFinding,
    TriageItem,
    fuse,
    rank,
    rules_flags_preserved,
)

__all__ = [
    "FusionConfig",
    "MlFinding",
    "TriageItem",
    "fuse",
    "rank",
    "rules_flags_preserved",
]
