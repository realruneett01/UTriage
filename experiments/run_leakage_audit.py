"""Leakage audit for ML-NDT dataset splits (project plan Phase 8.0).

Verifies whether physical cracks are shared across train, test, and validation splits.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        sys.modules["utriage"].__path__.insert(0, _pkg_dir)

from utriage.ml.datasets import (  # noqa: E402
    MLNDT_ROOT,
    list_batches,
    read_labels,
    split_holdout,
)
from utriage.ml.mlndt_meta import read_flaw_meta, source_crack  # noqa: E402

_LEAKAGE_NOTE = (
    "Old test split is a random batch split; every physical crack appears in "
    "train and test, so test scores measure interpolation within known cracks, "
    "not detection of unseen cracks."
)
_NEUTRAL_NOTE = "No shared physical cracks between train and test splits."


def _audit_batch(batch: Path) -> tuple[Counter, int, int]:
    """Audit single batch: returns crack counts, unflawed count, label mismatch count."""
    meta_list = read_flaw_meta(batch)
    labels = read_labels(batch)
    counts: Counter = Counter()
    unflawed = 0
    mismatches = 0

    for idx, meta in enumerate(meta_list):
        crack = source_crack(meta)
        is_flaw_label = int(labels[idx, 0]) == 1
        has_crack = crack is not None
        if is_flaw_label != has_crack:
            mismatches += 1
        if crack is not None:
            counts[crack] += 1
        else:
            unflawed += 1

    return counts, unflawed, mismatches


def audit_split(batches: list[Path]) -> dict[str, Any]:
    """Audit a split for crack counts, image count, and label consistency."""
    total_counts: Counter = Counter()
    total_unflawed = 0
    total_mismatches = 0

    for batch in batches:
        counts, unflawed, mismatches = _audit_batch(batch)
        total_counts.update(counts)
        total_unflawed += unflawed
        total_mismatches += mismatches

    flawed_total = sum(total_counts.values())
    return {
        "n_batches": len(batches),
        "n_images": total_unflawed + flawed_total,
        "images_per_crack": dict(sorted(total_counts.items())),
        "n_unflawed": total_unflawed,
        "labels_consistent": (total_mismatches == 0),
    }


def build_leakage_audit(root: Path = MLNDT_ROOT) -> dict[str, Any]:
    """Run leakage audit across train, test, and validation splits."""
    data_root = root if root is not None else MLNDT_ROOT
    training_all = list_batches("training", root=data_root)
    val_all = list_batches("validation", root=data_root)
    train_batches, test_batches = split_holdout(training_all, test_frac=0.1, seed=0)

    splits = {
        "train": audit_split(train_batches),
        "test": audit_split(test_batches),
        "validation": audit_split(val_all),
    }

    train_cracks = sorted(splits["train"]["images_per_crack"].keys())
    test_cracks = sorted(splits["test"]["images_per_crack"].keys())
    val_cracks = sorted(splits["validation"]["images_per_crack"].keys())

    shared_train_test = sorted(set(train_cracks) & set(test_cracks))
    shared_train_val = sorted(set(train_cracks) & set(val_cracks))
    leakage = len(shared_train_test) > 0

    note = _LEAKAGE_NOTE if leakage else _NEUTRAL_NOTE

    return {
        "label": "Real data (ML-NDT)",
        "splits": splits,
        "cracks_in_train": train_cracks,
        "cracks_in_test": test_cracks,
        "shared_cracks_train_test": shared_train_test,
        "shared_cracks_train_validation": shared_train_val,
        "leakage_by_physical_crack": leakage,
        "note": note,
    }


def main() -> None:
    """Execute leakage audit CLI and write results/leakage_audit.json."""
    parser = argparse.ArgumentParser(
        description="Audit ML-NDT dataset for crack leakage across splits"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=MLNDT_ROOT,
        help="ML-NDT data root path (default: datasets.MLNDT_ROOT)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/leakage_audit.json"),
        help="Output path for leakage audit JSON (default: results/leakage_audit.json)",
    )
    args = parser.parse_args()

    audit = build_leakage_audit(root=args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote leakage audit to {args.out}")


if __name__ == "__main__":
    main()
