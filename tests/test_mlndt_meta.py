"""Tests for ML-NDT metadata extraction and leakage audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from experiments.run_leakage_audit import (
    audit_split,
    build_leakage_audit,
)

from utriage.ml.mlndt_meta import image_index, read_flaw_meta, source_crack


def _create_fake_batch(
    batch_path: Path,
    flaws: list[dict[str, Any]],
    labels: list[tuple[float, float]],
) -> Path:
    """Write fake .bins, .jsons, and .labels files for a batch."""
    n = len(flaws)
    data = np.zeros((n, 256, 256), dtype=np.uint16)
    data.tofile(batch_path)

    jsons_text = "".join(json.dumps({"flaws": [f]}) for f in flaws)
    batch_path.with_suffix(".jsons").write_text(jsons_text, encoding="utf-8")

    labels_arr = np.array(labels, dtype=np.float32)
    np.savetxt(batch_path.with_suffix(".labels"), labels_arr, delimiter="\t")
    return batch_path


def _populate_fake_root(root: Path) -> None:
    """Populate fake ML-NDT root with training and validation batches."""
    tr_dir = root / "training"
    val_dir = root / "validation"
    tr_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    for i in range(10):
        _create_fake_batch(
            tr_dir / f"b_{i:02d}.bins",
            [{"size": "1.6"}, {"size": "0.0"}],
            [(1.0, 1.6), (0.0, 0.0)],
        )
    _create_fake_batch(
        val_dir / "val_01.bins",
        [{"size": "8.6"}, {"size": "0.0"}],
        [(1.0, 8.6), (0.0, 0.0)],
    )


def test_read_flaw_meta_concatenated(tmp_path: Path) -> None:
    """Verify parsing concatenated JSON objects without separators."""
    bpath = tmp_path / "batch_001.bins"
    flaws = [
        {"size": "1.6", "equivalent_flawsize": "1.8"},
        {"size": "0.0", "equivalent_flawsize": "0.0"},
        {"size": "8.6", "equivalent_flawsize": "8.0"},
    ]
    labels = [(1.0, 1.8), (0.0, 0.0), (1.0, 8.0)]
    _create_fake_batch(bpath, flaws, labels)

    meta = read_flaw_meta(bpath)
    assert len(meta) == 3
    assert meta[0]["size"] == "1.6"
    assert meta[1]["size"] == "0.0"
    assert meta[2]["size"] == "8.6"


def test_read_flaw_meta_mismatch_raises(tmp_path: Path) -> None:
    """Verify ValueError when JSON count differs from image count in .bins."""
    bpath = tmp_path / "mismatch.bins"
    flaws = [{"size": "1.6"}]
    labels = [(1.0, 1.6)]
    _create_fake_batch(bpath, flaws, labels)

    # Pad .bins to 2 images so size indicates 2 images, but .jsons only has 1
    two_images = np.zeros((2, 256, 256), dtype=np.uint16)
    two_images.tofile(bpath)

    with pytest.raises(ValueError, match="expected 2 images"):
        read_flaw_meta(bpath)


@pytest.mark.parametrize(
    ("meta", "expected"),
    [
        ({"size": "1.6"}, "crack_1.6"),
        ({"size": "4.0"}, "crack_4.0"),
        ({"size": "8.6"}, "crack_8.6"),
        ({"size": 1.6}, "crack_1.6"),
        ({"size": "0.0"}, None),
        ({"size": 0.0}, None),
        ({"size": "0"}, None),
        ({}, None),
        ({"size": "invalid"}, None),
    ],
)
def test_source_crack_mapping(meta: dict[str, Any], expected: str | None) -> None:
    """Verify crack id extraction and unflawed mapping to None."""
    assert source_crack(meta) == expected


def test_image_index(tmp_path: Path) -> None:
    """Verify indexing of batches as (batch, image_idx, crack_id) tuples."""
    b1 = _create_fake_batch(
        tmp_path / "b1.bins",
        [{"size": "1.6"}, {"size": "0.0"}],
        [(1.0, 1.6), (0.0, 0.0)],
    )
    b2 = _create_fake_batch(
        tmp_path / "b2.bins",
        [{"size": "4.0"}],
        [(1.0, 4.0)],
    )

    idx = image_index([b1, b2])
    assert len(idx) == 3
    assert idx[0] == (b1, 0, "crack_1.6")
    assert idx[1] == (b1, 1, None)
    assert idx[2] == (b2, 0, "crack_4.0")


def test_audit_leakage_on_fake_root(tmp_path: Path) -> None:
    """Verify leakage audit counting and shared crack detection on fake root."""
    root = tmp_path / "fake_mlndt"
    _populate_fake_root(root)
    audit = build_leakage_audit(root=root)
    train_split = audit["splits"]["train"]
    test_split = audit["splits"]["test"]

    assert audit["label"] == "Real data (ML-NDT)"
    assert (train_split["n_batches"], train_split["n_images"]) == (9, 18)
    assert train_split["images_per_crack"] == {"crack_1.6": 9}
    assert (train_split["n_unflawed"], train_split["labels_consistent"]) == (9, True)

    assert (test_split["n_batches"], test_split["n_images"]) == (1, 2)
    assert test_split["images_per_crack"] == {"crack_1.6": 1}
    assert (test_split["n_unflawed"], test_split["labels_consistent"]) == (1, True)

    assert audit["cracks_in_train"] == ["crack_1.6"]
    assert audit["cracks_in_test"] == ["crack_1.6"]
    assert audit["shared_cracks_train_test"] == ["crack_1.6"]
    assert audit["shared_cracks_train_validation"] == []
    assert audit["leakage_by_physical_crack"] is True
    assert "Old test split is a random batch split" in audit["note"]


def test_audit_labels_inconsistent(tmp_path: Path) -> None:
    """Verify labels_consistent flag is False when label and crack flag disagree."""
    bpath = _create_fake_batch(
        tmp_path / "inconsistent.bins",
        [{"size": "1.6"}],
        [(0.0, 0.0)],  # Label 0 but flaw size is 1.6
    )
    result = audit_split([bpath])
    assert result["labels_consistent"] is False


def test_audit_neutral_note_when_no_shared_cracks(tmp_path: Path) -> None:
    """Verify neutral note when train and test hold disjoint physical cracks."""
    root = tmp_path / "disjoint_mlndt"
    tr_dir = root / "training"
    val_dir = root / "validation"
    tr_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    _create_fake_batch(tr_dir / "b0.bins", [{"size": "1.6"}], [(1.0, 1.6)])
    _create_fake_batch(tr_dir / "b1.bins", [{"size": "4.0"}], [(1.0, 4.0)])
    _create_fake_batch(val_dir / "v0.bins", [{"size": "8.6"}], [(1.0, 8.6)])

    audit = build_leakage_audit(root=root)
    assert not audit["shared_cracks_train_test"]
    assert audit["leakage_by_physical_crack"] is False
    assert audit["note"] == "No shared physical cracks between train and test splits."
