"""ML-NDT metadata extraction and indexing (project plan Phase 8.0).

Parses concatenated per-image JSON flaw descriptors and maps source crack sizes.
"""

from __future__ import annotations

import json
from pathlib import Path

from utriage.ml.datasets import n_images


def read_flaw_meta(bins: Path) -> list[dict]:
    """Read per-image flaw metadata from the sibling .jsons file of a batch.

    Parses concatenated JSON objects with raw_decode and verifies that object
    count matches the number of images indicated by the .bins file size.
    """
    bins_path = Path(bins)
    jsons_path = bins_path.with_suffix(".jsons")
    text = jsons_path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    idx, length = 0, len(text)
    objs: list[dict] = []

    while idx < length:
        while idx < length and text[idx].isspace():
            idx += 1
        if idx >= length:
            break
        obj, end = decoder.raw_decode(text, idx)
        objs.append(obj)
        idx = end

    expected = n_images(bins_path)
    if len(objs) != expected:
        raise ValueError(
            f"Batch {bins_path.name}: expected {expected} images from .bins size, "
            f"found {len(objs)} JSON objects in {jsons_path.name}"
        )
    return [obj["flaws"][0] if obj.get("flaws") else {} for obj in objs]


def source_crack(meta: dict) -> str | None:
    """Return physical crack identifier 'crack_<size>' or None if unflawed (size == 0.0)."""
    size_raw = meta.get("size")
    if size_raw is None:
        return None
    try:
        if float(size_raw) == 0.0:
            return None
    except (ValueError, TypeError):
        # ponytail: non-numeric or malformed size treated as unflawed
        return None
    return f"crack_{size_raw}"


def image_index(batches: list[Path]) -> list[tuple[Path, int, str | None]]:
    """Return flat list of (batch_path, image_idx, crack_id) tuples across batches."""
    index: list[tuple[Path, int, str | None]] = []
    for batch in batches:
        meta_list = read_flaw_meta(batch)
        for idx, meta in enumerate(meta_list):
            index.append((batch, idx, source_crack(meta)))
    return index
