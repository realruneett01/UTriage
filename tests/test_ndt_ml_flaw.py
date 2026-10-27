"""Tests for NDT_ML_Flaw I/O loader and domain mapping (project plan Phase 8).

Pure synthetic tests: no real dataset archives or trained models required.
"""

from __future__ import annotations

import lzma
import subprocess
import sys
from pathlib import Path
from typing import Any

# Prioritize local worktree source tree over editable installs
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        sys.modules["utriage"].__path__.insert(0, _pkg_dir)
if "utriage.io" in sys.modules:
    _io_dir = str(Path(_SRC) / "utriage" / "io")
    if _io_dir not in sys.modules["utriage.io"].__path__:
        sys.modules["utriage.io"].__path__.insert(0, _io_dir)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from utriage.io.ndt_ml_flaw import (  # noqa: E402
    N_SCAN,
    N_TIME,
    is_real_batch,
    iter_images,
    list_real_batches,
    read_meta,
    to_mlndt_like,
)


def _write_synthetic_xz(path: Path, count: int = 3) -> list[int]:
    """Write synthetic .xz archive with constant-value images."""
    values = [10 * (i + 1) for i in range(count)]
    chunks = [np.full((N_SCAN, N_TIME), val, dtype="<i2").tobytes() for val in values]
    with lzma.open(path, "wb") as fh:
        fh.write(b"".join(chunks))
    return values


def _write_synthetic_txt(path: Path) -> None:
    """Write synthetic metadata text file."""
    lines = [
        "1\t0.472899\t1681.46\t210.52\t2.0\t1\tP41_01\n",
        "0\t0.823631\t0.00\t0.00\t0.0\t1\t-\n",
        "1\t0.662265\t1624.03\t226.58\t6.0\t1\tP41_06_notch\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


def test_read_meta_parses_records(tmp_path: Path) -> None:
    """read_meta accurately extracts flaw, sizes, and flaw types."""
    meta_path = tmp_path / "batch_013.txt"
    _write_synthetic_txt(meta_path)
    records = read_meta(meta_path)
    assert len(records) == 3
    assert records[0]["flaw"] == 1
    assert records[0]["flaw_type"] == "P41_01"
    assert records[1]["flaw"] == 0
    assert records[1]["flaw_type"] == "-"
    assert records[2]["flaw"] == 1
    assert records[2]["flaw_size"] == 6.0


def test_iter_images_sequential(tmp_path: Path) -> None:
    """iter_images streams correct (480, 7168) int16 arrays in sequence."""
    xz_path = tmp_path / "batch_013.xz"
    expected_vals = _write_synthetic_xz(xz_path, count=3)
    yielded = list(iter_images(xz_path))
    assert len(yielded) == 3
    for img, exp_val in zip(yielded, expected_vals, strict=True):
        assert img.shape == (N_SCAN, N_TIME)
        assert img.dtype == np.int16
        assert int(img[0, 0]) == exp_val


def test_iter_images_slice_bounds(tmp_path: Path) -> None:
    """iter_images respects start and stop slicing."""
    xz_path = tmp_path / "batch_013.xz"
    _write_synthetic_xz(xz_path, count=3)
    slice_yield = list(iter_images(xz_path, start=1, stop=2))
    assert len(slice_yield) == 1
    assert int(slice_yield[0][0, 0]) == 20


def test_to_mlndt_like_shape_and_finite() -> None:
    """to_mlndt_like produces finite (256, 256) float32 array with unit variance."""
    rng = np.random.default_rng(42)
    raw = rng.integers(-500, 500, size=(N_SCAN, N_TIME), dtype=np.int16)
    raw[:, 1500:1700] += 2000
    out = to_mlndt_like(raw)
    assert out.shape == (256, 256)
    assert out.dtype == np.float32
    assert bool(np.all(np.isfinite(out)))
    assert abs(float(np.mean(out))) < 1e-4
    assert abs(float(np.std(out)) - 1.0) < 1e-4


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("batch_013", True),
        ("batch_018", True),
        ("batch_019", True),
        ("batch_201", False),
        ("batch_208", False),
        ("batch_001", False),
        ("sample_random", False),
    ],
)
def test_is_real_batch_classification(name: str, expected: bool) -> None:
    """is_real_batch flags 013-019 as real and rejects 2xx simulated batches."""
    assert is_real_batch(name) is expected


def test_batch_selector_excludes_2xx(tmp_path: Path) -> None:
    """list_real_batches includes only real 013-019 pairs and excludes 2xx."""
    candidates = [
        "batch_013.xz",
        "batch_013.txt",
        "batch_019.xz",
        "batch_019.txt",
        "batch_201.xz",
        "batch_201.txt",
        "batch_202.lzma",
    ]
    for c in candidates:
        (tmp_path / c).touch()
    pairs = list_real_batches(tmp_path)
    pair_stems = [(xz.stem, txt.stem) for xz, txt in pairs]
    expected_stems = [("batch_013", "batch_013"), ("batch_019", "batch_019")]
    assert pair_stems == expected_stems


def test_repo_parent_root_timeout_records_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _repo_parent_root records/returns 'unknown' on subprocess timeout."""
    from experiments.run_cross_dataset import _repo_parent_root

    def _timeout_run(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd=["git"], timeout=30)

    monkeypatch.setattr(subprocess, "run", _timeout_run)
    assert _repo_parent_root() == "unknown"


def test_repo_parent_root_passes_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify _repo_parent_root invokes subprocess with timeout=30."""
    from experiments.run_cross_dataset import _repo_parent_root

    calls: list[dict[str, Any]] = []

    def _spy_run(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=str(tmp_path / ".git"))

    monkeypatch.setattr(subprocess, "run", _spy_run)
    res = _repo_parent_root()
    assert res == tmp_path
    assert len(calls) == 1
    assert calls[0].get("timeout") == 30


def test_evaluate_anomaly_weights_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify _evaluate_anomaly passes weights_only=True explicitly to torch.load."""
    from experiments.run_cross_dataset import _evaluate_anomaly

    from utriage.ml.anomaly import ConvAE2D

    ckpt_path = tmp_path / "anomaly.pt"
    ckpt_path.touch()
    calls: list[dict[str, Any]] = []
    dummy_state = ConvAE2D().state_dict()

    def _fake_load(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return {"state_dict": dummy_state}

    monkeypatch.setattr(torch, "load", _fake_load)
    images = np.zeros((1, 256, 256), dtype=np.float32)
    _evaluate_anomaly(ckpt_path, images, torch.device("cpu"))
    assert len(calls) == 1
    assert calls[0].get("weights_only") is True
