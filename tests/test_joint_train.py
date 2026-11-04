"""Tests for joint synthetic-real training and pretraining experiment (project plan Step 3.1 & 3.3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from experiments.run_joint_train import (
    JointConfig,
    MlNdtAE,
    SimDatasetAdapter,
    build_arg_parser,
    calc_detection,
    calc_false_calls,
    eval_pod,
    parse_cli_args,
    resolve_sim_count,
    run_joint_experiment,
)

from utriage.ml.models import MlNdtNet
from utriage.ml.pretrain import encoder_state
from utriage.ml.sim_to_2d import SimBscanConfig, SimBscanDataset


def _create_fake_batch(
    batch_path: Path,
    flaws: list[dict[str, Any]],
    labels: list[tuple[float, float]],
    seed: int,
) -> Path:
    """Write fake .bins, .jsons, and .labels files for a single batch."""
    rng = np.random.default_rng(seed)
    raw = rng.integers(100, 3000, size=(len(flaws), 256, 256), dtype=np.uint16)
    raw.tofile(batch_path)
    jsons_text = "".join(json.dumps({"flaws": [f]}) for f in flaws)
    batch_path.with_suffix(".jsons").write_text(jsons_text, encoding="utf-8")
    labels_arr = np.array(labels, dtype=np.float32)
    np.savetxt(batch_path.with_suffix(".labels"), labels_arr, delimiter="\t")
    return batch_path


@pytest.fixture
def fake_mlndt_root(tmp_path: Path) -> Path:
    """Populate minimal fake ML-NDT root with training and validation batches."""
    root = tmp_path / "fake_mlndt"
    tr_dir, val_dir = root / "training", root / "validation"
    tr_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    flaws = [{"size": "1.6"}, {"size": "4.0"}, {"size": "8.6"}, {"size": "0.0"}]
    labels = [(1.0, 1.6), (1.0, 4.0), (1.0, 8.6), (0.0, 0.0)]

    for i in range(10):
        _create_fake_batch(tr_dir / f"batch_tr_{i:02d}.bins", flaws, labels, seed=i)
    for j in range(2):
        _create_fake_batch(val_dir / f"batch_val_{j:02d}.bins", flaws, labels, seed=100 + j)
    return root


def test_autoencoder_shapes_and_transfer() -> None:
    """MlNdtAE reconstructs 256x256 image and its encoder transfers to MlNdtNet."""
    x = torch.randn(2, 1, 256, 256)
    ae = MlNdtAE(width=16, first_pool=(2, 2))
    recon = ae(x)
    assert recon.shape == x.shape

    enc = encoder_state(ae)
    net = MlNdtNet(width=16, first_pool=(2, 2))
    incompat = net.load_state_dict(enc, strict=False)
    assert len(incompat.unexpected_keys) == 0
    assert set(incompat.missing_keys) == {"head.weight", "head.bias"}


def test_sim_adapter_yields_three_elements() -> None:
    """SimDatasetAdapter adapts SimBscanDataset to yield (x, flaw, size)."""
    cfg = SimBscanConfig(length_mm=8.0, axial_step_mm=0.5)
    sim_ds = SimBscanDataset(length=2, base_seed=12, cfg=cfg)
    adapter = SimDatasetAdapter(sim_ds, flaw_size=1.5)
    assert len(adapter) == 2

    x, flaw, size = adapter[0]
    assert x.shape == (1, 256, 256)
    assert flaw in (0.0, 1.0)
    assert size == (1.5 if flaw == 1.0 else 0.0)


@pytest.mark.parametrize(
    ("frac", "n_sim", "n_real", "expected"),
    [
        (0.0, 100, 50, 0),
        (-0.5, 100, 50, 0),
        (0.5, 0, 50, 0),
        (0.5, 10, None, 5),
        (0.2, 100, 50, 10),
    ],
)
def test_resolve_sim_count(frac: float, n_sim: int, n_real: int | None, expected: int) -> None:
    """resolve_sim_count enforces 0 for non-positive fraction and scales otherwise."""
    assert resolve_sim_count(frac, n_sim, n_real) == expected


def _assert_detection_fields(det: dict[str, Any]) -> None:
    """Verify keys and bounds of detection dictionary."""
    assert {"k", "n", "rate", "lower95", "upper95"}.issubset(det.keys())
    assert 0.0 <= det["lower95"] <= det["upper95"] <= 1.0


def _assert_false_call_fields(fc: dict[str, Any]) -> None:
    """Verify keys and bounds of false calls dictionary."""
    assert {"k", "n", "rate", "upper95"}.issubset(fc.keys())
    assert 0.0 <= fc["rate"] <= fc["upper95"] <= 1.0


def test_metric_helpers() -> None:
    """Verify detection, false call, and POD estimation helper functions."""
    hits = np.array([1, 1, 0, 1, 0], dtype=float)
    flawed = np.array([True, True, True, False, False])
    unflawed = ~flawed
    sizes = np.array([1.5, 2.0, 3.0], dtype=float)

    det = calc_detection(hits, flawed)
    _assert_detection_fields(det)
    assert det["k"] == 2
    assert det["n"] == 3

    fc = calc_false_calls(hits, unflawed)
    _assert_false_call_fields(fc)
    assert fc["k"] == 1
    assert fc["n"] == 2

    pod = eval_pod(sizes, hits[flawed])
    assert "a90_95" in pod


def _assert_quick_metrics(metrics: dict[str, Any], models_dir: Path) -> None:
    """Verify quick run metrics dictionary and confirm no models on disk."""
    expected = "real metrics on ML-NDT (weld, 3 physical cracks); sim images are Synthetic data"
    assert metrics["label"] == expected
    assert set(metrics["arms"].keys()) == {"A", "B", "C", "D"}
    assert len(metrics["folds"]) == 1
    assert len(metrics["model_sha256"]) == 4
    assert not models_dir.exists() or len(list(models_dir.glob("*.pt"))) == 0


def test_quick_all_arms_and_json(fake_mlndt_root: Path, tmp_path: Path) -> None:
    """Run quick pipeline on 1 fold across all 4 arms and verify JSON schema."""
    out_dir = tmp_path / "results"
    models_dir = tmp_path / "models"
    cfg = JointConfig(
        sim_fraction=0.5,
        n_sim=4,
        pretrain_epochs=1,
        seed=42,
        quick=True,
        out_dir=out_dir,
        epochs=1,
        arms=("A", "B", "C", "D"),
        data_root=fake_mlndt_root,
        models_dir=models_dir,
        device="cpu",
    )
    metrics = run_joint_experiment(cfg)
    assert (out_dir / "joint_train_metrics.json").exists()
    _assert_quick_metrics(metrics, models_dir)


def test_sim_fraction_zero_no_sim_samples(fake_mlndt_root: Path, tmp_path: Path) -> None:
    """Verify sim_fraction=0 yields 0 synthetic samples and preserves dataset size."""
    out_dir = tmp_path / "zero_results"
    cfg = JointConfig(
        sim_fraction=0.0,
        n_sim=10,
        pretrain_epochs=1,
        seed=0,
        quick=True,
        out_dir=out_dir,
        epochs=1,
        arms=("A", "B"),
        data_root=fake_mlndt_root,
        device="cpu",
    )
    metrics = run_joint_experiment(cfg)
    fold_arms = metrics["folds"][0]["arms"]
    assert fold_arms["A"]["n_train"] == fold_arms["B"]["n_train"]


def test_cli_parsing(tmp_path: Path) -> None:
    """Verify build_arg_parser and parse_cli_args support all specified flags."""
    parser = build_arg_parser()
    argv = [
        "--quick",
        "--sim-fraction",
        "0.25",
        "--pretrain-epochs",
        "3",
        "--seed",
        "7",
        "--out-dir",
        str(tmp_path),
        "--arms",
        "A",
        "C",
    ]
    args = parser.parse_args(argv)
    cfg = parse_cli_args(args)
    assert cfg.quick is True
    assert cfg.sim_fraction == 0.25
    assert cfg.pretrain_epochs == 3
    assert cfg.seed == 7
    assert cfg.out_dir == tmp_path
    assert cfg.arms == ("A", "C")
