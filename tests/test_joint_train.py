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
    assert_no_held_out,
    build_arg_parser,
    build_experiment_summary,
    calc_detection,
    calc_false_calls,
    eval_pod,
    parse_cli_args,
    resolve_pretrain_dataset,
    resolve_sim_count,
    run_joint_experiment,
)

from utriage.ml.loco import IndexedMlNdtDataset, LocoFold
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
    ("frac", "n_real", "cap", "expected"),
    [
        (0.0, 50, None, 0),
        (-0.5, 50, None, 0),
        (0.5, 0, None, 0),
        (0.5, 50, None, 50),
        (0.5, 50, 10, 10),
        (0.2, 80, None, 20),
    ],
)
def test_resolve_sim_count(frac: float, n_real: int, cap: int | None, expected: int) -> None:
    """resolve_sim_count computes fraction correctly and respects optional cap."""
    assert resolve_sim_count(frac, n_real, cap) == expected


def _assert_detection_fields(det: dict[str, Any]) -> None:
    """Verify keys, interval type, and bounds of detection dictionary."""
    assert {"k", "n", "rate", "lower95", "upper95", "interval_type", "method"}.issubset(det.keys())
    assert det["interval_type"] == "two_sided_95"
    assert det["method"] == "Clopper-Pearson"
    assert 0.0 <= det["lower95"] <= det["upper95"] <= 1.0


def _assert_false_call_fields(fc: dict[str, Any]) -> None:
    """Verify keys, interval type, and bounds of false calls dictionary."""
    assert {"k", "n", "rate", "upper95", "interval_type", "method"}.issubset(fc.keys())
    assert fc["interval_type"] == "one_sided_95_upper"
    assert fc["method"] == "Clopper-Pearson"
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
    assert "method" in pod


def _generate_synthetic_pod_data(max_size: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate log-linear hit/miss data spanning up to max_size."""
    rng = np.random.default_rng(seed)
    sizes = np.linspace(0.5, max_size, 100)
    eta = -2.0 + 4.0 * np.log(sizes)
    prob = 1.0 / (1.0 + np.exp(-eta))
    hits = (rng.uniform(size=len(sizes)) < prob).astype(float)
    return sizes, hits


@pytest.mark.parametrize(
    ("max_size", "should_extrapolate"),
    [
        (10.0, False),
        (2.5, True),
    ],
)
def test_eval_pod_extrapolation_behavior(max_size: float, should_extrapolate: bool) -> None:
    """eval_pod returns a90_95 <= max_size when valid, or None with reason when extrapolating."""
    sizes, hits = _generate_synthetic_pod_data(max_size=max_size, seed=0)
    res = eval_pod(sizes, hits)
    assert "method" in res
    if should_extrapolate:
        assert res["a90_95"] is None
        assert res["reason"] == "a90_95 extrapolates beyond the largest tested flaw size"
    else:
        assert res["reason"] is None
        a90_95 = res["a90_95"]
        assert isinstance(a90_95, float)
        assert np.isfinite(a90_95)
        assert 0.0 < a90_95 <= float(np.max(sizes))


def test_arm_c_pretrain_dataset_has_no_sim(tmp_path: Path) -> None:
    """Arm C pretrains only on real images, while arm D includes sim images."""
    fake_b = tmp_path / "dummy.bins"
    fake_b.touch()
    real_ds = IndexedMlNdtDataset([(fake_b, 0), (fake_b, 1)])
    sim_ds = SimDatasetAdapter(SimBscanDataset(length=2, base_seed=1))

    ds_c = resolve_pretrain_dataset(real_ds, sim_ds, arm="C")
    ds_d = resolve_pretrain_dataset(real_ds, sim_ds, arm="D")

    assert ds_c is real_ds
    assert hasattr(ds_d, "datasets")
    assert sim_ds in ds_d.datasets


def test_held_out_assertion_detects_leak(tmp_path: Path) -> None:
    """assert_no_held_out raises AssertionError when test scan leaks into train."""
    b_tr = tmp_path / "tr.bins"
    b_te = tmp_path / "te.bins"
    b_tr.touch()
    b_te.touch()

    fold = LocoFold(held_out="crack_1.6", train=[(b_tr, 0)], val=[], test=[(b_te, 0)])
    ds_valid = IndexedMlNdtDataset([(b_tr, 0)])
    ds_leaked = IndexedMlNdtDataset([(b_te, 0)])

    assert_no_held_out(ds_valid, fold)
    with pytest.raises(AssertionError, match="Held-out"):
        assert_no_held_out(ds_leaked, fold)


def _assert_arm_record(arm_rec: dict[str, Any], arm_key: str) -> None:
    """Verify keys, interval types, and hashes in an arm record."""
    assert arm_rec["arm"] == arm_key
    _assert_detection_fields(arm_rec["detections"])
    _assert_false_call_fields(arm_rec["false_calls"])
    assert "method" in arm_rec["pod"]
    assert "model_sha256" in arm_rec
    if arm_key in ("C", "D"):
        assert isinstance(arm_rec["pretrained_encoder_sha256"], str)
        assert len(arm_rec["pretrained_encoder_sha256"]) == 64
    else:
        assert arm_rec["pretrained_encoder_sha256"] is None


def test_quick_all_arms_and_json(fake_mlndt_root: Path, tmp_path: Path) -> None:
    """Run quick pipeline across all 4 arms and verify fold metrics and JSON."""
    out_dir = tmp_path / "results"
    cfg = JointConfig(
        sim_fraction=0.5,
        n_sim_cap=4,
        pretrain_epochs=1,
        seed=42,
        quick=True,
        out_dir=out_dir,
        epochs=1,
        arms=("A", "B", "C", "D"),
        data_root=fake_mlndt_root,
        models_dir=tmp_path / "models",
        device="cpu",
    )
    metrics = run_joint_experiment(cfg)
    assert (out_dir / "joint_train_metrics.json").exists()
    assert metrics["requested_sim_fraction"] == 0.5
    assert "actual_sim_fraction" in metrics

    fold = metrics["folds"][0]
    assert "actual_sim_fraction" in fold
    assert fold["actual_sim_fraction"] == fold["n_sim_train"] / (
        fold["n_real_train"] + fold["n_sim_train"]
    )
    for arm_key in ("A", "B", "C", "D"):
        _assert_arm_record(fold["arms"][arm_key], arm_key)


def test_actual_sim_fraction_matches_requested(fake_mlndt_root: Path, tmp_path: Path) -> None:
    """Verify actual_sim_fraction matches requested fraction without cap."""
    cfg = JointConfig(
        sim_fraction=0.2,
        pretrain_epochs=1,
        quick=True,
        out_dir=tmp_path / "frac_results",
        epochs=1,
        arms=("A", "B"),
        data_root=fake_mlndt_root,
        device="cpu",
    )
    metrics = run_joint_experiment(cfg)
    assert metrics["requested_sim_fraction"] == 0.2
    assert "actual_sim_fraction" in metrics
    fold = metrics["folds"][0]
    expected_frac = fold["n_sim_train"] / (fold["n_real_train"] + fold["n_sim_train"])
    assert fold["actual_sim_fraction"] == pytest.approx(0.2, abs=0.05)
    assert fold["actual_sim_fraction"] == expected_frac


def test_multi_seeds_produces_per_seed_entries(fake_mlndt_root: Path, tmp_path: Path) -> None:
    """Configuring multiple seeds produces per-seed results while keeping single-seed keys."""
    cfg = JointConfig(
        sim_fraction=0.2,
        seeds=(0, 1),
        quick=True,
        out_dir=tmp_path / "seed_results",
        epochs=1,
        arms=("A",),
        data_root=fake_mlndt_root,
        device="cpu",
    )
    metrics = run_joint_experiment(cfg)
    assert metrics["seeds"] == [0, 1]
    arm_a = metrics["folds"][0]["arms"]["A"]
    assert "seeds" in arm_a
    assert len(arm_a["seeds"]) == 2
    assert arm_a["seeds"][0]["seed"] == 0
    assert arm_a["seeds"][1]["seed"] == 1
    assert "detections" in arm_a


def test_sim_fraction_zero_no_sim_samples(fake_mlndt_root: Path, tmp_path: Path) -> None:
    """Verify sim_fraction=0 yields 0 synthetic samples and preserves dataset size."""
    cfg = JointConfig(
        sim_fraction=0.0,
        n_sim_cap=10,
        pretrain_epochs=1,
        seed=0,
        quick=True,
        out_dir=tmp_path / "zero_results",
        epochs=1,
        arms=("A", "B"),
        data_root=fake_mlndt_root,
        device="cpu",
    )
    metrics = run_joint_experiment(cfg)
    fold = metrics["folds"][0]
    assert fold["n_sim_train"] == 0
    assert fold["actual_sim_fraction"] == 0.0
    assert fold["arms"]["A"]["n_train"] == fold["arms"]["B"]["n_train"]


def test_payload_builder_sim_fractions() -> None:
    """build_experiment_summary includes requested and actual sim fractions without training."""
    cfg = JointConfig(sim_fraction=0.25, arms=("A",))
    folds = [
        {
            "held_out": "crack_1.6",
            "n_real_train": 75,
            "n_sim_train": 25,
            "sim_fraction_reason": None,
            "arms": {"A": {"model_sha256": "sha_a"}},
        }
    ]
    summary = build_experiment_summary(cfg, folds, "cpu")
    assert summary["requested_sim_fraction"] == 0.25
    assert summary["actual_sim_fraction"] == 0.25


def test_cli_parsing(tmp_path: Path) -> None:
    """Verify build_arg_parser and parse_cli_args support seeds and n-sim-cap."""
    parser = build_arg_parser()
    argv = [
        "--quick",
        "--sim-fraction",
        "0.25",
        "--n-sim-cap",
        "50",
        "--seeds",
        "3",
        "5",
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
    assert cfg.n_sim_cap == 50
    assert cfg.seeds == (3, 5)
    assert cfg.seed == 3
    assert cfg.out_dir == tmp_path
    assert cfg.arms == ("A", "C")
