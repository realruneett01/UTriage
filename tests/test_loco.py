"""Tests for leave-one-crack-out (LOCO) dataset and fold partitioning."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        del sys.modules["utriage"]

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from utriage.ml.datasets import (  # noqa: E402
    MlNdtDataset,
    list_batches,
    split_holdout,
)
from utriage.ml.loco import (  # noqa: E402
    PHYSICAL_CRACKS,
    IndexedMlNdtDataset,
    make_folds,
)
from utriage.ml.mlndt_meta import image_index  # noqa: E402


def _create_fake_batch(
    batch_path: Path,
    flaws: list[dict[str, Any]],
    labels: list[tuple[float, float]],
    seed: int,
) -> Path:
    """Write fake .bins, .jsons, and .labels files for a single batch."""
    rng = np.random.default_rng(seed)
    n = len(flaws)
    raw = rng.integers(100, 3000, size=(n, 256, 256), dtype=np.uint16)
    raw.tofile(batch_path)

    jsons_text = "".join(json.dumps({"flaws": [f]}) for f in flaws)
    batch_path.with_suffix(".jsons").write_text(jsons_text, encoding="utf-8")

    labels_arr = np.array(labels, dtype=np.float32)
    np.savetxt(batch_path.with_suffix(".labels"), labels_arr, delimiter="\t")
    return batch_path


@pytest.fixture
def fake_mlndt_root(tmp_path: Path) -> Path:
    """Populate fake ML-NDT root with training and validation batches."""
    root = tmp_path / "fake_mlndt"
    tr_dir = root / "training"
    val_dir = root / "validation"
    tr_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    flaws = [{"size": "1.6"}, {"size": "4.0"}, {"size": "8.6"}, {"size": "0.0"}]
    labels = [(1.0, 1.6), (1.0, 4.0), (1.0, 8.6), (0.0, 0.0)]

    for i in range(10):
        _create_fake_batch(tr_dir / f"batch_tr_{i:02d}.bins", flaws, labels, seed=i)
    for j in range(2):
        _create_fake_batch(val_dir / f"batch_val_{j:02d}.bins", flaws, labels, seed=100 + j)
    return root


@pytest.mark.parametrize("size", [256, 128])
def test_indexed_dataset_matches_mlndt_dataset(fake_mlndt_root: Path, size: int) -> None:
    """Verify IndexedMlNdtDataset produces identical tensors to MlNdtDataset."""
    bpath = fake_mlndt_root / "training" / "batch_tr_00.bins"
    full_ds = MlNdtDataset([bpath], size=size)
    items = [(bpath, 0), (bpath, 3)]
    sub_ds = IndexedMlNdtDataset(items, size=size)

    assert len(sub_ds) == 2
    for sub_idx, full_idx in enumerate([0, 3]):
        x_sub, flaw_sub, size_sub = sub_ds[sub_idx]
        x_full, flaw_full, size_full = full_ds[full_idx]
        assert torch.allclose(x_sub, x_full)
        assert torch.equal(flaw_sub, flaw_full)
        assert torch.equal(size_sub, size_full)


def test_indexed_dataset_indexing_bounds(fake_mlndt_root: Path) -> None:
    """Verify negative indexing and IndexError on out-of-bounds access."""
    bpath = fake_mlndt_root / "training" / "batch_tr_00.bins"
    items = [(bpath, 1), (bpath, 2)]
    ds = IndexedMlNdtDataset(items)

    assert len(ds) == 2
    x_pos, _, _ = ds[1]
    x_neg, _, _ = ds[-1]
    assert torch.equal(x_pos, x_neg)

    with pytest.raises(IndexError):
        _ = ds[2]
    with pytest.raises(IndexError):
        _ = ds[-3]


@pytest.mark.parametrize("crack", PHYSICAL_CRACKS)
def test_folds_disjoint_and_held_out_absent(fake_mlndt_root: Path, crack: str) -> None:
    """Verify fold train, val, and test are mutually disjoint and crack is held out."""
    tr_batches = list_batches("training", root=fake_mlndt_root)
    val_batches = list_batches("validation", root=fake_mlndt_root)
    folds = make_folds(tr_batches, val_batches, seed=0)

    fold = next(f for f in folds if f.held_out == crack)
    crack_map = {(b, idx): c for b, idx, c in image_index(tr_batches + val_batches)}

    s_tr = set(fold.train)
    s_val = set(fold.val)
    s_te = set(fold.test)

    assert s_tr.isdisjoint(s_val)
    assert s_tr.isdisjoint(s_te)
    assert s_val.isdisjoint(s_te)
    assert all(crack_map[it] != crack for it in fold.train)
    assert all(crack_map[it] != crack for it in fold.val)


@pytest.mark.parametrize("crack", PHYSICAL_CRACKS)
def test_unflawed_test_only_from_holdout_batches(fake_mlndt_root: Path, crack: str) -> None:
    """Verify unflawed test images originate strictly from holdout training batches."""
    tr_batches = list_batches("training", root=fake_mlndt_root)
    val_batches = list_batches("validation", root=fake_mlndt_root)
    _, holdout_batches = split_holdout(tr_batches, test_frac=0.1, seed=0)
    holdout_set = set(holdout_batches)

    folds = make_folds(tr_batches, val_batches, seed=0)
    fold = next(f for f in folds if f.held_out == crack)
    crack_map = {(b, idx): c for b, idx, c in image_index(tr_batches + val_batches)}

    unflawed_test = [it for it in fold.test if crack_map[it] is None]
    assert len(unflawed_test) > 0
    assert all(bins in holdout_set for bins, _ in unflawed_test)


class _StubModel(torch.nn.Module):
    """Stub torch model returning fixed logits for testing."""

    def __init__(self, logit_val: float) -> None:
        super().__init__()
        self.logit_val = logit_val

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        logits = torch.full((b, 1), self.logit_val, dtype=torch.float32)
        sizes = torch.zeros((b, 1), dtype=torch.float32)
        return logits, sizes


@pytest.mark.parametrize(
    ("probs_input", "expected"),
    [
        ([[0.8, 0.2], [0.6, 0.4], [0.7, 0.3]], [0.7, 0.3]),
        ([[1.0, 0.0], [0.5, 0.5]], [0.75, 0.25]),
    ],
)
def test_ensemble_probabilities_averaging(
    probs_input: list[list[float]], expected: list[float]
) -> None:
    """Verify ensemble probability helper computes exact mean across seeds."""
    from experiments.run_loco import _ensemble_probabilities

    arrs = [np.array(p, dtype=np.float32) for p in probs_input]
    res = _ensemble_probabilities(arrs)
    assert np.allclose(res, np.array(expected, dtype=np.float32))


def test_ensemble_averaging_on_stub_models(fake_mlndt_root: Path) -> None:
    """Verify ensemble evaluation averages probabilities across stub models."""
    from experiments.run_loco import _eval_ensemble_fold

    tr_batches = list_batches("training", root=fake_mlndt_root)
    val_batches = list_batches("validation", root=fake_mlndt_root)
    folds = make_folds(tr_batches, val_batches, seed=0)
    fold = folds[0]

    models = [_StubModel(-2.0), _StubModel(0.0), _StubModel(2.0)]
    ev = _eval_ensemble_fold(models, fold)

    assert ev["n_test_flawed"] > 0
    assert ev["n_test_unflawed"] > 0
    assert "detections" in ev and "false_calls" in ev and "pod" in ev


def test_eval_only_loads_seed_weights(
    monkeypatch: pytest.MonkeyPatch, fake_mlndt_root: Path
) -> None:
    """Verify --eval-only skips training and loads existing seed checkpoints."""
    from experiments.run_loco import (
        EnsembleFoldContext,
        _run_ensemble_fold_workflow,
    )

    tr_batches = list_batches("training", root=fake_mlndt_root)
    val_batches = list_batches("validation", root=fake_mlndt_root)
    fold = make_folds(tr_batches, val_batches, seed=0)[0]

    loaded_paths: list[Path] = []

    def mock_load(path: Path, device: str = "auto") -> tuple[torch.nn.Module, dict]:
        loaded_paths.append(Path(path))
        return _StubModel(0.0), {}

    def fail_train(*_: Any) -> Any:
        pytest.fail("Training should not be called in eval-only mode")

    monkeypatch.setattr("experiments.run_loco.load_model", mock_load)
    monkeypatch.setattr("experiments.run_loco._run_single_seed", fail_train)

    ctx = EnsembleFoldContext(
        epochs=1,
        seeds=[0, 1, 2],
        dev_name="cpu",
        eval_only=True,
    )
    rec = _run_ensemble_fold_workflow(fold, ctx, single_best_data=None)

    assert len(loaded_paths) == 3
    assert all(p.name.startswith(f"loco_{fold.held_out}_seed") for p in loaded_paths)
    assert rec["seeds"] == [0, 1, 2]
    assert rec["single_best"] is None


def test_eval_only_with_single_best_comparison(
    monkeypatch: pytest.MonkeyPatch, fake_mlndt_root: Path
) -> None:
    """Verify single-best comparison is attached when loco_metrics.json exists."""
    from experiments.run_loco import (
        EnsembleFoldContext,
        _run_ensemble_fold_workflow,
    )

    tr_batches = list_batches("training", root=fake_mlndt_root)
    val_batches = list_batches("validation", root=fake_mlndt_root)
    fold = make_folds(tr_batches, val_batches, seed=0)[0]

    monkeypatch.setattr(
        "experiments.run_loco.load_model",
        lambda p, device="auto": (_StubModel(0.0), {}),
    )
    fake_single_best = {
        "folds": [
            {
                "held_out": fold.held_out,
                "best_val_loss": 0.005,
                "detections": {"rate": 0.95},
            }
        ]
    }
    ctx = EnsembleFoldContext(epochs=1, seeds=[0], dev_name="cpu", eval_only=True)
    rec = _run_ensemble_fold_workflow(fold, ctx, single_best_data=fake_single_best)

    assert rec["single_best"] == fake_single_best["folds"][0]
    assert rec["best_val_loss"] == 0.005


@pytest.mark.parametrize(
    ("cli_args", "expected_ens", "expected_eval"),
    [
        ([], False, False),
        (["--ensemble"], True, False),
        (["--eval-only"], False, True),
        (["--ensemble", "--eval-only"], True, True),
    ],
)
def test_ensemble_cli_arguments(
    monkeypatch: pytest.MonkeyPatch,
    cli_args: list[str],
    expected_ens: bool,
    expected_eval: bool,
) -> None:
    """Verify CLI flags --ensemble and --eval-only parse correctly."""
    from experiments.run_loco import _parse_args

    monkeypatch.setattr(sys, "argv", ["run_loco.py", *cli_args])
    args = _parse_args()
    assert args.ensemble is expected_ens
    assert args.eval_only is expected_eval


def test_ensemble_workflow_writes_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_mlndt_root: Path
) -> None:
    """Verify _run_ensemble_workflow writes loco_ensemble_metrics.json structure."""
    from experiments.run_loco import (
        EnsembleFoldContext,
        _run_ensemble_workflow,
    )

    tr_batches = list_batches("training", root=fake_mlndt_root)
    val_batches = list_batches("validation", root=fake_mlndt_root)
    folds = make_folds(tr_batches, val_batches, seed=0)

    monkeypatch.setattr(
        "experiments.run_loco.load_model",
        lambda p, device="auto": (_StubModel(0.0), {}),
    )
    monkeypatch.setattr(
        "experiments.run_loco._eval_part_a",
        lambda _: {"status": "ok", "flag": "test"},
    )

    out_file = tmp_path / "loco_ensemble_metrics.json"
    ctx = EnsembleFoldContext(epochs=1, seeds=[0, 1], dev_name="cpu", eval_only=True)
    metrics = _run_ensemble_workflow(folds, ctx, val_batches, out_path=out_file)

    assert out_file.exists()
    assert metrics["label"] == "Real data (ML-NDT)"
    assert metrics["seeds"] == [0, 1]
    assert len(metrics["folds"]) == len(folds)
    assert all("seeds" in f for f in metrics["folds"])
