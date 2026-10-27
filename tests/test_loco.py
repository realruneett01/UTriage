"""Tests for leave-one-crack-out (LOCO) dataset and fold partitioning."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from experiments.run_loco import (  # noqa: E402
    RUN_NOTE,
    EnsembleFoldContext,
    SingleFoldContext,
    _ensemble_probabilities,
    _eval_ensemble_fold,
    _file_mtime,
    _file_sha256,
    _parse_args,
    _run_ensemble_fold_workflow,
    _run_ensemble_workflow,
    _run_eval_single_workflow,
)

from utriage.ml.datasets import (  # noqa: E402
    MlNdtDataset,
    list_batches,
    split_holdout,
)
from utriage.ml.loco import (  # noqa: E402
    PHYSICAL_CRACKS,
    IndexedMlNdtDataset,
    LocoFold,
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
    raw = rng.integers(100, 3000, size=(len(flaws), 256, 256), dtype=np.uint16)
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


@pytest.fixture
def loco_data(fake_mlndt_root: Path) -> tuple[list[Path], list[Path], list[LocoFold]]:
    """Return training batches, validation batches, and LOCO folds for fake ML-NDT data."""
    tr = list_batches("training", root=fake_mlndt_root)
    val = list_batches("validation", root=fake_mlndt_root)
    return tr, val, make_folds(tr, val, seed=0)


@pytest.fixture
def loco_folds(loco_data: tuple[list[Path], list[Path], list[LocoFold]]) -> list[LocoFold]:
    """Return folds from loco_data."""
    return loco_data[2]


@pytest.fixture
def first_fold(loco_folds: list[LocoFold]) -> LocoFold:
    """Return first LOCO fold."""
    return loco_folds[0]


@pytest.fixture
def crack_map(
    loco_data: tuple[list[Path], list[Path], list[LocoFold]],
) -> dict[tuple[Path, int], str | None]:
    """Build mapping of (batch_path, index) to crack name."""
    tr, val, _ = loco_data
    return {(b, idx): c for b, idx, c in image_index(tr + val)}


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    """Provide isolated models directory under tmp_path."""
    d = tmp_path / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


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


def _mock_eval_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    logit_val: float = 0.0,
) -> None:
    """Mock model loading and Part-A evaluation for offline testing."""
    monkeypatch.setattr(
        "experiments.run_loco.load_model",
        lambda p, device="auto": (_StubModel(logit_val), {}),
    )
    monkeypatch.setattr(
        "experiments.run_loco._eval_part_a",
        lambda *_: {"status": "ok", "flag": "test"},
    )


def _write_weight(path: Path, content: bytes) -> None:
    """Write binary content to disk."""
    path.write_bytes(content)


def _setup_eval_single_mocks(
    monkeypatch: pytest.MonkeyPatch, models_dir: Path, folds: list[LocoFold]
) -> None:
    """Configure stub models and disallow training calls during evaluation tests."""
    monkeypatch.setattr(
        "experiments.run_loco.train_model",
        lambda *_: pytest.fail("train_model should not be called in eval-single mode"),
    )
    _mock_eval_dependencies(monkeypatch)
    if len(folds) > 0:
        _write_weight(
            models_dir / f"loco_{folds[0].held_out}.pt",
            f"weights_{folds[0].held_out}".encode(),
        )
    if len(folds) > 1:
        _write_weight(
            models_dir / f"loco_{folds[1].held_out}.pt",
            f"weights_{folds[1].held_out}".encode(),
        )
    if len(folds) > 2:
        _write_weight(
            models_dir / f"loco_{folds[2].held_out}.pt",
            f"weights_{folds[2].held_out}".encode(),
        )


def _write_stub_weights(models_dir: Path, held_out: str) -> None:
    """Write dummy seed and single weights files for testing."""
    _write_weight(models_dir / f"loco_{held_out}_seed0.pt", b"seed_0")
    _write_weight(models_dir / f"loco_{held_out}_seed1.pt", b"seed_1")
    _write_weight(models_dir / f"loco_{held_out}.pt", b"single_best_bytes")


def _verify_single_best_and_traceability(rec: dict[str, Any], held_out: str) -> None:
    """Check single-best comparison values and ensemble seed traceability."""
    sb = rec["single_best"]
    assert sb is not None and sb["held_out"] == held_out
    assert sb["model_sha256"] == hashlib.sha256(b"single_best_bytes").hexdigest()
    assert sb["best_val_loss"] is None and len(rec["model_mtime"]) == 2
    assert sb["note"] == "evaluated from saved weights, no training in this run"
    assert rec["model_sha256"] == [
        hashlib.sha256(b"seed_0").hexdigest(),
        hashlib.sha256(b"seed_1").hexdigest(),
    ]


def _verify_eval_single_results(res: dict[str, Any], n_folds: int) -> None:
    """Verify structural fields of --eval-single output record."""
    assert res["run_note"] == RUN_NOTE
    assert len(res["folds"]) == n_folds
    for entry in res["folds"]:
        assert entry["best_val_loss"] is None and entry["seconds"] is None
        assert entry["note"] == "evaluated from saved weights, no training in this run"
        assert len(entry["model_sha256"]) == 64
        assert isinstance(entry["model_mtime"], str) and len(entry["model_mtime"]) > 0


def _parse_cli(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> Any:
    """Mock sys.argv and call _parse_args."""
    monkeypatch.setattr(sys, "argv", ["run_loco.py", *args])
    return _parse_args()


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
    ds = IndexedMlNdtDataset([(bpath, 1), (bpath, 2)])

    assert len(ds) == 2
    x_pos, _, _ = ds[1]
    x_neg, _, _ = ds[-1]
    assert torch.equal(x_pos, x_neg)

    for idx in (2, -3):
        with pytest.raises(IndexError):
            _ = ds[idx]


@pytest.mark.parametrize("crack", PHYSICAL_CRACKS)
def test_folds_disjoint_and_held_out_absent(
    loco_folds: list[LocoFold],
    crack_map: dict[tuple[Path, int], str | None],
    crack: str,
) -> None:
    """Verify fold train, val, and test are mutually disjoint and crack is held out."""
    fold = next(f for f in loco_folds if f.held_out == crack)
    s_tr, s_val, s_te = set(fold.train), set(fold.val), set(fold.test)

    assert all([s_tr.isdisjoint(s_val), s_tr.isdisjoint(s_te), s_val.isdisjoint(s_te)])
    assert all(crack_map[it] != crack for it in fold.train)
    assert all(crack_map[it] != crack for it in fold.val)


@pytest.mark.parametrize("crack", PHYSICAL_CRACKS)
def test_unflawed_test_only_from_holdout_batches(
    loco_data: tuple[list[Path], list[Path], list[LocoFold]],
    crack_map: dict[tuple[Path, int], str | None],
    crack: str,
) -> None:
    """Verify unflawed test images originate strictly from holdout training batches."""
    tr, _, folds = loco_data
    _, holdout_batches = split_holdout(tr, test_frac=0.1, seed=0)
    holdout_set = set(holdout_batches)
    fold = next(f for f in folds if f.held_out == crack)

    unflawed_test = [it for it in fold.test if crack_map[it] is None]
    assert len(unflawed_test) > 0
    assert all(bins in holdout_set for bins, _ in unflawed_test)


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
    arrs = [np.array(p, dtype=np.float32) for p in probs_input]
    res = _ensemble_probabilities(arrs)
    assert np.allclose(res, np.array(expected, dtype=np.float32))


def test_ensemble_averaging_on_stub_models(first_fold: LocoFold) -> None:
    """Verify ensemble evaluation averages probabilities across stub models."""
    models = [_StubModel(-2.0), _StubModel(0.0), _StubModel(2.0)]
    ev = _eval_ensemble_fold(models, first_fold)

    assert ev["n_test_flawed"] > 0
    assert ev["n_test_unflawed"] > 0
    assert all(k in ev for k in ("detections", "false_calls", "pod"))


@pytest.mark.parametrize("with_single_best", [False, True])
def test_eval_only_loads_seed_weights(
    monkeypatch: pytest.MonkeyPatch,
    models_dir: Path,
    first_fold: LocoFold,
    with_single_best: bool,
) -> None:
    """Verify --eval-only skips training and loads seed weights and optional single-best."""
    if with_single_best:
        _write_weight(models_dir / f"loco_{first_fold.held_out}.pt", b"dummy")

    loaded_paths: list[Path] = []

    def mock_load(path: Path, device: str = "auto") -> tuple[torch.nn.Module, dict]:
        loaded_paths.append(Path(path))
        return _StubModel(0.0), {}

    monkeypatch.setattr("experiments.run_loco.load_model", mock_load)
    monkeypatch.setattr(
        "experiments.run_loco._run_single_seed",
        lambda *_: pytest.fail("Training in eval-only"),
    )

    ctx = EnsembleFoldContext(1, [0, 1, 2], "cpu", eval_only=True, models_dir=models_dir)
    rec = _run_ensemble_fold_workflow(first_fold, ctx, single_best_data=None)

    assert len(loaded_paths) == (4 if with_single_best else 3)
    assert all(p.name.startswith(f"loco_{first_fold.held_out}_seed") for p in loaded_paths[:3])
    assert (rec["single_best"] is not None) is with_single_best
    assert rec["seeds"] == [0, 1, 2]


def test_eval_only_with_single_best_comparison(
    monkeypatch: pytest.MonkeyPatch,
    models_dir: Path,
    first_fold: LocoFold,
) -> None:
    """Verify single-best comparison is attached when loco_metrics.json exists."""
    _mock_eval_dependencies(monkeypatch)
    fake_sb = {
        "folds": [
            {
                "held_out": first_fold.held_out,
                "best_val_loss": 0.005,
                "detections": {"rate": 0.95},
            }
        ]
    }
    ctx = EnsembleFoldContext(1, [0], "cpu", eval_only=True, models_dir=models_dir)
    rec = _run_ensemble_fold_workflow(first_fold, ctx, single_best_data=fake_sb)

    assert rec["single_best"] == fake_sb["folds"][0]
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
    args = _parse_cli(monkeypatch, cli_args)
    assert args.ensemble is expected_ens
    assert args.eval_only is expected_eval


def test_eval_single_cli_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI flag --eval-single parses correctly."""
    args = _parse_cli(monkeypatch, ["--eval-single"])
    assert args.eval_single is True
    assert not any([args.ensemble, args.eval_only])


def test_ensemble_workflow_writes_json(
    monkeypatch: pytest.MonkeyPatch,
    loco_data: tuple[list[Path], list[Path], list[LocoFold]],
    models_dir: Path,
) -> None:
    """Verify _run_ensemble_workflow writes loco_ensemble_metrics.json structure."""
    _, val, folds = loco_data
    _mock_eval_dependencies(monkeypatch)

    out_file = models_dir.parent / "loco_ensemble_metrics.json"
    ctx = EnsembleFoldContext(1, [0, 1], "cpu", eval_only=True, models_dir=models_dir)
    metrics = _run_ensemble_workflow(folds, ctx, val, out_path=out_file)

    assert out_file.exists() and metrics["run_note"] == RUN_NOTE
    assert metrics["label"] == "Real data (ML-NDT)" and metrics["seeds"] == [0, 1]
    assert len(metrics["folds"]) == len(folds) and all("seeds" in f for f in metrics["folds"])


def test_file_sha256_helper_on_tmp_file(tmp_path: Path) -> None:
    """Verify _file_sha256 computes correct sha256 digest on binary and missing files."""
    data = b"sample model weights binary data\x00\x01\x02\x03"
    target = tmp_path / "model.pt"
    _write_weight(target, data)

    expected = hashlib.sha256(data).hexdigest()
    assert _file_sha256(target) == expected
    assert _file_sha256(tmp_path / "missing.pt") == ""


def test_file_mtime_helper_on_tmp_file(tmp_path: Path) -> None:
    """Verify _file_mtime returns valid ISO timestamp string in UTC."""
    target = tmp_path / "weights.pt"
    _write_weight(target, b"sample")
    iso_time = _file_mtime(target)
    assert isinstance(iso_time, str) and len(iso_time) > 10
    assert _file_mtime(tmp_path / "missing.pt") == ""


def test_eval_single_path_calls_no_training(
    monkeypatch: pytest.MonkeyPatch,
    loco_data: tuple[list[Path], list[Path], list[LocoFold]],
    models_dir: Path,
) -> None:
    """Verify --eval-single skips training and evaluates saved models."""
    _, val, folds = loco_data
    _setup_eval_single_mocks(monkeypatch, models_dir, folds)

    out_file = models_dir.parent / "loco_metrics.json"
    ctx = SingleFoldContext(dev_name="cpu", models_dir=models_dir, out_path=out_file)
    res = _run_eval_single_workflow(folds, val, ctx=ctx)

    assert out_file.exists()
    _verify_eval_single_results(res, len(folds))


def test_single_best_comparison_computed_from_current_weights(
    monkeypatch: pytest.MonkeyPatch,
    first_fold: LocoFold,
    models_dir: Path,
) -> None:
    """Verify single-best comparison in ensemble evaluation uses current weights."""
    _write_stub_weights(models_dir, first_fold.held_out)
    _mock_eval_dependencies(monkeypatch, logit_val=1.5)

    ctx = EnsembleFoldContext(1, [0, 1], "cpu", eval_only=True, models_dir=models_dir)
    rec = _run_ensemble_fold_workflow(first_fold, ctx)
    _verify_single_best_and_traceability(rec, first_fold.held_out)


def test_guard_no_model_loading_under_repo_models(
    monkeypatch: pytest.MonkeyPatch,
    first_fold: LocoFold,
    models_dir: Path,
) -> None:
    """Verify test-time load paths resolve under tmp_path, never repository models."""
    _write_weight(models_dir / f"loco_{first_fold.held_out}.pt", b"dummy")
    repo_models = Path("models").resolve()
    base_tmp = models_dir.parent.resolve()

    def guarded_load(path: Path | str, device: str = "auto") -> tuple[torch.nn.Module, dict]:
        p = Path(path).resolve()
        assert not p.is_relative_to(repo_models), f"Resolved under repo models: {p}"
        assert p.is_relative_to(base_tmp), f"Path not under tmp_path: {p}"
        return _StubModel(0.0), {}

    monkeypatch.setattr("experiments.run_loco.load_model", guarded_load)
    ctx = EnsembleFoldContext(1, [0, 1], "cpu", eval_only=True, models_dir=models_dir)
    rec = _run_ensemble_fold_workflow(first_fold, ctx)
    assert rec["held_out"] == first_fold.held_out

    with pytest.raises(AssertionError, match="Resolved under repo models"):
        bad_ctx = EnsembleFoldContext(1, [0], "cpu", eval_only=True)
        _run_ensemble_fold_workflow(first_fold, bad_ctx)
