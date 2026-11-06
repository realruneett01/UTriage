"""Tests for invariant crack representation LOCO training and evaluation.

Follows TDD, code health (max 30 lines per function, CCN <= 6, max 4 parameters),
and runs on CPU in seconds using synthetic fixtures in tmp_path.
"""

from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure workspace experiments and src are on path
_TESTS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _TESTS_DIR.parent
_SRC_DIR = _ROOT_DIR / "src"
_EXP_DIR = _ROOT_DIR / "experiments"
for _p in (_ROOT_DIR, _SRC_DIR, _EXP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _mock_stretch_axis(x: torch.Tensor, factor: torch.Tensor, axis: int) -> torch.Tensor:
    """Mock stretch_axis returning input or resized tensor along axis."""
    if torch.allclose(factor, torch.ones_like(factor)):
        return x
    return x.clone()


def _mock_physics_view(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Mock physics_view applying deterministic augmentation."""
    from utriage.ml.lite_hybrid import augment_batch

    return augment_batch(x, gen)


def _mock_nt_xent_loss(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2
) -> torch.Tensor:
    """Mock SimCLR NT-Xent loss on L2-normalised [B, D] pairs."""
    b = z1.shape[0]
    if b == 0:
        return torch.tensor(0.0, device=z1.device, requires_grad=True)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    diag = torch.eye(2 * b, device=z1.device, dtype=torch.bool)
    sim = sim.masked_fill(diag, float("-inf"))
    targets = torch.cat(
        [
            torch.arange(b, 2 * b, device=z1.device),
            torch.arange(0, b, device=z1.device),
        ]
    )
    return F.cross_entropy(sim, targets)


def _mock_supcon_loss(
    z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.2
) -> torch.Tensor:
    """Mock supervised contrastive loss (Khosla et al. 2020) on normalised [N, D]."""
    n = z.shape[0]
    if n < 2:
        raise ValueError("N must be >= 2 for supcon_loss")
    lbl = labels.contiguous().view(-1, 1)
    mask = torch.eq(lbl, lbl.T).float()
    diag = torch.eye(n, device=z.device, dtype=torch.bool)
    pos_mask = mask * (~diag).float()
    sim = torch.div(torch.matmul(z, z.T), temperature)
    logits_max, _ = torch.max(sim, dim=1, keepdim=True)
    logits = sim - logits_max.detach()
    exp_logits = torch.exp(logits) * (~diag).float()
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
    valid = pos_mask.sum(1) > 0
    if not torch.any(valid):
        return torch.tensor(0.0, device=z.device, requires_grad=True)
    mean_log_prob_pos = (pos_mask * log_prob).sum(1)[valid] / pos_mask.sum(1)[valid]
    return -mean_log_prob_pos.mean()


class _MockLiteEncoder(nn.Module):
    """Mock encoder reusing LiteNet layers internally."""

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        from utriage.ml.lite_hybrid import LiteNet

        net = LiteNet(width=width)
        self.stem = net.stem
        self.block1 = net.block1
        self.block2 = net.block2
        self.block3 = net.block3
        self.block4 = net.block4
        self.pool = net.pool
        self._out_dim = 4 * width

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.pool(self.block4(self.block3(self.block2(self.block1(self.stem(x))))))
        return feat.flatten(start_dim=1)


class _MockProjectionHead(nn.Module):
    """Mock projection head mapping features to L2-normalised space."""

    def __init__(self, in_dim: int, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


class _MockInvariantClassifier(nn.Module):
    """Mock classifier wrapping LiteEncoder with linear readout head."""

    def __init__(self, encoder: _MockLiteEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(-1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def _install_mock_invariance_if_missing() -> None:
    """Install mock utriage.ml.invariance module if not yet created on disk."""
    try:
        import utriage.ml.invariance  # noqa: F401
    except ImportError:  # parallel module not yet merged
        mod = types.ModuleType("utriage.ml.invariance")
        mod.LiteEncoder = _MockLiteEncoder  # type: ignore[attr-defined]
        mod.ProjectionHead = _MockProjectionHead  # type: ignore[attr-defined]
        mod.InvariantClassifier = _MockInvariantClassifier  # type: ignore[attr-defined]
        mod.stretch_axis = _mock_stretch_axis  # type: ignore[attr-defined]
        mod.physics_view = _mock_physics_view  # type: ignore[attr-defined]
        mod.nt_xent_loss = _mock_nt_xent_loss  # type: ignore[attr-defined]
        mod.supcon_loss = _mock_supcon_loss  # type: ignore[attr-defined]
        sys.modules["utriage.ml.invariance"] = mod


_install_mock_invariance_if_missing()


def _write_single_batch(
    path: Path,
    flaws: list[dict[str, Any]],
    labels: list[tuple[float, float]],
) -> Path:
    """Write fake ML-NDT batch binary, metadata jsons, and label text files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(flaws)
    raw = np.zeros((n, 256, 256), dtype=np.uint16)
    raw.tofile(path)
    jsons = "".join(json.dumps({"flaws": [f]}) for f in flaws)
    path.with_suffix(".jsons").write_text(jsons, encoding="utf-8")
    np.savetxt(path.with_suffix(".labels"), np.array(labels, dtype=np.float32), delimiter="\t")
    return path


@pytest.fixture
def fake_batches(tmp_path: Path) -> tuple[list[Path], list[Path]]:
    """Create minimal valid ML-NDT training and validation batches."""
    tr_dir = tmp_path / "data" / "training"
    val_dir = tmp_path / "data" / "validation"
    flaws = [
        {"size": "1.6"},
        {"size": "4.0"},
        {"size": "8.6"},
        {"size": "0.0"},
        {"size": "0.0"},
        {"size": "0.0"},
    ]
    labels = [
        (1.0, 1.6),
        (1.0, 4.0),
        (1.0, 8.6),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ]
    tr_b = [_write_single_batch(tr_dir / f"b{i}.bins", flaws, labels) for i in range(10)]
    val_b = [_write_single_batch(val_dir / f"v{j}.bins", flaws, labels) for j in range(2)]
    return tr_b, val_b


def _check_payload_keys(payload: dict[str, Any]) -> None:
    """Validate top-level keys and identity fields of output payload."""
    expected = [
        "label",
        "preregistration",
        "note",
        "config",
        "gpu_name",
        "wall_time_s",
        "training_summary",
        "folds",
        "model_sha256",
    ]
    assert all(k in payload for k in expected)
    assert payload["label"] == "Real data (ML-NDT weld, 3 physical cracks)"
    assert (
        payload["preregistration"]
        == "docs/decisions/0014-invariant-representation-preregistration.md"
    )
    expected_note = (
        "transforms are approximations of physical variation; the held-out crack is never seen"
    )
    assert payload["note"] == expected_note


def _check_fold_structure(fold: dict[str, Any]) -> None:
    """Validate metric keys within a single fold."""
    for key in (
        "held_out",
        "n_flawed",
        "n_clean_calib",
        "n_clean_eval",
        "ranking",
        "operating_points",
        "fused",
    ):
        assert key in fold
    assert "0.01" in fold["operating_points"]
    assert "0.05" in fold["operating_points"]


def _run_quick_experiment(
    tmp_path: Path,
    fake_batches: tuple[list[Path], list[Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Execute quick invariant LOCO experiment in tmp_path."""
    from experiments.run_invariant_loco import main

    tr_b, val_b = fake_batches
    monkeypatch.setattr(
        "experiments.run_invariant_loco.list_batches",
        lambda split: tr_b if split == "training" else val_b,
    )
    out_file = tmp_path / "results" / "invariant_loco_metrics.json"
    models_dir = tmp_path / "models"
    argv = [
        "--quick",
        "--device",
        "cpu",
        "--width",
        "4",
        "--out",
        str(out_file),
        "--models-dir",
        str(models_dir),
    ]
    main(argv)
    return out_file


def test_end_to_end_quick_run_payload_keys(
    tmp_path: Path,
    fake_batches: tuple[list[Path], list[Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that end-to-end --quick run writes JSON payload with all expected keys."""
    out_file = _run_quick_experiment(tmp_path, fake_batches, monkeypatch)
    assert out_file.exists()
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    _check_payload_keys(payload)
    assert len(payload["folds"]) == 3
    for f in payload["folds"]:
        _check_fold_structure(f)


def test_held_out_crack_never_in_train_val_or_embedding_fit(
    fake_batches: tuple[list[Path], list[Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert held-out crack is excluded from train, val, and embedding fit."""
    from experiments.run_invariant_loco import extract_clean_train_embeddings, resolve_folds

    from utriage.ml.invariance import InvariantClassifier, LiteEncoder

    tr_b, val_b = fake_batches
    monkeypatch.setattr(
        "experiments.run_invariant_loco.list_batches",
        lambda split: tr_b if split == "training" else val_b,
    )
    folds = resolve_folds(quick=True)
    dev = torch.device("cpu")
    model = InvariantClassifier(LiteEncoder(width=4))

    for fold in folds:
        s_tr, s_val, s_te = set(fold.train), set(fold.val), set(fold.test)
        assert s_te.isdisjoint(s_tr)
        assert s_te.isdisjoint(s_val)
        clean_embs = extract_clean_train_embeddings(fold.train, model, dev)
        assert clean_embs.shape[0] > 0
        assert fold.held_out not in [b.stem for b, _ in fold.train]
        assert fold.held_out not in [b.stem for b, _ in fold.val]


def test_thresholds_only_from_clean_calib() -> None:
    """Assert that thresholds are derived solely from clean_calib scores."""
    from experiments.run_invariant_loco import evaluate_target_fcr_points

    p = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.85])
    anom = np.array([1.0, 1.5, 2.0, 2.5, 5.0, 4.5])
    flawed_idx = np.array([4, 5])
    calib = np.array([0, 1])
    ev = np.array([2, 3])

    res1 = evaluate_target_fcr_points((p, anom), (flawed_idx, calib, ev), 0.05)
    t_p_orig = res1["p"]["threshold"]
    t_anom_orig = res1["emb_anom"]["threshold"]

    # Mutate eval and flawed scores: thresholds must not change
    p_mutated = p.copy()
    p_mutated[flawed_idx] += 10.0
    p_mutated[ev] += 10.0
    anom_mutated = anom.copy()
    anom_mutated[flawed_idx] += 10.0
    anom_mutated[ev] += 10.0

    res2 = evaluate_target_fcr_points((p_mutated, anom_mutated), (flawed_idx, calib, ev), 0.05)
    assert res2["p"]["threshold"] == t_p_orig
    assert res2["emb_anom"]["threshold"] == t_anom_orig


def test_encoder_weights_change_during_finetuning() -> None:
    """Verify that encoder weights are adjusted (not frozen) during fine-tuning."""
    from experiments.run_invariant_loco import _train_finetune_cpu_epoch, build_finetune_optimizer

    from utriage.ml.invariance import InvariantClassifier, LiteEncoder

    encoder = LiteEncoder(width=4)
    model = InvariantClassifier(encoder)
    opt = build_finetune_optimizer(model)
    gen = torch.Generator().manual_seed(42)

    w_before = copy.deepcopy(model.encoder.state_dict())
    x = torch.randn(8, 1, 64, 64)
    y = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    fake_loader = [(x, y, None)]

    _train_finetune_cpu_epoch(model, fake_loader, opt, gen)  # type: ignore[arg-type]
    w_after = model.encoder.state_dict()

    changed = [not torch.equal(w_before[k], w_after[k]) for k in w_before]
    assert any(changed)


def test_two_param_groups_with_two_learning_rates() -> None:
    """Verify optimizer contains two param groups: encoder lr 1e-4, head lr 1e-3."""
    from experiments.run_invariant_loco import build_finetune_optimizer

    from utriage.ml.invariance import InvariantClassifier, LiteEncoder

    model = InvariantClassifier(LiteEncoder(width=4))
    opt = build_finetune_optimizer(model)

    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["lr"] == 1e-4
    assert opt.param_groups[1]["lr"] == 1e-3

    enc_params = set(model.encoder.parameters())
    head_params = set(model.head.parameters())
    assert set(opt.param_groups[0]["params"]) == enc_params
    assert set(opt.param_groups[1]["params"]) == head_params
