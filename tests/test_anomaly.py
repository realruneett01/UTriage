"""Tests for unsupervised anomaly detection models and gating utilities."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure local workspace src is prioritized over external paths
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        sys.modules["utriage"].__path__.insert(0, _pkg_dir)
if "utriage.ml" in sys.modules:
    _ml_dir = str(Path(_SRC) / "utriage" / "ml")
    if _ml_dir not in sys.modules["utriage.ml"].__path__:
        sys.modules["utriage.ml"].__path__.insert(0, _ml_dir)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from utriage.ml.anomaly import (  # noqa: E402
    ConvAE1D,
    ConvAE2D,
    anomaly_score,
    count_params,
    fit_threshold,
    gate_walls,
    train_autoencoder,
)


def _make_flat_noise(num_samples: int = 120, length: int = 128) -> torch.Tensor:
    """Generate synthetic zero-mean flat noise windows."""
    return torch.randn(num_samples, 1, length) * 0.05


def _inject_pulse(clean: torch.Tensor, center_idx: int = 64, amp: float = 1.5) -> torch.Tensor:
    """Inject a localized Gaussian echo pulse into 1D windows."""
    length = clean.shape[-1]
    t = torch.arange(length, dtype=torch.float32)
    pulse = amp * torch.exp(-0.5 * ((t - center_idx) / 3.0) ** 2)
    flawed = clean.clone()
    flawed[:, 0, :] += pulse
    return flawed


def test_conv_ae_1d_shapes_and_param_count() -> None:
    """ConvAE1D preserves input shape and stays under 50k parameters."""
    model = ConvAE1D()
    total_params = count_params(model)
    assert total_params < 50_000

    x = torch.randn(4, 1, 128)
    out = model(x)
    assert out.shape == (4, 1, 128)


def test_conv_ae_2d_shapes_and_param_count() -> None:
    """ConvAE2D preserves B-scan image shape and stays under 200k parameters."""
    model = ConvAE2D()
    total_params = count_params(model)
    assert total_params < 200_000

    x = torch.randn(2, 1, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 256, 256)


@pytest.mark.parametrize(
    ("length", "front_idx", "back_idx", "guard"),
    [
        (128, 20, 100, 5),
        (64, 10, 50, 3),
        (100, 0, 99, 4),
    ],
)
def test_gate_walls_masks_right_samples(
    length: int,
    front_idx: int,
    back_idx: int,
    guard: int,
) -> None:
    """gate_walls zeroes samples within guard distance of wall echoes."""
    ascan = np.ones(length, dtype=np.float32)
    mask = gate_walls(ascan, front_idx=front_idx, back_idx=back_idx, guard=guard)

    assert mask.shape == ascan.shape
    f_start = max(0, front_idx - guard)
    f_end = min(length, front_idx + guard + 1)
    b_start = max(0, back_idx - guard)
    b_end = min(length, back_idx + guard + 1)

    assert np.all(mask[f_start:f_end] == 0.0)
    assert np.all(mask[b_start:b_end] == 0.0)

    outside = np.ones(length, dtype=bool)
    outside[f_start:f_end] = False
    outside[b_start:b_end] = False
    assert np.all(mask[outside] == 1.0)


def test_training_lowers_loss_on_flat_noise() -> None:
    """Training ConvAE1D on synthetic flat noise monotonically lowers MSE loss."""
    torch.manual_seed(42)
    clean = _make_flat_noise(num_samples=160, length=128)
    loader = DataLoader(TensorDataset(clean), batch_size=32, shuffle=True)

    model = ConvAE1D()
    losses = train_autoencoder(model, loader, epochs=5, lr=1e-3, device="cpu")

    assert len(losses) == 5
    assert losses[-1] < losses[0]


def test_injected_gaussian_pulse_scores_higher() -> None:
    """Injected Gaussian flaw pulse scores higher than clean baseline after brief training."""
    torch.manual_seed(42)
    clean = _make_flat_noise(num_samples=160, length=128)
    loader = DataLoader(TensorDataset(clean), batch_size=32, shuffle=True)

    model = ConvAE1D()
    train_autoencoder(model, loader, epochs=5, lr=1e-3, device="cpu")

    test_clean = _make_flat_noise(num_samples=40, length=128)
    test_flawed = _inject_pulse(test_clean, center_idx=64, amp=1.5)

    clean_scores = anomaly_score(model, test_clean)
    flawed_scores = anomaly_score(model, test_flawed)

    assert flawed_scores.shape == (40,)
    assert clean_scores.shape == (40,)
    assert float(flawed_scores.mean().item()) > float(clean_scores.mean().item())


def test_gated_wall_echo_does_not_drive_score() -> None:
    """Wall echo inside the guard zone does not elevate the anomaly score when gated."""
    torch.manual_seed(42)
    clean = _make_flat_noise(num_samples=30, length=128)
    model = ConvAE1D()

    # Inject large wall reflection at index 15
    wall_data = _inject_pulse(clean, center_idx=15, amp=5.0)
    mask_np = gate_walls(np.ones(128, dtype=np.float32), front_idx=15, back_idx=115, guard=6)
    mask_tensor = torch.from_numpy(mask_np).view(1, 1, 128)

    ungated_score = anomaly_score(model, wall_data, mask=None)
    gated_score = anomaly_score(model, wall_data, mask=mask_tensor)

    assert float(gated_score.mean().item()) < float(ungated_score.mean().item())


@pytest.mark.parametrize("rate", [0.01, 0.05, 0.10])
def test_fit_threshold_quantile_property(rate: float) -> None:
    """Fitted threshold empirically isolates approximately rate fraction of scores."""
    np.random.seed(42)
    scores = np.random.uniform(0.0, 1.0, size=2000)
    thresh = fit_threshold(scores, false_call_rate=rate)

    emp_rate = float(np.mean(scores > thresh))
    assert pytest.approx(rate, abs=0.01) == emp_rate


@pytest.mark.parametrize("bad_rate", [-0.1, 1.1])
def test_fit_threshold_value_error_on_invalid_rate(bad_rate: float) -> None:
    """fit_threshold raises ValueError when false_call_rate is outside [0, 1]."""
    scores = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="false_call_rate"):
        fit_threshold(scores, false_call_rate=bad_rate)
