"""Tests for ML inference, checkpointing, and latency measurement."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from utriage.ml.infer import load_model, measure_latency, predict_images, save_model  # noqa: E402
from utriage.ml.models import MlNdtNet  # noqa: E402


class _MaliciousPayload:
    """Arbitrary non-allowlisted class simulating untrusted code execution."""


def test_save_load_round_trip(tmp_path: Path) -> None:
    """Save and load tiny model, ensuring weights and outputs match."""
    model = MlNdtNet(width=4)
    model.eval()
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        orig_logits, orig_sizes = model(x)

    meta = {"width": 4, "first_pool": (2, 2), "size": 64}
    ckpt_path = tmp_path / "subdir" / "tiny_model.pt"
    saved = save_model(model, ckpt_path, meta)
    assert saved.exists()

    loaded_model, loaded_meta = load_model(saved, device="cpu")
    assert loaded_meta["width"] == 4
    assert tuple(loaded_meta["first_pool"]) == (2, 2)
    assert loaded_meta["size"] == 64

    with torch.no_grad():
        loaded_logits, loaded_sizes = loaded_model(x)

    assert torch.allclose(orig_logits, loaded_logits, atol=1e-6)
    assert torch.allclose(orig_sizes, loaded_sizes, atol=1e-6)


def test_load_model_rejects_untrusted_object(tmp_path: Path) -> None:
    """Ensure load_model refuses checkpoints with non-allowlisted objects."""
    ckpt_path = tmp_path / "malicious.pt"
    payload = {"state_dict": {}, "meta": {"exploit": _MaliciousPayload()}}
    torch.save(payload, ckpt_path)
    with pytest.raises((pickle.UnpicklingError, RuntimeError)):
        load_model(ckpt_path)


def test_load_model_explicit_weights_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify load_model explicitly passes weights_only=True to torch.load."""
    ckpt_path = tmp_path / "valid.pt"
    model = MlNdtNet(width=4)
    save_model(model, ckpt_path, {"width": 4})
    calls: list[dict[str, Any]] = []
    real_torch_load = torch.load

    def tracked_load(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real_torch_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", tracked_load)
    load_model(ckpt_path)
    assert len(calls) == 1
    assert calls[0].get("weights_only") is True


@pytest.mark.parametrize("n_images", [1, 5])
def test_predict_images_shapes_and_range(n_images: int) -> None:
    """Validate predicted probabilities and sizes on raw uint16 images."""
    model = MlNdtNet(width=4)
    device = torch.device("cpu")
    images = np.random.randint(0, 1000, size=(n_images, 64, 64), dtype=np.uint16)

    probs, sizes = predict_images(model, images, device)
    assert probs.shape == (n_images,)
    assert sizes.shape == (n_images,)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_predict_images_empty() -> None:
    """Check empty image array edge case."""
    model = MlNdtNet(width=4)
    device = torch.device("cpu")
    empty = np.empty((0, 64, 64), dtype=np.float32)
    probs, sizes = predict_images(model, empty, device)
    assert len(probs) == 0
    assert len(sizes) == 0


def test_measure_latency() -> None:
    """Verify latency benchmark returns a positive float."""
    model = MlNdtNet(width=4)
    device = torch.device("cpu")
    latency_ms = measure_latency(model, device, batch=2, size=64, repeats=3)
    assert isinstance(latency_ms, float)
    assert latency_ms > 0.0
