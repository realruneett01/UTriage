"""Tests for model hash checks on application startup and loading."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_SRC = str(Path(__file__).resolve().parents[1] / "src")
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import typer  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from utriage.cli import _load_site_model, app  # noqa: E402
from utriage.integrity import (  # noqa: E402
    IntegrityError,
    model_sidecar,
    sha256_file,
    write_model_hash,
)
from utriage.ml.anomaly import ConvAE2D  # noqa: E402
from utriage.ml.bar_infer import load_window_model, save_window_model  # noqa: E402
from utriage.ml.bar_models import WindowNet  # noqa: E402


def _make_dummy_checkpoint(path: Path) -> Path:
    """Save small autoencoder weights and hash sidecar."""
    torch.manual_seed(0)
    model = ConvAE2D()
    torch.save(model.state_dict(), path)
    write_model_hash(path)
    return path


def _corrupt_file(path: Path) -> None:
    """Flip the first byte of target file on disk."""
    raw = bytearray(path.read_bytes())
    raw[0] ^= 0xFF
    path.write_bytes(raw)


def _tiny_window_model() -> WindowNet:
    """Instantiate a minimal WindowNet for test execution."""
    torch.manual_seed(0)
    return WindowNet(in_channels=1, width=4, n_classes=3, dropout=0.0)


def test_bar_infer_load_with_sidecar(tmp_path: Path) -> None:
    """Check that bar_infer loads a model with valid sidecar and attaches digest."""
    ckpt_path = tmp_path / "window.pt"
    model = _tiny_window_model()
    save_window_model(
        model,
        ckpt_path,
        {"width": 4, "in_channels": 1, "n_classes": 3, "dropout": 0.0},
    )

    sidecar = model_sidecar(ckpt_path)
    assert sidecar.is_file()

    loaded_model, meta = load_window_model(ckpt_path, device="cpu")
    assert isinstance(loaded_model, WindowNet)
    assert meta.get("sha256") == sha256_file(ckpt_path)


def test_bar_infer_tamper_rejected_before_torch_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check that tampered window model triggers IntegrityError before torch.load."""
    ckpt_path = tmp_path / "window.pt"
    model = _tiny_window_model()
    save_window_model(
        model,
        ckpt_path,
        {"width": 4, "in_channels": 1, "n_classes": 3, "dropout": 0.0},
    )
    _corrupt_file(ckpt_path)

    torch_load_reached = False

    def fail_load(*args: Any, **kwargs: Any) -> Any:
        nonlocal torch_load_reached
        torch_load_reached = True
        raise AssertionError("torch.load should not execute when hash check fails")

    monkeypatch.setattr(torch, "load", fail_load)

    with pytest.raises(IntegrityError, match=f"Model hash mismatch: {ckpt_path.name}"):
        load_window_model(ckpt_path, device="cpu")

    assert not torch_load_reached


@pytest.mark.parametrize(
    ("locked_env", "expect_error"),
    [
        (None, False),
        ("1", True),
    ],
    ids=["unlocked_allows_missing_sidecar", "locked_rejects_missing_sidecar"],
)
def test_bar_infer_missing_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    locked_env: str | None,
    expect_error: bool,
) -> None:
    """Check missing sidecar behavior in unlocked versus locked environments."""
    ckpt_path = tmp_path / "window.pt"
    model = _tiny_window_model()
    save_window_model(
        model,
        ckpt_path,
        {"width": 4, "in_channels": 1, "n_classes": 3, "dropout": 0.0},
    )
    model_sidecar(ckpt_path).unlink()

    if locked_env is not None:
        monkeypatch.setenv("UTRIAGE_LOCKED", locked_env)
    else:
        monkeypatch.delenv("UTRIAGE_LOCKED", raising=False)

    if expect_error:
        with pytest.raises(IntegrityError, match="Model hash file missing"):
            load_window_model(ckpt_path, device="cpu")
    else:
        loaded, meta = load_window_model(ckpt_path, device="cpu")
        assert isinstance(loaded, WindowNet)
        assert meta.get("sha256") == sha256_file(ckpt_path)


def test_cli_load_site_model_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Check that _load_site_model exits with code 1 and prints error on tampered file."""
    ckpt_path = _make_dummy_checkpoint(tmp_path / "site_ae.pt")
    _corrupt_file(ckpt_path)

    torch_load_reached = False

    def fail_load(*args: Any, **kwargs: Any) -> Any:
        nonlocal torch_load_reached
        torch_load_reached = True
        raise AssertionError("torch.load should not execute when hash check fails")

    monkeypatch.setattr(torch, "load", fail_load)

    with pytest.raises(typer.Exit) as exc_info:
        _load_site_model(ckpt_path, torch.device("cpu"))

    assert exc_info.value.exit_code == 1
    assert not torch_load_reached

    captured = capsys.readouterr()
    err_text = captured.out + captured.err
    assert f"Model hash mismatch: {ckpt_path.name}" in err_text


def test_cli_site_calibrate_tamper_command(tmp_path: Path) -> None:
    """Check that site-calibrate exits with code 1 on tampered model checkpoint."""
    ckpt_path = _make_dummy_checkpoint(tmp_path / "site_ae.pt")
    _corrupt_file(ckpt_path)

    clean_dir = tmp_path / "clean_scans"
    clean_dir.mkdir()
    np.save(clean_dir / "scan_00.npy", np.zeros((16, 16), dtype=np.float32))

    out_json = tmp_path / "profile.json"
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "site-calibrate",
            "--clean-dir",
            str(clean_dir),
            "--out",
            str(out_json),
            "--label",
            "Synthetic",
            "--model",
            str(ckpt_path),
            "--fcr",
            "0.05",
        ],
    )
    assert res.exit_code == 1
    assert "Model hash mismatch" in res.output
