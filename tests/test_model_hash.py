from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

_SRC = str(Path(__file__).resolve().parents[1] / "src")
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pytest  # noqa: E402
import torch  # noqa: E402

from utriage.integrity import (  # noqa: E402
    IntegrityError,
    check_model_hash,
    model_sidecar,
    sha256_file,
    write_model_hash,
)
from utriage.ml.infer import load_model, save_model  # noqa: E402
from utriage.ml.models import MlNdtNet  # noqa: E402


def test_model_sidecar_path() -> None:
    """Check that model_sidecar appends .sha256 to checkpoint path."""
    sample_path = Path("checkpoints") / "detector.pt"
    expected = Path("checkpoints") / "detector.pt.sha256"
    assert model_sidecar(sample_path) == expected


def test_direct_hash_helpers(tmp_path: Path) -> None:
    """Check direct write_model_hash and check_model_hash operations on raw file."""
    binary_file = tmp_path / "weights.bin"
    binary_file.write_bytes(b"dummy weights data")
    sidecar_path = write_model_hash(binary_file)
    assert sidecar_path.is_file()
    digest = check_model_hash(binary_file)
    assert digest == sha256_file(binary_file)


def test_sidecar_format(tmp_path: Path) -> None:
    """Check sidecar sha256sum format has 64 hex characters, two spaces, and name."""
    ckpt_path = tmp_path / "tiny.pt"
    model = MlNdtNet(width=4)
    save_model(model, ckpt_path, {"width": 4})

    sidecar = model_sidecar(ckpt_path)
    assert sidecar.is_file()

    content = sidecar.read_text(encoding="utf-8")
    pattern = r"^[0-9a-f]{64}  tiny\.pt\n$"
    assert re.match(pattern, content) is not None
    digest = sha256_file(ckpt_path)
    assert content == f"{digest}  {ckpt_path.name}\n"


def test_save_load_round_trip_digest(tmp_path: Path) -> None:
    """Check save and load round trip records matching sha256 in metadata."""
    ckpt_path = tmp_path / "tiny.pt"
    model = MlNdtNet(width=4)
    save_model(model, ckpt_path, {"width": 4})

    loaded_model, meta = load_model(ckpt_path, device="cpu")
    assert isinstance(loaded_model, MlNdtNet)
    assert meta.get("sha256") == sha256_file(ckpt_path)


def test_tampered_model_rejected_before_torch_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check that corrupted model bytes trigger IntegrityError before torch.load."""
    ckpt_path = tmp_path / "tiny.pt"
    save_model(MlNdtNet(width=4), ckpt_path, {"width": 4})

    raw_bytes = bytearray(ckpt_path.read_bytes())
    raw_bytes[0] ^= 0xFF
    ckpt_path.write_bytes(raw_bytes)

    torch_load_called = False

    def tracked_torch_load(*args: Any, **kwargs: Any) -> Any:
        nonlocal torch_load_called
        torch_load_called = True
        raise AssertionError("torch.load should not be reached when hash check fails")

    monkeypatch.setattr(torch, "load", tracked_torch_load)

    with pytest.raises(IntegrityError, match=f"Model hash mismatch: {ckpt_path.name}"):
        load_model(ckpt_path, device="cpu")

    assert not torch_load_called


@pytest.mark.parametrize(
    ("locked_env", "expect_error"),
    [
        (None, False),
        ("1", True),
    ],
    ids=["unlocked_loads_without_sidecar", "locked_fails_without_sidecar"],
)
def test_missing_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    locked_env: str | None,
    expect_error: bool,
) -> None:
    """Check missing sidecar behavior when unlocked versus locked."""
    ckpt_path = tmp_path / "tiny.pt"
    save_model(MlNdtNet(width=4), ckpt_path, {"width": 4})
    model_sidecar(ckpt_path).unlink()

    if locked_env is not None:
        monkeypatch.setenv("UTRIAGE_LOCKED", locked_env)
    else:
        monkeypatch.delenv("UTRIAGE_LOCKED", raising=False)

    if expect_error:
        with pytest.raises(IntegrityError, match="Model hash file missing"):
            load_model(ckpt_path, device="cpu")
    else:
        loaded, meta = load_model(ckpt_path, device="cpu")
        assert isinstance(loaded, MlNdtNet)
        assert meta.get("sha256") == sha256_file(ckpt_path)
