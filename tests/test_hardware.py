"""Tests for hardware benchmark and latency profiling (Phase 8)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

# Enable importing experiment scripts without package installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from run_hardware import (  # noqa: E402
    PROXY_HARDWARE_NOTE,
    BenchmarkSpec,
    _read_linux_cpu_name,
    benchmark_bar_model,
    benchmark_model_latency,
    get_cpu_name,
    get_inference_machine,
    get_training_machine,
    measure_latency,
    parse_args,
    run_hardware,
    setup_threads,
    write_hardware_json,
)


class _TinyFakeNet(nn.Module):
    """Minimal fake network for tests with zero external dependencies."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _FakeRegKey:
    """Fake registry key context manager for tests."""

    def __enter__(self) -> _FakeRegKey:
        return self

    def __exit__(self, *args: object) -> None:
        pass


class _FakeWinreg:
    """Mock winreg module returning predefined processor name."""

    HKEY_LOCAL_MACHINE = 1

    def __init__(self, name: str = "Mock CPU", raises: bool = False) -> None:
        self.name = name
        self.raises = raises

    def OpenKey(self, key: int, subkey: str) -> _FakeRegKey:
        if self.raises:
            raise OSError("Registry key not found")
        return _FakeRegKey()

    def QueryValueEx(self, key: _FakeRegKey, value_name: str) -> tuple[str, int]:
        return f"  {self.name}  ", 1


def test_measure_latency_trivial_fn() -> None:
    """Trivial function returns median <= p95 and counted iterations."""
    inputs = [1, 2, 3, 4]
    res = measure_latency(lambda x: x * 2, inputs, n=30, warmup=5)
    assert res["median"] <= res["p95"]
    assert res["median_ms"] <= res["p95_ms"]
    assert res["n"] == 30
    assert res["warmup"] == 5


def test_measure_latency_warmup_excluded() -> None:
    """Warmup passes are excluded from recorded latency percentiles."""
    call_count = 0

    def slow_warmup_fn(x: int) -> int:
        nonlocal call_count
        call_count += 1
        if call_count <= 4:
            time.sleep(0.015)
        return x

    res = measure_latency(slow_warmup_fn, [1], n=10, warmup=4)
    assert call_count == 14
    # Warmup calls took >15ms each; timed calls take <5ms
    assert res["p95"] < 10.0


def test_measure_latency_cycles_inputs() -> None:
    """Inputs are cycled sequentially across warmup and measurement."""
    order: list[int] = []

    def log_input(x: int) -> int:
        order.append(x)
        return x

    measure_latency(log_input, [10, 20], n=4, warmup=2)
    assert order == [10, 20, 10, 20, 10, 20]


def test_measure_latency_empty_inputs_raises() -> None:
    """Passing empty sequence to measure_latency raises ValueError."""
    with pytest.raises(ValueError, match="must not be empty"):
        measure_latency(lambda x: x, [], n=10, warmup=2)


def test_get_inference_machine() -> None:
    """Inference machine dictionary records required hardware fields."""
    info = get_inference_machine(threads=2)
    expected_keys = {
        "processor",
        "cpu_name",
        "platform",
        "cpu_count",
        "torch_version",
        "torch_num_threads",
    }
    assert expected_keys.issubset(info.keys())
    assert info["torch_num_threads"] == 2


def test_get_training_machine(tmp_path: Path) -> None:
    """Training machine reads device_name and records source file."""
    fake_metrics = tmp_path / "metrics.json"
    fake_metrics.write_text(
        json.dumps({"device_name": "Test RTX 5070 Ti"}),
        encoding="utf-8",
    )
    result = get_training_machine(fake_metrics)
    assert result["device_name"] == "Test RTX 5070 Ti"
    assert "metrics.json" in result["source_file"]


def test_json_writer_with_tiny_fake_model(tmp_path: Path) -> None:
    """Writer outputs valid JSON with tiny fake model without reading models/ or data/."""
    model = _TinyFakeNet()
    inputs = [torch.zeros((1, 4), dtype=torch.float32) for _ in range(5)]
    spec = BenchmarkSpec(n=10, warmup=2, threads=1)
    bench_res = benchmark_model_latency(model, inputs, spec)

    fake_report = {
        "inference_machine": get_inference_machine(threads=1),
        "training_machine": {"device_name": "Mock GPU", "source_file": "fake.json"},
        "cpu_latency_ms": {
            "mlndt": bench_res,
            "bar_1d": {"status": "skipped", "reason": "test mock skip"},
            "bar_2d": {"status": "skipped", "reason": "test mock skip"},
        },
    }

    out_json = tmp_path / "hardware.json"
    written = write_hardware_json(fake_report, out_json)
    assert written.exists()

    loaded = json.loads(written.read_text(encoding="utf-8"))
    assert loaded["training_machine"]["device_name"] == "Mock GPU"
    assert loaded["cpu_latency_ms"]["mlndt"]["n"] == 10
    assert loaded["cpu_latency_ms"]["mlndt"]["input_shape"] == [1, 4]
    assert (
        loaded["cpu_latency_ms"]["mlndt"]["median_ms"]
        <= loaded["cpu_latency_ms"]["mlndt"]["p95_ms"]
    )


def test_benchmark_bar_model_missing() -> None:
    """Missing model checkpoint returns a skipped dictionary with reason."""
    missing_path = Path("nonexistent_model_123.pt")
    spec = BenchmarkSpec(n=5, warmup=1, threads=1)
    res = benchmark_bar_model(missing_path, [], spec)
    assert res.get("status") == "skipped"
    assert "reason" in res
    assert "nonexistent_model_123.pt" in res["reason"]


def test_get_cpu_name_winreg_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows registry processor name string is returned stripped."""
    monkeypatch.setattr("run_hardware.winreg", _FakeWinreg("Intel Core i5-1135G7"))
    monkeypatch.setattr("platform.system", lambda: "Windows")
    assert get_cpu_name() == "Intel Core i5-1135G7"


@pytest.mark.parametrize(
    "winreg_mock",
    [
        _FakeWinreg(raises=True),
        None,
    ],
)
def test_get_cpu_name_fallback_none(
    monkeypatch: pytest.MonkeyPatch,
    winreg_mock: object,
) -> None:
    """Registry error or missing winreg falls back to None."""
    monkeypatch.setattr("run_hardware.winreg", winreg_mock)
    monkeypatch.setattr("platform.system", lambda: "Windows")
    assert get_cpu_name() is None


def test_read_linux_cpu_name(tmp_path: Path) -> None:
    """Linux cpuinfo extracts stripped model name and handles missing file."""
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nmodel name\t: AMD Ryzen 7 5800X\n", encoding="utf-8")
    assert _read_linux_cpu_name(cpuinfo) == "AMD Ryzen 7 5800X"
    assert _read_linux_cpu_name(tmp_path / "missing") is None


def test_parse_args_metadata_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Argument parser correctly sets metadata_only flag."""
    monkeypatch.setattr(sys, "argv", ["run_hardware.py", "--metadata-only"])
    args = parse_args()
    assert args.metadata_only is True


def test_metadata_only_preserves_latency(tmp_path: Path) -> None:
    """Metadata-only updates machine info and leaves latency metrics unchanged."""
    fake_path = tmp_path / "hardware.json"
    orig_latency = {
        "mlndt": {"median_ms": 14.2, "p95_ms": 20.9, "n": 500},
        "bar_1d": {"median_ms": 0.95, "p95_ms": 2.5, "n": 500},
    }
    fake_doc = {
        "inference_machine": {"processor": "old_proc", "cpu_count": 2},
        "training_machine": {"device_name": "old_gpu", "source_file": "old.json"},
        "cpu_latency_ms": orig_latency,
    }
    fake_path.write_text(json.dumps(fake_doc), encoding="utf-8")

    args = argparse.Namespace(
        output=str(fake_path),
        metadata_only=True,
        threads=2,
        n=500,
        warmup=20,
    )
    res = run_hardware(args)
    loaded = json.loads(fake_path.read_text(encoding="utf-8"))

    assert loaded["cpu_latency_ms"] == orig_latency
    assert res["cpu_latency_ms"] == orig_latency
    assert loaded["inference_machine"]["processor"] != "old_proc"
    assert "cpu_name" in loaded["inference_machine"]


@pytest.mark.parametrize(
    ("cli_args", "expected_threads"),
    [
        (["--threads", "4"], 4),
        ([], None),
        (["--n", "100"], None),
    ],
)
def test_parse_args_threads(cli_args: list[str], expected_threads: int | None) -> None:
    """Argument parser correctly extracts threads option or defaults to None."""
    args = parse_args(cli_args)
    assert args.threads == expected_threads


def test_torch_num_threads_after_setup() -> None:
    """Setup sets torch thread count to N and restores old thread count."""
    old_threads = torch.get_num_threads()
    target_threads = 2 if old_threads != 2 else 3
    try:
        setup_threads(target_threads)
        assert torch.get_num_threads() == target_threads
    finally:
        torch.set_num_threads(old_threads)
    assert torch.get_num_threads() == old_threads


def _assert_proxy_metadata(
    data: dict[str, Any],
    threads: int | None,
    expected: bool,
) -> None:
    """Verify presence or absence of proxy metadata fields."""
    if expected:
        assert data["threads"] == threads
        assert data["hardware_note"] == PROXY_HARDWARE_NOTE
    else:
        assert "threads" not in data
        assert "hardware_note" not in data


@pytest.mark.parametrize(
    ("threads_val", "expect_proxy_metadata"),
    [
        (3, True),
        (None, False),
    ],
)
def test_output_contains_threads_and_hardware_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    threads_val: int | None,
    expect_proxy_metadata: bool,
) -> None:
    """Output dict and saved JSON include proxy notes when threads is set."""
    monkeypatch.setattr(
        "run_hardware.benchmark_mlndt_model",
        lambda *a, **k: {"status": "skipped", "reason": "mock"},
    )
    monkeypatch.setattr(
        "run_hardware.benchmark_bar_model",
        lambda *a, **k: {"status": "skipped", "reason": "mock"},
    )
    monkeypatch.setattr("run_hardware.list_batches", lambda *a, **k: [])
    monkeypatch.setattr("run_hardware.load_bar_inputs", lambda *a, **k: [])

    old_threads = torch.get_num_threads()
    out_file = tmp_path / "hardware_out.json"
    args = argparse.Namespace(
        output=str(out_file),
        metadata_only=False,
        threads=threads_val,
        n=10,
        warmup=2,
    )
    try:
        res = run_hardware(args)
        _assert_proxy_metadata(res, threads_val, expect_proxy_metadata)
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize(
    ("threads_val", "expect_proxy_metadata"),
    [
        (4, True),
        (None, False),
    ],
)
def test_metadata_only_threads_and_hardware_note(
    tmp_path: Path,
    threads_val: int | None,
    expect_proxy_metadata: bool,
) -> None:
    """Metadata-only execution respects threads and proxy note presence."""
    fake_path = tmp_path / "hw.json"
    fake_doc = {
        "inference_machine": {"processor": "proc", "cpu_count": 4},
        "training_machine": {"device_name": "gpu", "source_file": "src.json"},
        "cpu_latency_ms": {"test": {"median_ms": 1.0}},
    }
    fake_path.write_text(json.dumps(fake_doc), encoding="utf-8")
    args = argparse.Namespace(
        output=str(fake_path),
        metadata_only=True,
        threads=threads_val,
        n=10,
        warmup=2,
    )
    old_threads = torch.get_num_threads()
    try:
        res = run_hardware(args)
        loaded = json.loads(fake_path.read_text(encoding="utf-8"))
        _assert_proxy_metadata(res, threads_val, expect_proxy_metadata)
        _assert_proxy_metadata(loaded, threads_val, expect_proxy_metadata)
    finally:
        torch.set_num_threads(old_threads)
