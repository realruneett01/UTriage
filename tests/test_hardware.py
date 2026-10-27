"""Tests for hardware benchmark and latency profiling (Phase 8)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Enable importing experiment scripts without package installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from run_hardware import (  # noqa: E402
    BenchmarkSpec,
    benchmark_bar_model,
    benchmark_model_latency,
    get_inference_machine,
    get_training_machine,
    measure_latency,
    write_hardware_json,
)


class _TinyFakeNet(nn.Module):
    """Minimal fake network for tests with zero external dependencies."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


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
