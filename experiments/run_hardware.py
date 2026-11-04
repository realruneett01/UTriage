"""Phase 8 hardware profiling and CPU latency benchmarking."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import winreg
except ImportError:  # Non-Windows platforms
    winreg = None  # type: ignore[assignment]

import numpy as np
import torch
import torch.nn as nn

from utriage.ml.bar_data import WindowSet
from utriage.ml.bar_infer import load_window_model
from utriage.ml.datasets import (
    MLNDT_ROOT,
    list_batches,
    n_images,
    read_image,
    standardise,
)
from utriage.ml.infer import load_model

PROXY_HARDWARE_NOTE = "CPU thread-limited proxy, NOT measured on the target i5-1135G7"
HARDWARE_NOTE = PROXY_HARDWARE_NOTE


@dataclass(frozen=True)
class BenchmarkSpec:
    """Benchmark execution parameters."""

    n: int = 500
    warmup: int = 20
    threads: int = 1


def setup_threads(threads: int | None = None) -> None:
    """Configure torch thread count and execution device before benchmarking."""
    if threads is not None:
        torch.set_num_threads(threads)
        torch.set_default_device("cpu")


setup_hardware = setup_threads


def _parse_cpuinfo_line(line: str) -> str | None:
    """Extract CPU model name from a single cpuinfo line."""
    stripped = line.strip()
    if not stripped.startswith("model name") or ":" not in stripped:
        return None
    name = stripped.split(":", 1)[1].strip()
    return name or None


def _parse_cpuinfo_model(lines: Sequence[str]) -> str | None:
    """Extract model name value from cpuinfo lines."""
    for line in lines:
        val = _parse_cpuinfo_line(line)
        if val is not None:
            return val
    return None


def _read_linux_cpu_name(cpuinfo: Path = Path("/proc/cpuinfo")) -> str | None:
    """Read CPU model name from /proc/cpuinfo if available."""
    if not cpuinfo.exists():
        return None
    try:
        lines = cpuinfo.read_text(encoding="utf-8").splitlines()
        return _parse_cpuinfo_model(lines)
    except OSError:  # /proc/cpuinfo read error
        return None


def _read_windows_cpu_name() -> str | None:
    """Read CPU name string from Windows registry."""
    if winreg is None:
        return None
    reg_key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_key) as key:
            val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(val).strip() if val else None
    except (OSError, AttributeError):  # Registry query failure or mock fallback
        return None


def get_cpu_name() -> str | None:
    """Read human-readable CPU model name across platforms."""
    if platform.system() == "Windows":
        return _read_windows_cpu_name()
    return _read_linux_cpu_name()


def measure_latency(
    fn: Callable[[Any], Any],
    inputs: Sequence[Any],
    n: int = 500,
    warmup: int = 20,
) -> dict[str, Any]:
    """Measure forward latency median and p95 in ms over n passes."""
    if not inputs:
        raise ValueError("inputs sequence must not be empty.")
    if n <= 0 or warmup < 0:
        raise ValueError(f"Invalid count: n={n}, warmup={warmup}")

    k = len(inputs)
    with torch.inference_mode():
        for i in range(warmup):
            fn(inputs[i % k])
        timings: list[float] = []
        for i in range(n):
            t0 = time.perf_counter()
            _ = fn(inputs[i % k])
            timings.append((time.perf_counter() - t0) * 1000.0)

    med = float(np.median(timings))
    p95 = float(np.percentile(timings, 95))
    return {
        "median_ms": med,
        "p95_ms": p95,
        "median": med,
        "p95": p95,
        "n": n,
        "warmup": warmup,
    }


def benchmark_model_latency(
    model: nn.Module,
    inputs: Sequence[torch.Tensor],
    spec: BenchmarkSpec,
) -> dict[str, Any]:
    """Benchmark model forward latency and attach execution metadata."""
    model.eval()
    res = measure_latency(model, inputs, n=spec.n, warmup=spec.warmup)
    res["threads"] = spec.threads
    res["input_shape"] = list(inputs[0].shape)
    return res


def _repo_parent_root() -> Path | None:
    """Find parent workspace root via git common dir if in a worktree."""
    try:
        cmd = ["git", "rev-parse", "--git-common-dir"]  # noqa: S607
        common = subprocess.check_output(  # noqa: S603
            cmd, text=True, stderr=subprocess.DEVNULL
        ).strip()
        return Path(common).resolve().parent
    except (subprocess.SubprocessError, OSError):  # Git inspection fallback
        return None


def resolve_model_path(name: str, models_dir: Path | None = None) -> Path:
    """Locate model checkpoint in models_dir, local models/ or parent."""
    if models_dir is not None:
        return models_dir / name
    local_p = Path("models") / name
    if local_p.exists():
        return local_p
    parent = _repo_parent_root()
    if parent is not None and (parent / "models" / name).exists():
        return parent / "models" / name
    wt_dir = Path(__file__).resolve().parents[2]
    for cand in wt_dir.glob(f"*/models/{name}"):
        if cand.exists():
            return cand
    return local_p


def resolve_metrics_path(filename: str = "mlndt_metrics.json") -> Path:
    """Locate metrics file in results/ or parent workspace."""
    local_p = Path("results") / filename
    if local_p.exists():
        return local_p
    parent = _repo_parent_root()
    if parent is not None and (parent / "results" / filename).exists():
        return parent / "results" / filename
    return local_p


def resolve_data_root() -> Path:
    """Locate ML-NDT dataset root directory."""
    if MLNDT_ROOT.exists():
        return MLNDT_ROOT
    parent = _repo_parent_root()
    if parent is not None:
        cand = parent / "data" / "external" / "ML-NDT" / "data"
        if cand.exists():
            return cand
    return MLNDT_ROOT


def resolve_synthetic_path(filename: str) -> Path:
    """Locate synthetic window file in data/synthetic/bar_windows."""
    local_p = Path("data") / "synthetic" / "bar_windows" / filename
    if local_p.exists():
        return local_p
    parent = _repo_parent_root()
    if parent is not None:
        cand = parent / "data" / "synthetic" / "bar_windows" / filename
        if cand.exists():
            return cand
    return local_p


def load_val_images(val_batches: Sequence[Path]) -> list[torch.Tensor]:
    """Load and standardise real validation images into tensors."""
    tensors: list[torch.Tensor] = []
    for b in val_batches:
        count = n_images(b)
        for idx in range(count):
            raw = read_image(b, idx)
            std = standardise(raw)
            tensors.append(torch.from_numpy(std).unsqueeze(0).unsqueeze(0))
    return tensors


def load_bar_inputs(filename: str, channels: int) -> list[torch.Tensor]:
    """Load synthetic bar windows or fallback to zero tensors."""
    path = resolve_synthetic_path(filename)
    if path.exists():
        wset = WindowSet.load(path)
        return [torch.from_numpy(wset.x[i : i + 1]) for i in range(len(wset.x))]
    return [torch.zeros((1, channels, 128), dtype=torch.float32) for _ in range(50)]


def get_inference_machine(threads: int) -> dict[str, Any]:
    """Capture inference platform hardware metadata."""
    return {
        "processor": platform.processor(),
        "cpu_name": get_cpu_name(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "torch_num_threads": threads,
    }


def get_training_machine(metrics_path: Path | None = None) -> dict[str, Any]:
    """Extract training machine details from mlndt_metrics.json."""
    path = metrics_path or resolve_metrics_path("mlndt_metrics.json")
    if not path.exists():
        return {
            "device_name": None,
            "source_file": "missing: results/mlndt_metrics.json",
        }
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "device_name": data.get("device_name"),
        "source_file": "results/mlndt_metrics.json",
    }


def benchmark_bar_model(
    model_path: Path,
    inputs: Sequence[torch.Tensor],
    spec: BenchmarkSpec,
) -> dict[str, Any]:
    """Benchmark bar WindowNet model or return skip reason."""
    if not model_path.exists():
        return {"status": "skipped", "reason": f"{model_path} not found"}
    if not inputs:
        return {"status": "skipped", "reason": "No inputs provided"}
    model, _ = load_window_model(model_path, device="cpu")
    return benchmark_model_latency(model, inputs, spec)


def benchmark_mlndt_model(
    model_path: Path,
    val_inputs: Sequence[torch.Tensor],
    spec: BenchmarkSpec,
) -> dict[str, Any]:
    """Benchmark ML-NDT model on validation inputs or return skip reason."""
    if not model_path.exists():
        return {"status": "skipped", "reason": f"{model_path} not found"}
    if not val_inputs:
        return {"status": "skipped", "reason": "No validation images found"}
    model, _ = load_model(model_path, device="cpu")
    return benchmark_model_latency(model, val_inputs, spec)


def write_hardware_json(
    data: dict[str, Any],
    out_path: Path = Path("results/hardware.json"),
) -> Path:
    """Serialize hardware benchmarks dictionary to JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return out_path


def update_hardware_metadata(
    path: Path,
    threads: int | None = None,
    proxy_threads: int | None = None,
) -> dict[str, Any]:
    """Rewrite machine metadata in existing hardware JSON while keeping latencies."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    existing_t = data.get("inference_machine", {}).get("torch_num_threads")
    t = threads or (existing_t or torch.get_num_threads())
    data["inference_machine"] = get_inference_machine(t)
    data["training_machine"] = get_training_machine()
    if proxy_threads is not None:
        data["threads"] = proxy_threads
        data["hardware_note"] = PROXY_HARDWARE_NOTE
    write_hardware_json(data, path)
    return data


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse benchmark command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark CPU inference latency.")
    parser.add_argument("--n", type=int, default=500, help="Iteration count (default: 500)")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup passes (default: 20)")
    parser.add_argument("--threads", type=int, default=None, help="Torch thread count")
    parser.add_argument("--output", type=str, default="results/hardware.json", help="Output path")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Update only machine metadata in existing output file without re-benchmarking",
    )
    return parser.parse_args(args)


def _collect_benchmark_latencies(spec: BenchmarkSpec) -> dict[str, Any]:
    """Benchmark ML-NDT and bar models using given spec."""
    val_batches = list_batches("validation", root=resolve_data_root())
    val_inputs = load_val_images(val_batches)
    mlndt_res = benchmark_mlndt_model(resolve_model_path("mlndt_best.pt"), val_inputs, spec)
    b1_inputs = load_bar_inputs("test_1d_full.npz", channels=1)
    bar1d_res = benchmark_bar_model(resolve_model_path("bar_1d.pt"), b1_inputs, spec)
    b2_inputs = load_bar_inputs("test_2d_full.npz", channels=9)
    bar2d_res = benchmark_bar_model(resolve_model_path("bar_2d.pt"), b2_inputs, spec)
    return {
        "mlndt": mlndt_res,
        "bar_1d": bar1d_res,
        "bar_2d": bar2d_res,
    }


def run_hardware(args: argparse.Namespace) -> dict[str, Any]:
    """Run full hardware benchmark suite or update metadata in-place."""
    setup_threads(args.threads)
    threads = torch.get_num_threads()
    out_path = Path(args.output)
    if getattr(args, "metadata_only", False):
        return update_hardware_metadata(out_path, threads=threads, proxy_threads=args.threads)

    spec = BenchmarkSpec(n=args.n, warmup=args.warmup, threads=threads)
    results: dict[str, Any] = {
        "inference_machine": get_inference_machine(threads),
        "training_machine": get_training_machine(),
        "cpu_latency_ms": _collect_benchmark_latencies(spec),
    }
    if args.threads is not None:
        results["threads"] = args.threads
        results["hardware_note"] = PROXY_HARDWARE_NOTE
    return results


def main() -> None:
    """Entry point for hardware benchmark experiment."""
    args = parse_args()
    results = run_hardware(args)
    if not args.metadata_only:
        write_hardware_json(results, Path(args.output))
    print(f"Hardware benchmarks written to {args.output}")


if __name__ == "__main__":
    main()
