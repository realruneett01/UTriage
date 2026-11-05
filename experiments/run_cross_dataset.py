"""Cross-dataset evaluation on real ultrasonic flaws (NDT_ML_Flaw).

Evaluates models trained on ML-NDT or simulator against real data from
NDT_ML_Flaw (batches 013-019), testing domain generalization.
Evaluation only: never trains on this dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Prioritize local worktree source tree over editable installs
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        sys.modules["utriage"].__path__.insert(0, _pkg_dir)
if "utriage.io" in sys.modules:
    _io_dir = str(Path(_SRC) / "utriage" / "io")
    if _io_dir not in sys.modules["utriage.io"].__path__:
        sys.modules["utriage.io"].__path__.insert(0, _io_dir)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from utriage.io.ndt_ml_flaw import (  # noqa: E402
    IMAGE_BYTES,
    N_SCAN,
    N_TIME,
    iter_images,
    list_real_batches,
    read_meta,
    to_mlndt_like,
)
from utriage.ml.infer import load_model, predict_images  # noqa: E402
from utriage.ml.train import pick_device  # noqa: E402
from utriage.pod.metrics import cell  # noqa: E402


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration options for cross-dataset evaluation."""

    max_per_batch: int | None
    quick: bool
    output_path: Path
    device: str


@dataclass(frozen=True)
class DatasetSample:
    """Preprocessed arrays and metadata for evaluated images."""

    images: np.ndarray
    labels: np.ndarray
    sizes: np.ndarray
    flaw_types: list[str]


def _repo_parent_root() -> Path | str | None:
    """Locate root directory of parent repository across worktrees."""
    try:
        cmd = ["git", "rev-parse", "--git-common-dir"]
        res = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return Path(res.stdout.strip()).resolve().parent
    except subprocess.TimeoutExpired:
        return "unknown"
    except (subprocess.SubprocessError, OSError):
        # Fall back if git directory inspection fails
        return None


def find_dataset_dir() -> Path:
    """Locate NDT_ML_Flaw datasets directory."""
    local_p = Path("data/external/NDT_ML_Flaw/datasets")
    if local_p.exists():
        return local_p
    parent = _repo_parent_root()
    if isinstance(parent, Path):
        cand = parent / "data" / "external" / "NDT_ML_Flaw" / "datasets"
        if cand.exists():
            return cand
    env_dir = os.environ.get("UTRIAGE_DATA_DIR")
    data_root = Path(env_dir) if env_dir else Path.home() / "utriage-data"
    user_p = data_root / "external" / "NDT_ML_Flaw" / "datasets"
    if user_p.exists():
        return user_p
    return local_p


def _resolve_model_path(name: str) -> Path | None:
    """Locate a single model checkpoint by name."""
    local_p = Path("models") / name
    if local_p.exists():
        return local_p
    parent = _repo_parent_root()
    if isinstance(parent, Path):
        cand = parent / "models" / name
        if cand.exists():
            return cand
    return None


def _collect_loco_models(models_dir: Path, target: dict[str, Path]) -> None:
    """Collect loco_crack_*.pt models from a specific directory into target dict."""
    if not models_dir.exists():
        return
    for p in sorted(models_dir.glob("loco_crack_*.pt")):
        if p.name not in target:
            target[p.name] = p


def find_all_model_paths() -> dict[str, Path]:
    """Find all candidate models: mlndt_best.pt, loco_crack_*.pt, anomaly_2d.pt."""
    models: dict[str, Path] = {}
    for fixed in ("mlndt_best.pt", "anomaly_2d.pt"):
        cand = _resolve_model_path(fixed)
        if cand is not None:
            models[fixed] = cand
    _collect_loco_models(Path("models"), models)
    parent = _repo_parent_root()
    if isinstance(parent, Path):
        _collect_loco_models(parent / "models", models)
    return models


def load_batch_data(
    xz_path: Path,
    txt_path: Path,
    limit: int | None = None,
) -> DatasetSample:
    """Stream and preprocess images and metadata from one batch pair."""
    meta = read_meta(txt_path)
    if limit is not None:
        meta = meta[:limit]
    imgs: list[np.ndarray] = []
    labels: list[int] = []
    sizes: list[float] = []
    types: list[str] = []
    for raw, rec in zip(iter_images(xz_path, stop=limit), meta, strict=False):
        imgs.append(to_mlndt_like(raw))
        labels.append(rec["flaw"])
        sizes.append(rec["flaw_size"])
        types.append(rec["flaw_type"])
    img_stack = np.stack(imgs, axis=0) if imgs else np.empty((0, 256, 256), dtype=np.float32)
    return DatasetSample(
        images=img_stack,
        labels=np.array(labels, dtype=int),
        sizes=np.array(sizes, dtype=np.float32),
        flaw_types=types,
    )


def collect_dataset_samples(
    pairs: list[tuple[Path, Path]],
    limit: int | None = None,
) -> DatasetSample:
    """Collect and concatenate preprocessed samples across all batches."""
    all_imgs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_sizes: list[np.ndarray] = []
    all_types: list[str] = []
    for xz, txt in pairs:
        batch_sample = load_batch_data(xz, txt, limit=limit)
        all_imgs.append(batch_sample.images)
        all_labels.append(batch_sample.labels)
        all_sizes.append(batch_sample.sizes)
        all_types.extend(batch_sample.flaw_types)
    return DatasetSample(
        images=np.concatenate(all_imgs, axis=0),
        labels=np.concatenate(all_labels, axis=0),
        sizes=np.concatenate(all_sizes, axis=0),
        flaw_types=all_types,
    )


def _evaluate_cnn(
    model_path: Path,
    images: np.ndarray,
    dev: torch.device,
) -> np.ndarray:
    """Load MlNdtNet checkpoint and run inference to get flaw probabilities."""
    model, _ = load_model(model_path, device=str(dev))
    probs, _ = predict_images(model, images, dev)
    return probs


def _resolve_anomaly_threshold() -> float:
    """Read anomaly threshold from results/anomaly_metrics.json if present."""
    metrics_p = Path("results/anomaly_metrics.json")
    if not metrics_p.exists():
        return 0.5
    try:
        with metrics_p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return float(data.get("threshold", data.get("threshold_2d", 0.5)))
    except (json.JSONDecodeError, OSError):
        # Fall back to default threshold on load failure
        return 0.5


def _evaluate_anomaly(
    model_path: Path,
    images: np.ndarray,
    dev: torch.device,
) -> np.ndarray:
    """Load ConvAE2D and compute normalised anomaly scores."""
    from utriage.ml.anomaly import ConvAE2D, anomaly_score

    ckpt = torch.load(model_path, map_location=dev, weights_only=True)
    model = ConvAE2D()
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.to(dev).eval()
    t_imgs = torch.from_numpy(images).unsqueeze(1).to(dev)
    scores = anomaly_score(model, t_imgs).cpu().numpy()
    thresh = _resolve_anomaly_threshold()
    return np.clip(scores / (2.0 * thresh), 0.0, 1.0).astype(np.float32)


def predict_for_model(
    name: str,
    path: Path,
    images: np.ndarray,
    dev: torch.device,
) -> np.ndarray:
    """Run model inference dispatched by model architecture."""
    if name.startswith("anomaly"):
        return _evaluate_anomaly(path, images, dev)
    return _evaluate_cnn(path, images, dev)


def _compute_type_cell(
    ft: str,
    preds: np.ndarray,
    probs: np.ndarray,
    sample: DatasetSample,
) -> dict[str, Any]:
    """Compute POD metrics cell for a single specific flaw type."""
    mask = np.array(
        [t == ft and lbl == 1 for t, lbl in zip(sample.flaw_types, sample.labels, strict=True)]
    )
    hits_ft = preds[mask]
    sizes_ft = sample.sizes[mask]
    probs_ft = probs[mask]
    labels_ft = sample.labels[mask]
    return cell(hits_ft, sizes_ft, probs_ft, labels_ft, n_false=0, n_scans=0, n_physical_flaws=1)


def _compute_flaw_type_metrics(
    preds: np.ndarray,
    probs: np.ndarray,
    sample: DatasetSample,
    distinct_types: list[str],
) -> dict[str, Any]:
    """Compute individual metrics cells for all present flaw types."""
    result: dict[str, Any] = {}
    for ft in distinct_types:
        result[ft] = _compute_type_cell(ft, preds, probs, sample)
    return result


def evaluate_model(
    probs: np.ndarray,
    sample: DatasetSample,
) -> dict[str, Any]:
    """Compute overall and per-flaw-type evaluation metrics."""
    preds = (probs >= 0.5).astype(int)
    flaw_mask = sample.labels == 1
    hits = preds[flaw_mask]
    sizes = sample.sizes[flaw_mask]
    n_false = int(np.sum(preds[~flaw_mask] == 1))
    n_scans = int(np.sum(~flaw_mask))
    distinct_types = sorted(
        {t for t, lbl in zip(sample.flaw_types, sample.labels, strict=True) if lbl == 1}
    )
    overall = cell(hits, sizes, probs, sample.labels, n_false, n_scans, len(distinct_types))
    by_type = _compute_flaw_type_metrics(preds, probs, sample, distinct_types)
    return {"overall": overall, "by_flaw_type": by_type}


def _build_format_facts() -> dict[str, Any]:
    """Compile verified format and dataset facts."""
    return {
        "image_shape": [N_SCAN, N_TIME],
        "dtype": "int16 (little-endian)",
        "bytes_per_image": IMAGE_BYTES,
        "images_per_batch": 1000,
        "decompressed_batch_bytes": IMAGE_BYTES * 1000,
        "flaw_area_sample_range": [1100, 3100],
        "time_axis": 1,
        "scan_axis": 0,
        "real_flaw_batches": "013-019",
        "simulated_flaw_batches": "2xx (excluded)",
    }


def _build_preprocessing_assumptions() -> dict[str, str]:
    """Compile documented domain mapping choices and physics assumptions."""
    return {
        "envelope_method": "scipy.signal.hilbert analytic magnitude along axis 1 (7168 samples)",
        "flaw_area_crop": "Axis 1 cropped to samples 1050-3150 (flaw area 1100-3100 + 50 margin)",
        "time_axis_determination": (
            "Axis 1 identified as time/depth: 1100-3100 sample range only fits in 7168 dimension"
        ),
        "resampling": "Bilinear spatial-temporal zoom to (256, 256) matching ML-NDT input space",
        "standardisation": (
            "Per-image zero-mean unit-variance standardisation matching ML-NDT pipeline"
        ),
    }


def build_report_payload(
    batches: list[str],
    results: dict[str, Any],
) -> dict[str, Any]:
    """Construct complete output metrics structure with audit metadata."""
    return {
        "label": "Real data (NDT_ML_Flaw, LGPL-3.0, not redistributed)",
        "dataset": "NDT_ML_Flaw",
        "licence": "LGPL-3.0",
        "batches_evaluated": batches,
        "verified_format_facts": _build_format_facts(),
        "preprocessing_assumptions": _build_preprocessing_assumptions(),
        "models": results,
    }


def run_cross_dataset(cfg: ExperimentConfig) -> dict[str, Any]:
    """Execute cross-dataset evaluation across discovered batches and models."""
    root = find_dataset_dir()
    pairs = list_real_batches(root)
    if not pairs:
        raise FileNotFoundError(f"No real batches found in {root}")
    limit = 20 if cfg.quick else cfg.max_per_batch
    sample = collect_dataset_samples(pairs, limit=limit)
    dev = pick_device(cfg.device)
    model_paths = find_all_model_paths()
    results: dict[str, Any] = {}
    for name, path in model_paths.items():
        probs = predict_for_model(name, path, sample.images, dev)
        results[name] = evaluate_model(probs, sample)
    batches = [xz.stem for xz, _ in pairs]
    payload = build_report_payload(batches, results)
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def parse_args() -> ExperimentConfig:
    """Parse command line arguments for cross-dataset evaluation."""
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation on NDT_ML_Flaw")
    parser.add_argument("--quick", action="store_true", help="Quick run (20 images per batch)")
    parser.add_argument("--max-per-batch", type=int, default=None, help="Max images per batch")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cross_dataset_metrics.json"),
        help="Path for output JSON metrics",
    )
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto/cuda/cpu)")
    args = parser.parse_args()
    return ExperimentConfig(
        max_per_batch=args.max_per_batch,
        quick=args.quick,
        output_path=args.output,
        device=args.device,
    )


def main() -> None:
    """Entry point for CLI execution."""
    cfg = parse_args()
    payload = run_cross_dataset(cfg)
    print(f"Cross-dataset evaluation complete. Evaluated {len(payload['models'])} models.")
    print(f"Results written to {cfg.output_path}")


if __name__ == "__main__":
    main()
