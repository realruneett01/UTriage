"""Unsupervised site adaptation evaluation experiment runner (project plan Phase 0 / ADR 0010).

Evaluates false-call rate drift under acoustic inspection environment shifts (gain, offset,
noise) on real ML-NDT data, and tests whether unsupervised site adaptation restores the target
false call rate without flaw labels.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Prioritize local workspace src
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from utriage.integrity import sha256_file  # noqa: E402
from utriage.ml.anomaly import ConvAE2D, anomaly_score  # noqa: E402
from utriage.ml.datasets import (  # noqa: E402
    MLNDT_ROOT,
    list_batches,
    n_images,
    read_labels,
    split_holdout,
    standardise,
)
from utriage.ml.site_adapt import (  # noqa: E402
    apply_site_stats,
    calibrate_site_threshold,
    fit_site_stats,
    recalibrate_batchnorm,
)
from utriage.ml.train import pick_device, set_seed  # noqa: E402
from utriage.pod.metrics import clopper_pearson  # noqa: E402

DATASET_LABEL: str = "Real (ML-NDT weld, 3 physical cracks; shifts are synthetic perturbations)"
LIMITS_TEXT: str = (
    "Real ML-NDT data contains only 3 physical thermal fatigue cracks in an austenitic "
    "pipe weld; all inspection environment shifts (gain, offset, noise) are synthetic "
    "perturbations applied in software. Advisory only: UTriage never accepts or passes a part."
)


@dataclass(frozen=True)
class SiteShift:
    """Parameters specifying an acoustic inspection environment perturbation."""

    gain: float
    offset: float
    noise_std: float
    name: str


FIXED_SHIFTS: tuple[SiteShift, ...] = (
    SiteShift(gain=1.0, offset=0.0, noise_std=0.0, name="identity"),
    SiteShift(gain=0.5, offset=0.0, noise_std=0.0, name="gain 0.5"),
    SiteShift(gain=2.0, offset=0.0, noise_std=0.0, name="gain 2.0"),
    SiteShift(gain=1.0, offset=0.5, noise_std=0.0, name="offset +0.5"),
    SiteShift(gain=1.0, offset=0.0, noise_std=0.2, name="noise 0.2"),
    SiteShift(gain=0.5, offset=0.5, noise_std=0.2, name="combined"),
)


@dataclass(frozen=True)
class ShiftedTestScans:
    """Shifted test split clean and flawed scans."""

    clean: list[np.ndarray]
    flawed: list[np.ndarray]


@dataclass(frozen=True)
class ShiftEvaluationData:
    """Pre-partitioned image scan arrays and baseline threshold."""

    cal_clean: list[np.ndarray]
    test_clean: list[np.ndarray]
    test_flawed: list[np.ndarray]
    t_orig: float


@dataclass(frozen=True)
class ShiftEvalSettings:
    """Execution parameters for shift evaluation."""

    target_fcr: float
    device: torch.device
    batch_size: int = 32
    seed: int = 0


@dataclass(frozen=True)
class SiteAdaptationConfig:
    """Configuration for site adaptation experiment execution."""

    root: Path = MLNDT_ROOT
    out_dir: Path = Path("results")
    models_dir: Path = Path("models")
    model_path: Path | None = None
    fcr: float = 0.01
    device: str = "auto"
    seed: int = 0
    quick: bool = False


def apply_shift(
    img: np.ndarray,
    shift: SiteShift,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Apply gain, offset, and additive Gaussian noise shift to an image."""
    arr = np.asarray(img, dtype=np.float32)
    shifted = arr * float(shift.gain) + float(shift.offset)
    if shift.noise_std > 0.0:
        gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        noise = gen.normal(0.0, float(shift.noise_std), size=shifted.shape).astype(np.float32)
        shifted = shifted + noise
    return shifted


def _summarize_counts(hits: np.ndarray, conf: float = 0.95) -> dict[str, Any]:
    """Compute hit count k, trials n, empirical rate, and Clopper-Pearson CI."""
    arr = np.asarray(hits, dtype=int)
    n = int(len(arr))
    k = int(np.sum(arr == 1))
    rate = float(k / n) if n > 0 else float("nan")
    ci_low, ci_high = clopper_pearson(k, n, conf=conf)
    return {
        "k": k,
        "n": n,
        "rate": rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "interval_95": [ci_low, ci_high],
        "method": "Clopper-Pearson two-sided 95",
    }


@torch.no_grad()
def score_images(
    model: nn.Module,
    images: Sequence[np.ndarray],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Compute anomaly scores for a sequence of 2D images."""
    if not images:
        return np.empty(0, dtype=np.float32)
    scores: list[float] = []
    model.eval()
    for i in range(0, len(images), batch_size):
        chunk = np.stack(images[i : i + batch_size], axis=0)
        t = torch.from_numpy(chunk).unsqueeze(1).to(device)
        s = anomaly_score(model, t)
        scores.extend(s.detach().cpu().numpy().tolist())
    return np.array(scores, dtype=np.float32)


def _make_loader(images: Sequence[np.ndarray], batch_size: int = 32) -> list[torch.Tensor]:
    """Prepare list of (B, 1, H, W) batch tensors for BatchNorm recalibration."""
    batches = []
    for i in range(0, len(images), batch_size):
        chunk = np.stack(images[i : i + batch_size], axis=0)
        batches.append(torch.from_numpy(chunk).unsqueeze(1))
    return batches


def _eval_baseline(
    model: nn.Module,
    scans: ShiftedTestScans,
    t_orig: float,
    dev: torch.device,
) -> dict[str, Any]:
    """Evaluate baseline performance on shifted test scans using unshifted threshold."""
    scores_clean = score_images(model, scans.clean, dev)
    scores_flawed = score_images(model, scans.flawed, dev)
    return {
        "threshold": float(t_orig),
        "false_calls": _summarize_counts(scores_clean > t_orig),
        "detection": _summarize_counts(scores_flawed > t_orig),
    }


def _eval_adapted(
    model: nn.Module,
    shifted_cal: list[np.ndarray],
    scans: ShiftedTestScans,
    settings: ShiftEvalSettings,
) -> dict[str, Any]:
    """Perform site adaptation on shifted calibration scans and evaluate shifted test."""
    stats = fit_site_stats(shifted_cal, label="Real")
    ad_cal = [apply_site_stats(x, stats) for x in shifted_cal]
    ad_tc = [apply_site_stats(x, stats) for x in scans.clean]
    ad_tf = [apply_site_stats(x, stats) for x in scans.flawed]

    ad_model = copy.deepcopy(model)
    recalibrate_batchnorm(
        ad_model, _make_loader(ad_cal, settings.batch_size), device=settings.device
    )
    ad_cal_scores = score_images(ad_model, ad_cal, settings.device, settings.batch_size)
    calib = calibrate_site_threshold(ad_cal_scores, false_call_rate=settings.target_fcr)

    sc_clean = score_images(ad_model, ad_tc, settings.device, settings.batch_size)
    sc_flawed = score_images(ad_model, ad_tf, settings.device, settings.batch_size)
    return {
        "threshold": float(calib.threshold),
        "upper95_fcr": float(calib.upper95_fcr),
        "false_calls": _summarize_counts(sc_clean > calib.threshold),
        "detection": _summarize_counts(sc_flawed > calib.threshold),
    }


def evaluate_shift(
    model: nn.Module,
    shift: SiteShift,
    data: ShiftEvaluationData,
    settings: ShiftEvalSettings,
) -> dict[str, Any]:
    """Evaluate baseline and adapted detection performance under a single site shift."""
    rng = np.random.default_rng(settings.seed)
    shifted_cal = [apply_shift(x, shift, rng) for x in data.cal_clean]
    shifted_tc = [apply_shift(x, shift, rng) for x in data.test_clean]
    shifted_tf = [apply_shift(x, shift, rng) for x in data.test_flawed]

    scans = ShiftedTestScans(clean=shifted_tc, flawed=shifted_tf)
    baseline_res = _eval_baseline(model, scans, data.t_orig, settings.device)
    adapted_res = _eval_adapted(model, shifted_cal, scans, settings)

    return {
        "shift": {
            "name": shift.name,
            "gain": shift.gain,
            "offset": shift.offset,
            "noise_std": shift.noise_std,
        },
        "baseline": baseline_res,
        "adapted": adapted_res,
    }


def _partition_val_items(
    batches: list[Path],
    seed: int,
    quick: bool,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Extract disjoint calibration clean, test clean, and test flawed item indices."""
    if len(batches) >= 2:
        cal_b, test_b = split_holdout(batches, test_frac=0.5, seed=seed)
        cal_clean = [(b, int(i)) for b in cal_b for i in np.where(read_labels(b)[:, 0] == 0)[0]]
        test_clean = [(b, int(i)) for b in test_b for i in np.where(read_labels(b)[:, 0] == 0)[0]]
        test_flawed = [(b, int(i)) for b in test_b for i in np.where(read_labels(b)[:, 0] == 1)[0]]
    else:
        b = batches[0]
        lbl = read_labels(b)
        cl_idx = np.where(lbl[:, 0] == 0)[0]
        mid = len(cl_idx) // 2
        cal_clean = [(b, int(i)) for i in cl_idx[:mid]]
        test_clean = [(b, int(i)) for i in cl_idx[mid:]]
        test_flawed = [(b, int(i)) for i in np.where(lbl[:, 0] == 1)[0]]

    lim = 16 if quick else sys.maxsize
    return cal_clean[:lim], test_clean[:lim], test_flawed[:lim]


def _stream_scan_images(items: list[tuple[Path, int]]) -> list[np.ndarray]:
    """Stream standardized 2D scans from disk using memory maps without loading full dataset."""
    mmaps: dict[Path, np.memmap] = {}
    images: list[np.ndarray] = []
    for b, idx in items:
        if b not in mmaps:
            shape = (n_images(b), 256, 256)
            mmaps[b] = np.memmap(b, dtype=np.uint16, mode="r", shape=shape)
        images.append(standardise(mmaps[b][idx]))
    return images


def _repo_parent_root() -> Path | None:
    """Locate root directory of parent repository across git worktrees."""
    try:
        cmd = ["git", "rev-parse", "--git-common-dir"]
        common = subprocess.check_output(  # noqa: S603
            cmd, text=True, stderr=subprocess.DEVNULL
        ).strip()
        return Path(common).resolve().parent
    except (subprocess.CalledProcessError, FileNotFoundError):  # git query unavailable
        return None


def _resolve_model_path(models_dir: Path, explicit_path: Path | None) -> Path:
    """Locate model checkpoint in models_dir or parent worktree fallback."""
    if explicit_path is not None and explicit_path.exists():
        return explicit_path
    local_p = models_dir / "anomaly_2d.pt"
    if local_p.exists():
        return local_p
    parent = _repo_parent_root()
    if parent is not None:
        fallback = parent / "models" / "anomaly_2d.pt"
        if fallback.exists():
            return fallback
    return local_p


def _load_model(path: Path, dev: torch.device) -> ConvAE2D:
    """Load trained ConvAE2D model from checkpoint onto target compute device."""
    model = ConvAE2D()
    state_dict = torch.load(path, map_location=dev, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()
    return model


def _build_payload(
    cfg: SiteAdaptationConfig,
    data: ShiftEvaluationData,
    model_sha: str,
    shifts: dict[str, Any],
) -> dict[str, Any]:
    """Construct complete JSON payload with limits and provenance information."""
    return {
        "label": DATASET_LABEL,
        "dataset_label": DATASET_LABEL,
        "limits": LIMITS_TEXT,
        "model_sha256": model_sha,
        "seed": cfg.seed,
        "target_fcr": cfg.fcr,
        "original_threshold": float(data.t_orig),
        "n_images": {
            "cal_clean": len(data.cal_clean),
            "test_clean": len(data.test_clean),
            "test_flawed": len(data.test_flawed),
            "total": len(data.cal_clean) + len(data.test_clean) + len(data.test_flawed),
        },
        "shifts": shifts,
    }


def run_site_adaptation(cfg: SiteAdaptationConfig) -> dict[str, Any]:
    """Execute site adaptation evaluation across all fixed shifts and save results."""
    set_seed(cfg.seed)
    dev = pick_device(cfg.device)
    model_path = _resolve_model_path(cfg.models_dir, cfg.model_path)
    model = _load_model(model_path, dev)
    model_sha = sha256_file(model_path)

    val_batches = list_batches("validation", root=cfg.root)
    cal_idx, tc_idx, tf_idx = _partition_val_items(val_batches, cfg.seed, cfg.quick)
    cal_clean = _stream_scan_images(cal_idx)
    test_clean = _stream_scan_images(tc_idx)
    test_flawed = _stream_scan_images(tf_idx)

    cal_scores = score_images(model, cal_clean, dev)
    t_orig = calibrate_site_threshold(cal_scores, false_call_rate=cfg.fcr).threshold
    data = ShiftEvaluationData(cal_clean, test_clean, test_flawed, t_orig)
    settings = ShiftEvalSettings(cfg.fcr, dev, batch_size=32, seed=cfg.seed)

    shift_results = {s.name: evaluate_shift(model, s, data, settings) for s in FIXED_SHIFTS}
    payload = _build_payload(cfg, data, model_sha, shift_results)

    out_file = cfg.out_dir / "site_adaptation.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _parse_args(args: list[str] | None = None) -> SiteAdaptationConfig:
    """Parse command line arguments into SiteAdaptationConfig."""
    parser = argparse.ArgumentParser(description="Evaluate unsupervised site adaptation.")
    parser.add_argument("--root", type=Path, default=MLNDT_ROOT, help="ML-NDT data root")
    parser.add_argument("--out-dir", type=Path, default=Path("results"), help="Output directory")
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="Models directory")
    parser.add_argument("--model-path", type=Path, default=None, help="Explicit model path")
    parser.add_argument("--fcr", type=float, default=0.01, help="Target false call rate")
    parser.add_argument("--device", type=str, default="auto", help="Compute device")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Quick run with subset of scans")
    ns = parser.parse_args(args)
    return SiteAdaptationConfig(
        root=ns.root,
        out_dir=ns.out_dir,
        models_dir=ns.models_dir,
        model_path=ns.model_path,
        fcr=ns.fcr,
        device=ns.device,
        seed=ns.seed,
        quick=ns.quick,
    )


def main() -> None:
    """CLI entrypoint for site adaptation evaluation."""
    cfg = _parse_args()
    run_site_adaptation(cfg)


if __name__ == "__main__":
    main()
