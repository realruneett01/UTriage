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
    min_clean_scans,
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


@dataclass(frozen=True)
class PartitionResult:
    """Disjoint calibration clean, test clean, and test flawed item partitions."""

    cal_clean: list[tuple[Path, int]]
    test_clean: list[tuple[Path, int]]
    test_flawed: list[tuple[Path, int]]
    overlaps_model_training: bool
    underpowered: bool
    min_required: int
    fcr_choice_reason: str


@dataclass(frozen=True)
class ExperimentPayloadInput:
    """Input metadata required to generate output JSON payload."""

    cfg: SiteAdaptationConfig
    data: ShiftEvaluationData
    part: PartitionResult
    model_sha: str


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


def _assert_disjoint_partitions(
    cal_clean: list[tuple[Path, int]],
    test_clean: list[tuple[Path, int]],
    test_flawed: list[tuple[Path, int]],
) -> None:
    """Assert zero index overlap between calibration clean items and test items."""
    cal_set = set(cal_clean)
    test_set = set(test_clean) | set(test_flawed)
    overlap = cal_set & test_set
    assert not overlap, f"Calibration and test overlap: {overlap}"  # noqa: S101


def _extract_clean_flawed(
    batches: Sequence[Path],
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Extract clean (label 0) and flawed (label 1) (batch, index) pairs from batches."""
    clean_items: list[tuple[Path, int]] = []
    flawed_items: list[tuple[Path, int]] = []
    for b in batches:
        lbl = read_labels(b)[:, 0]
        for i in np.where(lbl == 0)[0]:
            clean_items.append((b, int(i)))
        for i in np.where(lbl == 1)[0]:
            flawed_items.append((b, int(i)))
    return clean_items, flawed_items


def _partition_validation_split(
    val_batches: list[Path],
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Split validation batches into validation calibration clean, test clean, and test flawed."""
    if len(val_batches) >= 2:
        cal_b, test_b = split_holdout(val_batches, test_frac=0.5, seed=seed)
        val_cal_clean, _ = _extract_clean_flawed(cal_b)
        test_clean, test_flawed = _extract_clean_flawed(test_b)
        return val_cal_clean, test_clean, test_flawed
    if not val_batches:
        return [], [], []
    b = val_batches[0]
    lbl = read_labels(b)[:, 0]
    cl_idx = np.where(lbl == 0)[0]
    mid = len(cl_idx) // 2
    cal_clean = [(b, int(i)) for i in cl_idx[:mid]]
    test_clean = [(b, int(i)) for i in cl_idx[mid:]]
    test_flawed = [(b, int(i)) for i in np.where(lbl == 1)[0]]
    return cal_clean, test_clean, test_flawed


def _partition_training_batches(
    tr_batches: list[Path],
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], set[Path]]:
    """Split training batches into untrained holdout, trained clean, and trained batch paths."""
    if not tr_batches:
        return [], [], set()
    non_ho, ho = split_holdout(tr_batches, test_frac=0.1, seed=0)
    ho_clean, _ = _extract_clean_flawed(ho)
    trained_clean, _ = _extract_clean_flawed(non_ho)
    trained_batches = {b.resolve() for b in non_ho}
    return ho_clean, trained_clean, trained_batches


def _make_fcr_reason(fcr: float, n_cal: int, min_req: int, underpowered: bool) -> str:
    """Generate explanatory text for target false call rate power status."""
    if underpowered:
        return (
            f"Target FCR {fcr} requires {min_req} clean scans, "
            f"but only {n_cal} available (calibration underpowered)"
        )
    return f"Target FCR {fcr} requires {min_req} clean scans; calibration pool has {n_cal} scans"


def partition_adaptation_data(
    val_batches: list[Path],
    tr_batches: list[Path],
    cfg: SiteAdaptationConfig,
) -> PartitionResult:
    """Partition ML-NDT data into disjoint calibration clean, test clean, and test flawed sets."""
    val_cal, tc, tf = _partition_validation_split(val_batches, cfg.seed)
    ho_clean, tr_clean, trained_b = _partition_training_batches(tr_batches)

    untrained = val_cal + ho_clean
    min_req = min_clean_scans(cfg.fcr)
    candidate_cal = list(untrained) if len(untrained) >= min_req else untrained + tr_clean

    lim = 16 if cfg.quick else sys.maxsize
    cal_clean = candidate_cal[:lim]
    test_clean = tc[:lim]
    test_flawed = tf[:lim]

    _assert_disjoint_partitions(cal_clean, test_clean, test_flawed)
    overlaps = any(b.resolve() in trained_b for b, _ in cal_clean)
    underpowered = len(cal_clean) < min_req
    reason = _make_fcr_reason(cfg.fcr, len(cal_clean), min_req, underpowered)
    return PartitionResult(
        cal_clean, test_clean, test_flawed, overlaps, underpowered, min_req, reason
    )


def _partition_val_items(
    batches: list[Path],
    seed: int,
    quick: bool,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Extract disjoint calibration clean, test clean, and test flawed item indices."""
    cal_clean, test_clean, test_flawed = _partition_validation_split(batches, seed)
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
    inp: ExperimentPayloadInput,
    shifts: dict[str, Any],
) -> dict[str, Any]:
    """Construct complete JSON payload with limits and provenance information."""
    return {
        "label": DATASET_LABEL,
        "dataset_label": DATASET_LABEL,
        "limits": LIMITS_TEXT,
        "model_sha256": inp.model_sha,
        "seed": inp.cfg.seed,
        "target_fcr": inp.cfg.fcr,
        "original_threshold": float(inp.data.t_orig),
        "n_images": {
            "cal_clean": len(inp.data.cal_clean),
            "test_clean": len(inp.data.test_clean),
            "test_flawed": len(inp.data.test_flawed),
            "total": len(inp.data.cal_clean) + len(inp.data.test_clean) + len(inp.data.test_flawed),
        },
        "shifts": shifts,
        "n_calib_clean": len(inp.data.cal_clean),
        "min_clean_required": inp.part.min_required,
        "calibration_underpowered": inp.part.underpowered,
        "calib_overlaps_model_training": inp.part.overlaps_model_training,
        "fcr_choice_reason": inp.part.fcr_choice_reason,
    }


def _print_underpowered_warning(n_cal: int, min_req: int, fcr: float) -> None:
    """Print standard warning when calibration clean pool is underpowered."""
    print(
        f"WARNING: calibration clean pool ({n_cal}) below recommended minimum "
        f"({min_req}) for target FCR {fcr} at 95% confidence (underpowered)."
    )


def _write_site_payload(out_dir: Path, payload: dict[str, Any]) -> None:
    """Write site adaptation evaluation payload to site_adaptation.json."""
    out_file = out_dir / "site_adaptation.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_site_adaptation(cfg: SiteAdaptationConfig) -> dict[str, Any]:
    """Execute site adaptation evaluation across all fixed shifts and save results."""
    set_seed(cfg.seed)
    dev = pick_device(cfg.device)
    model_path = _resolve_model_path(cfg.models_dir, cfg.model_path)
    model = _load_model(model_path, dev)

    val_b, tr_b = list_batches("validation", root=cfg.root), list_batches("training", root=cfg.root)
    part = partition_adaptation_data(val_b, tr_b, cfg)
    cal_cl = _stream_scan_images(part.cal_clean)
    test_cl = _stream_scan_images(part.test_clean)
    test_fl = _stream_scan_images(part.test_flawed)

    if part.underpowered:
        _print_underpowered_warning(len(cal_cl), part.min_required, cfg.fcr)

    t_orig = calibrate_site_threshold(score_images(model, cal_cl, dev), cfg.fcr).threshold
    data = ShiftEvaluationData(cal_cl, test_cl, test_fl, t_orig)
    settings = ShiftEvalSettings(cfg.fcr, dev, batch_size=32, seed=cfg.seed)
    shifts = {s.name: evaluate_shift(model, s, data, settings) for s in FIXED_SHIFTS}

    payload = _build_payload(
        ExperimentPayloadInput(cfg, data, part, sha256_file(model_path)), shifts
    )
    _write_site_payload(cfg.out_dir, payload)
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
