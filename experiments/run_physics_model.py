"""Interpretable physics-feature detector on ML-NDT dataset.

Pre-registered design: candidate H3 of ADR 0010.
Streams images from disk without caching in RAM, extracts 8 hand-crafted physics
features, and fits a LogisticRegression classifier per LOCO fold with balanced
class weights and StandardScaler fitted strictly on fold training data.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        del sys.modules["utriage"]

import subprocess  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from utriage.detector_config import (  # noqa: E402
    DEFAULT_PATH,
    DetectorConfig,
    PhysicsConfig,
    config_fingerprint,
    load_detector_config,
)
from utriage.ml.datasets import (  # noqa: E402
    MLNDT_ROOT,
    list_batches,
    read_image,
    read_labels,
)
from utriage.ml.loco import LocoFold, make_folds  # noqa: E402
from utriage.ml.physics_features import (  # noqa: E402
    FEATURE_NAMES,
    extract_features,
)
from utriage.pod.metrics import cell, clopper_pearson  # noqa: E402

PROTOCOL_TEXT: str = (
    "Leave-one-crack-out (LOCO) cross-validation evaluating interpretable "
    "hand-crafted physics features with logistic regression (ADR 0010 H3). "
    "Features standardized with StandardScaler fitted on fold train set only; "
    "decision threshold at probability >= 0.5."
)


@dataclass(frozen=True)
class PhysicsModelConfig:
    """Configuration options for physics model experiment execution."""

    root: Path = MLNDT_ROOT
    out_path: Path = Path("results/physics_model_metrics.json")
    quick: bool = False
    detector_cfg: DetectorConfig = field(default_factory=load_detector_config)


@dataclass(frozen=True)
class FoldData:
    """Features, binary labels, and flaw sizes for a dataset split."""

    features: np.ndarray
    labels: np.ndarray
    sizes: np.ndarray


@dataclass(frozen=True)
class FitResult:
    """Trained classifier, fitted scaler, and extracted feature coefficients."""

    classifier: LogisticRegression
    scaler: StandardScaler
    coefficients: dict[str, float]


def _parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for physics feature LOCO experiment."""
    parser = argparse.ArgumentParser(description="Physics feature LOCO experiment")
    parser.add_argument("--quick", action="store_true", help="Quick run with subset")
    parser.add_argument("--root", type=str, default=str(MLNDT_ROOT), help="ML-NDT root")
    parser.add_argument(
        "--out",
        type=str,
        default="results/physics_model_metrics.json",
        help="Path to output JSON",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PATH,
        help="Path to detector YAML configuration file",
    )
    parser.add_argument(
        "--set",
        dest="set",
        action="append",
        default=[],
        help="Override configuration setting in dot-notation (e.g., physics.threshold=0.6)",
    )
    parsed = parser.parse_args(args)
    parsed.overrides = parsed.set
    return parsed


def _repo_parent_root() -> Path | None:
    """Locate root directory of parent repository across worktrees."""
    try:
        cmd = ["git", "rev-parse", "--git-common-dir"]
        common = subprocess.check_output(  # noqa: S603
            cmd, text=True, stderr=subprocess.DEVNULL
        ).strip()
        return Path(common).resolve().parent
    except (subprocess.CalledProcessError, FileNotFoundError):  # narrow reason: git unavailable
        return None


def _find_data_root(provided: Path) -> Path:
    """Locate ML-NDT dataset root across standalone clone or git worktree."""
    if provided.exists():
        return provided
    parent = _repo_parent_root()
    if parent is not None:
        cand = parent / "data" / "external" / "ML-NDT" / "data"
        if cand.exists():
            return cand
    return provided


def _resolve_batches(root: Path, quick: bool) -> tuple[list[Path], list[Path]]:
    """Resolve training and validation batches based on quick flag."""
    resolved_root = _find_data_root(root)
    tr_all = list_batches("training", root=resolved_root)
    val_all = list_batches("validation", root=resolved_root)
    if quick:
        return tr_all[:6], val_all[:1]
    return tr_all, val_all


def build_feature_dataset(
    items: list[tuple[Path, int]],
    noise_multiplier: float = 6.0,
    energy_window: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features, labels, and flaw sizes streaming images from disk."""
    feats: list[np.ndarray] = []
    flaws: list[int] = []
    sizes: list[float] = []
    labels_cache: dict[Path, np.ndarray] = {}

    for bins_path, idx in items:
        if bins_path not in labels_cache:
            labels_cache[bins_path] = read_labels(bins_path)
        lbl = labels_cache[bins_path]
        flaws.append(int(lbl[idx, 0]))
        sizes.append(float(lbl[idx, 1]))
        img = read_image(bins_path, idx)
        feats.append(
            extract_features(
                img,
                noise_multiplier=noise_multiplier,
                energy_window=energy_window,
            )
        )

    x_arr = np.array(feats, dtype=np.float64) if feats else np.empty((0, 8), dtype=np.float64)
    y_arr = np.array(flaws, dtype=np.int64) if flaws else np.empty(0, dtype=np.int64)
    s_arr = np.array(sizes, dtype=np.float32) if sizes else np.empty(0, dtype=np.float32)
    return x_arr, y_arr, s_arr


def fit_fold_classifier(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    scaler: StandardScaler,
    cfg: PhysicsConfig | None = None,
) -> FitResult:
    """Fit StandardScaler and LogisticRegression classifier on train fold split."""
    p_cfg = cfg if cfg is not None else PhysicsConfig()
    x_tr_scaled = scaler.fit_transform(x_tr)
    clf = LogisticRegression(
        class_weight=p_cfg.class_weight,
        C=p_cfg.C,
        max_iter=p_cfg.max_iter,
        random_state=0,
    )
    clf.fit(x_tr_scaled, y_tr)
    coefs = clf.coef_[0]
    intercept = float(clf.intercept_[0])
    coeff_dict = {name: float(coefs[i]) for i, name in enumerate(FEATURE_NAMES)}
    coeff_dict["intercept"] = intercept
    return FitResult(classifier=clf, scaler=scaler, coefficients=coeff_dict)


def _evaluate_fold(
    fit_res: FitResult,
    test_split: FoldData,
    threshold: float = 0.5,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Evaluate trained fold classifier on test split."""
    x_te, y_te, s_te = test_split.features, test_split.labels, test_split.sizes
    x_te_scaled = fit_res.scaler.transform(x_te)
    probs = fit_res.classifier.predict_proba(x_te_scaled)[:, 1]
    flags = (probs >= threshold).astype(int)

    f_mask = y_te == 1
    u_mask = y_te == 0
    k_det = int(np.sum(flags[f_mask] == 1))
    n_f = int(np.sum(f_mask))
    c_low, c_high = clopper_pearson(k_det, n_f, conf=0.95)
    det_rate = float(k_det / n_f) if n_f > 0 else 0.0
    det_summary = {"k": k_det, "n": n_f, "rate": det_rate, "ci_low": c_low, "ci_high": c_high}

    n_false = int(np.sum(flags[u_mask] == 1))
    n_clean = int(np.sum(u_mask))
    _, fc_up = clopper_pearson(n_false, n_clean, conf=0.95)
    fc_rate = float(n_false / n_clean) if n_clean > 0 else 0.0
    fc_summary = {"n_false": n_false, "n_clean": n_clean, "rate": fc_rate, "upper95": fc_up}

    c = cell(flags[f_mask], s_te[f_mask], probs, y_te, n_false, n_clean, 1)
    return det_summary, fc_summary, c


def _run_single_fold(
    fold: LocoFold,
    p_cfg: PhysicsConfig | None = None,
) -> dict[str, Any]:
    """Train and evaluate physics detector on a single LOCO fold."""
    cfg = p_cfg if p_cfg is not None else PhysicsConfig()
    x_tr, y_tr, _ = build_feature_dataset(fold.train, cfg.noise_multiplier, cfg.energy_window)
    x_te, y_te, s_te = build_feature_dataset(fold.test, cfg.noise_multiplier, cfg.energy_window)

    scaler = StandardScaler()
    fit_res = fit_fold_classifier(x_tr, y_tr, scaler, cfg)
    test_split = FoldData(features=x_te, labels=y_te, sizes=s_te)
    det, fc, fold_cell = _evaluate_fold(fit_res, test_split, cfg.threshold)

    return {
        "held_out": fold.held_out,
        "n_train": len(fold.train),
        "n_test_flawed": int(np.sum(y_te == 1)),
        "n_test_unflawed": int(np.sum(y_te == 0)),
        "detections": det,
        "false_calls": fc,
        "coefficients": fit_res.coefficients,
        "cell": fold_cell,
    }


def run_physics_experiment(cfg: PhysicsModelConfig) -> dict[str, Any]:
    """Execute complete LOCO physics feature experiment across 3 folds."""
    tr_batches, val_batches = _resolve_batches(cfg.root, cfg.quick)
    folds = make_folds(tr_batches, val_batches, seed=0, val_mode="batch")

    p_cfg = cfg.detector_cfg.physics
    fold_results: list[dict[str, Any]] = []
    for f in folds:
        res = _run_single_fold(f, p_cfg)
        fold_results.append(res)

    fp = config_fingerprint(cfg.detector_cfg)
    run_label = "pre-registered" if fp["is_default"] else "exploratory"
    payload = {
        "label": "Real data (ML-NDT)",
        "run_label": run_label,
        "detector_config": fp,
        "protocol": PROTOCOL_TEXT,
        "feature_names": list(FEATURE_NAMES),
        "folds": fold_results,
    }

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    """CLI entry point for physics model experiment."""
    args = _parse_args()
    overrides = args.overrides or None
    detector_cfg = load_detector_config(path=args.config, overrides=overrides)
    cfg = PhysicsModelConfig(
        root=Path(args.root),
        out_path=Path(args.out),
        quick=args.quick,
        detector_cfg=detector_cfg,
    )
    payload = run_physics_experiment(cfg)
    print(f"Physics model experiment complete. Folds evaluated: {len(payload['folds'])}")
    print(f"Results written to: {cfg.out_path}")


if __name__ == "__main__":
    main()
