"""LOCO failure mode analysis and failure case artifact generation (project plan Phase 8).

Re-scores held-out LOCO fold test sets, classifies misses and top false calls,
persists case plots and evidence metadata, and synthesises results/failure_modes.md.
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        del sys.modules["utriage"]

import numpy as np  # noqa: E402

from utriage.ml.datasets import MLNDT_ROOT, list_batches, read_image  # noqa: E402
from utriage.ml.infer import load_model  # noqa: E402
from utriage.ml.loco import IndexedMlNdtDataset, LocoFold, make_folds  # noqa: E402
from utriage.ml.mlndt_meta import read_flaw_meta  # noqa: E402
from utriage.ml.train import TrainConfig, pick_device, predict  # noqa: E402
from utriage.report.failures import (  # noqa: E402
    SMALL_FLAW_P10_DEFAULT,
    build_failure_modes_md,
    classify_failure,
    save_failure_case,
)


@dataclass(frozen=True)
class FoldContext:
    """Evaluation context for a single LOCO fold test partition."""

    fold_name: str
    test_items: list[tuple[Path, int]]
    flawed_mask: np.ndarray
    unflawed_mask: np.ndarray
    hits: np.ndarray
    probs: np.ndarray
    fold_dir: Path


@dataclass(frozen=True)
class FailureConfig:
    """Configuration for failure mode evaluation and artifact output."""

    out_dir: Path
    quick: bool
    device: str
    max_cases: int = 20


@dataclass(frozen=True)
class ProcessTarget:
    """Target category specifications for failure collection."""

    kind: str
    limit: int
    p10: float = SMALL_FLAW_P10_DEFAULT


@dataclass(frozen=True)
class CaseSpec:
    """Specification for recording a single failure case."""

    idx: int
    kind: str
    cause: str
    save_png: bool


def _build_case_id(kind: str, fold: str, batch_stem: str, img_idx: int) -> str:
    """Construct unique failure case identifier incorporating batch stem."""
    return f"{kind}_{fold}_{batch_stem}_{img_idx}"


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LOCO failure modes analysis")
    parser.add_argument("--quick", action="store_true", help="Quick run capping cases to 5")
    parser.add_argument(
        "--models-dir", type=str, default="models", help="Directory containing loco checkpoints"
    )
    parser.add_argument(
        "--out-dir", type=str, default="results/failures", help="Failure cases output folder"
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="results/failure_modes.md",
        help="Path for failure modes markdown",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')"
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=20,
        help="Maximum failure cases per category to save (default 20)",
    )
    return parser.parse_args()


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


def _find_parent_path(subpath: Path) -> Path | None:
    """Check whether subpath exists under parent repository root."""
    parent = _repo_parent_root()
    if parent is not None:
        target = parent / subpath
        if target.exists():
            return target
    return None


def _find_data_root() -> Path:
    """Locate ML-NDT dataset root across standalone clone or git worktree."""
    if MLNDT_ROOT.exists():
        return MLNDT_ROOT
    fallback = _find_parent_path(Path("data/external/ML-NDT/data"))
    return fallback if fallback is not None else MLNDT_ROOT


def _find_loco_models(models_dir: Path) -> list[Path]:
    """Find all loco_crack_*.pt checkpoints in local or parent models directory."""
    if models_dir.exists():
        local_models = sorted(models_dir.glob("loco_crack_*.pt"))
        if local_models:
            return local_models
    fallback = _find_parent_path(Path("models"))
    return sorted(fallback.glob("loco_crack_*.pt")) if fallback is not None else []


def _resolve_batches(quick: bool) -> tuple[list[Path], list[Path]]:
    """Partition ML-NDT dataset into training and validation batch lists."""
    root = _find_data_root()
    tr_all = list_batches("training", root=root)
    val_all = list_batches("validation", root=root)
    return (tr_all[:6], val_all[:1]) if quick else (tr_all, val_all)


def _format_evidence(kind: str, meta: dict[str, Any]) -> str:
    """Format failure evidence metadata string."""
    base = f"amp={meta.get('max_amplitude')} noise={meta.get('noise_threshold')}"
    if kind == "miss":
        return f"size={meta.get('size')} eq={meta.get('equivalent_flawsize')} {base}"
    return base


def _save_case(
    ctx: FoldContext,
    spec: CaseSpec,
    img: np.ndarray,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Save failure case plot and return structured evidence record."""
    bins_path, img_idx = ctx.test_items[spec.idx]
    score = float(ctx.probs[spec.idx])
    name = _build_case_id(spec.kind, ctx.fold_name, bins_path.stem, img_idx)
    save_failure_case(img, meta, score, ctx.fold_dir, name, save_png=spec.save_png)
    json_path = ctx.fold_dir / f"{name}.json"
    evidence_dict: dict[str, Any] = {}
    if json_path.exists():
        evidence_dict = json.loads(json_path.read_text(encoding="utf-8"))
        json_path.unlink()
    return {
        "name": name,
        "fold": ctx.fold_name,
        "kind": spec.kind,
        "score": score,
        "cause": spec.cause,
        "png_file": f"{name}.png",
        "has_png": spec.save_png,
        "evidence": evidence_dict,
        "evidence_str": _format_evidence(spec.kind, meta),
    }


def _evaluate_case(
    ctx: FoldContext,
    idx: int,
    kind: str,
    p10: float,
) -> tuple[str, np.ndarray, dict[str, Any]]:
    """Load case image and metadata and classify failure cause."""
    bins_path, img_idx = ctx.test_items[idx]
    img = read_image(bins_path, img_idx)
    meta = read_flaw_meta(bins_path)[img_idx]
    if kind == "miss":
        cause = classify_failure(meta, "miss", p10_threshold=p10, img=img)
    elif kind == "false_call":
        cause = classify_failure(meta, "false_call", img=img)
    elif kind == "near_miss_clean":
        cause = "clean_near_miss"
    else:
        cause = "unclassified"
    return cause, img, meta


def _collect_top_cases(
    ctx: FoldContext,
    indices: np.ndarray,
    target: ProcessTarget,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Count causes across all indices and save artifacts only for top cases."""
    counts: dict[str, int] = collections.defaultdict(int)
    saved: list[dict[str, Any]] = []
    for i, idx in enumerate(indices):
        cause, img, meta = _evaluate_case(ctx, int(idx), target.kind, target.p10)
        counts[cause] += 1
        if i < target.limit:
            spec = CaseSpec(idx=int(idx), kind=target.kind, cause=cause, save_png=True)
            saved.append(_save_case(ctx, spec, img, meta))
    return saved, dict(counts)


def _resolve_limit(max_cases: int | bool) -> int:
    """Resolve case limit from integer or quick boolean flag."""
    if isinstance(max_cases, bool):
        return 5 if max_cases else 20
    return max(0, int(max_cases))


def _process_misses(
    ctx: FoldContext,
    p10: float,
    max_cases: int | bool = 20,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Collect detection misses, saving only top cases while counting all."""
    miss_indices = np.where(ctx.flawed_mask & (~ctx.hits))[0]
    if len(miss_indices) == 0:
        return [], {}
    limit = _resolve_limit(max_cases)
    sorted_order = np.argsort(-ctx.probs[miss_indices])
    target = ProcessTarget(kind="miss", limit=limit, p10=p10)
    return _collect_top_cases(ctx, miss_indices[sorted_order], target)


def _process_false_calls(
    ctx: FoldContext,
    max_cases: int | bool = 20,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Collect false calls (clean with score >= 0.5), saving top while counting all."""
    actual_fc = np.where(ctx.unflawed_mask & ctx.hits)[0]
    if len(actual_fc) == 0:
        return [], {}
    limit = _resolve_limit(max_cases)
    sorted_order = np.argsort(-ctx.probs[actual_fc])
    target = ProcessTarget(kind="false_call", limit=limit)
    return _collect_top_cases(ctx, actual_fc[sorted_order], target)


def _process_near_misses_clean(
    ctx: FoldContext,
    max_cases: int | bool = 20,
) -> list[dict[str, Any]]:
    """Collect top clean images below threshold as near misses (not false calls)."""
    clean_indices = np.where(ctx.unflawed_mask & (~ctx.hits))[0]
    if len(clean_indices) == 0:
        return []
    limit = _resolve_limit(max_cases)
    sorted_order = np.argsort(-ctx.probs[clean_indices])
    top_indices = clean_indices[sorted_order][:limit]
    target = ProcessTarget(kind="near_miss_clean", limit=limit)
    saved, _ = _collect_top_cases(ctx, top_indices, target)
    return saved


def _score_fold_test(
    model_path: Path,
    fold: LocoFold,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score fold test set with trained model and extract labels."""
    test_ds = IndexedMlNdtDataset(fold.test)
    eval_cfg = TrainConfig(batch_size=64, device=device)
    model, _ = load_model(model_path, device=device)
    probs, _ = predict(model, test_ds, eval_cfg)
    flaw_labels, eq_sizes = test_ds.get_labels()
    return probs, flaw_labels, eq_sizes


def _write_cases_json(
    fold_dir: Path,
    misses: list[dict[str, Any]],
    false_calls: list[dict[str, Any]],
    near_misses_clean: list[dict[str, Any]] | None = None,
) -> Path:
    """Save top failure cases to cases.json in fold directory."""
    clean = near_misses_clean or []
    data = {
        "misses": misses,
        "false_calls": false_calls,
        "near_misses_clean": clean,
        "cases": misses + false_calls + clean,
    }
    cases_path = fold_dir / "cases.json"
    cases_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return cases_path


def _eval_single_fold(
    model_path: Path,
    fold: LocoFold,
    cfg: FailureConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate LOCO fold test set, categorize failures, and save case files."""
    fold_dir = cfg.out_dir / fold.held_out
    fold_dir.mkdir(parents=True, exist_ok=True)
    probs, flaw_labels, eq_sizes = _score_fold_test(model_path, fold, cfg.device)
    flawed = (flaw_labels == 1.0) & (eq_sizes > 0.0)
    unflawed = flaw_labels == 0.0
    hits = probs >= 0.5

    det_sizes = eq_sizes[flawed & hits]
    p10 = float(np.percentile(det_sizes, 10)) if len(det_sizes) > 0 else SMALL_FLAW_P10_DEFAULT

    ctx = FoldContext(fold.held_out, fold.test, flawed, unflawed, hits, probs, fold_dir)
    miss_recs, miss_counts = _process_misses(ctx, p10, cfg.max_cases)
    fc_recs, fc_counts = _process_false_calls(ctx, cfg.max_cases)
    clean_recs = _process_near_misses_clean(ctx, cfg.max_cases)
    _write_cases_json(fold_dir, miss_recs, fc_recs, clean_recs)

    summary = {
        "fold": fold.held_out,
        "n_misses": sum(miss_counts.values()),
        "miss_causes": miss_counts,
        "n_false_calls": sum(fc_counts.values()),
        "false_call_causes": fc_counts,
    }
    return summary, miss_recs + fc_recs + clean_recs


def _load_sim_gen(sim_path: Path) -> dict:
    """Load pre-computed synthetic generalization metrics if available."""
    if not sim_path.exists():
        return {}
    try:
        return json.loads(sim_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:  # narrow reason: corrupt json content
        return {}


def _write_summary_json(out_dir: Path, summaries: list[dict[str, Any]]) -> Path:
    """Save per-fold cause breakdown counts to summary.json."""
    summary_data = {
        s["fold"]: {
            "n_misses": s["n_misses"],
            "miss_causes": s["miss_causes"],
            "n_false_calls": s["n_false_calls"],
            "false_call_causes": s["false_call_causes"],
        }
        for s in summaries
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    return summary_path


def _process_folds(
    models: list[Path],
    fold_map: dict[str, LocoFold],
    cfg: FailureConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Process all available model folds and accumulate summaries and cases."""
    summaries: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for model_path in models:
        fold_name = model_path.stem.replace("loco_", "")
        if fold_name not in fold_map:
            continue
        print(f"Evaluating failure modes for {fold_name} on {cfg.device}...")
        summary, fold_cases = _eval_single_fold(model_path, fold_map[fold_name], cfg)
        summaries.append(summary)
        cases.extend(fold_cases)
    return summaries, cases


def _write_failure_modes_report(
    report_path: Path,
    summaries: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    cfg: FailureConfig,
) -> None:
    """Build and write failure modes report stating coverage and case cap."""
    sim_gen = _load_sim_gen(Path("results/sim_generalization.json"))
    png_count = sum(1 for c in cases if c.get("has_png", True))
    md_cases = [{**c, "evidence": c.get("evidence_str", c.get("evidence"))} for c in cases]
    md_content = build_failure_modes_md(summaries, md_cases, sim_gen, png_count=png_count)

    old_tail = "full JSON evidence retained for all evaluated failure cases."
    new_tail = (
        f"cause counts cover all cases while files are kept for the top {cfg.max_cases} of each"
        " kind."
    )
    md_content = md_content.replace(old_tail, new_tail)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md_content, encoding="utf-8")
    print(f"Wrote failure modes report to {report_path}")


def main() -> None:
    """Run failure mode extraction, plot generation, and report synthesis."""
    args = _parse_args()
    dev = pick_device(args.device)
    models = _find_loco_models(Path(args.models_dir))
    if not models:
        print("No LOCO models found (models are still training). Exiting.")
        return

    tr_batches, val_batches = _resolve_batches(args.quick)
    folds = make_folds(tr_batches, val_batches, seed=0)
    fold_map = {f.held_out: f for f in folds}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_cases = 5 if args.quick else args.max_cases
    cfg = FailureConfig(out_dir=out_dir, quick=args.quick, device=str(dev), max_cases=max_cases)
    summaries, cases = _process_folds(models, fold_map, cfg)
    _write_summary_json(out_dir, summaries)
    _write_failure_modes_report(Path(args.report_path), summaries, cases, cfg)


if __name__ == "__main__":
    main()
