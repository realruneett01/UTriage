"""Tests for pre-registration success rule evaluation.

Covers boundary evaluation, candidate presence/absence, latency flags,
schema detection, and CLI execution without writing to results/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from scripts.judge_preregistration import (
    EVIDENCE_NOTE,
    JudgeConfig,
    evaluate_preregistration,
    main,
)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Helper to write test payload dictionaries to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _make_baseline_data(
    auroc: float = 0.80,
    k: int = 3000,
    n: int = 4000,
    held_out: str = "crack_1.6",
) -> dict[str, Any]:
    """Build minimal baseline dictionary."""
    rate = k / n if n > 0 else float("nan")
    return {
        "protocols": {
            "batchval": {
                "folds": [
                    {
                        "held_out": held_out,
                        "auroc": auroc,
                        "operating_points": {
                            "0.05": {
                                "detection": {
                                    "k": k,
                                    "n": n,
                                    "rate": rate,
                                }
                            }
                        },
                    }
                ]
            }
        }
    }


def _make_robust_data(
    auroc: float = 0.85,
    k: int = 3000,
    n: int = 4000,
    held_out: str = "crack_1.6",
) -> dict[str, Any]:
    """Build minimal robust candidate dictionary."""
    rate = k / n if n > 0 else float("nan")
    return {
        "folds": [
            {
                "held_out": held_out,
                "ensemble_metrics": {
                    "auroc": auroc,
                    "fcr_0.05": {
                        "detections": {
                            "n_hit": k,
                            "n_flawed": n,
                            "rate": rate,
                        }
                    },
                },
            }
        ]
    }


def _make_lite_data(
    auroc: float = 0.85,
    k: int = 3000,
    n: int = 4000,
    held_out: str = "crack_1.6",
    lite_ms: float = 1.5,
    mlndt_ms: float = 3.0,
) -> dict[str, Any]:
    """Build minimal lite hybrid candidate dictionary."""
    rate = k / n if n > 0 else float("nan")
    return {
        "folds": {
            held_out: {
                "held_out": held_out,
                "threshold_free": {"fused": {"auroc": auroc}},
                "operating_points": {
                    "fcr_0.05": {"hybrid": {"detection": {"k": k, "n": n, "rate": rate}}}
                },
            }
        },
        "latency": {
            "total": {"median_ms": lite_ms},
            "mlndtnet": {"median_ms": mlndt_ms},
        },
    }


def _make_invariant_data(
    auroc: float = 0.86,
    k: int = 3100,
    n: int = 4000,
    held_out: str = "crack_1.6",
) -> dict[str, Any]:
    """Build minimal invariant representation candidate dictionary."""
    rate = k / n if n > 0 else float("nan")
    return {
        "folds": [
            {
                "held_out": held_out,
                "fused": {
                    "auroc": auroc,
                    "operating_points": {"0.05": {"detection": {"k": k, "n": n, "rate": rate}}},
                },
            }
        ]
    }


def test_missing_baseline_raises_clear_error(tmp_path: Path) -> None:
    """Missing baseline file must raise FileNotFoundError with clear message."""
    cfg = JudgeConfig(results_dir=tmp_path, out_path=tmp_path / "out.json")
    with pytest.raises(FileNotFoundError, match="Baseline file not found"):
        evaluate_preregistration(cfg)


def test_missing_candidate_reports_not_run(tmp_path: Path) -> None:
    """Missing candidate files must be reported with status not run."""
    _write_json(tmp_path / "separability_metrics.json", _make_baseline_data())
    cfg = JudgeConfig(results_dir=tmp_path, out_path=tmp_path / "out.json")
    res = evaluate_preregistration(cfg)

    cands = res["candidates"]
    for name in ("robust", "lite", "invariant"):
        assert cands[name]["status"] == "not run"
        assert cands[name]["folds"] == {}
        assert cands[name]["note"] == EVIDENCE_NOTE


@pytest.mark.parametrize(
    ("cand_auc", "cand_k", "expected_verdict"),
    [
        (0.85, 3000, "better"),
        (0.86, 3100, "better"),
        (0.8499, 3000, "no improvement"),
        (0.90, 2999, "no improvement"),
        (float("nan"), 3000, "not evaluable"),
        (0.86, -1, "not evaluable"),
    ],
)
def test_verdict_rule_outcomes(
    tmp_path: Path,
    cand_auc: float,
    cand_k: int,
    expected_verdict: str,
) -> None:
    """Test boundary condition, lower detection, and NaN handling."""
    _write_json(tmp_path / "separability_metrics.json", _make_baseline_data())
    _write_json(tmp_path / "loco_robust_metrics.json", _make_robust_data(cand_auc, cand_k))
    cfg = JudgeConfig(results_dir=tmp_path, out_path=tmp_path / "out.json")
    res = evaluate_preregistration(cfg)

    fold = res["candidates"]["robust"]["folds"]["crack_1.6"]
    assert fold["verdict"] == expected_verdict
    assert fold["base_auroc"] == 0.80
    assert fold["base_det_k"] == 3000


@pytest.mark.parametrize(
    ("lite_ms", "mlndt_ms", "expected_lighter"),
    [
        (1.5, 3.0, True),
        (3.5, 3.0, False),
        (3.0, 3.0, False),
    ],
)
def test_lite_lighter_flag(
    tmp_path: Path,
    lite_ms: float,
    mlndt_ms: float,
    expected_lighter: bool,
) -> None:
    """Lite candidate additionally sets lighter flag based on median latency."""
    _write_json(tmp_path / "separability_metrics.json", _make_baseline_data())
    _write_json(
        tmp_path / "lite_hybrid_metrics.json",
        _make_lite_data(lite_ms=lite_ms, mlndt_ms=mlndt_ms),
    )
    cfg = JudgeConfig(results_dir=tmp_path, out_path=tmp_path / "out.json")
    res = evaluate_preregistration(cfg)

    lite_block = res["candidates"]["lite"]
    assert lite_block["status"] == "evaluated"
    assert lite_block["lighter"] is expected_lighter
    assert lite_block["folds"]["crack_1.6"]["lighter"] is expected_lighter


def test_invariant_schema_unknown_when_keys_missing(tmp_path: Path) -> None:
    """Invariant file with unknown structure sets status schema unknown."""
    _write_json(tmp_path / "separability_metrics.json", _make_baseline_data())
    _write_json(tmp_path / "invariant_loco_metrics.json", {"unexpected_key": 123})
    cfg = JudgeConfig(results_dir=tmp_path, out_path=tmp_path / "out.json")
    res = evaluate_preregistration(cfg)

    inv = res["candidates"]["invariant"]
    assert inv["status"] == "schema unknown"
    assert inv["folds"] == {}


def test_main_cli_execution(tmp_path: Path) -> None:
    """CLI execution loads from results-dir and writes formatted payload to out."""
    _write_json(tmp_path / "separability_metrics.json", _make_baseline_data())
    _write_json(tmp_path / "loco_robust_metrics.json", _make_robust_data())
    _write_json(tmp_path / "lite_hybrid_metrics.json", _make_lite_data())
    _write_json(tmp_path / "invariant_loco_metrics.json", _make_invariant_data())

    out_file = tmp_path / "verdicts.json"
    argv = ["--results-dir", str(tmp_path), "--out", str(out_file)]
    payload = main(argv)

    assert out_file.exists()
    disk_data = json.loads(out_file.read_text(encoding="utf-8"))
    assert disk_data["candidates"]["robust"]["status"] == "evaluated"
    assert disk_data["candidates"]["lite"]["status"] == "evaluated"
    assert disk_data["candidates"]["invariant"]["status"] == "evaluated"
    assert payload["candidates"]["robust"]["folds"]["crack_1.6"]["verdict"] == "better"


def _make_shift_fold(
    held_out: str = "crack_1.6",
    status: str = "evaluated",
    det_rate: float = 0.80,
    fcr_rate: float = 0.02,
) -> dict[str, Any]:
    """Build fold dictionary for shift evaluation."""
    return {
        "held_out": held_out,
        "status": status,
        "summary": {
            "worst_case_detection_rate_0.01": det_rate,
            "worst_case_false_call_rate_0.01": fcr_rate,
        },
    }


def _make_shift_data(
    loco_folds: list[dict[str, Any]] | None = None,
    robust_folds: list[dict[str, Any]] | None = None,
    lite_folds: list[dict[str, Any]] | None = None,
    invariant_folds: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build test payload for shift robustness experiment."""
    families: dict[str, Any] = {}
    fams = [
        ("loco", loco_folds),
        ("robust", robust_folds),
        ("lite", lite_folds),
        ("invariant", invariant_folds),
    ]
    for name, folds in fams:
        if folds is not None:
            families[name] = {"status": "evaluated", "folds": folds}
        else:
            families[name] = {"status": "not run", "folds": []}
    return {"families": families}


def test_shift_missing_file_reports_not_run(tmp_path: Path) -> None:
    """Missing shift file sets candidates to not run without error."""
    _write_json(tmp_path / "separability_metrics.json", _make_baseline_data())
    cfg = JudgeConfig(results_dir=tmp_path, out_path=tmp_path / "out.json")
    res = evaluate_preregistration(cfg)

    assert "shift_robustness" in res
    shift_res = res["shift_robustness"]
    assert shift_res["baseline_family"] == "loco"
    assert shift_res["adr"] == "docs/decisions/0015-shift-robustness-preregistration.md"
    for name in ("robust", "lite", "invariant"):
        cand = shift_res["candidates"][name]
        assert cand["status"] == "not run"
        assert cand["folds"] == {}


@pytest.mark.parametrize(
    ("cand_det", "cand_fcr", "expected_verdict"),
    [
        (0.90, 0.02, "more shift-robust"),
        (0.90, 0.03, "more shift-robust"),
        (0.90, 0.0301, "no improvement"),
        (0.8999, 0.02, "no improvement"),
        (float("nan"), 0.02, "not evaluable"),
        (0.90, float("nan"), "not evaluable"),
    ],
)
def test_shift_verdict_rule_outcomes(
    tmp_path: Path,
    cand_det: float,
    cand_fcr: float,
    expected_verdict: str,
) -> None:
    """Test boundary condition, FCR limit, and NaN handling for shift robustness."""
    _write_json(tmp_path / "separability_metrics.json", _make_baseline_data())
    loco = [_make_shift_fold(det_rate=0.80, fcr_rate=0.02)]
    cand = [_make_shift_fold(det_rate=cand_det, fcr_rate=cand_fcr)]
    _write_json(tmp_path / "shift_robustness_metrics.json", _make_shift_data(loco, cand))
    cfg = JudgeConfig(results_dir=tmp_path, out_path=tmp_path / "out.json")
    res = evaluate_preregistration(cfg)

    fold = res["shift_robustness"]["candidates"]["robust"]["folds"]["crack_1.6"]
    assert fold["verdict"] == expected_verdict
    assert fold["base_worst_det"] == 0.80
    assert fold["base_worst_fcr"] == 0.02


@pytest.mark.parametrize(
    "shift_payload",
    [
        _make_shift_data(
            loco_folds=[_make_shift_fold("crack_4.0")],
            robust_folds=[_make_shift_fold("crack_1.6")],
        ),
        _make_shift_data(
            loco_folds=[_make_shift_fold("crack_1.6", status="not run")],
            robust_folds=[_make_shift_fold("crack_1.6")],
        ),
        {"families": {"robust": {"status": "evaluated", "folds": [_make_shift_fold()]}}},
    ],
)
def test_shift_missing_or_unevaluated_loco_not_evaluable(
    tmp_path: Path,
    shift_payload: dict[str, Any],
) -> None:
    """Missing or un-evaluated baseline fold leads to not evaluable verdict."""
    _write_json(tmp_path / "separability_metrics.json", _make_baseline_data())
    _write_json(tmp_path / "shift_robustness_metrics.json", shift_payload)
    cfg = JudgeConfig(results_dir=tmp_path, out_path=tmp_path / "out.json")
    res = evaluate_preregistration(cfg)

    fold = res["shift_robustness"]["candidates"]["robust"]["folds"]["crack_1.6"]
    assert fold["verdict"] == "not evaluable"
    assert fold["base_worst_det"] is None
    assert fold["delta_det"] is None


def test_shift_schema_fields_complete(tmp_path: Path) -> None:
    """Shift robustness payload contains all required schema keys."""
    _write_json(tmp_path / "separability_metrics.json", _make_baseline_data())
    loco = [_make_shift_fold()]
    rob = [_make_shift_fold(det_rate=0.91, fcr_rate=0.02)]
    _write_json(tmp_path / "shift_robustness_metrics.json", _make_shift_data(loco, rob))
    cfg = JudgeConfig(results_dir=tmp_path, out_path=tmp_path / "out.json")
    res = evaluate_preregistration(cfg)

    shift_block = res["shift_robustness"]
    assert "adr" in shift_block
    assert "rule" in shift_block
    assert "baseline_family" in shift_block
    assert "note" in shift_block
    fold_dict = shift_block["candidates"]["robust"]["folds"]["crack_1.6"]
    for key in (
        "base_worst_det",
        "cand_worst_det",
        "base_worst_fcr",
        "cand_worst_fcr",
        "delta_det",
        "verdict",
    ):
        assert key in fold_dict


def test_main_cli_with_shift_metrics(tmp_path: Path) -> None:
    """CLI execution includes shift robustness when metrics file exists."""
    _write_json(tmp_path / "separability_metrics.json", _make_baseline_data())
    loco = [_make_shift_fold(det_rate=0.80, fcr_rate=0.02)]
    rob = [_make_shift_fold(det_rate=0.90, fcr_rate=0.02)]
    _write_json(tmp_path / "shift_robustness_metrics.json", _make_shift_data(loco, rob))

    out_file = tmp_path / "verdicts.json"
    argv = ["--results-dir", str(tmp_path), "--out", str(out_file)]
    payload = main(argv)

    disk_data = json.loads(out_file.read_text(encoding="utf-8"))
    assert "shift_robustness" in disk_data
    assert disk_data["shift_robustness"]["candidates"]["robust"]["status"] == "evaluated"
    assert (
        payload["shift_robustness"]["candidates"]["robust"]["folds"]["crack_1.6"]["verdict"]
        == "more shift-robust"
    )
