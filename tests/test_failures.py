"""Unit tests for failure mode classification and artifact generation (project plan Phase 8).

Verifies classify_failure branches, image-based SNR, near_edge detection,
failure case rendering, unique case identifiers, and count-driven report generation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from experiments.run_failures import FoldContext  # noqa: E402

from utriage.report.failures import (  # noqa: E402
    build_failure_modes_md,
    classify_failure,
    compute_image_snr,
    is_near_edge,
    save_failure_case,
)


def _make_test_image(
    peak_row: int = 128, peak_val: float = 1000.0, noise_amp: float = 10.0
) -> np.ndarray:
    """Generate synthetic 256x256 image with known MAD and peak."""
    img = np.zeros((256, 256), dtype=np.float32)
    if noise_amp > 0.0:
        img[::2, :] = noise_amp
        img[1::2, :] = -noise_amp
    img[peak_row, 128] = peak_val
    return img


@pytest.mark.parametrize(
    ("peak_row", "expected"),
    [
        (5, True),  # near top edge (depth axis)
        (250, True),  # near bottom edge
        (128, False),  # center of image
        (25, True),  # exactly on default margin
        (26, False),  # just inside margin boundary
    ],
)
def test_near_edge_rule_branches(peak_row: int, expected: bool) -> None:
    """Verify is_near_edge on image arrays across edge and center locations."""
    img = _make_test_image(peak_row=peak_row, peak_val=500.0, noise_amp=10.0)
    assert is_near_edge(img) is expected


def test_near_edge_custom_margin() -> None:
    """Verify is_near_edge with custom margin parameter."""
    img = _make_test_image(peak_row=15, peak_val=500.0, noise_amp=10.0)
    assert is_near_edge(img, margin=10) is False
    assert is_near_edge(img, margin=20) is True


def test_compute_image_snr_branches() -> None:
    """Verify compute_image_snr on high SNR, low SNR, and zero noise arrays."""
    high_snr_img = _make_test_image(peak_row=128, peak_val=500.0, noise_amp=10.0)
    assert compute_image_snr(high_snr_img) == pytest.approx(50.0)

    low_snr_img = _make_test_image(peak_row=128, peak_val=15.0, noise_amp=10.0)
    assert compute_image_snr(low_snr_img) == pytest.approx(1.5)
    assert compute_image_snr(low_snr_img) < 2.0

    flat_img = np.zeros((256, 256), dtype=np.float32)
    assert compute_image_snr(flat_img) == 0.0


@pytest.mark.parametrize(
    ("meta", "peak_row", "peak_val", "expected"),
    [
        ({"equivalent_flawsize": 3.0}, 5, 500.0, "near_edge"),
        ({"equivalent_flawsize": 3.0}, 128, 15.0, "low_snr"),
        ({"equivalent_flawsize": 1.0}, 128, 500.0, "small_flaw"),
        ({"equivalent_flawsize": 3.0}, 128, 500.0, "unclassified"),
    ],
)
def test_classify_miss_branches(meta: dict, peak_row: int, peak_val: float, expected: str) -> None:
    """Verify classify_failure miss causes using synthetic image arrays."""
    img = _make_test_image(peak_row=peak_row, peak_val=peak_val, noise_amp=10.0)
    assert classify_failure(meta, "miss", p10_threshold=1.5, img=img) == expected


@pytest.mark.parametrize(
    ("meta", "kind", "expected"),
    [
        ({"near_edge": True}, "miss", "near_edge"),
        ({"snr": 1.5, "equivalent_flawsize": 3.0}, "miss", "low_snr"),
        ({"snr": 5.0, "equivalent_flawsize": 1.0}, "miss", "small_flaw"),
        ({"snr": 5.0, "equivalent_flawsize": 3.0}, "miss", "unclassified"),
        ({"image_max": 2500, "noise_threshold": 1000}, "false_call", "artefact_like"),
        ({"image_max": 800, "noise_threshold": 1000}, "false_call", "unclassified"),
        ({"image_max": 5000}, "unknown_kind", "unclassified"),
    ],
)
def test_classify_failure_meta_fallbacks(meta: dict, kind: str, expected: str) -> None:
    """Verify fallback classification when raw image array is not supplied."""
    assert classify_failure(meta, kind, p10_threshold=1.5) == expected


def test_classify_failure_custom_p10() -> None:
    """Verify custom detected-size p10 threshold override."""
    meta = {"snr": 5.0, "equivalent_flawsize": 1.8}
    assert classify_failure(meta, "miss", p10_threshold=2.0) == "small_flaw"
    assert classify_failure(meta, "miss", p10_threshold=1.5) == "unclassified"


def test_unique_case_ids_different_batches() -> None:
    """Verify case IDs from different batch stems do not overwrite each other."""
    from experiments.run_failures import _build_case_id

    id_batch1 = _build_case_id("miss", "crack_1.6", "batch_01", 16)
    id_batch2 = _build_case_id("miss", "crack_1.6", "batch_02", 16)

    assert id_batch1 != id_batch2
    assert "batch_01" in id_batch1
    assert "batch_02" in id_batch2
    assert id_batch1 == "miss_crack_1.6_batch_01_16"
    assert id_batch2 == "miss_crack_1.6_batch_02_16"


def test_save_failure_case_artifacts(tmp_path: Path) -> None:
    """Verify PNG plot and evidence JSON generation in destination directory."""
    img = np.zeros((256, 256), dtype=np.uint16)
    img[120, 80] = 4500
    meta = {"size": "2.5", "noise_threshold": "950"}

    saved_path = save_failure_case(img, meta, score=0.88, out_dir=tmp_path, name="case_001")
    assert saved_path == tmp_path / "case_001.png"
    assert saved_path.exists()
    assert saved_path.stat().st_size > 0

    json_path = tmp_path / "case_001.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["name"] == "case_001"
    assert payload["score"] == pytest.approx(0.88)
    assert payload["peak_amplitude"] == pytest.approx(4500.0)
    assert payload["peak_location"] == [120, 80]
    assert payload["png_saved"] is True


def test_save_failure_case_png_cap(tmp_path: Path) -> None:
    """Verify save_failure_case respects save_png=False by writing only JSON."""
    img = np.zeros((256, 256), dtype=np.uint16)
    img[50, 50] = 3000
    meta = {"size": "1.0"}

    ret_path = save_failure_case(
        img, meta, score=0.45, out_dir=tmp_path, name="case_no_png", save_png=False
    )
    assert ret_path == tmp_path / "case_no_png.json"
    assert not (tmp_path / "case_no_png.png").exists()
    assert (tmp_path / "case_no_png.json").exists()

    payload = json.loads(ret_path.read_text(encoding="utf-8"))
    assert payload["png_saved"] is False


def test_write_summary_json(tmp_path: Path) -> None:
    """Verify summary.json accurately records per-fold counts per cause."""
    from experiments.run_failures import _write_summary_json

    summaries = [
        {
            "fold": "crack_1.6",
            "n_misses": 15,
            "miss_causes": {"small_flaw": 10, "near_edge": 5},
            "n_false_calls": 8,
            "false_call_causes": {"artefact_like": 8},
        }
    ]
    out_file = _write_summary_json(tmp_path, summaries)
    assert out_file == tmp_path / "summary.json"
    assert out_file.exists()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert "crack_1.6" in data
    assert data["crack_1.6"]["n_misses"] == 15
    assert data["crack_1.6"]["miss_causes"] == {"small_flaw": 10, "near_edge": 5}
    assert data["crack_1.6"]["n_false_calls"] == 8
    assert data["crack_1.6"]["false_call_causes"] == {"artefact_like": 8}


def test_process_misses_png_capping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _process_misses caps saved cases to max_cases without per-case JSON."""
    from experiments.run_failures import FoldContext, _process_misses

    n_cases = 25
    fake_img = _make_test_image(peak_row=128, peak_val=500.0, noise_amp=10.0)
    fake_meta = [{"equivalent_flawsize": 3.0} for _ in range(n_cases)]

    monkeypatch.setattr("experiments.run_failures.read_image", lambda path, idx: fake_img)
    monkeypatch.setattr("experiments.run_failures.read_flaw_meta", lambda path: fake_meta)

    test_items = [(Path("batch_01.bins"), i) for i in range(n_cases)]
    flawed_mask = np.ones(n_cases, dtype=bool)
    unflawed_mask = np.zeros(n_cases, dtype=bool)
    hits = np.zeros(n_cases, dtype=bool)
    probs = np.linspace(0.01, 0.49, n_cases)

    ctx = FoldContext("crack_1.6", test_items, flawed_mask, unflawed_mask, hits, probs, tmp_path)
    recs, counts = _process_misses(ctx, p10=1.5, max_cases=20)

    assert len(recs) == 20
    assert sum(counts.values()) == n_cases
    assert sum(1 for r in recs if r["has_png"]) == 20
    assert len(list(tmp_path.glob("*.png"))) == 20
    assert len(list(tmp_path.glob("*.json"))) == 0


def _make_fake_fold_context(
    fold_name: str,
    n_misses: int,
    n_fc: int,
    fold_dir: Path,
) -> FoldContext:
    """Create synthetic FoldContext with specified miss and false call counts."""
    from experiments.run_failures import FoldContext

    total = n_misses + n_fc
    test_items = [(Path("batch_01.bins"), i) for i in range(total)]
    flawed = np.array([True] * n_misses + [False] * n_fc)
    unflawed = ~flawed
    hits = np.array([False] * n_misses + [True] * n_fc)
    probs = np.linspace(0.1, 0.9, total)
    return FoldContext(fold_name, test_items, flawed, unflawed, hits, probs, fold_dir)


def _verify_cases_json_structure(cases_path: Path, expected_count: int) -> None:
    """Verify cases.json schema, category counts, and case attributes."""
    data = json.loads(cases_path.read_text(encoding="utf-8"))
    assert len(data["misses"]) == expected_count
    assert len(data["false_calls"]) == expected_count

    first_case = data["misses"][0]
    for key in ("name", "fold", "kind", "score", "cause", "png_file", "evidence"):
        assert key in first_case
    assert isinstance(first_case["evidence"], dict)
    assert (cases_path.parent / first_case["png_file"]).exists()


def test_fake_fold_eval_counts_and_cases_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify cause counts cover all cases while only max_cases are written to cases.json."""
    from experiments.run_failures import (
        _process_false_calls,
        _process_misses,
        _write_cases_json,
    )

    fake_img = _make_test_image(peak_row=128, peak_val=500.0, noise_amp=10.0)
    fake_meta = [{"equivalent_flawsize": 3.0} for _ in range(50)]
    monkeypatch.setattr("experiments.run_failures.read_image", lambda path, idx: fake_img)
    monkeypatch.setattr("experiments.run_failures.read_flaw_meta", lambda path: fake_meta)

    ctx = _make_fake_fold_context("fold_a", 25, 25, tmp_path)
    miss_recs, miss_counts = _process_misses(ctx, p10=1.5, max_cases=20)
    fc_recs, fc_counts = _process_false_calls(ctx, max_cases=20)
    cases_file = _write_cases_json(tmp_path, miss_recs, fc_recs)

    assert sum(miss_counts.values()) == 25
    assert sum(fc_counts.values()) == 25
    assert len(miss_recs) == 20
    assert len(fc_recs) == 20
    _verify_cases_json_structure(cases_file, expected_count=20)
    assert len(list(tmp_path.glob("*.png"))) == 40
    assert list(tmp_path.glob("*.json")) == [cases_file]


def test_fake_fold_custom_max_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify custom max_cases option restricts saved cases while counts cover all."""
    from experiments.run_failures import (
        _process_false_calls,
        _process_misses,
        _write_cases_json,
    )

    fake_img = _make_test_image(peak_row=128, peak_val=500.0, noise_amp=10.0)
    fake_meta = [{"equivalent_flawsize": 3.0} for _ in range(50)]
    monkeypatch.setattr("experiments.run_failures.read_image", lambda path, idx: fake_img)
    monkeypatch.setattr("experiments.run_failures.read_flaw_meta", lambda path: fake_meta)

    ctx = _make_fake_fold_context("fold_b", 25, 25, tmp_path)
    miss_recs, miss_counts = _process_misses(ctx, p10=1.5, max_cases=5)
    fc_recs, fc_counts = _process_false_calls(ctx, max_cases=5)
    cases_file = _write_cases_json(tmp_path, miss_recs, fc_recs)

    assert sum(miss_counts.values()) == 25
    assert sum(fc_counts.values()) == 25
    assert len(miss_recs) == 5
    assert len(fc_recs) == 5
    _verify_cases_json_structure(cases_file, expected_count=5)
    assert len(list(tmp_path.glob("*.png"))) == 10
    assert list(tmp_path.glob("*.json")) == [cases_file]


def test_report_states_case_capping(tmp_path: Path) -> None:
    """Verify failure modes report text states that cause counts cover all cases."""
    from experiments.run_failures import FailureConfig, _write_failure_modes_report

    summaries, cases, _ = _make_fake_markdown_inputs()
    report_file = tmp_path / "failure_modes.md"
    cfg = FailureConfig(out_dir=tmp_path, quick=False, device="cpu", max_cases=20)
    _write_failure_modes_report(report_file, summaries, cases, cfg)

    content = report_file.read_text(encoding="utf-8")
    expected_statement = (
        "cause counts cover all cases while files are kept for the top 20 of each kind."
    )
    assert expected_statement in content
    assert "## Evaluated Failure Cases" in content


def _fake_sim_data() -> dict:
    """Construct mock simulation generalization payload."""
    return {
        "test_sets": {
            "in_distribution": {
                "rules": {"false_calls": {"rate": 0.03}},
                "ml_1d": {"false_calls": {"rate": 0.13, "n_false": 55, "n_scans": 410}},
                "union": {"false_calls": {"rate": 0.15}},
            },
            "freq_10mhz": {
                "rules": {"false_calls": {"rate": 0.64}},
                "ml_1d": {"false_calls": {"rate": 0.47, "n_false": 197, "n_scans": 420}},
                "union": {"false_calls": {"rate": 0.69}},
            },
            "noise_above_range": {
                "rules": {"false_calls": {"rate": 0.53}},
                "ml_1d": {"false_calls": {"rate": 0.47, "n_false": 211, "n_scans": 450}},
                "union": {"false_calls": {"rate": 0.66}},
            },
        }
    }


def _make_fake_markdown_inputs() -> tuple[list[dict], list[dict], dict]:
    """Construct mock data structures for failure modes markdown generation."""
    summaries = [
        {
            "fold": "crack_1.6",
            "n_misses": 4,
            "miss_causes": {"small_flaw": 2, "low_snr": 1, "near_edge": 1, "unclassified": 0},
            "n_false_calls": 3,
            "false_call_causes": {"artefact_like": 2, "unclassified": 1},
        }
    ]
    cases = [
        {
            "name": "miss_001",
            "fold": "crack_1.6",
            "kind": "miss",
            "score": 0.12,
            "cause": "small_flaw",
            "has_png": True,
            "evidence": "size=0.80mm, snr=3.2",
        }
    ]
    return summaries, cases, _fake_sim_data()


def test_build_failure_modes_md() -> None:
    """Verify markdown output contains required sections, counts, and descriptions."""
    summaries, cases, sim_data = _make_fake_markdown_inputs()
    md = build_failure_modes_md(summaries, cases, sim_data, png_count=1)

    assert "# Failure Modes and Generalization Boundaries" in md
    assert "## Plain-Language Summary" in md
    assert "## LOCO Folds Failure Breakdown" in md
    assert "| crack_1.6 | 4 | 1 | 1 | 2 | 0 | 3 | 2 | 1 |" in md
    assert "## Evaluated Failure Cases" in md
    assert "| miss_001 | crack_1.6 | miss | 0.120 | small_flaw | size=0.80mm, snr=3.2 |" in md
    assert "## Synthetic Held-Out Regions Generalization" in md
    assert "10 MHz Frequency Shift" in md
    assert "Noise Above Range" in md


def test_summary_text_built_from_counts_no_fixed_prose() -> None:
    """Verify summary section is dynamically generated from counts without fixed prose."""
    summaries, cases, sim_data = _make_fake_markdown_inputs()
    md = build_failure_modes_md(summaries, cases, sim_data, png_count=1)

    assert "**Fold crack_1.6**: 4 misses (top cause: small_flaw with 2 cases, 50.0%)" in md
    assert "top cause: artefact_like with 2 cases, 66.7%" in md

    # Ensure hand-written explanatory prose was removed
    assert "grain noise floor" not in md
    assert "acoustic interface reflections" not in md
    assert "Rayleigh scattering" not in md
    assert "Background grain scattering peaks" not in md


def test_false_calls_only_at_or_above_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify false calls are counted and collected only when clean score >= 0.5."""
    from experiments.run_failures import FoldContext, _process_false_calls

    fake_img = _make_test_image(peak_row=128, peak_val=500.0, noise_amp=10.0)
    monkeypatch.setattr("experiments.run_failures.read_image", lambda path, idx: fake_img)
    monkeypatch.setattr("experiments.run_failures.read_flaw_meta", lambda path: [{}] * 6)

    test_items = [(Path("batch_01.bins"), i) for i in range(6)]
    flawed = np.array([True, True, False, False, False, False])
    probs = np.array([0.90, 0.30, 0.80, 0.55, 0.48, 0.10])
    ctx = FoldContext("f1", test_items, flawed, ~flawed, probs >= 0.5, probs, tmp_path)

    fc_recs, fc_counts = _process_false_calls(ctx, max_cases=20)
    assert sum(fc_counts.values()) == 2
    assert len(fc_recs) == 2
    assert all(r["kind"] == "false_call" and r["score"] >= 0.5 for r in fc_recs)

    zero_probs = np.array([0.90, 0.30, 0.49, 0.40, 0.30, 0.10])
    zero_ctx = FoldContext(
        "f0", test_items, flawed, ~flawed, zero_probs >= 0.5, zero_probs, tmp_path
    )
    zero_recs, zero_counts = _process_false_calls(zero_ctx, max_cases=20)
    assert zero_recs == []
    assert zero_counts == {}


def test_near_misses_clean_saved_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify clean images below threshold are saved as near misses and not false calls."""
    from experiments.run_failures import (
        FoldContext,
        _process_near_misses_clean,
        _write_cases_json,
    )

    fake_img = _make_test_image(peak_row=128, peak_val=500.0, noise_amp=10.0)
    monkeypatch.setattr("experiments.run_failures.read_image", lambda path, idx: fake_img)
    monkeypatch.setattr("experiments.run_failures.read_flaw_meta", lambda path: [{}] * 6)

    test_items = [(Path("batch_01.bins"), i) for i in range(6)]
    flawed = np.array([True, True, False, False, False, False])
    probs = np.array([0.90, 0.30, 0.80, 0.55, 0.48, 0.10])
    ctx = FoldContext("f1", test_items, flawed, ~flawed, probs >= 0.5, probs, tmp_path)

    clean_recs = _process_near_misses_clean(ctx, max_cases=2)
    assert len(clean_recs) == 2
    assert [r["score"] for r in clean_recs] == [pytest.approx(0.48), pytest.approx(0.10)]
    assert all(r["kind"] == "near_miss_clean" for r in clean_recs)

    cases_file = _write_cases_json(tmp_path, [], [], clean_recs)
    data = json.loads(cases_file.read_text(encoding="utf-8"))
    assert len(data["false_calls"]) == 0
    assert len(data["near_misses_clean"]) == 2


def test_zero_false_calls_markdown_summary() -> None:
    """Verify markdown states '0 false calls; top clean scores listed as near misses.'"""
    summaries = [
        {
            "fold": "crack_1.6",
            "n_misses": 5,
            "miss_causes": {"small_flaw": 5},
            "n_false_calls": 0,
            "false_call_causes": {},
        }
    ]
    md = build_failure_modes_md(summaries, [], png_count=0)
    assert "0 false calls; top clean scores listed as near misses." in md


def test_png_file_size_under_60kb(tmp_path: Path) -> None:
    """Verify generated failure case PNG is under 60 KB for 256x256 image."""
    img = _make_test_image(peak_row=128, peak_val=1000.0, noise_amp=10.0)
    meta = {"max_amplitude": "1000", "noise_threshold": "500"}
    png_path = save_failure_case(img, meta, score=0.45, out_dir=tmp_path, name="size_check")
    assert png_path.exists()
    assert 0 < png_path.stat().st_size < 60 * 1024
