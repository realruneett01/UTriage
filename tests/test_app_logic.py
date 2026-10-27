"""Tests for load_evidence and evidence display in app_logic and Streamlit app."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        del sys.modules["utriage"]

import pytest  # noqa: E402

from utriage.app_logic import load_evidence  # noqa: E402


@dataclass(frozen=True)
class MissingCase:
    """Fixture case parameter for missing files."""

    inv: bool
    claims: bool
    fig: bool
    expected_missing: str


MISSING_CASES = (
    MissingCase(False, True, True, "data_inventory.md"),
    MissingCase(True, False, True, "claims.json"),
    MissingCase(True, True, False, "figures/generalization_matrix.png"),
)


def _populate_evidence_dir(dir_path: Path, case: MissingCase) -> None:
    """Write fixture files into a temporary directory based on MissingCase flags."""
    if case.inv:
        inv_file = dir_path / "data_inventory.md"
        inv_file.write_text(
            "# Real-data evidence limits\n\n"
            "Used in evaluation: 9 distinct physical flaws (ML-NDT: 3, NDT_ML_Flaw: 6)\n",
            encoding="utf-8",
        )
    if case.claims:
        claims_file = dir_path / "claims.json"
        sample_claims = [
            {
                "text": "Historical benchmark superseded.",
                "scope_limit": "one austenitic weld PAUT public dataset (ML-NDT)",
            }
        ]
        claims_file.write_text(json.dumps(sample_claims), encoding="utf-8")
    if case.fig:
        fig_dir = dir_path / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        (fig_dir / "generalization_matrix.png").write_bytes(b"dummy_png_bytes")


def test_load_evidence_all_present(tmp_path: Path) -> None:
    """Verify load_evidence loads all components when files are present."""
    all_present = MissingCase(True, True, True, "")
    _populate_evidence_dir(tmp_path, all_present)
    res = load_evidence(tmp_path)

    assert "Used in evaluation" in res["used_line"]
    assert len(res["claims"]) == 1
    assert res["claims"][0]["text"] == "Historical benchmark superseded."
    assert res["claims"][0]["scope_limit"] == "one austenitic weld PAUT public dataset (ML-NDT)"
    assert res["matrix_png"] == tmp_path / "figures" / "generalization_matrix.png"
    assert res["missing"] == []


@pytest.mark.parametrize("case", MISSING_CASES, ids=lambda c: c.expected_missing)
def test_load_evidence_missing_individual(tmp_path: Path, case: MissingCase) -> None:
    """Verify missing files are recorded in missing list without crashing."""
    _populate_evidence_dir(tmp_path, case)
    res = load_evidence(tmp_path)
    assert case.expected_missing in res["missing"]


def test_load_evidence_all_missing(tmp_path: Path) -> None:
    """Verify load_evidence returns safe empty values when directory is empty."""
    res = load_evidence(tmp_path)
    assert res["used_line"] == ""
    assert res["claims"] == []
    assert res["matrix_png"] is None
    assert set(res["missing"]) == {
        "data_inventory.md",
        "claims.json",
        "figures/generalization_matrix.png",
    }


def test_load_evidence_committed_results() -> None:
    """Verify load_evidence against real committed results directory."""
    res_dir = Path("results")
    if not res_dir.is_dir():
        pytest.skip("results directory not found")
    res = load_evidence(res_dir)
    assert res["used_line"].startswith("Used")
    assert len(res["claims"]) > 0
    assert res["matrix_png"] is not None
    assert res["missing"] == []


def test_streamlit_evidence_page() -> None:
    """Verify Page 7 renders without exception and displays advisory banner."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app_file = (Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py").resolve()
    at = AppTest.from_file(str(app_file))
    at.run()
    assert not at.exception

    at.sidebar.radio[0].set_value("7 Evidence and limits").run()
    assert not at.exception
    assert any("Advisory tool" in info.value for info in at.info)
