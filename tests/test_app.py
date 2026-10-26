"""Tests for UTriage Streamlit app and app_logic (Phase 6)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        del sys.modules["utriage"]

import matplotlib  # noqa: E402
import pytest  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utriage.app_logic import (  # noqa: E402
    Session,
    ascan_figure,
    cscan_figure,
    load_session,
    record_decision,
)

SAMPLE_PATH = Path("data/samples/ti64/bar_0000.npz")


def test_load_session_sample() -> None:
    """Verify load_session correctly initializes a Session with sample scan."""
    session = load_session(SAMPLE_PATH, model_path=None)
    assert isinstance(session, Session)
    assert session.scan is not None
    assert session.evaluation is not None
    assert isinstance(session.items, list)
    assert session.synthetic is True
    assert session.model_loaded is False


def test_record_decision_valid(tmp_path: Path) -> None:
    """Verify record_decision appends to session decisions and writes to audit CSV."""
    session = load_session(SAMPLE_PATH, model_path=None)
    log_csv = tmp_path / "audit.csv"

    record_decision(
        session=session,
        item_index=0,
        decision="accepted",
        comment="Cleared by Level II",
        inspector="Insp-42",
        log_path=log_csv,
    )

    assert len(session.decisions) == 1
    assert session.decisions[0].decision == "accepted"
    assert session.decisions[0].inspector == "Insp-42"
    assert log_csv.is_file()

    with log_csv.open("r", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))

    assert len(rows) == 2  # Header + 1 record
    assert rows[0] == [
        "timestamp",
        "bar_id",
        "item_index",
        "source",
        "category",
        "decision",
        "comment",
        "inspector",
    ]
    assert rows[1][2] == "0"
    assert rows[1][5] == "accepted"
    assert rows[1][6] == "Cleared by Level II"
    assert rows[1][7] == "Insp-42"


@pytest.mark.parametrize("bad_decision", ["pass", "fail", "ok", "APPROVED", ""])
def test_record_decision_invalid(tmp_path: Path, bad_decision: str) -> None:
    """Verify record_decision raises ValueError on invalid decisions."""
    session = load_session(SAMPLE_PATH, model_path=None)
    log_csv = tmp_path / "audit.csv"

    with pytest.raises(ValueError, match="decision must be accepted, rejected or rescan"):
        record_decision(
            session=session,
            item_index=0,
            decision=bad_decision,
            comment="bad",
            inspector="Insp-1",
            log_path=log_csv,
        )


def test_figures_build() -> None:
    """Verify A-scan and C-scan figures build without error and include Synthetic label."""
    session = load_session(SAMPLE_PATH, model_path=None)

    fig_c = cscan_figure(session)
    assert fig_c is not None
    assert "Synthetic data" in fig_c.axes[0].get_title()

    fig_a = ascan_figure(session, ia=0, ja=0)
    assert fig_a is not None
    assert "Synthetic data" in fig_a.axes[0].get_title()

    plt.close(fig_c)
    plt.close(fig_a)


def test_streamlit_app_smoke() -> None:
    """Verify AppTest executes app/streamlit_app.py without exception on default page."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    # Ensure relative path resolution from repository root
    app_target = (Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py").resolve()
    at = AppTest.from_file(str(app_target))
    at.run()
    assert not at.exception


def test_streamlit_app_all_pages() -> None:
    """Verify AppTest executes across all 6 navigation pages without exception."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app_target = (Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py").resolve()
    at = AppTest.from_file(str(app_target))
    at.run()
    assert not at.exception

    pages = [
        "1 Load scan",
        "2 C-scan view",
        "3 Indication list",
        "4 Data quality",
        "5 Decision log",
        "6 Report",
    ]
    for page_name in pages:
        at.sidebar.radio[0].set_value(page_name).run()
        assert not at.exception
