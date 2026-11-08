"""Tests for UTriage Streamlit app and app_logic (Phase 6)."""

from __future__ import annotations

import csv
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

_SRC = str(Path(__file__).resolve().parents[1] / "src")
_APP = str(Path(__file__).resolve().parents[1] / "app")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _APP not in sys.path:
    sys.path.insert(0, _APP)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        del sys.modules["utriage"]

import matplotlib  # noqa: E402
import pytest  # noqa: E402
from streamlit_app import _loadable_scans  # noqa: E402

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
EXPECTED_AUDIT_HEADERS = [
    "timestamp",
    "bar_id",
    "item_index",
    "source",
    "category",
    "decision",
    "comment",
    "inspector",
]
PAGES = [
    "1 Load scan",
    "2 C-scan view",
    "3 Indication list",
    "4 Data quality",
    "5 Decision log",
    "6 Report",
]


@pytest.fixture
def sample_session() -> Session:
    """Provide a Session initialized with the sample scan."""
    return load_session(SAMPLE_PATH, model_path=None)


@pytest.fixture
def running_app():
    """Initialize and run AppTest on streamlit_app.py."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app_target = (Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py").resolve()
    at = AppTest.from_file(str(app_target))
    at.run()
    assert not at.exception
    return at


def assert_synthetic_figure(fig: plt.Figure) -> None:
    """Verify figure builds and includes the synthetic data watermark."""
    assert fig is not None
    assert "Synthetic data" in fig.axes[0].get_title()


def _build_cscan(session: Session) -> plt.Figure:
    return cscan_figure(session)


def _build_ascan(session: Session) -> plt.Figure:
    return ascan_figure(session, ia=0, ja=0)


def test_load_session_sample(sample_session: Session) -> None:
    """Verify load_session correctly initializes a Session with sample scan."""
    assert isinstance(sample_session, Session)
    assert sample_session.scan is not None
    assert sample_session.evaluation is not None
    assert isinstance(sample_session.items, list)
    assert sample_session.synthetic is True
    assert sample_session.model_loaded is False


def test_record_decision_valid(sample_session: Session, tmp_path: Path) -> None:
    """Verify record_decision appends to session decisions and writes to audit CSV."""
    log_csv = tmp_path / "audit.csv"

    record_decision(
        session=sample_session,
        item_index=0,
        decision="accepted",
        comment="Cleared by Level II",
        inspector="Insp-42",
        log_path=log_csv,
    )

    assert len(sample_session.decisions) == 1
    assert sample_session.decisions[0].decision == "accepted"
    assert sample_session.decisions[0].inspector == "Insp-42"
    assert log_csv.is_file()

    with log_csv.open("r", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))

    assert len(rows) == 2  # Header + 1 record
    assert rows[0] == EXPECTED_AUDIT_HEADERS
    assert rows[1][2] == "0"
    assert rows[1][5] == "accepted"
    assert rows[1][6] == "Cleared by Level II"
    assert rows[1][7] == "Insp-42"


@pytest.mark.parametrize("bad_decision", ["pass", "fail", "ok", "APPROVED", ""])
def test_record_decision_invalid(
    sample_session: Session, tmp_path: Path, bad_decision: str
) -> None:
    """Verify record_decision raises ValueError on invalid decisions."""
    log_csv = tmp_path / "audit.csv"

    with pytest.raises(ValueError, match="decision must be accepted, rejected or rescan"):
        record_decision(
            session=sample_session,
            item_index=0,
            decision=bad_decision,
            comment="bad",
            inspector="Insp-1",
            log_path=log_csv,
        )


@pytest.mark.parametrize(
    "builder",
    [_build_cscan, _build_ascan],
    ids=["cscan", "ascan"],
)
def test_figures_build(sample_session: Session, builder: Callable[[Session], plt.Figure]) -> None:
    """Verify A-scan and C-scan figures build without error and include Synthetic label."""
    fig = builder(sample_session)
    try:
        assert_synthetic_figure(fig)
    finally:
        plt.close(fig)


def test_streamlit_app_smoke(running_app: Any) -> None:
    """Verify AppTest executes app/streamlit_app.py without exception on default page."""
    assert not running_app.exception


def test_streamlit_app_all_pages(running_app: Any) -> None:
    """Verify AppTest executes across all 6 navigation pages without exception."""
    for page_name in PAGES:
        running_app.sidebar.radio[0].set_value(page_name).run()
        assert not running_app.exception


@pytest.mark.parametrize(
    ("files", "expected_loadable_names", "expected_errors"),
    [
        (
            [Path("bar_0001.npz"), Path("bar_0001.json")],
            ["bar_0001.npz"],
            [],
        ),
        (
            [Path("bar_0001.npz")],
            [],
            ["bar_0001.npz: missing bar_0001.json - upload the .npz together with its .json file."],
        ),
        (
            [Path("scan.csv")],
            ["scan.csv"],
            [],
        ),
        (
            [Path("bar_0001.json")],
            [],
            [],
        ),
        (
            [
                Path("bar_0001.npz"),
                Path("bar_0001.json"),
                Path("bar_0002.npz"),
                Path("bar_0002.json"),
            ],
            ["bar_0001.npz", "bar_0002.npz"],
            [],
        ),
    ],
    ids=["npz_json_pair", "npz_alone", "csv_alone", "json_alone", "two_pairs"],
)
def test_loadable_scans_cases(
    files: list[Path],
    expected_loadable_names: list[str],
    expected_errors: list[str],
) -> None:
    """Check scan candidate filtering and missing-pair messages."""
    loadable, errors = _loadable_scans(files)
    assert [p.name for p in loadable] == expected_loadable_names
    assert errors == expected_errors


def test_loadable_scans_end_to_end(tmp_path: Path) -> None:
    """Check pairing and session loading from copied sample files."""
    sample_npz = Path("data/samples/ti64/bar_0000.npz")
    sample_json = Path("data/samples/ti64/bar_0000.json")
    dst_npz = tmp_path / "bar_0000.npz"
    dst_json = tmp_path / "bar_0000.json"
    dst_npz.write_bytes(sample_npz.read_bytes())
    dst_json.write_bytes(sample_json.read_bytes())

    loadable, errors = _loadable_scans([dst_npz, dst_json])
    assert errors == []
    assert len(loadable) == 1
    session = load_session(loadable[0], model_path=None)
    assert session.scan is not None
