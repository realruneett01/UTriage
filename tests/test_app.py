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
from streamlit_app import (  # noqa: E402
    _auto_pair,
    _find_matching_json,
    _load_all,
    _loadable_scans,
)

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
            [
                "bar_0001.npz: no matching bar_0001.json found automatically - "
                "upload the .npz together with its .json file."
            ],
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


def _setup_test_scan(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Copy sample scan files to root folder and upload folder."""
    root_dir = tmp_path / "root" / "a"
    root_dir.mkdir(parents=True)
    src_npz = Path("data/samples/ti64/bar_0000.npz")
    src_json = Path("data/samples/ti64/bar_0000.json")
    root_npz = root_dir / "bar_0000.npz"
    root_json = root_dir / "bar_0000.json"
    root_npz.write_bytes(src_npz.read_bytes())
    root_json.write_bytes(src_json.read_bytes())
    up_dir = tmp_path / "up"
    up_dir.mkdir(parents=True)
    up_npz = up_dir / "bar_0000.npz"
    up_npz.write_bytes(src_npz.read_bytes())
    return root_dir, up_dir, up_npz


def test_auto_pair_finds_and_loads(tmp_path: Path) -> None:
    """Test auto-pair finds matching sibling JSON and allows loading."""
    root_dir, _up_dir, up_npz = _setup_test_scan(tmp_path)
    saved, notes = _auto_pair([up_npz], [root_dir.parent])
    assert len(saved) == 2
    assert any("found automatically" in n for n in notes)
    copied_json = up_npz.parent / f"{up_npz.stem}.json"
    assert copied_json.is_file()
    loadable, errors = _loadable_scans(saved)
    assert errors == []
    assert loadable == [up_npz]
    session = load_session(loadable[0], model_path=None)
    assert session.scan is not None


def test_auto_pair_different_bytes(tmp_path: Path) -> None:
    """Test that same-name NPZ with mismatched bytes does not pair."""
    root_dir, _up_dir, up_npz = _setup_test_scan(tmp_path)
    (root_dir / "bar_0000.npz").write_bytes(b"different content")
    saved, notes = _auto_pair([up_npz], [root_dir.parent])
    assert saved == [up_npz]
    assert notes == []
    loadable, errors = _loadable_scans(saved)
    assert loadable == []
    assert errors == [
        "bar_0000.npz: no matching bar_0000.json found automatically - "
        "upload the .npz together with its .json file."
    ]


def test_auto_pair_already_paired(tmp_path: Path) -> None:
    """Test auto-pair adds nothing when JSON is already uploaded."""
    root_dir, _up_dir, up_npz = _setup_test_scan(tmp_path)
    up_json = up_npz.parent / "bar_0000.json"
    up_json.write_text("{}", encoding="utf-8")
    saved, notes = _auto_pair([up_npz, up_json], [root_dir.parent])
    assert saved == [up_npz, up_json]
    assert notes == []


def test_auto_pair_missing_root_dir(tmp_path: Path) -> None:
    """Test auto-pair handles nonexistent search root directories gracefully."""
    up_dir = tmp_path / "up"
    up_dir.mkdir()
    up_npz = up_dir / "sample.npz"
    up_npz.write_bytes(b"scan-data")
    missing_root = tmp_path / "does_not_exist"
    saved, notes = _auto_pair([up_npz], [missing_root])
    assert saved == [up_npz]
    assert notes == []


def test_find_matching_json_skips_self(tmp_path: Path) -> None:
    """Test helper skips file itself when candidate matches NPZ path."""
    npz = tmp_path / "bar_0000.npz"
    npz.write_bytes(b"data")
    (tmp_path / "bar_0000.json").write_text("{}", encoding="utf-8")
    match = _find_matching_json(npz, [tmp_path])
    assert match is None


def test_load_all_valid_pairs(tmp_path: Path) -> None:
    """Test loading multiple valid scan pairs with _load_all."""
    src_npz = Path("data/samples/ti64/bar_0000.npz")
    src_json = Path("data/samples/ti64/bar_0000.json")
    for prefix in ("a_0000", "b_0000"):
        (tmp_path / f"{prefix}.npz").write_bytes(src_npz.read_bytes())
        (tmp_path / f"{prefix}.json").write_bytes(src_json.read_bytes())
    paths = [tmp_path / "a_0000.npz", tmp_path / "b_0000.npz"]
    sessions, errors = _load_all(paths, model=None)
    assert errors == []
    assert len(sessions) == 2
    assert "a_0000.npz" in sessions
    assert "b_0000.npz" in sessions
    assert isinstance(sessions["a_0000.npz"], Session)
    assert isinstance(sessions["b_0000.npz"], Session)


def test_load_all_missing_json(tmp_path: Path) -> None:
    """Test _load_all error reporting when scan json metadata is missing."""
    src_npz = Path("data/samples/ti64/bar_0000.npz")
    src_json = Path("data/samples/ti64/bar_0000.json")
    (tmp_path / "good.npz").write_bytes(src_npz.read_bytes())
    (tmp_path / "good.json").write_bytes(src_json.read_bytes())
    (tmp_path / "missing.npz").write_bytes(src_npz.read_bytes())
    paths = [tmp_path / "good.npz", tmp_path / "missing.npz"]
    sessions, errors = _load_all(paths, model=None)
    assert len(sessions) == 1
    assert "good.npz" in sessions
    assert len(errors) == 1
    assert "missing.npz" in errors[0]


def test_streamlit_app_multi_session_switch(running_app: Any, sample_session: Session) -> None:
    """Test sidebar scan switcher and summary table with multiple sessions."""
    running_app.session_state.sessions = {
        "scan_a.npz": sample_session,
        "scan_b.npz": sample_session,
    }
    running_app.run()
    assert not running_app.exception
    assert len(running_app.sidebar.selectbox) == 1
    running_app.sidebar.selectbox[0].set_value("scan_b.npz").run()
    assert not running_app.exception
    assert running_app.session_state.session is sample_session


def test_sample_mode_summary_table(running_app: Any, sample_session: Session) -> None:
    """Check summary table in sample mode with multiple sessions across pages."""
    running_app.session_state.sessions = {
        "bar_0000.npz": sample_session,
        "bar_0001.npz": sample_session,
    }
    sample_opt = "Sample: data/samples/ti64/bar_0000.npz"
    running_app.radio[0].set_value(sample_opt).run()
    assert not running_app.exception
    assert len(running_app.dataframe) == 1
    assert len(running_app.dataframe[0].value) == 2
    for page_name in running_app.sidebar.radio[0].options:
        running_app.sidebar.radio[0].set_value(page_name).run()
        assert not running_app.exception
