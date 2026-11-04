"""Tests for Site calibration Streamlit page and profile_summary helper."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_SRC = str(Path(__file__).resolve().parents[1] / "src")
_APP = str(Path(__file__).resolve().parents[1] / "app")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _APP not in sys.path:
    sys.path.insert(0, _APP)

import pytest  # noqa: E402
from streamlit_app import profile_summary  # noqa: E402

from utriage.ml.site_adapt import (  # noqa: E402
    SiteCalibration,
    SiteStats,
    load_site_profile,
    min_clean_scans,
    save_site_profile,
)


def _create_profile(
    path: Path,
    n_clean: int,
    fcr: float = 0.05,
    label: str = "Synthetic",
) -> Path:
    """Save a dummy site profile JSON file to path."""
    stats = SiteStats(mean=12.5, std=3.2, n_scans=n_clean, label=label)
    calib = SiteCalibration(
        threshold=0.875,
        false_call_rate=fcr,
        n_clean=n_clean,
        upper95_fcr=0.08,
    )
    return save_site_profile(path, stats, calib, label=label)


def test_profile_summary_saved_profile(tmp_path: Path) -> None:
    """profile_summary extracts correct metrics from a saved profile."""
    profile_path = tmp_path / "profile.json"
    _create_profile(profile_path, n_clean=10, fcr=0.05, label="Synthetic")
    profile = load_site_profile(profile_path)
    summary = profile_summary(profile)

    assert summary["label"] == "Synthetic"
    assert summary["n_clean"] == 10
    assert summary["false_call_rate"] == 0.05
    assert summary["threshold"] == 0.875
    assert summary["upper95_fcr"] == 0.08
    assert bool(summary["created"])
    assert summary["min_clean_scans"] == min_clean_scans(0.05)
    assert summary["warning"] is True
    assert summary["warning_flag"] is True


@pytest.mark.parametrize(
    ("n_clean", "expected_warn"),
    [
        (10, True),
        (60, False),
    ],
)
def test_profile_summary_warning_flag(
    tmp_path: Path,
    n_clean: int,
    expected_warn: bool,
) -> None:
    """warning flag is true when n_clean < min_clean_scans(fcr), false otherwise."""
    profile_path = tmp_path / f"profile_{n_clean}.json"
    _create_profile(profile_path, n_clean=n_clean, fcr=0.05, label="Real")
    profile = load_site_profile(profile_path)
    summary = profile_summary(profile)

    assert summary["warning"] is expected_warn
    assert summary["warning_flag"] is expected_warn


def test_profile_summary_from_dict() -> None:
    """profile_summary handles plain dictionary inputs."""
    raw_dict: dict[str, Any] = {
        "label": "Real",
        "created": "2026-11-04T00:00:00Z",
        "calib": {
            "threshold": 0.92,
            "false_call_rate": 0.05,
            "n_clean": 5,
            "upper95_fcr": 0.12,
        },
    }
    summary = profile_summary(raw_dict)
    assert summary["label"] == "Real"
    assert summary["n_clean"] == 5
    assert summary["threshold"] == 0.92
    assert summary["warning"] is True


def test_site_calibration_page_apptest() -> None:
    """AppTest executes Page 8 Site calibration and renders required notices."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app_target = (Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py").resolve()
    at = AppTest.from_file(str(app_target))
    at.run()
    assert not at.exception

    at.sidebar.radio[0].set_value("8 Site calibration").run()
    assert not at.exception

    notices = [info.value for info in at.info]
    expected_notice = (
        "Advisory only. A site profile changes ML thresholds only, never rules limits. "
        "Accept/reject stays with the certified inspector. No plant validation exists."
    )
    assert any(expected_notice in n for n in notices)

    code_blocks = [c.value for c in at.code]
    assert any("utriage site-calibrate" in c for c in code_blocks)
