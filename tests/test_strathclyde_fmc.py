"""Tests for Strathclyde FMC dataset loader and fetch routes."""

from __future__ import annotations

import sys
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import scipy.io

_SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if "utriage" in sys.modules:
    _pkg_dir = str(Path(_SRC) / "utriage")
    if _pkg_dir not in sys.modules["utriage"].__path__:
        sys.modules["utriage"].__path__.insert(0, _pkg_dir)
if "utriage.io" in sys.modules:
    _io_dir = str(Path(_SRC) / "utriage" / "io")
    if _io_dir not in sys.modules["utriage.io"].__path__:
        sys.modules["utriage.io"].__path__.insert(0, _io_dir)

from fetch_external import (  # noqa: E402
    HONEST_USER_AGENT,
    _attempt_doi_route,
    _attempt_file_link_route,
    _fetch_page_html,
    download_file,
    fetch_strathclyde,
    fetch_tubers,
)

from utriage.io.strathclyde_fmc import FMCRecord, load_fmc  # noqa: E402


def _make_dummy_fmc_mat(path: Path, include_metadata: bool = True) -> dict[str, Any]:
    """Create a minimal .mat file with synthetic FMC data."""
    fmc = np.arange(64, dtype=np.float32).reshape((2, 2, 16))
    data: dict[str, Any] = {"fmc": fmc}
    if include_metadata:
        data.update(
            {
                "element_positions": np.array([0.0, 1.0]),
                "sampling_frequency": 50.0,
                "wave_speed": 5900.0,
            }
        )
    scipy.io.savemat(path, data)
    return data


def _make_mock_response(
    content: bytes,
    status: int = 200,
    content_type: str = "application/octet-stream",
) -> MagicMock:
    """Create a mock HTTP response object."""
    mock = MagicMock()
    mock.__enter__.return_value = mock
    mock.__exit__.return_value = None
    mock.status = status
    mock.headers = {"Content-Type": content_type}
    mock.read.side_effect = [content, b""]
    return mock


def test_load_fmc_full_metadata(tmp_path: Path) -> None:
    """Verify loader reads FMC array and all metadata parameters."""
    mat_file = tmp_path / "sample_fmc.mat"
    _make_dummy_fmc_mat(mat_file, include_metadata=True)
    record = load_fmc(mat_file)
    assert isinstance(record, FMCRecord)
    assert record.fmc is not None and record.fmc.shape == (2, 2, 16)
    assert record.element_positions is not None and record.element_positions.size == 2
    assert record.sampling_frequency == 50.0
    assert record.wave_speed == 5900.0


def test_load_fmc_missing_parameters(tmp_path: Path) -> None:
    """Verify missing metadata fields are None and never invented."""
    mat_file = tmp_path / "minimal_fmc.mat"
    _make_dummy_fmc_mat(mat_file, include_metadata=False)
    record = load_fmc(mat_file)
    assert record.fmc is not None
    assert record.element_positions is None
    assert record.sampling_frequency is None
    assert record.wave_speed is None


def test_load_fmc_v73_raises_not_implemented(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify load_fmc propagates NotImplementedError for MATLAB v7.3 HDF5 files."""
    dummy = tmp_path / "dummy_v73.mat"
    dummy.write_bytes(b"MATLAB 7.3 MAT-file")

    def _mock_loadmat(_path: Any) -> Any:
        raise NotImplementedError("Please use HDF reader for matlab v7.3 files, e.g. h5py")

    monkeypatch.setattr(scipy.io, "loadmat", _mock_loadmat)
    with pytest.raises(NotImplementedError):
        load_fmc(dummy)


def test_doi_route_stops_on_403(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify DOI route stops without retrying when server returns 403."""
    dest = tmp_path / "out.mat"
    call_count = 0

    def _mock_urlopen(_req: Any, **_kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        raise urllib.error.HTTPError("https://doi.org/10", 403, "Forbidden", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", _mock_urlopen)
    ok, status = _attempt_doi_route("10.15129/dummy", dest, timeout=5)
    assert not ok
    assert "403" in status
    assert call_count == 1
    assert not dest.exists()


def test_doi_route_stops_on_challenge_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify DOI route stops without downloading when receiving HTML/challenge page."""
    dest = tmp_path / "out.mat"
    resp = _make_mock_response(
        b"<!DOCTYPE html><html>Login Challenge</html>",
        content_type="text/html",
    )
    monkeypatch.setattr(urllib.request, "urlopen", lambda _req, **_kw: resp)
    ok, status = _attempt_doi_route("10.15129/dummy", dest, timeout=5)
    assert not ok
    assert any(term in status.lower() for term in ("challenge", "html", "login"))
    assert not dest.exists()


def test_file_link_route_stops_on_403(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify published file link route stops without retrying when receiving 403."""
    dest = tmp_path / "out.mat"
    call_count = 0

    def _mock_urlopen(req: Any, **_kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "datasets" in url:
            html = '<a href="/files/1/Lack_of_fusion_FMC_DORT_2016.mat">download</a>'
            return _make_mock_response(html.encode("utf-8"), content_type="text/html")
        raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", _mock_urlopen)
    ok, status = _attempt_file_link_route("https://pureportal.strath.ac.uk/en/datasets/x", dest, 5)
    assert not ok
    assert "403" in status
    assert call_count == 2
    assert not dest.exists()


def test_fetch_strathclyde_all_routes_fail_recording(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify all routes failing writes SOURCE.txt and prints access notice."""
    entry = {
        "name": "Strathclyde-FMC-Welded-316L",
        "url": "https://pureportal.strath.ac.uk/en/datasets/x",
        "doi": "10.15129/086404bd-eb69-429b-978c-2c35cdbfcf87",
        "licence": "CC BY 4.0",
        "accessed": "2026-10-27",
    }
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        MagicMock(side_effect=urllib.error.HTTPError("http", 403, "Forbidden", {}, None)),
    )
    res = fetch_strathclyde(entry, tmp_path, timeout=5)
    assert res is False
    source_file = tmp_path / "SOURCE.txt"
    assert source_file.exists()
    content = source_file.read_text(encoding="utf-8")
    assert "download_status: not_downloaded" in content
    assert "HTTP 403" in content
    captured = capsys.readouterr().out
    assert "request access from the data owner" in captured


def test_fetch_strathclyde_successful_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify successful route downloads file and records worked route."""
    entry = {
        "name": "Strathclyde-FMC-Welded-316L",
        "url": "https://pureportal.strath.ac.uk/en/datasets/x",
        "doi": "10.15129/086404bd-eb69-429b-978c-2c35cdbfcf87",
        "licence": "CC BY 4.0",
        "accessed": "2026-10-27",
    }
    dummy_bin = b"MATLAB 5.0 MAT-file payload"
    resp = _make_mock_response(dummy_bin, content_type="application/octet-stream")
    monkeypatch.setattr(urllib.request, "urlopen", lambda _req, **_kw: resp)
    res = fetch_strathclyde(entry, tmp_path, timeout=5)
    assert res is True
    dest = tmp_path / "Lack_of_fusion_FMC_DORT_2016.mat"
    assert dest.exists()
    assert dest.read_bytes() == dummy_bin
    source_txt = (tmp_path / "SOURCE.txt").read_text(encoding="utf-8")
    assert "download_status: downloaded" in source_txt


def _run_and_capture_requests(
    monkeypatch: pytest.MonkeyPatch,
    action: Any,
) -> list[urllib.request.Request]:
    """Capture all urllib.request.Request objects passed to urlopen."""
    captured: list[urllib.request.Request] = []
    dummy_zip = (
        b"PK\x05\x06\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    )
    zenodo_json = (
        b'{"files": [{"key": "Ultrasonic Data.zip", "links": {"self": "https://z/data"}}]}'
    )
    html = b'<a href="https://example.com/Lack_of_fusion_FMC_DORT_2016.mat">download</a>'

    def _mock_urlopen(req: Any, **_kwargs: Any) -> MagicMock:
        captured.append(req)
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "zenodo.org/api" in url:
            return _make_mock_response(zenodo_json, content_type="application/json")
        if "page" in url or "datasets" in url:
            return _make_mock_response(html, content_type="text/html")
        return _make_mock_response(dummy_zip, content_type="application/octet-stream")

    monkeypatch.setattr(urllib.request, "urlopen", _mock_urlopen)
    action()
    return captured


def test_source_has_no_mozilla() -> None:
    """Verify scripts/fetch_external.py contains no Mozilla User-Agent strings."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "fetch_external.py"
    source = script_path.read_text(encoding="utf-8")
    assert "Mozilla" not in source


@pytest.mark.parametrize(
    "action_fn",
    [
        lambda p: download_file("https://example.com/test", p / "file.bin"),
        lambda p: fetch_tubers({}, p, 5),
        lambda p: _attempt_doi_route("10.1234/x", p / "doi.bin", 5),
        lambda p: _fetch_page_html("https://example.com/page", 5),
        lambda p: _attempt_file_link_route("https://example.com/datasets/x", p / "mat.bin", 5),
    ],
)
def test_requests_use_honest_user_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action_fn: Any,
) -> None:
    """Verify that every Request built by fetch_external uses HONEST_USER_AGENT."""
    captured = _run_and_capture_requests(monkeypatch, lambda: action_fn(tmp_path))
    assert len(captured) > 0
    for req in captured:
        assert req.get_header("User-agent") == HONEST_USER_AGENT
