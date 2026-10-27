"""Tests for external data fetching and inventory reporting."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

_SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from fetch_external import (  # noqa: E402
    extract_strathclyde_download_url,
    extract_zenodo_file_url,
    filter_datasets_to_fetch,
    parse_size_gb,
    write_sha256sums,
    write_source_txt,
)
from make_data_inventory import (  # noqa: E402
    calculate_flaw_summary,
    generate_inventory_markdown,
)


@pytest.mark.parametrize(
    ("size_str", "expected_gb"),
    [
        ("18.52 GB", 18.52),
        ("1.0 GB", 1.0),
        ("512 MB", 0.5),
        ("260.05 MB", 260.05 / 1024.0),
        ("1048576 KB", 1.0),
        ("unknown (not stated on page)", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_size_gb(size_str: str | None, expected_gb: float | None) -> None:
    """Verify size string parsing into gigabytes."""
    parsed = parse_size_gb(size_str)
    if expected_gb is None:
        assert parsed is None
    else:
        assert parsed is not None
        assert abs(parsed - expected_gb) < 1e-4


def _create_fake_registry() -> list[dict]:
    """Create a test registry with accepted, rejected, and varying size entries."""
    return [
        {
            "name": "TUBERS-Ultrasonic-Crack-Dataset",
            "decision": "accept",
            "size_on_disk": "44.14 MB",
            "url": "https://zenodo.org/records/11239514",
            "n_distinct_physical_flaws": None,
        },
        {
            "name": "Strathclyde-FMC-Welded-316L",
            "decision": "accept",
            "size_on_disk": "260.05 MB",
            "url": "https://pureportal.strath.ac.uk/en/datasets/example",
            "n_distinct_physical_flaws": 1,
        },
        {
            "name": "Large-Dataset",
            "decision": "accept",
            "size_on_disk": "12.7 GB",
            "url": "https://example.com/large",
            "n_distinct_physical_flaws": None,
        },
        {
            "name": "No-Recipe-Dataset",
            "decision": "accept",
            "size_on_disk": "100 MB",
            "url": "https://example.com/norecipe",
            "n_distinct_physical_flaws": 2,
        },
        {
            "name": "Rejected-Dataset",
            "decision": "reject",
            "size_on_disk": "10 MB",
            "url": "https://example.com/rejected",
            "n_distinct_physical_flaws": None,
        },
    ]


def test_filter_datasets_max_gb_skipping() -> None:
    """Verify skipping datasets whose size exceeds max_gb threshold."""
    registry = _create_fake_registry()
    allowed_recipes = {
        "TUBERS-Ultrasonic-Crack-Dataset",
        "Strathclyde-FMC-Welded-316L",
        "Large-Dataset",
    }
    selected, skipped = filter_datasets_to_fetch(
        datasets=registry,
        available_recipes=allowed_recipes,
        max_gb=1.0,
        only=None,
    )
    selected_names = [d["name"] for d in selected]
    assert "TUBERS-Ultrasonic-Crack-Dataset" in selected_names
    assert "Strathclyde-FMC-Welded-316L" in selected_names
    assert "Large-Dataset" not in selected_names

    skip_reasons = {name: reason for name, reason in skipped}
    assert "Large-Dataset" in skip_reasons
    assert "exceeds" in skip_reasons["Large-Dataset"].lower()


def test_filter_datasets_recipes_and_only() -> None:
    """Verify recipe availability filtering and --only dataset targeting."""
    registry = _create_fake_registry()
    allowed_recipes = {"TUBERS-Ultrasonic-Crack-Dataset"}
    selected, skipped = filter_datasets_to_fetch(
        datasets=registry,
        available_recipes=allowed_recipes,
        max_gb=50.0,
        only=None,
    )
    selected_names = [d["name"] for d in selected]
    assert selected_names == ["TUBERS-Ultrasonic-Crack-Dataset"]
    skip_dict = dict(skipped)
    assert "No-Recipe-Dataset" in skip_dict

    selected_only, _ = filter_datasets_to_fetch(
        datasets=registry,
        available_recipes={"TUBERS-Ultrasonic-Crack-Dataset", "Strathclyde-FMC-Welded-316L"},
        max_gb=50.0,
        only="Strathclyde-FMC-Welded-316L",
    )
    assert [d["name"] for d in selected_only] == ["Strathclyde-FMC-Welded-316L"]


def test_write_sha256sums(tmp_path: Path) -> None:
    """Verify SHA256SUMS creation on tmp directory with nested files."""
    file1 = tmp_path / "sample.txt"
    file1.write_text("sample content 1", encoding="utf-8")
    subdir = tmp_path / "sub"
    subdir.mkdir()
    file2 = subdir / "sample2.bin"
    file2.write_bytes(b"\x00\x01\x02\x03")

    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "index").write_text("git internal", encoding="utf-8")

    out_file = write_sha256sums(tmp_path)
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")

    expected_hash1 = hashlib.sha256(b"sample content 1").hexdigest()
    expected_hash2 = hashlib.sha256(b"\x00\x01\x02\x03").hexdigest()
    assert f"{expected_hash1}  sample.txt" in content
    assert f"{expected_hash2}  sub/sample2.bin" in content
    assert ".git" not in content
    assert "SHA256SUMS" not in content


def test_write_source_txt(tmp_path: Path) -> None:
    """Verify SOURCE.txt writes provenance information."""
    entry = {
        "url": "https://example.org/dataset",
        "doi": "10.1234/test.doi",
        "licence": "MIT",
        "accessed": "2026-10-27",
    }
    src_path = write_source_txt(tmp_path, entry)
    assert src_path.exists()
    txt = src_path.read_text(encoding="utf-8")
    assert "url: https://example.org/dataset" in txt
    assert "doi: 10.1234/test.doi" in txt
    assert "licence: MIT" in txt
    assert "accessed: 2026-10-27" in txt


def test_extract_zenodo_file_url() -> None:
    """Verify Zenodo API payload parsing for specific file URL."""
    payload = {
        "files": [
            {
                "key": "Visual Data.zip",
                "links": {"self": "https://zenodo.org/api/records/1/files/Visual.zip/content"},
            },
            {
                "key": "Ultrasonic Data.zip",
                "links": {"self": "https://zenodo.org/api/records/1/files/Ultrasonic.zip/content"},
            },
        ]
    }
    url = extract_zenodo_file_url(payload, "Ultrasonic Data.zip")
    assert url == "https://zenodo.org/api/records/1/files/Ultrasonic.zip/content"
    assert extract_zenodo_file_url(payload, "Nonexistent.zip") is None


def test_extract_strathclyde_download_url() -> None:
    """Verify HTML parsing of Strathclyde Pure dataset page."""
    html = """
    <div class="document">
        <a href="/files/46166162/Lack_of_fusion_FMC_DORT_2016.mat" class="link">
            <span>Lack_of_fusion_FMC_DORT_2016</span>
        </a>
        <p class="property"><span>File</span>: <span>application/octet-stream, 260 MB</span></p>
    </div>
    """
    base_url = "https://pureportal.strath.ac.uk/en/datasets/test/"
    link = extract_strathclyde_download_url(html, base_url)
    expected = "https://pureportal.strath.ac.uk/files/46166162/Lack_of_fusion_FMC_DORT_2016.mat"
    assert link == expected

    no_link = extract_strathclyde_download_url("<html><body>No link here</body></html>", base_url)
    assert no_link is None


@pytest.mark.parametrize(
    ("datasets", "expected_flaws", "expected_unknown"),
    [
        (
            [
                {"decision": "accept", "n_distinct_physical_flaws": 3},
                {"decision": "accept", "n_distinct_physical_flaws": 1},
                {"decision": "accept", "n_distinct_physical_flaws": None},
                {"decision": "reject", "n_distinct_physical_flaws": 5},
            ],
            4,
            1,
        ),
        (
            [
                {"decision": "accept", "n_distinct_physical_flaws": None},
                {"decision": "accept", "n_distinct_physical_flaws": None},
            ],
            0,
            2,
        ),
    ],
)
def test_inventory_flaw_summary_counts(
    datasets: list[dict],
    expected_flaws: int,
    expected_unknown: int,
) -> None:
    """Verify flaw totals and unknown counts on accepted datasets."""
    summary = calculate_flaw_summary(datasets)
    assert summary["total_flaws"] == expected_flaws
    assert summary["unknown_count"] == expected_unknown


def test_inventory_markdown_generation() -> None:
    """Verify generated inventory markdown headers, rows, and summary text."""
    datasets = [
        {
            "name": "Dataset-A",
            "decision": "accept",
            "licence": "LGPL-3.0",
            "material": "316L",
            "technique": "PAUT",
            "n_distinct_physical_flaws": 3,
            "download_status": "downloaded",
        },
        {
            "name": "Dataset-B",
            "decision": "accept",
            "licence": "MIT",
            "material": "steel",
            "technique": "contact",
            "n_distinct_physical_flaws": None,
            "download_status": "not_downloaded",
        },
        {
            "name": "Dataset-C",
            "decision": "reject",
            "licence": "CC BY 4.0",
            "material": "concrete",
            "technique": "other",
            "n_distinct_physical_flaws": None,
            "download_status": "not_downloaded",
        },
    ]
    md = generate_inventory_markdown(datasets)
    lines = md.strip().splitlines()
    assert lines[0] == "# Real-data evidence limits"
    assert "| Dataset | Licence | Material | Technique |" in md
    assert "Dataset-A" in md
    assert "Dataset-B" in md
    assert "Dataset-C" not in md
    assert "3 distinct physical flaws" in md
    assert "1 accepted dataset has an unknown" in md
