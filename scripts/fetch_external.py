"""Fetch accepted external ultrasonic datasets into data/external/<name>/."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_UNIT_FACTORS = {
    "gb": 1.0,
    "mb": 1.0 / 1024.0,
    "kb": 1.0 / (1024.0 * 1024.0),
    "tb": 1024.0,
}

HONEST_USER_AGENT = "utriage-research-fetch (research project; contact: github.com/utriage)"
MAX_EXTRACT_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB; zip-bomb guard


@dataclass(frozen=True)
class FetchOptions:
    """Configuration options for external dataset fetching."""

    registry_path: Path
    output_dir: Path
    max_gb: float = 1.0
    timeout: int = 60
    only: str | None = None
    force: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class FetchResult:
    """Result status and metrics for a fetched dataset."""

    name: str
    status: str
    size_bytes: int = 0
    top_files: tuple[str, ...] = ()
    reason: str | None = None


def parse_size_gb(size_str: str | None) -> float | None:
    """Parse dataset size string (e.g. '260.05 MB', '18.52 GB') to gigabytes."""
    if not size_str:
        return None
    match = re.search(r"([\d.]+)\s*([a-z]+)", size_str.lower())
    if not match:
        return None
    val = float(match.group(1))
    factor = _UNIT_FACTORS.get(match.group(2))
    return val * factor if factor is not None else None


def compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Calculate SHA256 hex digest for a single file."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_sha256sums(directory: Path) -> Path:
    """Compute and write SHA256 of downloaded files to SHA256SUMS."""
    hashes: list[str] = []
    for file_path in sorted(directory.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(directory).as_posix()
        if any(part == ".git" for part in file_path.parts):
            continue
        if rel in {"SHA256SUMS", "SOURCE.txt"}:
            continue
        digest = compute_sha256(file_path)
        hashes.append(f"{digest}  {rel}")
    out_file = directory / "SHA256SUMS"
    out_file.write_text("\n".join(hashes) + "\n", encoding="utf-8")
    return out_file


def write_source_txt(
    directory: Path,
    entry: dict,
    extra_lines: list[str] | None = None,
) -> Path:
    """Write provenance information (url, doi, licence, access date) to SOURCE.txt."""
    lines = [
        f"url: {entry.get('url')}",
        f"doi: {entry.get('doi')}",
        f"licence: {entry.get('licence')}",
        f"accessed: {entry.get('accessed')}",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    out_file = directory / "SOURCE.txt"
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_file


def extract_zenodo_file_url(api_data: dict, filename: str = "Ultrasonic Data.zip") -> str | None:
    """Extract file download URL from Zenodo API record metadata."""
    for item in api_data.get("files", []):
        key = item.get("key") or item.get("filename")
        if key == filename:
            links = item.get("links", {})
            return links.get("self") or links.get("content")
    return None


def extract_strathclyde_download_url(html_content: str, base_url: str) -> str | None:
    """Find Lack_of_fusion_FMC_DORT_2016 .mat link on Pure portal HTML."""
    mat_match = re.search(
        r'href=[\'"]([^\'"]*Lack_of_fusion_FMC_DORT_2016[^\'"]*\.mat)[\'"]',
        html_content,
        re.IGNORECASE,
    )
    if mat_match:
        return urllib.parse.urljoin(base_url, mat_match.group(1))
    general_match = re.search(
        r'href=[\'"]([^\'"]*Lack_of_fusion_FMC_DORT_2016[^\'"]*)[\'"]',
        html_content,
        re.IGNORECASE,
    )
    if general_match:
        return urllib.parse.urljoin(base_url, general_match.group(1))
    return None


def download_file(url: str, dest_path: Path, timeout: int = 60) -> None:
    """Stream remote URL to local file path with timeout."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(  # noqa: S310
        url,
        headers={"User-Agent": HONEST_USER_AGENT},
    )
    with (
        urllib.request.urlopen(req, timeout=timeout) as resp,  # noqa: S310
        dest_path.open("wb") as out_file,
    ):
        while chunk := resp.read(65536):
            out_file.write(chunk)


def validate_member_path(name: str, target_dir: Path) -> None:
    """Ensure archive member path does not escape target directory."""
    if bool(re.match(r"^[a-zA-Z]:", name)):
        raise ValueError(f"Member '{name}' contains drive letter")
    if any((name.startswith(("/", "\\")), Path(name).is_absolute())):
        raise ValueError(f"Member '{name}' is an absolute path")
    resolved_target = target_dir.resolve()
    resolved_path = (target_dir / name).resolve()
    if not resolved_path.is_relative_to(resolved_target):
        raise ValueError(f"Member '{name}' resolves outside target directory")


def safe_extract(zf: zipfile.ZipFile, target_dir: Path) -> None:
    """Safely extract zip archive guarding against path traversal and zip bombs."""
    target_path = Path(target_dir)
    total_size = sum(info.file_size for info in zf.infolist())
    if total_size > MAX_EXTRACT_BYTES:
        raise ValueError(
            f"Archive uncompressed size ({total_size} bytes) "
            f"exceeds limit ({MAX_EXTRACT_BYTES} bytes)"
        )
    for info in zf.infolist():
        validate_member_path(info.filename, target_path)
    target_path.mkdir(parents=True, exist_ok=True)
    zf.extractall(target_path)


def fetch_tubers(entry: dict, target_dir: Path, timeout: int) -> None:
    """Download and unzip TUBERS Ultrasonic Crack Dataset from Zenodo API."""
    api_url = "https://zenodo.org/api/records/11239514"
    req = urllib.request.Request(  # noqa: S310
        api_url,
        headers={"User-Agent": HONEST_USER_AGENT},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        data = json.loads(resp.read().decode("utf-8"))
    file_url = extract_zenodo_file_url(data, "Ultrasonic Data.zip")
    if not file_url:
        raise ValueError("Could not find Ultrasonic Data.zip in Zenodo record files")
    zip_path = target_dir / "Ultrasonic Data.zip"
    download_file(file_url, zip_path, timeout=timeout)
    with zipfile.ZipFile(zip_path, "r") as zf:
        safe_extract(zf, target_dir)


def _is_challenge_or_html(content_type: str, initial_chunk: bytes) -> bool:
    """Check if response is HTML or a challenge/login page."""
    if "text/html" in content_type:
        return True
    lowered = initial_chunk.lower()
    markers = (b"<!doctype html", b"<html", b"<head", b"challenge", b"login")
    return any(marker in lowered for marker in markers)


def _save_stream_or_detect_page(
    resp: Any,
    dest_path: Path,
) -> tuple[bool, str]:
    """Inspect response and either save binary stream or detect challenge/HTML page."""
    status_code = getattr(resp, "status", 200)
    if status_code in (401, 403, 429):
        return False, f"HTTP {status_code}"
    content_type = str(resp.headers.get("Content-Type", "")).lower()
    first_chunk = resp.read(65536)
    if _is_challenge_or_html(content_type, first_chunk):
        return False, "login/challenge page (HTML content)"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_path.open("wb") as fh:
        fh.write(first_chunk)
        while chunk := resp.read(65536):
            fh.write(chunk)
    return True, "downloaded"


def _attempt_doi_route(
    doi: str,
    dest_path: Path,
    timeout: int,
) -> tuple[bool, str]:
    """Route A: Resolve DOI following normal redirects."""
    if not doi:
        return False, "missing DOI"
    url = f"https://doi.org/{doi}"
    req = urllib.request.Request(  # noqa: S310
        url,
        headers={"User-Agent": HONEST_USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return _save_stream_or_detect_page(resp, dest_path)
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code} {exc.reason}"
    except Exception as exc:  # reason: network or socket error during DOI resolution
        return False, str(exc)


def _fetch_page_html(url: str, timeout: int) -> tuple[str | None, str]:
    """Fetch HTML content from dataset page with honest user agent."""
    req = urllib.request.Request(  # noqa: S310
        url,
        headers={"User-Agent": HONEST_USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            if getattr(resp, "status", 200) in (401, 403, 429):
                return None, f"HTTP {resp.status}"
            html = resp.read().decode("utf-8", errors="ignore")
            return html, "ok"
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code} {exc.reason}"
    except Exception as exc:  # reason: network failure during page fetch
        return None, str(exc)


def _attempt_file_link_route(
    page_url: str,
    dest_path: Path,
    timeout: int,
) -> tuple[bool, str]:
    """Route B: Fetch file link published on dataset page."""
    html, status = _fetch_page_html(page_url, timeout)
    if html is None:
        return False, status
    file_url = extract_strathclyde_download_url(html, page_url)
    if not file_url:
        return False, "file link not found on dataset page"
    req = urllib.request.Request(  # noqa: S310
        file_url,
        headers={"User-Agent": HONEST_USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return _save_stream_or_detect_page(resp, dest_path)
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code} {exc.reason}"
    except Exception as exc:  # reason: network failure during file download
        return False, str(exc)


def _record_strathclyde_success(
    target_dir: Path,
    entry: dict,
    route_name: str,
) -> None:
    """Record successful download route and print confirmation."""
    extra = ["download_status: downloaded", f"route_worked: {route_name}"]
    write_source_txt(target_dir, entry, extra)
    print(f"[strathclyde] downloaded via {route_name}")


def _record_strathclyde_failure(
    target_dir: Path,
    entry: dict,
    status_a: str,
    status_b: str,
) -> None:
    """Record failing route statuses and print access notice."""
    extra = [
        "download_status: not_downloaded",
        f"route_a_doi: {status_a}",
        f"route_b_file_link: {status_b}",
        "action: request access from the data owner (Strathclyde Pure dataset page contact)",
    ]
    write_source_txt(target_dir, entry, extra)
    content = (target_dir / "SOURCE.txt").read_text(encoding="utf-8")
    print(content)
    print("request access from the data owner (Strathclyde Pure dataset page contact)")


def fetch_strathclyde(entry: dict, target_dir: Path, timeout: int) -> bool:
    """Fetch Strathclyde FMC dataset trying DOI and published file link routes."""
    page_url = entry.get("url", "")
    doi = entry.get("doi", "")
    dest_file = target_dir / "Lack_of_fusion_FMC_DORT_2016.mat"

    route_a_ok, status_a = _attempt_doi_route(doi, dest_file, timeout)
    if route_a_ok:
        _record_strathclyde_success(target_dir, entry, "route_a (doi_resolve)")
        return True

    route_b_ok, status_b = _attempt_file_link_route(page_url, dest_file, timeout)
    if route_b_ok:
        _record_strathclyde_success(target_dir, entry, "route_b (file_link)")
        return True

    _record_strathclyde_failure(target_dir, entry, status_a, status_b)
    return False


def fetch_ndt_ml_flaw(entry: dict, target_dir: Path, timeout: int) -> None:
    """Clone NDT_ML_Flaw shallow repository."""
    url = entry.get("url", "https://github.com/koomas/NDT_ML_Flaw")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    git_bin = shutil.which("git") or "git"
    subprocess.run(  # noqa: S603, S607
        [git_bin, "clone", "--depth", "1", url, str(target_dir)],
        check=True,
        timeout=timeout,
        capture_output=True,
        text=True,
    )


RECIPES = {
    "TUBERS-Ultrasonic-Crack-Dataset": fetch_tubers,
    "Strathclyde-FMC-Welded-316L": fetch_strathclyde,
    "NDT_ML_Flaw": fetch_ndt_ml_flaw,
}


def filter_datasets_to_fetch(
    datasets: list[dict],
    available_recipes: set[str],
    max_gb: float = 1.0,
    only: str | None = None,
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Filter registry datasets by decision, --only, size, and available recipe."""
    selected: list[dict] = []
    skipped: list[tuple[str, str]] = []
    for d in datasets:
        name = d.get("name", "")
        if d.get("decision") != "accept":
            continue
        if only is not None and name != only:
            continue
        size_gb = parse_size_gb(d.get("size_on_disk"))
        if size_gb is not None and size_gb > max_gb:
            skipped.append((name, f"Size {d.get('size_on_disk')} exceeds --max-gb {max_gb} GB"))
            continue
        if name not in available_recipes:
            skipped.append((name, f"No download recipe defined for {name}"))
            continue
        selected.append(d)
    return selected, skipped


def compute_dir_metrics(directory: Path) -> tuple[int, tuple[str, ...]]:
    """Calculate total size in bytes and list top-level files."""
    if not directory.exists():
        return 0, ()
    total_size = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
    top_items = tuple(sorted(p.name for p in directory.iterdir() if p.name != ".git"))
    return total_size, top_items


def process_dataset(entry: dict, options: FetchOptions) -> FetchResult:
    """Fetch single dataset, record provenance and compute file signatures."""
    name = entry["name"]
    target_dir = options.output_dir / name
    sums_file = target_dir / "SHA256SUMS"
    if sums_file.exists() and not options.force:
        size, files = compute_dir_metrics(target_dir)
        return FetchResult(name, "skipped", size, files, "already downloaded (SHA256SUMS exists)")
    if options.dry_run:
        return FetchResult(name, "dry_run", reason="would download via recipe")
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        recipe_fn = RECIPES[name]
        outcome = recipe_fn(entry, target_dir, options.timeout)
        if outcome is False:
            return FetchResult(
                name=name,
                status="not_downloaded",
                reason="request access from the data owner (Strathclyde Pure dataset page contact)",
            )
        write_source_txt(target_dir, entry)
        write_sha256sums(target_dir)
        size, files = compute_dir_metrics(target_dir)
        return FetchResult(name, "downloaded", size, files)
    except Exception as exc:  # reason: capture download failure per dataset without crashing run
        if target_dir.exists() and not any(target_dir.iterdir()):
            target_dir.rmdir()
        return FetchResult(name, "failed", reason=str(exc))


def run_fetch(options: FetchOptions) -> list[FetchResult]:
    """Execute dataset fetching across registry entries matching options."""
    with options.registry_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    datasets = data.get("datasets", [])
    selected, skipped = filter_datasets_to_fetch(
        datasets=datasets,
        available_recipes=set(RECIPES.keys()),
        max_gb=options.max_gb,
        only=options.only,
    )
    results: list[FetchResult] = []
    for name, reason in skipped:
        print(f"[skip] {name}: {reason}")
        results.append(FetchResult(name, "skipped", reason=reason))
    for entry in selected:
        print(f"[fetch] Processing {entry['name']}...")
        res = process_dataset(entry, options)
        print(f"[{res.status}] {res.name}: {res.reason or f'{res.size_bytes} bytes'}")
        results.append(res)
    return results


def main() -> None:
    """CLI entry point for external dataset fetching."""
    parser = argparse.ArgumentParser(description="Fetch accepted external UT datasets.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("data/registry.yaml"),
        help="Path to dataset registry YAML",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/external"),
        help="Target directory for external datasets",
    )
    parser.add_argument(
        "--max-gb",
        type=float,
        default=1.0,
        help="Skip datasets with size exceeding this threshold (default: 1.0 GB)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Network timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Fetch only the specified dataset name",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download datasets even if SHA256SUMS exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned actions without downloading",
    )
    args = parser.parse_args()
    opts = FetchOptions(
        registry_path=args.registry,
        output_dir=args.data_dir,
        max_gb=args.max_gb,
        timeout=args.timeout,
        only=args.only,
        force=args.force,
        dry_run=args.dry_run,
    )
    run_fetch(opts)


if __name__ == "__main__":
    main()
