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

import yaml

_UNIT_FACTORS = {
    "gb": 1.0,
    "mb": 1.0 / 1024.0,
    "kb": 1.0 / (1024.0 * 1024.0),
    "tb": 1024.0,
}


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


def write_source_txt(directory: Path, entry: dict) -> Path:
    """Write provenance information (url, doi, licence, access date) to SOURCE.txt."""
    lines = [
        f"url: {entry.get('url')}",
        f"doi: {entry.get('doi')}",
        f"licence: {entry.get('licence')}",
        f"accessed: {entry.get('accessed')}",
    ]
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
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) UTriage/1.0"},
    )
    with (
        urllib.request.urlopen(req, timeout=timeout) as resp,  # noqa: S310
        dest_path.open("wb") as out_file,
    ):
        while chunk := resp.read(65536):
            out_file.write(chunk)


def fetch_tubers(entry: dict, target_dir: Path, timeout: int) -> None:
    """Download and unzip TUBERS Ultrasonic Crack Dataset from Zenodo API."""
    api_url = "https://zenodo.org/api/records/11239514"
    req = urllib.request.Request(  # noqa: S310
        api_url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) UTriage/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        data = json.loads(resp.read().decode("utf-8"))
    file_url = extract_zenodo_file_url(data, "Ultrasonic Data.zip")
    if not file_url:
        raise ValueError("Could not find Ultrasonic Data.zip in Zenodo record files")
    zip_path = target_dir / "Ultrasonic Data.zip"
    download_file(file_url, zip_path, timeout=timeout)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


def fetch_strathclyde(entry: dict, target_dir: Path, timeout: int) -> None:
    """Parse Pure portal dataset page and download 260 MB lack-of-fusion .mat file."""
    page_url = entry.get("url")
    if not page_url:
        raise ValueError("Strathclyde dataset entry has no URL")
    req = urllib.request.Request(  # noqa: S310
        page_url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) UTriage/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        html = resp.read().decode("utf-8", errors="ignore")
    file_url = extract_strathclyde_download_url(html, page_url)
    if not file_url:
        raise ValueError("Could not find Lack_of_fusion_FMC_DORT_2016 link on Pure page")
    dest_file = target_dir / "Lack_of_fusion_FMC_DORT_2016.mat"
    download_file(file_url, dest_file, timeout=timeout)


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
        recipe_fn(entry, target_dir, options.timeout)
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
