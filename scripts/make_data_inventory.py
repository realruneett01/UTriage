"""Generate data inventory markdown from data/registry.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _is_used_dataset(dataset: dict[str, Any]) -> bool:
    """Check whether accepted dataset qualifies for used-in-evaluation count."""
    flaws = dataset.get("n_distinct_physical_flaws")
    relevance = dataset.get("relevance")
    rel_ok = relevance is not None and int(relevance) >= 2
    checks = [
        dataset.get("decision") == "accept",
        dataset.get("download_status") == "downloaded",
        flaws is not None,
        rel_ok,
    ]
    return all(checks)


def calculate_flaw_summary(datasets: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate total verified flaws and count of unknown accepted datasets."""
    total_flaws = 0
    known_count = 0
    unknown_count = 0
    used_datasets: list[tuple[str, int]] = []
    accepted = [d for d in datasets if d.get("decision") == "accept"]
    for entry in accepted:
        flaws = entry.get("n_distinct_physical_flaws")
        if flaws is not None:
            total_flaws += int(flaws)
            known_count += 1
        else:
            unknown_count += 1
        if _is_used_dataset(entry):
            name = str(entry.get("name", ""))
            used_datasets.append((name, int(flaws)))
    used_flaws = sum(cnt for _, cnt in used_datasets)
    return {
        "total_flaws": total_flaws,
        "known_count": known_count,
        "unknown_count": unknown_count,
        "accepted_count": len(accepted),
        "used_flaws": used_flaws,
        "used_datasets": used_datasets,
    }


def format_flaw_cell(flaws: int | None) -> str:
    """Format physical flaw count cell without guessing unknown values."""
    if flaws is None:
        return "unknown"
    return str(flaws)


def render_inventory_table(accepted: list[dict[str, Any]]) -> list[str]:
    """Render markdown table rows for accepted datasets."""
    lines = [
        "| Dataset | Licence | Material | Technique | Distinct Physical Flaws | Download Status |",
        "|---|---|---|---|---|---|",
    ]
    for d in accepted:
        name = d.get("name", "")
        licence = d.get("licence", "")
        material = d.get("material", "")
        technique = d.get("technique", "")
        flaws = format_flaw_cell(d.get("n_distinct_physical_flaws"))
        status = d.get("download_status", "not_downloaded")
        lines.append(f"| {name} | {licence} | {material} | {technique} | {flaws} | {status} |")
    return lines


def render_summary_text(summary: dict[str, Any]) -> str:
    """Render summary statement stating used evaluation flaws and listed flaw counts."""
    total = summary["total_flaws"]
    known = summary["known_count"]
    unknown = summary["unknown_count"]
    accepted = summary["accepted_count"]
    used_flaws = summary.get("used_flaws", 0)
    used_datasets = summary.get("used_datasets", [])
    breakdown = ", ".join(f"{name}: {cnt}" for name, cnt in used_datasets)
    flaw_unit = "flaw" if total == 1 else "flaws"
    ds_unit = "dataset" if unknown == 1 else "datasets"
    verb = "has" if unknown == 1 else "have"
    suffix = f" ({breakdown})" if breakdown else ""
    used_line = f"Used in evaluation: {used_flaws} distinct physical flaws{suffix}"
    listed_line = (
        f"Listed (accepted, known counts): across the {accepted} accepted datasets, "
        f"{known} datasets have verified "
        f"counts totaling {total} distinct physical {flaw_unit}. "
        f"The remaining {unknown} accepted {ds_unit} {verb} an unknown physical flaw count "
        "(unverified in public metadata; unknown values remain unknown and are never guessed)."
    )
    return f"{used_line}\n\n{listed_line}"


def generate_inventory_markdown(datasets: list[dict[str, Any]]) -> str:
    """Generate complete data inventory markdown document."""
    accepted = [d for d in datasets if d.get("decision") == "accept"]
    summary = calculate_flaw_summary(datasets)
    table_lines = render_inventory_table(accepted)
    summary_paragraph = render_summary_text(summary)
    return (
        "# Real-data evidence limits\n\n"
        + "\n".join(table_lines)
        + "\n\n"
        + summary_paragraph
        + "\n"
    )


def load_registry(path: Path) -> list[dict[str, Any]]:
    """Load dataset entries from YAML registry."""
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data.get("datasets", [])


def make_data_inventory(registry_path: Path, output_path: Path) -> None:
    """Generate and write inventory markdown file from registry."""
    datasets = load_registry(registry_path)
    content = generate_inventory_markdown(datasets)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def main() -> None:
    """CLI entry point for data inventory generation."""
    parser = argparse.ArgumentParser(description="Generate data inventory markdown.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("data/registry.yaml"),
        help="Path to dataset registry YAML",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/data_inventory.md"),
        help="Output path for markdown report",
    )
    args = parser.parse_args()
    make_data_inventory(args.registry, args.output)
    print(f"Data inventory generated at: {args.output}")


if __name__ == "__main__":
    main()
