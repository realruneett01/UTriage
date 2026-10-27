"""Generate results/claims.json and results/claims.md (project plan Phase 8)."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from utriage.report.claims import build_claims, render_markdown  # noqa: E402


def generate_claims(results_dir: Path | str = "results") -> list[dict]:
    """Build claims and write claims.json and claims.md into results directory."""
    r_dir = Path(results_dir)
    claims = build_claims(r_dir)
    claims_data = [dataclasses.asdict(c) for c in claims]

    json_path = r_dir / "claims.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(claims_data, fh, indent=2)

    md_content = render_markdown(claims)
    md_path = r_dir / "claims.md"
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write(md_content)

    return claims_data


def main() -> None:
    """CLI entrypoint for claims artifact generator."""
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    claims_data = generate_claims(results_dir)
    print(f"Generated {len(claims_data)} claims in {results_dir}")


if __name__ == "__main__":
    main()
