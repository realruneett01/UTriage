"""Generate results/claims.json and results/claims.md (project plan Phase 8)."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from utriage.report.claims import build_claims, render_markdown  # noqa: E402


def generate_claims(results_dir: Path | str = "results") -> list[dict]:
    """Build claims and write claims.json and claims.md into results directory."""
    r_dir = Path(results_dir)
    claims = build_claims(r_dir)
    claims_data = [dataclasses.asdict(c) for c in claims]

    (r_dir / "claims.json").write_text(json.dumps(claims_data, indent=2), encoding="utf-8")
    (r_dir / "claims.md").write_text(render_markdown(claims), encoding="utf-8")

    return claims_data


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for claims artifact generator."""
    args = sys.argv[1:] if argv is None else argv
    results_dir = Path(args[0]) if args else Path("results")
    claims_data = generate_claims(results_dir)
    print(f"Generated {len(claims_data)} claims in {results_dir}")


if __name__ == "__main__":
    main()
