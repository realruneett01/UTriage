"""Generate resume bullets strictly from results/claims.json (project plan section Resume entry)."""

from __future__ import annotations

import argparse
import json
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Any

from utriage.report.claims import check_forbidden


def round_ceil(val: float | int | str, decimals: int = 2) -> str:
    """Format number rounded up (ceiling) to avoid project-favour rounding."""
    step = Decimal("10") ** -decimals
    d = Decimal(str(val))
    return str(d.quantize(step, rounding=ROUND_CEILING))


def round_floor(val: float | int | str, decimals: int = 2) -> str:
    """Format number rounded down (floor) to avoid project-favour rounding."""
    step = Decimal("10") ** -decimals
    d = Decimal(str(val))
    return str(d.quantize(step, rounding=ROUND_FLOOR))


def _ensure_scope(text: str, scope: str) -> str:
    """Ensure bullet text ends with a scope tag in parentheses."""
    stripped = text.strip().rstrip(".")
    scope_tag = scope.strip("()")
    if stripped.endswith(f"({scope_tag})"):
        return f"{stripped}."
    return f"{stripped} ({scope_tag})."


def _build_framing_bullet() -> str:
    """Build independent prototype framing bullet."""
    text = (
        "Independent prototype: advisory, offline triage of ultrasonic indications "
        "for aerospace bar inspection; ML can only add flags to the rules engine"
    )
    return _ensure_scope(text, "independent prototype, advisory")


def _build_loco_bullet(claims: dict[str, dict[str, Any]]) -> str | None:
    """Build real-data leave-one-crack-out bullet from loco claims."""
    loco = [claims[k] for k in sorted(claims) if k.startswith("loco_crack_")]
    if not loco:
        return None
    frags = [c["text"].strip().rstrip(".") for c in loco if c.get("text")]
    if not frags:
        return None
    body = "Real-data leave-one-crack-out results across physical cracks: " + "; ".join(frags)
    scope = loco[0].get("scope_limit") or "public dataset, leave-one-crack-out"
    return _ensure_scope(body, scope)


def _build_anomaly_bullet(claims: dict[str, dict[str, Any]]) -> str | None:
    """Build anomaly detection bullet from anomaly claims."""
    anomaly = [claims[k] for k in sorted(claims) if k.startswith("anomaly_crack_")]
    if not anomaly and "anomaly" in claims:
        anomaly = [claims["anomaly"]]
    if not anomaly:
        return None
    frags = [c["text"].strip().rstrip(".") for c in anomaly if c.get("text")]
    if not frags:
        return None
    body = "Anomaly detection trained only on flaw-free scans: " + "; ".join(frags)
    scope = anomaly[0].get("scope_limit") or "public dataset, leave-one-crack-out"
    return _ensure_scope(body, scope)


def _build_cross_dataset_bullet(claims: dict[str, dict[str, Any]]) -> str | None:
    """Build cross-dataset bullet if cross_dataset claim exists."""
    c = claims.get("cross_dataset")
    if c is None or not c.get("text"):
        return None
    frag = c["text"].strip().rstrip(".")
    body = frag if "cross-dataset" in frag.lower() else f"Cross-dataset evaluation: {frag}"
    scope = c.get("scope_limit") or "cross-dataset"
    return _ensure_scope(body, scope)


def _build_sim_regions_bullet(claims: dict[str, dict[str, Any]]) -> str | None:
    """Build simulator held-out regions bullet from sim_regions claim."""
    c = claims.get("sim_regions")
    if c is None or not c.get("text"):
        return None
    frag = c["text"].strip().rstrip(".")
    has_sim_word = any(w in frag.lower() for w in ("simulator", "simulated"))
    body = (
        frag
        if has_sim_word
        else f"Simulator held-out parameter regions and false-call trade-off: {frag}"
    )
    scope = c.get("scope_limit") or "simulated"
    return _ensure_scope(body, scope)


def _resolve_engineering_scope(
    union_c: dict[str, Any] | None,
    latency_c: dict[str, Any] | None,
) -> str:
    """Resolve scope tag for engineering bullet from component claims."""
    if union_c and union_c.get("scope_limit"):
        return str(union_c["scope_limit"])
    if latency_c and latency_c.get("scope_limit"):
        return str(latency_c["scope_limit"])
    return "independent prototype, advisory"


def _build_engineering_bullet(claims: dict[str, dict[str, Any]]) -> str | None:
    """Build engineering bullet combining union property and latency claims."""
    union_c = claims.get("union_property")
    latency_c = claims.get("latency")
    if union_c is None and latency_c is None:
        return None
    frags = [
        c["text"].strip().rstrip(".")
        for c in (union_c, latency_c)
        if c is not None and c.get("text")
    ]
    if not frags:
        return None
    body = "; ".join(frags)
    scope = _resolve_engineering_scope(union_c, latency_c)
    return _ensure_scope(body, scope)


def _load_claims(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Load results/claims.json and index by claim id."""
    path = results_dir / "claims.json"
    if not path.exists():
        raise FileNotFoundError(f"Claims file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of claims in {path}, got {type(data).__name__}")
    return {c["id"]: c for c in data if isinstance(c, dict) and "id" in c}


def build_bullets(results_dir: Path) -> list[str]:
    """Build resume bullet points strictly from results/claims.json."""
    claims = _load_claims(results_dir)
    builders = [
        _build_loco_bullet,
        _build_anomaly_bullet,
        _build_cross_dataset_bullet,
        _build_sim_regions_bullet,
        _build_engineering_bullet,
    ]
    bullets = [_build_framing_bullet()]
    for builder in builders:
        b = builder(claims)
        if b is not None:
            bullets.append(b)
    for bullet in bullets:
        if found := check_forbidden(bullet):
            raise ValueError(f"Forbidden phrase(s) {found} found in bullet: {bullet!r}")
    return bullets


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: generate resume bullets strictly from claims.json."""
    parser = argparse.ArgumentParser(description="Generate resume bullets from claims.json.")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help="Path to results directory containing claims.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file to write bullets",
    )
    args = parser.parse_args(argv)
    bullets = build_bullets(args.results)
    for bullet in bullets:
        print(f"- {bullet}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(f"- {b}" for b in bullets) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
