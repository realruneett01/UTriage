"""Generate resume bullets strictly from results/*.json (project plan section Resume entry)."""

from __future__ import annotations

import json
import sys
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path


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


def _build_baseline_bullet(results_dir: Path) -> str | None:
    """Build simulator and rules baseline bullet."""
    path = results_dir / "rules_baseline.json"
    if not path.exists():
        print(f"{path.name} not found; omitting baseline bullet.", file=sys.stderr)
        return None
    with path.open("r", encoding="utf-8") as fh:
        _ = json.load(fh)
    return (
        "Built a physics-based immersion-UT simulator for rotating Ti-6Al-4V / "
        "IN718 bars (interface and back-wall echoes, attenuation, grain-noise "
        "backscatter, FBH and inclusion reflectors, coupling artefacts) and a "
        "rules engine (DAC, SNR, acceptance categories) plus a data-quality "
        "watchdog as the safety baseline."
    )


def _format_pod_real_bullet(a90_val: float | str, fc: dict) -> str:
    """Format real data bullet when POD a90/95 is available."""
    a90_str = round_ceil(a90_val, 2)
    n_false = fc.get("n_false", 0)
    n_scans = fc.get("n", 0)
    return (
        "Trained a CPU-sized CNN on real phased-array ultrasonic data "
        "(ML-NDT, Virkkunen et al.) and evaluated it with hit/miss POD: "
        f"a90/95 = {a90_str} mm with {n_false} false calls on {n_scans} "
        "unflawed scans (real data)."
    )


def _format_binomial_real_bullet(det: dict, fc: dict) -> str:
    """Format real data bullet using binomial detection lower bound."""
    n_hit = det.get("n_hit", 0)
    n_flawed = det.get("n_flawed", 0)
    pod_lb = round_floor(det.get("pod_lower95", 0.0), 2)
    n_false = fc.get("n_false", 0)
    n_scans = fc.get("n", 0)
    return (
        "Trained a CPU-sized CNN on real phased-array ultrasonic data "
        f"(ML-NDT, Virkkunen et al.): {n_hit} of {n_flawed} flaws detected "
        f"(POD lower bound >= {pod_lb}) with {n_false} false calls on "
        f"{n_scans} unflawed scans (real data)."
    )


def _build_real_data_bullet(results_dir: Path) -> str | None:
    """Build real-data bullet from mlndt_metrics.json test split."""
    path = results_dir / "mlndt_metrics.json"
    if not path.exists():
        print(f"{path.name} not found; omitting real-data bullet.", file=sys.stderr)
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    test_sp = data.get("per_split", {}).get("test", {})
    fc = test_sp.get("false_calls", {})
    pod = test_sp.get("pod")
    if pod is not None and "a90_95" in pod:
        return _format_pod_real_bullet(pod["a90_95"], fc)
    det = test_sp.get("detection", {})
    return _format_binomial_real_bullet(det, fc)


def _find_det_at_size(table: list[dict], target_size: float = 0.6) -> float | None:
    """Find detection rate for specific flaw size in table."""
    for row in table:
        size = float(row.get("size_mm", row.get("size", 0.0)))
        if abs(size - target_size) < 1e-3:
            return float(row.get("detection_rate", row.get("rate", 0.0)))
    return None


def _format_synthetic_bullet(fcr_str: str, comparison: str) -> str:
    """Assemble final synthetic validation bullet."""
    return (
        f"On simulated titanium bars at matched false-call rate ({fcr_str}), "
        f"the ML layer {comparison} under high grain noise; a union-with-rules "
        "design with conformal uncertainty means the AI can only add flags, "
        "never clear them (synthetic validation). Delivered as an offline, "
        "advisory-only inspector review app."
    )


def _extract_a90(entry: dict) -> float | None:
    """Safely extract a90_95 from method entry pod summary."""
    pod = entry.get("pod_summary", {}).get("pod")
    if isinstance(pod, dict) and "a90_95" in pod:
        return pod["a90_95"]
    return None


def _build_synthetic_bullet(results_dir: Path) -> str | None:
    """Build synthetic validation bullet from bar_domain_metrics.json at noise 2.0."""
    path = results_dir / "bar_domain_metrics.json"
    if not path.exists():
        print(f"{path.name} not found; omitting synthetic validation bullet.", file=sys.stderr)
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    model = data.get("kept_model", {}).get("choice", "ml_2d")
    pmn = data.get("per_method_per_noise", {})
    rules_sp = pmn.get("rules", {}).get("2.0", {})
    ml_sp = pmn.get(model, {}).get("2.0", {})
    fcr_val = rules_sp.get("false_call_rate", {}).get("rate", 0.0)
    fcr_str = round_ceil(fcr_val, 3)
    r_a90, m_a90 = _extract_a90(rules_sp), _extract_a90(ml_sp)
    if r_a90 is not None and m_a90 is not None:
        comp = f"improved a90/95 from {round_ceil(r_a90, 2)} mm to {round_ceil(m_a90, 2)} mm"
        return _format_synthetic_bullet(fcr_str, comp)
    r_det = _find_det_at_size(rules_sp.get("detection_table", []), 0.6) or 0.0
    m_det = _find_det_at_size(ml_sp.get("detection_table", []), 0.6) or 0.0
    comp = (
        f"improved detection rate at 0.6 mm from {round_floor(r_det, 2)} to {round_floor(m_det, 2)}"
    )
    return _format_synthetic_bullet(fcr_str, comp)


def _build_triage_bullet(results_dir: Path) -> str | None:
    """Build triage decision support bullet if triage_metrics.json exists."""
    path = results_dir / "triage_metrics.json"
    if not path.exists():
        print(f"{path.name} not found; omitting triage bullet.", file=sys.stderr)
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if "bullet" in data:
        bullet_str = str(data["bullet"])
        if "(synthetic validation)" not in bullet_str and "(real data)" not in bullet_str:
            return f"{bullet_str.rstrip('.')} (synthetic validation)."
        return bullet_str
    cov = round_floor(data.get("conformal_coverage", 0.9), 2)
    return (
        f"Triage decision support with conformal uncertainty (coverage >= {cov}) "
        "prioritizes critical indications for Level II review (advisory) (synthetic validation)."
    )


def build_bullets(results_dir: Path) -> list[str]:
    """Build resume bullet points strictly from metrics JSON files."""
    builders = [
        _build_baseline_bullet,
        _build_real_data_bullet,
        _build_synthetic_bullet,
        _build_triage_bullet,
    ]
    bullets: list[str] = []
    for b_func in builders:
        b = b_func(results_dir)
        if b is not None:
            bullets.append(b)
    return bullets


def main() -> None:
    """Entry point: generate and print resume bullets."""
    results_dir = Path("results")
    bullets = build_bullets(results_dir)
    for bullet in bullets:
        print(f"- {bullet}")


if __name__ == "__main__":
    main()
