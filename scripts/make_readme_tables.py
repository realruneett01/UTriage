"""Render markdown results tables from results/*.json into README.md."""

from __future__ import annotations

import json
import sys
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path

BEGIN_MARKER = "<!-- RESULTS:BEGIN -->"
END_MARKER = "<!-- RESULTS:END -->"


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


def load_json(path: Path) -> dict | None:
    """Load JSON from path if file exists, else return None."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_mlndt_latency(data: dict) -> str:
    """Extract latency string for CPU and GPU inference."""
    lat = data.get("latency_ms", {})
    cpu = lat.get("cpu_batch_1")
    gpu = lat.get("gpu_batch_1")
    if cpu is None and gpu is None:
        return "N/A"
    parts = []
    if cpu is not None:
        parts.append(f"{round_ceil(cpu, 1)} ms (CPU)")
    if gpu is not None:
        parts.append(f"{round_ceil(gpu, 1)} ms (GPU)")
    return ", ".join(parts)


def _extract_mlndt_a90(split_data: dict) -> str:
    """Extract a90/95 if estimable, else reason."""
    pod = split_data.get("pod")
    if isinstance(pod, dict) and "a90_95" in pod:
        return f"{round_ceil(pod['a90_95'], 2)} mm"
    reason = split_data.get("pod_reason") or "Separable data"
    return f"N/A ({reason})"


def _format_mlndt_row(split_name: str, split_data: dict, lat_str: str) -> str:
    """Format single row for ML-NDT real data split."""
    det = split_data.get("detection", {})
    fc = split_data.get("false_calls", {})
    n_hit, n_flawed = det.get("n_hit", 0), det.get("n_flawed", 0)
    det_str = f"{n_hit} / {n_flawed}"
    pod_lb = round_floor(det.get("pod_lower95", 0.0), 2)
    pod_str = f">= {pod_lb}"
    n_false, n_total = fc.get("n_false", 0), fc.get("n", 0)
    ub_str = round_ceil(fc.get("upper95", 0.0), 3)
    fc_str = f"{n_false} / {n_total} (<= {ub_str})"
    a90_str = _extract_mlndt_a90(split_data)
    return f"| {split_name} | {det_str} | {pod_str} | {fc_str} | {a90_str} | {lat_str} |"


def render_table_1(results_dir: Path) -> str:
    """Render Table 1: Real data (ML-NDT, Virkkunen et al.)."""
    data = load_json(results_dir / "mlndt_metrics.json")
    header = "### Table 1: Flaw detection and false calls on real weld data (Real data)\n\n"
    if data is None:
        return header + "*Experiment has not been run (mlndt_metrics.json not found).*\n"
    rows = [
        (
            "| Split | Flaws Detected (Hit / Total) | POD 95% LB | "
            "False Calls (95% UB) | a90/95 | Latency (Batch 1) |"
        ),
        "|---|---|---|---|---|---|",
    ]
    lat_str = _extract_mlndt_latency(data)
    per_split = data.get("per_split", {})
    for split_name, split_data in per_split.items():
        rows.append(_format_mlndt_row(split_name, split_data, lat_str))
    return header + "\n".join(rows) + "\n"


def _extract_bar_a90(entry: dict) -> str:
    """Extract a90_95 or reason from bar domain entry."""
    pod_summary = entry.get("pod_summary", {})
    pod = pod_summary.get("pod")
    if isinstance(pod, dict) and "a90_95" in pod:
        return f"{round_ceil(pod['a90_95'], 2)} mm"
    reason = pod_summary.get("reason") or "Separable"
    return f"N/A ({reason})"


def _format_bar_row(noise: str, method: str, entry: dict, sizes: list[float]) -> str:
    """Format single row of synthetic detection table."""
    fcr_val = entry.get("false_call_rate", {}).get("rate", 0.0)
    fcr_str = round_ceil(fcr_val, 3)
    det_map = {
        round(float(r.get("size_mm", r.get("size", 0.0))), 2): float(
            r.get("detection_rate", r.get("rate", 0.0))
        )
        for r in entry.get("detection_table", [])
    }
    size_cells = [round_floor(det_map.get(round(s, 2), 0.0), 2) for s in sizes]
    a90_str = _extract_bar_a90(entry)
    cells = [noise, method, fcr_str, *size_cells, a90_str]
    return "| " + " | ".join(cells) + " |"


def render_table_2(results_dir: Path) -> str:
    """Render Table 2: Synthetic data - rules vs ML detection rates."""
    data = load_json(results_dir / "bar_domain_metrics.json")
    header = "### Table 2: Rules vs ML at matched false-call rate (Synthetic data)\n\n"
    if data is None:
        return header + "*Experiment has not been run (bar_domain_metrics.json not found).*\n"
    sizes = data.get("configs", {}).get("sizes_mm") or [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2]
    size_headers = " | ".join(f"{s:.1f} mm" for s in sizes)
    rows = [
        f"| Noise Scale | Method | FCR | {size_headers} | a90/95 |",
        "|---|---|---|" + "---|" * len(sizes) + "---|",
    ]
    pmn = data.get("per_method_per_noise", {})
    noise_scales = sorted(pmn.get("rules", {}).keys(), key=float)
    for noise in noise_scales:
        for method in ("rules", "ml_1d", "ml_2d"):
            if method in pmn and noise in pmn[method]:
                rows.append(_format_bar_row(noise, method, pmn[method][noise], sizes))
    return header + "\n".join(rows) + "\n"


def _format_ranking_row(mode: str, mode_data: dict) -> str:
    """Format single row of triage ranking table."""
    thresh = round_floor(mode_data.get("threshold", 0.0), 3)
    r_found = round_floor(mode_data.get("flaws_found_rules_only", 0.0), 2)
    u_found = round_floor(mode_data.get("flaws_found_union", 0.0), 2)
    eff_rank = round_ceil(mode_data.get("mean_effort_ranked", 0.0), 1)
    eff_rand = round_ceil(mode_data.get("mean_effort_random", 0.0), 1)
    ff_bar = round_ceil(mode_data.get("ml_false_calls_per_bar", 0.0), 2)
    return f"| {mode} | {thresh} | {r_found} | {u_found} | {eff_rank} | {eff_rand} | {ff_bar} |"


def _format_conformal_summary(data: dict) -> list[str]:
    """Format conformal coverage and operating point summary lines."""
    conf = data.get("conformal", {})
    cov = round_floor(conf.get("coverage", 0.0), 2)
    alpha = conf.get("alpha", 0.1)
    unc = round_ceil(conf.get("uncertain_fraction", 0.0), 2)
    n_conf = conf.get("n", 0)

    op = data.get("operating_point", {})
    op_pod = round_floor(op.get("pod_lower95_at_ref", 0.0), 2)
    op_ref = op.get("ref_size", 0.8)
    op_thresh = round_floor(op.get("threshold", 0.0), 3)
    op_fcr = round_ceil(op.get("false_call_rate", 0.0), 3)

    return [
        "| Metric | Value | Reference / Target | Notes |",
        "|---|---|---|---|",
        (
            f"| Conformal Coverage | >= {cov} | >= {1 - alpha:.2f} (alpha = {alpha}) | "
            f"n = {n_conf}, uncertain fraction = {unc} |"
        ),
        (
            f"| Operating Point POD LB | >= {op_pod} | >= 0.90 at {op_ref} mm | "
            f"Threshold = {op_thresh}, FCR = {op_fcr} |"
        ),
        "",
    ]


def render_table_3(results_dir: Path) -> str:
    """Render Table 3: Synthetic data - triage and ranking performance."""
    data = load_json(results_dir / "triage_metrics.json")
    header = "### Table 3: Triage and ranking performance (Synthetic data)\n\n"
    if data is None:
        return header + "*Experiment has not been run (triage_metrics.json not found).*\n"
    rows = _format_conformal_summary(data)
    rows.extend(
        [
            (
                "| Ranking Mode | Threshold | Flaws Found (Rules) | Flaws Found (Union) | "
                "Effort (Ranked) | Effort (Random) | ML False Flags / Bar |"
            ),
            "|---|---|---|---|---|---|---|",
        ]
    )
    ranking = data.get("ranking", {})
    for mode, mode_data in ranking.items():
        rows.append(_format_ranking_row(mode, mode_data))
    return header + "\n".join(rows) + "\n"


def _format_sim_to_real_rows(sim_to_real: dict) -> list[str]:
    """Format rows for sim-to-real evaluation table."""
    rows = [
        "| Evaluation | Model | Test Data | ROC AUC | Sample Size | Axis / Domain Note |",
        "|---|---|---|---|---|---|",
    ]
    bscan_eval = sim_to_real.get("mlndt_on_simulated_bscan", {})
    if bscan_eval:
        auc = round_floor(bscan_eval.get("roc_auc", 0.0), 2)
        n_samples = bscan_eval.get("n_flawed", 0) + bscan_eval.get("n_unflawed", 0)
        model_str = bscan_eval.get("model", "N/A")
        rows.append(
            f"| ML-NDT on simulated B-scans | {model_str} | "
            f"Synthetic data | {auc} | {n_samples} | "
            "Chance level on uncalibrated synthetic data |"
        )
    mlndt_eval = sim_to_real.get("bar_1d_on_mlndt_val", {})
    if mlndt_eval:
        auc = round_floor(mlndt_eval.get("roc_auc", 0.0), 2)
        axis_note = mlndt_eval.get("image_axis_meaning", "UNVERIFIED - Ruri to check")
        model_str = mlndt_eval.get("model", "N/A")
        n_img = mlndt_eval.get("n_images", 0)
        rows.append(
            f"| Bar 1D on ML-NDT validation | {model_str} | "
            f"Real data (ML-NDT) | {auc} | {n_img} | "
            f"Image axis meaning: {axis_note} |"
        )
    return rows


def _format_domain_shift_row(name: str, shift_data: dict) -> str:
    """Format single row for domain shift robustness."""
    spec = shift_data.get("spec", {})
    rules = shift_data.get("rules", {})
    ml = shift_data.get("ml_2d_matched", {})
    r_fcr = round_ceil(rules.get("false_call_rate", 0.0), 3)
    m_fcr = round_ceil(ml.get("false_call_rate", 0.0), 3)
    r_det = rules.get("detection_per_size", {})
    m_det = ml.get("detection_per_size", {})
    r_str = f"{round_floor(r_det.get('0.6', 0.0), 2)} / {round_floor(r_det.get('0.8', 0.0), 2)}"
    m_str = f"{round_floor(m_det.get('0.6', 0.0), 2)} / {round_floor(m_det.get('0.8', 0.0), 2)}"
    mat = spec.get("material", "ti64")
    prb = spec.get("probe", "p5")
    noise = spec.get("noise_scale", 1.0)
    return f"| {name} | {mat} | {prb} | {noise} | {r_fcr} | {m_fcr} | {r_str} | {m_str} |"


def render_table_4(results_dir: Path) -> str:
    """Render Table 4: Synthetic data - robustness and sim-to-real AUCs."""
    data = load_json(results_dir / "robustness_metrics.json")
    header = "### Table 4: Robustness and sim-to-real domain shift (Synthetic data)\n\n"
    if data is None:
        return header + "*Experiment has not been run (robustness_metrics.json not found).*\n"
    rows = _format_sim_to_real_rows(data.get("sim_to_real", {}))
    domain_shift = data.get("domain_shift", {})
    if domain_shift:
        rows.extend(
            [
                "",
                (
                    "| Scenario | Material | Probe | Noise | Rules FCR | ML (2D) FCR | "
                    "Rules Det (0.6 / 0.8 mm) | ML Det (0.6 / 0.8 mm) |"
                ),
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for name, s_data in domain_shift.items():
            rows.append(_format_domain_shift_row(name, s_data))
    return header + "\n".join(rows) + "\n"


def render_all_tables(results_dir: Path) -> str:
    """Render all 4 results markdown tables from given results directory."""
    t1 = render_table_1(results_dir)
    t2 = render_table_2(results_dir)
    t3 = render_table_3(results_dir)
    t4 = render_table_4(results_dir)
    return f"{t1}\n{t2}\n{t3}\n{t4}"


def update_readme_text(readme_text: str, tables_markdown: str) -> str:
    """Update markdown text between RESULTS markers, adding markers if missing."""
    clean_tables = tables_markdown.strip()
    block = f"{BEGIN_MARKER}\n\n{clean_tables}\n\n{END_MARKER}"

    if BEGIN_MARKER in readme_text and END_MARKER in readme_text:
        start = readme_text.index(BEGIN_MARKER)
        end = readme_text.index(END_MARKER) + len(END_MARKER)
        return readme_text[:start] + block + readme_text[end:]

    if "## Results" in readme_text:
        pos = readme_text.index("## Results") + len("## Results")
        return readme_text[:pos] + f"\n\n{block}\n" + readme_text[pos:]

    target = "## Licence"
    if target in readme_text:
        pos = readme_text.index(target)
        return readme_text[:pos] + f"## Results\n\n{block}\n\n" + readme_text[pos:]

    return readme_text.rstrip() + f"\n\n## Results\n\n{block}\n"


def update_readme_file(readme_path: Path, results_dir: Path) -> None:
    """Render tables from results and update README.md file in place."""
    text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    tables_md = render_all_tables(results_dir)
    new_text = update_readme_text(text, tables_md)
    readme_path.write_text(new_text, encoding="utf-8")


def main() -> None:
    """Command line entry point to update README tables from results."""
    results_dir = Path("results")
    readme_path = Path("README.md")
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        readme_path = Path(sys.argv[2])
    update_readme_file(readme_path, results_dir)
    print(f"Updated {readme_path} with results tables from {results_dir}.")


if __name__ == "__main__":
    main()
