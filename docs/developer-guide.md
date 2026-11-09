# Developer Guide

Welcome to UTriage. This guide outlines setup instructions, coding conventions, testing protocols, and workflows for reproducing experiments.

## 1. Setup

UTriage requires Python version 3.11 or newer (as specified in `pyproject.toml`).

To install the editable package with machine learning and review application dependencies:

```bash
pip install -e ".[ml,app]"
```

To include development and testing tools defined in `pyproject.toml` (pytest, hypothesis, ruff, pytest-cov):

```bash
pip install -e ".[dev,ml,app]"
```

## 2. Repository Layout

See [architecture.md](architecture.md) for detailed architectural design and data flow diagrams.

- `configs/`: Probe specifications, material properties, and acceptance threshold configurations (placeholder, UNVERIFIED - Ruri to check).
- `data/`: Sample scans, synthetic datasets, and scan format export templates. External datasets reside in `data/external/` (gitignored).
- `src/utriage/`: Core package containing physics simulation, rules engine, data-quality watchdog, machine learning models, probability of detection (POD) analysis, triage fusion, file format adapters, and the command-line interface.
- `app/`: Streamlit review interface for scan inspection and report generation.
- `experiments/`: Experiment runner scripts for benchmarks, model training, and evaluations.
- `results/`: Metric JSON records, figures, failure mode catalogs, and claims documents.
- `tests/`: Test suite covering unit tests, integration workflows, and property-based checks.
- `docs/`: Technical notes, safety case documents, study notes, and architectural decision records.

## 3. Tests

Run the full test suite with pytest:

```bash
pytest
```

Run a single test file:

```bash
pytest tests/test_smoke.py
```

Run property-based tests using Hypothesis:

```bash
pytest tests/test_fusion.py
```

Run code style and format checks using Ruff:

```bash
ruff check .
ruff format --check .
```

To apply formatting automatically:

```bash
ruff format .
```

The ML-NDT dataset test (`test_real_mlndt_batches` in `tests/test_mlndt_dataset.py`) is skipped automatically when the local ML-NDT dataset directory is absent.

## 4. Data

### Synthetic Data Generation

Generate synthetic immersion ultrasonic bar scans with ground-truth reflectors using the command-line interface:

```bash
utriage simulate --material ti64 --bars 5 --out data/synthetic/ti64
```

All simulated signals, scans, and figures are labelled `Synthetic data`.

### ML-NDT Public Dataset

The public ML-NDT ultrasonic benchmark dataset is fetched using:

```bash
bash scripts/fetch_mlndt.sh
```

The ML-NDT dataset is licensed under LGPL-3.0. Downloaded files are placed in `data/external/ML-NDT` (gitignored) and are never committed to the repository or redistributed.

## 5. Reproducing Results

Each experiment script in `experiments/` executes a specific evaluation and writes output metrics to `results/`:

- `experiments/run_anomaly.py`: Unsupervised anomaly detection experiment runner. Writes `results/anomaly_metrics.json`.
- `experiments/run_cross_dataset.py`: Cross-dataset evaluation on real ultrasonic flaws (NDT_ML_Flaw). Writes `results/cross_dataset_metrics.json`.
- `experiments/run_failures.py`: LOCO failure mode analysis and failure case artifact generation. Writes `results/failures/summary.json` and `results/failure_modes.md`.
- `experiments/run_false_call_analysis.py`: LOCO false call analysis, cause classification, and trade-off curves (ADR 0010). Writes `results/false_call_analysis.json`.
- `experiments/run_fit_diagnostics.py`: Model fit diagnostics and learning curve analysis. Writes `results/fit_diagnostics.json`.
- `experiments/run_hardware.py`: Hardware profiling and CPU latency benchmarking. Writes `results/hardware.json`.
- `experiments/run_hybrid.py`: Hybrid detection experiment runner combining supervised LOCO and ConvAE2D (ADR 0010). Writes `results/hybrid_metrics.json`.
- `experiments/run_invariant_loco.py`: Two-stage training of an invariant crack representation on ML-NDT LOCO folds. Writes `results/invariant_loco_metrics.json`.
- `experiments/run_joint_train.py`: Joint simulated and real training with self-supervised pretraining. Writes `results/joint_train_metrics.json`.
- `experiments/run_leakage_audit.py`: Leakage audit for ML-NDT dataset splits. Writes `results/leakage_audit.json`.
- `experiments/run_lite_hybrid.py`: Lightweight hybrid detector training and evaluation on ML-NDT LOCO folds. Writes `results/lite_hybrid_metrics.json`.
- `experiments/run_loco.py`: Leave-one-crack-out (LOCO) training and evaluation experiment. Writes `results/loco_metrics.json`, `results/loco_crackval_metrics.json`, and `results/loco_ensemble_metrics.json`.
- `experiments/run_loco_robust.py`: Robust leave-one-crack-out (LOCO) training and evaluation experiment. Writes `results/loco_robust_metrics.json`.
- `experiments/run_phase2.py`: Baseline experiment: detection rate, false calls, and watchdog check. Writes `results/rules_baseline.json`.
- `experiments/run_phase3.py`: Real-data ML detector training, POD, and latency benchmarking. Writes `results/mlndt_metrics.json` and `results/mlndt_report.md`.
- `experiments/run_phase4.py`: Bar-domain ML vs rules engine under grain noise (Synthetic data). Writes `results/bar_domain_metrics.json`.
- `experiments/run_phase4_robust.py`: Robustness under domain shifts, unseen artefacts, and sim-to-real (Synthetic data). Writes `results/robustness_metrics.json`.
- `experiments/run_phase5.py`: Triage fusion, uncertainty, and ranking (Synthetic data). Writes `results/triage_metrics.json`.
- `experiments/run_phase8.py`: Pipeline orchestration and generalization evaluation. Writes `results/generalization_matrix.json`.
- `experiments/run_physics_model.py`: Interpretable physics-feature detector on ML-NDT dataset. Writes `results/physics_model_metrics.json`.
- `experiments/run_separability.py`: Leave-one-crack-out (LOCO) separability evaluation experiment. Writes `results/separability_metrics.json`.
- `experiments/run_shift_robustness.py`: Site shift robustness evaluation experiment (ADR 0015). Writes `results/shift_robustness_metrics.json`.
- `experiments/run_sim_generalization.py`: Held-out parameter-region generalization experiment. Writes `results/sim_generalization.json`.
- `experiments/run_site_adaptation.py`: Unsupervised site adaptation evaluation experiment runner (ADR 0011). Writes `results/site_adaptation.json`.
- `experiments/run_ti_zones.py`: Synthetic study of zone-wise calibration under titanium grain noise. Writes `results/ti_zone_metrics.json`.

### Regeneration of Summary Documents

Results summaries, tables, and claims are regenerated from `results/*.json` using dedicated scripts:

- `scripts/make_readme_tables.py`: Regenerates markdown results tables in `README.md` from `results/*.json`.
- `scripts/make_claims.py`: Regenerates `results/claims.json` and `results/claims.md` from `results/*.json`.
- `scripts/make_resume_bullets.py`: Regenerates summary bullets in `results/resume_bullets.md` from `results/*.json` and `results/claims.json`.

Rule: Never type a result by hand. All numerical results, detection rates, false call frequencies, and performance metrics must be generated by scripts directly from `results/*.json`. For full quantitative summaries, consult [results/claims.md](../results/claims.md) and the `README.md` Results section.

## 6. Pre-Registration

Experiments with a pre-registration Architectural Decision Record fix their hypotheses, candidate criteria, baseline models, comparison thresholds, and success rules before obtaining experimental results:

- ADR 0010: Hybrid detector evaluation (`docs/decisions/0010-hybrid-detector-preregistration.md`).
- ADR 0012: Robust LOCO training evaluation (`docs/decisions/0012-robust-loco-preregistration.md`).
- ADR 0013: LiteHybrid (H4) detector evaluation (`docs/decisions/0013-lite-hybrid-preregistration.md`).
- ADR 0014: Invariant representation learning evaluation (`docs/decisions/0014-invariant-representation-preregistration.md`).
- ADR 0015: Shift robustness evaluation (`docs/decisions/0015-shift-robustness-preregistration.md`).

Pre-registered decision rules are evaluated by `scripts/judge_preregistration.py`, which records outcomes in `results/preregistration_verdicts.json`. Once results are observed, the pre-registered success rules are never altered or retrofitted.

## 7. Coding Rules

- Type hints: All public functions, classes, and method signatures must include explicit type annotations.
- Docstrings: All public functions, classes, and modules must include descriptive docstrings detailing purpose, parameters, and return values.
- Small functions: Functions should remain short and focused (aiming for 30 lines or fewer, cyclomatic complexity at or below 6, nesting depth at or below 2, and 4 or fewer parameters grouped into frozen dataclasses where needed).
- Fixed seeds: Any randomized simulation, stochastic sampling, data partition, or model training must accept and set a fixed random seed.
- Allowed dependencies: Only use dependencies explicitly listed in `pyproject.toml` (`numpy`, `scipy`, `pandas`, `scikit-learn`, `pydantic`, `pyyaml`, `typer`, `matplotlib`, `torch`, `streamlit`, `pytest`, `hypothesis`, `ruff`, `pytest-cov`).
- New dependencies: Introducing an external dependency requires proposing and recording an ADR in `docs/decisions/`.
- Offline only: All modules and scripts must operate air-gapped without remote calls, external network sockets, or internet access.

## 8. Safety Rules a Change Must Keep

- ML only adds flags: The deterministic rules engine retains authority over indication detection. Machine learning models only provide advisory input and can only add candidate flags to the triage list. ML never suppresses, downgrades, or removes a flag raised by rules. This invariant is enforced by union fusion and checked by the property-based test `test_property_fusion_safety` in `tests/test_fusion.py` (which runs across 10,000 cases with Hypothesis).
- Never pass a part: UTriage is an offline advisory decision-support prototype. It never dispositions, approves, or passes a part. Every disposition decision (accept, reject, or re-scan) belongs solely to the human inspector.
- Label Synthetic data: All synthetic scans, signals, C-scan images, and evaluation plots generated by the simulator must be labelled `Synthetic data` across file headers, metadata, reports, and UI screens.

## 9. Writing an ADR and a Study Note

### Architectural Decision Records (ADRs)

- Location: `docs/decisions/`
- File naming: `NNNN-title.md` with a 4-digit zero-padded number and lowercase hyphenated title (for example, `docs/decisions/0001-safety-design.md`, `docs/decisions/0016-titanium-mode.md`).
- Contents: Title, Status (Proposed, Accepted, Rejected, Superseded), Date, Context (problem statement and constraints), Decision (the approach chosen), Rationale (why the choice was made over alternatives), and Consequences (trade-offs and impacts). Pre-registration ADRs must include hypotheses, baseline references, metric criteria, and locked success rules before running experiments.

### Study Notes

- Location: `docs/study/`
- File naming: `NN-title.md` with a 2-digit zero-padded number and lowercase hyphenated title (for example, `docs/study/00-ut-primer.md`, `docs/study/01-simulator-physics.md`, `docs/study/16-ti-grain-noise-and-zone-calibration.md`).
- Contents: Physics principles, mathematical formulas and derivations, ultrasonic domain context, exploratory notes, and reference equations before writing implementation code.
