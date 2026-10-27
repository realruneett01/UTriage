# PROGRESS.md

## Done
- **Phase 0** (tag `phase-0`): package skeleton, `pyproject.toml`, ruff/pytest config, `.gitignore`,
  ADR 0001 (safety design), ADR 0002 (prior art), `docs/study/00-ut-primer.md` (questions only),
  `scripts/fetch_mlndt.sh`, smoke tests, `python -m utriage --help`.
- **Phase 1** (tag `phase-1`): `physics/simulator.py` (A-scan/C-scan helical bar model, grain noise,
  FBH + inclusion reflectors, 5 artefact kinds), `physics/calibration.py` (E127-style block, DAC),
  `physics/dataset.py` (generator, npz + JSON truth), `physics/figures.py`, `utriage simulate` CLI,
  configs (`materials.yaml`, `probe.yaml`), `docs/study/01-simulator-physics.md`, ADR 0003,
  sample bar in `data/samples/ti64/`. Gate command runs: `python -m utriage simulate --material ti64 --bars 5`.

- **Phase 2** (tag `phase-2`): `rules/engine.py` (interface-synchronised gates, DAC %, local SNR,
  wrap-around clustering), `rules/acceptance.py` (categories), `rules/quality.py` (watchdog, 8 fault
  kinds), `rules/pipeline.py`, `utriage evaluate` CLI, `rules/scoring.py`, `experiments/run_phase2.py`
  -> `results/rules_baseline.json` + figures. Gate passed (from the JSON): detection at/above
  reference in low noise = True; all artefact kinds caught = True. Safety fix: data-quality faults
  never downgrade a rejectable/reportable indication.

- **Phase 3** (tag `phase-3`): `ml/datasets.py` (streams ML-NDT by file offset; one batch has 10
  images), `ml/models.py` (residual CNN, detection + size heads), `ml/train.py`, `ml/infer.py`,
  `pod/hitmiss.py` (logistic GLM, delta-method a90/95, Clopper-Pearson bounds),
  `experiments/run_phase3.py` (`--eval-only` reuses the trained model) -> `results/mlndt_metrics.json`,
  `results/mlndt_report.md`. Trained on GPU (ADR 0005). Every flawed image was detected, so the
  logistic POD is not estimable; a binomial POD lower bound is reported instead (see the JSON).

- **Phase 4 (tagged `phase-4`)**: `ml/bar_data.py` (DAC-normalised windows, virtual-flaw
  implants), `ml/bar_models.py` (WindowNet 1D / 3x3 patch), `ml/compare.py`, `pod/ahat.py`,
  `experiments/run_phase4.py` and `run_phase4_robust.py` -> `results/bar_domain_metrics.json`,
  `results/robustness_metrics.json` (noise 2.5x, IN718, 10 MHz, unseen artefacts, sim-to-real check).
- **Phase 5 (tagged `phase-5`)**: `triage/fusion.py` (union rule, 10,000-case property test),
  `triage/uncertainty.py` (ensemble, split conformal), `triage/operating_point.py`,
  `experiments/run_phase5.py` -> `results/triage_metrics.json`, `docs/safety-case.md`, ADR 0004.
  Limitation: ML scans only the inner flaw gate (a half-window from each gate edge); the
  near-surface and near-back-wall zones are covered by the rules engine alone.
- **Phase 6 (tagged `phase-6`)**: `io/adapters.py`, `io/validate.py`, `ml/bar_infer.py` (whole-bar
  scan, connected-component grouping, per-bar false-flag budget), `report/html.py`,
  `app/streamlit_app.py` + `app_logic.py`, hash-chained `audit.py`, `integrity.py` (config
  manifest, `UTRIAGE_LOCKED=1`), `workflow.py` + `utriage triage` CLI, `docs/scan_export_template.md`.
  Gate: `scripts/smoke_fresh_install.ps1` (fresh clone -> install -> app answers) passed:
  install 47 s, app start 1.9 s.
- **Phase 7**: generated README tables, `docs/technical-note.md` (from template, 0 missing values),
  resume-bullet script, pitch, demo script, interview Q&A, CI with coverage gate >= 80 % (ADR 0008),
  proprietary LICENSE (ADR 0006).

## Remaining (owner only)
- ADR 0007 (Nuitka offline build) is Proposed: approve or reject before a release build.
- Supply verified acceptance limits and material/probe constants (all configs are UNVERIFIED placeholders).
- Legal review: patent freedom-to-operate (US 9,927,404 B2) and ML-NDT LGPL-3.0 terms for derived weights.
- Retrain the product model on own/plant data; no plant validation or qualification exists.
Environment notes: venv `C:/Users/realr/.venvs/utriage` (Python 3.14, torch 2.14.1+cu130);
data junctions to `C:/Users/realr/utriage-data`.

## Gate items only Ruri can close
- Phase 0: answer `docs/study/00-ut-primer.md` yourself.
- Phase 1: explain every term of the A-scan equation on paper (`docs/study/01-simulator-physics.md`).

## Open questions (ask the MIDHANI contact)
- U1: Is UT evaluation or review time an actual queue in their flow? How long does UT and test clearance take per lot?
- U2: Which acceptance standard and class applies per product (AMS-STD-2154 / ASTM E2375 style)?
- U3: What data does their immersion UT system export (A-scan or C-scan files, format)? Can we get a sample?
- U4: How often is a bar re-scanned because of noise, a lost back-wall echo or a borderline call?

## Assumptions
- MIDHANI runs automated immersion UT of bars 8–300 mm diameter including titanium, 0.8 mm FBH sensitivity (from their 2021 tender corrigendum).
- Development machine differs from the original target: RTX 5070 Ti Laptop GPU (12 GB, CUDA), 32 GB RAM, Python 3.14
  (owner's instruction). Training uses the GPU; see ADR 0004 when Phase 3 lands. Inference stays CPU-capable.
- ML-NDT data lives outside OneDrive (`C:\Users\realr\utriage-data`), linked by junctions at
  `data/external` and `data/synthetic`, so the 2.7 GB is never synced.
- All material constants, simulator noise levels and acceptance limits are UNVERIFIED placeholders.
