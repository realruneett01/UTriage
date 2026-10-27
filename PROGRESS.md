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
  SUPERSEDED (Phase 8): the Phase 3 test-split detection numbers came from a random batch split that shares all 3 physical cracks between train and test (`results/leakage_audit.json`); current real-data evidence is leave-one-crack-out (`results/loco_metrics.json`).

- **Phase 4 (tagged `phase-4`)**: `ml/bar_data.py` (DAC-normalised windows, virtual-flaw
  implants), `ml/bar_models.py` (WindowNet 1D / 3x3 patch), `ml/compare.py`, `pod/ahat.py`,
  `experiments/run_phase4.py` and `run_phase4_robust.py` -> `results/bar_domain_metrics.json`,
  `results/robustness_metrics.json` (noise 2.5x, IN718, 10 MHz, unseen artefacts, sim-to-real check).
  SUPERSEDED (Phase 8): kept model is ml_1d after the _decide_kept_model units fix (`results/bar_domain_metrics.json` kept_model); Phase 5 triage numbers re-run with ml_1d (`results/triage_metrics.json`).
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

## Phase 8 - generalization hardening (2026-10-27)
- Leakage audit (`results/leakage_audit.json`): old random batch split shares all 3 physical cracks between train and test, so earlier test scores are superseded.
- Leave-one-crack-out (LOCO) single-best per crack (`results/loco_metrics.json`, 3 folds; model_sha256 is recorded per fold):
  - crack_1.6: detection 579/4014 (rate 0.14, 95% CI [0.13, 0.16]), false calls 0/778 (rate 0.000 per scan).
  - crack_4.0: detection 671/4078 (rate 0.16, 95% CI [0.15, 0.18]), false calls 0/778 (rate 0.000 per scan).
  - crack_8.6: detection 3925/3925 (rate 1.00, 95% CI [1.00, 1.00]), false calls 0/778 (rate 0.000 per scan).
- Deep ensemble (`results/loco_ensemble_metrics.json`, 3 folds, seeds 0, 1, 2):
  - crack_1.6: detection 243/4014 (rate 0.061), false calls 0/778 (rate 0.0).
  - crack_4.0: detection 143/4078 (rate 0.035), false calls 0/778 (rate 0.0).
  - crack_8.6: detection 3925/3925 (rate 1.00), false calls 0/778 (rate 0.0).
  - Observation: the 3-seed mean-probability ensemble detected fewer held-out cracks than the best single seed.
- Run-to-run variance: an earlier training run of the same protocol gave crack_1.6 19/4014 and crack_4.0 19/4078 (git history of `results/loco_metrics.json`, commit message "LOCO results"); GPU training is not bit-reproducible, so held-out detection is unstable across training runs.
- Anomaly detector (`results/anomaly_metrics.json`, ConvAE2D trained on flaw-free images only):
  - crack_1.6: detection 3649/4014 (rate 0.91, 95% CI [0.90, 0.92]), false calls 0.018 per scan.
  - crack_4.0: detection 1306/4078 (rate 0.32, 95% CI [0.31, 0.33]), false calls 0.018 per scan.
  - crack_8.6: detection 3925/3925 (rate 1.00, 95% CI [1.00, 1.00]), false calls 0.018 per scan.
- Cross-dataset NDT_ML_Flaw test only (`results/cross_dataset_metrics.json`, `results/claims.md`): the model flagged (nearly) every image, giving detections 3442/3442 and false calls 3558/3558; no cross-dataset transfer under the documented domain mapping.
- Simulator domain randomization and held-out regions (`results/sim_generalization.json`, `results/claims.md`):
  - In-distribution: ML det 0.89 vs rules 0.77, false calls 0.13 vs 0.029 (Synthetic data, UNVERIFIED constants).
  - Held-out regions (ML vs rules):
    - sim_freq_10mhz: ML det 0.89 vs rules 0.95, false calls 0.60 vs 0.75.
    - sim_material_third: ML det 0.92 vs rules 0.82, false calls 0.044 vs 0.040.
    - sim_noise_above_range: ML det 0.92 vs rules 0.90, false calls 0.41 vs 0.34.
    - sim_artefact_bubble: ML det 0.91 vs rules 0.87, false calls 0.18 vs 0.20.
    - sim_artefact_lost_backwall: ML det 0.89 vs rules 0.82, false calls 0.18 vs 0.17.
    - sim_artefact_waterpath_drift: ML det 0.89 vs rules 0.82, false calls 0.20 vs 0.10.
    - sim_artefact_roughness: ML det 0.91 vs rules 0.81, false calls 0.15 vs 0.11.
    - sim_artefact_missing_coverage: ML det 0.90 vs rules 0.86, false calls 0.19 vs 0.15.
    - Exceeds 0.3 in: sim_freq_10mhz, sim_noise_above_range.
  - Union property holds in all 9 synthetic simulator columns with 0 violations (ML adds flags, never removes a rules flag).
- Fit diagnostics (`results/fit_diagnostics.json`): underfitting if train metric < 0.9; overfitting if gap > 0.1; LOCO folds crack_1.6 (train 1.0 vs held-out 0.144, gap 0.856) and crack_4.0 (train 1.0 vs held-out 0.165, gap 0.835) show severe overfitting; crack_8.6 ok (train 1.0 vs held-out 1.0, gap 0.0); synthetic supervised shows underfitting+overfitting across heldout regions.
- Failure analysis (`results/failure_modes.md`): per-fold top causes:
  - crack_1.6: 3435 misses (top cause: small_flaw with 1865 cases, 54.3%), 20 false calls (top cause: artefact_like with 20 cases).
  - crack_4.0: 3407 misses (top cause: unclassified with 1750 cases, 51.4%), 20 false calls (top cause: artefact_like with 20 cases).
  - crack_8.6: 0 misses (top cause: none with 0 cases, 0.0%), 20 false calls (top cause: artefact_like with 20 cases).
- Generalization matrix + claims (`results/generalization_matrix.json`, `results/claims.md`): verified claims and scope limits.
- Data registry + inventory (`data/registry.yaml`, `results/data_inventory.md`): Used in evaluation: 9 distinct physical flaws (ML-NDT: 3, NDT_ML_Flaw: 6); 10 distinct physical flaws listed across accepted public datasets.
- TFM imager (`src/utriage/imaging/tfm.py`): Total Focusing Method imager for Full Matrix Capture (FMC) array data.
- Hardware/latency (`results/hardware.json`): CPU latency on Intel(R) Core(TM) Ultra 9 275HX: median 14.2 ms, p95 21.0 ms (development machine CPU, not target i5).
- ADR 0009 (`docs/decisions/0009-generalization-protocol.md`): generalization protocol.

### Phase 8 known gaps
- Strathclyde FMC download blocked (HTTP 403) so the TFM imager is only tested on simulated FMC.
- Step 3.1 joint sim+real training not done (1D bar windows vs 2D weld images).
- Step 3.3 self-supervised pretraining not attempted.
- LOCO early stopping is batch-held-out not crack-held-out (addressed by crack-held-out mode).
- Cross-dataset mapping is an assumption.
- All simulator constants UNVERIFIED.
- Latency measured on the development machine CPU, not the target i5.
- Held-out detection unstable across training runs.
- results/failures adds about 68 MB of evidence files to the repo (fixed: reduced to 7.3 MB).

## Post-Phase 8 gap fixes (branch gaps-p8, 2026-10-27)
- main branch created at the phase-8 tip;
- crack-held-out LOCO results (copy from `results/loco_crackval_metrics.json`, `experiments/run_loco.py --val-mode crack`):
  - crack_1.6 (val crack_4.0): detection 4012/4014 (rate 0.9995), false calls 753/778 (rate 0.9679, upper95 0.9776), a50=0.795, a90=0.849, a90_95=0.953 mm.
  - crack_4.0 (val crack_8.6): detection 2449/4078 (rate 0.6005), false calls 0/778 (rate 0.0, upper95 0.0038), a50=2.951, a90=3.284, a90_95=3.306 mm.
  - crack_8.6 (val crack_1.6): detection 3925/3925 (rate 1.0), false calls 11/778 (rate 0.0141, upper95 0.0233), separable data with Clopper-Pearson lower bound 0.9992 (a90_95 null).
- failure analysis corrected (false calls only at threshold, 0 per fold; near-miss clean images listed separately; 7.3 MB instead of 68 MB - `results/failure_modes.md`);
- Strathclyde still blocked (DOI route login/challenge page, file route HTTP 403 - request access from the data owner);
- security fixes: safe output names and size cap in workflow, explicit weights_only model loading, zip-slip guard in the downloader, tamper-evident audit log anchor, honest downloader User-Agent;
- docs: README sections, pitch and demo script corrected;
- app page 7 Evidence and limits;
- cli.py health stays 9.65 (Typer options).

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
- Phase 8: answer the 10 Phase 8 questions in `docs/study/interview-qa.md`.

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
