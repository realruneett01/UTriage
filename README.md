# UTriage — AI-assisted ultrasonic indication triage for aerospace bar inspection

Advisory, offline decision support for immersion UT of titanium and nickel-alloy bars.
**UTriage never accepts or passes a part. Every decision belongs to the inspector.**

> **Status:** Independent unvalidated advisory prototype; no plant data and no plant validation.

## Problem Statement
In automated immersion ultrasonic testing (UT) of titanium and nickel-alloy bars, coarse grain structures generate acoustic background noise that causes false detections and borderline indications. UT Level II/III inspectors must manually evaluate each borderline indication, leading to review queues and avoidable re-scans when data quality degrades.

As an unvalidated advisory prototype, UTriage explores whether software could provide consistent indication measurement, rank indications with calibrated confidence to separate likely grain noise from likely flaws, and perform target real-time data-quality checks before a bar leaves the immersion tank without reducing detection reliability. The prototype is tested only on simulated bar scans and public weld PAUT datasets; there is no plant data and no plant validation.

## Safety Architecture
UTriage operates under a safety-first architecture:
- **Rules engine in charge:** The deterministic rules engine (gates, distance-amplitude correction / DAC, comparison against reference flat-bottom holes, and SNR calculation) evaluates scans against configured acceptance standards. Every indication flagged by the rules engine remains flagged.
- **Union fusion (AI only adds):** Machine learning models can raise candidate indications that conventional gates might miss, but ML can never suppress or remove a rule-generated flag. This invariant is enforced by construction and checked via property-based testing.
- **Never accepts or passes a part:** UTriage is strictly an advisory decision-support system. It never auto-dispositions material; every accept, reject, or re-scan disposition belongs solely to the inspector.
- **Offline and advisory:** UTriage runs fully air-gapped with no connection to UT machine controllers, PLCs, or plant networks.
- **Data-quality watchdog:** Target real-time pre-screening checks for loss of front-wall echo (bubbles, water path drift), loss of back-wall echo, acoustic noise above limits, and missing scan coverage.

### Data Flow
```
           +-------------------------+
           |  Ultrasonic Scan (.npz) |
           +------------+------------+
                        |
                        v
           +-------------------------+
           |   Data-Quality Watchdog |
           |  (hardware/scan checks) |
           +------------+------------+
                        |
            +-----------+-----------+
            |                       |
            v                       v
 +---------------------+  +-------------------+
 |    Rules Engine     |  |     ML Models     |
 |  (gates, DAC, SNR)  |  | (supervised / AE) |
 +----------+----------+  +---------+---------+
            |                       |
            +-----------+-----------+
                        |
                        v
           +-------------------------+
           |       Union Fusion      |
           |   (ML adds flags only)  |
           +------------+------------+
                        |
                        v
           +-------------------------+
           |    Ranked Indication    |
           |   List for Inspector    |
           +------------+------------+
                        |
                        v
           +-------------------------+
           |        Inspector        |
           |  (sole decision-maker)  |
           +-------------------------+
```

## Quick-Start

### 1. Installation
Install the development environment:
```bash
pip install -e .
```
For standalone workstation setup and file integrity checks, see [docs/install.md](docs/install.md).

### 2. Simulate Synthetic Scans
Generate synthetic immersion UT bar scans with ground-truth reflectors:
```bash
utriage simulate --material ti64 --bars 5 --out data/synthetic/ti64
```

### 3. Run Triage
Run batch triage on exported scan files to generate indication summaries and reports:
```bash
utriage triage data/synthetic/ti64/*.npz --out out/
```
Or evaluate an individual scan against the rules engine:
```bash
utriage evaluate data/synthetic/ti64/bar_0000.npz --acceptance configs/acceptance.yaml
```

### 4. Inspector Review Application
Launch the local Streamlit review application for interactive inspection:
```bash
streamlit run app/streamlit_app.py
```

## Project Layout
```
utriage/
├── configs/            # Probe specs, material properties, acceptance limits (placeholders)
├── data/               # Synthetic datasets and templates (data/external is gitignored)
├── src/utriage/        # Core package
│   ├── physics/        # Ultrasonic simulator and synthetic bar generation
│   ├── rules/          # Rules engine, gates, DAC, and data-quality watchdog
│   ├── ml/             # PyTorch models (1D A-scan, 2D patch CNN, autoencoder)
│   ├── pod/            # Probability of Detection (hit/miss and a-hat) evaluation
│   ├── triage/         # Union fusion and triage logic
│   ├── io/             # File adapters and scan loaders
│   ├── report/         # Inspection reporting and export
│   └── cli.py          # Command-line interface
├── app/                # Streamlit inspector review application
├── experiments/        # Experiment scripts for benchmarking and model training
├── results/            # Benchmark metrics, claims, and figures
├── tests/              # Test suite (unit, integration, property-based tests)
└── docs/               # Technical documentation, safety case, and ADRs
```

## Documentation
- [Technical Note](docs/technical-note.md): Physics formulation, simulation models, and algorithms.
- [Safety Case](docs/safety-case.md): Hazard analysis, safety arguments, and failure mitigations.
- [Architecture Decisions](docs/decisions/): Architectural records, including safety design ([0001](docs/decisions/0001-safety-design.md)) and prior art ([0002](docs/decisions/0002-prior-art.md)).
- [Installation Guide](docs/install.md): Workstation system requirements, offline setup, and checksum checks.
- [Claims and Evidence Limits](results/claims.md): Full audit of model claims, dataset scope, and empirical bounds.

## Prior Art
UTriage builds upon concepts from published ultrasonic evaluation literature and patents:
- **US 9,927,404 B2:** "Phased array billet data evaluation software" (United Technologies / RTX).
- **Virkkunen et al. (2019):** "Augmented Ultrasonic Data for Machine Learning", arXiv:1903.11399.

UTriage is an independent prototype and portfolio project, not a commercial implementation of the patented system. It differs by integrating a rules-in-charge safety architecture with union fusion (ML can only add flags), a target real-time data-quality watchdog, and probability of detection (POD) evaluation.

## Prototype status
- **Advisory decision support:** UTriage provides offline indication ranking and triage evidence. It does not connect to machine controllers and never accepts or rejects a part.
- **Not qualified:** The software is not qualified under any aerospace or plant inspection standard.
- **Evaluation base:** Tested on synthetic data from the physics simulator and a public weld dataset (Virkkunen et al., arXiv:1903.11399); no plant data and no plant validation.
- **Prerequisites for a plant pilot:**
  - Calibration-block scans from plant reference standards.
  - Sample bar scans in the target alloys and diameters.
  - UT system scan export format specifications.
  - Level III inspector review of rules, gates, and thresholds.

## Acceptance limits are placeholders
Every value in `configs/acceptance.yaml` is a **PLACEHOLDER — UNVERIFIED**. None is taken from
AMS-STD-2154, ASTM E2375 or any other standard; those standards are paywalled and are not
reproduced here. The rules engine runs with these placeholders only so that the software can be
built and tested. Results that depend on them must not be read as compliance with any standard.

## Synthetic data
Everything produced by the simulator is **Synthetic data** and is labelled so in every file,
plot and report.

## Results

<!-- RESULTS:BEGIN -->

> ### Evidence limits
> - Total distinct real physical flaws: 10 accepted, 9 used in evaluation (across accepted real-data sources).
> - Real data are weld PAUT datasets not bars.
> - Simulator constants are UNVERIFIED - Ruri to check.
> - UTriage is an advisory independent prototype; inspectors make all decisions.

> [!NOTE]
> **Superseded:** Earlier test-split scores measured interpolation within known cracks and are superseded by the leave-one-crack-out results.

### Table 1: Generalization matrix (Real data columns)

| Method | mlndt_loco_crack_1.6 | mlndt_loco_crack_4.0 | mlndt_loco_crack_8.6 | mlndt_validation | cross_ndt_ml_flaw |
|---| --- | --- | --- | --- | --- |
| rules | n/a (rules engine operates on bar immersion A-scans, not on weld PAUT B-scan images) | n/a (rules engine operates on bar immersion A-scans, not on weld PAUT B-scan images) | n/a (rules engine operates on bar immersion A-scans, not on weld PAUT B-scan images) | n/a (rules engine operates on bar immersion A-scans, not on weld PAUT B-scan images) | n/a (rules engine operates on bar immersion A-scans, not on weld PAUT B-scan images) |
| supervised | 0.14 [0.13, 0.16] 579/4014, FCR: 0.000 | 0.16 [0.15, 0.18] 671/4078, FCR: 0.000 | 1.00 [0.99, 1.00] 3925/3925, FCR: 0.000 | 1.00 [0.96, 1.00] 111/111, FCR: 0.000 | 1.00 [0.99, 1.00] 3442/3442, FCR: 1.000 |
| anomaly | 0.90 [0.89, 0.92] 3649/4014, FCR: 0.018 | 0.32 [0.30, 0.34] 1306/4078, FCR: 0.018 | 1.00 [0.99, 1.00] 3925/3925, FCR: 0.018 | n/a (Anomaly model uses validation split for threshold calibration only) | 1.00 [0.99, 1.00] 3442/3442, FCR: 1.000 |
| fused | n/a (per-image fused scores not computed) | n/a (per-image fused scores not computed) | n/a (per-image fused scores not computed) | n/a (per-image fused scores not computed) | n/a (per-image fused scores not computed) |

### Table 2: Generalization matrix (Synthetic data columns)

| Method | sim_in_distribution | sim_freq_10mhz | sim_material_third | sim_noise_above_range | sim_artefact_bubble | sim_artefact_lost_backwall | sim_artefact_waterpath_drift | sim_artefact_roughness | sim_artefact_missing_coverage |
|---| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rules | 0.77 [0.72, 0.82] 309/400, FCR: 0.029 | 0.95 [0.92, 0.97] 380/400, FCR: 0.748 | 0.82 [0.78, 0.87] 330/400, FCR: 0.040 | 0.89 [0.86, 0.93] 358/400, FCR: 0.343 | 0.86 [0.83, 0.90] 347/400, FCR: 0.196 | 0.82 [0.77, 0.86] 328/400, FCR: 0.168 | 0.81 [0.77, 0.86] 327/400, FCR: 0.104 | 0.81 [0.77, 0.86] 326/400, FCR: 0.114 | 0.86 [0.82, 0.90] 346/400, FCR: 0.152 |
| supervised | 0.88 [0.85, 0.92] 355/400, FCR: 0.134 | 0.88 [0.84, 0.92] 354/400, FCR: 0.600 | 0.92 [0.89, 0.95] 369/400, FCR: 0.044 | 0.92 [0.88, 0.95] 368/400, FCR: 0.407 | 0.90 [0.87, 0.94] 363/400, FCR: 0.178 | 0.89 [0.85, 0.92] 356/400, FCR: 0.184 | 0.89 [0.85, 0.93] 357/400, FCR: 0.204 | 0.91 [0.88, 0.94] 365/400, FCR: 0.148 | 0.89 [0.86, 0.93] 359/400, FCR: 0.188 |
| anomaly | 0.43 [0.34, 0.54] 49/112, FCR: 0.027 | n/a (Anomaly model only evaluated on synthetic in-distribution data) | n/a (Anomaly model only evaluated on synthetic in-distribution data) | n/a (Anomaly model only evaluated on synthetic in-distribution data) | n/a (Anomaly model only evaluated on synthetic in-distribution data) | n/a (Anomaly model only evaluated on synthetic in-distribution data) | n/a (Anomaly model only evaluated on synthetic in-distribution data) | n/a (Anomaly model only evaluated on synthetic in-distribution data) | n/a (Anomaly model only evaluated on synthetic in-distribution data) |
| fused | 0.89 [0.85, 0.92] 356/400, FCR: 0.146 | 0.96 [0.94, 0.99] 386/400, FCR: 0.802 | 0.92 [0.89, 0.95] 370/400, FCR: 0.067 | 0.95 [0.92, 0.98] 382/400, FCR: 0.529 | 0.94 [0.91, 0.97] 378/400, FCR: 0.274 | 0.90 [0.87, 0.94] 362/400, FCR: 0.248 | 0.89 [0.86, 0.93] 359/400, FCR: 0.238 | 0.92 [0.88, 0.95] 368/400, FCR: 0.176 | 0.92 [0.89, 0.95] 370/400, FCR: 0.248 |

### Table 3: Site adaptation

*Real ML-NDT weld images with synthetic perturbations, single seed.*

| Shift | Baseline False Calls | Baseline Detection | Adapted False Calls | Adapted Detection | n_calib_clean |
|---|---|---|---|---|---|
| identity | 0/43 (0.000) | 42/57 (0.737) | 0/43 (0.000) | 42/57 (0.737) | 824 |
| gain 0.5 | 0/43 (0.000) | 19/57 (0.333) | 0/43 (0.000) | 42/57 (0.737) | 824 |
| gain 2.0 | 43/43 (1.000) | 57/57 (1.000) | 0/43 (0.000) | 42/57 (0.737) | 824 |
| offset +0.5 | 4/43 (0.093) | 44/57 (0.772) | 0/43 (0.000) | 42/57 (0.737) | 824 |
| noise 0.2 | 1/43 (0.023) | 45/57 (0.789) | 0/43 (0.000) | 41/57 (0.719) | 824 |
| combined | 0/43 (0.000) | 19/57 (0.333) | 0/43 (0.000) | 39/57 (0.684) | 824 |

### Table 4: Joint training metrics

*sim images are Synthetic data*

| Held-out Crack | Arm | Detections | False Calls |
|---|---|---|---|
| crack_1.6 | A | 165/4014 | 0/778 |
| crack_1.6 | B | 192/4014 | 0/778 |
| crack_1.6 | C | 107/4014 | 0/778 |
| crack_1.6 | D | 0/4014 | 0/778 |
| crack_4.0 | A | 0/4078 | 0/778 |
| crack_4.0 | B | 0/4078 | 0/778 |
| crack_4.0 | C | 2306/4078 | 0/778 |
| crack_4.0 | D | 2229/4078 | 0/778 |
| crack_8.6 | A | 3925/3925 | 0/778 |
| crack_8.6 | B | 3925/3925 | 0/778 |
| crack_8.6 | C | 3925/3925 | 0/778 |
| crack_8.6 | D | 3925/3925 | 0/778 |

### Table 5: False-call analysis

- Clean images evaluated: 778
- Flagged by any detector: 14

| Agreement Count | Scans Flagged |
|---|---|
| 1 | 7 |
| 2 | 7 |
| 3 | 0 |
| 4 | 0 |

Claims and evidence: [results/claims.md](results/claims.md) | Matrix figure: [results/figures/generalization_matrix.png](results/figures/generalization_matrix.png)

<!-- RESULTS:END -->

## Licence
Proprietary software. Copyright (c) 2026 Ruri. All rights reserved.
Private repository. No permission to use, copy, modify, distribute or sublicense without prior written permission.

### Third-Party Data
Evaluation on public weld ultrasonic data utilizes the ML-NDT dataset (Virkkunen et al., arXiv:1903.11399), licensed under LGPL-3.0. This dataset is not redistributed with this repository and must be fetched independently using the acquisition script.
