# UTriage — AI-assisted ultrasonic indication triage for aerospace bar inspection

Advisory, offline decision support for immersion UT of titanium and nickel-alloy bars.
**UTriage never accepts or passes a part. Every decision belongs to the certified inspector.**

> Status: work in progress (see `PROGRESS.md`). The full README with results tables is
> generated in Phase 7.

## Product status
- **Advisory decision support:** UTriage provides offline indication ranking and triage evidence. It does not connect to machine controllers and never accepts or rejects a part.
- **Not qualified:** The software is not qualified under any aerospace or plant inspection standard.
- **Validation base:** Evaluated on synthetic data from the physics simulator and a public weld dataset (Virkkunen et al., arXiv:1903.11399).
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
> - Total distinct real physical flaws: 10 (across accepted real-data sources).
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
| rules | 0.77 [0.72, 0.82] 309/400, FCR: 0.030 | 0.95 [0.92, 0.97] 380/400, FCR: 0.748 | 0.82 [0.78, 0.87] 330/400, FCR: 0.040 | 0.89 [0.86, 0.93] 358/400, FCR: 0.343 | 0.86 [0.83, 0.90] 347/400, FCR: 0.196 | 0.82 [0.77, 0.86] 328/400, FCR: 0.168 | 0.81 [0.77, 0.86] 327/400, FCR: 0.104 | 0.81 [0.77, 0.86] 326/400, FCR: 0.114 | 0.86 [0.82, 0.90] 346/400, FCR: 0.152 |
| supervised | 0.88 [0.85, 0.92] 355/400, FCR: 0.135 | 0.88 [0.84, 0.92] 354/400, FCR: 0.600 | 0.92 [0.89, 0.95] 369/400, FCR: 0.045 | 0.92 [0.88, 0.95] 368/400, FCR: 0.408 | 0.90 [0.87, 0.94] 363/400, FCR: 0.178 | 0.89 [0.85, 0.92] 356/400, FCR: 0.184 | 0.89 [0.85, 0.93] 357/400, FCR: 0.204 | 0.91 [0.88, 0.94] 365/400, FCR: 0.148 | 0.89 [0.86, 0.93] 359/400, FCR: 0.188 |
| anomaly | 0.43 [0.34, 0.54] 49/112, FCR: 0.027 | n/a (Anomaly model only evaluated on synthetic in-distribution data) | n/a (Anomaly model only evaluated on synthetic in-distribution data) | n/a (Anomaly model only evaluated on synthetic in-distribution data) | n/a (Anomaly model only evaluated on synthetic in-distribution data) | n/a (Anomaly model only evaluated on synthetic in-distribution data) | n/a (Anomaly model only evaluated on synthetic in-distribution data) | n/a (Anomaly model only evaluated on synthetic in-distribution data) | n/a (Anomaly model only evaluated on synthetic in-distribution data) |
| fused | 0.89 [0.85, 0.92] 356/400, FCR: 0.147 | 0.96 [0.94, 0.99] 386/400, FCR: 0.803 | 0.92 [0.89, 0.95] 370/400, FCR: 0.068 | 0.95 [0.92, 0.98] 382/400, FCR: 0.529 | 0.94 [0.91, 0.97] 378/400, FCR: 0.274 | 0.90 [0.87, 0.94] 362/400, FCR: 0.248 | 0.89 [0.86, 0.93] 359/400, FCR: 0.238 | 0.92 [0.88, 0.95] 368/400, FCR: 0.176 | 0.92 [0.89, 0.95] 370/400, FCR: 0.248 |

Claims and evidence: [results/claims.md](results/claims.md) | Matrix figure: [results/figures/generalization_matrix.png](results/figures/generalization_matrix.png)

<!-- RESULTS:END -->

## Licence
Proprietary software. Copyright (c) 2026 Ruri. All rights reserved.
Private repository. No permission to use, copy, modify, distribute or sublicense without prior written permission.
