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

### Table 1: Flaw detection and false calls on real weld data (Real data)

| Split | Flaws Detected (Hit / Total) | POD 95% LB | False Calls (95% UB) | a90/95 | Latency (Batch 1) |
|---|---|---|---|---|---|
| test | 1222 / 1222 | >= 0.99 | 0 / 778 (<= 0.004) | N/A (Data is separable: only hits or only misses present.) | 21.8 ms (CPU), 1.0 ms (GPU) |
| validation | 111 / 111 | >= 0.97 | 0 / 89 (<= 0.034) | N/A (Data is separable: only hits or only misses present.) | 21.8 ms (CPU), 1.0 ms (GPU) |

### Table 2: Rules vs ML at matched false-call rate (Synthetic data)

| Noise Scale | Method | FCR | 0.3 mm | 0.4 mm | 0.5 mm | 0.6 mm | 0.8 mm | 1.0 mm | 1.2 mm | a90/95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | rules | 0.000 | 0.00 | 0.00 | 0.00 | 0.37 | 1.00 | 1.00 | 1.00 | N/A (Data is separable: all hits are above all misses.) |
| 0.5 | ml_1d | 0.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | N/A (Data is separable: only hits or only misses present.) |
| 0.5 | ml_2d | 0.000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | N/A (Data is separable: only hits or only misses present.) |
| 1.0 | rules | 0.000 | 0.00 | 0.00 | 0.00 | 0.50 | 1.00 | 1.00 | 1.00 | N/A (Data is separable: all hits are above all misses.) |
| 1.0 | ml_1d | 0.000 | 0.25 | 0.12 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.62 mm |
| 1.0 | ml_2d | 0.000 | 0.50 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.59 mm |
| 1.5 | rules | 0.000 | 0.00 | 0.00 | 0.12 | 0.75 | 1.00 | 1.00 | 1.00 | 0.71 mm |
| 1.5 | ml_1d | 0.000 | 0.12 | 0.25 | 0.62 | 1.00 | 1.00 | 1.00 | 1.00 | 0.69 mm |
| 1.5 | ml_2d | 0.000 | 0.00 | 0.00 | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 | N/A (Data is separable: all hits are above all misses.) |
| 2.0 | rules | 0.018 | 0.00 | 0.25 | 0.12 | 0.75 | 1.00 | 1.00 | 1.00 | 0.84 mm |
| 2.0 | ml_1d | 0.018 | 0.00 | 0.00 | 0.00 | 0.87 | 1.00 | 1.00 | 1.00 | N/A (Data is separable: all hits are above all misses.) |
| 2.0 | ml_2d | 0.018 | 0.00 | 0.12 | 0.37 | 0.87 | 1.00 | 1.00 | 1.00 | 0.72 mm |

### Table 3: Triage and ranking performance (Synthetic data)

| Metric | Value | Reference / Target | Notes |
|---|---|---|---|
| Conformal Coverage | >= 0.91 | >= 0.90 (alpha = 0.1) | n = 336, uncertain fraction = 0.07 |
| Operating Point POD LB | >= 0.90 | >= 0.90 at 0.8 mm | Threshold = 0.999, FCR = 0.000 |

| Ranking Mode | Threshold | Flaws Found (Rules) | Flaws Found (Union) | Effort (Ranked) | Effort (Random) | ML False Flags / Bar |
|---|---|---|---|---|---|---|
| operating_point | 0.999 | 0.78 | 0.81 | 11.3 | 34.6 | 0.05 |
| matched_fcr | 0.837 | 0.78 | 0.95 | 32.5 | 103.0 | 135.50 |
| bar_budget | 4.514 | 0.78 | 0.78 | 12.0 | 38.4 | 0.00 |

### Table 4: Robustness and sim-to-real domain shift (Synthetic data)

| Evaluation | Model | Test Data | ROC AUC | Sample Size | Axis / Domain Note |
|---|---|---|---|---|---|
| ML-NDT on simulated B-scans | models/mlndt_best.pt | Synthetic data | 0.50 | 40 | Chance level on uncalibrated synthetic data |
| Bar 1D on ML-NDT validation | models/bar_1d.pt | Real data (ML-NDT) | 0.71 | 200 | Image axis meaning: UNVERIFIED - Ruri to check |

| Scenario | Material | Probe | Noise | Rules FCR | ML (2D) FCR | Rules Det (0.6 / 0.8 mm) | ML Det (0.6 / 0.8 mm) |
|---|---|---|---|---|---|---|---|
| noise_2.5x | ti64 | p5 | 2.5 | 0.157 | 0.157 | 0.87 / 1.00 | 1.00 / 1.00 |
| in718 | in718 | p5 | 1.0 | 0.000 | 0.000 | 1.00 / 1.00 | 1.00 / 1.00 |
| probe_p10 | ti64 | p10 | 1.0 | 0.907 | 0.907 | 1.00 / 1.00 | 1.00 / 0.87 |

<!-- RESULTS:END -->

## Licence
Proprietary software. Copyright (c) 2026 Ruri. All rights reserved.
Private repository. No permission to use, copy, modify, distribute or sublicense without prior written permission.
