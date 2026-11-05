# 0011. Site adaptation for zero-label site transfer

- Status: Proposed
- Date: 2026-11-04

## Context

In ultrasonic testing (UT) of aerospace bars and billets, using an advisory machine learning model across different inspection environments (immersion tanks, pulser-receivers, probes, alloy grades, and grain structures) introduces domain shift. Differences in electronic noise, acoustic attenuation, and material grain scattering alter feature distributions and degrade detector calibration.

The project owner has no plant data or target defect labels. Collecting and destructively confirming defect labels in practice is infeasible. To maintain reliable advisory performance when transitioning to a new environment, the model must adapt without requiring target defect labels.

To solve this, an unsupervised site adaptation module (`src/utriage/ml/site_adapt.py`), CLI command (`utriage site-calibrate`), and app page exist to adjust normalisation statistics and decision thresholds using exclusively unlabelled, inspector-confirmed clean scans.

## Decision

Adopt an unsupervised site adaptation protocol in `src/utriage/ml/site_adapt.py` governed by the following contract:

1. **Per-site z-score normalisation**:
   - `fit_site_stats`: calculates baseline mean and standard deviation from site calibration scans.
   - `apply_site_stats`: applies z-score standardisation to incoming scans to align input distributions.

2. **Adaptive Batch Normalisation (AdaBN)**:
   - `recalibrate_batchnorm`: re-estimates batch normalisation running mean and variance on inspector-confirmed clean scans from the target site. Model weights remain frozen.

3. **Site threshold calibration**:
   - `calibrate_site_threshold`: estimates an empirical quantile threshold corresponding to a target false-call rate, combined with a one-sided 95% Clopper-Pearson upper confidence bound.

4. **Sample size for clean calibration**:
   - `min_clean_scans`: enforces a zero-failure sample size calculation. 299 clean scans with zero false calls are needed to bound a 1% false-call rate target at 95% confidence.

5. **Site profile serialisation**:
   - `save_site_profile` and `load_site_profile`: serialise and load site adaptation parameters in JSON format. Every profile is explicitly tagged with metadata as `"Synthetic"` or `"Real"`.

## Alternatives Considered

- **Fine-tuning on plant labels**:
  Rejected because the owner has no plant data and collecting confirmed defect labels in practice is impossible without seeded defects or destructive validation.
- **Domain adversarial training (e.g. DANN)**:
  Rejected because adversarial methods require target domain samples during initial model training, introduce optimization instability, and are difficult to evaluate and audit in safety-critical NDT.
- **Do nothing (fixed thresholds)**:
  Rejected because acoustic variations across inspection lines cause distribution drift, leading to uncalibrated false-call rates or reduced defect sensitivity.

## Consequences

- **Unsupervised site transfer**: The advisory detector adapts to new acoustic environments using only unlabelled clean scans.
- **Auditability**: Normalisation and threshold updates are deterministic and separated from core model weights.
- **Traceability**: All site profiles are version-controlled, JSON-formatted, and explicitly identified as synthetic or real.
- **Operational requirement**: Establishing a site profile requires at least 299 confirmed clean scans from the target site for a 1% false-call rate target at 95% confidence.

## Safety Constraints

The site adaptation module must adhere to non-negotiable safety rules:
- **Rules engine stays in charge**: Conventional evaluation (gates, DAC, reference FBH, SNR) remains the baseline authority and is never modified by site adaptation.
- **ML only adds flags**: ML indications can only supplement rules flags in union; adapted ML can never suppress or dismiss a rules engine flag.
- **Threshold bounds**: A site profile can only adjust ML detection thresholds; it can never modify or relax rules limits.
- **Inspector-confirmed clean scans**: Calibration scans used for AdaBN or threshold estimation must be confirmed flaw-free by a certified human inspector.
- **No accept or pass decision**: Neither the site adaptation module nor adapted ML models can issue an accept or pass decision; every disposition belongs to the certified inspector.
- **Versioned and logged**: Every site profile is version-controlled, logged, and timestamped with calibration metadata.

## Validation Still Required

- **No plant validation exists**: UTriage has no plant validation, has not been tested on plant equipment, and has not been validated on MIDHANI production data or any live industrial line.
- **Empirical evaluation**: The site adaptation module, CLI command (`utriage site-calibrate`), app page, and experiment (`results/site_adaptation.json`) exist. In `experiments/run_site_adaptation.py` the evaluated model (`ConvAE2D`) has no BatchNorm layers, so `recalibrate_batchnorm` made no change (the adapted model sha256 in `results/site_adaptation.json` is identical across shifts; the experiment now records `adabn_applied=false`). The measured improvements come from per-site z-score normalisation and quantile threshold recalibration only. AdaBN is implemented and unit-tested but not evaluated in this experiment. Also `upper95_fcr` is computed on calibration clean scans; test-set false calls are reported separately. Status stays Proposed. Validation on real site changes is still required.
