# 0004. Operating point selection for advisory ML triage

- Status: Accepted
- Date: 2026-10-27

## Context

In ultrasonic inspection (UT) of aerospace bars and billets, coarse grain scattering generates acoustic noise that leads to false indications. An automated machine learning (ML) triage detector can assist inspectors by flagging candidate echoes and scoring likelihood. However, standard machine learning tuning criteria (such as raw classification accuracy or F1-score) are not acceptable for safety-critical non-destructive testing (NDT). 

In aerospace NDT, detection capability is governed by Probability of Detection (POD) and confidence bounds per MIL-HDBK-1823A. An operating threshold chosen arbitrarily or optimized purely for false-call suppression can degrade sensitivity at critical flaw dimensions. The operating point must guarantee statistical confidence at the governing reference sensitivity (0.8 mm FBH-equivalent, the tender sensitivity in project plan V1) before attempting to suppress false alarms.

## Decision

The ML operating point threshold is selected through a rigorous statistical search on validation data:

1. **Threshold sweep**: Candidate ML detection thresholds are scanned from highest (fewest false calls) downwards.
2. **Statistical lower bound evaluation**: For each candidate threshold, the one-sided 95% lower confidence bound of POD at the reference size (0.8 mm FBH-equivalent) is computed on validation data using:
   - The hit/miss logistic model with the delta method (`docs/study/03-pod.md`), or
   - A Clopper-Pearson exact binomial lower confidence bound over flaws at or above the reference size when the data are separable (e.g. perfect detection with zero misses).
3. **Threshold selection**: The highest candidate threshold whose one-sided 95% lower bound meets or exceeds the required target (default 0.90) is chosen.
4. **Implementation**: The selection logic is implemented in `src/utriage/triage/operating_point.py` through the function `choose_operating_point`.

Operational constraints and qualification boundaries:
- **Numerical values**: All empirical values and resulting thresholds come strictly from generated results files (`results/triage_metrics.json`). No numerical performance metrics are hard-coded or invented.
- **Synthetic validation**: Current validation is conducted strictly on synthetic simulated bar scan data.
- **Target threshold status**: The target lower bound value (default 0.90) is a software placeholder until formal acceptance criteria are established and signed off by a certified Level III examiner and the customer.

## Consequences

- **Detection lower bound on validation data only**: on the validation set used, the chosen threshold keeps the one-sided 95% lower bound of POD at the 0.8 mm FBH-equivalent reference size at or above the target. This is a property of that (synthetic) validation set, not a guarantee for real parts; real qualification needs a POD study on real flaws (see docs/safety-case.md).
- **Reproducibility**: Operating point selection is deterministic, data-driven, and recorded in `results/triage_metrics.json`.
- **Advisory limitation**: Because validation is performed on synthetic data, this operating point remains advisory and does not constitute plant qualification or production clearance.
