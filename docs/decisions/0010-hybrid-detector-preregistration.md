# 0010. Supervised + anomaly hybrid detector - pre-registered evaluation

- Status: Accepted
- Date: 2026-10-27

## Context

Leave-one-crack-out (LOCO) evaluations and model diagnostics show that the supervised CNN memorises known cracks and misses most unseen ones (`results/loco_metrics.json`, `results/fit_diagnostics.json`). Conversely, the anomaly autoencoder trained exclusively on flaw-free images exhibits varied behaviour across individual physical cracks (`results/anomaly_metrics.json`).

With only 3 physical cracks, exploring and selecting among many hybrid architectures or post-hoc combination rules based on test performance would cause data leakage. To ensure empirical validity, candidate architectures, fusion rules, decision thresholds, and evaluation protocols must be strictly pre-registered prior to running experiments and inspecting test outcomes.

## Decision

To evaluate hybrid detection without data leakage, exactly two candidate architectures are fixed before computing results:

- **Candidate H1 (OR-union)**:
  - An indication is flagged if the supervised probability >= 0.5 (as in `experiments/run_loco.py`) OR the anomaly score >= the anomaly threshold already fitted on clean calibration images at target false-call rate 0.01 (`results/anomaly_metrics.json`, `real.threshold`).
  - No threshold tuning is performed on test data.

- **Candidate H2 (Score fusion)**:
  - A fused score is calculated as `s = mean(p_supervised, p_anomaly)`, where `p_anomaly` is the empirical cumulative distribution function (CDF) of the anomaly score evaluated over the clean calibration images (official validation unflawed images).
  - The decision threshold is the `(1 - 0.01)` quantile of `s` evaluated over those same clean calibration images.
  - The threshold is determined strictly on clean calibration data and is never tuned on test data.

### Evaluation Protocol

- **Evaluation scope**:
  - Evaluated on every LOCO fold in batch mode using the existing fold test sets.
  - Evaluated for each of the 3 saved seeds per fold (`models/loco_<crack>_seed<k>.pt`) and for the single-best model.
- **Reported metrics**:
  - Detections `k/n` with Clopper-Pearson 95% confidence intervals per seed.
  - False calls `n_false/n_clean` per seed.
  - Summary statistics reporting the mean, min, and max across seeds.
  - Direct comparison against supervised-only and anomaly-only baselines on the exact same images.
- **Reporting commitment**:
  - Both candidates H1 and H2 are reported whatever the outcome.
  - No further candidates may be added after seeing results without a new ADR.

## Consequences

- **Evidence base remains small**: The hybrid detector still rests on 3 physical cracks from a single weld dataset; performance metrics do not imply generalisation beyond this domain.
- **Safety and false calls**: The OR-union satisfies the safety property (ADR 0001) that advisory models can only add candidate flags, but combining two detectors inherently raises false calls.
- **Output artifacts**: Evaluation results will be recorded in `results/hybrid_metrics.json` and fed into the claims table (`results/claims.json`).
