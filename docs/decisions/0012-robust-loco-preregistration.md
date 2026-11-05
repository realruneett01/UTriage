# 0012. Robust training leave-one-crack-out (LOCO) pre-registration

- Status: Proposed
- Date: 2026-11-05

## Context

In ultrasonic testing (UT), supervised deep learning models trained on small datasets tend to memorise training flaw signatures and perform inconsistently on unseen flaws (`results/loco_metrics.json`, `results/fit_diagnostics.json`).
Earlier explorations with crack-held-out early stopping (`results/loco_crackval_metrics.json`) resulted in degenerate operating points where high detection came at the expense of unacceptable false call rates exceeding 96%.
Moreover, evaluating multiple data augmentation policies, training configurations, and seed runs post-hoc risks metric shopping and data leakage.

To test whether physics-informed data augmentations (gain, offset, noise, time shift, scan flip) and seed ensembling improve model ranking and detection without compromising false call control, the evaluation protocol, baseline comparator, decision thresholds, and explicit success criteria must be pre-registered in this ADR before executing `experiments/run_loco_robust.py`.

No results currently exist for this experiment. This document records the experimental contract in advance.

## Decision

Pre-register the following evaluation protocol, metrics, and success criteria for `experiments/run_loco_robust.py`:

### 1. Cross-Crack Evaluation Protocol
- **Folds**: Exactly 3 leave-one-crack-out (LOCO) folds corresponding to the 3 physical cracks in ML-NDT (crack 1.6 mm, crack 3.1 mm, crack 4.0 mm).
- **Holdout discipline**: For each fold, models train only on the other two cracks. Early stopping and hyperparameter tuning use batch-held-out partitions from those training cracks (batchval protocol).
- **Zero test leakage**: The held-out crack is strictly unseen during training and tuning. It is never used for model selection, checkpoint selection, or early stopping.
- **Random seeds**: 5 fixed random seeds (seeds 0, 1, 2, 3, 4) per fold.
- **Ensemble aggregation**: Model predictions across all 5 seeds are averaged to form an ensemble prediction score for each test scan.
- **Test set parity**: The baseline run and the robust run evaluate on the exact same test images for each fold.

### 2. Metrics

- **Primary Metric**:
  - Per-fold Area Under the Receiver Operating Characteristic curve (AUROC) on the held-out crack test partition.
- **Secondary Metrics**:
  - Detection rate (hit rate) at clean-calibrated False Call Rate (FCR) target of 0.01.
  - Detection rate (hit rate) at clean-calibrated False Call Rate (FCR) target of 0.05.
- **Threshold Calibration**:
  - Decision thresholds for FCR 0.01 and FCR 0.05 are calculated solely on the clean calibration split as empirical quantiles at $(1 - \text{FCR})$. Flaw scans are never used to set thresholds.
  - Empirical false call rates are measured on the disjoint clean evaluation split.

### 3. Success Rule

Robust training is called **'better'** on a fold only if:
1. Its held-out crack AUROC exceeds the baseline from `results/separability_metrics.json` (batchval protocol) by at least **0.05** ($\Delta\text{AUROC} \ge +0.05$); **AND**
2. Its detection rate at clean-calibrated FCR 0.05 is not lower than the baseline detection rate at FCR 0.05.

If both conditions are not met on a fold, the outcome for that fold must be reported as **'no improvement'**.

### 4. Reporting Commitment
- All three folds are reported in full whatever the outcome.
- No folds or negative outcomes may be omitted or selectively filtered.
- Individual seed metrics (min, mean, max) and ensemble metrics will be recorded in `results/loco_robust_metrics.json`.

## Alternatives Considered

- **Crack-held-out early stopping**:
  Rejected because using the test crack or another crack for early stopping selects degenerate models with overwhelming false call rates (`results/loco_crackval_metrics.json`).
- **Post-hoc threshold tuning on test flaws**:
  Rejected because tuning decision thresholds on test data causes data leakage and exaggerates practical performance.
- **Single-seed evaluation**:
  Rejected because neural network training exhibits seed-dependent variance across small datasets; 5 seeds plus ensembling provides variance assessment and improved score stability.

## Consequences

- **Methodological discipline**: Fixing the success rule prior to running `experiments/run_loco_robust.py` prevents selective metric reporting.
- **Strict falsifiability**: If robust augmentations fail to beat the baseline AUROC by $\ge 0.05$ while preserving detection at FCR 0.05, the method will be documented as having no improvement.
- **Bounded evidence base**: The benchmark rests on 3 physical cracks in a single weld specimen. Outcomes apply solely to this dataset and do not show generalisation to unseen flaw types, diverse geometries, or plant environments.
- **Safety architecture maintained**: In alignment with ADR 0001, any advisory model flags can only add candidate indications for human review and cannot override or dismiss conventional rules.
