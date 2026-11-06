# 0015. Shift robustness evaluation pre-registration

- Status: Proposed
- Date: 2026-11-06

## Context

In ultrasonic non-destructive testing (UT), supervised deep learning models trained on small datasets risk performance collapse when exposed to acoustic shifts (such as variations in transducer coupling, electronic baseline drift, or microstructural grain noise).

In earlier evaluations recorded in the separability metrics (`results/separability_metrics.json`), the leave-one-crack-out baseline achieved AUROC at or near the maximum on every held-out crack.
Because the baseline AUROC is already at or near the ceiling across all held-out cracks, the pre-registered AUROC improvement rule specified in ADR 0012, ADR 0013, and ADR 0014 (which requires candidate AUROC to exceed the baseline by at least 0.05) can no longer be met by any candidate model.
Those earlier rules stay as written and will be reported as written ('no improvement' where AUROC cannot rise).

This ADR adds a separate, additional evaluation question.
This question is formulated after observing the baseline AUROC ceiling, but strictly before any shift robustness result exists for any model.

## Question

Does the flaw/clean separation survive an inspection site shift when the decision threshold is set on unshifted clean scans and not recalibrated?

## Decision

Pre-register the following evaluation protocol, shift perturbations, metrics, baselines, and success criteria for `experiments/run_shift_robustness.py`:

### 1. Evaluation Protocol

- **Execution script**: `experiments/run_shift_robustness.py`.
- **Folds**: Exactly 3 leave-one-crack-out folds using the ML-NDT dataset (held-out cracks crack_1.6, crack_4.0, and crack_8.6).
- **Holdout discipline**: In each fold, models train only on two cracks. Early stopping and hyperparameter tuning use batch-held-out partitions from those training cracks. The held-out crack is strictly unseen during training and tuning.
- **Identical test evaluation**: For each fold, all model families evaluate on the exact same shifted test scans.
- **Threshold calibration**:
  - Decision thresholds for target False Call Rates (FCR) of 0.01 and 0.05 are derived solely on unshifted clean calibration scans (`clean_calib`) as empirical quantiles at $(1 - \text{FCR})$.
  - Flaw scans are never used to set thresholds.
  - Crucially, thresholds are **not** recalibrated on shifted scans. The unshifted clean threshold is applied directly across all shifted test splits to test innate robustness.
  - Empirical false call rates under shift are measured on the disjoint clean evaluation split (`clean_eval`).

### 2. Shift Perturbations

Evaluations apply the exact six fixed site shift perturbations (`FIXED_SHIFTS`) from `experiments/run_site_adaptation.py`:
1. `identity`: gain = 1.0, offset = 0.0, noise_std = 0.0
2. `gain 0.5`: gain = 0.5, offset = 0.0, noise_std = 0.0
3. `gain 2.0`: gain = 2.0, offset = 0.0, noise_std = 0.0
4. `offset +0.5`: gain = 1.0, offset = 0.5, noise_std = 0.0
5. `noise 0.2`: gain = 1.0, offset = 0.0, noise_std = 0.2
6. `combined`: gain = 0.5, offset = 0.5, noise_std = 0.2

Each shift transforms raw image array $x$ via $x' = g \cdot x + b + n$, where $g$ is gain, $b$ is DC offset, and $n \sim \mathcal{N}(0, \sigma^2)$ is Gaussian noise.

### 3. Metrics

- **Primary Metric**:
  - Per fold, worst-case detection rate (hit rate) across all six shifts at the unshifted clean-calibrated FCR 0.01 threshold:
    $$\text{Detection}_{\text{worst}} = \min_{s \in \text{Shifts}} \text{Detection}(s; \tau_{0.01})$$
- **Secondary Metrics**:
  - Worst-case False Call Rate (FCR) across all six shifts at the unshifted clean-calibrated FCR 0.01 threshold:
    $$\text{FCR}_{\text{worst}} = \max_{s \in \text{Shifts}} \text{FCR}(s; \tau_{0.01})$$
  - AUROC under each shift condition.
  - Log-margin $m = \log_{10}(\text{median } s_{\text{flawed}}) - \log_{10}(\tau_{0.01})$ under each shift.

### 4. Comparison Baselines and Candidate Families

Performance is compared within each fold between candidate models and the baseline family on the same fold:
- **Baseline**: Leave-one-crack-out baseline family (`loco_baseline`).
- **Candidates**:
  - `robust`: physics-augmented robust training family (ADR 0012).
  - `lite`: lightweight hybrid architecture family (ADR 0013).
  - `invariant`: invariant representation learning family (ADR 0014).

### 5. Success Rule (Fixed in Advance)

A candidate model is designated **'more shift-robust'** on a fold if and only if:
1. Its worst-case detection rate across shifts at FCR 0.01 is at least 0.10 higher than the baseline worst-case detection rate on that fold:
   $$\text{Detection}_{\text{worst}}^{\text{candidate}} \ge \text{Detection}_{\text{worst}}^{\text{baseline}} + 0.10$$
   **AND**
2. Its worst-case false-call rate across shifts is not higher than the baseline worst-case false-call rate plus 0.01:
   $$\text{FCR}_{\text{worst}}^{\text{candidate}} \le \text{FCR}_{\text{worst}}^{\text{baseline}} + 0.01$$

If both conditions are not met on a fold, the outcome for that candidate on that fold must be reported as **'no improvement'**.

### 6. Reporting Commitment

- All three folds and all model families will be reported in full in `results/shift_robustness_metrics.json`.
- Outcomes will be reported regardless of whether candidate models show improvement, parity, or regression.
- No runs, folds, or shift conditions may be omitted.

## Alternatives Considered

- **Recalibrating thresholds per shift**:
  Rejected because adapting thresholds on target clean scans evaluates site adaptation (ADR 0011). This experiment specifically evaluates whether an unadapted model preserves score separation when exposed to unexpected acquisition shifts.
- **Evaluating only at FCR 0.05**:
  Rejected because ultrasonic flaw screening requires tight false-alarm control; evaluating worst-case detection at FCR 0.01 tests operational resilience under strict thresholding.
- **Average detection across shifts**:
  Rejected because an average metric can conceal complete failure under a specific acoustic shift (such as gain attenuation). A reliable inspection tool must be judged by its worst-case failure mode.

## Consequences and Scope Limits

- **Experimental discipline**: Fixing the hypothesis, metrics, and success criteria prior to running `experiments/run_shift_robustness.py` prevents selective reporting.
- **Safety architecture**: In accordance with ADR 0001, ML predictions serve strictly in an advisory role to assist human review. ML flags can only add candidate indications and never suppress rules-based flags.
- **Scope and dataset boundaries**:
  - Shifts represent synthetic perturbations (gain, DC offset, Gaussian noise) and do not capture all complex physical site variations.
  - The dataset contains 3 physical cracks in a single weld specimen; conclusions apply only to this benchmark and do not show generalisation across diverse alloys or geometries.
  - Unsupervised site calibration (ADR 0011) remains the intended operational approach; this test measures behaviour when site calibration is absent.
