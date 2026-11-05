# 0013. LiteHybrid (H4) detector pre-registration

- Status: Proposed
- Date: 2026-11-06

## Context

Ultrasonic testing (UT) in bar inspection requires fast, responsive triage on mobile laptop hardware with quad-core CPUs and no dedicated GPU.
Standard convolutional models (such as MlNdtNet) possess large parameter counts that increase inference latency and memory footprints.
At the same time, supervised models evaluated in earlier leave-one-crack-out experiments memorise known flaw signatures and exhibit variable sensitivity to unfamiliar flaw morphologies (`results/loco_metrics.json`, `results/fit_diagnostics.json`).
Earlier hybrid detectors H1 (OR-union) and H2 (score fusion) recorded in `results/hybrid_metrics.json`, and interpretable physics-feature models H3 recorded in `results/physics_model_metrics.json`, demonstrated that combining supervised and anomaly mechanisms can assist detection, but either raised false calls or suffered score dilution when a branch score collapsed.

To explore whether an ultra-lightweight neural network combined with a regularised physics-feature anomaly detector can reduce CPU latency while preserving detection sensitivity and controlling false alarms, candidate H4 (LiteHybrid) must be pre-registered.

Pre-registering the architecture, evaluation protocol, decision thresholds, and success rules prior to executing experiments or computing metrics prevents data leakage, metric shopping, and post-hoc threshold selection.
No result numbers currently exist for this experiment. This document records the experimental contract in advance.

## Decision

Pre-register the following architecture, protocol, baselines, and success criteria for H4 LiteHybrid:

### 1. Architecture: H4 LiteHybrid

Candidate H4 combines two complementary detectors via OR-fusion:

1. **Supervised Branch (LiteNet)**:
   - A convolutional neural network utilising depthwise-separable convolutions to cut spatial and channel arithmetic complexity.
   - Constrained to a parameter budget of at most 100,000 parameters ($\le 100\text{k}$ params).
   - Operates on raw B-scan acoustic patches to score defect likelihood.

2. **Anomaly Branch (Mahalanobis on Physics Features)**:
   - Computes an 8-dimensional physics feature vector (peak amplitude, noise MAD, SNR, peak depth index, echo duration, lateral extent, above-noise area, and echo energy ratio).
   - Fitted exclusively on sound, flaw-free calibration scans to estimate the clean centroid $\mu$ and covariance $\Sigma_{\text{LW}}$.
   - Utilises Ledoit-Wolf shrinkage to produce a well-conditioned covariance matrix that avoids inversion noise in small-sample regimes.
   - Calculates the distance $d_{\text{LW}}(x) = \sqrt{(x - \mu)^T \Sigma_{\text{LW}}^{-1} (x - \mu)}$.

3. **OR-Fusion Decision Rule**:
   - An indication is flagged if LiteNet exceeds its threshold OR the Mahalanobis distance exceeds its threshold:
     $$\text{Flag}_{\text{fused}}(x) = \mathbb{I}(s_{\text{LiteNet}}(x) \ge \tau_{\text{LiteNet}}) \lor \mathbb{I}(d_{\text{LW}}(x) \ge \tau_{\text{anomaly}})$$
   - Adheres to the safety principle that automated scoring can only add candidate flags for human inspector review and never suppress an indication.

### 2. Evaluation Protocol

- **Folds**: Exactly 3 leave-one-crack-out folds using the ML-NDT dataset (held-out cracks crack_1.6, crack_4.0 and crack_8.6, as named by make_folds).
- **Holdout Discipline**:
  - For each fold, models train only on the remaining two cracks using batch-held-out splits for tuning.
  - The held-out crack is strictly unseen during feature scaling, model training, hyperparameter selection, and early stopping. It is never used for model selection.
- **Random Seeds**: 5 fixed random seeds (seeds 0, 1, 2, 3, 4) per fold.
- **Clean Partitioning and Calibration**:
  - Available clean scans are split into two disjoint halves: a clean calibration split (`clean_calib`) and a clean evaluation split (`clean_eval`).
  - Thresholds are derived strictly on `clean_calib` by allocating half of the target False Call Rate (FCR) budget $\alpha = 0.05$ to each branch ($\alpha/2 = 0.025$ per branch):
    $$\tau_{\text{LiteNet}} = Q_{1 - 0.025}(\{s_{\text{LiteNet, clean\_cal}}\})$$
    $$\tau_{\text{anomaly}} = Q_{1 - 0.025}(\{d_{\text{LW, clean\_cal}}\})$$
  - Flaw scans are never used to set thresholds.
- **Evaluation Partition**:
  - The empirical false call rate is measured on `clean_eval`. By Boole's inequality, the union bound constrains empirical FCR to $\le \alpha = 0.05$.
  - Detection rate is measured on the held-out flawed scans using the clean-calibrated thresholds.

### 3. Metrics

- **Primary Metrics**:
  - Per-fold detection rate (hit rate) at clean-calibrated FCR target of 0.05 on the held-out crack partition.
  - Area Under the Receiver Operating Characteristic curve (AUROC) of the fused score on the held-out crack partition.
- **Secondary Metrics**:
  - CPU inference latency per scan (median time in milliseconds) measured with 4 worker threads (`torch.set_num_threads(4)`).
  - Total model parameter count for LiteNet (must satisfy $\le 100\text{k}$).

### 4. Comparison Baselines

Performance is benchmarked directly against:
1. **MlNdtNet (batchval)** from `results/separability_metrics.json`.
2. **Hybrid candidate H2 (score fusion)** from `results/hybrid_metrics.json`.

Evaluations occur on identical test scan partitions for direct comparison.

### 5. Success Rule (Fixed in Advance)

Candidate H4 is designated **'better'** on a fold only if:
1. Its fused score AUROC meets or exceeds the baseline AUROC by at least 0.05 ($\Delta\text{AUROC} \ge +0.05$); **AND**
2. Its detection rate at clean-calibrated FCR 0.05 is not lower than the baseline detection rate at FCR 0.05.

Candidate H4 is designated **'lighter'** if:
- Its median CPU latency is lower than MlNdtNet's median latency under the same 4-thread CPU benchmark.

If both conditions for 'better' are not satisfied on a fold, the outcome for that fold must be reported as **'no improvement'**.

### 6. Reporting Commitment and Evidence Limits

- **Complete reporting**: All 3 folds must be reported in full in `results/lite_hybrid_metrics.json`, regardless of whether outcomes are positive, neutral, or negative. No runs may be omitted.
- **Bounded evidence base**: The benchmark rests on 3 physical cracks within a single austenitic steel weld specimen. Findings apply strictly within this bounded experimental setting and do not show generalisation to diverse alloy grades, component shapes, or shop-floor conditions.

## Alternatives Considered

- **Standard convolutions in LiteNet**:
  Rejected because standard $3 \times 3$ convolutions across deep feature channels exceed the 100k parameter budget and increase FLOP counts on quad-core CPUs.
- **Supervised-only lightweight model**:
  Rejected because a purely supervised model remains blind to unusual flaw types not represented in the training set.
- **Sample covariance without shrinkage**:
  Rejected because the empirical covariance matrix on small sample sizes becomes ill-conditioned, amplifying inversion noise.
- **Post-hoc threshold selection on test flaws**:
  Rejected because selecting thresholds on test data introduces data leakage and exaggerates detection sensitivity.

## Consequences

- **Methodological discipline**: Fixing the evaluation contract and success rules prior to running experiments prevents selective metric reporting.
- **Computational efficiency**: Restricting LiteNet to $\le 100\text{k}$ parameters and using depthwise-separable convolutions targets low-latency execution on mobile quad-core laptop CPUs.
- **Safety architecture maintained**: In alignment with ADR 0001, OR-fusion keeps all candidate indications flagged by either branch, requiring that automated tools only add candidate regions for inspector evaluation and never suppress an indication.
- **Evidence boundary**: Findings remain strictly tied to the 3 physical cracks in the ML-NDT dataset.
