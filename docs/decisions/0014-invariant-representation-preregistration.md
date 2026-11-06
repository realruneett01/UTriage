# 0014. Invariant representation learning pre-registration

- Status: Proposed
- Date: 2026-11-06

## Context

Ultrasonic testing (UT) in aerospace bar inspection requires responsive indication triage on mobile laptop hardware with quad-core CPUs and no dedicated GPU.
Supervised models evaluated in earlier leave-one-crack-out experiments (`results/separability_metrics.json`, `results/lite_hybrid_metrics.json`) show significant score shifts and variable sensitivity when evaluated on unseen cracks (specifically the 1.6 mm and 4.0 mm cracks).

Supervised models trained on only two physical cracks learn the specific spatial echoes, facet orientations, and tip diffractions of "these two cracks" rather than a general representation of "an acoustic flaw".
Although the ML-NDT benchmark contains thousands of B-scan slices, these slices represent repeated spatial passes over only 3 physical defects, not thousands of independent flaws.
Standard training treats acquisition variations (gain, DC baseline drift, grain scatter, water path delay, sound velocity, probe speed, and scan direction) as arbitrary noise rather than physical invariances.

To investigate whether invariant representation learning—combining contrastive pretraining with physics-informed views, fine-tuning, and a clean-only Mahalanobis anomaly detector on the learned embedding—improves score separability on unseen flaws, this architecture, evaluation protocol, comparison baselines, and success criteria are pre-registered before running experiments.

No result numbers currently exist for this experiment.
This document fixes the experimental contract in advance to prevent post-hoc threshold selection or selective metric reporting.

## Decision

Pre-register the following architecture, protocol, metrics, baselines, and success criteria for invariant representation learning:

### 1. Architecture and Pipeline Design

The proposed system combines contrastive representation learning, supervised fine-tuning, and clean-only anomaly detection:

1. **Backbone and Projection Head**:
   - `LiteEncoder`: A depthwise-separable convolutional backbone restricted to at most 100,000 parameters ($\le 100\text{k}$ params).
   - A multi-layer perceptron projection head mapping latent feature representations $h \in \mathbb{R}^d$ to unit-normalised projection vectors $z \in \mathbb{R}^m$.

2. **Stage 1 — Contrastive Pretraining**:
   - The encoder and projection head undergo contrastive pretraining using seven physically grounded acoustic views:
     - Gain variation ($g$): acoustic coupling and receiver gain.
     - DC offset drift ($b$): pre-amplifier and converter baseline drift.
     - Noise injection ($n$): electronic noise and microstructural grain scatter.
     - Time shift ($\Delta t$): immersion water path variation.
     - Time stretch ($\alpha_t$): alloy sound velocity shifts.
     - Scan stretch ($\alpha_s$): probe scanning speed fluctuations and crack lateral extent.
     - Scan flip: probe scanning direction reversal.
   - Pretraining uses either self-supervised NT-Xent loss or Supervised Contrastive (SupCon) loss on the available training cracks.

3. **Stage 2 — Supervised Fine-Tuning**:
   - The projection head is removed, and a linear classification head is attached to encoder features $h$.
   - The network is fine-tuned with cross-entropy loss.
   - Encoder weights are adjusted with a smaller learning rate than the new classification head ($\eta_{\text{encoder}} \ll \eta_{\text{head}}$) to preserve the invariant latent geometry.

4. **Clean-Only Embedding Anomaly Branch**:
   - Operates on the encoder feature vectors $h = f_\theta(x)$ extracted from clean calibration scans.
   - Computes the clean centroid $\mu$ and covariance matrix $\Sigma_{\text{LW}}$ using Ledoit-Wolf shrinkage.
   - Evaluates the Mahalanobis distance $d_{\text{LW}}(x) = \sqrt{(x - \mu)^T \Sigma_{\text{LW}}^{-1} (x - \mu)}$.

5. **OR-Fusion Decision Rule**:
   - Combines the fine-tuned classifier and the embedding anomaly detector using an OR-rule:
     $$\text{Flag}_{\text{fused}}(x) = \mathbb{I}(s_{\text{classifier}}(x) \ge \tau_{\text{classifier}}) \lor \mathbb{I}(d_{\text{LW}}(x) \ge \tau_{\text{anomaly}})$$
   - Allocates the target false-call rate budget $\alpha$ equally between branches ($\alpha/2$ for the classifier, $\alpha/2$ for the anomaly branch).
   - Under Boole's inequality (union bound), the combined false-call rate on clean material satisfies $\text{FCR}_{\text{fused}} \le \alpha$.

### 2. Evaluation Protocol

- **Folds**: Exactly 3 leave-one-crack-out folds using the ML-NDT dataset (held-out cracks crack_1.6, crack_4.0, and crack_8.6).
- **Batch-Held-Out Holdout Discipline**:
  - In each fold, models train only on the remaining two cracks using batch-held-out splits.
  - The held-out crack is strictly unseen during Stage 1 pretraining, Stage 2 fine-tuning, embedding covariance fitting, calibration, and model selection.
- **Random Seeds**: 3 fixed random seeds (seeds 0, 1, 2) per fold.
- **Clean Partitioning and Calibration**:
  - Clean scans are split into two disjoint halves: a clean calibration set (`clean_calib`) and a clean evaluation set (`clean_eval`).
  - Decision thresholds are calibrated strictly on `clean_calib` at false-call allowance $\alpha/2$:
    $$\tau_{\text{classifier}} = Q_{1 - \alpha/2}(\{s_{\text{classifier, clean\_cal}}\})$$
    $$\tau_{\text{anomaly}} = Q_{1 - \alpha/2}(\{d_{\text{LW, clean\_cal}}\})$$
  - Flaw scans are never used to set or tune thresholds.
- **Evaluation Partition**:
  - False-call rate is evaluated on `clean_eval`.
  - Detection hit rate is evaluated on the held-out flawed scans using the clean-calibrated thresholds.

### 3. Metrics

- **Primary Metrics**:
  - Per-fold AUROC of the fused score on the held-out crack partition.
  - Per-fold AUROC of the fine-tuned classifier score on the held-out crack partition.
- **Secondary Metrics**:
  - Detection hit rate at clean-calibrated FCR 0.05 on the held-out crack partition.
  - Detection hit rate at clean-calibrated FCR 0.01 on the held-out crack partition.

### 4. Comparison Baselines

Performance is benchmarked directly against:
1. `results/separability_metrics.json` (batchval)
2. `results/lite_hybrid_metrics.json` (if present)

Evaluations occur on identical held-out test partitions for direct comparison.

### 5. Success Rule (Fixed in Advance)

Candidate invariant representation learning is designated **'better'** on a fold only if:
1. Its AUROC meets or exceeds the baseline AUROC by at least 0.05 ($\Delta\text{AUROC} \ge +0.05$); **AND**
2. Its detection hit rate at clean-calibrated FCR 0.05 is not lower than the baseline detection hit rate at FCR 0.05.

If both conditions are not met on a fold, the outcome for that fold must be designated **'no improvement'**.

All folds must be reported in full in the resulting metrics files whatever the outcome. No runs may be omitted or selectively chosen.

### 6. Reporting Commitment and Evidence Limits

- **Complete reporting**: All 3 folds must be reported in full, regardless of whether outcomes show improvement, parity, or regression.
- **Bounded evidence base**: The benchmark rests on 3 physical cracks within a single austenitic steel weld specimen. Findings apply strictly within this bounded experimental setting and do not show generalisation to all flaw types, materials, or component shapes.

## Alternatives Considered

- **Supervised training from scratch**:
  Rejected because standard cross-entropy training on two cracks overfits to those two flaw geometries and fails to learn invariance under acoustic acquisition shifts.
- **Generic vision transforms (random crops, color jitter)**:
  Rejected because B-scan acoustic data reflect time-of-flight physics; transformations must represent physical wave phenomena (coupling gain, water path delay, sound velocity, electronic drift, probe scanning direction).
- **Anomaly detection on handcrafted features alone**:
  Handcrafted physics features capture bulk echo metrics, but embedding representations from contrastive pretraining provide complementary spatial information learned directly from raw acoustic data.
- **Fixed default decision threshold ($\tau = 0.5$)**:
  Rejected because scores on novel crack geometries undergo scale compression and shift; clean-calibrated thresholding maintains controlled false-call rates without score scale assumptions.
- **Score averaging fusion**:
  Rejected because score degradation on an unfamiliar flaw orientation in one branch can dilute the combined score; OR-fusion preserves every candidate indication flagged by either branch.

## Consequences

- **Safety architecture maintained**: In alignment with ADR 0001, OR-fusion preserves all candidate indications flagged by either branch, upholding the principle that automated triage can only add candidate flags for human inspector review and never suppress an indication.
- **Methodological discipline**: Fixing the experimental contract, baselines, and success rule prior to execution prevents post-hoc threshold selection and selective reporting.
- **Two-stage pipeline**: Introduces a two-stage training workflow (contrastive pretraining followed by fine-tuning), but `LiteEncoder` retains a compact parameter budget ($\le 100\text{k}$ params) suitable for laptop CPU execution.
- **Bounded evidence boundary**: Conclusions remain strictly restricted to the 3 physical cracks in the ML-NDT dataset.
