# Technical Note: AI-Assisted Ultrasonic Indication Triage Architecture and Phase 8 Generalization Evaluation

**Document ID:** UTRIAGE-TN-2026-02  
**Target Audience:** ASNT / NAS 410 UT Level III Examiners, Aerospace NDT Quality Engineers  
**Classification:** Technical Architecture & Evidence Evaluation  
**Status:** Advisory Independent Prototype / Research Baseline  

---

## Summary

The target application is aerospace bar immersion UT, but all real-data evaluations use public weld PAUT datasets and all bar results are Synthetic data. In automated immersion ultrasonic testing (UT) of aerospace-grade titanium and nickel-base superalloy round bars, coarse alloy microstructures create significant acoustic grain backscatter on RF A-scans. This background noise causes false calls under conventional thresholding, increasing review burden and inspector fatigue.

UTriage is an offline, advisory decision-support prototype designed to aid Level II and Level III inspectors. Its architecture enforces three non-negotiable safety principles:
1. **Conventional rules remain authoritative:** Code-based evaluation (DAC curves, gating, reference reflectors, and SNR thresholds) represents the safety baseline (baseline gate passed: `yes`).
2. **The AI can only add candidate flags:** Machine learning models have zero software authority to suppress, clear, or downgrade any rule-flagged indication. The union property is enforced in code and property-tested on synthetic simulator columns; the real weld datasets have no rules baseline.
3. **Inspector authority:** UTriage is an advisory tool and never accepts a part. All final dispositions remain the sole responsibility of the Level II / Level III examiner.

Phase 8 conducted an evidence-driven audit of detection claims. When evaluated on random batch splits, machine learning models exhibit data leakage across physical cracks (leakage present: `yes`), measuring interpolation within known flaws. Under rigorous group-held-out evaluations across physical cracks, unseen datasets, and simulator parameter regions, supervised models exhibit severe generalization boundaries. Unsupervised anomaly detection trained solely on flaw-free data provides a more resilient alternative.

---

## What Changed

### Data Leakage Audit
In earlier phases, real-data machine learning benchmarks were evaluated using a random batch split of the ML-NDT dataset. A Phase 8 leakage audit (`results/leakage_audit.json`) revealed that all three physical fatigue cracks in the dataset were shared across training and test partitions:
- `crack_1.6`
- `crack_4.0`
- `crack_8.6`

Because augmented scans of the same three physical cracks appeared in both splits, test scores reflected intra-crack interpolation across familiar flaw geometries rather than genuine detection of novel, unseen flaws.

### Old Numbers Superseded
The previously reported ML-NDT test detection result (`1222` of `1222` detected, Clopper-Pearson 95% lower bound `0.9976`, with false-call rate `0`, upper 95% bound `0.003843`) is **superseded**. 

All real-data claims are now evaluated strictly under group-held-out protocols.

---

## Evaluation Protocol

To prevent leakage and define valid performance boundaries, all evaluations in Phase 8 use group-held-out splits:

1. **By Physical Crack (LOCO):** Leave-one-crack-out cross-validation across the three physical cracks in the ML-NDT dataset. Each fold trains on two physical cracks and clean background scans, and is tested strictly on the third held-out physical crack and held-out clean scans.
2. **By Dataset (External Test):** Direct evaluation on the independent NDT_ML_Flaw dataset containing `6` distinct physical flaws in dissimilar metal weld PAUT data without fine-tuning.
3. **By Simulator Parameter Region:** For synthetic bar simulations, holding out entire parameter regions, including elevated probe frequency (`freq_10mhz`), a novel alloy material matrix (`material_third`), and extreme acoustic backscatter noise (`noise_above_range`).

### Why Random Splits Do Not Count
In ultrasonic non-destructive testing, multiple A-scans and B-scans from a single flaw share spatial orientation, metallurgical morphology, probe wedge coupling, and acoustic diffraction signatures. Random splitting scatters correlated scans from the same physical crack into both training and test partitions. This evaluates interpolation within known flaw signatures, presenting an illusion of high detection capability while failing on genuinely novel flaws.

---

## Real-Data Results

### Leave-One-Crack-Out (LOCO) Supervised Evaluation
*Dataset: Real data (ML-NDT)*

Supervised convolutional networks trained on two cracks and tested on the held-out crack demonstrate severe generalization breakdown on smaller cracks:

- **Held-out `crack_1.6`:**
  - Detections: `579` of `4014` (detection rate: `0.1442`)
  - False-call rate on unflawed scans: `0`
  - Hit/miss POD $a_{90/95}$: `2.163` mm equivalent
  - *Observation:* Near-total miss on the held-out 1.6 mm crack when trained on 4.0 mm and 8.6 mm cracks.
- **Held-out `crack_4.0`:**
  - Detections: `671` of `4078` (detection rate: `0.1645`)
  - False-call rate on unflawed scans: `0`
  - Hit/miss POD $a_{90/95}$: `5.912` mm equivalent
  - *Observation:* Near-total miss on the held-out 4.0 mm crack when trained on 1.6 mm and 8.6 mm cracks.
- **Held-out `crack_8.6`:**
  - Detections: `3925` of `3925`
  - False-call rate on unflawed scans: `0`
  - Clopper-Pearson 95% lower bound: `0.9992`

In the LOCO evaluation above, early stopping was batch-held-out (validation batches contain the training cracks, ADR 0009). When early stopping uses a strictly held-out physical crack for validation (`results/loco_crackval_metrics.json`), results swing strongly with the validation crack choice:
- **Held-out `crack_1.6` (validation crack `crack_4.0`):**
  - Detections: `4012` of `4014` (detection rate: `0.9995`)
  - False-call rate on unflawed scans: `0.9679` (`753` of `778`)
  - *Observation:* On `crack_1.6` the detector flagged most clean scans, so the high detection there is not a usable operating point.
- **Held-out `crack_4.0` (validation crack `crack_8.6`):**
  - Detections: `2449` of `4078` (detection rate: `0.6005`)
  - False-call rate on unflawed scans: `0` (`0` of `778`)
- **Held-out `crack_8.6` (validation crack `crack_1.6`):**
  - Detections: `3925` of `3925` (detection rate: `1`)
  - False-call rate on unflawed scans: `0.01414` (`11` of `778`)

### Separability and threshold calibration
*Dataset: Real data (ML-NDT)*

AUROC is the probability that a random flawed scan scores above a random clean scan. Folds under the batchval protocol (`results/separability_metrics.json`):

- **Held-out `crack_1.6`:**
  - AUROC: `1`
  - Average Precision: `1`
  - Operating point 0.01 target false-call rate: detection `4010` of `4014`, false calls `6` of `389`
  - Operating point 0.05 target false-call rate: detection `4014` of `4014`, false calls `28` of `389`
- **Held-out `crack_4.0`:**
  - AUROC: `0.998`
  - Average Precision: `0.9996`
  - Operating point 0.01 target false-call rate: detection `3896` of `4078`, false calls `2` of `389`
  - Operating point 0.05 target false-call rate: detection `4038` of `4078`, false calls `18` of `389`
- **Held-out `crack_8.6`:**
  - AUROC: `1`
  - Average Precision: `1`
  - Operating point 0.01 target false-call rate: detection `3925` of `3925`, false calls `1` of `389`
  - Operating point 0.05 target false-call rate: detection `3925` of `3925`, false calls `35` of `389`

Thresholds are set on half of the clean test scans and evaluated on the other half; clean test scans come from held-out batches of the same weld; the score margin above the clean threshold is small, so this does not show robustness to site changes (see ADR 0015).

### Unsupervised Anomaly Detection
*Dataset: Real data (ML-NDT)*

An unsupervised convolutional autoencoder (`ConvAE2D`) was trained exclusively on flaw-free ultrasonic scans and calibrated on clean validation scans. Because it does not rely on supervised flaw signatures, it exhibits greater resilience on unseen geometries:
- **Crack `crack_1.6`:** Detection rate `0.9091`, false-call rate `0.01799`
- **Crack `crack_4.0`:** Detection rate `0.3203`, false-call rate `0.01799`
- **Crack `crack_8.6`:** Detection rate `1`, false-call rate `0.01799`

### Cross-Dataset External Evaluation
*Dataset: Real data (NDT_ML_Flaw)*

Models trained on ML-NDT were evaluated directly on `7000` scans from the independent NDT_ML_Flaw dataset (dissimilar metal weld PAUT) under the documented spatial-temporal mapping:
- Supervised model (`mlndt_best.pt`): Detection rate `1`, false-call rate `1`
- Anomaly detector (`anomaly_2d.pt`): Detection rate `1`, false-call rate `1`

*Outcome:* Every model flagged every image. Differences in probe wedge angles, weld bevel geometry, material grain texture, and pulser electronics resulted in zero transfer across datasets under the documented mapping.

---

## Hybrid and physics-feature detectors (pre-registered, ADR 0010)

*Dataset: Real data (ML-NDT)*

To address supervised crack memorization without post-hoc tuning on test data, three detector architectures were pre-registered under ADR 0010 and evaluated on leave-one-crack-out (LOCO) folds across the three physical cracks:

1. **H1 (OR-union):** Flags an indication if the supervised probability is $\ge 0.5$ OR the unsupervised reconstruction anomaly score is $\ge T_{\text{anom}}$ (`0.3815`), calibrated on clean validation scans at target false-call rate `0.01`.
2. **H2 (Score fusion):** Computes fused score $s = \frac{1}{2}(p_{\text{sup}} + p_{\text{anom}})$, where $p_{\text{anom}}$ is the empirical cumulative distribution function (CDF) of the anomaly score on clean calibration scans. The decision threshold $T_s$ is the clean-calibrated 0.99 quantile of $s$ on clean calibration scans (`0.4951` on crack 1.6 fold, `0.4951` on crack 4.0 fold, `0.4952` on crack 8.6 fold).
3. **H3 (Physics features + logistic regression):** Extracts eight hand-crafted physics features (`peak_amplitude`, `robust_noise`, `snr`, `peak_depth`, `echo_duration`, `lateral_extent`, `above_noise_area`, `energy_ratio`), standardized with `StandardScaler` fit strictly on the training fold, and classified via balanced-weight logistic regression at threshold probability $\ge 0.5$.

### Per-Crack Evaluation

- **Held-out `crack_1.6` (1.6 mm crack):**
  - Supervised (`loco_crack_1.6.pt`): `579` of `4014` (detection rate: `0.1442`), false-call rate: `0` (`0` of `778`)
  - Anomaly: `3649` of `4014` (detection rate: `0.9091`), false-call rate: `0.01799` (`14` of `778`)
  - Hybrid H1 (OR-union): `3653` of `4014` (detection rate: `0.9101`), false-call rate: `0.01799` (`14` of `778`)
  - Hybrid H2 (Fused score): `3761` of `4014` (detection rate: `0.937`), false-call rate: `0.008997` (`7` of `778`)
  - Physics H3 (Logistic regression): `2661` of `4014` (detection rate: `0.6629`), false-call rate: `0` (`0` of `778`)
- **Held-out `crack_4.0` (4.0 mm crack):**
  - Supervised (`loco_crack_4.0.pt`): `671` of `4078` (detection rate: `0.1645`), false-call rate: `0` (`0` of `778`)
  - Anomaly: `1306` of `4078` (detection rate: `0.3203`), false-call rate: `0.01799` (`14` of `778`)
  - Hybrid H1 (OR-union): `1610` of `4078` (detection rate: `0.3948`), false-call rate: `0.01799` (`14` of `778`)
  - Hybrid H2 (Fused score): `2272` of `4078` (detection rate: `0.5571`), false-call rate: `0.008997` (`7` of `778`)
  - Physics H3 (Logistic regression): `2578` of `4078` (detection rate: `0.6322`), false-call rate: `0` (`0` of `778`)
- **Held-out `crack_8.6` (8.6 mm crack):**
  - Supervised (`loco_crack_8.6.pt`): `3925` of `3925` (detection rate: `1`), false-call rate: `0` (`0` of `778`)
  - Anomaly: `3925` of `3925` (detection rate: `1`), false-call rate: `0.01799` (`14` of `778`)
  - Hybrid H1 (OR-union): `3925` of `3925` (detection rate: `1`), false-call rate: `0.01799` (`14` of `778`)
  - Hybrid H2 (Fused score): `3925` of `3925` (detection rate: `1`), false-call rate: `0.008997` (`7` of `778`)
  - Physics H3 (Logistic regression): `3925` of `3925` (detection rate: `1`), false-call rate: `0.001285` (`1` of `778`)

### Seed Sensitivity

Evaluating H2 across distinct random training seeds shows marked sensitivity on the intermediate crack size (`crack_4.0`), while remaining stable on large and small cracks:
- **`crack_1.6` Fold:** Seed detection rates range across seeds from `3687` of `4014` (`0.9185`) to `3766` of `4014` (`0.9382`), with false calls fixed at `7` of `778` (`0.008997`).
- **`crack_4.0` Fold:** Seed detection rates range across seeds from `899` of `4078` (`0.2205`) to `2272` of `4078` (`0.5571`), with false calls fixed at `7` of `778` (`0.008997`).
- **`crack_8.6` Fold:** Seed detection rates are identical across seeds at `3925` of `3925` (`1`), with false calls at `7` of `778` (`0.008997`).

### Comparison Across Architectures

Comparing the two alternative architectures per crack:
- On `crack_1.6`: H2 detection rate `0.937` (`3761` of `4014`) vs H3 detection rate `0.6629` (`2661` of `4014`).
- On `crack_4.0`: H2 detection rate `0.5571` (`2272` of `4078`) vs H3 detection rate `0.6322` (`2578` of `4078`).
- On `crack_8.6`: H2 detection rate `1` (`3925` of `3925`) vs H3 detection rate `1` (`3925` of `3925`).

H2 detection on the 4.0 mm crack varies widely across seeds (`0.2205` to `0.5571`, or `899` to `2272` of `4078`) and H3 coefficients change sign across folds, so no ranking is claimed.

### Coefficient Instability Across Folds

Although the interpretable physics model (H3) achieves zero false calls on the smaller held-out cracks (`0` on `crack_1.6` and `0` on `crack_4.0`), its logistic regression coefficients fluctuate drastically and flip signs across LOCO folds:
- **`energy_ratio` Coefficient:** `18.91` on fold 0 vs `-13.48` on fold 1 vs `1.317` on fold 2.
- **`peak_depth` Coefficient:** `4.232` on fold 0 vs `-5.624` on fold 1 vs `0.3489` on fold 2.
- **`peak_amplitude` Coefficient:** `-0.006443` on fold 0 vs `1.595` on fold 1 vs `0.1236` on fold 2.
- **`above_noise_area` Coefficient:** `10.58` on fold 0 vs `12.58` on fold 1 vs `4.287` on fold 2.

This instability underscores that the linear classifier over-fits idiosyncratic feature correlations of the specific two training flaws rather than learning invariant flaw physics across flaw geometries.

---

## False-Call Analysis

*Dataset: Real data (ML-NDT)*

Evaluation of false calls on clean unflawed scans (`results/false_call_analysis.json`):
- **Fold-scan clean evaluations (each distinct clean scan scored by every fold model):** `2334`
- **Flagged by any detector:** `43`
- **Per-fold flags:**
  - `crack_1.6` fold: `14` of `778` clean scans
  - `crack_4.0` fold: `14` of `778` clean scans
  - `crack_8.6` fold: `15` of `778` clean scans
- **Agreement counts:**
  - Single detector only: `22`
  - Two detectors: `21`
  - Three detectors: `0`
  - Four detectors: `0`
  - Total flags by supervised: `0`
  - Total flags by anomaly: `42`
  - Total flags by H2: `21`
  - Total flags by H3: `1`
- **Primary causes:**
  - High SNR structure: `43`

---

## Site Adaptation

*Dataset: real weld images with synthetic perturbations, single seed (`0`)*

Limits: `Real ML-NDT data contains only 3 physical thermal fatigue cracks in an austenitic pipe weld; all inspection environment shifts (gain, offset, noise) are synthetic perturbations applied in software. Advisory only: UTriage never accepts or passes a part.`

The evaluated ConvAE2D model has no BatchNorm layers, so AdaBN made no change in this experiment (adapted model hash identical across shifts); measured changes come from per-site normalisation and threshold recalibration.

Calibration clean scans: `434` (`n_calib_clean`); test clean scans: `433` (`n_test_clean`). The `upper95_fcr` is computed on calibration clean scans, not the test set. Performance across synthetic perturbation shifts compares baseline versus adapted models (`results/site_adaptation.json`):
- **Shift `identity`:**
  - False calls: baseline `11` of `433` (rate `0.0254`) vs adapted `11` of `433` (rate `0.0254`)
  - Detection: baseline `43` of `57` (rate `0.7544`) vs adapted `43` of `57` (rate `0.7544`)
- **Shift `gain 0.5`:**
  - False calls: baseline `0` of `433` (rate `0`) vs adapted `11` of `433` (rate `0.0254`)
  - Detection: baseline `19` of `57` (rate `0.3333`) vs adapted `43` of `57` (rate `0.7544`)
- **Shift `gain 2.0`:**
  - False calls: baseline `433` of `433` (rate `1`) vs adapted `11` of `433` (rate `0.0254`)
  - Detection: baseline `57` of `57` (rate `1`) vs adapted `43` of `57` (rate `0.7544`)
- **Shift `offset +0.5`:**
  - False calls: baseline `226` of `433` (rate `0.5219`) vs adapted `11` of `433` (rate `0.0254`)
  - Detection: baseline `48` of `57` (rate `0.8421`) vs adapted `43` of `57` (rate `0.7544`)
- **Shift `noise 0.2`:**
  - False calls: baseline `30` of `433` (rate `0.06928`) vs adapted `8` of `433` (rate `0.01848`)
  - Detection: baseline `49` of `57` (rate `0.8596`) vs adapted `43` of `57` (rate `0.7544`)
- **Shift `combined`:**
  - False calls: baseline `0` of `433` (rate `0`) vs adapted `7` of `433` (rate `0.01617`)
  - Detection: baseline `21` of `57` (rate `0.3684`) vs adapted `39` of `57` (rate `0.6842`)

---

## Joint Training and Self-Supervised Pretraining

*Dataset: real metrics on ML-NDT weld; sim images Synthetic data*

Joint training explores synthetic-data mixing and self-supervised masked pretraining across arms A through D (`results/joint_train_metrics.json`). Across all folds, no arm improved every crack:
- **Held-out `crack_1.6`:**
  - Arm A (`real_only`): detection `165` of `4014` (rate `0.04111`), false calls `0` of `778` (rate `0`)
  - Arm B (`real_plus_sim`): detection `878` of `4014` (rate `0.2187`), false calls `0` of `778` (rate `0`)
  - Arm C (`pretrain_real`): detection `2` of `4014` (rate `0.0004983`), false calls `0` of `778` (rate `0`)
  - Arm D (`pretrain_real_plus_sim`): detection `2421` of `4014` (rate `0.6031`), false calls `0` of `778` (rate `0`)
- **Held-out `crack_4.0`:**
  - Arm A (`real_only`): detection `0` of `4078` (rate `0`), false calls `0` of `778` (rate `0`)
  - Arm B (`real_plus_sim`): detection `0` of `4078` (rate `0`), false calls `0` of `778` (rate `0`)
  - Arm C (`pretrain_real`): detection `1264` of `4078` (rate `0.31`), false calls `0` of `778` (rate `0`)
  - Arm D (`pretrain_real_plus_sim`): detection `2158` of `4078` (rate `0.5292`), false calls `0` of `778` (rate `0`)
- **Held-out `crack_8.6`:**
  - Arm A (`real_only`): detection `3925` of `3925` (rate `1`), false calls `0` of `778` (rate `0`)
  - Arm B (`real_plus_sim`): detection `3925` of `3925` (rate `1`), false calls `0` of `778` (rate `0`)
  - Arm C (`pretrain_real`): detection `3925` of `3925` (rate `1`), false calls `0` of `778` (rate `0`)
  - Arm D (`pretrain_real_plus_sim`): detection `3925` of `3925` (rate `1`), false calls `0` of `778` (rate `0`)

---

## Synthetic Results

All synthetic evaluations are conducted on simulated cylindrical titanium and nickel alloy bars (`src/utriage/physics/simulator.py`).

### Parameter Region Generalization
*Dataset: Synthetic data*

When evaluated within the training parameter distribution:
- Rules baseline detection rate: `0.7725`, false-call rate: `0.02927` (Synthetic data)
- ML 1D model detection rate: `0.8875`, false-call rate: `0.1341` (Synthetic data)

However, shifting to held-out operational parameter regions exposes significant failure modes:

1. **Held-Out 10 MHz Transducer Frequency (`freq_10mhz`):**
   Acoustic Rayleigh scattering increases sharply at higher frequencies, causing severe false-call blow-up:
   - Rules baseline false-call rate: `0.7476` (Synthetic data)
   - ML 1D model false-call rate: `0.6` (Synthetic data)
   - Fused union false-call rate: `0.8024` (Synthetic data)
2. **Held-Out Noise Above Range (`noise_above_range`):**
   When material grain scattering exceeds nominal bounds, high background peaks breach decision gates:
   - Rules baseline false-call rate: `0.3429` (Synthetic data)
   - ML 1D model false-call rate: `0.4071` (Synthetic data)
   - Fused union false-call rate: `0.5286` (Synthetic data)

---

## Fit Diagnostics

Diagnostic checks (`results/fit_diagnostics.json`) quantify the degree of model overfitting to known training flaws. The evaluation criterion flags overfitting whenever the performance gap between training and held-out test groups exceeds 0.10.

### Supervised LOCO Overfitting
The supervised models detected only a small share of held-out cracks (`579`/`4014` on `crack_1.6` and `671`/`4078` on `crack_4.0`):
- **Crack `crack_1.6` Fold:** Training detection rate `1` vs held-out detection rate `0.1442` (Diagnosis: severe overfitting)
- **Crack `crack_4.0` Fold:** Training detection rate `1` vs held-out detection rate `0.1645` (Diagnosis: severe overfitting)

Because the network trained on only two crack shapes, it failed to form generalizable acoustic representations of smaller cracks.

### Synthetic Supervised Shift Gaps
*Dataset: Synthetic data*

In synthetic bar experiments, out-of-distribution parameter shifts reveal large generalization deficits:
- Frequency shift (`freq_10mhz`) generalization gap: `0.3675` (Synthetic data)
- Noise shift (`noise_above_range`) generalization gap: `0.03` (Synthetic data)

---

## Latency

Inference benchmarks were conducted on CPU on the local development machine (`results/hardware.json`):
- **Development Processor:** `Intel64 Family 6 Model 198 Stepping 2, GenuineIntel`
- **ML-NDT ResNet Single-Sample Latency:** Median `14.2` ms
- **Bar-Domain 1D Single-Sample Latency:** Median `0.971` ms

*Operational Constraint:* This development machine features a multi-core high-performance CPU. These figures do not represent execution latency on the target field laptop (Intel Core i5-1135G7, 8 GB RAM). Hardware re-benchmarking on the specific plant inspection computer remains an essential prerequisite.

---

## Evidence Limits

A rigorous evidence-based assessment requires explicit disclosure of technical and data boundaries:

1. **Total Distinct Real Physical Flaws:**
   Across accepted public ultrasonic datasets, the count of distinct physical flaws is documented in `results/data_inventory.md` (physical flaws used in evaluation comprise ML-NDT and NDT_ML_Flaw; remaining accepted datasets have unverified flaw counts).
2. **Weld Data Versus Bar Inspection:**
   All public real-data benchmarks are derived from pipe or plate welds, not aerospace round bars or billets. No public ultrasonic inspection datasets exist for rotating immersion testing of titanium or nickel superalloy bars.
3. **Unverified Simulator Constants:**
   Material acoustic velocities, attenuation curves, and grain noise parameters in `configs/materials.yaml` are handbook approximations labelled `UNVERIFIED — Ruri to check`. Acceptance thresholds in `configs/acceptance.yaml` are unverified placeholders and do not reproduce paywalled AMS-STD-2154 or ASTM E2375 tables.
4. **Advisory Independent Prototype:**
   UTriage is an advisory independent prototype. It never accepts a part; every final inspection disposition belongs exclusively to the Level II / Level III examiner. It operates strictly air-gapped and offline on exported scan files with no connection to UT instruments or plant machinery. No plant validation exists.

---

## Prior Art

1. **United Technologies / RTX Patent (US 9,927,404 B2):**
   "Phased array billet data evaluation software" — describes SNR-based indication classification and gating methods for ultrasonic inspection of titanium billets.
2. **Virkkunen et al. (arXiv:1903.11399):**
   "Augmented Ultrasonic Data for Machine Learning" — describes residual neural network architectures, early max-pooling envelope extraction, and virtual flaw augmentation on the ML-NDT dataset (LGPL-3.0).
3. **Tunukovic et al. (Ultrasonics 140 (2024) 107313):**
   Approach for unsupervised acoustic anomaly detection using convolutional autoencoders trained exclusively on flaw-free ultrasonic scans, evaluating reconstruction error after interface and wall gating (citation taken from docs/phase8-spec.md; details UNVERIFIED - Ruri to check).
