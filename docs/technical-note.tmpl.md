# Technical Note: AI-Assisted Ultrasonic Indication Triage Architecture and Phase 8 Generalization Evaluation

**Document ID:** UTRIAGE-TN-2026-02  
**Target Audience:** ASNT / NAS 410 UT Level III Examiners, Aerospace NDT Quality Engineers  
**Classification:** Technical Architecture & Evidence Evaluation  
**Status:** Advisory Independent Prototype / Research Baseline  

---

## Summary

In automated immersion ultrasonic testing (UT) of aerospace-grade titanium and nickel-base superalloy round bars, coarse alloy microstructures create significant acoustic grain backscatter on RF A-scans. This background noise causes false calls under conventional thresholding, increasing review burden and inspector fatigue.

UTriage is an offline, advisory decision-support prototype designed to aid certified Level II and Level III inspectors. Its architecture enforces three non-negotiable safety principles:
1. **Conventional rules remain authoritative:** Code-based evaluation (DAC curves, gating, reference reflectors, and SNR thresholds) represents the safety baseline (`{{rules_baseline.json: gate.passed}}`).
2. **The AI can only add candidate flags:** Machine learning models have zero software authority to suppress, clear, or downgrade any rule-flagged indication.
3. **Certified inspector authority:** UTriage is an advisory tool and never accepts a part. All final dispositions remain the sole responsibility of the certified inspector.

Phase 8 established an evidence-driven audit of detection claims. When evaluated on random batch splits, machine learning models exhibit data leakage across physical cracks (`{{leakage_audit.json: leakage_by_physical_crack}}`), measuring interpolation within known flaws. Under rigorous group-held-out evaluations across physical cracks, unseen datasets, and simulator parameter regions, supervised models exhibit severe generalization boundaries. Unsupervised anomaly detection trained solely on flaw-free data provides a more resilient alternative.

---

## What Changed

### Data Leakage Audit
In earlier phases, real-data machine learning benchmarks were evaluated using a random batch split of the ML-NDT dataset. A Phase 8 leakage audit (`results/leakage_audit.json`) revealed that all three physical fatigue cracks in the dataset were shared across training and test partitions:
- `{{leakage_audit.json: shared_cracks_train_test.0}}`
- `{{leakage_audit.json: shared_cracks_train_test.1}}`
- `{{leakage_audit.json: shared_cracks_train_test.2}}`

Because augmented scans of the same three physical cracks appeared in both splits, test scores reflected intra-crack interpolation across familiar flaw geometries rather than genuine detection of novel, unseen flaws.

### Old Numbers Superseded
The previously reported ML-NDT test detection result (`{{mlndt_metrics.json: per_split.test.detection.n_hit}}` of `{{mlndt_metrics.json: per_split.test.detection.n_flawed}}` detected, Clopper-Pearson 95% lower bound `{{mlndt_metrics.json: per_split.test.detection.pod_lower95}}`, with false-call rate `{{mlndt_metrics.json: per_split.test.false_calls.rate}}`, upper 95% bound `{{mlndt_metrics.json: per_split.test.false_calls.upper95}}`) is **superseded**. 

All real-data claims are now evaluated strictly under group-held-out protocols.

---

## Evaluation Protocol

To prevent leakage and establish valid performance boundaries, all evaluations in Phase 8 use group-held-out splits:

1. **By Physical Crack (LOCO):** Leave-one-crack-out cross-validation across the three physical cracks in the ML-NDT dataset. Each fold trains on two physical cracks and clean background scans, and is tested strictly on the third held-out physical crack and held-out clean scans.
2. **By Dataset (External Test):** Direct evaluation on the independent NDT_ML_Flaw dataset containing `{{cross_dataset_metrics.json: models.mlndt_best.pt.overall.n_physical_flaws}}` distinct physical flaws in dissimilar metal weld PAUT data without fine-tuning.
3. **By Simulator Parameter Region:** For synthetic bar simulations, holding out entire parameter regions, including elevated probe frequency (`freq_10mhz`), a novel alloy material matrix (`material_third`), and extreme acoustic backscatter noise (`noise_above_range`).

### Why Random Splits Do Not Count
In ultrasonic non-destructive testing, multiple A-scans and B-scans from a single flaw share spatial orientation, metallurgical morphology, probe wedge coupling, and acoustic diffraction signatures. Random splitting scatters correlated scans from the same physical crack into both training and test partitions. This evaluates interpolation within known flaw signatures, presenting an illusion of high detection capability while failing on genuinely novel flaws.

---

## Real-Data Results

### Leave-One-Crack-Out (LOCO) Supervised Evaluation
*Dataset: Real data (ML-NDT)*

Supervised convolutional networks trained on two cracks and tested on the held-out crack demonstrate severe generalization breakdown on smaller cracks:

- **Held-out `{{loco_metrics.json: folds.0.held_out}}`:**
  - Detections: `{{loco_metrics.json: folds.0.detections.n_hit}}` of `{{loco_metrics.json: folds.0.detections.n_flawed}}` (detection rate: `{{loco_metrics.json: folds.0.detections.rate}}`)
  - False-call rate on unflawed scans: `{{loco_metrics.json: folds.0.false_calls.rate}}`
  - Hit/miss POD $a_{90/95}$: `{{loco_metrics.json: folds.0.pod.a90_95}}` mm equivalent
  - *Observation:* Near-total miss on the held-out 1.6 mm crack when trained on 4.0 mm and 8.6 mm cracks.
- **Held-out `{{loco_metrics.json: folds.1.held_out}}`:**
  - Detections: `{{loco_metrics.json: folds.1.detections.n_hit}}` of `{{loco_metrics.json: folds.1.detections.n_flawed}}` (detection rate: `{{loco_metrics.json: folds.1.detections.rate}}`)
  - False-call rate on unflawed scans: `{{loco_metrics.json: folds.1.false_calls.rate}}`
  - Hit/miss POD $a_{90/95}$: `{{loco_metrics.json: folds.1.pod.a90_95}}` mm equivalent
  - *Observation:* Near-total miss on the held-out 4.0 mm crack when trained on 1.6 mm and 8.6 mm cracks.
- **Held-out `{{loco_metrics.json: folds.2.held_out}}`:**
  - Detections: `{{loco_metrics.json: folds.2.detections.n_hit}}` of `{{loco_metrics.json: folds.2.detections.n_flawed}}`
  - False-call rate on unflawed scans: `{{loco_metrics.json: folds.2.false_calls.rate}}`
  - Clopper-Pearson 95% lower bound: `{{loco_metrics.json: folds.2.pod.pod_lower95}}`

### Unsupervised Anomaly Detection
*Dataset: Real data (ML-NDT)*

An unsupervised convolutional autoencoder (`ConvAE2D`) was trained exclusively on flaw-free ultrasonic scans and calibrated on clean validation scans. Because it does not rely on supervised flaw signatures, it exhibits greater resilience on unseen geometries:
- **Crack `crack_1.6`:** Detection rate `{{anomaly_metrics.json: real.per_crack.crack_1.6.detection.rate}}`, false-call rate `{{anomaly_metrics.json: real.per_crack.crack_1.6.false_calls.rate}}`
- **Crack `crack_4.0`:** Detection rate `{{anomaly_metrics.json: real.per_crack.crack_4.0.detection.rate}}`, false-call rate `{{anomaly_metrics.json: real.per_crack.crack_4.0.false_calls.rate}}`
- **Crack `crack_8.6`:** Detection rate `{{anomaly_metrics.json: real.per_crack.crack_8.6.detection.rate}}`, false-call rate `{{anomaly_metrics.json: real.per_crack.crack_8.6.false_calls.rate}}`

### Cross-Dataset External Evaluation
*Dataset: Real data (NDT_ML_Flaw)*

Models trained on ML-NDT were evaluated directly on 7,000 scans from the independent NDT_ML_Flaw dataset (dissimilar metal weld PAUT) under the documented spatial-temporal mapping:
- Supervised model (`mlndt_best.pt`): Detection rate `{{cross_dataset_metrics.json: models.mlndt_best.pt.overall.detection.rate}}`, false-call rate `{{cross_dataset_metrics.json: models.mlndt_best.pt.overall.false_calls.rate}}`
- Anomaly detector (`anomaly_2d.pt`): Detection rate `{{cross_dataset_metrics.json: models.anomaly_2d.pt.overall.detection.rate}}`, false-call rate `{{cross_dataset_metrics.json: models.anomaly_2d.pt.overall.false_calls.rate}}`

*Outcome:* Every model flagged every image. Differences in probe wedge angles, weld bevel geometry, material grain texture, and pulser electronics resulted in zero transfer across datasets under the documented mapping.

---

## Hybrid and physics-feature detectors (pre-registered, ADR 0010)

*Dataset: Real data (ML-NDT)*

To address supervised crack memorization without post-hoc tuning on test data, three detector architectures were pre-registered under ADR 0010 and evaluated on leave-one-crack-out (LOCO) folds across the three physical cracks:

1. **H1 (OR-union):** Flags an indication if the supervised probability is $\ge 0.5$ OR the unsupervised reconstruction anomaly score is $\ge T_{\text{anom}}$ (`{{hybrid_metrics.json: t_anom}}`), calibrated on clean validation scans at target false-call rate `{{hybrid_metrics.json: target_fcr}}`.
2. **H2 (Score fusion):** Computes fused score $s = \frac{1}{2}(p_{\text{sup}} + p_{\text{anom}})$, where $p_{\text{anom}}$ is the empirical cumulative distribution function (CDF) of the anomaly score on clean calibration scans. The decision threshold $T_s$ is the clean-calibrated 0.99 quantile of $s$ on clean calibration scans (`{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6.pt.t_s}}` on crack 1.6 fold, `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0.pt.t_s}}` on crack 4.0 fold, `{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6.pt.t_s}}` on crack 8.6 fold).
3. **H3 (Physics features + logistic regression):** Extracts eight hand-crafted physics features (`{{physics_model_metrics.json: feature_names.0}}`, `{{physics_model_metrics.json: feature_names.1}}`, `{{physics_model_metrics.json: feature_names.2}}`, `{{physics_model_metrics.json: feature_names.3}}`, `{{physics_model_metrics.json: feature_names.4}}`, `{{physics_model_metrics.json: feature_names.5}}`, `{{physics_model_metrics.json: feature_names.6}}`, `{{physics_model_metrics.json: feature_names.7}}`), standardized with `StandardScaler` fit strictly on the training fold, and classified via balanced-weight logistic regression at threshold probability $\ge 0.5$.

### Per-Crack Evaluation

- **Held-out `{{hybrid_metrics.json: folds.crack_1.6.held_out}}` (1.6 mm crack):**
  - Supervised (`loco_crack_1.6.pt`): `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6.pt.methods.supervised.detection.k}}` of `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6.pt.methods.supervised.detection.n}}` (detection rate: `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6.pt.methods.supervised.detection.rate}}`), false-call rate: `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6.pt.methods.supervised.false_calls.rate}}` (`{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6.pt.methods.supervised.false_calls.n_false}}` of `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6.pt.methods.supervised.false_calls.n_scans}}`)
  - Anomaly: `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6.pt.methods.anomaly.detection.k}}` of `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6.pt.methods.anomaly.detection.n}}` (detection rate: `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6.pt.methods.anomaly.detection.rate}}`), false-call rate: `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6.pt.methods.anomaly.false_calls.rate}}` (`{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6.pt.methods.anomaly.false_calls.n_false}}` of `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6.pt.methods.anomaly.false_calls.n_scans}}`)
  - Hybrid H1 (OR-union): `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6.pt.methods.h1.detection.k}}` of `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6.pt.methods.h1.detection.n}}` (detection rate: `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6.pt.methods.h1.detection.rate}}`), false-call rate: `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6.pt.methods.h1.false_calls.rate}}` (`{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6.pt.methods.h1.false_calls.n_false}}` of `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6.pt.methods.h1.false_calls.n_scans}}`)
  - Hybrid H2 (Fused score): `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6.pt.methods.h2.detection.k}}` of `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6.pt.methods.h2.detection.n}}` (detection rate: `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6.pt.methods.h2.detection.rate}}`), false-call rate: `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6.pt.methods.h2.false_calls.rate}}` (`{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6.pt.methods.h2.false_calls.n_false}}` of `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6.pt.methods.h2.false_calls.n_scans}}`)
  - Physics H3 (Logistic regression): `{{physics_model_metrics.json: folds.0.detections.k}}` of `{{physics_model_metrics.json: folds.0.detections.n}}` (detection rate: `{{physics_model_metrics.json: folds.0.detections.rate}}`), false-call rate: `{{physics_model_metrics.json: folds.0.false_calls.rate}}` (`{{physics_model_metrics.json: folds.0.false_calls.n_false}}` of `{{physics_model_metrics.json: folds.0.false_calls.n_clean}}`)
- **Held-out `{{hybrid_metrics.json: folds.crack_4.0.held_out}}` (4.0 mm crack):**
  - Supervised (`loco_crack_4.0.pt`): `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0.pt.methods.supervised.detection.k}}` of `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0.pt.methods.supervised.detection.n}}` (detection rate: `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0.pt.methods.supervised.detection.rate}}`), false-call rate: `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0.pt.methods.supervised.false_calls.rate}}` (`{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0.pt.methods.supervised.false_calls.n_false}}` of `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0.pt.methods.supervised.false_calls.n_scans}}`)
  - Anomaly: `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0.pt.methods.anomaly.detection.k}}` of `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0.pt.methods.anomaly.detection.n}}` (detection rate: `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0.pt.methods.anomaly.detection.rate}}`), false-call rate: `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0.pt.methods.anomaly.false_calls.rate}}` (`{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0.pt.methods.anomaly.false_calls.n_false}}` of `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0.pt.methods.anomaly.false_calls.n_scans}}`)
  - Hybrid H1 (OR-union): `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0.pt.methods.h1.detection.k}}` of `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0.pt.methods.h1.detection.n}}` (detection rate: `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0.pt.methods.h1.detection.rate}}`), false-call rate: `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0.pt.methods.h1.false_calls.rate}}` (`{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0.pt.methods.h1.false_calls.n_false}}` of `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0.pt.methods.h1.false_calls.n_scans}}`)
  - Hybrid H2 (Fused score): `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0.pt.methods.h2.detection.k}}` of `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0.pt.methods.h2.detection.n}}` (detection rate: `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0.pt.methods.h2.detection.rate}}`), false-call rate: `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0.pt.methods.h2.false_calls.rate}}` (`{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0.pt.methods.h2.false_calls.n_false}}` of `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0.pt.methods.h2.false_calls.n_scans}}`)
  - Physics H3 (Logistic regression): `{{physics_model_metrics.json: folds.1.detections.k}}` of `{{physics_model_metrics.json: folds.1.detections.n}}` (detection rate: `{{physics_model_metrics.json: folds.1.detections.rate}}`), false-call rate: `{{physics_model_metrics.json: folds.1.false_calls.rate}}` (`{{physics_model_metrics.json: folds.1.false_calls.n_false}}` of `{{physics_model_metrics.json: folds.1.false_calls.n_clean}}`)
- **Held-out `{{hybrid_metrics.json: folds.crack_8.6.held_out}}` (8.6 mm crack):**
  - Supervised (`loco_crack_8.6.pt`): `{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6.pt.methods.supervised.detection.k}}` of `{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6.pt.methods.supervised.detection.n}}` (detection rate: `{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6.pt.methods.supervised.detection.rate}}`), false-call rate: `{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6.pt.methods.supervised.false_calls.rate}}` (`{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6.pt.methods.supervised.false_calls.n_false}}` of `{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6.pt.methods.supervised.false_calls.n_scans}}`)
  - Anomaly: `{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6.pt.methods.anomaly.detection.k}}` of `{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6.pt.methods.anomaly.detection.n}}` (detection rate: `{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6.pt.methods.anomaly.detection.rate}}`), false-call rate: `{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6.pt.methods.anomaly.false_calls.rate}}` (`{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6.pt.methods.anomaly.false_calls.n_false}}` of `{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6.pt.methods.anomaly.false_calls.n_scans}}`)
  - Hybrid H1 (OR-union): `{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6.pt.methods.h1.detection.k}}` of `{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6.pt.methods.h1.detection.n}}` (detection rate: `{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6.pt.methods.h1.detection.rate}}`), false-call rate: `{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6.pt.methods.h1.false_calls.rate}}` (`{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6.pt.methods.h1.false_calls.n_false}}` of `{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6.pt.methods.h1.false_calls.n_scans}}`)
  - Hybrid H2 (Fused score): `{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6.pt.methods.h2.detection.k}}` of `{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6.pt.methods.h2.detection.n}}` (detection rate: `{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6.pt.methods.h2.detection.rate}}`), false-call rate: `{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6.pt.methods.h2.false_calls.rate}}` (`{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6.pt.methods.h2.false_calls.n_false}}` of `{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6.pt.methods.h2.false_calls.n_scans}}`)
  - Physics H3 (Logistic regression): `{{physics_model_metrics.json: folds.2.detections.k}}` of `{{physics_model_metrics.json: folds.2.detections.n}}` (detection rate: `{{physics_model_metrics.json: folds.2.detections.rate}}`), false-call rate: `{{physics_model_metrics.json: folds.2.false_calls.rate}}` (`{{physics_model_metrics.json: folds.2.false_calls.n_false}}` of `{{physics_model_metrics.json: folds.2.false_calls.n_clean}}`)

### Seed Sensitivity

Evaluating H2 across distinct random training seeds shows marked sensitivity on the intermediate crack size (`crack_4.0`), while remaining stable on large and small cracks:
- **`crack_1.6` Fold:** Seed detection rates range across seeds from `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6_seed0.pt.methods.h2.detection.k}}` of `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6_seed0.pt.methods.h2.detection.n}}` (`{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6_seed0.pt.methods.h2.detection.rate}}`) to `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6_seed1.pt.methods.h2.detection.k}}` of `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6_seed1.pt.methods.h2.detection.n}}` (`{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6_seed1.pt.methods.h2.detection.rate}}`), with false calls fixed at `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6_seed0.pt.methods.h2.false_calls.n_false}}` of `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6_seed0.pt.methods.h2.false_calls.n_scans}}` (`{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6_seed0.pt.methods.h2.false_calls.rate}}`).
- **`crack_4.0` Fold:** Seed detection rates range across seeds from `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0_seed2.pt.methods.h2.detection.k}}` of `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0_seed2.pt.methods.h2.detection.n}}` (`{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0_seed2.pt.methods.h2.detection.rate}}`) to `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0_seed0.pt.methods.h2.detection.k}}` of `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0_seed0.pt.methods.h2.detection.n}}` (`{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0_seed0.pt.methods.h2.detection.rate}}`), with false calls fixed at `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0_seed0.pt.methods.h2.false_calls.n_false}}` of `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0_seed0.pt.methods.h2.false_calls.n_scans}}` (`{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0_seed0.pt.methods.h2.false_calls.rate}}`).
- **`crack_8.6` Fold:** Seed detection rates are identical across seeds at `{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6_seed0.pt.methods.h2.detection.k}}` of `{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6_seed0.pt.methods.h2.detection.n}}` (`{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6_seed0.pt.methods.h2.detection.rate}}`), with false calls at `{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6_seed0.pt.methods.h2.false_calls.n_false}}` of `{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6_seed0.pt.methods.h2.false_calls.n_scans}}` (`{{hybrid_metrics.json: folds.crack_8.6.models.loco_crack_8.6_seed0.pt.methods.h2.false_calls.rate}}`).

### Complementarity

Comparing the two alternative architectures reveals clear complementarity across flaw sizes: H2 is strongest on the 1.6 mm crack (detection rate `{{hybrid_metrics.json: folds.crack_1.6.models.loco_crack_1.6.pt.methods.h2.detection.rate}}` for H2 vs `{{physics_model_metrics.json: folds.0.detections.rate}}` for H3), whereas H3 is strongest on the 4.0 mm crack (detection rate `{{physics_model_metrics.json: folds.1.detections.rate}}` for H3 vs `{{hybrid_metrics.json: folds.crack_4.0.models.loco_crack_4.0.pt.methods.h2.detection.rate}}` for H2).

### Coefficient Instability Across Folds

Although the interpretable physics model (H3) achieves zero false calls on the smaller held-out cracks (`{{physics_model_metrics.json: folds.0.false_calls.rate}}` on `crack_1.6` and `{{physics_model_metrics.json: folds.1.false_calls.rate}}` on `crack_4.0`), its logistic regression coefficients fluctuate drastically and flip signs across LOCO folds:
- **`energy_ratio` Coefficient:** `{{physics_model_metrics.json: folds.0.coefficients.energy_ratio}}` on fold 0 vs `{{physics_model_metrics.json: folds.1.coefficients.energy_ratio}}` on fold 1 vs `{{physics_model_metrics.json: folds.2.coefficients.energy_ratio}}` on fold 2.
- **`peak_depth` Coefficient:** `{{physics_model_metrics.json: folds.0.coefficients.peak_depth}}` on fold 0 vs `{{physics_model_metrics.json: folds.1.coefficients.peak_depth}}` on fold 1 vs `{{physics_model_metrics.json: folds.2.coefficients.peak_depth}}` on fold 2.
- **`peak_amplitude` Coefficient:** `{{physics_model_metrics.json: folds.0.coefficients.peak_amplitude}}` on fold 0 vs `{{physics_model_metrics.json: folds.1.coefficients.peak_amplitude}}` on fold 1 vs `{{physics_model_metrics.json: folds.2.coefficients.peak_amplitude}}` on fold 2.
- **`above_noise_area` Coefficient:** `{{physics_model_metrics.json: folds.0.coefficients.above_noise_area}}` on fold 0 vs `{{physics_model_metrics.json: folds.1.coefficients.above_noise_area}}` on fold 1 vs `{{physics_model_metrics.json: folds.2.coefficients.above_noise_area}}` on fold 2.

This instability underscores that the linear classifier over-fits idiosyncratic feature correlations of the specific two training flaws rather than learning invariant flaw physics across flaw geometries.

---

## Synthetic Results

All synthetic evaluations are conducted on simulated cylindrical titanium and nickel alloy bars (`src/utriage/physics/simulator.py`).

### Parameter Region Generalization
*Dataset: Synthetic data*

When evaluated within the training parameter distribution:
- Rules baseline detection rate: `{{sim_generalization.json: test_sets.in_distribution.rules.detection.rate}}`, false-call rate: `{{sim_generalization.json: test_sets.in_distribution.rules.false_calls.rate}}` (Synthetic data)
- ML 1D model detection rate: `{{sim_generalization.json: test_sets.in_distribution.ml_1d.detection.rate}}`, false-call rate: `{{sim_generalization.json: test_sets.in_distribution.ml_1d.false_calls.rate}}` (Synthetic data)

However, shifting to held-out operational parameter regions exposes significant failure modes:

1. **Held-Out 10 MHz Transducer Frequency (`freq_10mhz`):**
   Acoustic Rayleigh scattering increases sharply at higher frequencies, causing severe false-call blow-up:
   - Rules baseline false-call rate: `{{sim_generalization.json: test_sets.freq_10mhz.rules.false_calls.rate}}` (Synthetic data)
   - ML 1D model false-call rate: `{{sim_generalization.json: test_sets.freq_10mhz.ml_1d.false_calls.rate}}` (Synthetic data)
   - Fused union false-call rate: `{{sim_generalization.json: test_sets.freq_10mhz.union.false_calls.rate}}` (Synthetic data)
2. **Held-Out Noise Above Range (`noise_above_range`):**
   When material grain scattering exceeds nominal bounds, high background peaks breach decision gates:
   - Rules baseline false-call rate: `{{sim_generalization.json: test_sets.noise_above_range.rules.false_calls.rate}}` (Synthetic data)
   - ML 1D model false-call rate: `{{sim_generalization.json: test_sets.noise_above_range.ml_1d.false_calls.rate}}` (Synthetic data)
   - Fused union false-call rate: `{{sim_generalization.json: test_sets.noise_above_range.union.false_calls.rate}}` (Synthetic data)

---

## Fit Diagnostics

Diagnostic checks (`results/fit_diagnostics.json`) quantify the degree of model overfitting to known training flaws. The evaluation criterion flags overfitting whenever the performance gap between training and held-out test groups exceeds 0.10.

### Supervised LOCO Overfitting
Supervised models exhibit complete memorization of training flaw morphology:
- **Crack `crack_1.6` Fold:** Training detection rate `{{fit_diagnostics.json: diagnostics.loco.folds.crack_1.6.train_detection_rate}}` vs held-out detection rate `{{fit_diagnostics.json: diagnostics.loco.folds.crack_1.6.heldout_detection_rate}}` (Diagnosis: severe overfitting)
- **Crack `crack_4.0` Fold:** Training detection rate `{{fit_diagnostics.json: diagnostics.loco.folds.crack_4.0.train_detection_rate}}` vs held-out detection rate `{{fit_diagnostics.json: diagnostics.loco.folds.crack_4.0.heldout_detection_rate}}` (Diagnosis: severe overfitting)

Because the network trained on only two crack shapes, it failed to form generalizable acoustic representations of smaller cracks.

### Synthetic Supervised Shift Gaps
*Dataset: Synthetic data*

In synthetic bar experiments, out-of-distribution parameter shifts reveal large generalization deficits:
- Frequency shift (`freq_10mhz`) generalization gap: `{{fit_diagnostics.json: diagnostics.synthetic_supervised.sets.freq_10mhz.gap}}` (Synthetic data)
- Noise shift (`noise_above_range`) generalization gap: `{{fit_diagnostics.json: diagnostics.synthetic_supervised.sets.noise_above_range.gap}}` (Synthetic data)

---

## Latency

Inference benchmarks were conducted on CPU on the local development machine (`results/hardware.json`):
- **Development Processor:** `{{hardware.json: inference_machine.processor}}`
- **ML-NDT ResNet Single-Sample Latency:** Median `{{hardware.json: cpu_latency_ms.mlndt.median_ms}}` ms
- **Bar-Domain 1D Single-Sample Latency:** Median `{{hardware.json: cpu_latency_ms.bar_1d.median_ms}}` ms

*Operational Constraint:* This development machine features a multi-core high-performance CPU. These figures do not represent execution latency on the target field laptop (Intel Core i5-1135G7, 8 GB RAM). Hardware re-benchmarking on the specific plant inspection computer remains an essential prerequisite.

---

## Evidence Limits

A rigorous evidence-based assessment requires explicit disclosure of technical and data boundaries:

1. **Total Distinct Real Physical Flaws:**
   Across all accepted public ultrasonic datasets, only ten distinct physical flaws exist across three verified datasets (three physical fatigue cracks in ML-NDT, one flaw in Strathclyde-FMC, and six physical flaws in NDT_ML_Flaw). The remaining accepted datasets have unverified flaw counts.
2. **Weld Data Versus Bar Inspection:**
   All public real-data benchmarks are derived from pipe or plate welds, not aerospace round bars or billets. No public ultrasonic inspection datasets exist for rotating immersion testing of titanium or nickel superalloy bars.
3. **Unverified Simulator Constants:**
   Material acoustic velocities, attenuation curves, and grain noise parameters in `configs/materials.yaml` are handbook approximations labelled `UNVERIFIED — Ruri to check`. Acceptance thresholds in `configs/acceptance.yaml` are unverified placeholders and do not reproduce paywalled AMS-STD-2154 or ASTM E2375 tables.
4. **Advisory Independent Prototype:**
   UTriage is an advisory independent prototype. It never accepts a part; every final inspection disposition belongs exclusively to the certified Level II / Level III examiner. It operates strictly air-gapped and offline on exported scan files with no connection to UT instruments or plant machinery.

---

## Prior Art

1. **United Technologies / RTX Patent (US 9,927,404 B2):**
   "Phased array billet data evaluation software" — describes SNR-based indication classification and gating methods for ultrasonic inspection of titanium billets.
2. **Virkkunen et al. (arXiv:1903.11399):**
   "Augmented Ultrasonic Data for Machine Learning" — establishes residual neural network architectures, early max-pooling envelope extraction, and virtual flaw augmentation on the ML-NDT dataset (LGPL-3.0).
3. **Tunukovic et al. (Ultrasonics 140 (2024) 107313):**
   Approach for unsupervised acoustic anomaly detection using convolutional autoencoders trained exclusively on flaw-free ultrasonic scans, evaluating reconstruction error after interface and wall gating.
