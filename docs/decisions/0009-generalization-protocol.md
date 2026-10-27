# 0009. Generalization evaluation protocol (Phase 8)

- Status: Accepted
- Date: 2026-10-27

## Context

Public ultrasonic testing (UT) data is scarce across non-destructive testing (NDT) literature. As documented in the PNNL/NRC review ML22284A071 (cited in `docs/phase8-spec.md`), most published studies evaluate small datasets with few specimens, a single material, and a single transducer frequency, leading to poor generalization and an absence of common public benchmarks.

In earlier development phases, the ML-NDT dataset evaluation used a random batch split. A subsequent audit documented in `results/leakage_audit.json` revealed that this random batch split leaked all 3 physical cracks across both training and test partitions. Consequently, past performance metrics reflected memorization of those specific crack geometries rather than true generalization to unseen flaws.

A rigorous evaluation protocol is required to ensure that reported generalization metrics reflect genuine out-of-distribution performance, comply with project safety design (ADR 0001), and maintain strict honesty regarding empirical limitations.

## Decision

To prevent data leakage and evaluate true generalization, the following protocol is established for all model training, threshold calibration, and evaluation:

- **Group-held-out evaluation on real data**:
  - Every real-data performance result must use group-held-out splits rather than random partitioning.
  - For physical cracks, leave-one-crack-out (LOCO) evaluation is mandatory: models are trained on remaining cracks and evaluated strictly on the held-out crack.
  - For external datasets, cross-dataset evaluation is strictly test-only; external real datasets are never included in training splits.
  - For the physics simulator, evaluation holds out entire parameter regions (transducer frequency, unseen material grades, and high acoustic noise regimes).
  - Random splits never count as generalization.

- **Honest early stopping protocol**:
  - Training early stopping uses batch-held-out validation data that excludes the held-out crack.
  - Documentation and reporting must state honestly that early stopping is batch-held-out, not crack-held-out.

- **Calibrated thresholding and anomaly detection**:
  - Decision thresholds are calibrated strictly on clean calibration data.
  - Anomaly detection models are trained exclusively on flaw-free scans.

- **Claim auditing and scope limits**:
  - All quantitative claims are exported strictly via `results/claims.json`.
  - Every claim must specify an explicit scope limit and pass automated forbidden-phrase auditing.
  - Forbidden phrases (such as claims of plant qualification or absolute certainty) are strictly prohibited.

- **Union property preservation**:
  - In accordance with ADR 0001, machine learning advisory outputs can only add candidate flags and cannot remove conventional rule flags.
  - The union property (ML only adds flags) is verified per evaluation column in `results/generalization_matrix.json`.

## Alternatives Considered

- **Random splits**: Rejected because random partitioning over correlated scans leaks source crack geometry between training and test sets (`results/leakage_audit.json`).
- **K-fold cross-validation over images**: Rejected because images sampled from the same specimen contain the same 3 physical cracks, reproducing data leakage across folds.
- **Training on NDT_ML_Flaw**: Rejected because NDT_ML_Flaw serves as the only independent public real test set for cross-dataset evaluation. Using it for training would eliminate external validation.

## Non-Attempted Spec Items

The following items from `docs/phase8-spec.md` were not implemented:
- **Step 3.1 joint training on simulated bar windows plus real weld B-scans**: Not attempted due to incompatible input domains (1D bar A-scan windows vs 2D weld PAUT images).
- **Step 3.3 optional self-supervised pretraining**: Not attempted; reserved unless cross-dataset transfer demonstrates clear benefit.

## Consequences

- **Superseded results**: Earlier perfect-looking test-split scores obtained under random splits are invalidated and superseded.
- **Small evidence base**: The real-data evidence base remains strictly bounded, as inventoried in `results/data_inventory.md`.
- **Model memorization documented**: Supervised models exhibit memorization of known cracks, as detailed in `results/fit_diagnostics.json`.
- **Cross-dataset domain gap**: Cross-dataset transfer to independent real test data failed under the documented feature mapping (`results/cross_dataset_metrics.json`), demonstrating that domain adaptation across differing UT inspection modalities remains unsolved.
- **Safety architecture maintained**: The rules engine remains the primary baseline authority, with ML acting solely in an advisory capacity.

## Update (2026-10-27): Crack-Held-Out Early-Stopping Mode

A crack-held-out early-stopping mode now exists (`experiments/run_loco.py --val-mode crack`, `results/loco_crackval_metrics.json`). In this mode, early stopping validates on the second remaining crack rather than a held-out batch of the training crack:

- **Fold crack_1.6** (val crack_4.0): detection 4012/4014 (rate 0.9995), false calls 753/778 (rate 0.9679, upper95 0.9776), a90/95 = 0.953 mm.
- **Fold crack_4.0** (val crack_8.6): detection 2449/4078 (rate 0.6005), false calls 0/778 (rate 0.0, upper95 0.0038), a90/95 = 3.306 mm.
- **Fold crack_8.6** (val crack_1.6): detection 3925/3925 (rate 1.0), false calls 11/778 (rate 0.0141, upper95 0.0233), separable data with Clopper-Pearson lower bound 0.9992 (a90/95 null).

Per-fold outcomes show that results swing strongly with the choice of early-stopping crack: crack_1.6 achieves high detection (4012/4014) but false calls rise sharply to 753/778, whereas crack_4.0 detection drops to 2449/4078 with 0/778 false calls, and crack_8.6 achieves 3925/3925 detection with 11/778 false calls. Batch mode stays the default for comparability.

