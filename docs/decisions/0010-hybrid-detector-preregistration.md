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

## Addendum 1 (2026-10-27, before any H1/H2/H3 result exists)

### Candidate H3 (Interpretable Physics-Feature Detector)

Pre-registers candidate H3 prior to running evaluations or computing results:

- **Features**: 8 hand-crafted features computed from the raw ML-NDT image:
  1. Peak amplitude
  2. Robust noise level (MAD)
  3. SNR
  4. Peak depth index
  5. Echo duration
  6. Lateral extent
  7. Above-noise area (pixels above 6 x noise)
  8. Energy ratio within +-8 samples of the peak row
- **Model**:
  - `StandardScaler` fitted on the fold train set only.
  - `LogisticRegression` (`class_weight="balanced"`, `C=1.0`, `max_iter=1000`).
  - Decision threshold: flag at probability >= 0.5.
- **Evaluation**: Evaluated on exactly the same LOCO batch-mode folds as H1/H2.
- **Implementation**:
  - Code: `src/utriage/ml/physics_features.py`
  - Runner: `experiments/run_physics_model.py`
  - Results destination: `results/physics_model_metrics.json`

### Configuration and Pre-registered Parameters

All decision thresholds and hyper-parameters of H1-H3 will be read from `configs/detectors.yaml`. The pre-registered values are the defaults in that file; any run with changed values is labelled exploratory in the claims and is not the pre-registered result.

## Results (2026-10-27)

Pre-registered evaluations for candidates H1 (OR-union) and H2 (score fusion) from `results/hybrid_metrics.json`, and candidate H3 (physics-feature logistic regression) from `results/physics_model_metrics.json`, were executed across the 3 physical crack leave-one-crack-out (LOCO) folds.

### Per-Crack Metrics

- **Crack 1.6 mm** (4014 flawed scans, 778 unflawed scans):
  - **H1 (OR-union)**:
    - Single-best model (`loco_crack_1.6.pt`): detection 3653/4014 (rate 0.9101, 95% CI [0.9008, 0.9187]), false calls 14/778 (rate 0.0180).
    - 3-seed range: detections 3649/4014 to 3660/4014 (rate 0.9091 to 0.9118, mean 0.9103), false calls 14/778 (rate 0.0180 across all seeds).
  - **H2 (Score fusion)**:
    - Single-best model (`loco_crack_1.6.pt`): detection 3761/4014 (rate 0.9370, 95% CI [0.9290, 0.9443]), false calls 7/778 (rate 0.0090).
    - 3-seed range: detections 3687/4014 to 3766/4014 (rate 0.9185 to 0.9382, mean 0.9312), false calls 7/778 (rate 0.0090 across all seeds).
  - **H3 (Physics features)**:
    - Detection 2661/4014 (rate 0.6629, 95% CI [0.6481, 0.6776]), false calls 0/778 (rate 0.0, upper95 0.0047).

- **Crack 4.0 mm** (4078 flawed scans, 778 unflawed scans):
  - **H1 (OR-union)**:
    - Single-best model (`loco_crack_4.0.pt`): detection 1610/4078 (rate 0.3948, 95% CI [0.3798, 0.4100]), false calls 14/778 (rate 0.0180).
    - 3-seed range: detections 1306/4078 to 1610/4078 (rate 0.3203 to 0.3948, mean 0.3619), false calls 14/778 (rate 0.0180 across all seeds).
  - **H2 (Score fusion)**:
    - Single-best model (`loco_crack_4.0.pt`): detection 2272/4078 (rate 0.5571, 95% CI [0.5417, 0.5725]), false calls 7/778 (rate 0.0090).
    - 3-seed range: detections 899/4078 to 2272/4078 (rate 0.2205 to 0.5571, mean 0.4385), false calls 7/778 (rate 0.0090 across all seeds).
    - *Losing case and seed sensitivity*: Under `loco_crack_4.0_seed2.pt`, supervised detection collapsed to 0/4078 (rate 0.0). Because supervised probability was near zero, the fused score was diluted, pulling H2 detection down to 899/4078 (rate 0.2205), below the anomaly-only baseline of 1306/4078 (rate 0.3203).
  - **H3 (Physics features)**:
    - Detection 2578/4078 (rate 0.6322, 95% CI [0.6172, 0.6470]), false calls 0/778 (rate 0.0, upper95 0.0047).

- **Crack 8.6 mm** (3925 flawed scans, 778 unflawed scans):
  - **H1 (OR-union)**:
    - Single-best model (`loco_crack_8.6.pt`): detection 3925/3925 (rate 1.0, 95% CI [0.9991, 1.0]), false calls 14/778 (rate 0.0180).
    - 3-seed range: detections 3925/3925 (rate 1.0 across all seeds), false calls 14/778 (rate 0.0180 across all seeds).
  - **H2 (Score fusion)**:
    - Single-best model (`loco_crack_8.6.pt`): detection 3925/3925 (rate 1.0, 95% CI [0.9991, 1.0]), false calls 7/778 (rate 0.0090).
    - 3-seed range: detections 3925/3925 (rate 1.0 across all seeds), false calls 7/778 (rate 0.0090 across all seeds).
  - **H3 (Physics features)**:
    - Detection 3925/3925 (rate 1.0, 95% CI [0.9991, 1.0]), false calls 1/778 (rate 0.0013, upper95 0.0071).

### Observations and Scope Limits

- results are mixed and seed-sensitive (H2 on the 4.0 mm crack varies across seeds, see results/hybrid_metrics.json; H3 feature weights change sign across folds, see results/physics_model_metrics.json); no ranking is claimed.
- **H3 coefficient instability**: H3 feature weights swing across LOCO training folds (e.g. `energy_ratio` is +18.91 on crack_1.6 but -13.48 on crack_4.0; `peak_depth` is +4.23 on crack_1.6 but -5.62 on crack_4.0), showing sensitivity to the training crack sample.
- **Pre-registration boundary**: Combining H2 and H3 is a new candidate that requires a new pre-registration before it is evaluated.
- **Evidence scope**: Evaluations are restricted to one weld PAUT public dataset across 3 physical cracks.
