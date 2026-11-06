# 15 — Shift Robustness and Score Margin in Ultrasonic Flaw Triage

## 1. Context and Problem Statement

In ultrasonic non-destructive testing (UT), acoustic scans are subject to acquisition shifts across probe assemblies, pulser-receiver units, couplant layers, and inspection instruments.
When an advisory model is trained on one set of scans, changes in acquisition parameters shift incoming signal amplitudes.

If the decision threshold $t$ is calibrated on unshifted clean material and applied to shifted scans without recalibration, two failure modes can occur:
1. **Detection collapse**: Scores assigned to true flaw scans drop below $t$, missing critical defects.
2. **False alarm surge**: Background noise or clean echoes receive scores exceeding $t$, overwhelming human inspectors with false calls.

Evaluating shift robustness tests whether flaw/clean score separation survives acquisition shifts without site adaptation.

## 2. Detection Probability and the Fragility of Small Score Margins

Let $x$ denote an acoustic B-scan patch and $s(x)$ denote the score assigned by a triage model.
For an operational decision threshold $t$, detection of flawed scans is given by:

$$\text{Detection}(t) = P(s_{\text{flawed}} \ge t)$$

In terms of the score probability density function $p(s \mid \text{flawed})$:

$$P(s_{\text{flawed}} \ge t) = \int_{t}^{\infty} p(s \mid \text{flawed}) \, ds$$

### Why Small Margins Make Detection Fragile

Consider an acoustic gain shift $x' = g \cdot x$.
Because ultrasonic echo amplitudes scale with acoustic gain, a reduction in gain ($g < 1$) attenuates both the flaw reflection and background scatter.
The triage model, processing compressed amplitude profiles, shifts its score distribution downward:
$$s(g \cdot x) < s(x)$$

If unshifted flaw scores lie close to threshold $t$, even a small downward shift pushes a significant fraction of $p(s \mid \text{flawed})$ below $t$.
Consequently, $P(s_{\text{flawed}} \ge t)$ drops sharply.
A model that achieves high AUROC on unshifted scans can experience severe detection loss under shift if its separation margin above $t$ is thin.

## 3. The Log-Margin Metric

To quantify how far flaw scores sit above the operating threshold, we define the log-margin $m$:

$$m = \log_{10}(\text{median } s_{\text{flawed}}) - \log_{10}(t)$$

### Physical and Mathematical Motivation

1. **Scale compression near zero**: In neural models with sigmoid activations or distance-based anomaly models, decision thresholds $t$ at low false call rates (such as FCR 0.01) sit at small numerical values.
2. **Relative buffer**: A linear difference $\text{median}(s_{\text{flawed}}) - t$ fails to reflect robustness across differing score scales. The logarithmic difference measures order-of-magnitude separation.
3. **Robustness buffer**:
   - If $m > 0$, the median flaw score exceeds the threshold by a factor of $10^m$.
   - If $m \le 0$, more than half of the flaw scans fall below the threshold, indicating severe detection impairment.

A larger log-margin indicates a deeper buffer against signal attenuation before detections fall below the decision boundary.

## 4. Worst-Case Evaluation Across Environmental Shifts

In safety-critical ultrasonic inspection, assessing models by average detection across shift conditions is unsafe.
If a model performs well under noise and offset shifts but suffers zero detection under gain reduction, its average score may appear acceptable while concealing a critical blind spot.

Therefore, shift robustness is evaluated under the worst-case shift:

$$\text{Detection}_{\text{worst}} = \min_{s \in \text{Shifts}} \text{Detection}(s; t)$$

$$\text{FCR}_{\text{worst}} = \max_{s \in \text{Shifts}} \text{FCR}(s; t)$$

where $\text{Shifts}$ is the set of fixed environmental perturbations (`FIXED_SHIFTS`):
- `identity`: baseline unshifted reference
- `gain 0.5`: 50% signal amplitude reduction (attenuation, weak acoustic coupling)
- `gain 2.0`: signal amplitude amplification
- `offset +0.5`: DC baseline electronic drift
- `noise 0.2`: elevated microstructural grain scatter or thermal noise
- `combined`: concurrent gain reduction, DC offset, and additive noise

A candidate model is considered more robust only if its worst-case detection rate improves substantially ($\Delta \ge 0.10$) without inflating worst-case false alarms ($\le +0.01$).

## 5. Relationship to Site Calibration (ADR 0011)

In operational practice, the recommended mitigation for domain shift is unsupervised site calibration (ADR 0011):
- Using unlabelled clean calibration scans from the target inspection setup to estimate site-specific normalisation statistics.
- Recalibrating the decision threshold to enforce the desired false call rate.

The shift robustness evaluation pre-registered in ADR 0015 investigates a distinct question:
*How resilient is the learned representation itself when site calibration is absent?*
Evaluating models with fixed, unadapted thresholds measures innate representation stability, identifying architectures that degrade gracefully under uncorrected acoustic shifts.

## 6. Scope Boundaries and Benchmark Constraints

- **Synthetic perturbations**: The shifts evaluated here are synthetic mathematical transformations (gain, offset, additive noise). They model first-order physical effects but do not simulate complex wave-structure interactions or non-linear material attenuation.
- **Single specimen benchmark**: The evaluation is conducted across 3 physical cracks within a single weld specimen from the ML-NDT dataset. Findings reflect relative model behaviour within this benchmark and do not show generalisation to other components, alloys, or inspection settings.
- **No result numbers**: In adherence to pre-registration standards, no empirical result figures are stated here. All findings will be computed by code and recorded in `results/shift_robustness_metrics.json`.
