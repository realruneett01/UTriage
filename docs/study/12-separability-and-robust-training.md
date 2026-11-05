# 12 — Flaw Separability Metrics and Robust Training Physics

## 1. Context and Problem Statement
In ultrasonic testing (UT) of aerospace components, classifying internal indications from acoustic scans requires discriminating true defect reflections from background microstructure grain scatter and electronic noise.
Earlier leave-one-crack-out (LOCO) runs with crack-held-out early stopping (`results/loco_crackval_metrics.json`) revealed that tuning models directly against an unseen crack resulted in degenerate operating points: near-perfect detection paired with overwhelming false call rates.
To address this, we define pre-registered separability metrics (AUROC and Average Precision) and physics-informed data augmentations, evaluated under strict clean-calibrated thresholds.

## 2. Separability Metrics: AUROC and Average Precision

### Area Under the ROC Curve (AUROC)
The Receiver Operating Characteristic (ROC) curve maps true positive rate (detection rate) against false positive rate (false call rate) across all possible decision thresholds.
The AUROC possesses a direct probabilistic rank interpretation: it is the probability that a randomly chosen flawed scan receives a higher anomaly or defect score than a randomly chosen clean scan:

$$AUROC = P(s_{\text{flawed}} > s_{\text{clean}})$$

Equivalently, AUROC is calculated from the Mann-Whitney $U$ statistic:

$$U = \sum_{i=1}^{n_{\text{flawed}}} \sum_{j=1}^{n_{\text{clean}}} \left[ \mathbb{I}(s_{\text{flawed}, i} > s_{\text{clean}, j}) + \frac{1}{2}\mathbb{I}(s_{\text{flawed}, i} = s_{\text{clean}, j}) \right]$$

$$AUROC = \frac{U}{n_{\text{flawed}} \cdot n_{\text{clean}}}$$

where $\mathbb{I}(\cdot)$ is the indicator function.
Unlike single-threshold classification metrics, AUROC measures global score separability across the entire spectrum of operating thresholds without fixing an arbitrary decision boundary.

### Average Precision (AP)
Average Precision summarizes the precision-recall curve by computing the weighted mean of precisions across recall increments:

$$AP = \sum_{k} (R_k - R_{k-1}) P_k = \int_0^1 p(r) \, dr$$

In non-destructive evaluation where flaws are rare anomalies among abundant sound material, class distributions are heavily imbalanced. While AUROC weights clean and flawed scans equally in ranking, AP focuses on positive predictive value across all recall levels, penalizing false alarms even under severe class imbalance.

## 3. Clean-Calibrated Thresholding and Disjoint Partitioning

### Setting Thresholds on Clean Scans Only
In non-destructive testing, inspection workflows cannot tolerate unconstrained false alarms. The operational decision threshold $\tau$ must be set relative to a target False Call Rate (FCR) $\alpha$ (e.g., $\alpha = 0.01$ or $\alpha = 0.05$) using only clean scans:

$$\tau = Q_{1 - \alpha}(\{s_{\text{clean, cal}}\})$$

where $Q_{1 - \alpha}$ represents the $(1 - \alpha)$ empirical quantile computed over clean calibration scans. Flaw scans are never used to set or tune this threshold.

### Disjoint Clean Evaluation Half
If the empirical false call rate were measured on the same clean scans used to calculate $\tau$, the resulting estimate would be optimistically biased due to sample overfitting.
To obtain an unbiased assessment:
1. Available clean scans are split into two disjoint halves: a clean calibration set and a clean evaluation set.
2. The calibration half is used solely to compute $\tau = Q_{1 - \alpha}(\{s_{\text{clean, cal}}\})$.
3. The evaluation half is used solely to assess the empirical false call rate:

$$\hat{\alpha} = \frac{1}{N_{\text{clean, eval}}} \sum_{i=1}^{N_{\text{clean, eval}}} \mathbb{I}(s_{\text{clean, eval}, i} \ge \tau)$$

4. Detection hit rate is evaluated on the held-out flawed scans using this clean-calibrated threshold $\tau$.

## 4. Clopper-Pearson Exact Confidence Intervals
For any observed binomial count $k$ out of $n$ trials (such as detection hits $k_{\text{hit}} / n_{\text{flawed}}$ or false calls $k_{\text{false}} / n_{\text{clean}}$), standard normal approximation intervals break down near 0 and 1.
The Clopper-Pearson method computes exact confidence limits for binomial parameter $p$ at confidence level $1 - \gamma$ (typically 95%, with $\gamma = 0.05$) by inverting the cumulative binomial distribution:

$$p_{\text{lower}} = B\left(\frac{\gamma}{2};\, k,\, n - k + 1\right)$$

$$p_{\text{upper}} = B\left(1 - \frac{\gamma}{2};\, k + 1,\, n - k\right)$$

where $B(q; a, b)$ is the $q$-quantile of the Beta distribution with shape parameters $a$ and $b$.
For zero observed events ($k = 0$), $p_{\text{lower}} = 0$ and $p_{\text{upper}} = 1 - (\gamma/2)^{1/n}$.

## 5. Degenerate Operating Points in Early Stopping
In `results/loco_crackval_metrics.json`, early stopping tuned directly on a held-out crack (crack 1.6 mm) yielded a detection rate of 99.95% (4012/4014), but simultaneously flagged 96.79% (753/778) of clean scans.
This is a degenerate operating point. A trivial detector predicting "flawed" for every scan achieves 100% detection while rendering the inspection system useless, as human inspectors would be forced to re-examine nearly every sound part.
Model selection and early stopping must never be driven by held-out crack metrics that compromise false call control. Detection rates must be evaluated at controlled, calibrated false call rates ($\alpha = 0.01$ and $\alpha = 0.05$).

## 6. Physics of Data Augmentations
Ultrasonic signals acquired across varying scan sessions, probe assemblies, and specimen positions experience physical variations. We model these variations with five physically grounded transformations:

### 1. Gain Variation ($g$)
Changes in pulser-receiver amplifier gain, transducer wear, couplant layer thickness, and surface contact pressure scale echo amplitudes:
$$x_{\text{scaled}} = g \cdot x, \quad g \sim \mathcal{U}(1 - \Delta_g, 1 + \Delta_g)$$

### 2. DC Offset Drift ($b$)
Baseline shifts arising from pre-amplifier thermal drift and analog-to-digital converter (ADC) bias:
$$x_{\text{offset}} = x + b, \quad b \sim \mathcal{U}(-\Delta_b, \Delta_b)$$

### 3. Electronic and Grain Scatter Noise ($n$)
High-frequency thermal noise from receiver electronics and backscatter from alloy microstructural grain boundaries:
$$x_{\text{noisy}} = x + n, \quad n \sim \mathcal{N}(0, \sigma_n^2)$$

Combined amplitude transformation:
$$x' = g \cdot x + b + n$$

### 4. Time Shift ($d$)
Variations in water path length in immersion tanks, couplant column height, or sound velocity changes due to temperature and material composition shift wave arrival times:
$$x'(t) = x(t - d)$$
where travel time delay relates to path variation $\Delta z$ and sound velocity $c$ by $\Delta t = 2 \Delta z / c$.

### 5. Scan Flip
Raster scanning alternates probe travel direction across consecutive passes. Flipping the scan axis accounts for probe travel reversal:
$$x_{\text{flipped}}(u, t) = x(L - u, t)$$
where $u$ is the lateral probe coordinate across scan aperture length $L$.

## 7. Bounded Evidence Base and Generalisation Limits
The ML-NDT benchmark contains data from only 3 physical cracks within a single austenitic steel weld specimen.
With only 3 physical cracks:
- No machine learning technique, training protocol, or augmentation scheme can claim or demonstrate generalisation to unseen flaw morphologies, differing crack orientations, volumetric voids, inclusions, or alternative alloy grades.
- The evaluation measures cross-crack ranking stability strictly within this bounded benchmark.
- All conclusions remain subject to this fundamental empirical constraint.
