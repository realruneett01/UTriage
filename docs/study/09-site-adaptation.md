# 09 — Unsupervised Site Adaptation and Calibration

## Plant context: transfer to new inspection environments
Immersion ultrasonic testing (UT) setups vary significantly between plants, inspection tanks, and product lines:
- **Transducer differences**: Nominal probe frequency, crystal diameter, focal length in water, and bandwidth.
- **Acoustic coupling & path**: Water path length, water temperature, aeration, and bar rotation speed.
- **Electronics & hardware**: Pulser excitation voltage, analog receiver gain, filter bandwidth, and digitiser sampling rate.
- **Material variations**: Alloy composition (e.g., Ti-6Al-4V vs. nickel superalloys), grain noise scattering levels, and acoustic velocity variations across production batches.

Supervised neural networks trained on a single inspection tank or synthetic simulator data suffer from domain shift when used at a different site. In practice, confirmed defect indications are rare events, making it impractical to collect labelled flaw datasets at every new site. 

Unsupervised site adaptation addresses domain shift using **only inspector-confirmed clean reference scans** (bars confirmed free of rejectable flaws) with zero labelled defect indications.

> **Safety constraint**: UTriage is advisory-only. The conventional rules engine stays in charge. ML models can only add candidate flags, never remove or downgrade a conventional rule flag. UTriage never accepts or passes a part; all accept/reject decisions remain strictly with the certified inspector.

---

## 1. Per-site z-score normalisation

Variations in pulser-receiver gain, couplant attenuation, and material attenuation translate and scale raw ultrasonic signal amplitudes. Before feature extraction or classification, inputs must be standardised to a zero-mean, unit-variance baseline.

### Formulation
For any raw scan or image $x$, let $\mu_{\text{site}}$ and $\sigma_{\text{site}}$ be the global pixel intensity mean and standard deviation estimated across all clean reference scans from the target site:

$$x' = \frac{x - \mu_{\text{site}}}{\sigma_{\text{site}} + \epsilon}$$

where $\epsilon = 10^{-5}$ prevents division by zero if standard deviation is near zero.

### Streaming accumulation
In `fit_site_stats`, baseline statistics are computed over $N_{\text{total}}$ pixels across $M$ clean scans using running sums without loading all scans into memory:

$$\mu_{\text{site}} = \frac{\sum_{i=1}^{N_{\text{total}}} x_i}{N_{\text{total}}}, \quad \sigma^2_{\text{site}} = \max\left(0,\, \frac{\sum_{i=1}^{N_{\text{total}}} x_i^2}{N_{\text{total}}} - \mu_{\text{site}}^2\right), \quad \sigma_{\text{site}} = \sqrt{\sigma^2_{\text{site}}}$$

This centers and rescales raw scan amplitudes so input distributions match the network's nominal input scale.

---

## 2. Adaptive Batch Normalisation (AdaBN)

Even after z-score normalisation, intermediate feature representations in deep convolutional networks drift across different acoustic environments due to subtle variations in pulse shape and grain noise textures. Fine-tuning convolutional weights with backpropagation on unlabelled data risks mode collapse or catastrophic forgetting.

Adaptive Batch Normalisation (AdaBN) realigns intermediate feature activations without altering any learned convolutional filters.

### Mechanism
In standard convolutional networks, each `BatchNorm2d` layer normalises its input activations $z$:

$$\hat{z} = \frac{z - \mu_{\text{run}}}{\sqrt{\sigma^2_{\text{run}} + \epsilon_{\text{bn}}}} \cdot \gamma + \beta$$

where $\gamma$ and $\beta$ are learned affine parameters, and $\mu_{\text{run}}, \sigma^2_{\text{run}}$ are layer-wise running statistics.

In AdaBN (`recalibrate_batchnorm`):
1. **Weights and affine parameters frozen**: All convolutional weights, biases, and BatchNorm affine parameters ($\gamma, \beta$) remain strictly fixed.
2. **Reset running statistics**: The model is set to evaluation mode (`model.eval()`), while BatchNorm modules are switched to training mode (`bn.train()`) with running statistics cleared (`bn.reset_running_stats()`) and momentum set to cumulative accumulation (`bn.momentum = None`).
3. **Forward calibration pass**: Unlabelled clean reference scans from the new site are streamed through the model in forward passes under `torch.no_grad()`.
4. **Site-specific statistics**: Each layer's running mean $\mu_{\text{run}}$ and variance $\sigma^2_{\text{run}}$ are recalculated to match the target site's empirical activation distribution.
5. **Freeze in evaluation mode**: The model is returned to `model.eval()`.

This aligns latent representations to the target site's acoustic characteristics without modifying trained feature representations.

> [!NOTE]
> In `experiments/run_site_adaptation.py` the evaluated model (`ConvAE2D`) has no BatchNorm layers, so `recalibrate_batchnorm` made no change (the adapted model sha256 in `results/site_adaptation.json` is identical across shifts). The measured improvements come from per-site z-score normalisation and quantile threshold recalibration only. AdaBN is implemented and unit-tested but not evaluated in this experiment.

---

## 3. Quantile threshold for a target false-call rate

Rather than selecting an arbitrary anomaly score cutoff, UTriage calibrates decision thresholds against an explicit false-call rate budget.

### Formulation
An anomaly detector or hybrid model assigns an anomaly score $s \in \mathbb{R}$ to each indication. Given $n$ clean calibration scores $\{s_1, s_2, \dots, s_n\}$ evaluated on confirmed flaw-free scans, and a target false-call rate $\alpha \in (0, 1)$ (e.g., $\alpha = 0.01$ for a 1% false-call budget), the empirical threshold $\theta$ is defined as the $(1 - \alpha)$ sample quantile:

$$\theta = Q_{1 - \alpha}\left(\{s_i\}_{i=1}^n\right)$$

An indication with score $s$ receives an advisory anomaly flag if:

$$s > \theta$$

By construction on the clean calibration sample:

$$P(s > \theta \mid \text{clean}) \approx \alpha$$

This provides an operationally controlled trade-off between detection sensitivity and the rate of benign indications flagged for human review.

---

## 4. One-sided 95% Clopper-Pearson upper bound

The empirical false-call rate measured on calibration scans is subject to finite-sample variability. To bound the false-call rate statistically (an estimate on the supplied clean scans, not a certification), UTriage calculates an exact one-sided 95% Clopper-Pearson upper confidence bound.

### Formulation
Let $n$ be the number of clean calibration scans evaluated, and let $k = \sum_{i=1}^n \mathbf{1}(s_i > \theta)$ be the observed count of false calls. Modeling false calls as independent Bernoulli trials with true rate $p$:

$$K \sim \text{Binomial}(n, p)$$

The one-sided $(1 - \gamma)$ upper confidence limit $p_{\text{upper}}$ (for 95% confidence, $\gamma = 0.05$) satisfies:

$$\sum_{j=0}^k \binom{n}{j} p_{\text{upper}}^j (1 - p_{\text{upper}})^{n - j} = \gamma = 0.05$$

Using the relationship between the cumulative binomial distribution and the incomplete beta function, this upper bound is computed via the Beta distribution quantile (`scipy.stats.beta.ppf`):

$$p_{\text{upper}} = \begin{cases} I^{-1}_{0.95}(k + 1, n - k) = \text{BetaQuantile}(0.95;\, k + 1,\, n - k), & \text{if } k < n \\ 1.0, & \text{if } k \ge n \end{cases}$$

This provides a 95% statistical upper bound ($p_{\text{upper}}$) on the true false-call rate in practice under the independent Bernoulli assumption on the supplied clean scans. Note that upper95_fcr is computed on calibration clean scans; test-set false calls are reported separately.

---

## 5. Zero-failure sample size and the Rule of Three

### Required calibration sample size
To bound the false-call rate at target $p$ with confidence $c$ (e.g., $c = 0.95$) assuming zero false calls are observed ($k = 0$), how many clean scans $n$ must be inspected?

Under true rate $p$, the probability of observing zero false calls in $n$ independent clean scans is:

$$P(K = 0) = (1 - p)^n$$

To ensure that this probability is at most $1 - c$:

$$(1 - p)^n \le 1 - c$$

Taking natural logarithms on both sides (noting $\ln(1 - p) < 0$):

$$n \ln(1 - p) \le \ln(1 - c) \implies n \ge \frac{\ln(1 - c)}{\ln(1 - p)}$$

Because $n$ must be an integer, the minimum sample size is:

$$n = \left\lceil \frac{\ln(1 - c)}{\ln(1 - p)} \right\rceil$$

### The Rule of Three
For small false-call rates $p \ll 1$, the first-order Taylor expansion yields $\ln(1 - p) \approx -p$. For a 95% confidence level ($c = 0.95$):

$$\ln(1 - c) = \ln(0.05) \approx -2.99573 \approx -3$$

Substituting into the sample size formula:

$$n \approx \frac{-3}{-p} = \frac{3}{p} \iff p \approx \frac{3}{n}$$

The Rule of Three states that if zero events occur in $n$ trials, the 95% upper confidence bound on the event rate is approximately $3/n$. If the number of calibration scans is less than $n_{\text{min}}$, UTriage raises a `UserWarning` alerting the operator of insufficient sample size.

---

## 6. Worked example

All numbers below are derived directly from the analytical formulas:

### Example 1: Target false-call rate $p = 0.01$ (1%), confidence $c = 0.95$
1. Exact zero-failure sample size:
   $$\ln(1 - c) = \ln(0.05) \approx -2.995732$$
   $$\ln(1 - p) = \ln(0.99) \approx -0.0100503$$
   $$n = \left\lceil \frac{-2.995732}{-0.0100503} \right\rceil = \lceil 298.073 \rceil = 299$$
2. Rule of three approximation:
   $$n \approx \frac{3}{0.01} = 300 \quad (\approx 299)$$
3. One-sided 95% Clopper-Pearson upper bound for $k = 0$ false calls in $n = 299$ scans:
   - Exact:
     $$p_{\text{upper}} = 1 - (1 - c)^{1/n} = 1 - (0.05)^{1/299} \approx 1 - 0.990031 = 0.009969 \le 0.01$$
   - Rule-of-three approximation ($e^{-3/n}$ with $n = 300$):
     $$p_{\text{upper}} \approx 1 - e^{-3/n} = 1 - e^{-3/300} \approx 1 - 0.990050 = 0.009950 \le 0.01$$

### Example 2: Target false-call rate $p = 0.05$ (5%), confidence $c = 0.95$
1. Exact zero-failure sample size:
   $$\ln(1 - p) = \ln(0.95) \approx -0.051293$$
   $$n = \left\lceil \frac{-2.995732}{-0.051293} \right\rceil = \lceil 58.404 \rceil = 59$$
2. Rule of three approximation:
   $$n \approx \frac{3}{0.05} = 60 \quad (\approx 59)$$

### Example 3: Target false-call rate $p = 0.10$ (10%), confidence $c = 0.95$
1. Exact zero-failure sample size:
   $$\ln(1 - p) = \ln(0.90) \approx -0.105361$$
   $$n = \left\lceil \frac{-2.995732}{-0.105361} \right\rceil = \lceil 28.433 \rceil = 29$$
2. Rule of three approximation:
   $$n \approx \frac{3}{0.10} = 30 \quad (\approx 29)$$

---

## 7. Structure of the site profile JSON

The calibrated parameters and metadata are stored in a persistent JSON file (`SiteProfile`) loaded via `load_site_profile` and serialized via `save_site_profile`.

```json
{
  "created": "2026-11-04T16:00:00+00:00",
  "label": "Synthetic",
  "advisory_only": true,
  "disclaimer": "Advisory only. UTriage never accepts or passes a part.",
  "stats": {
    "mean": 0.0421,
    "std": 0.1254,
    "n_scans": 300,
    "label": "Synthetic"
  },
  "calib": {
    "threshold": 0.0815,
    "false_call_rate": 0.01,
    "n_clean": 300,
    "upper95_fcr": 0.00995
  }
}
```

### Field definitions
- `created`: ISO 8601 UTC timestamp of calibration.
- `label`: Provenance label; strictly restricted to `"Synthetic"` or `"Real"`.
- `advisory_only`: Boolean flag set to `true`, enforcing that ML inferences remain advisory.
- `disclaimer`: Required statement clarifying that UTriage never accepts or passes inspected parts.
- `stats.mean`, `stats.std`: Global image normalisation parameters $\mu_{\text{site}}$ and $\sigma_{\text{site}}$.
- `stats.n_scans`: Number of clean reference scans used to estimate baseline statistics.
- `calib.threshold`: Empirical $(1 - \alpha)$ quantile decision threshold $\theta$.
- `calib.false_call_rate`: Target false-call rate budget $\alpha$.
- `calib.n_clean`: Number of clean calibration scans evaluated.
- `calib.upper95_fcr`: One-sided 95% Clopper-Pearson upper confidence limit on the site false-call rate (computed on calibration clean scans; test-set false calls are reported separately).

---

## 8. Limitations and safety boundaries

- **Shifts covered**: only the synthetic receiver gain, DC offset and additive noise perturbations in experiments/run_site_adaptation.py were tested. Sound velocity, coupling, temperature and material changes were not tested. Tested shifts are synthetic software perturbations of real ML-NDT weld images, not real plant/site changes.
- **Model architecture and AdaBN**: in experiments/run_site_adaptation.py the evaluated model (ConvAE2D) has no BatchNorm layers, so recalibrate_batchnorm made no change (the adapted model sha256 in results/site_adaptation.json is identical across shifts). The measured improvements come from per-site z-score normalisation and quantile threshold recalibration only. AdaBN is implemented and unit-tested but not evaluated in this experiment.
- **Cannot learn new flaw types**: Unsupervised adaptation uses only flaw-free clean scans. It does not train on defect indications and cannot learn the acoustic signatures or morphology of novel flaw mechanisms.
- **Requires inspector-confirmed clean scans**: If calibration scans inadvertently contain unflagged flaws, baseline statistics become distorted and the anomaly threshold is artificially elevated, desensitising the detector.
- **Rules engine stays in charge**: Safety-first architecture (ADR 0001). ML models can only raise advisory flags; they can never remove or downgrade a conventional rule flag.
- **No plant validation exists**: UTriage has not been validated or deployed on production plant lines or MIDHANI hardware.
- **Measured results**: See results/site_adaptation.json and the generated tables in README.md; results are single-seed, on real ML-NDT weld images (site adaptation uses synthetic gain/offset/noise perturbations).

---

## 9. Self-check questions for Ruri

1. **Why does AdaBN freeze convolutional weights and affine parameters while recomputing running mean and variance?**
   *Hint*: Consider what would happen if gradient descent were run on unlabelled scans versus updating only the running activation statistics.
2. **If an inspector provides 100 clean scans for calibration at a target false-call rate of 1%, why does UTriage emit a warning, and what is the minimum required number of clean scans?**
   *Hint*: Calculate $n = \lceil \ln(0.05)/\ln(0.99) \rceil$ and compare it to 100.
3. **How does the Rule of Three approximate the exact Clopper-Pearson sample size formula when $p$ is small?**
   *Hint*: Use the first-order Taylor expansion of $\ln(1 - p)$ around $p = 0$.
