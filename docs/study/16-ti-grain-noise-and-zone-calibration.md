# 16 — Titanium Grain Noise and Zone-Wise Reference Calibration

> All models and signals described in this note are **Synthetic data**.
> Parameter values from `configs/ti_mode.yaml` are **placeholders** (`UNVERIFIED - Ruri to check`).
> This note presents plain language, governing equations, and a symbolic worked example; no empirical result numbers are presented.

## 1. Problem Statement: Macrozones and Global Threshold Breakdown

In ultrasonic testing (UT) of rotating round bars and billets, titanium alloys present distinct acoustic challenges compared to fine-grained steels or nickel superalloys. During thermomechanical processing, titanium alloys can develop macrozones: extended regions of similarly oriented crystallographic grains that share an acoustic orientation.

These macrozones cause heterogeneous acoustic scattering:
- Sound traveling through a macrozone encounters coherent grain backscatter, producing elevated noise amplitude patches.
- Adjacent regions with different grain orientations exhibit significantly lower background noise.
- Acoustic attenuation and beam divergence cause noise and flaw amplitudes to vary with depth below the entry surface.

### Why a Single Global Threshold Fails

If an inspector or automated system applies a single global amplitude threshold across the entire volume of a titanium billet:
1. **Blindness in quiet zones**: If the threshold is set high enough to clear peak noise in loud macrozone patches, it becomes insensitive in low-noise regions. True defects with modest echo heights are missed.
2. **False alarm surge in loud zones**: If the threshold is set low enough to detect small reflectors in low-noise regions, noise peaks in loud macrozone patches cross the threshold continuously, creating an unmanageable flood of false calls.

Because noise levels vary both spatially across macrozone patches and radially with depth, reliable flaw triage requires dividing the depth range into discrete zones and calibrating each zone independently.

## 2. Structured Grain Noise Model

To simulate titanium inspection without physical plant data, UTriage-Ti introduces a structured grain noise model with three components:
1. Depth-dependent base noise levels.
2. A spatially correlated lognormal strength field representing macrozones.
3. Convolution with the ultrasonic pulse.

### The Lognormal Strength Field

Let $g(x, \theta, z)$ be a standard Gaussian random field with zero mean and unit variance:
$$\mathbb{E}[g] = 0, \quad \text{Var}(g) = 1$$

The local noise strength multiplier $m(x, \theta, z)$ is defined as:
$$m(x, \theta, z) = \exp\left(s \cdot g(x, \theta, z) - \frac{s^2}{2}\right)$$

where $s$ is the log-amplitude standard deviation parameter (`field_sigma: 0.1`, placeholder, UNVERIFIED - Ruri to check).

### Proof that the Multiplier Field has Mean 1

For a standard normal variable $g \sim \mathcal{N}(0, 1)$, its moment-generating function is:
$$\mathbb{E}[\exp(t g)] = \exp\left(\frac{t^2}{2}\right)$$

Setting $t = s$:
$$\mathbb{E}[\exp(s g)] = \exp\left(\frac{s^2}{2}\right)$$

Multiplying by the scalar factor $\exp(-s^2 / 2)$:
$$\mathbb{E}[m] = \mathbb{E}\left[\exp\left(s g - \frac{s^2}{2}\right)\right] = \exp\left(-\frac{s^2}{2}\right) \mathbb{E}[\exp(s g)] = \exp\left(-\frac{s^2}{2}\right) \exp\left(\frac{s^2}{2}\right) = 1$$

Because $\mathbb{E}[m] = 1$, the spatial field $m$ redistributes noise amplitude into louder and quieter patches (representing macrozones) without altering the nominal volume-averaged noise level.

### Gaussian Spatial Correlation

Macrozones are continuous physical structures extending across multiple millimeters. Adjacent scan lines therefore encounter correlated scattering.

The spatial correlation between two points separated by axial distance $\Delta x$ and circumferential angular distance $\Delta \theta$ follows a Gaussian covariance kernel:
$$K(\Delta x, \Delta \theta) = \exp\left( -\frac{\Delta x^2}{2 L_x^2} - \frac{\Delta \theta^2}{2 L_\theta^2} \right)$$

where:
- $L_x$ is the axial correlation length (`corr_axial_mm: 6.0`, placeholder, UNVERIFIED - Ruri to check).
- $L_\theta$ is the angular correlation length (`corr_angle_deg: 15.0`, placeholder, UNVERIFIED - Ruri to check).

Because the bar is cylindrical and rotates during immersion scanning, angular separation wraps periodically at 360 degrees:
$$\Delta \theta = \min\bigl(|\theta_1 - \theta_2|,\; 360^\circ - |\theta_1 - \theta_2|\bigr)$$

### Depth Zones and Literature Background

Depth-dependent noise scaling factors (`zone_levels: [1.0, 1.2, 1.1]`, placeholder, UNVERIFIED - Ruri to check) are assigned across radial depth boundaries (`zone_edges_mm: [2.0, 12.0, 22.0, 36.0]`, placeholder, UNVERIFIED - Ruri to check).

This phenomenological model draws on research by Margetan, Thompson and co-workers at Iowa State on titanium grain noise and the noise figure-of-merit (cite as 'Iowa State QNDE grain-noise papers, exact references UNVERIFIED - Ruri to check'). All simulated outputs are labelled **Synthetic data**.

## 3. Zone Gates and Reference Calibration

Rather than evaluating the entire metal path in a single time gate, the sound path is partitioned into radial depth zones $[z_k, z_{k+1}]$:
- **Surface dead zone** $[0, z_0)$: covers entry surface ring-down where flaw detection is not evaluated.
- **Depth zones** $[z_0, z_1], [z_1, z_2], \dots, [z_{K-1}, z_K]$: radial metal depth slices from near-surface to deep interior.

### Reference Block Calibration

Calibration uses a simulated reference standard block sharing the bar geometry and material sound velocity. The reference block contains one flat-bottom hole (FBH) per depth zone, drilled to reference diameter `ref_fbh_mm: 0.8` mm (placeholder, UNVERIFIED - Ruri to check).

Scanning the reference block provides zone-specific calibration quantities:
- Reference echo response $A$ for each zone.
- Background grain noise floor $B$ for each zone.
- Peak grain noise excursion $C$ for each zone.

## 4. Zone Signal-to-Noise Ratio (SNR)

For each depth zone, the signal-to-noise ratio is calculated as:
$$\text{SNR}_{\text{zone}} = \frac{A - B}{C - B}$$

### Definitions of Terms

- **$A$ (Reference Signal Response)**: The mean amplitude of the $3 \times 3$ C-scan pixels centred on the peak response of the reference hole (`signal_window_px: 1`, so N = 1, placeholder, UNVERIFIED - Ruri to check). Averaging over a larger window (as in US 6,401,537) suits a finer pixel grid than this simulator's.
- **$B$ (Mean Noise Floor)**: The mean envelope amplitude within the zone gate, measured across the reference block noise region while excluding pixels within `noise_exclude_px: 6` (placeholder, UNVERIFIED - Ruri to check) of the reference hole.
- **$C$ (Peak Noise Envelope)**: The maximum envelope amplitude observed within that same zone noise region.

### Mathematical Meaning and Prior Art

In conventional UT, SNR is often estimated as peak signal divided by root-mean-square noise ($A / n_{\text{rms}}$). In titanium billet inspection, $(A - B) / (C - B)$ measures the net flaw echo above the average noise background $(A - B)$ relative to the maximum noise excursion above that background $(C - B)$.

This specific form is published in patents US 6,370,956 and US 6,401,537 (titanium ultrasonic noise); whether a given specification uses exactly this form is UNVERIFIED - Ruri to check.

## 5. Alarm Level and the Capping Rule

The zone alarm threshold is determined by balancing noise clearance against flaw sensitivity:
$$\text{Alarm} = \min\bigl(C \times (1 + \text{noise\_margin}),\; \text{cap\_fraction} \times A\bigr)$$

where:
- `noise_margin: 0.10` is a placeholder (UNVERIFIED - Ruri to check).
- `cap_fraction: 0.70` is a placeholder taken from a secondary-source snippet attributed to ASTM E2375 that was not read in the original - UNVERIFIED - Ruri to check.

### Why the Cap Exists

The primary objective of calibration is detecting flaws that reflect sound with amplitude comparable to the reference hole $A$.

Consider what happens if no cap exists:
- If a noisy zone has a large peak noise $C$, an uncapped rule $C \times (1 + \text{noise\_margin})$ could place the alarm threshold above $A$ (e.g. at $1.2 A$ or $1.5 A$).
- An alarm threshold set higher than $A$ means a rejectable defect producing echo amplitude $1.0 A$ would never trigger an alarm.
- The inspection would be desensitized to rejectable flaws.

To prevent this hazard, governing specifications cap the alarm at a fraction of the reference echo: $\text{cap\_fraction} \times A$.

## 6. Worked Symbolic Example: The Inspectability Gate

To understand how the alarm rule operates without using fabricated result numbers, let us examine the mathematics symbolically.

### Case 1: Low-to-Moderate Noise (Normal Inspection)

Suppose the grain noise peak $C$ satisfies:
$$C \times (1 + \text{noise\_margin}) \le \text{cap\_fraction} \times A$$

Because the noise term is smaller, the minimum selects the noise-based threshold:
$$\text{Alarm} = C \times (1 + \text{noise\_margin})$$

In this regime:
1. Since $\text{noise\_margin} > 0$, we have $\text{Alarm} > C$. The threshold sits comfortably above peak noise $C$, suppressing false calls from grain noise.
2. Because $\text{Alarm} \le \text{cap\_fraction} \times A < A$, any flaw reflecting at reference strength $A$ crosses the alarm threshold.
3. The zone is inspectable and operates at target sensitivity.

### Case 2: High Noise (The Cap Binds)

Now suppose microstructural scattering is elevated (such as within an intense macrozone) so that:
$$C \times (1 + \text{noise\_margin}) > \text{cap\_fraction} \times A$$

Here the capping rule binds. The alarm threshold cannot exceed the cap:
$$\text{Alarm} = \text{cap\_fraction} \times A$$

Now examine where the alarm threshold sits relative to the peak noise $C$:
- Dividing the inequality by $(1 + \text{noise\_margin})$:
  $$C > \frac{\text{cap\_fraction} \times A}{1 + \text{noise\_margin}}$$
- Because $\text{noise\_margin} > 0$, the denominator exceeds 1.
- If noise is sufficiently severe, $\text{cap\_fraction} \times A \le C$.
- Under this condition, the alarm threshold satisfies:
  $$\text{Alarm} \le C$$

### Consequence: The Inspectability Gate

When $\text{Alarm} \le C$, the alarm threshold sits directly inside the background noise envelope.
- If inspection proceeded, background grain noise peaks would cross the threshold repeatedly, causing false calls to surge.
- Yet the threshold cannot be raised without violating the sensitivity cap $\text{cap\_fraction} \times A$ and missing reference-sized flaws.

Under the UTriage-Ti safety architecture:
1. When $\text{Alarm} \le C$, or when $\text{SNR}_{\text{zone}} < \text{min\_snr}$ (`min_snr: 2.0`, placeholder, UNVERIFIED - Ruri to check), the zone is flagged:
   'not inspectable at this sensitivity'
2. The scan is presented to the human inspector with this explicit flag for manual evaluation or alternative testing.
3. The zone is never silently passed.
4. Advisory ML models can add candidate indications but cannot override or clear the rules-engine inspectability flag.

## 7. Configuration Placeholders and Open Items

All configuration parameters from `configs/ti_mode.yaml` are summarized below:

| Parameter | Current Value | Status / Source |
|---|---|---|
| `corr_axial_mm` | 6.0 | placeholder, UNVERIFIED - Ruri to check |
| `corr_angle_deg` | 15.0 | placeholder, UNVERIFIED - Ruri to check |
| `field_sigma` | 0.1 | placeholder, UNVERIFIED - Ruri to check |
| `zone_edges_mm` | [2.0, 12.0, 22.0, 36.0] | placeholder, UNVERIFIED - Ruri to check |
| `zone_levels` | [1.0, 1.2, 1.1] | placeholder, UNVERIFIED - Ruri to check |
| `ref_fbh_mm` | 0.8 | placeholder, UNVERIFIED - Ruri to check |
| `signal_window_px` | 1 | placeholder, UNVERIFIED - Ruri to check |
| `noise_exclude_px` | 6 | placeholder, UNVERIFIED - Ruri to check |
| `noise_margin` | 0.10 | placeholder, UNVERIFIED - Ruri to check |
| `cap_fraction` | 0.70 | placeholder, UNVERIFIED - Ruri to check (secondary-source snippet attributed to ASTM E2375, not read in original) |
| `min_snr` | 2.0 | placeholder, UNVERIFIED - Ruri to check |
| `block_seed` | 1000 | placeholder, UNVERIFIED - Ruri to check |

### Questions for Ruri
1. How does the macrozone correlation length in physical titanium billets compare with the ultrasonic beam focal spot size at 5 MHz and 10 MHz?
2. In what specific alloy grades does the ASTM E2375 cap fraction bind in practice, and what transducer frequency or focal adjustments are made?
3. Why must a zone flagged 'not inspectable at this sensitivity' never be cleared or suppressed by an advisory machine learning model?

> **Change note (after the first calibration run, before any experiment result):** the defaults were first `field_sigma: 0.4`, `zone_levels: [1.0, 1.6, 1.3]` and `signal_window_px: 3`. On the simulated reference block the 3x3 mean cut the reference response to about a third (the simulator grid, 1 mm x 3 deg, is coarser than the beam), and with field_sigma >= 0.15 the hottest noise patch set the zone peak noise so high that no zone was inspectable. The defaults were changed to `field_sigma: 0.1`, `zone_levels: [1.0, 1.2, 1.1]`, `signal_window_px: 1`. These are simulator model choices, not measured values; `experiments/run_ti_zones.py` sweeps field_sigma so the loss of inspectability is reported, not hidden.
