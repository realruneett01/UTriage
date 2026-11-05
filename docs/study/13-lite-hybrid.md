# 13 — LiteHybrid: Efficient Neural Architectures and Anomaly Fusion

## 1. Context and Motivation

Ultrasonic testing (UT) in aerospace bar inspection demands fast triage on power-constrained mobile hardware (such as an Intel Core i5-1135G7 laptop with 8 GB RAM and no dedicated GPU).
Standard deep convolutional backbones (such as MlNdtNet) impose computational and memory overheads that slow triage queues.
Furthermore, purely supervised classifiers risk missing unfamiliar flaw types that depart from training examples.
To balance computational efficiency and sensitivity to unexpected acoustic reflections, we combine an ultra-lightweight convolutional neural network (LiteNet) with an interpretable physics-feature anomaly detector (Mahalanobis distance with Ledoit-Wolf shrinkage) using OR-fusion.

## 2. Depthwise-Separable Convolutions: Parameter and FLOP Reduction

Standard two-dimensional convolutions simultaneously map spatial correlations and cross-channel relationships.
For an input tensor with $C_{\text{in}}$ channels and an output tensor with $C_{\text{out}}$ channels using a spatial kernel of size $k \times k$, the standard convolutional layer requires $C_{\text{out}}$ filters, each of dimensions $k \times k \times C_{\text{in}}$.

### Parameter Count Comparison

The number of trainable weights (excluding bias terms) for a standard convolution is:

$$\text{Params}_{\text{standard}} = k \cdot k \cdot C_{\text{in}} \cdot C_{\text{out}}$$

A depthwise-separable convolution factorises this operation into two distinct stages:

1. **Depthwise convolution**: A single spatial filter of size $k \times k$ is applied to each input channel independently, yielding $C_{\text{in}}$ separate feature channels. This stage captures purely spatial patterns without mixing channels:
   $$\text{Params}_{\text{depthwise}} = k \cdot k \cdot C_{\text{in}}$$

2. **Pointwise convolution**: A $1 \times 1$ convolution linearly projects the $C_{\text{in}}$ depthwise output channels into $C_{\text{out}}$ output channels, capturing purely cross-channel linear combinations:
   $$\text{Params}_{\text{pointwise}} = 1 \cdot 1 \cdot C_{\text{in}} \cdot C_{\text{out}} = C_{\text{in}} \cdot C_{\text{out}}$$

Summing the two stages gives the total parameter count for depthwise-separable convolution:

$$\text{Params}_{\text{separable}} = k \cdot k \cdot C_{\text{in}} + C_{\text{in}} \cdot C_{\text{out}}$$

The relative parameter reduction ratio is:

$$\frac{\text{Params}_{\text{separable}}}{\text{Params}_{\text{standard}}} = \frac{k^2 C_{\text{in}} + C_{\text{in}} C_{\text{out}}}{k^2 C_{\text{in}} C_{\text{out}}} = \frac{1}{C_{\text{out}}} + \frac{1}{k^2}$$

For standard $3 \times 3$ kernels ($k = 3$) with $C_{\text{out}} \ge 32$ feature maps:

$$\frac{\text{Params}_{\text{separable}}}{\text{Params}_{\text{standard}}} \approx \frac{1}{32} + \frac{1}{9} \approx 0.031 + 0.111 = 0.142$$

This yields an approximate 85% to 88% reduction in trainable parameters.

### Computational Complexity (FLOPs)

For an output spatial grid of height $H$ and width $W$, the number of multiply-accumulate operations (FLOPs) scales proportionally:

$$\text{FLOPs}_{\text{standard}} = H \cdot W \cdot k \cdot k \cdot C_{\text{in}} \cdot C_{\text{out}}$$

$$\text{FLOPs}_{\text{separable}} = H \cdot W \cdot (k \cdot k \cdot C_{\text{in}} + C_{\text{in}} \cdot C_{\text{out}})$$

Factorising spatial and channel operations cuts arithmetic complexity while preserving the receptive field, allowing LiteNet to operate well within a 100k parameter budget on laptop CPUs.

## 3. Physics-Feature Anomaly Detection: Mahalanobis Distance and Ledoit-Wolf Shrinkage

While LiteNet operates directly on raw B-scan acoustic arrays, the anomaly branch extracts an 8-dimensional vector of hand-crafted physical features $x \in \mathbb{R}^d$ (including peak amplitude, noise level via median absolute deviation, signal-to-noise ratio, peak depth index, echo duration, lateral extent, above-noise area, and echo energy ratio).

### Mahalanobis Distance

Let $\mu \in \mathbb{R}^d$ denote the mean feature vector computed across clean (flaw-free) calibration scans.
The Mahalanobis distance measures how far an observation $x$ deviates from the clean centroid $\mu$, scaled by the feature covariance matrix $\Sigma$:

$$d(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$

Unlike Euclidean distance, the Mahalanobis distance accounts for differing measurement scales and correlations among acoustic features (for instance, correlations between peak amplitude and echo energy).

### Ledoit-Wolf Covariance Shrinkage

In small-sample regimes or when features exhibit high multicollinearity, the empirical sample covariance matrix $S$:

$$S = \frac{1}{N - 1} \sum_{i=1}^N (x_i - \bar{x})(x_i - \bar{x})^T$$

often becomes ill-conditioned or nearly singular.
Inverting an ill-conditioned sample covariance matrix amplifies estimation noise along low-variance eigenvectors, causing erratic distance estimates.

Ledoit-Wolf shrinkage addresses this instability by shrinking the empirical sample covariance $S$ toward a well-conditioned, structured target $F = \frac{\text{Tr}(S)}{d} I$ (the identity matrix scaled by average feature variance):

$$\Sigma_{\text{LW}} = (1 - \lambda) S + \lambda F$$

where $\lambda \in [0, 1]$ is the analytically optimal shrinkage intensity derived by Ledoit and Wolf (2004) to minimise expected mean squared error under the Frobenius norm.
The resulting matrix $\Sigma_{\text{LW}}$ is mathematically positive definite and well-conditioned, yielding a robust anomaly metric:

$$d_{\text{LW}}(x) = \sqrt{(x - \mu)^T \Sigma_{\text{LW}}^{-1} (x - \mu)}$$

## 4. Behaviour of the Clean-Only Anomaly Branch

### Response to Unseen Flaw Types

Supervised models learn decision boundaries that separate known training flaw classes from sound material.
When presented with an unmodelled flaw morphology (such as unexpected crack tilt angles, porous clusters, or slag inclusions), a supervised model may map the response to the sound region because its learned filters look only for specific flaw signatures.

In contrast, the anomaly detector trains exclusively on sound, flaw-free acoustic scans.
It builds a statistical envelope of normal material scatter.
Any internal condition that disrupts the sound baseline—whether through unexpected wave scatter, path attenuation, phase shifts, or prolonged ringing—generates a large distance $d_{\text{LW}}(x)$ and triggers an indication.
The anomaly branch therefore acts as a safety net for unexpected flaw geometries.

### Outlier Flagging on Unusual Sound Scans

The property that gives the anomaly branch broad sensitivity also introduces an operational trade-off:
The detector has no semantic understanding of defect physics; it flags statistical novelty.
Consequently, unusual but benign sound scans—caused by local coarse grain scatter, couplant thickness variations, slight surface roughness, or electrical transients—also exhibit large distances from $\mu$.
Without controlled threshold calibration, the anomaly branch can drive up the false call rate on atypical clean material.

## 5. OR-Fusion and False Call Budget Allocation

### Preserving Every Candidate Flag (Safety Design)

In UTriage safety architecture, conventional rule checks remain paramount, and machine learning components can only add candidate flags for human review.
Within the hybrid model itself, OR-fusion combines the supervised LiteNet classifier and the Mahalanobis anomaly detector:

$$\text{Flag}_{\text{fused}}(x) = \text{Flag}_{\text{LiteNet}}(x) \lor \text{Flag}_{\text{anomaly}}(x)$$

Every indication flagged by either branch is preserved.
If LiteNet misses an unfamiliar defect but the anomaly branch flags it, the indication reaches the inspector.
Conversely, if a known crack type produces borderline anomaly distance but triggers LiteNet, it is also flagged.
No alarm raised by either subsystem is discarded.

### Union Bound on the False Call Rate

Combining two detectors risks inflating the false call rate (FCR) on clean material.
To control this risk, we split the total permissible false call budget $\alpha$ (for example, $\alpha = 0.05$) equally between the two branches:

- Supervised branch false call allowance: $\alpha_1 = \alpha / 2 = 0.025$
- Anomaly branch false call allowance: $\alpha_2 = \alpha / 2 = 0.025$

Thresholds $\tau_{\text{LiteNet}}$ and $\tau_{\text{anomaly}}$ are calibrated independently on clean calibration scans:

$$\tau_{\text{LiteNet}} = Q_{1 - \alpha/2}(\{s_{\text{LiteNet, clean\_cal}}\})$$

$$\tau_{\text{anomaly}} = Q_{1 - \alpha/2}(\{d_{\text{LW, clean\_cal}}\})$$

Let $E_1$ denote the event that LiteNet raises a false call on a sound scan ($s_{\text{LiteNet}} \ge \tau_{\text{LiteNet}}$), and $E_2$ denote the event that the anomaly branch raises a false call ($d_{\text{LW}} \ge \tau_{\text{anomaly}}$).
By Boole's inequality (the union bound):

$$P(E_1 \cup E_2) \le P(E_1) + P(E_2) = \frac{\alpha}{2} + \frac{\alpha}{2} = \alpha$$

Hence, the fused false call rate under clean conditions satisfies:

$$\text{FCR}_{\text{fused}} \le \alpha$$

When both branches flag the same atypical sound scans simultaneously ($P(E_1 \cap E_2) > 0$), the actual false call rate drops below the conservative bound $\alpha$:

$$P(E_1 \cup E_2) = P(E_1) + P(E_2) - P(E_1 \cap E_2) = \alpha - P(E_1 \cap E_2) \le \alpha$$

## 6. AUROC: Meaning and Interpretation

The Area Under the Receiver Operating Characteristic curve (AUROC) measures the global ranking capability of a scoring function across all possible operating thresholds.
Mathematically, AUROC is equivalent to the probability that a randomly selected flawed scan receives a higher indication score than a randomly selected sound scan:

$$AUROC = P(s_{\text{flawed}} > s_{\text{clean}})$$

Computed via the Mann-Whitney $U$ statistic:

$$U = \sum_{i=1}^{n_{\text{flawed}}} \sum_{j=1}^{n_{\text{clean}}} \left[ \mathbb{I}(s_{\text{flawed}, i} > s_{\text{clean}, j}) + \frac{1}{2} \mathbb{I}(s_{\text{flawed}, i} = s_{\text{clean}, j}) \right]$$

$$AUROC = \frac{U}{n_{\text{flawed}} \cdot n_{\text{clean}}}$$

Unlike single-operating-point hit rates, AUROC reflects discrimination power across the entire spectrum of score thresholds without depending on a single threshold choice.

## 7. Execution Latency: Multithreaded CPU Benchmarks as a Proxy

The target hardware for UTriage is a mobile laptop equipped with an Intel Core i5-1135G7 CPU (4 physical cores, 8 threads, 8 GB RAM, no dedicated GPU).
In our development environment, latency benchmarks restrict PyTorch execution to 4 CPU worker threads using `torch.set_num_threads(4)` to simulate quad-core execution.

However, desktop and workstation processors differ substantially from mobile laptop silicon:
1. **Thermal and Power Constraints**: Mobile laptop CPUs operate under strict 15 W to 28 W thermal envelopes and throttle clock speeds under sustained multi-scan processing.
2. **Cache Hierarchy**: Desktop processors possess substantially larger L2 and L3 caches, reducing main-memory latency.
3. **Memory Bandwidth**: High-speed desktop memory channels achieve higher throughput than mobile low-power DDR configurations.

Restricting execution to 4 CPU threads on development hardware provides a reproducible proxy for relative comparison against heavier models, but it does not represent an exact measurement of latency on the physical i5 laptop.
