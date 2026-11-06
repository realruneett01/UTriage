# 14 — Invariant Representation Learning and Physics-Informed Views

## 1. Area Under the ROC Curve (AUROC) from Scratch

### Definition and Pairwise Ranking Probability

In ultrasonic non-destructive testing, an automated triage model assigns a scalar anomaly score to each ultrasonic B-scan.
A higher score indicates a higher likelihood of an internal flaw, while a lower score indicates sound, clean material.

Suppose we pick one flawed scan at random and one clean scan at random from an evaluation dataset, and compare their scores.
The Area Under the Receiver Operating Characteristic curve (AUROC) equals the probability that the flawed scan receives a higher score than the clean scan, with one-half awarded for ties:

$$\text{AUROC} = P(s_{\text{flawed}} > s_{\text{clean}}) + 0.5 \cdot P(s_{\text{flawed}} = s_{\text{clean}})$$

Equivalently, for an evaluation dataset containing $n_{\text{flawed}}$ flawed scans and $n_{\text{clean}}$ clean scans, AUROC corresponds to the normalized Mann-Whitney $U$ statistic:

$$U = \sum_{i=1}^{n_{\text{flawed}}} \sum_{j=1}^{n_{\text{clean}}} \left[ \mathbb{I}(s_{\text{flawed}, i} > s_{\text{clean}, j}) + 0.5 \cdot \mathbb{I}(s_{\text{flawed}, i} = s_{\text{clean}, j}) \right]$$

$$\text{AUROC} = \frac{U}{n_{\text{flawed}} \cdot n_{\text{clean}}}$$

where $\mathbb{I}(\cdot)$ is the indicator function.

- An AUROC of **0.5** represents a coin flip: the model ranks scans no better than random guessing, with clean and flawed distributions completely overlapping.
- An AUROC of **1.0** represents a perfect ranking: every flawed scan scores higher than every clean scan.

### The ROC Curve Across Varying Thresholds

To separate scans into "flawed" or "clean" in practice, an inspector applies a decision threshold $\tau$.
Scans with score $s \ge \tau$ are flagged as candidate flaws; scans with $s < \tau$ are classified as clean.

- **Detection Rate** (True Positive Rate): The fraction of flawed scans flagged ($s \ge \tau$).
- **False-Call Rate** (False Positive Rate): The fraction of clean scans mistakenly flagged ($s \ge \tau$).

Sweeping the threshold $\tau$ continuously from $+\infty$ down to $-\infty$ traces out the Receiver Operating Characteristic (ROC) curve:
- At $\tau = +\infty$, no scans are flagged (Detection Rate = 0, False-Call Rate = 0).
- At $\tau = -\infty$, all scans are flagged (Detection Rate = 1, False-Call Rate = 1).

The area beneath this trajectory is mathematically identical to the pairwise ranking probability.
Because AUROC integrates performance across all possible thresholds, it measures intrinsic score separability without depending on any single threshold choice.

### Score Shift on Unseen Cracks: High AUROC with Poor Fixed-Threshold Detection

A model can achieve a high AUROC yet exhibit poor detection when evaluated at a fixed threshold such as $\tau = 0.5$.

Consider a model where sound material scans yield scores between 0.05 and 0.15.
When evaluated on familiar training flaws, flawed scans yield scores between 0.70 and 0.90, so a fixed 0.5 threshold catches them easily.
However, when presented with an unseen crack (such as the held-out crack in the 1.6 mm or 4.0 mm folds), acoustic scattering differences, altered crack facet angles, or reduced tip diffractions may cause the model's scores to compress into the range 0.25 to 0.45.

In this scenario:
1. Every flawed scan ($0.25 \le s \le 0.45$) scores higher than every clean scan ($0.05 \le s \le 0.15$). The AUROC is 1.0 (perfect ranking).
2. If evaluated at the fixed default threshold $\tau = 0.50$, every flawed scan falls below 0.50. The detection rate is 0.0 (zero flaws detected).

This score shift on unseen flaws is exactly the question encountered in the 1.6 mm and 4.0 mm folds.
AUROC reveals whether clean and flawed distributions remain separable, while clean-calibrated thresholding adapts the decision boundary to the actual score scale.

### Small Worked Example (Illustrative, Not Project Data)

To illustrate pairwise ranking and threshold sensitivity, consider three flawed scans and three clean scans with synthetic scores *(illustrative, not project data)*:

- Flawed scan scores: $s_{f1} = 0.42$, $s_{f2} = 0.38$, $s_{f3} = 0.28$
- Clean scan scores: $s_{c1} = 0.18$, $s_{c2} = 0.12$, $s_{c3} = 0.08$

Evaluating all $3 \times 3 = 9$ pairwise comparisons:
- $s_{f1} = 0.42$: greater than $0.18, 0.12, 0.08 \implies 3$ wins
- $s_{f2} = 0.38$: greater than $0.18, 0.12, 0.08 \implies 3$ wins
- $s_{f3} = 0.28$: greater than $0.18, 0.12, 0.08 \implies 3$ wins

Total wins = 9 out of 9 pairs:

$$\text{AUROC} = \frac{3 + 3 + 3}{3 \times 3} = \frac{9}{9} = 1.0$$

At a fixed classification threshold $\tau = 0.50$:
- Flawed scans flagged ($s \ge 0.50$): 0 of 3 $\implies$ Detection Rate = 0.0
- Clean scans flagged ($s \ge 0.50$): 0 of 3 $\implies$ False-Call Rate = 0.0

At a clean-calibrated threshold chosen just above the highest clean score ($\tau = 0.20$):
- Flawed scans flagged ($s \ge 0.20$): 3 of 3 $\implies$ Detection Rate = 1.0
- Clean scans flagged ($s \ge 0.20$): 0 of 3 $\implies$ False-Call Rate = 0.0

### Average Precision

Average Precision (AP) summarizes the precision-recall curve across all recall operating points by calculating the area under the precision-recall curve:

$$\text{AP} = \sum_{k} (R_k - R_{k-1}) P_k$$

In non-destructive evaluation, clean material scans vastly outnumber flawed scans.
While AUROC weights clean and flawed scans equally in pairwise ranking across the full false-call spectrum, Average Precision focuses on positive predictive value, penalizing false alarms heavily when flaw indications are rare events in large volumes of sound material.

## 2. Why the Supervised Model is Weak on Unseen Cracks

Supervised deep learning models excel when test data share the exact distribution of training data.
However, in the benchmark ML-NDT dataset, there are only 3 physical cracks (nominal depths 1.6 mm, 4.0 mm, and 8.6 mm) introduced into a 316L pipe weld specimen.

When training on 2 cracks and evaluating on the third, the network does not learn an abstract, general concept of "an acoustic crack".
Instead, it learns the specific spatial reflections, facet angles, and tip diffractions of "these two cracks".
Although the dataset contains thousands of 2D B-scan images across multiple scan passes, these thousands of images are simply different views of one physical object per crack, not independent flaws.

When the network encounters the third physical crack—with a different orientation, facet roughness, or opening displacement—its feature activations shift.
Because the supervised loss only penalised misclassifying the two training cracks, the model never learned to disregard benign acoustic variations that do not alter the presence of a flaw.

## 3. Invariant Representation Learning with Physics Views

### The Invariance Analogy: The Same Graph in Differently Worded Questions

Consider a mathematics examination where the same quadratic graph appears in differently worded questions.
A student who memorises the specific wording of a sample question fails when the question is rephrased, even though the underlying graph is unchanged.
In contrast, a student who understands the governing equation recognises the relationship regardless of phrasing.

In ultrasonic inspection, the physical flaw is the underlying reality.
The observed B-scan changes because of acquisition conditions: transducer coupling pressure, receiver amplifier gain, electronic noise, water path variations in immersion inspection, slight sound velocity differences, probe scanning speed, and raster direction.
Invariant representation learning trains an encoder to map diverse acoustic presentations of the same scan to nearby points in feature space, isolating the structural flaw signature from acquisition disturbances.

### Seven Physically Plausible Transforms

To teach the model these invariances, each acoustic B-scan $x$ is transformed into augmented views using seven physically grounded operations:

1. **Gain Variation ($g$)**: Multiplicative scaling representing transducer coupling pressure, couplant layer thickness, and pulser-receiver gain adjustments:
   $$x' = g \cdot x, \quad g \sim \mathcal{U}(1 - \Delta_g, 1 + \Delta_g)$$

2. **DC Offset Drift ($b$)**: Additive baseline shift representing thermal drift in receiver pre-amplifiers and analog-to-digital converter bias:
   $$x' = x + b, \quad b \sim \mathcal{U}(-\Delta_b, \Delta_b)$$

3. **Noise Injection ($n$)**: Additive random variation representing thermal electronic noise and microstructural grain scatter:
   $$x' = x + n, \quad n \sim \mathcal{N}(0, \sigma_n^2)$$

4. **Time Shift ($\Delta t$)**: Translation along the time-of-flight axis representing variations in immersion water path or probe standoff:
   $$x'(t, s) = x(t - \Delta t, s)$$

5. **Time Stretch ($\alpha_t$)**: Dilation or compression along the time axis representing acoustic velocity shifts caused by alloy temperature gradients or component thickness:
   $$x'(t, s) = x(\alpha_t \cdot t, s)$$

6. **Scan Stretch ($\alpha_s$)**: Dilation or compression along the scan raster axis representing probe speed fluctuations, encoder pitch variations, or crack lateral extent:
   $$x'(t, s) = x(t, \alpha_s \cdot s)$$

7. **Scan Flip**: Spatial reversal along the scan raster axis representing reversing the probe scanning direction relative to the weld seam:
   $$x'(t, s) = x(t, -s)$$

### Contrastive Learning Formulations

Let $f_\theta(\cdot)$ denote a convolutional encoder (LiteEncoder) mapping scan $x$ to feature representation $h = f_\theta(x) \in \mathbb{R}^d$.
A projection head $g_\phi(\cdot)$ maps $h$ to a lower-dimensional embedding $z = g_\phi(h) \in \mathbb{R}^m$, normalised such that $\|z\|_2 = 1$.

#### NT-Xent (Normalized Temperature-scaled Cross Entropy Loss)

Each scan $x_i$ generates two augmented views, forming a batch of $2N$ views.
For positive pair $(z_i, z_j)$ originating from the same scan $x_i$, the NT-Xent loss is:

$$\mathcal{L}_{\text{NT-Xent}} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)}$$

where $\text{sim}(u, v) = u^T v$ is the cosine similarity between unit-normalised vectors, and $\tau > 0$ is a temperature hyperparameter that adjusts penalty sharpness for hard negatives.

#### Supervised Contrastive Loss (SupCon)

When training labels (sound vs flawed) are available for the training cracks, supervised contrastive learning pulls together all views of all training flaw scans while repelling clean material views.
For an anchor index $i \in I \equiv \{1, \dots, 2N\}$, let $P(i) \equiv \{p \in A(i) : y_p = y_i\}$ denote the set of positive indices sharing the same class label as $i$, where $A(i) \equiv I \setminus \{i\}$:

$$\mathcal{L}_{\text{SupCon}} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / \tau)}$$

This clusters representations of acoustic flaw echoes across both available training cracks into a compact subspace while separating them from sound grain scatter.

#### Fine-Tuning Stage

After contrastive pretraining:
1. The projection head $g_\phi(\cdot)$ is detached.
2. A linear classification head is attached to the encoder output $h$.
3. The network undergoes fine-tuning with cross-entropy loss.
4. To preserve the learned invariant embedding geometry, encoder weights are adjusted with a smaller learning rate than the new classification head:

$$\eta_{\text{encoder}} \ll \eta_{\text{head}}$$

This differential learning rate prevents destroying the invariant representation while aligning the decision boundary with the training data.

## 4. Limits of Invariant Representations

1. **Approximation Bounds**: The seven physical transforms are simplified mathematical models of wave mechanics. They do not simulate full 3D elastodynamic wave equations, mode conversion between longitudinal and shear waves, beam refraction at curved interfaces, or non-linear crack tip scattering.
2. **Unseen Defect Geometries**: Invariance training aligns representations under specified physical perturbations, but it cannot invent crack shapes or flaw morphologies absent from the two training cracks. If both training cracks are planar fatigue cracks, the encoder cannot invent the distinct scattering signatures of volumetric slag inclusions or spherical porosity.
3. **Evidence Boundary with Three Cracks**: With only 3 physical cracks in the ML-NDT dataset, no empirical result can show generalisation to all flaw types, materials, or geometries. Any conclusions remain strictly bounded by this 3-crack experimental setting.
