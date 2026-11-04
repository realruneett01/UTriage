# 11 — 2D B-Scan Generation from 1D Immersion Simulation (Synthetic data)

## 1. Context and Problem
The Phase 1 immersion ultrasonic simulator generates 1D radio-frequency (RF) A-scans
$s(t)$ across scan coordinates. However, external benchmark datasets such as ML-NDT
provide 2D ultrasonic images of size $256 \times 256$ pixels. To enable joint
synthetic-real dataset handling and cross-domain transfer experiments (project plan Step 3.1),
we synthesize 2D B-scans from simulated scan positions.

## 2. B-Scan Physics and Lateral Beam Profile
A B-scan displays ultrasound reflection amplitude as a function of lateral probe
position $x$ (scanning axis) and pulse travel time $t$ (depth axis).

For an isolated reflector (e.g., flat-bottom hole) located at position $x_{\text{flaw}}$
and depth $z_{\text{flaw}}$, the echo amplitude across lateral scan positions $x$ is
weighted by the probe's Gaussian lateral beam profile:

$$w(x) = \exp\left(-\frac{(x - x_{\text{flaw}})^2}{2\sigma_{\text{beam}}^2}\right)$$

where $\sigma_{\text{beam}}$ is the lateral beam standard deviation at focus loaded
from `configs/probe.yaml`. The raw RF echo across positions is:

$$s(x, t) = s_{\text{clean}}(x, t) + w(x) \cdot A_{\text{flaw}} \cdot p(t - t_{\text{flaw}})$$

where $A_{\text{flaw}}$ is calculated via `fbh_amplitude()`, $p(t)$ is the transducer
pulse waveform, and $s_{\text{clean}}(x, t)$ contains interface echoes, back-wall
echoes, and titanium grain backscatter.

## 3. Image Processing: Envelope, Resampling, and Standardisation
To convert raw RF matrices of shape $(N_{\text{positions}}, N_{\text{samples}})$ into
$256 \times 256$ input tensors:

1. **Analytic Envelope:** Compute the magnitude of the Hilbert transform along the fast-time axis:
   $$E(x, t) = |\mathcal{H}\{s(x, t)\}| = \sqrt{s(x, t)^2 + \tilde{s}(x, t)^2}$$
2. **Axis Resampling:** Resample both axes to 256 points via piecewise linear interpolation
   (`np.interp`) independently along the fast-time and probe-position dimensions.
3. **Standardisation:** Apply the same per-image transformation used for ML-NDT:
   $$I_{\text{std}} = \frac{I - \mu_I}{\sigma_I + 10^{-5}}$$
   yielding a float32 array of shape $(256, 256)$.

## 4. PyTorch Streaming Dataset
`SimBscanDataset` implements `torch.utils.data.Dataset`. For any index `idx`, it synthesizes
the sample on the fly using seed `base_seed + idx`. This keeps memory bounded (O(1) RAM)
and allows streaming concatenation with `MlNdtDataset` via `torch.utils.data.ConcatDataset`.

## 5. Limitations and Honesty
- **Synthetic data:** All outputs from this pipeline are simulated synthetic data.
- **Domain Gap:** Significant physics differences exist between simulated cylindrical bar
  immersion scans and real plate/weld contact or phased array scans (ML-NDT). Spatial
  grain correlation, beam skew, and surface roughness differ from plant conditions.
- **Unverified Constants:** Acoustic velocities, attenuation coefficients, and probe
  parameters are handbook figures or model choices from `configs/*.yaml` labelled
  `UNVERIFIED - Ruri to check`. Never claim plant qualification or MIDHANI validation.
