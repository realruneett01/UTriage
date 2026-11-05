# 10 — Self-Supervised Masked-Patch Pretraining

## Plant context: label scarcity in ultrasonic testing
Immersion ultrasonic inspection produces gigabytes of raw pulse-echo data: A-scan waveforms, B-scan cross sections, and C-scan top views across metres of bar stock. In production, sound material represents the vast majority of volume. Confirmed defect indications (such as voids, inclusions, or fatigue cracks) are rare and costly to label manually. Training deep neural networks exclusively on small supervised sets creates a risk of overfitting to specific flaw geometries.

Self-supervised pretraining addresses this scarcity by learning feature representations directly from abundant unlabelled ultrasonic scans without requiring any inspector annotations.

## Masked-patch reconstruction formulation
The pretraining strategy employs masked-patch denoising (analogous to masked autoencoding).

1. **Patch division**: An unlabelled ultrasonic image $x \in \mathbb{R}^{B \times 1 \times H \times W}$ is partitioned into non-overlapping spatial patches of size $P \times P$ (default $P = 16$). The number of patches along height and width is:
   $$N_h = \frac{H}{P}, \quad N_w = \frac{W}{P}, \quad N = N_h \times N_w$$

2. **Patch masking**: For each sample in the batch, a random subset of $K = \lfloor N \cdot r \rfloor$ patches is selected for masking, where $r$ is the masking ratio (default $r = 0.5$). A binary spatial mask $M \in \{0, 1\}^{B \times 1 \times H \times W}$ is formed by upsampling the patch mask such that $M_{i,j} = 1$ on masked pixels and $0$ on visible pixels.

3. **Masked input**: Masked pixels are zeroed out:
   $$\tilde{x} = x \odot (1 - M)$$

4. **Reconstruction**: An autoencoder $f_\theta$ (such as `ConvAE2D`) receives the masked image $\tilde{x}$ and produces reconstruction $\hat{x} = f_\theta(\tilde{x})$.

5. **Loss over masked pixels**: The training objective is mean squared error evaluated exclusively over the masked pixels:
   $$\mathcal{L}_{\text{masked}} = \frac{1}{\sum M} \sum_{b, c, i, j} M_{b, c, i, j} \cdot (\hat{x}_{b, c, i, j} - x_{b, c, i, j})^2$$
   Unmasked pixels are ignored in the loss, forcing the network to predict missing regions from surrounding acoustic context rather than reconstructing already-visible inputs.

## Why masked reconstruction helps ultrasonic feature learning
Ultrasonic data is governed by wave propagation physics:
- **Acoustic continuity**: Reflected waveforms and back-wall echoes exhibit spatial coherence across adjacent scan positions.
- **Grain noise texture**: Coarse grain structures (e.g., in titanium and nickel alloys) generate Rayleigh scattering with characteristic spatial correlation lengths.
- **Structural boundaries**: Interface boundaries and cylindrical bar surfaces impose geometric constraints on acoustic arrival times.

To infill a zeroed patch, the convolutional encoder cannot rely on trivial local interpolation. It must capture the underlying wavefield structure, velocity profile, and grain noise distribution from surrounding patches. This forces the encoder to learn robust acoustic representations before fine-tuning on downstream tasks.

## Downstream fine-tuning
Once pretraining finishes, `encoder_state(model)` extracts the encoder weights. The decoder is discarded; only supervised MlNdtNet classifier fine-tuning was run and evaluated.

## Current status and limitations
- **Measured results**: See results/joint_train_metrics.json (arms C and D) and the generated tables in README.md; results are single-seed. Arm D pretraining and fine-tuning include Synthetic data (SimBscanDataset); evaluation is on real ML-NDT weld images only; pretraining gave mixed results with no consistent gain.
- **Resolution alignment**: Patch size $P$ must match the spatial scale of ultrasonic beam spread and scan index. Very large patches remove too much contextual evidence, while very small patches permit trivial edge smoothing.
- **Domain limitation**: Pretraining on flaw-free material teaches normal acoustic background; it does not expose the network to rare flaw reflection phase inversions or corner reflections.
