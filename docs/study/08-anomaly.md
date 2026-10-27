# 08 — Unsupervised Anomaly Detection

## Plant context: extreme class imbalance
In industrial ultrasonic testing of bar stock, production lines inspect thousands of sound bars for every bar containing a defect. Real defects (such as inclusions, cracks, or porosity) are rare events. Supervised deep learning models require balanced sets of labelled flaw examples that are scarce in production. Unsupervised anomaly detection addresses this imbalance by training solely on abundant flaw-free data.

## The reconstruction-error concept
An autoencoder (`ConvAE1D` for A-scan windows or `ConvAE2D` for B-scan images) learns a compressed representation of sound material and reconstructs the input. Because it trains only on clean scans, it reconstructs sound material with low error. When an unusual reflection from an internal flaw appears, the network fails to reconstruct it accurately, producing elevated squared error. A moving-average filter of length 5 smooths high-frequency noise while highlighting coherent defect echoes.

## Wall gating
Ultrasonic signals contain large reflections from the front and back walls of the bar. Because wall echo amplitudes dwarf internal flaw echoes, slight variations in acoustic coupling or bar geometry would cause massive reconstruction errors at the boundaries. The `gate_walls` function masks out samples within a guard distance of the front and back wall echoes (`abs(idx - wall_idx) <= guard`), preventing wall reflections from dominating the anomaly score.

## Threshold selection from target false-call rate
Rather than choosing an arbitrary score threshold, `fit_threshold` computes the $(1 - \text{rate})$ empirical quantile of anomaly scores evaluated on flaw-free validation data. For a specified false-call rate $\alpha$ (such as 0.05), the threshold flags approximately the top $\alpha$ fraction of clean scans, providing a predictable operational trade-off between sensitivity and false alarm rate.

## Limitations
- **Non-specific flagging**: The detector flags any unusual acoustic signature, including couplant variations, surface roughness, grain noise, or electrical noise. It detects deviations from baseline, not specific flaw types.
- **No flaw sizing**: Reconstruction error magnitude reflects acoustic novelty rather than physical flaw dimensions (such as crack height or equivalent flat-bottom hole diameter).
- **Dead zones at boundaries**: Wall gating eliminates false alarms from wall echoes but leaves near-wall blind zones where near-surface flaws cannot be detected by this metric alone.

## References and prior art
- Approach as in: Tunukovic et al., Ultrasonics 140 (2024) 107313.
