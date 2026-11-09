# Documentation Index

Technical documentation, design records, study notes, and evaluation resources for the UTriage advisory decision-support prototype.

## Start here
- [Project Overview](../README.md): Problem statement, safety architecture, and quick-start instructions.
- [User Guide](user-guide.md): Inspector workflows and application review guidance.
- [Installation Guide](install.md): Workstation system requirements, offline setup, and checksum checks.

## Understand the design
- [Architecture](architecture.md): System components, data flow, and union fusion rules.
- [Technical Note](technical-note.md): Physics formulation, simulation models, and algorithms.
- [Safety Case](safety-case.md): Hazard analysis, safety arguments, and failure mitigations.

### Architecture decisions
- [0001. Safety design](decisions/0001-safety-design.md)
- [0002. Prior art evaluation and positioning](decisions/0002-prior-art.md)
- [0003. Simulator assumptions and physics simplifications](decisions/0003-simulator-assumptions.md)
- [0004. Operating point selection for advisory ML triage](decisions/0004-operating-point.md)
- [0005. Train on the GPU; keep inference CPU-capable](decisions/0005-gpu-training.md)
- [0006. Proprietary product decision and intellectual property strategy](decisions/0006-proprietary-product.md)
- [0007. Offline compiled Windows distribution with Nuitka](decisions/0007-distribution.md)
- [0008. Dev dependency: pytest-cov for test coverage](decisions/0008-coverage-tool.md)
- [0009. Generalization evaluation protocol (Phase 8)](decisions/0009-generalization-protocol.md)
- [0010. Supervised + anomaly hybrid detector - pre-registered evaluation](decisions/0010-hybrid-detector-preregistration.md)
- [0011. Site adaptation for zero-label site transfer](decisions/0011-site-adaptation.md)
- [0012. Robust training leave-one-crack-out (LOCO) pre-registration](decisions/0012-robust-loco-preregistration.md)
- [0013. LiteHybrid (H4) detector pre-registration](decisions/0013-lite-hybrid-preregistration.md)
- [0014. Invariant representation learning pre-registration](decisions/0014-invariant-representation-preregistration.md)
- [0015. Shift robustness evaluation pre-registration](decisions/0015-shift-robustness-preregistration.md)
- [0016. Titanium mode (UTriage-Ti) with structured grain noise and zone-wise reference calibration](decisions/0016-titanium-mode.md)

## Learn the physics and ML
- [Ultrasonic Testing Primer: Study Questions](study/00-ut-primer.md)
- [01 — The A-scan model used by the simulator](study/01-simulator-physics.md)
- [02 — The conventional rules engine and the data-quality watchdog](study/02-rules-engine.md)
- [03 — The real-data detector (ML-NDT)](study/03-model.md)
- [03 — Probability of detection (POD): hit/miss and a90/95](study/03-pod.md)
- [04 — â-versus-a POD (signal-response POD)](study/04-ahat-pod.md)
- [04 — Bar-domain ML under titanium grain noise (Synthetic data)](study/04-bar-domain-ml.md)
- [08 — Unsupervised Anomaly Detection](study/08-anomaly.md)
- [08-data-hunt.md — Public Ultrasonic NDT Dataset Survey](study/08-data-hunt.md)
- [08 — Full Matrix Capture (FMC) and Total Focusing Method (TFM)](study/08-tfm.md)
- [09 — Unsupervised Site Adaptation and Calibration](study/09-site-adaptation.md)
- [10 — Self-Supervised Masked-Patch Pretraining](study/10-self-supervised.md)
- [11 — 2D B-Scan Generation from 1D Immersion Simulation (Synthetic data)](study/11-sim-to-2d.md)
- [12 — Flaw Separability Metrics and Robust Training Physics](study/12-separability-and-robust-training.md)
- [13 — LiteHybrid: Efficient Neural Architectures and Anomaly Fusion](study/13-lite-hybrid.md)
- [14 — Invariant Representation Learning and Physics-Informed Views](study/14-invariant-representation.md)
- [15 — Shift Robustness and Score Margin in Ultrasonic Flaw Triage](study/15-shift-robustness.md)
- [16 — Titanium Grain Noise and Zone-Wise Reference Calibration](study/16-ti-grain-noise-and-zone-calibration.md)

## Develop
- [Developer Guide](developer-guide.md): Development environment setup, testing conventions, and codebase layout.
- [Verification Checklist](verification-checklist.md): Traceability checklist of parameters and placeholder values.

## Results
- [Claims and Evidence Limits](../results/claims.md): Full audit of model claims, dataset scope, and empirical bounds.
