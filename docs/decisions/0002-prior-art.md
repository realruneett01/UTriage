# 0002. Prior art evaluation and positioning

- Status: Accepted
- Date: 2026-10-26

## Context

Prior art relevant to ultrasonic inspection triage exists in commercial patents and academic research:

- US 9,927,404 B2 ("Phased array billet data evaluation software", United Technologies / RTX): in UT of titanium billets and bars, coarse grain structure creates background noise that causes many false detections. Borderline echoes need extra evaluation by the inspector. Industry has patented software to classify indications by signal-to-noise ratio (SNR).
Patent claims not reviewed in detail - UNVERIFIED, Ruri to check.

- Virkkunen et al. ("Augmented Ultrasonic Data for Machine Learning", arXiv:1903.11399; data at github.com/iikka-v/ML-NDT under LGPL-3.0): deep learning on real phased-array ultrasonic data has beaten human inspectors in a narrow, published setting: a90/95 of 0.9 mm against 2.7–5.6 mm for humans, with zero false calls. The setting was only 3 real cracks in a 316L pipe weld, and the authors state the limits clearly.

## Decision

UTriage is positioned as an open, offline portfolio and research project, not a commercial copy of the patented method.

How UTriage differs:
- POD-validated ML layer on top of rules (union, ML only adds): fusion operates strictly as a union where ML can only add flags and never remove a rules flag.
- Data-quality watchdog: monitors scan health before parts leave the tank.
- Open and offline: runs locally without cloud dependencies or machine controller connections.
- Portfolio and research project: developed for methodology demonstration rather than commercial replication.

## Consequences

- Licensing and data handling: ML-NDT data is fetched by script and never redistributed, respecting its LGPL-3.0 licence.
- Verification obligation: patent claims not reviewed in detail - UNVERIFIED, Ruri to check.
