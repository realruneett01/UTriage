# Interview Preparation: 25 Hard Technical Questions

> **Study Guide**: Answer these 25 questions in your own words without notes before technical interviews or discussions with Level III inspectors and aerospace QA managers.

1. What happens if your model misses a hard-alpha inclusion?
Answer:

2. Why should a Level III trust this?
Answer:

3. How is this different from the SNR software already sold?
Answer:

4. Your real data is a weld — why does it matter for bars?
Answer:

5. What POD study would qualify it?
Answer:

6. What is Probability of Detection (POD), and how does the hit/miss method formulate detection probability as a function of flaw size using a logistic generalized linear model (GLM)?
Answer:

7. What is the statistical definition of a90/95, how is it calculated from the covariance matrix of parameter estimates using the delta method, and why does a small sample of independent physical flaws cause a90/95 to be far above a90?
Answer:

8. In results/mlndt_report.md, why was the logistic POD model not estimable (reporting null) on the ML-NDT dataset, and what mathematical condition in logistic regression causes this when all flawed specimens are detected?
Answer:

9. What is a matched false-call rate (FCR), and why is it invalid to claim an ML detector outperforms a conventional rules baseline without matching their false-call operating points?
Answer:

10. What physical mechanisms (attenuation, beam divergence, transducer focusing) cause ultrasonic echo amplitude to decrease with depth, and how does a Distance-Amplitude Correction (DAC) curve constructed from reference blocks normalize reflector signals across varying metal paths?
Answer:

11. What microstructural features in titanium alloys (such as prior beta grains and alpha colonies) cause acoustic grain noise backscatter, and why must Signal-to-Noise Ratio (SNR) be calculated within a local spatial window rather than across the entire bar?
Answer:

12. How does conformal prediction produce calibrated prediction sets with distribution-free marginal coverage guarantees, and how does UTriage use it to tag uncertain indications for human inspector review?
Answer:

13. How does the union safety rule in triage/fusion.py ensure that machine learning can only add candidate flags and never suppress, downgrade, or remove a conventional rule-based indication?
Answer:

14. What specific hazard was addressed by the data-quality fault safety fix in rules/acceptance.py, and why must an acoustic fault (such as loss of back reflection) never overwrite or downgrade a rejectable or reportable flaw indication?
Answer:

15. What is the sim-to-real domain shift between synthetic acoustic simulations and actual immersion bar scans, and what specific acoustic variables (such as velocity dispersion, attenuation variation, probe wear, and beam distortion) cause models trained on synthetic data to fail on real plant scans?
Answer:

16. Why are the acceptance criteria and reject/record thresholds in configs/acceptance.yaml explicitly designated as UNVERIFIED placeholders, and why must an engineering tool never hard-code standards limits (such as AMS-STD-2154 or ASTM E2375) from memory?
Answer:

17. What are the key technical findings, strengths, and critical limitations of the Virkkunen et al. (arXiv:1903.11399) study on augmented ultrasonic data for machine learning?
Answer:

18. What specific method is claimed in US Patent 9,927,404 B2 ("Phased array billet data evaluation software"), and how does UTriage differ in its architecture, licensing, and technical scope?
Answer:

19. How does the â-vs-a (signal response versus flaw size) POD formulation differ from the binary hit/miss POD formulation, and under what conditions does MIL-HDBK-1823A recommend each?
Answer:

20. Why does the statistical "rule of three" dictate that observing zero false calls across 100 unflawed inspection scans only bounds the true false-call rate to approximately 3%, and why is that insufficient evidence for high-volume aerospace manufacturing?
Answer:

21. Why must ultrasonic flaw detection gates be dynamically synchronised to the front-wall (interface) echo rather than to a fixed electronic pulser trigger?
Answer:

22. How does the data-quality watchdog detect coupling aeration, water-path drift, and back-wall echo attenuation, and why must these faults trigger an immediate re-scan recommendation before the bar is removed from the immersion tank?
Answer:

23. What spatial cues does a 2D patch classifier (3x3 neighbouring A-scans across depth) exploit to differentiate localized physical reflectors from uncorrelated acoustic grain noise?
Answer:

24. How does spatial clustering with 8-connectivity and circumferential angle wrapping assemble individual A-scan gate crossings into discrete indications, and how does it compute peak amplitude and physical indication length?
Answer:

25. Why does UTriage strictly omit any automated part release or acceptance functionality within its code, user interface, and generated reports?
Answer:

## Phase 8 - generalization (answer these yourself)

1. Why does a naive random split across scan slices or extracted patches cause severe data leakage when multiple training and test examples originate from the same physical crack specimen?
Answer:

2. What shared background characteristics, such as transducer coupling variations, plate geometry echoes, or specimen grain structure, allow a model to memorize a specific specimen rather than learning general flaw acoustics?
Answer:

3. How does a leave-one-crack-out evaluation protocol isolate specimen memorization, and why is holding out an entire physical flaw required to establish an honest detection rate?
Answer:

4. What are the statistical limitations of leave-one-crack-out validation when the total pool of physical flaw specimens is small, and why does this prevent rigorous estimation of probability of detection curves?
Answer:

5. Why do ultrasonic classifiers trained to high nominal accuracy on a single laboratory dataset often suffer complete performance collapse when evaluated on external scans from different inspection systems?
Answer:

6. What differences in inspection hardware, center frequency, digitizer sampling rate, and probe beam profiles must be harmonized before an external cross-dataset test can be considered meaningful?
Answer:

7. Which physical acoustic phenomena, such as probe inclination, surface roughness, grain anisotropy, and couplant layer variations, are commonly omitted in synthetic simulators and cause synthetic-trained models to fail on plant scans?
Answer:

8. How do transient inspection artefacts, such as water path turbulence or front-wall aeration bubbles, trigger false calls in an anomaly detector trained solely on flaw-free reference scans?
Answer:

9. How should an inspector set the reconstruction error threshold for an unflawed-baseline model to guarantee an acceptable false-call rate without masking subtle, low-amplitude flaw indications?
Answer:

10. How do expected calibration error and reliability diagrams reveal overconfidence in ultrasonic indication triage, and why is an uncalibrated output probability dangerous when setting review thresholds?
Answer:
