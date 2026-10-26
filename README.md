# UTriage — AI-assisted ultrasonic indication triage for aerospace bar inspection

Advisory, offline decision support for immersion UT of titanium and nickel-alloy bars.
**UTriage never accepts or passes a part. Every decision belongs to the certified inspector.**

> Status: work in progress (see `PROGRESS.md`). The full README with results tables is
> generated in Phase 7.

## Product status
- **Advisory decision support:** UTriage provides offline indication ranking and triage evidence. It does not connect to machine controllers and never accepts or rejects a part.
- **Not qualified:** The software is not qualified under any aerospace or plant inspection standard.
- **Validation base:** Evaluated on synthetic data from the physics simulator and a public weld dataset (Virkkunen et al., arXiv:1903.11399).
- **Prerequisites for a plant pilot:**
  - Calibration-block scans from plant reference standards.
  - Sample bar scans in the target alloys and diameters.
  - UT system scan export format specifications.
  - Level III inspector review of rules, gates, and thresholds.

## Acceptance limits are placeholders
Every value in `configs/acceptance.yaml` is a **PLACEHOLDER — UNVERIFIED**. None is taken from
AMS-STD-2154, ASTM E2375 or any other standard; those standards are paywalled and are not
reproduced here. The rules engine runs with these placeholders only so that the software can be
built and tested. Results that depend on them must not be read as compliance with any standard.

## Synthetic data
Everything produced by the simulator is **Synthetic data** and is labelled so in every file,
plot and report.

## Licence
Proprietary software. Copyright (c) 2026 Ruri. All rights reserved.
Private repository. No permission to use, copy, modify, distribute or sublicense without prior written permission.
