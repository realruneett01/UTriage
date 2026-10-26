# Safety Case: UTriage Advisory Decision Support

## 1. Scope

UTriage is an offline, advisory decision-support tool designed for ultrasonic immersion testing (UT) of aerospace bars and billets. 

Non-negotiable scope boundaries:
- **Advisory and offline only**: UTriage operates purely offline on exported scan files. It has no physical or digital connection to UT machine controllers, pulser-receivers, PLCs, or plant networks.
- **Never passes or accepts a part**: UTriage does not possess an automated acceptance or release function. It cannot clear or accept an inspection piece.
- **Inspector authority**: Every accept or reject decision remains the sole legal and technical responsibility of a certified human inspector (ASNT / NAS 410 Level II or Level III).

## 2. Hazards, Mitigations, and Residual Risk

| Hazard ID | Hazard | Cause | Mitigation in UTriage | Residual Risk |
|---|---|---|---|---|
| **H1** | Missed flaw | ML detector produces a false negative, fails to score a real flaw, or an echo is masked by local grain scattering. | **Union rule**: Fusion operates strictly as a union (`triage/fusion.py`). The ML model can only add candidate flags, never remove or suppress a rules flag. Enforced by property-based testing over 10,000 cases (`test_fusion.py`). Final disposition remains with the certified inspector. | Flaws that fall below both conventional rule thresholds and ML sensitivity, or flaws located within physical transducer dead zones. Controlled by standard calibration blocks, gated physical procedures, and manual inspector evaluation. |
| **H2** | Over-trust / automation bias | Inspector relies uncritically on ML rankings or scores, skims triage output, or assumes unranked regions are defect-free. | **Inspector authority**: System enforces that UTriage never outputs an acceptance decision. **Review ranking** (`triage/fusion.py:rank`) orders indications by severity (rejectable first, then uncertain, then borderline) to guide attention without certifying unranked regions. **Audit log of inspector decisions** (Phase 6) records every review action and override. | Human fatigue or cognitive bias during high-volume manual review. Controlled by standard operating procedures governing maximum inspector shift durations and Level III oversight. |
| **H3** | Data drift / domain shift (sim-to-real) | Divergence between synthetic training data (or public weld data) and real alloy bar production scans (grain size variation, acoustic attenuation, probe wear). | **Conformal uncertainty**: Deep ensemble and conformal prediction flag predictions with high uncertainty as `inspector review: uncertain`. **Drift monitoring on noise statistics**: System requirement (planned) to track baseline grain noise RMS, water-path stability, and interface amplitudes across scans and alert on distribution shifts. **Union rule**: Rules baseline remains unaffected by ML drift. | Novel or unmodeled defect morphologies not represented in calibration or training distributions. Controlled by mandatory conventional calibration against ASTM reference blocks. |
| **H4** | Data-quality fault unnoticed | Scan corruption (interface echo missing, water-path drift, back-wall echo loss, saturation, coverage gap, excessive noise) obscures defects. | **Data-quality watchdog** (`rules/quality.py`): Scans every A-scan before evaluation, maps connected fault regions, and displays a prominent **re-scan banner**. **Safety fix in `rules/acceptance.py`**: Data-quality faults never downgrade rejectable or reportable indications; a fault only adds a re-scan reason (`data quality fault at this location (re-scan)`). Only sub-recording indications become fault regions. | Intermittent hardware or acoustic anomalies falling just below watchdog threshold triggers. Controlled by pre-scan and post-scan reference block verification runs. |
| **H5** | Software defect in fusion | Software bug, race condition, or logic error in fusion logic drops a rule indication or inverts ranking priorities. | **Functional immutability**: Fused representations use immutable frozen dataclasses (`TriageItem`, `Indication`, `MlFinding`). **Formal property test**: Property-based verification running 10,000 randomized cases (`test_fusion.py` / `rules_flags_preserved`) guarantees every rule flag is preserved unchanged. | Compiler, interpreter, or runtime library defects. Controlled by comprehensive test suites, deterministic builds, and strict dependency pinning. |
| **H6** | Misuse as an acceptance tool | Operators or production management treat triage summaries as automated part clearance or bypass human inspection. | **Architectural omission**: Software contains no mechanism or API to issue a "part pass" or "part accept" certificate. All UI screens, CLI outputs, and reports display prominent advisories: "Advisory triage — not an acceptance record". **Audit logging** (Phase 6) tracks inspector sign-offs. | Procedural non-compliance outside the software boundary. Controlled by plant quality management systems, ASNT/NAS 410 compliance audits, and customer oversight. |

## 3. What Real Qualification Would Require

Deploying automated or AI-assisted ultrasonic triage in aerospace production requires formal qualification under applicable industry standards (such as AMS-STD-2154, ASTM E2375, and MIL-HDBK-1823A). Real qualification would require:

1. **Empirical POD study on real flaws**:
   - A rigorous Probability of Detection (POD) study conducted on actual, full-scale representative bars of the target material (e.g., titanium alloy, nickel-base superalloy).
   - A statistically sufficient sample of **independent physical flaws** (~30+ independent flaws per `docs/study/03-pod.md`), rather than synthetic simulations or repeated scans of a few defects.
2. **Certified Level III sign-off**:
   - Formal technical review, procedure drafting, and written approval by a certified ASNT / NAS 410 Level III NDT examiner.
3. **Customer and regulatory approval**:
   - Formal qualification and written acceptance by the aerospace prime contractor, defense customer, and regulatory airworthiness authorities.
4. **Configuration and quality control**:
   - Strict software configuration control, validated calibration blocks (ASTM E127 / ASTM E428 flat-bottom hole standards), documented system stability procedures, and auditable parameter change logs.

## 4. Current Qualification Status and Limitations

- **Not qualified**: UTriage is **NOT qualified** for aerospace or defence production inspection.
- **No plant validation**: UTriage has **NO plant validation**, has not been tested on plant equipment, and has not been validated on MIDHANI production data or any live industrial line.
- **Dataset limitations**: Current development and testing rely exclusively on **synthetic simulated data** generated by the acoustic model, plus a public research weld dataset (ML-NDT: 316L pipe weld with 3 physical cracks).
- **Verifiable metrics only**: No performance metrics or physical constants are invented. All quantitative figures and evaluation results must be referenced directly from committed results files (`results/*.json`).
