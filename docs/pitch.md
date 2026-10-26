# Proposal: Offline Ultrasonic Indication Triage & Data-Quality Decision Support

**To:** Head of Quality Assurance & UT Level III Examiner  
**From:** UTriage Technical Team  
**Subject:** Feasibility Demonstration of Offline Indication Triage for Immersion UT of Aerospace Alloy Bars  
**Classification:** Proprietary / Under Non-Disclosure Agreement (NDA)  

---

## 1. Executive Summary & The Problem

Immersion ultrasonic testing (UT) of aerospace-grade titanium and nickel-base superalloy round bars and billets (8–300 mm diameter, up to 6 m length, per public plant specifications [V1]) is governed by stringent defect-detection thresholds, typically requiring sensitivity down to 0.8 mm flat-bottom-hole (FBH) equivalence [V1]. 

In manufacturing these alloys, coarse grain and colony microstructures scatter acoustic energy, generating high-amplitude background grain noise [V2]. This background scattering leads to two major operational friction points on the shop floor:
1. **Frequent false calls & borderline calls:** Grain-noise peaks frequently breach recording thresholds, forcing Level II and Level III inspectors to spend hours manually inspecting individual A-scans, determining local signal-to-noise ratios (SNR), and adjudicating borderline indications [V2].
2. **Avoidable re-scans & downstream delays:** Minor acoustic coupling instabilities, water-path drift, or loss of back-wall reflection that are detected only during post-scan review require re-chucking and re-scanning entire bars, compounding order cycle times and inventory holding [V3].

### Questions We Need to Confirm With Your Team
To assess whether automated triage provides meaningful operational relief at your facility, we need to clarify four practical questions:
- **[U1] Review bottleneck:** Does manual A-scan review and borderline indication triage represent a measurable queue in your bar-finishing throughput?
- **[U2] Acceptance specifications:** Which governing acceptance standards and classes apply to each alloy product line (e.g., AMS-STD-2154 or ASTM E2375 classes)? *(Note: Our development models use unverified placeholder limits; no paywalled standard values are assumed).*
- **[U3] Acquisition export capabilities:** What data formats (digitized RF A-scans, rectified envelopes, or vendor-proprietary C-scan streams) can your immersion tank consoles export?
- **[U4] Re-scan frequency:** What proportion of bar inspections currently require a re-scan due to grain-noise ambiguity, back-wall drop, or acoustic coupling anomalies?

---

## 2. The Proposed Solution: Advisory Decision Support

UTriage is a software tool developed specifically for offline immersion UT evaluation. It operates under a strict, non-negotiable safety architecture designed to satisfy aerospace quality governance:

1. **The rules stay in charge:** Conventional evaluation logic (interface gating, Distance-Amplitude Correction [DAC], reference FBH comparison, and local SNR measurement) executes first. Every indication flagged by standard acceptance rules remains flagged.
2. **AI can only add, never remove:** The machine learning detector is strictly restricted to flagging candidate indications that conventional thresholding might miss in high grain noise. It possesses no authority or software mechanism to dismiss, suppress, or clear a rule flag.
3. **Inspector retains sole disposition authority:** UTriage is an advisory decision-support system. It contains no mechanism to certify or accept a part. Every accept, reject, or re-scan disposition remains the exclusive decision of the certified human inspector.
4. **Data-quality watchdog:** A dedicated watchdog continuously checks signal integrity across every scan line (monitoring interface stability, water-path drift, back-wall reflection retention, electronic saturation, and acoustic coverage), flagging acquisition flaws while the bar is still in the tank.
5. **Automated evidence & pre-filled reports:** Detected indications are clustered in 3D, ranked by severity, and presented alongside local SNR and calibrated uncertainty sets, pre-populating standard inspection records to eliminate manual transcription time.
6. **Strictly offline and air-gapped:** UTriage executes as a standalone Windows application on standard plant PCs. It requires no network connection, transmits zero telemetry, and **never connects physically or digitally to UT pulser-receivers, machine controllers, or PLCs.**

---

## 3. What We Request From Your Plant

To conduct a plant-specific feasibility evaluation without disrupting production, we request sample data exported per our specification in `docs/scan_export_template.md`:

1. **Calibration-block reference scans:** One immersion scan of an ASTM E127 / ASTM E428 calibration standard matching production bar alloy and diameter, containing reference FBH reflectors (e.g., 0.8 mm FBH) across evaluated metal paths.
2. **Representative bar sample scans:** Two to three anonymized scan files of production or test bars in target alloys (e.g., Ti-6Al-4V or IN718) exhibiting typical grain backscatter, including any known natural or seeded indications.
3. **Scan format specifications:** Exported data in plain-text CSV RF format, or raw vendor console export files accompanied by header documentation (format specifications detailed in `docs/scan_export_template.md`).

---

## 4. What You Get Back

Within two weeks of receiving the sample scans, we provide a comprehensive, data-driven offline evaluation report on your own material:

- **Offline POD-style comparison:** A comparative analysis evaluating conventional thresholding against the fused rules-plus-ML triage layer on your actual material scans (benchmarking detection sensitivity, indication clustering, and false-call behaviour per the methodologies documented in `results/bar_domain_metrics.json` and `results/triage_metrics.json`).
- **Data-quality audit:** A full spatial mapping of acoustic coupling stability, water-path variance, and back-wall retention across the sample bars, identifying any acquisition anomalies caught by the watchdog.
- **Inspector review demonstration:** An offline demonstration showing how the triage decision-support interface ranks indications and pre-fills inspection documentation for Level II review.

---

## 5. What This Proposal Is Not

To maintain total technical and commercial integrity, we are explicit about the boundaries of this evaluation:

- **Not qualified:** UTriage is **not qualified** under AMS-STD-2154, ASTM E2375, MIL-HDBK-1823A, or any aerospace prime specification. It cannot be used for production part clearance.
- **Current validation base:** Current performance figures have been established strictly on physics-based synthetic bar simulations (`results/bar_domain_metrics.json`, `results/rules_baseline.json`) and a public research dataset of a 316L stainless steel pipe weld containing three physical cracks (`results/mlndt_metrics.json`). We make no claims of prior plant qualification or MIDHANI production data validation.
- **Proprietary commercial software under NDA:** UTriage is proprietary commercial software. All discussions, data exchanges, and evaluations are conducted under a mutual Non-Disclosure Agreement (NDA). All plant scan data remains the exclusive property of the plant and will not be retained or redistributed.

---

## 6. Next Steps

If your quality engineering team is interested in exploring whether automated triage can reduce inspection review time and eliminate unnecessary re-scans, we invite your Level III examiner to review the accompanying technical note (`docs/technical-note.md`) and data export specification (`docs/scan_export_template.md`), and schedule an introductory technical briefing.
