# 0001. Safety design

- Status: Accepted
- Date: 2026-10-26

## Context

In ultrasonic testing (UT) of aerospace and defence components, critical forgings, billets, and bars must meet stringent qualification requirements. Automated immersion UT systems evaluate reflections against calibrated acceptance thresholds. False calls occur due to material grain noise, while missed flaws present catastrophic risks.

Deploying machine learning (ML) models in safety-critical non-destructive testing (NDT) introduces specific risks:
- Aerospace and defence qualification: an unqualified ML component cannot clear a part or override code-based inspection standards.
- Accountability: legal and standard compliance requires a named, certified inspector to sign off on every inspection decision.
- Automation bias: human inspectors may over-rely on automated recommendations if the software appears to accept parts independently.

## Decision

The safety-first architecture of UTriage is defined by non-negotiable architectural boundaries:
- The rules engine stays in charge: conventional code-based evaluation (gates, distance-amplitude correction, reference comparison, and SNR) remains the baseline authority.
- ML can only add flags, never remove a rules flag: the fusion mechanism operates strictly as a union where ML can highlight additional candidate indications, but can never suppress or dismiss an indication flagged by the rules engine.
- Certified inspector authority: the certified inspector makes every accept or reject decision. UTriage is a decision-support tool and never passes or accepts a part.
- Advisory and offline only: the application executes purely offline on exported scan files with no connection to UT machine controllers, PLCs, or plant networks.

## Consequences

- Property-based testing: a property-based test must enforce that fusion never drops a rules flag (Phase 5).
- Reliability metric: detection capability is judged by probability of detection (POD) curves and a90/95, not classification accuracy.
- Operator workflow: the tool provides advisory triage assistance without altering legal liability or qualification boundaries.
