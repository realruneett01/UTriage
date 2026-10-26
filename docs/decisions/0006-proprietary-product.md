# 0006. Proprietary product decision and intellectual property strategy

- Status: Accepted
- Date: 2026-10-27

## Context

UTriage was initially scoped with open research components. The owner has decided that UTriage is a proprietary product to be proposed to industrial plants as a commercial solution, while also serving as a portfolio project shown on a resume. The repository will remain private.

Transitioning from an open portfolio project to a proprietary commercial offering introduces legal, licensing, and distribution considerations:
- Prior art: US 9,927,404 B2 ("Phased array billet data evaluation software", United Technologies / RTX) covers software evaluation of billet UT data.
- Third-party data licensing: the ML-NDT dataset (Virkkunen et al., arXiv:1903.11399) is licensed under LGPL-3.0.
- Asset protection: commercial distribution requires legal and technical safeguards against unauthorized copying.
- Safety boundaries: commercial positioning must not compromise safety constraints.

## Decision

UTriage is designated as a proprietary, closed-source product. The repository remains private under a proprietary copyright notice with all rights reserved.

Commercial engagements and demonstrations will be governed by non-disclosure agreements (NDAs) and proprietary licensing.

## Consequences

1. **Freedom-to-operate (FTO) review:**
   A formal freedom-to-operate review by a qualified patent attorney is required before any commercial deployment, specifically examining the claims of US 9,927,404 B2 (documented in `docs/decisions/0002-prior-art.md`). The patent claims have not been formally analysed by counsel (UNVERIFIED — Ruri to check). No commercial deployment will take place without legal clearance.

2. **Model weights trained on LGPL-3.0 data:**
   Whether neural network weights trained on LGPL-3.0 licensed data constitute a derivative work under copyright law remains an unsettled legal question. To avoid legal uncertainty, shipped commercial products should use models trained exclusively on proprietary owner or plant data. The model trained on ML-NDT data remains strictly research evidence demonstrating methodology.

3. **Intellectual property protection strategy:**
   Software protection relies on a combination of:
   - Private repository access controls.
   - Proprietary license terms forbidding unauthorized copying and distribution.
   - Non-disclosure agreements (NDAs) for plant pilots and customer evaluations.
   - Compiled or protected binary distribution (see ADR 0007, written in parallel).
   It is recognized that no technical software protection makes code completely uncopyable; operational and contractual protections form the primary defense.

4. **Safety architecture unchanged:**
   The safety-first architecture established in `docs/decisions/0001-safety-design.md` remains unchanged:
   - The conventional rules engine remains the baseline authority.
   - Machine learning can only add flags, never remove a rules flag.
   - The certified Level II/III inspector retains sole authority for accept or reject decisions. UTriage remains advisory and never passes a part.
   - The application runs offline without machine controller connections.
