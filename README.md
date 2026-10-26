# UTriage — AI-assisted ultrasonic indication triage for aerospace bar inspection

Advisory, offline decision support for immersion UT of titanium and nickel-alloy bars.
**UTriage never accepts or passes a part. Every decision belongs to the certified inspector.**

> Status: work in progress (see `PROGRESS.md`). The full README with results tables is
> generated in Phase 7.

## Acceptance limits are placeholders
Every value in `configs/acceptance.yaml` is a **PLACEHOLDER — UNVERIFIED**. None is taken from
AMS-STD-2154, ASTM E2375 or any other standard; those standards are paywalled and are not
reproduced here. The rules engine runs with these placeholders only so that the software can be
built and tested. Results that depend on them must not be read as compliance with any standard.

## Synthetic data
Everything produced by the simulator is **Synthetic data** and is labelled so in every file,
plot and report.
