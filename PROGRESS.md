# PROGRESS.md

## Done
- **Phase 0** (tag `phase-0`): package skeleton, `pyproject.toml`, ruff/pytest config, `.gitignore`,
  ADR 0001 (safety design), ADR 0002 (prior art), `docs/study/00-ut-primer.md` (questions only),
  `scripts/fetch_mlndt.sh`, smoke tests, `python -m utriage --help`.
- **Phase 1** (tag `phase-1`): `physics/simulator.py` (A-scan/C-scan helical bar model, grain noise,
  FBH + inclusion reflectors, 5 artefact kinds), `physics/calibration.py` (E127-style block, DAC),
  `physics/dataset.py` (generator, npz + JSON truth), `physics/figures.py`, `utriage simulate` CLI,
  configs (`materials.yaml`, `probe.yaml`), `docs/study/01-simulator-physics.md`, ADR 0003,
  sample bar in `data/samples/ti64/`. Gate command runs: `python -m utriage simulate --material ti64 --bars 5`.

## In progress
- Phase 2: rules engine, acceptance categories and watchdog merged; baseline experiment and
  `evaluate` CLI next.

## Gate items only Ruri can close
- Phase 0: answer `docs/study/00-ut-primer.md` yourself.
- Phase 1: explain every term of the A-scan equation on paper (`docs/study/01-simulator-physics.md`).

## Open questions (ask the MIDHANI contact)
- U1: Is UT evaluation or review time an actual queue in their flow? How long does UT and test clearance take per lot?
- U2: Which acceptance standard and class applies per product (AMS-STD-2154 / ASTM E2375 style)?
- U3: What data does their immersion UT system export (A-scan or C-scan files, format)? Can we get a sample?
- U4: How often is a bar re-scanned because of noise, a lost back-wall echo or a borderline call?

## Assumptions
- MIDHANI runs automated immersion UT of bars 8–300 mm diameter including titanium, 0.8 mm FBH sensitivity (from their 2021 tender corrigendum).
- Development machine differs from the original target: RTX 5070 Ti Laptop GPU (12 GB, CUDA), 32 GB RAM, Python 3.14
  (owner's instruction). Training uses the GPU; see ADR 0004 when Phase 3 lands. Inference stays CPU-capable.
- ML-NDT data lives outside OneDrive (`C:\Users\realr\utriage-data`), linked by junctions at
  `data/external` and `data/synthetic`, so the 2.7 GB is never synced.
- All material constants, simulator noise levels and acceptance limits are UNVERIFIED placeholders.
