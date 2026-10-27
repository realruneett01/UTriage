# Phase 8 — Data expansion & generalization hardening (owner spec, 2026-10-27)

All CONTRIBUTING.md rules apply: never invent numbers, datasets or licences; every reported number comes
from `results/*.json`; synthetic and real results are reported separately.

## Context
- Public ultrasonic NDT data is scarce (PNNL/NRC review ML22284A071: most studies about 1,000 signals
  or fewer, few specimens, one material, one frequency; poor generalization; no common datasets).
  Many papers give data "on request" (e.g. Tunukovic et al., Ultrasonics 2024).
- Generalization must come from: honest evaluation; every compatible public real dataset used mainly
  as an external test; a physics simulator with domain randomization; anomaly detection trained on
  flaw-free scans.
- Goal: a model whose generalization is measured, bounded and honestly reported, with known failure modes.

## Step 0 — Fix what is known to be wrong
- 0.1 `_decide_kept_model` (experiments/run_phase4.py): compare on the same metric and direction (POD
  lower bound) with the documented margin rule; unit test that fails on the old behaviour.
- 0.2 Leakage audit for ML-NDT: source crack per image from `.jsons` (`size` field); write
  `results/leakage_audit.json` (per split: images per source crack; same crack in train and test?).
- 0.3 Re-evaluate the real-data model on (a) the official `data/validation` only and (b)
  leave-one-crack-out (3 folds; train on 2 cracks + unflawed, test on held-out crack + held-out
  unflawed; unflawed split by source batch). Per fold: detections, false calls, hit/miss POD if
  fittable else Clopper-Pearson 95 % lower bound, with counts.
- 0.4 CPU latency on this laptop (median, p95 over 500 images, batch 1, threads recorded); training
  machine in `results/hardware.json`.
- 0.5 Re-run Phase 4 and 5 experiments; regenerate README tables and resume bullets from new results
  only; mark superseded numbers in PROGRESS.md.

## Step 1 — Public data hunt (licence-checked)
- 1.1 Search Zenodo, Mendeley Data, figshare, IEEE DataPort, Kaggle, Strathclyde Pure (e.g. DOI
  10.15129/086404bd-eb69-429b-978c-2c35cdbfcf87, 316L weld lack-of-fusion FMC, CC BY 4.0), Iowa State
  CNDE, GitHub, Papers-with-Code.
- 1.2 `data/registry.yaml`: every candidate (accepted or rejected): name, URL/DOI, licence, material,
  technique, frequency, data form, specimens, distinct physical flaws, label type, size, download
  status, relevance 0-3 with reason.
- 1.3 Reject (with reason): non-metal, guided-wave SHM, unclear licence, unlabelled unless useful for
  anomaly/pretraining.
- 1.4 Downloader per accepted dataset into `data/external/<name>/` (gitignored), checksums.
- 1.5 FMC: minimal TFM imager `src/utriage/imaging/tfm.py` (numpy; maths in `docs/study/08-tfm.md`);
  simulated point scatterer peak within 1 pixel.
- 1.6 `results/data_inventory.md`: total distinct physical flaws across accepted real data.

## Step 2 — Simulator at scale with domain randomization
- 2.1 Randomize material, frequency/bandwidth, focal depth, water path, diameter, gain, reflector
  type/size/depth/orientation, inclusion irregularity, artefacts; each range sourced or UNVERIFIED.
- 2.2 On-the-fly generation in the data loader, seeded workers, RAM under ~3 GB.
- 2.3 Hold out entire parameter regions (5 -> 10 MHz, 2 materials -> third, noise above range).

## Step 3 — Models (small)
- 3.1 Supervised: current architectures, simulated + real LOCO training folds.
- 3.2 Anomaly: conv autoencoder on flaw-free scans, reconstruction error after wall gating
  (Tunukovic et al., Ultrasonics 140 (2024) 107313); `docs/study/08-anomaly.md`.
- 3.3 Optional self-supervised pretraining, kept only if cross-dataset results improve.
- 3.4 Physics-aware augmentation, early stopping on group-held-out validation, weight decay,
  3-5 seed ensemble.
- 3.5 `results/fit_diagnostics.json` + plots: learning curves, data-size curve (10/25/50/100 %),
  train-vs-test gap; stated under/overfitting rule per model.

## Step 4 — Generalization protocol
`results/generalization_matrix.json` + figure. Rows: rules, supervised, anomaly, fused. Columns:
ML-NDT LOCO, ML-NDT validation, cross-dataset (test only), simulator held-out regions, artefact
stress. Metrics: detection rate + CP 95 % CI, false calls/scan, a90/95 where fittable, ECE +
reliability diagram, counts (images and distinct flaws).
- 4.1 Union property holds; fused POD vs rules per column.
- 4.2 `results/failures/` + `results/failure_modes.md` (near-wall, low SNR, unseen geometry,
  artefact, label error).
- 4.3 `results/claims.md`: claim, evidence file, scope limit.

## Step 5 — Docs and resume
- 5.1 README: generalization matrix, claims link, "Evidence limits" box (distinct real flaws).
- 5.2 Resume bullets only from `results/claims.md`. Allowed: "independent prototype", "public
  dataset", "simulated", "advisory", "leave-one-crack-out", "cross-dataset". Forbidden: "perfect",
  "100%", "for MIDHANI", "validated on plant data", "qualified", "deployed", "generalizes to all".
- 5.3 10 new interview questions (leakage, LOCO, cross-dataset, sim-to-real, anomaly, calibration).

## Gate
- Leakage audit done; leaked numbers marked superseded everywhere.
- `python experiments/run_phase8.py` regenerates matrix, fit diagnostics, claims from scratch.
- All real-data results use group-held-out splits (crack, dataset or parameter region).
- `results/data_inventory.md` states distinct real flaws.
- Resume bullets: no forbidden words; every number traces to `results/claims.md`.
- All tests pass, including the model-choice test and the union property test.

## Compute
Laptop i5-1135G7, 8 GB RAM; stream and generate on the fly. Training over ~2 h on CPU may use the
external GPU (record in `results/hardware.json`); inference is CPU-measured.
