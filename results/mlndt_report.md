# ML-NDT Real-Data Crack Detection & POD Report

> Superseded: the random batch split shares all physical cracks between train and test (results/leakage_audit.json); see results/loco_metrics.json for leave-one-crack-out results.

**Dataset Label:** Real data (ML-NDT, Virkkunen et al. arXiv:1903.11399; data LGPL-3.0, not redistributed)
**Device:** NVIDIA GeForce RTX 5070 Ti Laptop GPU | **Parameters:** 1226850 | **Kept Seed:** 2
**Flaw Size Metric:** equivalent size

## Dataset Splits
| Split | Batches | Images |
|---|---|---|
| Train | 179 | 17810 |
| Test | 20 | 2000 |
| Validation | 2 | 200 |

## Detection & POD Evaluation
| Split | POD Metrics | False Calls |
|---|---|---|
| Test | null (Data is separable: only hits or only misses present.) | 0/778 (0.00%, 95% UB: 0.38%) |
| Validation | null (Data is separable: only hits or only misses present.) | 0/89 (0.00%, 95% UB: 3.31%) |

## Flaw Detection (Binomial Bound)
| Split | n_flawed | n_hit | Size Range | POD Lower Bound |
|---|---|---|---|---|
| Test | 1222 | 1222 | 0.91 - 8.60 | 0.998 |
| Validation | 111 | 111 | 0.94 - 8.51 | 0.973 |

a90/95 is not estimable by the logistic model because every tested flaw was detected; the paper reports a90/95 0.9 mm; numbers are not directly comparable.

## Latency Benchmark
| Configuration | Median Latency (ms/image) |
|---|---|
| gpu_batch_1 | 0.996 |
| gpu_batch_64 | 0.601 |
| cpu_batch_1 | 21.732 |

## Comparison with Prior Art (Virkkunen et al. arXiv:1903.11399)
- **Paper Reference:** a90/95 of 0.9 mm with 0 false calls.
- **Our Model (Test):** POD: null (Data is separable: only hits or only misses present.); False calls: 0/778 (0.00%, 95% UB: 0.38%).

## Limitations
- **Specimen Count:** Dataset comprises augmented scans from only 3 physical cracks.
- **Component Geometry:** Planar weld inspection, not a rotating titanium bar.
- **Data Independence:** Augmented copies of the same cracks so not independent.
- **Scope:** Proves the method not MIDHANI applicability.
