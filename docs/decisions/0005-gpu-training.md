# 0005. Train on the GPU; keep inference CPU-capable

- Status: Accepted
- Date: 2026-10-27

## Context
The project plan and CONTRIBUTING.md target an i5-1135G7 laptop with no GPU and CPU-only PyTorch. The owner
develops on a different machine (RTX 5070 Ti Laptop GPU, 12 GB; 32 GB RAM) and asked for the
latest stack (Python 3.14, CUDA 13 PyTorch wheels) and GPU training. The project plan already allows a GPU
for training only if it is recorded in an ADR.

## Decision
- Training uses CUDA when available (`TrainConfig.device = "auto"`, bf16 autocast).
- Every model must still run on CPU; inference latency is measured on both and reported in
  `results/*.json`, so the CPU-only target machine stays supported.
- Python 3.14 with `torch 2.14.1+cu130`. `pyproject.toml` keeps `requires-python >= 3.11` and
  `torch` as an optional `ml` extra, so CPU-only installs still work.

## Consequences
- Training times in results are GPU times and are labelled with the device name; they are not
  the "under 2 hours on the laptop" budget from the project plan.
- Fixed seeds are kept, but GPU kernels can be non-deterministic; reruns may differ slightly.
