# 03 — The real-data detector (ML-NDT)

## Goal
Decide, for one 256 × 256 phased-array ultrasonic image, "crack present?" and estimate its
equivalent size — fast enough to feel instant to an inspector (target: a few milliseconds per
image on the GPU, and well under 100 ms on a laptop CPU; measured, not assumed, in
`results/mlndt_metrics.json`).

## Architecture: a small residual CNN with two heads ("hybrid" output)
1. **Stem.** 3×3 convolution → batch norm → ReLU, then a **max-pool**. Virkkunen et al. argue
   that an early max-pool sized to the ultrasonic wavelength acts like envelope detection: it
   keeps the peak of each oscillation and drops the phase. Our pool size is a configurable
   parameter; which image axis is time in ML-NDT is **UNVERIFIED — Ruri to check** in the data
   description, so the default pool is square (2 × 2).
2. **Four residual stages** (two 3×3 convolutions with a skip connection), each halving the
   resolution and doubling the channels (w, 2w, 4w, 8w). Residual skips make a deeper net
   train reliably with few epochs.
3. **Global pooling** — average and max pooling concatenated. Max pooling keeps a small, strong
   crack tip; average pooling describes the background.
4. **Two heads:** a detection logit (crack yes/no) and a size regressor (equivalent size, only
   trained on flawed images). The size head gives the inspector a number to compare with the
   rules engine; detection is what POD is computed on.

Parameter count is small (about a million), so inference is one GPU kernel sequence per batch.

## Training
- Binary cross-entropy on the logit + 0.1 × smooth-L1 on size (flawed images only).
- AdamW, cosine learning-rate decay, early stopping on validation loss, fixed seeds.
- Mixed precision on the GPU. **3 random restarts** (the paper reports sensitivity to
  initialisation); the restart with the best validation loss is kept.
- Splits: the provided `training` batches, minus a **held-out test split of whole batches**
  never used for tuning; the provided `validation` batches for early stopping and selection.

## Honest caveat
The augmented training images come from **only 3 physical cracks** in a 316L weld. Held-out
batches still contain augmented copies of the same cracks, so held-out performance is **not**
evidence of performance on new cracks. This proves the *method*, not plant applicability.

## Questions for Ruri
1. Why does a max-pool on raw RF behave like an envelope detector?
2. Why concatenate average and max global pooling?
3. Why train the size head only on flawed images?
4. Why are held-out batches from the same 3 cracks not independent test data?
