# 04 — Bar-domain ML under titanium grain noise (Synthetic data)

## 1. What the model sees: DAC-normalised windows
For a candidate position (one A-scan, one depth `z₀`) we cut a window of `L = 128` samples
(1.28 µs at 100 MHz, about ±3 pulse widths) centred on `t(z₀)`, and divide it by the DAC
reference at that depth:

    x(t) = s(t) / A_ref(z₀)

So `x` is in units of "reference FBH amplitude". The rules engine's decision on the same window
is the classic one: `100 · max envelope(x) ≥ record_pct` (% DAC). Both methods see **identical
inputs**, so the comparison is fair.

- **1D input:** one channel (the centre A-scan).
- **2D input ("patch"):** 9 channels — the 3 × 3 neighbouring A-scans (axial × angle). A real
  reflector appears in neighbours with the beam's lateral weight `exp(−r²/2w²)`; grain noise is
  independent between neighbours in our simulator, so the patch gives the model a spatial cue.

## 2. Virtual-flaw augmentation (after Virkkunen et al.)
Training flaws are **implanted**: a flaw echo computed from the simulator's reflector model
(`fbh_amplitude` × pulse, inclusion facets, lateral weights) is added to a noise-only window cut
from a simulated bar. Signals in our model add linearly, so the implant is exact except for
receiver clipping. Unflawed training windows go through the **same code path** with a zero
implant, so the model cannot learn the insertion itself.

Test flaws are **not** implanted: they are planted in full simulated bars and cut out, so the
test set exercises the whole simulator path.

## 3. Matched false-call rate
Detection rates only mean something at a stated false-call rate (FCR). We measure the rules FCR
on the test noise windows, then set the ML threshold so that its FCR on the same windows is
equal (or lower). POD (hit/miss, a90/95) is then computed for both at that operating point.

## 4. Robustness (domain shift)
Train on noise scales 0.5–2.0 × nominal Ti-6Al-4V, 5 MHz. Test outside: noise 2.5×, IN718
(different velocity and attenuation), 10 MHz probe, and windows from artefact types never seen
in training.

## Questions for Ruri
1. Why normalise by DAC before the model sees the window?
2. Why must unflawed windows go through the same implant code path?
3. Why compare at a matched false-call rate and not at each method's default?
4. What would change with real grain noise that is spatially correlated?
