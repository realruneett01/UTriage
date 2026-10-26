# 02 — The conventional rules engine and the data-quality watchdog

> This is the **safety backbone** and the **baseline** the ML must beat. Acceptance limits in
> `configs/acceptance.yaml` are **placeholders** (`UNVERIFIED — Ruri to check`) until checked
> against a legitimate copy of the governing standard.

## 1. Finding the interface echo (gate synchronisation)

The water path varies a little from A-scan to A-scan, so gates are set **relative to the
interface echo**, not to a fixed time. For each A-scan we take the envelope `e(t) = |s(t) + j H{s}(t)|`
(Hilbert transform `H`) and define the interface time as the first sample where `e(t)` crosses
a threshold `θ_if` (a fraction of full screen height):

    t_if = min { t : e(t) ≥ θ_if }.

If no sample crosses, the A-scan has **no interface** — a data-quality fault.

## 2. Depth and gates

Metal depth for time `t`: `z(t) = (t − t_if) · c_m / 2`.

- **Front-surface dead zone** `[0, z_dead)`: the interface echo rings here, so flaws are not
  evaluated (a known limitation of single-sided immersion UT).
- **Flaw gate** `[z_dead, D − z_back)`: where indications are evaluated.
- **Back-wall gate** `[D − z_back, D + z_back]`: used to check the back-wall echo is present.

## 3. Distance-amplitude correction (DAC)

A reference reflector (the FBH of the calibration block) echoes less strongly the deeper it is
(attenuation, beam spread, focus). The DAC curve `A_ref(z)` is measured on the block at several
metal paths and interpolated. An indication's amplitude is then expressed **relative to the
reference at the same depth**:

    a%(z) = 100 · e_peak(z) / A_ref(z)     [% of reference, "% DAC"]

100 % means "as big as the reference FBH at that depth". In dB: `20 log10(a%/100)`.

## 4. Signal-to-noise ratio (SNR) per indication

Noise is measured **locally**, because grain noise changes with depth and position. For an
indication with peak `e_peak` at depth `z₀`, take the envelope samples of the same A-scan and
its neighbours inside the window `|z − z₀| ≤ w_noise`, **excluding** `|z − z₀| ≤ w_excl` (the
echo itself). The noise level is the root-mean-square of those samples, `n_rms`. Then

    SNR_dB = 20 · log10(e_peak / n_rms).

## 5. Clustering

One reflector shows up in several neighbouring A-scans (the beam has a width). We threshold the
C-scan of `a%` at the recording level and join neighbouring A-scans (8-connectivity on the
axial × angle grid, with the angle axis wrapping round at 360°) into one **indication**. Each
indication reports: position of its peak, depth, peak `a%`, SNR, axial and circumferential
extent (length) and the number of A-scans.

## 6. Categories

Every indication gets exactly one category (limits come from `acceptance.yaml`):

| Category | Rule (placeholders) |
|---|---|
| rejectable | `a% ≥ reject_pct`, or length ≥ `max_length_mm`, or too many indications within `multiple_window_mm` |
| reportable | `record_pct ≤ a% < reject_pct` |
| insignificant | `a% < record_pct` |
| data-quality fault | below recording level and where the watchdog flags the data as unreliable. A fault never downgrades a rejectable or reportable indication; it adds a re-scan reason |

## 7. Data-quality watchdog

| Fault | Test |
|---|---|
| interface missing | no interface crossing in the A-scan |
| interface unstable / water-path drift | `|t_if − median(t_if)|` above a tolerance (in mm of water) |
| back-wall drop (loss of back reflection) | back-wall peak below a fraction of the bar's median back-wall peak |
| noise above limit | flaw-gate median envelope (robust noise level) above `noise_limit` × its bar median |
| coverage gap | A-scan is all (near) zero |
| saturation | any sample at the receiver's full scale |
| bubble / early echo | an echo before the interface echo |

Each fault is reported with its axial and angular location, so the re-scan can target it.

## Questions for Ruri
1. Why are gates synchronised to the interface echo and not to the trigger?
2. Why does the same 1 mm FBH give a smaller echo at 30 mm than at 15 mm (name three reasons)?
3. Why is SNR measured locally and not over the whole bar?
4. Why is loss of back reflection a reason to reject or re-scan, even with no flaw echo?
