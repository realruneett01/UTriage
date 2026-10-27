# 08 — Full Matrix Capture (FMC) and Total Focusing Method (TFM)

> Plain-language notes on synthetic aperture imaging for ultrasonic linear arrays.
> Output from simulated arrays is labelled **Synthetic data**.

## 1. What are FMC and TFM?

Conventional phased-array UT applies electronic time delays during transmission and reception
to form a single beam focused at a specific angle and depth. To inspect a full volume, the beam
must be swept through multiple angles (sectorial scan) or focal depths.

**Full Matrix Capture (FMC)** separates data collection from beamforming:
1. Element 1 transmits an omnidirectional wave; every element $1 \dots N$ records the reflected A-scans.
2. Element 2 transmits; every element records again.
3. This repeats until all $N$ elements have fired.

The result is a 3D data cube of raw A-scans:
$$\text{FMC}[tx, rx, t] \quad \text{of shape } (N, N, N_t)$$
No focusing happens in hardware. All spatial acoustic information is captured and stored.

**Total Focusing Method (TFM)** is the reconstruction algorithm applied to FMC data. It synthesizes
a focus at *every single pixel* in a target grid by calculating the exact two-way travel time
from each transmitter to the pixel and back to each receiver.

---

## 2. The Delay Equation

For a linear contact array placed along the specimen surface ($z = 0$):
- Transmitter $tx$ is at position $(x_{tx}, 0)$.
- Receiver $rx$ is at position $(x_{rx}, 0)$.
- Target pixel is at depth and lateral position $(x, z)$ with $z > 0$.

The one-way distances are:
$$d_{tx}(x, z) = \sqrt{(x - x_{tx})^2 + z^2}$$
$$d_{rx}(x, z) = \sqrt{(x - x_{rx})^2 + z^2}$$

Sound travels through the metal at compression velocity $c$. The total round-trip time of flight is:
$$t(tx, rx, x, z) = \frac{d_{tx}(x, z) + d_{rx}(x, z)}{c} - t_0$$
where $t_0$ is the start time of data recording.

Each transmitter-receiver pair contributes its signal amplitude evaluated at time $t$:
$$I(x, z) = \sum_{tx=1}^N \sum_{rx=1}^N s_{tx, rx}\big(t(tx, rx, x, z)\big)$$

If $t$ falls between sample indices, linear interpolation provides sub-sample precision.
Arrival times outside the recorded acquisition window contribute zero.

---

## 3. Why the Analytic Signal and Envelope?

Ultrasonic pulses are high-frequency oscillations: a carrier cosine modulated by a short pulse envelope:
$$s(t) = A(t) \cos(2\pi f_0 t + \phi)$$

If raw RF signals are summed directly across all pairs, slight sub-wavelength path length variations
cause destructive carrier interference. The resulting image exhibits high-frequency interference fringes
rather than a clean indication blob.

To remove carrier oscillations:
1. Compute the complex analytic signal along the time axis using the Hilbert transform:
   $$s_a(t) = s(t) + j \mathcal{H}\{s(t)\} = A(t) e^{j(2\pi f_0 t + \phi)}$$
2. Sum the complex analytic signals across all $tx$ and $rx$ pairs:
   $$S(x, z) = \sum_{tx=1}^N \sum_{rx=1}^N s_{a, tx, rx}\big(t(tx, rx, x, z)\big)$$
3. Compute the magnitude (envelope):
   $$I(x, z) = |S(x, z)|$$

This produces a non-negative image where the pixel intensity represents the true acoustic reflectivity
at $(x, z)$, free of RF cycle banding.

---

## 4. Resolution Limits

The spatial resolution of a TFM image is governed by diffraction and bandwidth:

### Lateral (Azimuthal) Resolution
Determined by array aperture $D = (N - 1) \cdot p$ (pitch $p$) and wavelength $\lambda = c / f_0$:
$$\Delta x \approx \frac{\lambda \cdot z}{2 D}$$
- The factor of 2 arises because both transmission and reception provide spatial diversity (synthetic aperture).
- Resolution degrades linearly with depth $z$, but is uniformly optimal across the entire reconstructed plane.

### Axial (Depth) Resolution
Determined by pulse duration and frequency bandwidth $B$:
$$\Delta z \approx \frac{c}{2 B} \approx \frac{c \cdot \tau}{2}$$
where $\tau$ is the effective pulse length in time.
A shorter tone burst (higher bandwidth $B$ or fewer cycles) sharpens axial resolution.
