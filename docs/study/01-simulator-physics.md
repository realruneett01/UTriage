# 01 — The A-scan model used by the simulator

> Everything here is a **simplified** model chosen so every term can be explained on paper.
> It is not a full wave simulation. Output is always labelled **Synthetic data**.
> The list of simplifications is in `docs/decisions/0003-simulator-assumptions.md`.

## 1. Geometry and time of flight

The probe sits in water at distance `w` (water path) from the bar surface and fires radially
into a round bar of diameter `D`. Sound travels at `c_w` in water and `c_m` in metal.

- Front-wall (interface) echo arrives at `t_if = 2 w / c_w`.
- An echo from depth `z` below the surface arrives at `t(z) = t_if + 2 z / c_m`.
- The back-wall echo (opposite side of the bar, through the centre) arrives at `t(D)`.

Pulse-echo means the sound travels there and back, hence the factor 2.

## 2. The pulse

A Gaussian-modulated cosine with centre frequency `f_c`:

    p(t) = exp(-t² / (2 s²)) · cos(2π f_c t)

Its spectrum is a Gaussian around `f_c` with standard deviation `σ_f = 1 / (2π s)`.
The −6 dB (half-amplitude) bandwidth is `B = 2 σ_f √(2 ln 2)`. Given the fractional
bandwidth `b` (e.g. 0.6 means 60 %), `B = b · f_c`, so

    σ_f = b f_c / (2 √(2 ln 2)),   s = 1 / (2π σ_f).

Wavelength in metal: `λ = c_m / f_c`.

## 3. Interface reflection and transmission

Acoustic impedance `Z = ρ c`. At normal incidence the pressure reflection coefficient from
medium 1 into medium 2 is

    R = (Z_2 − Z_1) / (Z_2 + Z_1).

Water → metal gives a large `R` (about 0.9), so the interface echo is the biggest signal.
Sound that enters the metal and comes back crosses the interface twice. The round-trip
pressure transmission is

    T_rt = T_12 · T_21 = 4 Z_1 Z_2 / (Z_1 + Z_2)² = 1 − R².

The back wall is a metal → water boundary; its reflection magnitude is also `|R|`.

## 4. Attenuation

Amplitude falls exponentially with metal path. Attenuation is given in dB per mm per MHz
(linear frequency law, a simplification):

    α_dB = a · f_MHz          [dB/mm]
    A(z) = 10^(−2 α_dB z / 20)   (round trip, z in mm)

## 5. Beam (focal) factor

A focused probe is strongest at its focal depth `z_f` in metal and weaker either side.
We use a Lorentzian-shaped on-axis factor with focal-zone half-length `z_R`:

    G(z) = 1 / sqrt(1 + ((z − z_f) / z_R)²).

Laterally, a reflector at lateral distance `r` from the beam axis is seen with weight
`exp(−r² / (2 w_b²))`, where `w_b` is the beam half-width (standard deviation).
For a reflector at depth `z` in a bar of radius `R_b`, its radius is `ρ = R_b − z`.
An angular offset `Δθ` becomes the arc `ρ Δθ`; an axial offset `Δx` stays `Δx`; so

    r² = Δx² + (ρ Δθ)².

## 6. Reflectors

- **Flat-bottom hole (FBH)**: echo amplitude proportional to the hole area, i.e. to `d²`:

      A_fbh = k_fbh · (d / 1 mm)² · T_rt · A(z) · G(z)

  `k_fbh` is a simulator gain constant (not a physical constant). The `d²` law is the
  classic far-field approximation for a small disc reflector.
- **Inclusion**: irregular, so the same law times a random factor `u ∈ [u_min, 1]`, and the
  echo is split into 2–3 pulses a fraction of a period apart (an uneven reflecting surface).

## 7. Grain noise

Many tiny grain boundaries scatter sound. With very many random scatterers per resolution
cell, their sum tends to a Gaussian random process (central limit theorem). We model it as
white Gaussian noise `n(t)` convolved with the pulse, then shaped by the same depth losses:

    g(t) = σ_g · (f_c / 5 MHz)² · [n ⊛ p](t) · T_rt · A(z(t)) · G(z(t))   for t > t_if

`σ_g` is a per-material grain-noise level (a model parameter). The `f²` factor encodes the
trend that grain noise rises with frequency (Rayleigh-region scattering grows faster, a
simplification).

## 8. Electronic noise, gain, and the receiver

    s(t) = 10^(gain_dB / 20) · [ interface + back wall + reflectors + grain noise ] + e(t)

`e(t)` is white Gaussian electronic noise with standard deviation `σ_e`. The receiver
clips at ±1 (full screen height), which is how saturation appears.

## 9. Scan geometry

The bar rotates (angle index `j`, `n_angle` steps over 360°) while the probe moves along
the bar (axial index `i`, step = scan index). The full data cube is `(n_axial, n_angle,
n_samples)`. The **C-scan** is the peak of the envelope inside a gate, per A-scan.

## 10. Calibration block (DAC)

A flat block with FBHs of one diameter at several metal paths. Each hole's peak echo gives
one point of the distance-amplitude correction (DAC) curve: amplitude versus depth.

## Questions for Ruri
1. Why is the interface echo so much bigger than any flaw echo?
2. Derive `T_rt = 1 − R²` from the two one-way transmission coefficients.
3. Why does doubling the FBH diameter give four times the amplitude in this model?
4. Why does grain noise get weaker with depth, but SNR of a flaw at the same depth may not?
