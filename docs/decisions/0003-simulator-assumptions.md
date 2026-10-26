# 0003. Simulator assumptions and physics simplifications

- Status: Accepted
- Date: 2026-10-26

## Context

Phase 1 of UTriage implements an immersion ultrasonic testing (UT) simulator for rotating round metal bars (`src/utriage/physics/simulator.py`). The simulator generates synthetic RF A-scans, peak-detected C-scans, and data-quality watchdog faults with known ground truth.

Developing and testing detection algorithms, the rules engine, and ML classifiers requires high volumes of diverse scan data with varying defect sizes, depths, and noise levels. Full elastodynamic numerical solvers (such as finite element analysis or finite-difference time-domain methods) require hours or days per bar scan and demand extensive computing infrastructure. UTriage must execute offline and train on a standard development laptop (i5-1135G7, 8 GB RAM, no GPU).

A computationally tractable 1D ray-convolution model with analytical beam profiles and heuristic noise shaping is implemented. Every simplification must be explicitly recorded to define the operational limits of synthetic data and prevent unjustified claims.

## Decision

We accept a simplified physical model for synthetic bar immersion UT generation. Every simplification implemented in `src/utriage/physics/simulator.py` is documented below with what we do, what reality does, and the effect on results:

1. **No full-wave simulation (no FE or FDTD)**
   - *What we do:* A 1D ray-convolution model combines analytic arrival times, a Gaussian-modulated cosine pulse, and depth-dependent loss factors.
   - *What reality does:* Full 3D elastodynamic wave equations govern sound propagation, producing complex diffraction, wavefield interactions, and scattered coda.
   - *Effect on results:* Fast scan generation (seconds per bar), but signals lack full-wave diffraction, edge scattering, and elastodynamic wavefield complexity.

2. **No mode conversion or shear waves**
   - *What we do:* Only longitudinal pressure waves are modelled using normal-incidence acoustic impedance mismatch ($R = (Z_2 - Z_1) / (Z_2 + Z_1)$).
   - *What reality does:* Wave reflections at curved boundaries, defect edges, and grain boundaries convert longitudinal energy into shear waves, surface waves, and creeping waves.
   - *Effect on results:* Mode-converted ghost indications, shear-wave echoes, and Rayleigh surface waves are absent from simulated A-scans.

3. **Normal incidence only, curved bar surface treated as flat for transmission**
   - *What we do:* The probe is assumed radially normal to the bar surface. Flat-interface reflection and round-trip transmission ($T_{rt} = 1 - R^2$) are applied uniformly across the circumference.
   - *What reality does:* The cylindrical surface acts as an acoustic lens, refracting non-normal rays and defocusing or focusing the sound beam depending on curvature and velocity ratios.
   - *Effect on results:* Cylindrical refraction, focal distortion, and circumferential beam curvature effects are omitted; transmission efficiency is constant.

4. **Simplified Lorentzian on-axis beam factor and Gaussian lateral profile**
   - *What we do:* On-axis beam focus follows a Lorentzian curve $G(z) = 1 / \sqrt{1 + ((z - z_f) / z_R)^2}$ and lateral beam sensitivity follows a Gaussian profile $\exp(-r^2 / 2\sigma_b^2)$.
   - *What reality does:* Transducer aperture diffraction produces near-field pressure oscillations, side lobes, and depth-dependent beam divergence.
   - *Effect on results:* Smooth, monotonic beam sensitivity without near-field nulls, side-lobe echoes, or depth-varying focal waist distortions.

5. **FBH amplitude uses the $d^2$ far-field law with a model gain constant $k_{fbh}$**
   - *What we do:* Flat-bottom hole (FBH) echo amplitude scales with hole area ($d^2$) multiplied by an empirical simulator gain constant $k_{fbh}$ ($0.8$), not a derived physical transducer voltage.
   - *What reality does:* Disc reflector response depends on the ratio of diameter to wavelength, showing near-field interference and transition to edge diffraction.
   - *Effect on results:* Echo height is strictly quadratic with diameter; absolute amplitudes reflect relative simulator units rather than calibrated instrument millivolts.

6. **Inclusions use the FBH law with a random factor and fixed 3-facet echo shape**
   - *What we do:* Inclusion amplitude is scaled by an amplitude factor $u \le 1.0$ applied to the FBH law, with the echo split across three discrete sub-echo facets (delays 0.0, 0.35, 0.7 periods; weights 1.0, -0.6, 0.4).
   - *What reality does:* Non-metallic inclusions (oxides, nitrides, silicates) have irregular 3D geometries, varying acoustic impedance contrasts, and complex multi-path specular/diffuse scattering.
   - *Effect on results:* Produces representative split-peak morphology and lower amplitudes, but does not simulate full irregular scatterer facet orientations or variable acoustic impedances.

7. **Linear-frequency attenuation**
   - *What we do:* Material attenuation is modelled with a linear frequency dependence ($\alpha = a \cdot f$ dB/mm) producing round-trip loss $10^{-2 \alpha z / 20}$.
   - *What reality does:* Polycrystalline metals exhibit multiple attenuation regimes: Rayleigh scattering ($\propto f^4$), stochastic scattering ($\propto f^2$), diffusion regimes, and absorption ($\propto f$).
   - *Effect on results:* High frequencies are attenuated smoothly with depth, but frequency-dependent dispersion and non-linear spectral filtering across broad bands are simplified.

8. **Grain noise is white Gaussian noise convolved with the pulse with an $f^2$ level law**
   - *What we do:* White Gaussian noise is convolved with the RF pulse (dense-scatterer limit), scaled by material parameter $\sigma_g \cdot (f / 5\,\text{MHz})^2 \cdot T_{rt} \cdot A(z) \cdot G(z)$. Noise is generated independently per A-scan with no grain-size model and no spatial correlation.
   - *What reality does:* Grain backscatter arises from finite crystallographic grain distributions and texture; adjacent A-scans overlap identical physical grains, producing spatially correlated speckle. Rayleigh scattering scales with $f^4$ and the third power of grain diameter.
   - *Effect on results:* Noise possesses realistic RF frequency content and depth decay, but adjacent scan lines lack physical speckle correlation across helical scan passes.

9. **Artefacts are heuristic multipliers**
   - *What we do:* Data-quality faults (bubbles, lost back wall, water-path drift, roughness, missing coverage) are simulated via heuristic regional multipliers and offsets on interface, metal, and back-wall amplitudes.
   - *What reality does:* Physical bubbles cause 3D acoustic shadowing and wave refraction; water-path changes alter acoustic focal alignment; surface roughness induces phase aberration and diffuse scattering.
   - *Effect on results:* Yields distinct, controllable defect signatures for testing watchdog detection thresholds, but does not capture physical multi-physics boundary degradation.

10. **Receiver clip at $\pm 1$ models saturation**
    - *What we do:* The combined RF signal is clipped hard at $[-1.0, 1.0]$ using `np.clip`.
    - *What reality does:* Preamplifiers and ADCs show non-linear compression curves, baseline DC shifts, slew-rate saturation, and recovery ring-down transients.
    - *Effect on results:* Saturated signals have perfectly flat plateaus without amplifier ring-down, DC recovery drift, or harmonic distortion.

11. **No multiple reflections or reverberations**
    - *What we do:* The simulator computes only first-order reflections: interface echo, direct flaw echo, and first back-wall echo.
    - *What reality does:* Acoustic energy reverberates across water-path boundaries and creates multiple internal bar reflections (second and third back walls) and corner reflections.
    - *Effect on results:* Simulated records are clean of secondary multiples and reverberation ghost indications in deeper gates.

12. **Material constants and noise levels are UNVERIFIED - Ruri to check**
    - *What we do:* Material parameters (`ti64`, `in718`, `maraging250`, `water`) in `configs/materials.yaml` and probe settings in `configs/probe.yaml` are handbook values or simulator model choices tagged `UNVERIFIED - Ruri to check`.
    - *What reality does:* Real metallurgical batches exhibit heat-treatment-dependent sound velocity, texture-induced velocity anisotropy, and microstructure-dependent grain backscatter.
    - *Effect on results:* Simulation outputs are qualitative synthetic proxies rather than plant-specific calibrated measurements.

## Consequences

- **Synthetic data designation:** All simulator outputs, plots, tables, and derived metrics must be strictly labelled **"Synthetic data"**. They cannot be used to make absolute probability of detection (POD) claims or qualify physical parts.
- **Valid for relative comparison:** Synthetic data is valid for validating the rules engine, measuring relative ranking performance, tuning watchdog triggers, and testing ML architectures under controlled conditions.
- **Sim-to-real gap:** A sim-to-real domain gap is expected (Phase 4.6). Models trained on synthetic data cannot be transferred directly to plant inspections without real-data evaluation and calibration.
- **Preserved safety architecture:** Because synthetic data has known limitations, the core safety architecture (ADR 0001) is reaffirmed: the rules engine remains in charge, ML can only add flags, and the certified inspector makes all disposition decisions.

## How to answer an expert who says real signals do not look like this

When a Level III inspector or NDT physics expert points out that real immersion UT signals exhibit complex phase aberration, multiple reverberations, or correlated grain speckle, respond with the following points:

1. **Acknowledge and agree:** "You are right. This is not a finite-element wave simulation and it does not replicate full elastodynamic physics."
2. **State the engineering purpose:** "The simulator exists to provide a reproducible, mathematically traceable test harness with known ground truth. It lets us rigorously verify gate logic, watchdog fault alarms, indication clustering, and fusion rules in automated CI before touching real data."
3. **Traceability over complexity:** "Every term in our A-scan model is derived analytically and explained on paper (`docs/study/01-simulator-physics.md`). We chose mathematical transparency and fast execution over black-box complexity."
4. **Clear labelling:** "Every synthetic scan and metric is explicitly labelled 'Synthetic data'. We make no claims of plant qualification, MIDHANI data validation, or physical acceptance."
5. **Sim-to-real validation:** "Real data testing is addressed separately using published physical benchmarks (such as the ML-NDT dataset in Phase 4). In our safety architecture, ML is restricted to adding candidate flags; it can never override or remove a rules-engine flag."
