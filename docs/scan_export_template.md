# Immersion Ultrasonic Testing (UT) Scan Export Specification

This document details the data requirements for exporting immersion ultrasonic bar scans for offline evaluation in UTriage. It is written for the plant UT Level II/III engineer or acquisition specialist.

> **Operational Safety Notice**: UTriage is an advisory, standalone software tool that operates strictly offline. It reads exported files only and never connects to ultrasonic pulser-receivers, instrument controllers, PLCs, or plant networks. UTriage does not certify, accept, or reject inspection material; every disposition remains the sole responsibility of the certified human inspector.

---

## 1. Required Parameters and Metadata

To reconstruct the cylindrical scan volume and evaluate indications against conventional baseline rules, the exported scan must include the following parameters:

### Material & Geometry
- **Material grade**: Alloy designation (e.g., Ti-6Al-4V, IN718), nominal longitudinal wave velocity, and heat-treatment condition.
- **Bar diameter**: Outer diameter in mm (`diameter_mm`).
- **Bar scan length**: Scanned axial length in mm.

### Transducer (Probe) & Immersion Setup
- **Probe frequency**: Nominal center frequency in MHz (e.g., 5.0 MHz).
- **Focusing**: Element dimensions, focusing type (flat, spherical, or cylindrical), and focal length in water.
- **Water path**: Nominal immersion water path distance in mm.

### Acquisition Electronics
- **Sampling rate**: Digitizer sampling rate in MHz (`fs_mhz`, e.g., 50.0 MHz or 100.0 MHz).
- **Time origin (`t0_us`)**: Time offset / start of the acquisition window in microseconds relative to the excitation pulse.
- **Instrument gain**: Hardware and software gain settings in dB.
- **Waveform format**: Digitized RF waveforms (preferred) or rectified envelope sample series per A-scan.

### Spatial Grid & Scan Index
- **Axial position (`axial_mm`)**: Linear translation carriage coordinate along the bar axis.
- **Angular position (`angle_deg`)**: Rotational position in degrees [0, 360).
- **Scan index (`axial_step_mm`)**: Axial pitch per bar revolution. Recording the scan index is required to verify beam overlap and ensure 100% volumetric inspection coverage.

---

## 2. Calibration-Block Reference Scans

To establish the Distance-Amplitude Correction (DAC) curve and verify sensitivity:
- **Reference block scan**: An immersion scan of a representative calibration block (e.g., ASTM E127 / ASTM E428 style) matching the production alloy grade, diameter, and acoustic velocity.
- **Reference reflector details**: Known flat-bottom hole (FBH) diameters (e.g., 0.8 mm FBH / #2 FBH) and their respective metal path depths.

---

## 3. Preferred Export Formats

Data can be supplied in either of two ways:

### Option A: Native UTriage CSV Format (Preferred)
A plain-text CSV file matching the specification in `utriage.io.adapters.CsvAscanAdapter`:
1. **Comment header**: Leading lines with `# key=value` pairs providing mandatory metadata:
   - `fs_mhz`: Sampling frequency in MHz (float).
   - `t0_us`: Acquisition start time in µs (float).
   - `material`: Alloy name (string, e.g., `ti64` or `in718`).
   - `probe`: Transducer identifier (string, e.g., `probe_5mhz`).
   - `diameter_mm`: Outer bar diameter in mm (float).
   - `axial_step_mm`: Optional axial index step in mm (float, defaults to 1.0).
2. **Column header row**: Exactly `axial_mm,angle_deg,s0,s1,s2,...,sN`
3. **Data rows**: One row per A-scan:
   - Column 1: Axial position in mm (`axial_mm`).
   - Column 2: Angular position in degrees (`angle_deg`).
   - Columns 3 onward: Digitized sample amplitudes (`s0`, `s1`, ..., `sN`).

#### Example Format:
```csv
# fs_mhz=100.0
# t0_us=12.50
# material=ti64
# probe=probe_5mhz
# diameter_mm=100.0
# axial_step_mm=1.0
axial_mm,angle_deg,s0,s1,s2,s3,s4,s5
0.0,0.0,0.012,-0.005,0.034,0.120,0.085,0.010
0.0,1.0,0.010,-0.008,0.029,0.115,0.080,0.008
1.0,0.0,0.014,-0.003,0.035,0.122,0.088,0.011
```

### Option B: Vendor Equipment Export with Format Documentation
If direct CSV export is not available on the inspection console:
- Provide raw data exported in the equipment manufacturer's native format (e.g., ScanMaster, Waygate/Krautkramer, Evident/Olympus, Sonatest, Tecnatom).
- Include the vendor format documentation, data dictionary, or header specification (record structures, byte offsets, scaling units) so that a custom offline reader adapter can be implemented.

---

## 4. Sample Data Request

To evaluate offline triage and benchmark baseline detection reliability:
1. **One production bar sample file**: An exported scan of a representative production bar (or a representative section >= 200 mm) exhibiting typical background grain noise and any natural borderline indications.
2. **One calibration-block file**: An exported scan of the calibration standard containing the reference FBH targets across the evaluated metal path depths.
