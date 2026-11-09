# UTriage User Guide

## 1. What UTriage Does and Does Not Do

UTriage is an offline decision-support tool for ultrasonic indication triage on immersion scans of metal bars.

### What UTriage Does
- **Advisory triage:** UTriage evaluates exported ultrasonic bar scans using a deterministic rules engine and machine learning models.
- **Rules engine in charge:** The deterministic rules engine detects indications using amplitude gates, distance-amplitude correction (DAC) curves, and clustering.
- **Union fusion (ML only adds):** Machine learning models can raise candidate indications that conventional gates might miss. ML can never suppress or remove a rule-generated flag. Every indication flagged by the rules engine remains flagged.
- **Data-quality screening:** A watchdog evaluates scans for hardware or couplant anomalies before manual review.

### What UTriage Does Not Do
- **Never accepts or passes a part:** UTriage is strictly advisory. It never automatically accepts, rejects, or dispositions inspection material. Every disposition decision belongs solely to the human inspector.
- **No live machine connection:** UTriage operates strictly offline and air-gapped. It reads exported files only. It never connects to pulser-receivers, instrument controllers, PLCs, or factory networks.
- **No plant trials or plant data:** UTriage is an independent prototype. It has not been tested on plant data or evaluated on live shop floors.
- **Synthetic data and public benchmark data only:** Results and performance metrics are derived exclusively from Synthetic data (simulated cylindrical bar scans) and public ML-NDT weld phased array ultrasonic testing (PAUT) datasets.
- **No quantitative result claims in this guide:** For benchmark findings, see [results/claims.md](../results/claims.md) and the [README Results section](../README.md#results).

---

## 2. Install

To install UTriage in a Python environment for local inspection and review:

```bash
pip install -e ".[ml,app]"
```

For development and test suites, use `pip install -e ".[dev,ml,app]"`.

For standalone workstation setup, file integrity checksum checks (`SHA256SUMS.txt`), and tamper-locked mode (`UTRIAGE_LOCKED=1`), see [install.md](install.md). UTriage runs fully air-gapped with no network connectivity required.

---

## 3. Input Files

UTriage accepts ultrasonic scans in two primary formats: the native `.npz` + `.json` pair, and native CSV exports.

### 3.1 The `.npz` + `.json` Scan Pair

A scan consists of an array archive (`.npz`) and a companion metadata file (`.json`) sharing the same file stem:

- **`.npz` Array Archive:**
  - `ascans`: 3D array of digitized ultrasonic waveforms [axial index, angle index, time samples] (float16 or float32).
  - `t0_us`: Time offset relative to probe excitation pulse in microseconds.
  - `fs_mhz`: Digitizer sampling frequency in MHz.
  - `axial_mm`: 1D array of axial translation positions along the bar axis in mm.
  - `angle_deg`: 1D array of rotational angles in degrees [0, 360).
  - `t_if_us`: 2D array of front-surface interface echo arrival times [axial, angle] in microseconds.
  - `missing`: 2D boolean mask [axial, angle] marking missing scan beams.

- **`.json` Ground Truth and Parameters:**
  - `label`: Dataset provenance description (e.g., `"Synthetic data"`).
  - `sim_version`: Simulation schema version.
  - `seed`: Random seed used for generation.
  - `material`: Material parameters dictionary (alloy name, acoustic velocity, density, attenuation, grain noise).
  - `probe`: Transducer parameters dictionary (frequency, crystal dimension, focus type, focal length in water).
  - `water`: Couplant parameters dictionary.
  - `diameter_mm`: Outer bar diameter in mm.
  - `length_mm`: Scanned axial length in mm.
  - `axial_step_mm`: Axial scan index pitch in mm.
  - `n_angle`: Number of angular steps per revolution.
  - `reflectors`: List of simulated reflectors (flat-bottom holes or inclusions).
  - `artefacts`: List of simulated acoustic artefacts.

### 3.2 Native CSV Export

Defined in `utriage.io.adapters.CsvAscanAdapter` and [scan_export_template.md](scan_export_template.md):
- **Comment header:** Leading `# key=value` lines providing mandatory metadata:
  - `fs_mhz`: Sampling frequency in MHz (float).
  - `t0_us`: Acquisition start time in microseconds (float).
  - `material`: Alloy identifier (e.g., `ti64` or `in718`).
  - `probe`: Transducer identifier (e.g., `probe_5mhz`).
  - `diameter_mm`: Outer bar diameter in mm (float).
  - `axial_step_mm`: Axial scan index step in mm (optional float, default: `1.0`).
- **Column header row:** Exactly `axial_mm,angle_deg,s0,s1,s2,...,sN`
- **Data rows:** One row per A-scan with linear axial position (`axial_mm`), rotational angle (`angle_deg`), and digitized sample amplitudes (`s0`, `s1`, ...).

### 3.3 Sample Scans Location

Sample scans in the repository reside in:
- `data/samples/`: Fixed sample scans for evaluation (e.g., `data/samples/ti64/bar_0000.npz`).
- `data/synthetic/`: Generated synthetic scans (e.g., `data/synthetic/ti64/`).

---

## 4. Command-Line Reference

UTriage provides command-line subcommands for evaluation, simulation, batch triage, calibration, and inspection.

### 4.1 `version`

- **Purpose:** Print the UTriage version.
- **Usage:**
  ```bash
  utriage version [OPTIONS]
  ```
- **Options:**
  | Option | Type | Default | Description |
  | --- | --- | --- | --- |
  | `--help` | - | - | Show this message and exit. |
- **Example:**
  ```bash
  utriage version
  ```

### 4.2 `simulate`

- **Purpose:** Generate synthetic bar UT scans with ground truth (Synthetic data).
- **Usage:**
  ```bash
  utriage simulate [OPTIONS]
  ```
- **Options:**
  | Option | Type | Default | Description |
  | --- | --- | --- | --- |
  | `--material` | TEXT | `ti64` | Material name |
  | `--probe` | TEXT | `p5` | Probe config |
  | `--bars` | INTEGER | `5` | Number of bars to simulate |
  | `--seed` | INTEGER | `0` | Random seed |
  | `--out` | PATH | `data/synthetic/<material>` | Output directory |
  | `--length-mm` | FLOAT | `64.0` | Bar length in mm |
  | `--n-angle` | INTEGER | `120` | Number of angular steps |
  | `--figures / --no-figures` | FLAG | `--figures` | Plot C-scans |
  | `--help` | - | - | Show this message and exit. |
- **Example:**
  ```bash
  utriage simulate --material ti64 --bars 5 --out data/synthetic/ti64
  ```

### 4.3 `evaluate`

- **Purpose:** Evaluate a bar scan against acceptance rules.
- **Usage:**
  ```bash
  utriage evaluate [OPTIONS] {scan}
  ```
- **Arguments:**
  | Argument | Type | Description |
  | --- | --- | --- |
  | `scan` | PATH (required) | Path to scan .npz file |
- **Options:**
  | Option | Type | Default | Description |
  | --- | --- | --- | --- |
  | `--acceptance` | PATH | `configs/acceptance.yaml` | Path to acceptance configuration |
  | `--json-out` | PATH | None | Optional path to output JSON file |
  | `--help` | - | - | Show this message and exit. |
- **Example:**
  ```bash
  utriage evaluate data/samples/ti64/bar_0000.npz --acceptance configs/acceptance.yaml --json-out out/eval.json
  ```

### 4.4 `triage`

- **Purpose:** Batch shop-floor triage: advisory evaluation, report and JSON generation.
- **Usage:**
  ```bash
  utriage triage [OPTIONS] {scans}...
  ```
- **Arguments:**
  | Argument | Type | Description |
  | --- | --- | --- |
  | `scans` | PATH... (required) | Paths to scan .npz files |
- **Options:**
  | Option | Type | Default | Description |
  | --- | --- | --- | --- |
  | `--out` | PATH | `out` | Output directory for reports and summaries |
  | `--model` | PATH | None | Optional ML model checkpoint path |
  | `--acceptance` | PATH | None | Optional path to acceptance configuration |
  | `--help` | - | - | Show this message and exit. |
- **Example:**
  ```bash
  utriage triage data/synthetic/ti64/*.npz --out out/
  ```

### 4.5 `site-calibrate`

- **Purpose:** Calibrate ML anomaly detection threshold on clean reference scans.
- **Usage:**
  ```bash
  utriage site-calibrate [OPTIONS]
  ```
- **Options:**
  | Option | Type | Default | Description |
  | --- | --- | --- | --- |
  | `--clean-dir` | PATH (required) | - | Folder of .npy clean scans |
  | `--out` | PATH (required) | - | Profile JSON output path |
  | `--label` | TEXT (required) | - | Data provenance ('Real' or 'Synthetic') |
  | `--model` | PATH | `models/anomaly_2d.pt` | Path to 2D autoencoder checkpoint |
  | `--fcr` | FLOAT | None | Target false call rate (default from detector config) |
  | `--help` | - | - | Show this message and exit. |
- **Example:**
  ```bash
  utriage site-calibrate --clean-dir data/clean_scans --out site_profile.json --label Synthetic
  ```

### 4.6 `ti-inspect`

- **Purpose:** Inspect titanium bar scan and write offline report and JSON. (Synthetic data)
- **Usage:**
  ```bash
  utriage ti-inspect [OPTIONS]
  ```
- **Options:**
  | Option | Type | Default | Description |
  | --- | --- | --- | --- |
  | `--scan` | PATH | None | Path to scan .npz file |
  | `--demo-seed` | INTEGER | None | Random seed for demo simulation |
  | `--out` | PATH | `out/ti` | Output directory |
  | `--bar-id` | TEXT | None | Bar identifier |
  | `--help` | - | - | Show this message and exit. |
- **Example:**
  ```bash
  utriage ti-inspect --scan data/samples/ti64/bar_0000.npz --out out/ti
  ```

---

## 5. Inspector App

The inspector review application provides an offline interactive graphical interface for reviewing scans, indications, waveforms, and data quality.

### 5.1 Starting the App

Launch the application with Streamlit:

```bash
streamlit run app/streamlit_app.py
```

The application runs locally in a web browser without network access or telemetry.

### 5.2 Uploading Scans and Automatic Pairing

On Page 1 (`1 Load scan`), inspectors can select the pre-loaded sample bar or upload scan files:
- **Uploading multiple files:** The file uploader accepts multiple `.npz`, `.json`, and `.csv` files simultaneously.
- **Automatic JSON discovery:** When an `.npz` file is uploaded alone, the application computes its SHA-256 digest and searches `data/samples` and `data/synthetic` for a matching file. When a match is found, the companion `.json` is paired automatically. If an `.npz` scan originates from an outside location, upload both `.npz` and `.json` files together.
- **Processing uploads:** Click the button **Process all uploaded scans (<count>)** to parse and initialize all candidate scans.
- **Active scan switcher:** When two or more scans are loaded, a dropdown titled **Active scan** appears in the sidebar. Inspectors can switch between scans at any time.
- **Summary table:** A summary table lists all loaded scans, showing scan filename, indication count, synthetic status, and whether the ML model is loaded.

### 5.3 Application Pages

The application is organized into eight pages in the sidebar:

1. **`1 Load scan`**: Select between sample bar and file upload (`.npz` or `.csv`). Manage multi-file uploads, auto-pairing, view the summary table, and inspect current bar dimensions.
2. **`2 C-scan view`**: View the 2D peak amplitude C-scan map over the unwrapped bar surface. Rules-based indications are marked with circles (`o`), and ML flags are marked with crosses (`x`). Select an indication from the dropdown to inspect its digitized A-scan waveform, gate limits, and DAC reference curve.
3. **`3 Indication list`**: View a ranked table of all detected indications. Rejectable indications and uncertain findings are placed at the top for priority review. Displays indication source, category, amplitude (% DAC), SNR (dB), ML score, uncertainty, and rule reasons.
4. **`4 Data quality`**: Review watchdog checks for couplant stability, interface echo, and back-wall retention. Displays either a green `Quality OK` status or a red `Re-scan recommended` notice along with a detailed fault table listing fault kind, axial range, angular range, and affected A-scans.
5. **`5 Decision log`**: Record inspector disposition decisions. Enter the Inspector name / ID, select an indication index, choose a disposition (`accepted`, `rejected`, or `rescan`), enter comments, and click **Record decision**. Decisions are saved to `logs/audit_log.csv`.
6. **`6 Report`**: Generate an inspection report pre-filled with data quality findings, indication triage, and inspector decisions. Click **Download Inspection Report (HTML)**. The report can be printed using browser print (`Ctrl+P` -> Save as PDF) for documentation archives.
7. **`7 Evidence and limits`**: Displays quantitative claims, real-data evidence limits, and the generalization matrix image read directly from benchmark results.
8. **`8 Site calibration`**: Review site calibration profiles. Upload or enter a path to a site profile JSON file to inspect provenance label, clean scan count, target false-call rate, threshold, and upper 95% bound.

---

## 6. Reading the Output

### 6.1 Indication Categories

Every indication evaluated by UTriage is assigned an advisory category by the rules engine:
- **`rejectable`:** Peak amplitude exceeds the reject threshold (placeholder value: 100% DAC), or linear length exceeds the maximum linear limit, or multiple recordable indications exceed the count limit within the defined axial window.
- **`reportable`:** Peak amplitude meets or exceeds the recording threshold (placeholder value: 50% DAC) but remains below the reject threshold.
- **`insignificant`:** Indication amplitude is below the recording threshold.
- **`data_quality_fault`:** An indication below recording threshold that falls within an acoustic data-quality fault region.

> [!NOTE]
> Acceptance limits in `configs/acceptance.yaml` are placeholder, UNVERIFIED - Ruri to check.

### 6.2 Data-Quality Watchdog Faults

The data-quality watchdog pre-screens ultrasonic scans for equipment or acoustic anomalies:
- **`interface_missing`:** The front-surface interface echo was not detected in the expected gate window.
- **`interface_drop`:** The interface echo amplitude dropped significantly below the scan median, indicating unstable acoustic couplant.
- **`waterpath_drift`:** The front-surface interface arrival time shifted beyond tolerance, indicating water path distance variation.
- **`backwall_loss`:** The back-wall echo dropped below the retention threshold, indicating severe attenuation, sound beam disruption, or geometry changes.
- **`noise_above_limit`:** Flaw-gate acoustic noise exceeds the allowable noise limit ratio.
- **`coverage_gap`:** Missing A-scans or amplitude below the coverage floor, indicating missing scan data.
- **`saturation`:** Sample amplitude reached or exceeded digitizer full scale (signal clipping).
- **`early_echo`:** Significant echo detected before the front-surface interface (e.g., air bubbles in the immersion couplant).

### 6.3 Watchdog Recommendation

- **`rescan_recommended`:** Evaluates to `True` when one or more data-quality faults are detected on the bar. The inspector should inspect couplant conditions and consider a re-scan.

---

## 7. Titanium Mode

Titanium mode provides a dedicated inspection workflow for titanium alloy bars exhibiting macrozone-like grain noise (Synthetic data).

### 7.1 How to Run

Run titanium mode inspection from the command line:

Using an existing scan:
```bash
utriage ti-inspect --scan data/samples/ti64/bar_0000.npz --out out/ti
```

Using demo simulation:
```bash
utriage ti-inspect --demo-seed 0 --out out/ti
```

### 7.2 Depth-Zone Calibration and SNR

Titanium mode segments the bar volume into depth zones:
1. **Reference calibration:** A simulated calibration block with flat-bottom holes (FBH) across depth zones is used to measure:
   - **Signal `A`:** Mean amplitude within the signal window around the reference FBH peak.
   - **Noise `B`:** Mean noise envelope outside reflector exclusion regions.
   - **Noise `C`:** Peak noise envelope outside reflector exclusion regions.
2. **SNR calculation:** Signal-to-noise ratio is calculated as:
   $$\text{SNR} = \frac{A - B}{C - B}$$
   (This form is published in US 6,370,956; whether a given specification uses it is UNVERIFIED - Ruri to check.)
3. **Zone alarm threshold:** The candidate alarm level is $C \times (1 + \text{noise\_margin})$, capped at $\text{cap\_fraction} \times A$.
4. **Inspectability check:** If the alarm cap falls below peak noise ($alarm \le C$) or SNR falls below the minimum limit ($\text{SNR} < \text{min\_snr}$), the zone is marked as `zone_not_inspectable`. This warns the inspector that grain noise prevents adequate sensitivity in that zone.
5. **Alarms:** Regions exceeding the alarm threshold in inspectable zones generate `alarm` records.

### 7.3 Outputs

Titanium mode outputs:
- `<bar_id>_ti.json`: Complete inspection results, calibration record, and zone alarms.
- `<bar_id>_ti_report.html`: Standalone printable HTML inspection report.

> [!WARNING]
> All titanium mode configuration parameters (including noise margins, cap fractions, min SNR, and reference FBH sizes) are placeholder, UNVERIFIED - Ruri to check.

---

## 8. Troubleshooting

This section lists real error messages produced by the codebase and the recommended actions to resolve them.

### 8.1 Configuration and Integrity Errors

- **`Locked configuration check failed: missing: <files>; mismatched: <files>`**
  - **Cause:** `UTRIAGE_LOCKED=1` is active, and files in `configs/` or `models/` were modified, missing, or have mismatched checksums relative to `configs/manifest.json`.
  - **Action:** Restore the original files from approved installation media. Do not edit config files on locked review workstations. Check that line endings were not modified by text editors.

- **`Model hash mismatch: <model_name>`** or **`Model hash file missing: <sidecar_name>`**
  - **Cause:** An ML model checkpoint in `models/` does not match its `.sha256` sidecar file, or the sidecar is missing in locked mode.
  - **Action:** Re-copy the authentic model checkpoint and its accompanying `.sha256` sidecar from approved offline media.

### 8.2 File Upload and Input Errors

- **`<filename>: size exceeds maximum limit of 524288000 bytes.`**
  - **Cause:** Uploaded scan file exceeds the 500 MB maximum upload size limit (`MAX_SCAN_BYTES`).
  - **Action:** Export a shorter bar scan segment or compress the arrays using `.npz` format before uploading.

- **`<scan_name>: no matching <stem>.json found automatically - upload the .npz together with its .json file.`**
  - **Cause:** An `.npz` file was uploaded without its companion `.json` metadata file, and no matching SHA-256 hash was found in `data/samples` or `data/synthetic`.
  - **Action:** Upload both the `.npz` file and its companion `.json` file together in the file uploader.

- **`Missing required metadata keys in CSV: <keys>`**
  - **Cause:** A CSV scan file is missing required `# key=value` comment header lines (`fs_mhz`, `t0_us`, `material`, `probe`, or `diameter_mm`).
  - **Action:** Edit the CSV file header to include all required metadata keys as described in Section 3.2 and [scan_export_template.md](scan_export_template.md).

- **`Header row must begin with 'axial_mm,angle_deg', got ...`**
  - **Cause:** The column header row of a CSV scan file does not start with `axial_mm,angle_deg`.
  - **Action:** Check the CSV column header row and format it as `axial_mm,angle_deg,s0,s1,...,sN`.

### 8.3 CLI Command Errors

- **`Error: exactly one of --scan or --demo-seed is required.`**
  - **Cause:** `utriage ti-inspect` was executed without either `--scan` or `--demo-seed`, or both were passed simultaneously.
  - **Action:** Provide either `--scan path/to/scan.npz` or `--demo-seed <int>`, but not both.

- **`Error: invalid --bar-id`**
  - **Cause:** The identifier provided to `--bar-id` in `ti-inspect` contains disallowed characters or path traversal elements (`..`).
  - **Action:** Use only alphanumeric characters, underscores, periods, and hyphens (e.g., `--bar-id bar_001`).

- **`Error: Invalid label '<label>'. Must be 'Real' or 'Synthetic'.`**
  - **Cause:** `utriage site-calibrate` received a `--label` argument other than `Real` or `Synthetic`.
  - **Action:** Pass either `--label Real` or `--label Synthetic`.

- **`Error: No .npy files found in <clean_dir>`**
  - **Cause:** The directory specified in `--clean-dir` contains no `.npy` clean scan files.
  - **Action:** Check the path and provide a directory containing exported `.npy` clean scan arrays.

- **`Error: --fcr is required when detector config does not exist.`**
  - **Cause:** `utriage site-calibrate` was run without `--fcr` and the default detector configuration file is not present.
  - **Action:** Supply an explicit target false-call rate using `--fcr <rate>` (for example, `--fcr 0.01`).
