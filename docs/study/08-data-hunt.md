# 08-data-hunt.md — Public Ultrasonic NDT Dataset Survey

## 1. Goal and Scope
This survey identifies public ultrasonic non-destructive testing (NDT) datasets of **metals** relevant to immersion and phased-array bar flaw detection. Candidate inspection modalities include:
- A-scan RF time series
- B-scan and C-scan planar slices
- Full Matrix Capture (FMC) and Total Focusing Method (TFM)
- Phased array ultrasonic testing (PAUT)

Target defect types include cracks, lack-of-fusion, inclusions, and flat-bottom holes in structural metals (titanium, nickel alloys, austenitic and ferritic steels, aluminium).

Per project honesty rules:
- Every dataset entry in `data/registry.yaml` was directly inspected via its source page in this session.
- Unstated properties (frequency, specimen count, flaw count) are recorded as `null` or `'unknown (not stated on page)'`.
- Candidates were filtered for clear open licences permitting research use and direct public download.

---

## 2. Search Methodology and Repository Queries

### University of Strathclyde Pure
- **Target:** FMC dataset for austenitic welds.
- **Entry point:** DOI `10.15129/086404bd-eb69-429b-978c-2c35cdbfcf87`.
- **Verified on source page:**
  - Title: *FMC dataset - Lack-of-fusion crack on welded 316L stainless steel plates*
  - Licence: CC BY 4.0
  - Files: `Lack_of_fusion_FMC_DORT_2016.mat` (260 MB) and `Lack_of_fusion_FMC_DORT_2016.ods` (53 KB).
  - Material: 316L stainless steel welded plates.
  - Technique: FMC phased array, 50-degree lack-of-fusion flaw.

### GitHub
- **Queries:**
  - `site:github.com "ultrasonic" dataset ("NDT" OR "phased array" OR "FMC" OR "A-scan" OR "flaw")`
- **Hits evaluated (4 looked at):**
  - `iikka-v/ML-NDT`: 316L weld PAUT, 3 thermal fatigue cracks, LGPL-3.0, 2.51 GB raw data. (Accepted)
  - `koomas/NDT_ML_Flaw`: Dissimilar metal weld PAUT, segmentation cracks and EDM notches, LGPL-3.0, 226.7 MB. (Accepted)
  - `ndtatbristol/arim`: Python array imaging library with sample configs, no standalone raw experimental defect repository.
  - `ndtatbristol/mfmc`: HDF5/MFMC file specification and reader tools, format repository.

### Zenodo
- **Queries via Zenodo REST API (`resource_type.type:dataset`):**
  - `(ultrasonic OR ultrasound) AND (NDT OR weld OR flaw OR defect)`: 33 total hits, 10 looked at.
  - `(ultrasonic OR ultrasound) AND (FMC OR TFM OR "phased array" OR "A-scan")`: 15 total hits, 10 looked at.
  - Targeted queries: `phased array NDT` (543 hits), `ultrasonic weld crack` (528 hits), `ultrasonic immersion` (528 hits), `FMC ultrasound` (25 hits).
- **Hits evaluated (14 looked at):**
  - Record 15692966: Acoustic lens pipe PAUT immersion, 140 mm steel pipe, 18.52 GB, CC BY 4.0. (Accepted)
  - Record 11239514: TUBERS ultrasonic and synthetic crack dataset (Univ. of Essex / TWI), steel plates & step wedge A-scans, 44.14 MB, MIT licence. (Accepted)
  - Record 15083865: Submerged Arc Welding Open Repository (PENELOPE), steel coupons, PAUT `.nde` files, 12.09 GB, CC BY 4.0. (Accepted)
  - Record 8030164: Weld map tomography dataset, grain orientation study, 643.7 MB. (Inspected)
  - Record 19462220: Electronic crosstalk deconvolution (ECNDT 2026), restricted access (HTTP 403 on files). (Rejected)
  - Records 4405106 & 4405277: CompInnova WP7/WP2 composite demonstrator panel & coupons. (Rejected: CFRP composite)
  - Records 15392443 & 16788440: Concrete compressive strength & concrete crack wavefield. (Rejected: concrete)
  - Record 22268109: Multi-modal composite inspection (CT, UT, DR). (Rejected: composite)
  - Record 15719377: Guided wave SHM on SMC samples. (Rejected: composite & guided-wave SHM)

### Mendeley Data
- **Queries:**
  - `site:data.mendeley.com "ultrasonic" ("A-scan" OR "B-scan" OR "phased array" OR "FMC")`: 0 direct hits.
  - `site:data.mendeley.com "ultrasonic" (flaw OR defect)`: 5 hits looked at.
- **Hits evaluated:**
  - `10.17632/ttb63krg6d.1`: Guided wave SHM for steel pipeline, 14–37 kHz, 106.4 MB, CC BY 4.0. (Rejected: guided-wave SHM, kHz range)
  - `10.17632/n5wxwncv6k.1`: Ship structure butt weld defect frequency/length statistics, CC BY 4.0. (Rejected: tabular statistics only, no waveforms)
  - `10.17632/5yx2kt56c7.1`: Dispersion U-Mo fuel plates UT methodology, 2.36 MB, CC BY 4.0. (Rejected: attenuation spreadsheets only)

### Figshare
- **Queries via API (`https://api.figshare.com/v2/articles/search`):**
  - `search_for: "ultrasonic NDT"`, page size 10: 10 hits looked at.
  - Targeted dataset search (`item_type: 3`): `FMC TFM ultrasonic` (0 hits), `ultrasonic flaw detection steel` (0 hits), `ultrasonic A-scan defect` (5 hits looked at).
- **Hits evaluated:**
  - Record 19375107: Nondestructive testing of thin composite structures. (Rejected: composite)
  - Other hits: academic paper tables and figures, no raw ultrasonic waveform datasets for metals.

### IEEE DataPort
- **Queries:**
  - `site:ieee-dataport.org "ultrasonic" OR "ultrasound"`
  - `site:ieee-dataport.org "non-destructive" OR NDT`
- **Hits evaluated (4 looked at):**
  - CUBDL (Challenge on Ultrasound Beamforming with Deep Learning, DOI `10.21227/2b1v-b228`): Medical ultrasound in vivo / phantom channel data. (Rejected: medical ultrasound)
  - Tactile displays / acoustic levitation control signals. (Rejected: non-NDT)
  - Hardware Trojan side-channel power data. (Rejected: electronic hardware)

### Kaggle
- **Queries:**
  - `site:kaggle.com/datasets "ultrasonic" OR "ultrasound" weld OR defect OR flaw OR NDT`
- **Hits evaluated (3 looked at):**
  - `sujaykapadnis/welding-defect-object-detection`: RGB photos of welds for YOLO. (Rejected: optical photography)
  - `ahmedabdelaty/radiographs-welding-defect-detection`: X-ray radiographs. (Rejected: radiographic testing)
  - `SDNET2018`: Concrete bridge and wall crack photos. (Rejected: non-metal, optical)

### Iowa State CNDE
- **Queries:**
  - `site:cnde.iastate.edu "benchmark" OR "dataset" OR "data" ultrasonic`
  - `site:cnde.iastate.edu "WFNDEC"`
- **Hits evaluated (4 looked at):**
  - WFNDEC simulation benchmark problem statements (AIP proceedings).
  - UTSIM simulation software modules.
  - NDE-ed educational tutorial waveform illustrations.
  - MAPOD archive references. (Rejected: simulation specifications and tools, no downloadable raw experimental scan files on open web portal)

### Papers-with-Code
- **Queries:**
  - `site:paperswithcode.com/dataset "ultrasonic" OR "ultrasound" OR "NDT"`
- **Hits evaluated (4 looked at):**
  - TN3K & GBCU: Thyroid nodule and gallbladder ultrasound segmentation. (Rejected: medical ultrasound)
  - CLIMB clinical multimodal datasets. (Rejected: medical ultrasound)
  - URFD robotic obstacle sensing. (Rejected: robotics distance telemetry)

---

## 3. Query and Hits Summary

| Repository | Primary Queries Run | Hits Looked At | Accepted | Rejected |
|---|---|---|---|---|
| **Univ. of Strathclyde Pure** | DOI `10.15129/086404bd-eb69-429b-978c-2c35cdbfcf87` | 1 | 1 | 0 |
| **GitHub** | `ultrasonic dataset NDT phased array FMC` | 4 | 2 | 0 |
| **Zenodo** | `resource_type.type:dataset AND ultrasonic (NDT/weld/flaw/FMC/PAUT)` | 24 | 3 | 7 |
| **Mendeley Data** | `ultrasonic flaw defect NDT`, `ttb63krg6d.1`, `n5wxwncv6k.1`, `5yx2kt56c7.1` | 5 | 0 | 3 |
| **Figshare** | `ultrasonic NDT`, `ultrasonic A-scan defect`, `FMC TFM ultrasonic` | 15 | 0 | 1 |
| **IEEE DataPort** | `ultrasonic`, `ultrasound`, `non-destructive NDT` | 4 | 0 | 1 |
| **Kaggle** | `ultrasonic weld defect`, `radiographs welding`, `welding object detection` | 3 | 0 | 3 |
| **Iowa State CNDE** | `benchmark`, `WFNDEC`, `UTSIM`, `dataset` | 4 | 0 | 1 |
| **Papers-with-Code**| `ultrasonic`, `ultrasound`, `NDT` | 4 | 0 | 2 |
| **Total** | — | **64** | **6** | **18** |

*(Detailed properties of each accepted and rejected entry are recorded in `data/registry.yaml`.)*

---

## 4. Accepted Datasets Overview

1. **ML-NDT** (`github.com/iikka-v/ML-NDT`)
   - Material: 316L stainless steel austenitic weld.
   - Modality: Phased array ultrasonic testing (PAUT) at 2.0 MHz.
   - Content: 1 specimen with 3 real thermal fatigue cracks; 2.51 GB raw binary data.
   - Status: Already downloaded in repo workflow; benchmark baseline.
2. **Strathclyde-FMC-Welded-316L** (DOI `10.15129/086404bd-eb69-429b-978c-2c35cdbfcf87`)
   - Material: 316L stainless steel welded plates.
   - Modality: Full Matrix Capture (FMC) phased array at 5.0 MHz.
   - Content: 1 real lack-of-fusion planar flaw oriented at 50 degrees; 260 MB raw MATLAB matrix.
   - Relevance: Direct compatibility with TFM reconstruction and immersion simulation comparison.
3. **NDT_ML_Flaw** (`github.com/koomas/NDT_ML_Flaw`)
   - Material: Dissimilar metal weld (austenitic/ferritic steel).
   - Modality: Phased array ultrasonic testing (PAUT).
   - Content: Real segmentation cracks and EDM notches alongside CIVA-simulated flaws; 226.7 MB.
4. **Zenodo-Pipe-Lens-PAUT** (DOI `10.5281/zenodo.15692966`)
   - Material: Submerged steel pipe (140 mm diameter).
   - Modality: Immersion phased array UT with acoustic lens at ~5.0 MHz.
   - Content: 140 mm pipe with machined wall-thinning / pitting defects; 18.52 GB archive.
5. **TUBERS-Ultrasonic-Crack-Dataset** (DOI `10.5281/zenodo.11239514`)
   - Material: Structural steel plates and step wedge (1–8 mm steps).
   - Modality: Contact pulse-echo A-scans at 5.0 MHz and 10.0 MHz.
   - Content: Oscilloscope time-series CSVs of cracked steel plates and thickness calibration steps; 44.14 MB.
6. **SAW-Open-Repository-PENELOPE** (DOI `10.5281/zenodo.15083865`)
   - Material: Submerged arc welding steel coupons.
   - Modality: Phased array ultrasonic testing (PAUT) in `.nde` format with defect location logs; 12.09 GB.

---

## 5. Why Public Ultrasonic NDT Data Is Scarce

Public ultrasonic NDT data of metals is exceptionally rare compared to medical ultrasound, computer vision, or seismic acoustics. Key reasons confirmed by literature (including PNNL/NRC NUREG reviews):

1. **Commercial and Defence IP Restrictions**
   - High-value alloy inspections (titanium billets, nickel superalloys, nuclear pressure vessels, aerospace forgings) are conducted under strict commercial non-disclosure or defence export controls. Raw scan files reveal part geometries, weld procedures, and proprietary manufacturing defect rates.
2. **Liability and Regulatory Conservatism**
   - In aerospace and nuclear sectors, inspection procedures must satisfy rigid qualification codes (ASME Section XI, AMS-STD-2154, ASTM E2375). Releasing scan files associated with components could create perceived liability or disclose proprietary flaw tolerances.
3. **Proprietary and Fragmented Data Formats**
   - Major equipment vendors (Evident/Olympus, Zetec, Sonatest, Eddyfi, Krautkramer/Baker Hughes) store raw A-scan and phased-array RF waveforms in encrypted or proprietary binary formats (`.opd`, `.fpd`, `.nde`, `.rtd`). Very few open-access pipelines export standardized HDF5/MFMC matrices.
4. **Data Acquisition and Flaw Introduction Costs**
   - Manufacturing realistic flawed specimens is costly. Introducing genuine fatigue cracks, lack of fusion, or controlled inclusions in thick metal sections requires specialized autoclave thermal cycling, EDM notch machining, or seeded defect welding. Consequently, research institutions treat physical specimens as prized capital assets and rarely distribute full raw RF volumes.
5. **Academic Tradition of "Data on Request"**
   - Unlike bioinformatics or machine learning where repository deposition is mandatory for peer-reviewed publication, NDT journals (e.g., *Ultrasonics*, *NDT&E International*) historically permitted "data available upon reasonable request". Requests frequently fail over time due to personnel moves, institutional storage deprecation, or licensing ambiguities.
