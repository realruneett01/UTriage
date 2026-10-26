# UTriage Plant Installation and Operator Guide

> **IMPORTANT: Safety and Advisory Notice**  
> UTriage is an advisory decision-support tool. **UTriage never accepts or rejects a part automatically.** Every disposition decision (accept, reject, or re-scan) belongs solely to the certified ultrasonic (UT) inspector.

This guide provides instructions for plant IT technicians deploying UTriage on shop-floor inspection workstations.

---

## 1. System Requirements

- **Operating System:** Microsoft Windows 10 (64-bit) or Windows 11 (64-bit).
- **Processor:** Intel Core i5 (11th Gen or newer) or equivalent x64 processor.
- **Memory:** 8 GB RAM minimum.
- **Storage:** 2 GB available disk space.
- **Graphics / Acceleration:** No GPU required. All inference runs on the CPU.
- **Network Connectivity:** None required. The application runs strictly offline and air-gapped with no connection to the internet, plant LAN, or UT machine controllers.

---

## 2. Installation and Verification

UTriage is delivered as a pre-compiled standalone zip archive (`utriage-<version>-windows-x64.zip`) via approved offline transfer media.

### Step 1: Extract Archive
Extract the archive into a dedicated local folder (for example, `C:\UTriage`):
```powershell
Expand-Archive -Path .\utriage-<version>-windows-x64.zip -DestinationPath C:\UTriage
cd C:\UTriage
```

### Step 2: Verify SHA256 Checksums
Before launching, verify file integrity against `SHA256SUMS.txt`:
```powershell
Get-Content .\SHA256SUMS.txt | ForEach-Object {
    $expectedHash, $relPath = $_ -split '\s+', 2
    if (Test-Path $relPath) {
        $actualHash = (Get-FileHash -Path $relPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne $expectedHash) {
            Write-Error "Checksum mismatch on $relPath"
        }
    } else {
        Write-Error "Missing file: $relPath"
    }
}
```
All files must match their expected checksums before putting the workstation into service.

---

## 3. Configuration and Integrity Locking

- **Tamper Protection:** Model weights (`models/`) and inspection configurations (`configs/`) are sealed with cryptographic hashes in `configs/manifest.json`.
- **Locked Mode:** Plant installations run with `UTRIAGE_LOCKED=1` by default. If any configuration file or model checkpoint is modified, missing, or corrupted, the application halts immediately with an integrity violation error.
- Do not edit files inside `configs/` or `models/` on plant workstations.

---

## 4. Running UTriage

### Command Line Interface (CLI)
To run automated triage on an exported ultrasonic scan file:
```powershell
.\utriage.exe triage path\to\scan.npz
```
To run rules-based evaluation against acceptance standards:
```powershell
.\utriage.exe evaluate path\to\scan.npz
```
To check the installed software version:
```powershell
.\utriage.exe version
```

### Inspector Review Application
To launch the interactive offline review application for the certified inspector:
```powershell
.\utriage.exe
```
The review application starts locally and displays indication details, A-scans, C-scans, and data-quality watchdog alerts. Telemetry and external web analytics are permanently disabled (`gatherUsageStats = false`).

---

## 5. Logs and Audit Trail

- **Application Logs:** Diagnostic logs from application runs are saved to the `logs/` directory.
- **Inspector Disposition Audit Log:** All inspector decisions recorded in the review interface are appended to `logs/audit_log.csv`.
  - Recorded fields: timestamp (UTC ISO format), bar identifier, indication index, source (rules/ML), category, disposition (`accepted`, `rejected`, `rescan`), inspector comments, and inspector name.
  - This audit log provides traceability for quality management and code compliance.

---

## 6. Troubleshooting and Reporting Problems

If an operator encounters an error or an integrity check fails:

1. **Do not attempt to bypass integrity checks** or modify files in `configs/`.
2. Collect the following diagnostic files:
   - All files in the `logs/` directory.
   - The audit log file: `logs/audit_log.csv`.
   - The scan file (`.npz` or `.csv`) that was being processed.
   - Exact error text or console screenshots.
3. Package the diagnostic files onto approved offline transfer media and report the issue to your plant engineering / IT contact.
