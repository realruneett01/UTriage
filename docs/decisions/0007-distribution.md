# 0007. Offline compiled Windows distribution with Nuitka

- Status: Proposed
- Date: 2026-10-27

## Context

UTriage is targeted for shop-floor evaluation on plant workstations in aerospace and defence manufacturing environments. Plant inspection workstations operate under strict constraints:
- Offline environment: workstations operate strictly air-gapped with no connection to the internet, plant local networks, or ultrasonic testing (UT) equipment controllers.
- Hardware baseline: standard Windows 10/11 x64 systems without dedicated GPUs (Intel Core i5, 8 GB RAM).
- Source protection: shipping raw Python source files or uncompiled bytecode (`.pyc`) exposes proprietary code, inspection logic, and model architectures to direct inspection or modification.
- Integrity: configuration files (acceptance criteria, probe calibrations) and model weights must be protected against accidental modification or tampering during shop-floor operation.
- Dependency governance: `CONTRIBUTING.md` requires that any dependency outside the core list requires an Architecture Decision Record (ADR) and owner approval. Introducing Nuitka as a build-time compiler dependency requires this approval request.

Honesty note: compiling Python code to native C extensions using Nuitka raises the reverse-engineering barrier substantially compared to plain source or PyInstaller archives. However, compiled Python binaries can still be reverse-engineered by skilled reverse engineers using disassemblers and binary analysis tools. Technical compilation is only one layer of defence; intellectual property protection relies equally on commercial software licenses, non-disclosure agreements (NDAs), and maintaining the primary source repository in private custody.

## Decision

1. **Standalone Nuitka compilation**:
   - Package UTriage as a standalone Windows x64 distribution using Nuitka (`python -m nuitka --standalone`).
   - The launcher entry point (`src/utriage/__main__.py`) compiles into native executable code (`utriage.exe`) along with necessary compiled C extensions, removing raw `.py` source files from the deployed distribution.
   - Nuitka is requested and approved strictly as a build-time dependency in the release environment; it is not shipped to or installed on the plant workstation.

2. **CPU-only PyTorch**:
   - In accordance with ADR 0005, the release distribution bundles the CPU-only build of PyTorch. Plant inspection workstations do not require GPU hardware or CUDA drivers.

3. **Tamper prevention and integrity verification**:
   - Pretrained model checkpoints (`models/`) and configuration files (`configs/`) ship alongside cryptographic SHA-256 checksums recorded in `configs/manifest.json`.
   - Runtime integrity checking (implemented in `src/utriage/integrity.py`) validates file checksums against `configs/manifest.json` on startup.
   - Plant builds enforce `UTRIAGE_LOCKED=1` by default. If any configuration file or model checkpoint differs from its registered checksum, the application halts immediately with an integrity violation error.

4. **Zero network footprint**:
   - No auto-update mechanisms. Updates are distributed strictly through manually verified offline installation packages.
   - Zero telemetry or analytics. Streamlit telemetry is permanently disabled (`gatherUsageStats = false` in `.streamlit/config.toml`).
   - No background sockets, network calls, or remote diagnostics.

5. **Release packaging**:
   - Release distribution is assembled via `scripts/build_release.ps1` into a single zip archive containing `utriage.exe`, bundled runtimes, `configs/`, `models/` (if present), `app/`, `.streamlit/config.toml`, `docs/install.md`, and a root `SHA256SUMS.txt`.

## Consequences

- Owner approval: adds Nuitka as an approved build-time dependency for CI/CD and release builds.
- Build overhead: native C compilation via Nuitka requires MSVC or MinGW64 on the build system and increases release build duration.
- Operational simplicity: plant IT personnel can deploy UTriage by unzipping a verified archive without configuring Python interpreters, virtual environments, or package managers.
- Security posture: satisfies offline aerospace plant security requirements while protecting proprietary algorithms.
