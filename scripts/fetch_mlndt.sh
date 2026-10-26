#!/usr/bin/env bash
set -euo pipefail

# scripts/fetch_mlndt.sh
# Shallow-clone iikka-v/ML-NDT into data/external/ML-NDT.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/data/external/ML-NDT"
REPO_URL="https://github.com/iikka-v/ML-NDT.git"

check_prerequisites() {
    if ! command -v git >/dev/null 2>&1; then
        echo "Error: git is required but was not found on PATH." >&2
        exit 1
    fi
}

handle_dry_run() {
    echo "[dry-run] Repository root: ${REPO_ROOT}"
    echo "[dry-run] Target directory: ${TARGET_DIR}"
    if [ -d "${TARGET_DIR}/.git" ]; then
        echo "[dry-run] Target directory already contains .git repository; clone would be skipped."
    else
        echo "[dry-run] Would create directory: ${REPO_ROOT}/data/external"
        echo "[dry-run] Would run: git clone --depth 1 ${REPO_URL} ${TARGET_DIR}"
    fi
    echo "[dry-run] Would print first 15 lines of LICENSE* or COPYING* from ${TARGET_DIR}"
    echo "[dry-run] Would display licence and citation information."
    exit 0
}

clone_or_skip() {
    if [ -d "${TARGET_DIR}/.git" ]; then
        local commit_hash
        commit_hash="$(git -C "${TARGET_DIR}" rev-parse HEAD 2>/dev/null || echo "unknown")"
        echo "already present: ${commit_hash}"
    else
        mkdir -p "${REPO_ROOT}/data/external"
        echo "Cloning ML-NDT repository into ${TARGET_DIR}..."
        git clone --depth 1 "${REPO_URL}" "${TARGET_DIR}"
    fi
}

print_licence() {
    local licence_file=""
    local f
    for f in "${TARGET_DIR}"/LICENSE* "${TARGET_DIR}"/COPYING*; do
        if [ -f "$f" ]; then
            licence_file="$f"
            break
        fi
    done

    if [ -n "$licence_file" ]; then
        echo "Licence file ($(basename "$licence_file")):"
        head -n 15 "$licence_file"
    else
        echo "Warning: No LICENSE* or COPYING* file found in ${TARGET_DIR}" >&2
    fi
}

print_notices() {
    echo "Data licence: LGPL-3.0. Do not redistribute. Cite: Virkkunen, Koskinen, Jessen-Juhler, Rinta-Aho, Augmented Ultrasonic Data for Machine Learning, arXiv:1903.11399."
    echo "Note: Dataset is approximately 2.5 GB and is gitignored."
}

main() {
    local dry_run=false

    for arg in "$@"; do
        case "$arg" in
            --dry-run)
                dry_run=true
                ;;
            *)
                echo "Unknown argument: $arg" >&2
                echo "Usage: $0 [--dry-run]" >&2
                exit 1
                ;;
        esac
    done

    if [ "$dry_run" = true ]; then
        handle_dry_run
    fi

    check_prerequisites
    clone_or_skip
    print_licence
    print_notices
}

main "$@"
