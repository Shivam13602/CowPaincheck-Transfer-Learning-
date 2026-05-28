#!/usr/bin/env bash
# Fir scratch paths — same layout as prepared for Rorqual.
# Create venv + PyTorch on a Fir *login* node, then run via sbatch/salloc (GPU nodes often have no outbound pip).
# Usage (after ssh + cd):  bash V2/run_task1_fir.sh
set -euo pipefail

ROOT="${PROJECT_DATA_ROOT:-/scratch/shiv136/project_data}"
export TARGET_MANIFEST="$ROOT/cow_face_sequences_10s_250/completed_manifest.csv"
export TARGET_ROOT="$ROOT/cow_face_sequences_10s_250"
export SRC_PROJECT="$ROOT/ucaps_source"
export SRC_SEQ="$ROOT/ucaps_source/sequence"
export CKPT_DIR="${CKPT_DIR:-$ROOT/v2.9_20260222_144752}"

mkdir -p "$ROOT/runs"
export OUT_DANN_BASE="${OUT_DANN_BASE:-$ROOT/runs/holstein_task1_dann_v2_run}"
export OUT_WEAK_GCE="${OUT_WEAK_GCE:-$ROOT/runs/holstein_task1_weak_gce_v2_run}"
export SSL_DIR="${SSL_DIR:-$HOME/Dann_transfer/holstein_ssl_outputs}"

cd "$HOME/Dann_transfer"
exec bash V2/run_task1_vast.sh
