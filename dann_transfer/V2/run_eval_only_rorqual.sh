#!/usr/bin/env bash
# Re-run DANN + weak-label ensemble/metrics/reports only (no training).
# Requires completed run under OUT_* with fold_*/best_*.pt and split JSONs.
#
# Do NOT run while another job is still writing the same OUT_* directories.
#
# On Rorqual (GPU node or batch): from repo root after exports, or:
#   bash ~/Dann_transfer/V2/run_eval_only_rorqual.sh
set -euo pipefail

export TARGET_MANIFEST=/scratch/shiv136/project_data/cow_face_sequences_10s_250/completed_manifest.csv
export TARGET_ROOT=/scratch/shiv136/project_data/cow_face_sequences_10s_250
export SRC_PROJECT=/scratch/shiv136/project_data/ucaps_source
export SRC_SEQ=/scratch/shiv136/project_data/ucaps_source/sequence
export CKPT_DIR=/scratch/shiv136/project_data/v2.9_20260222_144752

export OUT_DANN_BASE=/scratch/shiv136/project_data/runs/holstein_task1_dann_v2_run_sbatch
export OUT_WEAK_GCE=/scratch/shiv136/project_data/runs/holstein_task1_weak_gce_v2_run_sbatch

cd "${HOME}/Dann_transfer"
exec bash V2/run_eval_only_vast.sh
