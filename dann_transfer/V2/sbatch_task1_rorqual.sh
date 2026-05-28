#!/bin/bash
# Full V2 pipeline on Alliance Rorqual: DANN (14 folds) -> weak-label GCE.
# Submit from anywhere:  sbatch ~/Dann_transfer/V2/sbatch_task1_rorqual.sh
# Tune --time up to your GPU partition max (see: sinfo -o "%P %l" | grep -i gpu).
# Uses fresh OUT_* dirs so you do not mix with a partial interactive run.
#
# Partition must allow your --time (see: sinfo -o "%P %l" | grep gpu).
#   b5 = 7d   b4 = 3d   b3 = 1d   b2 = 12h   b1 = 3h
# Full pipeline often needs multi-day; 1d/12h may TIMEOUT — use fresh OUT_* dirs if retrying.
#
# Default below: 1 day on gpubase_bygpu_b3. For 12 hours instead:
#   #SBATCH --partition=gpubase_bygpu_b2
#   #SBATCH --time=12:00:00
# Or keep b5 and only shorten time (often OK): e.g. --partition=gpubase_bygpu_b5 --time=12:00:00
#
#SBATCH --partition=gpubase_bygpu_b3
#SBATCH --job-name=dann-v2-full
#SBATCH --account=def-sureshra_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=65536M
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/shiv136/project_data/runs/logs/%x-%j.out
#SBATCH --error=/scratch/shiv136/project_data/runs/logs/%x-%j.err

set -euo pipefail
mkdir -p /scratch/shiv136/project_data/runs/logs

module load cuda/12.6
source "${HOME}/ucaps_venv/bin/activate"

export TARGET_MANIFEST=/scratch/shiv136/project_data/cow_face_sequences_10s_250/completed_manifest.csv
export TARGET_ROOT=/scratch/shiv136/project_data/cow_face_sequences_10s_250
export SRC_PROJECT=/scratch/shiv136/project_data/ucaps_source
export SRC_SEQ=/scratch/shiv136/project_data/ucaps_source/sequence
export CKPT_DIR=/scratch/shiv136/project_data/v2.9_20260222_144752

export OUT_DANN_BASE=/scratch/shiv136/project_data/runs/holstein_task1_dann_v2_run_sbatch
export OUT_WEAK_GCE=/scratch/shiv136/project_data/runs/holstein_task1_weak_gce_v2_run_sbatch

cd "${HOME}/Dann_transfer"
bash V2/run_task1_vast.sh
