#!/bin/bash
# Full V3.1 pipeline on Alliance Rorqual: forked DANN -> forked weak-label GCE.
# Submit: sbatch ~/Dann_transfer/V3.1/sbatch_task1_v3_1_rorqual.sh
#
#SBATCH --partition=gpubase_bygpu_b3
#SBATCH --job-name=dann-v31-full
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

export OUT_DANN_BASE=/scratch/shiv136/project_data/runs/holstein_task1_dann_v31_run_sbatch
export OUT_WEAK_GCE=/scratch/shiv136/project_data/runs/holstein_task1_weak_gce_v31_run_sbatch

cd "${HOME}/Dann_transfer"
bash V3.1/run_task1_v3_1.sh
