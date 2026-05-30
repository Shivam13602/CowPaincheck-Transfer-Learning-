#!/usr/bin/env bash
#SBATCH --account=def-sureshra_gpu
#SBATCH --job-name=v5_732_8cow
#SBATCH --output=/scratch/shiv136/project_data/runs/logs/%x-%j.out
#SBATCH --error=/scratch/shiv136/project_data/runs/logs/%x-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#
# V5 matrix on enlarged 732-seq data. Can run in parallel with S4 job 13485054 (549).
# Prereq: upload v5 sequences + v5_split_v732.json + fixed run_v5_matrix_rorqual.sh
set -euo pipefail

mkdir -p /scratch/shiv136/project_data/runs/logs

if ! module load cuda/12.6 2>/dev/null; then
  module load cuda/12.4 2>/dev/null || module load cuda 2>/dev/null || true
fi

if [[ -f "${HOME}/ucaps_venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/ucaps_venv/bin/activate"
fi

cd "${SLURM_SUBMIT_DIR:-$HOME/Dann_transfer}"
bash "dann_transfer/V5/scripts/run_v5_matrix_rorqual_v732.sh"
