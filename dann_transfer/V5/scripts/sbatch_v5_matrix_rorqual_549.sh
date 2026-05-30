#!/usr/bin/env bash
#SBATCH --account=def-sureshra_gpu
#SBATCH --job-name=v5_549_8cow
#SBATCH --output=/scratch/shiv136/project_data/runs/logs/%x-%j.out
#SBATCH --error=/scratch/shiv136/project_data/runs/logs/%x-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#
# V5 core matrix on existing 549-seq thesis_stride8 (interim; extraction continues locally).
# Submit from ~/Dann_transfer:
#   sbatch dann_transfer/V5/scripts/sbatch_v5_matrix_rorqual_549.sh
#
# Partial matrix (e.g. weak-label only while testing):
#   sbatch --export=ALL,RUN_S4_DANN=0,RUN_S4_CORAL=0 dann_transfer/V5/scripts/sbatch_v5_matrix_rorqual_549.sh
set -euo pipefail

mkdir -p /scratch/shiv136/project_data/runs/logs

if ! module load cuda/12.6 2>/dev/null; then
  module load cuda/12.4 2>/dev/null || module load cuda 2>/dev/null || true
fi

if [[ -n "${VENV:-}" ]]; then
  if [[ -f "${VENV}" ]]; then
    VENV_ACT="${VENV}"
  elif [[ -f "${VENV}/bin/activate" ]]; then
    VENV_ACT="${VENV}/bin/activate"
  else
    echo "ERROR: VENV=${VENV} is not an activate file and has no bin/activate" >&2
    exit 1
  fi
else
  VENV_ACT="${HOME}/ucaps_venv/bin/activate"
fi
if [[ -f "${VENV_ACT}" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_ACT}"
else
  echo "ERROR: venv activate not found: ${VENV_ACT}" >&2
  exit 1
fi

cd "${SLURM_SUBMIT_DIR:-$HOME/Dann_transfer}"
bash "dann_transfer/V5/scripts/run_v5_matrix_rorqual_549.sh"
