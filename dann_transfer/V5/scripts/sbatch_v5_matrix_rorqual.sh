#!/usr/bin/env bash
#SBATCH --account=def-sureshra_gpu
#SBATCH --job-name=ucaps_v5_8cow
#SBATCH --output=/scratch/shiv136/project_data/runs/logs/%x-%j.out
#SBATCH --error=/scratch/shiv136/project_data/runs/logs/%x-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#
# Slurm wrapper for the V5 core matrix (S2-S4) on Alliance Rorqual.
# Submit from the Dann_transfer checkout root:
#   sbatch dann_transfer/V5/scripts/sbatch_v5_matrix_rorqual.sh
# H100 defaults (override on OOM): WEAK_BATCH_SIZE=128 DANN_BATCH_SIZE=48 NUM_WORKERS=24
#   sbatch --export=ALL,CKPT_DIR=/scratch/.../v2.9_20260502_181533,RUN_S4_DANN=0 \
#          dann_transfer/V5/scripts/sbatch_v5_matrix_rorqual.sh
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
if [[ ! -f "${VENV_ACT}" ]]; then
  echo "ERROR: venv activate not found: ${VENV_ACT}" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${VENV_ACT}"

cd "${SLURM_SUBMIT_DIR:-$HOME/Dann_transfer}"
bash "dann_transfer/V5/scripts/run_v5_matrix_rorqual.sh"
