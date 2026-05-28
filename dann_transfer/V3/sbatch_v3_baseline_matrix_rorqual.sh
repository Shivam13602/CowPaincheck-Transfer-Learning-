#!/usr/bin/env bash
# GPU account for Rorqual; omit --partition so Slurm picks a valid queue (partition names change by site).
# To pin a queue after checking `sinfo`, add: #SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --account=def-sureshra_gpu
#SBATCH --job-name=ucaps_v3_base
#SBATCH --output=/scratch/shiv136/project_data/runs/logs/%x-%j.out
#SBATCH --error=/scratch/shiv136/project_data/runs/logs/%x-%j.err
# Shorter walltime can help scheduling; raise if the full matrix hits TIMEOUT (Slurm: D-HH:MM:SS).
#SBATCH --time=2-00:00:00
# Extra CPUs for DataLoader workers (keep NUM_WORKERS <= cpus-per-task in run script).
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1

set -euo pipefail

mkdir -p /scratch/shiv136/project_data/runs/logs

# Compute nodes may not expose the same default modules as the login node; do not hard-fail the whole job.
if ! module load cuda/12.6 2>/dev/null; then
  module load cuda/12.4 2>/dev/null || module load cuda 2>/dev/null || true
fi

# Virtualenv: leave VENV unset to use ~/ucaps_venv, or set VENV to the venv *root* OR to .../bin/activate.
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

cd "${SLURM_SUBMIT_DIR:-$PWD}"
bash "V3/run_v3_baseline_matrix_rorqual.sh"
