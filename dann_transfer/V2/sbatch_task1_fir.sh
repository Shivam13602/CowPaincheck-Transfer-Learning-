#!/bin/bash
# Fir Slurm: full H100 + def-sureshra_gpu. Submit from ~/Dann_transfer:
#   sbatch V2/sbatch_task1_fir.sh
# Tune --partition / --time if scheduler rejects the request (see `sinfo`).
#SBATCH --job-name=dann-v2-t1
#SBATCH --account=def-sureshra_gpu
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=65536M
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/shiv136/project_data/runs/logs/%x-%j.out
#SBATCH --error=/scratch/shiv136/project_data/runs/logs/%x-%j.err

set -euo pipefail
mkdir -p /scratch/shiv136/project_data/runs/logs

module purge 2>/dev/null || true
# Optional: module load python cuda (see Fir docs for stack that matches your torch wheel)

if [[ -f "$HOME/ucaps_venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/ucaps_venv/bin/activate"
else
  echo "Missing ~/ucaps_venv — on login node run: python3 -m venv ~/ucaps_venv && pip install torch torchvision ... (CPU wheel ok for venv create; use Fir-approved CUDA build)"
  exit 1
fi

bash "$HOME/Dann_transfer/V2/run_task1_fir.sh"
