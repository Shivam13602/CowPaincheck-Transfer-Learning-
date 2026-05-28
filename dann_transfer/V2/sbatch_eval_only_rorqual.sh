#!/bin/bash
# Short GPU job: eval-only pass over completed *_sbatch outputs.
# Submit: sbatch ~/Dann_transfer/V2/sbatch_eval_only_rorqual.sh
#
#SBATCH --job-name=dann-v2-eval
#SBATCH --account=def-sureshra_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=65536M
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/shiv136/project_data/runs/logs/%x-%j.out
#SBATCH --error=/scratch/shiv136/project_data/runs/logs/%x-%j.err

set -euo pipefail
mkdir -p /scratch/shiv136/project_data/runs/logs

module load cuda/12.6
source "${HOME}/ucaps_venv/bin/activate"

bash "${HOME}/Dann_transfer/V2/run_eval_only_rorqual.sh"
