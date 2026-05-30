#!/usr/bin/env bash
#SBATCH --account=def-sureshra_gpu
#SBATCH --job-name=v5_extract_h100
#SBATCH --output=/scratch/shiv136/project_data/runs/logs/%x-%j.out
#SBATCH --error=/scratch/shiv136/project_data/runs/logs/%x-%j.err
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#
# Optional: run v5 extraction on Rorqual H100 (faster than laptop).
# Narval: copy header account to def-sureshra and submit from Narval login.
set -euo pipefail

mkdir -p /scratch/shiv136/project_data/runs/logs
export YOLO_BATCH_SIZE="${YOLO_BATCH_SIZE:-256}"

cd "${SLURM_SUBMIT_DIR:-$HOME/Dann_transfer}"
bash "dann_transfer/V5/scripts/run_extraction_gpu_h100.sh"
