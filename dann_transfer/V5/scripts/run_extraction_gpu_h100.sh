#!/usr/bin/env bash
# GPU extraction on Alliance H100 (Rorqual / Narval). Submit via sbatch or interactive salloc.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/Dann_transfer}}"
cd "${REPO_ROOT}"

DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}}"
OUTPUT="${OUTPUT:-/scratch/shiv136/project_data/cow_face_sequences_thesis_stride8_v5/output}"
YOLO_MODEL="${YOLO_MODEL:-${REPO_ROOT}/yolo_cow_face/yolo26s.pt}"
YOLO_BATCH="${YOLO_BATCH_SIZE:-256}"
DEVICE="${DEVICE:-0}"

if ! module load cuda/12.6 2>/dev/null; then
  module load cuda/12.4 2>/dev/null || module load cuda 2>/dev/null || true
fi

if [[ -n "${VENV:-}" ]]; then
  # shellcheck source=/dev/null
  source "${VENV}/bin/activate"
elif [[ -f "${HOME}/ucaps_venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/ucaps_venv/bin/activate"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

python CowPaincheck-Transfer-Learning/datasets/thesis_stride8_qa/create_thesis_stride8_sequences.py \
  --inventory cow_video_dataset_analysis.csv \
  --dataset-root "${DATASET_ROOT}" \
  --model "${YOLO_MODEL}" \
  --videos-per-cow 4 \
  --stride-seconds 8 \
  --seed 42 \
  --output "${OUTPUT}" \
  --overwrite \
  --device "${DEVICE}" \
  --yolo-batch-size "${YOLO_BATCH}" \
  --yolo-imgsz 640 \
  --yolo-half \
  --jpeg-quality 90
