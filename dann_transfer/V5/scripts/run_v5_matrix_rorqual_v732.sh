#!/usr/bin/env bash
# V5 full matrix on enlarged 732-seq dataset (4 videos/cow). Separate OUT_ROOT from 549 interim.
set -euo pipefail

export TARGET_MANIFEST="${TARGET_MANIFEST:-/scratch/shiv136/project_data/cow_face_sequences_thesis_stride8_v5/output/completed_manifest.csv}"
export TARGET_ROOT="${TARGET_ROOT:-/scratch/shiv136/project_data/cow_face_sequences_thesis_stride8_v5/output}"
export OUT_ROOT="${OUT_ROOT:-/scratch/shiv136/project_data/runs/v5_thesis_732_8cow}"
export DATASET_VERSION="${DATASET_VERSION:-thesis_stride8_qa_v732}"
export SPLIT_JSON="${SPLIT_JSON:-/scratch/shiv136/project_data/v5/splits/v5_split_v732.json}"
export CKPT_DIR="${CKPT_DIR:-/scratch/shiv136/project_data/v2.9_20260502_181533}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_v5_matrix_rorqual.sh"
