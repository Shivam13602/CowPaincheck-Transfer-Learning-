#!/usr/bin/env bash
# V5 matrix on the frozen 549-seq thesis_stride8 dataset already on Rorqual (V4 path).
# Same 8-cow test + v5_split.json protocol; does not wait for v5 extraction.
# Results go to OUT_ROOT=v5_thesis_8cow_549 so the enlarged-data run can use v5_thesis_8cow later.
set -euo pipefail

export TARGET_MANIFEST="${TARGET_MANIFEST:-/scratch/shiv136/project_data/cow_face_sequences_thesis_stride8/output/completed_manifest.csv}"
export TARGET_ROOT="${TARGET_ROOT:-/scratch/shiv136/project_data/cow_face_sequences_thesis_stride8/output}"
export OUT_ROOT="${OUT_ROOT:-/scratch/shiv136/project_data/runs/v5_thesis_8cow_549}"
export DATASET_VERSION="${DATASET_VERSION:-thesis_stride8_qa_549_interim}"
export SPLIT_JSON="${SPLIT_JSON:-/scratch/shiv136/project_data/v5/splits/v5_split.json}"
export CKPT_DIR="${CKPT_DIR:-/scratch/shiv136/project_data/v2.9_20260502_181533}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_v5_matrix_rorqual.sh"
