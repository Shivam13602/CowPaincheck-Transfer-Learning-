#!/usr/bin/env bash
# Regenerate V3 reports/metrics from saved fold checkpoints without training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${SCRIPT_DIR}/training_code"
cd "${CODE_DIR}"

TARGET_MANIFEST="${TARGET_MANIFEST:-/scratch/shiv136/project_data/cow_face_sequences_10s_250/completed_manifest.csv}"
TARGET_ROOT="${TARGET_ROOT:-/scratch/shiv136/project_data/cow_face_sequences_10s_250}"
DATASET_VERSION="${DATASET_VERSION:-baseline_10s_250}"
SRC_PROJECT="${SRC_PROJECT:-/scratch/shiv136/project_data/ucaps_source}"
SRC_SEQ="${SRC_SEQ:-/scratch/shiv136/project_data/ucaps_source/sequence}"
CKPT_DIR="${CKPT_DIR:-/scratch/shiv136/project_data/v2.9_20260222_144752}"
OUT_ROOT="${OUT_ROOT:-/scratch/shiv136/project_data/runs/v3_baseline_10s_250}"

V3_TEST_COW_IDS="${V3_TEST_COW_IDS:-363,403,404,408}"
NUM_WORKERS="${NUM_WORKERS:-12}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DIAG_BOOTSTRAP_SAMPLES="${DIAG_BOOTSTRAP_SAMPLES:-2000}"

COMMON_ARGS=(
  --manifest-csv "${TARGET_MANIFEST}"
  --sequence-root "${TARGET_ROOT}"
  --dataset-version "${DATASET_VERSION}"
  --label-column video_health_status
  --test-cows 4
  --test-cow-ids "${V3_TEST_COW_IDS}"
  --val-cows-per-fold 4
  --require-val-both-classes
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --diag-bootstrap-samples "${DIAG_BOOTSTRAP_SAMPLES}"
  --threshold-min-specificity 0.50
)

for DIR in "${OUT_ROOT}"/weak_*; do
  [[ -d "${DIR}" ]] || continue
  [[ -f "${DIR}/weak_label_cv_splits.json" ]] || continue
  echo "== Eval-only weak: ${DIR} =="
  python weak_label_adapt_v3.py \
    "${COMMON_ARGS[@]}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --ckpt-kind task1 \
    --out-dir "${DIR}" \
    --eval-only
done

for DIR in "${OUT_ROOT}"/dann_* "${OUT_ROOT}"/coral_* "${OUT_ROOT}"/domain_*; do
  [[ -d "${DIR}" ]] || continue
  [[ -f "${DIR}/dann_splits.json" ]] || continue
  echo "== Eval-only DANN/CORAL: ${DIR} =="
  python dann_adapt_v3.py \
    "${COMMON_ARGS[@]}" \
    --source-project-dir "${SRC_PROJECT}" \
    --source-sequence-dir "${SRC_SEQ}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --ckpt-kind task1 \
    --out-dir "${DIR}" \
    --eval-only
done

echo "V3 eval-only refresh complete: ${OUT_ROOT}"
