#!/usr/bin/env bash
# V3 dense_10s_stride5_qa matrix for Alliance Rorqual.
# Build the dense dataset first, then repeat only the best baseline settings.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${SCRIPT_DIR}/training_code"
DATA_CODE_DIR="${SCRIPT_DIR}/data_code"
cd "${CODE_DIR}"

DENSE_MANIFEST="${DENSE_MANIFEST:-/scratch/shiv136/project_data/cow_face_sequences_10s_v3_dense/completed_manifest.csv}"
DENSE_ROOT="${DENSE_ROOT:-/scratch/shiv136/project_data/cow_face_sequences_10s_v3_dense}"
INVENTORY="${INVENTORY:-/scratch/shiv136/project_data/cow_video_dataset_analysis.csv}"
DATASET_ROOT="${DATASET_ROOT:-/scratch/shiv136/project_data}"
YOLO_MODEL="${YOLO_MODEL:-/scratch/shiv136/project_data/yolo_cow_face/best.pt}"
SRC_PROJECT="${SRC_PROJECT:-/scratch/shiv136/project_data/ucaps_source}"
SRC_SEQ="${SRC_SEQ:-/scratch/shiv136/project_data/ucaps_source/sequence}"
CKPT_DIR="${CKPT_DIR:-/scratch/shiv136/project_data/v2.9_20260222_144752}"
OUT_ROOT="${OUT_ROOT:-/scratch/shiv136/project_data/runs/v3_dense_10s_stride5_qa}"
SSL_DIR="${SSL_DIR:-${OUT_ROOT}/ssl_fold_train}"

V3_TEST_COW_IDS="${V3_TEST_COW_IDS:-363,403,404,408}"
NUM_WORKERS="${NUM_WORKERS:-16}"
BATCH_SIZE="${BATCH_SIZE:-24}"
WEAK_EPOCHS="${WEAK_EPOCHS:-80}"
DANN_EPOCHS="${DANN_EPOCHS:-80}"
BEST_WEAK_LOSS="${BEST_WEAK_LOSS:-gce}"
BEST_DANN_DOMAIN_WEIGHT="${BEST_DANN_DOMAIN_WEIGHT:-0.10}"
BEST_ALIGNMENT_LOSS="${BEST_ALIGNMENT_LOSS:-domain}"
DIAG_BOOTSTRAP_SAMPLES="${DIAG_BOOTSTRAP_SAMPLES:-2000}"
BUILD_DENSE="${BUILD_DENSE:-1}"

mkdir -p "${OUT_ROOT}"

if [[ "${BUILD_DENSE}" == "1" ]]; then
  echo "== V3 dense dataset generation =="
  python "${DATA_CODE_DIR}/create_dense_10s_face_sequences_v3.py" \
    --inventory "${INVENTORY}" \
    --dataset-root "${DATASET_ROOT}" \
    --model "${YOLO_MODEL}" \
    --output "${DENSE_ROOT}" \
    --sequence-seconds 10 \
    --stride-seconds 5 \
    --target-fps 24 \
    --crop-size 224 \
    --conf 0.60 \
    --crop-pad 0.08 \
    --min-detection-rate 0.90 \
    --max-filled-rate 0.10 \
    --min-mean-confidence 0.80 \
    --min-min-confidence 0.60
fi

COMMON_TARGET_ARGS=(
  --manifest-csv "${DENSE_MANIFEST}"
  --sequence-root "${DENSE_ROOT}"
  --dataset-version dense_10s_stride5_qa
  --label-column video_health_status
  --test-cows 4
  --test-cow-ids "${V3_TEST_COW_IDS}"
  --val-cows-per-fold 4
  --require-val-both-classes
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --diag-bootstrap-samples "${DIAG_BOOTSTRAP_SAMPLES}"
  --threshold-min-specificity 0.50
  --qa-min-detection-rate 0.90
  --qa-max-filled-rate 0.10
  --qa-min-mean-confidence 0.80
  --qa-min-min-confidence 0.60
)

MAX_FRAMES_ARGS=()
if [[ -n "${MAX_FRAMES:-}" ]]; then
  MAX_FRAMES_ARGS=(--max-frames "${MAX_FRAMES}")
fi

SSL_ARGS=()
if [[ -f "${SSL_DIR}/fold_0/best_ssl_simsiam.pt" ]]; then
  SSL_ARGS=(--ssl-checkpoint-dir "${SSL_DIR}")
fi

echo "== V3 dense weak ${BEST_WEAK_LOSS} =="
python weak_label_adapt_v3.py \
  "${COMMON_TARGET_ARGS[@]}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --ckpt-kind task1 \
  --init-fold 0 \
  "${SSL_ARGS[@]}" \
  "${MAX_FRAMES_ARGS[@]}" \
  --out-dir "${OUT_ROOT}/weak_${BEST_WEAK_LOSS}" \
  --num-epochs "${WEAK_EPOCHS}" \
  --learning-rate "${WEAK_LR:-1e-4}" \
  --task1-loss "${BEST_WEAK_LOSS}" \
  --cow-balanced-sampler \
  --select-metric v3_composite

echo "== V3 dense DANN/CORAL best baseline setting =="
DOMAIN_WEIGHT="${BEST_DANN_DOMAIN_WEIGHT}"
CORAL_WEIGHT="${CORAL_WEIGHT:-0.10}"
if [[ "${BEST_ALIGNMENT_LOSS}" == "coral" ]]; then
  DOMAIN_WEIGHT="0.0"
fi
python dann_adapt_v3.py \
  "${COMMON_TARGET_ARGS[@]}" \
  --source-project-dir "${SRC_PROJECT}" \
  --source-sequence-dir "${SRC_SEQ}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --ckpt-kind task1 \
  --init-fold 0 \
  "${SSL_ARGS[@]}" \
  "${MAX_FRAMES_ARGS[@]}" \
  --out-dir "${OUT_ROOT}/${BEST_ALIGNMENT_LOSS}_dw_${DOMAIN_WEIGHT}" \
  --num-epochs "${DANN_EPOCHS}" \
  --learning-rate "${DANN_LR:-1e-5}" \
  --alignment-loss "${BEST_ALIGNMENT_LOSS}" \
  --domain-weight "${DOMAIN_WEIGHT}" \
  --coral-weight "${CORAL_WEIGHT}" \
  --source-task1-weight 1.0 \
  --source-task2-weight 0.0 \
  --target-weak-weight 0.0 \
  --source-task1-retention-floor 0.55 \
  --source-task1-retention-margin 0.03 \
  --source-task1-sanity-floor 0.70 \
  --select-metric v3_composite

echo "V3 dense matrix complete: ${OUT_ROOT}"
