#!/usr/bin/env bash
# V3 thesis_stride8_qa training on Alliance Rorqual — weak_focal + coral_w_0.10 only.
# Tuned for Rorqual H100: AMP + TF32 enabled in trainers; batch/workers set for high GPU util.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${SCRIPT_DIR}/training_code"
cd "${CODE_DIR}"

TARGET_MANIFEST="${TARGET_MANIFEST:-/scratch/shiv136/project_data/cow_face_sequences_thesis_stride8/output/completed_manifest.csv}"
TARGET_ROOT="${TARGET_ROOT:-/scratch/shiv136/project_data/cow_face_sequences_thesis_stride8/output}"
SRC_PROJECT="${SRC_PROJECT:-/scratch/shiv136/project_data/ucaps_source}"
SRC_SEQ="${SRC_SEQ:-/scratch/shiv136/project_data/ucaps_source/sequence}"
CKPT_DIR="${CKPT_DIR:-/scratch/shiv136/project_data/v2.9_20260222_144752}"
OUT_ROOT="${OUT_ROOT:-/scratch/shiv136/project_data/runs/v3_thesis_stride8_qa}"

V3_TEST_COW_IDS="${V3_TEST_COW_IDS:-363,403,404,408}"
# thesis_stride8_qa has 31 unique cows → 27 train-pool → 9 folds × 3 val cows.
VAL_COWS_PER_FOLD="${VAL_COWS_PER_FOLD:-3}"
# H100 + 24 CPUs: keep workers <= cpus-per-task; prefetch_factor=2 in loaders.
NUM_WORKERS="${NUM_WORKERS:-20}"
# 240-frame sequences @ 224px; raise toward 40 if memory allows (watch nvidia-smi).
BATCH_SIZE="${BATCH_SIZE:-32}"
WEAK_EPOCHS="${WEAK_EPOCHS:-80}"
DANN_EPOCHS="${DANN_EPOCHS:-80}"
LR_SCHEDULER_PATIENCE="${LR_SCHEDULER_PATIENCE:-12}"
DIAG_BOOTSTRAP_SAMPLES="${DIAG_BOOTSTRAP_SAMPLES:-2000}"
RUN_WEAK="${RUN_WEAK:-1}"
RUN_CORAL="${RUN_CORAL:-1}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

mkdir -p "${OUT_ROOT}"

COMMON_TARGET_ARGS=(
  --manifest-csv "${TARGET_MANIFEST}"
  --sequence-root "${TARGET_ROOT}"
  --dataset-version thesis_stride8_qa
  --label-column video_health_status
  --test-cows 4
  --test-cow-ids "${V3_TEST_COW_IDS}"
  --val-cows-per-fold "${VAL_COWS_PER_FOLD}"
  --require-val-both-classes
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --diag-bootstrap-samples "${DIAG_BOOTSTRAP_SAMPLES}"
  --threshold-min-specificity 0.50
)

echo "== Preflight =="
echo "Manifest: ${TARGET_MANIFEST}"
echo "Sequence root: ${TARGET_ROOT}"
echo "Output root: ${OUT_ROOT}"
echo "GPU batch size: ${BATCH_SIZE} | DataLoader workers: ${NUM_WORKERS}"
test -f "${TARGET_MANIFEST}" || { echo "ERROR: missing manifest ${TARGET_MANIFEST}" >&2; exit 2; }
test -d "${TARGET_ROOT}/sequences" || { echo "ERROR: missing sequences under ${TARGET_ROOT}" >&2; exit 2; }
test -d "${CKPT_DIR}" || { echo "ERROR: missing checkpoint dir ${CKPT_DIR}" >&2; exit 2; }
SEQ_COUNT="$(find "${TARGET_ROOT}/sequences" -name metadata.json 2>/dev/null | wc -l | tr -d ' ')"
echo "Sequence folders on scratch: ${SEQ_COUNT}"
if [[ "${SEQ_COUNT}" -lt 500 ]]; then
  echo "ERROR: expected ~549 sequences on scratch, found ${SEQ_COUNT}" >&2
  exit 2
fi

python weak_label_adapt_v3.py "${COMMON_TARGET_ARGS[@]}" --out-dir "${OUT_ROOT}/dry_run_weak" --from-scratch --dry-run
python dann_adapt_v3.py "${COMMON_TARGET_ARGS[@]}" --source-project-dir "${SRC_PROJECT}" --source-sequence-dir "${SRC_SEQ}" --out-dir "${OUT_ROOT}/dry_run_coral" --from-scratch --dry-run

if [[ "${RUN_WEAK}" == "1" ]]; then
  echo "== V3 weak_focal on thesis_stride8_qa =="
  python weak_label_adapt_v3.py \
    "${COMMON_TARGET_ARGS[@]}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --ckpt-kind task1 \
    --init-fold 0 \
    --out-dir "${OUT_ROOT}/weak_focal" \
    --num-epochs "${WEAK_EPOCHS}" \
    --lr-scheduler-patience "${LR_SCHEDULER_PATIENCE}" \
    --learning-rate "${WEAK_LR:-1e-4}" \
    --task1-loss focal \
    --focal-gamma "${FOCAL_GAMMA:-2.0}" \
    --cow-balanced-sampler \
    --select-metric v3_composite
fi

if [[ "${RUN_CORAL}" == "1" ]]; then
  echo "== V3 coral_w_0.10 on thesis_stride8_qa =="
  python dann_adapt_v3.py \
    "${COMMON_TARGET_ARGS[@]}" \
    --source-project-dir "${SRC_PROJECT}" \
    --source-sequence-dir "${SRC_SEQ}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --ckpt-kind task1 \
    --init-fold 0 \
    --out-dir "${OUT_ROOT}/coral_w_0.10" \
    --num-epochs "${DANN_EPOCHS}" \
    --lr-scheduler-patience "${LR_SCHEDULER_PATIENCE}" \
    --learning-rate "${DANN_LR:-1e-5}" \
    --alignment-loss coral \
    --domain-weight 0.0 \
    --coral-weight "${CORAL_WEIGHT:-0.10}" \
    --source-task1-weight 1.0 \
    --source-task2-weight 0.0 \
    --target-weak-weight 0.0 \
    --source-task1-retention-floor 0.55 \
    --source-task1-retention-margin 0.03 \
    --source-task1-sanity-floor 0.70 \
    --select-metric v3_composite
fi

echo "V3 thesis_stride8_qa training complete: ${OUT_ROOT}"
echo "Reports include sequence-, video-, and cow-level metrics."
