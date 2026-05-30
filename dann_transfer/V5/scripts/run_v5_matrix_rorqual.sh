#!/usr/bin/env bash
# V5 core matrix on Alliance Rorqual (H100): zero-shot re-baseline (S2) + weak-label
# losses (S3) + domain-alignment sweep (S4) on the enlarged thesis dataset with the
# 8-cow balanced split and the NEW complete v2.9 checkpoint set (v2.9_20260502_181533).
#
# Reuses the V3 trainers; this script only wires the V5 protocol (split, checkpoints,
# loss/alignment grid). Each cow is validated exactly once (5 folds x 4 val cows).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=h100_env.sh
source "${SCRIPT_DIR}/h100_env.sh"
V5_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# V3 trainers (superset with video metrics + spec-constrained thresholds).
CODE_DIR="${CODE_DIR:-${V5_DIR}/../V3/training_code}"

# ---- Paths (override via env) ----------------------------------------------
TARGET_MANIFEST="${TARGET_MANIFEST:-/scratch/shiv136/project_data/cow_face_sequences_thesis_stride8_v5/output/completed_manifest.csv}"
TARGET_ROOT="${TARGET_ROOT:-/scratch/shiv136/project_data/cow_face_sequences_thesis_stride8_v5/output}"
SRC_PROJECT="${SRC_PROJECT:-/scratch/shiv136/project_data/ucaps_source}"
SRC_SEQ="${SRC_SEQ:-/scratch/shiv136/project_data/ucaps_source/sequence}"
# NEW complete checkpoint set: 9 folds x {combined, task1, task2}, no missing files.
CKPT_DIR="${CKPT_DIR:-/scratch/shiv136/project_data/v2.9_20260502_181533}"
OUT_ROOT="${OUT_ROOT:-/scratch/shiv136/project_data/runs/v5_thesis_8cow}"
DATASET_VERSION="${DATASET_VERSION:-thesis_stride8_qa_v5}"

# ---- V5 split protocol ------------------------------------------------------
# 8-cow balanced test (legacy 4 + 4 more); 5 folds x 4 val cows (2H+2U); each cow once.
# The frozen split JSON (make_v5_splits.py output) is the source of truth: it forces
# cow 349 + surplus to train-only and guarantees each CV cow is validated exactly once.
# Regenerate it on the ENLARGED manifest before running, then upload it next to this script.
SPLIT_JSON="${SPLIT_JSON:-/scratch/shiv136/project_data/v5/splits/v5_split.json}"
V5_TEST_COW_IDS="${V5_TEST_COW_IDS:-363,370,378,403,404,408,433,436}"
VAL_COWS_PER_FOLD="${VAL_COWS_PER_FOLD:-4}"

# ---- Compute knobs (H100 defaults from h100_env.sh) -------------------------
WEAK_EPOCHS="${WEAK_EPOCHS:-80}"
DANN_EPOCHS="${DANN_EPOCHS:-80}"
LR_SCHEDULER_PATIENCE="${LR_SCHEDULER_PATIENCE:-12}"
DIAG_BOOTSTRAP_SAMPLES="${DIAG_BOOTSTRAP_SAMPLES:-2000}"

# ---- Stage toggles ----------------------------------------------------------
RUN_S3_WEAK="${RUN_S3_WEAK:-1}"     # weak-label loss grid (bce/focal/gce)
RUN_S4_DANN="${RUN_S4_DANN:-1}"     # domain-weight sweep
RUN_S4_CORAL="${RUN_S4_CORAL:-1}"   # coral-weight sweep

cd "${CODE_DIR}"
mkdir -p "${OUT_ROOT}"

COMMON_BASE=(
  --manifest-csv "${TARGET_MANIFEST}"
  --sequence-root "${TARGET_ROOT}"
  --dataset-version "${DATASET_VERSION}"
  --label-column video_health_status
  --split-json "${SPLIT_JSON}"
  --test-cows 8
  --test-cow-ids "${V5_TEST_COW_IDS}"
  --val-cows-per-fold "${VAL_COWS_PER_FOLD}"
  --require-val-both-classes
  --num-workers "${NUM_WORKERS}"
  --diag-bootstrap-samples "${DIAG_BOOTSTRAP_SAMPLES}"
  --threshold-min-specificity 0.50
)
# weak_label_adapt_v3 vs dann_adapt_v3 use different flag names for the same sampler.
COMMON_WEAK_ARGS=( "${COMMON_BASE[@]}" --cow-balanced-sampler )
COMMON_DANN_ARGS=( "${COMMON_BASE[@]}" --target-cow-balanced-sampler )

WEAK_ARGS=(--batch-size "${WEAK_BATCH_SIZE}")
DANN_ARGS=(--batch-size "${DANN_BATCH_SIZE}")

echo "== V5 preflight =="
echo "H100 workers    : ${NUM_WORKERS} prefetch=${DATALOADER_PREFETCH} persistent=${DATALOADER_PERSISTENT}"
echo "Weak batch      : ${WEAK_BATCH_SIZE}  |  DANN/CORAL batch: ${DANN_BATCH_SIZE}"
echo "Dataset tag   : ${DATASET_VERSION}"
echo "Manifest      : ${TARGET_MANIFEST}"
echo "Sequence root : ${TARGET_ROOT}"
echo "Checkpoints   : ${CKPT_DIR}"
echo "Output root   : ${OUT_ROOT}"
echo "Test cows (8) : ${V5_TEST_COW_IDS}"
test -f "${TARGET_MANIFEST}" || { echo "ERROR: missing manifest ${TARGET_MANIFEST}" >&2; exit 2; }
test -d "${TARGET_ROOT}/sequences" || { echo "ERROR: missing sequences under ${TARGET_ROOT}" >&2; exit 2; }
test -d "${CKPT_DIR}" || { echo "ERROR: missing checkpoint dir ${CKPT_DIR}" >&2; exit 2; }
test -f "${SPLIT_JSON}" || { echo "ERROR: missing split JSON ${SPLIT_JSON} (run make_v5_splits.py on the enlarged manifest and upload it)" >&2; exit 2; }
echo "Split JSON    : ${SPLIT_JSON}"
# Sanity: complete checkpoint set should have 9 task1 fold files.
T1_COUNT="$(find "${CKPT_DIR}" -name 'best_model_v2.9_task1_fold_*.pt' 2>/dev/null | wc -l | tr -d ' ')"
echo "task1 fold checkpoints found: ${T1_COUNT} (expected 9)"
[[ "${T1_COUNT}" -ge 9 ]] || echo "WARNING: fewer than 9 task1 fold checkpoints; per-fold init will fall back."

# Dry-run split audit (writes split plan without training).
python weak_label_adapt_v3.py "${COMMON_WEAK_ARGS[@]}" "${WEAK_ARGS[@]}" --out-dir "${OUT_ROOT}/dry_run_split" --from-scratch --dry-run

# ---------------------------------------------------------------------------
# S3 — weak-label loss grid (frozen CNN, class-balanced), per-fold init.
# ---------------------------------------------------------------------------
if [[ "${RUN_S3_WEAK}" == "1" ]]; then
  for LOSS in bce focal gce; do
    echo "== S3 weak_${LOSS} =="
    python weak_label_adapt_v3.py \
      "${COMMON_WEAK_ARGS[@]}" "${WEAK_ARGS[@]}" \
      --checkpoint-dir "${CKPT_DIR}" --ckpt-kind task1 --init-fold 0 \
      --out-dir "${OUT_ROOT}/S3_weak_${LOSS}" \
      --num-epochs "${WEAK_EPOCHS}" --lr-scheduler-patience "${LR_SCHEDULER_PATIENCE}" \
      --learning-rate "${WEAK_LR:-1e-4}" \
      --task1-loss "${LOSS}" --focal-gamma 2.0 --gce-q 0.7 \
      --freeze-cnn --class-balanced \
      --select-metric v3_composite
  done
fi

# ---------------------------------------------------------------------------
# S4 — domain-adversarial weight sweep (retention-gated).
# ---------------------------------------------------------------------------
if [[ "${RUN_S4_DANN}" == "1" ]]; then
  for DW in 0.10 0.25 0.50; do
    echo "== S4 dann_dw_${DW} =="
    python dann_adapt_v3.py \
      "${COMMON_DANN_ARGS[@]}" "${DANN_ARGS[@]}" \
      --source-project-dir "${SRC_PROJECT}" --source-sequence-dir "${SRC_SEQ}" \
      --checkpoint-dir "${CKPT_DIR}" --ckpt-kind task1 --init-fold 0 \
      --out-dir "${OUT_ROOT}/S4_dann_dw_${DW}" \
      --num-epochs "${DANN_EPOCHS}" --lr-scheduler-patience "${LR_SCHEDULER_PATIENCE}" \
      --learning-rate "${DANN_LR:-1e-5}" \
      --alignment-loss domain --domain-weight "${DW}" \
      --source-task1-weight 1.0 --source-task2-weight 0.0 --target-weak-weight 0.0 \
      --source-task1-retention-floor 0.55 --source-task1-retention-margin 0.03 \
      --source-task1-sanity-floor 0.70 \
      --select-metric v3_composite
  done
fi

# ---------------------------------------------------------------------------
# S4 — CORAL weight sweep (low weights; V4 showed collapse at higher dense data).
# ---------------------------------------------------------------------------
if [[ "${RUN_S4_CORAL}" == "1" ]]; then
  for CW in 0.01 0.05 0.10; do
    echo "== S4 coral_w_${CW} =="
    python dann_adapt_v3.py \
      "${COMMON_DANN_ARGS[@]}" "${DANN_ARGS[@]}" \
      --source-project-dir "${SRC_PROJECT}" --source-sequence-dir "${SRC_SEQ}" \
      --checkpoint-dir "${CKPT_DIR}" --ckpt-kind task1 --init-fold 0 \
      --out-dir "${OUT_ROOT}/S4_coral_w_${CW}" \
      --num-epochs "${DANN_EPOCHS}" --lr-scheduler-patience "${LR_SCHEDULER_PATIENCE}" \
      --learning-rate "${DANN_LR:-1e-5}" \
      --alignment-loss coral --domain-weight 0.0 --coral-weight "${CW}" \
      --source-task1-weight 1.0 --source-task2-weight 0.0 --target-weak-weight 0.0 \
      --source-task1-retention-floor 0.55 --source-task1-retention-margin 0.03 \
      --source-task1-sanity-floor 0.70 \
      --select-metric v3_composite
  done
fi

echo "V5 core matrix complete: ${OUT_ROOT}"
echo "Reports include sequence-, video-, and cow-level metrics on the 8-cow test."
