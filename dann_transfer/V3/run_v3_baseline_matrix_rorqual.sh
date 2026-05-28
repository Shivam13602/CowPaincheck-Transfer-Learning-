#!/usr/bin/env bash
# V3 baseline_10s_250 matrix for Alliance Rorqual.
# Run from anywhere: bash V3/run_v3_baseline_matrix_rorqual.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${SCRIPT_DIR}/training_code"
cd "${CODE_DIR}"

TARGET_MANIFEST="${TARGET_MANIFEST:-/scratch/shiv136/project_data/cow_face_sequences_10s_250/completed_manifest.csv}"
TARGET_ROOT="${TARGET_ROOT:-/scratch/shiv136/project_data/cow_face_sequences_10s_250}"
SRC_PROJECT="${SRC_PROJECT:-/scratch/shiv136/project_data/ucaps_source}"
SRC_SEQ="${SRC_SEQ:-/scratch/shiv136/project_data/ucaps_source/sequence}"
CKPT_DIR="${CKPT_DIR:-/scratch/shiv136/project_data/v2.9_20260222_144752}"
OUT_ROOT="${OUT_ROOT:-/scratch/shiv136/project_data/runs/v3_baseline_10s_250}"
SSL_DIR="${SSL_DIR:-${OUT_ROOT}/ssl_fold_train}"

V3_TEST_COW_IDS="${V3_TEST_COW_IDS:-363,403,404,408}"
# H100 + Slurm cpus-per-task=24: keep workers <= CPUs. Lower if the login node OOMs during local tests.
NUM_WORKERS="${NUM_WORKERS:-16}"
# Larger default batch improves GPU utilization on H100; if CUDA OOM, export BATCH_SIZE=16 (or 12).
BATCH_SIZE="${BATCH_SIZE:-24}"
WEAK_EPOCHS="${WEAK_EPOCHS:-80}"
DANN_EPOCHS="${DANN_EPOCHS:-80}"
SSL_EPOCHS="${SSL_EPOCHS:-40}"
LR_SCHEDULER_PATIENCE="${LR_SCHEDULER_PATIENCE:-12}"
DIAG_BOOTSTRAP_SAMPLES="${DIAG_BOOTSTRAP_SAMPLES:-2000}"
RUN_SSL="${RUN_SSL:-0}"
RUN_WEAK="${RUN_WEAK:-1}"
RUN_DANN="${RUN_DANN:-1}"
RUN_CORAL="${RUN_CORAL:-1}"

mkdir -p "${OUT_ROOT}"

COMMON_TARGET_ARGS=(
  --manifest-csv "${TARGET_MANIFEST}"
  --sequence-root "${TARGET_ROOT}"
  --dataset-version baseline_10s_250
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

MAX_FRAMES_ARGS=()
if [[ -n "${MAX_FRAMES:-}" ]]; then
  MAX_FRAMES_ARGS=(--max-frames "${MAX_FRAMES}")
fi

SLIDING_ARGS=()
if [[ -n "${INFER_SLIDING_RAW_SPAN:-}" ]]; then
  SLIDING_ARGS=(--infer-sliding-raw-span "${INFER_SLIDING_RAW_SPAN}")
  [[ -n "${INFER_SLIDING_STRIDE:-}" ]] && SLIDING_ARGS+=(--infer-sliding-stride "${INFER_SLIDING_STRIDE}")
  [[ -n "${INFER_SLIDING_AGGREGATE:-}" ]] && SLIDING_ARGS+=(--infer-sliding-aggregate "${INFER_SLIDING_AGGREGATE}")
fi

MC_ARGS=()
if [[ -n "${EVAL_MC_SAMPLES:-}" ]] && [[ "${EVAL_MC_SAMPLES}" != "0" ]]; then
  MC_ARGS=(--eval-mc-samples "${EVAL_MC_SAMPLES}")
fi

SSL_ARGS=()
if [[ -f "${SSL_DIR}/fold_0/best_ssl_simsiam.pt" ]]; then
  SSL_ARGS=(--ssl-checkpoint-dir "${SSL_DIR}")
else
  echo "No SSL checkpoint at ${SSL_DIR}/fold_0/best_ssl_simsiam.pt; baseline matrix starts from UCAPS checkpoint."
fi

echo "== V3 baseline dry runs =="
python weak_label_adapt_v3.py "${COMMON_TARGET_ARGS[@]}" --out-dir "${OUT_ROOT}/dry_run_weak" --from-scratch --dry-run
python dann_adapt_v3.py "${COMMON_TARGET_ARGS[@]}" --source-project-dir "${SRC_PROJECT}" --source-sequence-dir "${SRC_SEQ}" --out-dir "${OUT_ROOT}/dry_run_dann" --from-scratch --dry-run

if [[ "${RUN_SSL}" == "1" ]]; then
  echo "== V3 SimSiam SSL on fold-train target cows only =="
  python ssl_pretrain_holstein_v3.py \
    "${COMMON_TARGET_ARGS[@]}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --ckpt-kind task1 \
    --init-fold 0 \
    "${MAX_FRAMES_ARGS[@]}" \
    --out-dir "${SSL_DIR}" \
    --num-epochs "${SSL_EPOCHS}" \
    --learning-rate "${SSL_LR:-1e-4}"
  SSL_ARGS=(--ssl-checkpoint-dir "${SSL_DIR}")
fi

if [[ "${RUN_WEAK}" == "1" ]]; then
  for LOSS in bce gce focal; do
    echo "== V3 weak-label ${LOSS} on baseline_10s_250 =="
    EXTRA_LOSS_ARGS=()
    [[ "${LOSS}" == "gce" ]] && EXTRA_LOSS_ARGS=(--gce-q "${GCE_Q:-0.7}")
    [[ "${LOSS}" == "focal" ]] && EXTRA_LOSS_ARGS=(--focal-gamma "${FOCAL_GAMMA:-2.0}")
    python weak_label_adapt_v3.py \
      "${COMMON_TARGET_ARGS[@]}" \
      --checkpoint-dir "${CKPT_DIR}" \
      --ckpt-kind task1 \
      --init-fold 0 \
      "${SSL_ARGS[@]}" \
      "${MAX_FRAMES_ARGS[@]}" \
      "${SLIDING_ARGS[@]}" \
      "${MC_ARGS[@]}" \
      --out-dir "${OUT_ROOT}/weak_${LOSS}" \
      --num-epochs "${WEAK_EPOCHS}" \
      --lr-scheduler-patience "${LR_SCHEDULER_PATIENCE}" \
      --learning-rate "${WEAK_LR:-1e-4}" \
      --task1-loss "${LOSS}" \
      --cow-balanced-sampler \
      --select-metric v3_composite \
      "${EXTRA_LOSS_ARGS[@]}"
  done
fi

if [[ "${RUN_DANN}" == "1" ]]; then
  for DW in 0.0 0.05 0.10 0.25; do
    echo "== V3 source-retained DANN domain_weight=${DW} on baseline_10s_250 =="
    python dann_adapt_v3.py \
      "${COMMON_TARGET_ARGS[@]}" \
      --source-project-dir "${SRC_PROJECT}" \
      --source-sequence-dir "${SRC_SEQ}" \
      --checkpoint-dir "${CKPT_DIR}" \
      --ckpt-kind task1 \
      --init-fold 0 \
      "${SSL_ARGS[@]}" \
      "${MAX_FRAMES_ARGS[@]}" \
      "${SLIDING_ARGS[@]}" \
      "${MC_ARGS[@]}" \
      --out-dir "${OUT_ROOT}/dann_dw_${DW}" \
      --num-epochs "${DANN_EPOCHS}" \
      --lr-scheduler-patience "${LR_SCHEDULER_PATIENCE}" \
      --learning-rate "${DANN_LR:-1e-5}" \
      --alignment-loss domain \
      --domain-weight "${DW}" \
      --source-task1-weight 1.0 \
      --source-task2-weight 0.0 \
      --target-weak-weight 0.0 \
      --source-task1-retention-floor 0.55 \
      --source-task1-retention-margin 0.03 \
      --source-task1-sanity-floor 0.70 \
      --select-metric v3_composite
  done
fi

if [[ "${RUN_CORAL}" == "1" ]]; then
  echo "== V3 Deep CORAL alignment ablation on baseline_10s_250 =="
  python dann_adapt_v3.py \
    "${COMMON_TARGET_ARGS[@]}" \
    --source-project-dir "${SRC_PROJECT}" \
    --source-sequence-dir "${SRC_SEQ}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --ckpt-kind task1 \
    --init-fold 0 \
    "${SSL_ARGS[@]}" \
    "${MAX_FRAMES_ARGS[@]}" \
    "${SLIDING_ARGS[@]}" \
    "${MC_ARGS[@]}" \
    --out-dir "${OUT_ROOT}/coral_w_${CORAL_WEIGHT:-0.10}" \
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

echo "V3 baseline matrix complete: ${OUT_ROOT}"
