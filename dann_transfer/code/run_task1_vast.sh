#!/usr/bin/env bash
# Task1-only DANN + weak-label ablations on Vast.ai (Holstein CV & diagnostics plan).
# Protocol: 14 inner folds (2 val cows each), fixed V1 test cows 363,403,404,408.
#
# Defaults tuned for a single high-end GPU (e.g. A100 40GB): larger batch, more dataloader
# workers, long epoch budgets, and high ReduceLROnPlateau patience so LR does not collapse
# during noisy val metrics. There is no separate early-stop on epochs—each fold runs the
# full epoch count; checkpoint selection still tracks best validation score seen.
#
# Override via env: NUM_WORKERS, BATCH_SIZE, DANN_EPOCHS, WEAK_EPOCHS, LR_SCHEDULER_PATIENCE,
# MAX_FRAMES, TRAIN_LR, WEAK_LR, DIAG_BOOTSTRAP_SAMPLES (0=off), … — see V2/MAX_FRAMES_SWEEP.md
#
# Run from repo root after: cd /workspace/ucaps_transfer_dann/Dann_transfer
set -euo pipefail

TARGET_MANIFEST="${TARGET_MANIFEST:-/workspace/ucaps_transfer/cow_face_sequences_10s_250/completed_manifest.csv}"
TARGET_ROOT="${TARGET_ROOT:-/workspace/ucaps_transfer/cow_face_sequences_10s_250}"
SRC_PROJECT="${SRC_PROJECT:-/workspace/ucaps_transfer_dann/ucaps_source}"
SRC_SEQ="${SRC_SEQ:-/workspace/ucaps_transfer_dann/ucaps_source/sequence}"
CKPT_DIR="${CKPT_DIR:-/workspace/ucaps_transfer/v2.9/checkpoints_v2.9-20260502T160826Z-3-001/checkpoints_v2.9/v2.9_20260222_144752}"

SSL_DIR="${SSL_DIR:-./holstein_ssl_outputs}"
OUT_DANN_BASE="${OUT_DANN_BASE:-./holstein_task1_dann_v2_run}"
OUT_WEAK_GCE="${OUT_WEAK_GCE:-./holstein_task1_weak_gce_v2_run}"

# Throughput / capacity (override if OOM or CPU-bound)
NUM_WORKERS="${NUM_WORKERS:-12}"
BATCH_SIZE="${BATCH_SIZE:-16}"

# Long-run training (override for smoke tests)
DANN_EPOCHS="${DANN_EPOCHS:-80}"
WEAK_EPOCHS="${WEAK_EPOCHS:-80}"
LR_SCHEDULER_PATIENCE="${LR_SCHEDULER_PATIENCE:-12}"

MAX_FRAMES_ARGS=()
if [[ -n "${MAX_FRAMES:-}" ]]; then
  MAX_FRAMES_ARGS=(--max-frames "${MAX_FRAMES}")
fi

EXTRA_SSL=()
if [[ -f "${SSL_DIR}/fold_0/best_ssl_simsiam.pt" ]]; then
  EXTRA_SSL=(--ssl-checkpoint-dir "$SSL_DIR")
else
  echo "Note: no SSL checkpoint at ${SSL_DIR}/fold_0/best_ssl_simsiam.pt — continuing without SSL init."
fi

SLIDING_ARGS=()
if [[ -n "${INFER_SLIDING_RAW_SPAN:-}" ]]; then
  SLIDING_ARGS=(--infer-sliding-raw-span "${INFER_SLIDING_RAW_SPAN}")
  if [[ -n "${INFER_SLIDING_STRIDE:-}" ]]; then
    SLIDING_ARGS+=(--infer-sliding-stride "${INFER_SLIDING_STRIDE}")
  fi
  if [[ -n "${INFER_SLIDING_AGGREGATE:-}" ]]; then
    SLIDING_ARGS+=(--infer-sliding-aggregate "${INFER_SLIDING_AGGREGATE}")
  fi
fi

MC_ARGS=()
if [[ -n "${EVAL_MC_SAMPLES:-}" ]] && [[ "${EVAL_MC_SAMPLES}" != "0" ]]; then
  MC_ARGS=(--eval-mc-samples "${EVAL_MC_SAMPLES}")
fi

BOOT_ARGS=()
if [[ -n "${DIAG_BOOTSTRAP_SAMPLES+x}" ]]; then
  BOOT_ARGS=(--diag-bootstrap-samples "${DIAG_BOOTSTRAP_SAMPLES}")
fi

echo "== Capacity settings: NUM_WORKERS=$NUM_WORKERS BATCH_SIZE=$BATCH_SIZE DANN_EPOCHS=$DANN_EPOCHS WEAK_EPOCHS=$WEAK_EPOCHS LR_SCHEDULER_PATIENCE=$LR_SCHEDULER_PATIENCE =="

echo "== Task1-only DANN dry run =="
python dann_adapt_v2.9.py \
  --manifest-csv "$TARGET_MANIFEST" \
  --sequence-root "$TARGET_ROOT" \
  --num-workers "$NUM_WORKERS" \
  --out-dir "${OUT_DANN_BASE}_dry_run" \
  --dry-run

echo "== Task1-only DANN full run (14 folds, fixed test cows) =="
python dann_adapt_v2.9.py \
  --manifest-csv "$TARGET_MANIFEST" \
  --sequence-root "$TARGET_ROOT" \
  --source-project-dir "$SRC_PROJECT" \
  --source-sequence-dir "$SRC_SEQ" \
  --checkpoint-dir "$CKPT_DIR" \
  --ckpt-kind task1 \
  --init-fold 0 \
  "${EXTRA_SSL[@]}" \
  "${MAX_FRAMES_ARGS[@]}" \
  "${SLIDING_ARGS[@]}" \
  "${MC_ARGS[@]}" \
  "${BOOT_ARGS[@]}" \
  --out-dir "$OUT_DANN_BASE" \
  --label-column video_health_status \
  --test-cows 4 \
  --test-cow-ids 363,403,404,408 \
  --val-cows-per-fold 2 \
  --num-epochs "$DANN_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr-scheduler-patience "$LR_SCHEDULER_PATIENCE" \
  --learning-rate "${TRAIN_LR:-1e-5}" \
  --domain-weight 0.5 \
  --source-task2-weight 0.0 \
  --target-weak-weight 0.0 \
  --source-task1-sanity-floor 0.7

echo "== Weak-proxy GCE fine-tune (same protocol) =="
python weak_label_adapt_v2.9.py \
  --manifest-csv "$TARGET_MANIFEST" \
  --sequence-root "$TARGET_ROOT" \
  --checkpoint-dir "$CKPT_DIR" \
  --ckpt-kind task2 \
  --init-fold 0 \
  "${EXTRA_SSL[@]}" \
  "${MAX_FRAMES_ARGS[@]}" \
  "${SLIDING_ARGS[@]}" \
  "${MC_ARGS[@]}" \
  "${BOOT_ARGS[@]}" \
  --out-dir "$OUT_WEAK_GCE" \
  --label-column video_health_status \
  --test-cows 4 \
  --test-cow-ids 363,403,404,408 \
  --val-cows-per-fold 2 \
  --num-epochs "$WEAK_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr-scheduler-patience "$LR_SCHEDULER_PATIENCE" \
  --learning-rate "${WEAK_LR:-1e-4}" \
  --task1-loss gce \
  --gce-q 0.7

echo "Done. Sync out-dir folders and logs back to the workstation."
